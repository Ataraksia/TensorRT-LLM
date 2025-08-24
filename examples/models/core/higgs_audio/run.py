# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
from io import BytesIO
from urllib.request import urlopen

import librosa
import tensorrt as trt
import torch
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from utils import add_common_args

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm import logger
from tensorrt_llm.bindings import KVCacheType
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import (PYTHON_BINDINGS, ModelConfig, ModelRunner,
                                  SamplingConfig, Session, TensorInfo)

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp


def get_engine_name(rank: int) -> str:
    return f"rank{rank}.engine"


def trt_dtype_to_torch(dtype):
    if dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    elif dtype == trt.int32:
        return torch.int32
    else:
        raise TypeError(f"{dtype} is not supported")


class HiggsAudioInfer:

    def __init__(self,
                 audio_engine_path: str,
                 tokenizer_dir: str,
                 engine_dir: str,
                 log_level: str,
                 output_csv: str,
                 output_npy: str,
                 num_beams: int,
                 gpu_id: int = 0):
        self.audio_engine_path = audio_engine_path
        self.tokenizer_dir = tokenizer_dir
        self.engine_dir = engine_dir
        self.log_level = log_level
        self.max_seq_len = 0
        self.runner = None
        self.tokenizer = None
        self.processor = None
        self.config = None
        self.sampling_config = None
        self.output_csv = output_csv
        self.output_npy = output_npy
        self.num_beams = num_beams
        self.model_config = None
        self.gpu_device = torch.device("cuda", gpu_id)

    def get_model(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_dir,
            legacy=False,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(self.tokenizer_dir,
                                                  trust_remote_code=True)
        config_path = os.path.join(self.engine_dir, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        self.max_seq_len = config["build_config"]["max_seq_len"]
        assert self.max_seq_len > 0, "max_seq_len must be positive"

        gen_config_path = os.path.join(self.tokenizer_dir,
                                       "generation_config.json")
        with open(gen_config_path, "r") as f:
            gen_config = json.load(f)
        top_k = gen_config.get("top_k", 1)
        top_p = gen_config.get("top_p", 0.0)
        eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id
        pad_token_id = tokenizer.pad_token_id

        use_gpt_attention_plugin = config["build_config"]["plugin_config"][
            "gpt_attention_plugin"]
        remove_input_padding = config["build_config"]["plugin_config"][
            "remove_input_padding"]
        dtype = config["pretrained_config"]["dtype"]
        tp_size = config["pretrained_config"]["mapping"]["tp_size"]
        pp_size = config["pretrained_config"]["mapping"]["pp_size"]
        world_size = tp_size * pp_size
        assert (
            world_size == tensorrt_llm.mpi_world_size()
        ), f"Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})"
        num_heads = config["pretrained_config"]["num_attention_heads"] // world_size
        max_batch_size = config["build_config"]["max_batch_size"]
        hidden_size = config["pretrained_config"]["hidden_size"] // world_size
        vocab_size = config["pretrained_config"]["vocab_size"]
        num_layers = config["pretrained_config"]["num_hidden_layers"]
        num_kv_heads = config["pretrained_config"].get("num_key_value_heads",
                                                       num_heads)
        if "kv_cache_type" in config["build_config"]:
            kv_cache_type = KVCacheType.from_string(
                config["build_config"]["kv_cache_type"])
        else:
            kv_cache_type = KVCacheType.CONTINUOUS

        tokens_per_block = config["build_config"]["plugin_config"][
            "tokens_per_block"]
        max_prompt_embedding_table_size = config["build_config"].get(
            "max_prompt_embedding_table_size", 0)
        quant_mode = QuantMode.from_quant_algo(
            config["pretrained_config"]["quantization"]["quant_algo"],
            config["pretrained_config"]["quantization"][
                "kv_cache_quant_algo"],
        )
        if config["pretrained_config"].get("multi_query_mode", False):
            tensorrt_llm.logger.warning(
                "`multi_query_mode` config is deprecated. Please rebuild the engine.")
            num_kv_heads = 1

        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size=world_size,
                                               rank=runtime_rank,
                                               tp_size=tp_size,
                                               pp_size=pp_size)

        model_config = ModelConfig(
            max_batch_size=max_batch_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=num_layers,
            gpt_attention_plugin=use_gpt_attention_plugin,
            kv_cache_type=kv_cache_type,
            tokens_per_block=tokens_per_block,
            remove_input_padding=remove_input_padding,
            dtype=dtype,
            quant_mode=quant_mode,
            max_prompt_embedding_table_size=max_prompt_embedding_table_size,
            max_beam_width=self.num_beams,
        )
        sampling_config = SamplingConfig(
            end_id=eos_token_id,
            pad_id=pad_token_id,
            num_beams=self.num_beams,
            top_k=top_k,
            top_p=top_p,
            temperature=1.0,
        )

        engine_name = get_engine_name(runtime_rank)
        serialize_path = os.path.join(self.engine_dir, engine_name)
        logger.info(f"Loading LLM engine from {serialize_path}")
        return (
            model_config,
            sampling_config,
            runtime_mapping,
            runtime_rank,
            serialize_path,
            tokenizer,
            processor,
            eos_token_id,
            pad_token_id,
        )

    def model_init(self, args):
        logger.info(f"Loading audio engine from {self.audio_engine_path}")
        with open(self.audio_engine_path, "rb") as f:
            engine_buffer = f.read()
        logger.info(f"Creating session from engine {self.audio_engine_path}")
        self.session_audio = Session.from_serialized_engine(engine_buffer)

        self.config, _unused = AutoConfig.from_pretrained(
            self.tokenizer_dir,
            return_unused_kwargs=True,
            trust_remote_code=True,
        )

        (
            model_config,
            sampling_config,
            runtime_mapping,
            runtime_rank,
            serialize_path,
            tokenizer,
            processor,
            eos_token_id,
            pad_token_id,
        ) = self.get_model()
        runner_cls = ModelRunner if args.use_py_session else ModelRunnerCpp
        runner_kwargs = dict(
            engine_dir=args.engine_dir,
            lora_dir=args.lora_dir,
            rank=runtime_rank,
            debug_mode=args.debug_mode,
            lora_ckpt_source=args.lora_ckpt_source,
            gpu_weights_percent=args.gpu_weights_percent,
            max_output_len=args.max_new_tokens,
        )
        if not args.use_py_session:
            runner_kwargs.update(
                is_enc_dec=False,
                max_batch_size=model_config.max_batch_size,
                max_input_len=self.max_seq_len - args.max_new_tokens,
                max_beam_width=model_config.max_beam_width,
                max_attention_window_size=args.max_attention_window_size,
                sink_token_length=args.sink_token_length,
                max_tokens_in_paged_kv_cache=args.max_tokens_in_paged_kv_cache,
                kv_cache_enable_block_reuse=args.kv_cache_enable_block_reuse,
                kv_cache_free_gpu_memory_fraction=args.
                kv_cache_free_gpu_memory_fraction,
                cross_kv_cache_fraction=None,
                enable_chunked_context=args.enable_chunked_context,
                multi_block_mode=args.multi_block_mode,
                cuda_graph_mode=args.cuda_graph_mode,
                device_ids=[args.gpu_id])
        runner_kwargs.update(
            enable_context_fmha_fp32_acc=args.enable_context_fmha_fp32_acc)
        self.runner = runner_cls.from_dir(**runner_kwargs)
        self.tokenizer = tokenizer
        self.processor = processor
        self.sampling_config = sampling_config
        self.model_config = model_config

    def build_user_input(self, audio=None, text=None):
        assert isinstance(audio, str) or isinstance(
            text, str), "audio or text must be provided as user input"
        content = []
        if audio:
            content.append({'type': 'audio', 'audio_url': audio})
        if text:
            content.append({'type': 'text', 'text': text})
        user_input = {'role': 'user', 'content': content}
        return user_input

    def get_raw_audios(self, audio_url):
        audios = []
        for url in audio_url:
            if os.path.isfile(url):
                audio_data, _ = librosa.load(
                    url, sr=self.processor.feature_extractor.sampling_rate)
            else:
                audio_data, _ = librosa.load(
                    BytesIO(urlopen(url).read()),
                    sr=self.processor.feature_extractor.sampling_rate)
            audios.append(audio_data)
        return audios

    def audio_tower(self, audios, mask, stream, run_time=1):
        audios = audios.to(self.gpu_device)
        mask = mask.to(self.gpu_device)
        audio_inputs = {"input": audios.float(), "mask": mask}
        audio_output_info = self.session_audio.infer_shapes([
            TensorInfo("input", trt.DataType.FLOAT, audios.shape),
            TensorInfo("mask", trt.DataType.HALF, mask.shape)
        ])
        audio_outputs = {
            t.name:
            torch.empty(tuple(t.shape),
                        dtype=trt_dtype_to_torch(t.dtype),
                        device=self.gpu_device)
            for t in audio_output_info
        }
        profiler.start("Audio")
        for _ in range(run_time):
            ok = self.session_audio.run(audio_inputs, audio_outputs,
                                        stream.cuda_stream)
        stream.synchronize()
        audio_time = profiler.stop("Audio") / run_time
        logger.info(f"TensorRT-LLM Audio latency: {audio_time:3f} sec ")

        assert ok, "Runtime execution failed for audio session"

        audio_features = audio_outputs.get("output") or audio_outputs.get(
            "encoder_output")
        return audio_features

    def get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths

    def higgs_infer(self,
                    input_text,
                    audios,
                    audio_ids,
                    args,
                    stream,
                    history=None,
                    past_audio_features=None,
                    run_time=1):
        assert input_text, "input_text must be provided"
        assert torch.cuda.is_available(), "no gpu available"
        device = torch.device("cpu")
        if isinstance(history, list):
            history.append(input_text)
            full_text = self.processor.apply_chat_template(
                history, add_generation_prompt=True, tokenize=False)
        else:
            full_text = input_text
        inputs = self.processor(
            text=full_text,
            audios=audios,
            return_tensors="pt",
            padding=True,
            sampling_rate=self.processor.feature_extractor.sampling_rate)
        inputs = inputs.to(device)
        input_ids = inputs.input_ids

        if hasattr(inputs, 'input_features') and inputs.input_features is not None:
            batch_size, _, max_mel_seq_len = inputs.input_features.shape
            feature_attention_mask = inputs.feature_attention_mask

            audio_feat_lengths, num_audio_tokens = self.get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1))

            max_seq_len = (max_mel_seq_len - 2) // 2 + 1
            seq_range = (torch.arange(0,
                                      max_seq_len,
                                      dtype=audio_feat_lengths.dtype,
                                      device=device).unsqueeze(0).expand(
                                          batch_size, max_seq_len))
            lengths_expand = audio_feat_lengths.unsqueeze(1).expand(
                batch_size, max_seq_len)
            padding_mask = seq_range >= lengths_expand

            audio_attention_mask_ = padding_mask.view(
                batch_size, 1, 1, max_seq_len).expand(batch_size, 1,
                                                      max_seq_len, max_seq_len)
            audio_attention_mask = audio_attention_mask_.to(dtype=torch.float16,
                                                            device=device)
            audio_attention_mask[audio_attention_mask_] = float("-inf")

            audio_features = self.audio_tower(inputs.input_features,
                                              audio_attention_mask, stream,
                                              run_time)

            num_audios, max_audio_tokens, embed_dim = audio_features.shape
            audio_features_mask = torch.arange(max_audio_tokens,
                                               device=device).expand(
                                                   num_audios,
                                                   max_audio_tokens) < num_audio_tokens.unsqueeze(1)
            masked_audio_features = audio_features[audio_features_mask].view(
                -1, embed_dim)

            special_audio_token_mask = input_ids == self.config.audio_in_token_idx
            special_audio_token_num = special_audio_token_mask.sum().item()
            if past_audio_features is not None:
                assert isinstance(past_audio_features, list)
                assert special_audio_token_num == len(past_audio_features) + num_audios
                cur_audio_features = torch.split(masked_audio_features,
                                                 num_audio_tokens.tolist())
                if len(past_audio_features) > 0:
                    masked_audio_features = torch.cat(
                        (torch.cat(past_audio_features).to(
                            masked_audio_features.device), masked_audio_features))
                    past_num_audio_tokens = torch.tensor(
                        [past_feat.size(0) for past_feat in past_audio_features])
                    num_audio_tokens = torch.cat(
                        (past_num_audio_tokens.to(num_audio_tokens.device),
                         num_audio_tokens))
                past_audio_features.extend(
                    [cur_feat.cpu() for cur_feat in cur_audio_features])

            batch_indices, non_audio_indices = torch.where(input_ids !=
                                                           self.config.audio_in_token_idx)
            batch_indices, audio_indices = torch.where(input_ids ==
                                                       self.config.audio_in_token_idx)

            vocab_size = self.model_config.vocab_size
            fake_prompt_id = torch.arange(vocab_size,
                                          vocab_size + num_audio_tokens.sum(),
                                          device=device)

            input_ids[batch_indices, audio_indices] = fake_prompt_id
            input_lengths = torch.tensor(input_ids.size(1),
                                         dtype=torch.int32,
                                         device=self.gpu_device)
            dtype = self.model_config.dtype
            prompt_table = masked_audio_features

            assert isinstance(audio_ids, list), "audio_ids must be a list"
            assert (len(audio_ids) == num_audio_tokens.size(0))
            for i in audio_ids:
                assert isinstance(i, int) and i > 0
            extra_ids = torch.zeros_like(input_ids,
                                         dtype=torch.int64,
                                         device=device)
            seq_extra_ids = torch.cat([
                torch.full((n, ), audio_ids[i], dtype=torch.int64)
                for i, n in enumerate(num_audio_tokens)
            ]).to(device)
            extra_ids[batch_indices, audio_indices] = seq_extra_ids
            extra_ids = extra_ids.tolist()
        else:
            input_ids = input_ids.to(dtype=torch.int32, device=self.gpu_device)
            input_lengths = torch.tensor(input_ids.size(1),
                                         dtype=torch.int32,
                                         device=self.gpu_device)
            prompt_table = None
            extra_ids = torch.zeros_like(input_ids, dtype=torch.int64).tolist()

        max_input_length = torch.max(torch.tensor([input_ids.size(1)],
                                                 device=self.gpu_device,
                                                 dtype=torch.int32)).item()
        max_new_tokens = min(args.max_new_tokens,
                             self.max_seq_len - max_input_length)

        if prompt_table is None:
            prompt_table = torch.empty([1, self.model_config.hidden_size],
                                       device=self.gpu_device)

        prompt_table = prompt_table.unsqueeze(0)
        profiler.start("HiggsAudio")
        runner = self.runner
        outputs = runner.generate(
            batch_input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            max_attention_window_size=args.max_attention_window_size,
            sink_token_length=args.sink_token_length,
            end_id=self.sampling_config.end_id,
            pad_id=self.sampling_config.pad_id,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_sequences,
            length_penalty=args.length_penalty,
            early_stopping=args.early_stopping,
            repetition_penalty=args.repetition_penalty,
            presence_penalty=args.presence_penalty,
            frequency_penalty=args.frequency_penalty,
            stop_words_list=None,
            bad_words_list=self.sampling_config.bad_words_list,
            random_seed=args.random_seed,
            lora_uids=args.lora_task_uids,
            prompt_table=prompt_table,
            prompt_tasks="0",
            output_sequence_lengths=True,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            return_dict=True,
            return_all_generated_tokens=False,
            input_token_extra_ids=extra_ids)
        output_ids = outputs['output_ids']
        torch.cuda.synchronize()
        gen_time = profiler.stop("HiggsAudio")

        runtime_rank = tensorrt_llm.mpi_rank()
        if runtime_rank == 0:
            for b in range(output_ids.shape[0]):
                inputs_b = input_ids[b]
                if self.num_beams <= 1:
                    outputs_b = output_ids[b][0, len(inputs_b):].tolist()
                    output_text = self.tokenizer.decode(outputs_b,
                                                        skip_special_tokens=True)
                    print(f'Output: "{output_text}"')
                else:
                    for beam in range(self.num_beams):
                        outputs_b = output_ids[b][beam, len(inputs_b):].tolist()
                        output_text = self.tokenizer.decode(
                            outputs_b, skip_special_tokens=True)
                        print(f'Output(beam: {beam}): "{output_text}"')
        logger.info(f"TensorRT-LLM HiggsAudio time: {gen_time:3f} sec ")
        if isinstance(history, list):
            history.append({'role': 'assistant', 'content': output_text})
        return output_text, past_audio_features


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument(
        "--audio_engine_path",
        type=str,
        default="plan/audio_encoder/audio_encoder_fp16.plan",
    )
    parser.add_argument(
        "--input_text",
        type=str,
        default="<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:")
    parser.add_argument(
        "--audio_url",
        nargs="+",
        type=str,
        default=["./audio/sample.wav"],
    )
    parser.add_argument(
        "--input_tokens",
        dest="input_file",
        type=str,
        help="CSV or Numpy file containing tokenized input. Alternative to text input.",
        default=None,
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        help="CSV file where the tokenized output is stored.",
        default=None,
    )
    parser.add_argument(
        "--output_npy",
        type=str,
        help="Numpy file where the tokenized output is stored.",
        default=None,
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        help="Specify GPU device index for running.",
        default=0,
    )
    parser = add_common_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    tensorrt_llm.logger.set_level(args.log_level)
    torch.cuda.set_device(args.gpu_id)
    infer = HiggsAudioInfer(args.audio_engine_path, args.tokenizer_dir,
                            args.engine_dir, args.log_level, args.output_csv,
                            args.output_npy, args.num_beams, args.gpu_id)
    infer.model_init(args)

    audios = infer.get_raw_audios(args.audio_url)
    gpu_device = torch.device("cuda", args.gpu_id)
    stream = torch.cuda.current_stream(device=gpu_device)
    infer.higgs_infer(args.input_text, audios, [1], args, stream, None, None, 1)
