
import argparse
import json
import os
from io import BytesIO
from urllib.request import urlopen
from types import SimpleNamespace
import sys
import numpy as np
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
from tensorrt_llm.runtime import (PYTHON_BINDINGS, ModelConfig, ModelRunner,SamplingConfig, Session, TensorInfo, ModelRunnerCpp)

sys.path.append("/home/me/TTS/train-higgs-audio/")
from boson_multimodal.higgs_audio import *

def trt_dtype_to_torch(dtype):
    if dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.bfloat16:
        return torch.bfloat16
    else:
        raise TypeError(f"{dtype} is not supported")


class HiggsAudioInfer:
    def __init__(self, audio_tokenizer_path: str, audio_engine_path: str, text_engine_path: str, text_tokenizer_path: str, top_k: int, top_p: float, temperature: float, num_beams: int, max_new_tokens: int, device: torch.device):
        self.device = device
        self.max_new_tokens = max_new_tokens
        logger.info(f"Loading audio engine from {audio_engine_path}")
        with open(os.path.join(audio_engine_path, "model.engine"), "rb") as f:
            engine_buffer = f.read()
        self.session_audio = Session.from_serialized_engine(engine_buffer)
        self.audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer_path)
        self.tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_path)
        self.processor = AutoProcessor.from_pretrained(
                "openai/whisper-large-v3-turbo",
                trust_remote=True,
                device=self.device,
            )
        text_config_path = os.path.join(text_engine_path, "config.json")
        with open(text_config_path, "r") as f:
            text_config = json.load(f)
        audio_config_path = os.path.join(text_tokenizer_path, "config.json")
        with open(audio_config_path, "r") as f:
            audio_config = json.load(f)

        self.collator = HiggsAudioSampleCollator(
            whisper_processor=self.processor,
            encode_whisper_embed=True,
            audio_in_token_id=audio_config["audio_in_token_idx"],
            audio_out_token_id=audio_config["audio_out_token_idx"],
            audio_stream_bos_id=audio_config["audio_stream_bos_id"],
            audio_stream_eos_id=audio_config["audio_stream_eos_id"],
            pad_token_id=audio_config["pad_token_id"],
            return_audio_in_tokens=False,
            use_delay_pattern=audio_config["use_delay_pattern"],
            audio_num_codebooks=audio_config["audio_num_codebooks"],
            round_to=1,
        )
        
        self.eos_token_id = audio_config["text_config"]["eos_token_id"]
        self.pad_token_id = audio_config["audio_encoder_config"]["pad_token_id"]
        self.audio_in_token_idx = audio_config["audio_in_token_idx"]
        use_gpt_attention_plugin = text_config["build_config"]["plugin_config"]["gpt_attention_plugin"]
        remove_input_padding = text_config["build_config"]["plugin_config"]["remove_input_padding"]
        dtype = text_config["pretrained_config"]["dtype"]
        tp_size = text_config["pretrained_config"]["mapping"]["tp_size"]
        pp_size = text_config["pretrained_config"]["mapping"]["pp_size"]
        world_size = tp_size * pp_size
        assert (
            world_size == tensorrt_llm.mpi_world_size()
        ), f"Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})"
        num_heads = text_config["pretrained_config"]["num_attention_heads"] // world_size
        max_batch_size = text_config["build_config"]["max_batch_size"]
        hidden_size = text_config["pretrained_config"]["hidden_size"] // world_size
        vocab_size = text_config["pretrained_config"]["vocab_size"]
        num_layers = text_config["pretrained_config"]["num_hidden_layers"]
        num_kv_heads = text_config["pretrained_config"]["num_key_value_heads"]
        if "kv_cache_type" in text_config["build_config"]:
            kv_cache_type = KVCacheType.from_string(
                text_config["build_config"]["kv_cache_type"])
        else:
            kv_cache_type = KVCacheType.CONTINUOUS

        tokens_per_block = text_config["build_config"]["plugin_config"]["tokens_per_block"]
        max_prompt_embedding_table_size = text_config["build_config"]["max_prompt_embedding_table_size"]
        quant_mode = QuantMode.from_quant_algo(
            text_config["pretrained_config"]["quantization"]["quant_algo"],
            text_config["pretrained_config"]["quantization"]["kv_cache_quant_algo"],
        )
        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size=world_size,
                                               rank=runtime_rank,
                                               tp_size=tp_size,
                                               pp_size=pp_size)

        self.model_config = ModelConfig(
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
            max_beam_width=num_beams,
        )
        self.sampling_config = SamplingConfig(
            end_id=self.eos_token_id,
            pad_id=self.pad_token_id,
            num_beams=num_beams,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        logger.info(f"Loading LLM engine from {text_engine_path}")
        runner_kwargs = dict(
            engine_dir=text_engine_path,
            rank=runtime_rank,  
            is_enc_dec=False,
            max_batch_size=max_batch_size,
            max_beam_width=num_beams,
            max_output_len=max_new_tokens,
        )
        self.runner = ModelRunnerCpp.from_dir(**runner_kwargs)

    
    def prepare_inputs(self, audio=None, text=None):
        assert isinstance(audio, str) or isinstance(
            text, str), "audio or text must be provided as user input"
        messages = []
        system_content = f"You are an AI assistant designed to convert text into speech. Generate speech for the following text, using the specified speaker voice.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room\n\n\nSpeaker is an enthusiastic young Australian woman in her early 20s with a bright, clear voice.\n SPEAKER0: "
        system_message = Message(
            role="system",
            content=[TextContent(system_content), AudioContent(audio_url=audio), TextContent("\n<|scene_desc_end|>")],
        )
        user_message = Message(
            role="user",
            content=TextContent(text),
        )
        messages.append(system_message)
        messages.append(user_message)
        chatml_sample = ChatMLSample(messages=messages)
        input_tokens, _, audio_content, _ = prepare_chatml_sample(
            chatml_sample,
            self.tokenizer,
        )
        postfix = "<|start_header_id|>assistant<|end_header_id|>\n\n<|audio_out_bos|>"
        postfix = self.tokenizer.encode(postfix, add_special_tokens=False)
        input_tokens.extend(postfix)
        input_ids = torch.LongTensor(input_tokens)
        if audio_content[0].audio_url not in ["placeholder", ""]:
            raw_audio, _ = librosa.load(audio_content[0].audio_url, sr=self.audio_tokenizer.sampling_rate)
        elif audio_content[0].raw_audio is not None:
            raw_audio, _ = librosa.load(
                BytesIO(base64.b64decode(audio_content[0].raw_audio)), sr=self.audio_tokenizer.sampling_rate
            )
        audio_tokens = self.audio_tokenizer.encode(raw_audio, self.audio_tokenizer.sampling_rate)
        audio_ids = audio_tokens.squeeze(0)
        audio_ids_start = torch.tensor(
                np.cumsum(np.array([0] + [audio_ids.shape[1]])),
                dtype=torch.long,
                device=self.device,
            )[0:-1]
            
        sample = ChatMLDatasetSample(
            input_ids=input_tokens,
            label_ids=None,
            audio_ids_concat=audio_ids,
            audio_ids_start=audio_ids_start,
            audio_waveforms_concat=None,
            audio_waveforms_start=None,
            audio_sample_rate=None,
            audio_speaker_indices=None,
        )
        data = self.collator([sample])
        inputs = asdict(data)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
        print(inputs)
        return inputs

    def audio_tower(self, audios, mask, stream, run_time=1):
        audios = audios.to(self.device)
        mask = mask.to(self.device)
        audio_inputs = {"input": audios.float(), "mask": mask.float()}
        audio_output_info = self.session_audio.infer_shapes([
            TensorInfo("input", trt.DataType.FLOAT, audios.shape),
            TensorInfo("mask", trt.DataType.FLOAT, mask.shape)
        ])
        audio_outputs = {
            t.name:
            torch.empty(tuple(t.shape),
                        dtype=trt_dtype_to_torch(t.dtype),
                        device=self.device)
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

        audio_features = audio_outputs.get("output")
        if audio_features is None:
            audio_features = audio_outputs.get("encoder_output")
        return audio_features

    def get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths

    def higgs_infer(self,
                    input_ids,
                    audio_ids,
                    args,
                    stream,
                    past_audio_features=None,
                    run_time=1):
        assert input_ids, "input_ids must be provided"
        assert torch.cuda.is_available(), "no gpu available"
       
        proc_out = self.processor(
            input_ids,
            audio_ids,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )

        if hasattr(proc_out, 'input_features') and proc_out.input_features is not None:
            batch_size, _, max_mel_seq_len = proc_out.input_features.shape
            # Obtain per-frame attention mask; fall back gracefully if not provided
            if hasattr(proc_out, 'feature_attention_mask') and proc_out.feature_attention_mask is not None:
                feature_attention_mask = proc_out.feature_attention_mask
            elif hasattr(proc_out, 'attention_mask') and proc_out.attention_mask is not None:
                feature_attention_mask = proc_out.attention_mask
            else:
                # Derive mask from non-zero frames across feature dimension
                feature_attention_mask = (proc_out.input_features.abs().sum(dim=1) > 0).to(dtype=torch.long)

            #TODO Fix this so that it input_features don't get cut off
            # audio_mel_len = 3000
            # # Enforce fixed mel length for audio engine by padding/truncating
            # cur_len = proc_out.input_features.shape[-1]
            # if cur_len < audio_mel_len:
            #     pad_amt = audio_mel_len - cur_len
            #     pad_feats = torch.zeros(
            #         (batch_size, proc_out.input_features.shape[1], pad_amt),
            #         dtype=proc_out.input_features.dtype,
            #     )
            #     proc_out.input_features = torch.cat(
            #         [proc_out.input_features, pad_feats], dim=-1)
            #     pad_mask = torch.zeros((batch_size, pad_amt), dtype=feature_attention_mask.dtype)
            #     feature_attention_mask = torch.cat([feature_attention_mask, pad_mask], dim=-1)
            # elif cur_len > audio_mel_len:
            #     proc_out.input_features = proc_out.input_features[..., :audio_mel_len]
            #     feature_attention_mask = feature_attention_mask[..., :audio_mel_len]

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
            audio_attention_mask = audio_attention_mask_.to(dtype=torch.float32,
                                                            device=device)
            # Use large negative instead of -inf to avoid NaNs in some kernels
            audio_attention_mask[audio_attention_mask_] = -1e9

            audio_features = self.audio_tower(proc_out.input_features,
                                              audio_attention_mask, stream,
                                              run_time)

            num_audios, max_audio_tokens, embed_dim = audio_features.shape
            audio_features_mask = torch.arange(max_audio_tokens,
                                               device=device).expand(
                                                   num_audios,
                                                   max_audio_tokens) < num_audio_tokens.unsqueeze(1)
            masked_audio_features = audio_features[audio_features_mask].view(
                -1, embed_dim)

            # Prepare input_ids: prefer processor output; otherwise, build from tokenizer and expand AUDIO token
            if hasattr(proc_out, 'input_ids') and proc_out.input_ids is not None:
                input_ids = proc_out.input_ids
            else:
                text_tokens = self.tokenizer(
                    input_ids,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=True,
                )
                input_ids = text_tokens.input_ids
            special_audio_token_mask = input_ids == self.audio_in_token_idx
            special_audio_token_num = special_audio_token_mask.sum().item()
            # If the number of special tokens does not match total audio tokens, expand inline
            total_audio_tok = int(num_audio_tokens.sum().item())
            if special_audio_token_num != total_audio_tok:
                # Assume batch_size == 1 for current use case
                assert input_ids.size(0) == 1, "Batch >1 expansion not implemented"
                ids = input_ids[0]
                pos = (ids == self.audio_in_token_idx).nonzero(as_tuple=False).view(-1)
                assert pos.numel() >= 1, "Audio placeholder token not found in input_ids"
                p = pos[0].item()
                before = ids[:p]
                after = ids[p + 1:]
                expanded = torch.full((total_audio_tok,), self.audio_in_token_idx, dtype=ids.dtype)
                ids = torch.cat([before, expanded, after], dim=0)
                input_ids = ids.unsqueeze(0)
                special_audio_token_mask = input_ids == self.audio_in_token_idx
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
                                                           self.audio_in_token_idx)
            batch_indices, audio_indices = torch.where(input_ids ==
                                                       self.audio_in_token_idx)

            vocab_size = self.model_config.vocab_size
            fake_prompt_id = torch.arange(vocab_size,
                                          vocab_size + num_audio_tokens.sum(),
                                          device=device)

            input_ids[batch_indices, audio_indices] = fake_prompt_id
            input_lengths = torch.tensor(input_ids.size(1),
                                         dtype=torch.int32,
                                         device=self.device)
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
            input_ids = input_ids.to(dtype=torch.int32, device=self.device)
            input_lengths = torch.tensor(input_ids.size(1),
                                         dtype=torch.int32,
                                         device=self.device)
            prompt_table = None
            extra_ids = torch.zeros_like(input_ids, dtype=torch.int64).tolist()

        if prompt_table is None:
            prompt_table = torch.empty([1, self.model_config.hidden_size],
                                       device=self.device)

        max_input_length = torch.max(input_lengths).item()
        max_new_tokens = min(self.max_new_tokens,
                             max_seq_len - max_input_length)

        prompt_table = prompt_table.unsqueeze(0)
        profiler.start("HiggsAudio")
        # Move input IDs to GPU and cast to int32 for TRT-LLM runtime
        input_ids_gpu = input_ids.to(dtype=torch.int32, device=self.device)
        outputs = self.runner.generate(
            batch_input_ids=input_ids_gpu,
            max_new_tokens=max_new_tokens,
            max_attention_window_size=args.max_attention_window_size,
            sink_token_length=args.sink_token_length,
            end_id=self.eos_token_id,
            pad_id=self.pad_token_id,
            temperature=self.sampling_config.temperature,
            top_k=self.sampling_config.top_k,
            top_p=self.sampling_config.top_p,
            num_beams=self.sampling_config.num_beams,
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

        return output_text, past_audio_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_common_args(parser).parse_args()
    text_tokenizer_path = "/home/me/TTS/higgs-audio-v2-generation-3B-base"
    audio_tokenizer_path = "/home/me/TTS/higgs-audio-v2-generation-3B-base-tokenizer"
    audio_engine_path = "/home/me/TTS/higgs-audio-v2-generation-3B-base-FP32-TRT"
    text_engine_path = "/home/me/TTS/Llama-3.2-3B-TRT"
    audio_url = "/home/me/TTS/AussieGirl.wav"
    input_text = "Chat, stop backseating! I totally know what I'm doing... I think."
    top_k = 50
    top_p = 0.95
    temperature = 1.0
    max_new_tokens = 1024
    num_beams = 1
    device = torch.device("cuda", 0)
    infer = HiggsAudioInfer(audio_tokenizer_path, audio_engine_path, text_engine_path, text_tokenizer_path, top_k, top_p, temperature, num_beams, max_new_tokens, device)
    inputs = infer.prepare_inputs(audio_url, input_text)
    # stream = torch.cuda.current_stream(device=device)
    # infer.higgs_infer(input_ids, audio_ids, args, stream, None, None, 1)

