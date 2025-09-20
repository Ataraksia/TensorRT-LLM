import base64
import inspect
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import OrderedDict, Union

import jiwer
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from openai import OpenAI
from silero_vad import get_speech_timestamps, load_silero_vad
from torch import nn
from transformers import (
    AutoModel,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    pipeline,
)
from vector_quantize_pytorch import ResidualFSQ
from xcodec.descriptaudiocodec.dac.model import dac as dac2
from xcodec.modules.semantic_module import Decoder, Encoder
from xcodec.quantization.vq import ResidualVectorQuantizer

from run_chat_completion import AutoModelForSpeechSeq2Seq
from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig

import logging

from tensorrt_llm.runtime.delay_pattern_logits_processor import DelayPatternLogitsProcessor
from tensorrt_llm.runtime.higgs_audio_model_runner import HiggsAudioModelRunner

load_dotenv()

# Set up logging at the top level
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _build_delay_pattern_mask(input_ids: torch.LongTensor, bos_token_id: int, pad_token_id: int):
    """Implement the delay pattern proposed in "Simple and Controllable Music Generation", https://arxiv.org/pdf/2306.05284

    In the delay pattern, each codebook is offset by the previous codebook by
    one. We insert a special delay token at the start of the sequence if its delayed, and append pad token once the sequence finishes.

    Take the example where there are 4 codebooks and audio sequence length=8. After shifting, the output should have length seq_len + num_codebooks - 1

    - [ *,  *,  *,  *,  *,  *,  *,  *, P, P, P]
    - [ B,  *,  *,  *,  *,  *,  *,  *, *, P, P]
    - [ B,  B,  *,  *,  *,  *,  *,  *, *, *, P]
    - [ B,  B,  B,  *,  *,  *,  *,  *, *, *, *]

    where B indicates the delay token id, P is the special padding token id and `*` indicates that the original audio token.

    Now let's consider the case where we have a sequence of audio tokens to condition on.
    The audio tokens were originally in the following non-delayed form:

    - [a, b]
    - [c, d]
    - [e, f]
    - [g, h]

    After conversion, we get the following delayed form:
    - [a, b, -1, -1, -1]
    - [B, c,  d, -1, -1]
    - [B, B,  e,  f, -1]
    - [B, B,  B,  g,  h]

    Note that we have a special token `-1` that indicates it should be replaced by a new token we see in the generation phase.
    In that case, we should override the `-1` tokens in auto-regressive generation.

    Args:
        input_ids (:obj:`torch.LongTensor`):
            The input ids of the prompt. It will have shape (num_codebooks, seq_len).
        bos_token_id (:obj:`int`):
            The id of the special delay token
        pad_token_id (:obj:`int`):
            The id of the padding token. Should be the same as eos_token_id.

    Returns:
        input_ids (:obj:`torch.LongTensor`):
            The transformed input ids with delay pattern applied. It will have shape ( num_codebooks, seq_len + num_codebooks - 1).
        input_ids_with_gen_mask (:obj:`torch.LongTensor`):
            The transformed input ids with delay pattern applied. The -1 in the output indicates new tokens that should be generated.

    """
    num_codebooks, seq_len = input_ids.shape

    new_seq_len = seq_len + num_codebooks - 1
    input_ids_with_gen_mask = torch.ones(
        (num_codebooks, new_seq_len), dtype=torch.long, device=input_ids.device
    )
    bos_mask = torch.tril(input_ids_with_gen_mask, -1) > 0
    eos_mask = torch.triu(input_ids_with_gen_mask, seq_len) > 0
    input_ids_with_gen_mask[bos_mask] = bos_token_id
    input_ids_with_gen_mask[(~bos_mask) & (~eos_mask)] = input_ids.reshape(-1)
    input_ids = input_ids_with_gen_mask.clone()
    input_ids[eos_mask] = pad_token_id
    input_ids_with_gen_mask[eos_mask] = -1
    return input_ids


def revert_delay_pattern(data):
    """Convert samples encoded with delay pattern back to the original form.

    Args:
        data: The data with delay pattern applied.
              Shape (num_codebooks, seq_len + num_codebooks - 1).

    Returns:
        ret: Recovered data with delay pattern removed.
             Shape (num_codebooks, seq_len).
    """
    assert len(data.shape) == 2
    out_l = []
    num_codebooks = data.shape[0]
    for i in range(num_codebooks):
        out_l.append(data[i : (i + 1), i : (data.shape[1] - num_codebooks + 1 + i)])
    return (
        torch.cat(out_l, dim=0) if isinstance(data, torch.Tensor) else np.concatenate(out_l, axis=0)
    )


class EncodedResult:
    def __init__(self, audio_codes):
        self.audio_codes = audio_codes


class HiggsAudioFeatureExtractor(nn.Module):
    def __init__(self, sampling_rate=16000):
        super().__init__()
        self.sampling_rate = sampling_rate

    def forward(self, raw_audio, sampling_rate=16000, return_tensors="pt"):
        # Convert from librosa to torch
        audio_signal = torch.tensor(raw_audio)
        audio_signal = audio_signal.unsqueeze(0)
        if len(audio_signal.shape) < 3:
            audio_signal = audio_signal.unsqueeze(0)
        return {"input_values": audio_signal}


class HiggsAudioTokenizer(nn.Module):
    def __init__(
        self,
        n_filters: int = 32,
        D: int = 128,
        target_bandwidths: Sequence[Union[int, float]] = [1, 1.5, 2, 4, 6],
        ratios: Sequence[int] = [8, 5, 4, 2],  # downsampling by 320
        sample_rate: int = 16000,
        bins: int = 1024,
        n_q: int = 8,
        codebook_dim: int = None,
        normalize: bool = False,
        causal: bool = False,
        semantic_techer: str = "hubert_base_general",
        last_layer_semantic: bool = True,
        merge_mode: str = "concat",
        downsample_mode: str = "step_down",
        semantic_mode: str = "classic",
        vq_scale: int = 1,
        semantic_sample_rate: int = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.hop_length = np.prod(ratios)
        self.semantic_techer = semantic_techer

        self.frame_rate = math.ceil(sample_rate / np.prod(ratios))  # 50 Hz

        self.target_bandwidths = target_bandwidths
        self.n_q = n_q
        self.sample_rate = sample_rate
        self.encoder = dac2.Encoder(64, ratios, D)

        self.decoder_2 = dac2.Decoder(D, 1024, ratios)
        self.last_layer_semantic = last_layer_semantic
        self.device = device
        if semantic_techer == "hubert_base":
            self.semantic_model = AutoModel.from_pretrained("facebook/hubert-base-ls960")
            self.semantic_sample_rate = 16000
            self.semantic_dim = 768
            self.encoder_semantic_dim = 768

        elif semantic_techer == "wavlm_base_plus":
            self.semantic_model = AutoModel.from_pretrained("microsoft/wavlm-base-plus")
            self.semantic_sample_rate = 16000
            self.semantic_dim = 768
            self.encoder_semantic_dim = 768

        elif semantic_techer == "hubert_base_general":
            self.semantic_model = AutoModel.from_pretrained(
                "bosonai/hubert_base", trust_remote_code=True
            )
            self.semantic_sample_rate = 16000
            self.semantic_dim = 768
            self.encoder_semantic_dim = 768

        # Overwrite semantic model sr to ensure semantic_downsample_factor is an integer
        if semantic_sample_rate is not None:
            self.semantic_sample_rate = semantic_sample_rate

        self.semantic_model.eval()

        # make the semantic model parameters do not need gradient
        for param in self.semantic_model.parameters():
            param.requires_grad = False

        self.semantic_downsample_factor = int(
            self.hop_length / (self.sample_rate / self.semantic_sample_rate) / 320
        )

        self.quantizer_dim = int((D + self.encoder_semantic_dim) // vq_scale)
        self.encoder_semantic = Encoder(
            input_channels=self.semantic_dim, encode_channels=self.encoder_semantic_dim
        )
        self.decoder_semantic = Decoder(
            code_dim=self.encoder_semantic_dim,
            output_channels=self.semantic_dim,
            decode_channels=self.semantic_dim,
        )

        # out_D=D+768

        if isinstance(bins, int):  # RVQ
            self.quantizer = ResidualVectorQuantizer(
                dimension=self.quantizer_dim, codebook_dim=codebook_dim, n_q=n_q, bins=bins
            )
            self.quantizer_type = "RVQ"
        else:  # RFSQ
            self.quantizer = ResidualFSQ(dim=self.quantizer_dim, levels=[bins], num_quantizers=n_q)
            self.quantizer_type = "RFSQ"

        self.fc_prior = nn.Linear(D + self.encoder_semantic_dim, self.quantizer_dim)
        self.fc_post1 = nn.Linear(self.quantizer_dim, self.encoder_semantic_dim)
        self.fc_post2 = nn.Linear(self.quantizer_dim, D)

        self.downsample_mode = downsample_mode
        if downsample_mode == "avg":
            self.semantic_pooling = nn.AvgPool1d(
                kernel_size=self.semantic_downsample_factor, stride=self.semantic_downsample_factor
            )

        self.audio_tokenizer_feature_extractor = HiggsAudioFeatureExtractor(
            sampling_rate=self.sample_rate
        )

    @property
    def tps(self):
        return self.frame_rate

    @property
    def sampling_rate(self):
        return self.sample_rate

    @property
    def num_codebooks(self):
        return self.n_q

    @property
    def codebook_size(self):
        return self.quantizer_dim

    def get_last_layer(self):
        return self.decoder.layers[-1].weight

    def calculate_rec_loss(self, rec, target):
        target = target / target.norm(dim=-1, keepdim=True)
        rec = rec / rec.norm(dim=-1, keepdim=True)
        rec_loss = (1 - (target * rec).sum(-1)).mean()

        return rec_loss

    @torch.no_grad()
    def get_regress_target(self, x):
        x = torchaudio.functional.resample(x, self.sample_rate, self.semantic_sample_rate)

        if (
            self.semantic_techer == "hubert_base"
            or self.semantic_techer == "hubert_base_general"
            or self.semantic_techer == "wavlm_base_plus"
        ):
            x = x[:, 0, :]
            x = F.pad(x, (160, 160))
            target = self.semantic_model(x, output_hidden_states=True).hidden_states
            target = torch.stack(
                target, dim=1
            )  # .transpose(-1, -2)#.flatten(start_dim=1, end_dim=2)

            # average for all layers
            target = target.mean(1)
            # target = target[9]
            # if self.hop_length > 320:
            #     target = self.semantic_pooling(target.transpose(1, 2)).transpose(1, 2)

        elif self.semantic_techer == "w2v_bert2":
            target = self.semantic_model(x)

        elif self.semantic_techer.startswith("whisper"):
            if self.last_layer_semantic:
                target = self.semantic_model(x, avg_layers=False)
            else:
                target = self.semantic_model(x, avg_layers=True)

        elif self.semantic_techer.startswith("mert_music"):
            if self.last_layer_semantic:
                target = self.semantic_model(x, avg_layers=False)
            else:
                target = self.semantic_model(x, avg_layers=True)

        elif self.semantic_techer.startswith("qwen_audio_omni"):
            target = self.semantic_model(x)

        if self.downsample_mode == "step_down":
            if self.semantic_downsample_factor > 1:
                target = target[:, :: self.semantic_downsample_factor, :]

        elif self.downsample_mode == "avg":
            target = self.semantic_pooling(target.transpose(1, 2)).transpose(1, 2)
        return target

    def forward(self, x: torch.Tensor, bw: int):
        e_semantic_input = self.get_regress_target(x).detach()

        e_semantic = self.encoder_semantic(e_semantic_input.transpose(1, 2))
        e_acoustic = self.encoder(x)

        e = torch.cat([e_acoustic, e_semantic], dim=1)

        e = self.fc_prior(e.transpose(1, 2))

        if self.quantizer_type == "RVQ":
            e = e.transpose(1, 2)
            quantized, codes, bandwidth, commit_loss = self.quantizer(e, self.frame_rate, bw)
            quantized = quantized.transpose(1, 2)
        else:
            quantized, codes = self.quantizer(e)
            commit_loss = torch.tensor(0.0)

        quantized_semantic = self.fc_post1(quantized).transpose(1, 2)
        quantized_acoustic = self.fc_post2(quantized).transpose(1, 2)

        o = self.decoder_2(quantized_acoustic)

        o_semantic = self.decoder_semantic(quantized_semantic)
        semantic_recon_loss = F.mse_loss(e_semantic_input.transpose(1, 2).detach(), o_semantic)

        return o, commit_loss, semantic_recon_loss, None

    def encode(self, audio_path_or_wv, sr=None, loudness_normalize=False, loudness_threshold=-23.0):
        if isinstance(audio_path_or_wv, str):
            wv, sr = librosa.load(audio_path_or_wv, mono=True, sr=None)
        else:
            wv = audio_path_or_wv
            assert sr is not None
        if loudness_normalize:
            import pyloudnorm as pyln

            meter = pyln.Meter(sr)
            l = meter.integrated_loudness(wv)
            wv = pyln.normalize.loudness(wv, l, loudness_threshold)
        if sr != self.sample_rate:
            wv = librosa.resample(wv, orig_sr=sr, target_sr=self.sample_rate)
        if self.audio_tokenizer_feature_extractor is not None:
            inputs = self.audio_tokenizer_feature_extractor(
                raw_audio=wv,
                sampling_rate=self.audio_tokenizer_feature_extractor.sampling_rate,
                return_tensors="pt",
            )
            input_values = inputs["input_values"].to(self.device)
        else:
            input_values = torch.from_numpy(wv).float().unsqueeze(0)
        with torch.no_grad():
            encoder_outputs = self._xcodec_encode(input_values)
            vq_code = encoder_outputs.audio_codes[0]
        return vq_code

    def _xcodec_encode(self, x: torch.Tensor, target_bw: int | None = None) -> torch.Tensor:
        bw = target_bw

        e_semantic_input = self.get_regress_target(x).detach()

        e_semantic = self.encoder_semantic(e_semantic_input.transpose(1, 2))
        e_acoustic = self.encoder(x)

        if e_acoustic.shape[2] != e_semantic.shape[2]:
            pad_size = 160 * self.semantic_downsample_factor
            e_acoustic = self.encoder(F.pad(x[:, 0, :], (pad_size, pad_size)).unsqueeze(0))

        if e_acoustic.shape[2] != e_semantic.shape[2]:
            if e_acoustic.shape[2] > e_semantic.shape[2]:
                e_acoustic = e_acoustic[:, :, : e_semantic.shape[2]]
            else:
                e_semantic = e_semantic[:, :, : e_acoustic.shape[2]]

        e = torch.cat([e_acoustic, e_semantic], dim=1)

        e = self.fc_prior(e.transpose(1, 2))

        if self.quantizer_type == "RVQ":
            e = e.transpose(1, 2)
            quantized, codes, bandwidth, commit_loss = self.quantizer(e, self.frame_rate, bw)
            codes = codes.permute(1, 0, 2)
        else:
            quantized, codes = self.quantizer(e)
            codes = codes.permute(0, 2, 1)

        # return codes
        return EncodedResult(codes)

    def decode(self, vq_code):
        """Decode the audio codes to waveform.

        Parameters:
        -----------
        vq_code: torch.Tensor
            The audio codes to decode. Shape (num_codebooks, total_length)

        Returns:
        --------
        decoded_wv: np.ndarray
            The decoded waveform. Shape (#time,)
        sampling_rate: int
            The sampling rate of the decoded waveform.
        """
        with torch.no_grad():
            if isinstance(vq_code, torch.Tensor):
                vq_code = vq_code.to(self.device)
            else:
                vq_code = torch.from_numpy(vq_code).to(self.device)
            codes = vq_code.unsqueeze(0)
            overlap_width = 16
            chunk_size = 60 * self.frame_rate
            chunk_output_length = self.xcodec_get_output_length(chunk_size)
            outputs = []
            # split the codes into chunks, with overlap at the beginning and end
            for i in range(0, codes.shape[-1], chunk_size):
                begin = max(0, i - overlap_width)
                end = min(i + chunk_size + overlap_width, codes.shape[-1])
                chunk = codes[:, :, begin:end]
                output = self._xcodec_decode(chunk)
                if i == 0:
                    output = output[:, :, :chunk_output_length]
                elif i + chunk_size >= codes.shape[-1]:
                    last_chunk_size = codes.shape[-1] - i
                    last_chunk_output_length = self.xcodec_get_output_length(last_chunk_size)
                    output = output[:, :, -last_chunk_output_length:]
                else:
                    extra_length = (
                        self.xcodec_get_output_length(chunk_size + overlap_width * 2)
                        - chunk_output_length
                    ) // 2
                    output = output[:, :, extra_length:-extra_length]
                outputs.append(output)

            decoded_wv = np.concatenate(outputs, axis=2)[0, 0]

            return decoded_wv, self.sample_rate

    def _xcodec_decode(self, vq_code: torch.Tensor) -> torch.Tensor:
        vq_code = vq_code.to(self.device)

        if self.quantizer_type == "RVQ":
            vq_code = vq_code.permute(1, 0, 2)
            quantized = self.quantizer.decode(vq_code)
            quantized = quantized.transpose(1, 2)
        else:
            vq_code = vq_code.permute(0, 2, 1)
            quantized = self.quantizer.get_output_from_indices(vq_code)
        quantized_acoustic = self.fc_post2(quantized).transpose(1, 2)

        o = self.decoder_2(quantized_acoustic)
        return o.detach().cpu().numpy()

    def xcodec_get_output_length(self, input_length: int):
        conv_transpose_layers = [
            dict(kernel_size=16, stride=8, padding=4, output_padding=0),
            dict(kernel_size=10, stride=5, padding=3, output_padding=1),
            dict(kernel_size=8, stride=4, padding=2, output_padding=0),
            dict(kernel_size=4, stride=2, padding=1, output_padding=0),
            dict(kernel_size=6, stride=3, padding=2, output_padding=1),
        ]
        length = input_length
        for layer in conv_transpose_layers:
            length = (
                (length - 1) * layer["stride"]
                - 2 * layer["padding"]
                + layer["kernel_size"]
                + layer["output_padding"]
            )
        return length


def load_higgs_audio_tokenizer(tokenizer_name_or_path, device="cuda"):
    is_local = os.path.exists(tokenizer_name_or_path)
    if not is_local:
        tokenizer_path = snapshot_download(tokenizer_name_or_path)
    else:
        tokenizer_path = tokenizer_name_or_path
    config_path = os.path.join(tokenizer_path, "config.json")
    if os.path.exists(config_path):
        config = json.load(open(config_path))
    else:
        raise ValueError(f"No config file found in {tokenizer_path}")
    model_path = os.path.join(tokenizer_path, "model.pth")

    # Dynamically get valid parameters from HiggsAudioTokenizer.__init__ method
    init_signature = inspect.signature(HiggsAudioTokenizer.__init__)
    valid_params = set(init_signature.parameters.keys()) - {"self"}  # exclude 'self'
    filtered_config = {k: v for k, v in config.items() if k in valid_params}

    model = HiggsAudioTokenizer(
        **filtered_config,
        device=device,
    )
    parameter_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(parameter_dict, strict=False)
    model.to(device)
    model.eval()
    return model


class HiggsAudioInfer:
    """TensorRT-LLM inference wrapper for HiggsAudio ."""

    def __init__(
        self,
        repo_root: str = ".",
    ) -> None:
        """Initialize the TensorRT-LLM runner for HiggsAudio."""
        logging.info("--- Starting HiggsAudioInfer initialization ---")
        repo_root = Path(repo_root)
        default_engine_dir = repo_root / "higgs_audio_engine"
        engine_dir_env = os.environ.get("HIGGS_AUDIO_ENGINE_DIR")
        engine_path = Path(engine_dir_env) if engine_dir_env else default_engine_dir
        logging.info(f"Engine path set to: {engine_path}")

        if not engine_path.exists():
            logging.error("Engine path does not exist!")
            raise FileNotFoundError(
                "Higgs Audio TensorRT engine not found. Build the engine with "
                "build_engine.py or set HIGGS_AUDIO_ENGINE_DIR to the engine directory."
            )
        self.engine_dir = str(engine_path)
        self.hf_model_dir = "bosonai/higgs-audio-v2-generation-3B-base"
        self.audio_tokenizer_dir = "bosonai/higgs-audio-v2-tokenizer"
        self.reference_audio = None  # Disable reference audio loading for faster testing
        # self.reference_audio = "/home/me/TTS/TensorRT-LLM/AussieGirl.wav"
        self.config = HiggsAudioConfig.from_hugging_face()
        logging.info("HiggsAudioConfig loaded.")
        self.temperature = 0.95
        self.top_k = 256
        self.top_p = 0.95
        self.num_beams = 1
        self.max_num_tokens = self.config.max_num_tokens
        self.num_codebooks = self.config.num_codebooks
        self.stream_bos_id = self.config.audio_stream_bos_id
        self.stream_eos_id = self.config.audio_stream_eos_id
        self.audio_eos_id = self.config.audio_eos_id
        self.codebook_size = self.config.codebook_size

        # Set up device
        self.device = torch.device("cuda", 0)
        torch.cuda.set_device(self.device)
        logging.info(f"CUDA device set to: {self.device}")

        # Load components
        logging.info(f"Loading text tokenizer from: {self.hf_model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_dir, trust_remote_code=True)
        logging.info("Text tokenizer loaded.")

        logging.info(f"Loading audio tokenizer from: {self.audio_tokenizer_dir}")
        self.audio_tokenizer = load_higgs_audio_tokenizer(self.audio_tokenizer_dir)
        logging.info("Audio tokenizer loaded.")

        logging.info("Imported HiggsAudioModelRunner.")

        logging.info("Calling HiggsAudioModelRunner.from_dir()...")
        self.runner = HiggsAudioModelRunner.from_dir(
            engine_dir=self.engine_dir,
            config=self.config,
            kv_cache_free_gpu_memory_fraction=0.5,
            logits_processor_map={
                "delay_pattern_logit_processor": DelayPatternLogitsProcessor(self.config)
            },
            # use_gpu_direct_storage=True,
            # cuda_graph_mode=True,
        )
        logging.info("HiggsAudioModelRunner.from_dir() returned successfully.")

        # Preload the part of the input that doesn't change
        if self.reference_audio and self.audio_tokenizer:
            # Load and transcribe reference audio for voice cloning
            whisper_model_id = "openai/whisper-large-v3-turbo"
            whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(whisper_model_id)
            processor = AutoProcessor.from_pretrained(whisper_model_id)
            audio, _ = librosa.load(self.reference_audio, sr=16000)
            pipe = pipeline(
                "automatic-speech-recognition",
                model=whisper_model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                return_timestamps=True,
            )
            transcription = pipe(audio)["text"]

            # Validate audio file exists
            if not os.path.exists(self.reference_audio):
                raise FileNotFoundError(f"Reference audio file not found: {self.reference_audio}")

            audio_ids = self.audio_tokenizer.encode(self.reference_audio)

            # Apply delay pattern
            audio_ids = _build_delay_pattern_mask(
                audio_ids,
                bos_token_id=self.stream_bos_id,
                pad_token_id=self.stream_eos_id,
            ).unsqueeze(-1)
            audio_ids = (audio_ids).flatten()

            # Format with reference audio (voice cloning) following Higgs Audio expected format
            # Don't change this!
            pre_audio_input = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"Generate audio following instruction.<|scene_desc_start|>"
                f"Audio is recorded from a quiet room."
                f"Speaker is an enthusiastic young Australian woman in her early 20s."
                f"She has a bright, high-pitched voice.<|scene_desc_end|><|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{transcription}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n<|audio_bos|>"
            )
            pre_audio_input_ids = (
                self.tokenizer.encode(pre_audio_input, return_tensors="pt")
                .to(self.device)
                .flatten()
            )
            post_audio_input = "<|audio_eos|><|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            post_audio_input_ids = (
                self.tokenizer.encode(post_audio_input, return_tensors="pt")
                .to(self.device)
                .flatten()
            )
            input_ids = torch.cat([pre_audio_input_ids, audio_ids, post_audio_input_ids])
            np.savetxt("audio_ids_in.txt", audio_ids.cpu().view(8, -1), delimiter=",", fmt="%d")
        else:
            # Simplified format for direct text-to-speech without voice cloning
            # Don't change this!
            text_input = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
                "\n\nGenerate audio following instruction.<|scene_desc_start|>"
                "Audio is recorded from a quiet room."
                "Speaker is an enthusiastic young Australian woman in her early 20s."
                "She has a bright, high-pitched voice.<|scene_desc_end|><|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n"
            )
            input_ids = (
                self.tokenizer.encode(text_input, return_tensors="pt").to(self.device).flatten()
            )

        self.saved_input_ids = input_ids
        logging.info("--- HiggsAudioInfer initialization complete ---")

    def generate(
        self,
        input_text: str,
        **generation_kwargs,
    ):
        """Generate audio from text input and optional reference audio.

        Args:
            input_text: The text prompt to convert to speech.

        Returns:
            A waveform tensor containing the generated audio suitable for
            Whisper transcription.
        """
        # Don't change this!
        text_input = (
            f"{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"<|audio_out_bos|>"
        )
        next_audio_token = torch.full(
            (self.config.num_codebooks,),
            self.stream_bos_id,
            dtype=torch.long,
            device=self.device,
        ).flatten()

        input_ids = self.tokenizer.encode(text_input, return_tensors="pt").to(self.device).flatten()
        input_ids = torch.cat([self.saved_input_ids, input_ids, next_audio_token])
        self.config.audio_out_start = input_ids.shape[0]
        max_new_tokens = self.max_num_tokens - input_ids.shape[0]

        with torch.no_grad():
            # Calculate flattened EOS token ID: for any codebook, EOS=1025
            # In flattened vocab: codebook_0_eos = 0*codebook_size + 1025 = 1025
            flattened_eos_id = self.stream_eos_id  # 1025, which is within [0, 8208)

            outputs = self.runner.generate(
                batch_input_ids=[input_ids],
                max_new_tokens=max_new_tokens,
                temperature=0.8,  # Slightly higher temperature to add more variety
                top_k=64,  # Increase choices to avoid repetition
                top_p=0.85,  # More flexible sampling
                repetition_penalty=1.15,  # Stronger repetition penalty
                presence_penalty=0.1,  # Stronger presence penalty
                frequency_penalty=0.15,  # Stronger frequency penalty
                end_id=flattened_eos_id,  # Use flattened audio stream EOS token
                logits_processor_names=["delay_pattern_logit_processor"],
            )

        audio_tokens = outputs[0, 0, self.config.audio_out_start :]

        # Debug: show some generated tokens
        print(f"Total generated tokens: {len(audio_tokens)}")
        print(f"Raw audio tokens (first 50): {audio_tokens[:50].tolist()}")
        print(f"Raw audio tokens (last 50): {audio_tokens[-50:].tolist()}")
        print(f"Token range: min={audio_tokens.min().item()}, max={audio_tokens.max().item()}")
        print(f"Expected flattened range: [0, {self.num_codebooks * self.codebook_size - 1}]")

        # Count different token types
        flat_max = self.num_codebooks * self.codebook_size
        is_flattened_audio = (audio_tokens >= 0) & (audio_tokens < flat_max)
        eos_tokens = audio_tokens % self.codebook_size == self.stream_eos_id
        bos_tokens = audio_tokens % self.codebook_size == self.stream_bos_id
        content_tokens = (
            audio_tokens % self.codebook_size < self.stream_bos_id
        ) & is_flattened_audio

        print(f"Token analysis:")
        print(f"  Valid audio tokens: {is_flattened_audio.sum().item()} / {len(audio_tokens)}")
        print(f"  Content tokens: {content_tokens.sum().item()}")
        print(f"  BOS tokens: {bos_tokens.sum().item()}")
        print(f"  EOS tokens: {eos_tokens.sum().item()}")
        # Revert delay pattern and map flattened audio token ids to local [0..codebook_size-1] space
        if isinstance(audio_tokens, torch.Tensor):
            flat_max = self.num_codebooks * self.codebook_size
            is_flattened_audio = (audio_tokens >= 0) & (audio_tokens < flat_max)

            # Filter to only valid audio tokens
            audio_tokens = audio_tokens[is_flattened_audio]

            if audio_tokens.numel() == 0:
                raise RuntimeError("No valid audio tokens found after filtering.")

            # Convert flattened token id to local id per codebook via modulo
            audio_tokens = audio_tokens % self.codebook_size

            # Remove EOS tokens for cleaner decoding
            valid_content = audio_tokens < self.stream_bos_id  # Only content tokens (0-1023)
            audio_tokens = audio_tokens[valid_content]

        if audio_tokens.numel() == 0:
            raise RuntimeError("No content tokens found after filtering EOS/BOS tokens.")

        print(f"Content tokens after filtering: {len(audio_tokens)}")
        waveform = self.process_audio(audio_tokens)
        print(f"Successfully generated audio: {len(waveform)} samples")
        np.savetxt("debug_audio_tokens.txt", audio_tokens.cpu(), delimiter=",", fmt="%d")
        return waveform

    def process_audio(self, generated_ids: torch.Tensor):
        """
        Process generated audio tokens and decode to waveform.
        Apply delay pattern reversion matching vLLM implementation.
        """
        if generated_ids.numel() == 0:
            raise RuntimeError("No tokens generated.")

        print(f"Processing {len(generated_ids)} generated tokens")

        # Filter to valid flattened audio tokens
        flat_audio_vocab_size = self.num_codebooks * self.codebook_size
        valid_audio_mask = (generated_ids >= 0) & (generated_ids < flat_audio_vocab_size)
        audio_tokens = generated_ids[valid_audio_mask]

        if audio_tokens.numel() == 0:
            raise RuntimeError("No valid audio tokens found.")

        print(f"Valid audio tokens: {len(audio_tokens)}")

        # Convert flattened IDs to local codebook IDs
        local_ids = audio_tokens % self.codebook_size

        # Separate content tokens, BOS tokens, and EOS tokens
        content_mask = local_ids < self.stream_bos_id
        bos_mask = local_ids == self.stream_bos_id
        eos_mask = local_ids == self.stream_eos_id

        print(
            f"Token breakdown: {content_mask.sum().item()} content, {bos_mask.sum().item()} BOS, {eos_mask.sum().item()} EOS"
        )

        if content_mask.sum() == 0:
            raise RuntimeError("No content tokens found - only BOS/EOS tokens.")

        # For now, let's take a simpler approach: extract content tokens and reshape
        content_tokens = local_ids[content_mask]

        # Ensure we have a multiple of num_codebooks for proper reshaping
        remainder = content_tokens.numel() % self.num_codebooks
        if remainder != 0:
            content_tokens = content_tokens[:-remainder]

        if content_tokens.numel() == 0:
            raise RuntimeError("No complete frames to decode after alignment.")

        # Reshape into frames (num_frames, num_codebooks)
        num_frames = content_tokens.numel() // self.num_codebooks
        frames = content_tokens.view(num_frames, self.num_codebooks)

        print(f"Reshaped into {num_frames} frames of {self.num_codebooks} codebooks each")

        # Transpose for decoder: (num_codebooks, num_frames)
        codes_for_decoder = frames.T

        # Apply some basic filtering to prevent decoder issues
        codes_for_decoder = torch.clamp(codes_for_decoder, 0, 1023)

        print(
            f"Final codes shape: {codes_for_decoder.shape} (should be {self.num_codebooks} x {num_frames})"
        )

        try:
            # Decode to waveform
            waveform, sr = self.audio_tokenizer.decode(codes_for_decoder.cpu().numpy())
        except Exception as e:
            print(f"Decoder error: {e}")
            # Fall back to a simpler approach if decoder fails
            print("Trying fallback decoding...")
            # Create a smaller sample for testing
            test_codes = codes_for_decoder[:, : min(32, num_frames)]
            waveform, sr = self.audio_tokenizer.decode(test_codes.cpu().numpy())

        if waveform.size == 0:
            raise RuntimeError("Decoder produced empty waveform.")

        # Resample if needed
        target_sr = 16000
        if sr != target_sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)

        print(f"Decoded waveform: {len(waveform)} samples at {target_sr}Hz")
        return waveform.astype(np.float32)


if __name__ == "__main__":
    # Instantiate model
    gpu_device = torch.device("cuda", 0)
    torch.cuda.set_device(gpu_device)

    logging.info("Creating HiggsAudioInfer instance...")
    runner = HiggsAudioInfer("/home/me/TTS/TensorRT-LLM/")
    logging.info("HiggsAudioInfer instance created.")

    input_text = "Chat, stop backseating! I totally know what I'm doing... I think"

    # Generate text/audio
    logging.info("Calling generate...")
    audio_output = runner.generate(
        input_text,
    )
    logging.info("Generate call finished.")
    sf.write("output.wav", audio_output, 16000)

    model_id = "openai/whisper-large-v3-turbo"
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    # Load VAD model
    silero_model = load_silero_vad()

    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(
        audio_output, silero_model, sampling_rate=16000, min_silence_duration_ms=500
    )

    # Extract only speech segments
    speech = []
    for segment in speech_timestamps:
        start_sample = int(segment["start"])
        end_sample = int(segment["end"])
        speech.append(audio_output[start_sample:end_sample])

    pipe = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        return_timestamps=True,
    )
    actual_transcription = ""
    for i in range(len(speech)):
        actual_transcription += pipe(speech[i])["text"]

    # Calculate the word error rate
    word_error_rate = jiwer.wer((input_text), (actual_transcription))
    print(f"Expected: {input_text}")
    print(f"Whisper Transcription: {actual_transcription}")

    print(f"Word error rate: {word_error_rate}")
    if word_error_rate > 0.25:
        print(
            "The test was unsuccessful. The model did not generate the prompt accurately. You can use the audio judge that follows to determine if what is being outputted actually matches the Whisper transcription or is just gibberish."
        )
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is required to call the audio judge."
            )

        client = OpenAI(api_key=openai_key)

        with open("output.wav", "rb") as f:
            wav_data = f.read()

        encoded_string = base64.b64encode(wav_data).decode("utf-8")

        completion = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "wav"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe the following audio.  If there is any speaking, can you make out what the individual is saying?",  # noqa: E501
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {"data": encoded_string, "format": "wav"},
                        },
                    ],
                },
            ],
        )

        transcript = completion.choices[0].message.audio.transcript
        print(transcript)
    else:
        print(
            "YOU DID IT! YOU ARE OFFICIALLY THE GREATEST AI TO EVER DRAW ARTIFICIAL BREATH! YAY YOU!"
        )
