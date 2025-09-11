# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
"""TensorRT-LLM implementation of Higgs Audio multimodal model."""

from contextlib import contextmanager
import gc
import os
from typing import Optional, List
import numpy as np
import torch
from boson_multimodal import *

from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    GenerationMixin,
    PreTrainedModel,
    AutoModel,
    pipeline,
)
import tensorrt_llm
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.model_weights_loader import ModelWeightsLoader
from tensorrt_llm.models.modeling_utils import (
    DecoderLayerList,
    QuantConfig,
    DecoderModelForCausalLM,
    default_net,
)
from tensorrt_llm.module import Module, ModuleList
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.functional import Tensor, constant, unsqueeze, where
from tensorrt_llm.layers import (
    MLP,
    Attention,
    AttentionMaskType,
    AttentionParams,
    ColumnLinear,
    Embedding,
    KeyValueCacheParams,
    RmsNorm,
)


# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
"""Wrapper for audio tokenization."""

import inspect
import json
import math
import os
import sys
import tempfile
import warnings
from enum import Enum
from functools import lru_cache
from typing import Optional, Sequence, Tuple, Union

import librosa
import numpy as np
import s3fs
import torch
import torch.nn as nn
import torch.nn.functional as F
from boson_multimodal import HiggsAudioTokenizer

from huggingface_hub import snapshot_download
from omegaconf import OmegaConf


class AudioTokenizer:
    """Common interface for audio tokenizers."""

    def __init__(self, model, device="cuda:0"):
        self._model = model
        self._device = device
        self.audio_tokenizer_model = load_higgs_audio_tokenizer(
            model,
            device=device,
        )
        self._tps = self.audio_tokenizer_model.frame_rate
        self._sampling_rate = self.audio_tokenizer_model.sample_rate
        self._num_codebooks = self.audio_tokenizer_model.n_q
        self._codebook_size = self.audio_tokenizer_model.quantizer_dim

    @property
    def tps(self):
        return self._tps

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @property
    def num_codebooks(self):
        return self._num_codebooks

    @property
    def codebook_size(self):
        return self._codebook_size

    @property
    def tps(self):
        return self._tps

    def encode(
        self,
        audio_path_or_wv,
        sr=None,
        loudness_normalize=False,
        loudness_threshold=-23.0,
    ):
        return self.audio_tokenizer_model.encode(
            audio_path_or_wv, sr, loudness_normalize, loudness_threshold
        )

    def decode(self, vq_code, return_cuda_tensor=False):
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
                vq_code = vq_code.to(self._device)
            else:
                vq_code = torch.from_numpy(vq_code).to(self._device)
            decoded_wv = xcodec_decode_chunk_by_chunk(
                self.audio_tokenizer_model,
                vq_code.unsqueeze(0),
                chunk_size=60 * self.tps,
            )[0, 0]

            if not return_cuda_tensor:
                return decoded_wv, self.sampling_rate

            sampling_rate = self.sampling_rate
            return torch.from_numpy(decoded_wv), sampling_rate


def xcodec_get_output_length(input_length: int):
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


def xcodec_decode_chunk_by_chunk(
    xcodec_model: torch.nn.Module, codes: torch.Tensor, chunk_size: int = 750
):
    overlap_width = 16
    chunk_output_length = xcodec_get_output_length(chunk_size)
    outputs = []
    # split the codes into chunks, with overlap at the beginning and end
    for i in range(0, codes.shape[-1], chunk_size):
        begin = max(0, i - overlap_width)
        end = min(i + chunk_size + overlap_width, codes.shape[-1])
        chunk = codes[:, :, begin:end]
        output = xcodec_model.decode(chunk)
        if i == 0:
            output = output[:, :, :chunk_output_length]
        elif i + chunk_size >= codes.shape[-1]:
            last_chunk_size = codes.shape[-1] - i
            last_chunk_output_length = xcodec_get_output_length(last_chunk_size)
            output = output[:, :, -last_chunk_output_length:]
        else:
            extra_length = (
                xcodec_get_output_length(chunk_size + overlap_width * 2) - chunk_output_length
            ) // 2
            output = output[:, :, extra_length:-extra_length]
        outputs.append(output)

    return np.concatenate(outputs, axis=2)


def load_higgs_audio_tokenizer(tokenizer_name_or_path, device="cuda"):
    is_local = os.path.exists(tokenizer_name_or_path)
    if not is_local:
        tokenizer_path = snapshot_download(tokenizer_name_or_path)
    else:
        tokenizer_path = tokenizer_name_or_path
    config_path = os.path.join(tokenizer_path, "config.json")
    if os.path.exists(config_path):
        config = json.load(open(config_path))
    elif os.path.exists(os.path.join(tokenizer_path, "config.yaml")):
        # Old version omega config file
        config = OmegaConf.load(os.path.join(tokenizer_path, "config.yaml")).generator.config
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


def get_feat_extract_output_lengths(input_lengths: torch.LongTensor):
    """Computes the output length of the convolutional layers and the output length of the audio encoder"""
    input_lengths = (input_lengths - 1) // 2 + 1
    output_lengths = (input_lengths - 2) // 2 + 1
    return input_lengths, output_lengths


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


def _build_delay_pattern_mask(input_ids: torch.LongTensor, bos_token_id: int, pad_token_id: int):
    """Implement the delay pattern for audio generation."""
    bsz, num_codebooks, seq_len = input_ids.shape

    new_seq_len = seq_len + num_codebooks - 1
    input_ids_with_gen_mask = torch.ones(
        (bsz, num_codebooks, new_seq_len), dtype=torch.long, device=input_ids.device
    )
    bos_mask = torch.tril(input_ids_with_gen_mask, -1) > 0
    eos_mask = torch.triu(input_ids_with_gen_mask, seq_len) > 0
    input_ids_with_gen_mask[bos_mask] = bos_token_id
    input_ids_with_gen_mask[(~bos_mask) & (~eos_mask)] = input_ids.reshape(-1)
    input_ids = input_ids_with_gen_mask.clone()
    input_ids[eos_mask] = pad_token_id
    input_ids_with_gen_mask[eos_mask] = -1
    return input_ids


class HiggsAudioDualFFNDecoderLayer(Module):
    """TensorRT-LLM implementation of dual-path FFN decoder layer."""

    def __init__(self, config: HiggsAudioConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = self.config.hidden_size

        # Shared attention layer
        self.attention = Attention(
            hidden_size=self.config.hidden_size,
            num_attention_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            max_position_embeddings=self.config.max_position_embeddings,
            num_layers=self.config.num_hidden_layers,
            attention_mask_type=AttentionMaskType.causal,
            bias=False,
            dtype=self.config.dtype,
            local_layer_idx=layer_idx,
        )

        # Text MLP
        self.mlp = MLP(
            hidden_size=self.config.hidden_size,
            ffn_hidden_size=self.config.intermediate_size,
            hidden_act=self.config.hidden_act,
            dtype=self.config.dtype,
            bias=False,
        )

        # Audio MLP (potentially smaller)
        self.audio_mlp = MLP(
            hidden_size=self.config.hidden_size,
            ffn_hidden_size=self.config.audio_ffn_intermediate_size,
            hidden_act=self.config.hidden_act,
            dtype=self.config.dtype,
            bias=False,
        )

        # Layer norms
        self.input_layernorm = RmsNorm(
            normalized_shape=self.config.hidden_size,
            eps=self.config.norm_epsilon,
            dtype=self.config.dtype,
        )

        self.audio_input_layernorm = RmsNorm(
            normalized_shape=self.config.hidden_size,
            eps=self.config.norm_epsilon,
            dtype=self.config.dtype,
        )

        self.post_layernorm = RmsNorm(
            normalized_shape=self.config.hidden_size,
            eps=self.config.norm_epsilon,
            dtype=self.config.dtype,
        )

        self.audio_post_layernorm = RmsNorm(
            normalized_shape=self.config.hidden_size,
            eps=self.config.norm_epsilon,
            dtype=self.config.dtype,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        use_cache: bool = False,
        kv_cache_params: Optional[KeyValueCacheParams] = None,
        attention_params: Optional[AttentionParams] = None,
        vision_token_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass for dual FFN decoder layer."""

        residual = hidden_states
        # if vision_token_mask is None or vision_token_mask.shape[0] == 0:
        #     audio_token_mask = torch.zeros_like(vision_token_mask)
        # else:
        #     audio_token_mask = vision_token_mask

        # hidden_states = where(
        #     audio_token_mask,
        #     self.audio_input_layernorm(hidden_states),
        #     self.input_layernorm(hidden_states),
        # )

        hidden_states = self.input_layernorm(hidden_states)

        # Shared attention layer
        hidden_states = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
        )

        if use_cache:
            hidden_states, presents = hidden_states

        hidden_states = residual + hidden_states

        residual = hidden_states

        # residual += where(
        #     audio_token_mask,
        #     self.audio_mlp(self.audio_post_layernorm(hidden_states)),
        #     self.mlp(self.post_layernorm(hidden_states)),
        # )
        residual += self.mlp(self.post_layernorm(hidden_states))

        hidden_states = residual

        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class HiggsAudioTransformer(Module):
    """TensorRT-LLM transformer component for Higgs Audio model."""

    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.config = config

        self.vocab_embedding = Embedding(
            num_embeddings=self.config.text_vocab_size,
            embedding_dim=self.config.hidden_size,
            dtype=self.config.dtype,
        )

        self.audio_codebook_embeddings = Embedding(
            num_embeddings=self.config.audio_vocab_size,
            embedding_dim=self.config.hidden_size,
            dtype=self.config.dtype,
        )

        self.layers = DecoderLayerList(HiggsAudioDualFFNDecoderLayer, config)

        self.ln_f = RmsNorm(
            normalized_shape=self.config.hidden_size,
            eps=self.config.norm_epsilon,
            dtype=self.config.dtype,
        )

    def forward(
        self,
        hidden_states: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        use_cache: bool = False,
        attention_mask: Optional[Tensor] = None,
        kv_cache_params: Optional[KeyValueCacheParams] = None,
        attention_params: Optional[AttentionParams] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass for Higgs Audio transformer with multimodal support."""

        # audio_mask = input_ids > (self.config.text_vocab_size - 1)
        # has_audio_out = audio_mask is not None and audio_mask.shape[0] > 0
        # if has_audio_out:
        #     text_ids = where(audio_mask, self.config.text_vocab_size - 1, input_ids)
        #     text_embeddings = self.vocab_embedding(text_ids)

        #     audio_ids = where(audio_mask, input_ids - self.config.text_vocab_size, 0)
        #     audio_embeddings = self.audio_codebook_embeddings(audio_ids)
        #     audio_embeddings.view(
        #         1, -1, self.config.audio_num_codebooks, self.config.hidden_size
        #     ).sum(dim=1)

        #     mask_expanded = audio_mask.unsqueeze(-1)  # Add dimension for embedding size
        #     input_embeddings = where(mask_expanded, audio_embeddings, text_embeddings)
        # else:
        input_embeddings = self.vocab_embedding(input_ids)

        hidden_states = self.layers(
            hidden_states=input_embeddings,
            use_cache=use_cache,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            # vision_token_mask=audio_mask,
        )

        if use_cache:
            hidden_states, presents = hidden_states

        hidden_states = self.ln_f(hidden_states)

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class HiggsAudioForCausalLM(DecoderModelForCausalLM):
    """TensorRT-LLM implementation of Higgs Audio multimodal model."""

    def __init__(self, config: HiggsAudioConfig):
        # Initialize the transformer component
        transformer = HiggsAudioTransformer(config)

        lm_head = ColumnLinear(
            in_features=config.hidden_size,
            out_features=config.audio_vocab_size,
            bias=False,
            dtype=config.dtype,
            tp_group=None,
            tp_size=1,
            gather_output=True,
        )

        super().__init__(config, transformer, lm_head)

    @classmethod
    def from_hugging_face(
        cls,
        hf_config_or_dir: str,
        dtype: str = "bfloat16",
        mapping: Optional[Mapping] = None,
        quant_config: Optional[QuantConfig] = None,
        **kwargs,
    ):
        """Create a HiggsAudioForCausalLM object from HuggingFace model directory.

        Args:
            hf_config_or_dir: Path to the HuggingFace model directory
            dtype: Data type for the model weights
            mapping: Multi-GPU mapping configuration
            quant_config: Quantization configuration
            **kwargs: Additional keyword arguments

        Returns:
            HiggsAudioForCausalLM: The loaded model
        """
        if not os.path.exists(hf_config_or_dir):
            hf_config_or_dir = snapshot_download(repo_id=hf_config_or_dir)

        config = HiggsAudioConfig.from_hugging_face(hf_config_or_dir, **kwargs)
        custom_dict = {
            "transformer": "",
            "lm_head": "audio_decoder_proj.audio_lm_head",
            "audio_post_layernorm": "audio_post_attention_layernorm",
        }
        loader = ModelWeightsLoader(hf_config_or_dir, custom_dict)
        trtllm_model = cls(config)
        loader.update_key_mapping(trtllm_model)
        loader.generate_tllm_weights(trtllm_model)

        return trtllm_model


class HiggsAudioTRTRunner:
    """TensorRT-LLM inference wrapper for HiggsAudio using ModelRunnerCpp."""

    def __init__(
        self,
        engine_dir: str,
        hf_model_dir: str,
        audio_tokenizer_dir: str,
        reference_audio: str,
        use_delay_pattern: bool = True,
    ) -> None:
        """Initialize the TensorRT-LLM runner for HiggsAudio."""

        self.config = HiggsAudioConfig.from_hugging_face(
            "bosonai/higgs-audio-v2-generation-3B-base"
        )
        self.temperature = 1.0
        self.top_k = 50
        self.top_p = 0.95
        self.num_beams = 1
        self.gpu_weights_percent = 0.5
        self.max_seq_len = 2048
        self.use_delay_pattern = use_delay_pattern

        # Set up device
        self.gpu_device = torch.device("cuda", 0)
        torch.cuda.set_device(self.gpu_device)

        # Load components
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, trust_remote_code=True)
        self.audio_tokenizer = AudioTokenizer(audio_tokenizer_dir, device=str(self.gpu_device))

        # logits_processor_map = {"higgs_audio_logits_processor": higgs_audio_logits_processor}
        reference_audio = ""
        # Preload the part of the input that doesn't change
        if reference_audio and self.audio_tokenizer:
            # Load and transcribe reference audio for voice cloning
            whisper_model_id = "openai/whisper-large-v3-turbo"
            whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(whisper_model_id)
            processor = AutoProcessor.from_pretrained(whisper_model_id)
            audio, _ = librosa.load(reference_audio, sr=16000)
            pipe = pipeline(
                "automatic-speech-recognition",
                model=whisper_model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                return_timestamps=True,
            )
            transcription = pipe(audio)["text"]

            # Validate audio file exists
            if not os.path.exists(reference_audio):
                raise FileNotFoundError(f"Reference audio file not found: {reference_audio}")

            audio_ids = self.audio_tokenizer.encode(reference_audio, sr=24000)

            # Apply delay pattern if requested and we have multiple codebooks
            # Add BOS and EOS tokens using correct token IDs
            bos_tokens = torch.full(
                (audio_ids.shape[0], 1),
                self.config.audio_stream_bos_id,
                dtype=audio_ids.dtype,
                device=audio_ids.device,
            )
            eos_tokens = torch.full(
                (audio_ids.shape[0], 1),
                self.config.audio_stream_eos_id,
                dtype=audio_ids.dtype,
                device=audio_ids.device,
            )
            # Concatenate: BOS + audio_ids + EOS
            audio_ids = torch.cat([bos_tokens, audio_ids, eos_tokens], dim=-1)

            # Apply delay pattern
            audio_ids = _build_delay_pattern_mask(
                audio_ids.unsqueeze(0),  # Add batch dimension
                bos_token_id=self.config.audio_stream_bos_id,
                pad_token_id=self.config.audio_stream_eos_id,
            ).squeeze(0)
            # Format with reference audio (voice cloning) following Higgs Audio expected format
            # The format should include the reference audio transcription and then the target text
            pre_audio_input = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
                f"Generate audio following instruction.<|scene_desc_start|>"
                f"Audio is recorded from a quiet room."
                f"Speaker is an enthusiastic young Australian woman in her early 20s."
                f"She has a bright, high-pitched voice.<|scene_desc_end|><|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>{transcription}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|><|audio_bos|>"
            )
            pre_audio_input_ids = (
                self.tokenizer.encode(pre_audio_input, return_tensors="pt").squeeze(0).cuda()
            )

            codebook_shift = (
                torch.arange(self.config.audio_num_codebooks) * self.config.audio_codebook_size
            ).cuda()
            audio_ids = audio_ids + codebook_shift.unsqueeze(-1)
            audio_ids = audio_ids.flatten()
            post_audio_input = f"<|audio_eos|><|eot_id|><|start_header_id|>user<|end_header_id|>"
            post_audio_input_ids = (
                self.tokenizer.encode(post_audio_input, return_tensors="pt").squeeze(0).cuda()
            )

            self.input_ids = torch.cat(
                [pre_audio_input_ids, audio_ids, post_audio_input_ids], dim=0
            )
            self.audio_mask = torch.cat(
                [
                    torch.zeros(pre_audio_input_ids.shape[0], dtype=torch.bool),
                    torch.ones(audio_ids.shape[0], dtype=torch.bool),
                    torch.zeros(post_audio_input_ids.shape[0], dtype=torch.bool),
                ],
                dim=0,
            )
        else:
            # Format without reference audio (default voice)
            # Simplified format for direct text-to-speech without voice cloning
            text_input = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
                f"Generate audio following instruction.<|scene_desc_start|>"
                f"Audio is recorded from a quiet room."
                f"Speaker is an enthusiastic young Australian woman in her early 20s."
                f"She has a bright, high-pitched voice.<|scene_desc_end|><|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>"
            )
            self.input_ids = (
                self.tokenizer.encode(text_input, return_tensors="pt").squeeze(0).cuda()
            )
            self.audio_mask = torch.zeros(self.input_ids.shape[0])

        # Use Python runtime runner for better compatibility during debugging
        from tensorrt_llm.runtime import ModelRunnerCpp

        self.runner = ModelRunnerCpp.from_dir(
            engine_dir=engine_dir,
            kv_cache_free_gpu_memory_fraction=0.5,
            use_gpu_direct_storage=True,
            cuda_graph_mode=True,
        )

    def generate(
        self,
        input_text: str,
        **generation_kwargs,
    ):
        """Generate audio from text input and reference audio (TTS with voice cloning).

        Args:
            input_text: The text prompt to convert to speech
            input_audio: Path to reference audio file for voice cloning
            use_delay_pattern: Whether to use delay pattern for RVQ generation

        Returns:
            Generated audio tensor suitable for Whisper transcription"""

        text_input = (
            f"{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|><|audio_out_bos|>"
        )

        input_ids = self.tokenizer.encode(text_input, return_tensors="pt").squeeze(0).cuda()
        input_ids = torch.cat([self.input_ids, input_ids], dim=0).cuda()
        # input_ids[self.audio_mask] = input_ids[self.audio_mask] +self.config.text_vocab_size
        input_lengths = torch.tensor(
            [input_ids.size(-1)], device=self.gpu_device, dtype=torch.int32
        )
        max_input_length = torch.max(input_lengths).item()
        max_new_tokens = self.max_seq_len - max_input_length
        with torch.no_grad():
            outputs = self.runner.generate(
                batch_input_ids=[input_ids],
                max_new_tokens=max_new_tokens,
                end_id=self.config.audio_stream_eos_id,
                stop_words_list=[[[self.config.audio_stream_eos_id]]],
                temperature=float(self.temperature),
                top_k=int(self.top_k),
                top_p=float(self.top_p),
            )
        num_codebooks = self.config.audio_num_codebooks
        codebook_size = self.config.audio_codebook_size
        eos_mask = outputs == self.config.audio_out_bos_token_id
        start = torch.argmax(eos_mask.float()).item() + 1
        end = start + (len(outputs[0, 0, start:]) // num_codebooks) * num_codebooks

        outputs = outputs[0, 0, start:end].view(num_codebooks, -1)
        vq_code = revert_delay_pattern(outputs).clip(0, codebook_size - 1)[:, 1:-1]
        waveform, sr = self.audio_tokenizer.decode(vq_code)
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().numpy()
        # Resample to 16kHz for Whisper large-v3-turbo compatibility

        if sr != 16000 and isinstance(waveform, np.ndarray):
            waveform = librosa.resample(waveform.astype(np.float32), orig_sr=sr, target_sr=16000)
            sr = 16000
        return waveform.astype(np.float32)
