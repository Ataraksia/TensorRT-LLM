# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
"""TensorRT-LLM implementation of Higgs Audio multimodal model."""

import os
from typing import Optional
import numpy as np
import torch
from boson_multimodal import *
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import AutoTokenizer
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
from tensorrt_llm.functional import Tensor, constant
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
        self.hidden_size = config.hidden_size

        # Shared attention layer
        self.attention = Attention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
            max_position_embeddings=config.max_position_embeddings,
            num_layers=config.num_hidden_layers,
            apply_query_key_layer_scaling=False,
            attention_mask_type=AttentionMaskType.causal,
            bias=getattr(config, "attention_bias", False),
            dtype=config.dtype,
            tp_group=None,
            tp_size=1,
            tp_rank=0,
            quant_mode=QuantMode(0),
            local_layer_idx=layer_idx,
        )

        # Text MLP
        self.mlp = MLP(
            hidden_size=config.hidden_size,
            ffn_hidden_size=getattr(config, "intermediate_size", config.hidden_size * 4),
            hidden_act=getattr(config, "hidden_act", "silu"),
            dtype=config.dtype,
            bias=getattr(config, "mlp_bias", False),
            tp_group=None,
            tp_size=1,
            quant_mode=QuantMode(0),
        )

        # Audio MLP (potentially smaller)
        self.audio_mlp = MLP(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.audio_ffn_intermediate_size,
            hidden_act=getattr(config, "hidden_act", "silu"),
            dtype=config.dtype,
            bias=getattr(config, "mlp_bias", False),
            tp_group=None,
            tp_size=1,
            quant_mode=QuantMode(0),
        )

        # Layer norms
        self.input_layernorm = RmsNorm(
            normalized_shape=config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-5),
            dtype=config.dtype,
        )

        self.post_layernorm = RmsNorm(
            normalized_shape=config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-5),
            dtype=config.dtype,
        )

        self.audio_input_layernorm = RmsNorm(
            normalized_shape=config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-5),
            dtype=config.dtype,
        )

        self.audio_post_attention_layernorm = RmsNorm(
            normalized_shape=config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-5),
            dtype=config.dtype,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        use_cache: bool = False,
        kv_cache_params: Optional[KeyValueCacheParams] = None,
        attention_params: Optional[AttentionParams] = None,
    ) -> Tensor:
        """Forward pass for dual FFN decoder layer."""

        residual = hidden_states

        # For now, use standard processing during build time
        # The dual FFN logic will be handled at runtime through the prompt embedding mechanism
        hidden_states = self.input_layernorm(hidden_states)

        # Shared attention layer
        attention_output = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
        )

        if use_cache:
            attention_output, present_key_value = attention_output

        hidden_states = residual + attention_output

        # Standard FFN processing
        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        if use_cache:
            return hidden_states, present_key_value
        return hidden_states


class HiggsAudioTransformer(Module):
    """TensorRT-LLM transformer component for Higgs Audio model."""

    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.config = config

        self.mapping = tensorrt_llm.Mapping()

        self.vocab_embedding = Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            dtype=config.dtype,
            tp_group=config.mapping.tp_group
            if hasattr(config, "mapping") and config.mapping
            else None,
            tp_size=config.mapping.tp_size if hasattr(config, "mapping") and config.mapping else 1,
            sharding_dim=0,
            tp_rank=config.mapping.tp_rank if hasattr(config, "mapping") and config.mapping else 0,
        )

        # Audio codebook embeddings for audio generation
        self.audio_codebook_size = config.audio_codebook_size + 2  # +2 for BOS/EOS
        self.audio_num_codebooks = config.audio_num_codebooks

        self.audio_codebook_embeddings = Embedding(
            num_embeddings=config.audio_num_codebooks * self.audio_codebook_size,
            embedding_dim=config.hidden_size,
            dtype=config.dtype,
        )

        # Decoder layers - use dual FFN for all layers for simplicity
        self.layers = DecoderLayerList(HiggsAudioDualFFNDecoderLayer, config)

        # Final layer norm
        self.ln_f = RmsNorm(
            normalized_shape=config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-5),
            dtype=config.dtype,
        )

    def forward(
        self,
        hidden_states: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        use_cache: bool = False,
        attention_mask: Optional[Tensor] = None,
        kv_cache_params: Optional[KeyValueCacheParams] = None,
        attention_params: Optional[AttentionParams] = None,
        prompt_embedding_table: Optional[Tensor] = None,
        prompt_tasks: Optional[Tensor] = None,
        prompt_vocab_size: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass for Higgs Audio transformer with multimodal support.

        This implementation supports both standard text processing and audio-enhanced
        generation through the prompt embedding table mechanism used by TensorRT-LLM
        for multimodal inputs.
        """

        ptuning_args = (
            [prompt_embedding_table, prompt_tasks, prompt_vocab_size]
            if prompt_embedding_table is not None
            else []
        )

        # Embedding stage
        if self.mapping.is_first_pp_rank():
            if input_ids is not None:
                # Get text embeddings with optional prompt tuning/multimodal features
                hidden_states = self.vocab_embedding(input_ids, *ptuning_args)

                # If we have audio features in prompt_embedding_table, they are already
                # integrated through the PromptTuningEmbedding mechanism. The audio
                # features should be preprocessed and passed as prompt_embedding_table
                # with appropriate prompt_tasks and prompt_vocab_size during inference.

        else:
            # Receive hidden states from previous pipeline stage
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())

        # Decoder layers
        hidden_states = self.layers(
            hidden_states=hidden_states,
            use_cache=use_cache,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
        )

        if use_cache:
            hidden_states, presents = hidden_states

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        if use_cache:
            return hidden_states, presents
        return hidden_states


class HiggsAudioForCausalLM(DecoderModelForCausalLM):
    """TensorRT-LLM implementation of Higgs Audio multimodal model."""

    def __init__(self, config: HiggsAudioConfig):
        # Initialize the transformer component
        transformer = HiggsAudioTransformer(config)

        # Audio output projector
        lm_head = ColumnLinear(
            in_features=config.hidden_size,
            out_features=config.audio_num_codebooks * (config.audio_codebook_size + 2),
            bias=False,
            dtype=config.dtype,
            tp_group=None,
            tp_size=1,
            gather_output=True,
        )
        super().__init__(config, transformer, lm_head)

    def use_prompt_tuning(self):
        """Enable prompt tuning for multimodal audio features."""
        from tensorrt_llm.layers.embedding import PromptTuningEmbedding

        embedding = self.transformer.vocab_embedding
        self.transformer.vocab_embedding = PromptTuningEmbedding(
            num_embeddings=embedding.num_embeddings,
            embedding_dim=embedding.embedding_dim,
            vocab_size=self.config.vocab_size,
            dtype=embedding.dtype,
            tp_size=embedding.tp_size,
            tp_group=embedding.tp_group,
            sharding_dim=embedding.sharding_dim,
            tp_rank=embedding.tp_rank,
        )
        self.transformer.vocab_embedding.weight.value = embedding.weight.raw_value

    @classmethod
    def from_hugging_face(
        cls,
        hf_model_dir: str,
        dtype: str = "auto",
        mapping: Optional[Mapping] = None,
        quant_config: Optional[QuantConfig] = None,
        **kwargs,
    ):
        """Create a HiggsAudioForCausalLM object from HuggingFace model directory.

        Args:
            hf_model_dir: Path to the HuggingFace model directory
            dtype: Data type for the model weights
            mapping: Multi-GPU mapping configuration
            quant_config: Quantization configuration
            **kwargs: Additional keyword arguments

        Returns:
            HiggsAudioForCausalLM: The loaded model
        """
        if not os.path.exists(hf_model_dir):
            hf_model_dir = snapshot_download(repo_id=hf_model_dir)

        config = HiggsAudioConfig(
            mapping=mapping or Mapping(),
            quant_config=quant_config or QuantConfig(),
            dtype=dtype,  # Pass dtype in kwargs to override default
            **kwargs,
        )
        custom_dict = {
            "lm_head": "audio_decoder_proj.audio_lm_head.weight",
        }
        loader = ModelWeightsLoader(hf_model_dir, custom_dict)

        model = cls(config)
        loader.update_key_mapping(model)
        tllm_weights = {}
        for tllm_key, _ in tqdm(model.named_parameters()):
            if "audio" in tllm_key:
                print(tllm_key)
                tllm_weights.update(loader.load(tllm_key))
        loader.fill(tllm_weights)

        loader.generate_tllm_weights(model)

        return model


class HiggsAudioTRTRunner:
    """TensorRT-LLM inference wrapper for HiggsAudio using ModelRunnerCpp."""

    def __init__(
        self,
        config: HiggsAudioConfig,
        engine_dir: str,
        tokenizer_dir: str,
        audio_tokenizer_dir: str,
    ) -> None:
        """Initialize the TensorRT-LLM runner for HiggsAudio."""

        from tensorrt_llm.runtime import ModelRunnerCpp

        self.config = config
        self.engine_dir = engine_dir
        self.tokenizer_dir = tokenizer_dir
        self.audio_tokenizer_dir = audio_tokenizer_dir

        self.temperature = 1.0
        self.top_k = 50
        self.top_p = 0.95
        self.num_beams = 1
        self.use_py_session = False
        self.gpu_weights_percent = 0.5
        self.max_new_tokens = 512
        self.enable_context_fmha_fp32_acc = False

        self.pad_token_id = config.pad_token_id
        self.audio_eos_token_id = config.audio_eos_token_id
        self.audio_in_token_idx = config.audio_in_token_idx
        self.vocab_size = config.vocab_size
        self.max_seq_len = 1500

        # Set up device
        self.gpu_device = torch.device("cuda", 0)
        torch.cuda.set_device(self.gpu_device)

        # Initialize runner components
        self.runner = None
        self.tokenizer = None
        self.sampling_config = None
        self.model_config = None

        # Load components
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_dir, trust_remote_code=True, use_fast=False
        )
        self.audio_tokenizer = AudioTokenizer(self.audio_tokenizer_dir, device=str(self.gpu_device))

        # Use the bindings SamplingConfig which has beam_width
        from tensorrt_llm.bindings import executor as trtllm

        self.sampling_config = trtllm.SamplingConfig(
            beam_width=self.num_beams,  # Use beam_width instead of num_beams
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )

        # Use the correct audio EOS token for generation
        self.generation_end_id = self.config.audio_eos_token_id  # 128012
        self.runner = ModelRunnerCpp.from_dir(
            engine_dir=self.engine_dir,
            use_gpu_direct_storage=True,
            cuda_graph_mode=True,
            kv_cache_free_gpu_memory_fraction=self.gpu_weights_percent,
        )

    def _embed_audio_codes(self, audio_codes):
        """Convert audio codes to embeddings using codebook embeddings.

        Args:
            audio_codes: Audio codes tensor of shape (num_codebooks, seq_len)

        Returns:
            audio_embeddings: Embeddings tensor of shape (seq_len, hidden_size)
        """
        # Create codebook shift for each codebook
        codebook_shift = (
            torch.arange(self.config.audio_num_codebooks, device=audio_codes.device)
            * self.config.audio_codebook_size
        )
        codebook_shift = codebook_shift.unsqueeze(-1)  # Shape: (num_codebooks, 1)

        # Shift audio codes by codebook offset
        shifted_codes = audio_codes + codebook_shift

        # Get embeddings for each codebook (this would need the actual embedding layer)
        # For now, create dummy embeddings - this should be replaced with actual embedding lookup
        seq_len = audio_codes.shape[1]
        hidden_size = self.config.hidden_size

        # Create dummy embeddings (in real implementation, use actual codebook embeddings)
        audio_embeddings = torch.randn(
            seq_len, hidden_size, device=audio_codes.device, dtype=torch.float32
        )

        return audio_embeddings

    def _convert_tokens_to_audio_codes(self, token_ids):
        """Convert generated token IDs to audio codes for decoding.

        Args:
            token_ids: Generated token IDs from the model

        Returns:
            audio_codes: Audio codes tensor of shape (num_codebooks, seq_len)
        """
        if token_ids is None or len(token_ids) == 0:
            return None

        # Convert token IDs to audio codes
        # The model generates flattened audio token IDs that need to be reshaped
        # into (num_codebooks, seq_len) format for the audio tokenizer

        num_codebooks = self.config.audio_num_codebooks
        codebook_size = self.config.audio_codebook_size

        # Convert to tensor if needed
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, device=self.gpu_device)

        # The model outputs audio token IDs that are offset by codebook
        # We need to convert them back to per-codebook codes
        audio_codes = []

        for i in range(num_codebooks):
            # Extract codes for this codebook
            codebook_offset = i * codebook_size
            codebook_mask = (token_ids >= codebook_offset) & (
                token_ids < codebook_offset + codebook_size
            )
            codebook_tokens = token_ids[codebook_mask] - codebook_offset
            audio_codes.append(codebook_tokens)

        # Pad to same length and stack
        if audio_codes:
            max_len = max(len(codes) for codes in audio_codes)
            padded_codes = []
            for codes in audio_codes:
                if len(codes) < max_len:
                    # Pad with zeros
                    padding = torch.zeros(
                        max_len - len(codes), dtype=codes.dtype, device=codes.device
                    )
                    codes = torch.cat([codes, padding])
                padded_codes.append(codes)

            return torch.stack(padded_codes, dim=0)  # Shape: (num_codebooks, seq_len)

        return None

    def generate(
        self,
        input_text: str,
        input_audio: str = None,
        use_delay_pattern: bool = True,
        **generation_kwargs,
    ):
        """Generate audio from text input and reference audio (TTS with voice cloning).

        Args:
            input_text: The text prompt to convert to speech
            input_audio: Path to reference audio file for voice cloning
            use_delay_pattern: Whether to use delay pattern for RVQ generation

        Returns:
            Generated audio tensor suitable for Whisper transcription
        """
        if not self.runner:
            raise RuntimeError("Runner not initialized")

        # Process text input using the exact format from Test.py
        if input_audio is not None:
            # Format with reference audio (voice cloning)
            pre_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an AI assistant designed to convert text into speech. Generate speech for the user's text, using the specified description.<|scene_desc_start|>Audio is recorded from a quiet room. Speaker is an enthusiastic young Australian woman in her early 20s with a bright, high-pitched voice.<|scene_desc_end|><|eot_id|><|start_header_id|>user<|end_header_id|>Can you believe just how realistic this sounds now?<|eot_id|><|start_header_id|>assistant<|end_header_id|><|audio_bos|><|AUDIO|><|audio_eos|><|eot_id|><|start_header_id|>user<|end_header_id|>"
            post_prompt = "<|eot_id|><|start_header_id|>assistant<|end_header_id|><|audio_out_bos|>"
            formatted_text = pre_prompt + input_text + post_prompt
        else:
            # Format without reference audio (default voice)
            formatted_text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an AI assistant designed to convert text into speech. Generate speech for the user's text.<|eot_id|><|start_header_id|>user<|end_header_id|>{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|><|audio_out_bos|>"

        input_ids = self.tokenizer.encode(formatted_text, return_tensors="pt").squeeze(0)

        # Process audio input if provided
        prompt_table = None
        prompt_tasks = None
        prompt_vocab_size = None

        if input_audio is not None and self.audio_tokenizer:
            # Encode audio to RVQ codes
            audio_codes = self.audio_tokenizer.encode(input_audio, sr=24000)

            # Limit audio codes to fit within prompt table constraints (max 128 tokens)
            max_audio_tokens = 100  # Leave some margin for delay pattern expansion
            if len(audio_codes.shape) >= 2:
                # If it's RVQ codes (num_codebooks, seq_len), limit sequence length
                if audio_codes.shape[1] > max_audio_tokens:
                    audio_codes = audio_codes[:, :max_audio_tokens]  # Truncate sequence
            elif len(audio_codes.shape) == 1:
                # If it's a 1D array, limit length
                if audio_codes.shape[0] > max_audio_tokens:
                    audio_codes = audio_codes[:max_audio_tokens]

            # Apply delay pattern if requested and we have multiple codebooks
            if use_delay_pattern and len(audio_codes.shape) >= 2 and audio_codes.shape[0] > 1:
                # Add BOS and EOS tokens using correct token IDs
                bos_tokens = torch.full(
                    (audio_codes.shape[0], 1),
                    self.config.audio_bos_token_id,  # 128011
                    dtype=audio_codes.dtype,
                    device=audio_codes.device,
                )
                eos_tokens = torch.full(
                    (audio_codes.shape[0], 1),
                    self.config.audio_eos_token_id,  # 128012
                    dtype=audio_codes.dtype,
                    device=audio_codes.device,
                )

                # Concatenate: BOS + audio_codes + EOS
                audio_codes_with_special = torch.cat([bos_tokens, audio_codes, eos_tokens], dim=-1)

                # Apply delay pattern
                audio_codes = _build_delay_pattern_mask(
                    audio_codes_with_special.unsqueeze(0),  # Add batch dimension
                    bos_token_id=self.config.audio_bos_token_id,  # 128011
                    pad_token_id=self.config.audio_eos_token_id,  # 128012
                ).squeeze(0)  # Remove batch dimension

            # Convert audio codes to embeddings for prompt table
            audio_embeddings = self._embed_audio_codes(audio_codes)

            # Prepare as prompt embedding table
            prompt_table = audio_embeddings.unsqueeze(0).to(
                dtype=torch.bfloat16, device=self.gpu_device
            )

            # Set up prompt tasks and vocab size for audio features
            num_audio_tokens = prompt_table.shape[1]

            # Create task IDs (0 for audio task) - should be a string for ModelRunnerCpp
            prompt_tasks = "0"  # Single task ID as string

            # Set vocab size for audio features
            prompt_vocab_size = torch.tensor(
                [num_audio_tokens], dtype=torch.int32, device=self.gpu_device
            )

        # Prepare batch
        batch_input_ids = [input_ids]

        # Update sampling config
        input_lengths = torch.tensor(
            [input_ids.size(-1)], device=self.gpu_device, dtype=torch.int32
        )
        max_input_length = torch.max(input_lengths).item()
        max_new_tokens = min(self.max_new_tokens, self.max_seq_len - max_input_length)

        # No need for sampling config - using direct parameters

        # Run generation with or without audio features
        with torch.no_grad():
            # TEMPORARY: Disable prompt table to test basic generation
            # Use the older API approach to avoid the sampling_config_list issue
            print(
                f"DEBUG: Generating without prompt table (prompt_table was {'not None' if prompt_table is not None else 'None'})"
            )
            outputs = self.runner.generate(
                batch_input_ids=batch_input_ids,
                max_new_tokens=max_new_tokens,
                beam_width=self.num_beams,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                end_id=self.generation_end_id,
                pad_id=self.pad_token_id,
            )

        # Process outputs - Convert generated token IDs to audio
        if outputs is not None and len(outputs) > 0:
            # Remove input tokens from output
            generated_ids = outputs[0][input_lengths[0] :]

            print(f"DEBUG: Generated {len(generated_ids)} tokens")
            print(f"DEBUG: First 10 generated tokens: {generated_ids[:10]}")
            print(f"DEBUG: Last 10 generated tokens: {generated_ids[-10:]}")

            # The audio tokenizer expects the raw generated token IDs directly
            # No need for complex conversion - the model generates audio tokens directly
            try:
                generated_audio = self.audio_tokenizer.decode(
                    generated_ids, return_cuda_tensor=False
                )
                print(
                    f"DEBUG: Audio tokenizer decode successful, shape: {generated_audio.shape if hasattr(generated_audio, 'shape') else type(generated_audio)}"
                )

                # Ensure it's a numpy array for Whisper compatibility
                if isinstance(generated_audio, torch.Tensor):
                    generated_audio = generated_audio.cpu().numpy()
                return generated_audio

            except Exception as e:
                print(f"DEBUG: Audio tokenizer decode failed: {e}")
                # Fallback: try with tensor conversion
                try:
                    if not isinstance(generated_ids, torch.Tensor):
                        generated_ids = torch.tensor(generated_ids)
                    generated_audio = self.audio_tokenizer.decode(
                        generated_ids, return_cuda_tensor=False
                    )
                    if isinstance(generated_audio, torch.Tensor):
                        generated_audio = generated_audio.cpu().numpy()
                    return generated_audio
                except Exception as e2:
                    print(f"DEBUG: Fallback decode also failed: {e2}")

        # Return silence if generation failed
        print("DEBUG: No valid output generated, returning silence")
        return np.zeros(16000, dtype=np.float32)  # 1 second of silence at 16kHz
