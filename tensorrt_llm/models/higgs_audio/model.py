# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
"""TensorRT-LLM implementation of Higgs Audio multimodal model."""

import copy
import inspect
import json
import math
import os
import sys
import tempfile
import warnings
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from functools import lru_cache
from typing import Any, ClassVar, Dict, List, Literal, Optional, Set, Tuple, TypedDict, Union

import librosa
import numpy as np
import s3fs
import torch
import torch.nn as nn
import torch.nn.functional as F

from boson_multimodal import (
    HiggsAudioFeatureExtractor as BM_HiggsAudioFeatureExtractor,
    HiggsAudioTokenizer,
)
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
import transformers
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from transformers import AutoConfig, AutoFeatureExtractor, BatchFeature, ProcessorMixin
from transformers.models.whisper import WhisperFeatureExtractor
from transformers.tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput

# TensorRT-LLM imports
from .config import HiggsAudioConfig, HiggsAudioEncoderConfig
from ..modeling_utils import (
    DecoderModelForCausalLM,
    PretrainedConfig,
    PretrainedModel,
    DecoderLayerList,
)
from ...functional import (
    Tensor,
    ACT2FN,
    PositionEmbeddingType,
    concat,
    constant,
    gather_last_token_logits,
    shape,
    split,
    view,
    rms_norm,
)
from ...layers import (
    MLP,
    Attention,
    AttentionMaskType,
    AttentionParams,
    ColumnLinear,
    Embedding,
    GatedMLP,
    KeyValueCacheParams,
    Linear,
    LoraParams,
    RmsNorm,
    RowLinear,
)
from ...mapping import Mapping
from ...module import Module, ModuleList
from ...parameter import Parameter
from ...quantization import QuantMode

# TensorRT-LLM runtime imports (imported in TRTRunner class to avoid circular imports)
import tensorrt as trt  # type: ignore
import tensorrt_llm
from tensorrt_llm import logger


def revert_delay_pattern(data):
    """Convert samples encoded with delay pattern back to the original form."""
    assert len(data.shape) == 2
    out_l = []
    num_codebooks = data.shape[0]
    for i in range(num_codebooks):
        out_l.append(data[i, i:])
    return np.concatenate(out_l, axis=0)


class AudioTokenizer:
    """Common interface for audio tokenizers."""

    def __init__(self, model, device="cuda:0"):
        self.model = model
        self.device = device

    @property
    def sampling_rate(self):
        return self.model.sampling_rate

    @property
    def hop_length(self):
        return self.model.hop_length

    @property
    def chunk_length(self):
        return self.model.chunk_length

    @property
    def n_fft(self):
        return self.model.n_fft

    @property
    def feature_size(self):
        return self.model.feature_size

    def encode(self, audio_path_or_wv, sr=None, loudness_normalize=False, loudness_threshold=-23.0):
        return self.model.encode(audio_path_or_wv, sr, loudness_normalize, loudness_threshold)

    def decode(self, vq_code, return_cuda_tensor=False):
        return self.model.decode(vq_code, return_cuda_tensor)


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

    for i in range(0, codes.shape[-1], chunk_size):
        start = max(0, i - overlap_width)
        end = min(codes.shape[-1], i + chunk_size + overlap_width)

        chunk_codes = codes[:, :, start:end]
        chunk_output = xcodec_model.decode(chunk_codes)

        if i == 0:
            outputs.append(chunk_output[:, :, :chunk_output_length])
        else:
            outputs.append(chunk_output[:, :, overlap_width:])

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
        config = OmegaConf.load(os.path.join(tokenizer_path, "config.yaml"))
        config = OmegaConf.to_container(config, resolve=True)
    else:
        raise FileNotFoundError(f"No config found in {tokenizer_path}")
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


# Logger setup
logger.info("HiggsAudio TensorRT-LLM model loaded")

_KEYS_TO_MODIFY_MAPPING = {
    "audio_decoder_proj.audio_lm_head": "audio_lm_head",
    "audio_decoder_proj.text_lm_head": "text_lm_head",
    "model.": "",
}

AutoConfig.register("higgs_audio", HiggsAudioConfig)
AutoFeatureExtractor.register(HiggsAudioConfig, AudioTokenizer)
transformers._modules.add("AudioTokenizer")
transformers.AudioTokenizer = AudioTokenizer


# === Audio Inputs === #
class HiggsAudioInputs(TypedDict):
    # (num_audios, num_mel_bins, 3000)`
    audio_features: torch.Tensor
    # (num_audios, 3000)
    audio_feature_attention_mask: torch.Tensor
    # (num_audios, num_codebooks)
    audio_out_ids: torch.Tensor


def _validate_and_reshape_mm_tensor(
    mm_input: object, name: str, pad_with: Optional[int] = None
) -> torch.Tensor:
    if not isinstance(mm_input, (torch.Tensor, list)):
        raise TypeError(f"Invalid type for {name}: {type(mm_input)}")
    if isinstance(mm_input, torch.Tensor):
        return mm_input
    else:
        if pad_with is not None:
            # Pad sequences to same length if needed
            max_len = max(len(seq) for seq in mm_input)
            padded = []
            for seq in mm_input:
                if len(seq) < max_len:
                    padding = [pad_with] * (max_len - len(seq))
                    seq = seq + padding
                padded.append(seq)
            return torch.tensor(padded)
        else:
            return torch.tensor(mm_input)


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


class HiggsAudioFeatureProjector(Module):
    """Projector that maps audio features to text hidden state."""

    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.config = config

        audio_hidden_size = config.audio_encoder_config.d_model
        text_hidden_size = config.hidden_size

        self.linear = Linear(
            in_features=audio_hidden_size,
            out_features=text_hidden_size,
            bias=True,
            dtype=config.dtype,
            tp_group=None,
            tp_size=1,
            gather_output=True,
        )

    def forward(self, audio_features: Tensor) -> Tensor:
        """Project audio features to text hidden size."""
        hidden_states = self.linear(audio_features)
        return hidden_states


class HiggsAudioDecoderProjector(Module):
    """Projection layers that map hidden states from the LLM component to audio / text logits."""

    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.config = config

        # Audio projection head
        self.audio_lm_head = Linear(
            in_features=config.hidden_size,
            out_features=config.audio_num_codebooks * config.audio_codebook_size,
            bias=False,
            dtype=config.dtype,
            tp_group=None,
            tp_size=1,
            gather_output=True,
        )

        # Text projection head
        self.text_lm_head = Linear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            bias=False,
            dtype=config.dtype,
            tp_group=None,
            tp_size=1,
            gather_output=True,
        )

    def forward(self, hidden_states: Tensor, mode: str = "text") -> Tensor:
        """Project hidden states to logits."""
        if mode == "audio":
            logits = self.audio_lm_head(hidden_states)
            # Reshape to (batch, seq_len, num_codebooks, codebook_size)
            batch_size, seq_len = shape(hidden_states, 0), shape(hidden_states, 1)
            logits = view(
                logits,
                [
                    batch_size,
                    seq_len,
                    self.config.audio_num_codebooks,
                    self.config.audio_codebook_size,
                ],
            )
        else:  # text mode
            logits = self.text_lm_head(hidden_states)

        return logits


class HiggsAudioDualFFNDecoderLayer(Module):
    """TensorRT-LLM implementation of dual-path FFN decoder layer."""

    def __init__(self, config: HiggsAudioConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Shared attention layer
        self.self_attn = Attention(
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

        self.post_attention_layernorm = RmsNorm(
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
        audio_token_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass for dual FFN decoder layer."""

        residual = hidden_states

        # Apply appropriate input layer norm based on token type
        if audio_token_mask is not None:
            # Apply different layer norms for audio vs text tokens
            text_states = self.input_layernorm(hidden_states)
            audio_states = self.audio_input_layernorm(hidden_states)
            # Mix based on audio token mask
            hidden_states = audio_token_mask * audio_states + (1 - audio_token_mask) * text_states
        else:
            hidden_states = self.input_layernorm(hidden_states)

        # Shared attention layer
        attention_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
        )

        if use_cache:
            attention_output, present_key_value = attention_output

        hidden_states = residual + attention_output

        # Dual-path FFN
        residual = hidden_states

        if audio_token_mask is not None:
            # Separate processing for text and audio tokens
            text_states = self.post_attention_layernorm(hidden_states)
            audio_states = self.audio_post_attention_layernorm(hidden_states)

            # Apply appropriate MLPs
            text_output = self.mlp(text_states)
            audio_output = self.audio_mlp(audio_states)

            # Mix outputs based on token type
            mlp_output = audio_token_mask * audio_output + (1 - audio_token_mask) * text_output
        else:
            # Default to text processing if no audio mask provided
            hidden_states = self.post_attention_layernorm(hidden_states)
            mlp_output = self.mlp(hidden_states)

        hidden_states = residual + mlp_output

        if use_cache:
            return hidden_states, present_key_value
        return hidden_states


class HiggsAudioForCausalLM(DecoderModelForCausalLM):
    """TensorRT-LLM implementation of Higgs Audio multimodal model."""

    def __init__(self, config: HiggsAudioConfig):
        # Initialize the transformer component
        transformer = HiggsAudioTransformer(config)

        # Initialize language model head
        vocab_size_padded = config.vocab_size
        if hasattr(config, "mapping") and config.mapping is not None:
            if config.mapping.tp_size > 1:
                vocab_size_padded = vocab_size_padded // config.mapping.tp_size

        lm_head = ColumnLinear(
            in_features=config.hidden_size,
            out_features=vocab_size_padded,
            bias=False,
            dtype=config.dtype,
            tp_group=config.mapping.tp_group
            if hasattr(config, "mapping") and config.mapping
            else None,
            tp_size=config.mapping.tp_size if hasattr(config, "mapping") and config.mapping else 1,
            gather_output=True,
        )

        super().__init__(config, transformer, lm_head)

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
        from transformers import AutoConfig
        from ..modeling_utils import QuantConfig as DefaultQuantConfig

        # For now, create a basic HiggsAudio TensorRT-LLM config
        # In a full implementation, this would parse the HF config properly
        config = HiggsAudioConfig(
            mapping=mapping or Mapping(),
            quant_config=quant_config or DefaultQuantConfig(),
            dtype=dtype,  # Pass dtype in kwargs to override default
            **kwargs,
        )

        # Create model
        model = cls(config)

        # Load weights using the convert function
        if not kwargs.get("skip_loading_weights", False):
            try:
                from .convert import convert_hf_qwen

                # Load HuggingFace model for weight conversion
                from transformers import AutoModelForCausalLM

                hf_model = AutoModelForCausalLM.from_pretrained(
                    hf_model_dir, trust_remote_code=True, torch_dtype="auto"
                )

                # Convert weights
                weights = convert_hf_qwen(
                    hf_model=hf_model,
                    qwen_type="higgs_audio",  # Use custom type for HiggsAudio
                    mapping=mapping or Mapping(),
                    dtype=dtype,
                )

                # Load weights into the model
                model.load(weights)

            except Exception as e:
                logger.warning(f"Failed to load weights from HuggingFace model: {e}")
                logger.warning("Model created without weights loaded")

        return model


class HiggsAudioTransformer(Module):
    """TensorRT-LLM transformer component for Higgs Audio model."""

    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.config = config

        # Audio feature projector
        self.audio_encoder_proj = HiggsAudioFeatureProjector(config)

        # Text embedding
        vocab_size_padded = config.vocab_size
        if hasattr(config, "mapping") and config.mapping is not None:
            if config.mapping.tp_size > 1:
                vocab_size_padded = vocab_size_padded // config.mapping.tp_size

        self.embed_tokens = Embedding(
            num_embeddings=vocab_size_padded,
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

        # Audio output projector
        self.audio_decoder_proj = HiggsAudioDecoderProjector(config)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Optional[Tensor] = None,
        use_cache: bool = False,
        attention_mask: Optional[Tensor] = None,
        kv_cache_params: Optional[KeyValueCacheParams] = None,
        attention_params: Optional[AttentionParams] = None,
        audio_features: Optional[Tensor] = None,
        audio_feature_attention_mask: Optional[Tensor] = None,
        audio_out_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass for Higgs Audio transformer."""

        # Process input embeddings
        if audio_features is not None:
            # Audio comprehension mode - process audio features
            audio_embeddings = self.audio_tower(
                input_features=audio_features,
                attention_mask=audio_feature_attention_mask,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None,
            )

            # Project audio features to text hidden size
            audio_embeddings = self.audio_encoder_proj(audio_embeddings)

            # Get text embeddings
            text_embeddings = self.embed_tokens(input_ids)

            # Merge audio and text embeddings based on input_ids
            # This would need custom logic to determine where audio tokens are
            hidden_states = text_embeddings  # Simplified for now

        elif audio_out_ids is not None:
            # Audio generation mode - use audio codebook embeddings
            batch_size, seq_len, num_codebooks = (
                shape(audio_out_ids, 0),
                shape(audio_out_ids, 1),
                shape(audio_out_ids, 2),
            )

            # Flatten audio_out_ids to (batch_size, seq_len * num_codebooks)
            flat_audio_ids = view(audio_out_ids, [batch_size, seq_len * num_codebooks])

            # Get audio codebook embeddings
            audio_embeddings = self.audio_codebook_embeddings(flat_audio_ids)

            # Reshape back to (batch_size, seq_len, num_codebooks, hidden_size)
            audio_embeddings = view(
                audio_embeddings, [batch_size, seq_len, num_codebooks, self.config.hidden_size]
            )

            # Combine codebook embeddings (e.g., sum or average)
            hidden_states = audio_embeddings.sum(dim=2)  # Sum across codebooks

        else:
            # Text-only mode
            hidden_states = self.embed_tokens(input_ids)

        # Pass through decoder layers
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


class HiggsAudioTRTRunner:
    """TensorRT-LLM inference wrapper for HiggsAudio using ModelRunnerCpp."""

    def __init__(
        self,
        engine_dir: str,
        tokenizer_dir: str,
        audio_tokenizer_dir: str,
        gpu_id: int = 0,
        num_beams: int = 1,
        use_py_session: bool = False,
        debug_mode: bool = False,
        lora_dir: Optional[str] = None,
        lora_ckpt_source: Optional[str] = None,
        gpu_weights_percent: Optional[float] = None,
        max_new_tokens: int = 64,
        enable_context_fmha_fp32_acc: bool = False,
    ) -> None:
        """Initialize the TensorRT-LLM runner for HiggsAudio."""
        self.engine_dir = engine_dir
        self.tokenizer_dir = tokenizer_dir
        self.audio_tokenizer_dir = audio_tokenizer_dir
        self.num_beams = num_beams
        self.use_py_session = use_py_session
        self.debug_mode = debug_mode
        self.lora_dir = lora_dir
        self.lora_ckpt_source = lora_ckpt_source
        self.gpu_weights_percent = gpu_weights_percent
        self.max_new_tokens = max_new_tokens
        self.enable_context_fmha_fp32_acc = enable_context_fmha_fp32_acc

        # Set up device
        self.gpu_device = torch.device("cuda", gpu_id)
        torch.cuda.set_device(self.gpu_device)

        # Initialize runner components
        self.session_audio = None
        self.runner = None
        self.tokenizer = None
        self.sampling_config = None
        self.model_config = None

        # Load components
        self._load_tokenizers()
        self._load_model_config()
        self._setup_runner()

    def _load_tokenizers(self):
        """Load the tokenizer from the specified directory."""
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_dir, trust_remote_code=True, use_fast=False
            )
            self.audio_tokenizer = load_higgs_audio_tokenizer()
            logger.info(f"Loaded tokenizer from {self.tokenizer_dir}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

    def _load_model_config(self):
        """Load the model configuration."""
        try:
            from tensorrt_llm.runtime import ModelConfig

            config_path = os.path.join(self.engine_dir, "config.json")
            if os.path.exists(config_path):
                self.model_config = ModelConfig.from_json_file(config_path)
            else:
                self.model_config = ModelConfig()

            logger.info("Loaded model configuration")
        except Exception as e:
            logger.error(f"Failed to load model config: {e}")
            raise

    def _setup_runner(self):
        """Set up the TensorRT-LLM model runner."""
        try:
            from tensorrt_llm.runtime import (
                PYTHON_BINDINGS,
                ModelRunner,
                SamplingConfig as TRTSamplingConfig,
            )

            if PYTHON_BINDINGS:
                from tensorrt_llm.runtime import ModelRunnerCpp

            if PYTHON_BINDINGS and not self.use_py_session:
                self.runner = ModelRunnerCpp.from_dir(
                    engine_dir=self.engine_dir,
                    lora_dir=self.lora_dir,
                    rank=tensorrt_llm.mpi_rank(),
                    debug_mode=self.debug_mode,
                    lora_ckpt_source=self.lora_ckpt_source,
                    gpu_weights_percent=self.gpu_weights_percent,
                    enable_context_fmha_fp32_acc=self.enable_context_fmha_fp32_acc,
                )
            else:
                self.runner = ModelRunner.from_dir(
                    engine_dir=self.engine_dir,
                    lora_dir=self.lora_dir,
                    rank=tensorrt_llm.mpi_rank(),
                    debug_mode=self.debug_mode,
                    lora_ckpt_source=self.lora_ckpt_source,
                    gpu_weights_percent=self.gpu_weights_percent,
                )

            # Set up sampling config
            self.sampling_config = TRTSamplingConfig(
                end_id=self.tokenizer.eos_token_id if self.tokenizer else None,
                pad_id=self.tokenizer.pad_token_id if self.tokenizer else None,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
            )

            logger.info("Successfully set up TensorRT-LLM runner")
        except Exception as e:
            logger.error(f"Failed to set up runner: {e}")
            raise

    def generate(
        self,
        input_text: str,
        audio_data: Optional[np.ndarray] = None,
        max_new_tokens: Optional[int] = None,
        **generation_kwargs,
    ) -> str:
        """Generate text/audio response from input text and optional audio."""
        if not self.runner:
            raise RuntimeError("Runner not initialized")

        pre_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an AI assistant designed to convert text into speech. Generate speech for the user's text, using the specified description.<|scene_desc_start|>Audio is recorded from a quiet room. Speaker is an enthusiastic young Australian woman in her early 20s with a bright, high-pitched voice.<|scene_desc_end|><|eot_id|><|start_header_id|>user<|end_header_id|>Can you believe just how realistic this sounds now?<|eot_id|><|start_header_id|>assistant<|end_header_id|><|audio_bos|><|AUDIO|><|audio_eos|><|eot_id|><|start_header_id|>user<|end_header_id|>"  # noqa: E501
        post_prompt = "<|eot_id|><|start_header_id|>assistant<|end_header_id|><|audio_out_bos|>"

        input_text = pre_prompt + input_text + post_prompt

        # Tokenize input
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")

        # Process audio if provided
        if audio_data is not None and self.audio_tokenizer:
            # Run audio encoder
            input_audio_features = self.audio_tokenizer.encode(audio_data, sr=24000)
            # TODO: Integrate audio features into prompt

        # Set up generation parameters
        if max_new_tokens is not None:
            self.sampling_config.max_new_tokens = max_new_tokens

        # Run generation
        with torch.no_grad():
            outputs = self.runner.generate(
                batch_input_ids=[input_ids.squeeze(0).tolist()],
                sampling_config=self.sampling_config,
                **generation_kwargs,
            )

        # Decode output
        output_ids = outputs["output_ids"][0]
        generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        return generated_text

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, "session_audio") and self.session_audio:
            del self.session_audio
        if hasattr(self, "runner") and self.runner:
            del self.runner
