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
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoTokenizer,
    BatchFeature,
    ProcessorMixin,
)
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

import tensorrt_llm
from tensorrt_llm import logger

try:
    from tensorrt_llm.runtime import (
        PYTHON_BINDINGS,
        ModelRunnerCpp,
        SamplingConfig,
        ModelConfig,
    )
except ImportError:
    PYTHON_BINDINGS = False
    from tensorrt_llm.runtime import ModelRunner

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


def revert_delay_pattern(data):
    """Convert samples encoded with delay pattern back to the original form."""
    assert len(data.shape) == 2
    out_l = []
    num_codebooks = data.shape[0]
    for i in range(num_codebooks):
        out_l.append(data[i, i:])
    return np.concatenate(out_l, axis=0)


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
        last_token_ids=None,
        attention_mask: Optional[Tensor] = None,
        kv_cache_params: Optional[KeyValueCacheParams] = None,
        attention_params: Optional[AttentionParams] = None,
        prompt_embedding_table: Optional[Tensor] = None,
        prompt_tasks: Optional[Tensor] = None,
        prompt_vocab_size: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass for Higgs Audio transformer."""

        if prompt_embedding_table is not None:
            # Audio generation mode - use audio codebook embeddings
            batch_size, seq_len, num_codebooks = (
                shape(prompt_embedding_table, 0),
                shape(prompt_embedding_table, 1),
                shape(prompt_embedding_table, 2),
            )

            # Flatten prompt_embedding_table to (batch_size, seq_len * num_codebooks)
            flat_prompt_embeddings = view(
                prompt_embedding_table, [batch_size, seq_len * num_codebooks]
            )
            codebook_shift = (
                torch.arange(num_codebooks, device=audio_embeddings.device)
                * self.config.audio_codebook_size
            )
            codebook_shift = codebook_shift.unsqueeze(-1)

            # Get audio codebook embeddings
            audio_embeddings = self.audio_codebook_embeddings(
                flat_prompt_embeddings + codebook_shift
            )

            # Reshape back to (batch_size, seq_len, num_codebooks, hidden_size)
            audio_embeddings = view(
                audio_embeddings, [batch_size, seq_len, num_codebooks, self.config.hidden_size]
            )

            hidden_states = audio_embeddings.sum(dim=0)

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
                from .convert import convert_hf_higgs

                # Load HuggingFace model for weight conversion
                from transformers import AutoModelForCausalLM

                hf_model = AutoModelForCausalLM.from_pretrained(
                    hf_model_dir, trust_remote_code=True, torch_dtype="auto"
                )

                # Convert weights
                weights = convert_hf_higgs(
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
        self.config = config
        self.engine_dir = engine_dir
        self.tokenizer_dir = tokenizer_dir
        self.audio_tokenizer_dir = audio_tokenizer_dir

        self.temperature = 1.0
        self.top_k = 50
        self.top_p = 0.95
        self.num_beams = 1
        self.use_py_session = False
        self.gpu_weights_percent = None
        self.max_new_tokens = 512
        self.enable_context_fmha_fp32_acc = False

        self.audio_in_token_idx = config.audio_in_token_idx
        self.vocab_size = config.vocab_size

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
        self.audio_tokenizer = load_higgs_audio_tokenizer(
            self.audio_tokenizer_dir, device=str(self.gpu_device)
        )
        logger.info(f"Loaded tokenizer from {self.tokenizer_dir}")

        self.sampling_config = SamplingConfig(
            end_id=self.audio_eos_token_id,
            pad_id=self.pad_token_id,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )
        if PYTHON_BINDINGS and not self.use_py_session:
            self.runner = ModelRunnerCpp().from_dir(
                engine_dir=self.engine_dir,
                gpu_weights_percent=self.gpu_weights_percent,
                enable_context_fmha_fp32_acc=self.enable_context_fmha_fp32_acc,
                use_gpu_direct_storage=True,
                cuda_graph_mode=True,
            )
        else:
            self.runner = ModelRunner.from_dir(
                engine_dir=self.engine_dir,
                gpu_weights_percent=self.gpu_weights_percent,
            )

        logger.info("Successfully set up TensorRT-LLM runner")

    def generate(
        self,
        input_text: str,
        input_audio: str,
        max_new_tokens: Optional[int] = None,
        **generation_kwargs,
    ) -> str:
        """Generate response integrating text and optional audio inputs."""
        if not self.runner:
            raise RuntimeError("Runner not initialized")

        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").squeeze(0)

        prompt_table = None
        input_token_extra_ids = None

        if input_audio is not None and self.audio_tokenizer:
            audio_features = self.audio_tokenizer.encode(input_audio, sr=24000)
            num_audio_tokens = audio_features.shape[0]

            special_mask = input_ids == self.audio_in_token_idx
            if special_mask.sum() != 1:
                raise ValueError("Expected exactly one special audio token in input_ids")
            position = torch.where(special_mask)[1].item()

            fake_ids = torch.arange(
                self.vocab_size,
                self.vocab_size + num_audio_tokens,
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            input_ids = torch.cat([input_ids[:position], fake_ids, input_ids[position + 1 :]])

            # Prepare prompt_table from audio_features
            prompt_table = audio_features.to(
                dtype=self.model_config.dtype, device=self.gpu_device
            ).unsqueeze(0)

            seq_extra_ids = torch.full(
                (num_audio_tokens,), [1], dtype=torch.int64, device=self.gpu_device
            )

            # Insert into extra_ids at the fake positions
            extra_ids = torch.zeros_like(input_ids, dtype=torch.int64)
            extra_ids[position : position + num_audio_tokens] = seq_extra_ids
            input_token_extra_ids = [extra_ids.tolist()]
        print(input_token_extra_ids)
        # Prepare batch
        batch_input_ids = [input_ids.tolist()]

        # Update sampling config
        if max_new_tokens is not None:
            self.sampling_config.max_new_tokens = max_new_tokens

        # Run generation
        with torch.no_grad():
            outputs = self.runner.generate(
                batch_input_ids=batch_input_ids,
                sampling_config=self.sampling_config,
                prompt_table=prompt_table,
                prompt_tasks="0",
                input_token_extra_ids=input_token_extra_ids,
                **generation_kwargs,
            )

        # Decode new tokens only
        output_ids = outputs["output_ids"][0][0]
        input_len = len(input_ids)
        generated_ids = output_ids[input_len:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text
