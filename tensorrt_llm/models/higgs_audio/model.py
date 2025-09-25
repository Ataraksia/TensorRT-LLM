# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
"""TensorRT-LLM implementation of Higgs Audio multimodal model."""

import tensorrt as trt

import builtins
from tkinter import NO
import librosa
from collections import defaultdict, deque, Counter
from collections.abc import AsyncGenerator, Sequence
import os
from pathlib import Path
from typing import Any, Dict, Optional, List, OrderedDict, Union
import numpy as np
from openai.types.chat import ChatCompletionAudio
import tensorrt
import torch
from starlette.datastructures import State
from huggingface_hub import snapshot_download
from torch import nn
import torchaudio
import torch.nn.functional as F


from tensorrt_llm.bindings import INT32, ModelConfig
from tensorrt_llm.layers.attention import minimum
from tensorrt_llm.mapping import Mapping

from tensorrt_llm.models.generation_mixin import GenerationMixin
from tensorrt_llm.models.model_weights_loader import ModelWeightsLoader
from tensorrt_llm.models.modeling_utils import (
    DecoderLayerList,
    QuantConfig,
    DecoderModelForCausalLM,
    cp_split_plugin,
)
from tensorrt_llm.module import Module, ModuleList
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig

from tensorrt_llm.functional import (
    Tensor,
    allgather,
    arange,
    cast,
    categorical_sample,
    constant,
    constant_to_tensor_,
    cumsum,
    expand,
    expand_dims_like,
    gather_last_token_logits,
    int32_array,
    nonzero,
    not_op,
    op_or,
    pad,
    shape,
    softmax,
    unsqueeze,
    view,
    where,
    sum,
    mean,
    concat,
    index_select,
    op_and,
    slice,
)
from tensorrt_llm.layers import (
    MLP,
    Attention,
    AttentionMaskType,
    AttentionParams,
    ColumnLinear,
    Embedding,
    KeyValueCacheParams,
    RmsNorm,
    SpecDecodingParams,
)
import math
import copy
import inspect
import json
import os
from typing import Optional


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

        # Audio MLP
        self.audio_mlp = MLP(
            hidden_size=self.config.hidden_size,
            ffn_hidden_size=self.config.intermediate_size,
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

        hidden_states = where(
            vision_token_mask,
            self.audio_input_layernorm(hidden_states),
            self.input_layernorm(hidden_states),
        )

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

        residual += where(
            vision_token_mask,
            self.audio_mlp(self.audio_post_layernorm(hidden_states)),
            self.mlp(self.post_layernorm(hidden_states)),
        )

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

        self.codebook_embeddings = Embedding(
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

    def _embed_audio_ids(self, audio_ids: Tensor):
        """Embeds audio ids for all codebooks."""
        # Expecting shape (num_codebooks, seq_len) local ids; create vector of shifts
        codebook_shift = (
            arange(0, self.config.num_codebooks, dtype="int32") * self.config.codebook_size
        ).unsqueeze(-1)
        shifted_ids = audio_ids + codebook_shift  # broadcast to (num_codebooks, seq_len)
        audio_embed = sum(self.codebook_embeddings(shifted_ids), dim=0)

        return audio_embed

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
        mask = (self.config.audio_in_end >= position_ids >= self.config.audio_in_start) or (
            position_ids >= self.config.audio_out_start
        )
        audio_ids = where(mask == 1, input_ids, 0)
        audio_embed = self._embed_audio_ids(audio_ids)
        text_ids = where(mask == 0, input_ids, 0)
        text_embed = self.vocab_embedding(text_ids)
        input_embed = where(mask.unsqueeze(-1), audio_embed, text_embed)

        hidden_states = self.layers(
            hidden_states=input_embed,
            use_cache=use_cache,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            vision_token_mask=mask.unsqueeze(-1),
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

        # Main lm_head for the first codebook
        lm_head = ColumnLinear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            bias=False,
            dtype=config.dtype,
        )

        self.first_pass = True

        super().__init__(config, transformer, lm_head)

    def forward(self, *args, **kwargs):
        # Get transformer outputs
        # TODO might need to change this later when running server - need to find out if first_pass resets- split real batches up and just run theM?
        input_ids = kwargs["input_ids"]
        # if self.first_pass:
        #     self.first_pass = False
        #     input_ids = slice(input_ids, [0], [self.config.audio_out_start + 1])

        kwargs = {k: v for k, v in kwargs.items() if k != "input_ids"}
        lm_logits, hidden_states, all_hidden_states = super().forward(input_ids, *args, **kwargs)

        return lm_logits, hidden_states, all_hidden_states

    @classmethod
    def from_hugging_face(
        cls,
        hf_config_or_dir: str = "bosonai/higgs-audio-v2-generation-3B-base",
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
            # "text_lm_head": "audio_decoder_proj.text_lm_head",
            "audio_post_layernorm": "audio_post_attention_layernorm",
            # Ensure audio codebook embeddings are loaded from HF weights
            "codebook_embeddings": "audio_codebook_embeddings",
        }
        loader = ModelWeightsLoader(hf_config_or_dir, custom_dict)
        trtllm_model = cls(config)
        loader.update_key_mapping(trtllm_model)
        loader.generate_tllm_weights(trtllm_model)

        return trtllm_model
