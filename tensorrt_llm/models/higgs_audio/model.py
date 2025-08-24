# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

from ..._utils import pad_vocab_size
from ...functional import Tensor
from ...layers import (
    ColumnLinear,
    Embedding,
    PromptTuningEmbedding,
    RmsNorm,
    Attention,
    AttentionMaskType,
    PositionEmbeddingType,
    GatedMLP,
)
from ...module import Module
from ...top_model_mixin import TopModelMixin
from ..modeling_utils import DecoderLayerList, DecoderModelForCausalLM
from .config import HiggsAudioConfig
from .convert import build_config_from_hf


class HiggsAudioDecoderLayer(Module):
    """Minimal LLaMA-style decoder layer (baseline for Higgs-Audio).

    Audio adapters will be introduced later; this keeps the backbone functional.
    """

    def __init__(self, config: HiggsAudioConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.mapping = config.mapping

        # Norms
        self.input_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                       eps=config.norm_epsilon,
                                       dtype=config.dtype)
        self.post_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                      eps=config.norm_epsilon,
                                      dtype=config.dtype)

        # Attention
        self.attention = Attention(
            local_layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            attention_head_size=config.head_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            dtype=config.dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=config.attn_bias,
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            rotary_embedding_base=config.rotary_base,
            rotary_embedding_scaling=config.rotary_scaling,
            tp_group=self.mapping.tp_group,
            tp_size=self.mapping.tp_size,
            tp_rank=self.mapping.tp_rank,
            q_scaling=1.0,
            quant_mode=getattr(config, "quant_mode", None),
            cp_group=self.mapping.cp_group,
            cp_size=self.mapping.cp_size,
            cp_rank=self.mapping.cp_rank,
        )

        # MLP
        mlp_hidden_size = (config.hidden_size * 4
                           if config.intermediate_size is None
                           else config.intermediate_size)
        self.mlp = GatedMLP(hidden_size=config.hidden_size,
                            ffn_hidden_size=mlp_hidden_size,
                            hidden_act=config.hidden_act,
                            dtype=config.dtype,
                            bias=False,
                            tp_group=self.mapping.tp_group,
                            tp_size=self.mapping.tp_size,
                            quant_mode=getattr(config, "quant_mode", None))

    def forward(self,
                hidden_states,
                attention_mask=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None,
                lora_layer_params=None,
                position_ids=None,
                audio_token_mask=None):
        # Attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.attention(hidden_states,
                                  attention_mask=attention_mask,
                                  use_cache=use_cache,
                                  kv_cache_params=kv_cache_params,
                                  attention_params=attention_params,
                                  lora_layer_params=lora_layer_params)
        if use_cache:
            attn_out, presents = attn_out
        hidden_states = residual + attn_out

        # MLP block
        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states,
                                 lora_layer_params=lora_layer_params)
        hidden_states = residual + hidden_states

        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class HiggsAudioBackbone(Module):

    def __init__(self, config: HiggsAudioConfig) -> None:
        super().__init__()
        self.config = config
        self.mapping = config.mapping
        self.use_prompt_tuning = getattr(config, "use_prompt_tuning", False)
        self.vocab_size = config.vocab_size
        EmbeddingCls = PromptTuningEmbedding if self.use_prompt_tuning else Embedding
        if self.mapping.is_first_pp_rank():
            self.vocab_embedding = EmbeddingCls(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                dtype=config.dtype,
                tp_size=self.mapping.tp_size if getattr(config, "use_parallel_embedding", False) else 1,
                tp_group=self.mapping.tp_group if getattr(config, "use_parallel_embedding", False) else None,
                sharding_dim=getattr(config, "embedding_sharding_dim", None),
                tp_rank=self.mapping.tp_rank,
            )

        # Placeholder layer list to enable wiring; will be replaced with Llama-like layers with audio path
        self.layers = DecoderLayerList(HiggsAudioDecoderLayer, config)

        if self.mapping.is_last_pp_rank():
            self.ln_f = RmsNorm(normalized_shape=config.hidden_size,
                                eps=config.norm_epsilon,
                                dtype=config.dtype)

    def forward(self,
                input_ids,
                position_ids=None,
                use_cache=False,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                hidden_states=None,
                prompt_embedding_table: Optional[Tensor] = None,
                prompt_tasks: Optional[Tensor] = None,
                prompt_vocab_size: Optional[Tensor] = None,
                lora_params=None,
                input_token_extra_ids=None,
                audio_token_mask=None):
        # Fill kv cache structures
        kv_cache_params.fill_none_tensor_list(len(self.layers))

        # Embedding lookup (or receive from PP)
        if self.mapping.is_first_pp_rank():
            ptuning_args = [prompt_embedding_table, prompt_tasks, prompt_vocab_size] \
                if self.use_prompt_tuning else []
            hidden_states = self.vocab_embedding(input_ids, *ptuning_args)
        else:
            from ...functional import recv
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())

        # Forward layers; audio_token_mask will be used later
        hidden_states = self.layers.forward(hidden_states,
                                            use_cache=use_cache,
                                            attention_mask=attention_mask,
                                            kv_cache_params=kv_cache_params,
                                            attention_params=attention_params,
                                            lora_params=lora_params,
                                            position_ids=position_ids,
                                            audio_token_mask=audio_token_mask)

        if use_cache:
            hidden_states, presents = hidden_states

        if self.mapping.is_last_pp_rank():
            hidden_states = self.ln_f(hidden_states)
        else:
            from ...functional import send
            hidden_states = send(hidden_states, self.mapping.next_pp_rank())

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class HiggsAudioForCausalLM(DecoderModelForCausalLM, TopModelMixin):
    config_class = HiggsAudioConfig

    def __init__(self, config: HiggsAudioConfig):
        transformer = HiggsAudioBackbone(config)
        vocab_size_padded = pad_vocab_size(config.vocab_size, config.mapping.tp_size)
        if config.mapping.is_last_pp_rank():
            lm_head = ColumnLinear(config.hidden_size,
                                   vocab_size_padded,
                                   bias=False,
                                   dtype=config.dtype,
                                   tp_group=config.mapping.tp_group,
                                   tp_size=config.mapping.tp_size,
                                   gather_output=True)
        else:
            lm_head = None
        super().__init__(config, transformer, lm_head)

    def default_plugin_config(self, **kwargs):
        return super().default_plugin_config(**kwargs)

    @classmethod
    def from_hugging_face(cls,
                          hf_model_or_dir,
                          dtype: str = 'auto',
                          mapping=None,
                          quant_config=None,
                          **kwargs):
        """Construct model from a HuggingFace model directory.

        This builds a HiggsAudioConfig via `build_config_from_hf` and returns a model instance.
        """
        cfg = build_config_from_hf(hf_model_or_dir,
                                   dtype=dtype,
                                   mapping=mapping,
                                   quant_config=quant_config,
                                   **kwargs)
        return cls(cfg)
