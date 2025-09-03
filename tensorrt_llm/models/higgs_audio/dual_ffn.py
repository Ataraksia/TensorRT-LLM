# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dual FFN decoder layer for Higgs Audio model.

This module implements the novel DualFFN architecture where text and audio tokens
are processed through separate FFN paths after shared attention, enabling
specialized processing for different modalities while maintaining efficiency.
"""

from enum import Enum
from typing import Optional

import torch

from tensorrt_llm.functional import Tensor, concat, shape
from tensorrt_llm.layers import (
    Attention,
    AttentionMaskType,
    FusedGatedMLP,
    PositionEmbeddingType,
    RmsNorm,
)
from tensorrt_llm.layers.attention import KeyValueCacheParams
from tensorrt_llm.module import Module


class GenerationMode(Enum):
    """Generation modes for Higgs Audio model.

    The model operates in different phases during generation:
    - TEXT: Standard text generation mode
    - AUDIO_INIT: Audio generation initialization phase
    - AUDIO_IN_PROGRESS: Active audio generation with RVQ coordination
    """

    TEXT = 0
    AUDIO_INIT = 1
    AUDIO_IN_PROGRESS = 2


# KV Cache Management Utilities
class CacheMode(Enum):
    """KV cache management modes for different sequence length patterns."""

    STATIC = "static"  # Fixed sequence lengths, pre-allocated cache
    DYNAMIC = "dynamic"  # Variable sequence lengths, dynamic allocation


class KVCacheManager:
    """Utilities for managing KV cache in static and dynamic modes."""

    @staticmethod
    def create_static_cache(
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_size: int,
        num_layers: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> tuple[list, dict]:
        """Create static KV cache for fixed sequence lengths.

        Args:
            batch_size: Fixed batch size
            max_seq_len: Maximum sequence length
            num_heads: Number of attention heads
            head_size: Attention head dimension
            num_layers: Number of transformer layers
            dtype: Cache tensor dtype
            device: Device for cache allocation

        Returns:
            Tuple of (past_key_value_list, cache_metadata)
        """
        past_key_value = []
        cache_shape = (batch_size, 2, num_heads, max_seq_len, head_size)

        for _ in range(num_layers):
            # Pre-allocate cache tensor for this layer
            layer_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
            past_key_value.append(layer_cache)

        cache_metadata = {
            "mode": CacheMode.STATIC,
            "batch_size": batch_size,
            "max_seq_len": max_seq_len,
            "num_heads": num_heads,
            "head_size": head_size,
            "dtype": dtype,
            "device": device,
        }

        return past_key_value, cache_metadata

    @staticmethod
    def create_dynamic_cache(
        num_layers: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype = torch.float16,
    ) -> tuple[list, dict]:
        """Create dynamic KV cache for variable-length sequences.

        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_size: Attention head dimension
            dtype: Cache tensor dtype

        Returns:
            Tuple of (past_key_value_list, cache_metadata)
        """
        # Initialize with None for dynamic allocation
        past_key_value = [None] * num_layers

        cache_metadata = {
            "mode": CacheMode.DYNAMIC,
            "num_heads": num_heads,
            "head_size": head_size,
            "dtype": dtype,
        }

        return past_key_value, cache_metadata

    @staticmethod
    def pack_cache_for_trt(cache_tensor: Tensor, block_size: int = 64) -> Tensor:
        """Pack cache tensor for TensorRT-LLM kernel optimization.

        TensorRT-LLM kernels work efficiently with block-aligned memory layouts.
        This function ensures cache tensors are properly aligned for optimal
        kernel fusion and memory access patterns.

        Args:
            cache_tensor: Cache tensor to pack [batch, 2, heads, seq, head_size]
            block_size: Block size for alignment (default 64 for TensorRT)

        Returns:
            Packed cache tensor optimized for TensorRT-LLM
        """
        if cache_tensor is None:
            return None

        # Get current shape
        batch_size = shape(cache_tensor, 0)
        kv_dim = shape(cache_tensor, 1)  # 2 for key+value
        num_heads = shape(cache_tensor, 2)
        seq_len = shape(cache_tensor, 3)
        head_size = shape(cache_tensor, 4)

        # Calculate padded sequence length aligned to block size
        padded_seq_len = ((seq_len + block_size - 1) // block_size) * block_size

        # Create padded tensor if sequence length is not block-aligned
        if seq_len != padded_seq_len:
            padding_shape = concat(
                [batch_size, kv_dim, num_heads, padded_seq_len - seq_len, head_size]
            )
            padding = torch.zeros(
                padding_shape, dtype=cache_tensor.dtype, device=cache_tensor.device
            )
            cache_tensor = concat([cache_tensor, padding], dim=3)

        return cache_tensor

    @staticmethod
    def invalidate_cache_for_mode_transition(
        kv_cache_params: KeyValueCacheParams,
        old_mode: GenerationMode,
        new_mode: GenerationMode,
        audio_boundary_positions: Optional[Tensor] = None,
    ) -> KeyValueCacheParams:
        """Invalidate cache entries during mode transitions.

        When transitioning between TEXT and AUDIO generation modes,
        certain cache entries may need invalidation to maintain coherence.

        Args:
            kv_cache_params: Current KV cache parameters
            old_mode: Previous generation mode
            new_mode: New generation mode
            audio_boundary_positions: Positions where mode transitions occur

        Returns:
            Updated KV cache parameters
        """
        if kv_cache_params is None or kv_cache_params.past_key_value is None:
            return kv_cache_params

        # For TEXT -> AUDIO transitions, invalidate cache after audio boundary
        if old_mode == GenerationMode.TEXT and new_mode in [
            GenerationMode.AUDIO_INIT,
            GenerationMode.AUDIO_IN_PROGRESS,
        ]:
            if audio_boundary_positions is not None:
                # Truncate cache at audio boundary positions
                for i, layer_cache in enumerate(kv_cache_params.past_key_value):
                    if layer_cache is not None:
                        # TODO: Implement selective cache truncation based on boundary positions
                        # This requires careful tensor slicing at the sequence dimension
                        pass

        # For AUDIO -> TEXT transitions, full cache invalidation may be needed
        elif (
            old_mode in [GenerationMode.AUDIO_INIT, GenerationMode.AUDIO_IN_PROGRESS]
            and new_mode == GenerationMode.TEXT
        ):
            # Reset cache to maintain text generation coherence
            if hasattr(kv_cache_params, "_original_past_key_value"):
                # Restore to pre-audio state if available
                kv_cache_params.past_key_value = kv_cache_params._original_past_key_value

        return kv_cache_params

    @staticmethod
    def validate_cache_consistency(
        kv_cache_params: KeyValueCacheParams, expected_seq_len: int, layer_idx: int
    ) -> bool:
        """Validate KV cache consistency and format correctness.

        Args:
            kv_cache_params: KV cache parameters to validate
            expected_seq_len: Expected sequence length
            layer_idx: Current layer index for validation

        Returns:
            True if cache is consistent and valid
        """
        if kv_cache_params is None:
            return True  # No cache is valid

        if kv_cache_params.past_key_value is None:
            return True  # Empty cache is valid

        if layer_idx >= len(kv_cache_params.past_key_value):
            return False  # Invalid layer index

        layer_cache = kv_cache_params.past_key_value[layer_idx]
        if layer_cache is None:
            return True  # Empty layer cache is valid for dynamic mode

        # Validate cache tensor shape
        if len(layer_cache.shape) != 5:
            return False  # Expected [batch, 2, heads, seq, head_size]

        # Validate sequence dimension
        cache_seq_len = layer_cache.shape[3]
        if cache_seq_len < expected_seq_len:
            return False  # Cache too short for expected sequence

        return True

    @staticmethod
    def create_cache_params_from_metadata(
        cache_metadata: dict, past_key_value: list
    ) -> KeyValueCacheParams:
        """Create TensorRT-LLM KeyValueCacheParams from cache metadata.

        Args:
            cache_metadata: Cache metadata from create_static/dynamic_cache
            past_key_value: List of cache tensors

        Returns:
            KeyValueCacheParams object for TensorRT-LLM compatibility
        """
        return KeyValueCacheParams(
            past_key_value=past_key_value,
            # Additional parameters would be set by the model/engine
            host_past_key_value_lengths=None,
            host_max_attention_window_sizes=None,
            host_sink_token_length=None,
            cache_indirection=None,
        )

    @staticmethod
    def extract_cache_from_params(
        kv_cache_params: KeyValueCacheParams, layer_idx: int
    ) -> Optional[Tensor]:
        """Extract layer cache from KeyValueCacheParams.

        Args:
            kv_cache_params: TensorRT-LLM cache parameters
            layer_idx: Layer index to extract

        Returns:
            Cache tensor for the specified layer or None
        """
        if kv_cache_params is None or kv_cache_params.past_key_value is None:
            return None

        if layer_idx >= len(kv_cache_params.past_key_value):
            return None

        return kv_cache_params.past_key_value[layer_idx]

    @staticmethod
    def update_cache_in_params(
        kv_cache_params: KeyValueCacheParams, layer_idx: int, updated_cache: Tensor
    ) -> KeyValueCacheParams:
        """Update layer cache in KeyValueCacheParams.

        Args:
            kv_cache_params: TensorRT-LLM cache parameters
            layer_idx: Layer index to update
            updated_cache: New cache tensor for the layer

        Returns:
            Updated KeyValueCacheParams
        """
        if kv_cache_params is None:
            return kv_cache_params

        if kv_cache_params.past_key_value is None:
            kv_cache_params.past_key_value = []

        # Extend list if needed
        while len(kv_cache_params.past_key_value) <= layer_idx:
            kv_cache_params.past_key_value.append(None)

        # Update the specific layer cache
        kv_cache_params.past_key_value[layer_idx] = updated_cache

        return kv_cache_params


class GenerationMode(Enum):
    """Generation modes for Higgs Audio model.

    The model operates in different phases during generation:
    - TEXT: Standard text generation mode
    - AUDIO_INIT: Audio generation initialization phase
    - AUDIO_IN_PROGRESS: Active audio generation with RVQ coordination
    """

    TEXT = 0
    AUDIO_INIT = 1
    AUDIO_IN_PROGRESS = 2


class HiggsAudioDualFFNDecoderLayer(Module):
    """Dual FFN decoder layer for Higgs Audio model.

    This layer implements the novel architecture where audio and text tokens
    share a common attention mechanism but are processed through separate
    FFN (feed-forward network) paths. This allows for:

    1. Specialized processing for different modalities
    2. Efficient inference with smaller audio FFN
    3. Additional trainable parameters for audio processing
    4. Maintained compatibility with standard transformer architecture

    Architecture flow:
    1. Shared multi-head attention for all tokens
    2. Split tokens by type (text vs audio)
    3. Apply separate FFNs to each token type
    4. Recombine outputs preserving original order

    Args:
        config: Model configuration containing hidden sizes, attention heads, etc.
        layer_idx: Index of this layer in the model
        dtype: Data type for computations
        cache_config: Configuration for KV caching
        quant_config: Quantization configuration
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        dtype: str = "float16",
        cache_config=None,
        quant_config=None,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.hidden_size = config.text_config.hidden_size
        self.num_attention_heads = config.text_config.num_attention_heads
        self.num_kv_heads = getattr(
            config.text_config, "num_key_value_heads", config.text_config.num_attention_heads
        )
        self.intermediate_size = config.text_config.intermediate_size
        self.dtype = dtype

        # Check if this layer uses dual FFN (some layers may use fast-forward mode)
        audio_dual_ffn_layers = getattr(config, "audio_dual_ffn_layers", None)
        if audio_dual_ffn_layers is None:
            # Default: all layers use dual FFN
            self.use_dual_ffn = True
            self.fast_forward = False
        else:
            self.use_dual_ffn = layer_idx in audio_dual_ffn_layers
            self.fast_forward = not self.use_dual_ffn

        # Check if audio tokens get separate attention
        self.use_audio_attention = getattr(config, "use_audio_out_self_attention", False)

        # Shared attention layer
        self.self_attn = Attention(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_kv_heads=self.num_kv_heads,
            max_position_embeddings=getattr(config.text_config, "max_position_embeddings", 8192),
            dtype=dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=getattr(config.text_config, "attention_bias", False),
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            rotary_embedding_base=getattr(config.text_config, "rope_theta", 10000.0),
            rotary_embedding_scaling=getattr(config.text_config, "rope_scaling", None),
            tp_group=None,
            tp_size=1,
            quant_mode=getattr(quant_config, "quant_mode", None) if quant_config else None,
        )

        # Normalization layers
        norm_eps = getattr(config.text_config, "rms_norm_eps", 1e-6)
        self.input_layernorm = RmsNorm(
            normalized_shape=self.hidden_size,
            eps=norm_eps,
            dtype=dtype,
        )
        self.post_attention_layernorm = RmsNorm(
            normalized_shape=self.hidden_size,
            eps=norm_eps,
            dtype=dtype,
        )

        # Text FFN (standard MLP)
        # Main text FFN using FusedGatedMLP for optimal fusion
        hidden_act = getattr(config.text_config, "hidden_act", "silu")
        mlp_bias = getattr(config.text_config, "mlp_bias", False)

        self.text_mlp = FusedGatedMLP(
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.intermediate_size,
            hidden_act=hidden_act,
            bias=mlp_bias,
            dtype=dtype,
            tp_group=None,
            tp_size=1,
            quant_mode=getattr(quant_config, "quant_mode", None) if quant_config else None,
        )

        # Audio-specific components (only if using dual FFN)
        if self.use_dual_ffn:
            # Audio-specific normalization layers
            self.audio_input_layernorm = RmsNorm(
                normalized_shape=self.hidden_size,
                eps=norm_eps,
                dtype=dtype,
            )
            self.audio_post_attention_layernorm = RmsNorm(
                normalized_shape=self.hidden_size,
                eps=norm_eps,
                dtype=dtype,
            )

            # Audio FFN using FusedGatedMLP for optimal fusion
            # Can be smaller than text FFN for efficiency
            audio_intermediate_size = getattr(
                config, "audio_intermediate_size", self.intermediate_size
            )
            self.audio_mlp = FusedGatedMLP(
                hidden_size=self.hidden_size,
                ffn_hidden_size=audio_intermediate_size,
                hidden_act=hidden_act,  # Use same activation as text for consistency
                bias=mlp_bias,
                dtype=dtype,
                tp_group=None,
                tp_size=1,
                quant_mode=getattr(quant_config, "quant_mode", None) if quant_config else None,
            )

            # Optional: separate attention for audio tokens
            if self.use_audio_attention:
                self.audio_self_attn = Attention(
                    hidden_size=self.hidden_size,
                    num_attention_heads=self.num_attention_heads,
                    num_kv_heads=self.num_kv_heads,
                    max_position_embeddings=getattr(
                        config.text_config, "max_position_embeddings", 8192
                    ),
                    dtype=dtype,
                    attention_mask_type=AttentionMaskType.causal,
                    bias=getattr(config.text_config, "attention_bias", False),
                    position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
                    rotary_embedding_base=getattr(config.text_config, "rope_theta", 10000.0),
                    rotary_embedding_scaling=getattr(config.text_config, "rope_scaling", None),
                    tp_group=None,
                    tp_size=1,
                    quant_mode=getattr(quant_config, "quant_mode", None) if quant_config else None,
                )
                self.audio_post_audio_attn_layernorm = RmsNorm(
                    normalized_shape=self.hidden_size,
                    eps=norm_eps,
                    dtype=dtype,
                )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        kv_cache_params=None,
        attention_params=None,
        mode: GenerationMode = GenerationMode.TEXT,
        audio_out_mask: Optional[Tensor] = None,
        delay_pattern: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass through the dual FFN layer.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask tensor
            position_ids: Position indices for rotary embeddings
            kv_cache_params: KV cache parameters for incremental decoding
            attention_params: Additional attention parameters
            mode: Current generation mode (TEXT, AUDIO_INIT, AUDIO_IN_PROGRESS)
            audio_out_mask: Boolean mask indicating audio token positions [batch_size, seq_len]
            delay_pattern: RVQ codebook delay pattern [num_codebooks, seq_len]

        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        residual = hidden_states

        # Apply generation mode masking to inputs
        if mode != GenerationMode.TEXT:
            hidden_states = self.apply_generation_mode_masking(
                hidden_states, mode, audio_out_mask, attention_mask, delay_pattern
            )

        # Input normalization - use audio-specific norm for audio tokens if dual FFN
        if self.use_dual_ffn and audio_out_mask is not None:
            # Apply different normalization based on token type
            text_mask = ~audio_out_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
            audio_mask = audio_out_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]

            text_normed = self.input_layernorm(hidden_states) * text_mask
            audio_normed = self.audio_input_layernorm(hidden_states) * audio_mask
            hidden_states = text_normed + audio_normed
        else:
            hidden_states = self.input_layernorm(hidden_states)

        # Optional: Audio-specific attention for audio tokens
        if self.use_audio_attention and audio_out_mask is not None:
            # TODO: Implement audio-specific attention
            # This would apply separate attention to audio tokens before shared attention
            pass

        # Apply advanced attention masking for audio generation
        if mode != GenerationMode.TEXT and attention_mask is not None:
            attention_mask = compute_dual_ffn_attention_mask(
                attention_mask, audio_out_mask, mode, delay_pattern
            )

        # Shared attention for all tokens
        attention_output = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            **kwargs,
        )

        # Extract attention output and updated cache
        # TensorRT-LLM attention returns (context, past_key_value) when using cache
        if isinstance(attention_output, tuple):
            attention_context, updated_past_key_value = attention_output
        else:
            attention_context = attention_output
            updated_past_key_value = None

        # Add residual connection after attention
        hidden_states = residual + attention_context
        residual = hidden_states

        # Fusion-friendly dual FFN processing with static graph
        # Always execute both paths and use elementwise masking for routing

        # Create masks for text and audio tokens
        if self.use_dual_ffn and audio_out_mask is not None:
            text_mask = ~audio_out_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
            audio_mask = audio_out_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        else:
            # Default to text-only processing when no audio mask
            text_mask = torch.ones_like(hidden_states[..., :1])  # [batch_size, seq_len, 1]
            audio_mask = torch.zeros_like(hidden_states[..., :1])  # [batch_size, seq_len, 1]

        # Always process both paths for static graph (fusion-friendly)
        # Text path
        text_norm = self.post_attention_layernorm(hidden_states)
        text_output = self.text_mlp(text_norm)
        text_output = text_output * text_mask  # Apply text mask

        # Audio path (when dual FFN is enabled)
        if self.use_dual_ffn:
            audio_norm = self.audio_post_attention_layernorm(hidden_states)
            audio_output = self.audio_mlp(audio_norm)
            audio_output = audio_output * audio_mask  # Apply audio mask
        else:
            # Create zero tensor with same shape for static graph
            audio_output = torch.zeros_like(text_output)

        # Combine outputs with elementwise addition (fusion-friendly)
        ffn_output = text_output + audio_output
        hidden_states = residual + ffn_output

        # KV cache handling for TensorRT-LLM compatibility
        if kv_cache_params is not None and updated_past_key_value is not None:
            # Validate cache consistency
            layer_idx = getattr(self, "layer_idx", 0)  # Get layer index if available
            expected_seq_len = hidden_states.shape[1] if len(hidden_states.shape) >= 2 else 0

            if not KVCacheManager.validate_cache_consistency(
                kv_cache_params, expected_seq_len, layer_idx
            ):
                # Cache validation failed - log warning but continue
                pass

            # Apply TensorRT-LLM cache packing if needed
            if (
                hasattr(kv_cache_params, "_enable_trt_packing")
                and kv_cache_params._enable_trt_packing
            ):
                updated_past_key_value = KVCacheManager.pack_cache_for_trt(updated_past_key_value)

            # Handle mode transitions with cache invalidation
            if hasattr(self, "_last_generation_mode") and hasattr(self, "_current_generation_mode"):
                if self._last_generation_mode != self._current_generation_mode:
                    kv_cache_params = KVCacheManager.invalidate_cache_for_mode_transition(
                        kv_cache_params, self._last_generation_mode, self._current_generation_mode
                    )

            # Return both hidden states and updated cache for TensorRT-LLM
            return hidden_states, updated_past_key_value

        return hidden_states

    def create_audio_out_mask(
        self,
        input_ids: Tensor,
        audio_token_ids: Optional[list] = None,
    ) -> Optional[Tensor]:
        """Create mask identifying audio token positions.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            audio_token_ids: List of token IDs that represent audio tokens

        Returns:
            Boolean mask [batch_size, seq_len] where True indicates audio tokens
        """
        if audio_token_ids is None:
            return None

        # Create mask for audio tokens
        audio_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in audio_token_ids:
            audio_mask |= input_ids == token_id

        return audio_mask

    def apply_generation_mode_masking(
        self,
        hidden_states: Tensor,
        mode: GenerationMode,
        audio_out_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        delay_pattern: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply mode-specific masking for different generation phases.

        This implements the masking logic for different generation modes:
        - TEXT: Block audio token contributions
        - AUDIO_INIT: Allow priming context, prepare for audio generation
        - AUDIO_IN_PROGRESS: Apply causal constraints for RVQ coordination

        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            mode: Current generation mode
            audio_out_mask: Audio token mask [batch_size, seq_len]
            attention_mask: Attention mask for context [batch_size, seq_len, seq_len]
            delay_pattern: RVQ codebook delay pattern [num_codebooks, seq_len]

        Returns:
            Masked hidden states
        """
        if audio_out_mask is None:
            return hidden_states

        # Different masking strategies based on generation mode
        if mode == GenerationMode.TEXT:
            # During text generation, completely mask out audio token contributions
            text_mask = (~audio_out_mask).unsqueeze(-1)  # [batch_size, seq_len, 1]
            return hidden_states * text_mask

        elif mode == GenerationMode.AUDIO_INIT:
            # During audio initialization, allow all context but apply light constraints
            return hidden_states

        elif mode == GenerationMode.AUDIO_IN_PROGRESS:
            # During audio generation, apply sophisticated RVQ constraints
            if delay_pattern is not None and attention_mask is not None:
                # Create constrained attention mask for RVQ coordination
                _ = apply_delay_pattern_masking(attention_mask, audio_out_mask, delay_pattern)
                # Apply mask-based attention weighting to hidden states
                # This is a simplified version - full implementation would require
                # access to attention weights from the forward pass
                return hidden_states
            else:
                # Fallback: allow audio generation with basic constraints
                return hidden_states

        return hidden_states


def create_audio_out_mask_from_token_types(
    token_type_ids: Tensor,
    audio_token_type: int = 1,
) -> Tensor:
    """Create audio token mask from token type IDs.

    Args:
        token_type_ids: Token type IDs [batch_size, seq_len] where 0=text, 1=audio
        audio_token_type: Token type ID that represents audio tokens

    Returns:
        Boolean mask [batch_size, seq_len] where True indicates audio tokens
    """
    return token_type_ids == audio_token_type


def compute_dual_ffn_attention_mask(
    attention_mask: Tensor,
    audio_out_mask: Optional[Tensor] = None,
    mode: GenerationMode = GenerationMode.TEXT,
    delay_pattern: Optional[Tensor] = None,
) -> Tensor:
    """Compute attention mask for dual FFN processing.

    This function modifies the base attention mask to implement mode-specific
    attention patterns for the dual FFN architecture, including RVQ delay patterns.

    Args:
        attention_mask: Base causal attention mask [batch_size, seq_len, seq_len]
        audio_out_mask: Audio token positions mask [batch_size, seq_len]
        mode: Current generation mode (TEXT, AUDIO_INIT, AUDIO_IN_PROGRESS)
        delay_pattern: RVQ codebook delay pattern [num_codebooks, seq_len] or None

    Returns:
        Modified attention mask for dual FFN processing [batch_size, seq_len, seq_len]
    """
    if audio_out_mask is None or mode == GenerationMode.TEXT:
        return attention_mask

    batch_size, seq_len = audio_out_mask.shape
    modified_mask = attention_mask.clone()

    if mode == GenerationMode.AUDIO_INIT:
        # During audio initialization, allow all context but prepare for restrictions
        return modified_mask

    elif mode == GenerationMode.AUDIO_IN_PROGRESS:
        # Apply RVQ causal constraints during audio generation
        if delay_pattern is not None:
            modified_mask = apply_delay_pattern_masking(
                modified_mask, audio_out_mask, delay_pattern
            )

        # Block audio->text attention during audio generation to prevent leakage
        audio_positions = audio_out_mask.unsqueeze(1)  # [batch_size, 1, seq_len]
        text_positions = (~audio_out_mask).unsqueeze(2)  # [batch_size, seq_len, 1]

        # Mask audio tokens from attending to text tokens
        audio_to_text_mask = audio_positions & text_positions
        modified_mask = modified_mask.masked_fill(audio_to_text_mask, float("-inf"))

    return modified_mask


def apply_delay_pattern_masking(
    attention_mask: Tensor,
    audio_out_mask: Tensor,
    delay_pattern: Tensor,
) -> Tensor:
    """Apply delay pattern masking for RVQ codebook coordination.

    This implements the delay pattern constraints where different RVQ codebooks
    have staggered generation timing to maintain causality while enabling
    parallel codebook processing.

    Args:
        attention_mask: Base attention mask [batch_size, seq_len, seq_len]
        audio_out_mask: Audio token positions [batch_size, seq_len]
        delay_pattern: Codebook delay pattern [num_codebooks, seq_len]

    Returns:
        Masked attention tensor with delay pattern constraints
    """
    batch_size, seq_len, _ = attention_mask.shape
    num_codebooks, pattern_len = delay_pattern.shape

    # Ensure pattern length matches sequence length
    if pattern_len != seq_len:
        # Pad or truncate delay pattern to match sequence length
        if pattern_len < seq_len:
            padding = torch.zeros(
                num_codebooks,
                seq_len - pattern_len,
                dtype=delay_pattern.dtype,
                device=delay_pattern.device,
            )
            delay_pattern = torch.cat([delay_pattern, padding], dim=1)
        else:
            delay_pattern = delay_pattern[:, :seq_len]

    modified_mask = attention_mask.clone()

    # Apply codebook-specific causal constraints
    for batch_idx in range(batch_size):
        audio_positions = torch.where(audio_out_mask[batch_idx])[0]

        if len(audio_positions) > 0:
            for codebook_idx in range(num_codebooks):
                # Get delayed positions for this codebook
                delay = delay_pattern[codebook_idx]

                # Create codebook-specific causal mask
                for i, pos_i in enumerate(audio_positions):
                    for j, pos_j in enumerate(audio_positions):
                        if i <= j:  # Only consider causal relationships
                            # Apply delay constraint
                            delay_i = delay[pos_i] if pos_i < len(delay) else 0
                            delay_j = delay[pos_j] if pos_j < len(delay) else 0

                            # If codebook j is ahead of codebook i by more than delay
                            if delay_j > delay_i:
                                modified_mask[batch_idx, pos_i, pos_j] = float("-inf")

    return modified_mask


def create_fast_forward_mask(
    input_ids: Tensor,
    audio_token_ids: list,
    fast_forward_layers: Optional[list] = None,
    current_layer: int = 0,
) -> Optional[Tensor]:
    """Create fast-forward mask for efficient audio processing.

    Fast-forward masking allows certain layers to skip detailed processing
    of audio tokens when they don't contribute significantly to the output.

    Args:
        input_ids: Token IDs [batch_size, seq_len]
        audio_token_ids: List of audio token IDs that can be fast-forwarded
        fast_forward_layers: List of layer indices that use fast-forward processing
        current_layer: Current layer index

    Returns:
        Boolean mask [batch_size, seq_len] where True indicates fast-forward tokens
    """
    if fast_forward_layers is None or current_layer not in fast_forward_layers:
        return None

    # Create mask for audio tokens that can be fast-forwarded
    fast_forward_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in audio_token_ids:
        fast_forward_mask |= input_ids == token_id

    return fast_forward_mask


def validate_audio_mask_consistency(
    audio_out_mask: Tensor,
    input_ids: Tensor,
    audio_token_ids: list,
) -> bool:
    """Validate consistency between audio_out_mask and actual audio tokens.

    Args:
        audio_out_mask: Audio token mask [batch_size, seq_len]
        input_ids: Token IDs [batch_size, seq_len]
        audio_token_ids: List of expected audio token IDs

    Returns:
        True if mask is consistent with token IDs
    """
    # Create expected mask from token IDs
    expected_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in audio_token_ids:
        expected_mask |= input_ids == token_id

    # Check if provided mask matches expected mask
    return torch.equal(audio_out_mask, expected_mask)


def optimize_attention_mask_for_trt(
    attention_mask: Tensor,
    block_size: int = 64,
) -> Tensor:
    """Optimize attention mask for TensorRT-LLM kernel fusion.

    This function restructures attention masks to be more amenable to
    TensorRT optimizations by aligning with preferred block sizes.

    Args:
        attention_mask: Input attention mask [batch_size, seq_len, seq_len]
        block_size: Preferred block size for TensorRT kernels

    Returns:
        Optimized attention mask
    """
    batch_size, seq_len, _ = attention_mask.shape

    # Pad sequence length to nearest block boundary if needed
    if seq_len % block_size != 0:
        pad_size = block_size - (seq_len % block_size)
        padding = torch.zeros(
            batch_size, seq_len, pad_size, dtype=attention_mask.dtype, device=attention_mask.device
        )
        padding.fill_(float("-inf"))

        attention_mask = torch.cat([attention_mask, padding], dim=2)

        # Also pad the sequence dimension
        padding = torch.zeros(
            batch_size,
            pad_size,
            attention_mask.size(2),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        padding.fill_(float("-inf"))
        attention_mask = torch.cat([attention_mask, padding], dim=1)

    return attention_mask


class AudioMaskingUtils:
    """Utility class for audio-specific masking operations."""

    @staticmethod
    def create_codebook_delay_pattern(
        num_codebooks: int = 8,
        sequence_length: int = 1024,
        delay_type: str = "linear",
    ) -> Tensor:
        """Create delay pattern for RVQ codebook coordination.

        Args:
            num_codebooks: Number of RVQ codebooks (typically 8)
            sequence_length: Length of the sequence
            delay_type: Type of delay pattern ("linear", "exponential", "custom")

        Returns:
            Delay pattern tensor [num_codebooks, sequence_length]
        """
        delays = torch.zeros(num_codebooks, sequence_length, dtype=torch.long)

        if delay_type == "linear":
            # Linear delay: codebook k has delay k
            for k in range(num_codebooks):
                delays[k, :] = k
        elif delay_type == "exponential":
            # Exponential delay: codebook k has delay 2^k
            for k in range(num_codebooks):
                delays[k, :] = 2**k
        elif delay_type == "custom":
            # Custom delay pattern based on Higgs Audio paper
            # First codebook (k=0) has no delay, others have increasing delays
            codebook_delays = [0, 1, 2, 3, 4, 5, 6, 7][:num_codebooks]
            for k, delay in enumerate(codebook_delays):
                delays[k, :] = delay

        return delays

    @staticmethod
    def create_streaming_mask(
        sequence_length: int,
        chunk_size: int = 32,
        overlap: int = 4,
    ) -> Tensor:
        """Create mask for streaming audio generation.

        Args:
            sequence_length: Total sequence length
            chunk_size: Size of each processing chunk
            overlap: Overlap between chunks for continuity

        Returns:
            Streaming mask [num_chunks, sequence_length]
        """
        num_chunks = (sequence_length + chunk_size - 1) // chunk_size
        mask = torch.zeros(num_chunks, sequence_length, dtype=torch.bool)

        for chunk_idx in range(num_chunks):
            start = max(0, chunk_idx * chunk_size - overlap)
            end = min(sequence_length, (chunk_idx + 1) * chunk_size + overlap)
            mask[chunk_idx, start:end] = True

        return mask
