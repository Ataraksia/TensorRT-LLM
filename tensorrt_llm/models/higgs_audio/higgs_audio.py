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

"""Main HiggsAudio model implementation for TensorRT-LLM.

HiggsAudio is a multimodal TTS model that combines Llama-3.2-3B text generation
with audio processing through a novel DualFFN architecture. The model processes:
- Text input through standard transformer layers
- Audio input through Whisper-based encoder
- Dual-headed output for simultaneous text and 8-codebook audio generation

The model operates in different generation modes:
- TEXT: Standard text generation
- AUDIO_INIT: Audio generation initialization
- AUDIO_IN_PROGRESS: Active audio generation with RVQ coordination
"""

from typing import Any, Dict, Optional, Tuple

import torch

from tensorrt_llm.functional import Tensor, gather_last_token_logits
from tensorrt_llm.layers import Embedding, Linear, RmsNorm
from tensorrt_llm.layers.attention import KeyValueCacheParams
from tensorrt_llm.module import Module, ModuleList

from .audio_encoder import HiggsAudioEncoder, HiggsAudioFeatureProjector
from .audio_tokenizer import HiggsAudioTokenizer
from .config import HiggsAudioConfig
from .dual_ffn import GenerationMode, HiggsAudioDualFFNDecoderLayer

__all__ = [
    "HiggsAudioModel",
    "HiggsAudioDecoderProjector",
    "GenerationMode",
]


class HiggsAudioDecoderProjector(Module):
    """Dual-headed projector for text and audio token generation.

    Projects hidden states to both text vocabulary logits and 8-codebook
    audio token logits simultaneously during generation.
    """

    def __init__(self, config: HiggsAudioConfig, dtype: str = "float16"):
        super().__init__()
        self.config = config

        # Get text config parameters
        if isinstance(config.text_config, dict):
            text_vocab_size = config.text_config.get("vocab_size", 128256)
            hidden_size = config.text_config.get("hidden_size", 3072)
        else:
            text_vocab_size = getattr(config.text_config, "vocab_size", 128256)
            hidden_size = getattr(config.text_config, "hidden_size", 3072)

        # Text head for standard vocabulary
        self.text_lm_head = Linear(
            in_features=hidden_size,
            out_features=text_vocab_size,
            bias=False,
            dtype=dtype,
        )

        # Audio head for RVQ codebooks
        # Each codebook has audio_codebook_size tokens + 2 for stream BOS/EOS
        audio_vocab_size = config.audio_num_codebooks * (config.audio_codebook_size + 2)
        self.audio_lm_head = Linear(
            in_features=hidden_size,
            out_features=audio_vocab_size,
            bias=False,
            dtype=dtype,
        )

        # Optional projection layers for audio output embeddings
        if config.use_audio_out_embed_projector:
            self.audio_out_embed_projector = Linear(
                in_features=hidden_size,
                out_features=hidden_size,
                bias=False,
                dtype=dtype,
            )
        else:
            self.audio_out_embed_projector = None

    def forward(
        self,
        hidden_states: Tensor,
        mode: GenerationMode = GenerationMode.TEXT,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass producing text and audio logits.

        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            mode: Generation mode determining which heads to compute

        Returns:
            Tuple of (text_logits, audio_logits)
            audio_logits is None in TEXT mode, shaped
            [batch_size, seq_len, num_codebooks, codebook_size+2] otherwise
        """
        # Always compute text logits
        text_logits = self.text_lm_head(hidden_states)

        # Compute audio logits for audio generation modes
        if mode in [GenerationMode.AUDIO_INIT, GenerationMode.AUDIO_IN_PROGRESS]:
            # Apply optional projection before audio head
            audio_hidden = hidden_states
            if self.audio_out_embed_projector is not None:
                audio_hidden = self.audio_out_embed_projector(audio_hidden)

            # Get raw audio logits
            audio_logits = self.audio_lm_head(audio_hidden)

            # Reshape to [batch, seq_len, num_codebooks, codebook_size+2]
            batch_size, seq_len = audio_logits.shape[:2]
            audio_logits = audio_logits.view(
                batch_size,
                seq_len,
                self.config.audio_num_codebooks,
                self.config.audio_codebook_size + 2,
            )
        else:
            audio_logits = None

        return text_logits, audio_logits


class HiggsAudioModel(Module):
    """Main HiggsAudio multimodal model for TensorRT-LLM.

    Combines Llama-3.2-3B text generation backbone with audio processing
    through Whisper encoder and novel DualFFN architecture for simultaneous
    text and audio token generation.
    """

    def __init__(self, config: HiggsAudioConfig, dtype: str = "float16"):
        super().__init__()

        # Validate and store configuration
        if not isinstance(config, HiggsAudioConfig):
            raise TypeError(f"Expected HiggsAudioConfig, got {type(config)}")
        config.validate()  # Ensure config is valid
        self.config = config
        self.dtype = dtype

        # Extract key configuration values
        if isinstance(config.text_config, dict):
            self.hidden_size = config.text_config.get("hidden_size", 3072)
            self.num_layers = config.text_config.get("num_hidden_layers", 28)
            self.vocab_size = config.text_config.get("vocab_size", 128256)
            self.num_attention_heads = config.text_config.get("num_attention_heads", 24)
            self.num_key_value_heads = config.text_config.get("num_key_value_heads", 8)
        else:
            self.hidden_size = getattr(config.text_config, "hidden_size", 3072)
            self.num_layers = getattr(config.text_config, "num_hidden_layers", 28)
            self.vocab_size = getattr(config.text_config, "vocab_size", 128256)
            self.num_attention_heads = getattr(config.text_config, "num_attention_heads", 24)
            self.num_key_value_heads = getattr(config.text_config, "num_key_value_heads", 8)

        # Store special token IDs for easy access
        self.audio_bos_token_id = config.audio_bos_token_id
        self.audio_eos_token_id = config.audio_eos_token_id
        self.audio_out_bos_token_id = config.audio_out_bos_token_id
        self.audio_in_token_idx = config.audio_in_token_idx
        self.audio_out_token_idx = config.audio_out_token_idx
        self.audio_stream_bos_id = config.audio_stream_bos_id
        self.audio_stream_eos_id = config.audio_stream_eos_id
        self.pad_token_id = config.pad_token_id

        # Audio codebook parameters
        self.audio_num_codebooks = config.audio_num_codebooks
        self.audio_codebook_size = config.audio_codebook_size

        # ========== Model Components ==========

        # 1. Text Token Embedding (with extended vocabulary for audio tokens)
        # Extended vocab to include audio-specific tokens
        extended_vocab_size = self.vocab_size + 64  # Add space for audio tokens
        self.embed_tokens = Embedding(
            num_embeddings=extended_vocab_size,
            embedding_dim=self.hidden_size,
            dtype=dtype,
        )

        # 2. Position handling will be managed by individual attention layers
        # RoPE embeddings are handled within the DualFFN layers themselves

        # 3. Audio Encoder (Whisper-based) - processes mel-spectrograms to features

        # 3. Audio Encoder (Whisper-based) - processes mel-spectrograms to features
        self.audio_encoder = HiggsAudioEncoder(config, dtype=dtype)

        # 4. Audio Feature Projector - maps audio features to text model hidden size
        self.audio_feature_projector = HiggsAudioFeatureProjector(config, dtype=dtype)

        # 5. Audio Tokenizer - handles RVQ encoding/decoding (loaded separately)
        self.audio_tokenizer = None  # Will be loaded with load_audio_tokenizer()

        # 6. Decoder Layers - using DualFFN layers for audio-aware processing
        self.layers = ModuleList()
        for i in range(self.num_layers):
            layer = HiggsAudioDualFFNDecoderLayer(
                config=config,
                layer_idx=i,
                dtype=dtype,
            )
            self.layers.append(layer)

        # 7. Layer Normalization (RMS norm following Llama architecture)
        if isinstance(config.text_config, dict):
            norm_eps = config.text_config.get("rms_norm_eps", 1e-6)
        else:
            norm_eps = getattr(config.text_config, "rms_norm_eps", 1e-6)
        self.norm = RmsNorm(normalized_shape=self.hidden_size, eps=norm_eps, dtype=dtype)

        # 8. Dual-headed output projector
        self.decoder_proj = HiggsAudioDecoderProjector(config, dtype=dtype)

        # 9. Audio codebook embeddings for input audio tokens
        # Each codebook needs embeddings for its tokens plus stream BOS/EOS
        audio_vocab_size = self.audio_num_codebooks * (self.audio_codebook_size + 2)
        self.audio_codebook_embeddings = Embedding(
            num_embeddings=audio_vocab_size,
            embedding_dim=self.hidden_size,
            dtype=dtype,
        )

        # ========== State Management ==========

        # Current generation mode
        self.generation_mode = GenerationMode.TEXT

        # Audio feature cache for multimodal processing
        self._audio_feature_cache = None
        self._audio_feature_lengths = None

        # Masks and cache metadata buffers
        self._audio_out_mask = None
        self._attention_mask_cache = None
        self._past_key_values = None

        # Performance optimization flags
        self.use_static_cache = True
        self.use_flash_attention = True
        self.fast_forward_layers = getattr(config, "audio_dual_ffn_layers", list(range(self.num_layers)))

    def set_generation_mode(self, mode: GenerationMode):
        """Set the current generation mode."""
        if not isinstance(mode, GenerationMode):
            raise TypeError(f"Expected GenerationMode, got {type(mode)}")
        self.generation_mode = mode

    def load_audio_tokenizer(self, tokenizer_path: str):
        """Load the audio tokenizer from the specified path."""
        self.audio_tokenizer = HiggsAudioTokenizer(tokenizer_path)

    def get_audio_out_mask(
        self, input_ids: Tensor, token_type_ids: Optional[Tensor] = None
    ) -> Optional[Tensor]:
        """Generate mask for audio output tokens.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            token_type_ids: Optional token type IDs [batch_size, seq_len]
                           1 for audio tokens, 0 for text tokens

        Returns:
            Boolean mask [batch_size, seq_len] where True indicates audio output positions
        """
        if self.generation_mode == GenerationMode.TEXT:
            return None

        if token_type_ids is not None:
            # Use provided token types
            return token_type_ids == 1  # 1 indicates audio tokens
        else:
            # Infer from special tokens in input_ids
            audio_token_mask = input_ids == self.audio_out_token_idx
            return audio_token_mask

    def _prepare_attention_mask(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor],
        audio_out_mask: Optional[Tensor],
    ) -> Tensor:
        """Prepare attention mask with audio-specific causality."""
        batch_size, seq_len = input_ids.shape

        if attention_mask is None:
            # Default causal mask
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device)
            )
            attention_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)

        # No special masking if not in audio mode or no audio tokens
        if self.generation_mode == GenerationMode.TEXT or audio_out_mask is None or not audio_out_mask.any():
            return attention_mask

        # TODO: Implement full RVQ delay pattern causality here.
        # This is a placeholder for the complex masking required for staggered
        # codebook generation. For now, we use a standard causal mask.
        # A real implementation would modify the attention_mask based on the
        # relative positions of audio tokens and their codebook indices.

        return attention_mask

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        audio_features: Optional[Tensor] = None,
        audio_feature_attention_mask: Optional[Tensor] = None,
        audio_out_ids: Optional[Tensor] = None,
        kv_cache_params: Optional[KeyValueCacheParams] = None,
        use_cache: bool = True,
        last_token_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass through the HiggsAudio model.

        Args:
            input_ids: Text token IDs [batch_size, seq_len]
            position_ids: Position indices [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len, seq_len]
            token_type_ids: Token type IDs (0=text, 1=audio) [batch_size, seq_len]
            audio_features: Mel-spectrogram features [batch_size, mel_bins, time]
            audio_feature_attention_mask: Audio attention mask [batch_size, time]
            audio_out_ids: Audio output token IDs [batch_size, seq_len, num_codebooks]
            kv_cache_params: KV cache parameters for incremental decoding
            use_cache: Whether to use KV cache
            last_token_ids: Last token IDs for gather operations [batch_size]

        Returns:
            Tuple of (text_logits, audio_logits)
        """
        batch_size, seq_len = input_ids.shape

        # ========== 1. Input Token Embedding ==========

        # Get text token embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Get audio output mask for dual processing (needed for later processing)
        audio_out_mask = self.get_audio_out_mask(input_ids, token_type_ids)

        # ========== 2. Audio Feature Processing ==========

        audio_features_embedded = None
        if audio_features is not None:
            # Encode audio features through Whisper encoder
            audio_input_lengths = None
            if audio_feature_attention_mask is not None:
                audio_input_lengths = audio_feature_attention_mask.sum(dim=-1)

            audio_encoded = self.audio_encoder(
                inputs=audio_features,
                input_lengths=audio_input_lengths,
                use_flash_attn=self.use_flash_attention,
                return_projected=False,  # We'll project manually
            )

            # Project audio features to text model hidden size
            audio_features_embedded = self.audio_feature_projector(audio_encoded)

            # Cache audio features for potential reuse
            self._audio_feature_cache = audio_features_embedded
            if audio_feature_attention_mask is not None:
                self._audio_feature_lengths = audio_feature_attention_mask.sum(dim=-1)

        # ========== 3. Audio Output Token Embedding ==========

        if audio_out_ids is not None and audio_out_mask is not None:
            # Embed audio output tokens from codebooks
            # audio_out_ids shape: [batch_size, seq_len, num_codebooks]
            batch_size_ao, seq_len_ao, num_codebooks = audio_out_ids.shape

            # Flatten for embedding lookup: [batch_size * seq_len * num_codebooks]
            audio_flat = audio_out_ids.flatten()
            audio_embedded_flat = self.audio_codebook_embeddings(audio_flat)

            # Reshape back: [batch_size, seq_len, num_codebooks, hidden_size]
            audio_embedded = audio_embedded_flat.view(
                batch_size_ao, seq_len_ao, num_codebooks, self.hidden_size
            )

            # Sum across codebooks: [batch_size, seq_len, hidden_size]
            audio_out_embedded = audio_embedded.sum(dim=2)

            # Merge with text embeddings where audio tokens are present
            audio_mask_expanded = audio_out_mask.unsqueeze(-1).expand_as(hidden_states)
            text_mask_expanded = ~audio_mask_expanded
            hidden_states = (
                hidden_states * text_mask_expanded + audio_out_embedded * audio_mask_expanded
            )

        # ========== 2.5. Merge Audio Input Features ==========

        if audio_features_embedded is not None:
            # Find audio input token positions in input_ids
            audio_input_mask = (input_ids == self.audio_in_token_idx)

            if audio_input_mask.any():
                # Get positions where audio features should be inserted
                audio_positions = torch.where(audio_input_mask)

                if len(audio_positions[0]) > 0:
                    # For simplicity, replace audio input tokens with mean-pooled audio features
                    # In a full implementation, this would be more sophisticated sequence alignment
                    audio_feature_summary = audio_features_embedded.mean(dim=1)  # [batch_size, hidden_size]

                    # Broadcast to match positions
                    for batch_idx, seq_idx in zip(audio_positions[0], audio_positions[1]):
                        if batch_idx < audio_feature_summary.shape[0]:
                            hidden_states[batch_idx, seq_idx] = audio_feature_summary[batch_idx]

        # ========== 4. Generate Position IDs and Attention Masks ==========

        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Generate attention mask for causal modeling
        if attention_mask is None:
            # Create causal attention mask with proper device placement
            attention_mask = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device)
            ).unsqueeze(0).expand(batch_size, -1, -1)

        # Handle audio-specific attention masking for different generation modes
        if self.generation_mode != GenerationMode.TEXT and audio_out_mask is not None:
            # Apply audio-specific masking logic
            # For audio generation, we may need different causal constraints
            pass

        # ========== 5. Transformer Decoder Layers ==========

        # Process through all decoder layers with dual FFN support
        presents = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            # Apply layer
            layer_output = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache_params=kv_cache_params,
                mode=self.generation_mode,
                audio_out_mask=audio_out_mask,
            )

            # Handle layer output (may include updated cache)
            if isinstance(layer_output, tuple):
                hidden_states, layer_present = layer_output
                if use_cache:
                    presents.append(layer_present)
            else:
                hidden_states = layer_output
                if use_cache:
                    presents.append(None)

        # Store updated cache
        if use_cache and presents is not None:
            self._past_key_values = presents

        # ========== 6. Layer Normalization ==========

        hidden_states = self.norm(hidden_states)

        # ========== 7. Output Projection ==========

        # Apply dual-headed projector
        text_logits, audio_logits = self.decoder_proj(
            hidden_states=hidden_states,
            mode=self.generation_mode,
        )

        # ========== 8. Gather Last Token Logits (if needed) ==========

        if last_token_ids is not None:
            text_logits = gather_last_token_logits(text_logits, last_token_ids)
            if audio_logits is not None:
                audio_logits = gather_last_token_logits(audio_logits, last_token_ids)

        return text_logits, audio_logits

    def prepare_inputs_for_generation(
        self,
        input_ids: Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare inputs for generation step.

        This method is called during incremental generation to prepare
        the inputs for the next forward pass.
        """
        # TODO: Implement input preparation logic for generation
        # This should handle KV cache updates, position IDs, etc.

        return {
            "input_ids": input_ids,
            **kwargs,
        }

    def sample(
        self,
        hidden_states: Tensor,
        logits: Tensor,
        audio_logits: Optional[Tensor] = None,
        **sampling_kwargs,
    ) -> Tensor:
        """Sample next tokens from logits.

        Args:
            hidden_states: Current hidden states
            logits: Text logits for sampling
            audio_logits: Audio logits for sampling (if in audio mode)
            **sampling_kwargs: Additional sampling parameters

        Returns:
            Next token IDs
        """
        # TODO: Implement sampling logic with support for:
        # - Text token sampling with temperature/top-k/top-p
        # - Audio token sampling with delay pattern coordination
        # - Mode-aware sampling strategies

        # For now, just do greedy sampling on text logits
        if logits.dim() == 3:  # [batch, seq_len, vocab]
            logits = logits[:, -1, :]  # Take last position

        next_tokens = torch.argmax(logits, dim=-1)
        return next_tokens
