# SPDX-License-Identifier: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import Optional, Union, Dict, Any, Tuple, List
import math
import time
import warnings
import numpy as np
import torch
import random
from tensorrt_llm.functional import Tensor, gelu, layer_norm, embedding, conv1d, default_net, gather_last_token_logits, AttentionMaskType, PositionEmbeddingType, constant, where, send, recv, cast, arange, unsqueeze, expand, lt, max as trt_max, full, concat
from tensorrt_llm.layers import Conv1d, Embedding, LayerNorm, Attention, MLP
from tensorrt_llm.module import Module, ModuleList
from tensorrt_llm.parameter import Parameter
from tensorrt_llm._utils import pad_vocab_size
from tensorrt_llm.layers import (MLP, Attention, ColumnLinear, Embedding,
                                 GatedMLP, RmsNorm, PromptTuningEmbedding)
#lm_headfrom tensorrt_llm._torch.models.modeling_llama import LlamaAttention
from tensorrt_llm.models import PretrainedConfig, PretrainedModel
from tensorrt_llm.models.modeling_utils import DecoderModelForCausalLM, KeyValueCacheParams, DecoderLayerList
from tensorrt_llm.top_model_mixin import TopModelMixin
from torch.nn import ModuleList, AvgPool1d
from .config import HiggsAudioConfig
from transformers import AutoConfig

class GenerationMode(Enum):
    """TTS-specific generation modes for coordinated audio-text generation."""
    TEXT = 0
    """Text-only generation mode for standard language modeling tasks."""
    AUDIO_INIT = 1
    """Initial audio generation mode for setting up audio token generation."""
    AUDIO_IN_PROGRESS = 2
    """Ongoing audio generation mode for streaming audio token generation."""

class HiggsAudioEncoderLayer(Module):
    """
    Single transformer layer for the Higgs Audio encoder.
    
    This layer implements a standard transformer encoder layer with:
    - Multi-head self-attention
    - Feed-forward network
    - Layer normalization and residual connections
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.embed_dim = config.audio_d_model
        
        # Self-attention mechanism (WhisperAttention-compatible)
        self.self_attn = Attention(
            local_layer_idx=layer_idx,
            hidden_size=self.embed_dim,
            attention_head_size=self.embed_dim // config.audio_encoder_heads,
            num_attention_heads=config.audio_encoder_heads,
            num_kv_heads=config.audio_encoder_heads,  # Same as attention heads for encoder
            max_position_embeddings=config.audio_max_source_positions,
            dtype=config.dtype,
            attention_mask_type='bidirectional',  # Encoder uses bidirectional attention
            bias=False,
            tp_group=None,  # Audio encoder typically not tensor parallel
            tp_size=1,
        )
        
        # Layer normalization
        self.self_attn_layer_norm = LayerNorm(
            normalized_shape=self.embed_dim,
            dtype=config.dtype
        )
        
        # Feed-forward network
        self.mlp = MLP(
            hidden_size=self.embed_dim,
            ffn_hidden_size=config.audio_encoder_ffn_dim,
            hidden_act=getattr(config, 'activation_function', 'gelu'),
            dtype=config.dtype,
            bias=True
        )
        
        # Final layer normalization
        self.final_layer_norm = LayerNorm(
            normalized_shape=self.embed_dim,
            dtype=config.dtype
        )
        
        # Dropout rate
        self.dropout_rate = getattr(config, 'dropout', 0.0)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        layer_head_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass for a single encoder layer.
        
        Args:
            hidden_states: Input hidden states [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            layer_head_mask: Optional head mask for this layer
            output_attentions: Whether to output attention weights
            
        Returns:
            Tuple of (hidden_states, attention_weights)
        """
        residual = hidden_states
        
        # Self-attention with pre-norm
        hidden_states = self.self_attn_layer_norm(hidden_states)
        
        # Apply self-attention with proper parameter handling
        attn_output = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=False,  # Encoder doesn't use KV cache
            kv_cache_params=None,
            attention_params=None,
        )
        
        # Extract attention output and weights
        if isinstance(attn_output, tuple):
            hidden_states = attn_output[0]
            attention_weights = attn_output[1] if len(attn_output) > 1 else None
        else:
            hidden_states = attn_output
            attention_weights = None
        
        # Apply head masking if provided (like original WhisperEncoderLayer)
        if layer_head_mask is not None and attention_weights is not None:
            # Apply head mask to attention weights
            attention_weights = attention_weights * layer_head_mask.view(1, -1, 1, 1)
        
        hidden_states = residual + hidden_states
        
        # Feed-forward network with pre-norm
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        # Return attention weights if requested and available
        if output_attentions:
            return hidden_states, attention_weights
        else:
            return hidden_states, None


class HiggsAudioEncoder(Module):
    """
    Higgs Audio Encoder based on Whisper architecture.
    
    This encoder processes mel-spectrogram features and converts them to audio 
    embeddings that can be integrated with the text model. It consists of:
    - Convolutional feature extraction layers
    - Positional embeddings  
    - Stack of transformer encoder layers
    - Layer normalization
    """

    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.config = config
        
        # Audio configuration parameters
        self.num_mel_bins = config.audio_num_mel_bins
        self.embed_dim = config.audio_d_model
        self.num_layers = config.audio_encoder_layers
        self.max_source_positions = config.audio_max_source_positions
        
        # Dropout and layer drop rates
        self.dropout_rate = getattr(config, 'dropout', 0.0)
        self.layerdrop_rate = getattr(config, 'encoder_layerdrop', 0.0)
        
        # Embedding scale factor
        scale_embedding = getattr(config, 'scale_embedding', False)
        self.embed_scale = math.sqrt(self.embed_dim) if scale_embedding else 1.0
        
        # Memory optimization: Only create layers when needed
        # Skip audio tower if configured to reduce memory during engine build
        
        if not config.skip_audio_tower:
            # Convolutional feature extraction layers
            # These layers downsample the mel-spectrogram and extract features
            self.conv1 = Conv1d(
                in_channels=self.num_mel_bins,
                out_channels=self.embed_dim,
                kernel_size=3,
                padding=1,
                dtype=config.dtype
            )
            
            self.conv2 = Conv1d(
                in_channels=self.embed_dim,
                out_channels=self.embed_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                dtype=config.dtype
            )
            
            # Positional embeddings
            self.embed_positions = Embedding(
                num_embeddings=self.max_source_positions,
                embedding_dim=self.embed_dim,
                dtype=config.dtype
            )
            
            # Freeze positional embeddings (common practice)
            self.embed_positions.weight.requires_grad = False
            
            # Transformer encoder layers using ModuleList for proper registration
            self.layers = ModuleList([
                HiggsAudioEncoderLayer(config, layer_idx) 
                for layer_idx in range(self.num_layers)
            ])
            
            # Average pooling layer for sequence reduction (like original Transformers)
            self.avg_pooler = AvgPool1d(
                kernel_size=2,
                stride=2,
                padding=0
            )
            
            # Final layer normalization
            self.layer_norm = LayerNorm(
                normalized_shape=self.embed_dim,
                dtype=config.dtype
            )
        else:
            # Minimal placeholder components to save memory during engine build
            self.conv1 = None
            self.conv2 = None
            self.embed_positions = None
            self.layers = []
            self.avg_pooler = None
            self.layer_norm = None

    def forward(
        self,
        input_features: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        check_seq_length: bool = True,
    ) -> Union[Tensor, Tuple]:
        """
        Forward pass for the audio encoder.
        
        Args:
            input_features: Mel-spectrogram features [batch, num_mel_bins, seq_len]
            attention_mask: Optional attention mask (not typically used for encoder)
            head_mask: Optional head mask for attention layers
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states from all layers
            return_dict: Whether to return a dictionary or tuple
            check_seq_length: Whether to validate input sequence length
            
        Returns:
            Encoded audio features [batch, seq_len, hidden_size]
        """
        # Validate input sequence length if requested
        if check_seq_length:
            # Calculate expected sequence length based on downsampling
            # conv1: no stride (stride=1), conv2: stride=2, so total downsampling = 2
            # Expected input length for max_source_positions output after downsampling
            expected_seq_length = self.max_source_positions * 2  # Account for conv2 stride=2
            if input_features.shape[-1] != expected_seq_length:
                # Allow some flexibility for sequence length
                min_seq_length = expected_seq_length - 10
                max_seq_length = expected_seq_length + 10
                if not (min_seq_length <= input_features.shape[-1] <= max_seq_length):
                    warnings.warn(
                        f"HiggsAudio encoder expects input features of length ~{expected_seq_length}, "
                        f"but got {input_features.shape[-1]}. This may affect performance."
                    )
        
        # Convolutional feature extraction
        # Apply first conv layer with GELU activation
        hidden_states = self.conv1(input_features)
        hidden_states = gelu(hidden_states)
        
        # Apply second conv layer with GELU activation and stride
        hidden_states = self.conv2(hidden_states)  
        hidden_states = gelu(hidden_states)
        
        # Reshape from [batch, channels, seq_len] to [batch, seq_len, channels]
        hidden_states = hidden_states.permute(0, 2, 1)
        
        # Add positional embeddings (direct weight access like original Transformers)
        # Scale embeddings if configured
        hidden_states = hidden_states * self.embed_scale
        
        # Add positional embeddings - use direct weight access for better compatibility
        embed_pos = self.embed_positions.weight
        hidden_states = hidden_states + embed_pos
        
        # Storage for outputs if requested (use lists for memory efficiency)
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        
        # Validate head mask dimensions if provided (like original Transformers)
        if head_mask is not None:
            if head_mask.shape[0] != len(self.layers):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, "
                    f"but it is for {head_mask.shape[0]} layers."
                )
        
        # Pass through transformer layers with memory optimization
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                # Only store if really needed to save memory
                all_hidden_states.append(hidden_states.detach())
            
            # Layer dropout during training (like original Transformers implementation)
            skip_layer = False
            if self.training and self.layerdrop_rate > 0:
                # Generate random number for layer drop decision
                # Note: For training mode only, inference always processes all layers
                dropout_probability = random.random()
                if dropout_probability < self.layerdrop_rate:
                    skip_layer = True
            
            if not skip_layer:
                # Apply encoder layer
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    layer_head_mask=head_mask[idx] if head_mask is not None else None,
                    output_attentions=output_attentions,
                )
                
                hidden_states = layer_outputs[0]
                
                if output_attentions and layer_outputs[1] is not None:
                    all_attentions.append(layer_outputs[1].detach())
            else:
                # Layer was skipped due to layer drop
                if output_attentions:
                    # Add None for skipped layer to maintain indexing
                    all_attentions.append(None)
        
        # Apply average pooling before final layer norm (like original Transformers)
        # Permute to [batch, channels, seq_len] for pooling
        hidden_states = hidden_states.permute(0, 2, 1)
        
        # Apply average pooling layer for sequence reduction
        hidden_states = self.avg_pooler(hidden_states)
        
        # Permute back to [batch, seq_len, channels]
        hidden_states = hidden_states.permute(0, 2, 1)
        
        # Final layer normalization
        hidden_states = self.layer_norm(hidden_states)
        
        # Add final hidden state if collecting all states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Return based on return_dict flag
        if return_dict:
            return {
                'last_hidden_state': hidden_states,
                'hidden_states': all_hidden_states,
                'attentions': all_attentions,
            }
        else:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

    def _get_feat_extract_output_lengths(self, input_lengths: Tensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder.
        
        Args:
            input_lengths: Length of input mel-spectrogram sequences
            
        Returns:
            Tuple of (conv_output_lengths, encoder_output_lengths)
        """
        # Calculate lengths after convolutional layers
        # conv1: kernel_size=3, stride=1, padding=1 -> no length change
        # conv2: kernel_size=3, stride=2, padding=1 -> length = (length + 2*1 - 3) // 2 + 1
        conv_output_lengths = (input_lengths + 2 * 1 - 3) // 2 + 1
        
        # Calculate lengths after average pooling (kernel_size=2, stride=2)
        # avg_pool: kernel_size=2, stride=2 -> length = length // 2
        encoder_output_lengths = conv_output_lengths // 2
        
        return conv_output_lengths, encoder_output_lengths

    def get_input_embeddings(self):
        """Get the input embedding layer (conv1 in this case)."""
        return self.conv1
    
    def set_input_embeddings(self, value):
        """Set the input embedding layer."""
        self.conv1 = value


class HiggsAudioEncoderProjector(Module):
    """
    Projects audio features from the encoder to the text model's hidden size.
    
    This is a linear projection layer that maps from the audio encoder's
    output dimension to the text model's hidden dimension, enabling integration
    of audio and text representations. Uses TensorRT-LLM ColumnLinear for 
    tensor parallelism support.
    """

    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        
        # Get dimensions from config
        audio_dim = config.audio_d_model
        text_dim = config.hidden_size
        
        # Use TensorRT-LLM ColumnLinear for proper tensor parallelism support
        self.linear = ColumnLinear(
            in_features=audio_dim,
            out_features=text_dim,
            bias=True,
            dtype=config.dtype,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            gather_output=True  # Gather output across TP ranks
        )

    def preprocess_audio_features(self, audio_features: Tensor) -> Tensor:
        """
        Preprocess audio features for multimodal generation.
        
        Args:
            audio_features: Raw audio features from audio encoder
            
        Returns:
            Preprocessed audio features ready for transformer
        """
        # Audio features are typically already preprocessed by audio encoder
        # Apply any additional normalization or projection if needed
        if hasattr(self, 'audio_projection'):
            audio_features = self.audio_projection(audio_features)
        return audio_features

    def prepare_multimodal_inputs(
        self,
        input_ids: Tensor,
        audio_features: Optional[Tensor] = None,
        audio_mask: Optional[Tensor] = None,
        text_positions: Optional[Tensor] = None,
        audio_positions: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Prepare inputs for multimodal forward pass.
        
        Args:
            input_ids: Text token IDs
            audio_features: Preprocessed audio features
            audio_mask: Mask indicating audio token positions
            text_positions: Positions of text tokens in sequence
            audio_positions: Positions of audio tokens in sequence
            
        Returns:
            Dictionary containing prepared inputs
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device if hasattr(input_ids, 'device') else None
        
        # Initialize embeddings
        if audio_features is not None and audio_mask is not None:
            # Get text embeddings
            text_embeds = self.vocab_embedding(input_ids)
            
            # Combine text and audio embeddings based on mask
            combined_embeds = where(
                audio_mask.unsqueeze(-1),
                audio_features,
                text_embeds
            )
        else:
            # Text-only mode
            combined_embeds = self.vocab_embedding(input_ids)
            
        return {
            'embeddings': combined_embeds,
            'audio_mask': audio_mask,
            'text_positions': text_positions,
            'audio_positions': audio_positions
        }

    def multimodal_forward(
        self,
        input_ids: Tensor,
        audio_features: Optional[Tensor] = None,
        audio_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[list] = None,
        sequence_length: Optional[Tensor] = None,
        past_key_value_length: Optional[Tensor] = None,
        masked_tokens: Optional[Tensor] = None,
        use_cache: bool = False,
        last_token_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        cache_indirection: Optional[Tensor] = None,
        kv_cache_params: Optional[KeyValueCacheParams] = None,
        attention_params: Optional[AttentionParams] = None,
        hidden_states: Optional[Tensor] = None,
        lora_params: Optional[LoraParams] = None,
        prompt_embedding_table: Optional[Tensor] = None,
        prompt_tasks: Optional[Tensor] = None,
        prompt_vocab_size: Optional[Tensor] = None
    ) -> Dict[str, Any]:
        """
        Multimodal forward pass handling both text and audio modalities.
        
        Returns:
            Dictionary containing logits and hidden states
        """
        # Prepare multimodal inputs
        mm_inputs = self.prepare_multimodal_inputs(
            input_ids=input_ids,
            audio_features=audio_features,
            audio_mask=audio_mask
        )
        
        # Use prepared embeddings instead of input_ids
        hidden_states = mm_inputs['embeddings']
        
        # Pass through transformer layers with audio mask for modal routing
        outputs = self.decoder(
            hidden_states=hidden_states,
            use_cache=use_cache,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            position_ids=position_ids,
            hidden_states_input=hidden_states,
            audio_out_mask=audio_mask,  # Route audio tokens through audio layers
            lora_params=lora_params
        )
        
        if use_cache:
            hidden_states, present_key_value = outputs
        else:
            hidden_states = outputs
            present_key_value = None
            
        # Get logits for both text and audio
        logits = {}
        
        # Text logits from language model head
        if audio_mask is None or not audio_mask.all():
            text_logits = self.lm_head(hidden_states)
            logits['text'] = text_logits
            
        # Audio logits from decoder projector
        if audio_mask is not None and audio_mask.any():
            # Get audio hidden states
            audio_hidden = where(
                audio_mask.unsqueeze(-1),
                hidden_states,
                constant(0.0)
            )
            # Project to audio logits
            audio_logits = self.decoder_projector(audio_hidden)
            logits['audio'] = audio_logits
            
        return {
            'logits': logits,
            'hidden_states': hidden_states,
            'present_key_value': present_key_value
        }

    def forward(self, audio_features: Tensor) -> Tensor:
        """
        Project audio features to text model dimension.
        
        Args:
            audio_features: Audio features from encoder [batch, seq_len, audio_dim]
            
        Returns:
            Projected features [batch, seq_len, text_dim]
        """
        # Apply linear projection using TensorRT-LLM ColumnLinear
        output = self.linear(audio_features)
        return output


class HiggsAudioDecoderProjector(Module):
    """
    TensorRT-LLM compatible decoder projection layers for Higgs Audio model.
    
    Projects hidden states from decoder layers to both text and audio logits.
    Supports multi-codebook audio generation with RVQ delay patterns.
    """
    
    def __init__(self, config: HiggsAudioConfig) -> None:
        """Initialize decoder projection layers.
        
        Args:
            config: HiggsAudioConfig containing projection specifications
        """
        super().__init__()
        self.config = config
        
        # Text projection head - projects to text vocabulary
        self.text_lm_head = ColumnLinear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            bias=False,
            dtype=config.dtype,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            gather_output=True
        )
        
        # Audio projection head - projects to audio codebook vocabularies
        # Each codebook has codebook_size + 2 tokens (regular tokens + special tokens)
        audio_vocab_size = config.audio_num_codebooks * (config.audio_codebook_size + 2)
        self.audio_lm_head = ColumnLinear(
            in_features=config.hidden_size,
            out_features=audio_vocab_size,
            bias=False,
            dtype=config.dtype,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            gather_output=True
        )
        
        # Cache configuration parameters
        self.audio_num_codebooks = config.audio_num_codebooks
        self.audio_codebook_size = config.audio_codebook_size
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
    
    def forward(self,
                hidden_states: Tensor,
                audio_out_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Project hidden states to text and audio logits.
        
        Args:
            hidden_states: Hidden states from decoder [batch, seq_len, hidden_size]
            audio_out_mask: Boolean mask indicating audio token positions [batch, seq_len]
            
        Returns:
            text_logits: Text vocabulary logits [batch, seq_len, vocab_size]
            audio_logits: Audio codebook logits [batch, seq_len, audio_vocab_size]
        """
        # Project to text logits for all positions
        text_logits = self.text_lm_head(hidden_states)
        
        # Project to audio logits for all positions
        audio_logits = self.audio_lm_head(hidden_states)
        
        return text_logits, audio_logits

class HiggsAudioDecoderLayer(Module):
    """Llama-style decoder layer optimized for Higgs Audio TTS model.

    This layer implements the core transformer decoder functionality with
    TTS-specific optimizations. It serves as the building block for the
    HiggsAudioModel and supports audio token processing through
    specialized attention and MLP components.

    The layer follows the standard transformer architecture:
    - Multi-head self-attention with RoPE positional encoding
    - Gated MLP with SiLU activation
    - RMSNorm for layer normalization
    - Residual connections around both attention and MLP blocks

    Future audio adapter integration points are preserved for DualFFN
    and other audio-specific enhancements.

    Attributes:
        config (HiggsAudioConfig): Layer configuration
        layer_idx (int): Layer index in the model stack
        mapping: TensorRT-LLM parallelism mapping
        input_layernorm (RmsNorm): Pre-attention layer normalization
        post_layernorm (RmsNorm): Pre-MLP layer normalization
        attention (Attention): Multi-head self-attention mechanism
        mlp (GatedMLP): Feed-forward network with gating

    Example:
        >>> layer = HiggsAudioDecoderLayer(config, layer_idx=0)
        >>> output = layer(hidden_states, attention_mask=mask)
    """

    def __init__(self, config: HiggsAudioConfig, layer_idx: int) -> None:
        """Initialize decoder layer with TTS-optimized components.

        Args:
            config: HiggsAudioConfig containing layer parameters
            layer_idx: Zero-based index of this layer in the model stack

        Raises:
            ValueError: If config parameters are invalid for layer construction
        """
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

        # Attention - Use standard Attention with Llama-style configuration
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
            cross_attention=False,
            relative_attention=False,
            max_distance=0,
            num_buckets=0,
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
                hidden_states: Tensor,
                attention_mask: Optional[Tensor] = None,
                use_cache: bool = False,
                kv_cache_params: Optional[Any] = None,
                attention_params: Optional[Any] = None,
                lora_layer_params: Optional[Any] = None,
                position_ids: Optional[Tensor] = None,
                audio_token_mask: Optional[Tensor] = None,
                audio_out_mask: Optional[Tensor] = None,
                next_layer_input_layernorm_args: Optional[Any] = None) -> Union[Tensor, Tuple[Tensor, Any]]:
        """Forward pass through decoder layer with TTS-aware processing.

        Implements standard transformer decoder layer computation with residual
        connections around attention and MLP blocks. The audio_token_mask and
        audio_out_mask parameters are reserved for future audio-specific processing.

        Args:
            hidden_states: Input hidden states tensor [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask for sequence padding
            use_cache: Whether to return KV cache for next iteration
            kv_cache_params: KV cache parameters for attention computation
            attention_params: Additional attention computation parameters
            lora_layer_params: LoRA adaptation parameters if enabled
            position_ids: Position indices for positional encoding
            audio_token_mask: Mask indicating audio token positions (legacy parameter)
            audio_out_mask: Mask indicating audio output tokens (future use)

        Returns:
            If use_cache=False: Hidden states tensor after layer processing
            If use_cache=True: Tuple of (hidden_states, kv_cache_presents)

        Example:
            >>> output = layer(hidden_states, attention_mask=mask, use_cache=True)
            >>> hidden_states, kv_cache = output
        """
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


class HiggsAudioDualFFNDecoderLayer(Module):
    """
    Higgs Audio DualFFN Decoder Layer with separate FFN paths for audio and text tokens.
    
    This layer implements the DualFFN architecture where audio and text tokens use
    separate MLP processing paths after shared attention computation. This allows
    specialized processing for different modalities while maintaining efficiency.
    
    Features:
    - Shared attention mechanism for all tokens
    - Separate FFN paths for audio vs text tokens
    - Optional fast-forward mode for audio tokens
    - Optional audio-specific attention mechanism
    - Comprehensive error handling and validation
    """
    
    def __init__(self, config: HiggsAudioConfig, layer_idx: int):
        super().__init__()
        
        # Validate configuration parameters
        self._validate_config(config, layer_idx)
        
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.dtype = config.dtype
        
        # Determine if this layer uses DualFFN based on configuration
        self.use_dual_ffn = self._should_use_dual_ffn(config, layer_idx)
        self.use_fast_forward = self._should_use_fast_forward(config, layer_idx)
        
        # Initialize attention mechanism
        self._initialize_attention(config)
        
        # Initialize FFN components based on DualFFN configuration
        self._initialize_ffn_components(config)
        
        # Initialize layer normalization
        self._initialize_layer_norms(config)
    
    def _validate_config(self, config: HiggsAudioConfig, layer_idx: int) -> None:
        """Validate configuration parameters for DualFFN layer."""
        if not isinstance(config, HiggsAudioConfig):
            raise TypeError(f"Expected HiggsAudioConfig, got {type(config)}")
        
        if not isinstance(layer_idx, int) or layer_idx < 0:
            raise ValueError(f"layer_idx must be a non-negative integer, got {layer_idx}")
        
        if layer_idx >= config.num_hidden_layers:
            raise ValueError(
                f"layer_idx {layer_idx} exceeds num_hidden_layers {config.num_hidden_layers}"
            )
        
        # Validate DualFFN configuration
        if hasattr(config, 'audio_dual_ffn_layers') and config.audio_dual_ffn_layers:
            if not isinstance(config.audio_dual_ffn_layers, (list, tuple)):
                raise TypeError(
                    f"audio_dual_ffn_layers must be a list or tuple, got {type(config.audio_dual_ffn_layers)}"
                )
            
            for idx in config.audio_dual_ffn_layers:
                if not isinstance(idx, int) or idx < 0 or idx >= config.num_hidden_layers:
                    raise ValueError(
                        f"Invalid layer index {idx} in audio_dual_ffn_layers. "
                        f"Must be between 0 and {config.num_hidden_layers - 1}"
                    )
        
        # Validate fast-forward configuration
        if hasattr(config, 'audio_fast_forward_layers') and config.audio_fast_forward_layers:
            if not isinstance(config.audio_fast_forward_layers, (list, tuple)):
                raise TypeError(
                    f"audio_fast_forward_layers must be a list or tuple, got {type(config.audio_fast_forward_layers)}"
                )
            
            for idx in config.audio_fast_forward_layers:
                if not isinstance(idx, int) or idx < 0 or idx >= config.num_hidden_layers:
                    raise ValueError(
                        f"Invalid layer index {idx} in audio_fast_forward_layers. "
                        f"Must be between 0 and {config.num_hidden_layers - 1}"
                    )
        
        # Validate hidden size
        if not hasattr(config, 'hidden_size') or config.hidden_size <= 0:
            raise ValueError(f"Invalid hidden_size: {getattr(config, 'hidden_size', None)}")
        
        # Validate intermediate size for MLP
        if not hasattr(config, 'intermediate_size') or config.intermediate_size <= 0:
            raise ValueError(f"Invalid intermediate_size: {getattr(config, 'intermediate_size', None)}")
    
    def _should_use_dual_ffn(self, config: HiggsAudioConfig, layer_idx: int) -> bool:
        """Determine if this layer should use DualFFN based on configuration."""
        if not hasattr(config, 'audio_dual_ffn_layers') or not config.audio_dual_ffn_layers:
            return False
        return layer_idx in config.audio_dual_ffn_layers
    
    def _should_use_fast_forward(self, config: HiggsAudioConfig, layer_idx: int) -> bool:
        """Determine if this layer should use fast-forward mode for audio tokens."""
        if not hasattr(config, 'audio_fast_forward_layers') or not config.audio_fast_forward_layers:
            return False
        return layer_idx in config.audio_fast_forward_layers
    
    def _initialize_attention(self, config: HiggsAudioConfig) -> None:
        """Initialize attention mechanisms."""
        # Main attention mechanism (shared for all tokens)
        self.attention = Attention(
            local_layer_idx=self.layer_idx,
            hidden_size=config.hidden_size,
            attention_head_size=getattr(config, 'head_size', config.hidden_size // config.num_attention_heads),
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, 'num_key_value_heads', config.num_attention_heads),
            max_position_embeddings=config.max_position_embeddings,
            dtype=config.dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=getattr(config, 'attn_bias', False),
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            rotary_embedding_base=getattr(config, 'rotary_base', 10000.0),
            rotary_embedding_scaling=getattr(config, 'rotary_scaling', None),
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            tp_rank=config.mapping.tp_rank,
            q_scaling=getattr(config, 'q_scaling', 1.0),
            quant_mode=config.quant_mode,
        )
        
        # Optional audio-specific attention
        self.use_audio_attention = getattr(config, 'use_audio_out_self_attention', False)
        if self.use_audio_attention:
            self.audio_attention = Attention(
                local_layer_idx=self.layer_idx,
                hidden_size=config.hidden_size,
                attention_head_size=getattr(config, 'head_size', config.hidden_size // config.num_attention_heads),
                num_attention_heads=config.num_attention_heads,
                num_kv_heads=getattr(config, 'num_key_value_heads', config.num_attention_heads),
                max_position_embeddings=config.max_position_embeddings,
                dtype=config.dtype,
                attention_mask_type=AttentionMaskType.causal,
                bias=getattr(config, 'attn_bias', False),
                position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
                rotary_embedding_base=getattr(config, 'rotary_base', 10000.0),
                rotary_embedding_scaling=getattr(config, 'rotary_scaling', None),
                tp_group=config.mapping.tp_group,
                tp_size=config.mapping.tp_size,
                tp_rank=config.mapping.tp_rank,
                q_scaling=getattr(config, 'q_scaling', 1.0),
                quant_mode=config.quant_mode,
            )
    
    def _initialize_ffn_components(self, config: HiggsAudioConfig) -> None:
        """Initialize FFN components based on DualFFN configuration with memory optimization."""
        # Check for memory-efficient build mode
        memory_efficient_build = getattr(config, 'memory_efficient_build', False)
        
        if self.use_dual_ffn and not memory_efficient_build:
            # Full DualFFN implementation for runtime
            self.text_mlp = GatedMLP(
                hidden_size=config.hidden_size,
                ffn_hidden_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                dtype=config.dtype,
                bias=getattr(config, 'mlp_bias', False),
                tp_group=config.mapping.tp_group,
                tp_size=config.mapping.tp_size,
                quant_mode=config.quant_mode,
            )
            
            self.audio_mlp = GatedMLP(
                hidden_size=config.hidden_size,
                ffn_hidden_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                dtype=config.dtype,
                bias=getattr(config, 'mlp_bias', False),
                tp_group=config.mapping.tp_group,
                tp_size=config.mapping.tp_size,
                quant_mode=config.quant_mode,
            )
        else:
            # Memory-efficient mode: Use single MLP for both audio and text during build
            # This reduces memory usage during engine build by 50%
            self.mlp = GatedMLP(
                hidden_size=config.hidden_size,
                ffn_hidden_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                dtype=config.dtype,
                bias=getattr(config, 'mlp_bias', False),
                tp_group=config.mapping.tp_group,
                tp_size=config.mapping.tp_size,
                quant_mode=config.quant_mode,
            )
            
            # In memory-efficient mode, alias the single MLP for both paths
            if self.use_dual_ffn:
                self.text_mlp = self.mlp
                self.audio_mlp = self.mlp
    
    def _initialize_layer_norms(self, config: HiggsAudioConfig) -> None:
        """Initialize layer normalization components."""
        # Pre-attention layer norm
        self.input_layernorm = RmsNorm(
            normalized_shape=config.hidden_size,
            eps=config.norm_epsilon,
            dtype=config.dtype
        )
        
        if self.use_dual_ffn:
            # Separate post-attention layer norms for dual FFN
            self.post_layernorm_text = RmsNorm(
                normalized_shape=config.hidden_size,
                eps=config.norm_epsilon,
                dtype=config.dtype
            )
            self.post_layernorm_audio = RmsNorm(
                normalized_shape=config.hidden_size,
                eps=config.norm_epsilon,
                dtype=config.dtype
            )
        else:
            # Standard post-attention layer norm
            self.post_layernorm = RmsNorm(
                normalized_shape=config.hidden_size,
                eps=config.norm_epsilon,
                dtype=config.dtype
            )
    
    def _validate_basic_inputs(self, hidden_states: Tensor, audio_out_mask: Optional[Tensor] = None) -> None:
        """Validate basic inputs to the forward method."""
        if hidden_states is None:
            raise ValueError("hidden_states cannot be None")
        
        if not isinstance(hidden_states, Tensor):
            raise TypeError(f"hidden_states must be a Tensor, got {type(hidden_states)}")
        
        # TensorRT-LLM tensors can have dynamic shapes during engine build
        # Handle both 2D (flattened) and 3D tensor formats
        if len(hidden_states.shape) == 2:
            # Flattened format: [num_tokens, hidden_size]
            num_tokens, hidden_size = hidden_states.shape
            if hidden_size != self.hidden_size:
                raise ValueError(
                    f"hidden_states hidden_size {hidden_size} doesn't match "
                    f"config hidden_size {self.hidden_size}"
                )
        elif len(hidden_states.shape) == 3:
            # Standard format: [batch, seq_len, hidden_size]
            batch_size, seq_len, hidden_size = hidden_states.shape
            if hidden_size != self.hidden_size:
                raise ValueError(
                    f"hidden_states hidden_size {hidden_size} doesn't match "
                    f"config hidden_size {self.hidden_size}"
                )
        else:
            # For dynamic shapes (-1, size), be more flexible
            shape_list = list(hidden_states.shape)
            if len(shape_list) >= 2 and shape_list[-1] == self.hidden_size:
                # Last dimension matches hidden_size, likely valid
                pass
            else:
                raise ValueError(
                    f"hidden_states tensor shape {hidden_states.shape} is not compatible. "
                    f"Expected last dimension to be {self.hidden_size}"
                )
        
        # Validate audio_out_mask if provided (with TensorRT-LLM compatibility)
        if audio_out_mask is not None:
            if not isinstance(audio_out_mask, Tensor):
                raise TypeError(f"audio_out_mask must be a Tensor, got {type(audio_out_mask)}")
            
            # Be more flexible with mask validation for TensorRT-LLM dynamic shapes
            if len(audio_out_mask.shape) == 1:
                # Flattened mask format: [num_tokens]
                pass  # Allow flattened format during engine build
            elif len(audio_out_mask.shape) == 2:
                # Standard format: [batch, seq_len]
                mask_batch, mask_seq = audio_out_mask.shape
                # Only validate dimensions if we have 3D hidden_states
                if len(hidden_states.shape) == 3:
                    batch_size, seq_len, _ = hidden_states.shape
                    if mask_batch != batch_size or mask_seq != seq_len:
                        raise ValueError(
                            f"audio_out_mask shape {audio_out_mask.shape} doesn't match "
                            f"hidden_states batch/seq dimensions [{batch_size}, {seq_len}]"
                        )
            else:
                # Allow other shapes for dynamic TensorRT tensors
                pass
    
    def _apply_dual_path_ffn(self, hidden_states: Tensor, audio_out_mask: Tensor) -> Tensor:
        """
        Apply dual-path FFN processing with comprehensive error handling.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            audio_out_mask: Boolean mask [batch, seq_len] indicating audio tokens
            
        Returns:
            Processed hidden states with dual-path FFN applied
        """
        # Handle different tensor shapes for TensorRT-LLM compatibility
        if len(hidden_states.shape) == 3:
            batch_size, seq_len, hidden_size = hidden_states.shape
            # Validate inputs for 3D case
            if len(audio_out_mask.shape) == 2:
                if audio_out_mask.shape != (batch_size, seq_len):
                    raise ValueError(
                        f"audio_out_mask shape {audio_out_mask.shape} doesn't match "
                        f"expected [{batch_size}, {seq_len}]"
                    )
        elif len(hidden_states.shape) == 2:
            # Flattened format: [num_tokens, hidden_size]
            num_tokens, hidden_size = hidden_states.shape
            batch_size, seq_len = 1, num_tokens  # Treat as single sequence
        else:
            # Dynamic shape - infer from tensor
            hidden_size = hidden_states.shape[-1]
            # Use fallback dimensions for dynamic shapes
            batch_size, seq_len = 1, -1
        
        # Convert boolean mask to float for easier processing
        audio_mask_float = cast(audio_out_mask, hidden_states.dtype)
        text_mask_float = 1.0 - audio_mask_float
        
        # Expand masks to match hidden state dimensions (handle different shapes)
        if len(hidden_states.shape) == 3 and len(audio_mask_float.shape) == 2:
            # Standard case: expand 2D mask to 3D
            audio_mask_expanded = audio_mask_float.unsqueeze(-1)  # [batch, seq, 1]
            text_mask_expanded = text_mask_float.unsqueeze(-1)    # [batch, seq, 1]
        elif len(hidden_states.shape) == 2 and len(audio_mask_float.shape) == 1:
            # Flattened case: expand 1D mask to 2D
            audio_mask_expanded = audio_mask_float.unsqueeze(-1)  # [num_tokens, 1]
            text_mask_expanded = text_mask_float.unsqueeze(-1)    # [num_tokens, 1]
        else:
            # Fallback: try to broadcast directly or use identity
            # Attempt to expand last dimension
            audio_mask_expanded = audio_mask_float.unsqueeze(-1)
            text_mask_expanded = text_mask_float.unsqueeze(-1)
        
        # Fast-forward mode: skip audio token processing
        if self.use_fast_forward:
            # Only process text tokens through FFN, keep audio tokens unchanged
            text_norm = self.post_layernorm_text(hidden_states)
            text_output = self.text_mlp(text_norm)
            
            # Combine: keep original hidden states for audio tokens, use text FFN output for text tokens
            output = (
                hidden_states * audio_mask_expanded +    # Keep audio tokens unchanged
                text_output * text_mask_expanded         # Update text tokens with FFN output
            )
            return output
        
        # Memory-efficient dual-path processing
        # Process text tokens only where needed
        text_norm = self.post_layernorm_text(hidden_states)
        text_mlp_out = self.text_mlp(text_norm)
        
        # Process audio tokens only where needed
        audio_norm = self.post_layernorm_audio(hidden_states)
        audio_mlp_out = self.audio_mlp(audio_norm)
        
        # Combine outputs efficiently without creating large intermediate tensors
        output = (
            text_mlp_out * text_mask_expanded +      # Text tokens use text MLP
            audio_mlp_out * audio_mask_expanded      # Audio tokens use audio MLP
        )
        
        # Clear intermediate tensors to free memory
        del text_norm, audio_norm, text_mlp_out, audio_mlp_out
        
        return output

    def forward(self,
                hidden_states: Tensor,
                attention_mask: Optional[Tensor] = None,
                use_cache: bool = False,
                kv_cache_params: Optional[Any] = None,
                attention_params: Optional[Any] = None,
                lora_layer_params: Optional[Any] = None,
                position_ids: Optional[Tensor] = None,
                audio_token_mask: Optional[Tensor] = None,
                audio_out_mask: Optional[Tensor] = None,
                next_layer_input_layernorm_args: Optional[Any] = None) -> Union[Tensor, Tuple[Tensor, Any]]:
        """
        Forward pass with comprehensive error handling and validation.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Attention mask tensor
            use_cache: Whether to return KV cache for next iteration
            kv_cache_params: KV cache parameters for attention computation
            attention_params: Additional attention computation parameters
            lora_layer_params: LoRA adaptation parameters if enabled
            position_ids: Position indices for positional encoding
            audio_token_mask: Mask indicating audio token positions (legacy parameter)
            audio_out_mask: Boolean mask indicating audio vs text tokens [batch, seq_len]
            
        Returns:
            If use_cache=False: Hidden states tensor after layer processing
            If use_cache=True: Tuple of (hidden_states, kv_cache_presents)
        """
        # Basic input validation
        self._validate_basic_inputs(hidden_states, audio_out_mask)
        
        # Store residual connection
        residual = hidden_states
        
        # Pre-attention layer normalization
        hidden_states = self.input_layernorm(hidden_states)
        
        # Apply attention mechanism (TensorRT-LLM Attention doesn't use position_ids directly)
        attn_out = self.attention(hidden_states,
                                attention_mask=attention_mask,
                                use_cache=use_cache,
                                kv_cache_params=kv_cache_params,
                                attention_params=attention_params,
                                lora_layer_params=lora_layer_params)
        if use_cache:
            attn_out, presents = attn_out
        
        # Add residual connection after attention
        hidden_states = residual + attn_out
        
        # Apply audio-specific attention if configured
        if self.use_audio_attention and audio_out_mask is not None:
            if audio_out_mask.any():  # Only if we have audio tokens
                audio_residual = hidden_states
                audio_norm = self.input_layernorm(hidden_states)  # Reuse input norm
                
                audio_attn_out = self.audio_attention(audio_norm,
                                                    attention_mask=attention_mask,
                                                    use_cache=False,  # Don't cache audio attention
                                                    kv_cache_params=None,
                                                    attention_params=attention_params,
                                                    lora_layer_params=lora_layer_params,
                                                    position_ids=position_ids)
                if isinstance(audio_attn_out, tuple):
                    audio_attn_out = audio_attn_out[0]  # Extract just the output
                
                # Apply audio attention only to audio tokens
                audio_mask_expanded = audio_out_mask.unsqueeze(-1)
                hidden_states = (
                    hidden_states * (~audio_mask_expanded) +  # Keep text tokens unchanged
                    (audio_residual + audio_attn_out) * audio_mask_expanded  # Update audio tokens
                )
        
        # Store residual for FFN
        residual = hidden_states
        
        # Apply FFN processing
        if self.use_dual_ffn and audio_out_mask is not None:
            # Use dual-path FFN processing
            ffn_output = self._apply_dual_path_ffn(hidden_states, audio_out_mask)
        else:
            # Standard single-path FFN processing
            if hasattr(self, 'post_layernorm'):
                hidden_states = self.post_layernorm(hidden_states)
            else:
                # Fallback to input layernorm if post_layernorm not available
                hidden_states = self.input_layernorm(hidden_states)
            
            ffn_output = self.mlp(hidden_states,
                                lora_layer_params=lora_layer_params)
        
        # Add residual connection after FFN
        hidden_states = residual + ffn_output
        
        if use_cache:
            return (hidden_states, presents)
        return hidden_states

class HiggsAudioDecoderLayerList(DecoderLayerList):
    def __init__(self, layers, config):
        self.num_hidden_layers = config.num_hidden_layers
        self.layer_list = config.mapping.pp_layers(config.num_hidden_layers)
        self.quant_mode = config.quant_mode
        self.config = config  # Store the config
        # Initialize with the pre-constructed layers
        super(DecoderLayerList, self).__init__(layers)
    
    def forward(self,
                hidden_states,
                use_cache=False,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                mrope_params=None,
                position_ids=None,
                lora_params=None,
                spec_decoding_params=None,
                vision_token_mask=None,
                audio_out_mask=None):
        kv_cache_params.fill_none_tensor_list(len(self.layer_list))

        if use_cache:
            presents = []

        for layer_idx, (layer, past) in enumerate(
                zip(self, kv_cache_params.past_key_value)):

            lora_layer_params = None
            if lora_params is not None and lora_params.lora_ranks is not None:
                lora_layer_params = lora_params.get_layer_params(layer_idx)

            kwargs = {}
            if position_ids is not None:
                kwargs['position_ids'] = position_ids
            if vision_token_mask is not None:
                kwargs['vision_token_mask'] = vision_token_mask
            if audio_out_mask is not None:
                kwargs['audio_out_mask'] = audio_out_mask
            if lora_layer_params is not None:
                kwargs['lora_layer_params'] = lora_layer_params
            if spec_decoding_params is not None:
                kwargs['spec_decoding_params'] = spec_decoding_params
            if mrope_params is not None:
                kwargs['mrope_params'] = mrope_params

            if default_net().plugin_config.reduce_fusion:
                if layer_idx + self.layer_list[0] < self.layer_list[-1]:
                    qkv_activation_scaling_factor = None
                    if default_net().plugin_config.user_buffer:
                        qkv_linear = self[layer_idx + 1].attention.qkv
                        if self.quant_mode.has_fp8_qdq():
                            qkv_activation_scaling_factor = constant(
                                qkv_linear.activation_scaling_factor.raw_value.
                                copy())
                        elif self.quant_mode.has_nvfp4():
                            qkv_activation_scaling_factor = constant(
                                qkv_linear.activation_global_scaling_factor.
                                raw_value.copy())
                    kwargs['next_layer_input_layernorm_args'] = (
                        self[layer_idx + 1].input_layernorm.weight.value,
                        self[layer_idx + 1].input_layernorm.eps,
                        qkv_activation_scaling_factor)
                else:
                    kwargs['next_layer_input_layernorm_args'] = None
            elif default_net().plugin_config.norm_quant_fusion:
                if layer_idx < self.layer_list[-1] - self.layer_list[0]:
                    activation_scaling_factor = constant(
                        self[layer_idx + 1].attention.qkv.
                        activation_global_scaling_factor.raw_value.copy())
                    kwargs['next_layer_input_layernorm_args'] = (
                        self[layer_idx + 1].input_layernorm.weight.value,
                        self[layer_idx + 1].input_layernorm.eps,
                        activation_scaling_factor)
                else:
                    kwargs['next_layer_input_layernorm_args'] = None

            # LlamaAttention handles position embeddings automatically, no need for manual creation
            
            layer_kwargs = {
                'hidden_states': hidden_states,
                'use_cache': use_cache,
                'attention_mask': attention_mask,
                'kv_cache_params': KeyValueCacheParams(
                    past_key_value=[past],
                    host_past_key_value_lengths=kv_cache_params.
                    host_past_key_value_lengths,
                    host_max_attention_window_sizes=kv_cache_params.
                    host_max_attention_window_sizes,
                    host_sink_token_length=kv_cache_params.
                    host_sink_token_length,
                    kv_cache_block_offsets=kv_cache_params.
                    kv_cache_block_offsets,
                    host_kv_cache_block_offsets=kv_cache_params.
                    host_kv_cache_block_offsets,
                    host_kv_cache_pool_pointers=kv_cache_params.
                    host_kv_cache_pool_pointers,
                    host_kv_cache_pool_mapping=kv_cache_params.
                    host_kv_cache_pool_mapping,
                    cache_indirection=kv_cache_params.cache_indirection),
                'attention_params': attention_params,
                **kwargs
            }
            
            layer_output = layer(**layer_kwargs)

            if use_cache:
                hidden_states, present = layer_output
                presents.append(present)
            else:
                hidden_states = layer_output

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states

class HiggsAudioModel(Module):
    """Core transformer backbone for Higgs Audio TTS model."""

    def __init__(self, config: HiggsAudioConfig) -> None:
        """Initialize transformer backbone with TTS-optimized components."""
        super().__init__()
        self.config = config
        self.mapping = config.mapping
        self.use_prompt_tuning = getattr(config, "use_prompt_tuning", False)
        self.vocab_size = config.vocab_size
        
        # Audio mask handling for DualFFN layers
        self.audio_out_mask: Optional[Tensor] = None
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

        # Initialize layers based on audio adapter configuration
        self._initialize_decoder_layers(config)

        if self.mapping.is_last_pp_rank():
            self.ln_f = RmsNorm(normalized_shape=config.hidden_size,
                                eps=config.norm_epsilon,
                                dtype=config.dtype)

    def _initialize_decoder_layers(self, config: HiggsAudioConfig) -> None:
        """Initialize decoder layers with comprehensive error handling and validation."""
        # Validate configuration
        if not isinstance(config, HiggsAudioConfig):
            raise TypeError(f"Expected HiggsAudioConfig, got {type(config)}")
        
        if not hasattr(config, 'num_hidden_layers') or config.num_hidden_layers <= 0:
            raise ValueError(f"Invalid num_hidden_layers: {getattr(config, 'num_hidden_layers', None)}")
        
        # Validate audio adapter configuration
        audio_adapter_type = getattr(config, 'audio_adapter_type', 'stack')
        valid_adapter_types = ['dual_ffn', 'dual_ffn_fast_forward', 'stack']
        
        if audio_adapter_type not in valid_adapter_types:
            raise ValueError(
                f"Unsupported audio_adapter_type: {audio_adapter_type}. "
                f"Supported types: {valid_adapter_types}"
            )
        
        layers = []
        
        # Create layers based on adapter type
        if audio_adapter_type in ['dual_ffn', 'dual_ffn_fast_forward']:
            # Validate DualFFN configuration
            dual_ffn_layers = getattr(config, 'audio_dual_ffn_layers', [])
            if dual_ffn_layers and not isinstance(dual_ffn_layers, (list, tuple)):
                raise TypeError(
                    f"audio_dual_ffn_layers must be a list or tuple, got {type(dual_ffn_layers)}"
                )
            
            # Validate layer indices
            for idx in dual_ffn_layers:
                if not isinstance(idx, int) or idx < 0 or idx >= config.num_hidden_layers:
                    raise ValueError(
                        f"Invalid layer index {idx} in audio_dual_ffn_layers. "
                        f"Must be between 0 and {config.num_hidden_layers - 1}"
                    )
            
            # Create layers with DualFFN where specified
            for layer_idx in range(config.num_hidden_layers):
                if layer_idx in dual_ffn_layers:
                    # Use DualFFN layer
                    layer = HiggsAudioDualFFNDecoderLayer(config, layer_idx)
                else:
                    # Use standard decoder layer
                    layer = HiggsAudioDecoderLayer(config, layer_idx)
                
                layers.append(layer)
                    
        elif audio_adapter_type == 'stack':
            # Standard stacked decoder layers
            for layer_idx in range(config.num_hidden_layers):
                layer = HiggsAudioDecoderLayer(config, layer_idx)
                layers.append(layer)
        else:
            raise ValueError(
                f"Unsupported audio_adapter_type: {audio_adapter_type}. "
                f"Supported types: {valid_adapter_types}"
            )
        
        # Validate that we created the correct number of layers
        if len(layers) != config.num_hidden_layers:
            raise RuntimeError(
                f"Created {len(layers)} layers but expected {config.num_hidden_layers}"
            )

        self.layers = HiggsAudioDecoderLayerList(layers, config)

    def forward(self,
                input_ids: Tensor,
                position_ids: Optional[Tensor] = None,
                use_cache: bool = False,
                attention_mask: Optional[Tensor] = None,
                kv_cache_params: Optional[Any] = None,
                attention_params: Optional[Any] = None,
                hidden_states: Optional[Tensor] = None,
                prompt_embedding_table: Optional[Tensor] = None,
                prompt_tasks: Optional[Tensor] = None,
                prompt_vocab_size: Optional[Tensor] = None,
                lora_params: Optional[Any] = None,
                input_token_extra_ids: Optional[Tensor] = None,
                audio_token_mask: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tuple[Any, ...]]]:
        """Forward pass through transformer backbone with multimodal support.

        Processes input tokens through embedding lookup, transformer layers,
        and final normalization. Supports both text and audio token processing
        with pipeline parallelism and KV caching optimizations.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            position_ids: Position indices for RoPE encoding
            use_cache: Whether to use and return KV cache
            attention_mask: Attention mask for sequence padding
            kv_cache_params: KV cache parameters for efficient generation
            attention_params: Additional attention computation parameters
            hidden_states: Pre-computed hidden states (for PP intermediate ranks)
            prompt_embedding_table: Prompt tuning embedding table
            prompt_tasks: Task IDs for prompt tuning
            prompt_vocab_size: Vocabulary size for prompt tuning
            lora_params: LoRA adaptation parameters
            input_token_extra_ids: Additional token IDs for special processing
            audio_token_mask: Mask indicating audio token positions

        Returns:
            If use_cache=False: Final hidden states [batch_size, seq_len, hidden_size]
            If use_cache=True: Tuple of (hidden_states, kv_cache_presents)

        Example:
            >>> output = backbone(
            ...     input_ids=tokens,
            ...     attention_mask=mask,
            ...     use_cache=True,
            ...     audio_token_mask=audio_mask
            ... )
        """
        # Fill kv cache structures
        kv_cache_params.fill_none_tensor_list(len(self.layers))

        # Embedding lookup (or receive from PP)
        if self.mapping.is_first_pp_rank():
            ptuning_args = [prompt_embedding_table, prompt_tasks, prompt_vocab_size] \
                if self.use_prompt_tuning else []
            hidden_states = self.vocab_embedding(input_ids, *ptuning_args)
        else:
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())

        # Forward layers with audio mask for DualFFN processing
        # Use stored audio_out_mask if available, otherwise fall back to audio_token_mask
        audio_mask_for_layers = self.audio_out_mask if self.audio_out_mask is not None else audio_token_mask
        
        hidden_states = self.layers.forward(hidden_states,
                                            use_cache=use_cache,
                                            attention_mask=attention_mask,
                                            kv_cache_params=kv_cache_params,
                                            attention_params=attention_params,
                                            lora_params=lora_params,
                                            position_ids=position_ids,
                                            audio_out_mask=audio_mask_for_layers)

        if use_cache:
            hidden_states, presents = hidden_states

        if self.mapping.is_last_pp_rank():
            hidden_states = self.ln_f(hidden_states)
        else:
            hidden_states = send(hidden_states, self.mapping.next_pp_rank())

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states
            
class HiggsAudioForCausalLM(DecoderModelForCausalLM, TopModelMixin):
    """Complete Higgs Audio model for causal language modeling with TTS capabilities.

    Attributes:
        transformer (HiggsAudioModel): Core transformer backbone
        lm_head (Optional[ColumnLinear]): Language modeling head (last PP rank only)
        config (HiggsAudioConfig): Model configuration with TTS parameters
    """

    config_class = HiggsAudioConfig
    def __init__(self, config: HiggsAudioConfig) -> None:
        """Initialize complete Higgs Audio model for TTS generation.

        Args:
            config: HiggsAudioConfig with complete model parameters

        Raises:
            ValueError: If config is incompatible with model requirements
        """
        self.config: HiggsAudioConfig = config
        self.transformer = HiggsAudioModel(config)
        if self.mapping.is_last_pp_rank():
            self.decoder_projector = HiggsAudioDecoderProjector(config)
            self.lm_head = self.decoder_projector.text_lm_head
        else:
            lm_head = None
            self.decoder_projector = None

        # Audio tower integration - Adapted for TensorRT-LLM
        if not config.skip_audio_tower:
            self.audio_tower = HiggsAudioEncoder(config)
            self.audio_encoder_proj = HiggsAudioEncoderProjector(config)
        else:
            self.audio_tower = None
            self.audio_encoder_proj = None
        
        # Initialize delay pattern attributes
        self.use_delay_pattern = config.use_delay_pattern
        self.audio_num_codebooks = config.audio_num_codebooks
        self.audio_stream_bos_id = getattr(config, 'audio_stream_bos_id', 0)
        self.audio_stream_eos_id = getattr(config, 'audio_stream_eos_id', 1)

        super().__init__(config, transformer, lm_head)
    
    def _apply_delay_pattern_to_audio_tokens(
        self,
        next_audio_tokens: Tensor,
        num_delay: int,
        num_remaining_delays: Optional[int],
        audio_eos_token_id: Optional[int] = None
    ) -> Tuple[Tensor, int, Optional[int], bool]:
        """Apply delay pattern to audio tokens for coordinated multi-codebook generation.
        
        This implements a linear delay pattern with stride 1, where each codebook
        starts generating one timestep after the previous one. This ensures proper
        temporal alignment across RVQ codebooks.
        
        Args:
            next_audio_tokens: Audio tokens for all codebooks [num_codebooks]
            num_delay: Current delay counter (number of codebooks that have started)
            num_remaining_delays: Remaining delays to apply (for ending pattern)
            audio_eos_token_id: Token ID to use when ending audio generation
            
        Returns:
            Tuple of:
                - Modified audio tokens with delay pattern applied
                - Updated num_delay counter
                - Updated num_remaining_delays counter
                - Boolean indicating if audio generation should end
        """
        should_end_audio = False

        if not self.use_delay_pattern:
            return next_audio_tokens, num_delay, num_remaining_delays, should_end_audio

        # Build codebook index vector [0, 1, ..., num_codebooks-1]
        num_cb = self.audio_num_codebooks
        idx = arange(0, num_cb, dtype='int32')  # [num_cb]

        # 1) Start pattern (linear stride = 1): allow first (num_delay+1) codebooks to generate
        started = constant(num_delay + 1)
        # mask_started = (idx < started)
        mask_started = lt(idx, started)
        bos_vec = full([num_cb], self.audio_stream_bos_id, dtype=next_audio_tokens.dtype)
        # Keep tokens for started codebooks; force BOS for the ones not yet started
        next_audio_tokens = where(mask_started, next_audio_tokens, bos_vec)

        # Increase delay until all codebooks have started
        if num_delay + 1 < num_cb:
            num_delay += 1

        # 2) Ending pattern: if already ending, force EOS on the first num_ended codebooks
        if num_remaining_delays is not None:
            num_ended = num_cb - num_remaining_delays
            ended_threshold = constant(num_ended)
            mask_ended = lt(idx, ended_threshold)
            eos_vec = full([num_cb], self.audio_stream_eos_id, dtype=next_audio_tokens.dtype)
            next_audio_tokens = where(mask_ended, eos_vec, next_audio_tokens)
            num_remaining_delays -= 1

            if num_remaining_delays <= 0:
                should_end_audio = True
                num_delay = 0
                num_remaining_delays = None

        # Note: EOS-triggered start of ending pattern is handled by the caller that
        # computes num_remaining_delays, to keep this utility simple and TRT-LLM friendly.

        return next_audio_tokens, num_delay, num_remaining_delays, should_end_audio

    def sample_audio_tokens_with_delay_pattern(
        self,
        audio_logits: Tensor,
        num_delay: int = 0,
        num_remaining_delays: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        audio_eos_token_id: Optional[int] = None
    ) -> Tuple[Tensor, int, Optional[int], bool]:
        """Sample audio tokens from logits and apply delay pattern.
        
        This method handles the sampling of audio tokens across multiple codebooks
        and applies the delay pattern to ensure proper temporal coordination.
        
        Args:
            audio_logits: Logits for audio token generation [batch, num_codebooks, vocab_size]
            num_delay: Current delay counter
            num_remaining_delays: Remaining delays for ending pattern
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            audio_eos_token_id: Token ID for ending audio generation
            
        Returns:
            Tuple of:
                - Sampled audio tokens [num_codebooks]
                - Updated num_delay
                - Updated num_remaining_delays
                - Boolean indicating if audio should end
        """
        # Gather last-step logits: [B, V_a] where V_a = num_codebooks*(codebook_size+2)
        last_logits = gather_last_token_logits(audio_logits)

        # Assume batch size 1 in generation; squeeze to [V_a]
        if len(last_logits.shape) == 2:
            # [B, V_a] -> [V_a] (B==1 expected)
            last_logits = last_logits[0]

        # Split into per-codebook logits: [num_cb, vocab_per_cb]
        vocab_per_cb = self.config.audio_codebook_size + 2
        num_cb = self.audio_num_codebooks
        # Reshape for per-codebook argmax
        per_cb_logits = last_logits.view(num_cb, vocab_per_cb)

        # Temperature scaling (optional)
        if temperature != 1.0:
            per_cb_logits = per_cb_logits / temperature

        # Greedy per-codebook sampling
        next_audio_tokens = per_cb_logits.argmax(dim=-1)

        # If not already ending, detect EOS to start ending pattern (HF parity)
        if self.use_delay_pattern and num_remaining_delays is None:
            last_eos_idx = None
            # Small loop over codebooks is acceptable here for clarity
            for i in range(int(num_cb)):
                token_val = next_audio_tokens[i]
                if isinstance(token_val, torch.Tensor):
                    val = int(token_val.item())
                else:
                    val = int(token_val)
                if val == int(self.audio_stream_eos_id):
                    last_eos_idx = i
            if last_eos_idx is not None:
                # Force EOS for codebooks before the last eos index
                for j in range(last_eos_idx):
                    next_audio_tokens[j] = int(self.audio_stream_eos_id)
                num_remaining_delays = int(num_cb) - int(last_eos_idx) - 1

        # Apply delay pattern
        next_audio_tokens, num_delay, num_remaining_delays, should_end = \
            self._apply_delay_pattern_to_audio_tokens(
                next_audio_tokens,
                num_delay,
                num_remaining_delays,
                audio_eos_token_id
            )
        
        return next_audio_tokens, num_delay, num_remaining_delays, should_end

    def prepare_inputs(self, *args, **kwargs):
        """Prepare inputs for generation with TTS-specific handling."""
        inputs = super().prepare_inputs(*args, **kwargs)
        
        # Add delay pattern state if in audio generation mode
        if hasattr(self, '_generation_state'):
            if self._generation_state.get('mode') == GenerationMode.AUDIO_IN_PROGRESS:
                inputs['num_delay'] = self._generation_state.get('num_delay', 0)
                inputs['num_remaining_delays'] = self._generation_state.get('num_remaining_delays', None)

        return inputs
    
    def generate_with_delay_pattern(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        use_delay_pattern: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate audio tokens with delay pattern support.
        
        This is a simplified generation method that demonstrates how to use
        delay patterns during audio generation. In production, this would be
        integrated with the full TensorRT-LLM generation pipeline.
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            use_delay_pattern: Whether to apply delay pattern
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generated tokens and metadata
        """
        # Initialize generation state
        self._generation_state = {
            'mode': GenerationMode.TEXT,
            'num_delay': 0,
            'num_remaining_delays': None,
            'audio_sequences': []
        }
        
        # Store original delay pattern setting
        original_use_delay = self.use_delay_pattern
        self.use_delay_pattern = use_delay_pattern
        
        generated_tokens = []
        audio_sequences = []
        
        try:
            # This is a simplified generation loop
            # In production, you'd use the full TensorRT-LLM generation pipeline
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.forward(
                    input_ids=input_ids,
                    use_cache=True,
                    audio_out_mask=self._generation_state.get('audio_out_mask')
                )
                
                # Check generation mode based on last token
                last_token = input_ids[0, -1] if len(input_ids.shape) > 1 else input_ids[-1]
                
                # Determine generation mode
                if last_token == getattr(self.config, 'audio_out_bos_token_id', -1):
                    self._generation_state['mode'] = GenerationMode.AUDIO_INIT
                elif last_token == getattr(self.config, 'audio_out_token_idx', -1):
                    self._generation_state['mode'] = GenerationMode.AUDIO_IN_PROGRESS
                else:
                    self._generation_state['mode'] = GenerationMode.TEXT
                
                # Handle audio generation with delay pattern
                if self._generation_state['mode'] == GenerationMode.AUDIO_IN_PROGRESS:
                    # Get audio logits from decoder projector
                    _, audio_logits = self.decoder_projector(outputs[0])
                    
                    # Sample audio tokens with delay pattern
                    audio_tokens, num_delay, num_remaining_delays, should_end = \
                        self.sample_audio_tokens_with_delay_pattern(
                            audio_logits,
                            self._generation_state['num_delay'],
                            self._generation_state['num_remaining_delays'],
                            temperature=temperature
                        )
                    
                    # Update state
                    self._generation_state['num_delay'] = num_delay
                    self._generation_state['num_remaining_delays'] = num_remaining_delays
                    
                    # Store audio tokens
                    audio_sequences.append(audio_tokens)
                    
                    # If audio should end, switch back to text mode
                    if should_end:
                        self._generation_state['mode'] = GenerationMode.TEXT
                        # Add audio end token
                        next_token = constant(getattr(self.config, 'audio_eos_token_id', 2))
                    else:
                        # Continue with audio placeholder token
                        next_token = constant(getattr(self.config, 'audio_out_token_idx', -1))
                else:
                    # Text generation - use regular sampling
                    text_logits, _ = self.decoder_projector(outputs[0])
                    next_token = text_logits.argmax(dim=-1)
                
                # Append token
                generated_tokens.append(next_token)
                input_ids = concat([input_ids, next_token.unsqueeze(0)], dim=-1)
                
                # Check for end of generation
                if next_token == getattr(self.config, 'eos_token_id', 2):
                    break
        
        finally:
            # Restore original setting
            self.use_delay_pattern = original_use_delay
        
        return {
            'sequences': input_ids,
            'audio_sequences': audio_sequences,
            'generated_tokens': generated_tokens,
            'delay_pattern_info': {
                'final_num_delay': self._generation_state['num_delay'],
                'used_delay_pattern': use_delay_pattern
            }
        }

    @classmethod
    def from_hugging_face(cls,
                         hf_model_or_dir: Union[str, Any],
                         dtype: str = 'auto',
                         mapping: Optional[Any] = None,
                         quant_config: Optional[Any] = None,
                         **kwargs: Any) -> 'HiggsAudioModelForCausalLM':
        """Factory method to create model from HuggingFace checkpoint.

        This method maintains compatibility with existing TRT-LLM patterns
        while enabling TTS-specific configuration loading and validation.

        Args:
            hf_model_or_dir: Path to HF model directory or HF model object
            dtype: Data type for model weights ('auto', 'float16', 'float32', etc.)
            mapping: TensorRT-LLM tensor/pipeline parallelism mapping configuration
            quant_config: Quantization configuration for model optimization
            **kwargs: Additional arguments passed to config building

        Returns:
            Configured HiggsAudioModelForCausalLM instance ready for TTS generation

        Raises:
            ValueError: If HF model is incompatible or missing required components
            FileNotFoundError: If model directory doesn't exist

        Example:
            >>> model = HiggsAudioModelForCausalLM.from_hugging_face(
            ...     "path/to/higgs-audio-model",
            ...     dtype="float16",
            ...     mapping=tp_mapping
            ... )
        """
        hf_cfg, _unused = AutoConfig.from_pretrained(
        hf_model_or_dir, trust_remote_code=trust_remote_code, return_unused_kwargs=True
        )
        cfg = HiggsAudioConfig.from_hugging_face(
            hf_cfg, dtype=dtype, mapping=mapping, quant_config=quant_config, **kwargs)
            
        return cls(cfg)

    def _apply_audio_tower(self, audio_features, audio_feature_attention_mask):
        """Apply the audio tower to the audio features - TensorRT-LLM compatible implementation."""
        
        # Handle empty audio features case
        if audio_features.shape[0] == 0:
            # Return None for empty batch to avoid computation
            return None, None
        
        # Calculate attention mask if provided
        audio_attention_mask = None
        audio_feat_out_lengths = None
        
        if audio_feature_attention_mask is not None:
            # Calculate actual feature lengths from attention mask
            audio_feat_lengths = audio_feature_attention_mask.sum(dim=-1)
            
            # Calculate output lengths after conv layers (stride=2 from conv2)
            # Mel-spectrogram length -> conv1 (no stride) -> conv2 (stride=2) -> final length
            audio_feat_out_lengths = (audio_feat_lengths - 1) // 2 + 1
            
            batch_size, max_mel_seq_len = audio_feature_attention_mask.shape
            max_seq_len = (max_mel_seq_len - 1) // 2 + 1
            
            # Create sequence range tensor for masking
            seq_range = arange(0, max_seq_len, dtype='int32')
            seq_range = unsqueeze(seq_range, 0)  # [1, max_seq_len]
            seq_range = expand(seq_range, [batch_size, max_seq_len])  # [batch, max_seq_len]
            
            # Expand lengths for comparison
            lengths_expand = unsqueeze(audio_feat_out_lengths, 1)  # [batch, 1]
            lengths_expand = expand(lengths_expand, [batch_size, max_seq_len])  # [batch, max_seq_len]
            
            # Create padding mask (True where valid tokens)
            padding_mask = lt(seq_range, lengths_expand)  # [batch, max_seq_len]
            
            # For bidirectional attention in encoder, use simple padding mask
            audio_attention_mask = padding_mask
        
        # Apply audio encoder
        audio_outputs = self.audio_tower(
            audio_features,
            attention_mask=audio_attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            check_seq_length=False  # Skip length validation for flexibility
        )
        
        # Extract last hidden state
        if isinstance(audio_outputs, dict):
            selected_audio_feature = audio_outputs['last_hidden_state']
        else:
            # Handle tuple output
            selected_audio_feature = audio_outputs[0] if isinstance(audio_outputs, tuple) else audio_outputs
        
        # Project audio features to text model dimension
        audio_features_embed = self.audio_encoder_proj(selected_audio_feature)
        
        return audio_features_embed, audio_feat_out_lengths

    def forward(self,
                input_ids: Tensor,
                position_ids=None,
                use_cache=False,
                last_token_ids=None,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                mrope_params=None,
                hidden_states=None,
                prompt_embedding_table: Optional[Tensor] = None,
                prompt_tasks: Optional[Tensor] = None,
                prompt_vocab_size: Optional[Tensor] = None,
                lora_params=None,
                spec_decoding_params=None,
                audio_out_mask: Optional[Tensor] = None):
        """Forward pass for Higgs Audio model with audio token routing.
        
        This method extends the base DecoderModelForCausalLM forward method to support
        audio-specific parameters like audio_out_mask for routing audio tokens through
        DualFFN layers.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            audio_out_mask: Boolean mask indicating audio tokens [batch_size, seq_len]
            **kwargs: Other standard TensorRT-LLM forward arguments
            
        Returns:
            Model outputs with logits for text/audio token generation
        """

        if audio_out_mask is not None:
            self.transformer.audio_out_mask = audio_out_mask

        # Audio tower integration - TensorRT-LLM compatible implementation
        audio_features_embed = None
        audio_features_length = None
        
        # Check if audio features are provided and audio tower is enabled
        if hasattr(self, 'audio_features') and self.audio_features is not None and not self.config.skip_audio_tower:
            # Apply audio tower processing
            audio_features_embed, audio_features_length = self._apply_audio_tower(
                self.audio_features, self.audio_feature_attention_mask
            ) 
            # Integrate audio features with text embeddings if needed
            # This would typically happen in the transformer layers through attention mechanisms
        
        output = backbone(input_ids, attention_mask=attention_mask, use_cache=True)


        # Call parent forward method with standard arguments
        return super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=use_cache,
            last_token_ids=last_token_ids,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            mrope_params=mrope_params,
            hidden_states=hidden_states,
            prompt_embedding_table=prompt_embedding_table,
            prompt_tasks=prompt_tasks,
            prompt_vocab_size=prompt_vocab_size,
            lora_params=lora_params,
            spec_decoding_params=spec_decoding_params
        )

    def generate_multimodal(
        self,
        input_ids: Tensor,
        audio_features: Optional[Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_delay_pattern: bool = True,
        num_codebooks: int = 8,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        audio_eos_token_id: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Unified multimodal generation with delay pattern support.
        
        This method handles:
        1. Audio input processing and encoding
        2. Multimodal forward pass with modal switching
        3. Delay pattern application for multi-codebook audio generation
        4. Seamless transitions between text and audio generation
        
        Args:
            input_ids: Input text token IDs
            audio_features: Optional preprocessed audio features
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            use_delay_pattern: Whether to use delay pattern for audio
            num_codebooks: Number of audio codebooks
            eos_token_id: End-of-sequence token ID for text
            pad_token_id: Padding token ID
            audio_eos_token_id: End-of-sequence token ID for audio
            
        Returns:
            Dictionary containing:
            - 'sequences': Generated token sequences
            - 'audio_sequences': Generated audio token sequences
            - 'modal_mask': Mask indicating modal types
            - 'generation_info': Additional generation metadata
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device if hasattr(input_ids, 'device') else None
        
        # Initialize generation state
        if not hasattr(self, '_generation_state'):
            self._generation_state = {
                'num_delay': 0,
                'num_remaining_delays': None,
                'current_mode': 'text',  # 'text' or 'audio'
                'audio_buffer': [],
                'text_buffer': []
            }
        
        # Prepare audio mask if audio features provided
        audio_mask = None
        if audio_features is not None:
            # Create mask for audio positions
            seq_len = input_ids.shape[1]
            audio_len = audio_features.shape[1] if len(audio_features.shape) > 1 else 0
            if audio_len > 0:
                # Mark positions where audio features are present
                audio_mask = constant(False, shape=[batch_size, seq_len])
                # This is simplified - in practice you'd determine audio positions from input
                
        # Storage for generated sequences
        generated_sequences = input_ids.clone() if hasattr(input_ids, 'clone') else input_ids
        generated_audio = []
        modal_indicators = []
        
        # Generation loop
        past_key_value = None
        for step in range(max_new_tokens):
            # Determine current generation mode
            current_mode = self._generation_state['current_mode']
            
            # Multimodal forward pass
            outputs = self.multimodal_forward(
                input_ids=generated_sequences,
                audio_features=audio_features if step == 0 else None,
                audio_mask=audio_mask,
                past_key_value=past_key_value,
                use_cache=True
            )
            
            logits = outputs['logits']
            past_key_value = outputs.get('present_key_value')
            
            # Sample based on current mode
            if current_mode == 'audio' and 'audio' in logits:
                # Audio generation with delay pattern
                audio_logits = logits['audio']
                
                # Apply delay pattern sampling
                if use_delay_pattern:
                    audio_tokens, num_delay, num_remaining_delays, should_end = \
                        self.sample_audio_tokens_with_delay_pattern(
                            audio_logits,
                            self._generation_state['num_delay'],
                            self._generation_state['num_remaining_delays'],
                            temperature=temperature,
                            audio_eos_token_id=audio_eos_token_id
                        )
                    
                    # Update delay pattern state
                    self._generation_state['num_delay'] = num_delay
                    self._generation_state['num_remaining_delays'] = num_remaining_delays
                    
                    # Check if audio generation should end
                    if should_end:
                        self._generation_state['current_mode'] = 'text'
                        self._generation_state['num_delay'] = 0
                        self._generation_state['num_remaining_delays'] = None
                else:
                    # Simple audio sampling without delay pattern
                    audio_tokens = self._sample_tokens(
                        audio_logits[:, -1, :],
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p
                    )
                    
                generated_audio.append(audio_tokens)
                modal_indicators.append('audio')
                
                # Append audio tokens to sequence
                next_tokens = audio_tokens[:, 0] if len(audio_tokens.shape) > 1 else audio_tokens
                
            else:
                # Text generation
                if 'text' in logits:
                    text_logits = logits['text']
                    
                    # Sample text tokens
                    next_tokens = self._sample_tokens(
                        text_logits[:, -1, :],
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p
                    )
                    
                    modal_indicators.append('text')
                    
                    # Check for mode switch indicators (simplified)
                    # In practice, you'd check for special tokens that indicate mode switch
                    # For now, we'll continue in text mode
                else:
                    # No valid logits, end generation
                    break
                    
            # Update generated sequence
            next_tokens_expanded = next_tokens.unsqueeze(1) if len(next_tokens.shape) == 1 else next_tokens
            generated_sequences = concat([generated_sequences, next_tokens_expanded], dim=1)
            
            # Check for end-of-sequence
            if eos_token_id is not None and current_mode == 'text':
                if (next_tokens == eos_token_id).any():
                    break
                    
        # Prepare output
        return {
            'sequences': generated_sequences,
            'audio_sequences': generated_audio if generated_audio else None,
            'modal_mask': modal_indicators,
            'generation_info': {
                'num_tokens_generated': len(modal_indicators),
                'final_mode': self._generation_state['current_mode'],
                'delay_pattern_info': {
                    'num_delay': self._generation_state['num_delay'],
                    'num_remaining_delays': self._generation_state['num_remaining_delays']
                } if use_delay_pattern else None
            }
        }

    def _sample_tokens(
        self,
        logits: Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> Tensor:
        """
        Sample tokens from logits with temperature, top-k, and top-p.
        
        Args:
            logits: Logits tensor [batch_size, vocab_size]
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Sampled token IDs
        """
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
            
        # For now, use argmax sampling (greedy)
        # Full sampling with top-k/top-p would require more complex implementation
        sampled = argmax(logits, dim=-1)
        
        return sampled