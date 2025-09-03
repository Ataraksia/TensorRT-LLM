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

"""Configuration classes for HiggsAudio multimodal model.

HiggsAudio combines text generation with audio processing through a dual-FFN architecture.
The model includes:
- HiggsAudioConfig: Main configuration for the multimodal model
- HiggsAudioEncoderConfig: Configuration for the Whisper-based audio encoder
"""

import json
from typing import Dict, Optional, Union

from ..modeling_utils import PretrainedConfig

__all__ = [
    "HiggsAudioConfig",
    "HiggsAudioEncoderConfig",
]


class HiggsAudioEncoderConfig(PretrainedConfig):
    """Configuration class for the HiggsAudio audio encoder (Whisper-based).

    This configuration class stores the configuration parameters for the
    audio encoder component of the HiggsAudio multimodal model. The encoder is based
    on a Whisper architecture and processes mel-spectrogram audio features.
    """

    def __init__(
        self,
        *,
        # Audio encoder specific parameters
        num_mel_bins: int = 128,
        encoder_layers: int = 32,
        encoder_attention_heads: int = 20,
        encoder_ffn_dim: int = 5120,
        encoder_layerdrop: float = 0.0,
        d_model: int = 1280,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_function: str = "gelu",
        activation_dropout: float = 0.0,
        scale_embedding: bool = False,
        init_std: float = 0.02,
        max_source_positions: int = 1500,
        pad_token_id: int = 128001,
        **kwargs,
    ):
        # Initialize base config first
        super().__init__(
            architecture="higgs_audio_encoder",
            dtype="float16",
            hidden_size=d_model,
            num_hidden_layers=encoder_layers,
            num_attention_heads=encoder_attention_heads,
            vocab_size=None,  # Audio encoder doesn't use text vocab
            hidden_act=activation_function,
            max_position_embeddings=max_source_positions,
            intermediate_size=encoder_ffn_dim,
        )

        # Audio encoder specific attributes
        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.encoder_layerdrop = encoder_layerdrop
        self.init_std = init_std
        self.scale_embedding = scale_embedding
        self.max_source_positions = max_source_positions
        self.pad_token_id = pad_token_id

        # Validate configuration
        self.validate()

    def validate(self):
        """Validate audio encoder configuration parameters."""
        # Core architecture validation
        if not isinstance(self.encoder_layers, int) or self.encoder_layers <= 0:
            raise ValueError(
                f"encoder_layers must be a positive integer, got {self.encoder_layers}"
            )

        if not isinstance(self.encoder_attention_heads, int) or self.encoder_attention_heads <= 0:
            raise ValueError(
                f"encoder_attention_heads must be a positive integer, got {self.encoder_attention_heads}"
            )

        if not isinstance(self.d_model, int) or self.d_model <= 0:
            raise ValueError(f"d_model must be a positive integer, got {self.d_model}")

        if self.d_model % self.encoder_attention_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"encoder_attention_heads ({self.encoder_attention_heads})"
            )

        if not isinstance(self.encoder_ffn_dim, int) or self.encoder_ffn_dim <= 0:
            raise ValueError(
                f"encoder_ffn_dim must be a positive integer, got {self.encoder_ffn_dim}"
            )

        if not isinstance(self.num_mel_bins, int) or self.num_mel_bins <= 0:
            raise ValueError(f"num_mel_bins must be a positive integer, got {self.num_mel_bins}")

        if not isinstance(self.max_source_positions, int) or self.max_source_positions <= 0:
            raise ValueError(
                f"max_source_positions must be a positive integer, got {self.max_source_positions}"
            )

        # Dropout validation
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError(f"dropout must be between 0 and 1, got {self.dropout}")

        if not 0.0 <= self.attention_dropout <= 1.0:
            raise ValueError(
                f"attention_dropout must be between 0 and 1, got {self.attention_dropout}"
            )

        if not 0.0 <= self.activation_dropout <= 1.0:
            raise ValueError(
                f"activation_dropout must be between 0 and 1, got {self.activation_dropout}"
            )

        if not 0.0 <= self.encoder_layerdrop <= 1.0:
            raise ValueError(
                f"encoder_layerdrop must be between 0 and 1, got {self.encoder_layerdrop}"
            )

        # Initialization and scaling validation
        if not isinstance(self.init_std, (int, float)) or self.init_std <= 0:
            raise ValueError(f"init_std must be a positive number, got {self.init_std}")

        if not isinstance(self.scale_embedding, bool):
            raise ValueError(f"scale_embedding must be a boolean, got {type(self.scale_embedding)}")

        # Token ID validation
        if not isinstance(self.pad_token_id, int) or self.pad_token_id < 0:
            raise ValueError(
                f"pad_token_id must be a non-negative integer, got {self.pad_token_id}"
            )

        # Activation function validation
        valid_activations = ["relu", "gelu", "silu", "swish", "gelu_new", "mish", "tanh"]
        if self.activation_function not in valid_activations:
            import warnings

            warnings.warn(
                f"activation_function '{self.activation_function}' is not in common list "
                f"{valid_activations}. Please ensure it's supported by your implementation.",
                UserWarning,
            )

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        output = super().to_dict()

        # Add audio encoder specific attributes
        output.update(
            {
                "num_mel_bins": self.num_mel_bins,
                "encoder_layers": self.encoder_layers,
                "encoder_attention_heads": self.encoder_attention_heads,
                "encoder_ffn_dim": self.encoder_ffn_dim,
                "encoder_layerdrop": self.encoder_layerdrop,
                "d_model": self.d_model,
                "dropout": self.dropout,
                "attention_dropout": self.attention_dropout,
                "activation_function": self.activation_function,
                "activation_dropout": self.activation_dropout,
                "scale_embedding": self.scale_embedding,
                "init_std": self.init_std,
                "max_source_positions": self.max_source_positions,
                "pad_token_id": self.pad_token_id,
            }
        )

        return output

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "HiggsAudioEncoderConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def to_json_string(self):
        """Return JSON string representation of the config."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


class HiggsAudioConfig(PretrainedConfig):
    """Configuration class for HiggsAudio multimodal model.

    HiggsAudio is a multimodal model combining text generation with audio processing
    capabilities. It uses a dual-FFN architecture that can process both text and
    audio tokens through separate processing paths.
    """

    def __init__(
        self,
        *,
        # Core configurations
        text_config: Optional[Union[PretrainedConfig, Dict]] = None,
        audio_encoder_config: Optional[Union[HiggsAudioEncoderConfig, Dict]] = None,
        audio_tokenizer_config: Optional[Dict] = None,
        # Adapter configuration
        audio_adapter_type: str = "stack",
        audio_ffn_hidden_size: int = 4096,
        audio_ffn_intermediate_size: int = 14336,
        audio_dual_ffn_layers: Optional[list] = None,
        audio_decoder_proj_num_layers: int = 0,
        # Processing flags
        encode_audio_in_tokens: bool = False,
        use_delay_pattern: bool = False,
        skip_audio_tower: bool = False,
        use_audio_out_embed_projector: bool = False,
        use_audio_out_self_attention: bool = False,
        # Audio codebook configuration
        audio_num_codebooks: int = 8,
        audio_codebook_size: int = 1024,
        audio_stream_bos_id: int = 1024,
        audio_stream_eos_id: int = 1025,
        # Special tokens
        audio_bos_token: str = "<|audio_bos|>",
        audio_bos_token_id: int = 128011,
        audio_eos_token: str = "<|audio_eos|>",
        audio_eos_token_id: int = 128012,
        audio_out_bos_token: str = "<|audio_out_bos|>",
        audio_out_bos_token_id: int = 128013,
        audio_in_token: str = "<|AUDIO|>",
        audio_out_token: str = "<|AUDIO_OUT|>",
        audio_in_token_idx: int = 128015,
        audio_out_token_idx: int = 128016,
        pad_token_id: int = 128001,
        **kwargs,
    ):
        # Handle nested configurations
        if audio_encoder_config is None:
            audio_encoder_config = HiggsAudioEncoderConfig()
        elif isinstance(audio_encoder_config, dict):
            audio_encoder_config = HiggsAudioEncoderConfig(**audio_encoder_config)

        # For text_config, we'll use a default LLaMA-like config if none provided
        if text_config is None:
            # Default LLaMA-3.2-3B configuration
            text_config = {
                "architecture": "llama",
                "hidden_size": 3072,
                "num_hidden_layers": 28,
                "num_attention_heads": 24,
                "intermediate_size": 8192,
                "vocab_size": 128256,
                "max_position_embeddings": 131072,
                "num_key_value_heads": 8,
            }
        elif isinstance(text_config, dict) and "architecture" not in text_config:
            text_config["architecture"] = "llama"

        # Determine base configuration parameters from text config
        if isinstance(text_config, dict):
            hidden_size = text_config.get("hidden_size", 3072)
            num_layers = text_config.get("num_hidden_layers", 28)
            vocab_size = text_config.get("vocab_size", 128256)
        else:
            hidden_size = getattr(text_config, "hidden_size", 3072)
            num_layers = getattr(text_config, "num_hidden_layers", 28)
            vocab_size = getattr(text_config, "vocab_size", 128256)

        # Initialize base PretrainedConfig
        super().__init__(
            architecture="higgs_audio",
            dtype="float16",
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=24,  # Default for LLaMA-3.2-3B
            vocab_size=vocab_size,
            hidden_act="silu",
        )

        # Store nested configurations
        self.text_config = text_config
        self.audio_encoder_config = audio_encoder_config
        self.audio_tokenizer_config = audio_tokenizer_config

        # Adapter configuration
        self.audio_adapter_type = audio_adapter_type
        self.audio_ffn_hidden_size = audio_ffn_hidden_size
        self.audio_ffn_intermediate_size = audio_ffn_intermediate_size
        self.audio_dual_ffn_layers = audio_dual_ffn_layers
        self.audio_decoder_proj_num_layers = audio_decoder_proj_num_layers

        # Processing flags
        self.encode_audio_in_tokens = encode_audio_in_tokens
        self.use_delay_pattern = use_delay_pattern
        self.skip_audio_tower = skip_audio_tower
        self.use_audio_out_embed_projector = use_audio_out_embed_projector
        self.use_audio_out_self_attention = use_audio_out_self_attention

        # Audio codebook parameters
        self.audio_num_codebooks = audio_num_codebooks
        self.audio_codebook_size = audio_codebook_size
        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id

        # Special tokens
        self.audio_bos_token = audio_bos_token
        self.audio_bos_token_id = audio_bos_token_id
        self.audio_eos_token = audio_eos_token
        self.audio_eos_token_id = audio_eos_token_id
        self.audio_out_bos_token = audio_out_bos_token
        self.audio_out_bos_token_id = audio_out_bos_token_id
        self.audio_in_token = audio_in_token
        self.audio_out_token = audio_out_token
        self.audio_in_token_idx = audio_in_token_idx
        self.audio_out_token_idx = audio_out_token_idx
        self.pad_token_id = pad_token_id

        # Validate configuration
        self.validate()

    def validate(self):
        """Validate configuration parameters comprehensively."""
        import warnings

        # 1. Validate adapter type and related configurations
        valid_adapter_types = ["stack", "dual_ffn", "dual_ffn_fast_forward"]
        if self.audio_adapter_type not in valid_adapter_types:
            raise ValueError(
                f"Invalid audio_adapter_type: {self.audio_adapter_type}. "
                f"Must be one of {valid_adapter_types}"
            )

        # Validate dual FFN configuration
        if self.audio_adapter_type.startswith("dual_ffn"):
            if self.audio_dual_ffn_layers is None:
                raise ValueError(
                    f"audio_dual_ffn_layers must be specified when using "
                    f"audio_adapter_type='{self.audio_adapter_type}'"
                )
            if not isinstance(self.audio_dual_ffn_layers, (list, tuple)):
                raise ValueError(
                    f"audio_dual_ffn_layers must be a list or tuple, got {type(self.audio_dual_ffn_layers)}"
                )

        # 2. Validate audio parameters - core requirements
        if not isinstance(self.audio_num_codebooks, int) or self.audio_num_codebooks <= 0:
            raise ValueError(
                f"audio_num_codebooks must be a positive integer, got {self.audio_num_codebooks}"
            )

        if not isinstance(self.audio_codebook_size, int) or self.audio_codebook_size <= 0:
            raise ValueError(
                f"audio_codebook_size must be a positive integer, got {self.audio_codebook_size}"
            )

        # Optional: Check if codebook size is a power of two (common requirement)
        if self.audio_codebook_size & (self.audio_codebook_size - 1) != 0:
            warnings.warn(
                f"audio_codebook_size ({self.audio_codebook_size}) is not a power of two. "
                f"This may impact performance in some implementations.",
                UserWarning,
            )

        # 3. Validate all token IDs are non-negative integers
        token_ids = [
            ("audio_bos_token_id", self.audio_bos_token_id),
            ("audio_eos_token_id", self.audio_eos_token_id),
            ("audio_out_bos_token_id", self.audio_out_bos_token_id),
            ("audio_in_token_idx", self.audio_in_token_idx),
            ("audio_out_token_idx", self.audio_out_token_idx),
            ("pad_token_id", self.pad_token_id),
            ("audio_stream_bos_id", self.audio_stream_bos_id),
            ("audio_stream_eos_id", self.audio_stream_eos_id),
        ]

        for name, token_id in token_ids:
            if not isinstance(token_id, int) or token_id < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {token_id}")

        # 4. Validate FFN dimensions are positive integers
        if not isinstance(self.audio_ffn_hidden_size, int) or self.audio_ffn_hidden_size <= 0:
            raise ValueError(
                f"audio_ffn_hidden_size must be a positive integer, got {self.audio_ffn_hidden_size}"
            )

        if (
            not isinstance(self.audio_ffn_intermediate_size, int)
            or self.audio_ffn_intermediate_size <= 0
        ):
            raise ValueError(
                f"audio_ffn_intermediate_size must be a positive integer, got {self.audio_ffn_intermediate_size}"
            )

        if (
            not isinstance(self.audio_decoder_proj_num_layers, int)
            or self.audio_decoder_proj_num_layers < 0
        ):
            raise ValueError(
                f"audio_decoder_proj_num_layers must be a non-negative integer, "
                f"got {self.audio_decoder_proj_num_layers}"
            )

        # 5. Validate nested configurations are proper types
        if self.text_config is not None:
            if not isinstance(self.text_config, (dict, PretrainedConfig)):
                raise ValueError(
                    f"text_config must be a dict or PretrainedConfig instance, got {type(self.text_config)}"
                )

        if self.audio_encoder_config is not None:
            if not isinstance(self.audio_encoder_config, (dict, HiggsAudioEncoderConfig)):
                raise ValueError(
                    f"audio_encoder_config must be a dict or HiggsAudioEncoderConfig instance, "
                    f"got {type(self.audio_encoder_config)}"
                )

        if self.audio_tokenizer_config is not None:
            if not isinstance(self.audio_tokenizer_config, dict):
                raise ValueError(
                    f"audio_tokenizer_config must be a dict, got {type(self.audio_tokenizer_config)}"
                )

        # 6. Cross-validation: Check consistency between configs
        # Check hidden size consistency between text config and audio FFN
        if isinstance(self.text_config, dict):
            text_hidden_size = self.text_config.get("hidden_size")
        elif hasattr(self.text_config, "hidden_size"):
            text_hidden_size = self.text_config.hidden_size
        else:
            text_hidden_size = None

        if text_hidden_size is not None:
            if text_hidden_size != self.audio_ffn_hidden_size:
                warnings.warn(
                    f"text_config.hidden_size ({text_hidden_size}) differs from "
                    f"audio_ffn_hidden_size ({self.audio_ffn_hidden_size}). "
                    f"This may cause dimension mismatch issues.",
                    UserWarning,
                )

        # Check audio encoder output size consistency
        if isinstance(self.audio_encoder_config, HiggsAudioEncoderConfig):
            if hasattr(self.audio_encoder_config, "d_model") and text_hidden_size is not None:
                if self.audio_encoder_config.d_model != text_hidden_size:
                    warnings.warn(
                        f"audio_encoder_config.d_model ({self.audio_encoder_config.d_model}) "
                        f"differs from text_config.hidden_size ({text_hidden_size}). "
                        f"This may require projection layers for compatibility.",
                        UserWarning,
                    )

        # 7. Validate boolean flags
        boolean_flags = [
            ("encode_audio_in_tokens", self.encode_audio_in_tokens),
            ("use_delay_pattern", self.use_delay_pattern),
            ("skip_audio_tower", self.skip_audio_tower),
            ("use_audio_out_embed_projector", self.use_audio_out_embed_projector),
            ("use_audio_out_self_attention", self.use_audio_out_self_attention),
        ]

        for name, flag in boolean_flags:
            if not isinstance(flag, bool):
                raise ValueError(f"{name} must be a boolean, got {type(flag)}")

        # 8. Validate string tokens are non-empty
        string_tokens = [
            ("audio_bos_token", self.audio_bos_token),
            ("audio_eos_token", self.audio_eos_token),
            ("audio_out_bos_token", self.audio_out_bos_token),
            ("audio_in_token", self.audio_in_token),
            ("audio_out_token", self.audio_out_token),
        ]

        for name, token in string_tokens:
            if not isinstance(token, str) or len(token.strip()) == 0:
                raise ValueError(f"{name} must be a non-empty string, got '{token}'")

        # 9. Range validation for stream IDs vs codebook size
        if self.audio_stream_bos_id >= self.audio_codebook_size:
            warnings.warn(
                f"audio_stream_bos_id ({self.audio_stream_bos_id}) is >= "
                f"audio_codebook_size ({self.audio_codebook_size}). "
                f"This may indicate a configuration issue.",
                UserWarning,
            )

        if self.audio_stream_eos_id >= self.audio_codebook_size:
            warnings.warn(
                f"audio_stream_eos_id ({self.audio_stream_eos_id}) is >= "
                f"audio_codebook_size ({self.audio_codebook_size}). "
                f"This may indicate a configuration issue.",
                UserWarning,
            )

        # 10. Validate audio encoder config if present and callable
        if hasattr(self.audio_encoder_config, "validate"):
            self.audio_encoder_config.validate()

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        # Validate before serialization
        self.validate()

        output = super().to_dict()

        # Handle nested configs
        if hasattr(self.audio_encoder_config, "to_dict"):
            output["audio_encoder_config"] = self.audio_encoder_config.to_dict()
        else:
            output["audio_encoder_config"] = self.audio_encoder_config

        # Ensure text_config is JSON-serializable
        if hasattr(self.text_config, "to_dict"):
            output["text_config"] = self.text_config.to_dict()
        else:
            output["text_config"] = self.text_config

        # Add all our custom attributes
        custom_attrs = [
            "audio_tokenizer_config",
            "audio_adapter_type",
            "audio_ffn_hidden_size",
            "audio_ffn_intermediate_size",
            "audio_dual_ffn_layers",
            "audio_decoder_proj_num_layers",
            "encode_audio_in_tokens",
            "use_delay_pattern",
            "skip_audio_tower",
            "use_audio_out_embed_projector",
            "use_audio_out_self_attention",
            "audio_num_codebooks",
            "audio_codebook_size",
            "audio_stream_bos_id",
            "audio_stream_eos_id",
            "audio_bos_token",
            "audio_bos_token_id",
            "audio_eos_token",
            "audio_eos_token_id",
            "audio_out_bos_token",
            "audio_out_bos_token_id",
            "audio_in_token",
            "audio_out_token",
            "audio_in_token_idx",
            "audio_out_token_idx",
            "pad_token_id",
        ]

        for attr in custom_attrs:
            if hasattr(self, attr):
                output[attr] = getattr(self, attr)

        # Ensure model_type is included
        output["model_type"] = self.model_type

        return output

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "HiggsAudioConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def to_json_string(self, indent: int = 2) -> str:
        """Convert configuration to JSON string."""
        import json

        return json.dumps(self.to_dict(), indent=indent)

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save configuration to a directory."""
        import os

        os.makedirs(save_directory, exist_ok=True)
        config_file = os.path.join(save_directory, "config.json")
        self.to_json_file(config_file)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> "HiggsAudioConfig":
        """Load configuration from a pretrained model path."""
        import os

        if os.path.isdir(model_name_or_path):
            config_file = os.path.join(model_name_or_path, "config.json")
            if os.path.isfile(config_file):
                return cls.from_json_file(config_file)
        # If it's not a directory or config.json doesn't exist,
        # assume it's the model_name_or_path itself is the config file
        if os.path.isfile(model_name_or_path):
            return cls.from_json_file(model_name_or_path)

        raise ValueError(f"Could not find config.json in {model_name_or_path}")

    @property
    def model_type(self) -> str:
        """Return model type for HuggingFace compatibility."""
        return self.architecture

    @classmethod
    def from_json_string(cls, json_string: str):
        """Create a new config from a JSON string."""
        return cls.from_dict(json.loads(json_string))
