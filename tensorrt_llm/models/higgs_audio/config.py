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
"""

import json
from typing import Dict, Optional, Union

from tensorrt_llm.models.modeling_utils import PretrainedConfig, QuantConfig


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
        quant_config: Optional[Union[QuantConfig, Dict]] = None,
        audio_tokenizer_config: Optional[Dict] = None,
        dtype: str = "bfloat16",  # Add dtype parameter
        # Adapter configuration
        audio_adapter_type: str = "dual_ffn_fast_forward",
        audio_ffn_hidden_size: int = 4096,
        audio_ffn_intermediate_size: int = 8192,
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

        if quant_config is None:
            quant_config = QuantConfig()
        elif isinstance(quant_config, dict):
            quant_config = QuantConfig(**quant_config)

        # Remove conflicting parameters from kwargs
        filtered_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["hidden_size", "num_hidden_layers", "vocab_size", "dtype"]
        }

        # Initialize base PretrainedConfig
        super().__init__(
            architecture="HiggsAudioForCausalLM",
            dtype=dtype,  # Use the passed dtype parameter
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=24,  # Default for LLaMA-3.2-3B
            vocab_size=vocab_size,
            hidden_act="swiglu",
            **filtered_kwargs,
        )

        # Store nested configurations
        self.text_config = text_config
        self.audio_tokenizer_config = audio_tokenizer_config

        # Additional attributes required by TensorRT-LLM layers
        self.layer_idx_offset = (
            getattr(text_config, "layer_idx_offset", 0)
            if not isinstance(text_config, dict)
            else text_config.get("layer_idx_offset", 0)
        )
        self.max_position_embeddings = (
            getattr(text_config, "max_position_embeddings", 2048)
            if not isinstance(text_config, dict)
            else text_config.get("max_position_embeddings", 2048)
        )

        # LLaMA-specific attributes
        self.use_input_layernorm_in_first_layer = (
            getattr(text_config, "use_input_layernorm_in_first_layer", True)
            if not isinstance(text_config, dict)
            else text_config.get("use_input_layernorm_in_first_layer", True)
        )
        self.rms_norm_eps = (
            getattr(text_config, "rms_norm_eps", 1e-6)
            if not isinstance(text_config, dict)
            else text_config.get("rms_norm_eps", 1e-6)
        )
        self.intermediate_size = (
            getattr(text_config, "intermediate_size", 11008)
            if not isinstance(text_config, dict)
            else text_config.get("intermediate_size", 11008)
        )
        self.num_key_value_heads = (
            getattr(text_config, "num_key_value_heads", self.num_attention_heads)
            if not isinstance(text_config, dict)
            else text_config.get("num_key_value_heads", self.num_attention_heads)
        )
        self.rope_theta = (
            getattr(text_config, "rope_theta", 10000.0)
            if not isinstance(text_config, dict)
            else text_config.get("rope_theta", 10000.0)
        )
        self.rotary_base = (
            getattr(text_config, "rotary_base", 10000.0)
            if not isinstance(text_config, dict)
            else text_config.get("rotary_base", 10000.0)
        )
        self.mlp_bias = (
            getattr(text_config, "mlp_bias", False)
            if not isinstance(text_config, dict)
            else text_config.get("mlp_bias", False)
        )
        self.attention_bias = (
            getattr(text_config, "attention_bias", False)
            if not isinstance(text_config, dict)
            else text_config.get("attention_bias", False)
        )
        self.attn_bias = (
            getattr(text_config, "attn_bias", False)
            if not isinstance(text_config, dict)
            else text_config.get("attn_bias", False)
        )

        # Adapter configuration
        self.audio_adapter_type = audio_adapter_type
        self.audio_ffn_hidden_size = audio_ffn_hidden_size
        self.audio_ffn_intermediate_size = audio_ffn_intermediate_size

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

    def to_dict(self) -> Dict:
        output = super().to_dict()

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
