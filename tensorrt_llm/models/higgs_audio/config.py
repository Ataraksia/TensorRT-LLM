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

from tensorrt_llm.models.modeling_utils import PretrainedConfig


class HiggsAudioConfig(PretrainedConfig):
    """Configuration class for HiggsAudio multimodal model.

    HiggsAudio is a multimodal model combining text generation with audio processing
    capabilities. It uses a dual-FFN architecture that can process both text and
    audio tokens through separate processing paths.
    """

    def __init__(
        self,
        *,
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
        self.text_vocab_size = text_config.get("vocab_size", 128256)
        self.audio_vocab_size = audio_num_codebooks * (audio_codebook_size + 2)

        # Initialize base PretrainedConfig
        super().__init__(
            architecture="HiggsAudioForCausalLM",
            hidden_size=text_config.get("hidden_size", 3072),
            num_hidden_layers=text_config.get("num_hidden_layers", 28),
            num_attention_heads=24,  # Default for LLaMA-3.2-3B
            vocab_size=text_config.get("vocab_size", 128256),
            hidden_act="swiglu",
            dtype="bfloat16",
            **kwargs,
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

    @classmethod
    def from_hugging_face(
        cls,
        hf_config_or_dir: str,
        **kwargs,
    ) -> "HiggsAudioConfig":
        return cls(
            **kwargs,
        )
