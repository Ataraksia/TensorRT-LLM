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
import os
from pathlib import Path
from typing import Dict, Union

import transformers
from huggingface_hub import hf_hub_download, repo_exists

from tensorrt_llm.models.convert_utils import infer_dtype
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
        audio_dual_ffn_layers: list = None,
        audio_ffn_hidden_size: int = 3072,
        audio_ffn_intermediate_size: int = 8192,
        text_vocab_size: int = 128256,
        audio_vocab_size: int = 8208,
        # Audio codebook configuration
        num_codebooks: int = 8,
        codebook_size: int = 1024 + 2,  # +2 for special tokens
        audio_stream_bos_id: int = 1024,
        audio_stream_eos_id: int = 1025,
        # Special tokens
        audio_bos_token: str = "<|audio_bos|>",
        audio_bos_id: int = 128011,
        audio_eos_token: str = "<|audio_eos|>",
        audio_eos_id: int = 128012,
        audio_out_bos_token: str = "<|audio_out_bos|>",
        audio_out_bos_id: int = 128013,
        audio_in_token: str = "<|AUDIO|>",
        audio_out_token: str = "<|AUDIO_OUT|>",
        audio_in_idx: int = 128015,
        audio_out_idx: int = 128016,
        pad_token_id: int = 128001,
        max_num_tokens=2048,
        **kwargs,
    ):
        self.text_vocab_size = text_vocab_size
        self.audio_vocab_size = audio_vocab_size

        # Initialize base PretrainedConfig
        super().__init__(**kwargs)
        self.model_type = "higgs_audio"
        # Adapter configuration
        self.audio_adapter_type = audio_adapter_type
        self.audio_dual_ffn_layers = audio_dual_ffn_layers or list(range(28))  # Default: all layers
        self.audio_ffn_hidden_size = audio_ffn_hidden_size
        self.audio_ffn_intermediate_size = audio_ffn_intermediate_size

        # Audio codebook parameters
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id

        # Special tokens
        self.audio_bos_token = audio_bos_token
        self.audio_bos_id = audio_bos_id
        self.audio_eos_token = audio_eos_token
        self.audio_eos_id = audio_eos_id
        self.audio_out_bos_token = audio_out_bos_token
        self.audio_out_bos_id = audio_out_bos_id
        self.audio_in_token = audio_in_token
        self.audio_out_token = audio_out_token
        self.audio_in_idx = audio_in_idx
        self.audio_out_idx = audio_out_idx
        self.pad_token_id = pad_token_id

        self.max_num_tokens = max_num_tokens

    def to_dict(self) -> Dict:
        output = super().to_dict()
        # Add all our custom attributes
        custom_attrs = [
            "text_vocab_size",
            "audio_vocab_size",
            "audio_adapter_type",
            "audio_dual_ffn_layers",
            "audio_ffn_hidden_size",
            "audio_ffn_intermediate_size",
            "num_codebooks",
            "codebook_size",
            "audio_stream_bos_id",
            "audio_stream_eos_id",
            "audio_bos_token",
            "audio_bos_id",
            "audio_eos_token",
            "audio_eos_id",
            "audio_out_bos_token",
            "audio_out_bos_id",
            "audio_in_token",
            "audio_out_token",
            "audio_in_idx",
            "audio_out_idx",
            "pad_token_id",
        ]
        for attr in custom_attrs:
            if hasattr(self, attr):
                output[attr] = getattr(self, attr)

        return output

    @classmethod
    def from_hugging_face(
        cls,
        hf_config_or_path_or_dir: Union[
            str, "transformers.PretrainedConfig"
        ] = "bosonai/higgs-audio-v2-generation-3B-base",
        engine_dir: str = "./higgs_audio_engine",
        dtype: str = "bfloat16",
        **kwargs,
    ) -> "HiggsAudioConfig":
        if isinstance(hf_config_or_path_or_dir, transformers.PretrainedConfig):
            hf_config = hf_config_or_path_or_dir
            hf_config = hf_config.text_config
        else:
            hf_path_or_dir = str(hf_config_or_path_or_dir)
            if os.path.isdir(hf_path_or_dir):
                hf_config = json.load(Path(hf_path_or_dir + "/config.json").open("r"))
            elif repo_exists(hf_path_or_dir):
                config_file = hf_hub_download(repo_id=hf_path_or_dir, filename="config.json")
                hf_config = json.load(Path(config_file).open("r"))
            hf_config = hf_config["text_config"]
        # Keep the custom architecture name for our model; we'll extend the TRT-LLM
        # builder to recognize it as a decoder-only model when preparing inputs.
        hf_config["architectures"] = ["HiggsAudioForCausalLM"]
        attn_bias = False  # All existing Qwen models have attn bias
        rotary_scaling = getattr(hf_config, "rope_scaling", None)
        seq_length = getattr(hf_config, "seq_length", 2048)
        use_logn_attn = getattr(hf_config, "use_logn_attn", False)
        disable_weight_only_quant_plugin = kwargs.pop("disable_weight_only_quant_plugin", False)
        rotary_base = getattr(hf_config, "rope_theta", 100000.0)
        num_labels = 1
        dtype = infer_dtype(dtype, getattr(hf_config, "torch_dtype", None))
        rotary_embedding_dim = None
        return cls(
            architecture=hf_config["architectures"][0],
            dtype=dtype,
            num_hidden_layers=hf_config["num_hidden_layers"],
            num_attention_heads=hf_config["num_attention_heads"],
            hidden_size=hf_config["hidden_size"],
            intermediate_size=hf_config["intermediate_size"],
            num_key_value_heads=hf_config["num_key_value_heads"],
            head_size=hf_config["head_dim"],
            # IMPORTANT: We generate only audio tokens in TensorRT-LLM, so the
            # runtime's sampling vocabulary must match the audio head size.
            # Use audio vocab (num_codebooks * (codebook_size)) instead of text vocab.
            vocab_size=(1024 + 2) * 8,
            max_position_embeddings=hf_config["max_position_embeddings"],
            rotary_embedding_dim=rotary_embedding_dim,
            hidden_act=hf_config["hidden_act"],
            norm_epsilon=hf_config["rms_norm_eps"],
            attn_bias=attn_bias,
            rotary_base=rotary_base,
            rotary_scaling=rotary_scaling,
            disable_weight_only_quant_plugin=disable_weight_only_quant_plugin,
            seq_length=seq_length,
            use_logn_attn=use_logn_attn,
            num_labels=num_labels,
            tie_word_embeddings=hf_config["tie_word_embeddings"],
            **kwargs,
        )
