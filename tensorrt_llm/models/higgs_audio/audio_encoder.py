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
import torch

from tensorrt_llm.functional import gelu
from tensorrt_llm.layers import MLP, Attention, Conv1d, Embedding, LayerNorm, Linear
from tensorrt_llm.module import Module, ModuleList

from .config import HiggsAudioConfig


class HiggsAudioFeatureProjector(Module):
    def __init__(self, config: HiggsAudioConfig, dtype: str = "float16"):
        super().__init__()
        self.linear = Linear(
            in_features=config.audio_encoder_config.d_model,
            out_features=config.text_config.hidden_size,
            dtype=dtype,
        )

    def forward(self, hidden_states):
        return self.linear(hidden_states)


class WhisperEncoderLayer(Module):
    def __init__(self, config: HiggsAudioConfig, dtype: str = "float16"):
        super().__init__()
        self.config = config.audio_encoder_config
        self.dtype = dtype

        self.self_attn = Attention(
            hidden_size=self.config.d_model,
            num_attention_heads=self.config.encoder_attention_heads,
            attention_head_size=self.config.d_model // self.config.encoder_attention_heads,
            dtype=self.dtype,
        )

        self.self_attn_layer_norm = LayerNorm(
            normalized_shape=self.config.d_model, dtype=self.dtype
        )

        self.mlp = MLP(
            hidden_size=self.config.d_model,
            ffn_hidden_size=self.config.encoder_ffn_dim,
            hidden_act=self.config.activation_function,
            dtype=self.dtype,
        )

        self.final_layer_norm = LayerNorm(normalized_shape=self.config.d_model, dtype=self.dtype)

    def forward(
        self, hidden_states, attention_mask=None, use_flash_attn=True, gradient_checkpointing=False
    ):
        def custom_forward(*inputs):
            hidden_states = inputs[0]
            attention_mask = inputs[1]

            residual = hidden_states
            hidden_states = self.self_attn_layer_norm(hidden_states)

            hidden_states = self.self_attn(
                hidden_states, attention_mask=attention_mask, use_flash_attn=use_flash_attn
            )[0]

            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.final_layer_norm(hidden_states)

            hidden_states = self.mlp(hidden_states)

            hidden_states = residual + hidden_states

            return hidden_states

        if gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(custom_forward, hidden_states, attention_mask)
        else:
            return custom_forward(hidden_states, attention_mask)


class HiggsAudioEncoder(Module):
    def __init__(self, config: HiggsAudioConfig, dtype: str = "float16"):
        super().__init__()
        self.config = config.audio_encoder_config
        self.dtype = dtype

        self.embed_dim = self.config.d_model
        self.num_mel_bins = self.config.num_mel_bins

        self.conv1 = Conv1d(self.num_mel_bins, self.embed_dim, kernel_size=3, padding=1)
        self.conv2 = Conv1d(self.embed_dim, self.embed_dim, kernel_size=3, stride=2, padding=1)

        # Using functional gelu; no module needed

        self.embed_positions = Embedding(self.config.max_source_positions, self.embed_dim)

        self.layers = ModuleList(
            [WhisperEncoderLayer(config, dtype) for _ in range(self.config.encoder_layers)]
        )

        self.layer_norm = LayerNorm(normalized_shape=self.embed_dim, dtype=self.dtype)

        self.gradient_checkpointing = False

    def forward(self, inputs, input_lengths=None, use_flash_attn=True, return_projected=True):
        # TODO: Add zero-shape tensor support

        hidden_states = self.conv1(inputs)
        hidden_states = gelu(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = gelu(hidden_states)

        hidden_states = hidden_states.permute(0, 2, 1)

        positions = self.embed_positions(hidden_states.shape[1])
        hidden_states = hidden_states + positions

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=None,
                use_flash_attn=use_flash_attn,
                gradient_checkpointing=self.gradient_checkpointing,
            )

        hidden_states = self.layer_norm(hidden_states)

        # Optional temporal pooling removed for now to avoid dependency on AvgPool1d

        return hidden_states
