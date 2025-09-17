#!/usr/bin/env python3
"""Script to build TensorRT-LLM engine for HiggsAudio model."""

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

from tensorrt_llm.builder import BuildConfig, build
from tensorrt_llm.logger import logger
from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.models.higgs_audio.model import HiggsAudioForCausalLM


def main():
    logger.set_level("info")

    gpu_device = torch.device("cuda", 0)
    torch.cuda.set_device(gpu_device)
    trtllm_config = HiggsAudioConfig.from_hugging_face("bosonai/higgs-audio-v2-generation-3B-base")
    trtllm_model = HiggsAudioForCausalLM.from_hugging_face()

    max_num_tokens = trtllm_config.max_num_tokens
    print(f"max_num_tokens: {max_num_tokens}")
    build_config = BuildConfig()
    build_config.max_batch_size = 1
    build_config.max_input_len = max_num_tokens
    build_config.max_num_tokens = max_num_tokens
    build_config.opt_num_tokens = max_num_tokens // 2
    build_config.max_seq_len = max_num_tokens
    build_config.plugin_config.remove_input_padding = True
    build_config.plugin_config.dtype = "bfloat16"
    build_config.plugin_config.gpt_attention_plugin = "bfloat16"
    build_config.plugin_config.gemm_plugin = "bfloat16"
    # build_config.plugin_config.use_fp8_context_fmha = True
    # build_config.plugin_config._multiple_profiles = True
    # build_config.strongly_typed = False
    # build_config.plugin_config._gemm_swiglu_plugin = "FP8"
    # build_config.plugin_config._fp8_rowwise_gemm_plugin = "bfloat16"
    # build_config.plugin_config._low_latency_gemm_swiglu_plugin = "FP8"
    # build_config.plugin_config.low_latency_gemm_plugin = FP8
    # build_config.plugin_config.gemm_allreduce_plugin = "bfloat16"
    # build_config.plugin_config.context_fmha = True
    # build_config.plugin_config.norm_quant_fusion = True
    # build_config.plugin_config.user_buffer = True
    # build_config.plugin_config._use_paged_context_fmha = True
    # build_config.plugin_config._use_fp8_context_fmha = True
    # build_config.plugin_config._fuse_fp4_quant = True
    # build_config.plugin_config.paged_state = True
    # build_config.plugin_config._streamingllm = True
    # build_config.plugin_config.use_fused_mlp = True
    # build_config.plugin_config._pp_reduce_scatter = True
    # build_config.plugin_config._use_fused_mlp = True

    trtllm_model.config.max_position_embeddings = max_num_tokens

    engine = build(trtllm_model, build_config)
    engine.save("./higgs_audio_engine")


if __name__ == "__main__":
    main()
