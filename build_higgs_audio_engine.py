#!/usr/bin/env python3
"""Script to build TensorRT-LLM engine for HiggsAudio model."""

import torch

from tensorrt_llm.builder import BuildConfig, build
from tensorrt_llm.logger import logger
from tensorrt_llm.models.higgs_audio.model import HiggsAudioForCausalLM


def main():
    logger.set_level("info")

    gpu_device = torch.device("cuda", 0)
    torch.cuda.set_device(gpu_device)

    max_num_tokens = 2048

    build_config = BuildConfig()
    build_config.max_batch_size = 1
    build_config.max_input_len = max_num_tokens
    build_config.max_num_tokens = max_num_tokens
    build_config.plugin_config._remove_input_padding = True
    # build_config.plugin_config.dtype = "bfloat16"
    # build_config.plugin_config.gpt_attention_plugin = "bfloat16"
    # build_config.plugin_config.gemm_plugin = "bfloat16"
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

    trtllm_model = HiggsAudioForCausalLM.from_hugging_face()
    trtllm_model.config.max_position_embeddings = max_num_tokens

    engine = build(trtllm_model, build_config)
    engine.save("./higgs_audio_engine")


if __name__ == "__main__":
    main()
