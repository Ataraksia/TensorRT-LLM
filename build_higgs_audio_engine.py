#!/usr/bin/env python3
"""Script to build TensorRT-LLM engine for HiggsAudio model."""

import torch

from tensorrt_llm.builder import BuildConfig, KVCacheType, PluginConfig, build
from tensorrt_llm.models.higgs_audio.model import HiggsAudioForCausalLM


def main():
    max_len = 2048
    plugin_config = PluginConfig()
    plugin_config.dtype = "bfloat16"
    plugin_config.use_fp8_context_fmha = False
    plugin_config.gpt_attention_plugin = "bfloat16"
    plugin_config.gemm_plugin = "bfloat16"
    plugin_config.remove_input_padding = True
    # plugin_config._multiple_profiles = True
    build_config = BuildConfig(
        max_batch_size=1,
        max_num_tokens=max_len,
        max_seq_len=max_len,
        kv_cache_type=KVCacheType.PAGED,
        gather_context_logits=False,
        gather_generation_logits=False,
        strongly_typed=False,
        plugin_config=plugin_config,
    )
    gpu_device = torch.device("cuda", 0)
    torch.cuda.set_device(gpu_device)
    trtllm_model = HiggsAudioForCausalLM.from_hugging_face(
        "bosonai/higgs-audio-v2-generation-3B-base"
    )
    # trtllm_model.save_checkpoint("./higgs_audio_engine")
    engine = build(trtllm_model, build_config)
    engine.save("./higgs_audio_engine")


if __name__ == "__main__":
    main()
