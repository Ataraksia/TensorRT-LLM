#!/usr/bin/env python3
"""Script to build TensorRT-LLM engine for HiggsAudio model."""

import torch

from tensorrt_llm.builder import BuildConfig, KVCacheType, PluginConfig, build
from tensorrt_llm.models.higgs_audio.model import HiggsAudioForCausalLM


def main():
    max_mel_seq_len = 3000
    max_seq_len = (max_mel_seq_len - 2) // 2 + 1
    plugin_config = PluginConfig()
    plugin_config.dtype = "bfloat16"
    plugin_config.use_fp8_context_fmha = False
    plugin_config.gpt_attention_plugin = "bfloat16"
    plugin_config.gemm_plugin = "bfloat16"
    plugin_config.remove_input_padding = True
    build_config = BuildConfig(
        max_seq_len=max_seq_len,
        max_batch_size=1,
        max_beam_width=1,
        max_num_tokens=max_seq_len,
        max_prompt_embedding_table_size=1024,
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
    engine = build(trtllm_model, build_config)
    engine.save("./higgs_audio_engine")


if __name__ == "__main__":
    main()
