#!/usr/bin/env python3
"""Script to build TensorRT-LLM engine for HiggsAudio model."""

from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig


def main():
    # max_mel_seq_len = 3000
    # max_seq_len = (max_mel_seq_len - 2) // 2 + 1

    # # Create output directory

    # plugin_config = PluginConfig()
    # plugin_config.dtype = "bfloat16"
    # plugin_config.use_fp8_context_fmha = False
    # plugin_config.paged_kv_cache = True
    # plugin_config.gpt_attention_plugin = "bfloat16"
    # plugin_config.gemm_plugin = "bfloat16"
    # plugin_config.remove_input_padding = True
    # build_config = BuildConfig(
    #     max_seq_len=max_seq_len,
    #     max_batch_size=1,
    #     max_beam_width=1,
    #     max_num_tokens=max_seq_len,
    #     max_prompt_embedding_table_size=1024,
    #     kv_cache_type=KVCacheType.PAGED,
    #     gather_context_logits=False,
    #     gather_generation_logits=False,
    #     strongly_typed=False,
    #     plugin_config=plugin_config,
    # )
    # config = HiggsAudioConfig.from_hugging_face(**kwargs)
    # hf_model = HiggsAudioModel.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
    # trtllm_model = HiggsAudioForCausalLM(config)
    config = HiggsAudioConfig.from_hugging_face("bosonai/higgs-audio-v2-generation-3B-base")
    print(config.to_dict())
    # trtllm_model = HiggsAudioForCausalLM.from_hugging_face(
    #     "bosonai/higgs-audio-v2-generation-3B-base"
    # )
    # trtllm_model.save_checkpoint("./higgs_audio_engine")

    # weights = load_weights_from_hf_model(hf_model, config)
    # trtllm_model.load(weights)
    # engine = build(trtllm_model, build_config)
    # engine.save("./higgs_audio_engine")


if __name__ == "__main__":
    main()

    # print("Build TRT-LLM engine...")
    # build_cmd = [
    #     "trtllm-build",
    #     f"--checkpoint_dir={workspace}/Qwen2-Audio",
    #     f"--gemm_plugin=float16",
    #     f"--gpt_attention_plugin=float16",
    #     f"--max_prompt_embedding_table_size=4096",
    #     f"--output_dir={engine_dir}",
    #     f"--max_batch_size={1}",
    # ]
