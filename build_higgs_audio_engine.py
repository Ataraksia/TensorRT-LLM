#!/usr/bin/env python3
"""Script to build TensorRT-LLM engine for HiggsAudio model."""

import argparse

from tensorrt_llm import logger
from tensorrt_llm.bindings import KVCacheType
from tensorrt_llm.builder import BuildConfig, build
from tensorrt_llm.models.higgs_audio import HiggsAudioConfig
from tensorrt_llm.models.higgs_audio.model import HiggsAudioForCausalLM
from tensorrt_llm.plugin import PluginConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Build HiggsAudio TensorRT-LLM engine")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="bosonai/higgs-audio-v2-generation-3B-base",
        help="HuggingFace model name or local model directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./higgs_audio_engine",
        help="Output directory for the TensorRT engine",
    )
    parser.add_argument(
        "--max_batch_size", type=int, default=1, help="Maximum batch size for the engine"
    )
    parser.add_argument(
        "--max_input_len", type=int, default=512, help="Maximum input sequence length"
    )
    parser.add_argument(
        "--max_output_len", type=int, default=512, help="Maximum output sequence length"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for the model",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="info",
        choices=["verbose", "info", "warning", "error", "internal_error"],
        help="Log level",
    )
    parser.add_argument(
        "--world_size", type=int, default=1, help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline parallelism size")
    return parser.parse_args()


def find_bytes_in_dict(d, path=""):
    for key, value in d.items():
        current_path = f"{path}.{key}" if path else key
        if isinstance(value, bytes):
            print(f"Found bytes at: {current_path}")
        elif isinstance(value, dict):
            find_bytes_in_dict(value, current_path)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, bytes):
                    print(f"Found bytes at: {current_path}[{i}]")
                elif isinstance(item, dict):
                    find_bytes_in_dict(item, f"{current_path}[{i}]")


def main():
    args = parse_args()
    logger.set_level(args.log_level)
    num_mul_bins = 128
    max_mel_seq_len = 3000
    max_seq_len = (max_mel_seq_len - 2) // 2 + 1

    # Create output directory

    plugin_config = PluginConfig()
    plugin_config.dtype = args.dtype
    plugin_config.use_fp8_context_fmha = False
    plugin_config.paged_kv_cache = True
    plugin_config.gpt_attention_plugin = "auto"
    plugin_config.gemm_plugin = args.dtype
    plugin_config.remove_input_padding = True
    build_config = BuildConfig(
        max_input_len=max_seq_len,
        max_seq_len=max_seq_len,
        max_batch_size=1,
        max_beam_width=1,
        max_num_tokens=max_seq_len,
        max_prompt_embedding_table_size=num_mul_bins,
        kv_cache_type=KVCacheType.PAGED,
        gather_context_logits=False,
        gather_generation_logits=False,
        strongly_typed=False,
        plugin_config=plugin_config,
    )
    config = HiggsAudioConfig()
    model = HiggsAudioForCausalLM(config)
    engine = build(model, build_config)
    engine.save(args.output_dir)


if __name__ == "__main__":
    main()
