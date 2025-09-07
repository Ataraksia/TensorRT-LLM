#!/usr/bin/env python3
"""Script to build TensorRT-LLM engine for HiggsAudio model."""

import argparse
import sys
from pathlib import Path
from pickle import FALSE

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


def main():
    args = parse_args()
    logger.set_level(args.log_level)
    num_mul_bins = 128
    max_mel_seq_len = 3000
    max_seq_len = (max_mel_seq_len - 2) // 2 + 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        strongly_typed=FALSE,
        plugin_config=plugin_config,
    )
    config = HiggsAudioConfig()
    model = HiggsAudioForCausalLM(config)
    engine = build(model, build_config)

    if engine is None:
        logger.error(f"Failed to build engine for rank {0}")
        sys.exit(1)

    # Save the engine
    engine_path = output_dir / "rank0.engine"
    engine.save(engine_path)


if __name__ == "__main__":
    main()
