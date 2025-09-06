#!/usr/bin/env python3
"""Script to build TensorRT-LLM engine for HiggsAudio model."""

import argparse
import sys
import time
from pathlib import Path

from tensorrt_llm import Mapping, logger
from tensorrt_llm._utils import to_json_file
from tensorrt_llm.builder import Builder
from tensorrt_llm.models.higgs_audio import HiggsAudioConfig
from tensorrt_llm.models.higgs_audio.model import HiggsAudioForCausalLM
from tensorrt_llm.network import net_guard
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

    # Set up logging
    if args.log_level == "info":
        logger.set_level("info")
    elif args.log_level == "verbose":
        logger.set_level("verbose")
    elif args.log_level == "warning":
        logger.set_level("warning")
    elif args.log_level == "error":
        logger.set_level("error")
    else:
        logger.set_level("info")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
    plugin_config = PluginConfig()
    plugin_config.dtype = args.dtype
    plugin_config.use_fp8_context_fmha = False
    plugin_config.paged_kv_cache = True
    plugin_config.gpt_attention_plugin = "auto"
    plugin_config.gemm_plugin = args.dtype
    plugin_config.remove_input_padding = True
    builder = Builder()
    max_len = 1024
    builder_config = builder.create_builder_config(
        name="higgs_audio",
        precision=args.dtype,
        tensor_parallel=mapping.tp_size,
        pipeline_parallel=mapping.pp_size,
        plugin_config=plugin_config,
        max_multimodal_length=128,
        max_batch_size=1,
        max_input_len=max_len // 2,
        max_output_len=max_len // 2,
        max_beam_width=1,
        max_num_tokens=max_len,
        kv_cache_type="PAGED",
        builder_optimization_level=5,
        timing_cache=None,
        parallel_build=True,
    )
    rank = 0
    # Build engines for all ranks
    output_dir.mkdir(exist_ok=True)

    logger.info(f"Building engine for rank {rank}...")
    start_time = time.time()

    network = builder.create_network()
    network.plugin_config = builder_config.plugin_config
    with net_guard(network):
        # Create the complete Higgs Audio model
        config = HiggsAudioConfig()
        higgs_audio_model = HiggsAudioForCausalLM(config)
        # Prepare inputs for network building with cache enabled
        inputs = higgs_audio_model.prepare_inputs(
            max_batch_size=builder_config.max_batch_size,
            max_input_len=builder_config.max_input_len,
            max_seq_len=builder_config.max_input_len + builder_config.max_output_len,
            max_num_tokens=builder_config.max_num_tokens,
            use_cache=True,
            max_beam_width=builder_config.max_beam_width,
        )
        outputs = higgs_audio_model(**inputs)
        if outputs is None:
            raise RuntimeError("Model forward pass returned None")
        if isinstance(outputs, tuple):
            if len(outputs) == 3:
                # (lm_logits, presents, hidden_states)
                logits = outputs[0]
            elif len(outputs) == 2:
                # (hidden_states, presents) or (lm_logits, hidden_states)
                logits = outputs[0]
            else:
                logits = outputs[0]
        else:
            logits = outputs
        logits.mark_output("lm_logits", args.dtype)

    logger.info("  → Starting engine compilation (this may take several minutes)...")
    engine = builder.build_engine(network, builder_config)

    if engine is None:
        logger.error(f"Failed to build engine for rank {rank}")
        sys.exit(1)

    # Save the engine
    engine_path = output_dir / "rank0.engine"
    with open(engine_path, "wb") as f:
        f.write(engine)

    build_time = time.time() - start_time
    logger.info(f"Engine for rank {rank} built successfully in {build_time:.2f}s")
    logger.info(f"Engine saved to {engine_path}")

    # Save config
    config_path = output_dir / "config.json"
    config_dict = {
        "version": "1.0",
        "pretrained_config": {
            **config.to_dict(),
            "architecture": "HiggsAudioForCausalLM",
            "dtype": str(args.dtype),
            "logits_dtype": "float32",
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": getattr(
                config,
                "num_key_value_heads",
                config.num_attention_heads,
            ),
            "head_size": config.hidden_size // config.num_attention_heads,
            "intermediate_size": getattr(config, "intermediate_size", config.hidden_size * 4),
            "norm_epsilon": getattr(config, "norm_epsilon", 1e-5),
            "position_embedding_type": "rope_gpt_neox",
            "world_size": mapping.tp_size * mapping.pp_size,
            "tp_size": mapping.tp_size,
            "pp_size": mapping.pp_size,
            "max_position_embeddings": getattr(config, "max_position_embeddings", 131072),
            "use_parallel_embedding": False,
            "embedding_sharding_dim": 0,
            "share_embedding_table": False,
            "quantization": {
                "quant_algo": None,
                "kv_cache_quant_algo": None,
            },
            "mapping": {
                "world_size": mapping.tp_size * mapping.pp_size,
                "tp_size": mapping.tp_size,
                "pp_size": mapping.pp_size,
            },
        },
        "build_config": {
            **builder_config.to_dict(),
            "max_num_tokens": builder_config.max_num_tokens,
            "kv_cache_type": "PAGED",
            "plugin_config": {**builder_config.plugin_config.to_dict()},
            "lora_config": {},
        },
    }
    to_json_file(config_dict, str(config_path))
    logger.info("  → Building TensorRT engine...")

    logger.info(f"Build completed! Engine saved to {output_dir}")
    logger.info(f"Config saved to {config_path}")


if __name__ == "__main__":
    main()
