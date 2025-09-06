#!/usr/bin/env python3

import argparse
import os
import time

import tensorrt_llm
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.higgs_audio.model import HiggsAudioForCausalLM
from tensorrt_llm.models.modeling_utils import QuantConfig


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./higgs-audio-v2-generation-3B-base",
        help="The huggingface model directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./higgs_trt_checkpoint",
        help="The output directory to save TRT checkpoint",
    )
    parser.add_argument("--tp_size", type=int, default=1, help="N-way tensor parallelism size")
    parser.add_argument("--pp_size", type=int, default=1, help="N-way pipeline parallelism size")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model weights and activations",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes for conversion"
    )
    return parser.parse_args()


def convert_and_save_rank(args, rank):
    """Convert and save checkpoint for a specific rank."""
    mapping = Mapping(
        world_size=args.tp_size * args.pp_size,
        rank=rank,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
    )

    logger.info(f"Converting HiggsAudio model from {args.model_dir}")
    logger.info(f"Rank {rank}, Mapping: {mapping}")

    # Create QuantConfig with default settings (no quantization)
    quant_config = QuantConfig()

    # Load the HiggsAudio model from HuggingFace
    higgs_audio = HiggsAudioForCausalLM.from_hugging_face(
        hf_model_dir=args.model_dir,
        dtype=args.dtype,
        mapping=mapping,
        quant_config=quant_config,
        skip_loading_weights=False,  # Skip weight loading for now
    )

    # Save the checkpoint
    logger.info(f"Saving checkpoint to {args.output_dir}")
    higgs_audio.save_checkpoint(args.output_dir, save_config=(rank == 0))

    logger.info(f"Rank {rank} conversion completed")
    del higgs_audio


def main():
    print(f"TensorRT-LLM version: {tensorrt_llm.__version__}")
    args = parse_arguments()

    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    world_size = args.tp_size * args.pp_size

    logger.info("Starting HiggsAudio model conversion...")
    logger.info(f"World size: {world_size}")
    logger.info(f"TP size: {args.tp_size}")
    logger.info(f"PP size: {args.pp_size}")

    # For now, only support single rank conversion
    if world_size == 1:
        convert_and_save_rank(args, 0)
    else:
        logger.error("Multi-rank conversion not implemented yet")
        return

    tok = time.time()
    t = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
    logger.info(f"Total time: {t}")
    logger.info(f"Checkpoint saved to {args.output_dir}")


if __name__ == "__main__":
    main()
