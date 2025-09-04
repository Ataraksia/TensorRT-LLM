#!/usr/bin/env python3
"""Convert Higgs Audio model weights from vLLM/HuggingFace format to TensorRT-LLM format.

Single conversion script for individual model directories.
"""

import argparse
import sys
from pathlib import Path

# Add the tensorrt_llm package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from tensorrt_llm.logger import logger
from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.models.higgs_audio.weight import (
    export_trtllm_tokenizer,
    generate_conversion_report,
    load_from_vllm_checkpoint,
    load_tokenizers,
    save_trtllm,
    validate_converted_weights,
)


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(
        description="Convert Higgs Audio model weights from vLLM/HF to TensorRT-LLM format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "source_dir", type=str, help="Source directory containing vLLM/HuggingFace model weights"
    )
    parser.add_argument(
        "output_dir", type=str, help="Output directory for converted TensorRT-LLM weights"
    )

    # Optional arguments
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float16",
        help="Data type for converted weights",
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu", help="Device for weight processing"
    )
    parser.add_argument(
        "--text-tokenizer-dir",
        type=str,
        default="/home/me/TTS/TensorRT-LLM/higgs-audio-v2-generation-3B-base",
        help="Text tokenizer directory",
    )
    parser.add_argument(
        "--audio-tokenizer-dir",
        type=str,
        default="/home/me/TTS/TensorRT-LLM/higgs-audio-v2-generation-3B-base-tokenizer",
        help="Audio tokenizer directory",
    )
    parser.add_argument("--strict", action="store_true", help="Fail on any validation errors")
    parser.add_argument(
        "--skip-tokenizer", action="store_true", help="Skip tokenizer loading and export"
    )
    parser.add_argument("--skip-validation", action="store_true", help="Skip weight validation")
    parser.add_argument(
        "--report-path",
        type=str,
        help="Path to save conversion report (default: output_dir/conversion_report.json)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        import logging

        logging.basicConfig(level=logging.INFO)

    # Validate paths
    source_path = Path(args.source_dir)
    output_path = Path(args.output_dir)

    if not source_path.exists():
        logger.error(f"Source directory does not exist: {source_path}")
        return 1

    if not source_path.is_dir():
        logger.error(f"Source path is not a directory: {source_path}")
        return 1

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Converting weights from {source_path} to {output_path}")
        logger.info(f"Using dtype: {args.dtype}, device: {args.device}")

        # Load configuration
        config = HiggsAudioConfig()
        logger.info("Loaded HiggsAudioConfig")

        # Load weights from checkpoint
        logger.info("Loading weights from vLLM checkpoint...")
        trt_llm_weights = load_from_vllm_checkpoint(
            model_dir=str(source_path), config=config, dtype=args.dtype, device=args.device
        )
        logger.info(f"Loaded {len(trt_llm_weights)} weight tensors")

        # Validate weights
        validation_result = None
        if not args.skip_validation:
            logger.info("Validating converted weights...")
            validation_result = validate_converted_weights(
                trt_llm_weights=trt_llm_weights, config=config, strict=args.strict
            )

            if not validation_result["valid"] and args.strict:
                logger.error("Validation failed with strict mode enabled")
                return 1

        # Save converted weights
        logger.info("Saving converted weights...")
        save_trtllm(weights=trt_llm_weights, output_dir=str(output_path), config=config)

        # Handle tokenizers
        if not args.skip_tokenizer:
            logger.info("Loading and exporting tokenizers...")
            tokenizer_info = load_tokenizers(
                text_tokenizer_dir=args.text_tokenizer_dir,
                audio_tokenizer_dir=args.audio_tokenizer_dir,
                config=config,
            )

            if tokenizer_info["errors"]:
                logger.error(f"Tokenizer loading failed: {tokenizer_info['errors']}")
                if args.strict:
                    return 1
            else:
                # Export tokenizer artifacts
                tokenizer_output_dir = output_path / "tokenizer"
                export_trtllm_tokenizer(tokenizer_info, str(tokenizer_output_dir))

        # Generate report
        if validation_result:
            report_path = args.report_path or str(output_path / "conversion_report.json")
            generate_conversion_report(
                validation_result=validation_result,
                output_path=report_path,
                source_dir=str(source_path),
                target_dir=str(output_path),
            )

        logger.info("Conversion completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
