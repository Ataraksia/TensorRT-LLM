#!/usr/bin/env python3
"""Batch convert multiple Higgs Audio model weights from vLLM/HuggingFace to TensorRT-LLM format.

Batch conversion script for processing multiple model directories in parallel.
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Add the tensorrt_llm package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from tensorrt_llm.logger import logger


def convert_single_model(args_dict):
    """Convert a single model using subprocess.

    Args:
        args_dict: Dictionary containing conversion arguments

    Returns:
        tuple: (model_name, success, error_message)
    """
    import subprocess

    model_name = args_dict["model_name"]
    source_dir = args_dict["source_dir"]
    output_dir = args_dict["output_dir"]

    # Build command
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "convert_weights.py"),
        source_dir,
        output_dir,
    ]

    # Add optional arguments
    for arg_name in ["dtype", "device", "text_tokenizer_dir", "audio_tokenizer_dir"]:
        if arg_name in args_dict and args_dict[arg_name]:
            cmd.extend([f"--{arg_name.replace('_', '-')}", args_dict[arg_name]])

    # Add boolean flags
    for flag in ["strict", "skip_tokenizer", "skip_validation", "verbose"]:
        if args_dict.get(flag):
            cmd.append(f"--{flag.replace('_', '-')}")

    if args_dict.get("report_path"):
        cmd.extend(["--report-path", args_dict["report_path"]])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout per model
        )

        if result.returncode == 0:
            return (model_name, True, None)
        else:
            return (model_name, False, result.stderr or result.stdout)

    except subprocess.TimeoutExpired:
        return (model_name, False, "Conversion timed out after 1 hour")
    except Exception as e:
        return (model_name, False, str(e))


def main():
    """Main batch conversion function."""
    parser = argparse.ArgumentParser(
        description="Batch convert Higgs Audio models from vLLM/HF to TensorRT-LLM format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input specification
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str, help="JSON config file specifying models to convert")
    group.add_argument(
        "--input-dir", type=str, help="Directory containing multiple model subdirectories"
    )

    # Output directory
    parser.add_argument(
        "output_base_dir", type=str, help="Base output directory for converted models"
    )

    # Processing options
    parser.add_argument(
        "--max-workers", type=int, default=2, help="Maximum number of parallel conversion processes"
    )
    parser.add_argument(
        "--filter-pattern",
        type=str,
        help="Only process directories matching this pattern (for --input-dir)",
    )

    # Conversion options (passed to individual conversions)
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
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        import logging

        logging.basicConfig(level=logging.INFO)

    # Determine models to convert
    models_to_convert = []

    if args.config:
        # Load from config file
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Config file does not exist: {config_path}")
            return 1

        try:
            with open(config_path, "r") as f:
                config = json.load(f)

            for model_config in config.get("models", []):
                models_to_convert.append(
                    {
                        "model_name": model_config["name"],
                        "source_dir": model_config["source_dir"],
                        "output_dir": str(Path(args.output_base_dir) / model_config["name"]),
                        **{
                            k: v for k, v in model_config.items() if k not in ["name", "source_dir"]
                        },
                    }
                )
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            return 1

    else:
        # Scan input directory
        input_dir = Path(args.input_dir)
        if not input_dir.exists() or not input_dir.is_dir():
            logger.error(f"Input directory does not exist or is not a directory: {input_dir}")
            return 1

        for model_dir in input_dir.iterdir():
            if not model_dir.is_dir():
                continue

            # Apply filter pattern if specified
            if args.filter_pattern and args.filter_pattern not in model_dir.name:
                continue

            # Check if it looks like a model directory
            config_file = model_dir / "config.json"
            if not config_file.exists():
                logger.warning(f"Skipping {model_dir.name} - no config.json found")
                continue

            models_to_convert.append(
                {
                    "model_name": model_dir.name,
                    "source_dir": str(model_dir),
                    "output_dir": str(Path(args.output_base_dir) / model_dir.name),
                }
            )

    if not models_to_convert:
        logger.error("No models found to convert")
        return 1

    logger.info(f"Found {len(models_to_convert)} models to convert")

    # Add common arguments to all conversions
    common_args = {
        "dtype": args.dtype,
        "device": args.device,
        "text_tokenizer_dir": args.text_tokenizer_dir,
        "audio_tokenizer_dir": args.audio_tokenizer_dir,
        "strict": args.strict,
        "skip_tokenizer": args.skip_tokenizer,
        "skip_validation": args.skip_validation,
        "verbose": args.verbose,
    }

    for model_args in models_to_convert:
        model_args.update(common_args)
        # Add individual report path
        model_args["report_path"] = str(Path(model_args["output_dir"]) / "conversion_report.json")

    # Create output directories
    for model_args in models_to_convert:
        Path(model_args["output_dir"]).mkdir(parents=True, exist_ok=True)

    # Run conversions in parallel
    successful_conversions = 0
    failed_conversions = []

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all jobs
        future_to_model = {
            executor.submit(convert_single_model, model_args): model_args["model_name"]
            for model_args in models_to_convert
        }

        # Process completed jobs
        for future in as_completed(future_to_model):
            model_name, success, error_message = future.result()

            if success:
                logger.info(f"✓ Successfully converted {model_name}")
                successful_conversions += 1
            else:
                logger.error(f"✗ Failed to convert {model_name}: {error_message}")
                failed_conversions.append((model_name, error_message))

    # Generate summary report
    summary = {
        "total_models": len(models_to_convert),
        "successful_conversions": successful_conversions,
        "failed_conversions": len(failed_conversions),
        "failures": failed_conversions,
    }

    summary_path = Path(args.output_base_dir) / "batch_conversion_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    logger.info("\nBatch conversion completed:")
    logger.info(f"  Total models: {len(models_to_convert)}")
    logger.info(f"  Successful: {successful_conversions}")
    logger.info(f"  Failed: {len(failed_conversions)}")
    logger.info(f"  Summary saved to: {summary_path}")

    if failed_conversions:
        logger.info("\nFailed conversions:")
        for model_name, error in failed_conversions:
            logger.info(f"  - {model_name}: {error}")

    return 0 if len(failed_conversions) == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
