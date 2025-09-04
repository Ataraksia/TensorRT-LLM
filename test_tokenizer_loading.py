#!/usr/bin/env python3
"""Test script to validate tokenizer loading functionality for Higgs Audio model."""

import sys

# Add the tensorrt_llm package to Python path
sys.path.insert(0, "/home/me/TTS/TensorRT-LLM")

from pathlib import Path

from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.models.higgs_audio.weight import (
    export_trtllm_tokenizer,
    load_tokenizers,
    validate_special_tokens,
)


def test_tokenizer_loading():
    """Test the tokenizer loading functionality."""
    print("=" * 60)
    print("Testing Higgs Audio Tokenizer Loading")
    print("=" * 60)

    # Test 1: Load tokenizers with default paths
    print("\n1. Loading tokenizers with default paths...")
    tokenizer_info = load_tokenizers()

    print(f"Errors: {len(tokenizer_info['errors'])}")
    print(f"Warnings: {len(tokenizer_info['warnings'])}")

    if tokenizer_info["errors"]:
        print("ERRORS:")
        for error in tokenizer_info["errors"]:
            print(f"  - {error}")

    if tokenizer_info["warnings"]:
        print("WARNINGS:")
        for warning in tokenizer_info["warnings"]:
            print(f"  - {warning}")

    print(f"Vocab size: {tokenizer_info['vocab_size']}")
    print(f"Special tokens validated: {tokenizer_info['special_tokens_validated']}")

    # Test 2: Validate special tokens directly
    print("\n2. Testing special token validation...")
    if tokenizer_info["text_tokenizer"] and "config" in tokenizer_info["text_tokenizer"]:
        validation_result = validate_special_tokens(tokenizer_info["text_tokenizer"]["config"])
        print(f"Special tokens valid: {validation_result['valid']}")
        if validation_result["errors"]:
            print("Validation errors:")
            for error in validation_result["errors"]:
                print(f"  - {error}")
        if validation_result["warnings"]:
            print("Validation warnings:")
            for warning in validation_result["warnings"]:
                print(f"  - {warning}")

    # Test 3: Export TRT-LLM tokenizer artifacts
    print("\n3. Testing TRT-LLM tokenizer export...")
    if not tokenizer_info["errors"]:
        try:
            test_output_dir = "/tmp/test_tokenizer_export"
            export_trtllm_tokenizer(tokenizer_info, test_output_dir)

            # Check if files were created
            output_path = Path(test_output_dir)
            files_created = list(output_path.glob("*.json"))
            print(f"Files created: {[f.name for f in files_created]}")

        except Exception as e:
            print(f"Export failed: {e}")

    # Test 4: Config validation
    print("\n4. Testing with HiggsAudioConfig...")
    try:
        config = HiggsAudioConfig()
        tokenizer_info_with_config = load_tokenizers(config=config)
        print(
            f"Config validation completed with {len(tokenizer_info_with_config['warnings'])} warnings"
        )
    except Exception as e:
        print(f"Config validation failed: {e}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

    return tokenizer_info


if __name__ == "__main__":
    test_tokenizer_loading()
