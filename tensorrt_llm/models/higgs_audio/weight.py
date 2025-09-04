# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import json
from pathlib import Path
from typing import Dict

import torch
from safetensors import safe_open

from tensorrt_llm.logger import logger

from .config import HiggsAudioConfig

# Special token IDs for Higgs Audio model
SPECIAL_TOKENS = {
    "audio_bos_token_id": 128011,  # <|audio_bos|>
    "audio_eos_token_id": 128012,  # <|audio_eos|>
    "audio_out_bos_token_id": 128013,  # <|audio_out_bos|>
    "audio_in_token_id": 128015,  # <|AUDIO|>
    "audio_out_token_id": 128016,  # <|AUDIO_OUT|>
    "audio_stream_bos_id": 1024,  # <|audio_stream_bos|>
    "audio_stream_eos_id": 1025,  # <|audio_stream_eos|>
}


def load_tokenizers(
    text_tokenizer_dir: str = "/home/me/TTS/TensorRT-LLM/higgs-audio-v2-generation-3B-base",
    audio_tokenizer_dir: str = "/home/me/TTS/TensorRT-LLM/higgs-audio-v2-generation-3B-base-tokenizer",
    config: HiggsAudioConfig = None,
):
    """Load both text and audio tokenizers and verify special token alignment.

    Args:
        text_tokenizer_dir: Path to the text tokenizer directory
        audio_tokenizer_dir: Path to the audio tokenizer directory
        config: HiggsAudioConfig instance for validation

    Returns:
        Dict containing tokenizer info and validation results
    """
    logger.info(f"Loading text tokenizer from {text_tokenizer_dir}...")
    logger.info(f"Loading audio tokenizer from {audio_tokenizer_dir}...")

    tokenizer_info = {
        "text_tokenizer": None,
        "audio_tokenizer": None,
        "special_tokens_validated": False,
        "vocab_size": None,
        "errors": [],
        "warnings": [],
    }

    # Load text tokenizer configuration
    text_tokenizer_dir = Path(text_tokenizer_dir)
    audio_tokenizer_dir = Path(audio_tokenizer_dir)

    # Check if directories exist
    if not text_tokenizer_dir.exists():
        error_msg = f"Text tokenizer directory not found: {text_tokenizer_dir}"
        logger.error(error_msg)
        tokenizer_info["errors"].append(error_msg)
        return tokenizer_info

    if not audio_tokenizer_dir.exists():
        error_msg = f"Audio tokenizer directory not found: {audio_tokenizer_dir}"
        logger.error(error_msg)
        tokenizer_info["errors"].append(error_msg)
        return tokenizer_info

    # Load text tokenizer configuration files
    try:
        # Load tokenizer config
        tokenizer_config_path = text_tokenizer_dir / "tokenizer_config.json"
        if tokenizer_config_path.exists():
            with open(tokenizer_config_path, "r") as f:
                tokenizer_config = json.load(f)
            tokenizer_info["text_tokenizer"] = {"config": tokenizer_config}

            # Extract vocab size from config
            if "vocab_size" in tokenizer_config:
                tokenizer_info["vocab_size"] = tokenizer_config["vocab_size"]
            elif "added_tokens_decoder" in tokenizer_config:
                # Calculate vocab size from added tokens decoder
                max_token_id = max(int(k) for k in tokenizer_config["added_tokens_decoder"].keys())
                tokenizer_info["vocab_size"] = max_token_id + 1

            logger.info(f"Text tokenizer vocab size: {tokenizer_info['vocab_size']}")
        else:
            error_msg = f"tokenizer_config.json not found in {text_tokenizer_dir}"
            logger.error(error_msg)
            tokenizer_info["errors"].append(error_msg)
            return tokenizer_info

        # Load special tokens map
        special_tokens_path = text_tokenizer_dir / "special_tokens_map.json"
        if special_tokens_path.exists():
            with open(special_tokens_path, "r") as f:
                special_tokens_map = json.load(f)
            tokenizer_info["text_tokenizer"]["special_tokens_map"] = special_tokens_map

        # Validate special tokens
        validation_result = validate_special_tokens(tokenizer_config)
        tokenizer_info["special_tokens_validated"] = validation_result["valid"]
        if not validation_result["valid"]:
            tokenizer_info["errors"].extend(validation_result["errors"])
        if validation_result["warnings"]:
            tokenizer_info["warnings"].extend(validation_result["warnings"])

    except Exception as e:
        error_msg = f"Failed to load text tokenizer: {str(e)}"
        logger.error(error_msg)
        tokenizer_info["errors"].append(error_msg)
        return tokenizer_info

    # Load audio tokenizer configuration
    try:
        audio_config_path = audio_tokenizer_dir / "config.json"
        if audio_config_path.exists():
            with open(audio_config_path, "r") as f:
                audio_config = json.load(f)
            tokenizer_info["audio_tokenizer"] = {"config": audio_config}

            # Validate audio tokenizer config
            required_audio_keys = ["n_q", "bins", "sample_rate"]
            missing_keys = [k for k in required_audio_keys if k not in audio_config]
            if missing_keys:
                error_msg = f"Audio tokenizer config missing required keys: {missing_keys}"
                logger.error(error_msg)
                tokenizer_info["errors"].append(error_msg)
            else:
                logger.info(
                    f"Audio tokenizer: {audio_config['n_q']} codebooks, "
                    f"{audio_config['bins']} codebook size, "
                    f"{audio_config['sample_rate']} Hz sample rate"
                )
        else:
            error_msg = f"config.json not found in {audio_tokenizer_dir}"
            logger.error(error_msg)
            tokenizer_info["errors"].append(error_msg)

        # Check for audio tokenizer model file
        model_path = audio_tokenizer_dir / "model.pth"
        if model_path.exists():
            tokenizer_info["audio_tokenizer"]["model_path"] = str(model_path)
            logger.info(f"Audio tokenizer model found at: {model_path}")
        else:
            warning_msg = f"Audio tokenizer model.pth not found in {audio_tokenizer_dir}"
            logger.warning(warning_msg)
            tokenizer_info["warnings"].append(warning_msg)

    except Exception as e:
        error_msg = f"Failed to load audio tokenizer: {str(e)}"
        logger.error(error_msg)
        tokenizer_info["errors"].append(error_msg)

    # Validate against config if provided
    if config and tokenizer_info["vocab_size"]:
        expected_vocab_size = getattr(config, "vocab_size", None)
        if expected_vocab_size and tokenizer_info["vocab_size"] != expected_vocab_size:
            warning_msg = (
                f"Vocab size mismatch: tokenizer has {tokenizer_info['vocab_size']}, "
                f"config expects {expected_vocab_size}"
            )
            logger.warning(warning_msg)
            tokenizer_info["warnings"].append(warning_msg)

    if not tokenizer_info["errors"]:
        logger.info("Successfully loaded both text and audio tokenizers")
    else:
        logger.error(f"Failed to load tokenizers with {len(tokenizer_info['errors'])} errors")

    return tokenizer_info


def validate_special_tokens(tokenizer_config: Dict) -> Dict:
    """Validate that all required special tokens are present with correct IDs.

    Args:
        tokenizer_config: Loaded tokenizer configuration

    Returns:
        Dict with validation results
    """
    result = {"valid": True, "errors": [], "warnings": []}

    if "added_tokens_decoder" not in tokenizer_config:
        result["valid"] = False
        result["errors"].append("No added_tokens_decoder found in tokenizer config")
        return result

    added_tokens = tokenizer_config["added_tokens_decoder"]

    # Check for required audio special tokens
    required_tokens = [
        (128011, "<|audio_bos|>"),
        (128012, "<|audio_eos|>"),
        (128013, "<|audio_out_bos|>"),
        (128015, "<|AUDIO|>"),
        (128016, "<|AUDIO_OUT|>"),
    ]

    for token_id, expected_content in required_tokens:
        token_id_str = str(token_id)
        if token_id_str not in added_tokens:
            result["valid"] = False
            result["errors"].append(f"Missing required token ID {token_id} ({expected_content})")
        else:
            actual_content = added_tokens[token_id_str].get("content", "")
            if actual_content != expected_content:
                result["valid"] = False
                result["errors"].append(
                    f"Token ID {token_id} has content '{actual_content}', expected '{expected_content}'"
                )

    # Check for streaming tokens (may not be present in all models)
    streaming_tokens = [(1024, "<|audio_stream_bos|>"), (1025, "<|audio_stream_eos|>")]

    for token_id, expected_content in streaming_tokens:
        token_id_str = str(token_id)
        if token_id_str not in added_tokens:
            result["warnings"].append(
                f"Streaming token ID {token_id} ({expected_content}) not found"
            )
        else:
            actual_content = added_tokens[token_id_str].get("content", "")
            if actual_content != expected_content:
                result["warnings"].append(
                    f"Streaming token ID {token_id} has content '{actual_content}', expected '{expected_content}'"
                )

    return result


def export_trtllm_tokenizer(tokenizer_info: Dict, output_dir: str):
    """Export tokenizer artifacts required by TensorRT-LLM runtime.

    Args:
        tokenizer_info: Tokenizer info dict from load_tokenizers()
        output_dir: Directory to save TRT-LLM compatible tokenizer files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not tokenizer_info["text_tokenizer"]:
        logger.error("No text tokenizer loaded, cannot export")
        return

    logger.info(f"Exporting TensorRT-LLM tokenizer artifacts to {output_dir}")

    # Export tokenizer configuration
    tokenizer_config = tokenizer_info["text_tokenizer"]["config"].copy()

    # Add TRT-LLM specific fields
    tokenizer_config["model_type"] = "higgs_audio"
    tokenizer_config["audio_special_tokens"] = SPECIAL_TOKENS

    # Save config
    config_path = output_dir / "tokenizer_config.json"
    with open(config_path, "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    logger.info(f"Saved tokenizer config to {config_path}")

    # Copy special tokens map if present
    if "special_tokens_map" in tokenizer_info["text_tokenizer"]:
        special_tokens_path = output_dir / "special_tokens_map.json"
        with open(special_tokens_path, "w") as f:
            json.dump(tokenizer_info["text_tokenizer"]["special_tokens_map"], f, indent=2)
        logger.info(f"Saved special tokens map to {special_tokens_path}")

    # Export audio tokenizer info if present
    if tokenizer_info["audio_tokenizer"]:
        audio_config_path = output_dir / "audio_tokenizer_config.json"
        with open(audio_config_path, "w") as f:
            json.dump(tokenizer_info["audio_tokenizer"]["config"], f, indent=2)
        logger.info(f"Saved audio tokenizer config to {audio_config_path}")

    logger.info("Successfully exported TensorRT-LLM tokenizer artifacts")


def validate_converted_weights(
    trt_llm_weights: Dict[str, torch.Tensor], config: HiggsAudioConfig, strict: bool = True
) -> Dict:
    """Validate converted weights against expected shapes and configuration.

    Args:
        trt_llm_weights: Dictionary of converted TensorRT-LLM weights
        config: HiggsAudioConfig for validation
        strict: Whether to fail on any validation errors

    Returns:
        Dict containing validation results and statistics
    """
    logger.info("Validating converted weights...")

    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "statistics": {
            "total_weights": len(trt_llm_weights),
            "total_parameters": 0,
            "memory_mb": 0,
            "checksums": {},
        },
        "expected_shapes": {},
        "actual_shapes": {},
        "missing_weights": [],
        "unexpected_weights": [],
    }

    # Calculate statistics
    for name, tensor in trt_llm_weights.items():
        num_params = tensor.numel()
        validation_result["statistics"]["total_parameters"] += num_params
        validation_result["statistics"]["memory_mb"] += (
            tensor.element_size() * num_params / (1024 * 1024)
        )

        # Calculate checksum
        tensor_bytes = tensor.cpu().numpy().tobytes()
        checksum = hashlib.md5(tensor_bytes).hexdigest()
        validation_result["statistics"]["checksums"][name] = checksum

        validation_result["actual_shapes"][name] = list(tensor.shape)

    # Define expected shapes based on config
    expected_shapes = get_expected_weight_shapes(config)
    validation_result["expected_shapes"] = expected_shapes

    # Check for missing weights
    expected_keys = set(expected_shapes.keys())
    actual_keys = set(trt_llm_weights.keys())

    missing_keys = expected_keys - actual_keys
    unexpected_keys = actual_keys - expected_keys

    validation_result["missing_weights"] = list(missing_keys)
    validation_result["unexpected_weights"] = list(unexpected_keys)

    # Report missing weights
    if missing_keys:
        for key in missing_keys:
            error_msg = f"Missing expected weight: {key}"
            validation_result["errors"].append(error_msg)
            if strict:
                logger.error(error_msg)
            else:
                logger.warning(error_msg)

    # Report unexpected weights
    if unexpected_keys:
        for key in unexpected_keys:
            warning_msg = f"Unexpected weight found: {key}"
            validation_result["warnings"].append(warning_msg)
            logger.warning(warning_msg)

    # Validate shapes for matching keys
    common_keys = expected_keys & actual_keys
    for key in common_keys:
        expected_shape = expected_shapes[key]
        actual_shape = list(trt_llm_weights[key].shape)

        if expected_shape != actual_shape:
            error_msg = f"Shape mismatch for {key}: expected {expected_shape}, got {actual_shape}"
            validation_result["errors"].append(error_msg)
            if strict:
                logger.error(error_msg)
            else:
                logger.warning(error_msg)

    # Check for NaN/Inf values
    for name, tensor in trt_llm_weights.items():
        if torch.isnan(tensor).any():
            error_msg = f"NaN values found in {name}"
            validation_result["errors"].append(error_msg)
            logger.error(error_msg)

        if torch.isinf(tensor).any():
            error_msg = f"Inf values found in {name}"
            validation_result["errors"].append(error_msg)
            logger.error(error_msg)

    # Overall validation status
    validation_result["valid"] = len(validation_result["errors"]) == 0

    logger.info(
        f"Validation complete: {validation_result['statistics']['total_weights']} weights, "
        f"{validation_result['statistics']['total_parameters']:,} parameters, "
        f"{validation_result['statistics']['memory_mb']:.2f} MB"
    )

    if validation_result["errors"]:
        logger.error(f"Validation failed with {len(validation_result['errors'])} errors")
    else:
        logger.info("All validation checks passed")

    return validation_result


def get_expected_weight_shapes(config: HiggsAudioConfig) -> Dict:
    """Get expected weight shapes based on model configuration.

    Args:
        config: HiggsAudioConfig instance

    Returns:
        Dict mapping weight names to expected shapes
    """
    expected_shapes = {}

    # Text model parameters
    hidden_size = config.text_config.hidden_size  # 3072
    vocab_size = config.vocab_size  # 128256
    num_layers = config.text_config.num_hidden_layers  # 28
    intermediate_size = config.text_config.intermediate_size  # 8192

    # Embeddings
    expected_shapes["transformer.vocab_embedding.weight"] = [vocab_size, hidden_size]

    # LM Heads
    expected_shapes["lm_head.weight"] = [vocab_size, hidden_size]
    expected_shapes["audio_lm_head.weight"] = [
        config.audio_num_codebooks * config.audio_codebook_size,
        hidden_size,
    ]

    # Layer norm
    expected_shapes["transformer.ln_f.weight"] = [hidden_size]

    # Transformer layers
    for i in range(num_layers):
        # Attention (QKV packed)
        expected_shapes[f"transformer.layers.{i}.attention.qkv.weight"] = [
            3 * hidden_size,
            hidden_size,
        ]
        expected_shapes[f"transformer.layers.{i}.attention.dense.weight"] = [
            hidden_size,
            hidden_size,
        ]

        # Layer norms
        expected_shapes[f"transformer.layers.{i}.input_layernorm.weight"] = [hidden_size]
        expected_shapes[f"transformer.layers.{i}.post_layernorm.weight"] = [hidden_size]

        # MLP/FFN (SwiGLU)
        expected_shapes[f"transformer.layers.{i}.mlp.fc.weight"] = [intermediate_size, hidden_size]
        expected_shapes[f"transformer.layers.{i}.mlp.gate.weight"] = [
            intermediate_size,
            hidden_size,
        ]
        expected_shapes[f"transformer.layers.{i}.mlp.proj.weight"] = [
            hidden_size,
            intermediate_size,
        ]

    # Audio feature projector
    audio_encoder_dim = config.audio_encoder_config.d_model  # 1280
    expected_shapes["audio_feature_projector.projector.0.weight"] = [hidden_size, audio_encoder_dim]
    expected_shapes["audio_feature_projector.projector.2.weight"] = [hidden_size, hidden_size]

    return expected_shapes


def generate_conversion_report(
    validation_result: Dict, output_path: str, source_dir: str, target_dir: str
):
    """Generate a comprehensive conversion report.

    Args:
        validation_result: Results from validate_converted_weights()
        output_path: Path to save the report
        source_dir: Source checkpoint directory
        target_dir: Target TensorRT-LLM directory
    """
    report = {
        "conversion_info": {
            "source_directory": source_dir,
            "target_directory": target_dir,
            "timestamp": str(torch.cuda.Event().query() if torch.cuda.is_available() else "N/A"),
            "validation_passed": validation_result["valid"],
        },
        "statistics": validation_result["statistics"],
        "validation_summary": {
            "total_errors": len(validation_result["errors"]),
            "total_warnings": len(validation_result["warnings"]),
            "missing_weights_count": len(validation_result["missing_weights"]),
            "unexpected_weights_count": len(validation_result["unexpected_weights"]),
        },
        "errors": validation_result["errors"],
        "warnings": validation_result["warnings"],
        "missing_weights": validation_result["missing_weights"],
        "unexpected_weights": validation_result["unexpected_weights"],
        "weight_shapes": {
            "expected": validation_result["expected_shapes"],
            "actual": validation_result["actual_shapes"],
        },
    }

    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Conversion report saved to: {output_path}")


def load_from_vllm_checkpoint(
    model_dir: str,
    config: HiggsAudioConfig,
    dtype: str = "float16",
    device: str = "cpu",
):
    """Load weights from a vLLM/HuggingFace checkpoint and convert them to the TensorRT-LLM format."""
    logger.info(f"Loading weights from vLLM checkpoint in {model_dir}...")

    trt_llm_weights = {}
    name_mapping = vllm_to_trt_llm_name_mapping(config)

    # Find the checkpoint file(s)
    model_dir = Path(model_dir)
    if (model_dir / "model.safetensors").exists():
        checkpoint_paths = [model_dir / "model.safetensors"]
    elif (model_dir / "pytorch_model.bin").exists():
        checkpoint_paths = [model_dir / "pytorch_model.bin"]
    else:
        # Sharded checkpoint
        if (model_dir / "model.safetensors.index.json").exists():
            index_file = model_dir / "model.safetensors.index.json"
            with open(index_file, "r") as f:
                index = json.load(f)
            shard_files = list(set(index["weight_map"].values()))
            checkpoint_paths = [model_dir / f for f in shard_files]
        else:
            raise FileNotFoundError("Could not find checkpoint files in the specified directory.")

    # Load weights from all shards
    for ckpt_path in checkpoint_paths:
        logger.info(f"Loading checkpoint shard: {ckpt_path}")
        with safe_open(ckpt_path, framework="pt", device=device) as f:
            for key in f.keys():
                if key in name_mapping:
                    trt_llm_name = name_mapping[key]
                    tensor = f.get_tensor(key)

                    # TODO: Add tensor transformations (e.g., for QKV packing)

                    trt_llm_weights[trt_llm_name] = tensor.to(dtype=getattr(torch, dtype))
                else:
                    logger.warning(f"Weight {key} not found in name mapping, skipping.")

    return trt_llm_weights


def load_audio_encoder(
    model_dir: str,
    config: HiggsAudioConfig,
    dtype: str = "float16",
    device: str = "cpu",
):
    """Load audio encoder weights."""
    logger.info("Loading audio encoder weights...")
    # Placeholder for audio encoder weight loading
    pass


def save_trtllm(
    weights: Dict[str, torch.Tensor],
    output_dir: str,
    config: HiggsAudioConfig,
):
    """Save weights in the TensorRT-LLM format."""
    logger.info(f"Saving weights to {output_dir}...")
    # Placeholder for saving weights
    pass


def vllm_to_trt_llm_name_mapping(config: HiggsAudioConfig, is_audio_encoder: bool = False):
    """Creates a name mapping from vLLM/HF format to TensorRT-LLM format."""
    mapping = {}

    # Common mapping for Llama-3.2-3B backbone
    for i in range(config.text_config.num_hidden_layers):
        # Attention layers
        mapping[f"model.layers.{i}.self_attn.q_proj.weight"] = (
            f"transformer.layers.{i}.attention.qkv.weight"
        )
        mapping[f"model.layers.{i}.self_attn.k_proj.weight"] = (
            f"transformer.layers.{i}.attention.qkv.weight"
        )
        mapping[f"model.layers.{i}.self_attn.v_proj.weight"] = (
            f"transformer.layers.{i}.attention.qkv.weight"
        )
        mapping[f"model.layers.{i}.self_attn.o_proj.weight"] = (
            f"transformer.layers.{i}.attention.dense.weight"
        )

        # Dual FFN layers
        mapping[f"model.layers.{i}.mlp.gate_proj.weight"] = f"transformer.layers.{i}.mlp.fc.weight"
        mapping[f"model.layers.{i}.mlp.up_proj.weight"] = f"transformer.layers.{i}.mlp.gate.weight"
        mapping[f"model.layers.{i}.mlp.down_proj.weight"] = (
            f"transformer.layers.{i}.mlp.proj.weight"
        )

        # Layer norms
        mapping[f"model.layers.{i}.input_layernorm.weight"] = (
            f"transformer.layers.{i}.input_layernorm.weight"
        )
        mapping[f"model.layers.{i}.post_attention_layernorm.weight"] = (
            f"transformer.layers.{i}.post_layernorm.weight"
        )

    # Embeddings and final layer norm
    mapping["model.embed_tokens.weight"] = "transformer.vocab_embedding.weight"
    mapping["model.norm.weight"] = "transformer.ln_f.weight"

    # LM Heads
    mapping["lm_head.weight"] = "lm_head.weight"
    mapping["audio_lm_head.weight"] = "audio_lm_head.weight"

    # Audio components
    if not is_audio_encoder:
        # Audio Feature Projector
        mapping["model.audio_feature_projector.0.weight"] = (
            "audio_feature_projector.projector.0.weight"
        )
        mapping["model.audio_feature_projector.2.weight"] = (
            "audio_feature_projector.projector.2.weight"
        )

    return mapping
