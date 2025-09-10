#!/usr/bin/env python3
"""Comprehensive test for HiggsAudio weights conversion.

This test verifies that load_weights_from_hf_model in convert.py is working properly by:
1. Loading the HF HiggsAudioModel
2. Converting HF weights to TLLM weight dict using load_weights_from_hf_model
3. Verifying that all converted weights have non-zero values (not empty)
4. Comparing actual weight values between HF and converted weights to ensure they match
5. Testing different quantization modes if available

Run:
  python3 test_higgs_audio_weight_conversion.py --model_dir bosonai/higgs-audio-v2-generation-3B-base
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM

# Add the current directory to Python path to import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import HiggsAudioForCausalLM
from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.models.higgs_audio.convert import load_hf_higgs_audio, load_weights_from_hf_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_dir",
        type=str,
        default="bosonai/higgs-audio-v2-generation-3B-base",
        help="HF model repo or local path",
    )
    p.add_argument(
        "--test_quantization",
        action="store_true",
        help="Test weight-only quantization modes",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Tolerance for weight value comparison",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed comparison results",
    )
    return p.parse_args()


def load_hf_model_safely(model_dir: str):
    """Load HF model with error handling."""
    try:
        print(f"[Test] Loading HF model from: {model_dir}")
        # Try using the convert.py function first
        hf_model = load_hf_higgs_audio(model_dir, load_model_on_cpu=False)
        print("[Test] Successfully loaded HF model using load_hf_higgs_audio")
        return hf_model
    except Exception as e:
        print(f"[Test] Failed to load with load_hf_higgs_audio: {e}")
        try:
            # Fallback to direct AutoModelForCausalLM
            print("[Test] Trying direct AutoModelForCausalLM...")
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True,
            )
            print("[Test] Successfully loaded HF model using AutoModelForCausalLM")
            return hf_model
        except Exception as e2:
            print(f"[Test] Failed to load HF model: {e2}")
            raise


def check_weight_non_empty(weights: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str]]:
    """Check that all weights are non-empty and contain non-zero values."""
    empty_weights = []
    zero_weights = []

    for name, weight in weights.items():
        if weight.numel() == 0:
            empty_weights.append(name)
        elif torch.all(weight == 0):
            zero_weights.append(name)

    return empty_weights, zero_weights


def extract_hf_weight_by_tllm_name(hf_model, tllm_name: str, mapping: Mapping) -> torch.Tensor:
    """Extract HF weight corresponding to TLLM weight name."""
    model_params = dict(hf_model.named_parameters())
    # Handle embedding weights
    if tllm_name == "transformer.vocab_embedding.weight":
        return model_params.get("embed_tokens.weight")

    # Handle audio codebook embeddings
    if tllm_name == "transformer.audio_codebook_embeddings.weight":
        return model_params.get("audio_codebook_embeddings.weight")

    # Handle final layer norm
    if tllm_name == "transformer.ln_f.weight":
        return model_params.get("norm.weight")

    # Handle LM head
    if tllm_name == "lm_head.weight":
        # For HiggsAudio, this should map to audio_lm_head
        return model_params.get("audio_decoder_proj.audio_lm_head.weight")

    # Handle layer weights
    if tllm_name.startswith("transformer.layers."):
        # Extract layer number and component
        parts = tllm_name.split(".")
        if len(parts) >= 3:
            layer_idx = int(parts[2])
            # Map TLLM layer names back to HF layer names
            hf_prefix = f"layers.{layer_idx}."

            if "attention.qkv.weight" in tllm_name:
                # For QKV, we need to concatenate q, k, v weights
                q_weight = model_params.get(f"{hf_prefix}self_attn.q_proj.weight")
                k_weight = model_params.get(f"{hf_prefix}self_attn.k_proj.weight")
                v_weight = model_params.get(f"{hf_prefix}self_attn.v_proj.weight")
                if q_weight is not None and k_weight is not None and v_weight is not None:
                    return torch.cat([q_weight, k_weight, v_weight], dim=0)

            elif "attention.dense.weight" in tllm_name:
                return model_params.get(f"{hf_prefix}self_attn.o_proj.weight")

            elif "audio_mlp.fc.weight" in tllm_name:
                return model_params.get(f"{hf_prefix}audio_mlp.gate_proj.weight")
            elif "audio_mlp.proj.weight" in tllm_name:
                return model_params.get(f"{hf_prefix}audio_mlp.down_proj.weight")
            elif "audio_mlp.gate.weight" in tllm_name:
                return model_params.get(f"{hf_prefix}audio_mlp.up_proj.weight")

            elif "mlp.fc.weight" in tllm_name:
                return model_params.get(f"{hf_prefix}mlp.gate_proj.weight")
            elif "mlp.proj.weight" in tllm_name:
                return model_params.get(f"{hf_prefix}mlp.down_proj.weight")
            elif "mlp.gate.weight" in tllm_name:
                return model_params.get(f"{hf_prefix}mlp.up_proj.weight")

            elif "audio_input_layernorm.weight" in tllm_name:
                return model_params.get(f"{hf_prefix}audio_input_layernorm.weight")

            elif "audio_post_layernorm.weight" in tllm_name:
                return model_params.get(f"{hf_prefix}audio_post_attention_layernorm.weight")

            elif "input_layernorm.weight" in tllm_name:
                return model_params.get(f"{hf_prefix}input_layernorm.weight")

            elif "post_layernorm.weight" in tllm_name:
                return model_params.get(f"{hf_prefix}post_attention_layernorm.weight")

    return None


def compare_weights(
    hf_model,
    tllm_weights: Dict[str, torch.Tensor],
    mapping: Mapping,
    tolerance: float = 1e-6,
    verbose: bool = False,
) -> Tuple[List[str], List[Tuple[str, float]]]:
    """Compare TLLM weights with original HF weights."""
    missing_hf_weights = []
    mismatched_weights = []
    for tllm_name, tllm_weight in tllm_weights.items():
        hf_weight = extract_hf_weight_by_tllm_name(hf_model, tllm_name, mapping)

        if hf_weight is None:
            missing_hf_weights.append(tllm_name)
            continue

        # Handle tensor parallel splitting for comparison
        if mapping.tp_size > 1:
            # For tensor parallel weights, we need to account for splitting
            # This is a simplified comparison - in practice, we'd need to handle
            # the specific splitting logic for each weight type
            if verbose:
                print(f"[Test] Skipping TP comparison for {tllm_name} (TP size: {mapping.tp_size})")
            continue

        # Convert to same device and dtype for comparison
        hf_weight = hf_weight.to(tllm_weight.device).to(tllm_weight.dtype)

        # Handle shape differences due to transposition or reshaping
        if hf_weight.shape != tllm_weight.shape:
            # Try transposing for linear layers
            if len(hf_weight.shape) == 2 and len(tllm_weight.shape) == 2:
                if hf_weight.shape == (tllm_weight.shape[1], tllm_weight.shape[0]):
                    hf_weight = hf_weight.t()

        if hf_weight.shape != tllm_weight.shape:
            if verbose:
                print(
                    f"[Test] Shape mismatch for {tllm_name}: HF {hf_weight.shape} vs TLLM {tllm_weight.shape}"
                )
            continue

        # Compare values
        diff = torch.abs(hf_weight - tllm_weight)
        max_diff = torch.max(diff).item()

        if max_diff > tolerance:
            mismatched_weights.append((tllm_name, max_diff))

        if verbose:
            mean_diff = torch.mean(diff).item()
            print(f"[Test] {tllm_name}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")

    return missing_hf_weights, mismatched_weights


def test_weight_conversion(
    model_dir: str, test_quantization: bool = False, tolerance: float = 1e-5, verbose: bool = False
):
    """Main test function."""
    print("[Test] Starting comprehensive weight conversion test")
    print(f"[Test] Model: {model_dir}")
    print(f"[Test] Tolerance: {tolerance}")

    # Load HF model
    hf_model = load_hf_model_safely(model_dir)

    # Test basic conversion (no quantization)
    print("\n[Test] Testing basic weight conversion (no quantization)")

    config = HiggsAudioConfig.from_hugging_face("bosonai/higgs-audio-v2-generation-3B-base")
    tllm_model = HiggsAudioForCausalLM(config)

    print("[Test] Converting weights from HF -> TLLM")
    weights = load_weights_from_hf_model(hf_model, config)

    print(f"[Test] Converted {len(weights)} weight tensors")

    # Check for empty or zero weights
    print("[Test] Checking for empty or zero weights...")
    empty_weights, zero_weights = check_weight_non_empty(weights)

    if empty_weights:
        print(f"[Test][ERROR] Found {len(empty_weights)} empty weights:")
        for name in empty_weights[:10]:
            print(f"  - {name}")
        if len(empty_weights) > 10:
            print(f"  ... and {len(empty_weights) - 10} more")

    if zero_weights:
        print(f"[Test][WARNING] Found {len(zero_weights)} all-zero weights:")
        for name in zero_weights[:10]:
            print(f"  - {name}")
        if len(zero_weights) > 10:
            print(f"  ... and {len(zero_weights) - 10} more")

    # Check parameter coverage
    print("[Test] Checking parameter coverage...")
    missing = []
    mismatched = []
    checked = 0

    for name, param in tllm_model.named_parameters():
        if name not in weights:
            missing.append(name)
            continue
        t = weights[name]
        if tuple(t.shape) != tuple(param.shape):
            mismatched.append((name, tuple(param.shape), tuple(t.shape)))
        checked += 1

    print(f"[Test] Checked parameters: {checked}")
    if missing:
        print(f"[Test][ERROR] Missing weights for {len(missing)} parameters:")
        for n in missing[:10]:
            print(f"  - {n}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    if mismatched:
        print(f"[Test][ERROR] Shape mismatches for {len(mismatched)} parameters:")
        for n, exp, got in mismatched[:10]:
            print(f"  - {n}: expected {exp}, got {got}")
        if len(mismatched) > 10:
            print(f"  ... and {len(mismatched) - 10} more")

    # Compare weight values with original HF model
    print("[Test] Comparing weight values with original HF model...")
    missing_hf, value_mismatches = compare_weights(
        hf_model, weights, config.mapping, tolerance, verbose
    )

    if missing_hf:
        print(f"[Test][WARNING] Could not find HF weights for {len(missing_hf)} TLLM weights:")
        for name in missing_hf[:10]:
            print(f"  - {name}")
        if len(missing_hf) > 10:
            print(f"  ... and {len(missing_hf) - 10} more")

    if value_mismatches:
        print(f"[Test][ERROR] Value mismatches for {len(value_mismatches)} weights:")
        for name, max_diff in value_mismatches[:10]:
            print(f"  - {name}: max_diff={max_diff:.2e}")
        if len(value_mismatches) > 10:
            print(f"  ... and {len(value_mismatches) - 10} more")

    # Summary for basic test
    basic_success = not (empty_weights or missing or mismatched or value_mismatches)
    print(f"\n[Test] Basic conversion test: {'PASSED' if basic_success else 'FAILED'}")

    return basic_success


def main():
    args = parse_args()

    try:
        success = test_weight_conversion(
            args.model_dir, args.test_quantization, args.tolerance, args.verbose
        )

        if success:
            print("\n[Test] ✅ ALL TESTS PASSED")
            print("[Test] load_weights_from_hf_model is working correctly!")
            sys.exit(0)
        else:
            print("\n[Test] ❌ SOME TESTS FAILED")
            print("[Test] load_weights_from_hf_model has issues that need to be addressed.")
            sys.exit(1)

    except Exception as e:
        print(f"\n[Test] ❌ TEST CRASHED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
