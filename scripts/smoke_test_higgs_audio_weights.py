#!/usr/bin/env python3
"""Smoke test for HiggsAudio weights conversion.

- Loads the HF HiggsAudioModel
- Builds a TensorRT-LLM HiggsAudioForCausalLM config/model
- Converts HF weights to TLLM weight dict
- Verifies that every expected TLLM parameter has a matching weight with the same shape

Run:
  python3 scripts/smoke_test_higgs_audio_weights.py --model_dir bosonai/higgs-audio-v2-generation-3B-base
"""

import argparse
import sys

import torch

from boson_multimodal import HiggsAudioModel
from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.models.higgs_audio.convert import load_weights_from_hf_model
from tensorrt_llm.models import HiggsAudioForCausalLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_dir",
        type=str,
        default="bosonai/higgs-audio-v2-generation-3B-base",
        help="HF model repo or local path",
    )
    return p.parse_args()


def main():
    args = parse_args()
    print("[SmokeTest] Loading HF model:", args.model_dir)
    hf_model = HiggsAudioModel.from_pretrained(args.model_dir)

    print("[SmokeTest] Building TLLM config/model")
    config = HiggsAudioConfig(dtype="bfloat16")
    tllm_model = HiggsAudioForCausalLM(config)

    print("[SmokeTest] Converting weights from HF -> TLLM")
    weights = load_weights_from_hf_model(hf_model, config)

    print("[SmokeTest] Verifying parameter names and shapes...")
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

    print(f"[SmokeTest] Checked parameters: {checked}")
    if missing:
        print("[SmokeTest][ERROR] Missing weights for:")
        for n in missing[:20]:
            print("  -", n)
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
    if mismatched:
        print("[SmokeTest][ERROR] Shape mismatches:")
        for n, exp, got in mismatched[:20]:
            print(f"  - {n}: expected {exp}, got {got}")
        if len(mismatched) > 20:
            print(f"  ... and {len(mismatched) - 20} more")

    if missing or mismatched:
        print("[SmokeTest] FAILED")
        sys.exit(1)

    print("[SmokeTest] SUCCESS: All checked parameters have matching weights and shapes.")


if __name__ == "__main__":
    main()

