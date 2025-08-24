#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from tensorrt_llm.models.higgs_audio.convert import build_config_from_hf, load_weights_from_hf_model


def find_index_json(model_dir: Path):
    # Prefer safetensors index, fallback to PyTorch index
    cand = list(model_dir.glob("*.safetensors.index.json"))
    if cand:
        return cand[0]
    cand = list(model_dir.glob("pytorch_model.bin.index.json"))
    if cand:
        return cand[0]
    return None


def main():
    ap = argparse.ArgumentParser(description="Higgs-Audio HF->TRT-LLM conversion dry-run checker")
    ap.add_argument("--hf_model_dir", required=True, type=str, help="Path to local HF model directory")
    ap.add_argument("--trust_remote_code", action="store_true", help="Pass through to HF AutoConfig")
    args = ap.parse_args()

    model_dir = Path(args.hf_model_dir)
    assert model_dir.exists(), f"Model dir not found: {model_dir}"

    # Build TRT-LLM config and retrieve expected key template
    config = build_config_from_hf(str(model_dir), trust_remote_code=args.trust_remote_code)
    ckpt_meta = load_weights_from_hf_model(str(model_dir), config)
    expected = ckpt_meta.get("expected_keys", [])

    print("=== TRT-LLM Expected Key Template (backbone) ===")
    print(f"count: {len(expected)}")
    for k in (expected[:10] + (["..."] if len(expected) > 10 else [])):
        print(k)

    # Try to list HF weight parameter names via index json
    index_path = find_index_json(model_dir)
    if index_path is None:
        print("\n[info] No weight index json found (*.safetensors.index.json or pytorch_model.bin.index.json).\n"
              "      Skipping HF key listing to avoid loading full weights.")
        return

    try:
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        hf_keys = sorted(weight_map.keys())
        print("\n=== HF Parameter Keys from index ===")
        print(f"count: {len(hf_keys)}  (from {os.path.basename(index_path)})")
        for k in (hf_keys[:15] + (["..."] if len(hf_keys) > 15 else [])):
            print(k)
    except Exception as e:
        print(f"[warn] Failed to read index json: {e}")


if __name__ == "__main__":
    main()
