#!/usr/bin/env python
import sys
import os

print(f"Python executable: {sys.executable}")
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

try:
    import tensorrt_llm  # Replace with your actual module name
    print(f"Module imported from: {tensorrt_llm.__file__}")
except ImportError as e:
    print(f"Import failed: {e}")
    
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


def expect_shape(name, tensor, shape):
    if tensor is None:
        raise AssertionError(f"Missing tensor: {name}")
    if tuple(tensor.shape) != tuple(shape):
        raise AssertionError(f"Shape mismatch for {name}: got {tuple(tensor.shape)} != expected {tuple(shape)}")


def validate_shapes(config, ckpt_tensors):
    tp = config.mapping.tp_size
    tp_rank = config.mapping.tp_rank
    hidden = config.hidden_size
    interm = config.intermediate_size
    vocab = config.vocab_size
    n_heads = config.num_attention_heads
    head_size = config.head_size

    # Vocab embedding on first PP
    if config.mapping.is_first_pp_rank():
        t = ckpt_tensors.get('transformer.vocab_embedding.weight')
        if t is not None:
            expect_shape('transformer.vocab_embedding.weight', t, (vocab // tp, hidden))

    # Per-layer on this PP rank
    layers = config.mapping.pp_layers(config.num_hidden_layers)
    for idx, l in enumerate(layers):
        base = f'transformer.layers.{idx}'
        qkv_t = ckpt_tensors.get(f'{base}.attention.qkv.weight')
        if qkv_t is None:
            raise AssertionError(f"Missing tensor: {base}.attention.qkv.weight")
        rows = qkv_t.shape[0]
        # Infer kv heads from rows: rows_per_rank = (n_heads + 2*n_kv) * head_size / tp
        inferred_n_kv = (rows * tp // head_size - n_heads) // 2
        # Sanity: recompute expected rows
        qkv_rows = (n_heads + 2 * inferred_n_kv) * head_size // tp
        expect_shape(f'{base}.attention.qkv.weight', qkv_t, (qkv_rows, hidden))
        expect_shape(f'{base}.attention.dense.weight', ckpt_tensors.get(f'{base}.attention.dense.weight'), (hidden, hidden // tp))
        expect_shape(f'{base}.mlp.fc.weight', ckpt_tensors.get(f'{base}.mlp.fc.weight'), (interm // tp, hidden))
        expect_shape(f'{base}.mlp.gate.weight', ckpt_tensors.get(f'{base}.mlp.gate.weight'), (interm // tp, hidden))
        expect_shape(f'{base}.mlp.proj.weight', ckpt_tensors.get(f'{base}.mlp.proj.weight'), (hidden, interm // tp))
        expect_shape(f'{base}.input_layernorm.weight', ckpt_tensors.get(f'{base}.input_layernorm.weight'), (hidden,))
        expect_shape(f'{base}.post_layernorm.weight', ckpt_tensors.get(f'{base}.post_layernorm.weight'), (hidden,))

    # Final norm and lm_head on last PP
    if config.mapping.is_last_pp_rank():
        t = ckpt_tensors.get('transformer.ln_f.weight')
        if t is not None:
            expect_shape('transformer.ln_f.weight', t, (hidden,))
        t = ckpt_tensors.get('lm_head.weight')
        if t is not None:
            expect_shape('lm_head.weight', t, (vocab // tp, hidden))


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

    # Validate shapes of mapped tensors
    try:
        validate_shapes(config, ckpt_meta.get("tensors", {}))
        print("\n[ok] TRT-LLM tensor shapes validated for this rank.")
    except AssertionError as e:
        print(f"\n[fail] Shape validation error: {e}")

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
