#!/usr/bin/env python3
"""Runtime smoke test for HiggsAudio TensorRT-LLM engine.

Steps:
- Ensure an engine exists (optionally build if missing)
- Load tokenizer
- Run a tiny generation (a few tokens) with ModelRunnerCpp

Run:
  python3 scripts/smoke_test_higgs_audio_runtime.py \
      --model_dir bosonai/higgs-audio-v2-generation-3B-base \
      --engine_dir ./higgs_audio_engine_test4 \
      --max_new_tokens 8
"""

import argparse
import os
import subprocess

import torch
from transformers import AutoTokenizer

from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.runtime import ModelRunnerCpp


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, default="bosonai/higgs-audio-v2-generation-3B-base")
    p.add_argument("--engine_dir", type=str, default="./higgs_audio_engine_test4")
    p.add_argument("--max_new_tokens", type=int, default=8)
    return p.parse_args()


def ensure_engine(model_dir: str, engine_dir: str):
    if os.path.isdir(engine_dir) and os.listdir(engine_dir):
        return
    print(f"[RuntimeSmoke] Engine not found at {engine_dir}, building...")
    cmd = [
        "python3",
        "build_higgs_audio_engine.py",
        "--model_dir",
        model_dir,
        "--output_dir",
        engine_dir,
    ]
    subprocess.check_call(cmd, cwd=os.getcwd())


def main():
    args = parse_args()
    ensure_engine(args.model_dir, args.engine_dir)

    print("[RuntimeSmoke] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True, use_fast=False
    )

    print("[RuntimeSmoke] Loading ModelRunnerCpp...")
    runner = ModelRunnerCpp.from_dir(
        engine_dir=args.engine_dir,
        use_gpu_direct_storage=True,
        cuda_graph_mode=True,
        kv_cache_free_gpu_memory_fraction=0.5,
    )

    config = HiggsAudioConfig()
    end_id = config.audio_eos_id  # audio eos
    pad_id = config.pad_token_id

    # Use the same prompt style as in the model's runner (no audio reference for simplicity)
    input_text = "Hello from a runtime smoke test."
    formatted_text = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an AI assistant designed to convert text into speech. Generate speech for the user's text.<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|><|audio_out_bos|>"
    )

    input_ids = tokenizer.encode(formatted_text, return_tensors="pt").squeeze(0)
    batch_input_ids = [input_ids]

    print("[RuntimeSmoke] Generating ...")
    outputs = runner.generate(
        batch_input_ids=batch_input_ids,
        max_new_tokens=args.max_new_tokens,
        beam_width=1,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        end_id=end_id,
        pad_id=pad_id,
    )

    gen = None
    if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
        gen = outputs[0]
    elif torch.is_tensor(outputs):
        gen = outputs

    if gen is not None:
        try:
            n = gen.shape[0] if hasattr(gen, "shape") else len(gen)
        except Exception:
            n = len(gen)
        tail_len = min(16, n)
        tail = gen[-tail_len:]
        print(f"[RuntimeSmoke] Generated {n} tokens (including prompt). Tail: {tail}")
    else:
        print("[RuntimeSmoke] No output returned from runner.generate")


if __name__ == "__main__":
    main()
