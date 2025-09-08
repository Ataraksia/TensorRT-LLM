#!/usr/bin/env python3
"""Debug engine weights to see what was actually loaded."""

import os
import safetensors
import torch
from collections import defaultdict


def analyze_engine_weights():
    """Analyze the weights in the TensorRT engine to see what was loaded."""
    print("Analyzing TensorRT engine weights...")
    
    engine_dir = "/home/me/TTS/TensorRT-LLM/higgs_audio_engine"
    rank0_weights_path = os.path.join(engine_dir, "rank0.safetensors")
    
    if not os.path.exists(rank0_weights_path):
        print(f"❌ Weights file not found: {rank0_weights_path}")
        return
    
    print(f"Loading weights from: {rank0_weights_path}")
    
    # Load the weights
    weights = safetensors.torch.load_file(rank0_weights_path)
    
    print(f"Total weight tensors: {len(weights)}")
    
    # Analyze weights by component
    total_params = 0
    param_groups = defaultdict(list)
    
    for name, tensor in weights.items():
        total_params += tensor.numel()
        
        # Group parameters by component
        if "vocab_embedding" in name:
            param_groups["embeddings"].append((name, tensor.shape, tensor.numel()))
        elif "layers" in name:
            if "attention" in name:
                param_groups["attention"].append((name, tensor.shape, tensor.numel()))
            elif "mlp" in name:
                param_groups["mlp"].append((name, tensor.shape, tensor.numel()))
            elif "layernorm" in name or "ln_" in name:
                param_groups["normalization"].append((name, tensor.shape, tensor.numel()))
            else:
                param_groups["other_layers"].append((name, tensor.shape, tensor.numel()))
        elif "lm_head" in name:
            param_groups["output_head"].append((name, tensor.shape, tensor.numel()))
        elif "audio" in name:
            param_groups["audio_components"].append((name, tensor.shape, tensor.numel()))
        else:
            param_groups["other"].append((name, tensor.shape, tensor.numel()))
    
    print(f"\nTotal parameters in engine: {total_params:,}")
    print(f"Total size (BF16): {total_params * 2 / (1024**3):.2f} GB")
    
    for group, params in param_groups.items():
        group_size = sum(p[2] for p in params)
        print(f"\n{group.upper()} ({len(params)} tensors, {group_size:,} params):")
        for name, shape, size in params:
            print(f"  {name}: {shape} ({size:,})")
    
    # Check for specific patterns
    print("\nWEIGHT ANALYSIS:")
    
    # Count layers
    layer_count = len([name for name in weights.keys() if "layers." in name and "attention.qkv.weight" in name])
    print(f"Number of transformer layers: {layer_count}")
    
    # Check MLP patterns
    standard_mlp_count = len([name for name in weights.keys() if "mlp.gate_up_proj.weight" in name])
    audio_mlp_count = len([name for name in weights.keys() if "audio_mlp" in name])
    print(f"Standard MLP layers: {standard_mlp_count}")
    print(f"Audio MLP layers: {audio_mlp_count}")
    
    # Check largest weights
    print("\nLARGEST WEIGHTS:")
    all_weights = [(name, tensor.shape, tensor.numel()) for name, tensor in weights.items()]
    all_weights.sort(key=lambda x: x[2], reverse=True)
    for name, shape, size in all_weights[:15]:
        print(f"  {name}: {shape} ({size:,})")
    
    # Check if we have the expected HiggsAudio components
    print("\nHIGGSAUDIO COMPONENT CHECK:")
    expected_components = [
        "transformer.vocab_embedding.weight",
        "lm_head.weight",
        "transformer.ln_f.weight",
    ]
    
    for comp in expected_components:
        if comp in weights:
            print(f"  ✅ {comp}: {weights[comp].shape}")
        else:
            print(f"  ❌ {comp}: MISSING")
    
    # Check for audio-specific components
    audio_components = [name for name in weights.keys() if "audio" in name]
    print(f"\nAudio components found: {len(audio_components)}")
    for comp in audio_components:
        print(f"  {comp}: {weights[comp].shape}")
    
    return weights


def compare_expected_vs_actual():
    """Compare expected model size vs actual engine size."""
    print("\n" + "="*60)
    print("SIZE COMPARISON")
    print("="*60)
    
    # Expected sizes for a 3B parameter model
    expected_total_params = 3_000_000_000  # 3B parameters
    expected_size_gb = expected_total_params * 2 / (1024**3)  # BF16
    
    # Actual engine size
    engine_dir = "/home/me/TTS/TensorRT-LLM/higgs_audio_engine"
    engine_file = os.path.join(engine_dir, "rank0.engine")
    weights_file = os.path.join(engine_dir, "rank0.safetensors")
    
    engine_size_gb = 0
    weights_size_gb = 0
    
    if os.path.exists(engine_file):
        engine_size_gb = os.path.getsize(engine_file) / (1024**3)
    
    if os.path.exists(weights_file):
        weights_size_gb = os.path.getsize(weights_file) / (1024**3)
    
    total_engine_size = engine_size_gb + weights_size_gb
    
    print(f"Expected model size: ~{expected_size_gb:.2f} GB")
    print(f"Actual engine size: {engine_size_gb:.2f} GB")
    print(f"Actual weights size: {weights_size_gb:.2f} GB")
    print(f"Total actual size: {total_engine_size:.2f} GB")
    print(f"Size ratio: {total_engine_size / expected_size_gb:.2f}x")
    
    if total_engine_size < expected_size_gb * 0.7:
        print("⚠️  Engine is significantly smaller than expected - likely missing weights!")
    else:
        print("✅ Engine size looks reasonable")


if __name__ == "__main__":
    try:
        weights = analyze_engine_weights()
        compare_expected_vs_actual()
        print("\n✅ Engine weight analysis completed!")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
