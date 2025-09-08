#!/usr/bin/env python3
"""Debug model weights to see what's missing."""

import torch
from transformers import AutoModelForCausalLM
from collections import defaultdict


def analyze_hf_model_weights():
    """Analyze the HuggingFace model weights to see what we might be missing."""
    print("Loading HuggingFace model...")
    
    model = AutoModelForCausalLM.from_pretrained(
        "bosonai/higgs-audio-v2-generation-3B-base",
        device_map="cpu",  # Load on CPU to avoid GPU memory issues
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    print(f"Model type: {type(model)}")
    print(f"Model config: {model.config}")
    
    # Analyze model parameters
    total_params = 0
    param_groups = defaultdict(list)
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        # Group parameters by component
        if "embed" in name:
            param_groups["embeddings"].append((name, param.shape, param.numel()))
        elif "layers" in name or "transformer" in name:
            param_groups["transformer_layers"].append((name, param.shape, param.numel()))
        elif "audio" in name:
            param_groups["audio_components"].append((name, param.shape, param.numel()))
        elif "lm_head" in name or "decoder" in name:
            param_groups["output_heads"].append((name, param.shape, param.numel()))
        else:
            param_groups["other"].append((name, param.shape, param.numel()))
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Total size (BF16): {total_params * 2 / (1024**3):.2f} GB")
    
    for group, params in param_groups.items():
        group_size = sum(p[2] for p in params)
        print(f"\n{group.upper()} ({len(params)} parameters, {group_size:,} total):")
        for name, shape, size in params[:10]:  # Show first 10
            print(f"  {name}: {shape} ({size:,})")
        if len(params) > 10:
            print(f"  ... and {len(params) - 10} more")
    
    # Check specific audio components
    print("\nAUDIO-SPECIFIC COMPONENTS:")
    audio_params = [name for name, _ in model.named_parameters() if "audio" in name]
    for name in audio_params:
        param = dict(model.named_parameters())[name]
        print(f"  {name}: {param.shape} ({param.numel():,})")
    
    # Check if there are any large components we might be missing
    print("\nLARGEST PARAMETERS:")
    all_params = [(name, param.shape, param.numel()) for name, param in model.named_parameters()]
    all_params.sort(key=lambda x: x[2], reverse=True)
    for name, shape, size in all_params[:20]:
        print(f"  {name}: {shape} ({size:,})")
    
    return model


def compare_with_convert_script():
    """Compare what the convert script loads vs what's in the model."""
    print("\n" + "="*60)
    print("COMPARING WITH CONVERT SCRIPT")
    print("="*60)
    
    # These are the components the convert script loads
    convert_script_components = [
        "transformer.vocab_embedding.weight",
        "audio_codebook_embeddings.weight", 
        "lm_head.weight",  # from audio_decoder_proj.audio_lm_head.weight
        "transformer.layers.*.attention.*",
        "transformer.layers.*.mlp.*",
        "transformer.layers.*.audio_mlp.*",
        "transformer.layers.*.input_layernorm.*",
        "transformer.layers.*.post_attention_layernorm.*",
        "transformer.ln_f.*",
    ]
    
    print("Components loaded by convert script:")
    for comp in convert_script_components:
        print(f"  ✓ {comp}")
    
    print("\nPotential missing components:")
    print("  - Audio encoder/decoder networks")
    print("  - Audio feature projection layers")
    print("  - Cross-attention between text and audio")
    print("  - Audio-specific normalization layers")
    print("  - Position embeddings for audio")


if __name__ == "__main__":
    try:
        model = analyze_hf_model_weights()
        compare_with_convert_script()
        print("\n✅ Model weight analysis completed!")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
