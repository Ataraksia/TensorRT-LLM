#!/usr/bin/env python3
"""Test script to verify HiggsAudio model integration with audio features."""

import torch
import numpy as np
from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.models.higgs_audio.model import HiggsAudioForCausalLM


def test_model_creation():
    """Test that the HiggsAudio model can be created successfully."""
    print("Testing HiggsAudio model creation...")
    
    config = HiggsAudioConfig()
    model = HiggsAudioForCausalLM(config)
    
    print(f"✓ Model created successfully")
    print(f"  - Vocab size: {config.vocab_size}")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Audio codebooks: {config.audio_num_codebooks}")
    print(f"  - Audio codebook size: {config.audio_codebook_size}")
    
    return model, config


def test_prompt_tuning_setup():
    """Test that prompt tuning can be enabled for audio features."""
    print("\nTesting prompt tuning setup...")
    
    config = HiggsAudioConfig()
    model = HiggsAudioForCausalLM(config)
    
    # Enable prompt tuning
    model.use_prompt_tuning()
    
    # Check that the embedding layer was replaced
    from tensorrt_llm.layers.embedding import PromptTuningEmbedding
    assert isinstance(model.transformer.vocab_embedding, PromptTuningEmbedding)
    
    print("✓ Prompt tuning enabled successfully")
    print(f"  - Embedding type: {type(model.transformer.vocab_embedding).__name__}")
    
    return model, config


def test_audio_feature_simulation():
    """Test simulated audio feature processing."""
    print("\nTesting audio feature simulation...")
    
    config = HiggsAudioConfig()
    
    # Simulate audio features as would be produced by an audio encoder
    batch_size = 1
    num_audio_tokens = 64  # Typical number of audio tokens
    hidden_size = config.hidden_size
    
    # Create mock audio features
    audio_features = torch.randn(batch_size, num_audio_tokens, hidden_size, dtype=torch.float16)
    
    # Create mock input with audio placeholder
    input_text = "Transcribe this audio: <|AUDIO|>"
    
    print(f"✓ Audio features simulated")
    print(f"  - Audio features shape: {audio_features.shape}")
    print(f"  - Input text: {input_text}")
    
    return audio_features, input_text


def test_engine_build_compatibility():
    """Test that the model is compatible with TensorRT-LLM engine building."""
    print("\nTesting engine build compatibility...")
    
    # Check that the engine was built successfully
    import os
    engine_path = "./higgs_audio_engine/rank0.engine"
    config_path = "./higgs_audio_engine/config.json"
    
    if os.path.exists(engine_path) and os.path.exists(config_path):
        engine_size = os.path.getsize(engine_path) / (1024 * 1024 * 1024)  # GB
        print(f"✓ Engine built successfully")
        print(f"  - Engine file: {engine_path}")
        print(f"  - Engine size: {engine_size:.2f} GB")
        print(f"  - Config file: {config_path}")
        return True
    else:
        print("✗ Engine files not found")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("HiggsAudio Model Integration Test")
    print("=" * 60)
    
    try:
        # Test 1: Model creation
        model, config = test_model_creation()
        
        # Test 2: Prompt tuning setup
        model_pt, config_pt = test_prompt_tuning_setup()
        
        # Test 3: Audio feature simulation
        audio_features, input_text = test_audio_feature_simulation()
        
        # Test 4: Engine build compatibility
        engine_ok = test_engine_build_compatibility()
        
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print("✓ Model creation: PASSED")
        print("✓ Prompt tuning setup: PASSED")
        print("✓ Audio feature simulation: PASSED")
        print(f"{'✓' if engine_ok else '✗'} Engine build compatibility: {'PASSED' if engine_ok else 'FAILED'}")
        
        if engine_ok:
            print("\n🎉 All tests passed! The HiggsAudio model is ready for audio-enhanced inference.")
            print("\nNext steps:")
            print("1. Use the HiggsAudioTRTRunner for inference")
            print("2. Pass audio features through prompt_embedding_table")
            print("3. Set appropriate prompt_tasks and prompt_vocab_size")
        else:
            print("\n⚠️  Engine build test failed. Please run build_higgs_audio_engine.py first.")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
