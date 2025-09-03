#!/usr/bin/env python3
"""Test script for HiggsAudio model basic functionality."""

import sys
from pathlib import Path

import torch

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
    from tensorrt_llm.models.higgs_audio.higgs_audio import GenerationMode, HiggsAudioModel

    print("✅ Successfully imported HiggsAudio modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_config_creation():
    """Test configuration creation."""
    try:
        config = HiggsAudioConfig()
        print("✅ Config creation successful")
        print(f"   - Hidden size: {config.text_config['hidden_size']}")
        print(f"   - Num layers: {config.text_config['num_hidden_layers']}")
        print(f"   - Audio codebooks: {config.audio_num_codebooks}")
        return config
    except Exception as e:
        print(f"❌ Config creation failed: {e}")
        return None


def test_model_creation(config):
    """Test model creation."""
    try:
        model = HiggsAudioModel(config, dtype="float16")
        print("✅ Model creation successful")
        print(f"   - Number of parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"   - Number of layers: {len(model.layers)}")
        return model
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return None


def test_forward_pass(model):
    """Test a simple forward pass."""
    try:
        batch_size = 2
        seq_len = 32
        vocab_size = model.vocab_size

        # Create dummy input
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

        # Forward pass
        with torch.no_grad():
            text_logits, audio_logits = model(input_ids)

        print("✅ Forward pass successful")
        print(f"   - Text logits shape: {text_logits.shape}")
        print(
            f"   - Audio logits shape: {audio_logits.shape if audio_logits is not None else 'None'}"
        )
        return True
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_generation_modes(model):
    """Test switching generation modes."""
    try:
        # Test mode switching
        model.set_generation_mode(GenerationMode.TEXT)
        assert model.generation_mode == GenerationMode.TEXT

        model.set_generation_mode(GenerationMode.AUDIO_INIT)
        assert model.generation_mode == GenerationMode.AUDIO_INIT

        model.set_generation_mode(GenerationMode.AUDIO_IN_PROGRESS)
        assert model.generation_mode == GenerationMode.AUDIO_IN_PROGRESS

        print("✅ Generation mode switching successful")
        return True
    except Exception as e:
        print(f"❌ Generation mode switching failed: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 Testing HiggsAudio Model Implementation")
    print("=" * 50)

    # Test 1: Configuration
    config = test_config_creation()
    if config is None:
        return 1

    # Test 2: Model creation
    model = test_model_creation(config)
    if model is None:
        return 1

    # Test 3: Generation modes
    if not test_generation_modes(model):
        return 1

    # Test 4: Forward pass
    if not test_forward_pass(model):
        return 1

    print("\n🎉 All tests passed!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
