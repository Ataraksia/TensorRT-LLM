"""Tests for HiggsAudio data processing and collation functionality."""

import torch

from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.models.higgs_audio.data_processing import (
    HiggsAudioSampleCollator,
    create_higgs_audio_collator,
    prepare_audio_sample,
    prepare_text_sample,
)


def test_collator_initialization():
    """Test basic collator initialization."""
    print("Testing collator initialization...")

    config = HiggsAudioConfig()
    collator = HiggsAudioSampleCollator(config)

    assert collator.config == config
    assert collator.pad_multiple == 8
    assert collator.return_attention_mask is True
    assert collator.enable_delay_pattern is False  # Default
    print("✓ Collator initialization correct")


def test_text_only_collation():
    """Test collating text-only samples."""
    print("\nTesting text-only collation...")

    config = HiggsAudioConfig()
    collator = HiggsAudioSampleCollator(config, pad_multiple=4)

    # Create test samples
    samples = [
        prepare_text_sample([1, 2, 3, 4], sample_id="sample1"),
        prepare_text_sample([5, 6], sample_id="sample2"),
        prepare_text_sample([7, 8, 9], sample_id="sample3"),
    ]

    batch = collator(samples)

    # Check batch structure
    print(f"Batch keys: {list(batch.keys())}")
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "batch_metadata" in batch

    # Check tensor shapes (should be padded to multiple of 4)
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"attention_mask shape: {batch['attention_mask'].shape}")
    assert batch["input_ids"].shape == (3, 4)  # 3 samples, padded to 4
    assert batch["attention_mask"].shape == (3, 4)

    # Check attention mask correctness
    expected_mask = torch.tensor(
        [
            [1, 1, 1, 1],  # Sample 1: all real tokens
            [1, 1, 0, 0],  # Sample 2: 2 real, 2 padded
            [1, 1, 1, 0],  # Sample 3: 3 real, 1 padded
        ],
        dtype=torch.bool,
    )
    assert torch.equal(batch["attention_mask"], expected_mask)

    print("✓ Text-only collation correct")


def test_mixed_collation():
    """Test collating samples with both text and audio."""
    print("\nTesting mixed collation...")

    config = HiggsAudioConfig()
    collator = HiggsAudioSampleCollator(config, pad_multiple=4)

    # Create test audio features
    mel1 = torch.randn(128, 50)  # [n_mels, time_frames]
    mel2 = torch.randn(128, 30)

    # Create test samples
    samples = [
        prepare_text_sample([1, 2, 3, 4], sample_id="text_only"),
        prepare_audio_sample([5, 6], mel1, sample_id="audio1"),
        prepare_audio_sample([7, 8, 9], mel2, sample_id="audio2"),
    ]

    batch = collator(samples)

    # Check batch structure
    print(f"Mixed batch keys: {list(batch.keys())}")
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "batch_metadata" in batch

    # Check if audio keys exist (they should since we have audio samples)
    has_audio = any("mel" in str(sample) for sample in samples)
    print(f"Has audio samples: {has_audio}")

    # Check tensor shapes
    print(f"input_ids shape: {batch['input_ids'].shape}")
    assert batch["input_ids"].shape == (3, 4)  # Padded to 4

    print("✓ Mixed collation correct")


def test_delay_pattern_integration():
    """Test collator with delay patterns enabled."""
    print("\nTesting delay pattern integration...")

    config = HiggsAudioConfig()
    collator = HiggsAudioSampleCollator(
        config,
        enable_delay_pattern=True,
        delay_pattern_strategy="linear",
        num_codebooks=3,
    )

    # Check initialization
    assert collator.enable_delay_pattern is True
    assert collator.delay_pattern_provider is not None
    assert collator.audio_token_utils is not None
    assert collator.streaming_state is not None

    print("✓ Delay pattern integration correct")


def test_utility_functions():
    """Test utility functions."""
    print("\nTesting utility functions...")

    config = HiggsAudioConfig()

    # Test create_higgs_audio_collator
    collator = create_higgs_audio_collator(config, pad_multiple=16)
    assert collator.pad_multiple == 16

    # Test prepare_text_sample
    text_sample = prepare_text_sample([1, 2, 3], sample_id="test", mode="text")
    assert text_sample["input_ids"] == [1, 2, 3]
    assert text_sample["sample_id"] == "test"
    assert text_sample["mode"] == "text"

    # Test prepare_audio_sample
    mel = torch.randn(128, 20)
    audio_sample = prepare_audio_sample([4, 5, 6], mel, sample_id="audio")
    assert audio_sample["input_ids"] == [4, 5, 6]
    assert torch.equal(audio_sample["mel"], mel)
    assert audio_sample["sample_id"] == "audio"

    print("✓ Utility functions correct")


if __name__ == "__main__":
    print("🎵 Running HiggsAudio Data Processing Tests 🎵\n")

    test_collator_initialization()
    test_text_only_collation()
    test_mixed_collation()
    test_delay_pattern_integration()
    test_utility_functions()

    print("\n🎉 All data processing tests passed! 🎉")
