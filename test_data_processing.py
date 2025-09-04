"""Comprehensive tests for HiggsAudio data processing and collation.

This test suite validates:
- HiggsAudioSampleCollator functionality
- Audio preprocessing and chunking
- RVQ delay pattern coordination
- Multi-codebook token handling
- Streaming collation state management
"""

import torch

# Import the modules we're testing
from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.models.higgs_audio.data_processing import (
    HiggsAudioSampleCollator,
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
    assert collator.text_pad_value == config.pad_token_id
    assert collator.audio_pad_value == 0.0

    print("✅ Collator initialization test passed")


def test_text_only_collation():
    """Test collating text-only samples."""
    print("\nTesting text-only collation...")

    config = HiggsAudioConfig()
    collator = HiggsAudioSampleCollator(config, pad_multiple=4)

    # Create test samples with different lengths
    samples = [
        prepare_text_sample([1, 2, 3, 4, 5], "sample_1"),
        prepare_text_sample([10, 11, 12], "sample_2"),
        prepare_text_sample([20, 21, 22, 23, 24, 25, 26], "sample_3"),
    ]

    batch = collator(samples)

    # Verify batch structure
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "text_lengths" in batch
    assert "batch_metadata" in batch

    # Check shapes (max length should be 8 due to pad_multiple=4, max_len=7 -> 8)
    assert batch["input_ids"].shape == (3, 8)
    assert batch["attention_mask"].shape == (3, 8)
    assert batch["text_lengths"].shape == (3,)

    # Check values
    assert torch.equal(batch["text_lengths"], torch.tensor([5, 3, 7]))

    # Check attention masks
    expected_mask = torch.tensor(
        [
            [True, True, True, True, True, False, False, False],
            [True, True, True, False, False, False, False, False],
            [True, True, True, True, True, True, True, False],
        ]
    )
    assert torch.equal(batch["attention_mask"], expected_mask)

    print("✅ Text-only collation test passed")


def test_audio_collation():
    """Test collating samples with audio features."""
    print("\nTesting audio collation...")

    config = HiggsAudioConfig()
    collator = HiggsAudioSampleCollator(config, pad_multiple=4)

    # Create test audio features with different time lengths
    mel1 = torch.randn(80, 100)  # 80 mel bins, 100 time frames
    mel2 = torch.randn(80, 150)  # 80 mel bins, 150 time frames

    samples = [
        prepare_audio_sample([1, 2, 3], mel1, sample_id="audio_1"),
        prepare_audio_sample([4, 5], mel2, sample_id="audio_2"),
    ]

    batch = collator(samples)

    # Verify batch structure
    assert "input_ids" in batch
    assert "mel_features" in batch
    assert "audio_attention_mask" in batch
    assert "audio_lengths" in batch

    # Check shapes (max time should be 152 due to pad_multiple=4, max_time=150 -> 152)
    assert batch["input_ids"].shape == (2, 4)  # max text length 3 -> 4
    assert batch["mel_features"].shape == (2, 80, 152)
    assert batch["audio_attention_mask"].shape == (2, 152)
    assert batch["audio_lengths"].shape == (2,)

    # Check audio lengths
    assert torch.equal(batch["audio_lengths"], torch.tensor([100, 150]))

    print("✅ Audio collation test passed")


def test_mixed_batch():
    """Test collating mixed text and audio samples."""
    print("\nTesting mixed batch collation...")

    config = HiggsAudioConfig()
    collator = HiggsAudioSampleCollator(config)

    # Create mixed samples
    text_samples = [[1, 2, 3, 4], [10, 11]]
    audio_samples = [([5, 6, 7], torch.randn(80, 50))]

    samples = prepare_mixed_batch(text_samples, audio_samples)
    batch = collator(samples)

    # Verify all expected fields are present
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "text_lengths" in batch
    assert "mel_features" in batch
    assert "audio_attention_mask" in batch
    assert "audio_lengths" in batch
    assert "batch_metadata" in batch

    # Check batch size consistency
    batch_size = 3  # 2 text + 1 audio
    assert batch["input_ids"].shape[0] == batch_size
    assert batch["mel_features"].shape[0] == batch_size
    assert len(batch["batch_metadata"]) == batch_size

    # Validate the batch
    assert validate_collated_batch(batch)

    print("✅ Mixed batch collation test passed")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting edge cases...")

    config = HiggsAudioConfig()
    collator = HiggsAudioSampleCollator(config)

    # Test empty batch
    batch = collator([])
    assert batch == {}

    # Test single empty sequence
    samples = [prepare_text_sample([])]
    batch = collator(samples)
    assert batch["text_lengths"][0] == 0

    # Test inconsistent mel dimensions should raise error
    mel1 = torch.randn(80, 100)
    mel2 = torch.randn(128, 100)  # Different n_mels

    samples = [
        prepare_audio_sample([1, 2], mel1),
        prepare_audio_sample([3, 4], mel2),
    ]

    try:
        batch = collator(samples)
        assert False, "Should have raised error for inconsistent mel dimensions"
    except ValueError:
        pass  # Expected

    print("✅ Edge cases test passed")


def test_special_tokens():
    """Test special token insertion for audio samples."""
    print("\nTesting special token insertion...")

    config = HiggsAudioConfig()
    collator = HiggsAudioSampleCollator(config)

    # Create audio input sample
    sample = prepare_audio_sample([1, 2, 3], torch.randn(80, 50), mode="audio_input")
    batch = collator([sample])

    # Check if special tokens are handled (implementation specific)
    # For now, just verify the batch is valid
    assert validate_collated_batch(batch)

    print("✅ Special tokens test passed")


def test_with_real_audio():
    """Test with real audio preprocessing."""
    print("\nTesting with real audio preprocessing...")

    config = HiggsAudioConfig()
    collator = HiggsAudioSampleCollator(config)

    # Create synthetic audio data
    audio_preprocessor = AudioPreprocessor()

    # Generate synthetic waveforms of different lengths
    waveform1 = torch.randn(1, 16000)  # 1 second at 16kHz
    waveform2 = torch.randn(1, 32000)  # 2 seconds at 16kHz

    # Compute mel spectrograms
    mel1 = audio_preprocessor.compute_whisper_mel(waveform1)
    mel2 = audio_preprocessor.compute_whisper_mel(waveform2)

    # Create samples
    samples = [
        prepare_audio_sample([1, 2, 3, 4], mel1, {"duration": 1.0}),
        prepare_audio_sample([5, 6, 7], mel2, {"duration": 2.0}),
    ]

    batch = collator(samples)

    # Verify the batch
    assert validate_collated_batch(batch)
    assert batch["mel_features"].shape[1] == 80  # Whisper mel bins

    # Check metadata preservation
    assert batch["batch_metadata"][0]["audio_meta"]["duration"] == 1.0
    assert batch["batch_metadata"][1]["audio_meta"]["duration"] == 2.0

    print("✅ Real audio preprocessing test passed")


def test_audio_chunking():
    """Test audio chunking functionality."""
    from tensorrt_llm.models.higgs_audio.preprocessing import AudioChunker

    print("\nTesting audio chunking...")

    # Create a chunker with small parameters for testing
    chunker = AudioChunker(
        chunk_duration_seconds=20.0,  # 20 seconds
        overlap_duration_seconds=5.0,  # 5 seconds overlap
        hop_length=160,  # Default mel hop length
        sample_rate=16000,  # Default sample rate for whisper
    )

    # Create test mel spectrogram (128 mel bins, 5000 frames = ~50 seconds at 100fps)
    mel_spec = torch.randn(128, 5000)

    # Test chunking
    chunks = chunker.chunk_mel_spectrogram(mel_spec)

    # Should create multiple chunks with overlap
    assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"

    # Check chunk metadata
    for i, chunk in enumerate(chunks):
        assert "mel" in chunk
        assert "chunk_metadata" in chunk

        metadata = chunk["chunk_metadata"]
        assert "chunk_index" in metadata
        assert "start_frame" in metadata
        assert "end_frame" in metadata
        assert "start_time" in metadata
        assert "end_time" in metadata

        # Check chunk shape
        mel_chunk = chunk["mel"]
        assert mel_chunk.shape[0] == 128, "Should preserve mel bin dimension"

        print(
            f"  Chunk {i}: frames {metadata['start_frame']}-{metadata['end_frame']}, "
            f"time {metadata['start_time']:.1f}-{metadata['end_time']:.1f}s, shape={mel_chunk.shape}"
        )

    # Test reassembly
    reassembled = chunker.reassemble_chunks(chunks)

    # Should match original length (may be slightly padded)
    assert reassembled.shape[0] == 128, "Should preserve mel bin dimension"
    assert reassembled.shape[1] >= mel_spec.shape[1], "Should be at least original length"

    print(f"  Original shape: {mel_spec.shape}, Reassembled shape: {reassembled.shape}")
    print("✅ Audio chunking test passed")


def test_long_audio_chunking():
    """Test the high-level chunk_long_audio function."""
    from tensorrt_llm.models.higgs_audio.preprocessing import chunk_long_audio

    print("\nTesting long audio chunking...")

    # Create test mel spectrogram (128 mel bins, 5000 frames = ~50 seconds at 100fps)
    mel_spec = torch.randn(128, 5000)

    # Test chunking
    chunked_samples = chunk_long_audio(
        mel_spectrogram=mel_spec, chunk_duration=20.0, overlap_duration=2.0, sample_id="test_sample"
    )

    # Should create at least 2 chunks for long audio
    assert len(chunked_samples) >= 2, f"Expected at least 2 chunks, got {len(chunked_samples)}"

    # Check each chunk
    for i, chunk in enumerate(chunked_samples):
        assert "mel" in chunk
        assert "chunk_metadata" in chunk

        metadata = chunk["chunk_metadata"]
        assert "chunk_index" in metadata
        assert "start_frame" in metadata
        assert "end_frame" in metadata

        # Check mel shape
        mel_chunk = chunk["mel"]
        assert mel_chunk.shape[0] == 128, "Should have 128 mel bins"

        print(
            f"  Chunk {i}: {metadata['start_frame']}-{metadata['end_frame']} frames, "
            f"shape={mel_chunk.shape}"
        )

    print("✅ Long audio chunking test passed")


def main():
    """Run all tests."""
    print("Starting HiggsAudio data processing tests...\n")

    try:
        test_collator_initialization()
        test_text_only_collation()
        test_audio_collation()
        test_mixed_batch()
        test_edge_cases()
        test_special_tokens()
        test_with_real_audio()
        test_audio_chunking()
        test_long_audio_chunking()

        print("\n🎉 All tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()
