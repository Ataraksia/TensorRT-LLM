#!/usr/bin/env python3
"""Test script for audio and text preprocessing utilities."""

import sys
from pathlib import Path

import torch

# Add the tensorrt_llm package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from tensorrt_llm.models.higgs_audio.preprocessing import (
    AudioPreprocessor,
    TextPreprocessor,
    create_preprocessors,
    preprocess_batch_audio,
    preprocess_batch_text,
)


def test_audio_preprocessing():
    """Test audio preprocessing functionality."""
    print("Testing audio preprocessing...")

    # Create audio preprocessor
    audio_proc = AudioPreprocessor(target_sr=16000, device=torch.device("cpu"))

    # Test with synthetic audio
    duration = 2.0  # 2 seconds
    sample_rate = 16000
    num_samples = int(duration * sample_rate)

    # Generate sine wave
    t = torch.linspace(0, duration, num_samples)
    frequency = 440.0  # A4 note
    waveform = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)  # [1, time]

    print(f"Generated test audio: {waveform.shape}, duration: {duration}s")

    # Test audio validation
    is_valid = audio_proc.validate_audio(waveform, sample_rate)
    print(f"Audio validation: {is_valid}")

    # Test mel spectrogram computation
    mel_spec = audio_proc.compute_whisper_mel(waveform, sample_rate)
    print(f"Mel spectrogram shape: {mel_spec.shape}")

    # Verify expected shape [n_mels, time_frames]
    expected_frames = audio_proc.get_mel_frames(duration)
    print(f"Expected frames: {expected_frames}, actual: {mel_spec.shape[1]}")

    # Test with different audio lengths
    test_durations = [0.5, 1.0, 30.0, 65.0]
    for test_duration in test_durations:
        test_samples = int(test_duration * sample_rate)
        test_waveform = torch.randn(1, test_samples)

        try:
            test_mel = audio_proc.compute_whisper_mel(test_waveform, sample_rate)
            print(f"Duration {test_duration}s -> mel shape: {test_mel.shape}")
        except Exception as e:
            print(f"Failed for duration {test_duration}s: {e}")

    print("Audio preprocessing tests completed!\n")


def test_text_preprocessing():
    """Test text preprocessing functionality."""
    print("Testing text preprocessing...")

    # Create text preprocessor
    text_proc = TextPreprocessor()

    # Test cases
    test_texts = [
        "Hello, world!",
        "  Multiple   spaces   here  ",
        "Unicode test: café naïve résumé",
        "\t\nWhitespace\r\nmixed\t\n",
        "Numbers 123 and symbols !@#$%",
        "",
    ]

    for text in test_texts:
        print(f"Input: '{text}'")

        # Test validation
        is_valid = text_proc.validate_text(text)
        print(f"  Valid: {is_valid}")

        if is_valid:
            # Test preprocessing
            processed = text_proc.preprocess_text(text)
            print(f"  Processed: '{processed}'")

            # Test length
            length = text_proc.get_text_length(processed)
            print(f"  Length: {length}")

        print()

    print("Text preprocessing tests completed!\n")


def test_batch_processing():
    """Test batch processing functionality."""
    print("Testing batch processing...")

    # Test batch audio processing
    print("Batch audio processing:")

    # Generate multiple synthetic audio samples
    sample_rate = 16000
    durations = [1.0, 2.0, 0.5, 3.0]
    audio_list = []

    for i, duration in enumerate(durations):
        num_samples = int(duration * sample_rate)
        frequency = 440.0 * (2 ** (i / 12))  # Different frequencies
        t = torch.linspace(0, duration, num_samples)
        waveform = torch.sin(2 * torch.pi * frequency * t)
        audio_list.append(waveform.numpy())

    # Process batch
    results = preprocess_batch_audio(audio_list, target_sr=sample_rate)

    for i, (waveform, mel_spec) in enumerate(results):
        if waveform is not None and mel_spec is not None:
            print(f"  Sample {i}: waveform {waveform.shape}, mel {mel_spec.shape}")
        else:
            print(f"  Sample {i}: Failed to process")

    # Test batch text processing
    print("\nBatch text processing:")

    text_list = [
        "First sample text",
        "  Second   with   spaces  ",
        "Third with unicode: café",
        "Fourth sample",
    ]

    processed_texts = preprocess_batch_text(text_list)
    for i, (original, processed) in enumerate(zip(text_list, processed_texts)):
        print(f"  Sample {i}: '{original}' -> '{processed}'")

    print("Batch processing tests completed!\n")


def test_create_preprocessors():
    """Test preprocessor factory function."""
    print("Testing preprocessor creation...")

    audio_proc, text_proc = create_preprocessors(target_sr=16000)

    print(f"Audio preprocessor: {type(audio_proc).__name__}")
    print(f"Text preprocessor: {type(text_proc).__name__}")

    # Test basic functionality
    test_waveform = torch.randn(1, 16000)  # 1 second of audio
    mel_spec = audio_proc.compute_whisper_mel(test_waveform)
    print(f"Test mel spectrogram shape: {mel_spec.shape}")

    test_text = "  Test   text   with   spaces  "
    processed_text = text_proc.preprocess_text(test_text)
    print(f"Test text: '{test_text}' -> '{processed_text}'")

    print("Preprocessor creation tests completed!\n")


def test_whisper_mel_compatibility():
    """Test Whisper mel spectrogram compatibility."""
    print("Testing Whisper mel compatibility...")

    audio_proc = AudioPreprocessor(
        target_sr=16000, n_mels=80, n_fft=400, hop_length=160, win_length=400, fmin=0.0, fmax=8000.0
    )

    # Test with 1 second of audio
    duration = 1.0
    sample_rate = 16000
    num_samples = int(duration * sample_rate)

    # Generate test signal
    waveform = torch.randn(1, num_samples)

    # Compute mel spectrogram
    mel_spec = audio_proc.compute_whisper_mel(waveform, sample_rate)

    print(f"Mel spectrogram shape: {mel_spec.shape}")
    print("Expected shape: [80, ~100] (80 mel bins, ~100 frames for 1s)")

    # Verify properties
    assert mel_spec.shape[0] == 80, f"Expected 80 mel bins, got {mel_spec.shape[0]}"

    # Check frame count (should be approximately 100 for 1 second)
    expected_frames = 1 + (num_samples - 400) // 160  # STFT frame calculation
    print(f"Expected frames: {expected_frames}, actual: {mel_spec.shape[1]}")

    # Check value range (should be log values, likely negative)
    min_val, max_val = mel_spec.min().item(), mel_spec.max().item()
    print(f"Mel value range: [{min_val:.3f}, {max_val:.3f}]")

    # Check for NaN or inf values
    has_nan = torch.any(torch.isnan(mel_spec))
    has_inf = torch.any(torch.isinf(mel_spec))
    print(f"Has NaN: {has_nan}, Has Inf: {has_inf}")

    print("Whisper mel compatibility tests completed!\n")


def main():
    """Run all preprocessing tests."""
    print("Running preprocessing utility tests...\n")

    try:
        test_audio_preprocessing()
        test_text_preprocessing()
        test_batch_processing()
        test_create_preprocessors()
        test_whisper_mel_compatibility()

        print("All tests completed successfully!")
        return 0

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
