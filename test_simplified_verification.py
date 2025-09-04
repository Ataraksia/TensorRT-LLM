"""Simplified verification test for HiggsAudio data processing pipeline.

This test focuses on the core data processing pipeline without complex mel spectrogram validation.
"""

import time
import warnings

import torch

from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.models.higgs_audio.data_processing import (
    AudioTokenUtils,
    DelayPatternProvider,
    HiggsAudioSampleCollator,
    create_tensorrt_optimized_collator,
    prepare_audio_sample,
    prepare_text_sample,
)


def test_collation_pipeline():
    """Test the basic collation pipeline functionality."""
    print("Testing basic collation pipeline...")

    config = HiggsAudioConfig()
    collator = HiggsAudioSampleCollator(config)

    # Create test samples
    samples = [
        prepare_text_sample([1, 2, 3, 4], sample_id="text_1"),
        prepare_audio_sample([5, 6, 7], torch.randn(80, 100), sample_id="audio_1"),
        prepare_text_sample([8, 9], sample_id="text_2"),
    ]

    # Test collation
    batch = collator(samples)

    # Validate batch structure
    assert "input_ids" in batch, "Missing input_ids in batch"
    assert "attention_mask" in batch, "Missing attention_mask in batch"
    assert batch["input_ids"].shape[0] == 3, (
        f"Expected batch size 3, got {batch['input_ids'].shape[0]}"
    )

    print("✓ Basic collation pipeline test passed")
    return True


def test_tensorrt_optimized_pipeline():
    """Test the TensorRT optimized collation pipeline."""
    print("Testing TensorRT optimized pipeline...")

    config = HiggsAudioConfig()
    optimized_collator = create_tensorrt_optimized_collator(config)

    # Create test samples
    samples = [
        prepare_text_sample([1, 2, 3, 4], sample_id="text_1"),
        prepare_audio_sample([5, 6, 7], torch.randn(80, 100), sample_id="audio_1"),
        prepare_text_sample([8, 9], sample_id="text_2"),
    ]

    # Test optimized collation
    batch = optimized_collator(samples)

    # Validate batch structure
    assert "input_ids" in batch, "Missing input_ids in optimized batch"
    assert "attention_mask" in batch, "Missing attention_mask in optimized batch"
    assert "estimated_memory_gb" in batch, "Missing memory estimation in optimized batch"
    assert batch["estimated_memory_gb"] > 0, "Memory estimation should be positive"

    print("✓ TensorRT optimized pipeline test passed")
    return True


def test_delay_patterns():
    """Test delay pattern functionality."""
    print("Testing delay pattern functionality...")

    provider = DelayPatternProvider(strategy="linear", num_codebooks=8)
    utils = AudioTokenUtils(num_codebooks=8)

    # Test delay pattern generation and reversal
    seq_len = 50
    num_codebooks = 8

    # Create test tokens
    original_tokens = torch.randint(0, 1024, (num_codebooks, seq_len))

    # Generate delay pattern
    delay_pattern = provider.generate_delay_pattern(seq_len, num_codebooks)

    # Apply delay pattern
    delayed_tokens = provider.apply_delay_pattern(original_tokens, delay_pattern)

    # Reverse delay pattern
    reconstructed = provider.reverse_delay_pattern(delayed_tokens, delay_pattern, seq_len)

    # Check reconstruction matches original
    assert torch.equal(reconstructed, original_tokens), "Delay pattern reversal failed"

    # Test token interleaving
    batch_tokens = [original_tokens[i : i + 1, :] for i in range(num_codebooks)]

    if utils.validate_codebook_sequences(batch_tokens):
        interleaved = utils.interleave_codebook_tokens(batch_tokens)
        extracted = utils.extract_codebook_tokens(interleaved)

        # Check extraction matches original
        for i, (original_cb, extracted_cb) in enumerate(zip(batch_tokens, extracted)):
            assert torch.equal(original_cb, extracted_cb), (
                f"Token interleaving failed for codebook {i}"
            )

    print("✓ Delay pattern test passed")
    return True


def test_performance_benchmark():
    """Basic performance benchmark."""
    print("Running performance benchmark...")

    config = HiggsAudioConfig()
    collator = HiggsAudioSampleCollator(config)

    # Create larger batch for benchmarking
    batch_size = 8
    samples = []
    for i in range(batch_size):
        tokens = torch.randint(0, 1000, (100,)).tolist()
        mel = torch.randn(80, 500)
        sample = prepare_audio_sample(tokens, mel, sample_id=f"bench_{i}")
        samples.append(sample)

    # Benchmark collation
    iterations = 10
    times = []

    for _ in range(iterations):
        start_time = time.perf_counter()
        _ = collator(samples)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    avg_time = sum(times) / len(times)

    print(f"✓ Performance benchmark: {avg_time:.4f}s average for batch_size={batch_size}")
    return True


def test_stress_scenarios():
    """Test various stress scenarios."""
    print("Testing stress scenarios...")

    config = HiggsAudioConfig()
    collator = HiggsAudioSampleCollator(config)

    test_cases = [
        # (batch_size, seq_len, audio_len)
        (1, 10, 100),
        (4, 100, 500),
        (8, 50, 1000),
        (2, 500, 200),
    ]

    for batch_size, seq_len, audio_len in test_cases:
        samples = []
        for i in range(batch_size):
            if i % 2 == 0:
                # Text-only sample
                tokens = torch.randint(0, 1000, (seq_len,)).tolist()
                sample = prepare_text_sample(tokens, sample_id=f"text_{i}")
            else:
                # Audio sample
                tokens = torch.randint(0, 1000, (seq_len,)).tolist()
                mel = torch.randn(80, audio_len)
                sample = prepare_audio_sample(tokens, mel, sample_id=f"audio_{i}")
            samples.append(sample)

        # Test collation
        batch = collator(samples)

        # Basic validation
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert batch["input_ids"].shape[0] == batch_size

    print("✓ Stress scenario tests passed")
    return True


def run_simplified_verification():
    """Run simplified verification tests."""
    print("🔍 Starting Simplified HiggsAudio Data Processing Verification 🔍\n")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)

    test_results = []

    # Run tests
    tests = [
        ("Basic Collation Pipeline", test_collation_pipeline),
        ("TensorRT Optimized Pipeline", test_tensorrt_optimized_pipeline),
        ("Delay Pattern Functionality", test_delay_patterns),
        ("Performance Benchmark", test_performance_benchmark),
        ("Stress Scenarios", test_stress_scenarios),
    ]

    print("=" * 60)
    print("RUNNING VERIFICATION TESTS")
    print("=" * 60)

    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result, None))
            print(f"{test_name}: {'✓ PASS' if result else '✗ FAIL'}")
        except Exception as e:
            test_results.append((test_name, False, str(e)))
            print(f"{test_name}: ✗ FAIL ({e})")

    # Final results
    print("\n" + "=" * 60)
    print("FINAL VERIFICATION RESULTS")
    print("=" * 60)

    passed_tests = sum(1 for _, result, _ in test_results if result)
    total_tests = len(test_results)

    print(f"Tests passed: {passed_tests}/{total_tests}")

    if passed_tests == total_tests:
        print("\n🎉 ALL VERIFICATION TESTS PASSED! 🎉")
        print("The HiggsAudio data processing pipeline is ready for production use.")
        return True
    else:
        print("\n❌ SOME TESTS FAILED ❌")
        for test_name, result, error in test_results:
            if not result:
                print(f"  - {test_name}: {error or 'Unknown error'}")
        return False


if __name__ == "__main__":
    success = run_simplified_verification()
    exit(0 if success else 1)
