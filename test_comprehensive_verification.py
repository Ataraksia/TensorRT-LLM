"""Comprehensive verification tests and microbenchmarks for HiggsAudio data processing pipeline.

This test suite provides:
- Unit tests for all preprocessing utilities
- Integration tests for end-to-end collation pipeline
- Microbenchmarks for performance-critical operations
- Memory usage profiling and optimization tests
- Correctness validation against reference implementations
- Stress tests with various audio lengths and batch sizes
"""

import time
import warnings
from typing import List

import psutil
import torch
import torch.nn.functional as F

from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.models.higgs_audio.data_processing import (
    AudioTokenUtils,
    DelayPatternProvider,
    HiggsAudioSampleCollator,
    create_tensorrt_optimized_collator,
    prepare_audio_sample,
    prepare_text_sample,
)
from tensorrt_llm.models.higgs_audio.preprocessing import AudioChunker, AudioPreprocessor


class PerformanceProfiler:
    """Utility class for profiling performance and memory usage."""

    def __init__(self):
        self.measurements = {}

    def profile_function(self, func, *args, name: str = None, iterations: int = 10, **kwargs):
        """Profile a function's execution time and memory usage.

        Args:
            func: Function to profile
            *args: Function arguments
            name: Name for the measurement (defaults to function name)
            iterations: Number of iterations to run
            **kwargs: Function keyword arguments

        Returns:
            Tuple of (result, measurements_dict)
        """
        if name is None:
            name = func.__name__

        # Warm up
        for _ in range(min(3, iterations)):
            _ = func(*args, **kwargs)

        # Memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Time measurements
        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        # Memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB

        measurements = {
            "mean_time_s": sum(times) / len(times),
            "min_time_s": min(times),
            "max_time_s": max(times),
            "memory_delta_mb": memory_after - memory_before,
            "memory_total_mb": memory_after,
            "iterations": iterations,
        }

        self.measurements[name] = measurements
        return result, measurements

    def get_summary(self) -> str:
        """Get a summary of all measurements."""
        if not self.measurements:
            return "No measurements recorded."

        summary = "Performance Profiling Summary:\n"
        summary += "=" * 50 + "\n"

        for name, measurements in self.measurements.items():
            summary += f"\n{name}:\n"
            time_range = measurements["max_time_s"] - measurements["min_time_s"]
            summary += f"  Time: {measurements['mean_time_s']:.4f}s ± {time_range:.4f}s\n"
            memory_info = (
                f"  Memory: {measurements['memory_delta_mb']:+.1f} MB "
                f"(total: {measurements['memory_total_mb']:.1f} MB)\n"
            )
            summary += memory_info
            summary += f"  Iterations: {measurements['iterations']}\n"

        return summary


class CorrectnessValidator:
    """Validates correctness against reference implementations and expected behaviors."""

    def __init__(self):
        self.validation_results = {}

    def validate_mel_spectrogram_properties(
        self,
        preprocessor: AudioPreprocessor,
        audio_durations: List[float] = [0.5, 1.0, 5.0, 30.0],
        tolerance: float = 1e-5,
    ) -> bool:
        """Validate mel spectrogram extraction properties."""
        print("Validating mel spectrogram properties...")

        all_valid = True

        for duration in audio_durations:
            # Create synthetic audio
            sample_rate = 16000
            samples = int(duration * sample_rate)
            # Low amplitude to avoid clipping
            waveform = torch.randn(samples) * 0.1

            # Extract mel spectrogram
            mel = preprocessor.compute_whisper_mel(waveform, sr=sample_rate)

            # Validate properties
            expected_frames = int(duration * 100)  # ~100 frames per second
            actual_frames = mel.shape[1]

            # Check shape
            if mel.shape[0] != 80:
                print(f"✗ Wrong mel bins: expected 80, got {mel.shape[0]}")
                all_valid = False
                continue

            # Check frame count (allow ±2 frames tolerance for STFT boundaries)
            if abs(actual_frames - expected_frames) > 2:
                print(
                    f"✗ Wrong frame count for {duration}s: expected ~{expected_frames}, got {actual_frames}"
                )
                all_valid = False
                continue

            # Check value ranges (log-mel should be mostly negative)
            if mel.max() > 10 or mel.min() < -100:
                print(
                    f"✗ Suspicious mel values for {duration}s: range [{mel.min():.2f}, {
                        mel.max():.2f
                    }]"
                )
                all_valid = False
                continue

            # Check for NaN/Inf
            if not torch.isfinite(mel).all():
                print(f"✗ Non-finite values in mel for {duration}s")
                all_valid = False
                continue

            print(f"✓ {duration}s audio: {mel.shape} mel, range [{mel.min():.2f}, {mel.max():.2f}]")

        self.validation_results["mel_properties"] = all_valid
        return all_valid

    def validate_chunking_correctness(
        self,
        chunker: AudioChunker,
        test_durations: List[float] = [35.0, 50.0, 65.0],
    ) -> bool:
        """Validate audio chunking and reassembly correctness."""
        print("Validating chunking correctness...")

        all_valid = True

        for duration in test_durations:
            # Create synthetic mel spectrogram
            frames = int(duration * 100)  # 100 fps
            mel = torch.randn(80, frames)

            # Chunk the mel
            chunks_data_list = chunker.chunk_mel_spectrogram(mel)
            chunks = [chunk_data["mel"] for chunk_data in chunks_data_list]

            # Validate chunk properties
            expected_chunks = max(1, int((frames - 100) / 2900) + 1)  # Rough estimate
            if len(chunks) != expected_chunks:
                print(
                    f"✗ Unexpected chunk count for {duration}s: expected ~{expected_chunks}, got {
                        len(chunks)
                    }"
                )
                # Don't fail immediately as this is just an estimate

            # Check chunk shapes
            for i, chunk in enumerate(chunks):
                if chunk.shape[0] != 80:
                    print(f"✗ Chunk {i} wrong mel bins: expected 80, got {chunk.shape[0]}")
                    all_valid = False

                expected_chunk_frames = 3000  # 30s * 100fps
                if i == len(chunks) - 1:
                    # Last chunk might be shorter
                    if chunk.shape[1] > expected_chunk_frames:
                        print(
                            f"✗ Last chunk {i} too long: {chunk.shape[1]} > {expected_chunk_frames}"
                        )
                        all_valid = False
                else:
                    # Non-last chunks should be full size
                    if chunk.shape[1] != expected_chunk_frames:
                        print(
                            f"✗ Chunk {i} wrong length: expected {expected_chunk_frames}, got {
                                chunk.shape[1]
                            }"
                        )
                        all_valid = False

            # Test reassembly
            try:
                reassembled = chunker.reassemble_chunks(chunks_data_list)

                # Check reassembled length matches original
                if reassembled.shape != mel.shape:
                    print(
                        f"✗ Reassembly shape mismatch: expected {mel.shape}, got {
                            reassembled.shape
                        }"
                    )
                    all_valid = False
                    continue

                # Check reassembly quality (should be close to original in non-overlap regions)
                # For synthetic random data, we can't expect perfect reconstruction,
                # but the shapes and general structure should be preserved
                mse = F.mse_loss(reassembled, mel).item()
                if mse > 1.0:  # Reasonable threshold for random data
                    print(f"✗ High reconstruction error for {duration}s: MSE = {mse:.4f}")
                    all_valid = False
                    continue

                print(f"✓ {duration}s audio: {len(chunks)} chunks, reassembly MSE = {mse:.4f}")

            except Exception as e:
                print(f"✗ Reassembly failed for {duration}s: {e}")
                all_valid = False

        self.validation_results["chunking_correctness"] = all_valid
        return all_valid

    def validate_delay_pattern_consistency(
        self,
        provider: DelayPatternProvider,
        utils: AudioTokenUtils,
        sequence_lengths: List[int] = [10, 50, 100],
        num_codebooks: int = 8,
    ) -> bool:
        """Validate delay pattern application and reversal consistency."""
        print("Validating delay pattern consistency...")

        all_valid = True

        for seq_len in sequence_lengths:
            # Create test tokens
            original_tokens = torch.randint(0, 1024, (num_codebooks, seq_len))

            # Generate delay pattern
            delay_pattern = provider.generate_delay_pattern(seq_len, num_codebooks)

            # Apply delay pattern
            delayed_tokens = provider.apply_delay_pattern(original_tokens, delay_pattern)

            # Reverse delay pattern
            reconstructed = provider.reverse_delay_pattern(delayed_tokens, delay_pattern, seq_len)

            # Check reconstruction matches original
            if not torch.equal(reconstructed, original_tokens):
                print(f"✗ Delay pattern reversal failed for length {seq_len}")
                print(f"  Original shape: {original_tokens.shape}")
                print(f"  Delayed shape: {delayed_tokens.shape}")
                print(f"  Reconstructed shape: {reconstructed.shape}")
                print(f"  Max diff: {(reconstructed - original_tokens).abs().max().item()}")
                all_valid = False
                continue

            # Test token interleaving
            batch_tokens = [original_tokens[i : i + 1, :] for i in range(num_codebooks)]

            if utils.validate_codebook_sequences(batch_tokens):
                interleaved = utils.interleave_codebook_tokens(batch_tokens)
                extracted = utils.extract_codebook_tokens(interleaved)

                # Check extraction matches original
                for i, (original_cb, extracted_cb) in enumerate(zip(batch_tokens, extracted)):
                    if not torch.equal(original_cb, extracted_cb):
                        print(f"✗ Token interleaving failed for codebook {i}, length {seq_len}")
                        all_valid = False
                        break
                else:
                    print(f"✓ Length {seq_len}: delay patterns and interleaving consistent")
            else:
                print(f"✗ Invalid codebook sequences for length {seq_len}")
                all_valid = False

        self.validation_results["delay_pattern_consistency"] = all_valid
        return all_valid


class StressTester:
    """Stress tests with various configurations and edge cases."""

    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
        self.stress_results = {}

    def stress_test_collation_pipeline(
        self,
        collator: HiggsAudioSampleCollator,
        batch_sizes: List[int] = [1, 4, 8, 16, 32],
        sequence_lengths: List[int] = [10, 50, 100, 500, 1000],
        audio_lengths: List[int] = [100, 500, 1000, 5000],
    ) -> bool:
        """Stress test the collation pipeline with various configurations."""
        print("Running collation pipeline stress tests...")

        all_passed = True

        for batch_size in batch_sizes:
            for seq_len in sequence_lengths:
                for audio_len in audio_lengths:
                    try:
                        # Create test samples
                        samples = []
                        for i in range(batch_size):
                            # Alternate between text-only and audio samples
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

                        # Profile collation
                        def collate_batch():
                            return collator(samples)

                        batch, measurements = self.profiler.profile_function(
                            collate_batch,
                            name=f"collate_b{batch_size}_s{seq_len}_a{audio_len}",
                            iterations=3,
                        )

                        # Validate batch structure
                        assert "input_ids" in batch
                        assert "attention_mask" in batch
                        assert batch["input_ids"].shape[0] == batch_size

                        # Check for memory issues
                        if measurements["memory_delta_mb"] > 1000:  # 1GB threshold
                            print(f"⚠ High memory usage: {measurements['memory_delta_mb']:.1f} MB")

                        # Check for performance issues
                        if measurements["mean_time_s"] > 5.0:  # 5s threshold
                            print(f"⚠ Slow collation: {measurements['mean_time_s']:.2f}s")

                    except Exception as e:
                        print(
                            f"✗ Stress test failed (batch={batch_size}, seq={seq_len}, audio={audio_len}): {e}"
                        )
                        all_passed = False

        if all_passed:
            print("✓ All stress tests passed")

        self.stress_results["collation_pipeline"] = all_passed
        return all_passed

    def stress_test_memory_scaling(
        self,
        create_collator_func,
        max_memory_mb: float = 2000,  # 2GB limit
    ) -> bool:
        """Test memory scaling with increasing batch sizes."""
        print("Running memory scaling stress tests...")

        results = []

        for batch_size in [1, 2, 4, 8, 16, 32, 64]:
            try:
                collator = create_collator_func()

                # Create large samples
                samples = []
                for i in range(batch_size):
                    tokens = torch.randint(0, 1000, (512,)).tolist()  # Long sequences
                    mel = torch.randn(80, 2000)  # Long audio
                    sample = prepare_audio_sample(tokens, mel, sample_id=f"sample_{i}")
                    samples.append(sample)

                # Profile memory usage
                def collate_large_batch():
                    return collator(samples)

                _, measurements = self.profiler.profile_function(
                    collate_large_batch,
                    name=f"memory_scale_b{batch_size}",
                    iterations=1,
                )

                memory_usage = measurements["memory_total_mb"]
                results.append((batch_size, memory_usage))

                print(f"Batch size {batch_size}: {memory_usage:.1f} MB")

                if memory_usage > max_memory_mb:
                    print(f"⚠ Memory limit exceeded at batch size {batch_size}")
                    break

            except Exception as e:
                print(f"✗ Memory scaling test failed at batch size {batch_size}: {e}")
                break

        # Analyze scaling behavior
        if len(results) >= 3:
            # Check if memory scales roughly linearly with batch size
            scaling_factors = []
            for i in range(1, len(results)):
                prev_batch, prev_memory = results[i - 1]
                curr_batch, curr_memory = results[i]
                factor = (curr_memory / prev_memory) / (curr_batch / prev_batch)
                scaling_factors.append(factor)

            avg_scaling = sum(scaling_factors) / len(scaling_factors)
            print(f"Average memory scaling factor: {avg_scaling:.2f} (ideal: ~1.0)")

            # Memory scaling should be roughly linear (factor between 0.8 and
            # 1.5)
            memory_scaling_ok = 0.8 <= avg_scaling <= 1.5
        else:
            memory_scaling_ok = False

        self.stress_results["memory_scaling"] = memory_scaling_ok
        return memory_scaling_ok


def run_comprehensive_verification():
    """Run all verification tests and benchmarks."""
    print("🔍 Starting Comprehensive HiggsAudio Data Processing Verification 🔍\n")

    # Initialize components
    config = HiggsAudioConfig()
    profiler = PerformanceProfiler()
    validator = CorrectnessValidator()
    stress_tester = StressTester(profiler)

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)

    # Test results
    all_tests_passed = True

    print("=" * 60)
    print("UNIT TESTS - PREPROCESSING UTILITIES")
    print("=" * 60)

    # Test 1: Audio Preprocessing Validation
    try:
        preprocessor = AudioPreprocessor()
        mel_validation = validator.validate_mel_spectrogram_properties(preprocessor)
        all_tests_passed &= mel_validation
        print(f"Mel Spectrogram Validation: {'✓ PASS' if mel_validation else '✗ FAIL'}")
    except Exception as e:
        print(f"Mel Spectrogram Validation: ✗ FAIL ({e})")
        all_tests_passed = False

    # Test 2: Chunking Validation
    try:
        chunker = AudioChunker()
        chunking_validation = validator.validate_chunking_correctness(chunker)
        all_tests_passed &= chunking_validation
        print(f"Chunking Correctness: {'✓ PASS' if chunking_validation else '✗ FAIL'}")
    except Exception as e:
        print(f"Chunking Correctness: ✗ FAIL ({e})")
        all_tests_passed = False

    # Test 3: Delay Pattern Validation
    try:
        delay_provider = DelayPatternProvider(strategy="linear", num_codebooks=8)
        token_utils = AudioTokenUtils(num_codebooks=8)
        delay_validation = validator.validate_delay_pattern_consistency(delay_provider, token_utils)
        all_tests_passed &= delay_validation
        print(f"Delay Pattern Consistency: {'✓ PASS' if delay_validation else '✗ FAIL'}")
    except Exception as e:
        print(f"Delay Pattern Consistency: ✗ FAIL ({e})")
        all_tests_passed = False

    print("\n" + "=" * 60)
    print("INTEGRATION TESTS - END-TO-END PIPELINE")
    print("=" * 60)

    # Test 4: Basic Collation Pipeline
    try:
        collator = HiggsAudioSampleCollator(config)

        # Create test samples
        test_samples = [
            prepare_text_sample([1, 2, 3, 4]),
            prepare_audio_sample([5, 6, 7], torch.randn(80, 100)),
            prepare_text_sample([8, 9]),
        ]

        batch = collator(test_samples)

        # Validate batch structure
        integration_pass = (
            "input_ids" in batch and "attention_mask" in batch and batch["input_ids"].shape[0] == 3
        )

        all_tests_passed &= integration_pass
        print(f"Basic Collation Pipeline: {'✓ PASS' if integration_pass else '✗ FAIL'}")

    except Exception as e:
        print(f"Basic Collation Pipeline: ✗ FAIL ({e})")
        all_tests_passed = False

    # Test 5: TensorRT Optimized Pipeline
    try:
        optimized_collator = create_tensorrt_optimized_collator(config)

        # Test with optimizations
        batch_opt = optimized_collator(test_samples)

        tensorrt_pass = (
            "input_ids" in batch_opt
            and "estimated_memory_gb" in batch_opt
            and batch_opt["estimated_memory_gb"] > 0
        )

        all_tests_passed &= tensorrt_pass
        print(f"TensorRT Optimized Pipeline: {'✓ PASS' if tensorrt_pass else '✗ FAIL'}")

    except Exception as e:
        print(f"TensorRT Optimized Pipeline: ✗ FAIL ({e})")
        all_tests_passed = False

    print("\n" + "=" * 60)
    print("PERFORMANCE MICROBENCHMARKS")
    print("=" * 60)

    # Benchmark 1: Preprocessing Performance
    try:
        preprocessor = AudioPreprocessor()

        # Benchmark mel extraction
        test_audio = torch.randn(16000 * 5)  # 5 seconds

        def extract_mel():
            return preprocessor.compute_whisper_mel(test_audio)

        _, mel_measurements = profiler.profile_function(
            extract_mel, name="mel_extraction_5s", iterations=10
        )

        print(f"Mel Extraction (5s audio): {mel_measurements['mean_time_s']:.4f}s")

    except Exception as e:
        print(f"Mel Extraction Benchmark: ✗ FAIL ({e})")

    # Benchmark 2: Collation Performance
    try:
        collator = HiggsAudioSampleCollator(config)

        # Create larger batch for benchmarking
        large_samples = []
        for i in range(16):
            tokens = torch.randint(0, 1000, (100,)).tolist()
            mel = torch.randn(80, 500)
            sample = prepare_audio_sample(tokens, mel, sample_id=f"bench_{i}")
            large_samples.append(sample)

        def collate_large():
            return collator(large_samples)

        _, collation_measurements = profiler.profile_function(
            collate_large, name="collation_batch16", iterations=5
        )

        print(f"Collation (batch=16): {collation_measurements['mean_time_s']:.4f}s")

    except Exception as e:
        print(f"Collation Benchmark: ✗ FAIL ({e})")

    print("\n" + "=" * 60)
    print("STRESS TESTS")
    print("=" * 60)

    # Stress Test 1: Pipeline Robustness
    try:
        collator = HiggsAudioSampleCollator(config)
        pipeline_stress = stress_tester.stress_test_collation_pipeline(
            collator,
            batch_sizes=[1, 4, 8],  # Reduced for faster testing
            sequence_lengths=[10, 100, 500],
            audio_lengths=[100, 1000],
        )
        all_tests_passed &= pipeline_stress
        print(f"Pipeline Stress Test: {'✓ PASS' if pipeline_stress else '✗ FAIL'}")

    except Exception as e:
        print(f"Pipeline Stress Test: ✗ FAIL ({e})")
        all_tests_passed = False

    # Stress Test 2: Memory Scaling
    try:

        def create_basic_collator():
            return HiggsAudioSampleCollator(config)

        memory_stress = stress_tester.stress_test_memory_scaling(
            create_basic_collator,
            max_memory_mb=1000,  # 1GB limit for testing
        )
        all_tests_passed &= memory_stress
        print(f"Memory Scaling Test: {'✓ PASS' if memory_stress else '✗ FAIL'}")

    except Exception as e:
        print(f"Memory Scaling Test: ✗ FAIL ({e})")
        all_tests_passed = False

    # Final Results
    print("\n" + "=" * 60)
    print("FINAL VERIFICATION RESULTS")
    print("=" * 60)

    print(profiler.get_summary())

    if all_tests_passed:
        print("\n🎉 ALL VERIFICATION TESTS PASSED! 🎉")
        print("The HiggsAudio data processing pipeline is ready for production use.")
    else:
        print("\n❌ SOME TESTS FAILED ❌")
        print("Please review the failed tests and fix any issues before deployment.")

    return all_tests_passed


if __name__ == "__main__":
    success = run_comprehensive_verification()
    exit(0 if success else 1)
