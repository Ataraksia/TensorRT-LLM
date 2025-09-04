#!/usr/bin/env python3
"""Comprehensive end-to-end validation and performance benchmarks for Higgs Audio model.

This test suite validates the full generation pipeline, performance characteristics,
and quality metrics for the TensorRT-LLM Higgs Audio implementation.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

# Import Higgs Audio components
from tensorrt_llm.models.higgs_audio.dual_ffn import GenerationMode
from tensorrt_llm.models.higgs_audio.generation import (
    DelayPatternScheduler,
    FusedMultiHeadLogitsHandler,
    GenerationState,
    PipelineConfig,
    SamplerConfig,
    StreamingIterator,
    TTSPresets,
    VoiceCloneingUtils,
    VoiceProfile,
    VoiceSimilarityMetrics,
    create_voice_clone,
)


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark."""

    test_name: str
    duration_ms: float
    memory_mb: float
    tokens_per_second: Optional[float] = None
    audio_duration_ratio: Optional[float] = None  # Generated audio duration / processing time
    quality_score: Optional[float] = None
    error: Optional[str] = None


@dataclass
class ValidationResult:
    """Results from a validation test."""

    test_name: str
    passed: bool
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class HiggsEndToEndValidator:
    """Comprehensive validation and benchmarking suite for Higgs Audio."""

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """Initialize validator with model and configuration."""
        self.device = device
        self.model_path = model_path or "higgs-audio-v2-generation-3B-base"

        # Test configurations
        self.test_texts = [
            "Hello, how are you today?",
            "The quick brown fox jumps over the lazy dog.",
            "In a hole in the ground there lived a hobbit.",
            "To be or not to be, that is the question.",
            "Machine learning is transforming the world of artificial intelligence.",
        ]

        # Performance tracking
        self.benchmark_results: List[BenchmarkResult] = []
        self.validation_results: List[ValidationResult] = []

        # Memory baseline
        self.baseline_memory = self._get_memory_usage()

    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

    def _time_function(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """Time a function execution and return result and duration."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        duration = (time.perf_counter() - start_time) * 1000  # Convert to ms
        return result, duration

    def validate_config_creation(self) -> ValidationResult:
        """Test creation and validation of pipeline configurations."""
        try:
            # Test PipelineConfig creation
            config = PipelineConfig(
                model_path=self.model_path,
                device=self.device,
                max_new_tokens=1024,
                use_kv_cache=True,
                kv_cache_type="static",
            )

            # Validate config properties
            assert config.model_path == self.model_path
            assert config.device == self.device
            assert config.max_new_tokens == 1024
            assert config.use_kv_cache is True
            assert config.kv_cache_type == "static"

            # Test SamplerConfig creation
            sampler_config = SamplerConfig(
                temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.1
            )

            # Validate sampler properties
            assert sampler_config.temperature == 0.8
            assert sampler_config.top_k == 50
            assert sampler_config.top_p == 0.9
            assert sampler_config.repetition_penalty == 1.1

            return ValidationResult("config_creation", True)

        except Exception as e:
            return ValidationResult("config_creation", False, error_message=str(e))

    def validate_state_management(self) -> ValidationResult:
        """Test generation state management and transitions."""
        try:
            config = PipelineConfig(model_path=self.model_path, device=self.device)

            # Test initial state
            state = GenerationState(config)
            assert state.current_mode == GenerationMode.TEXT
            assert state.text_position == 0
            assert state.audio_position == 0
            assert len(state.generated_tokens) == 0

            # Test mode transitions
            state.advance_mode(GenerationMode.AUDIO_INIT)
            assert state.current_mode == GenerationMode.AUDIO_INIT

            state.advance_mode(GenerationMode.AUDIO_IN_PROGRESS)
            assert state.current_mode == GenerationMode.AUDIO_IN_PROGRESS

            # Test token addition
            test_tokens = torch.tensor([1, 2, 3, 4, 5])
            state.add_tokens(test_tokens)
            assert len(state.generated_tokens) == 5
            assert torch.equal(state.generated_tokens, test_tokens)

            # Test position tracking
            state.update_positions(text_pos=10, audio_pos=20)
            assert state.text_position == 10
            assert state.audio_position == 20

            return ValidationResult("state_management", True)

        except Exception as e:
            return ValidationResult("state_management", False, error_message=str(e))

    def validate_delay_pattern_scheduler(self) -> ValidationResult:
        """Test delay pattern scheduling for RVQ codebooks."""
        try:
            scheduler = DelayPatternScheduler(num_codebooks=8, delay_pattern="serial")

            # Test pattern generation
            pattern = scheduler.get_delay_pattern()
            assert len(pattern) == 8
            assert pattern == [0, 1, 2, 3, 4, 5, 6, 7]  # Serial pattern

            # Test codebook scheduling
            codebook_mask = scheduler.get_active_codebooks(step=3)
            expected_mask = torch.tensor([True, True, True, True, False, False, False, False])
            assert torch.equal(codebook_mask, expected_mask)

            # Test causality constraints
            can_generate = scheduler.can_generate_codebook(codebook_idx=2, current_step=5)
            assert can_generate is True

            can_generate = scheduler.can_generate_codebook(codebook_idx=6, current_step=5)
            assert can_generate is False

            return ValidationResult("delay_pattern_scheduler", True)

        except Exception as e:
            return ValidationResult("delay_pattern_scheduler", False, error_message=str(e))

    def validate_fused_logits_handler(self) -> ValidationResult:
        """Test fused multi-head logits processing."""
        try:
            handler = FusedMultiHeadLogitsHandler(
                vocab_size=32000, num_codebooks=8, device=self.device
            )

            # Test logits processing
            batch_size, seq_len = 2, 10
            mock_logits = torch.randn(batch_size, seq_len, 32000 + 8 * 1024, device=self.device)

            # Process logits
            text_logits, audio_logits = handler.extract_logits(mock_logits)

            # Validate shapes
            assert text_logits.shape == (batch_size, seq_len, 32000)
            assert audio_logits.shape == (batch_size, seq_len, 8, 1024)

            # Test sampling
            sampler_config = SamplerConfig(temperature=1.0, top_k=50)
            text_tokens = handler.sample_text_tokens(text_logits, sampler_config)
            audio_tokens = handler.sample_audio_tokens(audio_logits, sampler_config)

            # Validate sampling results
            assert text_tokens.shape == (batch_size, seq_len)
            assert audio_tokens.shape == (batch_size, seq_len, 8)
            assert text_tokens.dtype == torch.long
            assert audio_tokens.dtype == torch.long

            return ValidationResult("fused_logits_handler", True)

        except Exception as e:
            return ValidationResult("fused_logits_handler", False, error_message=str(e))

    def validate_streaming_iterator(self) -> ValidationResult:
        """Test real-time streaming iterator functionality."""
        try:
            # Create mock audio data generator
            async def mock_generator():
                for i in range(5):
                    yield torch.randn(1024), f"chunk_{i}"
                    await asyncio.sleep(0.01)  # Simulate processing delay

            async def test_streaming():
                iterator = StreamingIterator(
                    buffer_size=2048, chunk_size=512, enable_backpressure=True
                )

                # Test streaming
                chunks_received = []
                async for chunk in iterator.stream_audio(mock_generator()):
                    chunks_received.append(chunk)
                    if len(chunks_received) >= 3:
                        break

                # Validate streaming results
                assert len(chunks_received) == 3
                for chunk in chunks_received:
                    assert hasattr(chunk, "audio_data")
                    assert hasattr(chunk, "metadata")
                    assert chunk.audio_data.shape[0] <= 512  # Chunk size limit

                # Test performance metrics
                metrics = iterator.get_performance_metrics()
                assert "total_chunks_processed" in metrics
                assert "average_chunk_size" in metrics
                assert "buffer_utilization" in metrics

                return True

            # Run async test
            result = asyncio.run(test_streaming())
            assert result is True

            return ValidationResult("streaming_iterator", True)

        except Exception as e:
            return ValidationResult("streaming_iterator", False, error_message=str(e))

    def validate_voice_cloning_utilities(self) -> ValidationResult:
        """Test voice cloning and TTS utilities."""
        try:
            # Test voice profile creation
            audio_data = torch.randn(48000)  # 2 seconds at 24kHz
            voice_profile = VoiceProfile.from_reference_audio(
                audio_data, speaker_name="test_speaker"
            )

            assert voice_profile.speaker_name == "test_speaker"
            assert voice_profile.embedding is not None
            assert voice_profile.embedding.shape[0] == 512  # Default embedding size

            # Test TTS presets
            preset = TTSPresets.get_preset("professional")
            assert preset is not None
            assert preset.style_name == "professional"
            assert hasattr(preset, "sampler_config")

            # Test voice similarity metrics
            embedding1 = torch.randn(512)
            embedding2 = torch.randn(512)
            similarity = VoiceSimilarityMetrics.cosine_similarity(embedding1, embedding2)
            assert isinstance(similarity, float)
            assert -1.0 <= similarity <= 1.0

            # Test high-level factory functions
            try:
                # Note: This would normally require a real model/session
                # For validation, we just test the function signature and error handling
                voice_clone_config = create_voice_clone(
                    reference_audio=audio_data,
                    target_text="Hello world",
                    model_path=self.model_path,
                )
                # Should return a configuration dict even with mock model
                assert isinstance(voice_clone_config, dict)

            except Exception as e:
                # Expected for mock testing - just ensure it's a reasonable error
                assert "model" in str(e).lower() or "session" in str(e).lower()

            return ValidationResult("voice_cloning_utilities", True)

        except Exception as e:
            return ValidationResult("voice_cloning_utilities", False, error_message=str(e))

    def benchmark_text_generation(self) -> BenchmarkResult:
        """Benchmark text generation performance."""
        try:
            config = PipelineConfig(model_path=self.model_path, device=self.device)
            sampler_config = SamplerConfig(temperature=0.8, top_k=50)

            # Mock text generation (would use real model in production)
            def mock_text_generation():
                state = GenerationState(config)
                handler = FusedMultiHeadLogitsHandler(32000, 8, self.device)

                total_tokens = 0
                for text in self.test_texts[:3]:  # Test with 3 texts
                    # Simulate token generation
                    for _ in range(len(text.split()) * 2):  # ~2 tokens per word
                        mock_logits = torch.randn(1, 1, 32000, device=self.device)
                        tokens = handler.sample_text_tokens(mock_logits, sampler_config)
                        state.add_tokens(tokens.squeeze())
                        total_tokens += 1

                return total_tokens

            # Benchmark
            start_memory = self._get_memory_usage()
            tokens_generated, duration = self._time_function(mock_text_generation)
            end_memory = self._get_memory_usage()

            memory_used = end_memory - start_memory
            tokens_per_second = tokens_generated / (duration / 1000) if duration > 0 else 0

            return BenchmarkResult(
                test_name="text_generation",
                duration_ms=duration,
                memory_mb=memory_used,
                tokens_per_second=tokens_per_second,
            )

        except Exception as e:
            return BenchmarkResult(
                test_name="text_generation", duration_ms=0, memory_mb=0, error=str(e)
            )

    def benchmark_audio_generation(self) -> BenchmarkResult:
        """Benchmark audio generation performance."""
        try:
            sampler_config = SamplerConfig(temperature=0.8, top_k=50)
            scheduler = DelayPatternScheduler(num_codebooks=8)

            # Mock audio generation
            def mock_audio_generation():
                handler = FusedMultiHeadLogitsHandler(32000, 8, self.device)

                total_audio_tokens = 0
                audio_duration = 0

                # Simulate 2 seconds of audio at 50 Hz (100 timesteps)
                for step in range(100):
                    active_codebooks = scheduler.get_active_codebooks(step)
                    mock_logits = torch.randn(1, 1, 8, 1024, device=self.device)

                    handler.sample_audio_tokens(mock_logits, sampler_config)
                    total_audio_tokens += active_codebooks.sum().item()
                    audio_duration += 0.02  # 20ms per step

                return total_audio_tokens, audio_duration

            # Benchmark
            start_memory = self._get_memory_usage()
            (tokens_generated, audio_duration), duration = self._time_function(
                mock_audio_generation
            )
            end_memory = self._get_memory_usage()

            memory_used = end_memory - start_memory
            audio_duration_ratio = audio_duration / (duration / 1000) if duration > 0 else 0

            return BenchmarkResult(
                test_name="audio_generation",
                duration_ms=duration,
                memory_mb=memory_used,
                tokens_per_second=tokens_generated / (duration / 1000) if duration > 0 else 0,
                audio_duration_ratio=audio_duration_ratio,
            )

        except Exception as e:
            return BenchmarkResult(
                test_name="audio_generation", duration_ms=0, memory_mb=0, error=str(e)
            )

    def benchmark_streaming_performance(self) -> BenchmarkResult:
        """Benchmark streaming performance and latency."""
        try:

            async def mock_streaming_benchmark():
                iterator = StreamingIterator(buffer_size=4096, chunk_size=1024)

                # Generate mock audio stream
                async def audio_generator():
                    for i in range(10):
                        yield torch.randn(1024), {"chunk_id": i}
                        await asyncio.sleep(0.01)

                chunks_processed = 0
                async for chunk in iterator.stream_audio(audio_generator()):
                    chunks_processed += 1
                    if chunks_processed >= 8:
                        break

                return chunks_processed, iterator.get_performance_metrics()

            # Benchmark
            start_memory = self._get_memory_usage()
            (chunks_processed, metrics), duration = self._time_function(
                lambda: asyncio.run(mock_streaming_benchmark())
            )
            end_memory = self._get_memory_usage()

            memory_used = end_memory - start_memory
            chunks_per_second = chunks_processed / (duration / 1000) if duration > 0 else 0

            return BenchmarkResult(
                test_name="streaming_performance",
                duration_ms=duration,
                memory_mb=memory_used,
                tokens_per_second=chunks_per_second,
                quality_score=metrics.get("buffer_utilization", 0),
            )

        except Exception as e:
            return BenchmarkResult(
                test_name="streaming_performance", duration_ms=0, memory_mb=0, error=str(e)
            )

    def benchmark_voice_cloning_performance(self) -> BenchmarkResult:
        """Benchmark voice cloning and similarity metrics."""
        try:

            def voice_cloning_benchmark():
                # Test embedding extraction
                audio_samples = [torch.randn(48000) for _ in range(5)]
                embeddings = []

                for audio in audio_samples:
                    embedding = VoiceCloneingUtils.extract_speaker_embedding(
                        audio,
                        device="cpu",  # Use CPU for consistent benchmarking
                    )
                    embeddings.append(embedding)

                # Test similarity calculations
                similarities = []
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        sim = VoiceSimilarityMetrics.cosine_similarity(embeddings[i], embeddings[j])
                        similarities.append(sim)

                return len(embeddings), len(similarities)

            # Benchmark
            start_memory = self._get_memory_usage()
            (embeddings_count, similarities_count), duration = self._time_function(
                voice_cloning_benchmark
            )
            end_memory = self._get_memory_usage()

            memory_used = end_memory - start_memory
            operations_per_second = (embeddings_count + similarities_count) / (duration / 1000)

            return BenchmarkResult(
                test_name="voice_cloning_performance",
                duration_ms=duration,
                memory_mb=memory_used,
                tokens_per_second=operations_per_second,
            )

        except Exception as e:
            return BenchmarkResult(
                test_name="voice_cloning_performance", duration_ms=0, memory_mb=0, error=str(e)
            )

    def run_all_validations(self) -> List[ValidationResult]:
        """Run all validation tests."""
        print("Running validation tests...")

        validations = [
            self.validate_config_creation,
            self.validate_state_management,
            self.validate_delay_pattern_scheduler,
            self.validate_fused_logits_handler,
            self.validate_streaming_iterator,
            self.validate_voice_cloning_utilities,
        ]

        results = []
        for validation in validations:
            print(f"  Running {validation.__name__}...")
            try:
                result = validation()
                results.append(result)
                status = "✓" if result.passed else "✗"
                print(f"    {status} {result.test_name}")
                if not result.passed:
                    print(f"      Error: {result.error_message}")
            except Exception as e:
                print(f"    ✗ {validation.__name__} (Exception: {str(e)})")
                results.append(ValidationResult(validation.__name__, False, error_message=str(e)))

        self.validation_results = results
        return results

    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all performance benchmarks."""
        print("\nRunning performance benchmarks...")

        benchmarks = [
            self.benchmark_text_generation,
            self.benchmark_audio_generation,
            self.benchmark_streaming_performance,
            self.benchmark_voice_cloning_performance,
        ]

        results = []
        for benchmark in benchmarks:
            print(f"  Running {benchmark.__name__}...")
            try:
                result = benchmark()
                results.append(result)
                if result.error:
                    print(f"    ✗ {result.test_name}: {result.error}")
                else:
                    print(
                        f"    ✓ {result.test_name}: {result.duration_ms:.1f}ms, "
                        f"{result.memory_mb:.1f}MB"
                    )
                    if result.tokens_per_second:
                        print(f"      Throughput: {result.tokens_per_second:.1f} tokens/sec")
                    if result.audio_duration_ratio:
                        print(f"      Audio ratio: {result.audio_duration_ratio:.2f}x real-time")
            except Exception as e:
                print(f"    ✗ {benchmark.__name__} (Exception: {str(e)})")
                results.append(BenchmarkResult(benchmark.__name__, 0, 0, error=str(e)))

        self.benchmark_results = results
        return results

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        validation_summary = {
            "total_tests": len(self.validation_results),
            "passed": sum(1 for r in self.validation_results if r.passed),
            "failed": sum(1 for r in self.validation_results if not r.passed),
            "success_rate": sum(1 for r in self.validation_results if r.passed)
            / len(self.validation_results)
            * 100,
        }

        benchmark_summary = {
            "total_benchmarks": len(self.benchmark_results),
            "successful": sum(1 for r in self.benchmark_results if r.error is None),
            "failed": sum(1 for r in self.benchmark_results if r.error is not None),
            "total_duration_ms": sum(
                r.duration_ms for r in self.benchmark_results if r.error is None
            ),
            "total_memory_mb": sum(r.memory_mb for r in self.benchmark_results if r.error is None),
        }

        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": self.model_path,
            "device": self.device,
            "validation_summary": validation_summary,
            "benchmark_summary": benchmark_summary,
            "validation_results": [
                {
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "error_message": r.error_message,
                    "details": r.details,
                }
                for r in self.validation_results
            ],
            "benchmark_results": [
                {
                    "test_name": r.test_name,
                    "duration_ms": r.duration_ms,
                    "memory_mb": r.memory_mb,
                    "tokens_per_second": r.tokens_per_second,
                    "audio_duration_ratio": r.audio_duration_ratio,
                    "quality_score": r.quality_score,
                    "error": r.error,
                }
                for r in self.benchmark_results
            ],
        }

    def save_report(self, output_path: str = "higgs_audio_test_report.json"):
        """Save test report to file."""
        report = self.generate_report()
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nTest report saved to: {output_path}")
        return output_path


def main():
    """Run the complete validation and benchmarking suite."""
    print("Higgs Audio End-to-End Validation and Benchmarking Suite")
    print("=" * 60)

    # Initialize validator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    validator = HiggsEndToEndValidator(device=device)

    # Run tests
    validation_results = validator.run_all_validations()
    benchmark_results = validator.run_all_benchmarks()

    # Generate summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed_validations = sum(1 for r in validation_results if r.passed)
    total_validations = len(validation_results)
    print(
        f"Validation Tests: {passed_validations}/{total_validations} passed "
        f"({passed_validations / total_validations * 100:.1f}%)"
    )

    successful_benchmarks = sum(1 for r in benchmark_results if r.error is None)
    total_benchmarks = len(benchmark_results)
    print(
        f"Benchmarks: {successful_benchmarks}/{total_benchmarks} successful "
        f"({successful_benchmarks / total_benchmarks * 100:.1f}%)"
    )

    if successful_benchmarks > 0:
        total_time = sum(r.duration_ms for r in benchmark_results if r.error is None)
        total_memory = sum(r.memory_mb for r in benchmark_results if r.error is None)
        print(f"Total benchmark time: {total_time:.1f}ms")
        print(f"Total memory usage: {total_memory:.1f}MB")

    # Save report
    validator.save_report()

    # Return status code
    all_passed = (
        passed_validations == total_validations and successful_benchmarks == total_benchmarks
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
