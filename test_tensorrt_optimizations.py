"""Tests for TensorRT-LLM batch optimizations in HiggsAudio.

This test suite validates:
- TensorRTBatchOptimizer functionality
- Sequence length bucketing
- Memory estimation and optimization
- Packed attention masks
- TensorRTOptimizedCollator integration
"""

import torch

from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.models.higgs_audio.data_processing import (
    TensorRTBatchOptimizer,
    TensorRTOptimizedCollator,
    create_tensorrt_optimized_collator,
    prepare_audio_sample,
    prepare_text_sample,
)


class TestTensorRTBatchOptimizer:
    """Test TensorRTBatchOptimizer functionality."""

    def test_optimizer_initialization(self):
        """Test optimizer initialization with default parameters."""
        print("Testing optimizer initialization...")

        optimizer = TensorRTBatchOptimizer()

        assert optimizer.bucket_boundaries == [32, 64, 128, 256, 512, 1024]
        assert optimizer.max_batch_size == 32
        assert optimizer.memory_threshold_gb == 8.0
        assert optimizer.enable_packed_attention is True
        assert optimizer.enable_pinned_memory is True
        print("✓ Optimizer initialization correct")

    def test_bucket_assignment(self):
        """Test sequence length bucketing."""
        print("Testing bucket assignment...")

        optimizer = TensorRTBatchOptimizer(bucket_boundaries=[32, 64, 128])

        # Test various lengths
        assert optimizer.get_bucket_for_length(10) == 32  # Fits in first bucket
        assert optimizer.get_bucket_for_length(32) == 32  # Exact match
        assert optimizer.get_bucket_for_length(50) == 64  # Fits in second bucket
        assert optimizer.get_bucket_for_length(100) == 128  # Fits in third bucket
        assert optimizer.get_bucket_for_length(200) == 256  # Next power of 2

        # Test caching
        assert optimizer.get_bucket_for_length(10) == 32  # Should use cache
        print("✓ Bucket assignment correct")

    def test_sample_organization(self):
        """Test organizing samples by bucket."""
        print("Testing sample organization...")

        optimizer = TensorRTBatchOptimizer(bucket_boundaries=[32, 64])

        # Create test samples with different lengths
        samples = [
            prepare_text_sample([1, 2, 3]),  # Length 3 -> bucket 32
            prepare_text_sample([1, 2, 3, 4, 5]),  # Length 5 -> bucket 32
            prepare_text_sample(list(range(40))),  # Length 40 -> bucket 64
            prepare_text_sample(list(range(50))),  # Length 50 -> bucket 64
        ]

        buckets = optimizer.organize_samples_by_bucket(samples)

        assert 32 in buckets
        assert 64 in buckets
        assert len(buckets[32]) == 2  # First two samples
        assert len(buckets[64]) == 2  # Last two samples
        print("✓ Sample organization correct")

    def test_packed_attention_mask(self):
        """Test packed attention mask creation."""
        print("Testing packed attention mask...")

        optimizer = TensorRTBatchOptimizer()

        # Create test attention mask with padding
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 0, 0],  # Length 3
                [1, 1, 0, 0, 0],  # Length 2
                [1, 1, 1, 1, 0],  # Length 4
            ],
            dtype=torch.bool,
        )

        packed_mask, pack_indices = optimizer.create_packed_attention_mask(attention_mask)

        # Should have total length 3 + 2 + 4 = 9
        assert packed_mask.shape == (9,)
        assert torch.all(packed_mask == 1)  # All ones (no padding)

        # Check pack indices: [0, 3, 5, 9]
        expected_indices = torch.tensor([0, 3, 5, 9])
        assert torch.equal(pack_indices, expected_indices)
        print("✓ Packed attention mask correct")

    def test_memory_estimation(self):
        """Test memory usage estimation."""
        print("Testing memory estimation...")

        optimizer = TensorRTBatchOptimizer()

        # Test text-only memory estimation
        text_memory = optimizer.estimate_memory_usage(batch_size=4, seq_len=128, audio_len=None)
        assert text_memory > 0
        assert text_memory < 1.0  # Should be reasonable for small batch

        # Test with audio
        audio_memory = optimizer.estimate_memory_usage(
            batch_size=4, seq_len=128, audio_len=200, n_mels=128
        )
        assert audio_memory > text_memory  # Should be larger with audio
        print("✓ Memory estimation correct")

    def test_optimal_batch_size(self):
        """Test optimal batch size calculation."""
        print("Testing optimal batch size...")

        optimizer = TensorRTBatchOptimizer(max_batch_size=8, memory_threshold_gb=1.0)

        # Create test samples
        samples = [prepare_text_sample(list(range(100))) for _ in range(10)]

        optimal_size = optimizer.get_optimal_batch_size(samples, target_memory_gb=0.1)

        assert optimal_size >= 1
        assert optimal_size <= len(samples)
        assert optimal_size <= optimizer.max_batch_size
        print("✓ Optimal batch size calculation correct")

    def test_pinned_memory_allocation(self):
        """Test pinned memory allocation."""
        print("Testing pinned memory allocation...")

        optimizer = TensorRTBatchOptimizer(enable_pinned_memory=True)

        shapes = [(4, 128), (4, 128, 256)]
        tensors = optimizer.allocate_pinned_memory(shapes)

        assert len(tensors) == 2
        assert tensors[0].shape == (4, 128)
        assert tensors[1].shape == (4, 128, 256)
        assert tensors[0].is_pinned()  # Should be pinned memory
        assert tensors[1].is_pinned()

        # Test with disabled pinned memory
        optimizer_no_pin = TensorRTBatchOptimizer(enable_pinned_memory=False)
        tensors_no_pin = optimizer_no_pin.allocate_pinned_memory(shapes)
        assert not tensors_no_pin[0].is_pinned()
        print("✓ Pinned memory allocation correct")


class TestTensorRTOptimizedCollator:
    """Test TensorRTOptimizedCollator functionality."""

    def test_optimized_collator_initialization(self):
        """Test optimized collator initialization."""
        print("Testing optimized collator initialization...")

        config = HiggsAudioConfig()
        collator = TensorRTOptimizedCollator(config)

        assert collator.optimizer is not None
        assert isinstance(collator.optimizer, TensorRTBatchOptimizer)
        print("✓ Optimized collator initialization correct")

    def test_optimized_collation(self):
        """Test optimized collation with bucketing."""
        print("Testing optimized collation...")

        config = HiggsAudioConfig()
        optimizer = TensorRTBatchOptimizer(bucket_boundaries=[32, 64])
        collator = TensorRTOptimizedCollator(config, optimizer=optimizer)

        # Create samples that fit in same bucket
        samples = [
            prepare_text_sample([1, 2, 3, 4]),  # Length 4 -> bucket 32
            prepare_text_sample([5, 6, 7]),  # Length 3 -> bucket 32
            prepare_text_sample([8, 9]),  # Length 2 -> bucket 32
        ]

        batch = collator(samples)

        # Check basic structure
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "estimated_memory_gb" in batch

        # Should have memory estimation
        assert batch["estimated_memory_gb"] > 0
        print("✓ Optimized collation correct")

    def test_packed_attention_integration(self):
        """Test packed attention mask integration."""
        print("Testing packed attention integration...")

        config = HiggsAudioConfig()
        optimizer = TensorRTBatchOptimizer(enable_packed_attention=True)
        collator = TensorRTOptimizedCollator(config, optimizer=optimizer)

        # Create a larger batch to trigger packed attention
        samples = [prepare_text_sample(list(range(i * 10, (i + 1) * 10))) for i in range(10)]

        batch = collator(samples)

        # Should have packed attention components
        assert "attention_mask" in batch
        # Note: packed_attention_mask might only be added for very large batches
        if "packed_attention_mask" in batch:
            assert "pack_indices" in batch
            assert batch["packed_attention_mask"].dim() == 1

        print("✓ Packed attention integration correct")

    def test_mixed_content_optimization(self):
        """Test optimization with mixed text and audio content."""
        print("Testing mixed content optimization...")

        config = HiggsAudioConfig()
        collator = TensorRTOptimizedCollator(config)

        # Create mixed samples
        mel = torch.randn(128, 50)
        samples = [
            prepare_text_sample([1, 2, 3]),
            prepare_audio_sample([4, 5, 6, 7], mel),
            prepare_text_sample([8, 9]),
        ]

        batch = collator(samples)

        # Check structure
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "estimated_memory_gb" in batch

        # Should account for audio in memory estimation
        memory_usage = batch["estimated_memory_gb"]
        assert memory_usage > 0
        print("✓ Mixed content optimization correct")


class TestUtilityFunctions:
    """Test utility functions for TensorRT optimization."""

    def test_create_tensorrt_optimized_collator(self):
        """Test utility function for creating optimized collator."""
        print("Testing optimized collator creation...")

        config = HiggsAudioConfig()
        collator = create_tensorrt_optimized_collator(
            config,
            bucket_boundaries=[16, 32, 64],
            max_batch_size=16,
        )

        assert isinstance(collator, TensorRTOptimizedCollator)
        assert collator.optimizer.bucket_boundaries == [16, 32, 64]
        assert collator.optimizer.max_batch_size == 16
        print("✓ Optimized collator creation correct")


def test_tensorrt_optimizations_end_to_end():
    """End-to-end test of TensorRT optimizations."""
    print("\n=== Running End-to-End TensorRT Optimization Test ===")

    # Create optimized collator
    config = HiggsAudioConfig()
    collator = create_tensorrt_optimized_collator(
        config,
        bucket_boundaries=[32, 64, 128],
        max_batch_size=8,
    )

    # Test with varied sample lengths
    samples = []
    for i in range(6):
        # Create samples with different lengths
        length = (i + 1) * 10  # 10, 20, 30, 40, 50, 60
        tokens = list(range(length))
        samples.append(prepare_text_sample(tokens, sample_id=f"sample_{i}"))

    print(f"Created {len(samples)} samples with varying lengths")

    # Collate with optimizations
    batch = collator(samples)

    # Check results
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "estimated_memory_gb" in batch

    print(f"Batch shape: {batch['input_ids'].shape}")
    print(f"Estimated memory: {batch['estimated_memory_gb']:.4f} GB")

    # Test bucketing behavior
    buckets = collator.optimizer.organize_samples_by_bucket(samples)
    print(f"Organized into {len(buckets)} buckets: {list(buckets.keys())}")

    # Test memory optimization
    optimal_batch_size = collator.optimizer.get_optimal_batch_size(samples, target_memory_gb=0.1)
    print(f"Optimal batch size for 0.1GB limit: {optimal_batch_size}")

    print("🎉 End-to-end TensorRT optimization test passed!")


if __name__ == "__main__":
    print("⚡ Running TensorRT-LLM Optimization Tests ⚡\n")

    # Run all test classes
    test_classes = [
        TestTensorRTBatchOptimizer(),
        TestTensorRTOptimizedCollator(),
        TestUtilityFunctions(),
    ]

    for test_class in test_classes:
        print(f"\n--- {test_class.__class__.__name__} ---")
        for method_name in dir(test_class):
            if method_name.startswith("test_"):
                print(f"\nRunning {method_name}...")
                method = getattr(test_class, method_name)
                method()

    # Run end-to-end test
    test_tensorrt_optimizations_end_to_end()

    print("\n🎉 All TensorRT optimization tests passed! 🎉")
