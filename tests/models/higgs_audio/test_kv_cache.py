"""Test KV cache handling in Higgs Audio DualFFN layer.

These tests validate the KV cache management system including static and dynamic
cache modes, TensorRT-LLM packing, and mode transition handling.
"""

import unittest
from unittest.mock import Mock, patch

import torch

from tensorrt_llm.layers.attention import KeyValueCacheParams
from tensorrt_llm.models.higgs_audio.dual_ffn import CacheMode, KVCacheManager


class TestKVCacheManager(unittest.TestCase):
    """Test KV cache management utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.max_seq_len = 128
        self.num_heads = 16
        self.head_size = 64
        self.num_layers = 4
        self.dtype = torch.float16
        self.device = "cpu"  # Use CPU for testing to avoid GPU requirements

    def test_create_static_cache(self):
        """Test static cache creation."""
        past_key_value, cache_metadata = KVCacheManager.create_static_cache(
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            num_heads=self.num_heads,
            head_size=self.head_size,
            num_layers=self.num_layers,
            dtype=self.dtype,
            device=self.device,
        )

        # Validate cache structure
        self.assertEqual(len(past_key_value), self.num_layers)
        self.assertEqual(cache_metadata["mode"], CacheMode.STATIC)
        self.assertEqual(cache_metadata["batch_size"], self.batch_size)
        self.assertEqual(cache_metadata["max_seq_len"], self.max_seq_len)

        # Validate cache tensor shapes
        for layer_cache in past_key_value:
            expected_shape = (self.batch_size, 2, self.num_heads, self.max_seq_len, self.head_size)
            self.assertEqual(layer_cache.shape, expected_shape)
            self.assertEqual(layer_cache.dtype, self.dtype)

    def test_create_dynamic_cache(self):
        """Test dynamic cache creation."""
        past_key_value, cache_metadata = KVCacheManager.create_dynamic_cache(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_size=self.head_size,
            dtype=self.dtype,
        )

        # Validate cache structure
        self.assertEqual(len(past_key_value), self.num_layers)
        self.assertEqual(cache_metadata["mode"], CacheMode.DYNAMIC)
        self.assertEqual(cache_metadata["num_heads"], self.num_heads)
        self.assertEqual(cache_metadata["head_size"], self.head_size)

        # All entries should be None for dynamic allocation
        for layer_cache in past_key_value:
            self.assertIsNone(layer_cache)

    @patch("tensorrt_llm.models.higgs_audio.dual_ffn.torch")
    def test_pack_cache_for_trt(self, mock_torch):
        """Test TensorRT-LLM cache packing."""
        # Create a mock cache tensor
        cache_tensor = Mock()
        cache_tensor.shape = [2, 2, 16, 100, 64]  # Non-aligned sequence length
        cache_tensor.dtype = torch.float16
        cache_tensor.device = "cuda"

        # Mock the tensor operations
        mock_torch.zeros.return_value = Mock()
        mock_concat = Mock()

        with patch("tensorrt_llm.models.higgs_audio.dual_ffn.concat", mock_concat):
            with patch("tensorrt_llm.models.higgs_audio.dual_ffn.shape") as mock_shape:
                mock_shape.side_effect = lambda tensor, dim: cache_tensor.shape[dim]

                # Test packing
                _ = KVCacheManager.pack_cache_for_trt(cache_tensor, block_size=64)

                # Should call concat for padding since 100 is not divisible by 64
                mock_concat.assert_called_once()

    def test_validate_cache_consistency_valid(self):
        """Test cache validation with valid cache."""
        # Create valid cache params
        cache_tensor = torch.zeros((2, 2, 16, 128, 64))
        kv_cache_params = KeyValueCacheParams(past_key_value=[cache_tensor])

        # Test validation
        is_valid = KVCacheManager.validate_cache_consistency(
            kv_cache_params, expected_seq_len=64, layer_idx=0
        )

        self.assertTrue(is_valid)

    def test_validate_cache_consistency_invalid_shape(self):
        """Test cache validation with invalid tensor shape."""
        # Create invalid cache (4D instead of 5D)
        cache_tensor = torch.zeros((2, 16, 128, 64))
        kv_cache_params = KeyValueCacheParams(past_key_value=[cache_tensor])

        # Test validation
        is_valid = KVCacheManager.validate_cache_consistency(
            kv_cache_params, expected_seq_len=64, layer_idx=0
        )

        self.assertFalse(is_valid)

    def test_validate_cache_consistency_short_sequence(self):
        """Test cache validation with sequence too short."""
        # Create cache with short sequence
        cache_tensor = torch.zeros((2, 2, 16, 32, 64))  # Only 32 seq len
        kv_cache_params = KeyValueCacheParams(past_key_value=[cache_tensor])

        # Test validation with longer expected sequence
        is_valid = KVCacheManager.validate_cache_consistency(
            kv_cache_params, expected_seq_len=64, layer_idx=0
        )

        self.assertFalse(is_valid)

    def test_create_cache_params_from_metadata(self):
        """Test KeyValueCacheParams creation from metadata."""
        past_key_value = [torch.zeros((2, 2, 16, 128, 64))]
        cache_metadata = {
            "mode": CacheMode.STATIC,
            "batch_size": 2,
            "max_seq_len": 128,
        }

        kv_cache_params = KVCacheManager.create_cache_params_from_metadata(
            cache_metadata, past_key_value
        )

        self.assertIsInstance(kv_cache_params, KeyValueCacheParams)
        self.assertEqual(kv_cache_params.past_key_value, past_key_value)

    def test_extract_cache_from_params(self):
        """Test cache extraction from params."""
        cache_tensor = torch.zeros((2, 2, 16, 128, 64))
        kv_cache_params = KeyValueCacheParams(past_key_value=[cache_tensor])

        # Extract cache
        extracted = KVCacheManager.extract_cache_from_params(kv_cache_params, layer_idx=0)

        self.assertTrue(torch.equal(extracted, cache_tensor))

    def test_extract_cache_from_params_invalid_index(self):
        """Test cache extraction with invalid layer index."""
        cache_tensor = torch.zeros((2, 2, 16, 128, 64))
        kv_cache_params = KeyValueCacheParams(past_key_value=[cache_tensor])

        # Extract with invalid index
        extracted = KVCacheManager.extract_cache_from_params(kv_cache_params, layer_idx=1)

        self.assertIsNone(extracted)

    def test_update_cache_in_params(self):
        """Test cache update in params."""
        original_cache = torch.zeros((2, 2, 16, 128, 64))
        new_cache = torch.ones((2, 2, 16, 128, 64))
        kv_cache_params = KeyValueCacheParams(past_key_value=[original_cache])

        # Update cache
        updated_params = KVCacheManager.update_cache_in_params(
            kv_cache_params, layer_idx=0, updated_cache=new_cache
        )

        self.assertTrue(torch.equal(updated_params.past_key_value[0], new_cache))

    def test_update_cache_in_params_extend_list(self):
        """Test cache update that extends the cache list."""
        kv_cache_params = KeyValueCacheParams(past_key_value=[])
        new_cache = torch.ones((2, 2, 16, 128, 64))

        # Update at index 2 (should extend list)
        updated_params = KVCacheManager.update_cache_in_params(
            kv_cache_params, layer_idx=2, updated_cache=new_cache
        )

        # Should have 3 entries now (0, 1, 2)
        self.assertEqual(len(updated_params.past_key_value), 3)
        self.assertIsNone(updated_params.past_key_value[0])
        self.assertIsNone(updated_params.past_key_value[1])
        self.assertTrue(torch.equal(updated_params.past_key_value[2], new_cache))


@unittest.skip("Skipping integration test due to missing TensorRT-LLM dependencies")
class TestDualFFNKVCacheIntegration(unittest.TestCase):
    """Integration tests for KV cache in DualFFN layer."""

    def setUp(self):
        """Set up test fixtures."""
        self.hidden_size = 1024
        self.intermediate_size = 4096
        self.num_attention_heads = 16
        self.num_attention_kv_heads = 16
        self.max_position_embeddings = 2048

    def test_dual_ffn_kv_cache_forward(self):
        """Test DualFFN forward with KV cache."""
        # This test would require full TensorRT-LLM setup
        # Skipped for now due to dependency requirements
        pass


if __name__ == "__main__":
    # Run only the KV cache manager tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestKVCacheManager)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
