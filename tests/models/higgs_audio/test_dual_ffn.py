# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for HiggsAudioDualFFNDecoderLayer."""

import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

# Add the tensorrt_llm models path for direct import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Mock TensorRT-LLM components to avoid circular imports
sys.modules["tensorrt_llm.functional"] = MagicMock()
sys.modules["tensorrt_llm.layers"] = MagicMock()
sys.modules["tensorrt_llm.module"] = MagicMock()

# Import after setting up mocks
from tensorrt_llm.models.higgs_audio.dual_ffn import (
    GenerationMode,
    compute_dual_ffn_attention_mask,
    create_audio_out_mask_from_token_types,
)


class MockConfig:
    """Mock configuration for testing."""

    def __init__(self):
        self.text_config = Mock()
        self.text_config.hidden_size = 3072
        self.text_config.num_attention_heads = 24
        self.text_config.num_key_value_heads = 24
        self.text_config.intermediate_size = 8192
        self.text_config.max_position_embeddings = 8192
        self.text_config.attention_bias = False
        self.text_config.rope_theta = 10000.0
        self.text_config.rope_scaling = None
        self.text_config.rms_norm_eps = 1e-6
        self.text_config.hidden_act = "silu"
        self.text_config.mlp_bias = False

        # Audio-specific config
        self.audio_dual_ffn_layers = None
        self.audio_intermediate_size = 4096
        self.use_audio_out_self_attention = False


@pytest.fixture
def config():
    """Fixture providing mock configuration."""
    return MockConfig()


@pytest.fixture
def layer_config(config):
    """Fixture providing layer with configuration."""
    return {"config": config, "layer_idx": 0, "dtype": "float16"}


class TestGenerationMode:
    """Test GenerationMode enum."""

    def test_generation_mode_values(self):
        """Test that GenerationMode has correct values."""
        assert GenerationMode.TEXT.value == 0
        assert GenerationMode.AUDIO_INIT.value == 1
        assert GenerationMode.AUDIO_IN_PROGRESS.value == 2


class TestHiggsAudioDualFFNDecoderLayer:
    """Test HiggsAudioDualFFNDecoderLayer."""

    @patch("tensorrt_llm.models.higgs_audio.dual_ffn.Attention")
    @patch("tensorrt_llm.models.higgs_audio.dual_ffn.RmsNorm")
    @patch("tensorrt_llm.models.higgs_audio.dual_ffn.MLP")
    def test_layer_initialization(self, mock_mlp, mock_rmsnorm, mock_attention, layer_config):
        """Test layer initialization with default configuration."""
        layer = HiggsAudioDualFFNDecoderLayer(**layer_config)

        # Check basic attributes
        assert layer.layer_idx == 0
        assert layer.hidden_size == 3072
        assert layer.num_attention_heads == 24
        assert layer.num_kv_heads == 24
        assert layer.intermediate_size == 8192
        assert layer.dtype == "float16"

        # Check dual FFN is enabled by default
        assert layer.use_dual_ffn is True
        assert layer.fast_forward is False

        # Check components are created
        mock_attention.assert_called()
        assert mock_rmsnorm.call_count >= 2  # At least input and post attention norms
        assert mock_mlp.call_count >= 1  # At least text MLP

    @patch("tensorrt_llm.models.higgs_audio.dual_ffn.Attention")
    @patch("tensorrt_llm.models.higgs_audio.dual_ffn.RmsNorm")
    @patch("tensorrt_llm.models.higgs_audio.dual_ffn.MLP")
    def test_layer_initialization_with_specific_layers(
        self, mock_mlp, mock_rmsnorm, mock_attention, config
    ):
        """Test layer initialization with specific dual FFN layers."""
        config.audio_dual_ffn_layers = [0, 2, 4]  # Only certain layers use dual FFN

        # Test layer that should use dual FFN
        layer = HiggsAudioDualFFNDecoderLayer(config, layer_idx=0)
        assert layer.use_dual_ffn is True
        assert layer.fast_forward is False

        # Test layer that should use fast forward
        layer = HiggsAudioDualFFNDecoderLayer(config, layer_idx=1)
        assert layer.use_dual_ffn is False
        assert layer.fast_forward is True

    @patch("tensorrt_llm.models.higgs_audio.dual_ffn.Attention")
    @patch("tensorrt_llm.models.higgs_audio.dual_ffn.RmsNorm")
    @patch("tensorrt_llm.models.higgs_audio.dual_ffn.MLP")
    def test_layer_initialization_with_audio_attention(
        self, mock_mlp, mock_rmsnorm, mock_attention, config
    ):
        """Test layer initialization with audio-specific attention."""
        config.use_audio_out_self_attention = True

        layer = HiggsAudioDualFFNDecoderLayer(config, layer_idx=0)
        assert layer.use_audio_attention is True

        # Should create additional attention layer for audio
        assert mock_attention.call_count >= 2  # Self attn + audio self attn

    def test_create_audio_out_mask(self, layer_config):
        """Test audio token mask creation."""
        with (
            patch("tensorrt_llm.models.higgs_audio.dual_ffn.Attention"),
            patch("tensorrt_llm.models.higgs_audio.dual_ffn.RmsNorm"),
            patch("tensorrt_llm.models.higgs_audio.dual_ffn.MLP"),
        ):
            layer = HiggsAudioDualFFNDecoderLayer(**layer_config)

            # Test with no audio token IDs
            input_ids = torch.tensor([[1, 2, 3, 4, 5]])
            mask = layer.create_audio_out_mask(input_ids, audio_token_ids=None)
            assert mask is None

            # Test with audio token IDs
            audio_token_ids = [2, 4]
            mask = layer.create_audio_out_mask(input_ids, audio_token_ids=audio_token_ids)
            expected = torch.tensor([[False, True, False, True, False]])
            assert torch.equal(mask, expected)

    def test_apply_generation_mode_masking(self, layer_config):
        """Test generation mode masking."""
        with (
            patch("tensorrt_llm.models.higgs_audio.dual_ffn.Attention"),
            patch("tensorrt_llm.models.higgs_audio.dual_ffn.RmsNorm"),
            patch("tensorrt_llm.models.higgs_audio.dual_ffn.MLP"),
        ):
            layer = HiggsAudioDualFFNDecoderLayer(**layer_config)
            hidden_states = torch.randn(1, 5, 3072)

            # Test TEXT mode - should return unchanged
            result = layer.apply_generation_mode_masking(
                hidden_states, GenerationMode.TEXT, audio_out_mask=None
            )
            assert torch.equal(result, hidden_states)

            # Test with audio mask but TEXT mode
            audio_mask = torch.tensor([[False, True, False, True, False]])
            result = layer.apply_generation_mode_masking(
                hidden_states, GenerationMode.TEXT, audio_out_mask=audio_mask
            )
            assert torch.equal(result, hidden_states)

            # Test AUDIO_INIT mode
            result = layer.apply_generation_mode_masking(
                hidden_states, GenerationMode.AUDIO_INIT, audio_out_mask=audio_mask
            )
            assert torch.equal(result, hidden_states)  # For now, no specific masking


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_audio_out_mask_from_token_types(self):
        """Test creating audio mask from token type IDs."""
        token_type_ids = torch.tensor([[0, 1, 0, 1, 0]])  # 0=text, 1=audio

        mask = create_audio_out_mask_from_token_types(token_type_ids)
        expected = torch.tensor([[False, True, False, True, False]])
        assert torch.equal(mask, expected)

        # Test with different audio token type
        mask = create_audio_out_mask_from_token_types(token_type_ids, audio_token_type=0)
        expected = torch.tensor([[True, False, True, False, True]])
        assert torch.equal(mask, expected)

    def test_compute_dual_ffn_attention_mask(self):
        """Test dual FFN attention mask computation."""
        attention_mask = torch.tril(torch.ones(5, 5))  # Causal mask

        # Test TEXT mode - should return original mask
        result = compute_dual_ffn_attention_mask(attention_mask, mode=GenerationMode.TEXT)
        assert torch.equal(result, attention_mask)

        # Test with audio mask but no special processing yet
        audio_mask = torch.tensor([False, True, False, True, False])
        result = compute_dual_ffn_attention_mask(
            attention_mask, audio_out_mask=audio_mask, mode=GenerationMode.AUDIO_INIT
        )
        assert torch.equal(result, attention_mask)  # No modification yet


class TestDualFFNForwardPass:
    """Test forward pass functionality."""

    @patch("tensorrt_llm.models.higgs_audio.dual_ffn.Attention")
    @patch("tensorrt_llm.models.higgs_audio.dual_ffn.RmsNorm")
    @patch("tensorrt_llm.models.higgs_audio.dual_ffn.MLP")
    def test_forward_text_mode_only(self, mock_mlp, mock_rmsnorm, mock_attention, layer_config):
        """Test forward pass in text-only mode."""
        # Setup mocks
        mock_attention_instance = Mock()
        mock_attention_instance.return_value = torch.randn(1, 5, 3072)
        mock_attention.return_value = mock_attention_instance

        mock_norm_instance = Mock()
        mock_norm_instance.return_value = torch.randn(1, 5, 3072)
        mock_rmsnorm.return_value = mock_norm_instance

        mock_mlp_instance = Mock()
        mock_mlp_instance.return_value = torch.randn(1, 5, 3072)
        mock_mlp.return_value = mock_mlp_instance

        layer = HiggsAudioDualFFNDecoderLayer(**layer_config)

        # Test forward pass
        hidden_states = torch.randn(1, 5, 3072)
        result = layer.forward(hidden_states, mode=GenerationMode.TEXT)

        # Should call attention and normalization
        mock_attention_instance.assert_called_once()
        assert mock_norm_instance.call_count >= 1

        # Should have correct output shape
        assert result.shape == (1, 5, 3072)

    @patch("tensorrt_llm.models.higgs_audio.dual_ffn.Attention")
    @patch("tensorrt_llm.models.higgs_audio.dual_ffn.RmsNorm")
    @patch("tensorrt_llm.models.higgs_audio.dual_ffn.MLP")
    def test_forward_with_audio_mask(self, mock_mlp, mock_rmsnorm, mock_attention, layer_config):
        """Test forward pass with audio token mask."""
        # Setup mocks
        mock_attention_instance = Mock()
        mock_attention_instance.return_value = torch.randn(1, 5, 3072)
        mock_attention.return_value = mock_attention_instance

        mock_norm_instance = Mock()
        mock_norm_instance.return_value = torch.randn(1, 5, 3072)
        mock_rmsnorm.return_value = mock_norm_instance

        mock_mlp_instance = Mock()
        mock_mlp_instance.return_value = torch.randn(1, 5, 3072)
        mock_mlp.return_value = mock_mlp_instance

        layer = HiggsAudioDualFFNDecoderLayer(**layer_config)

        # Test forward pass with audio mask
        hidden_states = torch.randn(1, 5, 3072)
        audio_mask = torch.tensor([[False, True, False, True, False]])

        result = layer.forward(
            hidden_states, mode=GenerationMode.AUDIO_IN_PROGRESS, audio_out_mask=audio_mask
        )

        # Should have correct output shape
        assert result.shape == (1, 5, 3072)

        # Should call both text and audio MLPs due to dual FFN processing
        assert mock_mlp_instance.call_count >= 2  # Text and audio MLPs


class TestIntegration:
    """Integration tests for the dual FFN layer."""

    def test_layer_compatibility_with_tensorrt_llm(self):
        """Test that the layer is compatible with TensorRT-LLM patterns."""
        config = MockConfig()

        # Should be able to create layer without errors
        with (
            patch("tensorrt_llm.models.higgs_audio.dual_ffn.Attention"),
            patch("tensorrt_llm.models.higgs_audio.dual_ffn.RmsNorm"),
            patch("tensorrt_llm.models.higgs_audio.dual_ffn.MLP"),
        ):
            layer = HiggsAudioDualFFNDecoderLayer(config, layer_idx=0)

            # Should have expected attributes for TensorRT-LLM compatibility
            assert hasattr(layer, "hidden_size")
            assert hasattr(layer, "num_attention_heads")
            assert hasattr(layer, "dtype")
            assert hasattr(layer, "layer_idx")

    def test_memory_efficiency_features(self):
        """Test memory efficiency features."""
        config = MockConfig()
        config.audio_intermediate_size = 2048  # Smaller than text FFN

        with (
            patch("tensorrt_llm.models.higgs_audio.dual_ffn.Attention"),
            patch("tensorrt_llm.models.higgs_audio.dual_ffn.RmsNorm"),
            patch("tensorrt_llm.models.higgs_audio.dual_ffn.MLP") as mock_mlp,
        ):
            layer = HiggsAudioDualFFNDecoderLayer(config, layer_idx=0)

            # Check that audio MLP was created with smaller size
            mlp_calls = mock_mlp.call_args_list
            audio_mlp_call = None
            for call in mlp_calls:
                if call.kwargs.get("ffn_hidden_size") == 2048:
                    audio_mlp_call = call
                    break

            assert audio_mlp_call is not None, "Audio MLP should be created with smaller size"


if __name__ == "__main__":
    pytest.main([__file__])
