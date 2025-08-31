
"""
Comprehensive tests for Higgs Audio delay pattern functionality.

This test suite validates the delay pattern implementation across different
configurations including various delay strategies, codebook counts, sequence
lengths, and generation modes.
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from unittest.mock import patch, MagicMock

# Import the classes we need to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.models.higgs_audio.model import (
    DelayPatternProvider,
    AudioTokenUtils,
    DelayAwareAttentionUtils,
    GenerationMode,
    DelayPatternError,
    AudioTokenError,
    AttentionError,
    HiggsAudioForCausalLM
)


class MockTensor:
    """Mock tensor class for testing without full TensorRT-LLM dependency."""
    
    def __init__(self, data, dtype=None, device='cpu'):
        if isinstance(data, (list, tuple)):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array([data])
        
        self._shape = self.data.shape
        self.dtype = dtype or 'float32'
        self.device = device
    
    @property
    def shape(self):
        return self._shape
    
    def dim(self):
        return len(self._shape)
    
    def item(self):
        return self.data.item() if self.data.size == 1 else self.data
    
    def numpy(self):
        return self.data
    
    def detach(self):
        return self
    
    def cpu(self):
        return self
    
    def unsqueeze(self, dim):
        new_data = np.expand_dims(self.data, dim)
        return MockTensor(new_data, self.dtype, self.device)
    
    def __getitem__(self, key):
        return MockTensor(self.data[key], self.dtype, self.device)


def create_mock_constant(data):
    """Create a mock constant tensor."""
    return MockTensor(data)


def create_mock_config(
    num_codebooks: int = 4,
    delay_strategy: str = 'linear',
    delay_stride: int = 1,
    max_delay: Optional[int] = None,
    custom_delays: Optional[List[int]] = None
) -> HiggsAudioConfig:
    """Create a mock HiggsAudioConfig for testing."""
    
    # Create minimal mock config
    config = MagicMock(spec=HiggsAudioConfig)
    config.audio_num_codebooks = num_codebooks
    config.audio_delay_pattern_strategy = delay_strategy
    config.audio_delay_pattern_stride = delay_stride
    config.audio_delay_pattern_max_delay = max_delay
    config.audio_delay_pattern_custom_delays = custom_delays
    config.hidden_size = 4096
    config.num_hidden_layers = 32
    config.vocab_size = 32000
    config.pad_token_id = 0
    config.eos_token_id = 2
    config.audio_start_token_id = 128010
    config.audio_end_token_id = 128011
    
    return config


class TestDelayPatternProvider:
    """Test suite for DelayPatternProvider class."""
    
    def test_initialization_valid_params(self):
        """Test DelayPatternProvider initialization with valid parameters."""
        
        # Test linear strategy
        provider = DelayPatternProvider(
            strategy='linear',
            stride=1,
            max_delay=10,
            pad_token_id=0
        )
        assert provider.strategy == 'linear'
        assert provider.stride == 1
        assert provider.max_delay == 10
        assert provider.pad_token_id == 0
        assert not provider.fallback_used
        assert len(provider.get_validation_warnings()) == 0
    
    def test_initialization_invalid_strategy_fallback(self):
        """Test initialization with invalid strategy uses fallback."""
        
        provider = DelayPatternProvider(
            strategy='invalid_strategy',
            stride=1,
            enable_fallback=True
        )
        assert provider.strategy == 'linear'  # Should fallback to linear
        assert provider.fallback_used or len(provider.get_validation_warnings()) > 0
    
    def test_initialization_invalid_strategy_no_fallback(self):
        """Test initialization with invalid strategy without fallback raises error."""
        
        with pytest.raises(DelayPatternError) as exc_info:
            DelayPatternProvider(
                strategy='invalid_strategy',
                enable_fallback=False
            )
        assert "INVALID_STRATEGY" in str(exc_info.value)
    
    def test_custom_delays_validation(self):
        """Test custom delays validation and processing."""
        
        # Valid custom delays
        provider = DelayPatternProvider(
            strategy='custom',
            custom_delays=[0, 1, 2, 3],
            enable_fallback=False
        )
        assert provider.strategy == 'custom'
        assert provider.custom_delays == [0, 1, 2, 3]
        
        # Invalid custom delays with fallback
        provider = DelayPatternProvider(
            strategy='custom',
            custom_delays=[0, -1, 'invalid', 2],
            max_delay=5,
            enable_fallback=True
        )
        assert len(provider.get_validation_warnings()) > 0
    
    def test_generate_delay_pattern_linear(self):
        """Test delay pattern generation with linear strategy."""
        
        with patch('tensorrt_llm.models.higgs_audio.model.constant', side_effect=create_mock_constant):
            provider = DelayPatternProvider(strategy='linear', stride=1)
            
            pattern = provider.generate_delay_pattern(n_codebooks=4, sequence_length=10)
            
            assert pattern.shape == (4, 10)
            pattern_data = pattern.numpy()
            
            # Check linear delays: [0, 1, 2, 3]
            expected = np.array([
                [0] * 10,  # Codebook 0: no delay
                [1] * 10,  # Codebook 1: 1-step delay
                [2] * 10,  # Codebook 2: 2-step delay
                [3] * 10   # Codebook 3: 3-step delay
            ])
            np.testing.assert_array_equal(pattern_data, expected)
    
    def test_generate_delay_pattern_exponential(self):
        """Test delay pattern generation with exponential strategy."""
        
        with patch('tensorrt_llm.models.higgs_audio.model.constant', side_effect=create_mock_constant):
            provider = DelayPatternProvider(strategy='exponential', stride=1)
            
            pattern = provider.generate_delay_pattern(n_codebooks=4, sequence_length=10)
            
            assert pattern.shape == (4, 10)
            pattern_data = pattern.numpy()
            
            # Check exponential delays: [0, 1, 3, 7] = [2^0-1, 2^1-1, 2^2-1, 2^3-1]
            expected = np.array([
                [0] * 10,  # Codebook 0: 2^0-1 = 0
                [1] * 10,  # Codebook 1: 2^1-1 = 1
                [3] * 10,  # Codebook 2: 2^2-1 = 3
                [7] * 10   # Codebook 3: 2^3-1 = 7
            ])
            np.testing.assert_array_equal(pattern_data, expected)
    
    def test_generate_delay_pattern_custom(self):
        """Test delay pattern generation with custom strategy."""
        
        with patch('tensorrt_llm.models.higgs_audio.model.constant', side_effect=create_mock_constant):
            custom_delays = [0, 2, 1, 4]
            provider = DelayPatternProvider(
                strategy='custom',
                custom_delays=custom_delays
            )
            
            pattern = provider.generate_delay_pattern(n_codebooks=4, sequence_length=10)
            
            assert pattern.shape == (4, 10)
            pattern_data = pattern.numpy()
            
            # Check custom delays
            expected = np.array([
                [0] * 10,  # Codebook 0: delay 0
                [2] * 10,  # Codebook 1: delay 2
                [1] * 10,  # Codebook 2: delay 1
                [4] * 10   # Codebook 3: delay 4
            ])
            np.testing.assert_array_equal(pattern_data, expected)
    
    def test_generate_delay_pattern_none(self):
        """Test delay pattern generation with none strategy."""
        
        with patch('tensorrt_llm.models.higgs_audio.model.constant', side_effect=create_mock_constant):
            provider = DelayPatternProvider(strategy='none')
            
            pattern = provider.generate_delay_pattern(n_codebooks=4, sequence_length=10)
            
            assert pattern.shape == (4, 10)
            pattern_data = pattern.numpy()
            
            # All delays should be 0
            expected = np.zeros((4, 10))
            np.testing.assert_array_equal(pattern_data, expected)
    
    def test_generate_delay_pattern_validation_errors(self):
        """Test validation errors in delay pattern generation."""
        
        provider = DelayPatternProvider(strategy='linear')
        
        # Invalid n_codebooks
        with pytest.raises(DelayPatternError) as exc_info:
            provider.generate_delay_pattern(n_codebooks=0, sequence_length=10)
        assert "INVALID_PARAMETERS" in str(exc_info.value)
        
        # Invalid sequence_length
        with pytest.raises(DelayPatternError) as exc_info:
            provider.generate_delay_pattern(n_codebooks=4, sequence_length=-1)
        assert "INVALID_PARAMETERS" in str(exc_info.value)
    
    def test_apply_delay_pattern(self):
        """Test applying delay pattern to token sequences."""
        
        with patch('tensorrt_llm.models.higgs_audio.model.constant', side_effect=create_mock_constant):
            provider = DelayPatternProvider(strategy='linear', stride=1, pad_token_id=0)
            
            # Create test tokens
            tokens = MockTensor([
                [1, 2, 3, 4, 5],  # Codebook 0
                [6, 7, 8, 9, 10], # Codebook 1
                [11, 12, 13, 14, 15], # Codebook 2
                [16, 17, 18, 19, 20]  # Codebook 3
            ])
            
            # Create delay pattern
            pattern = provider.generate_delay_pattern(n_codebooks=4, sequence_length=5)
            
            # Apply delay pattern
            delayed_tokens = provider.apply_delay_pattern(tokens, pattern)
            
            # Check delayed output
            delayed_data = delayed_tokens.numpy()
            expected = np.array([
                [1, 2, 3, 4, 5, 0, 0, 0],      # Codebook 0: no delay
                [0, 6, 7, 8, 9, 10, 0, 0],     # Codebook 1: 1-step delay
                [0, 0, 11, 12, 13, 14, 15, 0], # Codebook 2: 2-step delay
                [0, 0, 0, 16, 17, 18, 19, 20]  # Codebook 3: 3-step delay
            ])
            np.testing.assert_array_equal(delayed_data, expected)
    
    def test_reverse_delay_pattern(self):
        """Test reversing delay pattern to recover original tokens."""
        
        with patch('tensorrt_llm.models.higgs_audio.model.constant', side_effect=create_mock_constant):
            provider = DelayPatternProvider(strategy='linear', stride=1, pad_token_id=0)
            
            # Create delayed tokens
            delayed_tokens = MockTensor([
                [1, 2, 3, 4, 5, 0, 0, 0],      # Codebook 0: no delay
                [0, 6, 7, 8, 9, 10, 0, 0],     # Codebook 1: 1-step delay
                [0, 0, 11, 12, 13, 14, 15, 0], # Codebook 2: 2-step delay
                [0, 0, 0, 16, 17, 18, 19, 20]  # Codebook 3: 3-step delay
            ])
            
            # Create delay pattern
            pattern = provider.generate_delay_pattern(n_codebooks=4, sequence_length=5)
            
            # Reverse delay pattern
            original_tokens = provider.reverse_delay_pattern(
                delayed_tokens, pattern, original_length=5
            )
            
            # Check recovered tokens
            original_data = original_tokens.numpy()
            expected = np.array([
                [1, 2, 3, 4, 5],   # Codebook 0
                [6, 7, 8, 9, 10],  # Codebook 1
                [11, 12, 13, 14, 15], # Codebook 2
                [16, 17, 18, 19, 20]  # Codebook 3
            ])
            np.testing.assert_array_equal(original_data, expected)


class TestAudioTokenUtils:
    """Test suite for AudioTokenUtils class."""
    
    def test_initialization(self):
        """Test AudioTokenUtils initialization."""
        
        utils = AudioTokenUtils(
            num_codebooks=4,
            pad_token_id=0,
            eos_token_id=2,
            audio_start_token_id=128010,
            audio_end_token_id=128011
        )
        
        assert utils.num_codebooks == 4
        assert utils.pad_token_id == 0
        assert utils.eos_token_id == 2
        assert utils.audio_start_token_id == 128010
        assert utils.audio_end_token_id == 128011
    
    def test_split_audio_tokens_by_codebook(self):
        """Test splitting unified audio tokens by codebook."""
        
        with patch('tensorrt_llm.models.higgs_audio.model.constant', side_effect=create_mock_constant):
            utils = AudioTokenUtils(num_codebooks=4)
            
            # Create unified tokens: [batch_size=2, total_seq_len=8]
            # Interleaved: [t0_cb0, t0_cb1, t0_cb2, t0_cb3, t1_cb0, t1_cb1, t1_cb2, t1_cb3]
            unified_tokens = MockTensor([
                [10, 20, 30, 40, 11, 21, 31, 41],  # Batch 0
                [50, 60, 70, 80, 51, 61, 71, 81]   # Batch 1
            ])
            
            codebook_tokens = utils.split_audio_tokens_by_codebook(unified_tokens)
            
            # Should get 4 codebook tensors, each [2, 2] (batch_size, time_steps)
            assert len(codebook_tokens) == 4
            
            # Check codebook 0
            cb0_data = codebook_tokens[0].numpy()
            expected_cb0 = np.array([[10, 11], [50, 51]])
            np.testing.assert_array_equal(cb0_data, expected_cb0)
            
            # Check codebook 1
            cb1_data = codebook_tokens[1].numpy()
            expected_cb1 = np.array([[20, 21], [60, 61]])
            np.testing.assert_array_equal(cb1_data, expected_cb1)
    
    def test_merge_codebook_tokens(self):
        """Test merging per-codebook tokens into unified sequence."""
        
        with patch('tensorrt_llm.models.higgs_audio.model.constant', side_effect=create_mock_constant):
            utils = AudioTokenUtils(num_codebooks=4)
            
            # Create per-codebook tokens
            codebook_tokens = [
                MockTensor([[10, 11], [50, 51]]),  # Codebook 0
                MockTensor([[20, 21], [60, 61]]),  # Codebook 1
                MockTensor([[30, 31], [70, 71]]),  # Codebook 2
                MockTensor([[40, 41], [80, 81]])   # Codebook 3
            ]
            
            unified_tokens = utils.merge_codebook_tokens(codebook_tokens)
            
            # Should get [2, 8] tensor with interleaved tokens
            unified_data = unified_tokens.numpy()
            expected = np.array([
                [10, 20, 30, 40, 11, 21, 31, 41],  # Batch 0
                [50, 60, 70, 80, 51, 61, 71, 81]   # Batch 1
            ])
            np.testing.assert_array_equal(unified_data, expected)
    
    def test_validate_audio_tokens_single_tensor(self):
        """Test audio token validation with single tensor input."""
        
        utils = AudioTokenUtils(num_codebooks=4, pad_token_id=0)
        
        # Valid single tensor (unified format)
        audio_tokens = MockTensor([
            [10, 20, 30, 40, 11, 21, 31, 41],  # 4 codebooks, 2 time steps
            [50, 60, 70, 80, 51, 61, 71, 81]
        ])
        
        # Should pass validation
        result = utils.validate_audio_tokens(
            audio_tokens,
            expected_codebooks=4,
            expected_sequence_length=2
        )
        assert result is True
    
    def test_validate_audio_tokens_list_format(self):
        """Test audio token validation with list format input."""
        
        utils = AudioTokenUtils(num_codebooks=4, pad_token_id=0)
        
        # Valid list format
        audio_tokens = [
            MockTensor([[10, 11], [50, 51]]),  # Codebook 0
            MockTensor([[20, 21], [60, 61]]),  # Codebook 1
            MockTensor([[30, 31], [70, 71]]),  # Codebook 2
            MockTensor([[40, 41], [80, 81]])   # Codebook 3
        ]
        
        # Should pass validation
        result = utils.validate_audio_tokens(
            audio_tokens,
            expected_codebooks=4,
            expected_sequence_length=2
        )
        assert result is True
    
    def test_validate_audio_tokens_errors(self):
        """Test audio token validation error cases."""
        
        utils = AudioTokenUtils(num_codebooks=4, pad_token_id=0)
        
        # None input
        with pytest.raises(AudioTokenError) as exc_info:
            utils.validate_audio_tokens(None)
        assert "INVALID_INPUT" in str(exc_info.value)
        
        # Wrong codebook count
        with pytest.raises(AudioTokenError) as exc_info:
            audio_tokens = [MockTensor([[1, 2]]), MockTensor([[3, 4]])]  # Only 2 codebooks
            utils.validate_audio_tokens(audio_tokens, expected_codebooks=4)
        assert "CODEBOOK_COUNT_MISMATCH" in str(exc_info.value)
        
        # Shape mismatch in list
        with pytest.raises(AudioTokenError) as exc_info:
            audio_tokens = [
                MockTensor([[1, 2], [3, 4]]),      # Shape [2, 2]
                MockTensor([[5, 6, 7], [8, 9, 10]]) # Shape [2, 3] - mismatch!
            ]
            utils.validate_audio_tokens(audio_tokens)
        assert "SHAPE_MISMATCH_IN_LIST" in str(exc_info.value)
    
    def test_get_audio_token_statistics(self):
        """Test getting comprehensive audio token statistics."""
        
        utils = AudioTokenUtils(num_codebooks=2, pad_token_id=0, eos_token_id=2)
        
        # Create test tokens with some padding and special tokens
        audio_tokens = [
            MockTensor([[1, 2, 0, 0], [3, 4, 5, 2]]),  # Codebook 0: with padding and EOS
            MockTensor([[6, 7, 0, 0], [8, 9, 10, 2]])  # Codebook 1: with padding and EOS
        ]
        
        stats = utils.get_audio_token_statistics(audio_tokens)
        
        # Check basic statistics
        assert stats['num_codebooks'] == 2
        assert stats['batch_size'] == 2
        assert stats['sequence_length'] == 4
        assert stats['total_tokens'] == 16  # 2 * 2 * 4
        
        # Check per-codebook statistics
        assert 'codebook_0' in stats['codebook_statistics']
        assert 'codebook_1' in stats['codebook_statistics']
        
        cb0_stats = stats['codebook_statistics']['codebook_0']
        assert cb0_stats['min_token'] == 0  # Pad token
        assert cb0_stats['max_token'] == 5
        assert cb0_stats['pad_token_count'] == 2
        assert cb0_stats['non_pad_tokens'] == 6
        assert cb0_stats['eos_token_count'] == 1
        
        # Check cross-codebook statistics
        assert 'cross_codebook_statistics' in stats
        assert 'synchronization_quality' in stats['cross_codebook_statistics']


class TestDelayAwareAttentionUtils:
    """Test suite for DelayAwareAttentionUtils class."""
    
    def test_initialization(self):
        """Test DelayAwareAttentionUtils initialization."""
        
        utils = DelayAwareAttentionUtils(
            num_codebooks=4,
            causal_attention=True,
            cross_codebook_attention=False,
            max_delay=8
        )
        
        assert utils.num_codebooks == 4
        assert utils.causal_attention is True
        assert utils.cross_codebook_attention is False
        assert utils.max_delay == 8
    
    def test_create_delay_aware_attention_mask(self):
        """Test creating delay-aware attention masks."""
        
        # Mock torch functions
        with patch('torch.tril') as mock_tril, \
             patch('torch.ones') as mock_ones:
            
            # Setup mocks
            mock_ones.return_value = MockTensor(np.ones((5, 5), dtype=bool))
            mock_tril.return_value = MockTensor(np.tril(np.ones((5, 5)), k=0).astype(bool))
            
            utils = DelayAwareAttentionUtils(num_codebooks=3, causal_attention=True)
            
            # Create delay pattern
            delay_pattern = MockTensor([
                [0, 0, 0, 0, 0],  # Codebook 0: no delay
                [1, 1, 1, 1, 1],  # Codebook 1: 1-step delay
                [2, 2, 2, 2, 2]   # Codebook 2: 2-step delay
            ])
            
            # Create attention mask
            mask = utils.create_delay_aware_attention_mask(
                batch_size=2,
                seq_len=5,
                delay_pattern=delay_pattern
            )
            
            # Should return proper mask shape
            assert mask.shape == (2, 5, 5)
    
    def test_create_codebook_routing_mask(self):
        """Test creating codebook routing masks."""
        
        with patch('torch.zeros') as mock_zeros:
            mock_zeros.return_value = MockTensor(np.zeros((2, 5, 3), dtype=bool))
            
            utils = DelayAwareAttentionUtils(num_codebooks=3)
            
            input_ids = MockTensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
            delay_pattern = MockTensor([
                [0, 0, 0, 0, 0],  # Codebook 0: no delay
                [1, 1, 1, 1, 1],  # Codebook 1: 1-step delay
                [2, 2, 2, 2, 2]   # Codebook 2: 2-step delay
            ])
            
            routing_mask = utils.create_codebook_routing_mask(
                input_ids=input_ids,
                delay_pattern=delay_pattern,
                audio_start_position=2
            )
            
            # Should return proper routing mask shape
            assert routing_mask.shape == (2, 5, 3)


class TestIntegrationScenarios:
    """Integration tests for complete delay pattern workflows."""
    
    def test_end_to_end_delay_pattern_workflow(self):
        """Test complete delay pattern workflow from generation to decoding."""
        
        with patch('tensorrt_llm.models.higgs_audio.model.constant', side_effect=create_mock_constant):
            # Setup components
            delay_provider = DelayPatternProvider(strategy='linear', stride=1)
            audio_utils = AudioTokenUtils(num_codebooks=4, pad_token_id=0)
            
            # Generate delay pattern
            pattern = delay_provider.generate_delay_pattern(n_codebooks=4, sequence_length=6)
            
            # Create original tokens
            original_tokens = MockTensor([
                [1, 2, 3, 4, 5, 6],    # Codebook 0
                [7, 8, 9, 10, 11, 12], # Codebook 1
                [13, 14, 15, 16, 17, 18], # Codebook 2
                [19, 20, 21, 22, 23, 24]  # Codebook 3
            ])
            
            # Apply delay pattern
            delayed_tokens = delay_provider.apply_delay_pattern(original_tokens, pattern)
            
            # Validate delayed tokens
            audio_utils.validate_codebook_sequences([delayed_tokens[i] for i in range(4)])
            
            # Reverse delay pattern
            recovered_tokens = delay_provider.reverse_delay_pattern(
                delayed_tokens, pattern, original_length=6
            )
            
            # Verify round-trip consistency
            np.testing.assert_array_equal(
                original_tokens.numpy(),
                recovered_tokens.numpy()
            )
    
    def test_streaming_delay_pattern_coordination(self):
        """Test delay pattern coordination in streaming scenarios."""
        
        with patch('tensorrt_llm.models.higgs_audio.model.constant', side_effect=create_mock_constant):
            delay_provider = DelayPatternProvider(strategy='linear', stride=1)
            
            # Test streaming-compatible delay patterns
            chunk_size = 4
            total_length = 12
            
            for chunk_start in range(0, total_length, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_length)
                chunk_length = chunk_end - chunk_start
                
                # Generate pattern for chunk
                pattern = delay_provider.generate_delay_pattern(
                    n_codebooks=3,
                    sequence_length=chunk_length
                )
                
                # Validate pattern properties
                assert pattern.shape == (3, chunk_length)
                pattern_info = delay_provider.get_pattern_info(pattern)
                assert pattern_info['strategy'] == 'linear'
                assert pattern_info['is_uniform_per_codebook'] is True
    
    def test_error_recovery_scenarios(self):
        """Test error handling and recovery in various failure scenarios."""
        
        # Test DelayPatternProvider fallback
        provider = DelayPatternProvider(
            strategy='invalid',
            custom_delays=[],
            enable_fallback=True
        )
        assert provider.fallback_used or len(provider.get_validation_warnings()) > 0
        
        # Test AudioTokenUtils error handling
        utils = AudioTokenUtils(num_codebooks=4)
        
        with pytest.raises(AudioTokenError):
            utils.validate_audio_tokens("invalid_input")
        
        with pytest.raises(AudioTokenError):
            utils.validate_audio_tokens([])  # Empty list
    
    def test_performance_edge_cases(self):
        """Test performance with edge cases like large codebook counts and sequences."""
        
        with patch('tensorrt_llm.models.higgs_audio.model.constant', side_effect=create_mock_constant):
            # Test with many codebooks
            provider = DelayPatternProvider(
                strategy='linear',
                stride=1,
                enable_fallback=True,
                validate_memory_usage=True
            )
            
            # Should handle reasonable sizes without error
            pattern = provider.generate_delay_pattern(n_codebooks=8, sequence_length=64)
            assert pattern.shape == (8, 64)
            
            # Large sizes should trigger warnings but not fail
            pattern = provider.generate_delay_pattern(n_codebooks=16, sequence_length=128)
            assert pattern.shape == (16, 128)
            
            # Very large sizes with validation should warn
            warnings = provider.get_validation_warnings()
            # Should have warnings about large configurations
    
    def test_different_delay_strategies_consistency(self):
        """Test consistency across different delay strategies."""
        
        with patch('tensorrt_llm.models.higgs_audio.model.constant', side_effect=create_mock_constant):
            n_codebooks = 4
            seq_len = 8
            
            strategies = ['linear', 'exponential', 'none']
            patterns = {}
            
            for strategy in strategies:
                provider = DelayPatternProvider(strategy=strategy, stride=1)
                pattern = provider.generate_delay_pattern(n_codebooks, seq_len)
                patterns[strategy] = pattern
                
                # All patterns should have correct shape
                assert pattern.shape == (n_codebooks, seq_len)
                
                # All patterns should have non-negative delays
                pattern_data = pattern.numpy()
                assert np.all(pattern_data >= 0), f"{strategy} strategy produced negative delays"
                
                # Test pattern info
                pattern_info = provider.get_pattern_info(pattern)
                assert pattern_info['strategy'] == strategy
                assert pattern_info['num_codebooks'] == n_codebooks
                assert pattern_info['sequence_length'] == seq_len
            
            # Compare strategy differences
            linear_data = patterns['linear'].numpy()
            exp_data = patterns['exponential'].numpy()
            none_data = patterns['none'].numpy()
            
            # Linear should have increasing delays per codebook
            for cb in range(n_codebooks):
                assert np.all(linear_data[cb] == cb), f"Linear strategy codebook {cb} should have delay {cb}"
            
            # Exponential should have exponentially increasing delays
            for cb in range(n_codebooks):
                expected_delay = (2 ** cb) - 1 if cb > 0 else 0
                assert np.all(exp_data[cb] == expected_delay), f"Exponential strategy codebook {cb} should have delay {expected_delay}"
            
            # None should have all zeros
            assert np.all(none_data == 0), "None strategy should have all zero delays"


class TestComplexScenarios:
    """Test complex real-world scenarios and edge cases."""
    
    def test_multicodebook_streaming_coordination(self):
        """Test coordination of multiple codebooks in streaming scenarios."""
        
        with patch('tensorrt_llm.models.higgs_audio.model.constant', side_effect=create_mock_constant):
            # Setup streaming scenario with 6 codebooks
            provider = DelayPatternProvider(strategy='exponential', stride=1)
            audio_utils = AudioTokenUtils(num_codebooks=6, pad_token_id=0)
            attention_utils = DelayAwareAttentionUtils(
                num_codebooks=6,
                causal_attention=True,
                cross_codebook_attention=True
            )
            
            # Simulate streaming chunks
            chunk_size = 8
            total_chunks = 4
            
            all_patterns = []
            all_delayed_tokens = []
            
            for chunk_idx in range(total_chunks):
                # Generate pattern for this chunk
                pattern = provider.generate_delay_pattern(
                    n_codebooks=6,
                    sequence_length=chunk_size
                )
                all_patterns.append(pattern)
                
                # Create chunk tokens (simulate generated audio)
                chunk_tokens = MockTensor([
                    [100 + chunk_idx * 10 + i for i in range(chunk_size)]  # Codebook tokens
                    for _ in range(6)
                ])
                
                # Apply delay pattern
                delayed_chunk = provider.apply_delay_pattern(chunk_tokens, pattern)
                all_delayed_tokens.append(delayed_chunk)
                
                # Validate chunk consistency
                chunk_stats = audio_utils.get_audio_token_statistics([
                    delayed_chunk[cb] for cb in range(6)
                ])
                assert chunk_stats['num_codebooks'] == 6
                assert chunk_stats['batch_size'] == 1
            
            # Test cross-chunk consistency
            assert len(all_patterns) == total_chunks
            assert len(all_delayed_tokens) == total_chunks
            
            # All patterns should be consistent
            for i in range(1, total_chunks):
                np.testing.assert_array_equal(
                    all_patterns[0].numpy(),
                    all_patterns[i].numpy(),
                    "Streaming patterns should be consistent across chunks"
                )
    
    def test_mixed_strategy_fallback_scenarios(self):
        """Test complex fallback scenarios with mixed configurations."""
        
        with patch('tensorrt_llm.models.higgs_audio.model.constant', side_effect=create_mock_constant):
            # Test invalid custom strategy with fallback
            provider = DelayPatternProvider(
                strategy='custom',
                custom_delays=[-1, 'invalid', 2.5, None],  # Invalid delays
                max_delay=10,
                enable_fallback=True
            )
            
            # Should fallback to valid configuration
            assert provider.fallback_used or len(provider.get_validation_warnings()) > 0
            
            # Should still generate valid patterns
            pattern = provider.generate_delay_pattern(n_codebooks=4, sequence_length=6)
            assert pattern.shape == (4, 6)
            
            pattern_data = pattern.numpy()
            assert np.all(pattern_data >= 0), "Fallback should produce non-negative delays"
            assert np.all(pattern_data <= 10), "Fallback should respect max_delay"
    
    def test_memory_constraint_validation(self):
        """Test memory constraint validation for large configurations."""
        
        with patch('tensorrt_llm.models.higgs_audio.model.constant', side_effect=create_mock_constant):
            # Test reasonable size (should pass)
            provider = DelayPatternProvider(
                strategy='linear',
                validate_memory_usage=True,
                enable_fallback=True
            )
            
            pattern = provider.generate_delay_pattern(n_codebooks=8, sequence_length=128)
            assert pattern.shape == (8, 128)
            
            # Test very large size (should warn but work with fallback)
            large_pattern = provider.generate_delay_pattern(n_codebooks=32, sequence_length=512)
            assert large_pattern.shape == (32, 512)
            
            warnings = provider.get_validation_warnings()
            # Should have memory warnings for large configurations
    
    def test_cross_codebook_synchronization_validation(self):
        """Test validation of cross-codebook synchronization quality."""
        
        utils = AudioTokenUtils(num_codebooks=4, pad_token_id=0, eos_token_id=2)
        
        # Create well-synchronized tokens (all codebooks have same pattern)
        well_sync_tokens = [
            MockTensor([[1, 2, 0, 2], [3, 4, 0, 2]]),  # Codebook 0
            MockTensor([[5, 6, 0, 2], [7, 8, 0, 2]]),  # Codebook 1
            MockTensor([[9, 10, 0, 2], [11, 12, 0, 2]]), # Codebook 2
            MockTensor([[13, 14, 0, 2], [15, 16, 0, 2]]) # Codebook 3
        ]
        
        stats = utils.get_audio_token_statistics(well_sync_tokens)
        sync_quality = stats['cross_codebook_statistics']['synchronization_quality']
        
        # Well-synchronized tokens should have high sync quality
        assert sync_quality >= 0.8, "Well-synchronized tokens should have high sync quality"
        
        # Create poorly synchronized tokens (different patterns)
        poorly_sync_tokens = [
            MockTensor([[1, 2, 3, 4], [5, 6, 7, 8]]),     # No padding/EOS
            MockTensor([[0, 0, 0, 2], [0, 0, 0, 2]]),     # Mostly padding
            MockTensor([[1, 0, 2, 0], [3, 0, 2, 0]]),     # Mixed pattern
            MockTensor([[1, 1, 1, 1], [2, 2, 2, 2]])      # No special tokens
        ]
        
        poor_stats = utils.get_audio_token_statistics(poorly_sync_tokens)
        poor_sync_quality = poor_stats['cross_codebook_statistics']['synchronization_quality']
        
        # Poorly synchronized tokens should have lower sync quality
        assert poor_sync_quality < sync_quality, "Poorly synchronized tokens should have lower sync quality"


class TestRealWorldUseCases:
    """Test realistic TTS use cases and scenarios."""
    
    def test_voice_cloning_scenario(self):
        """Test delay patterns in voice cloning scenario."""
        
        with patch('tensorrt_llm.models.higgs_audio.model.constant', side_effect=create_mock_constant):
            # Setup for voice cloning (requires high-quality audio generation)
            provider = DelayPatternProvider(
                strategy='exponential',  # Better for quality
                stride=1,
                max_delay=16
            )
            
            utils = AudioTokenUtils(
                num_codebooks=8,  # High-quality audio needs more codebooks
                pad_token_id=0,
                audio_start_token_id=128010,
                audio_end_token_id=128011
            )
            
            # Test long sequence (typical for voice cloning)
            seq_len = 256
            pattern = provider.generate_delay_pattern(n_codebooks=8, sequence_length=seq_len)
            
            # Validate pattern for voice cloning requirements
            pattern_info = provider.get_pattern_info(pattern)
            assert pattern_info['max_delay'] <= 16, "Max delay should be reasonable for real-time"
            assert pattern_info['is_uniform_per_codebook'], "Should have uniform delays per codebook"
            
            # Test with realistic audio tokens
            voice_tokens = []
            for cb in range(8):
                # Simulate realistic audio token ranges (1024-2048 per codebook)
                cb_tokens = MockTensor([
                    [1024 + cb * 100 + i for i in range(seq_len)]
                ])
                voice_tokens.append(cb_tokens)
            
            # Validate token statistics
            stats = utils.get_audio_token_statistics(voice_tokens)
            assert stats['num_codebooks'] == 8
            assert stats['sequence_length'] == seq_len
            
            # Check token range validity
            for cb in range(8):
                cb_stats = stats['codebook_statistics'][f'codebook_{cb}']
                assert cb_stats['min_token'] >= 1024, "Audio tokens should be in valid range"
                assert cb_stats['max_token'] < 2048, "Audio tokens should be in valid range"
    
    def test_real_time_streaming_scenario(self):
        """Test delay patterns for real-time streaming TTS."""
        
        with patch('tensorrt_llm.models.higgs_audio.model.constant', side_effect=create_mock_constant):
            # Setup for real-time streaming (latency-optimized)
            provider = DelayPatternProvider(
                strategy='linear',  # More predictable latency
                stride=1,
                max_delay=4  # Low max delay for real-time
            )
            
            attention_utils = DelayAwareAttentionUtils(
                num_codebooks=4,
                causal_attention=True,  # Required for streaming
                cross_codebook_attention=False,  # Disabled for speed
                max_delay=4
            )
            
            # Simulate real-time streaming chunks (small chunks)
            chunk_size = 16
            num_chunks = 8
            
            for chunk_idx in range(num_chunks):
                # Generate low-latency pattern
                pattern = provider.generate_delay_pattern(n_codebooks=4, sequence_length=chunk_size)
                
                # Verify real-time constraints
                max_delay = np.max(pattern.numpy())
                assert max_delay <= 4, f"Chunk {chunk_idx}: Max delay {max_delay} too high for real-time"
                
                # Test attention mask for streaming
                with patch('torch.ones', return_value=MockTensor(np.ones((chunk_size, chunk_size), dtype=bool))), \
                     patch('torch.tril', return_value=MockTensor(np.tril(np.ones((chunk_size, chunk_size)), k=0).astype(bool))):
                    
                    mask = attention_utils.create_delay_aware_attention_mask(
                        batch_size=1,
                        seq_len=chunk_size,
                        delay_pattern=pattern
                    )
                    
                    assert mask.shape == (1, chunk_size, chunk_size)
    
    def test_batch_processing_scenario(self):
        """Test delay patterns for batch processing of multiple TTS requests."""
        
        with patch('tensorrt_llm.models.higgs_audio.model.constant', side_effect=create_mock_constant):
            # Setup for batch processing
            provider = DelayPatternProvider(strategy='linear', stride=2)  # Different stride
            utils = AudioTokenUtils(num_codebooks=6, pad_token_id=0, eos_token_id=2)
            
            batch_size = 4
            seq_len = 64
            
            # Generate pattern (same for all batch items)
            pattern = provider.generate_delay_pattern(n_codebooks=6, sequence_length=seq_len)
            
            # Create batch of audio tokens with different lengths
            batch_tokens = []
            for batch_idx in range(batch_size):
                # Each batch item might have different actual length
                actual_length = seq_len - (batch_idx * 8)  # Decreasing lengths
                
                item_tokens = []
                for cb in range(6):
                    # Create tokens with padding
                    tokens = [100 + batch_idx * 10 + cb * 1000 + i for i in range(actual_length)]
                    tokens.extend([0] * (seq_len - actual_length))  # Pad to seq_len
                    item_tokens.append(MockTensor([tokens]))
                
                batch_tokens.append(item_tokens)
            
            # Validate each batch item
            for batch_idx in range(batch_size):
                stats = utils.get_audio_token_statistics(batch_tokens[batch_idx])
                assert stats['batch_size'] == 1
                assert stats['num_codebooks'] == 6
                assert stats['sequence_length'] == seq_len
                
                # Check padding is correctly handled
                for cb in range(6):
                    cb_stats = stats['codebook_statistics'][f'codebook_{cb}']
                    expected_padding = seq_len - (seq_len - batch_idx * 8)
                    assert cb_stats['pad_token_count'] == expected_padding


if __name__ == "__main__":
    # Run specific test suites
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Quick test mode - run basic functionality tests
        pytest.main([
            "tests/test_higgs_audio_delay_patterns.py::TestDelayPatternProvider::test_initialization_valid_params",
            "tests/test_higgs_audio_delay_patterns.py::TestDelayPatternProvider::test_generate_delay_pattern_linear",
            "tests/test_higgs_audio_delay_patterns.py::TestAudioTokenUtils::test_initialization",
            "tests/test_higgs_audio_delay_patterns.py::TestDelayAwareAttentionUtils::test_initialization",
            "-v"
        ])
    elif len(sys.argv) > 1 and sys.argv[1] == "integration":
        # Integration test mode
        pytest.main([
            "tests/test_higgs_audio_delay_patterns.py::TestIntegrationScenarios",
            "-v"
        ])
    elif len(sys.argv) > 1 and sys.argv[1] == "real-world":
        # Real-world scenario tests
        pytest.main([
            "tests/test_higgs_audio_delay_patterns.py::TestRealWorldUseCases",
            "-v"
        ])
    else:
        # Full test suite
        pytest.main([
            "tests/test_higgs_audio_delay_patterns.py",
            "-v",
            "--tb=short"
        ])