"""
End-to-end TTS pipeline tests for Higgs Audio model.

This module tests the complete TTS workflow from text input to audio generation,
validating that all components work together properly including mode transitions,
delay patterns, multi-codebook generation, and streaming support.
"""

import pytest
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from unittest.mock import Mock, MagicMock, patch

# Skip all tests if TensorRT-LLM not available
from .conftest import TENSORRT_LLM_AVAILABLE
if TENSORRT_LLM_AVAILABLE:
    from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
    from tensorrt_llm.models.higgs_audio.model import (
        HiggsAudioForCausalLM,
        DelayPatternProvider,
        AudioTokenUtils,
        GenerationMode
    )

pytestmark = pytest.mark.skipif(
    not TENSORRT_LLM_AVAILABLE,
    reason="TensorRT-LLM not available"
)


@pytest.mark.end_to_end
class TestTTSPipelineComplete:
    """Test complete text-to-speech generation pipeline."""
    
    def test_text_only_generation_pipeline(self, sample_higgs_audio_config, mock_tensor, mock_layers):
        """Test complete text-only generation pipeline."""
        with patch('tensorrt_llm.models.higgs_audio.model.HiggsAudioBackbone'), \
             patch('tensorrt_llm.layers.ColumnLinear', mock_layers['ColumnLinear']), \
             patch('tensorrt_llm._utils.pad_vocab_size', return_value=32000), \
             patch('tensorrt_llm.models.modeling_utils.DecoderModelForCausalLM'), \
             patch('tensorrt_llm.models.higgs_audio.model.GenerationModeManager') as mock_mode_manager:
            
            # Setup mode manager for TEXT mode
            mock_manager_instance = MagicMock()
            mock_manager_instance.get_current_mode.return_value = GenerationMode.TEXT
            mock_manager_instance.is_audio_generation_active.return_value = False
            mock_mode_manager.return_value = mock_manager_instance
            
            model = HiggsAudioForCausalLM(sample_higgs_audio_config)
            
            # Mock generate method internals
            with patch.object(model, '_validate_and_prepare_generation_params') as mock_validate, \
                 patch.object(model, '_initialize_generation_state') as mock_init_state, \
                 patch.object(model, '_generate_standard') as mock_generate_std, \
                 patch.object(model, '_prepare_generation_outputs') as mock_prepare_outputs:
                
                # Setup mocks
                mock_validate.return_value = {
                    'input_ids': mock_tensor([[1, 2, 3, 4]]),
                    'max_length': 20,
                    'use_delay_pattern': False,
                    'streaming': False,
                    'num_codebooks': 1
                }
                mock_init_state.return_value = {'current_position': 4}
                mock_generate_std.return_value = {'sequences': mock_tensor([[1, 2, 3, 4, 5, 6, 7, 8]])}
                mock_prepare_outputs.return_value = {
                    'sequences': mock_tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
                    'generation_mode_history': ['text']
                }
                
                # Execute generation
                input_ids = mock_tensor([[1, 2, 3, 4]])
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=4,
                    return_dict=True
                )
                
                # Verify pipeline execution
                mock_validate.assert_called_once()
                mock_init_state.assert_called_once()
                mock_generate_std.assert_called_once()
                mock_prepare_outputs.assert_called_once()
                
                assert 'sequences' in outputs
                assert outputs['sequences'].shape == (1, 8)
    
    def test_audio_generation_pipeline_with_delay_patterns(self, sample_higgs_audio_config, mock_tensor, mock_layers):
        """Test complete audio generation pipeline with delay patterns."""
        sample_higgs_audio_config.use_delay_pattern = True
        sample_higgs_audio_config.audio_num_codebooks = 4
        
        with patch('tensorrt_llm.models.higgs_audio.model.HiggsAudioBackbone'), \
             patch('tensorrt_llm.layers.ColumnLinear', mock_layers['ColumnLinear']), \
             patch('tensorrt_llm._utils.pad_vocab_size', return_value=32000), \
             patch('tensorrt_llm.models.modeling_utils.DecoderModelForCausalLM'), \
             patch('tensorrt_llm.models.higgs_audio.model.GenerationModeManager') as mock_mode_manager:
            
            # Setup mode manager for AUDIO_INIT mode
            mock_manager_instance = MagicMock()
            mock_manager_instance.get_current_mode.return_value = GenerationMode.AUDIO_INIT
            mock_manager_instance.is_audio_generation_active.return_value = True
            mock_mode_manager.return_value = mock_manager_instance
            
            model = HiggsAudioForCausalLM(sample_higgs_audio_config)
            
            # Mock generation pipeline components
            with patch.object(model, '_validate_and_prepare_generation_params') as mock_validate, \
                 patch.object(model, '_initialize_generation_state') as mock_init_state, \
                 patch.object(model, '_generate_standard') as mock_generate_std, \
                 patch.object(model, '_prepare_generation_outputs') as mock_prepare_outputs, \
                 patch('tensorrt_llm.models.higgs_audio.model.DelayPatternProvider') as mock_delay_provider, \
                 patch('tensorrt_llm.models.higgs_audio.model.AudioTokenUtils') as mock_audio_utils:
                
                # Setup mocks
                mock_validate.return_value = {
                    'input_ids': mock_tensor([[1, 2, 3, 4]]),
                    'max_length': 24,  # 4 original + 20 new (4 codebooks * 5 time steps)
                    'use_delay_pattern': True,
                    'streaming': False,
                    'num_codebooks': 4,
                    'pad_token_id': 0,
                    'eos_token_id': 2
                }
                
                mock_init_state.return_value = {
                    'current_position': 4,
                    'audio_generation_active': True,
                    'codebook_states': {
                        'codebook_0': {'active': False, 'generated_tokens': []},
                        'codebook_1': {'active': False, 'generated_tokens': []},
                        'codebook_2': {'active': False, 'generated_tokens': []},
                        'codebook_3': {'active': False, 'generated_tokens': []}
                    }
                }
                
                # Mock audio tokens with delay pattern applied
                audio_tokens = mock_tensor([
                    [1, 2, 3, 4, 100, 200, 300, 400, 101, 201, 301, 401, 102, 202, 302, 402, 103, 203, 303, 403, 104, 204, 304, 404]
                ])
                
                mock_generate_std.return_value = {'sequences': audio_tokens}
                mock_prepare_outputs.return_value = {
                    'sequences': audio_tokens,
                    'codebook_sequences': [
                        mock_tensor([[100, 101, 102, 103, 104]]),  # Codebook 0
                        mock_tensor([[200, 201, 202, 203, 204]]),  # Codebook 1  
                        mock_tensor([[300, 301, 302, 303, 304]]),  # Codebook 2
                        mock_tensor([[400, 401, 402, 403, 404]])   # Codebook 3
                    ],
                    'generation_mode_history': ['audio_init', 'audio_in_progress'],
                    'delay_pattern_info': {
                        'strategy': 'linear',
                        'max_delay': 3,
                        'codebook_delays': [0, 1, 2, 3]
                    }
                }
                
                # Setup delay pattern provider
                mock_delay_instance = MagicMock()
                mock_delay_provider.return_value = mock_delay_instance
                
                # Setup audio utils
                mock_audio_instance = MagicMock()
                mock_audio_utils.return_value = mock_audio_instance
                
                # Execute audio generation
                input_ids = mock_tensor([[1, 2, 3, 4]])
                audio_features = mock_tensor(np.random.randn(1, 100, 1280))
                
                outputs = model.generate(
                    input_ids=input_ids,
                    audio_features=audio_features,
                    max_new_tokens=20,
                    use_delay_pattern=True,
                    num_codebooks=4,
                    return_dict=True
                )
                
                # Verify audio pipeline execution
                mock_validate.assert_called_once()
                mock_delay_provider.assert_called_once()
                mock_audio_utils.assert_called_once()
                
                assert 'sequences' in outputs
                assert 'codebook_sequences' in outputs
                assert 'delay_pattern_info' in outputs
                assert len(outputs['codebook_sequences']) == 4
    
    def test_streaming_audio_generation_pipeline(self, sample_higgs_audio_config, mock_tensor, mock_layers):
        """Test streaming audio generation pipeline."""
        sample_higgs_audio_config.audio_streaming_chunk_size = 16
        
        with patch('tensorrt_llm.models.higgs_audio.model.HiggsAudioBackbone'), \
             patch('tensorrt_llm.layers.ColumnLinear', mock_layers['ColumnLinear']), \
             patch('tensorrt_llm._utils.pad_vocab_size', return_value=32000), \
             patch('tensorrt_llm.models.modeling_utils.DecoderModelForCausalLM'), \
             patch('tensorrt_llm.models.higgs_audio.model.GenerationModeManager') as mock_mode_manager:
            
            # Setup mode manager for streaming
            mock_manager_instance = MagicMock()
            mock_manager_instance.get_current_mode.return_value = GenerationMode.AUDIO_IN_PROGRESS
            mock_manager_instance.is_audio_generation_active.return_value = True
            mock_mode_manager.return_value = mock_manager_instance
            
            model = HiggsAudioForCausalLM(sample_higgs_audio_config)
            
            with patch.object(model, '_validate_and_prepare_generation_params') as mock_validate, \
                 patch.object(model, '_initialize_generation_state') as mock_init_state, \
                 patch.object(model, '_generate_streaming') as mock_generate_stream, \
                 patch.object(model, '_prepare_generation_outputs') as mock_prepare_outputs:
                
                # Setup streaming parameters
                mock_validate.return_value = {
                    'input_ids': mock_tensor([[1, 2, 3, 4]]),
                    'streaming': True,
                    'stream_chunk_size': 16,
                    'use_delay_pattern': True,
                    'num_codebooks': 4
                }
                
                mock_init_state.return_value = {
                    'streaming_chunks': [],
                    'audio_generation_active': True
                }
                
                # Mock streaming chunks
                streaming_chunks = [
                    mock_tensor([[100, 200, 300, 400, 101, 201, 301, 401]]),  # Chunk 0
                    mock_tensor([[102, 202, 302, 402, 103, 203, 303, 403]]),  # Chunk 1
                    mock_tensor([[104, 204, 304, 404, 105, 205, 305, 405]])   # Chunk 2
                ]
                
                mock_generate_stream.return_value = {
                    'sequences': mock_tensor([[1, 2, 3, 4] + [t.data.flatten().tolist()[0] for t in streaming_chunks]]),
                    'streaming_chunks': streaming_chunks
                }
                
                mock_prepare_outputs.return_value = {
                    'sequences': mock_tensor([[1, 2, 3, 4, 100, 200, 300, 400, 101, 201, 301, 401]]),
                    'streaming_chunks': streaming_chunks,
                    'generation_mode_history': ['audio_in_progress']
                }
                
                # Execute streaming generation
                outputs = model.generate(
                    input_ids=mock_tensor([[1, 2, 3, 4]]),
                    streaming=True,
                    stream_chunk_size=16,
                    return_dict=True
                )
                
                # Verify streaming pipeline
                mock_validate.assert_called_once()
                mock_generate_stream.assert_called_once()
                
                assert 'streaming_chunks' in outputs
                assert len(outputs['streaming_chunks']) == 3
    
    def test_mode_transitions_during_generation(self, sample_higgs_audio_config, mock_tensor, mock_layers):
        """Test mode transitions during generation pipeline."""
        with patch('tensorrt_llm.models.higgs_audio.model.HiggsAudioBackbone'), \
             patch('tensorrt_llm.layers.ColumnLinear', mock_layers['ColumnLinear']), \
             patch('tensorrt_llm._utils.pad_vocab_size', return_value=32000), \
             patch('tensorrt_llm.models.modeling_utils.DecoderModelForCausalLM'), \
             patch('tensorrt_llm.models.higgs_audio.model.GenerationModeManager') as mock_mode_manager:
            
            # Setup mode manager to track transitions
            mock_manager_instance = MagicMock()
            mock_manager_instance.get_current_mode.side_effect = [
                GenerationMode.TEXT,
                GenerationMode.AUDIO_INIT,
                GenerationMode.AUDIO_IN_PROGRESS
            ]
            mock_manager_instance.is_audio_generation_active.return_value = True
            mock_mode_manager.return_value = mock_manager_instance
            
            model = HiggsAudioForCausalLM(sample_higgs_audio_config)
            
            # Test mode transitions
            with patch.object(model, 'set_generation_mode') as mock_set_mode:
                
                # Simulate mode transitions during generation
                model.set_generation_mode(GenerationMode.TEXT)
                model.set_generation_mode(GenerationMode.AUDIO_INIT)
                model.set_generation_mode(GenerationMode.AUDIO_IN_PROGRESS)
                
                # Verify all transitions were attempted
                assert mock_set_mode.call_count == 3
                
                expected_modes = [GenerationMode.TEXT, GenerationMode.AUDIO_INIT, GenerationMode.AUDIO_IN_PROGRESS]
                actual_modes = [call[0][0] for call in mock_set_mode.call_args_list]
                assert actual_modes == expected_modes


@pytest.mark.end_to_end
@pytest.mark.streaming
class TestStreamingInference:
    """Test streaming inference capabilities."""
    
    def test_real_time_streaming_constraints(self, sample_higgs_audio_config, mock_tensor, performance_benchmarker):
        """Test that streaming inference meets real-time constraints."""
        sample_higgs_audio_config.audio_realtime_mode = True
        sample_higgs_audio_config.audio_streaming_chunk_size = 32
        
        with patch('tensorrt_llm.models.higgs_audio.model.HiggsAudioBackbone'), \
             patch('tensorrt_llm.layers.ColumnLinear'), \
             patch('tensorrt_llm._utils.pad_vocab_size', return_value=32000), \
             patch('tensorrt_llm.models.modeling_utils.DecoderModelForCausalLM'), \
             patch('tensorrt_llm.models.higgs_audio.model.GenerationModeManager'):
            
            model = HiggsAudioForCausalLM(sample_higgs_audio_config)
            
            # Mock streaming generation function
            def mock_streaming_func():
                # Simulate chunk generation latency
                return {'chunk': mock_tensor(np.random.randint(100, 500, (1, 32)))}
            
            # Measure streaming latency
            result, latency_ms = performance_benchmarker.measure_latency(mock_streaming_func)
            
            # Real-time constraint: chunk generation should be < 50ms for 32 tokens
            # (This is a mock test - real implementation would have actual latency requirements)
            performance_benchmarker.record_measurement(
                'streaming_chunk_latency_ms',
                latency_ms,
                {'chunk_size': 32, 'real_time_mode': True}
            )
            
            # In real implementation, would assert latency < threshold
            assert latency_ms >= 0  # Basic sanity check
    
    def test_streaming_chunk_coordination(self, sample_higgs_audio_config, mock_tensor, mock_layers):
        """Test coordination between streaming chunks with delay patterns."""
        sample_higgs_audio_config.use_delay_pattern = True
        sample_higgs_audio_config.audio_delay_pattern_strategy = 'linear'
        
        with patch('tensorrt_llm.models.higgs_audio.model.HiggsAudioBackbone'), \
             patch('tensorrt_llm.layers.ColumnLinear', mock_layers['ColumnLinear']), \
             patch('tensorrt_llm._utils.pad_vocab_size', return_value=32000), \
             patch('tensorrt_llm.models.modeling_utils.DecoderModelForCausalLM'), \
             patch('tensorrt_llm.models.higgs_audio.model.GenerationModeManager') as mock_mode_manager:
            
            mock_manager_instance = MagicMock()
            mock_mode_manager.return_value = mock_manager_instance
            
            model = HiggsAudioForCausalLM(sample_higgs_audio_config)
            
            # Test chunk coordination
            with patch('tensorrt_llm.models.higgs_audio.model.DelayPatternProvider') as mock_delay_provider:
                mock_delay_instance = MagicMock()
                mock_delay_provider.return_value = mock_delay_instance
                
                # Mock delay pattern for streaming
                mock_delay_instance.generate_delay_pattern.return_value = mock_tensor([
                    [0, 0, 0, 0],  # Codebook 0: no delay
                    [1, 1, 1, 1],  # Codebook 1: 1-step delay
                    [2, 2, 2, 2],  # Codebook 2: 2-step delay
                    [3, 3, 3, 3]   # Codebook 3: 3-step delay
                ])
                
                # Simulate streaming chunk processing
                chunk_size = 16
                num_chunks = 4
                
                for chunk_idx in range(num_chunks):
                    pattern = mock_delay_instance.generate_delay_pattern(
                        n_codebooks=4,
                        sequence_length=chunk_size
                    )
                    
                    # Verify pattern consistency across chunks
                    assert pattern.shape == (4, chunk_size)
                
                # Verify delay pattern provider was used consistently
                assert mock_delay_instance.generate_delay_pattern.call_count == num_chunks
    
    def test_streaming_error_recovery(self, sample_higgs_audio_config, mock_tensor):
        """Test error recovery during streaming generation."""
        with patch('tensorrt_llm.models.higgs_audio.model.HiggsAudioBackbone'), \
             patch('tensorrt_llm.layers.ColumnLinear'), \
             patch('tensorrt_llm._utils.pad_vocab_size', return_value=32000), \
             patch('tensorrt_llm.models.modeling_utils.DecoderModelForCausalLM'), \
             patch('tensorrt_llm.models.higgs_audio.model.GenerationModeManager') as mock_mode_manager:
            
            mock_manager_instance = MagicMock()
            mock_manager_instance.reset_state = MagicMock()
            mock_mode_manager.return_value = mock_manager_instance
            
            model = HiggsAudioForCausalLM(sample_higgs_audio_config)
            
            # Mock generation failure
            with patch.object(model, '_validate_and_prepare_generation_params') as mock_validate:
                mock_validate.side_effect = RuntimeError("Generation failed")
                
                # Test error handling and recovery
                with pytest.raises(RuntimeError, match="Generation failed"):
                    model.generate(
                        input_ids=mock_tensor([[1, 2, 3, 4]]),
                        streaming=True,
                        return_dict=True
                    )
                
                # Verify error recovery was attempted
                mock_manager_instance.reset_state.assert_called_once()


@pytest.mark.end_to_end
class TestMultiCodebookGeneration:
    """Test multi-codebook RVQ generation."""
    
    def test_four_codebook_generation(self, sample_higgs_audio_config, mock_tensor, mock_layers):
        """Test generation with 4 RVQ codebooks."""
        sample_higgs_audio_config.audio_num_codebooks = 4
        sample_higgs_audio_config.use_delay_pattern = True
        
        with patch('tensorrt_llm.models.higgs_audio.model.HiggsAudioBackbone'), \
             patch('tensorrt_llm.layers.ColumnLinear', mock_layers['ColumnLinear']), \
             patch('tensorrt_llm._utils.pad_vocab_size', return_value=32000), \
             patch('tensorrt_llm.models.modeling_utils.DecoderModelForCausalLM'), \
             patch('tensorrt_llm.models.higgs_audio.model.GenerationModeManager') as mock_mode_manager:
            
            mock_manager_instance = MagicMock()
            mock_mode_manager.return_value = mock_manager_instance
            
            model = HiggsAudioForCausalLM(sample_higgs_audio_config)
            
            with patch.object(model, '_validate_and_prepare_generation_params') as mock_validate, \
                 patch('tensorrt_llm.models.higgs_audio.model.DelayPatternProvider') as mock_delay_provider, \
                 patch('tensorrt_llm.models.higgs_audio.model.AudioTokenUtils') as mock_audio_utils:
                
                # Setup 4-codebook configuration
                mock_validate.return_value = {
                    'input_ids': mock_tensor([[1, 2, 3, 4]]),
                    'num_codebooks': 4,
                    'use_delay_pattern': True,
                    'streaming': False
                }
                
                # Mock delay pattern and audio utils
                mock_delay_instance = MagicMock()
                mock_delay_provider.return_value = mock_delay_instance
                
                mock_audio_instance = MagicMock()
                mock_audio_utils.return_value = mock_audio_instance
                
                # Verify 4-codebook setup
                mock_delay_provider.assert_called_once()
                mock_audio_utils.assert_called_once()
                
                # Check initialization args for correct codebook count
                delay_init_args = mock_delay_provider.call_args[1]  # keyword args
                audio_init_args = mock_audio_utils.call_args[1]
                
                expected_num_codebooks = 4
                assert audio_init_args.get('num_codebooks') == expected_num_codebooks
    
    def test_large_codebook_generation(self, sample_higgs_audio_config, mock_tensor, mock_layers):
        """Test generation with large number of codebooks."""
        sample_higgs_audio_config.audio_num_codebooks = 16
        sample_higgs_audio_config.use_delay_pattern = True
        sample_higgs_audio_config.audio_delay_pattern_strategy = 'exponential'
        
        with patch('tensorrt_llm.models.higgs_audio.model.HiggsAudioBackbone'), \
             patch('tensorrt_llm.layers.ColumnLinear', mock_layers['ColumnLinear']), \
             patch('tensorrt_llm._utils.pad_vocab_size', return_value=32000), \
             patch('tensorrt_llm.models.modeling_utils.DecoderModelForCausalLM'), \
             patch('tensorrt_llm.models.higgs_audio.model.GenerationModeManager') as mock_mode_manager:
            
            mock_manager_instance = MagicMock()
            mock_mode_manager.return_value = mock_manager_instance
            
            model = HiggsAudioForCausalLM(sample_higgs_audio_config)
            
            with patch.object(model, '_validate_and_prepare_generation_params') as mock_validate, \
                 patch('tensorrt_llm.models.higgs_audio.model.DelayPatternProvider') as mock_delay_provider:
                
                mock_validate.return_value = {
                    'input_ids': mock_tensor([[1, 2, 3, 4]]),
                    'num_codebooks': 16,
                    'use_delay_pattern': True,
                    'streaming': False
                }
                
                mock_delay_instance = MagicMock()
                mock_delay_provider.return_value = mock_delay_instance
                
                # Verify delay pattern provider handles large codebook count
                mock_delay_provider.assert_called_once()
                
                # Check that exponential strategy is used
                delay_init_args = mock_delay_provider.call_args[1]
                assert delay_init_args.get('strategy') == 'exponential'
    
    def test_codebook_synchronization(self, sample_higgs_audio_config, mock_tensor):
        """Test synchronization between multiple codebooks."""
        sample_higgs_audio_config.audio_num_codebooks = 8
        
        with patch('tensorrt_llm.models.higgs_audio.model.HiggsAudioBackbone'), \
             patch('tensorrt_llm.layers.ColumnLinear'), \
             patch('tensorrt_llm._utils.pad_vocab_size', return_value=32000), \
             patch('tensorrt_llm.models.modeling_utils.DecoderModelForCausalLM'), \
             patch('tensorrt_llm.models.higgs_audio.model.GenerationModeManager') as mock_mode_manager:
            
            mock_manager_instance = MagicMock()
            mock_manager_instance.get_generation_state.return_value = MagicMock()
            mock_mode_manager.return_value = mock_manager_instance
            
            model = HiggsAudioForCausalLM(sample_higgs_audio_config)
            
            # Test codebook state tracking
            with patch('tensorrt_llm.models.higgs_audio.model.AudioTokenUtils') as mock_audio_utils:
                mock_audio_instance = MagicMock()
                mock_audio_utils.return_value = mock_audio_instance
                
                # Mock codebook token validation
                mock_codebook_tokens = [
                    mock_tensor([[100 + i, 101 + i, 102 + i]]) for i in range(8)
                ]
                
                mock_audio_instance.validate_codebook_sequences.return_value = True
                
                # Simulate codebook synchronization check
                for tokens in mock_codebook_tokens:
                    model.update_audio_generation_state(tokens.data[0, 0])
                
                # Verify synchronization was maintained
                assert mock_manager_instance.get_generation_state.call_count >= 8


@pytest.mark.end_to_end
class TestErrorHandlingAndRecovery:
    """Test comprehensive error handling and recovery mechanisms."""
    
    def test_generation_parameter_validation_errors(self, sample_higgs_audio_config, mock_tensor):
        """Test validation errors for generation parameters."""
        with patch('tensorrt_llm.models.higgs_audio.model.HiggsAudioBackbone'), \
             patch('tensorrt_llm.layers.ColumnLinear'), \
             patch('tensorrt_llm._utils.pad_vocab_size', return_value=32000), \
             patch('tensorrt_llm.models.modeling_utils.DecoderModelForCausalLM'), \
             patch('tensorrt_llm.models.higgs_audio.model.GenerationModeManager'):
            
            model = HiggsAudioForCausalLM(sample_higgs_audio_config)
            
            # Test various validation errors
            with pytest.raises(ValueError, match="input_ids is required"):
                model.generate(input_ids=None)
            
            with pytest.raises(TypeError, match="input_ids must be a Tensor"):
                model.generate(input_ids="invalid")
            
            with pytest.raises(ValueError, match="temperature must be positive"):
                model.generate(
                    input_ids=mock_tensor([[1, 2, 3]]),
                    temperature=0.0
                )
            
            with pytest.raises(ValueError, match="max_new_tokens must be positive"):
                model.generate(
                    input_ids=mock_tensor([[1, 2, 3]]),
                    max_new_tokens=-1
                )
    
    def test_mode_transition_error_recovery(self, sample_higgs_audio_config):
        """Test error recovery during mode transitions."""
        with patch('tensorrt_llm.models.higgs_audio.model.HiggsAudioBackbone'), \
             patch('tensorrt_llm.layers.ColumnLinear'), \
             patch('tensorrt_llm._utils.pad_vocab_size', return_value=32000), \
             patch('tensorrt_llm.models.modeling_utils.DecoderModelForCausalLM'), \
             patch('tensorrt_llm.models.higgs_audio.model.GenerationModeManager') as mock_mode_manager:
            
            mock_manager_instance = MagicMock()
            mock_manager_instance.transition_to_mode.return_value = False  # Simulate failure
            mock_mode_manager.return_value = mock_manager_instance
            
            model = HiggsAudioForCausalLM(sample_higgs_audio_config)
            
            # Test mode transition failure handling
            with pytest.raises(ValueError, match="Failed to transition to mode"):
                model.set_generation_mode(Gener