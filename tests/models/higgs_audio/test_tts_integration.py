"""
Comprehensive integration tests for Higgs Audio TTS core components.

This module tests the integration and interaction between major TTS components
including the TTS-optimized model classes, DualFFN architecture, generation
mode management, and CUDA graph integration.
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
        HiggsAudioModelForCausalLM,
        HiggsAudioDualFFNDecoderLayer,
        HiggsAudioDecoderLayer,
        HiggsAudioBackbone,
        HiggsAudioForCausalLM,
        GenerationMode
    )

pytestmark = pytest.mark.skipif(
    not TENSORRT_LLM_AVAILABLE,
    reason="TensorRT-LLM not available"
)


@pytest.mark.integration
class TestHiggsAudioModelForCausalLM:
    """Test the TTS-optimized base model class."""
    
    def test_initialization_with_valid_config(self, sample_higgs_audio_config):
        """Test model initialization with valid configuration."""
        with patch('tensorrt_llm.models.higgs_audio.model.GenerationModeManager') as mock_manager:
            mock_manager.return_value = MagicMock()
            
            model = HiggsAudioModelForCausalLM(sample_higgs_audio_config)
            
            assert model.config == sample_higgs_audio_config
            assert hasattr(model, 'generation_mode_manager')
            mock_manager.assert_called_once()
    
    def test_generation_mode_transitions(self, sample_higgs_audio_config):
        """Test generation mode transitions with comprehensive validation."""
        with patch('tensorrt_llm.models.higgs_audio.model.GenerationModeManager') as mock_manager:
            mock_mode_manager = MagicMock()
            mock_manager.return_value = mock_mode_manager
            
            model = HiggsAudioModelForCausalLM(sample_higgs_audio_config)
            
            # Test mode transition
            mock_mode_manager.transition_to_mode.return_value = True
            model.set_generation_mode(GenerationMode.AUDIO_INIT)
            
            mock_mode_manager.transition_to_mode.assert_called_with(
                target_mode=GenerationMode.AUDIO_INIT,
                validation_level='standard',
                preserve_context=True
            )
    
    def test_generation_mode_transition_failure(self, sample_higgs_audio_config):
        """Test handling of failed generation mode transitions."""
        with patch('tensorrt_llm.models.higgs_audio.model.GenerationModeManager') as mock_manager:
            mock_mode_manager = MagicMock()
            mock_manager.return_value = mock_mode_manager
            mock_mode_manager.transition_to_mode.return_value = False
            
            model = HiggsAudioModelForCausalLM(sample_higgs_audio_config)
            
            with pytest.raises(ValueError, match="Failed to transition to mode"):
                model.set_generation_mode(GenerationMode.AUDIO_INIT)
    
    def test_audio_generation_state_management(self, sample_higgs_audio_config):
        """Test audio generation state management."""
        with patch('tensorrt_llm.models.higgs_audio.model.GenerationModeManager') as mock_manager:
            mock_mode_manager = MagicMock()
            mock_manager.return_value = mock_mode_manager
            
            # Mock generation state
            mock_state = MagicMock()
            mock_state.current_mode = GenerationMode.AUDIO_IN_PROGRESS
            mock_state.generated_tokens = 5
            mock_state.current_position = 10
            mock_state.codebook_states = {'codebook_0': {'active': True}}
            mock_mode_manager.get_generation_state.return_value = mock_state
            mock_mode_manager.is_audio_generation_active.return_value = True
            
            model = HiggsAudioModelForCausalLM(sample_higgs_audio_config)
            
            # Test state update
            model.update_audio_generation_state(123)
            
            # Verify state was updated
            assert mock_state.generated_tokens == 6
            assert mock_state.current_position == 11
    
    def test_performance_summary(self, sample_higgs_audio_config):
        """Test performance summary retrieval."""
        with patch('tensorrt_llm.models.higgs_audio.model.GenerationModeManager') as mock_manager:
            mock_mode_manager = MagicMock()
            mock_manager.return_value = mock_mode_manager
            mock_mode_manager.get_performance_summary.return_value = {
                'avg_latency_ms': 50.0,
                'total_tokens': 100
            }
            
            model = HiggsAudioModelForCausalLM(sample_higgs_audio_config)
            summary = model.get_performance_summary()
            
            assert 'avg_latency_ms' in summary
            assert summary['avg_latency_ms'] == 50.0
    
    def test_state_checkpoint_operations(self, sample_higgs_audio_config):
        """Test state checkpoint creation and restoration."""
        with patch('tensorrt_llm.models.higgs_audio.model.GenerationModeManager') as mock_manager:
            mock_mode_manager = MagicMock()
            mock_manager.return_value = mock_mode_manager
            mock_mode_manager.create_state_checkpoint.return_value = "checkpoint_data"
            mock_mode_manager.restore_state_from_checkpoint.return_value = True
            
            model = HiggsAudioModelForCausalLM(sample_higgs_audio_config)
            
            # Test checkpoint creation
            checkpoint = model.create_state_checkpoint()
            assert checkpoint == "checkpoint_data"
            
            # Test checkpoint restoration
            success = model.restore_state_from_checkpoint(checkpoint)
            assert success is True
    
    def test_cuda_graph_integration(self, sample_higgs_audio_config):
        """Test CUDA graph manager integration."""
        # Enable CUDA graphs in config
        sample_higgs_audio_config.cuda_graph_enable = True
        
        with patch('tensorrt_llm.models.higgs_audio.model.CudaGraphManager') as mock_cuda_manager, \
             patch('tensorrt_llm.models.higgs_audio.model.CUDA_GRAPHS_AVAILABLE', True), \
             patch('tensorrt_llm.models.higgs_audio.model.GenerationModeManager') as mock_mode_manager:
            
            mock_cuda_instance = MagicMock()
            mock_cuda_manager.return_value = mock_cuda_instance
            mock_mode_manager.return_value = MagicMock()
            
            model = HiggsAudioModelForCausalLM(sample_higgs_audio_config)
            
            assert model.cuda_graph_manager is not None
            mock_cuda_manager.assert_called_once()
    
    @patch('tensorrt_llm.models.higgs_audio.convert.build_config_from_hf')
    def test_from_hugging_face_factory(self, mock_build_config):
        """Test factory method for loading from HuggingFace."""
        mock_config = MagicMock(spec=HiggsAudioConfig)
        mock_build_config.return_value = mock_config
        
        with patch('tensorrt_llm.models.higgs_audio.model.GenerationModeManager'):
            model = HiggsAudioModelForCausalLM.from_hugging_face(
                "test/model/path",
                dtype="float16"
            )
            
            mock_build_config.assert_called_once_with(
                "test/model/path",
                dtype="float16",
                mapping=None,
                quant_config=None
            )


@pytest.mark.integration
@pytest.mark.dualffn
class TestHiggsAudioDualFFNDecoderLayer:
    """Test DualFFN decoder layer architecture."""
    
    def test_initialization_with_dualffn_config(self, sample_higgs_audio_config, mock_layers):
        """Test DualFFN layer initialization."""
        # Set layer 1 to use DualFFN
        sample_higgs_audio_config.audio_dual_ffn_layers = [1]
        
        with patch.multiple('tensorrt_llm.layers',
                          RmsNorm=mock_layers['RmsNorm'],
                          Attention=mock_layers['Attention'],
                          GatedMLP=mock_layers['GatedMLP']):
            
            layer = HiggsAudioDualFFNDecoderLayer(sample_higgs_audio_config, layer_idx=1)
            
            assert layer.use_dual_ffn is True
            assert hasattr(layer, 'text_mlp')
            assert hasattr(layer, 'audio_mlp')
            assert hasattr(layer, 'post_layernorm_text')
            assert hasattr(layer, 'post_layernorm_audio')
    
    def test_initialization_without_dualffn_config(self, sample_higgs_audio_config, mock_layers):
        """Test standard layer initialization when not using DualFFN."""
        # Layer 0 is not in DualFFN layers list
        sample_higgs_audio_config.audio_dual_ffn_layers = [1]
        
        with patch.multiple('tensorrt_llm.layers',
                          RmsNorm=mock_layers['RmsNorm'],
                          Attention=mock_layers['Attention'],
                          GatedMLP=mock_layers['GatedMLP']):
            
            layer = HiggsAudioDualFFNDecoderLayer(sample_higgs_audio_config, layer_idx=0)
            
            assert layer.use_dual_ffn is False
            assert hasattr(layer, 'mlp')
            assert not hasattr(layer, 'text_mlp')
    
    def test_forward_with_audio_mask(self, sample_higgs_audio_config, mock_layers, mock_tensor):
        """Test forward pass with audio output mask."""
        sample_higgs_audio_config.audio_dual_ffn_layers = [1]
        
        with patch.multiple('tensorrt_llm.layers',
                          RmsNorm=mock_layers['RmsNorm'],
                          Attention=mock_layers['Attention'],
                          GatedMLP=mock_layers['GatedMLP']), \
             patch('tensorrt_llm.functional.where') as mock_where, \
             patch('tensorrt_llm.functional.cast') as mock_cast:
            
            # Setup mocks
            mock_cast.side_effect = lambda x, dtype: x
            mock_where.return_value = mock_tensor(np.random.randn(2, 8, 512))
            
            layer = HiggsAudioDualFFNDecoderLayer(sample_higgs_audio_config, layer_idx=1)
            
            hidden_states = mock_tensor(np.random.randn(2, 8, 512))
            audio_mask = mock_tensor(np.array([[True, False, True, False, True, False, True, False],
                                             [False, True, False, True, False, True, False, True]]))
            
            output = layer.forward(hidden_states, audio_out_mask=audio_mask)
            
            assert output.shape == (2, 8, 512)
    
    def test_fast_forward_mode(self, sample_higgs_audio_config, mock_layers, mock_tensor):
        """Test fast-forward mode for audio tokens."""
        sample_higgs_audio_config.audio_dual_ffn_layers = [1]
        sample_higgs_audio_config.audio_fast_forward_layers = [1]
        
        with patch.multiple('tensorrt_llm.layers',
                          RmsNorm=mock_layers['RmsNorm'],
                          Attention=mock_layers['Attention'],
                          GatedMLP=mock_layers['GatedMLP']), \
             patch('tensorrt_llm.functional.cast') as mock_cast:
            
            mock_cast.side_effect = lambda x, dtype: x
            
            layer = HiggsAudioDualFFNDecoderLayer(sample_higgs_audio_config, layer_idx=1)
            
            assert layer.use_fast_forward is True
    
    def test_validation_errors(self, sample_higgs_audio_config):
        """Test configuration validation errors."""
        # Invalid layer index
        with pytest.raises(ValueError, match="layer_idx .* exceeds num_hidden_layers"):
            HiggsAudioDualFFNDecoderLayer(sample_higgs_audio_config, layer_idx=99)
        
        # Invalid config type
        with pytest.raises(TypeError):
            HiggsAudioDualFFNDecoderLayer("invalid_config", layer_idx=0)


@pytest.mark.integration
class TestHiggsAudioBackbone:
    """Test the core transformer backbone."""
    
    def test_initialization_with_standard_layers(self, sample_higgs_audio_config, mock_layers):
        """Test backbone initialization with standard decoder layers."""
        sample_higgs_audio_config.audio_adapter_type = 'stack'
        
        with patch.multiple('tensorrt_llm.layers',
                          Embedding=mock_layers['Embedding'],
                          RmsNorm=mock_layers['RmsNorm']), \
             patch('tensorrt_llm.models.higgs_audio.model.HiggsAudioDecoderLayer') as mock_decoder_layer, \
             patch('tensorrt_llm.models.higgs_audio.model.ModuleList') as mock_module_list:
            
            mock_decoder_layer.return_value = MagicMock()
            mock_module_list.return_value = MagicMock()
            
            backbone = HiggsAudioBackbone(sample_higgs_audio_config)
            
            # Should create standard decoder layers
            assert mock_decoder_layer.call_count == sample_higgs_audio_config.num_hidden_layers
    
    def test_initialization_with_dualffn_layers(self, sample_higgs_audio_config, mock_layers):
        """Test backbone initialization with DualFFN layers."""
        sample_higgs_audio_config.audio_adapter_type = 'dual_ffn'
        sample_higgs_audio_config.audio_dual_ffn_layers = [1, 3]
        
        with patch.multiple('tensorrt_llm.layers',
                          Embedding=mock_layers['Embedding'],
                          RmsNorm=mock_layers['RmsNorm']), \
             patch('tensorrt_llm.models.higgs_audio.model.HiggsAudioDecoderLayer') as mock_std_layer, \
             patch('tensorrt_llm.models.higgs_audio.model.HiggsAudioDualFFNDecoderLayer') as mock_dual_layer, \
             patch('tensorrt_llm.models.higgs_audio.model.ModuleList') as mock_module_list:
            
            mock_std_layer.return_value = MagicMock()
            mock_dual_layer.return_value = MagicMock()
            mock_module_list.return_value = MagicMock()
            
            backbone = HiggsAudioBackbone(sample_higgs_audio_config)
            
            # Should create both standard and DualFFN layers
            assert mock_std_layer.call_count == 2  # Layers 0, 2
            assert mock_dual_layer.call_count == 2  # Layers 1, 3
    
    def test_audio_out_mask_management(self, sample_higgs_audio_config, mock_tensor):
        """Test audio output mask storage and retrieval."""
        with patch.multiple('tensorrt_llm.layers',
                          Embedding=MagicMock(),
                          RmsNorm=MagicMock()), \
             patch('tensorrt_llm.models.higgs_audio.model.ModuleList'):
            
            backbone = HiggsAudioBackbone(sample_higgs_audio_config)
            
            # Test setting and getting audio mask
            audio_mask = mock_tensor(np.array([[True, False, True, False]]))
            
            backbone.set_audio_out_mask(audio_mask)
            retrieved_mask = backbone.get_audio_out_mask()
            
            assert retrieved_mask is audio_mask
    
    def test_forward_with_pipeline_parallelism(self, sample_higgs_audio_config, mock_tensor, mock_layers):
        """Test forward pass with pipeline parallelism simulation."""
        # Mock mapping for first PP rank
        sample_higgs_audio_config.mapping.is_first_pp_rank.return_value = True
        sample_higgs_audio_config.mapping.is_last_pp_rank.return_value = True
        
        with patch.multiple('tensorrt_llm.layers',
                          Embedding=mock_layers['Embedding'],
                          RmsNorm=mock_layers['RmsNorm']), \
             patch('tensorrt_llm.models.higgs_audio.model.ModuleList') as mock_module_list:
            
            # Mock layers forward
            mock_layers_instance = MagicMock()
            mock_layers_instance.forward.return_value = mock_tensor(np.random.randn(2, 8, 512))
            mock_module_list.return_value = mock_layers_instance
            
            backbone = HiggsAudioBackbone(sample_higgs_audio_config)
            
            input_ids = mock_tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
            kv_cache_params = MagicMock()
            kv_cache_params.fill_none_tensor_list = MagicMock()
            
            output = backbone.forward(
                input_ids=input_ids,
                kv_cache_params=kv_cache_params,
                use_cache=False
            )
            
            assert output.shape == (2, 4, 512)  # Updated from embedding


@pytest.mark.integration
class TestHiggsAudioForCausalLM:
    """Test the complete TTS model."""
    
    def test_initialization(self, sample_higgs_audio_config, mock_layers):
        """Test complete model initialization."""
        with patch('tensorrt_llm.models.higgs_audio.model.HiggsAudioBackbone') as mock_backbone, \
             patch('tensorrt_llm.layers.ColumnLinear', mock_layers['ColumnLinear']), \
             patch('tensorrt_llm._utils.pad_vocab_size', return_value=32000):
            
            mock_backbone.return_value = MagicMock()
            
            model = HiggsAudioForCausalLM(sample_higgs_audio_config)
            
            assert hasattr(model, 'transformer')
            assert hasattr(model, 'lm_head') or sample_higgs_audio_config.mapping.is_last_pp_rank() is False
    
    def test_forward_with_audio_mask(self, sample_higgs_audio_config, mock_tensor, mock_layers):
        """Test forward pass with audio output mask routing."""
        with patch('tensorrt_llm.models.higgs_audio.model.HiggsAudioBackbone') as mock_backbone, \
             patch('tensorrt_llm.layers.ColumnLinear', mock_layers['ColumnLinear']), \
             patch('tensorrt_llm._utils.pad_vocab_size', return_value=32000), \
             patch('tensorrt_llm.models.modeling_utils.DecoderModelForCausalLM') as mock_base:
            
            # Setup mocks
            mock_backbone_instance = MagicMock()
            mock_backbone_instance.set_audio_out_mask = MagicMock()
            mock_backbone.return_value = mock_backbone_instance
            
            mock_base.__init__ = MagicMock(return_value=None)
            mock_base.forward = MagicMock(return_value=mock_tensor(np.random.randn(2, 8, 32000)))
            
            model = HiggsAudioForCausalLM(sample_higgs_audio_config)
            model.transformer = mock_backbone_instance
            
            input_ids = mock_tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
            audio_mask = mock_tensor(np.array([[True, False, True, False], [False, True, False, True]]))
            
            output = model.forward(input_ids=input_ids, audio_out_mask=audio_mask)
            
            # Verify audio mask was set on transformer
            mock_backbone_instance.set_audio_out_mask.assert_called_once_with(audio_mask)
    
    def test_prepare_inputs_for_generation_text_mode(self, sample_higgs_audio_config, mock_tensor):
        """Test input preparation for text generation mode."""
        with patch('tensorrt_llm.models.higgs_audio.model.HiggsAudioBackbone'), \
             patch('tensorrt_llm.layers.ColumnLinear'), \
             patch('tensorrt_llm._utils.pad_vocab_size', return_value=32000), \
             patch('tensorrt_llm.models.modeling_utils.DecoderModelForCausalLM'):
            
            model = HiggsAudioForCausalLM(sample_higgs_audio_config)
            
            # Mock generation mode as TEXT
            with patch.object(model, 'is_audio_generation_mode', return_value=False), \
                 patch('tensorrt_llm.models.modeling_utils.DecoderModelForCausalLM.prepare_inputs_for_generation') as mock_prepare:
                
                mock_prepare.return_value = {'input_ids': mock_tensor([[1, 2, 3]])}
                
                input_ids = mock_tensor([[1, 2, 3]])
                inputs = model.prepare_inputs_for_generation(input_ids)
                
                # Should not add audio-specific inputs
                assert 'audio_init_mode' not in inputs
                assert 'audio_stream_mode' not in inputs


@pytest.mark.integration 
@pytest.mark.cuda_graphs
class TestCudaGraphIntegration:
    """Test CUDA graph integration with TTS components."""
    
    def test_cuda_graph_manager_initialization(self, sample_higgs_audio_config):
        """Test CUDA graph manager initialization."""
        sample_higgs_audio_config.cuda_graph_enable = True
        
        with patch('tensorrt_llm.models.higgs_audio.model.CudaGraphManager') as mock_manager, \
             patch('tensorrt_llm.models.higgs_audio.model.CUDA_GRAPHS_AVAILABLE', True), \
             patch('tensorrt_llm.models.higgs_audio.model.GenerationModeManager'):
            
            mock_manager.return_value = MagicMock()
            
            model = HiggsAudioModelForCausalLM(sample_higgs_audio_config)
            
            assert model.cuda_graph_manager is not None
            mock_manager.assert_called_once()
    
    def test_cuda_graph_prewarming(self, sample_higgs_audio_config):
        """Test CUDA graph prewarming process."""
        sample_higgs_audio_config.cuda_graph_enable = True
        sample_higgs_audio_config.cuda_graph_hrewarm = True
        
        with patch('tensorrt_llm.models.higgs_audio.model.CudaGraphManager') as mock_manager, \
             patch('tensorrt_llm.models.higgs_audio.model.CUDA_GRAPHS_AVAILABLE', True), \
             patch('tensorrt_llm.models.higgs_audio.model.GenerationModeManager'):
            
            mock_graph_manager = MagicMock()
            mock_graph_manager.prewarm_graphs = MagicMock()
            mock_manager.return_value = mock_graph_manager
            
            model = HiggsAudioModelForCausalLM(sample_higgs_audio_config)
            
            # Verify prewarming was called
            mock_graph_manager.prewarm_graphs.assert_called_once()
    
    def test_cuda_graph_fallback_on_error(self, sample_higgs_audio_config):
        """Test graceful fallback when CUDA graphs fail."""
        sample_higgs_audio_config.cuda_graph_enable = True
        
        with patch('tensorrt_llm.models.higgs_audio.model.CudaGraphManager') as mock_manager, \
             patch('tensorrt_llm.models.higgs_audio.model.CUDA_GRAPHS_AVAILABLE', True), \
             patch('tensorrt_llm.models.higgs_audio.model.GenerationModeManager'), \
             patch('warnings.warn') as mock_warn:
            
            mock_manager.side_effect = Exception("CUDA graph initialization failed")
            
            model = HiggsAudioModelForCausalLM(sample_higgs_audio_config)
            
            # Should fallback gracefully
            assert model.cuda_graph_manager is None
            mock_warn.assert_called()


@pytest.mark.integration
@pytest.mark.generation_modes
class TestGenerationModeIntegration:
    """Test generation mode management integration."""
    
    def test_mode_transition_with_state_preservation(self, sample_higgs_audio_config):
        """Test mode transitions preserve necessary state."""
        with patch('tensorrt_llm.models.higgs_audio.model.GenerationModeManager') as mock_manager:
            mock_mode_manager = MagicMock()
            mock_manager.return_value = mock_mode_manager
            mock_mode_manager.transition_to_mode.return_value = True
            
            model = HiggsAudioModelForCausalLM(sample_higgs_audio_config)
            
            # Test transition with state preservation
            model.set_generation_mode(
                GenerationMode.AUDIO_INIT,
                validation_level='comprehensive',
                preserve_context=True
            )
            
            mock_mode_manager.transition_to_mode.assert_called_with(
                target_mode=GenerationMode.AUDIO_INIT,
                validation_level='comprehensive',
                preserve_context=True
            )
    
    def test_audio_generation_state_tracking(self, sample_higgs_audio_config):
        """Test comprehensive audio generation state tracking."""
        with patch('tensorrt_llm.models.higgs_audio.model.GenerationModeManager') as mock_manager:
            mock_mode_manager = MagicMock()
            mock_manager.return_value = mock_mode_manager
            mock_mode_manager.is_audio_generation_active.return_value = True
            
            # Mock generation state with comprehensive tracking
            mock_state = MagicMock()
            mock_state.generated_tokens = 0
            mock_state.current_position = 10
            mock_state.current_codebook_index = 0
            mock_state.delay_pattern_offset = 0
            mock_state.codebook_states = {}
            mock_state.latency_history = []
            mock_mode_manager.get_generation_state.return_value = mock_state
            mock_mode_manager.enable_performance_monitoring = True
            
            model = HiggsAudioModelForCausalLM(sample_higgs_audio_config)
            
            # Update state multiple times
            for token_id in [100, 101, 102]:
                model.update_audio_generation_state(token_id)
            
            # Verify comprehensive state tracking
            assert mock_state.generated_tokens == 3
            assert mock_state.current_position == 13
            assert len(mock_state.latency_history) == 3
    
    def test_mode_specific_configuration_application(self, sample_higgs_audio_config):
        """Test that mode-specific configurations are properly applied."""
        with patch('tensorrt_llm.models.higgs_audio.model.GenerationModeManager') as mock_manager:
            mock_mode_manager = MagicMock()
            mock_manager.return_value = mock_mode_manager
            
            # Set up mode-specific configurations
            sample_higgs_audio_config.text_mode_temperature = 0.8
            sample_higgs_audio_config.audio_init_mode_warmup_tokens = 10
            sample_higgs_audio_config.audio_progress_mode_chunk_size = 32
            
            model = HiggsAudioModelForCausalLM(sample_higgs_audio_config)
            
            # Verify mode manager was initialized with mode configurations
            call_args = mock_manager.call_args
            config_param = call_args[1]['config']
            
            assert 'text_mode' in config_param
            assert 'audio_init_mode' in config_param
            assert 'audio_progress_mode' in config_param


@pytest.mark.integration
@pytest.mark.performance
class TestTTSOptimizationIntegration:
    """Test integration of TTS-specific optimizations."""
    
    def test_delay_pattern_integration(self, sample_higgs_audio_config):
        """Test delay pattern integration with model components."""
        sample_higgs_audio_config.use_delay_pattern = True
        sample_higgs_audio_config.audio_delay_pattern_strategy = 'linear'
        
        with patch('tensorrt_llm.models.higgs_audio.model.GenerationModeManager'):
            model = HiggsAudioModelForCausalLM(sample_higgs_audio_config)
            
            # Verify delay pattern configuration is accessible
            assert model.config.use_delay_pattern is True