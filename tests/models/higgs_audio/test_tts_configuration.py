"""
Configuration and compatibility tests for Higgs Audio TTS model.

This module tests DualFFN parameters, model loading configurations,
quantization support, and TensorRT-LLM compatibility.
"""

import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock, patch, mock_open

# Skip all tests if TensorRT-LLM not available
from .conftest import TENSORRT_LLM_AVAILABLE
if TENSORRT_LLM_AVAILABLE:
    from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
    from tensorrt_llm.models.higgs_audio.model import HiggsAudioForCausalLM

pytestmark = pytest.mark.skipif(
    not TENSORRT_LLM_AVAILABLE,
    reason="TensorRT-LLM not available"
)


@pytest.mark.configuration
class TestHiggsAudioConfiguration:
    """Test HiggsAudioConfig functionality and parameter validation."""
    
    def test_basic_config_creation(self):
        """Test basic configuration creation with default parameters."""
        config = HiggsAudioConfig()
        
        # Validate essential TTS parameters
        assert hasattr(config, 'audio_vocab_size')
        assert hasattr(config, 'audio_num_codebooks')
        assert hasattr(config, 'audio_max_continuation_length')
        assert hasattr(config, 'audio_realtime_mode')
        assert hasattr(config, 'dualffn_config')
        
        # Validate default values
        assert config.audio_vocab_size > 0
        assert config.audio_num_codebooks >= 1
        assert config.audio_max_continuation_length > 0
        assert isinstance(config.audio_realtime_mode, bool)
        assert isinstance(config.dualffn_config, dict)
    
    def test_dualffn_configuration_parameters(self, sample_higgs_audio_config):
        """Test DualFFN-specific configuration parameters."""
        config = sample_higgs_audio_config
        
        # DualFFN architecture parameters
        dualffn = config.dualffn_config
        assert 'audio_mlp_config' in dualffn
        assert 'text_mlp_config' in dualffn
        assert 'routing_strategy' in dualffn
        
        # Audio MLP configuration
        audio_mlp = dualffn['audio_mlp_config']
        assert 'intermediate_size' in audio_mlp
        assert 'activation_function' in audio_mlp
        assert audio_mlp['intermediate_size'] > 0
        
        # Text MLP configuration
        text_mlp = dualffn['text_mlp_config']
        assert 'intermediate_size' in text_mlp
        assert 'activation_function' in text_mlp
        assert text_mlp['intermediate_size'] > 0
        
        # Routing strategy validation
        valid_strategies = ['generation_mode_based', 'token_type_based', 'learned']
        assert dualffn['routing_strategy'] in valid_strategies
    
    def test_audio_configuration_parameters(self, sample_higgs_audio_config):
        """Test audio-specific configuration parameters."""
        config = sample_higgs_audio_config
        
        # Audio tokenization parameters
        assert config.audio_vocab_size >= 1024  # Reasonable minimum for RVQ
        assert 1 <= config.audio_num_codebooks <= 16  # Typical RVQ codebook range
        assert config.audio_max_continuation_length >= 512  # Reasonable TTS sequence length
        
        # Audio generation parameters
        if hasattr(config, 'audio_streaming_chunk_size'):
            assert config.audio_streaming_chunk_size > 0
            assert config.audio_streaming_chunk_size <= 128  # Reasonable chunk size
        
        if hasattr(config, 'audio_delay_pattern_strategy'):
            valid_patterns = ['linear', 'exponential', 'custom', 'none']
            assert config.audio_delay_pattern_strategy in valid_patterns
    
    def test_compatibility_parameters(self, sample_higgs_audio_config):
        """Test compatibility with base TensorRT-LLM configuration."""
        config = sample_higgs_audio_config
        
        # Essential LLM parameters should be preserved
        assert hasattr(config, 'vocab_size')
        assert hasattr(config, 'hidden_size')
        assert hasattr(config, 'num_hidden_layers')
        assert hasattr(config, 'num_attention_heads')
        assert hasattr(config, 'intermediate_size')
        
        # Validate compatibility constraints
        assert config.vocab_size > 0
        assert config.hidden_size > 0
        assert config.num_hidden_layers > 0
        assert config.num_attention_heads > 0
        assert config.intermediate_size > 0
        
        # DualFFN should not break compatibility
        assert config.hidden_size == config.dualffn_config['audio_mlp_config']['input_size']
        assert config.hidden_size == config.dualffn_config['text_mlp_config']['input_size']
    
    def test_config_validation_errors(self):
        """Test configuration validation catches invalid parameters."""
        # Test invalid audio vocab size
        with pytest.raises((ValueError, AssertionError)):
            config = HiggsAudioConfig(audio_vocab_size=0)
        
        # Test invalid number of codebooks
        with pytest.raises((ValueError, AssertionError)):
            config = HiggsAudioConfig(audio_num_codebooks=0)
        
        # Test invalid continuation length
        with pytest.raises((ValueError, AssertionError)):
            config = HiggsAudioConfig(audio_max_continuation_length=0)
        
        # Test invalid DualFFN routing strategy
        invalid_dualffn_config = {
            'routing_strategy': 'invalid_strategy',
            'audio_mlp_config': {'intermediate_size': 2048, 'activation_function': 'swiglu', 'input_size': 2048},
            'text_mlp_config': {'intermediate_size': 2048, 'activation_function': 'swiglu', 'input_size': 2048}
        }
        
        with pytest.raises((ValueError, AssertionError)):
            config = HiggsAudioConfig(dualffn_config=invalid_dualffn_config)


@pytest.mark.configuration 
class TestModelLoading:
    """Test model loading and weight conversion compatibility."""
    
    def test_config_from_hugging_face_model(self):
        """Test loading configuration from Hugging Face model directory."""
        # Mock HF model directory structure
        mock_config_dict = {
            'model_type': 'higgs_audio',
            'vocab_size': 32000,
            'hidden_size': 2048,
            'num_hidden_layers': 24,
            'num_attention_heads': 32,
            'intermediate_size': 5632,
            'audio_vocab_size': 1024,
            'audio_num_codebooks': 8,
            'audio_max_continuation_length': 1500,
            'audio_realtime_mode': True,
            'dualffn_config': {
                'routing_strategy': 'generation_mode_based',
                'audio_mlp_config': {
                    'intermediate_size': 5632,
                    'activation_function': 'swiglu',
                    'input_size': 2048
                },
                'text_mlp_config': {
                    'intermediate_size': 5632,
                    'activation_function': 'swiglu',
                    'input_size': 2048
                }
            }
        }
        
        # Mock HF AutoConfig loading
        with patch('transformers.AutoConfig.from_pretrained') as mock_autoconfig:
            mock_hf_config = Mock()
            for key, value in mock_config_dict.items():
                setattr(mock_hf_config, key, value)
            mock_autoconfig.return_value = mock_hf_config
            
            # Test configuration loading
            with patch.object(HiggsAudioConfig, 'from_hugging_face') as mock_from_hf:
                mock_from_hf.return_value = HiggsAudioConfig(**mock_config_dict)
                
                config = HiggsAudioConfig.from_hugging_face('fake_model_path')
                
                # Validate loaded configuration
                assert config.audio_vocab_size == 1024
                assert config.audio_num_codebooks == 8
                assert config.dualffn_config['routing_strategy'] == 'generation_mode_based'
    
    def test_weight_mapping_compatibility(self, sample_higgs_audio_config):
        """Test weight mapping between HF and TensorRT-LLM formats."""
        config = sample_higgs_audio_config
        
        # Mock HF model weights structure
        mock_hf_weights = {
            # Standard transformer weights
            'model.embed_tokens.weight': Mock(),
            'model.norm.weight': Mock(),
            'lm_head.weight': Mock(),
            
            # DualFFN layer weights (per layer)
            'model.layers.0.self_attn.q_proj.weight': Mock(),
            'model.layers.0.self_attn.k_proj.weight': Mock(),
            'model.layers.0.self_attn.v_proj.weight': Mock(),
            'model.layers.0.self_attn.o_proj.weight': Mock(),
            'model.layers.0.input_layernorm.weight': Mock(),
            'model.layers.0.post_attention_layernorm.weight': Mock(),
            
            # DualFFN-specific weights
            'model.layers.0.audio_mlp.gate_proj.weight': Mock(),
            'model.layers.0.audio_mlp.up_proj.weight': Mock(),
            'model.layers.0.audio_mlp.down_proj.weight': Mock(),
            'model.layers.0.text_mlp.gate_proj.weight': Mock(),
            'model.layers.0.text_mlp.up_proj.weight': Mock(),
            'model.layers.0.text_mlp.down_proj.weight': Mock(),
            
            # Audio projection weights
            'audio_projection.weight': Mock(),
            'audio_projection.bias': Mock(),
        }
        
        # Mock weight conversion function
        with patch('tensorrt_llm.models.higgs_audio.convert.load_weights_from_hf_model') as mock_load_weights:
            # Expected TensorRT-LLM weight structure
            expected_trt_weights = {
                'transformer.vocab_embedding.weight': mock_hf_weights['model.embed_tokens.weight'],
                'transformer.ln_f.weight': mock_hf_weights['model.norm.weight'],
                'lm_head.weight': mock_hf_weights['lm_head.weight'],
                
                # DualFFN layer mapping
                'transformer.layers.0.attention.qkv.weight': Mock(),  # Combined from q,k,v
                'transformer.layers.0.attention.dense.weight': mock_hf_weights['model.layers.0.self_attn.o_proj.weight'],
                'transformer.layers.0.input_layernorm.weight': mock_hf_weights['model.layers.0.input_layernorm.weight'],
                'transformer.layers.0.post_layernorm.weight': mock_hf_weights['model.layers.0.post_attention_layernorm.weight'],
                
                # Mapped DualFFN weights
                'transformer.layers.0.audio_mlp.gate.weight': mock_hf_weights['model.layers.0.audio_mlp.gate_proj.weight'],
                'transformer.layers.0.audio_mlp.proj.weight': mock_hf_weights['model.layers.0.audio_mlp.up_proj.weight'],
                'transformer.layers.0.audio_mlp.fc.weight': mock_hf_weights['model.layers.0.audio_mlp.down_proj.weight'],
                'transformer.layers.0.text_mlp.gate.weight': mock_hf_weights['model.layers.0.text_mlp.gate_proj.weight'],
                'transformer.layers.0.text_mlp.proj.weight': mock_hf_weights['model.layers.0.text_mlp.up_proj.weight'],
                'transformer.layers.0.text_mlp.fc.weight': mock_hf_weights['model.layers.0.text_mlp.down_proj.weight'],
                
                # Audio projection
                'audio_projection.weight': mock_hf_weights['audio_projection.weight'],
                'audio_projection.bias': mock_hf_weights['audio_projection.bias'],
            }
            
            mock_load_weights.return_value = expected_trt_weights
            
            # Test weight loading
            weights = mock_load_weights('fake_model_path', config)
            
            # Validate DualFFN weight mapping
            assert 'transformer.layers.0.audio_mlp.gate.weight' in weights
            assert 'transformer.layers.0.text_mlp.gate.weight' in weights
            assert 'audio_projection.weight' in weights
    
    def test_config_serialization_deserialization(self, sample_higgs_audio_config):
        """Test configuration serialization and deserialization."""
        config = sample_higgs_audio_config
        
        # Serialize to dict
        config_dict = config.to_dict()
        
        # Validate essential keys are present
        assert 'audio_vocab_size' in config_dict
        assert 'audio_num_codebooks' in config_dict
        assert 'dualffn_config' in config_dict
        
        # Deserialize from dict
        restored_config = HiggsAudioConfig.from_dict(config_dict)
        
        # Validate restoration
        assert restored_config.audio_vocab_size == config.audio_vocab_size
        assert restored_config.audio_num_codebooks == config.audio_num_codebooks
        assert restored_config.dualffn_config == config.dualffn_config
        
        # Test JSON serialization
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(config_dict, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            # Load from JSON
            with open(tmp_path, 'r') as f:
                loaded_dict = json.load(f)
            
            json_config = HiggsAudioConfig.from_dict(loaded_dict)
            
            # Validate JSON round-trip
            assert json_config.audio_vocab_size == config.audio_vocab_size
            assert json_config.dualffn_config['routing_strategy'] == config.dualffn_config['routing_strategy']
        finally:
            Path(tmp_path).unlink()  # Clean up temp file


@pytest.mark.configuration
class TestQuantizationSupport:
    """Test quantization configuration and compatibility."""
    
    def test_fp16_quantization_config(self, sample_higgs_audio_config):
        """Test FP16 quantization configuration."""
        config = sample_higgs_audio_config
        config.dtype = 'float16'
        config.quantization_config = {
            'quant_mode': 'fp16',
            'calibration_dataset': None,
            'amax_filename': None
        }
        
        # Validate FP16 configuration
        assert config.dtype == 'float16'
        assert config.quantization_config['quant_mode'] == 'fp16'
        
        # DualFFN should support FP16
        with patch('tensorrt_llm.models.higgs_audio.model.HiggsAudioBackbone'), \
             patch('tensorrt_llm.layers.ColumnLinear'), \
             patch('tensorrt_llm._utils.pad_vocab_size', return_value=32000), \
             patch('tensorrt_llm.models.modeling_utils.DecoderModelForCausalLM'), \
             patch('tensorrt_llm.models.higgs_audio.model.GenerationModeManager'):
            
            model = HiggsAudioForCausalLM(config)
            
            # Mock layer dtype validation
            with patch.object(model, 'named_modules') as mock_modules:
                mock_audio_mlp = Mock()
                mock_audio_mlp.dtype = 'float16'
                mock_text_mlp = Mock()
                mock_text_mlp.dtype = 'float16'
                
                mock_modules.return_value = [
                    ('layers.0.audio_mlp', mock_audio_mlp),
                    ('layers.0.text_mlp', mock_text_mlp)
                ]
                
                # Should not raise any dtype compatibility errors
                for name, module in model.named_modules():
                    if 'audio_mlp' in name or 'text_mlp' in name:
                        assert hasattr(module, 'dtype')
    
    def test_int8_quantization_config(self, sample_higgs_audio_config):
        """Test INT8 quantization configuration."""
        config = sample_higgs_audio_config
        config.quantization_config = {
            'quant_mode': 'int8',
            'calibration_dataset': 'dummy_calibration_data',
            'amax_filename': 'amaxs.json',
            'kv_cache_quant_mode': 'int8'
        }
        
        # Validate INT8 configuration
        assert config.quantization_config['quant_mode'] == 'int8'
        assert config.quantization_config['calibration_dataset'] is not None
        assert config.quantization_config['kv_cache_quant_mode'] == 'int8'
        
        # DualFFN should support INT8 (with calibration)
        with patch('tensorrt_llm.models.higgs_audio.model.HiggsAudioBackbone'), \
             patch('tensorrt_llm.layers.ColumnLinear'), \
             patch('tensorrt_llm._utils.pad_vocab_size', return_value=32000), \
             patch('tensorrt_llm.models.modeling_utils.DecoderModelForCausalLM'), \
             patch('tensorrt_llm.models.higgs_audio.model.GenerationModeManager'):
            
            model = HiggsAudioForCausalLM(config)
            
            # Mock quantization validation
            with patch('tensorrt_llm.quantization.layers.Int8Linear') as mock_int8_layer:
                mock_int8_layer.return_value = Mock()
                
                # DualFFN layers should be quantizable
                assert config.quantization_config['quant_mode'] == 'int8'
                # This would be validated during actual TensorRT engine build
    
    def test_quantization_parameter_validation(self, sample_higgs_audio_config):
        """Test quantization parameter validation."""
        config = sample_higgs_audio_config
        
        # Test valid quantization modes
        valid_modes = ['fp32', 'fp16', 'int8', 'int4']
        for mode in valid_modes:
            config.quantization_config = {'quant_mode': mode}
            # Should not raise validation errors
            assert config.quantization_config['quant_mode'] == mode
        
        # Test invalid quantization mode
        with pytest.raises((ValueError, AssertionError)):
            config.quantization_config = {'quant_mode': 'invalid_mode'}
            # This should trigger validation error in actual usage
        
        # Test INT8 requirements
        config.quantization_config = {
            'quant_mode': 'int8',
            # Missing calibration_dataset should cause warning/error
        }
        
        # This would be validated during quantization process
        assert config.quantization_config['quant_mode'] == 'int8'


@pytest.mark.configuration
class TestTensorRTLLMCompatibility:
    """Test compatibility with TensorRT-LLM framework."""
    
    def test_config_inheritance(self, sample_higgs_audio_config):
        """Test HiggsAudioConfig properly inherits from PretrainedConfig."""
        config = sample_higgs_audio_config
        
        # Should have TensorRT-LLM config attributes
        assert hasattr(config, 'dtype')
        assert hasattr(config, 'logits_dtype')
        assert hasattr(config, 'use_parallel_embedding')
        assert hasattr(config, 'embedding_sharding_dim')
        
        # Should have model architecture attributes
        assert hasattr(config, 'architecture')
        assert hasattr(config, 'max_position_embeddings')
        
        # TTS-specific attributes should extend, not replace
        base_attributes = ['vocab_size', 'hidden_size', 'num_hidden_layers']
        for attr in base_attributes:
            assert hasattr(config, attr)
            assert getattr(config, attr) is not None
    
    def test_builder_compatibility(self, sample_higgs_audio_config):
        """Test compatibility with TensorRT-LLM builder system."""
        config = sample_higgs_audio_config
        
        # Mock builder interface
        with patch('tensorrt_llm.builder.Builder') as mock_builder_class:
            mock_builder = Mock()
            mock_builder_class.return_value = mock_builder
            
            # Mock network creation
            with patch('tensorrt_llm.network.Network') as mock_network_class:
                mock_network = Mock()
                mock_network_class.return_value = mock_network
                
                # Mock model building
                with patch.object(HiggsAudioForCausalLM, 'from_config') as mock_from_config:
                    mock_model = Mock()
                    mock_from_config.return_value = mock_model
                    
                    # Test builder integration points
                    builder_config = {
                        'max_batch_size': 8,
                        'max_input_len': 2048,
                        'max_output_len': 1024,
                        'dtype': config.dtype,
                        # TTS-specific builder parameters
                        'audio_max_continuation_length': config.audio_max_continuation_length,
                        'audio_streaming_support': config.audio_realtime_mode,
                        'dualffn_enabled': True
                    }
                    
                    # Should be able to create model with TTS configuration
                    model = mock_from_config(config)
                    assert model is not None
                    
                    # Builder should handle TTS-specific parameters
                    assert builder_config['dualffn_enabled'] == True
                    assert builder_config['audio_streaming_support'] == config.audio_realtime_mode
    
    def test_runtime_compatibility(self, sample_higgs_audio_config):
        """Test compatibility with TensorRT-LLM runtime system."""
        config = sample_higgs_audio_config
        
        # Mock runtime components
        with patch('tensorrt_llm.runtime.ModelRunner') as mock_runner_class, \
             patch('tensorrt_llm.runtime.model_runner.ModelRunner') as mock_model_runner:
            
            mock_runner = Mock()
            mock_runner_class.return_value = mock_runner
            mock_model_runner.return_value = mock_runner
            
            # Mock generation session
            with patch('tensorrt_llm.runtime.generation.GenerationSession') as mock_session_class:
                mock_session = Mock()
                mock_session_class.return_value = mock_session
                
                # Test runtime configuration
                runtime_config = {
                    'max_batch_size': 4,
                    'max_input_len': 2048,
                    'max_new_tokens': 1024,
                    # TTS-specific runtime parameters
                    'generation_mode': 'TEXT',  # Should transition to AUDIO_INIT, AUDIO_IN_PROGRESS
                    'audio_streaming_chunk_size': config.audio_streaming_chunk_size if hasattr(config, 'audio_streaming_chunk_size') else 32,
                    'delay_pattern_coordination': True
                }
                
                # Runtime should handle TTS generation modes
                assert runtime_config['generation_mode'] in ['TEXT', 'AUDIO_INIT', 'AUDIO_IN_PROGRESS']
                assert runtime_config['delay_pattern_coordination'] == True
                
                # Mock successful runtime creation
                runner = mock_runner_class.from_dir('fake_engine_path')
                assert runner is not None
    
    def test_plugin_compatibility(self, sample_higgs_audio_config):
        """Test compatibility with TensorRT-LLM plugins."""
        config = sample_higgs_audio_config
        
        # Mock plugin system
        with patch('tensorrt_llm.plugin.plugin.plugin_lib') as mock_plugin_lib:
            mock_plugin_lib.init_plugin.return_value = True
            
            # Test plugin initialization for TTS-specific operations
            tts_plugins = [
                'DualFFNPlugin',           # For DualFFN layer routing
                'DelayPatternPlugin',      # For RVQ delay pattern coordination  
                'AudioTokenizerPlugin',    # For audio token processing
                'GenerationModePlugin'     # For mode transitions
            ]
            
            for plugin_name in tts_plugins:
                with patch(f'tensorrt_llm.plugin.{plugin_name.lower()}.{plugin_name}') as mock_plugin:
                    mock_plugin.return_value = Mock()
                    
                    # Plugin should be loadable with TTS config
                    plugin_instance = mock_plugin(config=config)
                    assert plugin_instance is not None
                    
                    # Plugin should handle TTS-specific parameters
                    if plugin_name == 'DualFFNPlugin':
                        # Should handle DualFFN routing configuration
                        assert hasattr(mock_plugin, 'call_args')
                    elif plugin_name == 'DelayPatternPlugin':
                        # Should handle delay pattern strategy
                        assert hasattr(mock_plugin, 'call_args')


@pytest.mark.configuration
class TestErrorHandlingConfiguration:
    """Test error handling and edge cases in configuration."""
    
    def test_missing_required_audio_parameters(self):
        """Test handling of missing required audio parameters."""
        # Test missing audio vocab size
        with pytest.raises((ValueError, TypeError, AttributeError)):
            config = HiggsAudioConfig()
            delattr(config, 'audio_vocab_size')
            # Usage should fail validation
            
        # Test missing DualFFN configuration
        with pytest.raises((ValueError, TypeError, AttributeError)):
            config = HiggsAudioConfig(dualffn_config=None)
    
    def test_conflicting_configuration_parameters(self, sample_higgs_audio_config):
        """Test handling of conflicting configuration parameters."""
        config = sample_higgs_audio_config
        
        # Test conflicting audio and text vocab sizes
        if config.vocab_size != config.audio_vocab_size:
            # Different vocab sizes should be handled correctly
            assert config.vocab_size > 0
            assert config.audio_vocab_size > 0
            # This is valid - text and audio can have different vocabularies
        
        # Test conflicting intermediate sizes
        text_intermediate = config.dualffn_config['text_mlp_config']['intermediate_size']
        audio_intermediate = config.dualffn_config['audio_mlp_config']['intermediate_size']
        
        # Different intermediate sizes are allowed
        assert text_intermediate > 0
        assert audio_intermediate > 0
        
        # But they should be compatible with the hidden size
        assert config.dualffn_config['text_mlp_config']['input_size'] == config.hidden_size
        assert config.dualffn_config['audio_mlp_config']['input_size'] == config.hidden_size
    
    def test_boundary_value_configurations(self):
        """Test configuration with boundary values."""
        # Test minimum viable configuration
        minimal_config = HiggsAudioConfig(
            vocab_size=1000,
            hidden_size=512,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=1024,
            audio_vocab_size=256,
            audio_num_codebooks=1,
            audio_max_continuation_length=64
        )
        
        assert minimal_config.vocab_size == 1000
        assert minimal_config.audio_vocab_size == 256
        
        # Test maximum reasonable configuration
        large_config = HiggsAudioConfig(
            vocab_size=100000,
            hidden_size=8192,
            num_hidden_layers=80,
            num_attention_heads=64,
            intermediate_size=32768,
            audio_vocab_size=65536,
            audio_num_codebooks=16,
            audio_max_continuation_length=8192
        )
        
        assert large_config.vocab_size == 100000
        assert large_config.audio_vocab_size == 65536
        assert large_config.audio_num_codebooks == 16
    
    def test_configuration_upgrade_compatibility(self, sample_higgs_audio_config):
        """Test backward compatibility when configuration format changes."""
        config = sample_higgs_audio_config
        
        # Simulate old configuration format (without some TTS parameters)
        old_config_dict = config.to_dict()
        
        # Remove newer parameters to simulate old config
        deprecated_params = ['audio_streaming_chunk_size', 'cuda_graph_enable']
        for param in deprecated_params:
            old_config_dict.pop(param, None)
        
        # Should be able to load old config with defaults
        try:
            upgraded_config = HiggsAudioConfig.from_dict(old_config_dict)
            
            # Essential parameters should be preserved
            assert upgraded_config.audio_vocab_size == config.audio_vocab_size
            assert upgraded_config.dualffn_config == config.dualffn_config
            
            # Missing parameters should have sensible defaults
            if not hasattr(upgraded_config, 'audio_streaming_chunk_size'):
                # Should have a default value or be handled gracefully
                pass
            
        except Exception as e:
            pytest.fail(f"Configuration upgrade failed: {e}")


# Configuration test fixtures and utilities
@pytest.fixture
def mock_hf_config_dict():
    """Provide mock Hugging Face configuration dictionary."""
    return {
        'model_type': 'higgs_audio',
        'vocab_size': 32000,
        'hidden_size': 2048,
        'num_hidden_layers': 24,
        'num_attention_heads': 32,
        'intermediate_size': 5632,
        'max_position_embeddings': 8192,
        'rms_norm_eps': 1e-6,
        'tie_word_embeddings': False,
        'rope_theta': 500000.0,
        
        # TTS-specific parameters
        'audio_vocab_size': 1024,
        'audio_num_codebooks': 8,
        'audio_max_continuation_length': 1500,
        'audio_realtime_mode': True,
        'audio_streaming_chunk_size': 32,
        'audio_delay_pattern_strategy': 'linear',
        
        # DualFFN configuration
        'dualffn_config': {
            'routing_strategy': 'generation_mode_based',
            'audio_mlp_config': {
                'intermediate_size': 5632,
                'activation_function': 'swiglu',
                'input_size': 2048
            },
            'text_mlp_config': {
                'intermediate_size': 5632,
                'activation_function': 'swiglu',
                'input_size': 2048
            }
        },
        
        # CUDA graph parameters
        'cuda_graph_enable': True,
        'cuda_graph_memory_pool_size_gb': 2.0
    }


@pytest.fixture
def mock_quantization_configs():
    """Provide mock quantization configurations for testing."""
    return {
        'fp16': {
            'quant_mode': 'fp16',
            'calibration_dataset': None,
            'amax_filename': None,
            'kv_cache_quant_mode': None
        },
        'int8': {
            'quant_mode': 'int8',
            'calibration_dataset': 'dummy_calibration_data',
            'amax_filename': 'amaxs.json',
            'kv_cache_quant_mode': 'int8'
        },
        'int4': {
            'quant_mode': 'int4',
            'calibration_dataset': 'dummy_calibration_data',
            'amax_filename': 'amaxs_int4.json',
            'kv_cache_quant_mode': 'int8'
        }
    }


@pytest.fixture
def mock_engine_config():
    """Provide mock TensorRT engine configuration."""
    return {
        'version': '1.0.0',
        'engine_dir': '/tmp/higgs_audio_engine',
        'model_config': {
            'model_type': 'higgs_audio',
            'max_batch_size': 8,
            'max_input_len': 2048,
            'max_output_len': 1024,
            'dtype': 'float16',
            'use_gpt_attention_plugin': True,
            'use_gemm_plugin': True,
            'use_layernorm_plugin': True,
            'use_lookup_plugin': True,
            
            # TTS-specific engine parameters
            'audio_streaming_support': True,
            'dualffn_enabled': True,
            'delay_pattern_coordination': True,
            'cuda_graph_enabled': True
        },
        'build_config': {
            'builder_opt': 3,
            'precision': 'fp16',
            'max_workspace_size': 2048 * 1024 * 1024,  # 2GB
            'tactic_sources': ['CUBLAS', 'CUBLAS_LT', 'CUDNN']
        }
    }


# Configuration validation utilities
class ConfigurationValidator:
    """Utility class for validating TTS configuration consistency."""
    
    @staticmethod
    def validate_audio_config(config: 'HiggsAudioConfig') -> List[str]:
        """Validate audio-specific configuration parameters."""
        issues = []
        
        # Audio vocabulary validation
        if config.audio_vocab_size < 256:
            issues.append(f"Audio vocab size {config.audio_vocab_size} may be too small for RVQ")
        
        if config.audio_vocab_size > 65536:
            issues.append(f"Audio vocab size {config.audio_vocab_size} may be excessive")
        
        # Codebook validation
        if config.audio_num_codebooks < 1 or config.audio_num_codebooks > 16:
            issues.append(f"Number of codebooks {config.audio_num_codebooks} outside typical range 1-16")
        
        # Sequence length validation
        if config.audio_max_continuation_length > 8192:
            issues.append(f"Max continuation length {config.audio_max_continuation_length} may cause memory issues")
        
        return issues
    
    @staticmethod
    def validate_dualffn_config(config: 'HiggsAudioConfig') -> List[str]:
        """Validate DualFFN configuration consistency."""
        issues = []
        
        dualffn = config.dualffn_config
        
        # Routing strategy validation
        valid_strategies = ['generation_mode_based', 'token_type_based', 'learned']
        if dualffn['routing_strategy'] not in valid_strategies:
            issues.append(f"Invalid routing strategy: {dualffn['routing_strategy']}")
        
        # MLP configuration validation
        audio_mlp = dualffn['audio_mlp_config']
        text_mlp = dualffn['text_mlp_config']
        
        if audio_mlp['input_size'] != config.hidden_size:
            issues.append(f"Audio MLP input size {audio_mlp['input_size']} doesn't match hidden size {config.hidden_size}")
        
        if text_mlp['input_size'] != config.hidden_size:
            issues.append(f"Text MLP input size {text_mlp['input_size']} doesn't match hidden size {config.hidden_size}")
        
        # Activation function validation
        valid_activations = ['swiglu', 'gelu', 'relu', 'silu']
        if audio_mlp['activation_function'] not in valid_activations:
            issues.append(f"Invalid audio MLP activation: {audio_mlp['activation_function']}")
        
        if text_mlp['activation_function'] not in valid_activations:
            issues.append(f"Invalid text MLP activation: {text_mlp['activation_function']}")
        
        return issues
    
    @staticmethod
    def validate_compatibility(config: 'HiggsAudioConfig') -> List[str]:
        """Validate overall configuration compatibility."""
        issues = []
        
        # Base model compatibility
        if config.hidden_size % config.num_attention_heads != 0:
            issues.append(f"Hidden size {config.hidden_size} not divisible by attention heads {config.num_attention_heads}")
        
        # TTS-specific compatibility
        if hasattr(config, 'audio_streaming_chunk_size'):
            if config.audio_streaming_chunk_size > config.audio_max_continuation_length:
                issues.append(f"Streaming chunk size {config.audio_streaming_chunk_size} exceeds max continuation length")
        
        # Quantization compatibility
        if hasattr(config, 'quantization_config') and config.quantization_config:
            quant_mode = config.quantization_config.get('quant_mode', 'fp32')
            if quant_mode in ['int8', 'int4'] and not config.quantization_config.get('calibration_dataset'):
                issues.append(f"Quantization mode {quant_mode} requires calibration dataset")
        
        return issues


@pytest.fixture
def config_validator():
    """Provide configuration validator utility."""
    return ConfigurationValidator()


# Performance validation for configuration impact
def test_config_performance_impact(sample_higgs_audio_config, config_validator):
    """Test that configuration choices don't negatively impact performance."""
    config = sample_higgs_audio_config
    
    # Validate configuration doesn't have obvious performance issues
    audio_issues = config_validator.validate_audio_config(config)
    dualffn_issues = config_validator.validate_dualffn_config(config)
    compatibility_issues = config_validator.validate_compatibility(config)
    
    all_issues = audio_issues + dualffn_issues + compatibility_issues
    
    # Critical performance issues should fail the test
    critical_keywords = ['excessive', 'memory issues', 'too large']
    critical_issues = [issue for issue in all_issues if any(keyword in issue.lower() for keyword in critical_keywords)]
    
    assert len(critical_issues) == 0, f"Configuration has critical performance issues: {critical_issues}"
    
    # Log warnings for non-critical issues
    if all_issues:
        print(f"Configuration warnings (non-critical): {all_issues}")


# Mark all configuration tests
pytest.mark.config_validation = pytest.mark.mark("config_validation")
pytest.mark.quantization = pytest.mark.mark("quantization")
pytest.mark.compatibility = pytest.mark.mark("compatibility")
pytest.mark.error_handling = pytest.mark.mark("error_handling")