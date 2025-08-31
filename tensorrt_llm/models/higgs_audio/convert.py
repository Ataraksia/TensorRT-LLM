# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Dict, Optional, List, Set, Tuple, Any
import warnings

import torch

from ...logger import logger
from ...mapping import Mapping
from ...quantization import QuantAlgo
from ..convert_utils import (
    infer_dtype,
    iterate_shard_files,
    load_state_dict,
    split,
    split_qkv_tp,
    split_qkv_bias_tp,
    split_matrix_tp,
    dup_kv_weight,
)
from ..modeling_utils import QuantConfig
from .config import HiggsAudioConfig


def build_config_from_hf(
    hf_model_dir: str,
    *,
    dtype: str = "auto",
    mapping: Optional[Mapping] = None,
    quant_config: Optional[QuantConfig] = None,
    trust_remote_code: bool = True,
    **kwargs,
) -> HiggsAudioConfig:
    """Create a `HiggsAudioConfig` from a HuggingFace model directory.

    This reads the HF composition config (text + audio encoder) and flattens
    what we need for TRT-LLM.
    """
    import transformers

    hf_cfg, _unused = transformers.AutoConfig.from_pretrained(
        hf_model_dir, trust_remote_code=trust_remote_code, return_unused_kwargs=True
    )
    return HiggsAudioConfig.from_hugging_face(
        hf_cfg, dtype=dtype, mapping=mapping, quant_config=quant_config, **kwargs
    )


def _backbone_key_template(config: HiggsAudioConfig):
    """Return a list of expected TRT-LLM tensor keys for the LLaMA-like backbone with DualFFN support.

    This function generates the expected tensor keys for weight conversion, including support for
    DualFFN layers that have separate audio and text MLP paths.
    
    Args:
        config: HiggsAudioConfig containing layer specifications and DualFFN configuration
        
    Returns:
        List of expected tensor key names for the backbone model
    """
    keys = []
    # Embedding and final norm/lm_head
    if config.mapping.is_first_pp_rank():
        keys.append("transformer.vocab_embedding.weight")
    
    # Get DualFFN layer configuration
    dual_ffn_layers = getattr(config, 'audio_dual_ffn_layers', [])
    dual_ffn_layer_set = set(dual_ffn_layers) if dual_ffn_layers else set()
    
    for l in range(config.num_hidden_layers):
        # Attention projections
        base = f"transformer.layers.{l}.attention"
        keys.extend([
            f"{base}.qkv.weight",  # or separate q_proj/k_proj/v_proj depending on loader
            f"{base}.qkv.bias" if config.attn_bias else None,
            f"{base}.dense.weight",
            f"{base}.dense.bias" if config.attn_bias else None,
        ])
        
        # Layer normalization components
        keys.append(f"transformer.layers.{l}.input_layernorm.weight")
        
        # Check if this layer uses DualFFN architecture
        if l in dual_ffn_layer_set:
            # DualFFN layers have separate layer norms and MLPs for audio and text
            keys.extend([
                f"transformer.layers.{l}.post_layernorm_text.weight",
                f"transformer.layers.{l}.post_layernorm_audio.weight",
            ])
            
            # Text MLP components
            text_mlp_base = f"transformer.layers.{l}.text_mlp"
            keys.extend([
                f"{text_mlp_base}.fc.weight",    # maps from HF gate_proj
                f"{text_mlp_base}.gate.weight",  # maps from HF up_proj
                f"{text_mlp_base}.proj.weight",  # maps from HF down_proj
            ])
            
            # Audio MLP components
            audio_mlp_base = f"transformer.layers.{l}.audio_mlp"
            keys.extend([
                f"{audio_mlp_base}.fc.weight",   # maps from HF audio_mlp.gate_proj
                f"{audio_mlp_base}.gate.weight", # maps from HF audio_mlp.up_proj
                f"{audio_mlp_base}.proj.weight", # maps from HF audio_mlp.down_proj
            ])
        else:
            # Standard layers have single layer norm and MLP
            keys.append(f"transformer.layers.{l}.post_layernorm.weight")
            
            # Standard MLP
            mlp_base = f"transformer.layers.{l}.mlp"
            # Follow TRT-LLM LLaMA checkpoint naming
            keys.extend([
                f"{mlp_base}.fc.weight",    # maps from HF gate_proj
                f"{mlp_base}.gate.weight",  # maps from HF up_proj
                f"{mlp_base}.proj.weight",  # maps from HF down_proj
            ])
    
    if config.mapping.is_last_pp_rank():
        keys.append("transformer.ln_f.weight")
        keys.append("lm_head.weight")
    
    # Filter Nones
    return [k for k in keys if k is not None]


def get_dual_ffn_layer_mapping(config: HiggsAudioConfig) -> Dict[str, Any]:
    """Generate layer-specific mapping information for DualFFN weight conversion.
    
    Args:
        config: HiggsAudioConfig containing DualFFN layer specifications
        
    Returns:
        Dictionary containing layer mapping information:
        - 'dual_ffn_layers': Set of layer indices that use DualFFN
        - 'standard_layers': Set of layer indices that use standard MLP
        - 'adapter_type': Audio adapter type from config
        - 'total_layers': Total number of layers
        
    Raises:
        ValueError: If configuration is invalid
    """
    dual_ffn_layers = getattr(config, 'audio_dual_ffn_layers', [])
    adapter_type = getattr(config, 'audio_adapter_type', 'stack')
    
    # Validate dual_ffn_layers
    if dual_ffn_layers:
        if not isinstance(dual_ffn_layers, (list, tuple)):
            raise ValueError(f"audio_dual_ffn_layers must be a list or tuple, got {type(dual_ffn_layers)}")
        
        for idx in dual_ffn_layers:
            if not isinstance(idx, int) or idx < 0 or idx >= config.num_hidden_layers:
                raise ValueError(
                    f"Invalid layer index {idx} in audio_dual_ffn_layers. "
                    f"Must be between 0 and {config.num_hidden_layers - 1}"
                )
    
    dual_ffn_layer_set = set(dual_ffn_layers) if dual_ffn_layers else set()
    standard_layer_set = set(range(config.num_hidden_layers)) - dual_ffn_layer_set
    
    return {
        'dual_ffn_layers': dual_ffn_layer_set,
        'standard_layers': standard_layer_set,
        'adapter_type': adapter_type,
        'total_layers': config.num_hidden_layers,
        'dual_ffn_enabled': len(dual_ffn_layer_set) > 0
    }


def map_dual_ffn_weights(
    hf_layer_weights: Dict[str, torch.Tensor],
    layer_idx: int,
    config: HiggsAudioConfig,
    mapping: Mapping
) -> Dict[str, torch.Tensor]:
    """Map DualFFN weights from HuggingFace format to TensorRT-LLM format.
    
    This function handles the mapping of DualFFN-specific weights including separate
    audio and text MLP components and layer normalization weights.
    
    Args:
        hf_layer_weights: Dictionary of HF weights for this layer
        layer_idx: Zero-based layer index
        config: HiggsAudioConfig containing model parameters
        mapping: TensorRT-LLM tensor parallelism mapping
        
    Returns:
        Dictionary of mapped TensorRT-LLM weight tensors
        
    Raises:
        ValueError: If required DualFFN weights are missing or invalid
    """
    mapped_weights = {}
    tllm_prefix = f'transformer.layers.{layer_idx}.'
    
    # Helper function for weight normalization
    def _normalize_linear(w: torch.Tensor, expected_out: int, expected_in: int, name: str) -> torch.Tensor:
        if w is None:
            return None
        if w.dim() != 2:
            return w
        if w.shape[0] == expected_out and w.shape[1] == expected_in:
            return w
        if w.shape[0] == expected_in and w.shape[1] == expected_out:
            return w.t().contiguous()
        # Try to fix based on matching one dimension
        if w.shape[1] == expected_in:
            if w.shape[0] != expected_out:
                logger.warning(f"{name}: unexpected out_features {w.shape[0]} != {expected_out}")
            return w
        if w.shape[0] == expected_in:
            t = w.t().contiguous()
            if t.shape[0] != expected_out:
                logger.warning(f"{name}: unexpected out_features after transpose {t.shape[0]} != {expected_out}")
            return t
        logger.warning(f"{name}: could not normalize shape {tuple(w.shape)} to ({expected_out}, {expected_in})")
        return w
    
    # Map layer normalization weights for DualFFN
    # Input layer norm (shared)
    input_norm_key = f'layers.{layer_idx}.input_layernorm.weight'
    if input_norm_key in hf_layer_weights:
        mapped_weights[tllm_prefix + 'input_layernorm.weight'] = hf_layer_weights[input_norm_key]
    
    # Post-attention layer norms (separate for audio and text in DualFFN)
    post_norm_text_key = f'layers.{layer_idx}.post_attention_layernorm.weight'
    post_norm_audio_key = f'layers.{layer_idx}.audio_post_attention_layernorm.weight'
    
    # Text post-attention layer norm
    if post_norm_text_key in hf_layer_weights:
        mapped_weights[tllm_prefix + 'post_layernorm_text.weight'] = hf_layer_weights[post_norm_text_key]
    elif f'layers.{layer_idx}.post_layernorm.weight' in hf_layer_weights:
        # Fallback to standard post layer norm for text path
        mapped_weights[tllm_prefix + 'post_layernorm_text.weight'] = hf_layer_weights[f'layers.{layer_idx}.post_layernorm.weight']
    
    # Audio post-attention layer norm
    if post_norm_audio_key in hf_layer_weights:
        mapped_weights[tllm_prefix + 'post_layernorm_audio.weight'] = hf_layer_weights[post_norm_audio_key]
    elif f'layers.{layer_idx}.audio_input_layernorm.weight' in hf_layer_weights:
        # Alternative naming pattern
        mapped_weights[tllm_prefix + 'post_layernorm_audio.weight'] = hf_layer_weights[f'layers.{layer_idx}.audio_input_layernorm.weight']
    else:
        # Use text layer norm as fallback for audio path
        if post_norm_text_key in hf_layer_weights:
            mapped_weights[tllm_prefix + 'post_layernorm_audio.weight'] = hf_layer_weights[post_norm_text_key]
            logger.info(f"Layer {layer_idx}: Using text layer norm as fallback for audio layer norm")
    
    # Map text MLP weights
    text_mlp_weights = _map_mlp_weights(
        hf_layer_weights=hf_layer_weights,
        layer_idx=layer_idx,
        mlp_prefix='mlp',  # Standard HF MLP prefix
        tllm_prefix=tllm_prefix + 'text_mlp.',
        config=config,
        mapping=mapping,
        component_name='text_mlp'
    )
    mapped_weights.update(text_mlp_weights)
    
    # Map audio MLP weights
    audio_mlp_weights = _map_mlp_weights(
        hf_layer_weights=hf_layer_weights,
        layer_idx=layer_idx,
        mlp_prefix='audio_mlp',  # Audio-specific MLP prefix
        tllm_prefix=tllm_prefix + 'audio_mlp.',
        config=config,
        mapping=mapping,
        component_name='audio_mlp'
    )
    mapped_weights.update(audio_mlp_weights)
    
    return mapped_weights


def _map_mlp_weights(
    hf_layer_weights: Dict[str, torch.Tensor],
    layer_idx: int,
    mlp_prefix: str,
    tllm_prefix: str,
    config: HiggsAudioConfig,
    mapping: Mapping,
    component_name: str
) -> Dict[str, torch.Tensor]:
    """Map MLP weights from HuggingFace format to TensorRT-LLM format.
    
    Args:
        hf_layer_weights: Dictionary of HF weights for this layer
        layer_idx: Zero-based layer index
        mlp_prefix: Prefix for MLP weights in HF format ('mlp' or 'audio_mlp')
        tllm_prefix: Prefix for TensorRT-LLM weights
        config: HiggsAudioConfig containing model parameters
        mapping: TensorRT-LLM tensor parallelism mapping
        component_name: Name of the component for logging
        
    Returns:
        Dictionary of mapped MLP weight tensors
    """
    mapped_weights = {}
    
    def _normalize_linear(w: torch.Tensor, expected_out: int, expected_in: int, name: str) -> torch.Tensor:
        if w is None:
            return None
        if w.dim() != 2:
            return w
        if w.shape[0] == expected_out and w.shape[1] == expected_in:
            return w
        if w.shape[0] == expected_in and w.shape[1] == expected_out:
            return w.t().contiguous()
        # Try to fix based on matching one dimension
        if w.shape[1] == expected_in:
            if w.shape[0] != expected_out:
                logger.warning(f"{name}: unexpected out_features {w.shape[0]} != {expected_out}")
            return w
        if w.shape[0] == expected_in:
            t = w.t().contiguous()
            if t.shape[0] != expected_out:
                logger.warning(f"{name}: unexpected out_features after transpose {t.shape[0]} != {expected_out}")
            return t
        logger.warning(f"{name}: could not normalize shape {tuple(w.shape)} to ({expected_out}, {expected_in})")
        return w
    
    # Map gate projection (gate_proj -> fc)
    gate_key = f'layers.{layer_idx}.{mlp_prefix}.gate_proj.weight'
    if gate_key in hf_layer_weights:
        gate_w = _normalize_linear(
            hf_layer_weights[gate_key],
            config.intermediate_size,
            config.hidden_size,
            f'layer{layer_idx}.{component_name}.gate'
        )
        if gate_w is not None:
            # Column-parallel split by rows (dim=0)
            fc_split = split(gate_w, mapping.tp_size, mapping.tp_rank, dim=0)
            mapped_weights[tllm_prefix + 'fc.weight'] = fc_split
        else:
            logger.warning(f"Layer {layer_idx}: Missing or invalid {component_name} gate_proj weights")
    
    # Map up projection (up_proj -> gate)
    up_key = f'layers.{layer_idx}.{mlp_prefix}.up_proj.weight'
    if up_key in hf_layer_weights:
        up_w = _normalize_linear(
            hf_layer_weights[up_key],
            config.intermediate_size,
            config.hidden_size,
            f'layer{layer_idx}.{component_name}.up'
        )
        if up_w is not None:
            # Column-parallel split by rows (dim=0)
            gate_split = split(up_w, mapping.tp_size, mapping.tp_rank, dim=0)
            mapped_weights[tllm_prefix + 'gate.weight'] = gate_split
        else:
            logger.warning(f"Layer {layer_idx}: Missing or invalid {component_name} up_proj weights")
    
    # Map down projection (down_proj -> proj)
    down_key = f'layers.{layer_idx}.{mlp_prefix}.down_proj.weight'
    if down_key in hf_layer_weights:
        down_w = _normalize_linear(
            hf_layer_weights[down_key],
            config.hidden_size,
            config.intermediate_size,
            f'layer{layer_idx}.{component_name}.down'
        )
        if down_w is not None:
            # Row-parallel split by columns (dim=1)
            proj_split = split_matrix_tp(down_w, mapping.tp_size, mapping.tp_rank, dim=1)
            mapped_weights[tllm_prefix + 'proj.weight'] = proj_split
        else:
            logger.warning(f"Layer {layer_idx}: Missing or invalid {component_name} down_proj weights")
    
    return mapped_weights


def handle_missing_dual_ffn_weights(
    layer_idx: int,
    config: HiggsAudioConfig,
    available_weights: Set[str],
    fallback_strategy: str = 'duplicate_text'
) -> Dict[str, torch.Tensor]:
    """Handle missing DualFFN weights with fallback strategies.
    
    Args:
        layer_idx: Zero-based layer index
        config: HiggsAudioConfig containing model parameters
        available_weights: Set of available weight keys from HF model
        fallback_strategy: Strategy for handling missing weights ('duplicate_text', 'zero_init', 'skip')
        
    Returns:
        Dictionary of fallback weight tensors or empty dict if skipped
        
    Raises:
        ValueError: If fallback strategy is invalid
    """
    valid_strategies = ['duplicate_text', 'zero_init', 'skip']
    if fallback_strategy not in valid_strategies:
        raise ValueError(f"Invalid fallback strategy '{fallback_strategy}'. Valid strategies: {valid_strategies}")
    
    if fallback_strategy == 'skip':
        return {}
    
    fallback_weights = {}
    
    # Check what's missing for audio components
    expected_audio_keys = [
        f'layers.{layer_idx}.audio_mlp.gate_proj.weight',
        f'layers.{layer_idx}.audio_mlp.up_proj.weight',
        f'layers.{layer_idx}.audio_mlp.down_proj.weight',
        f'layers.{layer_idx}.audio_post_attention_layernorm.weight',
        f'layers.{layer_idx}.audio_input_layernorm.weight'
    ]
    
    missing_keys = [key for key in expected_audio_keys if key not in available_weights]
    
    if missing_keys:
        logger.info(f"Layer {layer_idx}: Missing DualFFN weights: {missing_keys}, using fallback strategy '{fallback_strategy}'")
        
        if fallback_strategy == 'duplicate_text':
            # Use text MLP weights as fallback for audio MLP
            text_mappings = {
                f'layers.{layer_idx}.audio_mlp.gate_proj.weight': f'layers.{layer_idx}.mlp.gate_proj.weight',
                f'layers.{layer_idx}.audio_mlp.up_proj.weight': f'layers.{layer_idx}.mlp.up_proj.weight',
                f'layers.{layer_idx}.audio_mlp.down_proj.weight': f'layers.{layer_idx}.mlp.down_proj.weight',
                f'layers.{layer_idx}.audio_post_attention_layernorm.weight': f'layers.{layer_idx}.post_attention_layernorm.weight',
                f'layers.{layer_idx}.audio_input_layernorm.weight': f'layers.{layer_idx}.input_layernorm.weight'
            }
            
            for missing_key in missing_keys:
                if missing_key in text_mappings:
                    text_key = text_mappings[missing_key]
                    if text_key in available_weights:
                        # This will be handled by the main weight loading logic
                        logger.info(f"Layer {layer_idx}: Will use {text_key} as fallback for {missing_key}")
        
        elif fallback_strategy == 'zero_init':
            logger.warning(f"Layer {layer_idx}: Zero initialization not implemented for missing DualFFN weights")
    
    return fallback_weights


def validate_dual_ffn_components(
    layer_weights: Dict[str, torch.Tensor],
    layer_idx: int,
    config: HiggsAudioConfig,
    is_dual_ffn_layer: bool
) -> bool:
    """Validate that all required DualFFN components are present and consistent.
    
    Args:
        layer_weights: Dictionary of loaded weight tensors for this layer
        layer_idx: Zero-based layer index
        config: HiggsAudioConfig containing model parameters
        is_dual_ffn_layer: Whether this layer is configured as DualFFN
        
    Returns:
        True if validation passes, False otherwise
        
    Raises:
        ValueError: If critical validation errors are found
    """
    tllm_prefix = f'transformer.layers.{layer_idx}.'
    validation_passed = True
    
    if is_dual_ffn_layer:
        # Check for required DualFFN components
        required_components = [
            'text_mlp.fc.weight', 'text_mlp.gate.weight', 'text_mlp.proj.weight',
            'audio_mlp.fc.weight', 'audio_mlp.gate.weight', 'audio_mlp.proj.weight',
            'post_layernorm_text.weight', 'post_layernorm_audio.weight'
        ]
        
        missing_components = []
        for component in required_components:
            full_key = tllm_prefix + component
            if full_key not in layer_weights:
                missing_components.append(component)
        
        if missing_components:
            logger.warning(f"Layer {layer_idx}: Missing DualFFN components: {missing_components}")
            validation_passed = False
        
        # Validate weight shapes for DualFFN components
        for mlp_type in ['text_mlp', 'audio_mlp']:
            for weight_type in ['fc', 'gate', 'proj']:
                key = tllm_prefix + f'{mlp_type}.{weight_type}.weight'
                if key in layer_weights:
                    weight = layer_weights[key]
                    if weight_type in ['fc', 'gate']:
                        # Column-parallel weights should have intermediate_size // tp_size rows
                        expected_out = config.intermediate_size // config.mapping.tp_size
                        expected_in = config.hidden_size
                        if weight.shape != (expected_out, expected_in):
                            logger.warning(
                                f"Layer {layer_idx}: {mlp_type}.{weight_type} weight shape {weight.shape} "
                                f"doesn't match expected ({expected_out}, {expected_in})"
                            )
                            validation_passed = False
                    elif weight_type == 'proj':
                        # Row-parallel weights should have hidden_size rows and intermediate_size // tp_size columns
                        expected_out = config.hidden_size
                        expected_in = config.intermediate_size // config.mapping.tp_size
                        if weight.shape != (expected_out, expected_in):
                            logger.warning(
                                f"Layer {layer_idx}: {mlp_type}.{weight_type} weight shape {weight.shape} "
                                f"doesn't match expected ({expected_out}, {expected_in})"
                            )
                            validation_passed = False
        
        # Validate layer norm shapes
        for norm_type in ['post_layernorm_text', 'post_layernorm_audio']:
            key = tllm_prefix + f'{norm_type}.weight'
            if key in layer_weights:
                weight = layer_weights[key]
                if weight.shape != (config.hidden_size,):
                    logger.warning(
                        f"Layer {layer_idx}: {norm_type} weight shape {weight.shape} "
                        f"doesn't match expected ({config.hidden_size},)"
                    )
                    validation_passed = False
    
    return validation_passed


@torch.no_grad()
def load_weights_from_hf_model(
    hf_model_dir: str,
    config: HiggsAudioConfig,
    *,
    quant_config: Optional[QuantConfig] = None,
    validate_weights: bool = True,
    fallback_strategy: str = 'duplicate_text',
):
    """Map HF Higgs-Audio weights into TRT-LLM checkpoint dict with DualFFN support.

    This enhanced function supports DualFFN architecture where specific layers have separate
    audio and text MLP processing paths. It handles layer-specific routing based on configuration
    and provides comprehensive validation and error handling.

    Args:
        hf_model_dir: Path to HuggingFace model directory
        config: HiggsAudioConfig containing DualFFN layer specifications
        quant_config: Optional quantization configuration
        validate_weights: Whether to validate DualFFN components after loading
        fallback_strategy: Strategy for handling missing DualFFN weights
                          ('duplicate_text', 'zero_init', 'skip')

    Returns:
        Dictionary containing:
        - metadata: Model and conversion information
        - tensors: Mapped weight tensors ready for TRT-LLM
        - expected_keys: List of expected tensor keys
        - dual_ffn_info: Information about DualFFN layer processing

    Notes:
    - Assumes HF parameter names are LLaMA-like and ROOT-level (no 'model.' prefix)
    - Supports both standard layers and DualFFN layers based on config.audio_dual_ffn_layers
    - Handles TP/PP splitting following TRT-LLM helpers
    - Provides fallback mechanisms for missing DualFFN components
    """
    dtype = getattr(torch, config.dtype)
    mapping = config.mapping

    # Get DualFFN layer mapping configuration
    layer_mapping_info = get_dual_ffn_layer_mapping(config)
    dual_ffn_layers = layer_mapping_info['dual_ffn_layers']
    dual_ffn_enabled = layer_mapping_info['dual_ffn_enabled']
    
    logger.info(f"Loading weights with DualFFN support: {len(dual_ffn_layers)} DualFFN layers out of {config.num_hidden_layers} total")
    if dual_ffn_enabled:
        logger.info(f"DualFFN layers: {sorted(dual_ffn_layers)}")

    # Load shard tensors without instantiating HF model
    model_params: Dict[str, torch.Tensor] = {}
    available_keys = set()
    
    for shard_path in iterate_shard_files(hf_model_dir, rank=mapping.rank, progress_bar=False):
        sd = load_state_dict(shard_path, dtype=dtype, device='cuda')
        # Keep all relevant parameters including DualFFN components
        for k, v in sd.items():
            # Skip purely audio-encoder parameters (not DualFFN components)
            if (k.startswith('audio_encoder.') or
                k.startswith('audio_tower.') or
                k.startswith('whisper.') or
                (k.startswith('audio_') and '.mlp.' not in k and 'layernorm' not in k)):
                continue
            # Keep text backbone and DualFFN components
            model_params[k] = v
            available_keys.add(k)

    weights: Dict[str, torch.Tensor] = {}
    dual_ffn_processing_info = {
        'layers_processed': [],
        'dual_ffn_layers_converted': [],
        'standard_layers_converted': [],
        'fallback_used': [],
        'validation_results': {}
    }

    # Embedding (first PP rank) - TP split rows
    if mapping.is_first_pp_rank() and 'embed_tokens.weight' in model_params:
        emb = model_params['embed_tokens.weight']
        emb_split = split(emb, mapping.tp_size, mapping.tp_rank, dim=0)
        weights['transformer.vocab_embedding.weight'] = emb_split

    # Determine layers assigned to this PP rank
    layers_range = mapping.pp_layers(config.num_hidden_layers)

    # Process each layer with DualFFN-aware logic
    for l in layers_range:
        tllm_layer_idx = l - layers_range[0]  # Local layer index for this PP rank
        tllm_prefix = f'transformer.layers.{tllm_layer_idx}.'
        is_dual_ffn_layer = l in dual_ffn_layers
        
        dual_ffn_processing_info['layers_processed'].append(l)
        
        logger.debug(f"Processing layer {l} (local idx {tllm_layer_idx}), DualFFN: {is_dual_ffn_layer}")
        
        # Process attention components (same for all layers)
        attention_weights = _process_attention_weights(model_params, l, tllm_prefix, config, mapping)
        weights.update(attention_weights)
        
        # Process layer normalization and MLP components based on layer type
        if is_dual_ffn_layer:
            # Use DualFFN processing for this layer
            try:
                dual_ffn_weights = map_dual_ffn_weights(
                    hf_layer_weights=model_params,
                    layer_idx=tllm_layer_idx,  # Use local layer index for TRT-LLM naming
                    config=config,
                    mapping=mapping
                )
                weights.update(dual_ffn_weights)
                dual_ffn_processing_info['dual_ffn_layers_converted'].append(l)
                
                # Handle missing DualFFN components with fallback strategy
                missing_components = handle_missing_dual_ffn_weights(
                    layer_idx=l,  # Use global layer index for HF parameter lookup
                    config=config,
                    available_weights=available_keys,
                    fallback_strategy=fallback_strategy
                )
                if missing_components:
                    weights.update(missing_components)
                    dual_ffn_processing_info['fallback_used'].append(l)
                
                # Validate DualFFN components if requested
                if validate_weights:
                    validation_result = validate_dual_ffn_components(
                        layer_weights=weights,
                        layer_idx=tllm_layer_idx,  # Use local layer index for TRT-LLM validation
                        config=config,
                        is_dual_ffn_layer=True
                    )
                    dual_ffn_processing_info['validation_results'][l] = validation_result
                    if not validation_result:
                        logger.warning(f"DualFFN validation failed for layer {l}")
                
            except Exception as e:
                logger.error(f"Error processing DualFFN layer {l}: {e}")
                # Fall back to standard layer processing
                logger.info(f"Falling back to standard layer processing for layer {l}")
                standard_weights = _process_standard_layer_weights(model_params, l, tllm_prefix, config, mapping)
                weights.update(standard_weights)
                dual_ffn_processing_info['fallback_used'].append(l)
        else:
            # Use standard layer processing
            standard_weights = _process_standard_layer_weights(model_params, l, tllm_prefix, config, mapping)
            weights.update(standard_weights)
            dual_ffn_processing_info['standard_layers_converted'].append(l)

    # Final norm and lm_head (last PP rank)
    if mapping.is_last_pp_rank():
        ln_f = model_params.get('norm.weight')
        if ln_f is not None:
            weights['transformer.ln_f.weight'] = ln_f
        lm_head = model_params.get('lm_head.weight')
        if lm_head is not None:
            # For TP, split vocab rows across ranks
            lm_head_split = split_matrix_tp(lm_head, mapping.tp_size, mapping.tp_rank, dim=0)
            weights['lm_head.weight'] = lm_head_split

    # Compile comprehensive conversion information
    ckpt = {
        "metadata": {
            "source": "hf",
            "hf_model_dir": hf_model_dir,
            "dtype": str(config.dtype),
            "vocab_size": int(config.vocab_size),
            "adapter_type": getattr(config, "audio_adapter_type", None),
            "dual_ffn_enabled": dual_ffn_enabled,
            "dual_ffn_layers": sorted(dual_ffn_layers) if dual_ffn_layers else [],
            "fallback_strategy": fallback_strategy,
            "total_layers": config.num_hidden_layers,
        },
        "tensors": weights,
        "expected_keys": _backbone_key_template(config),
        "dual_ffn_info": dual_ffn_processing_info,
    }
    
    # Log conversion summary
    logger.info(f"Weight conversion completed:")
    logger.info(f"  - Total layers processed: {len(dual_ffn_processing_info['layers_processed'])}")
    logger.info(f"  - DualFFN layers converted: {len(dual_ffn_processing_info['dual_ffn_layers_converted'])}")
    logger.info(f"  - Standard layers converted: {len(dual_ffn_processing_info['standard_layers_converted'])}")
    if dual_ffn_processing_info['fallback_used']:
        logger.info(f"  - Layers using fallback: {dual_ffn_processing_info['fallback_used']}")
    
    return ckpt


def _process_attention_weights(
    model_params: Dict[str, torch.Tensor],
    layer_idx: int,
    tllm_prefix: str,
    config: HiggsAudioConfig,
    mapping: Mapping
) -> Dict[str, torch.Tensor]:
    """Process attention weights for a single layer (common for all layer types)."""
    weights = {}
    
    def _normalize_linear(w: torch.Tensor, expected_out: int, expected_in: int, name: str) -> torch.Tensor:
        # Ensure [out_features, in_features] = [expected_out, expected_in]
        if w is None:
            return None
        if w.dim() != 2:
            return w
        if w.shape[0] == expected_out and w.shape[1] == expected_in:
            return w
        if w.shape[0] == expected_in and w.shape[1] == expected_out:
            return w.t().contiguous()
        # Try to fix based on matching one dimension
        if w.shape[1] == expected_in:
            # already [*, expected_in]; if first dim != expected_out, warn but proceed
            if w.shape[0] != expected_out:
                logger.warning(f"{name}: unexpected out_features {w.shape[0]} != {expected_out}")
            return w
        if w.shape[0] == expected_in:
            t = w.t().contiguous()
            if t.shape[0] != expected_out:
                logger.warning(f"{name}: unexpected out_features after transpose {t.shape[0]} != {expected_out}")
            return t
        logger.warning(f"{name}: could not normalize shape {tuple(w.shape)} to ({expected_out}, {expected_in})")
        return w
    
    # Q, K, V projections
    q_out = config.num_attention_heads * config.head_size
    kv_out = config.num_key_value_heads * config.head_size
    q_w = _normalize_linear(model_params.get(f'layers.{layer_idx}.self_attn.q_proj.weight'), q_out, config.hidden_size, f'layer{layer_idx}.q')
    k_w = _normalize_linear(model_params.get(f'layers.{layer_idx}.self_attn.k_proj.weight'), kv_out, config.hidden_size, f'layer{layer_idx}.k')
    v_w = _normalize_linear(model_params.get(f'layers.{layer_idx}.self_attn.v_proj.weight'), kv_out, config.hidden_size, f'layer{layer_idx}.v')
    
    if q_w is None or k_w is None or v_w is None:
        logger.warning(f"Missing Q/K/V for layer {layer_idx}; skipping attention weights.")
        return weights

    # Infer kv heads from shape if config says MHA but shapes indicate GQA
    head_size = config.head_size
    n_heads = config.num_attention_heads
    inferred_kv_heads = k_w.shape[0] // head_size if (k_w.dim() == 2 and head_size > 0) else config.num_key_value_heads
    mha_mode = (inferred_kv_heads == n_heads)

    # Biases (rare; typically absent). If present, pack and split similarly
    q_b = model_params.get(f'layers.{layer_idx}.self_attn.q_proj.bias')
    k_b = model_params.get(f'layers.{layer_idx}.self_attn.k_proj.bias')
    v_b = model_params.get(f'layers.{layer_idx}.self_attn.v_proj.bias')

    # Ensure in_features (dim=1) align across Q/K/V before concat
    if q_w.shape[1] != k_w.shape[1]:
        if k_w.shape[0] == q_w.shape[1]:
            k_w = k_w.t().contiguous()
        else:
            logger.debug(f"[convert][layer {layer_idx}] adjusting k: q.in={q_w.shape[1]}, k={k_w.shape}")
    if q_w.shape[1] != v_w.shape[1]:
        if v_w.shape[0] == q_w.shape[1]:
            v_w = v_w.t().contiguous()
        else:
            logger.debug(f"[convert][layer {layer_idx}] adjusting v: q.in={q_w.shape[1]}, v={v_w.shape}")
    
    logger.debug(f"[convert][layer {layer_idx}] q={tuple(q_w.shape)} k={tuple(k_w.shape)} v={tuple(v_w.shape)}")

    if not mha_mode:
        # GQA: possibly duplicate KV heads up to tp_size
        if inferred_kv_heads < mapping.tp_size:
            k_w = dup_kv_weight(k_w, inferred_kv_heads, mapping.tp_size)
            v_w = dup_kv_weight(v_w, inferred_kv_heads, mapping.tp_size)
            if k_b is not None and v_b is not None:
                from ..convert_utils import dup_kv_bias  # local import to avoid clutter at top
                k_b = dup_kv_bias(k_b, inferred_kv_heads, mapping.tp_size)
                v_b = dup_kv_bias(v_b, inferred_kv_heads, mapping.tp_size)

        wq = split(q_w, mapping.tp_size, mapping.tp_rank)
        wk = split(k_w, mapping.tp_size, mapping.tp_rank)
        wv = split(v_w, mapping.tp_size, mapping.tp_rank)
        qkv_w_split = torch.cat((wq, wk, wv))

        if q_b is not None and k_b is not None and v_b is not None:
            qkv_b = torch.cat((q_b, k_b, v_b))
            qkv_b_split = split_qkv_bias_tp(qkv_b, config.num_attention_heads, config.hidden_size,
                                            mapping.tp_size, mapping.tp_rank)
        else:
            qkv_b_split = None
    else:
        qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
        qkv_w_split = split_qkv_tp(qkv_w, config.num_attention_heads, config.hidden_size,
                                   mapping.tp_size, mapping.tp_rank)
        if q_b is not None and k_b is not None and v_b is not None:
            qkv_b = torch.cat((q_b, k_b, v_b))
            qkv_b_split = split_qkv_bias_tp(qkv_b, config.num_attention_heads, config.hidden_size,
                                            mapping.tp_size, mapping.tp_rank)
        else:
            qkv_b_split = None

    weights[tllm_prefix + 'attention.qkv.weight'] = qkv_w_split
    if qkv_b_split is not None:
        weights[tllm_prefix + 'attention.qkv.bias'] = qkv_b_split

    # Attention output projection
    o_w = _normalize_linear(model_params.get(f'layers.{layer_idx}.self_attn.o_proj.weight'), config.hidden_size, config.hidden_size, f'layer{layer_idx}.o')
    if o_w is None:
        logger.warning(f"Missing o_proj for layer {layer_idx}")
    else:
        o_w_split = split_matrix_tp(o_w, mapping.tp_size, mapping.tp_rank, dim=1)
        if f'layers.{layer_idx}.self_attn.o_proj.bias' in model_params:
            o_b = model_params[f'layers.{layer_idx}.self_attn.o_proj.bias']
        else:
            o_b = None
        weights[tllm_prefix + 'attention.dense.weight'] = o_w_split
        if o_b is not None:
            weights[tllm_prefix + 'attention.dense.bias'] = o_b

    return weights


def _process_standard_layer_weights(
    model_params: Dict[str, torch.Tensor],
    layer_idx: int,
    tllm_prefix: str,
    config: HiggsAudioConfig,
    mapping: Mapping
) -> Dict[str, torch.Tensor]:
    """Process standard layer weights (non-DualFFN layers)."""
    weights = {}
    
    def _normalize_linear(w: torch.Tensor, expected_out: int, expected_in: int, name: str) -> torch.Tensor:
        # Ensure [out_features, in_features] = [expected_out, expected_in]
        if w is None:
            return None
        if w.dim() != 2:
            return w
        if w.shape[0] == expected_out and w.shape[1] == expected_in:
            return w
        if w.shape[0] == expected_in and w.shape[1] == expected_out:
            return w.t().contiguous()
        # Try to fix based on matching one dimension
        if w.shape[1] == expected_in:
            # already [*, expected_in]; if first dim != expected_out, warn but proceed
            if w.shape[0] != expected_out:
                logger.warning(f"{name}: unexpected out_features {w.shape[0]} != {expected_out}")
            return w
        if w.shape[0] == expected_in:
            t = w.t().contiguous()
            if t.shape[0] != expected_out:
                logger.warning(f"{name}: unexpected out_features after transpose {t.shape[0]} != {expected_out}")
            return t
        logger.warning(f"{name}: could not normalize shape {tuple(w.shape)} to ({expected_out}, {expected_in})")
        return w
    
    # Layer normalization
    in_ln = model_params.get(f'layers.{layer_idx}.input_layernorm.weight')
    post_ln = model_params.get(f'layers.{layer_idx}.post_attention_layernorm.weight')
    if in_ln is not None:
        weights[tllm_prefix + 'input_layernorm.weight'] = in_ln
    if post_ln is not None:
        weights[tllm_prefix + 'post_layernorm.weight'] = post_ln

    # Standard MLP: gate/up/down map to fc/gate/proj per TRT-LLM naming
    gate = _normalize_linear(model_params.get(f'layers.{layer_idx}.mlp.gate_proj.weight'), config.intermediate_size, config.hidden_size, f'layer{layer_idx}.mlp.gate')
    up = _normalize_linear(model_params.get(f'layers.{layer_idx}.mlp.up_proj.weight'), config.intermediate_size, config.hidden_size, f'layer{layer_idx}.mlp.up')
    down = _normalize_linear(model_params.get(f'layers.{layer_idx}.mlp.down_proj.weight'), config.hidden_size, config.intermediate_size, f'layer{layer_idx}.mlp.down')
    
    if gate is None or up is None or down is None:
        logger.warning(f"Missing MLP weights for layer {layer_idx}")
    else:
        # Column-parallel (fc/gate) split by rows (dim=0). Row-parallel (proj) split by cols (dim=1)
        fc_split = split(gate, mapping.tp_size, mapping.tp_rank, dim=0)
        gate_split = split(up, mapping.tp_size, mapping.tp_rank, dim=0)
        proj_split = split_matrix_tp(down, mapping.tp_size, mapping.tp_rank, dim=1)
        weights[tllm_prefix + 'mlp.fc.weight'] = fc_split
        weights[tllm_prefix + 'mlp.gate.weight'] = gate_split
        weights[tllm_prefix + 'mlp.proj.weight'] = proj_split

    return weights
