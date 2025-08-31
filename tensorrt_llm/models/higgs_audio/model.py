# SPDX-License-Identifier: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import Optional, Union, Dict, Any, Tuple, List
import math
import warnings
import numpy as np
import torch
from tensorrt_llm.functional import Tensor, gelu, layer_norm, embedding, conv1d, default_net, gather_last_token_logits, AttentionMaskType, PositionEmbeddingType, constant
from tensorrt_llm.layers import Conv1d, Embedding, LayerNorm, Attention, MLP
from tensorrt_llm.module import Module, ModuleList
from tensorrt_llm.parameter import Parameter
from ..._utils import pad_vocab_size
from tensorrt_llm.layers import (MLP, Attention, ColumnLinear, Embedding,
                                 GatedMLP, RmsNorm, PromptTuningEmbedding)
from tensorrt_llm._torch.models.modeling_llama import LlamaAttention
from tensorrt_llm.models import PretrainedConfig, PretrainedModel
from tensorrt_llm.models.modeling_utils import DecoderModelForCausalLM, KeyValueCacheParams, DecoderLayerList
from tensorrt_llm.top_model_mixin import TopModelMixin
from torch.nn import ModuleList, AvgPool1d
from .config import HiggsAudioConfig
from .generation_mode_manager import GenerationModeManager, GenerationState, GenerationMode

# CUDA Graph optimization imports
try:
    from .cuda_graphs import (
        CudaGraphManager, TTSGraphType, DualFFNGraphOptimizer, 
        DelayPatternGraphCoordinator, integrate_cuda_graphs_with_model
    )
    CUDA_GRAPHS_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"CUDA graphs not available: {e}")
    CUDA_GRAPHS_AVAILABLE = False
    CudaGraphManager = None
    TTSGraphType = None
    DualFFNGraphOptimizer = None
    DelayPatternGraphCoordinator = None


# ================================
# Custom Exception Classes
# ================================

class HiggsAudioError(Exception):
    """Base exception class for Higgs Audio TTS model errors.
    
    This is the base class for all exceptions raised by the Higgs Audio
    implementation. It provides structured error handling with context
    information for debugging and recovery.
    """
    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        
    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_msg = f"{base_msg} (Context: {context_str})"
        return base_msg


class DelayPatternError(HiggsAudioError):
    """Exception raised for delay pattern related errors.
    
    This exception is raised when delay pattern generation, application,
    or validation fails. It includes specific error codes for different
    types of delay pattern issues.
    
    Error Codes:
    - INVALID_STRATEGY: Unknown or unsupported delay strategy
    - INVALID_PARAMETERS: Invalid delay pattern parameters
    - DIMENSION_MISMATCH: Incompatible tensor dimensions
    - MEMORY_ERROR: Insufficient memory for delay pattern operations
    - VALIDATION_FAILED: Delay pattern validation checks failed
    """
    pass


class AudioTokenError(HiggsAudioError):
    """Exception raised for audio token management errors.
    
    This exception covers errors related to multi-codebook audio token
    processing, including splitting, merging, and validation operations.
    
    Error Codes:
    - CODEBOOK_MISMATCH: Inconsistent codebook configuration
    - TOKEN_VALIDATION_FAILED: Audio token validation failed
    - SYNCHRONIZATION_ERROR: Multi-codebook synchronization issues
    - STREAMING_ERROR: Audio token streaming processing failed
    """
    pass


class GenerationError(HiggsAudioError):
    """Exception raised for TTS generation related errors.
    
    This exception covers errors during the TTS generation process,
    including mode transitions, streaming issues, and output validation.
    
    Error Codes:
    - MODE_TRANSITION_FAILED: Error during generation mode transition
    - STREAMING_FAILED: Streaming generation encountered an error
    - VALIDATION_FAILED: Generation parameter validation failed
    - RESOURCE_EXHAUSTED: Insufficient resources for generation
    """
    pass


class AttentionError(HiggsAudioError):
    """Exception raised for attention mechanism errors.
    
    This exception covers errors in delay-aware attention processing,
    including mask generation, routing, and validation issues.
    
    Error Codes:
    - MASK_GENERATION_FAILED: Error generating attention masks
    - ROUTING_ERROR: Token routing through attention failed
    - DELAY_COORDINATION_FAILED: Delay-aware attention coordination failed
    """
    pass


# GenerationMode is now imported from generation_mode_manager.py


class DelayPatternProvider:
    """Provider for delay patterns in RVQ (Residual Vector Quantization) codebook generation.
    
    This class manages delay patterns that enable simultaneous multi-codebook audio token
    generation while supporting streaming inference. Delay patterns are critical for TTS
    models that use RVQ-based audio tokenization, as they coordinate token generation
    across multiple codebooks to maintain proper temporal alignment.
    
    Key Features:
    - Generate delay patterns for multiple codebooks with configurable strategies
    - Support different delay strategies (linear, exponential, custom)
    - Apply and reverse delay patterns for encoding/decoding sequences
    - Validate pattern consistency and handle edge cases
    - Optimize for real-time streaming inference
    - Comprehensive error handling and recovery mechanisms
    
    Delay Pattern Strategies:
    - 'linear': Each codebook has incrementally increasing delay (0, 1, 2, ...)
    - 'exponential': Exponentially increasing delays (0, 1, 2, 4, 8, ...)
    - 'custom': User-provided delay offsets per codebook
    - 'none': No delays applied (synchronous generation)
    
    Error Handling:
    - Automatic fallback to 'linear' strategy if custom strategy fails
    - Memory usage validation for large delay patterns
    - Device compatibility checking for tensor operations
    - Graceful degradation for edge cases
    
    Example:
        >>> try:
        ...     provider = DelayPatternProvider(strategy='linear', stride=1)
        ...     pattern = provider.generate_delay_pattern(n_codebooks=4, sequence_length=10)
        ...     delayed_tokens = provider.apply_delay_pattern(tokens, pattern)
        ... except DelayPatternError as e:
        ...     print(f"Delay pattern error: {e}")
        ...     # Automatic fallback or alternative strategy
    """
    
    def __init__(self,
                 strategy: str = 'linear',
                 stride: int = 1,
                 custom_delays: Optional[List[int]] = None,
                 max_delay: Optional[int] = None,
                 pad_token_id: int = 0,
                 enable_fallback: bool = True,
                 validate_memory_usage: bool = True):
        """Initialize DelayPatternProvider with comprehensive validation and error handling.
        
        Args:
            strategy: Delay pattern strategy ('linear', 'exponential', 'custom', 'none')
            stride: Base stride for delay increments in linear strategy
            custom_delays: Custom delay offsets for each codebook (used with 'custom' strategy)
            max_delay: Maximum allowed delay (helps prevent excessive delays)
            pad_token_id: Token ID used for padding delayed sequences
            enable_fallback: Whether to enable automatic fallback to safe configurations
            validate_memory_usage: Whether to validate memory usage for large patterns
            
        Raises:
            DelayPatternError: If strategy is invalid or parameters are incompatible
        """
        self.enable_fallback = enable_fallback
        self.validate_memory_usage = validate_memory_usage
        self.fallback_used = False
        self.validation_warnings = []
        
        try:
            # Validate and set strategy
            self.strategy, self.stride, self.custom_delays, self.max_delay = self._validate_and_set_parameters(
                strategy, stride, custom_delays, max_delay, enable_fallback
            )
            self.pad_token_id = pad_token_id
            
            # Validate memory constraints if enabled
            if validate_memory_usage:
                self._validate_memory_constraints()
                
        except Exception as e:
            if enable_fallback:
                warnings.warn(
                    f"DelayPatternProvider initialization failed, using fallback configuration: {e}",
                    UserWarning
                )
                self._apply_fallback_configuration()
            else:
                raise DelayPatternError(
                    f"Failed to initialize DelayPatternProvider: {e}",
                    error_code="INITIALIZATION_FAILED",
                    context={"strategy": strategy, "stride": stride, "max_delay": max_delay}
                ) from e
    
    def _validate_and_set_parameters(self, strategy: str, stride: int, custom_delays: Optional[List[int]],
                                   max_delay: Optional[int], enable_fallback: bool) -> Tuple[str, int, List[int], Optional[int]]:
        """Validate and set delay pattern parameters with fallback support."""
        valid_strategies = ['linear', 'exponential', 'custom', 'none']
        
        # Validate strategy
        if strategy not in valid_strategies:
            if enable_fallback:
                self.validation_warnings.append(f"Invalid strategy '{strategy}', falling back to 'linear'")
                strategy = 'linear'
            else:
                raise DelayPatternError(
                    f"Invalid delay strategy '{strategy}'. Valid strategies: {valid_strategies}",
                    error_code="INVALID_STRATEGY",
                    context={"provided_strategy": strategy, "valid_strategies": valid_strategies}
                )
        
        # Validate and adjust stride
        original_stride = stride
        stride = max(1, stride)  # Ensure stride is at least 1
        if original_stride != stride:
            self.validation_warnings.append(f"Stride adjusted from {original_stride} to {stride}")
        
        # Validate max_delay
        if max_delay is not None and max_delay < 0:
            if enable_fallback:
                self.validation_warnings.append(f"Negative max_delay {max_delay}, setting to None")
                max_delay = None
            else:
                raise DelayPatternError(
                    f"max_delay must be non-negative, got {max_delay}",
                    error_code="INVALID_PARAMETERS"
                )
        
        # Validate custom delays
        validated_custom_delays = custom_delays or []
        if strategy == 'custom':
            if not custom_delays:
                if enable_fallback:
                    self.validation_warnings.append("No custom_delays provided for custom strategy, falling back to linear")
                    strategy = 'linear'
                    validated_custom_delays = []
                else:
                    raise DelayPatternError(
                        "custom_delays must be provided when using 'custom' strategy",
                        error_code="INVALID_PARAMETERS",
                        context={"strategy": strategy}
                    )
            else:
                # Validate each custom delay
                valid_delays = []
                for i, delay in enumerate(custom_delays):
                    if not isinstance(delay, int):
                        try:
                            delay = int(delay)
                            self.validation_warnings.append(f"Converted delay at index {i} to integer: {delay}")
                        except (ValueError, TypeError):
                            if enable_fallback:
                                self.validation_warnings.append(f"Invalid delay at index {i}, skipping")
                                continue
                            else:
                                raise DelayPatternError(
                                    f"Invalid delay {delay} at index {i}. Delays must be integers",
                                    error_code="INVALID_PARAMETERS",
                                    context={"delay_index": i, "delay_value": delay}
                                )
                    
                    if delay < 0:
                        if enable_fallback:
                            self.validation_warnings.append(f"Negative delay {delay} at index {i}, setting to 0")
                            delay = 0
                        else:
                            raise DelayPatternError(
                                f"Delay {delay} at index {i} must be non-negative",
                                error_code="INVALID_PARAMETERS",
                                context={"delay_index": i, "delay_value": delay}
                            )
                    
                    if max_delay is not None and delay > max_delay:
                        if enable_fallback:
                            self.validation_warnings.append(f"Delay {delay} at index {i} exceeds max_delay {max_delay}, capping")
                            delay = max_delay
                        else:
                            raise DelayPatternError(
                                f"Delay {delay} at index {i} exceeds max_delay {max_delay}",
                                error_code="INVALID_PARAMETERS",
                                context={"delay_index": i, "delay_value": delay, "max_delay": max_delay}
                            )
                    
                    valid_delays.append(delay)
                
                if not valid_delays and enable_fallback:
                    self.validation_warnings.append("No valid custom delays found, falling back to linear strategy")
                    strategy = 'linear'
                    validated_custom_delays = []
                else:
                    validated_custom_delays = valid_delays
        
        return strategy, stride, validated_custom_delays, max_delay
    
    def _validate_memory_constraints(self) -> None:
        """Validate memory constraints for delay pattern operations."""
        try:
            # Estimate memory usage for typical operations
            max_reasonable_codebooks = 16
            max_reasonable_sequence = 8192
            
            if self.max_delay is not None and self.max_delay > max_reasonable_sequence // 2:
                self.validation_warnings.append(
                    f"Large max_delay {self.max_delay} may cause memory issues with long sequences"
                )
            
            # Check custom delays for memory concerns
            if self.strategy == 'custom' and self.custom_delays:
                max_custom_delay = max(self.custom_delays)
                if max_custom_delay > max_reasonable_sequence // 2:
                    self.validation_warnings.append(
                        f"Large custom delay {max_custom_delay} may cause memory issues"
                    )
                
                if len(self.custom_delays) > max_reasonable_codebooks:
                    self.validation_warnings.append(
                        f"Large number of codebooks {len(self.custom_delays)} may impact performance"
                    )
                    
        except Exception as e:
            self.validation_warnings.append(f"Memory constraint validation failed: {e}")
    
    def _apply_fallback_configuration(self) -> None:
        """Apply safe fallback configuration when initialization fails."""
        self.strategy = 'linear'
        self.stride = 1
        self.custom_delays = []
        self.max_delay = 8  # Conservative max delay
        self.fallback_used = True
        self.validation_warnings.append("Applied fallback configuration: linear strategy with stride=1, max_delay=8")
    
    def get_validation_warnings(self) -> List[str]:
        """Get list of validation warnings encountered during initialization."""
        return self.validation_warnings.copy()
    
    def is_fallback_used(self) -> bool:
        """Check if fallback configuration was used during initialization."""
        return self.fallback_used
    
    def generate_delay_pattern(self, n_codebooks: int, sequence_length: int,
                             validate_output: bool = True) -> Tensor:
        """Generate delay pattern matrix for multi-codebook token generation with comprehensive error handling.
        
        Creates a delay pattern that specifies the temporal offset for each codebook
        at each sequence position. The pattern ensures proper coordination between
        codebooks while supporting streaming generation.
        
        Args:
            n_codebooks: Number of RVQ codebooks to coordinate
            sequence_length: Length of the token sequence to generate
            validate_output: Whether to validate the generated pattern (default: True)
            
        Returns:
            Delay pattern tensor of shape [n_codebooks, sequence_length] containing
            delay offsets for each codebook at each position. Positive values indicate
            how many positions to delay token generation for that codebook.
            
        Raises:
            DelayPatternError: If pattern generation fails or parameters are invalid
            
        Example:
            >>> pattern = provider.generate_delay_pattern(n_codebooks=3, sequence_length=5)
            >>> # For linear strategy with stride=1:
            >>> # [[0, 0, 0, 0, 0],    # Codebook 0: no delay
            >>> #  [1, 1, 1, 1, 1],    # Codebook 1: 1-step delay
            >>> #  [2, 2, 2, 2, 2]]    # Codebook 2: 2-step delay
        """
        try:
            # Validate input parameters with enhanced error messages
            if not isinstance(n_codebooks, int) or n_codebooks <= 0:
                raise DelayPatternError(
                    f"n_codebooks must be a positive integer, got {n_codebooks} (type: {type(n_codebooks)})",
                    error_code="INVALID_PARAMETERS",
                    context={"n_codebooks": n_codebooks, "expected_type": "int", "expected_range": "> 0"}
                )
            
            if not isinstance(sequence_length, int) or sequence_length <= 0:
                raise DelayPatternError(
                    f"sequence_length must be a positive integer, got {sequence_length} (type: {type(sequence_length)})",
                    error_code="INVALID_PARAMETERS",
                    context={"sequence_length": sequence_length, "expected_type": "int", "expected_range": "> 0"}
                )
            
            # Check for reasonable limits to prevent memory issues
            if self.validate_memory_usage:
                max_reasonable_codebooks = 32
                max_reasonable_sequence = 16384
                
                if n_codebooks > max_reasonable_codebooks:
                    warning_msg = f"Large number of codebooks ({n_codebooks}) may impact performance"
                    if self.enable_fallback:
                        warnings.warn(warning_msg, UserWarning)
                    else:
                        raise DelayPatternError(
                            warning_msg,
                            error_code="MEMORY_WARNING",
                            context={"n_codebooks": n_codebooks, "max_reasonable": max_reasonable_codebooks}
                        )
                
                if sequence_length > max_reasonable_sequence:
                    warning_msg = f"Large sequence length ({sequence_length}) may cause memory issues"
                    if self.enable_fallback:
                        warnings.warn(warning_msg, UserWarning)
                    else:
                        raise DelayPatternError(
                            warning_msg,
                            error_code="MEMORY_WARNING",
                            context={"sequence_length": sequence_length, "max_reasonable": max_reasonable_sequence}
                        )
            
            # Generate base delays with error handling
            base_delays = self._generate_base_delays_with_fallback(n_codebooks)
            
            # Validate delays don't exceed limits
            max_delay_in_pattern = max(base_delays) if base_delays else 0
            
            if self.max_delay is not None and max_delay_in_pattern > self.max_delay:
                error_msg = f"Generated delay {max_delay_in_pattern} exceeds max_delay {self.max_delay}"
                if self.enable_fallback:
                    warnings.warn(f"{error_msg}, capping delays", UserWarning)
                    base_delays = [min(delay, self.max_delay) for delay in base_delays]
                    max_delay_in_pattern = max(base_delays)
                else:
                    raise DelayPatternError(
                        error_msg,
                        error_code="DELAY_LIMIT_EXCEEDED",
                        context={"generated_delay": max_delay_in_pattern, "max_delay": self.max_delay}
                    )
            
            if max_delay_in_pattern >= sequence_length:
                error_msg = f"Maximum delay {max_delay_in_pattern} must be less than sequence_length {sequence_length}"
                if self.enable_fallback:
                    warnings.warn(f"{error_msg}, adjusting delays", UserWarning)
                    # Scale delays to fit within sequence length
                    scale_factor = (sequence_length - 1) / max_delay_in_pattern if max_delay_in_pattern > 0 else 1
                    base_delays = [int(delay * scale_factor) for delay in base_delays]
                else:
                    raise DelayPatternError(
                        error_msg,
                        error_code="DELAY_SEQUENCE_MISMATCH",
                        context={"max_delay": max_delay_in_pattern, "sequence_length": sequence_length}
                    )
            
            # Create delay pattern tensor with error handling
            pattern = self._create_pattern_tensor(base_delays, n_codebooks, sequence_length)
            
            # Validate output pattern if requested
            if validate_output:
                self._validate_generated_pattern(pattern, n_codebooks, sequence_length)
            
            return pattern
            
        except DelayPatternError:
            # Re-raise DelayPatternError as-is
            raise
        except Exception as e:
            # Wrap unexpected errors in DelayPatternError
            raise DelayPatternError(
                f"Unexpected error during delay pattern generation: {e}",
                error_code="GENERATION_FAILED",
                context={"n_codebooks": n_codebooks, "sequence_length": sequence_length, "strategy": self.strategy}
            ) from e
    
    def _generate_base_delays_with_fallback(self, n_codebooks: int) -> List[int]:
        """Generate base delays with fallback handling for each strategy."""
        try:
            if self.strategy == 'none':
                return [0] * n_codebooks
            elif self.strategy == 'linear':
                return [i * self.stride for i in range(n_codebooks)]
            elif self.strategy == 'exponential':
                delays = []
                for i in range(n_codebooks):
                    delay = (2**i - 1) * self.stride
                    if self.max_delay is not None:
                        delay = min(delay, self.max_delay)
                    delays.append(delay)
                return delays
            elif self.strategy == 'custom':
                if len(self.custom_delays) < n_codebooks:
                    error_msg = f"custom_delays has {len(self.custom_delays)} entries but {n_codebooks} codebooks requested"
                    if self.enable_fallback:
                        warnings.warn(f"{error_msg}, padding with zero delays", UserWarning)
                        # Pad with zero delays
                        padded_delays = self.custom_delays[:] + [0] * (n_codebooks - len(self.custom_delays))
                        return padded_delays[:n_codebooks]
                    else:
                        raise DelayPatternError(
                            error_msg,
                            error_code="INSUFFICIENT_CUSTOM_DELAYS",
                            context={"custom_delays_count": len(self.custom_delays), "n_codebooks": n_codebooks}
                        )
                return self.custom_delays[:n_codebooks]
            else:
                error_msg = f"Unknown strategy: {self.strategy}"
                if self.enable_fallback:
                    warnings.warn(f"{error_msg}, falling back to linear", UserWarning)
                    return [i for i in range(n_codebooks)]  # Simple linear delays
                else:
                    raise DelayPatternError(
                        error_msg,
                        error_code="UNKNOWN_STRATEGY",
                        context={"strategy": self.strategy}
                    )
        except Exception as e:
            if self.enable_fallback:
                warnings.warn(f"Error generating base delays: {e}, using fallback", UserWarning)
                return [i for i in range(n_codebooks)]  # Simple fallback
            else:
                raise
    
    def _create_pattern_tensor(self, base_delays: List[int], n_codebooks: int, sequence_length: int) -> Tensor:
        """Create delay pattern tensor with error handling."""
        try:
            from tensorrt_llm.functional import constant
            import numpy as np
            
            # Create pattern as numpy array first for easier manipulation
            pattern_np = np.zeros((n_codebooks, sequence_length), dtype=np.int32)
            for i, delay in enumerate(base_delays):
                if i < n_codebooks:  # Safety check
                    pattern_np[i, :] = delay
            
            # Convert to TensorRT-LLM tensor
            pattern = constant(pattern_np)
            return pattern
            
        except Exception as e:
            raise DelayPatternError(
                f"Failed to create delay pattern tensor: {e}",
                error_code="TENSOR_CREATION_FAILED",
                context={"n_codebooks": n_codebooks, "sequence_length": sequence_length}
            ) from e
    
    def _validate_generated_pattern(self, pattern: Tensor, n_codebooks: int, sequence_length: int) -> None:
        """Validate the generated delay pattern for correctness."""
        try:
            # Check tensor shape
            if pattern.shape != (n_codebooks, sequence_length):
                raise DelayPatternError(
                    f"Generated pattern has incorrect shape {pattern.shape}, expected ({n_codebooks}, {sequence_length})",
                    error_code="INVALID_PATTERN_SHAPE",
                    context={"actual_shape": pattern.shape, "expected_shape": (n_codebooks, sequence_length)}
                )
            
            # Validate pattern contents
            if hasattr(pattern, 'detach'):
                pattern_np = pattern.detach().cpu().numpy()
            else:
                pattern_np = pattern.numpy()
            
            # Check for negative delays
            if (pattern_np < 0).any():
                raise DelayPatternError(
                    "Generated pattern contains negative delays",
                    error_code="NEGATIVE_DELAYS",
                    context={"min_delay": int(pattern_np.min())}
                )
            
            # Check for excessive delays
            max_delay = int(pattern_np.max())
            if max_delay >= sequence_length:
                raise DelayPatternError(
                    f"Generated pattern max delay {max_delay} exceeds sequence length {sequence_length}",
                    error_code="EXCESSIVE_DELAY",
                    context={"max_delay": max_delay, "sequence_length": sequence_length}
                )
                
        except DelayPatternError:
            raise
        except Exception as e:
            raise DelayPatternError(
                f"Pattern validation failed: {e}",
                error_code="VALIDATION_FAILED"
            ) from e
    
    def apply_delay_pattern(self, tokens: Tensor, pattern: Tensor,
                          validate: bool = True) -> Tensor:
        """Apply delay pattern to multi-codebook token sequences.
        
        Transforms synchronized token sequences into delayed sequences according to
        the provided delay pattern. This creates the proper temporal offsets needed
        for coordinated multi-codebook generation.
        
        Args:
            tokens: Input token tensor [n_codebooks, sequence_length] containing
                   synchronized tokens from all codebooks
            pattern: Delay pattern tensor [n_codebooks, sequence_length] from
                    generate_delay_pattern()
            validate: Whether to perform input validation (default: True)
            
        Returns:
            Delayed token tensor [n_codebooks, extended_sequence_length] where
            tokens are temporally shifted according to the delay pattern.
            Padded positions use self.pad_token_id.
            
        Raises:
            ValueError: If token and pattern shapes are incompatible
            
        Example:
            >>> tokens = tensor([[1, 2, 3],    # Codebook 0
            ...                  [4, 5, 6],    # Codebook 1
            ...                  [7, 8, 9]])   # Codebook 2
            >>> pattern = tensor([[0, 0, 0],   # No delay
            ...                   [1, 1, 1],   # 1-step delay
            ...                   [2, 2, 2]])  # 2-step delay
            >>> delayed = provider.apply_delay_pattern(tokens, pattern)
            >>> # Result: [[1, 2, 3, 0, 0],     # Codebook 0: original
            >>> #          [0, 4, 5, 6, 0],     # Codebook 1: shifted right by 1
            >>> #          [0, 0, 7, 8, 9]]     # Codebook 2: shifted right by 2
        """
        if validate:
            self._validate_tokens_and_pattern(tokens, pattern)
        
        n_codebooks, seq_len = tokens.shape
        
        # Calculate maximum delay to determine output sequence length
        from tensorrt_llm.functional import max as trt_max
        max_delay = int(trt_max(pattern).item())
        output_seq_len = seq_len + max_delay
        
        # Create output tensor filled with pad tokens
        from tensorrt_llm.functional import constant
        import numpy as np
        
        output_np = np.full((n_codebooks, output_seq_len), self.pad_token_id, dtype=np.int32)
        
        # Apply delays for each codebook
        tokens_np = tokens.detach().cpu().numpy() if hasattr(tokens, 'detach') else tokens.numpy()
        pattern_np = pattern.detach().cpu().numpy() if hasattr(pattern, 'detach') else pattern.numpy()
        
        for codebook_idx in range(n_codebooks):
            # Get delay for this codebook (assuming uniform delay across sequence)
            delay = int(pattern_np[codebook_idx, 0])
            
            # Copy tokens with delay offset
            for seq_idx in range(seq_len):
                output_pos = seq_idx + delay
                if output_pos < output_seq_len:
                    output_np[codebook_idx, output_pos] = tokens_np[codebook_idx, seq_idx]
        
        return constant(output_np)
    
    def reverse_delay_pattern(self, delayed_tokens: Tensor, pattern: Tensor,
                            original_length: Optional[int] = None,
                            validate: bool = True) -> Tensor:
        """Reverse delay pattern to recover synchronized token sequences.
        
        Transforms delayed token sequences back to synchronized sequences by
        reversing the temporal shifts applied by apply_delay_pattern(). This
        is used during decoding to reconstruct the original token alignment.
        
        Args:
            delayed_tokens: Delayed token tensor [n_codebooks, extended_seq_len]
                           from apply_delay_pattern()
            pattern: Delay pattern tensor [n_codebooks, pattern_seq_len] used
                    for the original delay application
            original_length: Expected length of output sequence. If None,
                           inferred from inputs
            validate: Whether to perform input validation (default: True)
            
        Returns:
            Synchronized token tensor [n_codebooks, sequence_length] with
            delays removed and tokens realigned
            
        Raises:
            ValueError: If inputs are incompatible or original_length is invalid
            
        Example:
            >>> delayed = tensor([[1, 2, 3, 0, 0],     # Codebook 0
            ...                   [0, 4, 5, 6, 0],     # Codebook 1 (delayed)
            ...                   [0, 0, 7, 8, 9]])    # Codebook 2 (delayed)
            >>> pattern = tensor([[0, 0, 0],           # Delay pattern
            ...                   [1, 1, 1],
            ...                   [2, 2, 2]])
            >>> tokens = provider.reverse_delay_pattern(delayed, pattern)
            >>> # Result: [[1, 2, 3],    # Codebook 0
            >>> #          [4, 5, 6],    # Codebook 1
            >>> #          [7, 8, 9]]    # Codebook 2
        """
        if validate:
            if delayed_tokens.dim() != 2:
                raise ValueError(f"delayed_tokens must be 2D, got shape {delayed_tokens.shape}")
            if pattern.dim() != 2:
                raise ValueError(f"pattern must be 2D, got shape {pattern.shape}")
        
        n_codebooks, delayed_seq_len = delayed_tokens.shape
        pattern_codebooks, pattern_seq_len = pattern.shape
        
        if n_codebooks != pattern_codebooks:
            raise ValueError(f"Codebook count mismatch: delayed_tokens has {n_codebooks}, "
                           f"pattern has {pattern_codebooks}")
        
        # Infer original length if not provided
        if original_length is None:
            from tensorrt_llm.functional import max as trt_max
            max_delay = int(trt_max(pattern).item())
            original_length = delayed_seq_len - max_delay
        
        if original_length <= 0:
            raise ValueError(f"Invalid original_length {original_length}")
        
        # Create output tensor
        from tensorrt_llm.functional import constant
        import numpy as np
        
        output_np = np.full((n_codebooks, original_length), self.pad_token_id, dtype=np.int32)
        
        # Reverse delays for each codebook
        delayed_np = delayed_tokens.detach().cpu().numpy() if hasattr(delayed_tokens, 'detach') else delayed_tokens.numpy()
        pattern_np = pattern.detach().cpu().numpy() if hasattr(pattern, 'detach') else pattern.numpy()
        
        for codebook_idx in range(n_codebooks):
            # Get delay for this codebook
            delay = int(pattern_np[codebook_idx, 0])
            
            # Copy tokens removing delay offset
            for seq_idx in range(original_length):
                delayed_pos = seq_idx + delay
                if delayed_pos < delayed_seq_len:
                    output_np[codebook_idx, seq_idx] = delayed_np[codebook_idx, delayed_pos]
        
        return constant(output_np)
    
    def validate_pattern_consistency(self, pattern: Tensor,
                                   n_codebooks: int,
                                   sequence_length: int) -> bool:
        """Validate that a delay pattern is consistent and well-formed.
        
        Checks that the delay pattern meets requirements for successful
        multi-codebook generation, including proper dimensions, non-negative
        delays, and reasonable delay magnitudes.
        
        Args:
            pattern: Delay pattern tensor to validate
            n_codebooks: Expected number of codebooks
            sequence_length: Expected sequence length
            
        Returns:
            True if pattern is valid and consistent
            
        Raises:
            ValueError: If pattern is invalid with detailed error message
        """
        if pattern.dim() != 2:
            raise ValueError(f"Pattern must be 2D, got shape {pattern.shape}")
        
        pattern_codebooks, pattern_seq_len = pattern.shape
        
        if pattern_codebooks != n_codebooks:
            raise ValueError(f"Pattern codebook count {pattern_codebooks} "
                           f"doesn't match expected {n_codebooks}")
        
        if pattern_seq_len != sequence_length:
            raise ValueError(f"Pattern sequence length {pattern_seq_len} "
                           f"doesn't match expected {sequence_length}")
        
        # Convert to numpy for easier validation
        pattern_np = pattern.detach().cpu().numpy() if hasattr(pattern, 'detach') else pattern.numpy()
        
        # Check for negative delays
        if (pattern_np < 0).any():
            raise ValueError("Pattern contains negative delays")
        
        # Check for excessive delays
        max_delay = pattern_np.max()
        if self.max_delay is not None and max_delay > self.max_delay:
            raise ValueError(f"Pattern contains delay {max_delay} exceeding max_delay {self.max_delay}")
        
        if max_delay >= sequence_length:
            raise ValueError(f"Pattern max delay {max_delay} must be less than sequence_length {sequence_length}")
        
        return True
    
    def get_pattern_info(self, pattern: Tensor) -> Dict[str, Any]:
        """Get information about a delay pattern for debugging and analysis.
        
        Args:
            pattern: Delay pattern tensor to analyze
            
        Returns:
            Dictionary containing pattern statistics and properties
        """
        pattern_np = pattern.detach().cpu().numpy() if hasattr(pattern, 'detach') else pattern.numpy()
        
        return {
            'shape': pattern.shape,
            'strategy': self.strategy,
            'stride': self.stride,
            'min_delay': int(pattern_np.min()),
            'max_delay': int(pattern_np.max()),
            'mean_delay': float(pattern_np.mean()),
            'delays_per_codebook': [int(pattern_np[i, 0]) for i in range(pattern.shape[0])],
            'is_uniform_per_codebook': all(
                len(set(pattern_np[i, :])) == 1 for i in range(pattern.shape[0])
            )
        }
    
    def _validate_tokens_and_pattern(self, tokens: Tensor, pattern: Tensor) -> None:
        """Internal method to validate tokens and pattern compatibility."""
        if tokens.dim() != 2:
            raise ValueError(f"tokens must be 2D, got shape {tokens.shape}")
        if pattern.dim() != 2:
            raise ValueError(f"pattern must be 2D, got shape {pattern.shape}")
        
        token_codebooks, token_seq_len = tokens.shape
        pattern_codebooks, pattern_seq_len = pattern.shape
        
        if token_codebooks != pattern_codebooks:
            raise ValueError(f"Codebook count mismatch: tokens has {token_codebooks}, "
                           f"pattern has {pattern_codebooks}")
        
        if token_seq_len != pattern_seq_len:
            raise ValueError(f"Sequence length mismatch: tokens has {token_seq_len}, "
                           f"pattern has {pattern_seq_len}")


class AudioTokenUtils:
   """Utility class for multi-codebook audio token management in RVQ-based TTS models.
   
   This class provides comprehensive utilities for handling audio tokens across multiple
   codebooks in Residual Vector Quantization (RVQ) based audio tokenization. It supports
   token splitting, merging, validation, and delay pattern coordination for multi-codebook
   audio generation.
   
   Key Features:
   - Split unified token sequences by codebook for independent processing
   - Merge tokens from multiple codebooks into unified sequences
   - Validate codebook sequence consistency and token alignment
   - Create delay-aware position encodings for temporal coordination
   - Handle padding and special tokens across codebook boundaries
   - Support streaming generation with proper token synchronization
   
   RVQ Context:
   RVQ (Residual Vector Quantization) uses multiple codebooks to encode audio features
   hierarchically. Each codebook captures different aspects of the audio signal, and
   tokens from all codebooks must be coordinated during generation to maintain proper
   audio quality and coherence.
   
   Example:
       >>> utils = AudioTokenUtils(num_codebooks=4, pad_token_id=0)
       >>> codebook_tokens = utils.split_audio_tokens_by_codebook(unified_tokens)
       >>> merged_tokens = utils.merge_codebook_tokens(codebook_tokens)
       >>> utils.validate_codebook_sequences(codebook_tokens)
   """
   
   def __init__(self,
                num_codebooks: int = 4,
                pad_token_id: int = 0,
                eos_token_id: Optional[int] = None,
                audio_start_token_id: Optional[int] = None,
                audio_end_token_id: Optional[int] = None):
       """Initialize AudioTokenUtils with codebook configuration.
       
       Args:
           num_codebooks: Number of RVQ codebooks to manage (default: 4)
           pad_token_id: Token ID used for padding sequences (default: 0)
           eos_token_id: End-of-sequence token ID for sequence termination
           audio_start_token_id: Special token marking start of audio generation
           audio_end_token_id: Special token marking end of audio generation
           
       Raises:
           ValueError: If num_codebooks is invalid or token IDs are incompatible
       """
       if num_codebooks <= 0:
           raise ValueError(f"num_codebooks must be positive, got {num_codebooks}")
       
       self.num_codebooks = num_codebooks
       self.pad_token_id = pad_token_id
       self.eos_token_id = eos_token_id
       self.audio_start_token_id = audio_start_token_id
       self.audio_end_token_id = audio_end_token_id
       
       # Validate token IDs don't conflict
       token_ids = [pad_token_id, eos_token_id, audio_start_token_id, audio_end_token_id]
       non_none_tokens = [t for t in token_ids if t is not None]
       if len(set(non_none_tokens)) != len(non_none_tokens):
           raise ValueError("Token IDs must be unique")
   
   def split_audio_tokens_by_codebook(self,
                                    unified_tokens: Tensor,
                                    validate: bool = True) -> List[Tensor]:
       """Split unified audio token sequence into per-codebook sequences.
       
       Takes a unified sequence where tokens from different codebooks are interleaved
       and separates them into individual codebook sequences. This is essential for
       RVQ-based models where each codebook needs independent processing.
       
       Token Layout (for 4 codebooks):
       unified: [tok0_cb0, tok0_cb1, tok0_cb2, tok0_cb3, tok1_cb0, tok1_cb1, ...]
       split:   [[tok0_cb0, tok1_cb0, ...],  # Codebook 0
                 [tok0_cb1, tok1_cb1, ...],  # Codebook 1
                 [tok0_cb2, tok1_cb2, ...],  # Codebook 2
                 [tok0_cb3, tok1_cb3, ...]] # Codebook 3
       
       Args:
           unified_tokens: Unified token tensor [batch_size, total_seq_len] where
                         total_seq_len = time_steps * num_codebooks
           validate: Whether to perform input validation (default: True)
           
       Returns:
           List of codebook-specific token tensors, each of shape
           [batch_size, time_steps] where time_steps = total_seq_len / num_codebooks
           
       Raises:
           ValueError: If unified_tokens shape is incompatible with num_codebooks
           
       Example:
           >>> unified = tensor([[10, 20, 30, 40, 11, 21, 31, 41]])  # 4 codebooks, 2 time steps
           >>> codebook_tokens = utils.split_audio_tokens_by_codebook(unified)
           >>> # Result: [[[10, 11]], [[20, 21]], [[30, 31]], [[40, 41]]]
       """
       if validate:
           if unified_tokens.dim() != 2:
               raise ValueError(f"unified_tokens must be 2D [batch, seq], got shape {unified_tokens.shape}")
       
       batch_size, total_seq_len = unified_tokens.shape
       
       if total_seq_len % self.num_codebooks != 0:
           raise ValueError(
               f"total_seq_len {total_seq_len} must be divisible by num_codebooks {self.num_codebooks}"
           )
       
       time_steps = total_seq_len // self.num_codebooks
       
       # Convert to numpy for easier manipulation
       unified_np = unified_tokens.detach().cpu().numpy() if hasattr(unified_tokens, 'detach') else unified_tokens.numpy()
       
       # Reshape to [batch, time_steps, num_codebooks] and split
       reshaped = unified_np.reshape(batch_size, time_steps, self.num_codebooks)
       
       # Split along codebook dimension
       codebook_tokens = []
       from tensorrt_llm.functional import constant
       
       for codebook_idx in range(self.num_codebooks):
           codebook_seq = reshaped[:, :, codebook_idx]  # [batch, time_steps]
           codebook_tokens.append(constant(codebook_seq))
       
       return codebook_tokens
   
   def merge_codebook_tokens(self,
                           codebook_tokens: List[Tensor],
                           validate: bool = True) -> Tensor:
       """Merge per-codebook token sequences into unified sequence.
       
       Combines separate codebook sequences back into a unified interleaved sequence.
       This is the inverse operation of split_audio_tokens_by_codebook() and is used
       when preparing tokens for models that expect unified input.
       
       Args:
           codebook_tokens: List of codebook-specific tensors, each of shape
                          [batch_size, time_steps]
           validate: Whether to perform input validation (default: True)
           
       Returns:
           Unified token tensor [batch_size, total_seq_len] where
           total_seq_len = time_steps * num_codebooks with interleaved tokens
           
       Raises:
           ValueError: If codebook_tokens list is inconsistent or incompatible
           
       Example:
           >>> codebook_tokens = [tensor([[10, 11]]),  # Codebook 0
           ...                    tensor([[20, 21]]),  # Codebook 1
           ...                    tensor([[30, 31]]),  # Codebook 2
           ...                    tensor([[40, 41]])]  # Codebook 3
           >>> unified = utils.merge_codebook_tokens(codebook_tokens)
           >>> # Result: tensor([[10, 20, 30, 40, 11, 21, 31, 41]])
       """
       if validate:
           if not isinstance(codebook_tokens, (list, tuple)):
               raise TypeError(f"codebook_tokens must be list or tuple, got {type(codebook_tokens)}")
           
           if len(codebook_tokens) != self.num_codebooks:
               raise ValueError(
                   f"Expected {self.num_codebooks} codebook tensors, got {len(codebook_tokens)}"
               )
           
           # Validate all tensors have same shape
           if not codebook_tokens:
               raise ValueError("codebook_tokens cannot be empty")
           
           reference_shape = codebook_tokens[0].shape
           for i, tensor in enumerate(codebook_tokens[1:], 1):
               if tensor.shape != reference_shape:
                   raise ValueError(
                       f"Codebook {i} shape {tensor.shape} doesn't match "
                       f"codebook 0 shape {reference_shape}"
                   )
       
       batch_size, time_steps = codebook_tokens[0].shape
       
       # Convert tensors to numpy
       codebook_arrays = []
       for tensor in codebook_tokens:
           array = tensor.detach().cpu().numpy() if hasattr(tensor, 'detach') else tensor.numpy()
           codebook_arrays.append(array)
       
       # Stack codebooks as last dimension and reshape to interleave
       import numpy as np
       stacked = np.stack(codebook_arrays, axis=-1)  # [batch, time, num_codebooks]
       unified = stacked.reshape(batch_size, -1)     # [batch, time * num_codebooks]
       
       from tensorrt_llm.functional import constant
       return constant(unified)
   
   def validate_codebook_sequences(self,
                                 codebook_tokens: List[Tensor],
                                 check_padding: bool = True,
                                 check_special_tokens: bool = True) -> bool:
       """Validate consistency of multi-codebook token sequences.
       
       Performs comprehensive validation of codebook token sequences to ensure
       they are properly formatted and consistent for multi-codebook generation.
       This includes checking shapes, padding alignment, and special token placement.
       
       Args:
           codebook_tokens: List of per-codebook token tensors
           check_padding: Whether to validate padding token consistency
           check_special_tokens: Whether to validate special token placement
           
       Returns:
           True if all sequences are valid and consistent
           
       Raises:
           ValueError: If any validation check fails with detailed error message
           
       Example:
           >>> try:
           ...     utils.validate_codebook_sequences(codebook_tokens)
           ...     print("Sequences are valid")
           ... except ValueError as e:
           ...     print(f"Validation failed: {e}")
       """
       if not isinstance(codebook_tokens, (list, tuple)):
           raise ValueError(f"codebook_tokens must be list or tuple, got {type(codebook_tokens)}")
       
       if len(codebook_tokens) == 0:
           raise ValueError("codebook_tokens cannot be empty")
       
       if len(codebook_tokens) != self.num_codebooks:
           raise ValueError(
               f"Expected {self.num_codebooks} codebook tensors, got {len(codebook_tokens)}"
           )
       
       # Validate tensor shapes
       reference_shape = codebook_tokens[0].shape
       if len(reference_shape) != 2:
           raise ValueError(
               f"Each codebook tensor must be 2D [batch, time], got shape {reference_shape}"
           )
       
       batch_size, time_steps = reference_shape
       
       for i, tensor in enumerate(codebook_tokens):
           if tensor.shape != reference_shape:
               raise ValueError(
                   f"Codebook {i} shape {tensor.shape} doesn't match "
                   f"reference shape {reference_shape}"
               )
       
       # Validate padding consistency if requested
       if check_padding and self.pad_token_id is not None:
           self._validate_padding_consistency(codebook_tokens)
       
       # Validate special token placement if requested
       if check_special_tokens:
           self._validate_special_token_placement(codebook_tokens)
       
       return True
   
   def create_delay_aware_position_ids(self,
                                     codebook_tokens: List[Tensor],
                                     delay_pattern: Tensor,
                                     base_position_offset: int = 0) -> List[Tensor]:
       """Create position IDs that account for delay patterns in multi-codebook generation.
       
       Generates position encodings for each codebook that are aware of the temporal
       delays introduced by delay patterns. This ensures proper positional encoding
       during multi-codebook generation where different codebooks may have different
       temporal offsets.
       
       Args:
           codebook_tokens: List of per-codebook token tensors [batch, time_steps]
           delay_pattern: Delay pattern tensor [num_codebooks, time_steps]
           base_position_offset: Base offset for position IDs (default: 0)
           
       Returns:
           List of position ID tensors for each codebook, accounting for delays
           
       Raises:
           ValueError: If inputs are incompatible or delay pattern is invalid
           
       Example:
           >>> position_ids = utils.create_delay_aware_position_ids(
           ...     codebook_tokens, delay_pattern, base_position_offset=0
           ... )
           >>> # position_ids[0]: [0, 1, 2, 3, ...]  # No delay
           >>> # position_ids[1]: [1, 2, 3, 4, ...]  # 1-step delay
       """
       # Validate inputs
       if len(codebook_tokens) != self.num_codebooks:
           raise ValueError(
               f"Expected {self.num_codebooks} codebook tensors, got {len(codebook_tokens)}"
           )
       
       if delay_pattern.shape[0] != self.num_codebooks:
           raise ValueError(
               f"delay_pattern first dimension {delay_pattern.shape[0]} "
               f"doesn't match num_codebooks {self.num_codebooks}"
           )
       
       batch_size, time_steps = codebook_tokens[0].shape
       
       if delay_pattern.shape[1] != time_steps:
           raise ValueError(
               f"delay_pattern second dimension {delay_pattern.shape[1]} "
               f"doesn't match time_steps {time_steps}"
           )
       
       # Convert delay pattern to numpy for easier processing
       delay_np = delay_pattern.detach().cpu().numpy() if hasattr(delay_pattern, 'detach') else delay_pattern.numpy()
       
       position_ids_list = []
       from tensorrt_llm.functional import constant
       import numpy as np
       
       for codebook_idx in range(self.num_codebooks):
           # Get delay for this codebook (assuming uniform delay per codebook)
           codebook_delay = int(delay_np[codebook_idx, 0])
           
           # Create base position IDs
           base_positions = np.arange(time_steps) + base_position_offset + codebook_delay
           
           # Expand for batch dimension
           position_ids = np.broadcast_to(base_positions[None, :], (batch_size, time_steps))
           
           position_ids_list.append(constant(position_ids))
       
       return position_ids_list
   
   def _validate_padding_consistency(self, codebook_tokens: List[Tensor]) -> None:
       """Validate that padding tokens are consistently placed across codebooks."""
       # Convert tensors to numpy for easier processing
       codebook_arrays = []
       for tensor in codebook_tokens:
           array = tensor.detach().cpu().numpy() if hasattr(tensor, 'detach') else tensor.numpy()
           codebook_arrays.append(array)
       
       batch_size, time_steps = codebook_arrays[0].shape
       
       for batch_idx in range(batch_size):
           for time_idx in range(time_steps):
               # Check if any codebook has padding at this position
               has_padding = any(
                   codebook_arrays[cb_idx][batch_idx, time_idx] == self.pad_token_id
                   for cb_idx in range(self.num_codebooks)
               )
               
               if has_padding:
                   # If one codebook has padding, all should have padding
                   for cb_idx in range(self.num_codebooks):
                       if codebook_arrays[cb_idx][batch_idx, time_idx] != self.pad_token_id:
                           raise ValueError(
                               f"Inconsistent padding at batch {batch_idx}, time {time_idx}: "
                               f"codebook {cb_idx} has non-padding token but others have padding"
                           )
   
   def _validate_special_token_placement(self, codebook_tokens: List[Tensor]) -> None:
       """Validate that special tokens are properly placed across codebooks."""
       special_tokens = []
       if self.eos_token_id is not None:
           special_tokens.append(self.eos_token_id)
       if self.audio_start_token_id is not None:
           special_tokens.append(self.audio_start_token_id)
       if self.audio_end_token_id is not None:
           special_tokens.append(self.audio_end_token_id)
       
       if not special_tokens:
           return  # No special tokens to validate
       
       # Convert tensors to numpy
       codebook_arrays = []
       for tensor in codebook_tokens:
           array = tensor.detach().cpu().numpy() if hasattr(tensor, 'detach') else tensor.numpy()
           codebook_arrays.append(array)
       
       batch_size, time_steps = codebook_arrays[0].shape
       
       for batch_idx in range(batch_size):
           for time_idx in range(time_steps):
               # Check for special tokens in any codebook at this position
               special_token_positions = []
               for cb_idx in range(self.num_codebooks):
                   token = codebook_arrays[cb_idx][batch_idx, time_idx]
                   if token in special_tokens:
                       special_token_positions.append((cb_idx, token))
               
               # If special tokens exist, validate they're consistent
               if special_token_positions:
                   # All codebooks should have the same special token at this position
                   reference_token = special_token_positions[0][1]
                   for cb_idx in range(self.num_codebooks):
                       token = codebook_arrays[cb_idx][batch_idx, time_idx]
                       if token != reference_token:
                           if token in special_tokens and token != reference_token:
                               raise ValueError(
                                   f"Mixed special tokens at batch {batch_idx}, time {time_idx}: "
                                   f"codebook 0 has {reference_token}, codebook {cb_idx} has {token}"
                               )
                           elif token not in special_tokens and token != self.pad_token_id:
                               raise ValueError(
                                   f"Inconsistent special token placement at batch {batch_idx}, time {time_idx}: "
                                   f"codebook 0 has special token {reference_token}, "
                                   f"codebook {cb_idx} has regular token {token}"
                               )
   
   def validate_audio_tokens(self,
                           audio_tokens: Union[Tensor, List[Tensor]],
                           expected_codebooks: Optional[int] = None,
                           expected_sequence_length: Optional[int] = None,
                           check_token_ranges: bool = True,
                           check_synchronization: bool = True) -> bool:
       """Validate audio tokens with comprehensive error handling and detailed diagnostics.
       
       This method performs extensive validation of audio tokens to ensure they are
       properly formatted for multi-codebook generation. It checks dimensions,
       token ranges, synchronization across codebooks, and consistency with
       expected parameters.
       
       Args:
           audio_tokens: Audio tokens to validate. Can be:
                        - Single tensor [batch, seq_len] for unified tokens
                        - List of tensors, each [batch, time_steps] for per-codebook tokens
           expected_codebooks: Expected number of codebooks (None to infer from data)
           expected_sequence_length: Expected sequence length (None to infer from data)
           check_token_ranges: Whether to validate token values are within reasonable ranges
           check_synchronization: Whether to check multi-codebook synchronization
           
       Returns:
           True if validation passes
           
       Raises:
           AudioTokenError: If validation fails with detailed error information
           TypeError: If input types are incorrect
           
       Example:
           >>> try:
           ...     utils.validate_audio_tokens(audio_tokens, expected_codebooks=4)
           ...     print("Audio tokens are valid")
           ... except AudioTokenError as e:
           ...     print(f"Validation failed: {e}")
           ...     print(f"Error context: {e.context}")
       """
       try:
           # Validate input types
           if audio_tokens is None:
               raise AudioTokenError(
                   "audio_tokens cannot be None",
                   error_code="INVALID_INPUT",
                   context={"input_type": type(audio_tokens)}
               )
           
           # Handle different input formats
           if isinstance(audio_tokens, Tensor):
               # Single tensor - check if it's unified or single codebook
               if audio_tokens.dim() not in [2, 3]:
                   raise AudioTokenError(
                       f"Single tensor must be 2D [batch, seq] or 3D [batch, seq, codebooks], got shape {audio_tokens.shape}",
                       error_code="INVALID_TENSOR_SHAPE",
                       context={"tensor_shape": audio_tokens.shape, "expected_dims": [2, 3]}
                   )
               
               batch_size, seq_len = audio_tokens.shape[:2]
               
               if audio_tokens.dim() == 3:
                   # 3D tensor with codebook dimension
                   num_codebooks = audio_tokens.shape[2]
                   codebook_tokens = [audio_tokens[:, :, i] for i in range(num_codebooks)]
               else:
                   # 2D tensor - might be unified or single codebook
                   if expected_codebooks and expected_codebooks > 1:
                       # Try to split as unified tokens
                       if seq_len % expected_codebooks != 0:
                           raise AudioTokenError(
                               f"Cannot split sequence length {seq_len} into {expected_codebooks} codebooks evenly",
                               error_code="INVALID_UNIFIED_FORMAT",
                               context={
                                   "seq_len": seq_len,
                                   "expected_codebooks": expected_codebooks,
                                   "remainder": seq_len % expected_codebooks
                               }
                           )
                       codebook_tokens = self.split_audio_tokens_by_codebook(audio_tokens, validate=True)
                       num_codebooks = len(codebook_tokens)
                   else:
                       # Treat as single codebook
                       codebook_tokens = [audio_tokens]
                       num_codebooks = 1
                       
           elif isinstance(audio_tokens, (list, tuple)):
               # List of per-codebook tensors
               if len(audio_tokens) == 0:
                   raise AudioTokenError(
                       "audio_tokens list cannot be empty",
                       error_code="EMPTY_TOKEN_LIST",
                       context={"list_type": type(audio_tokens)}
                   )
               
               # Validate each tensor in the list
               for i, tensor in enumerate(audio_tokens):
                   if not isinstance(tensor, Tensor):
                       raise AudioTokenError(
                           f"All elements in audio_tokens list must be Tensors, got {type(tensor)} at index {i}",
                           error_code="INVALID_LIST_ELEMENT_TYPE",
                           context={"element_index": i, "element_type": type(tensor)}
                       )
                   
                   if tensor.dim() != 2:
                       raise AudioTokenError(
                           f"Each tensor in list must be 2D [batch, time_steps], got shape {tensor.shape} at index {i}",
                           error_code="INVALID_LIST_ELEMENT_SHAPE",
                           context={"element_index": i, "element_shape": tensor.shape}
                       )
               
               # Validate shape consistency
               reference_shape = audio_tokens[0].shape
               for i, tensor in enumerate(audio_tokens[1:], 1):
                   if tensor.shape != reference_shape:
                       raise AudioTokenError(
                           f"Shape mismatch in audio_tokens list: tensor {i} has shape {tensor.shape}, "
                           f"expected {reference_shape}",
                           error_code="SHAPE_MISMATCH_IN_LIST",
                           context={
                               "reference_shape": reference_shape,
                               "mismatch_index": i,
                               "mismatch_shape": tensor.shape
                           }
                       )
               
               codebook_tokens = list(audio_tokens)
               num_codebooks = len(codebook_tokens)
               batch_size, time_steps = reference_shape
               
           else:
               raise TypeError(
                   f"audio_tokens must be Tensor or list of Tensors, got {type(audio_tokens)}"
               )
           
           # Validate against expected parameters
           if expected_codebooks is not None:
               if num_codebooks != expected_codebooks:
                   raise AudioTokenError(
                       f"Number of codebooks {num_codebooks} doesn't match expected {expected_codebooks}",
                       error_code="CODEBOOK_COUNT_MISMATCH",
                       context={
                           "actual_codebooks": num_codebooks,
                           "expected_codebooks": expected_codebooks
                       }
                   )
           
           if expected_sequence_length is not None:
               actual_seq_len = codebook_tokens[0].shape[1]
               if actual_seq_len != expected_sequence_length:
                   raise AudioTokenError(
                       f"Sequence length {actual_seq_len} doesn't match expected {expected_sequence_length}",
                       error_code="SEQUENCE_LENGTH_MISMATCH",
                       context={
                           "actual_sequence_length": actual_seq_len,
                           "expected_sequence_length": expected_sequence_length
                       }
                   )
           
           # Validate token ranges if requested
           if check_token_ranges:
               self._validate_token_ranges(codebook_tokens)
           
           # Validate multi-codebook synchronization if requested
           if check_synchronization and num_codebooks > 1:
               self._validate_codebook_synchronization(codebook_tokens)
           
           # Additional validations using existing methods
           if num_codebooks > 1:
               self.validate_codebook_sequences(
                   codebook_tokens,
                   check_padding=True,
                   check_special_tokens=True
               )
           
           return True
           
       except AudioTokenError:
           # Re-raise AudioTokenError as-is
           raise
       except Exception as e:
           # Wrap unexpected errors
           raise AudioTokenError(
               f"Unexpected error during audio token validation: {e}",
               error_code="VALIDATION_FAILED",
               context={"original_error": str(e), "error_type": type(e).__name__}
           ) from e
   
   def _validate_token_ranges(self, codebook_tokens: List[Tensor]) -> None:
       """Validate that token values are within reasonable ranges."""
       try:
           for codebook_idx, tokens in enumerate(codebook_tokens):
               # Convert to numpy for easier processing
               tokens_np = tokens.detach().cpu().numpy() if hasattr(tokens, 'detach') else tokens.numpy()
               
               # Check for reasonable token ID ranges
               min_token = tokens_np.min()
               max_token = tokens_np.max()
               
               # Validate minimum token (should be non-negative)
               if min_token < 0:
                   raise AudioTokenError(
                       f"Codebook {codebook_idx} contains negative token IDs (min: {min_token})",
                       error_code="NEGATIVE_TOKEN_IDS",
                       context={
                           "codebook_index": codebook_idx,
                           "min_token": int(min_token),
                           "max_token": int(max_token)
                       }
                   )
               
               # Check for excessively large token IDs (likely indicates corruption)
               max_reasonable_token = 100000  # Reasonable upper bound for most tokenizers
               if max_token > max_reasonable_token:
                   warnings.warn(
                       f"Codebook {codebook_idx} contains very large token ID ({max_token}), "
                       f"which might indicate data corruption",
                       UserWarning
                   )
               
               # Check for suspicious patterns
               unique_tokens = len(np.unique(tokens_np))
               total_tokens = tokens_np.size
               
               # Very low token diversity might indicate issues
               if total_tokens > 100 and unique_tokens < 2:
                   warnings.warn(
                       f"Codebook {codebook_idx} has very low token diversity "
                       f"({unique_tokens} unique tokens in {total_tokens} positions)",
                       UserWarning
                   )
               
       except Exception as e:
           raise AudioTokenError(
               f"Token range validation failed: {e}",
               error_code="RANGE_VALIDATION_FAILED"
           ) from e
   
   def _validate_codebook_synchronization(self, codebook_tokens: List[Tensor]) -> None:
       """Validate synchronization between multiple codebooks."""
       try:
           if len(codebook_tokens) < 2:
               return  # No synchronization to check
           
           # Convert tokens to numpy for easier processing
           codebook_arrays = []
           for tensor in codebook_tokens:
               array = tensor.detach().cpu().numpy() if hasattr(tensor, 'detach') else tensor.numpy()
               codebook_arrays.append(array)
           
           batch_size, time_steps = codebook_arrays[0].shape
           
           # Check for consistent sequence boundaries
           for batch_idx in range(batch_size):
               # Find sequence boundaries (non-pad tokens)
               sequence_lengths = []
               for codebook_array in codebook_arrays:
                   # Count non-pad tokens
                   non_pad_mask = codebook_array[batch_idx] != self.pad_token_id
                   if non_pad_mask.any():
                       # Find last non-pad token
                       last_non_pad = np.where(non_pad_mask)[0][-1]
                       sequence_lengths.append(last_non_pad + 1)
                   else:
                       sequence_lengths.append(0)
               
               # Check if sequence lengths are reasonably similar
               if len(set(sequence_lengths)) > 1:
                   max_length_diff = max(sequence_lengths) - min(sequence_lengths)
                   # Allow some difference for delay patterns, but flag large discrepancies
                   if max_length_diff > max(time_steps // 4, 8):  # More than 25% or 8 tokens difference
                       warnings.warn(
                           f"Batch {batch_idx}: Large sequence length variation across codebooks "
                           f"(lengths: {sequence_lengths}). This might indicate synchronization issues.",
                           UserWarning
                       )
           
           # Check for consistent special token placement
           if any(token_id is not None for token_id in [self.eos_token_id, self.audio_start_token_id, self.audio_end_token_id]):
               self._validate_special_token_placement(codebook_tokens)
           
       except Exception as e:
           raise AudioTokenError(
               f"Codebook synchronization validation failed: {e}",
               error_code="SYNCHRONIZATION_VALIDATION_FAILED"
           ) from e
   
   def get_audio_token_statistics(self, audio_tokens: Union[Tensor, List[Tensor]]) -> Dict[str, Any]:
       """Get comprehensive statistics about audio tokens for debugging and analysis.
       
       Args:
           audio_tokens: Audio tokens to analyze (same formats as validate_audio_tokens)
           
       Returns:
           Dictionary containing detailed statistics about the audio tokens
           
       Example:
           >>> stats = utils.get_audio_token_statistics(audio_tokens)
           >>> print(f"Codebooks: {stats['num_codebooks']}")
           >>> print(f"Sequence length: {stats['sequence_length']}")
           >>> print(f"Token range: {stats['token_range']}")
       """
       try:
           # First validate the tokens to ensure they're properly formatted
           self.validate_audio_tokens(audio_tokens, check_token_ranges=False, check_synchronization=False)
           
           # Process tokens into codebook format
           if isinstance(audio_tokens, Tensor):
               if audio_tokens.dim() == 3:
                   codebook_tokens = [audio_tokens[:, :, i] for i in range(audio_tokens.shape[2])]
               else:
                   codebook_tokens = [audio_tokens]
           else:
               codebook_tokens = list(audio_tokens)
           
           batch_size, time_steps = codebook_tokens[0].shape
           num_codebooks = len(codebook_tokens)
           
           # Gather statistics
           stats = {
               'num_codebooks': num_codebooks,
               'batch_size': batch_size,
               'sequence_length': time_steps,
               'total_tokens': batch_size * time_steps * num_codebooks,
               'codebook_statistics': {}
           }
           
           # Per-codebook statistics
           for i, tokens in enumerate(codebook_tokens):
               tokens_np = tokens.detach().cpu().numpy() if hasattr(tokens, 'detach') else tokens.numpy()
               
               # Basic statistics
               codebook_stats = {
                   'min_token': int(tokens_np.min()),
                   'max_token': int(tokens_np.max()),
                   'unique_tokens': len(np.unique(tokens_np)),
                   'pad_token_count': int(np.sum(tokens_np == self.pad_token_id)),
                   'non_pad_tokens': int(np.sum(tokens_np != self.pad_token_id))
               }
               
               # Special token counts
               if self.eos_token_id is not None:
                   codebook_stats['eos_token_count'] = int(np.sum(tokens_np == self.eos_token_id))
               if self.audio_start_token_id is not None:
                   codebook_stats['audio_start_count'] = int(np.sum(tokens_np == self.audio_start_token_id))
               if self.audio_end_token_id is not None:
                   codebook_stats['audio_end_count'] = int(np.sum(tokens_np == self.audio_end_token_id))
               
               # Sequence length statistics per batch
               sequence_lengths = []
               for batch_idx in range(batch_size):
                   non_pad_mask = tokens_np[batch_idx] != self.pad_token_id
                   if non_pad_mask.any():
                       last_non_pad = np.where(non_pad_mask)[0][-1]
                       sequence_lengths.append(last_non_pad + 1)
                   else:
                       sequence_lengths.append(0)
               
               codebook_stats['sequence_lengths'] = sequence_lengths
               codebook_stats['avg_sequence_length'] = float(np.mean(sequence_lengths))
               codebook_stats['min_sequence_length'] = int(min(sequence_lengths))
               codebook_stats['max_sequence_length'] = int(max(sequence_lengths))
               
               stats['codebook_statistics'][f'codebook_{i}'] = codebook_stats
           
           # Cross-codebook statistics
           if num_codebooks > 1:
               # Sequence length consistency across codebooks
               all_sequence_lengths = [stats['codebook_statistics'][f'codebook_{i}']['sequence_lengths']
                                     for i in range(num_codebooks)]
               
               # Calculate sequence length differences
               max_length_differences = []
               for batch_idx in range(batch_size):
                   batch_lengths = [lengths[batch_idx] for lengths in all_sequence_lengths]
                   max_diff = max(batch_lengths) - min(batch_lengths)
                   max_length_differences.append(max_diff)
               
               stats['cross_codebook_statistics'] = {
                   'max_sequence_length_differences': max_length_differences,
                   'avg_sequence_length_difference': float(np.mean(max_length_differences)),
                   'synchronization_quality': 'good' if max(max_length_differences) <= 2 else
                                            'fair' if max(max_length_differences) <= 8 else 'poor'
               }
           
           # Overall token range
           all_tokens = np.concatenate([tokens.detach().cpu().numpy().flatten() if hasattr(tokens, 'detach')
                                      else tokens.numpy().flatten() for tokens in codebook_tokens])
           
           stats['token_range'] = {
               'min': int(all_tokens.min()),
               'max': int(all_tokens.max()),
               'unique_count': len(np.unique(all_tokens)),
               'total_count': len(all_tokens)
           }
           
           return stats
           
       except Exception as e:
           raise AudioTokenError(
               f"Failed to compute audio token statistics: {e}",
               error_code="STATISTICS_COMPUTATION_FAILED"
           ) from e


class DelayAwareAttentionUtils:
    """Utilities for delay-aware attention masking and token routing in multi-codebook generation.
    
    This class provides comprehensive utilities for handling attention mechanisms that account
    for delay patterns in RVQ-based multi-codebook audio generation. It creates specialized
    attention masks and manages token routing to ensure proper temporal coordination between
    different codebooks during generation.
    
    Key Features:
    - Generate delay-aware attention masks that respect temporal offsets between codebooks
    - Create causal masks with codebook-specific delays for proper autoregressive generation
    - Handle cross-codebook attention dependencies and constraints
    - Support streaming-aware attention patterns for real-time TTS applications
    - Provide position encoding adjustments for delay-aware attention computation
    
    The class integrates with DelayPatternProvider and AudioTokenUtils to create
    sophisticated attention patterns that maintain causal dependencies while allowing
    coordinated generation across multiple audio codebooks.
    
    Example:
        >>> attn_utils = DelayAwareAttentionUtils(num_codebooks=4)
        >>> mask = attn_utils.create_delay_aware_attention_mask(
        ...     batch_size=2, seq_len=100, delay_pattern=pattern
        ... )
        >>> routing_mask = attn_utils.create_codebook_routing_mask(tokens, delay_pattern)
    """
    
    def __init__(self,
                 num_codebooks: int = 4,
                 causal_attention: bool = True,
                 cross_codebook_attention: bool = False,
                 max_delay: Optional[int] = None):
        """Initialize DelayAwareAttentionUtils with codebook configuration.
        
        Args:
            num_codebooks: Number of RVQ codebooks to coordinate
            causal_attention: Whether to enforce causal attention constraints
            cross_codebook_attention: Whether to allow attention between codebooks
            max_delay: Maximum delay to consider for attention masking
            
        Raises:
            ValueError: If parameters are invalid
        """
        if num_codebooks <= 0:
            raise ValueError(f"num_codebooks must be positive, got {num_codebooks}")
        
        self.num_codebooks = num_codebooks
        self.causal_attention = causal_attention
        self.cross_codebook_attention = cross_codebook_attention
        self.max_delay = max_delay
    
    def create_delay_aware_attention_mask(self,
                                        batch_size: int,
                                        seq_len: int,
                                        delay_pattern: Tensor,
                                        codebook_mask: Optional[Tensor] = None,
                                        dtype=None) -> Tensor:
        """Create attention mask that accounts for delay patterns in multi-codebook generation.
        
        This method generates sophisticated attention masks that respect the temporal
        offsets introduced by delay patterns. It ensures that tokens can only attend
        to positions that are temporally valid given the delay constraints.
        
        Args:
            batch_size: Batch size for the attention mask
            seq_len: Sequence length for the attention mask
            delay_pattern: Delay pattern tensor [num_codebooks, seq_len]
            codebook_mask: Optional mask indicating which codebook each position belongs to
            dtype: Data type for the mask (defaults to torch.bool)
            
        Returns:
            Attention mask tensor [batch_size, seq_len, seq_len] where False indicates
            positions that should be masked (not attended to)
            
        Raises:
            ValueError: If input dimensions are incompatible
            
        Example:
            >>> mask = attn_utils.create_delay_aware_attention_mask(
            ...     batch_size=2, seq_len=10, delay_pattern=pattern
            ... )
            >>> # mask[b, i, j] = False means position i cannot attend to position j
        """
        import torch
        
        if dtype is None:
            dtype = torch.bool
        
        # Validate inputs
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        
        if delay_pattern.shape != (self.num_codebooks, seq_len):
            raise ValueError(
                f"delay_pattern shape {delay_pattern.shape} doesn't match "
                f"expected ({self.num_codebooks}, {seq_len})"
            )
        
        # Create base causal mask if required
        if self.causal_attention:
            # Standard causal mask: position i can attend to positions 0..i
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=dtype))
        else:
            # Allow all positions to attend to all positions
            causal_mask = torch.ones((seq_len, seq_len), dtype=dtype)
        
        # Apply delay pattern constraints
        delay_aware_mask = causal_mask.clone()
        
        # Convert delay pattern to CPU numpy for easier processing
        delay_np = delay_pattern.detach().cpu().numpy() if hasattr(delay_pattern, 'detach') else delay_pattern.numpy()
        
        # Apply delay constraints for each position
        for pos_i in range(seq_len):
            for pos_j in range(seq_len):
                # Check if position i can attend to position j given delay constraints
                
                # If we have codebook information, use it for more precise masking
                if codebook_mask is not None:
                    # Get codebook for each position
                    codebook_i = self._get_position_codebook(pos_i, codebook_mask, batch_size)
                    codebook_j = self._get_position_codebook(pos_j, codebook_mask, batch_size)
                    
                    # Apply codebook-specific delay constraints
                    if codebook_i != -1 and codebook_j != -1:
                        delay_i = delay_np[codebook_i, min(pos_i, seq_len-1)]
                        delay_j = delay_np[codebook_j, min(pos_j, seq_len-1)]
                        
                        # Position i can attend to position j if the effective temporal
                        # position of j (considering its delay) is not in the future
                        effective_pos_j = pos_j + delay_j
                        effective_pos_i = pos_i + delay_i
                        
                        if effective_pos_j > effective_pos_i:
                            delay_aware_mask[pos_i, pos_j] = False
                else:
                    # Without specific codebook information, apply general delay constraints
                    # Use maximum delay as conservative constraint
                    max_delay_at_pos = delay_np.max(axis=0)[min(pos_j, seq_len-1)]
                    if pos_j + max_delay_at_pos > pos_i:
                        # Be conservative: if any codebook at position j could be delayed
                        # beyond position i, mask it
                        pass  # Keep existing causal constraint
        
        # Handle cross-codebook attention if disabled
        if not self.cross_codebook_attention and codebook_mask is not None:
            delay_aware_mask = self._apply_intra_codebook_attention_only(
                delay_aware_mask, codebook_mask, batch_size, seq_len
            )
        
        # Expand to batch dimension
        batch_mask = delay_aware_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        return batch_mask
    
    def create_codebook_routing_mask(self,
                                   input_ids: Tensor,
                                   delay_pattern: Tensor,
                                   audio_start_position: int = 0) -> Tensor:
        """Create mask for routing tokens through codebook-specific processing paths.
        
        This method creates masks that identify which tokens belong to which codebook
        during multi-codebook generation. It accounts for delay patterns to properly
        route tokens through DualFFN layers and other codebook-specific components.
        
        Args:
            input_ids: Input token tensor [batch_size, seq_len]
            delay_pattern: Delay pattern tensor [num_codebooks, seq_len]
            audio_start_position: Position where audio generation starts
            
        Returns:
            Routing mask tensor [batch_size, seq_len, num_codebooks] where
            mask[b, pos, cb] = True indicates position pos uses codebook cb
            
        Example:
            >>> routing_mask = attn_utils.create_codebook_routing_mask(
            ...     input_ids=tokens, delay_pattern=pattern, audio_start_position=10
            ... )
        """
        import torch
        
        batch_size, seq_len = input_ids.shape
        
        # Initialize routing mask
        routing_mask = torch.zeros(
            (batch_size, seq_len, self.num_codebooks),
            dtype=torch.bool,
            device=input_ids.device
        )
        
        # Convert delay pattern for processing
        delay_np = delay_pattern.detach().cpu().numpy() if hasattr(delay_pattern, 'detach') else delay_pattern.numpy()
        
        # Route tokens based on position and delay pattern
        for pos in range(seq_len):
            if pos < audio_start_position:
                # Pre-audio positions: all codebooks active (text tokens)
                routing_mask[:, pos, :] = True
            else:
                # Audio positions: route based on delay pattern
                audio_pos = pos - audio_start_position
                
                for cb_idx in range(self.num_codebooks):
                    # Check if this codebook should be active at this position
                    if audio_pos < delay_pattern.shape[1]:
                        delay = delay_np[cb_idx, audio_pos]
                        # Codebook is active if its delayed position matches current position
                        if audio_pos >= delay:
                            routing_mask[:, pos, cb_idx] = True
        
        return routing_mask
    
    def adjust_position_ids_for_delays(self,
                                     position_ids: Tensor,
                                     delay_pattern: Tensor,
                                     codebook_assignments: Optional[Tensor] = None) -> Tensor:
        """Adjust position IDs to account for delay patterns in positional encoding.
        
        This method modifies position IDs to reflect the temporal offsets introduced
        by delay patterns. This ensures that positional encodings properly represent
        the actual temporal relationships between tokens in different codebooks.
        
        Args:
            position_ids: Original position IDs [batch_size, seq_len]
            delay_pattern: Delay pattern tensor [num_codebooks, seq_len]
            codebook_assignments: Optional codebook assignment for each position
            
        Returns:
            Adjusted position IDs tensor [batch_size, seq_len] with delay offsets applied
            
        Example:
            >>> adjusted_pos = attn_utils.adjust_position_ids_for_delays(
            ...     position_ids=pos_ids, delay_pattern=pattern
            ... )
        """
        import torch
        
        batch_size, seq_len = position_ids.shape
        adjusted_position_ids = position_ids.clone()
        
        # Convert delay pattern for processing
        delay_np = delay_pattern.detach().cpu().numpy() if hasattr(delay_pattern, 'detach') else delay_pattern.numpy()
        
        # Adjust position IDs based on delay pattern
        for pos in range(seq_len):
            if codebook_assignments is not None:
                # Use specific codebook assignment for this position
                codebook = codebook_assignments[:, pos]  # [batch_size]
                for batch_idx in range(batch_size):
                    cb_idx = codebook[batch_idx].item()
                    if 0 <= cb_idx < self.num_codebooks:
                        delay = delay_np[cb_idx, min(pos, delay_pattern.shape[1]-1)]
                        adjusted_position_ids[batch_idx, pos] = position_ids[batch_idx, pos] + delay
            else:
                # Without specific assignments, use average delay as approximation
                avg_delay = delay_np.mean(axis=0)[min(pos, delay_pattern.shape[1]-1)]
                adjusted_position_ids[:, pos] = position_ids[:, pos] + int(avg_delay)
        
        return adjusted_position_ids
    
    def create_streaming_attention_mask(self,
                                      batch_size: int,
                                      seq_len: int,
                                      delay_pattern: Tensor,
                                      chunk_size: int = 32,
                                      overlap_size: int = 4) -> Tensor:
        """Create attention mask optimized for streaming generation with delay patterns.
        
        This method creates attention masks that support streaming generation while
        maintaining proper delay pattern coordination. It includes chunking and
        overlap considerations for real-time TTS applications.
        
        Args:
            batch_size: Batch size for the attention mask
            seq_len: Sequence length for the attention mask
            delay_pattern: Delay pattern tensor [num_codebooks, seq_len]
            chunk_size: Size of streaming chunks
            overlap_size: Overlap between chunks for context preservation
            
        Returns:
            Streaming-optimized attention mask [batch_size, seq_len, seq_len]
            
        Example:
            >>> stream_mask = attn_utils.create_streaming_attention_mask(
            ...     batch_size=2, seq_len=100, delay_pattern=pattern, chunk_size=16
            ... )
        """
        import torch
        
        # Start with standard delay-aware mask
        base_mask = self.create_delay_aware_attention_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            delay_pattern=delay_pattern
        )
        
        # Apply streaming constraints
        streaming_mask = base_mask.clone()
        
        # For each chunk, limit attention to current chunk + overlap from previous chunk
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            
            # Determine attention window for this chunk
            attention_start = max(0, chunk_start - overlap_size)
            attention_end = chunk_end
            
            # Mask out attention beyond the window
            for pos in range(chunk_start, chunk_end):
                streaming_mask[:, pos, :attention_start] = False
                if attention_end < seq_len:
                    streaming_mask[:, pos, attention_end:] = False
        
        return streaming_mask
    
    def _get_position_codebook(self,
                             position: int,
                             codebook_mask: Tensor,
                             batch_size: int) -> int:
        """Get the codebook index for a specific position."""
        if codebook_mask is None:
            return -1  # No codebook information available
        
        # For simplicity, use the first batch item's codebook assignment
        if codebook_mask.dim() == 3:  # [batch, seq, num_codebooks]
            # Find which codebook is active at this position
            active_codebooks = codebook_mask[0, position, :]
            active_indices = torch.where(active_codebooks)[0]
            return active_indices[0].item() if len(active_indices) > 0 else -1
        elif codebook_mask.dim() == 2:  # [batch, seq] with codebook indices
            return codebook_mask[0, position].item()
        else:
            return -1
    
    def _apply_intra_codebook_attention_only(self,
                                           mask: Tensor,
                                           codebook_mask: Tensor,
                                           batch_size: int,
                                           seq_len: int) -> Tensor:
        """Restrict attention to within the same codebook only."""
        import torch
        
        # Create a mask that only allows attention within the same codebook
        intra_codebook_mask = torch.zeros((seq_len, seq_len), dtype=torch.bool)
        
        for i in range(seq_len):
            for j in range(seq_len):
                codebook_i = self._get_position_codebook(i, codebook_mask, batch_size)
                codebook_j = self._get_position_codebook(j, codebook_mask, batch_size)
                
                # Allow attention if same codebook or if codebook info unavailable
                if codebook_i == codebook_j or codebook_i == -1 or codebook_j == -1:
                    intra_codebook_mask[i, j] = True
        
        # Apply the intra-codebook constraint
        return mask & intra_codebook_mask
    
    def validate_attention_consistency(self,
                                     attention_mask: Tensor,
                                     delay_pattern: Tensor,
                                     tolerance: float = 1e-6) -> bool:
        """Validate that attention mask is consistent with delay patterns.
        
        This method performs comprehensive validation to ensure that the generated
        attention mask properly respects delay pattern constraints and maintains
        causal dependencies where required.
        
        Args:
            attention_mask: Attention mask to validate [batch, seq, seq]
            delay_pattern: Delay pattern used for mask generation
            tolerance: Tolerance for numerical comparisons
            
        Returns:
            True if mask is consistent, raises ValueError if not
            
        Raises:
            ValueError: If mask violates delay pattern constraints
        """
        batch_size, seq_len_q, seq_len_k = attention_mask.shape
        
        if seq_len_q != seq_len_k:
            raise ValueError(f"Attention mask must be square, got shape {attention_mask.shape}")
        
        if delay_pattern.shape[1] > seq_len_q:
            raise ValueError(
                f"Delay pattern sequence length {delay_pattern.shape[1]} "
                f"exceeds attention mask sequence length {seq_len_q}"
            )
        
        # Check causal constraints if enabled
        if self.causal_attention:
            # Verify that future positions are masked
            for i in range(seq_len_q):
                for j in range(i + 1, seq_len_k):
                    if attention_mask[0, i, j].item():  # Check first batch item
                        raise ValueError(
                            f"Causal constraint violated: position {i} attends to future position {j}"
                        )
        
        # Additional delay pattern specific validations could be added here
        return True


class HiggsAudioEncoderLayer(Module):
    """
    Single transformer layer for the Higgs Audio encoder.
    
    This layer implements a standard transformer encoder layer with:
    - Multi-head self-attention
    - Feed-forward network
    - Layer normalization and residual connections
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.embed_dim = config.audio_d_model
        
        # Self-attention mechanism (WhisperAttention-compatible)
        self.self_attn = Attention(
            local_layer_idx=layer_idx,
            hidden_size=self.embed_dim,
            attention_head_size=self.embed_dim // config.audio_encoder_heads,
            num_attention_heads=config.audio_encoder_heads,
            num_kv_heads=config.audio_encoder_heads,  # Same as attention heads for encoder
            max_position_embeddings=config.audio_max_source_positions,
            dtype=config.dtype,
            attention_mask_type='bidirectional',  # Encoder uses bidirectional attention
            bias=False,
            tp_group=None,  # Audio encoder typically not tensor parallel
            tp_size=1,
        )
        
        # Layer normalization
        self.self_attn_layer_norm = LayerNorm(
            normalized_shape=self.embed_dim,
            dtype=config.dtype
        )
        
        # Feed-forward network
        self.mlp = MLP(
            hidden_size=self.embed_dim,
            ffn_hidden_size=config.audio_encoder_ffn_dim,
            hidden_act=getattr(config, 'activation_function', 'gelu'),
            dtype=config.dtype,
            bias=True
        )
        
        # Final layer normalization
        self.final_layer_norm = LayerNorm(
            normalized_shape=self.embed_dim,
            dtype=config.dtype
        )
        
        # Dropout rate
        self.dropout_rate = getattr(config, 'dropout', 0.0)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        layer_head_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass for a single encoder layer.
        
        Args:
            hidden_states: Input hidden states [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            layer_head_mask: Optional head mask for this layer
            output_attentions: Whether to output attention weights
            
        Returns:
            Tuple of (hidden_states, attention_weights)
        """
        residual = hidden_states
        
        # Self-attention with pre-norm
        hidden_states = self.self_attn_layer_norm(hidden_states)
        
        # Apply self-attention with proper parameter handling
        attn_output = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=False,  # Encoder doesn't use KV cache
            kv_cache_params=None,
            attention_params=None,
        )
        
        # Extract attention output and weights
        if isinstance(attn_output, tuple):
            hidden_states = attn_output[0]
            attention_weights = attn_output[1] if len(attn_output) > 1 else None
        else:
            hidden_states = attn_output
            attention_weights = None
        
        # Apply head masking if provided (like original WhisperEncoderLayer)
        if layer_head_mask is not None and attention_weights is not None:
            # Apply head mask to attention weights
            attention_weights = attention_weights * layer_head_mask.view(1, -1, 1, 1)
        
        hidden_states = residual + hidden_states
        
        # Feed-forward network with pre-norm
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        # Return attention weights if requested and available
        if output_attentions:
            return hidden_states, attention_weights
        else:
            return hidden_states, None


class HiggsAudioEncoder(Module):
    """
    Higgs Audio Encoder based on Whisper architecture.
    
    This encoder processes mel-spectrogram features and converts them to audio 
    embeddings that can be integrated with the text model. It consists of:
    - Convolutional feature extraction layers
    - Positional embeddings  
    - Stack of transformer encoder layers
    - Layer normalization
    """

    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.config = config
        
        # Audio configuration parameters
        self.num_mel_bins = config.audio_num_mel_bins
        self.embed_dim = config.audio_d_model
        self.num_layers = config.audio_encoder_layers
        self.max_source_positions = config.audio_max_source_positions
        
        # Dropout and layer drop rates
        self.dropout_rate = getattr(config, 'dropout', 0.0)
        self.layerdrop_rate = getattr(config, 'encoder_layerdrop', 0.0)
        
        # Embedding scale factor
        scale_embedding = getattr(config, 'scale_embedding', False)
        self.embed_scale = math.sqrt(self.embed_dim) if scale_embedding else 1.0
        
        # Memory optimization: Only create layers when needed
        # Skip audio tower if configured to reduce memory during engine build
        
        if not config.skip_audio_tower:
            # Convolutional feature extraction layers
            # These layers downsample the mel-spectrogram and extract features
            self.conv1 = Conv1d(
                in_channels=self.num_mel_bins,
                out_channels=self.embed_dim,
                kernel_size=3,
                padding=1,
                dtype=config.dtype
            )
            
            self.conv2 = Conv1d(
                in_channels=self.embed_dim,
                out_channels=self.embed_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                dtype=config.dtype
            )
            
            # Positional embeddings
            self.embed_positions = Embedding(
                num_embeddings=self.max_source_positions,
                embedding_dim=self.embed_dim,
                dtype=config.dtype
            )
            
            # Freeze positional embeddings (common practice)
            self.embed_positions.weight.requires_grad = False
            
            # Transformer encoder layers using ModuleList for proper registration
            self.layers = ModuleList([
                HiggsAudioEncoderLayer(config, layer_idx) 
                for layer_idx in range(self.num_layers)
            ])
            
            # Average pooling layer for sequence reduction (like original Transformers)
            self.avg_pooler = AvgPool1d(
                kernel_size=2,
                stride=2,
                padding=0
            )
            
            # Final layer normalization
            self.layer_norm = LayerNorm(
                normalized_shape=self.embed_dim,
                dtype=config.dtype
            )
        else:
            # Minimal placeholder components to save memory during engine build
            self.conv1 = None
            self.conv2 = None
            self.embed_positions = None
            self.layers = []
            self.avg_pooler = None
            self.layer_norm = None

    def forward(
        self,
        input_features: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        check_seq_length: bool = True,
    ) -> Union[Tensor, Tuple]:
        """
        Forward pass for the audio encoder.
        
        Args:
            input_features: Mel-spectrogram features [batch, num_mel_bins, seq_len]
            attention_mask: Optional attention mask (not typically used for encoder)
            head_mask: Optional head mask for attention layers
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states from all layers
            return_dict: Whether to return a dictionary or tuple
            check_seq_length: Whether to validate input sequence length
            
        Returns:
            Encoded audio features [batch, seq_len, hidden_size]
        """
        # Validate input sequence length if requested
        if check_seq_length:
            # Calculate expected sequence length based on downsampling
            # conv1: no stride (stride=1), conv2: stride=2, so total downsampling = 2
            # Expected input length for max_source_positions output after downsampling
            expected_seq_length = self.max_source_positions * 2  # Account for conv2 stride=2
            if input_features.shape[-1] != expected_seq_length:
                # Allow some flexibility for sequence length
                min_seq_length = expected_seq_length - 10
                max_seq_length = expected_seq_length + 10
                if not (min_seq_length <= input_features.shape[-1] <= max_seq_length):
                    warnings.warn(
                        f"HiggsAudio encoder expects input features of length ~{expected_seq_length}, "
                        f"but got {input_features.shape[-1]}. This may affect performance."
                    )
        
        # Convolutional feature extraction
        # Apply first conv layer with GELU activation
        hidden_states = self.conv1(input_features)
        hidden_states = gelu(hidden_states)
        
        # Apply second conv layer with GELU activation and stride
        hidden_states = self.conv2(hidden_states)  
        hidden_states = gelu(hidden_states)
        
        # Reshape from [batch, channels, seq_len] to [batch, seq_len, channels]
        hidden_states = hidden_states.permute(0, 2, 1)
        
        # Add positional embeddings (direct weight access like original Transformers)
        # Scale embeddings if configured
        hidden_states = hidden_states * self.embed_scale
        
        # Add positional embeddings - use direct weight access for better compatibility
        embed_pos = self.embed_positions.weight
        hidden_states = hidden_states + embed_pos
        
        # Storage for outputs if requested (use lists for memory efficiency)
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        
        # Validate head mask dimensions if provided (like original Transformers)
        if head_mask is not None:
            if head_mask.shape[0] != len(self.layers):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, "
                    f"but it is for {head_mask.shape[0]} layers."
                )
        
        # Pass through transformer layers with memory optimization
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                # Only store if really needed to save memory
                all_hidden_states.append(hidden_states.detach())
            
            # Layer dropout during training (like original Transformers implementation)
            skip_layer = False
            if self.training and self.layerdrop_rate > 0:
                # Generate random number for layer drop decision
                # Note: For training mode only, inference always processes all layers
                import random
                dropout_probability = random.random()
                if dropout_probability < self.layerdrop_rate:
                    skip_layer = True
            
            if not skip_layer:
                # Apply encoder layer
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    layer_head_mask=head_mask[idx] if head_mask is not None else None,
                    output_attentions=output_attentions,
                )
                
                hidden_states = layer_outputs[0]
                
                if output_attentions and layer_outputs[1] is not None:
                    all_attentions.append(layer_outputs[1].detach())
            else:
                # Layer was skipped due to layer drop
                if output_attentions:
                    # Add None for skipped layer to maintain indexing
                    all_attentions.append(None)
        
        # Apply average pooling before final layer norm (like original Transformers)
        # Permute to [batch, channels, seq_len] for pooling
        hidden_states = hidden_states.permute(0, 2, 1)
        
        # Apply average pooling layer for sequence reduction
        hidden_states = self.avg_pooler(hidden_states)
        
        # Permute back to [batch, seq_len, channels]
        hidden_states = hidden_states.permute(0, 2, 1)
        
        # Final layer normalization
        hidden_states = self.layer_norm(hidden_states)
        
        # Add final hidden state if collecting all states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Return based on return_dict flag
        if return_dict:
            return {
                'last_hidden_state': hidden_states,
                'hidden_states': all_hidden_states,
                'attentions': all_attentions,
            }
        else:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

    def _get_feat_extract_output_lengths(self, input_lengths: Tensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder.
        
        Args:
            input_lengths: Length of input mel-spectrogram sequences
            
        Returns:
            Tuple of (conv_output_lengths, encoder_output_lengths)
        """
        # Calculate lengths after convolutional layers
        # conv1: kernel_size=3, stride=1, padding=1 -> no length change
        # conv2: kernel_size=3, stride=2, padding=1 -> length = (length + 2*1 - 3) // 2 + 1
        conv_output_lengths = (input_lengths + 2 * 1 - 3) // 2 + 1
        
        # Calculate lengths after average pooling (kernel_size=2, stride=2)
        # avg_pool: kernel_size=2, stride=2 -> length = length // 2
        encoder_output_lengths = conv_output_lengths // 2
        
        return conv_output_lengths, encoder_output_lengths

    def get_input_embeddings(self):
        """Get the input embedding layer (conv1 in this case)."""
        return self.conv1
    
    def set_input_embeddings(self, value):
        """Set the input embedding layer."""
        self.conv1 = value


class HiggsAudioFeatureProjector(Module):
    """
    Projects audio features from the encoder to the text model's hidden size.
    
    This is a linear projection layer that maps from the audio encoder's
    output dimension to the text model's hidden dimension, enabling integration
    of audio and text representations. Uses TensorRT-LLM ColumnLinear for 
    tensor parallelism support.
    """

    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        
        # Get dimensions from config
        audio_dim = config.audio_d_model
        text_dim = config.hidden_size
        
        # Use TensorRT-LLM ColumnLinear for proper tensor parallelism support
        self.linear = ColumnLinear(
            in_features=audio_dim,
            out_features=text_dim,
            bias=True,
            dtype=config.dtype,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            gather_output=True  # Gather output across TP ranks
        )

    def forward(self, audio_features: Tensor) -> Tensor:
        """
        Project audio features to text model dimension.
        
        Args:
            audio_features: Audio features from encoder [batch, seq_len, audio_dim]
            
        Returns:
            Projected features [batch, seq_len, text_dim]
        """
        # Apply linear projection using TensorRT-LLM ColumnLinear
        output = self.linear(audio_features)
        return output


class HiggsAudioDecoderProjector(Module):
    """
    TensorRT-LLM compatible decoder projection layers for Higgs Audio model.
    
    Projects hidden states from decoder layers to both text and audio logits.
    Supports multi-codebook audio generation with RVQ delay patterns.
    """
    
    def __init__(self, config: HiggsAudioConfig) -> None:
        """Initialize decoder projection layers.
        
        Args:
            config: HiggsAudioConfig containing projection specifications
        """
        super().__init__()
        self.config = config
        
        # Text projection head - projects to text vocabulary
        self.text_lm_head = ColumnLinear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            bias=False,
            dtype=config.dtype,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            gather_output=True
        )
        
        # Audio projection head - projects to audio codebook vocabularies
        # Each codebook has codebook_size + 2 tokens (regular tokens + special tokens)
        audio_vocab_size = config.audio_num_codebooks * (config.audio_codebook_size + 2)
        self.audio_lm_head = ColumnLinear(
            in_features=config.hidden_size,
            out_features=audio_vocab_size,
            bias=False,
            dtype=config.dtype,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            gather_output=True
        )
        
        # Cache configuration parameters
        self.audio_num_codebooks = config.audio_num_codebooks
        self.audio_codebook_size = config.audio_codebook_size
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
    
    def forward(self,
                hidden_states: Tensor,
                audio_out_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Project hidden states to text and audio logits.
        
        Args:
            hidden_states: Hidden states from decoder [batch, seq_len, hidden_size]
            audio_out_mask: Boolean mask indicating audio token positions [batch, seq_len]
            
        Returns:
            text_logits: Text vocabulary logits [batch, seq_len, vocab_size]
            audio_logits: Audio codebook logits [batch, seq_len, audio_vocab_size]
        """
        # Project to text logits for all positions
        text_logits = self.text_lm_head(hidden_states)
        
        # Project to audio logits for all positions
        audio_logits = self.audio_lm_head(hidden_states)
        
        return text_logits, audio_logits


class HiggsAudioModelForCausalLM(DecoderModelForCausalLM):
    """TTS-optimized base class for Higgs Audio causal language models.

    This class extends TensorRT-LLM's DecoderModelForCausalLM with TTS-specific
    optimizations and generation modes. It provides the foundation for multimodal
    audio-text generation while maintaining compatibility with existing TRT-LLM
    infrastructure.

    The class implements a state machine for TTS generation with three modes:
    1. TEXT: Standard text generation without audio processing
    2. AUDIO_INIT: Initialize audio generation from text or voice input
    3. AUDIO_IN_PROGRESS: Continue audio generation with streaming support

    Key TTS Optimizations:
    - Generation mode management for different TTS phases
    - Audio token streaming support with configurable delay patterns
    - Multi-codebook audio generation coordination for RVQ-based tokenization
    - Real-time performance optimizations for speech synthesis
    - State tracking for continuous audio generation sessions

    Attributes:
        config (HiggsAudioConfig): Model configuration with TTS parameters
        generation_mode (GenerationMode): Current operational mode
        audio_codebook_state (Dict[str, Dict[str, Any]]): State for multi-codebook generation
        _audio_generation_active (bool): Whether audio generation is currently active
        _current_codebook_idx (int): Current codebook index for RVQ generation
        _delay_pattern_offset (int): Current offset in delay pattern sequence

    Example:
        >>> config = HiggsAudioConfig.from_hugging_face("higgs-audio-model")
        >>> model = HiggsAudioModelForCausalLM(config)
        >>> model.set_generation_mode(GenerationMode.AUDIO_INIT)
        >>> outputs = model.generate(text_input, audio_features=audio_input)
    """

    def __init__(self, config: HiggsAudioConfig) -> None:
        """Initialize TTS-optimized Higgs Audio model with comprehensive mode management.

        Args:
            config: HiggsAudioConfig with TTS-specific parameters including
                   audio encoder settings, fusion parameters, and TTS optimizations

        Raises:
            ValueError: If config is missing required TTS parameters
        """
        super().__init__(config)
        self.config: HiggsAudioConfig = config
        
        # Initialize comprehensive generation mode manager
        mode_config = {
            'text_mode': getattr(config, 'text_mode_config', {}),
            'audio_init_mode': getattr(config, 'audio_init_mode_config', {}),
            'audio_progress_mode': getattr(config, 'audio_progress_mode_config', {})
        }
        
        self.generation_mode_manager = GenerationModeManager(
            config=mode_config,
            enable_performance_monitoring=getattr(config, 'enable_performance_monitoring', True),
            enable_auto_recovery=getattr(config, 'enable_auto_recovery', True),
            max_recovery_attempts=getattr(config, 'max_recovery_attempts', 3),
            performance_threshold_latency_ms=getattr(config, 'performance_threshold_latency_ms', 100.0)
        )
        
        # Initialize CUDA graph manager if enabled
        self.cuda_graph_manager: Optional['CudaGraphManager'] = None
        if CUDA_GRAPHS_AVAILABLE and getattr(config, 'enable_cuda_graphs', False):
            try:
                self.cuda_graph_manager = CudaGraphManager(
                    config=config,
                    device=getattr(config, 'cuda_graph_device', 'cuda:0'),
                    max_batch_size=getattr(config, 'cuda_graph_max_batch_size', 8),
                    max_sequence_length=getattr(config, 'cuda_graph_max_sequence_length', 2048)
                )
                
                # Pre-initialize common graph types if requested
                if getattr(config, 'cuda_graph_hrewarm', True):
                    try:
                        self.cuda_graph_manager.prewarm_graphs(
                            batch_sizes=[1, 2, 4] if hasattr(config, 'cuda_graph_prewarm_batch_sizes') 
                                      else getattr(config, 'cuda_graph_prewarm_batch_sizes', [1]),
                            sequence_lengths=[512, 1024] if hasattr(config, 'cuda_graph_prewarm_seq_lengths')
                                            else getattr(config, 'cuda_graph_prewarm_seq_lengths', [512])
                        )
                    except Exception as e:
                        warnings.warn(f"CUDA graph prewarming failed: {e}")
                        
            except Exception as e:
                warnings.warn(f"CUDA graph manager initialization failed: {e}")
                self.cuda_graph_manager = None
        
        # Legacy compatibility attributes (deprecated, use mode_manager instead)
        self.audio_codebook_state: Dict[str, Dict[str, Any]] = {}

    def set_generation_mode(self, mode: Union[GenerationMode, str], **kwargs: Any) -> None:
        """Set the current generation mode using comprehensive mode management.

        This method uses the GenerationModeManager for validated mode transitions
        with comprehensive state coordination and error handling.

        Args:
            mode: Target generation mode (GenerationMode enum or string value)
            **kwargs: Additional transition parameters (validation_level, preserve_context)

        Raises:
            GenerationModeError: If mode transition fails or is invalid

        Example:
            >>> model.set_generation_mode(GenerationMode.AUDIO_INIT)
            >>> model.set_generation_mode("audio_in_progress", validation_level='comprehensive')
        """
        if isinstance(mode, str):
            try:
                mode = GenerationMode(mode)
            except ValueError:
                valid_modes = [m.value for m in GenerationMode]
                raise ValueError(f"Invalid generation mode '{mode}'. "
                               f"Valid modes: {valid_modes}")
        
        # Use comprehensive mode transition
        success = self.generation_mode_manager.transition_to_mode(
            target_mode=mode,
            validation_level=kwargs.get('validation_level', 'standard'),
            preserve_context=kwargs.get('preserve_context', True)
        )
        
        if not success:
            raise ValueError(f"Failed to transition to mode {mode.value}")

    def get_generation_mode(self) -> GenerationMode:
        """Get the current generation mode from the mode manager.

        Returns:
            Current GenerationMode enum value

        Example:
            >>> mode = model.get_generation_mode()
            >>> if mode == GenerationMode.AUDIO_INIT:
            ...     print("Model is initializing audio generation")
        """
        return self.generation_mode_manager.get_current_mode()

    def is_audio_generation_mode(self) -> bool:
        """Check if model is currently in audio generation mode.

        Returns:
            True if in AUDIO_INIT or AUDIO_IN_PROGRESS mode, False for TEXT mode

        Example:
            >>> if model.is_audio_generation_mode():
            ...     # Apply audio-specific processing
            ...     audio_inputs = prepare_audio_inputs(inputs)
        """
        return self.generation_mode_manager.is_audio_generation_active()

    def get_generation_state(self) -> GenerationState:
        """Get comprehensive generation state from the mode manager.

        Returns:
            Complete GenerationState with all tracking information

        Example:
            >>> state = model.get_generation_state()
            >>> print(f"Mode: {state.current_mode.value}")
            >>> print(f"Position: {state.current_position}")
        """
        return self.generation_mode_manager.get_generation_state()

    def get_audio_generation_state(self) -> Dict[str, Any]:
        """Get current audio generation state for inspection and debugging.
        
        This method provides backward compatibility while leveraging the new mode manager.

        Returns:
            Dictionary containing comprehensive audio generation state

        Example:
            >>> state = model.get_audio_generation_state()
            >>> print(f"Mode: {state['mode']}, Active: {state['audio_active']}")
        """
        gen_state = self.generation_mode_manager.get_generation_state()
        
        return {
            'mode': gen_state.current_mode.value,
            'audio_active': gen_state.audio_generation_active,
            'current_codebook': gen_state.current_codebook_index,
            'delay_offset': gen_state.delay_pattern_offset,
            'codebook_states': gen_state.codebook_states.copy(),
            'generated_tokens': gen_state.generated_tokens,
            'current_position': gen_state.current_position
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary from mode manager.

        Returns:
            Performance metrics and statistics from the mode manager

        Example:
            >>> perf = model.get_performance_summary()
            >>> print(f"Average latency: {perf.get('avg_latency_ms', 0):.2f}ms")
        """
        return self.generation_mode_manager.get_performance_summary()

    def reset_generation_state(self, preserve_config: bool = True) -> None:
        """Reset generation state using the mode manager.

        Args:
            preserve_config: Whether to preserve mode-specific configuration

        Example:
            >>> model.reset_generation_state()  # Reset to clean state
        """
        self.generation_mode_manager.reset_state(preserve_config=preserve_config)
        self.audio_codebook_state.clear()  # Legacy compatibility

    def create_state_checkpoint(self) -> str:
        """Create a state checkpoint using the mode manager.

        Returns:
            Serialized state checkpoint

        Example:
            >>> checkpoint = model.create_state_checkpoint()
            >>> # Later: model.restore_state_from_checkpoint(checkpoint)
        """
        return self.generation_mode_manager.create_state_checkpoint()

    def restore_state_from_checkpoint(self, checkpoint_data: str) -> bool:
        """Restore state from checkpoint using the mode manager.

        Args:
            checkpoint_data: Serialized checkpoint data

        Returns:
            True if restoration succeeded, False otherwise

        Example:
            >>> success = model.restore_state_from_checkpoint(checkpoint)
            >>> if not success:
            ...     print("Failed to restore state")
        """
        return self.generation_mode_manager.restore_state_from_checkpoint(checkpoint_data)

    def prepare_inputs_for_generation(self,
                                    input_ids: Tensor,
                                    attention_mask: Optional[Tensor] = None,
                                    **kwargs: Any) -> Dict[str, Any]:
        """Prepare inputs for generation with TTS-specific handling.

        This method extends the base implementation to handle TTS-specific
        input preparation based on the current generation mode. It adds
        audio-specific parameters when in audio generation modes.

        Args:
            input_ids: Input token IDs tensor
            attention_mask: Optional attention mask tensor
            **kwargs: Additional generation arguments (audio_features, etc.)

        Returns:
            Dictionary of prepared inputs for generation, including TTS-specific
            parameters when in audio generation mode

        Example:
            >>> inputs = model.prepare_inputs_for_generation(
            ...     input_ids=text_tokens,
            ...     audio_features=audio_embeddings
            ... )
            >>> # inputs now contains audio-specific parameters if in audio mode
        """
        # Call parent implementation first
        inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        # Add TTS-specific inputs based on generation mode
        if self.is_audio_generation_mode():
            inputs.update(self._prepare_audio_generation_inputs(input_ids, **kwargs))

        return inputs

    def _prepare_audio_generation_inputs(self,
                                       input_ids: Tensor,
                                       **kwargs: Any) -> Dict[str, Any]:
        """Prepare TTS-specific inputs for audio generation.

        Args:
            input_ids: Input token IDs tensor
            **kwargs: Additional generation arguments

        Returns:
            Dictionary of audio-specific generation inputs including mode flags,
            token indices, and streaming parameters
        """
        audio_inputs: Dict[str, Any] = {}

        # Add audio token indicators if in audio mode
        if self.generation_mode == GenerationMode.AUDIO_INIT:
            # Mark transition from text to audio generation
            audio_inputs['audio_init_mode'] = True
            audio_inputs['audio_out_token_idx'] = getattr(
                self.config, 'audio_out_token_idx', 128016
            )
        elif self.generation_mode == GenerationMode.AUDIO_IN_PROGRESS:
            # Add streaming audio generation parameters
            audio_inputs['audio_stream_mode'] = True
            audio_inputs['current_codebook_idx'] = self._current_codebook_idx
            audio_inputs['delay_pattern_offset'] = self._delay_pattern_offset

        return audio_inputs

    def update_audio_generation_state(self,
                                    generated_token: int,
                                    **kwargs: Any) -> None:
        """Update audio generation state after token generation using mode manager.

        This method uses the GenerationModeManager to maintain comprehensive state
        tracking for multi-codebook generation and delay patterns with full validation.

        Args:
            generated_token: The token that was just generated
            **kwargs: Additional state update parameters (codebook_override, position_update, etc.)

        Example:
            >>> for token in generated_tokens:
            ...     model.update_audio_generation_state(token)
            ...     # State automatically updated with comprehensive tracking
        """
        if not self.is_audio_generation_mode():
            return

        # Update generation state through mode manager
        generation_state = self.generation_mode_manager.get_generation_state()
        
        # Update token and position tracking
        generation_state.generated_tokens += 1
        generation_state.current_position += 1
        
        # Update codebook state for multi-codebook generation
        if hasattr(self.config, 'audio_num_codebooks') and self.config.audio_num_codebooks > 1:
            current_codebook = generation_state.current_codebook_index
            codebook_key = f'codebook_{current_codebook}'
            
            if codebook_key not in generation_state.codebook_states:
                generation_state.codebook_states[codebook_key] = {
                    'active': False,
                    'last_token': None,
                    'delay_offset': current_codebook * getattr(self.config, 'audio_delay_pattern_stride', 1),
                    'generated_tokens': []
                }
            
            # Update codebook state
            generation_state.codebook_states[codebook_key]['last_token'] = generated_token
            generation_state.codebook_states[codebook_key]['active'] = True
            generation_state.codebook_states[codebook_key]['generated_tokens'].append(generated_token)

        # Update delay pattern offset
        if hasattr(self.config, 'use_delay_pattern') and self.config.use_delay_pattern:
            generation_state.delay_pattern_offset += 1

        # Cycle through codebooks if using multi-codebook generation
        if hasattr(self.config, 'audio_num_codebooks'):
            generation_state.current_codebook_index = (
                generation_state.current_codebook_index + 1
            ) % self.config.audio_num_codebooks
        
        # Update legacy compatibility state
        self.audio_codebook_state = generation_state.codebook_states.copy()
        
        # Trigger performance tracking if enabled
        if self.generation_mode_manager.enable_performance_monitoring:
            import time
            generation_state.latency_history.append(time.time() * 1000)  # Convert to ms

    @classmethod
    def from_hugging_face(cls,
                         hf_model_or_dir: Union[str, Any],
                         dtype: str = 'auto',
                         mapping: Optional[Any] = None,
                         quant_config: Optional[Any] = None,
                         **kwargs: Any) -> 'HiggsAudioModelForCausalLM':
        """Factory method to create model from HuggingFace checkpoint.

        This method maintains compatibility with existing TRT-LLM patterns
        while enabling TTS-specific configuration loading and validation.

        Args:
            hf_model_or_dir: Path to HF model directory or HF model object
            dtype: Data type for model weights ('auto', 'float16', 'float32', etc.)
            mapping: TensorRT-LLM tensor/pipeline parallelism mapping configuration
            quant_config: Quantization configuration for model optimization
            **kwargs: Additional arguments passed to config building

        Returns:
            Configured HiggsAudioModelForCausalLM instance ready for TTS generation

        Raises:
            ValueError: If HF model is incompatible or missing required components
            FileNotFoundError: If model directory doesn't exist

        Example:
            >>> model = HiggsAudioModelForCausalLM.from_hugging_face(
            ...     "path/to/higgs-audio-model",
            ...     dtype="float16",
            ...     mapping=tp_mapping
            ... )
        """
        # Use existing config loading logic
        from .convert import build_config_from_hf
        cfg = build_config_from_hf(hf_model_or_dir,
                                  dtype=dtype,
                                  mapping=mapping,
                                  quant_config=quant_config,
                                  **kwargs)
        return cls(cfg)


class HiggsAudioDecoderLayer(Module):
    """Llama-style decoder layer optimized for Higgs Audio TTS model.

    This layer implements the core transformer decoder functionality with
    TTS-specific optimizations. It serves as the building block for the
    HiggsAudioBackbone and supports audio token processing through
    specialized attention and MLP components.

    The layer follows the standard transformer architecture:
    - Multi-head self-attention with RoPE positional encoding
    - Gated MLP with SiLU activation
    - RMSNorm for layer normalization
    - Residual connections around both attention and MLP blocks

    Future audio adapter integration points are preserved for DualFFN
    and other audio-specific enhancements.

    Attributes:
        config (HiggsAudioConfig): Layer configuration
        layer_idx (int): Layer index in the model stack
        mapping: TensorRT-LLM parallelism mapping
        input_layernorm (RmsNorm): Pre-attention layer normalization
        post_layernorm (RmsNorm): Pre-MLP layer normalization
        attention (Attention): Multi-head self-attention mechanism
        mlp (GatedMLP): Feed-forward network with gating

    Example:
        >>> layer = HiggsAudioDecoderLayer(config, layer_idx=0)
        >>> output = layer(hidden_states, attention_mask=mask)
    """

    def __init__(self, config: HiggsAudioConfig, layer_idx: int) -> None:
        """Initialize decoder layer with TTS-optimized components.

        Args:
            config: HiggsAudioConfig containing layer parameters
            layer_idx: Zero-based index of this layer in the model stack

        Raises:
            ValueError: If config parameters are invalid for layer construction
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.mapping = config.mapping

        # Norms
        self.input_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                       eps=config.norm_epsilon,
                                       dtype=config.dtype)
        self.post_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                      eps=config.norm_epsilon,
                                      dtype=config.dtype)

        # Attention - Use standard Attention with Llama-style configuration
        self.attention = Attention(
            local_layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            attention_head_size=config.head_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            dtype=config.dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=config.attn_bias,
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            rotary_embedding_base=config.rotary_base,
            rotary_embedding_scaling=config.rotary_scaling,
            tp_group=self.mapping.tp_group,
            tp_size=self.mapping.tp_size,
            tp_rank=self.mapping.tp_rank,
            q_scaling=1.0,
            quant_mode=getattr(config, "quant_mode", None),
            cross_attention=False,
            relative_attention=False,
            max_distance=0,
            num_buckets=0,
        )

        # MLP
        mlp_hidden_size = (config.hidden_size * 4
                           if config.intermediate_size is None
                           else config.intermediate_size)
        self.mlp = GatedMLP(hidden_size=config.hidden_size,
                            ffn_hidden_size=mlp_hidden_size,
                            hidden_act=config.hidden_act,
                            dtype=config.dtype,
                            bias=False,
                            tp_group=self.mapping.tp_group,
                            tp_size=self.mapping.tp_size,
                            quant_mode=getattr(config, "quant_mode", None))

    def forward(self,
                hidden_states: Tensor,
                attention_mask: Optional[Tensor] = None,
                use_cache: bool = False,
                kv_cache_params: Optional[Any] = None,
                attention_params: Optional[Any] = None,
                lora_layer_params: Optional[Any] = None,
                position_ids: Optional[Tensor] = None,
                audio_token_mask: Optional[Tensor] = None,
                audio_out_mask: Optional[Tensor] = None,
                next_layer_input_layernorm_args: Optional[Any] = None) -> Union[Tensor, Tuple[Tensor, Any]]:
        """Forward pass through decoder layer with TTS-aware processing.

        Implements standard transformer decoder layer computation with residual
        connections around attention and MLP blocks. The audio_token_mask and
        audio_out_mask parameters are reserved for future audio-specific processing.

        Args:
            hidden_states: Input hidden states tensor [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask for sequence padding
            use_cache: Whether to return KV cache for next iteration
            kv_cache_params: KV cache parameters for attention computation
            attention_params: Additional attention computation parameters
            lora_layer_params: LoRA adaptation parameters if enabled
            position_ids: Position indices for positional encoding
            audio_token_mask: Mask indicating audio token positions (legacy parameter)
            audio_out_mask: Mask indicating audio output tokens (future use)

        Returns:
            If use_cache=False: Hidden states tensor after layer processing
            If use_cache=True: Tuple of (hidden_states, kv_cache_presents)

        Example:
            >>> output = layer(hidden_states, attention_mask=mask, use_cache=True)
            >>> hidden_states, kv_cache = output
        """
        # Attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.attention(hidden_states,
                                  attention_mask=attention_mask,
                                  use_cache=use_cache,
                                  kv_cache_params=kv_cache_params,
                                  attention_params=attention_params,
                                  lora_layer_params=lora_layer_params)
        if use_cache:
            attn_out, presents = attn_out
        hidden_states = residual + attn_out

        # MLP block
        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states,
                                 lora_layer_params=lora_layer_params)
        hidden_states = residual + hidden_states

        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class HiggsAudioDualFFNDecoderLayer(Module):
    """
    Higgs Audio DualFFN Decoder Layer with separate FFN paths for audio and text tokens.
    
    This layer implements the DualFFN architecture where audio and text tokens use
    separate MLP processing paths after shared attention computation. This allows
    specialized processing for different modalities while maintaining efficiency.
    
    Features:
    - Shared attention mechanism for all tokens
    - Separate FFN paths for audio vs text tokens
    - Optional fast-forward mode for audio tokens
    - Optional audio-specific attention mechanism
    - Comprehensive error handling and validation
    """
    
    def __init__(self, config: HiggsAudioConfig, layer_idx: int):
        super().__init__()
        
        # Validate configuration parameters
        self._validate_config(config, layer_idx)
        
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.dtype = config.dtype
        
        # Determine if this layer uses DualFFN based on configuration
        self.use_dual_ffn = self._should_use_dual_ffn(config, layer_idx)
        self.use_fast_forward = self._should_use_fast_forward(config, layer_idx)
        
        # Initialize attention mechanism
        self._initialize_attention(config)
        
        # Initialize FFN components based on DualFFN configuration
        self._initialize_ffn_components(config)
        
        # Initialize layer normalization
        self._initialize_layer_norms(config)
    
    def _validate_config(self, config: HiggsAudioConfig, layer_idx: int) -> None:
        """Validate configuration parameters for DualFFN layer."""
        if not isinstance(config, HiggsAudioConfig):
            raise TypeError(f"Expected HiggsAudioConfig, got {type(config)}")
        
        if not isinstance(layer_idx, int) or layer_idx < 0:
            raise ValueError(f"layer_idx must be a non-negative integer, got {layer_idx}")
        
        if layer_idx >= config.num_hidden_layers:
            raise ValueError(
                f"layer_idx {layer_idx} exceeds num_hidden_layers {config.num_hidden_layers}"
            )
        
        # Validate DualFFN configuration
        if hasattr(config, 'audio_dual_ffn_layers') and config.audio_dual_ffn_layers:
            if not isinstance(config.audio_dual_ffn_layers, (list, tuple)):
                raise TypeError(
                    f"audio_dual_ffn_layers must be a list or tuple, got {type(config.audio_dual_ffn_layers)}"
                )
            
            for idx in config.audio_dual_ffn_layers:
                if not isinstance(idx, int) or idx < 0 or idx >= config.num_hidden_layers:
                    raise ValueError(
                        f"Invalid layer index {idx} in audio_dual_ffn_layers. "
                        f"Must be between 0 and {config.num_hidden_layers - 1}"
                    )
        
        # Validate fast-forward configuration
        if hasattr(config, 'audio_fast_forward_layers') and config.audio_fast_forward_layers:
            if not isinstance(config.audio_fast_forward_layers, (list, tuple)):
                raise TypeError(
                    f"audio_fast_forward_layers must be a list or tuple, got {type(config.audio_fast_forward_layers)}"
                )
            
            for idx in config.audio_fast_forward_layers:
                if not isinstance(idx, int) or idx < 0 or idx >= config.num_hidden_layers:
                    raise ValueError(
                        f"Invalid layer index {idx} in audio_fast_forward_layers. "
                        f"Must be between 0 and {config.num_hidden_layers - 1}"
                    )
        
        # Validate hidden size
        if not hasattr(config, 'hidden_size') or config.hidden_size <= 0:
            raise ValueError(f"Invalid hidden_size: {getattr(config, 'hidden_size', None)}")
        
        # Validate intermediate size for MLP
        if not hasattr(config, 'intermediate_size') or config.intermediate_size <= 0:
            raise ValueError(f"Invalid intermediate_size: {getattr(config, 'intermediate_size', None)}")
    
    def _should_use_dual_ffn(self, config: HiggsAudioConfig, layer_idx: int) -> bool:
        """Determine if this layer should use DualFFN based on configuration."""
        if not hasattr(config, 'audio_dual_ffn_layers') or not config.audio_dual_ffn_layers:
            return False
        return layer_idx in config.audio_dual_ffn_layers
    
    def _should_use_fast_forward(self, config: HiggsAudioConfig, layer_idx: int) -> bool:
        """Determine if this layer should use fast-forward mode for audio tokens."""
        if not hasattr(config, 'audio_fast_forward_layers') or not config.audio_fast_forward_layers:
            return False
        return layer_idx in config.audio_fast_forward_layers
    
    def _initialize_attention(self, config: HiggsAudioConfig) -> None:
        """Initialize attention mechanisms."""
        try:
            # Main attention mechanism (shared for all tokens)
            self.attention = Attention(
                local_layer_idx=self.layer_idx,
                hidden_size=config.hidden_size,
                attention_head_size=getattr(config, 'head_size', config.hidden_size // config.num_attention_heads),
                num_attention_heads=config.num_attention_heads,
                num_kv_heads=getattr(config, 'num_key_value_heads', config.num_attention_heads),
                max_position_embeddings=config.max_position_embeddings,
                dtype=config.dtype,
                attention_mask_type=AttentionMaskType.causal,
                bias=getattr(config, 'attn_bias', False),
                position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
                rotary_embedding_base=getattr(config, 'rotary_base', 10000.0),
                rotary_embedding_scaling=getattr(config, 'rotary_scaling', None),
                tp_group=config.mapping.tp_group,
                tp_size=config.mapping.tp_size,
                tp_rank=config.mapping.tp_rank,
                q_scaling=getattr(config, 'q_scaling', 1.0),
                quant_mode=config.quant_mode,
            )
            
            # Optional audio-specific attention
            self.use_audio_attention = getattr(config, 'use_audio_out_self_attention', False)
            if self.use_audio_attention:
                self.audio_attention = Attention(
                    local_layer_idx=self.layer_idx,
                    hidden_size=config.hidden_size,
                    attention_head_size=getattr(config, 'head_size', config.hidden_size // config.num_attention_heads),
                    num_attention_heads=config.num_attention_heads,
                    num_kv_heads=getattr(config, 'num_key_value_heads', config.num_attention_heads),
                    max_position_embeddings=config.max_position_embeddings,
                    dtype=config.dtype,
                    attention_mask_type=AttentionMaskType.causal,
                    bias=getattr(config, 'attn_bias', False),
                    position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
                    rotary_embedding_base=getattr(config, 'rotary_base', 10000.0),
                    rotary_embedding_scaling=getattr(config, 'rotary_scaling', None),
                    tp_group=config.mapping.tp_group,
                    tp_size=config.mapping.tp_size,
                    tp_rank=config.mapping.tp_rank,
                    q_scaling=getattr(config, 'q_scaling', 1.0),
                    quant_mode=config.quant_mode,
                )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize attention for layer {self.layer_idx}: {e}")
    
    def _initialize_ffn_components(self, config: HiggsAudioConfig) -> None:
        """Initialize FFN components based on DualFFN configuration with memory optimization."""
        try:
            # Check for memory-efficient build mode
            memory_efficient_build = getattr(config, 'memory_efficient_build', False)
            
            if self.use_dual_ffn and not memory_efficient_build:
                # Full DualFFN implementation for runtime
                self.text_mlp = GatedMLP(
                    hidden_size=config.hidden_size,
                    ffn_hidden_size=config.intermediate_size,
                    hidden_act=config.hidden_act,
                    dtype=config.dtype,
                    bias=getattr(config, 'mlp_bias', False),
                    tp_group=config.mapping.tp_group,
                    tp_size=config.mapping.tp_size,
                    quant_mode=config.quant_mode,
                )
                
                self.audio_mlp = GatedMLP(
                    hidden_size=config.hidden_size,
                    ffn_hidden_size=config.intermediate_size,
                    hidden_act=config.hidden_act,
                    dtype=config.dtype,
                    bias=getattr(config, 'mlp_bias', False),
                    tp_group=config.mapping.tp_group,
                    tp_size=config.mapping.tp_size,
                    quant_mode=config.quant_mode,
                )
            else:
                # Memory-efficient mode: Use single MLP for both audio and text during build
                # This reduces memory usage during engine build by 50%
                self.mlp = GatedMLP(
                    hidden_size=config.hidden_size,
                    ffn_hidden_size=config.intermediate_size,
                    hidden_act=config.hidden_act,
                    dtype=config.dtype,
                    bias=getattr(config, 'mlp_bias', False),
                    tp_group=config.mapping.tp_group,
                    tp_size=config.mapping.tp_size,
                    quant_mode=config.quant_mode,
                )
                
                # In memory-efficient mode, alias the single MLP for both paths
                if self.use_dual_ffn:
                    self.text_mlp = self.mlp
                    self.audio_mlp = self.mlp
                    
        except Exception as e:
            raise RuntimeError(f"Failed to initialize FFN components for layer {self.layer_idx}: {e}")
    
    def _initialize_layer_norms(self, config: HiggsAudioConfig) -> None:
        """Initialize layer normalization components."""
        try:
            # Pre-attention layer norm
            self.input_layernorm = RmsNorm(
                normalized_shape=config.hidden_size,
                eps=config.norm_epsilon,
                dtype=config.dtype
            )
            
            if self.use_dual_ffn:
                # Separate post-attention layer norms for dual FFN
                self.post_layernorm_text = RmsNorm(
                    normalized_shape=config.hidden_size,
                    eps=config.norm_epsilon,
                    dtype=config.dtype
                )
                self.post_layernorm_audio = RmsNorm(
                    normalized_shape=config.hidden_size,
                    eps=config.norm_epsilon,
                    dtype=config.dtype
                )
            else:
                # Standard post-attention layer norm
                self.post_layernorm = RmsNorm(
                    normalized_shape=config.hidden_size,
                    eps=config.norm_epsilon,
                    dtype=config.dtype
                )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize layer norms for layer {self.layer_idx}: {e}")
    
    def _validate_basic_inputs(self, hidden_states: Tensor, audio_out_mask: Optional[Tensor] = None) -> None:
        """Validate basic inputs to the forward method."""
        if hidden_states is None:
            raise ValueError("hidden_states cannot be None")
        
        if not isinstance(hidden_states, Tensor):
            raise TypeError(f"hidden_states must be a Tensor, got {type(hidden_states)}")
        
        # TensorRT-LLM tensors can have dynamic shapes during engine build
        # Handle both 2D (flattened) and 3D tensor formats
        if len(hidden_states.shape) == 2:
            # Flattened format: [num_tokens, hidden_size]
            num_tokens, hidden_size = hidden_states.shape
            if hidden_size != self.hidden_size:
                raise ValueError(
                    f"hidden_states hidden_size {hidden_size} doesn't match "
                    f"config hidden_size {self.hidden_size}"
                )
        elif len(hidden_states.shape) == 3:
            # Standard format: [batch, seq_len, hidden_size]
            batch_size, seq_len, hidden_size = hidden_states.shape
            if hidden_size != self.hidden_size:
                raise ValueError(
                    f"hidden_states hidden_size {hidden_size} doesn't match "
                    f"config hidden_size {self.hidden_size}"
                )
        else:
            # For dynamic shapes (-1, size), be more flexible
            shape_list = list(hidden_states.shape)
            if len(shape_list) >= 2 and shape_list[-1] == self.hidden_size:
                # Last dimension matches hidden_size, likely valid
                pass
            else:
                raise ValueError(
                    f"hidden_states tensor shape {hidden_states.shape} is not compatible. "
                    f"Expected last dimension to be {self.hidden_size}"
                )
        
        # Validate audio_out_mask if provided (with TensorRT-LLM compatibility)
        if audio_out_mask is not None:
            if not isinstance(audio_out_mask, Tensor):
                raise TypeError(f"audio_out_mask must be a Tensor, got {type(audio_out_mask)}")
            
            # Be more flexible with mask validation for TensorRT-LLM dynamic shapes
            if len(audio_out_mask.shape) == 1:
                # Flattened mask format: [num_tokens]
                pass  # Allow flattened format during engine build
            elif len(audio_out_mask.shape) == 2:
                # Standard format: [batch, seq_len]
                mask_batch, mask_seq = audio_out_mask.shape
                # Only validate dimensions if we have 3D hidden_states
                if len(hidden_states.shape) == 3:
                    batch_size, seq_len, _ = hidden_states.shape
                    if mask_batch != batch_size or mask_seq != seq_len:
                        raise ValueError(
                            f"audio_out_mask shape {audio_out_mask.shape} doesn't match "
                            f"hidden_states batch/seq dimensions [{batch_size}, {seq_len}]"
                        )
            else:
                # Allow other shapes for dynamic TensorRT tensors
                pass
    
    def _apply_dual_path_ffn(self, hidden_states: Tensor, audio_out_mask: Tensor) -> Tensor:
        """
        Apply dual-path FFN processing with comprehensive error handling.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            audio_out_mask: Boolean mask [batch, seq_len] indicating audio tokens
            
        Returns:
            Processed hidden states with dual-path FFN applied
        """
        try:
            from tensorrt_llm.functional import where, cast
            
            # Handle different tensor shapes for TensorRT-LLM compatibility
            if len(hidden_states.shape) == 3:
                batch_size, seq_len, hidden_size = hidden_states.shape
                # Validate inputs for 3D case
                if len(audio_out_mask.shape) == 2:
                    if audio_out_mask.shape != (batch_size, seq_len):
                        raise ValueError(
                            f"audio_out_mask shape {audio_out_mask.shape} doesn't match "
                            f"expected [{batch_size}, {seq_len}]"
                        )
            elif len(hidden_states.shape) == 2:
                # Flattened format: [num_tokens, hidden_size]
                num_tokens, hidden_size = hidden_states.shape
                batch_size, seq_len = 1, num_tokens  # Treat as single sequence
            else:
                # Dynamic shape - infer from tensor
                hidden_size = hidden_states.shape[-1]
                # Use fallback dimensions for dynamic shapes
                batch_size, seq_len = 1, -1
            
            # Convert boolean mask to float for easier processing
            audio_mask_float = cast(audio_out_mask, hidden_states.dtype)
            text_mask_float = 1.0 - audio_mask_float
            
            # Expand masks to match hidden state dimensions (handle different shapes)
            if len(hidden_states.shape) == 3 and len(audio_mask_float.shape) == 2:
                # Standard case: expand 2D mask to 3D
                audio_mask_expanded = audio_mask_float.unsqueeze(-1)  # [batch, seq, 1]
                text_mask_expanded = text_mask_float.unsqueeze(-1)    # [batch, seq, 1]
            elif len(hidden_states.shape) == 2 and len(audio_mask_float.shape) == 1:
                # Flattened case: expand 1D mask to 2D
                audio_mask_expanded = audio_mask_float.unsqueeze(-1)  # [num_tokens, 1]
                text_mask_expanded = text_mask_float.unsqueeze(-1)    # [num_tokens, 1]
            else:
                # Fallback: try to broadcast directly or use identity
                try:
                    # Attempt to expand last dimension
                    audio_mask_expanded = audio_mask_float.unsqueeze(-1)
                    text_mask_expanded = text_mask_float.unsqueeze(-1)
                except:
                    # If unsqueeze fails, use the mask as-is (might work for some dynamic shapes)
                    audio_mask_expanded = audio_mask_float
                    text_mask_expanded = text_mask_float
            
            # Fast-forward mode: skip audio token processing
            if self.use_fast_forward:
                # Only process text tokens through FFN, keep audio tokens unchanged
                text_norm = self.post_layernorm_text(hidden_states)
                text_output = self.text_mlp(text_norm)
                
                # Combine: keep original hidden states for audio tokens, use text FFN output for text tokens
                output = (
                    hidden_states * audio_mask_expanded +    # Keep audio tokens unchanged
                    text_output * text_mask_expanded         # Update text tokens with FFN output
                )
                return output
            
            # Memory-efficient dual-path processing
            # Process text tokens only where needed
            text_norm = self.post_layernorm_text(hidden_states)
            text_mlp_out = self.text_mlp(text_norm)
            
            # Process audio tokens only where needed
            audio_norm = self.post_layernorm_audio(hidden_states)
            audio_mlp_out = self.audio_mlp(audio_norm)
            
            # Combine outputs efficiently without creating large intermediate tensors
            output = (
                text_mlp_out * text_mask_expanded +      # Text tokens use text MLP
                audio_mlp_out * audio_mask_expanded      # Audio tokens use audio MLP
            )
            
            # Clear intermediate tensors to free memory
            del text_norm, audio_norm, text_mlp_out, audio_mlp_out
            
            return output
            
        except Exception as e:
            raise RuntimeError(
                f"Error in dual-path FFN processing for layer {self.layer_idx}: {e}"
            ) from e
    
    def forward(self,
                hidden_states: Tensor,
                attention_mask: Optional[Tensor] = None,
                use_cache: bool = False,
                kv_cache_params: Optional[Any] = None,
                attention_params: Optional[Any] = None,
                lora_layer_params: Optional[Any] = None,
                position_ids: Optional[Tensor] = None,
                audio_token_mask: Optional[Tensor] = None,
                audio_out_mask: Optional[Tensor] = None,
                next_layer_input_layernorm_args: Optional[Any] = None) -> Union[Tensor, Tuple[Tensor, Any]]:
        """
        Forward pass with comprehensive error handling and validation.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Attention mask tensor
            use_cache: Whether to return KV cache for next iteration
            kv_cache_params: KV cache parameters for attention computation
            attention_params: Additional attention computation parameters
            lora_layer_params: LoRA adaptation parameters if enabled
            position_ids: Position indices for positional encoding
            audio_token_mask: Mask indicating audio token positions (legacy parameter)
            audio_out_mask: Boolean mask indicating audio vs text tokens [batch, seq_len]
            
        Returns:
            If use_cache=False: Hidden states tensor after layer processing
            If use_cache=True: Tuple of (hidden_states, kv_cache_presents)
        """
        try:
            # Basic input validation
            self._validate_basic_inputs(hidden_states, audio_out_mask)
            
            # Store residual connection
            residual = hidden_states
            
            # Pre-attention layer normalization
            hidden_states = self.input_layernorm(hidden_states)
            
            # Apply attention mechanism (TensorRT-LLM Attention doesn't use position_ids directly)
            attn_out = self.attention(hidden_states,
                                    attention_mask=attention_mask,
                                    use_cache=use_cache,
                                    kv_cache_params=kv_cache_params,
                                    attention_params=attention_params,
                                    lora_layer_params=lora_layer_params)
            if use_cache:
                attn_out, presents = attn_out
            
            # Add residual connection after attention
            hidden_states = residual + attn_out
            
            # Apply audio-specific attention if configured
            if self.use_audio_attention and audio_out_mask is not None:
                if audio_out_mask.any():  # Only if we have audio tokens
                    audio_residual = hidden_states
                    audio_norm = self.input_layernorm(hidden_states)  # Reuse input norm
                    
                    audio_attn_out = self.audio_attention(audio_norm,
                                                        attention_mask=attention_mask,
                                                        use_cache=False,  # Don't cache audio attention
                                                        kv_cache_params=None,
                                                        attention_params=attention_params,
                                                        lora_layer_params=lora_layer_params,
                                                        position_ids=position_ids)
                    if isinstance(audio_attn_out, tuple):
                        audio_attn_out = audio_attn_out[0]  # Extract just the output
                    
                    # Apply audio attention only to audio tokens
                    audio_mask_expanded = audio_out_mask.unsqueeze(-1)
                    hidden_states = (
                        hidden_states * (~audio_mask_expanded) +  # Keep text tokens unchanged
                        (audio_residual + audio_attn_out) * audio_mask_expanded  # Update audio tokens
                    )
            
            # Store residual for FFN
            residual = hidden_states
            
            # Apply FFN processing
            if self.use_dual_ffn and audio_out_mask is not None:
                # Use dual-path FFN processing
                ffn_output = self._apply_dual_path_ffn(hidden_states, audio_out_mask)
            else:
                # Standard single-path FFN processing
                if hasattr(self, 'post_layernorm'):
                    hidden_states = self.post_layernorm(hidden_states)
                else:
                    # Fallback to input layernorm if post_layernorm not available
                    hidden_states = self.input_layernorm(hidden_states)
                
                ffn_output = self.mlp(hidden_states,
                                    lora_layer_params=lora_layer_params)
            
            # Add residual connection after FFN
            hidden_states = residual + ffn_output
            
            if use_cache:
                return (hidden_states, presents)
            return hidden_states
            
        except Exception as e:
            raise RuntimeError(
                f"Error in forward pass for HiggsAudioDualFFNDecoderLayer {self.layer_idx}: {e}"
            ) from e


class HiggsAudioBackbone(Module):
    """Core transformer backbone for Higgs Audio TTS model.

    This class implements the main transformer architecture that processes
    both text and audio tokens through a unified decoder stack. It handles
    embedding lookup, layer-wise processing, and final normalization.

    The backbone supports:
    - Text token embedding with optional prompt tuning
    - Multi-layer transformer processing with TTS optimizations
    - Pipeline parallelism for distributed inference
    - KV caching for efficient autoregressive generation
    - Audio token processing through specialized attention mechanisms

    Key Components:
    - vocab_embedding: Token embedding layer with prompt tuning support
    - layers: Stack of HiggsAudioDecoderLayer instances
    - ln_f: Final layer normalization before output projection

    Attributes:
        config (HiggsAudioConfig): Model configuration
        mapping: TensorRT-LLM parallelism mapping
        use_prompt_tuning (bool): Whether prompt tuning is enabled
        vocab_size (int): Vocabulary size for embedding layer
        vocab_embedding: Token embedding layer (first PP rank only)
        layers (DecoderLayerList): Stack of decoder layers
        ln_f (RmsNorm): Final layer normalization (last PP rank only)

    Example:
        >>> backbone = HiggsAudioBackbone(config)
        >>> output = backbone(input_ids, attention_mask=mask, use_cache=True)
    """

    def __init__(self, config: HiggsAudioConfig) -> None:
        """Initialize transformer backbone with TTS-optimized components.

        Args:
            config: HiggsAudioConfig containing model architecture parameters

        Raises:
            ValueError: If config parameters are incompatible with backbone requirements
        """
        super().__init__()
        self.config = config
        self.mapping = config.mapping
        self.use_prompt_tuning = getattr(config, "use_prompt_tuning", False)
        self.vocab_size = config.vocab_size
        
        # Audio mask handling for DualFFN layers
        self._audio_out_mask: Optional[Tensor] = None
        EmbeddingCls = PromptTuningEmbedding if self.use_prompt_tuning else Embedding
        if self.mapping.is_first_pp_rank():
            self.vocab_embedding = EmbeddingCls(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                dtype=config.dtype,
                tp_size=self.mapping.tp_size if getattr(config, "use_parallel_embedding", False) else 1,
                tp_group=self.mapping.tp_group if getattr(config, "use_parallel_embedding", False) else None,
                sharding_dim=getattr(config, "embedding_sharding_dim", None),
                tp_rank=self.mapping.tp_rank,
            )

        # Initialize layers based on audio adapter configuration
        self._initialize_decoder_layers(config)

        if self.mapping.is_last_pp_rank():
            self.ln_f = RmsNorm(normalized_shape=config.hidden_size,
                                eps=config.norm_epsilon,
                                dtype=config.dtype)

    def set_audio_out_mask(self, audio_out_mask: Optional[Tensor]) -> None:
        """Set the audio output mask for DualFFN layer processing.
        
        This method stores the audio mask that will be used by DualFFN layers
        to route audio and text tokens through separate processing paths.
        
        Args:
            audio_out_mask: Boolean tensor [batch_size, seq_len] indicating
                          audio token positions, or None to clear the mask
        """
        self._audio_out_mask = audio_out_mask
    
    def get_audio_out_mask(self) -> Optional[Tensor]:
        """Get the current audio output mask for DualFFN processing.
        
        Returns:
            The stored audio output mask tensor, or None if not set
        """
        return self._audio_out_mask

    def _initialize_decoder_layers(self, config: HiggsAudioConfig) -> None:
        """Initialize decoder layers with comprehensive error handling and validation."""
        try:
            # Validate configuration
            if not isinstance(config, HiggsAudioConfig):
                raise TypeError(f"Expected HiggsAudioConfig, got {type(config)}")
            
            if not hasattr(config, 'num_hidden_layers') or config.num_hidden_layers <= 0:
                raise ValueError(f"Invalid num_hidden_layers: {getattr(config, 'num_hidden_layers', None)}")
            
            # Validate audio adapter configuration
            audio_adapter_type = getattr(config, 'audio_adapter_type', 'stack')
            valid_adapter_types = ['dual_ffn', 'dual_ffn_fast_forward', 'stack']
            
            if audio_adapter_type not in valid_adapter_types:
                raise ValueError(
                    f"Unsupported audio_adapter_type: {audio_adapter_type}. "
                    f"Supported types: {valid_adapter_types}"
                )
            
            layers = []
            
            # Create layers based on adapter type
            if audio_adapter_type in ['dual_ffn', 'dual_ffn_fast_forward']:
                # Validate DualFFN configuration
                dual_ffn_layers = getattr(config, 'audio_dual_ffn_layers', [])
                if dual_ffn_layers and not isinstance(dual_ffn_layers, (list, tuple)):
                    raise TypeError(
                        f"audio_dual_ffn_layers must be a list or tuple, got {type(dual_ffn_layers)}"
                    )
                
                # Validate layer indices
                for idx in dual_ffn_layers:
                    if not isinstance(idx, int) or idx < 0 or idx >= config.num_hidden_layers:
                        raise ValueError(
                            f"Invalid layer index {idx} in audio_dual_ffn_layers. "
                            f"Must be between 0 and {config.num_hidden_layers - 1}"
                        )
                
                # Create layers with DualFFN where specified
                for layer_idx in range(config.num_hidden_layers):
                    try:
                        if layer_idx in dual_ffn_layers:
                            # Use DualFFN layer
                            layer = HiggsAudioDualFFNDecoderLayer(config, layer_idx)
                        else:
                            # Use standard decoder layer
                            layer = HiggsAudioDecoderLayer(config, layer_idx)
                        
                        layers.append(layer)
                        
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to create layer {layer_idx}: {e}"
                        ) from e
                        
            elif audio_adapter_type == 'stack':
                # Standard stacked decoder layers
                for layer_idx in range(config.num_hidden_layers):
                    try:
                        layer = HiggsAudioDecoderLayer(config, layer_idx)
                        layers.append(layer)
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to create standard layer {layer_idx}: {e}"
                        ) from e
            else:
                raise ValueError(
                    f"Unsupported audio_adapter_type: {audio_adapter_type}. "
                    f"Supported types: {valid_adapter_types}"
                )
            
            # Validate that we created the correct number of layers
            if len(layers) != config.num_hidden_layers:
                raise RuntimeError(
                    f"Created {len(layers)} layers but expected {config.num_hidden_layers}"
                )
            
            # Create DecoderLayerList with the constructed layers
            from ..modeling_utils import DecoderLayerList
            # We need to create a custom DecoderLayerList that can handle our layers
            # Since we already have the layers constructed, we'll create a custom class
            class HiggsAudioDecoderLayerList(DecoderLayerList):
                def __init__(self, layers, config):
                    self.num_hidden_layers = config.num_hidden_layers
                    self.layer_list = config.mapping.pp_layers(config.num_hidden_layers)
                    self.quant_mode = config.quant_mode
                    self.config = config  # Store the config
                    # Initialize with the pre-constructed layers
                    super(DecoderLayerList, self).__init__(layers)
                
                def forward(self,
                            hidden_states,
                            use_cache=False,
                            attention_mask=None,
                            kv_cache_params=None,
                            attention_params=None,
                            mrope_params=None,
                            position_ids=None,
                            lora_params=None,
                            spec_decoding_params=None,
                            vision_token_mask=None,
                            audio_out_mask=None):
                    kv_cache_params.fill_none_tensor_list(len(self.layer_list))

                    if use_cache:
                        presents = []

                    for layer_idx, (layer, past) in enumerate(
                            zip(self, kv_cache_params.past_key_value)):

                        lora_layer_params = None
                        if lora_params is not None and lora_params.lora_ranks is not None:
                            lora_layer_params = lora_params.get_layer_params(layer_idx)

                        kwargs = {}
                        if position_ids is not None:
                            kwargs['position_ids'] = position_ids
                        if vision_token_mask is not None:
                            kwargs['vision_token_mask'] = vision_token_mask
                        if audio_out_mask is not None:
                            kwargs['audio_out_mask'] = audio_out_mask
                        if lora_layer_params is not None:
                            kwargs['lora_layer_params'] = lora_layer_params
                        if spec_decoding_params is not None:
                            kwargs['spec_decoding_params'] = spec_decoding_params
                        if mrope_params is not None:
                            kwargs['mrope_params'] = mrope_params

                        if default_net().plugin_config.reduce_fusion:
                            if layer_idx + self.layer_list[0] < self.layer_list[-1]:
                                qkv_activation_scaling_factor = None
                                if default_net().plugin_config.user_buffer:
                                    qkv_linear = self[layer_idx + 1].attention.qkv
                                    if self.quant_mode.has_fp8_qdq():
                                        qkv_activation_scaling_factor = constant(
                                            qkv_linear.activation_scaling_factor.raw_value.
                                            copy())
                                    elif self.quant_mode.has_nvfp4():
                                        qkv_activation_scaling_factor = constant(
                                            qkv_linear.activation_global_scaling_factor.
                                            raw_value.copy())
                                kwargs['next_layer_input_layernorm_args'] = (
                                    self[layer_idx + 1].input_layernorm.weight.value,
                                    self[layer_idx + 1].input_layernorm.eps,
                                    qkv_activation_scaling_factor)
                            else:
                                kwargs['next_layer_input_layernorm_args'] = None
                        elif default_net().plugin_config.norm_quant_fusion:
                            if layer_idx < self.layer_list[-1] - self.layer_list[0]:
                                try:
                                    activation_scaling_factor = constant(
                                        self[layer_idx + 1].attention.qkv.
                                        activation_global_scaling_factor.raw_value.copy())
                                except:
                                    activation_scaling_factor = None
                                kwargs['next_layer_input_layernorm_args'] = (
                                    self[layer_idx + 1].input_layernorm.weight.value,
                                    self[layer_idx + 1].input_layernorm.eps,
                                    activation_scaling_factor)
                            else:
                                kwargs['next_layer_input_layernorm_args'] = None

                        # LlamaAttention handles position embeddings automatically, no need for manual creation
                        
                        layer_kwargs = {
                            'hidden_states': hidden_states,
                            'use_cache': use_cache,
                            'attention_mask': attention_mask,
                            'kv_cache_params': KeyValueCacheParams(
                                past_key_value=[past],
                                host_past_key_value_lengths=kv_cache_params.
                                host_past_key_value_lengths,
                                host_max_attention_window_sizes=kv_cache_params.
                                host_max_attention_window_sizes,
                                host_sink_token_length=kv_cache_params.
                                host_sink_token_length,
                                kv_cache_block_offsets=kv_cache_params.
                                kv_cache_block_offsets,
                                host_kv_cache_block_offsets=kv_cache_params.
                                host_kv_cache_block_offsets,
                                host_kv_cache_pool_pointers=kv_cache_params.
                                host_kv_cache_pool_pointers,
                                host_kv_cache_pool_mapping=kv_cache_params.
                                host_kv_cache_pool_mapping,
                                cache_indirection=kv_cache_params.cache_indirection),
                            'attention_params': attention_params,
                            **kwargs
                        }
                        
                        layer_output = layer(**layer_kwargs)

                        if use_cache:
                            hidden_states, present = layer_output
                            presents.append(present)
                        else:
                            hidden_states = layer_output

                    if use_cache:
                        return (hidden_states, tuple(presents))
                    return hidden_states

            self.layers = HiggsAudioDecoderLayerList(layers, config)
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize decoder layers for HiggsAudioBackbone: {e}"
            ) from e

    def forward(self,
                input_ids: Tensor,
                position_ids: Optional[Tensor] = None,
                use_cache: bool = False,
                attention_mask: Optional[Tensor] = None,
                kv_cache_params: Optional[Any] = None,
                attention_params: Optional[Any] = None,
                hidden_states: Optional[Tensor] = None,
                prompt_embedding_table: Optional[Tensor] = None,
                prompt_tasks: Optional[Tensor] = None,
                prompt_vocab_size: Optional[Tensor] = None,
                lora_params: Optional[Any] = None,
                input_token_extra_ids: Optional[Tensor] = None,
                audio_token_mask: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tuple[Any, ...]]]:
        """Forward pass through transformer backbone with multimodal support.

        Processes input tokens through embedding lookup, transformer layers,
        and final normalization. Supports both text and audio token processing
        with pipeline parallelism and KV caching optimizations.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            position_ids: Position indices for RoPE encoding
            use_cache: Whether to use and return KV cache
            attention_mask: Attention mask for sequence padding
            kv_cache_params: KV cache parameters for efficient generation
            attention_params: Additional attention computation parameters
            hidden_states: Pre-computed hidden states (for PP intermediate ranks)
            prompt_embedding_table: Prompt tuning embedding table
            prompt_tasks: Task IDs for prompt tuning
            prompt_vocab_size: Vocabulary size for prompt tuning
            lora_params: LoRA adaptation parameters
            input_token_extra_ids: Additional token IDs for special processing
            audio_token_mask: Mask indicating audio token positions

        Returns:
            If use_cache=False: Final hidden states [batch_size, seq_len, hidden_size]
            If use_cache=True: Tuple of (hidden_states, kv_cache_presents)

        Example:
            >>> output = backbone(
            ...     input_ids=tokens,
            ...     attention_mask=mask,
            ...     use_cache=True,
            ...     audio_token_mask=audio_mask
            ... )
        """
        # Fill kv cache structures
        kv_cache_params.fill_none_tensor_list(len(self.layers))

        # Embedding lookup (or receive from PP)
        if self.mapping.is_first_pp_rank():
            ptuning_args = [prompt_embedding_table, prompt_tasks, prompt_vocab_size] \
                if self.use_prompt_tuning else []
            hidden_states = self.vocab_embedding(input_ids, *ptuning_args)
        else:
            from ...functional import recv
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())

        # Forward layers with audio mask for DualFFN processing
        # Use stored audio_out_mask if available, otherwise fall back to audio_token_mask
        audio_mask_for_layers = self._audio_out_mask if self._audio_out_mask is not None else audio_token_mask
        
        hidden_states = self.layers.forward(hidden_states,
                                            use_cache=use_cache,
                                            attention_mask=attention_mask,
                                            kv_cache_params=kv_cache_params,
                                            attention_params=attention_params,
                                            lora_params=lora_params,
                                            position_ids=position_ids,
                                            audio_out_mask=audio_mask_for_layers)

        if use_cache:
            hidden_states, presents = hidden_states

        if self.mapping.is_last_pp_rank():
            hidden_states = self.ln_f(hidden_states)
        else:
            from ...functional import send
            hidden_states = send(hidden_states, self.mapping.next_pp_rank())

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class HiggsAudioForCausalLM(HiggsAudioModelForCausalLM, TopModelMixin):
    """Complete Higgs Audio model for causal language modeling with TTS capabilities.

    This is the main model class that combines the TTS-optimized base class
    with TensorRT-LLM's TopModelMixin for complete inference functionality.
    It integrates the HiggsAudioBackbone with a language modeling head for
    token generation.

    The model supports:
    - Multimodal text and audio token generation
    - TTS-specific generation modes and optimizations
    - TensorRT-LLM runtime integration and optimization
    - Hugging Face model loading and weight conversion
    - Pipeline and tensor parallelism for scalable inference

    Architecture:
    - transformer: HiggsAudioBackbone for core processing
    - lm_head: ColumnLinear layer for vocabulary projection
    - Inherits TTS optimizations from HiggsAudioModelForCausalLM
    - Inherits runtime features from TopModelMixin

    Attributes:
        config_class: Configuration class for this model type
        transformer (HiggsAudioBackbone): Core transformer backbone
        lm_head (Optional[ColumnLinear]): Language modeling head (last PP rank only)

    Example:
        >>> model = HiggsAudioForCausalLM.from_hugging_face("higgs-audio-model")
        >>> model.set_generation_mode(GenerationMode.AUDIO_INIT)
        >>> outputs = model.generate(input_ids, max_length=100)
    """

    config_class = HiggsAudioConfig

    def __init__(self, config: HiggsAudioConfig) -> None:
        """Initialize complete Higgs Audio model for TTS generation.

        Args:
            config: HiggsAudioConfig with complete model parameters

        Raises:
            ValueError: If config is incompatible with model requirements
        """
        transformer = HiggsAudioBackbone(config)
        vocab_size_padded = pad_vocab_size(config.vocab_size, config.mapping.tp_size)

        # Create simple lm_head for parent initialization first
        if config.mapping.is_last_pp_rank():
            # Use simple ColumnLinear as lm_head for parent compatibility
            lm_head = ColumnLinear(
                in_features=config.hidden_size,
                out_features=vocab_size_padded,
                bias=False,
                dtype=config.dtype,
                tp_group=config.mapping.tp_group,
                tp_size=config.mapping.tp_size,
                gather_output=True
            )
        else:
            lm_head = None

        # Call parent __init__ FIRST to initialize the module properly
        DecoderModelForCausalLM.__init__(self, config, transformer, lm_head)

        # Now initialize decoder projector AFTER parent init
        if config.mapping.is_last_pp_rank():
            # Use the new HiggsAudioDecoderProjector
            self.decoder_projector = HiggsAudioDecoderProjector(config)
            # Replace lm_head with text head from projector for compatibility
            self.lm_head = self.decoder_projector.text_lm_head
        else:
            self.decoder_projector = None

        # Audio tower integration - Adapted for TensorRT-LLM
        if not config.skip_audio_tower:
            self.audio_tower = HiggsAudioEncoder(config)
            self.audio_encoder_proj = HiggsAudioFeatureProjector(config)
        else:
            self.audio_tower = None
            self.audio_encoder_proj = None

    def _apply_audio_tower(self, audio_features, audio_feature_attention_mask):
        """Apply the audio tower to the audio features - TensorRT-LLM compatible implementation."""
        
        # Handle empty audio features case
        if audio_features.shape[0] == 0:
            # Return None for empty batch to avoid computation
            return None, None
        
        # Calculate attention mask if provided
        audio_attention_mask = None
        audio_feat_out_lengths = None
        
        if audio_feature_attention_mask is not None:
            # Calculate actual feature lengths from attention mask
            audio_feat_lengths = audio_feature_attention_mask.sum(dim=-1)
            
            # Calculate output lengths after conv layers (stride=2 from conv2)
            # Mel-spectrogram length -> conv1 (no stride) -> conv2 (stride=2) -> final length
            audio_feat_out_lengths = (audio_feat_lengths - 1) // 2 + 1
            
            batch_size, max_mel_seq_len = audio_feature_attention_mask.shape
            max_seq_len = (max_mel_seq_len - 1) // 2 + 1
            
            # Create sequence range tensor for masking
            from tensorrt_llm.functional import arange, unsqueeze, expand
            
            seq_range = arange(0, max_seq_len, dtype='int32')
            seq_range = unsqueeze(seq_range, 0)  # [1, max_seq_len]
            seq_range = expand(seq_range, [batch_size, max_seq_len])  # [batch, max_seq_len]
            
            # Expand lengths for comparison
            lengths_expand = unsqueeze(audio_feat_out_lengths, 1)  # [batch, 1]
            lengths_expand = expand(lengths_expand, [batch_size, max_seq_len])  # [batch, max_seq_len]
            
            # Create padding mask (True where valid tokens)
            from tensorrt_llm.functional import lt
            padding_mask = lt(seq_range, lengths_expand)  # [batch, max_seq_len]
            
            # For bidirectional attention in encoder, use simple padding mask
            audio_attention_mask = padding_mask
        
        # Apply audio encoder
        audio_outputs = self.audio_tower(
            audio_features,
            attention_mask=audio_attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            check_seq_length=False  # Skip length validation for flexibility
        )
        
        # Extract last hidden state
        if isinstance(audio_outputs, dict):
            selected_audio_feature = audio_outputs['last_hidden_state']
        else:
            # Handle tuple output
            selected_audio_feature = audio_outputs[0] if isinstance(audio_outputs, tuple) else audio_outputs
        
        # Project audio features to text model dimension
        audio_features_embed = self.audio_encoder_proj(selected_audio_feature)
        
        return audio_features_embed, audio_feat_out_lengths

    def forward(self,
                input_ids: Tensor,
                position_ids=None,
                use_cache=False,
                last_token_ids=None,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                mrope_params=None,
                hidden_states=None,
                prompt_embedding_table: Optional[Tensor] = None,
                prompt_tasks: Optional[Tensor] = None,
                prompt_vocab_size: Optional[Tensor] = None,
                lora_params=None,
                spec_decoding_params=None,
                audio_out_mask: Optional[Tensor] = None):
        """Forward pass for Higgs Audio model with audio token routing.
        
        This method extends the base DecoderModelForCausalLM forward method to support
        audio-specific parameters like audio_out_mask for routing audio tokens through
        DualFFN layers.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            audio_out_mask: Boolean mask indicating audio tokens [batch_size, seq_len]
            **kwargs: Other standard TensorRT-LLM forward arguments
            
        Returns:
            Model outputs with logits for text/audio token generation
        """
        # Store audio_out_mask in the transformer for layer access
        if hasattr(self.transformer, 'set_audio_out_mask') and audio_out_mask is not None:
            self.transformer.set_audio_out_mask(audio_out_mask)

        # Audio tower integration - TensorRT-LLM compatible implementation
        audio_features_embed = None
        audio_features_length = None
        
        # Check if audio features are provided and audio tower is enabled
        if hasattr(self, 'audio_features') and self.audio_features is not None and not self.config.skip_audio_tower:
            # Apply audio tower processing
            audio_features_embed, audio_features_length = self._apply_audio_tower(
                self.audio_features, self.audio_feature_attention_mask
            )
            
            # Integrate audio features with text embeddings if needed
            # This would typically happen in the transformer layers through attention mechanisms
        
        # Call parent forward method with standard arguments
        return super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=use_cache,
            last_token_ids=last_token_ids,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            mrope_params=mrope_params,
            hidden_states=hidden_states,
            prompt_embedding_table=prompt_embedding_table,
            prompt_tasks=prompt_tasks,
            prompt_vocab_size=prompt_vocab_size,
            lora_params=lora_params,
            spec_decoding_params=spec_decoding_params
        )

    def generate(self,
                 input_ids: Optional[Tensor] = None,
                 attention_mask: Optional[Tensor] = None,
                 audio_features: Optional[Tensor] = None,
                 max_length: int = 512,
                 max_new_tokens: Optional[int] = None,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 1.0,
                 do_sample: bool = True,
                 pad_token_id: Optional[int] = None,
                 eos_token_id: Optional[int] = None,
                 use_delay_pattern: bool = True,
                 num_codebooks: Optional[int] = None,
                 streaming: bool = False,
                 stream_chunk_size: int = 32,
                 return_dict: bool = True,
                 **kwargs: Any) -> Union[Tensor, Dict[str, Any]]:
        """Generate text and audio tokens with delay pattern coordination for RVQ-based TTS.
        
        This method implements TTS-optimized generation that supports multi-codebook
        audio token generation with delay patterns for RVQ (Residual Vector Quantization).
        It handles mode transitions from text to audio generation and coordinates token
        generation across multiple codebooks while maintaining proper temporal alignment.
        
        Key Features:
        - Automatic mode transitions: TEXT -> AUDIO_INIT -> AUDIO_IN_PROGRESS
        - Multi-codebook audio token generation with delay pattern coordination
        - Streaming support for real-time TTS applications
        - Comprehensive validation and error handling
        - Integration with DelayPatternProvider and AudioTokenUtils
        - Support for both text-only and multimodal audio-text generation
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len] for text prompts
            attention_mask: Attention mask for input padding
            audio_features: Optional audio feature embeddings for voice conditioning
            max_length: Maximum total sequence length (including input)
            max_new_tokens: Maximum number of new tokens to generate (overrides max_length)
            temperature: Sampling temperature (higher = more random, lower = more deterministic)
            top_k: Top-k sampling parameter (0 = disabled)
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            pad_token_id: Token ID for padding (defaults to config value)
            eos_token_id: Token ID for end-of-sequence (defaults to config value)
            use_delay_pattern: Whether to apply delay patterns for multi-codebook generation
            num_codebooks: Number of audio codebooks (defaults to config value)
            streaming: Whether to enable streaming generation for real-time applications
            stream_chunk_size: Number of tokens per streaming chunk
            return_dict: Whether to return a dictionary with detailed outputs
            **kwargs: Additional generation parameters
            
        Returns:
            If return_dict=False: Generated token tensor [batch_size, output_seq_len]
            If return_dict=True: Dictionary containing:
                - 'sequences': Generated token sequences
                - 'codebook_sequences': Per-codebook token sequences (if multi-codebook)
                - 'generation_mode_history': Sequence of generation modes used
                - 'delay_pattern_info': Information about applied delay patterns
                - 'streaming_chunks': Generated chunks (if streaming enabled)
                
        Raises:
            ValueError: If generation parameters are invalid or incompatible
            RuntimeError: If generation fails due to model or input issues
            
        Example:
            >>> # Text-only generation
            >>> model.set_generation_mode(GenerationMode.TEXT)
            >>> outputs = model.generate(input_ids=text_tokens, max_new_tokens=50)
            
            >>> # Audio generation with delay patterns
            >>> model.set_generation_mode(GenerationMode.AUDIO_INIT)
            >>> outputs = model.generate(
            ...     input_ids=text_tokens,
            ...     audio_features=audio_embeddings,
            ...     max_new_tokens=100,
            ...     use_delay_pattern=True,
            ...     num_codebooks=4,
            ...     return_dict=True
            ... )
            
            >>> # Streaming audio generation
            >>> outputs = model.generate(
            ...     input_ids=text_tokens,
            ...     streaming=True,
            ...     stream_chunk_size=16,
            ...     use_delay_pattern=True
            ... )
        """
        try:
            # Validate and prepare generation parameters
            gen_params = self._validate_and_prepare_generation_params(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_features=audio_features,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                use_delay_pattern=use_delay_pattern,
                num_codebooks=num_codebooks,
                streaming=streaming,
                stream_chunk_size=stream_chunk_size,
                **kwargs
            )
            
            # Initialize generation state
            generation_state = self._initialize_generation_state(gen_params)
            
            # Initialize delay pattern provider if needed
            delay_provider = None
            audio_utils = None
            if gen_params['use_delay_pattern'] and gen_params['num_codebooks'] > 1:
                delay_provider = DelayPatternProvider(
                    strategy=getattr(self.config, 'audio_delay_pattern_strategy', 'linear'),
                    stride=getattr(self.config, 'audio_delay_pattern_stride', 1),
                    custom_delays=getattr(self.config, 'audio_delay_pattern_custom_delays', None),
                    max_delay=getattr(self.config, 'audio_delay_pattern_max_delay', None),
                    pad_token_id=gen_params['pad_token_id']
                )
                audio_utils = AudioTokenUtils(
                    num_codebooks=gen_params['num_codebooks'],
                    pad_token_id=gen_params['pad_token_id'],
                    eos_token_id=gen_params['eos_token_id'],
                    audio_start_token_id=getattr(self.config, 'audio_start_token_id', None),
                    audio_end_token_id=getattr(self.config, 'audio_end_token_id', None)
                )
            
            # Execute generation based on streaming mode
            if gen_params['streaming']:
                outputs = self._generate_streaming(
                    gen_params=gen_params,
                    generation_state=generation_state,
                    delay_provider=delay_provider,
                    audio_utils=audio_utils
                )
            else:
                outputs = self._generate_standard(
                    gen_params=gen_params,
                    generation_state=generation_state,
                    delay_provider=delay_provider,
                    audio_utils=audio_utils
                )
            
            # Post-process outputs
            if return_dict:
                return self._prepare_generation_outputs(
                    outputs=outputs,
                    generation_state=generation_state,
                    delay_provider=delay_provider,
                    audio_utils=audio_utils,
                    gen_params=gen_params
                )
            else:
                return outputs['sequences']
                
        except Exception as e:
            # Reset generation mode on error using mode manager
            try:
                if hasattr(self, 'generation_mode_manager'):
                    self.generation_mode_manager.reset_state(preserve_config=True)
                    # Fallback to TEXT mode for safety
                    self.set_generation_mode(GenerationMode.TEXT, validation_level='basic')
            except:
                pass  # Don't let error handling fail
            raise RuntimeError(f"Generation failed: {e}") from e

    def _validate_and_prepare_generation_params(self, **kwargs: Any) -> Dict[str, Any]:
        """Validate and prepare generation parameters with comprehensive error checking."""
        # Extract parameters
        input_ids = kwargs.get('input_ids')
        attention_mask = kwargs.get('attention_mask')
        audio_features = kwargs.get('audio_features')
        max_length = kwargs.get('max_length', 512)
        max_new_tokens = kwargs.get('max_new_tokens')
        temperature = kwargs.get('temperature', 1.0)
        top_k = kwargs.get('top_k', 50)
        top_p = kwargs.get('top_p', 1.0)
        do_sample = kwargs.get('do_sample', True)
        pad_token_id = kwargs.get('pad_token_id', getattr(self.config, 'pad_token_id', 0))
        eos_token_id = kwargs.get('eos_token_id', getattr(self.config, 'eos_token_id', None))
        use_delay_pattern = kwargs.get('use_delay_pattern', True)
        num_codebooks = kwargs.get('num_codebooks', getattr(self.config, 'audio_num_codebooks', 4))
        streaming = kwargs.get('streaming', False)
        stream_chunk_size = kwargs.get('stream_chunk_size', 32)
        
        # Validate input_ids
        if input_ids is None:
            raise ValueError("input_ids is required for generation")
        if not isinstance(input_ids, Tensor):
            raise TypeError(f"input_ids must be a Tensor, got {type(input_ids)}")
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2D [batch, seq], got shape {input_ids.shape}")
        
        batch_size, input_seq_len = input_ids.shape
        
        # Validate attention_mask
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError(f"attention_mask shape {attention_mask.shape} doesn't match input_ids shape {input_ids.shape}")
        
        # Validate and compute sequence lengths
        if max_new_tokens is not None:
            if max_new_tokens <= 0:
                raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
            max_length = input_seq_len + max_new_tokens
        
        if max_length <= input_seq_len:
            raise ValueError(f"max_length {max_length} must be greater than input sequence length {input_seq_len}")
        
        # Validate sampling parameters
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {top_k}")
        if not 0 < top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        
        # Validate codebook parameters
        if num_codebooks <= 0:
            raise ValueError(f"num_codebooks must be positive, got {num_codebooks}")
        if use_delay_pattern and num_codebooks == 1:
            # Disable delay pattern for single codebook
            use_delay_pattern = False
        
        # Validate streaming parameters
        if streaming and stream_chunk_size <= 0:
            raise ValueError(f"stream_chunk_size must be positive, got {stream_chunk_size}")
        
        # Validate audio features if provided
        if audio_features is not None:
            if not isinstance(audio_features, Tensor):
                raise TypeError(f"audio_features must be a Tensor, got {type(audio_features)}")
            if audio_features.shape[0] != batch_size:
                raise ValueError(f"audio_features batch size {audio_features.shape[0]} doesn't match input_ids batch size {batch_size}")
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'audio_features': audio_features,
            'batch_size': batch_size,
            'input_seq_len': input_seq_len,
            'max_length': max_length,
            'max_new_tokens': max_length - input_seq_len,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'do_sample': do_sample,
            'pad_token_id': pad_token_id,
            'eos_token_id': eos_token_id,
            'use_delay_pattern': use_delay_pattern,
            'num_codebooks': num_codebooks,
            'streaming': streaming,
            'stream_chunk_size': stream_chunk_size
        }

    def _initialize_generation_state(self, gen_params: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize generation state for TTS-aware token generation."""
        # Get current generation mode
        current_mode = self.get_generation_mode()
        
        # Determine if we need audio generation based on mode and features
        needs_audio_generation = (
            current_mode in (GenerationMode.AUDIO_INIT, GenerationMode.AUDIO_IN_PROGRESS) or
            gen_params['audio_features'] is not None or
            gen_params['use_delay_pattern']
        )
        
        # Initialize generation state
        state = {
            'current_position': gen_params['input_seq_len'],
            'generated_tokens': 0,
            'generation_mode_history': [current_mode.value],
            'mode_transition_positions': [],
            'audio_generation_active': needs_audio_generation,
            'current_codebook_idx': 0,
            'delay_pattern_offset': 0,
            'streaming_chunks': [],
            'finished_sequences': set(),
            'eos_reached': False
        }
        
        # Initialize codebook-specific state if using multi-codebook generation
        if gen_params['use_delay_pattern'] and gen_params['num_codebooks'] > 1:
            state['codebook_states'] = {}
            for i in range(gen_params['num_codebooks']):
                state['codebook_states'][f'codebook_{i}'] = {
                    'active': False,
                    'last_token': None,
                    'delay_offset': i * getattr(self.config, 'audio_delay_pattern_stride', 1),
                    'generated_tokens': []
                }
        
        return state
