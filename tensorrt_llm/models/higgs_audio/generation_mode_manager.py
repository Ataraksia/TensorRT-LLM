
# SPDX-License-Identifier: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive Generation Mode Management for Higgs Audio TTS Model.

This module provides centralized management for different TTS generation phases,
state coordination, and mode transitions with validation and recovery mechanisms.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Union, Dict, Any, List, Tuple
import math

from tensorrt_llm.functional import Tensor


class GenerationModeError(Exception):
    """Exception class for generation mode management errors.
    
    This exception is raised when mode transitions fail, state validation
    errors occur, or mode-specific operations encounter issues.
    
    Error Codes:
    - INVALID_MODE_TRANSITION: Attempted transition between incompatible modes
    - STATE_CORRUPTION: Generation state is corrupted or inconsistent
    - VALIDATION_FAILED: Mode transition validation checks failed
    - CONTEXT_PRESERVATION_FAILED: Failed to preserve context during transition
    - RECOVERY_FAILED: Automatic recovery mechanisms failed
    - CHECKPOINT_FAILED: State checkpointing/restoration failed
    - PERFORMANCE_DEGRADED: Mode performance below acceptable thresholds
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


class GenerationMode(Enum):
    """TTS-specific generation modes for coordinated audio-text generation."""
    
    TEXT = "text"
    """Text-only generation mode for standard language modeling tasks."""
    
    AUDIO_INIT = "audio_init"
    """Initial audio generation mode for setting up audio token generation."""
    
    AUDIO_IN_PROGRESS = "audio_in_progress"
    """Ongoing audio generation mode for streaming audio token generation."""


@dataclass
class GenerationState:
    """
    Comprehensive state container for TTS generation across mode transitions.
    
    This class maintains all necessary state for coordinated multi-mode TTS generation,
    including context preservation, performance tracking, and recovery mechanisms.
    
    Features:
    - Cross-mode state preservation with serialization support
    - Validation and consistency checking
    - Performance monitoring and optimization tracking
    - Automatic recovery mechanisms with state checkpointing
    - Streaming support with chunk coordination
    - Multi-codebook audio generation state
    - Text and audio conditioning state management
    """
    
    # Core generation state
    current_mode: GenerationMode = GenerationMode.TEXT
    previous_mode: Optional[GenerationMode] = None
    mode_transition_count: int = 0
    mode_transition_history: List[Tuple[str, float]] = field(default_factory=list)
    
    # Position and sequence tracking
    current_position: int = 0
    input_sequence_length: int = 0
    generated_tokens: int = 0
    max_sequence_length: int = 2048
    
    # Audio generation state
    audio_generation_active: bool = False
    current_codebook_index: int = 0
    delay_pattern_offset: int = 0
    audio_token_count: int = 0
    
    # Multi-codebook coordination
    codebook_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    delay_pattern_active: bool = False
    
    # Context preservation
    text_context: Optional[Dict[str, Any]] = None
    audio_context: Optional[Dict[str, Any]] = None
    conditioning_state: Optional[Dict[str, Any]] = None
    
    # KV cache and attention state
    kv_cache_state: Optional[Dict[str, Any]] = None
    attention_patterns: Optional[Dict[str, Any]] = None
    
    # Performance monitoring
    mode_performance_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    latency_history: List[float] = field(default_factory=list)
    throughput_history: List[float] = field(default_factory=list)
    
    # Recovery and checkpointing
    checkpoint_data: Optional[Dict[str, Any]] = None
    recovery_attempts: int = 0
    last_known_good_state: Optional[Dict[str, Any]] = None
    
    # Streaming state
    streaming_active: bool = False
    streaming_chunks: List[Dict[str, Any]] = field(default_factory=list)
    chunk_coordination_state: Optional[Dict[str, Any]] = None
    
    # Error handling and validation
    validation_errors: List[str] = field(default_factory=list)
    warning_count: int = 0
    last_validation_timestamp: float = field(default_factory=time.time)
    
    # Generation parameters
    generation_config: Dict[str, Any] = field(default_factory=dict)
    mode_specific_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def validate_state(self) -> bool:
        """
        Validate state consistency and integrity with comprehensive error checking.
        
        Returns:
            True if state is valid and consistent
            
        Raises:
            GenerationModeError: If state validation fails
        """
        try:
            self.validation_errors.clear()
            
            # Validate core state
            if self.current_position < 0:
                self.validation_errors.append(f"Invalid current_position: {self.current_position}")
            
            if self.generated_tokens < 0:
                self.validation_errors.append(f"Invalid generated_tokens: {self.generated_tokens}")
            
            if self.input_sequence_length < 0:
                self.validation_errors.append(f"Invalid input_sequence_length: {self.input_sequence_length}")
            
            if self.current_position < self.input_sequence_length:
                self.validation_errors.append(
                    f"current_position {self.current_position} < input_sequence_length {self.input_sequence_length}"
                )
            
            # Validate mode consistency
            if self.current_mode not in GenerationMode:
                self.validation_errors.append(f"Invalid current_mode: {self.current_mode}")
            
            # Validate audio state consistency
            if self.audio_generation_active:
                if self.current_codebook_index < 0:
                    self.validation_errors.append(f"Invalid current_codebook_index: {self.current_codebook_index}")
                
                if self.delay_pattern_offset < 0:
                    self.validation_errors.append(f"Invalid delay_pattern_offset: {self.delay_pattern_offset}")
                
                if self.audio_token_count < 0:
                    self.validation_errors.append(f"Invalid audio_token_count: {self.audio_token_count}")
            
            # Validate codebook states if present
            if self.codebook_states:
                for codebook_id, state in self.codebook_states.items():
                    if not isinstance(state, dict):
                        self.validation_errors.append(f"Invalid codebook state for {codebook_id}: not a dict")
                    elif 'active' not in state:
                        self.validation_errors.append(f"Codebook {codebook_id} missing 'active' field")
            
            # Validate performance metrics
            if self.mode_performance_metrics:
                for mode, metrics in self.mode_performance_metrics.items():
                    if not isinstance(metrics, dict):
                        self.validation_errors.append(f"Invalid performance metrics for {mode}: not a dict")
                    for metric_name, value in metrics.items():
                        if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
                            self.validation_errors.append(f"Invalid {metric_name} for {mode}: {value}")
            
            # Validate streaming state
            if self.streaming_active and not isinstance(self.streaming_chunks, list):
                self.validation_errors.append("streaming_chunks must be a list when streaming is active")
            
            # Update validation timestamp
            self.last_validation_timestamp = time.time()
            
            if self.validation_errors:
                error_summary = "; ".join(self.validation_errors[:5])  # Limit to first 5 errors
                if len(self.validation_errors) > 5:
                    error_summary += f" (and {len(self.validation_errors) - 5} more)"
                
                raise GenerationModeError(
                    f"State validation failed: {error_summary}",
                    error_code="VALIDATION_FAILED",
                    context={
                        "total_errors": len(self.validation_errors),
                        "current_mode": self.current_mode.value,
                        "current_position": self.current_position
                    }
                )
            
            return True
            
        except GenerationModeError:
            raise
        except Exception as e:
            raise GenerationModeError(
                f"Unexpected error during state validation: {e}",
                error_code="VALIDATION_FAILED"
            ) from e
    
    def create_checkpoint(self) -> Dict[str, Any]:
        """
        Create a checkpoint of the current state for recovery purposes.
        
        Returns:
            Dictionary containing serialized state data
        """
        try:
            checkpoint = {
                'timestamp': time.time(),
                'state_version': '1.0',
                'core_state': {
                    'current_mode': self.current_mode.value,
                    'previous_mode': self.previous_mode.value if self.previous_mode else None,
                    'current_position': self.current_position,
                    'input_sequence_length': self.input_sequence_length,
                    'generated_tokens': self.generated_tokens,
                    'max_sequence_length': self.max_sequence_length,
                },
                'audio_state': {
                    'audio_generation_active': self.audio_generation_active,
                    'current_codebook_index': self.current_codebook_index,
                    'delay_pattern_offset': self.delay_pattern_offset,
                    'audio_token_count': self.audio_token_count,
                    'delay_pattern_active': self.delay_pattern_active,
                },
                'context_state': {
                    'text_context': self.text_context,
                    'audio_context': self.audio_context,
                    'conditioning_state': self.conditioning_state,
                },
                'codebook_states': self.codebook_states.copy(),
                'performance_metrics': self.mode_performance_metrics.copy(),
                'streaming_state': {
                    'streaming_active': self.streaming_active,
                    'chunk_count': len(self.streaming_chunks),
                    'chunk_coordination_state': self.chunk_coordination_state,
                },
                'generation_config': self.generation_config.copy(),
                'mode_specific_config': self.mode_specific_config.copy(),
            }
            
            self.checkpoint_data = checkpoint
            self.last_known_good_state = checkpoint.copy()
            
            return checkpoint
            
        except Exception as e:
            raise GenerationModeError(
                f"Failed to create state checkpoint: {e}",
                error_code="CHECKPOINT_FAILED"
            ) from e
    
    def restore_from_checkpoint(self, checkpoint: Optional[Dict[str, Any]] = None) -> bool:
        """
        Restore state from a checkpoint.
        
        Args:
            checkpoint: Checkpoint data to restore from. If None, uses last known good state.
            
        Returns:
            True if restoration was successful
            
        Raises:
            GenerationModeError: If restoration fails
        """
        try:
            if checkpoint is None:
                checkpoint = self.last_known_good_state
            
            if checkpoint is None:
                raise GenerationModeError(
                    "No checkpoint data available for restoration",
                    error_code="CHECKPOINT_FAILED"
                )
            
            # Validate checkpoint format
            required_keys = ['timestamp', 'state_version', 'core_state', 'audio_state']
            for key in required_keys:
                if key not in checkpoint:
                    raise GenerationModeError(
                        f"Invalid checkpoint: missing {key}",
                        error_code="CHECKPOINT_FAILED"
                    )
            
            # Restore core state
            core_state = checkpoint['core_state']
            self.current_mode = GenerationMode(core_state['current_mode'])
            self.previous_mode = GenerationMode(core_state['previous_mode']) if core_state['previous_mode'] else None
            self.current_position = core_state['current_position']
            self.input_sequence_length = core_state['input_sequence_length']
            self.generated_tokens = core_state['generated_tokens']
            self.max_sequence_length = core_state['max_sequence_length']
            
            # Restore audio state
            audio_state = checkpoint['audio_state']
            self.audio_generation_active = audio_state['audio_generation_active']
            self.current_codebook_index = audio_state['current_codebook_index']
            self.delay_pattern_offset = audio_state['delay_pattern_offset']
            self.audio_token_count = audio_state['audio_token_count']
            self.delay_pattern_active = audio_state['delay_pattern_active']
            
            # Restore context state
            if 'context_state' in checkpoint:
                context_state = checkpoint['context_state']
                self.text_context = context_state.get('text_context')
                self.audio_context = context_state.get('audio_context')
                self.conditioning_state = context_state.get('conditioning_state')
            
            # Restore other states
            if 'codebook_states' in checkpoint:
                self.codebook_states = checkpoint['codebook_states'].copy()
            
            if 'performance_metrics' in checkpoint:
                self.mode_performance_metrics = checkpoint['performance_metrics'].copy()
            
            if 'streaming_state' in checkpoint:
                streaming_state = checkpoint['streaming_state']
                self.streaming_active = streaming_state['streaming_active']
                self.chunk_coordination_state = streaming_state.get('chunk_coordination_state')
            
            if 'generation_config' in checkpoint:
                self.generation_config = checkpoint['generation_config'].copy()
            
            if 'mode_specific_config' in checkpoint:
                self.mode_specific_config = checkpoint['mode_specific_config'].copy()
            
            # Validate restored state
            self.validate_state()
            
            return True
            
        except GenerationModeError:
            raise
        except Exception as e:
            self.recovery_attempts += 1
            raise GenerationModeError(
                f"Failed to restore from checkpoint: {e}",
                error_code="CHECKPOINT_FAILED",
                context={"recovery_attempts": self.recovery_attempts}
            ) from e
    
    def serialize(self) -> str:
        """
        Serialize the state to JSON string.
        
        Returns:
            JSON string representation of the state
        """
        try:
            # Convert enum values to strings for JSON serialization
            serializable_data = asdict(self)
            serializable_data['current_mode'] = self.current_mode.value
            serializable_data['previous_mode'] = self.previous_mode.value if self.previous_mode else None
            
            return json.dumps(serializable_data, indent=2, default=str)
            
        except Exception as e:
            raise GenerationModeError(
                f"Failed to serialize state: {e}",
                error_code="SERIALIZATION_FAILED"
            ) from e
    
    @classmethod
    def deserialize(cls, data: str) -> 'GenerationState':
        """
        Deserialize state from JSON string.
        
        Args:
            data: JSON string containing serialized state
            
        Returns:
            GenerationState instance with restored data
        """
        try:
            parsed_data = json.loads(data)
            
            # Convert string mode values back to enums
            parsed_data['current_mode'] = GenerationMode(parsed_data['current_mode'])
            if parsed_data['previous_mode']:
                parsed_data['previous_mode'] = GenerationMode(parsed_data['previous_mode'])
            else:
                parsed_data['previous_mode'] = None
            
            return cls(**parsed_data)
            
        except Exception as e:
            raise GenerationModeError(
                f"Failed to deserialize state: {e}",
                error_code="DESERIALIZATION_FAILED"
            ) from e
    
    def update_performance_metrics(self, mode: GenerationMode, metrics: Dict[str, float]) -> None:
        """
        Update performance metrics for a specific mode.
        
        Args:
            mode: Generation mode to update metrics for
            metrics: Dictionary of metric names and values
        """
        try:
            if mode.value not in self.mode_performance_metrics:
                self.mode_performance_metrics[mode.value] = {}
            
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)) and not (math.isnan(value) or math.isinf(value)):
                    self.mode_performance_metrics[mode.value][metric_name] = float(value)
                    
                    # Track specific metrics in history
                    if metric_name == 'latency_ms':
                        self.latency_history.append(value)
                        # Keep only recent history
                        if len(self.latency_history) > 100:
                            self.latency_history = self.latency_history[-100:]
                    elif metric_name == 'throughput_tokens_per_sec':
                        self.throughput_history.append(value)
                        if len(self.throughput_history) > 100:
                            self.throughput_history = self.throughput_history[-100:]
                            
        except Exception as e:
            warnings.warn(f"Failed to update performance metrics: {e}", UserWarning)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of performance metrics across all modes.
        
        Returns:
            Dictionary containing performance summary statistics
        """
        try:
            summary = {
                'total_modes_used': len(self.mode_performance_metrics),
                'mode_metrics': self.mode_performance_metrics.copy(),
                'overall_stats': {}
            }
            
            if self.latency_history:
                summary['overall_stats']['avg_latency_ms'] = sum(self.latency_history) / len(self.latency_history)
                summary['overall_stats']['min_latency_ms'] = min(self.latency_history)
                summary['overall_stats']['max_latency_ms'] = max(self.latency_history)
            
            if self.throughput_history:
                summary['overall_stats']['avg_throughput'] = sum(self.throughput_history) / len(self.throughput_history)
                summary['overall_stats']['min_throughput'] = min(self.throughput_history)
                summary['overall_stats']['max_throughput'] = max(self.throughput_history)
            
            summary['overall_stats']['mode_transitions'] = self.mode_transition_count
            summary['overall_stats']['generated_tokens'] = self.generated_tokens
            summary['overall_stats']['audio_tokens'] = self.audio_token_count
            summary['overall_stats']['validation_errors'] = len(self.validation_errors)
            summary['overall_stats']['warnings'] = self.warning_count
            
            return summary
            
        except Exception as e:
            warnings.warn(f"Failed to generate performance summary: {e}", UserWarning)
            return {'error': str(e)}


class GenerationModeManager:
    """
    Central coordinator for generation mode management in Higgs Audio TTS model.
    
    This class provides comprehensive mode management including state coordination,
    transition validation, performance monitoring, and automatic recovery mechanisms.
    It ensures smooth transitions between TEXT, AUDIO_INIT, and AUDIO_IN_PROGRESS modes
    while maintaining context and optimizing for TTS-specific requirements.
    
    Key Features:
    - Validated mode transitions with precondition checking
    - Context preservation across mode changes
    - Performance monitoring and optimization
    - Automatic recovery from failure states
    - Streaming support with mode-aware coordination
    - Multi-codebook audio generation management
    - KV cache coordination during mode transitions
    - Comprehensive logging and debugging support
    """
    
    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 enable_performance_monitoring: bool = True,
                 enable_auto_recovery: bool = True,
                 max_recovery_attempts: int = 3,
                 performance_threshold_latency_ms: float = 1000.0,
                 performance_threshold_throughput: float = 10.0):
        """
        Initialize GenerationModeManager with comprehensive configuration.
        
        Args:
            config: Configuration dictionary with mode-specific settings
            enable_performance_monitoring: Whether to track performance metrics
            enable_auto_recovery: Whether to enable automatic recovery from errors
            max_recovery_attempts: Maximum number of recovery attempts
            performance_threshold_latency_ms: Latency threshold for performance warnings
            performance_threshold_throughput: Throughput threshold for performance warnings
        """
        self.config = config or {}
        self.enable_performance_monitoring = enable_performance_monitoring
        self.enable_auto_recovery = enable_auto_recovery
        self.max_recovery_attempts = max_recovery_attempts
        self.performance_threshold_latency_ms = performance_threshold_latency_ms
        self.performance_threshold_throughput = performance_threshold_throughput
        
        # Initialize state
        self.current_state = GenerationState()
        self.transition_history: List[Dict[str, Any]] = []
        self.performance_data: Dict[str, List[float]] = {
            'transition_times': [],
            'validation_times': [],
            'recovery_times': []
        }
        
        # Recovery state
        self.in_recovery_mode = False
        self.recovery_start_time: Optional[float] = None
        
        # Configuration for different modes
        self._initialize_mode_configurations()
    
    def _initialize_mode_configurations(self) -> None:
        """Initialize default configurations for each generation mode."""
        default_mode_configs = {
            GenerationMode.TEXT.value: {
                'max_sequence_length': 2048,
                'use_kv_cache': True,
                'attention_optimization': 'standard',
                'memory_optimization_level': 'balanced'
            },
            GenerationMode.AUDIO_INIT.value: {
                'max_sequence_length': 4096,
                'use_kv_cache': True,
                'attention_optimization': 'multimodal',
                'memory_optimization_level': 'aggressive',
                'codebook_initialization_delay': 0.1,
                'context_preservation_strength': 'high'
            },
            GenerationMode.AUDIO_IN_PROGRESS.value: {
                'max_sequence_length': 8192,
                'use_kv_cache': True,
                'attention_optimization': 'streaming',
                'memory_optimization_level': 'streaming',
                'streaming_chunk_size': 32,
                'delay_pattern_coordination': True,
                'context_preservation_strength': 'medium'
            }
        }
        
        # Merge with provided config
        for mode_name, default_config in default_mode_configs.items():
            if mode_name not in self.config:
                self.config[mode_name] = {}
            
            for key, value in default_config.items():
                if key not in self.config[mode_name]:
                    self.config[mode_name][key] = value
        
        # Store in current state
        self.current_state.mode_specific_config = self.config.copy()
    
    def transition_to_mode(self,
                          target_mode: Union[GenerationMode, str],
                          force: bool = False,
                          preserve_context: bool = True,
                          validation_level: str = 'standard') -> bool:
        """
        Transition to a new generation mode with comprehensive validation and recovery.
        
        Args:
            target_mode: Target generation mode to transition to
            force: Whether to force transition even if validation fails
            preserve_context: Whether to preserve generation context during transition
            validation_level: Level of validation ('minimal', 'standard', 'comprehensive')
            
        Returns:
            True if transition was successful
            
        Raises:
            GenerationModeError: If transition fails and auto-recovery is disabled
        """
        try:
            # Convert string to enum if needed
            if isinstance(target_mode, str):
                target_mode = GenerationMode(target_mode)
            
            # Record transition start time for performance monitoring
            transition_start_time = time.time()
            
            # Check if already in target mode
            if self.current_state.current_mode == target_mode:
                return True
            
            # Create checkpoint before transition
            if preserve_context:
                self.current_state.create_checkpoint()
            
            # Validate transition
            if not force:
                validation_start_time = time.time()
                self._validate_transition_preconditions(target_mode, validation_level)
                validation_time = time.time() - validation_start_time
                
                if self.enable_performance_monitoring:
                    self.performance_data['validation_times'].append(validation_time * 1000)  # Convert to ms
            
            # Preserve context if requested
            if preserve_context:
                self._preserve_transition_context(target_mode)
            
            # Apply mode-specific configuration
            self._apply_mode_configuration(target_mode)
            
            # Execute the transition
            previous_mode = self.current_state.current_mode
            self.current_state.previous_mode = previous_mode
            self.current_state.current_mode = target_mode
            self.current_state.mode_transition_count += 1
            
            # Update transition history
            transition_time = time.time() - transition_start_time
            transition_record = {
                'timestamp': time.time(),
                'from_mode': previous_mode.value,
                'to_mode': target_mode.value,
                'transition_time_ms': transition_time * 1000,
                'preserve_context': preserve_context,
                'validation_level': validation_level,
                'forced': force
            }
            
            self.transition_history.append(transition_record)
            self.current_state.mode_transition_history.append((target_mode.value, time.time()))
            
            if self.enable_performance_monitoring:
                self.performance_data['transition_times'].append(transition_time * 1000)
            
            # Post-transition validation
            self.current_state.validate_state()
            
            # Log successful transition
            self._log_transition_success(previous_mode, target_mode, transition_time)
            
            return True
            
        except Exception as e:
            # Handle transition failure
            return self._handle_transition_failure(target_mode, e, preserve_context)
    
    def _validate_transition_preconditions(self,
                                         target_mode: GenerationMode,
                                         validation_level: str = 'standard') -> None:
        """
        Validate preconditions for mode transition with comprehensive error checking.
        
        Args:
            target_mode: Target mode to validate transition to
            validation_level: Level of validation to perform
            
        Raises:
            GenerationModeError: If validation fails
        """
        try:
            current_mode = self.current_state.current_mode
            
            # Basic validation - check if transition is allowed
            valid_transitions = {
                GenerationMode.TEXT: [GenerationMode.AUDIO_INIT, GenerationMode.TEXT],
                GenerationMode.AUDIO_INIT: [GenerationMode.AUDIO_IN_PROGRESS, GenerationMode.TEXT],
                GenerationMode.AUDIO_IN_PROGRESS: [GenerationMode.AUDIO_IN_PROGRESS, GenerationMode.TEXT]
            }
            
            if target_mode not in valid_transitions[current_mode]:
                raise GenerationModeError(
                    f"Invalid mode transition from {current_mode.value} to {target_mode.value}",
                    error_code="INVALID_MODE_TRANSITION",
                    context={
                        "current_mode": current_mode.value,
                        "target_mode": target_mode.value,
                        "valid_targets": [mode.value for mode in valid_transitions[current_mode]]
                    }
                )
            
            # Standard validation
            if validation_level in ['standard', 'comprehensive']:
                # Validate current state integrity
                self.current_state.validate_state()
                
                # Check sequence position constraints
                if target_mode == GenerationMode.AUDIO_INIT:
                    if self.current_state.audio_generation_active:
                        raise GenerationModeError(
                            "Cannot transition to AUDIO_INIT while audio generation is already active",
                            error_code="INVALID_MODE_TRANSITION",
                            context={
                                "current_mode": current_mode.value,
                                "target_mode": target_mode.value,
                                "audio_generation_active": self.current_state.audio_generation_active
                            }
                        )
                
                elif target_mode == GenerationMode.AUDIO_IN_PROGRESS:
                    if not self.current_state.audio_generation_active and current_mode != GenerationMode.AUDIO_INIT:
                        raise GenerationModeError(
                            "Cannot transition to AUDIO_IN_PROGRESS without initializing audio generation first",
                            error_code="INVALID_MODE_TRANSITION",
                            context={
                                "current_mode": current_mode.value,
                                "target_mode": target_mode.value,
                                "audio_generation_active": self.current_state.audio_generation_active
                            }
                        )
            
            # Comprehensive validation
            if validation_level == 'comprehensive':
                # Check resource constraints
                self._validate_resource_constraints(target_mode)
                
                # Check performance thresholds
                self._validate_performance_thresholds()
                
                # Check context integrity
                self._validate_context_integrity(target_mode)
                
        except GenerationModeError:
            raise
        except Exception as e:
            raise GenerationModeError(
                f"Unexpected error during transition validation: {e}",
                error_code="VALIDATION_FAILED"
            ) from e
    
    def _validate_resource_constraints(self, target_mode: GenerationMode) -> None:
        """Validate that system resources are sufficient for target mode."""
        mode_config = self.config.get(target_mode.value, {})
        
        # Check sequence length constraints
        max_seq_len = mode_config.get('max_sequence_length', 2048)
        if self.current_state.current_position > max_seq_len:
            raise GenerationModeError(
                f"Current position {self.current_state.current_position} exceeds maximum for {target_mode.value} mode ({max_seq_len})",
                error_code="RESOURCE_CONSTRAINT_VIOLATION"
            )
    
    def _validate_performance_thresholds(self) -> None:
        """Validate that performance is within acceptable thresholds."""
        if not self.enable_performance_monitoring or not self.current_state.latency_history:
            return
        
        recent_latency = sum(self.current_state.latency_history[-10:]) / min(10, len(self.current_state.latency_history))
        if recent_latency > self.performance_threshold_latency_ms:
            warnings.warn(
                f"Recent average latency ({recent_latency:.2f}ms) exceeds threshold ({self.performance_threshold_latency_ms}ms)",
                UserWarning
            )
    
    def _validate_context_integrity(self, target_mode: GenerationMode) -> None:
        """Validate context integrity for the target mode."""
        if target_mode in [GenerationMode.AUDIO_INIT, GenerationMode.AUDIO_IN_PROGRESS]:
            # Check that we have necessary context for audio generation
            if self.current_state.text_context is None:
                raise GenerationModeError(
                    f"Missing text context required for {target_mode.value} mode",
                    error_code="CONTEXT_VALIDATION_FAILED"
                )
    
    def _preserve_transition_context(self, target_mode: GenerationMode) -> None:
        """Preserve context during mode transition."""
        try:
            # Create context snapshot
            context_snapshot = {
                'timestamp': time.time(),
                'from_mode': self.current_state.current_mode.value,
                'to_mode': target_mode.value,
                'position': self.current_state.current_position,
                'generated_tokens': self.current_state.generated_tokens
            }
            
            # Preserve text context
            if self.current_state.text_context:
                context_snapshot['text_context'] = self.current_state.text_context.copy()
            
            # Preserve audio context
            if self.current_state.audio_context:
                context_snapshot['audio_context'] = self.current_state.audio_context.copy()
            
            # Preserve conditioning state
            if self.current_state.conditioning_state:
                context_snapshot['conditioning_state'] = self.current_state.conditioning_state.copy()
            
            # Store the context snapshot
            self.current_state.checkpoint_data = context_snapshot
            
        except Exception as e:
            raise GenerationModeError(
                f"Failed to preserve transition context: {e}",
                error_code="CONTEXT_PRESERVATION_FAILED"
            ) from e
    
    def _apply_mode_configuration(self, target_mode: GenerationMode) -> None:
        """Apply mode-specific configuration settings."""
        try:
            mode_config = self.config.get(target_mode.value, {})
            
            # Update generation config with mode-specific settings
            self.current_state.generation_config.update(mode_config)
            
            # Apply mode-specific optimizations
            if target_mode == GenerationMode.AUDIO_INIT:
                self.current_state.audio_generation_active = True
                self.current_state.current_codebook_index = 0
                self.current_state.delay_pattern_offset = 0
                
                # Initialize codebook states
                if 'num_codebooks' in mode_config:
                    num_codebooks = mode_config['num_codebooks']
                    for i in range(num_codebooks):
                        self.current_state.codebook_states[f'codebook_{i}'] = {
                            'active': False,
                            'last_token': None,
                            'delay_offset': i * mode_config.get('delay_pattern_stride', 1),
                            'generated_tokens': []
                        }
            
            elif target_mode == GenerationMode.AUDIO_IN_PROGRESS:
                # Enable delay pattern coordination
                self.current_state.delay_pattern_active = mode_config.get('delay_pattern_coordination', True)
                
                # Configure streaming if enabled
                if mode_config.get('streaming_optimization', False):
                    self.current_state.streaming_active = True
                    self.current_state.chunk_coordination_state = {
                        'chunk_size': mode_config.get('streaming_chunk_size', 32),
                        'overlap_size': mode_config.get('streaming_overlap_size', 4)
                    }
            
            elif target_mode == GenerationMode.TEXT:
                # Reset audio-specific state
                self.current_state.audio_generation_active = False
                self.current_state.delay_pattern_active = False
                self.current_state.streaming_active = False
                self.current_state.current_codebook_index = 0
                self.current_state.delay_pattern_offset = 0
                
        except Exception as e:
            raise GenerationModeError(
                f"Failed to apply mode configuration: {e}",
                error_code="CONFIGURATION_APPLICATION_FAILED"
            ) from e
    
    def _handle_transition_failure(self,
                                 target_mode: GenerationMode,
                                 error: Exception,
                                 preserve_context: bool) -> bool:
        """Handle mode transition failure with recovery mechanisms."""
        try:
            if not self.enable_auto_recovery:
                raise GenerationModeError(
                    f"Mode transition failed: {error}",
                    error_code="MODE_TRANSITION_FAILED"
                ) from error
            
            # Check recovery attempts
            if self.current_state.recovery_attempts >= self.max_recovery_attempts:
                raise GenerationModeError(
                    f"Maximum recovery attempts ({self.max_recovery_attempts}) exceeded",
                    error_code="RECOVERY_FAILED",
                    context={"recovery_attempts": self.current_state.recovery_attempts}
                ) from error
            
            # Start recovery process
            self.in_recovery_mode = True
            self.recovery_start_time = time.time()
            self.current_state.recovery_attempts += 1
            
            # Attempt to restore from checkpoint
            if preserve_context and self.current_state.last_known_good_state:
                try:
                    self.current_state.restore_from_checkpoint()
                    recovery_time = time.time() - self.recovery_start_time
                    
                    if self.enable_performance_monitoring:
                        self.performance_data['recovery_times'].append(recovery_time * 1000)
                    
                    self.in_recovery_mode = False
                    self.recovery_start_time = None
                    
                    warnings.warn(
                        f"Recovered from transition failure using checkpoint after {recovery_time:.3f}s",
                        UserWarning
                    )
                    return False  # Transition failed but recovery succeeded
                    
                except Exception as recovery_error:
                    warnings.warn(
                        f"Checkpoint recovery failed: {recovery_error}",
                        UserWarning
                    )
            
            # Fallback: reset to safe state
            try:
                self.current_state.current_mode = GenerationMode.TEXT
                self.current_state.audio_generation_active = False
                self.current_state.delay_pattern_active = False
                self.current_state.streaming_active = False
                
                recovery_time = time.time() - self.recovery_start_time
                if self.enable_performance_monitoring:
                    self.performance_data['recovery_times'].append(recovery_time * 1000)
                
                self.in_recovery_mode = False
                self.recovery_start_time = None
                
                warnings.warn(
                    f"Reset to TEXT mode for recovery after {recovery_time:.3f}s",
                    UserWarning
                )
                return False
                
            except Exception as fallback_error:
                self.in_recovery_mode = False
                self.recovery_start_time = None
                raise GenerationModeError(
                    f"All recovery attempts failed. Original error: {error}. Fallback error: {fallback_error}",
                    error_code="RECOVERY_FAILED"
                ) from error
                
        except GenerationModeError:
            raise
        except Exception as e:
            raise GenerationModeError(
                f"Recovery process failed: {e}",
                error_code="RECOVERY_FAILED"
            ) from e
    
    def _log_transition_success(self,
                              from_mode: GenerationMode,
                              to_mode: GenerationMode,
                              transition_time: float) -> None:
        """Log successful mode transition."""
        # This would typically integrate with a logging system
        # For now, we'll just store in performance data
        pass
    
    def get_current_mode(self) -> GenerationMode:
        """Get the current generation mode."""
        return self.current_state.current_mode
    
    def get_generation_state(self) -> GenerationState:
        """Get the current generation state."""
        return self.current_state
    
    def is_audio_generation_active(self) -> bool:
        """Check if audio generation is currently active."""
        return self.current_state.audio_generation_active
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = self.current_state.get_performance_summary()
        
        # Add manager-specific metrics
        if self.performance_data['transition_times']:
            summary['transition_metrics'] = {
                'avg_transition_time_ms': sum(self.performance_data['transition_times']) / len(self.performance_data['transition_times']),
                'min_transition_time_ms': min(self.performance_data['transition_times']),
                'max_transition_time_ms': max(self.performance_data['transition_times']),
                'total_transitions': len(self.performance_data['transition_times'])
            }
        
        if self.performance_data['recovery_times']:
            summary['recovery_metrics'] = {
                'avg_recovery_time_ms': sum(self.performance_data['recovery_times']) / len(self.performance_data['recovery_times']),
                'total_recoveries': len(self.performance_data['recovery_times']),
                'recovery_success_rate': (len(self.performance_data['recovery_times']) / max(1, self.current_state.recovery_attempts))
            }
        
        summary['manager_status'] = {
            'in_recovery_mode': self.in_recovery_mode,
            'total_transitions': len(self.transition_history),
            'current_mode': self.current_state.current_mode.value,
            'auto_recovery_enabled': self.enable_auto_recovery
        }
        
        return summary
    
    def reset_state(self, preserve_config: bool = True) -> None:
        """Reset generation state to initial values."""
        config_backup = self.current_state.mode_specific_config.copy() if preserve_config else {}
        
        self.current_state = GenerationState()
        
        if preserve_config:
            self.current_state.mode_specific_config = config_backup
        
        self.transition_history.clear()
        self.in_recovery_mode = False
        self.recovery_start_time = None
    
    def create_state_checkpoint(self) -> str:
        """Create a serialized checkpoint of the current state."""
        self.current_state.create_checkpoint()
        return self.current_state.serialize()
    
    def restore_state_from_checkpoint(self, checkpoint_data: str) -> bool:
        """Restore state from a serialized checkpoint."""
        try:
            restored_state = GenerationState.deserialize(checkpoint_data)
            self.current_state = restored_state
            return True
        except Exception as e:
            warnings.warn(f"Failed to restore state from checkpoint: {e}", UserWarning)
            return False