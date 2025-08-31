# SPDX-License-Identifier: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""CUDA Graph optimizations for Higgs Audio TTS model.

This module implements comprehensive CUDA graph optimizations specifically designed
for TTS (Text-to-Speech) workloads. It provides mode-aware graph management,
streaminging inference optimization, and integration with DualFFN architecture
and delay pattern coordination for multi-codebook RVQ generation.

Key Features:
- Mode-specific CUDA graphs for TEXT/AUDIO_INIT/AUDIO_IN_PROGRESS phases
- Streaming inference optimization with low-latency execution paths
- DualFFN-aware graph creation with audio/text path optimization
- Multi-codebook delay pattern coordination
- Memory pool management and graph caching
- Performance monitoring and fallback mechanisms
"""

from __future__ import annotations

import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set

import torch
import torch.nn.functional as F
from tensorrt_llm.functional import Tensor

from .generation_mode_manager import GenerationMode, GenerationState
from .config import HiggsAudioConfig


class TTSGraphType(Enum):
    """TTS-specific CUDA graph types for different generation phases."""
    TEXT = "text"
    AUDIO_INIT = "audio_init"
    AUDIO_IN_PROGRESS = "audio_in_progress" 
    STREAMING_CHUNK = "streaming_chunk"
    DELAY_PATTERN = "delay_pattern"


@dataclass
class GraphCacheKey:
    """Cache key for CUDA graphs with TTS-specific parameters."""
    graph_type: TTSGraphType
    batch_size: int
    sequence_length: int
    num_codebooks: int = 1
    chunk_size: int = 0
    audio_mask_pattern: Optional[str] = None
    
    def __hash__(self) -> int:
        return hash((
            self.graph_type,
            self.batch_size,
            self.sequence_length,
            self.num_codebooks,
            self.chunk_size,
            self.audio_mask_pattern
        ))


@dataclass
class CudaGraphInstance:
    """Represents a cached CUDA graph instance with metadata."""
    graph: torch.cuda.CUDAGraph
    input_tensors: Dict[str, torch.Tensor]
    output_tensors: Dict[str, torch.Tensor]
    cache_key: GraphCacheKey
    creation_time: float = field(default_factory=time.time)
    last_used_time: float = field(default_factory=time.time)
    usage_count: int = 0
    memory_pool: Optional[torch.cuda.graph_pool_handle] = None
    
    def execute(self) -> None:
        """Execute the CUDA graph and update usage statistics."""
        self.last_used_time = time.time()
        self.usage_count += 1
        self.graph.replay()
    
    def update_inputs(self, inputs: Dict[str, torch.Tensor]) -> None:
        """Update input tensors for graph execution."""
        for name, tensor in inputs.items():
            if name in self.input_tensors:
                if tensor.shape != self.input_tensors[name].shape:
                    raise ValueError(
                        f"Input tensor '{name}' shape mismatch: expected {self.input_tensors[name].shape}, "
                        f"got {tensor.shape}"
                    )
                self.input_tensors[name].copy_(tensor)
            else:
                warnings.warn(f"Unexpected input tensor '{name}' not found in graph inputs")


class CudaGraphManager:
    """TTS-optimized CUDA graph manager for Higgs Audio model.
    
    This class provides comprehensive CUDA graph optimization specifically designed
    for TTS workloads. It manages mode-specific graphs, streaming inference,
    and coordinates with DualFFN architecture and delay pattern systems.
    
    Key Features:
    - Mode-aware graph creation and caching
    - Streaming inference optimization
    - Memory pool management
    - Performance monitoring and fallback mechanisms
    - DualFFN and delay pattern integration
    
    Example:
        >>> config = HiggsAudioConfig()
        >>> manager = CudaGraphManager(config)
        >>> outputs = manager.execute_graph(graph, inputs)
    """
    
    def __init__(self,
                 config: HiggsAudioConfig,
                 enable_caching: bool = True,
                 max_cache_size: int = 32,
                 enable_performance_monitoring: bool = True,
                 memory_pool_handle: Optional[torch.cuda.graph_pool_handle] = None,
                 fallback_enabled: bool = True):
        """Initialize TTS-optimized CUDA graph manager.
        
        Args:
            config: Higgs Audio configuration with CUDA graph parameters
            enable_caching: Whether to enable graph caching for reuse
            max_cache_size: Maximum number of graphs to cache
            enable_performance_monitoring: Whether to collect execution statistics
            memory_pool_handle: Optional shared memory pool for graph allocation
            fallback_enabled: Whether to enable fallback to non-graph execution
        """
        self.config = config
        self.enable_caching = enable_caching
        self.max_cache_size = max_cache_size
        self.enable_performance_monitoring = enable_performance_monitoring
        self.fallback_enabled = fallback_enabled
        
        # Graph cache and management
        self.graph_cache: Dict[GraphCacheKey, CudaGraphInstance] = {}
        self.memory_pool = memory_pool_handle or torch.cuda.graph_pool_handle()

        
        # TTS-specific configuration
        self.tts_batch_sizes = getattr(config, 'cuda_graph_tts_batch_sizes', [1])
        self.tts_sequence_lengths = getattr(config, 'cuda_graph_tts_sequence_lengths', [1024])
        self.streaming_chunk_sizes = getattr(config, 'cuda_graph_streaming_chunk_sizes', [32])
        self.enable_streaming_graphs = getattr(config, 'cuda_graph_enable_streaming', True)
        self.enable_delay_pattern_graphs = getattr(config, 'cuda_graph_enable_delay_patterns', True)
        
        # Warmup tracking
        self.warmup_completed = False
        self.warmup_graphs: Set[GraphCacheKey] = set()
    
    def create_tts_graphs(self, 
                         model: torch.nn.Module,
                         validate_graphs: bool = True) -> Dict[TTSGraphType, List[CudaGraphInstance]]:
        """Generate TTS-optimized CUDA graphs for common patterns.
        
        This method pre-creates CUDA graphs for typical TTS workloads including
        different generation modes, batch sizes, and sequence lengths. This
        reduces latency during actual inference.
        
        Args:
            model: The Higgs Audio model to create graphs for
            validate_graphs: Whether to validate created graphs
            
        Returns:
            Dictionary mapping graph types to lists of created graph instances
            
        Raises:
            RuntimeError: If graph creation fails for critical configurations
        """
        created_graphs: Dict[TTSGraphType, List[CudaGraphInstance]] = {
            graph_type: [] for graph_type in TTSGraphType
        }
        
        try:
            # Create mode-specific graphs
            for graph_type in [TTSGraphType.TEXT, TTSGraphType.AUDIO_INIT, TTSGraphType.AUDIO_IN_PROGRESS]:
                for batch_size in self.tts_batch_sizes:
                    for seq_length in self.tts_sequence_lengths:
                        try:
                            cache_key = GraphCacheKey(
                                graph_type=graph_type,
                                batch_size=batch_size,
                                sequence_length=seq_length,
                                num_codebooks=self.config.audio_num_codebooks
                            )
                            
                            # Create forward function for this configuration
                            forward_fn = self._create_mode_specific_forward_fn(model, graph_type)
                            
                            # Create the graph
                            graph_instance = self._create_graph_instance(
                                cache_key=cache_key,
                                forward_fn=forward_fn,
                                model=model
                            )
                            
                            if validate_graphs:
                                self._validate_graph_instance(graph_instance)
                            
                            created_graphs[graph_type].append(graph_instance)
                            self.warmup_graphs.add(cache_key)
                            
                        except Exception as e:
                            warnings.warn(
                                f"Failed to create {graph_type.value} graph for batch_size={batch_size}, "
                                f"seq_length={seq_length}: {e}"
                            )
                            continue
            
            # Create streaming graphs if enabled
            if self.enable_streaming_graphs:
                created_graphs[TTSGraphType.STREAMING_CHUNK] = self._create_streaming_graphs(model)
            
            # Create delay pattern graphs if enabled
            if self.enable_delay_pattern_graphs:
                created_graphs[TTSGraphType.DELAY_PATTERN] = self._create_delay_pattern_graphs(model)
            
            self.warmup_completed = True
            return created_graphs
            
        except Exception as e:
            raise RuntimeError(f"Failed to create TTS CUDA graphs: {e}") from e
    
    def get_optimal_graph(self,
                         graph_type: TTSGraphType,
                         batch_size: int,
                         sequence_length: int,
                         num_codebooks: int = 1,
                         chunk_size: int = 0,
                         audio_mask_pattern: Optional[str] = None) -> Optional[CudaGraphInstance]:
        """Select the optimal CUDA graph for the given parameters.
        
        This method finds the best matching cached graph or returns None if
        no suitable graph is available. It considers exact matches first,
        then falls back to compatible graphs with larger capacities.
        
        Args:
            graph_type: Type of TTS graph needed
            batch_size: Batch size for inference
            sequence_length: Sequence length for inference
            num_codebooks: Number of audio codebooks (for multi-codebook generation)
            chunk_size: Chunk size (for streaming inference)
            audio_mask_pattern: Audio mask pattern identifier
            
        Returns:
            Optimal graph instance or None if no suitable graph is found
        """
        # Try exact match first
        exact_key = GraphCacheKey(
            graph_type=graph_type,
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_codebooks=num_codebooks,
            chunk_size=chunk_size,
            audio_mask_pattern=audio_mask_pattern
        )
        
        if exact_key in self.graph_cache:
            return self.graph_cache[exact_key]
        
        # Find compatible graph with larger capacity
        best_match = None
        best_score = float('inf')
        
        for cache_key, graph_instance in self.graph_cache.items():
            if (cache_key.graph_type == graph_type and
                cache_key.batch_size >= batch_size and
                cache_key.sequence_length >= sequence_length and
                cache_key.num_codebooks >= num_codebooks):
                
                # Score based on "waste" (how much larger the cached graph is)
                size_diff = (
                    (cache_key.batch_size - batch_size) * 1000 +
                    (cache_key.sequence_length - sequence_length) +
                    (cache_key.num_codebooks - num_codebooks) * 100
                )
                
                if size_diff < best_score:
                    best_score = size_diff
                    best_match = graph_instance
        
        return best_match
    
    def execute_graph(self,
                     graph_instance: CudaGraphInstance,
                     inputs: Dict[str, torch.Tensor],
                     stream: Optional[torch.cuda.Stream] = None) -> Dict[str, torch.Tensor]:
        """Execute CUDA graph with comprehensive error handling and performance tracking.
        
        Args:
            graph_instance: The graph instance to execute
            inputs: Input tensors for the graph
            stream: Optional CUDA stream for execution
            
        Returns:
            Output tensors from graph execution
            
        Raises:
            RuntimeError: If graph execution fails
        """
        
        # Update input tensors
        graph_instance.update_inputs(inputs)
        
        # Execute graph
        if stream is not None:
            with torch.cuda.stream(stream):
                graph_instance.execute()
            stream.synchronize()
        else:
            graph_instance.execute()
            torch.cuda.synchronize()
        
        return graph_instance.output_tensors.copy()
            

    
    def create_streaming_graph(self,
                              model: torch.nn.Module,
                              chunk_size: int,
                              batch_size: int = 1,
                              overlap_size: int = 4) -> CudaGraphInstance:
        """Create CUDA graph optimized for streaming TTS inference.
        
        This method creates a specialized graph for real-time streaming TTS
        applications with minimal latency and optimized memory usage.
        
        Args:
            model: Higgs Audio model
            chunk_size: Size of streaming chunks
            batch_size: Batch size for streaming
            overlap_size: Overlap between chunks for context preservation
            
        Returns:
            Streaming-optimized graph instance
        """
        cache_key = GraphCacheKey(
            graph_type=TTSGraphType.STREAMING_CHUNK,
            batch_size=batch_size,
            sequence_length=chunk_size + overlap_size,
            chunk_size=chunk_size
        )
        
        def streaming_forward_fn(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            """Streaming-optimized forward function."""
            input_ids = inputs['input_ids']
            attention_mask = inputs.get('attention_mask')
            audio_out_mask = inputs.get('audio_out_mask')
            
            # Apply streaming-specific optimizations
            with torch.no_grad():
                outputs = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    audio_out_mask=audio_out_mask,
                    use_cache=False  # Disable KV cache for streaming
                )
                
                # Extract logits for next token generation
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                return {'logits': logits}
        
        return self._create_graph_instance(cache_key, streaming_forward_fn, model)
    
    def create_delay_pattern_graph(self,
                                 model: torch.nn.Module,
                                 num_codebooks: int,
                                 sequence_length: int,
                                 batch_size: int = 1) -> CudaGraphInstance:
        """Create CUDA graph for multi-codebook delay pattern coordination.
        
        This method creates a specialized graph that optimizes multi-codebook
        RVQ generation with proper delay pattern coordination.
        
        Args:
            model: Higgs Audio model
            num_codebooks: Number of RVQ codebooks
            sequence_length: Sequence length for delay pattern
            batch_size: Batch size
            
        Returns:
            Delay pattern optimized graph instance
        """
        cache_key = GraphCacheKey(
            graph_type=TTSGraphType.DELAY_PATTERN,
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_codebooks=num_codebooks
        )
        
        def delay_pattern_forward_fn(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            """Delay pattern optimized forward function."""
            input_ids = inputs['input_ids']
            attention_mask = inputs.get('attention_mask')
            audio_out_mask = inputs.get('audio_out_mask')
            codebook_ids = inputs.get('codebook_ids')
            
            with torch.no_grad():
                outputs = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    audio_out_mask=audio_out_mask,
                    use_cache=False
                )
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                # Apply delay pattern coordination
                if codebook_ids is not None:
                    # Multi-codebook token generation with delay patterns
                    codebook_logits = []
                    for cb_idx in range(num_codebooks):
                        cb_mask = (codebook_ids == cb_idx)
                        if cb_mask.any():
                            cb_logits = logits[cb_mask]
                            codebook_logits.append(cb_logits)
                    
                    return {
                        'logits': logits,
                        'codebook_logits': codebook_logits if codebook_logits else None
                    }
                
                return {'logits': logits}
        
        return self._create_graph_instance(cache_key, delay_pattern_forward_fn, model)
    
    def manage_memory_pools(self, 
                          cleanup_threshold: float = 0.8,
                          max_pool_size_gb: float = 4.0) -> Dict[str, Any]:
        """Manage CUDA graph memory pools to prevent OOM and optimize performance.
        
        Args:
            cleanup_threshold: Memory usage threshold to trigger cleanup (0-1)
            max_pool_size_gb: Maximum memory pool size in GB
            
        Returns:
            Dictionary with memory management statistics
        """
        try:
            # Get current GPU memory usage
            allocated_memory = torch.cuda.memory_allocated() / 1e9  # GB
            cached_memory = torch.cuda.memory_reserved() / 1e9  # GB
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            
            memory_usage_ratio = allocated_memory / total_memory
            
            stats = {
                'allocated_memory_gb': allocated_memory,
                'cached_memory_gb': cached_memory,
                'total_memory_gb': total_memory,
                'usage_ratio': memory_usage_ratio,
                'cleanup_performed': False,
                'graphs_cleaned': 0
            }
            
            # Perform cleanup if necessary
            if memory_usage_ratio > cleanup_threshold or cached_memory > max_pool_size_gb:
                graphs_cleaned = self._cleanup_unused_graphs()
                torch.cuda.empty_cache()
                
                stats['cleanup_performed'] = True
                stats['graphs_cleaned'] = graphs_cleaned
                stats['memory_after_cleanup'] = torch.cuda.memory_allocated() / 1e9
            
            return stats
            
        except Exception as e:
            warnings.warn(f"Memory pool management failed: {e}")
            return {'error': str(e)}
    
    def cleanup_cache(self, max_age_hours: float = 24.0, max_unused_hours: float = 1.0) -> int:
        """Clean up old or unused CUDA graphs from cache.
        
        Args:
            max_age_hours: Maximum age in hours before graph is removed
            max_unused_hours: Maximum time unused in hours before removal
            
        Returns:
            Number of graphs removed from cache
        """
        current_time = time.time()
        graphs_to_remove = []
        
        for cache_key, graph_instance in self.graph_cache.items():
            age_hours = (current_time - graph_instance.creation_time) / 3600
            unused_hours = (current_time - graph_instance.last_used_time) / 3600
            
            if (age_hours > max_age_hours or 
                (unused_hours > max_unused_hours and graph_instance.usage_count == 0)):
                graphs_to_remove.append(cache_key)
        
        # Remove identified graphs
        for cache_key in graphs_to_remove:
            del self.graph_cache[cache_key]
        
        return len(graphs_to_remove)
    
    def _create_graph_instance(self,
                              cache_key: GraphCacheKey,
                              forward_fn: Callable,
                              model: torch.nn.Module) -> CudaGraphInstance:
        """Create a new CUDA graph instance with the given parameters."""
        try:
            # Create sample inputs for graph capture
            sample_inputs = self._create_sample_inputs(cache_key)
            
            # Warm up the model (required for CUDA graph capture)
            with torch.no_grad():
                for _ in range(3):  # Multiple warmup iterations
                    _ = forward_fn(sample_inputs)
                torch.cuda.synchronize()
            
            # Create CUDA graph
            graph = torch.cuda.CUDAGraph()
            
            # Capture graph execution
            with torch.cuda.graph(graph, pool=self.memory_pool):
                sample_outputs = forward_fn(sample_inputs)
            
            # Create graph instance
            graph_instance = CudaGraphInstance(
                graph=graph,
                input_tensors=sample_inputs.copy(),
                output_tensors=sample_outputs.copy() if isinstance(sample_outputs, dict) else {'output': sample_outputs},
                cache_key=cache_key,
                memory_pool=self.memory_pool
            )
            
            # Add to cache if caching is enabled
            if self.enable_caching:
                self._add_to_cache(cache_key, graph_instance)
            
            return graph_instance
            
        except Exception as e:
            raise RuntimeError(f"Failed to create CUDA graph instance: {e}") from e
    
    def _create_sample_inputs(self, cache_key: GraphCacheKey) -> Dict[str, torch.Tensor]:
        """Create sample input tensors for graph capture."""
        batch_size = cache_key.batch_size
        sequence_length = cache_key.sequence_length
        device = torch.cuda.current_device()
        
        # Base inputs
        inputs = {
            'input_ids': torch.randint(
                0, self.config.vocab_size, 
                (batch_size, sequence_length), 
                dtype=torch.int32, 
                device=device
            ),
            'attention_mask': torch.ones(
                (batch_size, sequence_length), 
                dtype=torch.bool, 
                device=device
            )
        }
        
        # Add TTS-specific inputs based on graph type
        if cache_key.graph_type in [TTSGraphType.AUDIO_INIT, TTSGraphType.AUDIO_IN_PROGRESS, TTSGraphType.DELAY_PATTERN]:
            inputs['audio_out_mask'] = torch.zeros(
                (batch_size, sequence_length), 
                dtype=torch.bool, 
                device=device
            )
            
            # Mark some positions as audio tokens
            audio_start = sequence_length // 2  # Simple heuristic
            inputs['audio_out_mask'][:, audio_start:] = True
            
        if cache_key.graph_type == TTSGraphType.DELAY_PATTERN:
            inputs['codebook_ids'] = torch.randint(
                0, cache_key.num_codebooks,
                (batch_size, sequence_length),
                dtype=torch.int32,
                device=device
            )
        
        if cache_key.graph_type == TTSGraphType.STREAMING_CHUNK:
            # Streaming-specific inputs
            inputs['chunk_position'] = torch.tensor([0], dtype=torch.int32, device=device)
            inputs['overlap_mask'] = torch.ones(
                (batch_size, cache_key.chunk_size),
                dtype=torch.bool,
                device=device
            )
        
        return inputs
    
    def _create_mode_specific_forward_fn(self, 
                                       model: torch.nn.Module, 
                                       graph_type: TTSGraphType) -> Callable:
        """Create mode-specific forward function optimized for graph type."""
        
        def mode_forward_fn(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            """Mode-aware forward function with TTS optimizations."""
            input_ids = inputs['input_ids']
            attention_mask = inputs.get('attention_mask')
            audio_out_mask = inputs.get('audio_out_mask')
            
            with torch.no_grad():
                # Set model to appropriate generation mode context
                if hasattr(model, 'set_generation_mode'):
                    if graph_type == TTSGraphType.TEXT:
                        mode = GenerationMode.TEXT
                    elif graph_type == TTSGraphType.AUDIO_INIT:
                        mode = GenerationMode.AUDIO_INIT
                    elif graph_type == TTSGraphType.AUDIO_IN_PROGRESS:
                        mode = GenerationMode.AUDIO_IN_PROGRESS
                    else:
                        mode = GenerationMode.TEXT  # Default fallback
                    
                    # Note: We can't actually change mode during graph capture
                    # This is for documentation/future enhancement
                
                # Forward pass with mode-specific optimizations
                outputs = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    audio_out_mask=audio_out_mask,
                    use_cache=False  # Disable KV cache for graph consistency
                )
                
                # Extract and return relevant outputs
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    # Handle tuple/list outputs
                    logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                
                return {'logits': logits}
        
        return mode_forward_fn
    
    def _create_streaming_graphs(self, model: torch.nn.Module) -> List[CudaGraphInstance]:
        """Create streaming-optimized CUDA graphs for real-time TTS."""
        streaming_graphs = []
        
        for chunk_size in self.streaming_chunk_sizes:
            for batch_size in [1, 2]:  # Focus on low-latency batch sizes
                try:
                    graph_instance = self.create_streaming_graph(
                        model=model,
                        chunk_size=chunk_size,
                        batch_size=batch_size,
                        overlap_size=4
                    )
                    streaming_graphs.append(graph_instance)
                    
                except Exception as e:
                    warnings.warn(
                        f"Failed to create streaming graph for chunk_size={chunk_size}, "
                        f"batch_size={batch_size}: {e}"
                    )
                    continue
        
        return streaming_graphs
    
    def _create_delay_pattern_graphs(self, model: torch.nn.Module) -> List[CudaGraphInstance]:
        """Create delay pattern optimized CUDA graphs for multi-codebook generation."""
        delay_pattern_graphs = []
        
        num_codebooks = self.config.audio_num_codebooks
        
        for batch_size in [1, 2, 4]:  # Common batch sizes for TTS
            for seq_length in [128, 256, 512]:  # Moderate sequence lengths
                try:
                    graph_instance = self.create_delay_pattern_graph(
                        model=model,
                        num_codebooks=num_codebooks,
                        sequence_length=seq_length,
                        batch_size=batch_size
                    )
                    delay_pattern_graphs.append(graph_instance)
                    
                except Exception as e:
                    warnings.warn(
                        f"Failed to create delay pattern graph for batch_size={batch_size}, "
                        f"seq_length={seq_length}: {e}"
                    )
                    continue
        
        return delay_pattern_graphs
    
    def _add_to_cache(self, cache_key: GraphCacheKey, graph_instance: CudaGraphInstance) -> None:
        """Add graph instance to cache with size management."""
        # Check if we need to make room in cache
        if len(self.graph_cache) >= self.max_cache_size:
            # Remove least recently used graph
            lru_key = min(
                self.graph_cache.keys(),
                key=lambda k: self.graph_cache[k].last_used_time
            )
            del self.graph_cache[lru_key]
        
        # Add new graph to cache
        self.graph_cache[cache_key] = graph_instance
    
    def _cleanup_unused_graphs(self) -> int:
        """Clean up unused graphs to free memory."""
        current_time = time.time()
        graphs_to_remove = []
        
        # Identify graphs that haven't been used recently
        for cache_key, graph_instance in self.graph_cache.items():
            time_since_use = current_time - graph_instance.last_used_time
            
            # Remove graphs unused for more than 30 minutes with low usage
            if (time_since_use > 1800 and graph_instance.usage_count < 5) or time_since_use > 7200:
                graphs_to_remove.append(cache_key)
        
        # Remove identified graphs
        for cache_key in graphs_to_remove:
            del self.graph_cache[cache_key]
        
        return len(graphs_to_remove)

    def __del__(self):
        """Cleanup resources when manager is destroyed."""
        try:
            # Clear cache to release CUDA graphs
            self.graph_cache.clear()
            torch.cuda.empty_cache()
        except:
            pass  # Ignore errors during cleanup



class DualFFNGraphOptimizer:
    """CUDA graph optimizer specifically designed for DualFFN architecture.
    
    This class creates optimized CUDA graphs that account for the dual MLP paths
    in DualFFN layers, providing separate optimization strategies for audio and
    text processing paths based on token routing patterns.
    """
    
    def __init__(self, 
                 config: HiggsAudioConfig,
                 cuda_graph_manager: CudaGraphManager):
        """Initialize DualFFN CUDA graph optimizer.
        
        Args:
            config: Higgs Audio configuration
            cuda_graph_manager: Parent CUDA graph manager
        """
        self.config = config
        self.graph_manager = cuda_graph_manager
        self.dual_ffn_layers = set(config.audio_dual_ffn_layers or [])
        
        # DualFFN-specific configuration
        self.separate_graphs = config.cuda_graph_dualffn_separate_graphs
        self.audio_text_ratio_threshold = config.cuda_graph_dualffn_audio_text_ratio_threshold
        
        # Cache for DualFFN-specific graphs
        self.audio_path_graphs: Dict[GraphCacheKey, CudaGraphInstance] = {}
        self.text_path_graphs: Dict[GraphCacheKey, CudaGraphInstance] = {}
        self.mixed_path_graphs: Dict[GraphCacheKey, CudaGraphInstance] = {}
    
    def create_dualffn_graphs(self, 
                             model: torch.nn.Module,
                             layer_indices: Optional[List[int]] = None) -> Dict[str, List[CudaGraphInstance]]:
        """Create CUDA graphs optimized for DualFFN layers.
        
        Args:
            model: Higgs Audio model with DualFFN layers
            layer_indices: Specific layer indices to optimize (None for all DualFFN layers)
            
        Returns:
            Dictionary with audio_path, text_path, and mixed_path graph lists
        """
        if not self.separate_graphs:
            return {'mixed_path': []}
        
        target_layers = layer_indices or list(self.dual_ffn_layers)
        created_graphs = {
            'audio_path': [],
            'text_path': [],
            'mixed_path': []
        }
        
        for layer_idx in target_layers:
            if layer_idx not in self.dual_ffn_layers:
                warnings.warn(f"Layer {layer_idx} is not a DualFFN layer, skipping")
                continue
            
            try:
                # Create graphs for different audio/text ratios
                for batch_size in self.config.cuda_graph_tts_batch_sizes:
                    for seq_length in self.config.cuda_graph_tts_sequence_lengths:
                        # Audio-dominant graph (>70% audio tokens)
                        audio_graph = self._create_path_specific_graph(
                            model=model,
                            layer_idx=layer_idx,
                            batch_size=batch_size,
                            seq_length=seq_length,
                            path_type='audio',
                            audio_ratio=0.8
                        )
                        created_graphs['audio_path'].append(audio_graph)
                        
                        # Text-dominant graph (>70% text tokens) 
                        text_graph = self._create_path_specific_graph(
                            model=model,
                            layer_idx=layer_idx,
                            batch_size=batch_size,
                            seq_length=seq_length,
                            path_type='text',
                            audio_ratio=0.2
                        )
                        created_graphs['text_path'].append(text_graph)
                        
                        # Mixed graph (balanced audio/text)
                        mixed_graph = self._create_path_specific_graph(
                            model=model,
                            layer_idx=layer_idx,
                            batch_size=batch_size,
                            seq_length=seq_length,
                            path_type='mixed',
                            audio_ratio=0.5
                        )
                        created_graphs['mixed_path'].append(mixed_graph)
                        
            except Exception as e:
                warnings.warn(f"Failed to create DualFFN graphs for layer {layer_idx}: {e}")
                continue
        
        return created_graphs
    
    def get_optimal_dualffn_graph(self,
                                 batch_size: int,
                                 sequence_length: int,
                                 audio_out_mask: torch.Tensor,
                                 layer_idx: int) -> Optional[CudaGraphInstance]:
        """Select optimal DualFFN graph based on audio/text token distribution.
        
        Args:
            batch_size: Batch size
            sequence_length: Sequence length
            audio_out_mask: Boolean mask indicating audio tokens
            layer_idx: DualFFN layer index
            
        Returns:
            Optimal graph instance or None if no suitable graph found
        """
        if layer_idx not in self.dual_ffn_layers:
            return None
        
        # Calculate audio token ratio
        audio_ratio = float(audio_out_mask.sum()) / audio_out_mask.numel()
        
        # Determine optimal path type
        if audio_ratio > 0.7:
            path_type = 'audio'
            graph_cache = self.audio_path_graphs
        elif audio_ratio < 0.3:
            path_type = 'text'
            graph_cache = self.text_path_graphs
        else:
            path_type = 'mixed'
            graph_cache = self.mixed_path_graphs
        
        # Create cache key
        cache_key = GraphCacheKey(
            graph_type=TTSGraphType.AUDIO_IN_PROGRESS,  # DualFFN typically used in audio generation
            batch_size=batch_size,
            sequence_length=sequence_length,
            audio_mask_pattern=f"dualffn_{path_type}_{layer_idx}"
        )
        
        return graph_cache.get(cache_key)
    
    def _create_path_specific_graph(self,
                                   model: torch.nn.Module,
                                   layer_idx: int,
                                   batch_size: int,
                                   seq_length: int,
                                   path_type: str,
                                   audio_ratio: float) -> CudaGraphInstance:
        """Create CUDA graph optimized for specific DualFFN path."""
        cache_key = GraphCacheKey(
            graph_type=TTSGraphType.AUDIO_IN_PROGRESS,
            batch_size=batch_size,
            sequence_length=seq_length,
            audio_mask_pattern=f"dualffn_{path_type}_{layer_idx}"
        )
        
        def dualffn_forward_fn(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            """DualFFN-optimized forward function."""
            input_ids = inputs['input_ids']
            attention_mask = inputs.get('attention_mask')
            audio_out_mask = inputs.get('audio_out_mask')
            
            with torch.no_grad():
                # Execute model forward with DualFFN-specific optimizations
                outputs = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    audio_out_mask=audio_out_mask,
                    use_cache=False
                )
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                return {
                    'logits': logits,
                    'path_type': path_type,
                    'layer_idx': layer_idx
                }
        
        # Create sample inputs with appropriate audio/text distribution
        sample_inputs = self.graph_manager._create_sample_inputs(cache_key)
        
        # Adjust audio_out_mask to match desired ratio
        if 'audio_out_mask' in sample_inputs:
            mask_size = sample_inputs['audio_out_mask'].numel()
            audio_tokens = int(mask_size * audio_ratio)
            sample_inputs['audio_out_mask'].fill_(False)
            sample_inputs['audio_out_mask'].view(-1)[:audio_tokens] = True
        
        return self.graph_manager._create_graph_instance(cache_key, dualffn_forward_fn, model)


class DelayPatternGraphCoordinator:
    """Coordinates CUDA graph execution with RVQ delay patterns for multi-codebook generation.
    
    This class manages the complex interaction between CUDA graphs and delay patterns,
    ensuring that multi-codebook RVQ generation maintains proper temporal coordination
    while benefiting from graph optimization.
    """
    
    def __init__(self,
                 config: HiggsAudioConfig,
                 cuda_graph_manager: CudaGraphManager):
        """Initialize delay pattern graph coordinator.
        
        Args:
            config: Higgs Audio configuration
            cuda_graph_manager: Parent CUDA graph manager
        """
        self.config = config
        self.graph_manager = cuda_graph_manager
        self.num_codebooks = config.audio_num_codebooks
        self.max_codebooks = config.cuda_graph_delay_pattern_max_codebooks
        
        # Delay pattern configuration
        self.delay_strategy = config.audio_delay_pattern_strategy
        self.delay_stride = config.audio_delay_pattern_stride
        self.custom_delays = config.audio_delay_pattern_custom_delays
        self.max_delay = config.audio_delay_pattern_max_delay
        
        # Graph cache for different delay patterns
        self.delay_pattern_graphs: Dict[str, Dict[GraphCacheKey, CudaGraphInstance]] = {}
        
        # Pre-computed delay patterns for common configurations
        self.cached_delay_patterns: Dict[Tuple[int, int], torch.Tensor] = {}
    
    def create_delay_pattern_graphs(self, 
                                  model: torch.nn.Module) -> Dict[str, List[CudaGraphInstance]]:
        """Create CUDA graphs for different delay pattern configurations.
        
        Args:
            model: Higgs Audio model
            
        Returns:
            Dictionary mapping delay pattern types to graph lists
        """
        created_graphs = {
            'linear': [],
            'exponential': [],
            'custom': [],
            'streaming': []
        }
        
        # Create graphs for different delay strategies
        for strategy in ['linear', 'exponential']:
            if not self.config.cuda_graph_delay_pattern_optimization_enabled:
                continue
                
            for batch_size in [1, 2, 4]:  # Focus on smaller batches for delay patterns
                for seq_length in [128, 256, 512]:
                    for num_codebooks in [4, 8, 12]:
                        if num_codebooks > self.max_codebooks:
                            continue
                        
                        try:
                            graph_instance = self._create_delay_strategy_graph(
                                model=model,
                                batch_size=batch_size,
                                seq_length=seq_length,
                                num_codebooks=num_codebooks,
                                strategy=strategy
                            )
                            created_graphs[strategy].append(graph_instance)
                            
                        except Exception as e:
                            warnings.warn(
                                f"Failed to create {strategy} delay pattern graph for "
                                f"batch_size={batch_size}, seq_length={seq_length}, "
                                f"num_codebooks={num_codebooks}: {e}"
                            )
                            continue
        
        # Create streaming delay pattern graphs
        if self.config.cuda_graph_enable_streaming:
            created_graphs['streaming'] = self._create_streaming_delay_graphs(model)
        
        return created_graphs
    
    def execute_with_delay_coordination(self,
                                      graph_instance: CudaGraphInstance,
                                      inputs: Dict[str, torch.Tensor],
                                      codebook_states: Dict[str, Any],
                                      current_position: int) -> Dict[str, torch.Tensor]:
        """Execute CUDA graph with delay pattern coordination.
        
        Args:
            graph_instance: Graph to execute
            inputs: Input tensors
            codebook_states: Current codebook generation states
            current_position: Current generation position
            
        Returns:
            Outputs with delay pattern coordination applied
        """
        try:
            # Prepare delay-aware inputs
            delay_inputs = self._prepare_delay_inputs(
                inputs, codebook_states, current_position
            )
            
            # Execute graph with delay coordination
            outputs = self.graph_manager.execute_graph(graph_instance, delay_inputs)
            
            # Apply delay pattern to outputs
            if 'logits' in outputs:
                coordinated_outputs = self._apply_delay_coordination(
                    outputs, codebook_states, current_position
                )
                return coordinated_outputs
            
            return outputs
            
        except Exception as e:
            warnings.warn(f"Delay pattern coordination failed: {e}")
            return self.graph_manager.execute_graph(graph_instance, inputs)
    
    def get_delay_pattern(self, 
                         num_codebooks: int, 
                         sequence_length: int,
                         strategy: Optional[str] = None) -> torch.Tensor:
        """Get or create delay pattern for the given configuration.
        
        Args:
            num_codebooks: Number of codebooks
            sequence_length: Sequence length
            strategy: Delay strategy override
            
        Returns:
            Delay pattern tensor [num_codebooks, sequence_length]
        """
        cache_key = (num_codebooks, sequence_length)
        
        if cache_key in self.cached_delay_patterns:
            return self.cached_delay_patterns[cache_key]
        
        strategy = strategy or self.delay_strategy
        device = torch.cuda.current_device()
        
        if strategy == 'linear':
            delays = torch.arange(num_codebooks, device=device) * self.delay_stride
        elif strategy == 'exponential':
            delays = torch.tensor(
                [(2**i - 1) * self.delay_stride for i in range(num_codebooks)],
                device=device
            )
        elif strategy == 'custom' and self.custom_delays:
            delays = torch.tensor(
                self.custom_delays[:num_codebooks] + [0] * max(0, num_codebooks - len(self.custom_delays)),
                device=device
            )
        else:
            delays = torch.zeros(num_codebooks, device=device)
        
        # Apply max delay constraint
        if self.max_delay is not None:
            delays = torch.clamp(delays, max=self.max_delay)
        
        # Create delay pattern matrix
        delay_pattern = delays.unsqueeze(1).expand(num_codebooks, sequence_length)
        
        # Cache the pattern
        self.cached_delay_patterns[cache_key] = delay_pattern
        
        return delay_pattern
    
    def _create_delay_strategy_graph(self,
                                   model: torch.nn.Module,
                                   batch_size: int,
                                   seq_length: int,
                                   num_codebooks: int,
                                   strategy: str) -> CudaGraphInstance:
        """Create CUDA graph for specific delay strategy."""
        cache_key = GraphCacheKey(
            graph_type=TTSGraphType.DELAY_PATTERN,
            batch_size=batch_size,
            sequence_length=seq_length,
            num_codebooks=num_codebooks,
            audio_mask_pattern=f"delay_{strategy}"
        )
        
        def delay_strategy_forward_fn(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            """Forward function with specific delay strategy."""
            input_ids = inputs['input_ids']
            attention_mask = inputs.get('attention_mask')
            audio_out_mask = inputs.get('audio_out_mask')
            delay_pattern = inputs.get('delay_pattern')
            
            with torch.no_grad():
                outputs = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    audio_out_mask=audio_out_mask,
                    use_cache=False
                )
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                # Apply delay-specific post-processing
                processed_logits = self._apply_delay_strategy_processing(
                    logits, delay_pattern, strategy
                )
                
                return {
                    'logits': processed_logits,
                    'delay_strategy': strategy,
                    'num_codebooks': num_codebooks
                }
        
        return self.graph_manager._create_graph_instance(cache_key, delay_strategy_forward_fn, model)
    
    def _create_streaming_delay_graphs(self, model: torch.nn.Module) -> List[CudaGraphInstance]:
        """Create streaming-specific delay pattern graphs."""
        streaming_graphs = []
        
        for chunk_size in self.config.cuda_graph_streaming_chunk_sizes:
            for num_codebooks in [4, 8]:
                try:
                    cache_key = GraphCacheKey(
                        graph_type=TTSGraphType.STREAMING_CHUNK,
                        batch_size=1,  # Focus on single batch for streaming
                        sequence_length=chunk_size,
                        num_codebooks=num_codebooks,
                        chunk_size=chunk_size,
                        audio_mask_pattern="streaming_delay"
                    )
                    
                    def streaming_delay_forward_fn(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
                        """Streaming forward function with delay coordination."""
                        input_ids = inputs['input_ids']
                        attention_mask = inputs.get('attention_mask')
                        audio_out_mask = inputs.get('audio_out_mask')
                        
                        with torch.no_grad():
                            outputs = model.forward(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                audio_out_mask=audio_out_mask,
                                use_cache=False
                            )
                            
                            if hasattr(outputs, 'logits'):
                                logits = outputs.logits
                            else:
                                logits = outputs
                            
                            # Apply streaming delay coordination
                            streaming_logits = self._apply_streaming_delay_coordination(
                                logits, num_codebooks, chunk_size
                            )
                            
                            return {
                                'logits': streaming_logits,
                                'chunk_size': chunk_size,
                                'num_codebooks': num_codebooks
                            }
                    
                    graph_instance = self.graph_manager._create_graph_instance(
                        cache_key, streaming_delay_forward_fn, model
                    )
                    streaming_graphs.append(graph_instance)
                    
                except Exception as e:
                    warnings.warn(
                        f"Failed to create streaming delay graph for chunk_size={chunk_size}, "
                        f"num_codebooks={num_codebooks}: {e}"
                    )
                    continue
        
        return streaming_graphs
    
    def _prepare_delay_inputs(self,
                             inputs: Dict[str, torch.Tensor],
                             codebook_states: Dict[str, Any],
                             current_position: int) -> Dict[str, torch.Tensor]:
        """Prepare inputs with delay pattern information."""
        delay_inputs = inputs.copy()
        
        # Add delay pattern to inputs
        batch_size, seq_length = inputs['input_ids'].shape
        delay_pattern = self.get_delay_pattern(self.num_codebooks, seq_length)
        delay_inputs['delay_pattern'] = delay_pattern
        
        # Add codebook state information
        active_codebooks = []
        for i in range(self.num_codebooks):
            codebook_key = f'codebook_{i}'
            if codebook_key in codebook_states and codebook_states[codebook_key].get('active', False):
                active_codebooks.append(i)
        
        if active_codebooks:
            codebook_mask = torch.zeros_like(inputs['input_ids'])
            for cb_idx in active_codebooks:
                codebook_mask[:, :] = cb_idx  # Simplified codebook assignment
            delay_inputs['codebook_ids'] = codebook_mask
        
        return delay_inputs
    
    def _apply_delay_coordination(self,
                                outputs: Dict[str, torch.Tensor],
                                codebook_states: Dict[str, Any],
                                current_position: int) -> Dict[str, torch.Tensor]:
        """Apply delay pattern coordination to model outputs."""
        if 'logits' not in outputs:
            return outputs
        
        logits = outputs['logits']
        batch_size, seq_length, vocab_size = logits.shape
        
        # Create codebook-specific logits
        coordinated_outputs = outputs.copy()
        codebook_logits = []
        
        for i in range(self.num_codebooks):
            codebook_key = f'codebook_{i}'
            if codebook_key in codebook_states:
                cb_state = codebook_states[codebook_key]
                delay_offset = cb_state.get('delay_offset', 0)
                
                # Apply delay-specific processing
                if current_position >= delay_offset:
                    effective_position = current_position - delay_offset
                    if effective_position < seq_length:
                        cb_logits = logits[:, effective_position:effective_position+1, :]
                        codebook_logits.append(cb_logits)
        
        if codebook_logits:
            coordinated_outputs['codebook_logits'] = torch.cat(codebook_logits, dim=1)
        
        return coordinated_outputs
    
    def _apply_delay_strategy_processing(self,
                                       logits: torch.Tensor,
                                       delay_pattern: Optional[torch.Tensor],
                                       strategy: str) -> torch.Tensor:
        """Apply strategy-specific processing to logits."""
        if delay_pattern is None:
            return logits
        
        # Strategy-specific post-processing
        if strategy == 'linear':
            # Linear delay processing - no special modification needed
            return logits
        elif strategy == 'exponential':
            # Exponential delay processing - could apply different temperature scaling
            return logits
        else:
            return logits
    
    def _apply_streaming_delay_coordination(self,
                                          logits: torch.Tensor,
                                          num_codebooks: int,
                                          chunk_size: int) -> torch.Tensor:
        """Apply delay coordination specifically for streaming generation."""
        # For streaming, we ensure that codebooks generate tokens in proper order
        batch_size, seq_length, vocab_size = logits.shape
        
        # Create streaming-aware logits processing
        # This is a placeholder for more sophisticated streaming coordination
        streaming_logits = logits.clone()
        
        return streaming_logits


# Integration helper functions

def integrate_cuda_graphs_with_model(model: torch.nn.Module, 
                                   config: HiggsAudioConfig) -> CudaGraphManager:
    """Integrate CUDA graph optimization with Higgs Audio model.
    
    Args:
        model: Higgs Audio model instance
        config: Model configuration with CUDA graph settings
        
    Returns:
        Configured CUDA graph manager
    """
    if not config.cuda_graph_enable:
        return None
    
    # Create CUDA graph manager
    manager = CudaGraphManager(
        config=config,
        enable_caching=True,
        max_cache_size=config.cuda_graph_max_cache_size,
        enable_performance_monitoring=config.cuda_graph_enable_performance_monitoring,
        fallback_enabled=config.cuda_graph_fallback_enabled
    )
    
    # Create TTS-optimized graphs
    try:
        manager.create_tts_graphs(model, validate_graphs=config.cuda_graph_validation_enabled)
    except Exception as e:
        if config.cuda_graph_fallback_enabled:
            warnings.warn(f"CUDA graph creation failed, proceeding without graphs: {e}")
            return None
        else:
            raise
    
    return manager