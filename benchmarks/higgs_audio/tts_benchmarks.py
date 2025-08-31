#!/usr/bin/env python3
# SPDX-License-Identifier: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
TTS-Specific Benchmarking Modules for Higgs Audio TensorRT-LLM Implementation

This module provides specialized benchmarking capabilities for TTS-specific features
including generation modes, delay patterns, streaming performance, and DualFFN
architecture validation.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from .benchmark_suite import BenchmarkConfiguration, BenchmarkResult, BenchmarkType, ArchitectureType
from tensorrt_llm.models.higgs_audio.generation_mode_manager import GenerationMode, GenerationModeManager
from tensorrt_llm.models.higgs_audio.model import DelayPatternProvider, AudioTokenUtils
from tensorrt_llm.models.higgs_audio.cuda_graphs import DualFFNGraphOptimizer, DelayPatternGraphCoordinator


logger = logging.getLogger(__name__)


@dataclass
class TTSBenchmarkResult(BenchmarkResult):
    """Extended benchmark result for TTS-specific metrics."""
    
    # TTS-specific metrics
    audio_quality_score: float = 0.0
    voice_similarity_score: float = 0.0
    streaming_latency_ms: float = 0.0
    codebook_synchronization_score: float = 0.0
    delay_pattern_efficiency: float = 0.0
    
    # Generation mode metrics
    mode_transition_latency_ms: float = 0.0
    mode_transition_success_rate: float = 0.0
    
    # DualFFN metrics
    audio_path_utilization: float = 0.0
    text_path_utilization: float = 0.0
    dualffn_routing_efficiency: float = 0.0


class GenerationModeBenchmark:
    """Benchmark for TTS generation mode transitions and performance."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.mode_manager = GenerationModeManager()
        
    async def benchmark_generation_modes(self, 
                                       config: BenchmarkConfiguration) -> List[TTSBenchmarkResult]:
        """Benchmark performance across different generation modes."""
        logger.info("Benchmarking generation modes")
        
        results = []
        
        for batch_size in config.batch_sizes:
            for seq_len in config.sequence_lengths:
                for mode in config.generation_modes:
                    result = await self._benchmark_single_mode(
                        config, batch_size, seq_len, mode
                    )
                    results.append(result)
        
        return results
    
    async def benchmark_mode_transitions(self, 
                                       config: BenchmarkConfiguration) -> List[TTSBenchmarkResult]:
        """Benchmark generation mode transition performance."""
        logger.info("Benchmarking mode transitions")
        
        results = []
        
        # Test transitions between all mode pairs
        mode_transitions = [
            (GenerationMode.TEXT, GenerationMode.AUDIO_INIT),
            (GenerationMode.AUDIO_INIT, GenerationMode.AUDIO_IN_PROGRESS),
            (GenerationMode.AUDIO_IN_PROGRESS, GenerationMode.TEXT)
        ]
        
        for batch_size in config.batch_sizes:
            for seq_len in config.sequence_lengths:
                for from_mode, to_mode in mode_transitions:
                    result = await self._benchmark_mode_transition(
                        config, batch_size, seq_len, from_mode, to_mode
                    )
                    results.append(result)
        
        return results
    
    async def _benchmark_single_mode(self, 
                                   config: BenchmarkConfiguration,
                                   batch_size: int,
                                   seq_len: int,
                                   mode: GenerationMode) -> TTSBenchmarkResult:
        """Benchmark performance for a single generation mode."""
        
        benchmark_id = f"generation_mode_{mode.value}_{batch_size}_{seq_len}"
        
        result = TTSBenchmarkResult(
            benchmark_id=benchmark_id,
            benchmark_type=BenchmarkType.GENERATION_MODES,
            architecture_type=ArchitectureType.UNIFIED,
            config=config,
            test_parameters={
                'batch_size': batch_size,
                'sequence_length': seq_len,
                'generation_mode': mode.value
            }
        )
        
        # Warmup
        for _ in range(config.warmup_runs):
            await self._execute_with_mode(batch_size, seq_len, mode, warmup=True)
        
        # Benchmark runs
        measurements = []
        mode_latencies = []
        
        for run_idx in range(config.num_runs):
            try:
                start_time = time.time()
                
                # Execute with specific mode
                mode_start = time.time()
                await self._execute_with_mode(batch_size, seq_len, mode, warmup=False)
                mode_end = time.time()
                
                end_time = time.time()
                
                total_latency = (end_time - start_time) * 1000
                mode_latency = (mode_end - mode_start) * 1000
                
                measurements.append(total_latency)
                mode_latencies.append(mode_latency)
                
            except Exception as e:
                logger.error(f"Mode benchmark run {run_idx} failed: {e}")
                result.failed_runs += 1
        
        result.raw_measurements = measurements
        result.compute_statistics()
        
        # Add mode-specific metrics
        if mode_latencies:
            result.mode_transition_latency_ms = np.mean(mode_latencies)
        
        return result
    
    async def _benchmark_mode_transition(self, 
                                       config: BenchmarkConfiguration,
                                       batch_size: int,
                                       seq_len: int,
                                       from_mode: GenerationMode,
                                       to_mode: GenerationMode) -> TTSBenchmarkResult:
        """Benchmark transition between two generation modes."""
        
        benchmark_id = f"mode_transition_{from_mode.value}_to_{to_mode.value}_{batch_size}_{seq_len}"
        
        result = TTSBenchmarkResult(
            benchmark_id=benchmark_id,
            benchmark_type=BenchmarkType.GENERATION_MODES,
            architecture_type=ArchitectureType.UNIFIED,
            config=config,
            test_parameters={
                'batch_size': batch_size,
                'sequence_length': seq_len,
                'from_mode': from_mode.value,
                'to_mode': to_mode.value
            }
        )
        
        # Benchmark runs
        measurements = []
        success_count = 0
        
        for run_idx in range(config.num_runs):
            try:
                start_time = time.time()
                
                # Execute transition
                success = await self._execute_mode_transition(
                    batch_size, seq_len, from_mode, to_mode
                )
                
                end_time = time.time()
                
                if success:
                    success_count += 1
                    transition_latency = (end_time - start_time) * 1000
                    measurements.append(transition_latency)
                
            except Exception as e:
                logger.error(f"Mode transition benchmark run {run_idx} failed: {e}")
                result.failed_runs += 1
        
        result.raw_measurements = measurements
        result.compute_statistics()
        
        # Add transition-specific metrics
        result.mode_transition_success_rate = success_count / config.num_runs
        
        return result
    
    async def _execute_with_mode(self, 
                               batch_size: int, 
                               seq_len: int, 
                               mode: GenerationMode,
                               warmup: bool = False) -> None:
        """Execute inference with specific generation mode."""
        
        # Create sample input
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len), dtype=torch.int32)
        
        # Set generation mode
        self.mode_manager.set_generation_mode(mode)
        
        # Create mode-specific inputs
        if mode == GenerationMode.AUDIO_INIT:
            # Add audio initialization tokens
            audio_init_tokens = torch.tensor([[self.config.audio_in_token_idx]] * batch_size, 
                                           dtype=torch.int32, device=input_ids.device)
            input_ids = torch.cat([audio_init_tokens, input_ids], dim=1)
        elif mode == GenerationMode.AUDIO_IN_PROGRESS:
            # Add audio continuation tokens
            audio_cont_tokens = torch.tensor([[self.config.audio_stream_bos_id]] * batch_size,
                                           dtype=torch.int32, device=input_ids.device)
            input_ids = torch.cat([audio_cont_tokens, input_ids], dim=1)
        
        # Execute inference (simplified for benchmarking)
        if not warmup:
            # Simulate model forward pass
            await asyncio.sleep(0.001)  # Minimal delay for simulation
    
    async def _execute_mode_transition(self, 
                                     batch_size: int, 
                                     seq_len: int,
                                     from_mode: GenerationMode,
                                     to_mode: GenerationMode) -> bool:
        """Execute generation mode transition."""
        
        try:
            # Set initial mode
            self.mode_manager.set_generation_mode(from_mode)
            
            # Create transition input
            input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len), dtype=torch.int32)
            
            # Simulate transition
            await asyncio.sleep(0.0005)  # Minimal transition delay
            
            # Set target mode
            self.mode_manager.set_generation_mode(to_mode)
            
            # Execute with new mode
            await self._execute_with_mode(batch_size, seq_len, to_mode, warmup=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Mode transition failed: {e}")
            return False


class DelayPatternBenchmark:
    """Benchmark for RVQ delay pattern performance and efficiency."""
    
    def __init__(self, config):
        self.config = config
        self.delay_provider = DelayPatternProvider(
            strategy='linear',
            stride=1,
            max_delay=config.audio_delay_pattern_max_delay
        )
        self.audio_utils = AudioTokenUtils(
            num_codebooks=config.audio_num_codebooks,
            pad_token_id=config.pad_token_id
        )
    
    async def benchmark_delay_patterns(self, 
                                     config: BenchmarkConfiguration) -> List[TTSBenchmarkResult]:
        """Benchmark delay pattern generation and application performance."""
        logger.info("Benchmarking delay patterns")
        
        results = []
        
        for batch_size in config.batch_sizes:
            for seq_len in config.sequence_lengths:
                for strategy in config.delay_pattern_strategies:
                    result = await self._benchmark_delay_strategy(
                        config, batch_size, seq_len, strategy
                    )
                    results.append(result)
        
        return results
    
    async def benchmark_delay_efficiency(self, 
                                       config: BenchmarkConfiguration) -> List[TTSBenchmarkResult]:
        """Benchmark delay pattern efficiency for streaming."""
        logger.info("Benchmarking delay pattern efficiency")
        
        results = []
        
        for num_codebooks in [4, 8, 12, 16]:
            for seq_len in config.sequence_lengths:
                result = await self._benchmark_streaming_efficiency(
                    config, num_codebooks, seq_len
                )
                results.append(result)
        
        return results
    
    async def _benchmark_delay_strategy(self, 
                                      config: BenchmarkConfiguration,
                                      batch_size: int,
                                      seq_len: int,
                                      strategy: str) -> TTSBenchmarkResult:
        """Benchmark specific delay pattern strategy."""
        
        benchmark_id = f"delay_pattern_{strategy}_{batch_size}_{seq_len}"
        
        result = TTSBenchmarkResult(
            benchmark_id=benchmark_id,
            benchmark_type=BenchmarkType.DELAY_PATTERNS,
            architecture_type=ArchitectureType.UNIFIED,
            config=config,
            test_parameters={
                'batch_size': batch_size,
                'sequence_length': seq_len,
                'delay_strategy': strategy
            }
        )
        
        # Create delay provider for this strategy
        provider = DelayPatternProvider(
            strategy=strategy,
            stride=1,
            max_delay=config.max_delay if hasattr(config, 'max_delay') else None
        )
        
        # Benchmark runs
        measurements = []
        efficiency_scores = []
        
        for run_idx in range(config.num_runs):
            try:
                start_time = time.time()
                
                # Generate delay pattern
                pattern = provider.generate_delay_pattern(
                    n_codebooks=self.config.audio_num_codebooks,
                    sequence_length=seq_len
                )
                
                # Create sample tokens
                tokens = torch.randint(0, 1000, 
                                     (self.config.audio_num_codebooks, seq_len), 
                                     dtype=torch.int32)
                
                # Apply delay pattern
                delayed_tokens = provider.apply_delay_pattern(tokens, pattern)
                
                # Reverse delay pattern
                synchronized_tokens = provider.reverse_delay_pattern(
                    delayed_tokens, pattern, original_length=seq_len
                )
                
                end_time = time.time()
                
                latency = (end_time - start_time) * 1000
                measurements.append(latency)
                
                # Calculate efficiency score (lower delay overhead is better)
                delay_overhead = delayed_tokens.shape[1] - seq_len
                efficiency = 1.0 - (delay_overhead / seq_len)
                efficiency_scores.append(efficiency)
                
            except Exception as e:
                logger.error(f"Delay pattern benchmark run {run_idx} failed: {e}")
                result.failed_runs += 1
        
        result.raw_measurements = measurements
        result.compute_statistics()
        
        # Add delay-specific metrics
        if efficiency_scores:
            result.delay_pattern_efficiency = np.mean(efficiency_scores)
        
        return result
    
    async def _benchmark_streaming_efficiency(self, 
                                            config: BenchmarkConfiguration,
                                            num_codebooks: int,
                                            seq_len: int) -> TTSBenchmarkResult:
        """Benchmark delay pattern efficiency for streaming scenarios."""
        
        benchmark_id = f"streaming_delay_efficiency_{num_codebooks}_{seq_len}"
        
        result = TTSBenchmarkResult(
            benchmark_id=benchmark_id,
            benchmark_type=BenchmarkType.DELAY_PATTERNS,
            architecture_type=ArchitectureType.UNIFIED,
            config=config,
            test_parameters={
                'num_codebooks': num_codebooks,
                'sequence_length': seq_len,
                'streaming_optimized': True
            }
        )
        
        # Benchmark runs
        measurements = []
        sync_scores = []
        
        for run_idx in range(config.num_runs):
            try:
                start_time = time.time()
                
                # Simulate streaming chunks
                chunk_size = 64
                num_chunks = seq_len // chunk_size
                
                total_delay_overhead = 0
                
                for chunk_idx in range(num_chunks):
                    chunk_start = chunk_idx * chunk_size
                    chunk_end = min((chunk_idx + 1) * chunk_size, seq_len)
                    
                    # Generate delay pattern for chunk
                    pattern = self.delay_provider.generate_delay_pattern(
                        n_codebooks=num_codebooks,
                        sequence_length=chunk_size
                    )
                    
                    # Create chunk tokens
                    chunk_tokens = torch.randint(0, 1000, 
                                               (num_codebooks, chunk_size), 
                                               dtype=torch.int32)
                    
                    # Apply delay pattern
                    delayed_chunk = self.delay_provider.apply_delay_pattern(
                        chunk_tokens, pattern
                    )
                    
                    total_delay_overhead += (delayed_chunk.shape[1] - chunk_size)
                
                end_time = time.time()
                
                latency = (end_time - start_time) * 1000
                measurements.append(latency)
                
                # Calculate synchronization score
                avg_delay_overhead = total_delay_overhead / num_chunks
                sync_score = 1.0 - (avg_delay_overhead / chunk_size)
                sync_scores.append(sync_score)
                
            except Exception as e:
                logger.error(f"Streaming efficiency benchmark run {run_idx} failed: {e}")
                result.failed_runs += 1
        
        result.raw_measurements = measurements
        result.compute_statistics()
        
        # Add streaming-specific metrics
        if sync_scores:
            result.codebook_synchronization_score = np.mean(sync_scores)
        
        return result


class StreamingBenchmark:
    """Benchmark for real-time streaming TTS performance."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
    async def benchmark_streaming_latency(self, 
                                        config: BenchmarkConfiguration) -> List[TTSBenchmarkResult]:
        """Benchmark streaming latency for real-time TTS."""
        logger.info("Benchmarking streaming latency")
        
        results = []
        
        for batch_size in config.batch_sizes:
            for chunk_size in config.streaming_chunk_sizes:
                result = await self._benchmark_chunk_streaming(
                    config, batch_size, chunk_size
                )
                results.append(result)
        
        return results
    
    async def benchmark_streaming_quality(self, 
                                        config: BenchmarkConfiguration) -> List[TTSBenchmarkResult]:
        """Benchmark streaming quality and consistency."""
        logger.info("Benchmarking streaming quality")
        
        results = []
        
        for chunk_size in config.streaming_chunk_sizes:
            result = await self._benchmark_streaming_consistency(
                config, chunk_size
            )
            results.append(result)
        
        return results
    
    async def _benchmark_chunk_streaming(self, 
                                       config: BenchmarkConfiguration,
                                       batch_size: int,
                                       chunk_size: int) -> TTSBenchmarkResult:
        """Benchmark streaming performance for specific chunk size."""
        
        benchmark_id = f"streaming_chunk_{batch_size}_{chunk_size}"
        
        result = TTSBenchmarkResult(
            benchmark_id=benchmark_id,
            benchmark_type=BenchmarkType.STREAMING,
            architecture_type=ArchitectureType.UNIFIED,
            config=config,
            test_parameters={
                'batch_size': batch_size,
                'chunk_size': chunk_size,
                'streaming_mode': True
            }
        )
        
        # Benchmark runs
        measurements = []
        
        for run_idx in range(config.num_runs):
            try:
                start_time = time.time()
                
                # Simulate streaming generation
                total_chunks = 10  # Simulate 10 chunks
                
                for chunk_idx in range(total_chunks):
                    # Create chunk input
                    chunk_input = torch.randint(0, self.config.vocab_size, 
                                              (batch_size, chunk_size), dtype=torch.int32)
                    
                    # Process chunk (simplified for benchmarking)
                    await self._process_streaming_chunk(chunk_input, chunk_idx)
                    
                    # Simulate real-time constraint (sub-100ms per chunk)
                    await asyncio.sleep(0.001)  # Minimal processing delay
                
                end_time = time.time()
                
                total_latency = (end_time - start_time) * 1000
                avg_chunk_latency = total_latency / total_chunks
                
                measurements.append(avg_chunk_latency)
                
            except Exception as e:
                logger.error(f"Streaming chunk benchmark run {run_idx} failed: {e}")
                result.failed_runs += 1
        
        result.raw_measurements = measurements
        result.compute_statistics()
        
        # Add streaming-specific metrics
        result.streaming_latency_ms = result.mean
        
        return result
    
    async def _benchmark_streaming_consistency(self, 
                                             config: BenchmarkConfiguration,
                                             chunk_size: int) -> TTSBenchmarkResult:
        """Benchmark streaming consistency across chunks."""
        
        benchmark_id = f"streaming_consistency_{chunk_size}"
        
        result = TTSBenchmarkResult(
            benchmark_id=benchmark_id,
            benchmark_type=BenchmarkType.STREAMING,
            architecture_type=ArchitectureType.UNIFIED,
            config=config,
            test_parameters={
                'chunk_size': chunk_size,
                'consistency_test': True
            }
        )
        
        # Benchmark runs
        measurements = []
        quality_scores = []
        
        for run_idx in range(config.num_runs):
            try:
                start_time = time.time()
                
                # Generate multiple chunks and measure consistency
                chunks = []
                total_chunks = 8
                
                for chunk_idx in range(total_chunks):
                    chunk_input = torch.randint(0, self.config.vocab_size, 
                                              (1, chunk_size), dtype=torch.int32)
                    
                    # Process chunk
                    chunk_output = await self._process_streaming_chunk(chunk_input, chunk_idx)
                    chunks.append(chunk_output)
                
                end_time = time.time()
                
                latency = (end_time - start_time) * 1000
                measurements.append(latency)
                
                # Calculate consistency score (simplified)
                if len(chunks) > 1:
                    # Measure variance in chunk processing times
                    chunk_variances = []
                    for i in range(len(chunks) - 1):
                        # Simplified consistency metric
                        consistency = 1.0 - abs(len(chunks[i]) - len(chunks[i + 1])) / max(len(chunks[i]), len(chunks[i + 1]))
                        chunk_variances.append(consistency)
                    
                    avg_consistency = np.mean(chunk_variances)
                    quality_scores.append(avg_consistency)
                
            except Exception as e:
                logger.error(f"Streaming consistency benchmark run {run_idx} failed: {e}")
                result.failed_runs += 1
        
        result.raw_measurements = measurements
        result.compute_statistics()
        
        # Add quality metrics
        if quality_scores:
            result.audio_quality_score = np.mean(quality_scores)
        
        return result
    
    async def _process_streaming_chunk(self, chunk_input: torch.Tensor, chunk_idx: int) -> torch.Tensor:
        """Process a single streaming chunk (simplified for benchmarking)."""
        
        # Simulate chunk processing with realistic delay
        processing_delay = 0.001 + (chunk_idx * 0.0001)  # Slight increase per chunk
        await asyncio.sleep(processing_delay)
        
        # Return processed chunk (simplified)
        return chunk_input * 2  # Dummy processing


class DualFFNBenchmark:
    """Benchmark for DualFFN architecture performance."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
    async def benchmark_dualffn_routing(self, 
                                      config: BenchmarkConfiguration) -> List[TTSBenchmarkResult]:
        """Benchmark DualFFN routing efficiency."""
        logger.info("Benchmarking DualFFN routing")
        
        results = []
        
        for batch_size in config.batch_sizes:
            for seq_len in config.sequence_lengths:
                for audio_ratio in config.audio_token_ratios:
                    result = await self._benchmark_routing_efficiency(
                        config, batch_size, seq_len, audio_ratio
                    )
                    results.append(result)
        
        return results
    
    async def benchmark_path_specialization(self, 
                                          config: BenchmarkConfiguration) -> List[TTSBenchmarkResult]:
        """Benchmark DualFFN path specialization performance."""
        logger.info("Benchmarking DualFFN path specialization")
        
        results = []
        
        for batch_size in config.batch_sizes:
            for seq_len in config.sequence_lengths:
                # Test pure audio path
                result = await self._benchmark_path_performance(
                    config, batch_size, seq_len, 'audio', 1.0
                )
                results.append(result)
                
                # Test pure text path
                result = await self._benchmark_path_performance(
                    config, batch_size, seq_len, 'text', 0.0
                )
                results.append(result)
                
                # Test mixed path
                result = await self._benchmark_path_performance(
                    config, batch_size, seq_len, 'mixed', 0.5
                )
                results.append(result)
        
        return results
    
    async def _benchmark_routing_efficiency(self, 
                                          config: BenchmarkConfiguration,
                                          batch_size: int,
                                          seq_len: int,
                                          audio_ratio: float) -> TTSBenchmarkResult:
        """Benchmark DualFFN routing efficiency for specific audio/text ratio."""
        
        benchmark_id = f"dualffn_routing_{batch_size}_{seq_len}_{audio_ratio:.1f}"
        
        result = TTSBenchmarkResult(
            benchmark_id=benchmark_id,
            benchmark_type=BenchmarkType.DUALFFN,
            architecture_type=ArchitectureType.UNIFIED,
            config=config,
            test_parameters={
                'batch_size': batch_size,
                'sequence_length': seq_len,
                'audio_ratio': audio_ratio
            }
        )
        
        # Benchmark runs
        measurements = []
        audio_utilizations = []
        text_utilizations = []
        
        for run_idx in range(config.num_runs):
            try:
                start_time = time.time()
                
                # Create input with specific audio/text ratio
                input_ids = torch.randint(0, self.config.vocab_size, 
                                        (batch_size, seq_len), dtype=torch.int32)
                
                # Create audio/text mask based on ratio
                audio_tokens = int(seq_len * audio_ratio)
                audio_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
                audio_mask[:, :audio_tokens] = True
                
                # Execute DualFFN routing (simplified for benchmarking)
                await self._execute_dualffn_routing(input_ids, audio_mask)
                
                end_time = time.time()
                
                latency = (end_time - start_time) * 1000
                measurements.append(latency)
                
                # Track path utilization
                audio_utilizations.append(audio_ratio)
                text_utilizations.append(1.0 - audio_ratio)
                
            except Exception as e:
                logger.error(f"DualFFN routing benchmark run {run_idx} failed: {e}")
                result.failed_runs += 1
        
        result.raw_measurements = measurements
        result.compute_statistics()
        
        # Add DualFFN-specific metrics
        if audio_utilizations:
            result.audio_path_utilization = np.mean(audio_utilizations)
            result.text_path_utilization = np.mean(text_utilizations)
            result.dualffn_routing_efficiency = 1.0 - abs(result.audio_path_utilization - 0.5)  # Efficiency based on balance
        
        return result
    
    async def _benchmark_path_performance(self, 
                                        config: BenchmarkConfiguration,
                                        batch_size: int,
                                        seq_len: int,
                                        path_type: str,
                                        audio_ratio: float) -> TTSBenchmarkResult:
        """Benchmark performance of specific DualFFN path."""
        
        benchmark_id = f"dualffn_path_{path_type}_{batch_size}_{seq_len}"
        
        result = TTSBenchmarkResult(
            benchmark_id=benchmark_id,
            benchmark_type=BenchmarkType.DUALFFN,
            architecture_type=ArchitectureType.UNIFIED,
            config=config,
            test_parameters={
                'batch_size': batch_size,
                'sequence_length': seq_len,
                'path_type': path_type,
                'audio_ratio': audio_ratio
            }
        )
        
        # Benchmark runs
        measurements = []
        
        for run_idx in range(config.num_runs):
            try:
                start_time = time.time()
                
                # Create input optimized for specific path
                input_ids = torch.randint(0, self.config.vocab_size, 
                                        (batch_size, seq_len), dtype=torch.int32)
                
                # Create path-specific mask
                if path_type == 'audio':
                    audio_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
                elif path_type == 'text':
                    audio_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
                else:  # mixed
                    audio_tokens = int(seq_len * audio_ratio)
                    audio_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
                    audio_mask[:, :audio_tokens] = True
                
                # Execute path-specific processing
                await self._execute_path_specific_processing(input_ids, audio_mask, path_type)
                
                end_time = time.time()
                
                latency = (end_time - start_time) * 1000
                measurements.append(latency)
                
            except Exception as e:
                logger.error(f"DualFFN path benchmark run {run_idx} failed: {e}")
                result.failed_runs += 1
        
        result.raw_measurements = measurements
        result.compute_statistics()
        
        return result
    
    async def _execute_dualffn_routing(self, input_ids: torch.Tensor, audio_mask: torch.Tensor) -> None:
        """Execute DualFFN routing (simplified for benchmarking)."""
        
        # Simulate routing decision and processing
        audio_count = audio_mask.sum().item()
        total_tokens = input_ids.numel()
        audio_ratio = audio_count / total_tokens
        
        # Simulate processing delay based on routing complexity
        routing_delay = 0.0005 + (abs(audio_ratio - 0.5) * 0.0002)  # More complex routing for balanced loads
        await asyncio.sleep(routing_delay)
    
    async def _execute_path_specific_processing(self, 
                                             input_ids: torch.Tensor, 
                                             audio_mask: torch.Tensor,
                                             path_type: str) -> None:
        """Execute path-specific processing (simplified for benchmarking)."""
        
        # Simulate path-specific processing
        if path_type == 'audio':
            # Audio path: simulate audio token processing
            await asyncio.sleep(0.0008)
        elif path_type == 'text':
            # Text path: simulate text token processing
            await asyncio.sleep(0.0006)
        else:  # mixed
            # Mixed path: simulate combined processing
            await asyncio.sleep(0.0007)


# Main TTS benchmarking orchestrator

class TTSBenchmarkOrchestrator:
    """Orchestrator for all TTS-specific benchmarks."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Initialize benchmark modules
        self.generation_mode_benchmark = GenerationModeBenchmark(model, config)
        self.delay_pattern_benchmark = DelayPatternBenchmark(config)
        self.streaming_benchmark = StreamingBenchmark(model, config)
        self.dualffn_benchmark = DualFFNBenchmark(model, config)
    
    async def run_tts_benchmarks(self, 
                               config: BenchmarkConfiguration) -> Dict[str, List[TTSBenchmarkResult]]:
        """Run all TTS-specific benchmarks."""
        logger.info("Running comprehensive TTS benchmarks")
        
        results = {}
        
        # Generation mode benchmarks
        if config.benchmark_type == BenchmarkType.GENERATION_MODES:
            results['generation_modes'] = await self.generation_mode_benchmark.benchmark_generation_modes(config)
            results['mode_transitions'] = await self.generation_mode_benchmark.benchmark_mode_transitions(config)
        
        # Delay pattern benchmarks
        elif config.benchmark_type == BenchmarkType.DELAY_PATTERNS:
            results['delay_patterns'] = await self.delay_pattern_benchmark.benchmark_delay_patterns(config)
            results['delay_efficiency'] = await self.delay_pattern_benchmark.benchmark_delay_efficiency(config)
        
        # Streaming benchmarks
        elif config.benchmark_type == BenchmarkType.STREAMING:
            results['streaming_latency'] = await self.streaming_benchmark.benchmark_streaming_latency(config)
            results['streaming_quality'] = await self.streaming_benchmark.benchmark_streaming_quality(config)
        
        # DualFFN benchmarks
        elif config.benchmark_type == BenchmarkType.DUALFFN:
            results['dualffn_routing'] = await self.dualffn_benchmark.benchmark_dualffn_routing(config)
            results['path_specialization'] = await self.dualffn_benchmark.benchmark_path_specialization(config)
        
        return results
    
    def generate_tts_report(self, 
                          results: Dict[str, List[TTSBenchmarkResult]]) -> Dict[str, Any]:
        """Generate comprehensive TTS benchmark report."""
        
        report = {
            'timestamp': time.time(),
            'tts_benchmark_summary': {},
            'performance_highlights': [],
            'areas_for_improvement': [],
            'recommendations': []
        }
        
        # Analyze each benchmark category
        for category, category_results in results.items():
            if not category_results:
                continue
            
            category_summary = self._analyze_category_results(category, category_results)
            report['tts_benchmark_summary'][category] = category_summary
            
            # Extract highlights and concerns
            highlights, concerns = self._extract_insights(category, category_results)
            report['performance_highlights'].extend(highlights)
            report['areas_for_improvement'].extend(concerns)
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(results)
        
        return report
    
    def _analyze_category_results(self, 
                                category: str, 
                                results: List[TTSBenchmarkResult]) -> Dict[str, Any]:
        """Analyze results for a specific benchmark category."""
        
        if not results:
            return {}
        
        # Aggregate metrics
        latencies = []
        for result in results:
            latencies.extend(result.raw_measurements)
        
        if not latencies:
            return {}
        
        return {
            'num_results': len(results),
            'total_measurements': len(latencies),
            'mean_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'std_dev_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'percentile_95_ms': np.percentile(latencies, 95),
            'percentile_99_ms': np.percentile(latencies, 99)
        }
    
    def _extract_insights(self, 
                        category: str, 
                        results: List[TTSBenchmarkResult]) -> Tuple[List[str], List[str]]:
        """Extract performance highlights and concerns."""
        
        highlights = []
        concerns = []
        
        for result in results:
            if result.mean < 10:  # Very fast (< 10ms)
                highlights.append(f"Excellent performance in {category}: {result.mean:.2f}ms")
            elif result.mean > 100:  # Slow (> 100ms)
                concerns.append(f"Performance concern in {category}: {result.mean:.2f}ms")
            
            if result.std_dev / result.mean > 0.2:  # High variability
                concerns.append(f"High variability in {category}: {result.std_dev/result.mean:.2f} CV")
        
        return highlights, concerns
    
    def _generate_recommendations(self, results: Dict[str, List[TTSBenchmarkResult]]) -> List[str]:
        """Generate recommendations based on benchmark results."""
        
        recommendations = []
        
        # Check streaming performance
        if 'streaming_latency' in results:
            streaming_results = results['streaming_latency']
            avg_streaming_latency = np.mean([r.mean for r in streaming_results])
            
            if avg_streaming_latency > 50:
                recommendations.append("Consider optimizing streaming pipeline for lower latency")
            else:
                recommendations.append("Streaming latency is within acceptable range")
        
        # Check delay pattern efficiency
        if 'delay_efficiency' in results:
            delay_results = results['delay_efficiency']
            avg_efficiency = np.mean([r.delay_pattern_efficiency for r in delay_results if hasattr(r, 'delay_pattern_efficiency')])
            
            if avg_efficiency < 0.8:
                recommendations.append("Investigate delay pattern optimization opportunities")
        
        # Check DualFFN routing
        if 'dualffn_routing' in results:
            routing_results = results['dualffn_routing']
            avg_routing_eff = np.mean([r.dualffn_routing_efficiency for r in routing_results if hasattr(r, 'dualffn_routing_efficiency')])
            
            if avg_routing_eff < 0.9:
                recommendations.append("Consider DualFFN routing optimization")
        
        return recommendations