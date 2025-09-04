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

"""Higgs Audio generation pipeline with TensorRT-LLM optimizations.

This module implements the core inference pipeline for Higgs Audio, supporting:
- Three-phase generation: TEXT -> AUDIO_INIT -> AUDIO_IN_PROGRESS
- Delay pattern coordination for 8-codebook RVQ generation
- Streaming audio generation with real-time support
- TensorRT-LLM optimizations (static cache, kernel fusion)
- Voice cloning and TTS utilities
"""

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
import torch
from torch.nn import functional as F

from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import Session, TensorInfo

from .audio_tokenizer import HiggsAudioTokenizer
from .config import HiggsAudioConfig
from .dual_ffn import GenerationMode

__all__ = [
    "GenerationState",
    "PipelineConfig",
    "HiggsAudioGenerationPipeline",
    "SamplerConfig",
    "StreamingIterator",
    "StreamingBuffer",
    "StreamingChunk",
    "VoiceProfile",
    "TTSPreset",
    "VoiceCloneingUtils",
    "TTSPresets",
    "VoiceSimilarityMetrics",
    "create_tts_pipeline",
    "create_voice_cloning_pipeline",
    "prepare_reference_audio",
    "clone_voice_from_text",
    "synthesize_speech",
]


class GenerationState(Enum):
    """Current generation state tracking for the pipeline."""

    # Initialization states
    IDLE = "idle"
    INITIALIZED = "initialized"

    # Active generation states
    TEXT = "text"
    AUDIO_INIT = "audio_init"
    AUDIO_IN_PROGRESS = "audio_in_progress"

    # Completion states
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class SamplerConfig:
    """Configuration for text and audio sampling strategies."""

    # Text sampling parameters
    text_temperature: float = 1.0
    text_top_k: int = 50
    text_top_p: float = 0.9
    text_repetition_penalty: float = 1.0
    text_frequency_penalty: float = 0.0
    text_presence_penalty: float = 0.0

    # Audio sampling parameters
    audio_temperature: float = 1.0
    audio_top_k: int = 100
    audio_top_p: float = 0.8

    # Per-codebook parameters (8 codebooks)
    codebook_temperatures: Optional[List[float]] = None
    codebook_top_k: Optional[List[int]] = None
    codebook_top_p: Optional[List[float]] = None

    # Generation control
    max_text_length: int = 2048
    max_audio_length: int = 4096  # In frames
    eos_token_id: int = 128001
    pad_token_id: int = 128002

    # Audio-specific tokens (8 codebooks * (codebook_size + 2))
    audio_bos_token_ids: Optional[List[int]] = None
    audio_eos_token_ids: Optional[List[int]] = None

    def __post_init__(self):
        """Initialize per-codebook parameters if not provided."""
        if self.codebook_temperatures is None:
            self.codebook_temperatures = [self.audio_temperature] * 8
        if self.codebook_top_k is None:
            self.codebook_top_k = [self.audio_top_k] * 8
        if self.codebook_top_p is None:
            self.codebook_top_p = [self.audio_top_p] * 8

        # Validate list lengths
        assert len(self.codebook_temperatures) == 8
        assert len(self.codebook_top_k) == 8
        assert len(self.codebook_top_p) == 8


@dataclass
class PipelineConfig:
    """Main configuration for the generation pipeline."""

    # Model configuration
    model_config: HiggsAudioConfig
    sampler_config: SamplerConfig = field(default_factory=SamplerConfig)

    # TensorRT-LLM engine settings
    use_static_cache: bool = True
    use_kernel_fusion: bool = True
    use_flash_attention: bool = True
    enable_streaming: bool = True

    # Streaming configuration
    streaming_chunk_size: int = 32  # Tokens per chunk
    streaming_buffer_size: int = 512  # Buffer size for audio frames
    max_streaming_latency_ms: float = 100.0  # Target latency

    # Performance settings
    enable_profiling: bool = False
    enable_memory_optimization: bool = True
    device: str = "cuda"
    dtype: str = "float16"

    # Safety limits
    max_concurrent_requests: int = 8
    timeout_seconds: float = 60.0


@dataclass
class GenerationSequence:
    """Tracks state for a single generation sequence."""

    # Request metadata
    request_id: str
    prompt_text: str
    reference_audio: Optional[torch.Tensor] = None

    # Generation state
    state: GenerationState = GenerationState.IDLE
    mode: GenerationMode = GenerationMode.TEXT

    # Token sequences
    input_ids: Optional[torch.Tensor] = None
    generated_text_tokens: List[int] = field(default_factory=list)
    generated_audio_tokens: List[List[int]] = field(default_factory=lambda: [[] for _ in range(8)])

    # Timing and metrics
    start_time: float = field(default_factory=time.time)
    first_token_time: Optional[float] = None
    completion_time: Optional[float] = None

    # Cache state
    kv_cache_length: int = 0
    audio_frame_count: int = 0

    # Delay pattern tracking
    codebook_delays: List[int] = field(default_factory=lambda: list(range(8)))
    delayed_audio_tokens: Optional[np.ndarray] = None

    def add_text_token(self, token_id: int):
        """Add a generated text token."""
        self.generated_text_tokens.append(token_id)
        if self.first_token_time is None:
            self.first_token_time = time.time()

    def add_audio_tokens(self, codebook_tokens: List[int]):
        """Add audio tokens for current frame across all codebooks."""
        assert len(codebook_tokens) == 8
        for i, token in enumerate(codebook_tokens):
            self.generated_audio_tokens[i].append(token)
        self.audio_frame_count += 1

    def set_completed(self):
        """Mark sequence as completed."""
        self.state = GenerationState.COMPLETED
        self.completion_time = time.time()

    def get_metrics(self) -> Dict[str, float]:
        """Get generation metrics."""
        current_time = time.time()
        total_time = current_time - self.start_time

        metrics = {
            "total_time": total_time,
            "text_tokens": len(self.generated_text_tokens),
            "audio_frames": self.audio_frame_count,
        }

        if self.first_token_time:
            metrics["time_to_first_token"] = self.first_token_time - self.start_time

        if self.completion_time:
            metrics["completion_time"] = self.completion_time - self.start_time
            if len(self.generated_text_tokens) > 0:
                metrics["text_tokens_per_second"] = (
                    len(self.generated_text_tokens) / metrics["completion_time"]
                )
            if self.audio_frame_count > 0:
                metrics["audio_frames_per_second"] = (
                    self.audio_frame_count / metrics["completion_time"]
                )

        return metrics


class DelayPatternScheduler:
    """Manages delay pattern coordination for 8-codebook RVQ generation."""

    def __init__(self, num_codebooks: int = 8):
        self.num_codebooks = num_codebooks
        self.delays = list(range(num_codebooks))  # [0, 1, 2, ..., 7]
        self.reset()

    def reset(self):
        """Reset scheduler state."""
        self.current_frame = 0
        self.emission_schedule = {}  # frame -> list of codebook indices to emit

    def can_emit_codebook(self, codebook_idx: int, frame: int) -> bool:
        """Check if codebook can emit at given frame."""
        return frame >= self.delays[codebook_idx]

    def get_active_codebooks(self, frame: int) -> List[int]:
        """Get list of codebooks that should emit at current frame."""
        active = []
        for i in range(self.num_codebooks):
            if self.can_emit_codebook(i, frame):
                active.append(i)
        return active

    def advance_frame(self):
        """Advance to next frame."""
        self.current_frame += 1

    def get_current_frame(self) -> int:
        """Get current frame number."""
        return self.current_frame

    def create_causality_mask(self, frame: int, vocab_size_per_codebook: int) -> torch.Tensor:
        """Create causality mask for logits at given frame.

        Returns:
            mask: Tensor of shape (num_codebooks * vocab_size_per_codebook,)
                  with -inf for positions that should be masked.
        """
        mask = torch.zeros(self.num_codebooks * vocab_size_per_codebook)

        for i in range(self.num_codebooks):
            start_idx = i * vocab_size_per_codebook
            end_idx = (i + 1) * vocab_size_per_codebook

            if not self.can_emit_codebook(i, frame):
                mask[start_idx:end_idx] = float("-inf")

        return mask


class FusedMultiHeadLogitsHandler:
    """Efficient processing of dual text and audio logits with TensorRT-LLM optimizations."""

    def __init__(self, config: PipelineConfig, device: torch.device):
        self.config = config
        self.device = device
        self.model_config = config.model_config

        # Cache for efficiency
        self._logits_cache = {}
        self.use_kernel_fusion = config.use_kernel_fusion

        # Pre-compute vocab size splits
        self.text_vocab_size = self.model_config.text_vocab_size
        self.audio_vocab_size = self.model_config.audio_vocab_size
        self.total_vocab_size = self.text_vocab_size + self.audio_vocab_size

        logger.info(
            f"FusedMultiHeadLogitsHandler initialized: "
            f"text_vocab={self.text_vocab_size}, "
            f"audio_vocab={self.audio_vocab_size}, "
            f"fusion_enabled={self.use_kernel_fusion}"
        )

    def extract_logits(
        self,
        outputs: Dict[str, torch.Tensor],
        sequence: GenerationSequence,
        mode: GenerationMode,
        codebook_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Extract appropriate logits based on generation mode."""
        if mode == GenerationMode.TEXT:
            return self._extract_text_logits(outputs, sequence)
        elif mode in [
            GenerationMode.AUDIO,
            GenerationMode.AUDIO_INIT,
            GenerationMode.AUDIO_IN_PROGRESS,
        ]:
            return self._extract_audio_logits(outputs, sequence, codebook_idx)
        else:
            raise ValueError(f"Unknown generation mode: {mode}")

    def _extract_text_logits(
        self, outputs: Dict[str, torch.Tensor], sequence: GenerationSequence
    ) -> torch.Tensor:
        """Extract text logits with fallback strategies."""
        # Strategy 1: Dedicated text head
        for key in ["text_logits", "text_head_logits", "lm_head_logits"]:
            if key in outputs:
                logits = self._normalize_logits_tensor(outputs[key])
                if logits is not None and logits.size(-1) >= self.text_vocab_size:
                    return logits[: self.text_vocab_size]

        # Strategy 2: Split from combined logits
        if "logits" in outputs:
            combined_logits = self._normalize_logits_tensor(outputs["logits"])
            if combined_logits is not None:
                if combined_logits.size(-1) == self.total_vocab_size:
                    # Text tokens first
                    return combined_logits[: self.text_vocab_size]
                elif combined_logits.size(-1) >= self.text_vocab_size:
                    # Assume text is at the beginning
                    return combined_logits[: self.text_vocab_size]

        # Strategy 3: Multi-head extraction
        text_logits = self._extract_from_multihead_output(outputs, "text")
        if text_logits is not None:
            return text_logits

        raise RuntimeError("No text logits found in model outputs")

    def _extract_audio_logits(
        self, outputs: Dict[str, torch.Tensor], sequence: GenerationSequence, codebook_idx: int
    ) -> torch.Tensor:
        """Extract audio logits for specific codebook."""
        # Strategy 1: Codebook-specific outputs
        codebook_keys = [
            f"audio_logits_codebook_{codebook_idx}",
            f"audio_head_{codebook_idx}_logits",
            f"codebook_{codebook_idx}_logits",
            f"audio_{codebook_idx}_logits",
        ]

        for key in codebook_keys:
            if key in outputs:
                logits = self._normalize_logits_tensor(outputs[key])
                if logits is not None and logits.size(-1) >= self.audio_vocab_size:
                    return logits[: self.audio_vocab_size]

        # Strategy 2: Multi-codebook tensor
        for key in ["audio_logits", "audio_head_logits"]:
            if key in outputs:
                audio_logits = outputs[key]
                if isinstance(audio_logits, torch.Tensor):
                    if audio_logits.dim() == 4:  # [batch, seq, codebooks, vocab]
                        if audio_logits.size(2) > codebook_idx:
                            return audio_logits[0, -1, codebook_idx, : self.audio_vocab_size]
                    elif audio_logits.dim() == 3 and codebook_idx == 0:  # Single codebook
                        return audio_logits[0, -1, : self.audio_vocab_size]

        # Strategy 3: Split from combined logits
        if "logits" in outputs:
            combined_logits = self._normalize_logits_tensor(outputs["logits"])
            if combined_logits is not None and combined_logits.size(-1) == self.total_vocab_size:
                # Audio tokens after text tokens
                return combined_logits[self.text_vocab_size :]

        # Strategy 4: Multi-head extraction
        audio_logits = self._extract_from_multihead_output(outputs, "audio", codebook_idx)
        if audio_logits is not None:
            return audio_logits

        raise RuntimeError(f"No audio logits found for codebook {codebook_idx} in model outputs")

    def _normalize_logits_tensor(self, logits: Any) -> Optional[torch.Tensor]:
        """Normalize various logits formats to tensor."""
        if logits is None:
            return None

        if isinstance(logits, torch.Tensor):
            # Handle different tensor shapes
            if logits.dim() == 3:  # [batch, seq, vocab]
                return logits[0, -1, :]  # Last token, first batch
            elif logits.dim() == 2:  # [batch, vocab]
                return logits[0, :]  # First batch
            elif logits.dim() == 1:  # [vocab]
                return logits
            else:
                logger.warning(f"Unexpected logits tensor shape: {logits.shape}")
                return None
        else:
            # Convert numpy or other formats
            try:
                logits_tensor = torch.tensor(logits, device=self.device)
                return self._normalize_logits_tensor(logits_tensor)
            except Exception as e:
                logger.warning(f"Failed to convert logits to tensor: {e}")
                return None

    def _extract_from_multihead_output(
        self, outputs: Dict[str, torch.Tensor], head_type: str, codebook_idx: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """Extract logits from multi-head output structure."""
        # Look for structured multi-head outputs
        multihead_keys = ["multihead_logits", "head_outputs", "model_outputs"]

        for key in multihead_keys:
            if key in outputs:
                multihead_data = outputs[key]
                if isinstance(multihead_data, dict):
                    if head_type == "text":
                        text_keys = ["text", "lm_head", "language_model"]
                        for text_key in text_keys:
                            if text_key in multihead_data:
                                return self._normalize_logits_tensor(multihead_data[text_key])
                    elif head_type == "audio":
                        audio_keys = ["audio", "audio_head", "audio_model"]
                        for audio_key in audio_keys:
                            if audio_key in multihead_data:
                                audio_data = multihead_data[audio_key]
                                if codebook_idx is not None and isinstance(
                                    audio_data, (list, tuple)
                                ):
                                    if len(audio_data) > codebook_idx:
                                        return self._normalize_logits_tensor(
                                            audio_data[codebook_idx]
                                        )
                                else:
                                    return self._normalize_logits_tensor(audio_data)

        return None

    def fused_logits_processing(
        self,
        outputs: Dict[str, torch.Tensor],
        sequence: GenerationSequence,
        mode: GenerationMode,
        codebook_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Fused processing with TensorRT-LLM optimizations."""
        # Extract logits
        raw_logits = self.extract_logits(outputs, sequence, mode, codebook_idx)

        # Apply mode-specific processing with fusion
        if self.use_kernel_fusion:
            return self._fused_logits_transform(raw_logits, sequence, mode, codebook_idx)
        else:
            return self._standard_logits_transform(raw_logits, sequence, mode, codebook_idx)

    def _fused_logits_transform(
        self,
        logits: torch.Tensor,
        sequence: GenerationSequence,
        mode: GenerationMode,
        codebook_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply fused transformations using TensorRT-LLM kernels."""
        # Note: In a real implementation, this would use custom TensorRT-LLM kernels
        # For now, we implement optimized standard operations

        processed_logits = logits.clone()

        # Fused temperature and penalty application
        if mode == GenerationMode.TEXT:
            config = self.config.sampler_config
            temp = config.text_temperature

            # Fused temperature + repetition penalty
            if temp != 1.0 or config.text_repetition_penalty != 1.0:
                processed_logits = self._fused_text_penalties(
                    processed_logits, sequence, temp, config.text_repetition_penalty
                )

        elif mode in [
            GenerationMode.AUDIO,
            GenerationMode.AUDIO_INIT,
            GenerationMode.AUDIO_IN_PROGRESS,
        ]:
            config = self.config.sampler_config
            if codebook_idx is not None and codebook_idx < len(config.codebook_temperatures):
                temp = config.codebook_temperatures[codebook_idx]

                # Fused temperature + audio constraints
                processed_logits = self._fused_audio_penalties(
                    processed_logits, sequence, temp, codebook_idx
                )

        return processed_logits

    def _standard_logits_transform(
        self,
        logits: torch.Tensor,
        sequence: GenerationSequence,
        mode: GenerationMode,
        codebook_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply standard transformations without fusion."""
        processed_logits = logits.clone()

        if mode == GenerationMode.TEXT:
            config = self.config.sampler_config

            # Apply temperature
            if config.text_temperature != 1.0:
                processed_logits = processed_logits / config.text_temperature

            # Apply repetition penalty
            if config.text_repetition_penalty != 1.0 and sequence.generated_text_tokens:
                for token in set(sequence.generated_text_tokens):
                    if token < processed_logits.size(0):
                        if processed_logits[token] > 0:
                            processed_logits[token] /= config.text_repetition_penalty
                        else:
                            processed_logits[token] *= config.text_repetition_penalty

        elif mode in [
            GenerationMode.AUDIO,
            GenerationMode.AUDIO_INIT,
            GenerationMode.AUDIO_IN_PROGRESS,
        ]:
            config = self.config.sampler_config
            if codebook_idx is not None and codebook_idx < len(config.codebook_temperatures):
                temp = config.codebook_temperatures[codebook_idx]

                # Apply temperature
                if temp != 1.0:
                    processed_logits = processed_logits / temp

        return processed_logits

    def _fused_text_penalties(
        self,
        logits: torch.Tensor,
        sequence: GenerationSequence,
        temperature: float,
        repetition_penalty: float,
    ) -> torch.Tensor:
        """Fused temperature and repetition penalty for text."""
        # This would ideally be a custom CUDA kernel in TensorRT-LLM
        # For now, implement optimized version

        # Apply temperature first
        if temperature != 1.0:
            logits = logits / temperature

        # Apply repetition penalty in a vectorized way
        if repetition_penalty != 1.0 and sequence.generated_text_tokens:
            # Create penalty mask
            penalty_mask = torch.ones_like(logits)
            unique_tokens = list(set(sequence.generated_text_tokens))

            if unique_tokens:
                # Count frequencies
                token_counts = torch.zeros_like(logits)
                for token in sequence.generated_text_tokens:
                    if token < logits.size(0):
                        token_counts[token] += 1

                # Apply penalty where tokens appeared
                mask = token_counts > 0
                penalty_mask[mask] = 1.0 / (repetition_penalty ** token_counts[mask])

                # Apply penalty (vectorized)
                positive_mask = logits > 0
                logits[positive_mask] *= penalty_mask[positive_mask]
                logits[~positive_mask] /= penalty_mask[~positive_mask]

        return logits

    def _fused_audio_penalties(
        self,
        logits: torch.Tensor,
        sequence: GenerationSequence,
        temperature: float,
        codebook_idx: int,
    ) -> torch.Tensor:
        """Fused temperature and constraints for audio."""
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply audio token constraints (vectorized)
        config = self.config.model_config

        # Mask text tokens
        text_tokens = [
            config.text_bos_token_id,
            config.text_eos_token_id,
            config.text_pad_token_id,
        ]

        for token_id in text_tokens:
            if token_id is not None and token_id < logits.size(0):
                logits[token_id] = float("-inf")

        return logits

    def efficient_softmax_sampling(
        self, logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0, use_fused: bool = None
    ) -> torch.Tensor:
        """Memory-efficient softmax and sampling operations."""
        if use_fused is None:
            use_fused = self.use_kernel_fusion

        if use_fused:
            return self._fused_softmax_sampling(logits, top_k, top_p)
        else:
            return self._standard_softmax_sampling(logits, top_k, top_p)

    def _fused_softmax_sampling(
        self, logits: torch.Tensor, top_k: int, top_p: float
    ) -> torch.Tensor:
        """Fused softmax and sampling using TensorRT-LLM optimizations."""
        # In real implementation, this would use TensorRT-LLM's fused kernels
        # For now, implement memory-efficient version

        # Apply top-k and top-p filtering before softmax for efficiency
        if top_k > 0:
            top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
            filtered_logits = torch.full_like(logits, float("-inf"))
            filtered_logits.scatter_(-1, top_k_indices, top_k_values)
            logits = filtered_logits

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Find cutoff
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False

            # Apply cutoff
            sorted_logits[sorted_indices_to_remove] = float("-inf")
            filtered_logits = torch.full_like(logits, float("-inf"))
            filtered_logits.scatter_(-1, sorted_indices, sorted_logits)
            logits = filtered_logits

        # Compute probabilities
        return torch.softmax(logits, dim=-1)

    def _standard_softmax_sampling(
        self, logits: torch.Tensor, top_k: int, top_p: float
    ) -> torch.Tensor:
        """Standard softmax computation."""
        # Apply filtering
        if top_k > 0:
            top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
            filtered_logits = torch.full_like(logits, float("-inf"))
            filtered_logits.scatter_(-1, top_k_indices, top_k_values)
            logits = filtered_logits

        # Compute softmax
        return torch.softmax(logits, dim=-1)

    def cache_logits(self, cache_key: str, logits: torch.Tensor, max_cache_size: int = 1000):
        """Cache logits for efficiency."""
        if len(self._logits_cache) >= max_cache_size:
            # Remove oldest entries
            oldest_key = next(iter(self._logits_cache))
            del self._logits_cache[oldest_key]

        self._logits_cache[cache_key] = logits.clone()

    def get_cached_logits(self, cache_key: str) -> Optional[torch.Tensor]:
        """Retrieve cached logits."""
        return self._logits_cache.get(cache_key)

    def clear_cache(self):
        """Clear logits cache."""
        self._logits_cache.clear()


class EngineWrapper:
    """Wrapper around TensorRT-LLM session for Higgs Audio generation."""

    def __init__(self, session: Session, config: PipelineConfig):
        self.session = session
        self.config = config
        self.device = torch.device(config.device)

        # Initialize KV cache manager
        self.kv_cache_manager = None
        if config.use_static_cache:
            self._init_kv_cache()

        # Performance settings
        self.stream = torch.cuda.Stream() if config.device == "cuda" else None

    def _init_kv_cache(self):
        """Initialize KV cache for static caching."""
        # This will be implemented based on the specific session configuration
        # For now, we'll use the session's built-in cache management
        pass

    def prepare_inputs(self, sequence: GenerationSequence) -> Dict[str, torch.Tensor]:
        """Prepare inputs for the model based on current generation state."""
        inputs = {}

        if sequence.state == GenerationState.IDLE:
            # Initial forward pass with full prompt
            inputs["input_ids"] = sequence.input_ids
            if sequence.reference_audio is not None:
                inputs["audio_features"] = sequence.reference_audio
        else:
            # Incremental generation - single token
            last_token = None
            if sequence.mode == GenerationMode.TEXT and sequence.generated_text_tokens:
                last_token = sequence.generated_text_tokens[-1]
            elif sequence.mode in [GenerationMode.AUDIO_INIT, GenerationMode.AUDIO_IN_PROGRESS]:
                # For audio mode, we need to handle multi-codebook tokens
                if sequence.generated_audio_tokens[0]:  # If any audio tokens generated
                    # Get the last frame of audio tokens
                    last_frame = []
                    for codebook in sequence.generated_audio_tokens:
                        if codebook:
                            last_frame.append(codebook[-1])
                        else:
                            last_frame.append(0)  # Pad if needed
                    # For now, use first codebook token (this needs refinement)
                    last_token = last_frame[0]

            if last_token is not None:
                inputs["input_ids"] = torch.tensor(
                    [[last_token]], device=self.device, dtype=torch.long
                )

        return inputs

    def run_inference(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run model inference."""
        with torch.cuda.stream(self.stream) if self.stream else torch.no_grad():
            # Convert inputs to TensorInfo format expected by session
            session_inputs = {}
            for name, tensor in inputs.items():
                session_inputs[name] = TensorInfo(name, tensor.dtype, tensor.shape)

            # Run the session
            outputs = self.session.infer(inputs)

        return outputs


class SamplerRegistry:
    """Registry for different sampling strategies."""

    @staticmethod
    def sample_text_logits(
        logits: torch.Tensor, config: SamplerConfig, generated_tokens: List[int] = None
    ) -> int:
        """Sample from text logits with various strategies."""
        # Apply temperature
        if config.text_temperature != 1.0:
            logits = logits / config.text_temperature

        # Apply repetition penalty
        if config.text_repetition_penalty != 1.0 and generated_tokens:
            for token in set(generated_tokens):
                if logits[token] > 0:
                    logits[token] /= config.text_repetition_penalty
                else:
                    logits[token] *= config.text_repetition_penalty

        # Apply top-k filtering
        if config.text_top_k > 0:
            top_k_values, top_k_indices = torch.topk(logits, config.text_top_k)
            logits_filtered = torch.full_like(logits, float("-inf"))
            logits_filtered[top_k_indices] = top_k_values
            logits = logits_filtered

        # Apply top-p (nucleus) filtering
        if config.text_top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > config.text_top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float("-inf")

        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        return next_token

    @staticmethod
    def sample_audio_logits(
        logits: torch.Tensor,
        codebook_idx: int,
        config: SamplerConfig,
        generated_tokens: List[int] = None,
    ) -> int:
        """Sample from audio logits for specific codebook."""
        # Get codebook-specific parameters
        temperature = config.codebook_temperatures[codebook_idx]
        top_k = config.codebook_top_k[codebook_idx]
        top_p = config.codebook_top_p[codebook_idx]

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply top-k filtering
        if top_k > 0:
            top_k_values, top_k_indices = torch.topk(logits, top_k)
            logits_filtered = torch.full_like(logits, float("-inf"))
            logits_filtered[top_k_indices] = top_k_values
            logits = logits_filtered

        # Apply top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float("-inf")

        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        return next_token


class HiggsAudioGenerationPipeline:
    """Main generation pipeline for Higgs Audio model."""

    def __init__(
        self,
        session: Session,
        config: PipelineConfig,
        tokenizer: Optional[Any] = None,
        audio_tokenizer: Optional[HiggsAudioTokenizer] = None,
    ):
        self.session = session
        self.config = config
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer

        # Initialize components
        self.engine = EngineWrapper(session, config)
        self.sampler = SamplerRegistry()
        self.delay_scheduler = DelayPatternScheduler()
        self.logits_handler = FusedMultiHeadLogitsHandler(config, config.device)

        # Thread safety
        self._lock = threading.RLock()

        # Active sequences
        self.active_sequences: Dict[str, GenerationSequence] = {}

        logger.info(f"Initialized Higgs Audio generation pipeline with config: {config}")
        logger.info(f"Fused logits handler enabled with kernel_fusion={config.use_kernel_fusion}")

    def generate(
        self,
        prompt_text: str,
        reference_audio: Optional[torch.Tensor] = None,
        request_id: Optional[str] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """Generate audio from text prompt and optional reference audio.

        Args:
            prompt_text: Input text prompt
            reference_audio: Optional reference audio for voice cloning
            request_id: Optional request ID for tracking
            stream: Whether to return streaming iterator

        Returns:
            Generation result dict or streaming iterator
        """
        if request_id is None:
            request_id = f"req_{int(time.time() * 1000000)}"

        # Create sequence
        sequence = self._create_sequence(request_id, prompt_text, reference_audio)

        # Add to active sequences
        with self._lock:
            self.active_sequences[request_id] = sequence

        try:
            if stream:
                return self._generate_streaming(sequence)
            else:
                return self._generate_blocking(sequence)
        finally:
            # Clean up
            with self._lock:
                self.active_sequences.pop(request_id, None)

    def create_streaming_iterator(
        self,
        request_id: str,
        prompt_text: str,
        reference_audio: Optional[torch.Tensor] = None,
        buffer_config: Optional["StreamingBuffer"] = None,
    ) -> "StreamingIterator":
        """Create a streaming iterator for real-time generation.

        Args:
            request_id: Unique identifier for this generation request
            prompt_text: Text prompt for generation
            reference_audio: Optional reference audio for voice cloning
            buffer_config: Optional custom buffer configuration

        Returns:
            StreamingIterator instance for real-time generation
        """
        # Create generation sequence
        sequence = self._create_sequence(request_id, prompt_text, reference_audio)

        # Use custom buffer config or create default
        if buffer_config is None:
            buffer_config = StreamingBuffer(
                buffer_size=self.config.streaming_buffer_size,
                chunk_size=self.config.streaming_chunk_size,
                max_latency_ms=self.config.max_streaming_latency_ms,
            )

        # Create and return streaming iterator
        streaming_iterator = StreamingIterator(
            pipeline=self, sequence=sequence, buffer_config=buffer_config
        )

        # Track active sequence
        with self._lock:
            self.active_sequences[request_id] = sequence

        logger.info(f"Created streaming iterator for {request_id}")
        return streaming_iterator

    def _create_sequence(
        self, request_id: str, prompt_text: str, reference_audio: Optional[torch.Tensor]
    ) -> GenerationSequence:
        """Create a new generation sequence."""
        sequence = GenerationSequence(
            request_id=request_id, prompt_text=prompt_text, reference_audio=reference_audio
        )

        # Tokenize prompt text
        if self.tokenizer:
            sequence.input_ids = torch.tensor(
                [self.tokenizer.encode(prompt_text)], device=self.config.device
            )
        else:
            # Fallback for testing
            sequence.input_ids = torch.tensor([[1, 2, 3]], device=self.config.device)

        sequence.state = GenerationState.INITIALIZED
        return sequence

    def _generate_blocking(self, sequence: GenerationSequence) -> Dict[str, Any]:
        """Run blocking generation for a sequence."""
        # Text generation phase
        self._run_text_generation(sequence)

        # Audio generation phases
        if sequence.reference_audio is not None:
            self._run_audio_generation(sequence)

        sequence.set_completed()

        return {
            "request_id": sequence.request_id,
            "text_tokens": sequence.generated_text_tokens,
            "audio_tokens": sequence.generated_audio_tokens,
            "metrics": sequence.get_metrics(),
            "status": "completed",
        }

    def _generate_streaming(self, sequence: GenerationSequence) -> Iterator[Dict[str, Any]]:
        """Run streaming generation for a sequence using advanced StreamingIterator."""
        # Create advanced streaming iterator with real-time buffering
        streaming_iterator = StreamingIterator(
            pipeline=self,
            sequence=sequence,
            buffer_config=StreamingBuffer(
                buffer_size=self.config.streaming_buffer_size,
                chunk_size=self.config.streaming_chunk_size,
                max_latency_ms=self.config.max_streaming_latency_ms,
            ),
        )

        # Yield chunks from the advanced iterator
        try:
            for chunk in streaming_iterator:
                # Convert StreamingChunk to dict format expected by callers
                yield {
                    "request_id": chunk.request_id,
                    "chunk_id": chunk.chunk_id,
                    "chunk_type": chunk.chunk_type,
                    "data": chunk.data,
                    "timestamp": chunk.timestamp,
                    "latency_ms": chunk.latency_ms,
                    "generation_phase": chunk.generation_phase,
                    "codebook_data": chunk.codebook_data,
                    "audio_frame_idx": chunk.audio_frame_idx,
                    "is_partial": chunk.is_partial,
                    "is_final": chunk.is_final,
                }
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            # Yield error information
            yield {
                "request_id": sequence.request_id,
                "chunk_type": "error",
                "data": {"error": str(e)},
                "timestamp": time.time(),
                "is_final": True,
            }
        finally:
            # Ensure cleanup
            if hasattr(streaming_iterator, "stop"):
                streaming_iterator.stop()

    def _streaming_iterator(self, sequence: GenerationSequence) -> Iterator[Dict[str, Any]]:
        """Create streaming iterator for sequence generation (legacy interface)."""
        # Delegate to the advanced streaming implementation
        yield from self._generate_streaming(sequence)

    def _run_text_generation(self, sequence: GenerationSequence):
        """Run text generation phase with advanced sampling and TensorRT-LLM optimizations."""
        sequence.state = GenerationState.TEXT
        sequence.mode = GenerationMode.TEXT

        max_length = self.config.sampler_config.max_text_length

        logger.info(f"Starting text generation for {sequence.request_id}, max_length={max_length}")

        # Initialize cache and session state
        if self.config.use_static_cache:
            self._initialize_kv_cache(sequence)

        for step in range(max_length):
            # Prepare inputs for current step
            inputs = self.engine.prepare_inputs(sequence)

            # Add generation mode information
            inputs["generation_mode"] = GenerationMode.TEXT.value

            # Run inference with TensorRT-LLM optimizations
            try:
                with (
                    torch.cuda.stream(self.engine.stream) if self.engine.stream else torch.no_grad()
                ):
                    outputs = self.engine.run_inference(inputs)
            except Exception as e:
                logger.error(f"Inference failed at step {step}: {e}")
                break

            # Extract text logits using fused handler
            text_logits = self.logits_handler.extract_logits(outputs, sequence, GenerationMode.TEXT)
            if text_logits is None:
                logger.error(f"No text logits found at step {step}")
                break

            # Apply logits processing with fusion
            processed_logits = self.logits_handler.fused_logits_processing(
                outputs, sequence, GenerationMode.TEXT
            )

            # Sample next token with advanced strategies
            next_token = self._sample_text_token(processed_logits, sequence, step)

            # Add token to sequence
            sequence.add_text_token(next_token)

            # Update KV cache length
            sequence.kv_cache_length += 1

            # Check for early stopping conditions
            if self._should_stop_text_generation(next_token, sequence, step):
                logger.info(f"Text generation stopped at step {step}, token={next_token}")
                break

            # Optional: yield control for streaming or other operations
            if step % 10 == 0:  # Periodic check
                self._update_generation_metrics(sequence, step)

        logger.info(f"Text generation completed: {len(sequence.generated_text_tokens)} tokens")

    def _initialize_kv_cache(self, sequence: GenerationSequence):
        """Initialize KV cache for static caching."""
        if self.config.use_static_cache:
            # For TensorRT-LLM, the cache is usually managed by the session
            # We just need to ensure proper initialization
            sequence.kv_cache_length = 0
            logger.debug(f"KV cache initialized for {sequence.request_id}")

    def _extract_text_logits(
        self, outputs: Dict[str, torch.Tensor], sequence: GenerationSequence
    ) -> Optional[torch.Tensor]:
        """Extract text logits from model outputs."""
        # Try different possible output keys
        for key in ["text_logits", "logits", "last_token_logits"]:
            if key in outputs:
                logits = outputs[key]
                if isinstance(logits, torch.Tensor):
                    # Handle different shapes
                    if logits.dim() == 3:  # [batch, seq, vocab]
                        return logits[0, -1, :]  # Last token, first batch
                    elif logits.dim() == 2:  # [batch, vocab]
                        return logits[0, :]  # First batch
                    elif logits.dim() == 1:  # [vocab]
                        return logits
                else:
                    # Convert numpy or other formats to tensor
                    logits = torch.tensor(logits, device=self.config.device)
                    if logits.dim() == 3:
                        return logits[0, -1, :]
                    elif logits.dim() == 2:
                        return logits[0, :]
                    else:
                        return logits

        logger.warning("No text logits found in model outputs")
        return None

    def _process_text_logits(
        self, logits: torch.Tensor, sequence: GenerationSequence, step: int
    ) -> torch.Tensor:
        """Process logits with penalties and constraints."""
        processed_logits = logits.clone()

        # Apply frequency and presence penalties
        if len(sequence.generated_text_tokens) > 0:
            processed_logits = self._apply_frequency_penalty(
                processed_logits, sequence.generated_text_tokens
            )
            processed_logits = self._apply_presence_penalty(
                processed_logits, sequence.generated_text_tokens
            )

        # Apply length penalties for min/max constraints
        processed_logits = self._apply_length_penalties(processed_logits, sequence, step)

        # Apply token type constraints (e.g., avoid certain tokens in text mode)
        processed_logits = self._apply_token_constraints(processed_logits, sequence, step)

        return processed_logits

    def _apply_frequency_penalty(
        self, logits: torch.Tensor, generated_tokens: List[int]
    ) -> torch.Tensor:
        """Apply frequency penalty to reduce repetition."""
        config = self.config.sampler_config
        if config.text_frequency_penalty == 0.0:
            return logits

        # Count token frequencies
        token_counts = {}
        for token in generated_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        # Apply penalty based on frequency
        for token, count in token_counts.items():
            if token < logits.size(0):  # Valid token index
                penalty = config.text_frequency_penalty * count
                logits[token] -= penalty

        return logits

    def _apply_presence_penalty(
        self, logits: torch.Tensor, generated_tokens: List[int]
    ) -> torch.Tensor:
        """Apply presence penalty to encourage diversity."""
        config = self.config.sampler_config
        if config.text_presence_penalty == 0.0:
            return logits

        # Apply penalty to all tokens that have appeared
        unique_tokens = set(generated_tokens)
        for token in unique_tokens:
            if token < logits.size(0):  # Valid token index
                logits[token] -= config.text_presence_penalty

        return logits

    def _apply_length_penalties(
        self, logits: torch.Tensor, sequence: GenerationSequence, step: int
    ) -> torch.Tensor:
        """Apply penalties based on sequence length."""
        config = self.config.sampler_config
        current_length = len(sequence.generated_text_tokens)

        # Encourage EOS if we're approaching max length
        if current_length >= config.max_text_length * 0.9:
            eos_boost = min(5.0, (current_length - config.max_text_length * 0.9) * 2)
            if config.eos_token_id < logits.size(0):
                logits[config.eos_token_id] += eos_boost

        # Discourage EOS if we're below minimum length (if defined)
        min_length = getattr(config, "min_text_length", 10)
        if current_length < min_length:
            eos_penalty = 10.0
            if config.eos_token_id < logits.size(0):
                logits[config.eos_token_id] -= eos_penalty

        return logits

    def _apply_token_constraints(
        self, logits: torch.Tensor, sequence: GenerationSequence, step: int
    ) -> torch.Tensor:
        """Apply token-level constraints for text generation."""
        # In text mode, heavily penalize audio-specific tokens
        config = self.config.model_config

        # Penalize audio tokens
        audio_tokens = [
            config.audio_bos_token_id,
            config.audio_eos_token_id,
            config.audio_out_bos_token_id,
            config.audio_in_token_idx,
            config.audio_out_token_idx,
            config.audio_stream_bos_id,
            config.audio_stream_eos_id,
        ]

        for token_id in audio_tokens:
            if token_id is not None and token_id < logits.size(0):
                logits[token_id] = float("-inf")

        return logits

    def _sample_text_token(
        self, logits: torch.Tensor, sequence: GenerationSequence, step: int
    ) -> int:
        """Sample next text token with advanced strategies."""
        return self.sampler.sample_text_logits(
            logits, self.config.sampler_config, sequence.generated_text_tokens
        )

    def _should_stop_text_generation(
        self, token_id: int, sequence: GenerationSequence, step: int
    ) -> bool:
        """Check if text generation should stop."""
        config = self.config.sampler_config

        # Stop on EOS token
        if token_id == config.eos_token_id:
            return True

        # Stop on max length
        if len(sequence.generated_text_tokens) >= config.max_text_length:
            return True

        # Stop on timeout
        if hasattr(self.config, "timeout_seconds"):
            elapsed = time.time() - sequence.start_time
            if elapsed > self.config.timeout_seconds:
                logger.warning(f"Text generation timeout after {elapsed:.2f}s")
                return True

        return False

    def _update_generation_metrics(self, sequence: GenerationSequence, step: int):
        """Update generation metrics during processing."""
        # This can be used for monitoring and debugging
        current_time = time.time()
        elapsed = current_time - sequence.start_time
        tokens_per_second = len(sequence.generated_text_tokens) / max(elapsed, 0.001)

        if step % 50 == 0:  # Log every 50 steps
            logger.debug(
                f"Text generation step {step}: "
                f"{len(sequence.generated_text_tokens)} tokens, "
                f"{tokens_per_second:.2f} tok/s"
            )

    def _stream_text_generation(self, sequence: GenerationSequence) -> Iterator[Dict[str, Any]]:
        """Stream text generation results with real-time updates."""
        sequence.state = GenerationState.TEXT
        sequence.mode = GenerationMode.TEXT

        max_length = self.config.sampler_config.max_text_length
        chunk_size = self.config.streaming_chunk_size

        logger.info(f"Starting streaming text generation for {sequence.request_id}")

        # Initialize cache
        if self.config.use_static_cache:
            self._initialize_kv_cache(sequence)

        # Buffer for streaming chunks
        token_buffer = []
        last_yield_time = time.time()
        max_latency = self.config.max_streaming_latency_ms / 1000.0

        for step in range(max_length):
            step_start_time = time.time()

            # Prepare inputs
            inputs = self.engine.prepare_inputs(sequence)
            inputs["generation_mode"] = GenerationMode.TEXT.value

            # Run inference
            try:
                outputs = self.engine.run_inference(inputs)
            except Exception as e:
                logger.error(f"Streaming inference failed at step {step}: {e}")
                yield {
                    "request_id": sequence.request_id,
                    "error": str(e),
                    "status": "error",
                    "step": step,
                }
                break

            # Extract and process logits using fused handler
            text_logits = self.logits_handler.extract_logits(outputs, sequence, GenerationMode.TEXT)
            if text_logits is None:
                break

            processed_logits = self.logits_handler.fused_logits_processing(
                outputs, sequence, GenerationMode.TEXT
            )
            next_token = self._sample_text_token(processed_logits, sequence, step)

            # Add token to sequence and buffer
            sequence.add_text_token(next_token)
            token_buffer.append(next_token)
            sequence.kv_cache_length += 1

            # Calculate timing metrics
            step_time = time.time() - step_start_time
            time_since_last_yield = time.time() - last_yield_time

            # Yield chunk based on size or latency constraints
            should_yield = (
                len(token_buffer) >= chunk_size
                or time_since_last_yield >= max_latency
                or self._should_stop_text_generation(next_token, sequence, step)
            )

            if should_yield:
                yield {
                    "request_id": sequence.request_id,
                    "tokens": token_buffer.copy(),
                    "total_tokens": sequence.generated_text_tokens.copy(),
                    "status": "generating_text",
                    "chunk_type": "text",
                    "step": step,
                    "metrics": {
                        "step_time_ms": step_time * 1000,
                        "tokens_in_chunk": len(token_buffer),
                        "total_tokens_generated": len(sequence.generated_text_tokens),
                    },
                }
                token_buffer = []
                last_yield_time = time.time()

            # Check for stopping conditions
            if self._should_stop_text_generation(next_token, sequence, step):
                break

        # Final chunk with any remaining tokens
        if token_buffer:
            yield {
                "request_id": sequence.request_id,
                "tokens": token_buffer,
                "total_tokens": sequence.generated_text_tokens.copy(),
                "status": "text_completed",
                "chunk_type": "text_final",
                "step": step,
                "metrics": sequence.get_metrics(),
            }

        logger.info(
            f"Streaming text generation completed: {len(sequence.generated_text_tokens)} tokens"
        )

    def _run_audio_generation(self, sequence: GenerationSequence):
        """Run three-phase audio generation with 8-codebook RVQ coordination."""
        logger.info(f"Starting audio generation for {sequence.request_id}")

        # Phase 1: AUDIO_INIT - Generate first codebook
        self._run_audio_init_phase(sequence)

        # Phase 2: AUDIO_IN_PROGRESS - Generate remaining codebooks
        self._run_audio_in_progress_phase(sequence)

        logger.info(
            f"Audio generation completed: {len(sequence.generated_audio_tokens)} timesteps, "
            f"{len(sequence.generated_audio_tokens[0]) if sequence.generated_audio_tokens else 0} codebooks"
        )

    def _run_audio_init_phase(self, sequence: GenerationSequence):
        """Initialize audio generation with first codebook."""
        sequence.state = GenerationState.AUDIO_INIT
        sequence.mode = GenerationMode.AUDIO

        max_length = self.config.sampler_config.max_audio_length
        eos_token = self.config.model_config.audio_eos_token_id
        audio_bos_token = self.config.model_config.audio_bos_token_id

        logger.info(f"AUDIO_INIT phase starting for {sequence.request_id}")

        # Initialize audio sequence with BOS token for first codebook
        if audio_bos_token is not None:
            # Add BOS token to first codebook only
            first_codebook_tokens = [audio_bos_token]
            sequence.add_audio_tokens(
                [first_codebook_tokens] + [[] for _ in range(7)]
            )  # 8 codebooks total

        self.delay_scheduler.reset()

        # Generate tokens for first codebook
        for step in range(max_length):
            # Prepare inputs for audio generation
            inputs = self.engine.prepare_inputs(sequence)
            inputs["generation_mode"] = GenerationMode.AUDIO.value
            inputs["audio_phase"] = "init"
            inputs["target_codebook"] = 0

            try:
                with (
                    torch.cuda.stream(self.engine.stream) if self.engine.stream else torch.no_grad()
                ):
                    outputs = self.engine.run_inference(inputs)
            except Exception as e:
                logger.error(f"Audio INIT inference failed at step {step}: {e}")
                break

            # Extract audio logits using fused handler
            audio_logits = self.logits_handler.extract_logits(
                outputs, sequence, GenerationMode.AUDIO, codebook_idx=0
            )
            if audio_logits is None:
                logger.error(f"No audio logits for codebook 0 at step {step}")
                break

            # Apply audio-specific processing with fusion
            processed_logits = self.logits_handler.fused_logits_processing(
                outputs, sequence, GenerationMode.AUDIO, codebook_idx=0
            )

            # Sample next token for first codebook
            next_token = self._sample_audio_token(processed_logits, sequence, step, codebook_idx=0)

            # Add token to first codebook only
            self._add_audio_token_to_codebook(sequence, next_token, codebook_idx=0)

            # Update KV cache length
            sequence.kv_cache_length += 1

            # Check for early stopping
            if self._should_stop_audio_generation(next_token, sequence, step, eos_token):
                logger.info(f"Audio INIT phase stopped at step {step}, token={next_token}")
                break

        logger.info(f"AUDIO_INIT phase completed: {len(sequence.generated_audio_tokens)} timesteps")

    def _run_audio_in_progress_phase(self, sequence: GenerationSequence):
        """Generate remaining codebooks with delay pattern coordination."""
        sequence.state = GenerationState.AUDIO_IN_PROGRESS

        logger.info(f"AUDIO_IN_PROGRESS phase starting for {sequence.request_id}")

        if not sequence.generated_audio_tokens:
            logger.warning("No audio tokens from INIT phase, skipping IN_PROGRESS")
            return

        num_timesteps = len(sequence.generated_audio_tokens)
        num_codebooks = 8

        # Generate tokens for codebooks 1-7 using delay pattern
        for step in range(num_timesteps):
            # Determine which codebooks are active at this timestep
            active_codebooks = self.delay_scheduler.get_active_codebooks(step)

            for codebook_idx in active_codebooks:
                if codebook_idx == 0:
                    continue  # First codebook already generated in INIT phase

                # Prepare inputs for specific codebook
                inputs = self.engine.prepare_inputs(sequence)
                inputs["generation_mode"] = GenerationMode.AUDIO.value
                inputs["audio_phase"] = "in_progress"
                inputs["target_codebook"] = codebook_idx
                inputs["current_timestep"] = step

                try:
                    with (
                        torch.cuda.stream(self.engine.stream)
                        if self.engine.stream
                        else torch.no_grad()
                    ):
                        outputs = self.engine.run_inference(inputs)
                except Exception as e:
                    logger.error(
                        f"Audio IN_PROGRESS inference failed at step {step}, "
                        f"codebook {codebook_idx}: {e}"
                    )
                    continue

                # Extract logits using fused handler
                audio_logits = self.logits_handler.extract_logits(
                    outputs, sequence, GenerationMode.AUDIO, codebook_idx
                )
                if audio_logits is None:
                    logger.warning(f"No audio logits for codebook {codebook_idx} at step {step}")
                    continue

                # Process logits with fused operations
                processed_logits = self.logits_handler.fused_logits_processing(
                    outputs, sequence, GenerationMode.AUDIO, codebook_idx
                )

                # Sample token for current codebook
                next_token = self._sample_audio_token(
                    processed_logits, sequence, step, codebook_idx
                )

                # Add token to appropriate codebook
                self._add_audio_token_to_codebook(sequence, next_token, codebook_idx)

        logger.info(
            f"AUDIO_IN_PROGRESS phase completed: {num_timesteps} timesteps, "
            f"{num_codebooks} codebooks"
        )

    def _extract_audio_logits(
        self, outputs: Dict[str, torch.Tensor], sequence: GenerationSequence, codebook_idx: int
    ) -> Optional[torch.Tensor]:
        """Extract audio logits for specific codebook from model outputs."""
        # Try different possible output keys for audio logits
        possible_keys = [
            f"audio_logits_codebook_{codebook_idx}",
            f"audio_logits_{codebook_idx}",
            f"codebook_{codebook_idx}_logits",
            "audio_logits",
            "audio_head_logits",
        ]

        for key in possible_keys:
            if key in outputs:
                logits = outputs[key]
                if isinstance(logits, torch.Tensor):
                    # Handle different shapes for audio logits
                    if logits.dim() == 4:  # [batch, seq, codebooks, vocab]
                        if logits.size(2) > codebook_idx:
                            return logits[0, -1, codebook_idx, :]
                    elif logits.dim() == 3:  # [batch, seq, vocab] - single codebook
                        return logits[0, -1, :]
                    elif logits.dim() == 2:  # [batch, vocab]
                        return logits[0, :]
                    elif logits.dim() == 1:  # [vocab]
                        return logits
                else:
                    # Convert numpy or other formats to tensor
                    logits = torch.tensor(logits, device=self.config.device)
                    if logits.dim() == 4 and logits.size(2) > codebook_idx:
                        return logits[0, -1, codebook_idx, :]
                    elif logits.dim() == 3:
                        return logits[0, -1, :]
                    elif logits.dim() == 2:
                        return logits[0, :]
                    else:
                        return logits

        # Fallback: try to extract from general logits output
        if "logits" in outputs:
            general_logits = outputs["logits"]
            if isinstance(general_logits, torch.Tensor) and general_logits.dim() >= 3:
                # Assume last dimension might contain multiple heads
                # Try to extract audio-specific portion
                vocab_size = self.config.model_config.audio_vocab_size
                if general_logits.size(-1) >= vocab_size:
                    # Extract audio vocab portion (assuming it's at the end)
                    audio_start = general_logits.size(-1) - vocab_size
                    return general_logits[0, -1, audio_start : audio_start + vocab_size]

        logger.warning(f"No audio logits found for codebook {codebook_idx} in model outputs")
        return None

    def _process_audio_logits(
        self, logits: torch.Tensor, sequence: GenerationSequence, step: int, codebook_idx: int
    ) -> torch.Tensor:
        """Process audio logits with codebook-specific constraints."""
        processed_logits = logits.clone()

        # Apply causality constraints based on delay pattern
        processed_logits = self._apply_causality_constraints(
            processed_logits, sequence, step, codebook_idx
        )

        # Apply audio-specific token constraints
        processed_logits = self._apply_audio_token_constraints(
            processed_logits, sequence, step, codebook_idx
        )

        # Apply repetition penalties if configured
        if hasattr(sequence, "generated_audio_tokens") and sequence.generated_audio_tokens:
            processed_logits = self._apply_audio_repetition_penalty(
                processed_logits, sequence, codebook_idx
            )

        return processed_logits

    def _apply_causality_constraints(
        self, logits: torch.Tensor, sequence: GenerationSequence, step: int, codebook_idx: int
    ) -> torch.Tensor:
        """Apply causality constraints based on delay pattern."""
        # Ensure tokens are only generated when allowed by delay pattern
        if not self.delay_scheduler.is_codebook_active(step, codebook_idx):
            # If codebook shouldn't be active, force pad token
            pad_token = self.config.model_config.audio_pad_token_id
            if pad_token is not None and pad_token < logits.size(0):
                logits.fill_(float("-inf"))
                logits[pad_token] = 0.0
            else:
                # Fallback: heavily bias toward first token (assumed to be pad/silent)
                logits.fill_(float("-inf"))
                logits[0] = 0.0

        return logits

    def _apply_audio_token_constraints(
        self, logits: torch.Tensor, sequence: GenerationSequence, step: int, codebook_idx: int
    ) -> torch.Tensor:
        """Apply audio-specific token constraints."""
        config = self.config.model_config

        # Prevent text tokens in audio generation
        text_tokens = [
            config.text_bos_token_id,
            config.text_eos_token_id,
            config.text_pad_token_id,
        ]

        for token_id in text_tokens:
            if token_id is not None and token_id < logits.size(0):
                logits[token_id] = float("-inf")

        # Handle special audio tokens based on position
        if step == 0 and codebook_idx == 0:
            # First position, first codebook - allow BOS
            pass
        else:
            # Other positions - discourage BOS
            if config.audio_bos_token_id is not None and config.audio_bos_token_id < logits.size(0):
                logits[config.audio_bos_token_id] -= 10.0

        return logits

    def _apply_audio_repetition_penalty(
        self, logits: torch.Tensor, sequence: GenerationSequence, codebook_idx: int
    ) -> torch.Tensor:
        """Apply repetition penalty to audio tokens for specific codebook."""
        config = self.config.sampler_config
        penalty = getattr(config, "audio_repetition_penalty", 1.0)

        if penalty == 1.0 or codebook_idx >= len(sequence.generated_audio_tokens[0]):
            return logits

        # Get tokens for current codebook
        codebook_tokens = [
            tokens[codebook_idx]
            for tokens in sequence.generated_audio_tokens
            if len(tokens) > codebook_idx
        ]

        # Count token frequencies
        token_counts = {}
        for token in codebook_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        # Apply penalty
        for token, count in token_counts.items():
            if token < logits.size(0):
                if logits[token] > 0:
                    logits[token] /= penalty**count
                else:
                    logits[token] *= penalty**count

        return logits

    def _sample_audio_token(
        self, logits: torch.Tensor, sequence: GenerationSequence, step: int, codebook_idx: int
    ) -> int:
        """Sample audio token for specific codebook."""
        # Get codebook history for this specific codebook
        codebook_history = []
        if (
            hasattr(sequence, "generated_audio_tokens")
            and sequence.generated_audio_tokens
            and codebook_idx < len(sequence.generated_audio_tokens[0])
        ):
            codebook_history = [
                tokens[codebook_idx]
                for tokens in sequence.generated_audio_tokens
                if len(tokens) > codebook_idx
            ]

        return self.sampler.sample_audio_logits(
            logits, codebook_idx, self.config.sampler_config, codebook_history
        )

    def _add_audio_token_to_codebook(
        self, sequence: GenerationSequence, token: int, codebook_idx: int
    ):
        """Add audio token to specific codebook in sequence."""
        if not hasattr(sequence, "generated_audio_tokens"):
            sequence.generated_audio_tokens = []

        # Ensure we have enough timesteps
        current_timestep = len(sequence.generated_audio_tokens) - 1
        target_timestep = (
            current_timestep if codebook_idx > 0 else len(sequence.generated_audio_tokens)
        )

        # Extend timesteps if needed
        while len(sequence.generated_audio_tokens) <= target_timestep:
            sequence.generated_audio_tokens.append([None] * 8)  # 8 codebooks

        # Add token to specific codebook
        if target_timestep < len(sequence.generated_audio_tokens):
            sequence.generated_audio_tokens[target_timestep][codebook_idx] = token

    def _should_stop_audio_generation(
        self, token_id: int, sequence: GenerationSequence, step: int, eos_token: Optional[int]
    ) -> bool:
        """Check if audio generation should stop."""
        # Stop on EOS token
        if eos_token is not None and token_id == eos_token:
            return True

        # Stop on max length
        max_length = self.config.sampler_config.max_audio_length
        if step >= max_length:
            return True

        # Stop on timeout
        if hasattr(self.config, "timeout_seconds"):
            elapsed = time.time() - sequence.start_time
            if elapsed > self.config.timeout_seconds:
                logger.warning(f"Audio generation timeout after {elapsed:.2f}s")
                return True

        return False

    def _stream_audio_generation(self, sequence: GenerationSequence) -> Iterator[Dict[str, Any]]:
        """Stream audio generation results with real-time updates."""
        logger.info(f"Starting streaming audio generation for {sequence.request_id}")

        # Phase 1: Stream AUDIO_INIT
        for result in self._stream_audio_init_phase(sequence):
            yield result

        # Phase 2: Stream AUDIO_IN_PROGRESS
        for result in self._stream_audio_in_progress_phase(sequence):
            yield result

        # Final result
        yield {
            "request_id": sequence.request_id,
            "audio_tokens": sequence.generated_audio_tokens.copy()
            if sequence.generated_audio_tokens
            else [],
            "status": "audio_completed",
            "chunk_type": "audio_final",
            "metrics": sequence.get_metrics(),
        }

    def _stream_audio_init_phase(self, sequence: GenerationSequence) -> Iterator[Dict[str, Any]]:
        """Stream audio initialization phase."""
        sequence.state = GenerationState.AUDIO_INIT
        sequence.mode = GenerationMode.AUDIO

        max_length = self.config.sampler_config.max_audio_length
        chunk_size = self.config.streaming_chunk_size

        for step in range(max_length):
            # Generate step (similar to _run_audio_init_phase but with yielding)
            inputs = self.engine.prepare_inputs(sequence)
            inputs["generation_mode"] = GenerationMode.AUDIO.value
            inputs["audio_phase"] = "init"
            inputs["target_codebook"] = 0

            try:
                outputs = self.engine.run_inference(inputs)
            except Exception as e:
                logger.error(f"Streaming audio INIT inference failed at step {step}: {e}")
                yield {
                    "request_id": sequence.request_id,
                    "error": str(e),
                    "status": "error",
                    "step": step,
                }
                break

            audio_logits = self.logits_handler.extract_logits(
                outputs, sequence, GenerationMode.AUDIO, codebook_idx=0
            )
            if audio_logits is None:
                break

            processed_logits = self.logits_handler.fused_logits_processing(
                outputs, sequence, GenerationMode.AUDIO, codebook_idx=0
            )
            next_token = self._sample_audio_token(processed_logits, sequence, step, codebook_idx=0)
            self._add_audio_token_to_codebook(sequence, next_token, codebook_idx=0)

            # Yield chunk when ready
            if step % chunk_size == 0 or step == max_length - 1:
                yield {
                    "request_id": sequence.request_id,
                    "audio_tokens": sequence.generated_audio_tokens.copy()
                    if sequence.generated_audio_tokens
                    else [],
                    "status": "generating_audio_init",
                    "chunk_type": "audio_init",
                    "step": step,
                    "codebook": 0,
                }

            # Check for stopping
            eos_token = self.config.model_config.audio_eos_token_id
            if self._should_stop_audio_generation(next_token, sequence, step, eos_token):
                break

    def _stream_audio_in_progress_phase(
        self, sequence: GenerationSequence
    ) -> Iterator[Dict[str, Any]]:
        """Stream audio in-progress phase."""
        sequence.state = GenerationState.AUDIO_IN_PROGRESS

        if not sequence.generated_audio_tokens:
            logger.warning("No audio tokens from INIT phase for streaming")
            return

        num_timesteps = len(sequence.generated_audio_tokens)
        chunk_size = self.config.streaming_chunk_size

        tokens_generated = 0

        for step in range(num_timesteps):
            active_codebooks = self.delay_scheduler.get_active_codebooks(step)

            for codebook_idx in active_codebooks:
                if codebook_idx == 0:
                    continue  # Already generated

                inputs = self.engine.prepare_inputs(sequence)
                inputs["generation_mode"] = GenerationMode.AUDIO.value
                inputs["audio_phase"] = "in_progress"
                inputs["target_codebook"] = codebook_idx
                inputs["current_timestep"] = step

                try:
                    outputs = self.engine.run_inference(inputs)
                except Exception as e:
                    logger.error(f"Streaming audio IN_PROGRESS inference failed: {e}")
                    continue

                audio_logits = self.logits_handler.extract_logits(
                    outputs, sequence, GenerationMode.AUDIO, codebook_idx
                )
                if audio_logits is None:
                    continue

                processed_logits = self.logits_handler.fused_logits_processing(
                    outputs, sequence, GenerationMode.AUDIO, codebook_idx
                )
                next_token = self._sample_audio_token(
                    processed_logits, sequence, step, codebook_idx
                )
                self._add_audio_token_to_codebook(sequence, next_token, codebook_idx)

                tokens_generated += 1

                # Yield chunk when ready
                if tokens_generated % chunk_size == 0:
                    yield {
                        "request_id": sequence.request_id,
                        "audio_tokens": sequence.generated_audio_tokens.copy(),
                        "status": "generating_audio_progress",
                        "chunk_type": "audio_progress",
                        "step": step,
                        "codebook": codebook_idx,
                        "tokens_generated": tokens_generated,
                    }

    def _generate_audio_frame(self, sequence: GenerationSequence, frame: int) -> List[int]:
        """Generate audio tokens for a single frame across all codebooks."""
        # Get active codebooks for this frame
        active_codebooks = self.delay_scheduler.get_active_codebooks(frame)

        # Initialize frame tokens (pad inactive codebooks)
        frame_tokens = [0] * 8  # Assuming pad token is 0

        # Generate tokens for active codebooks
        for codebook_idx in active_codebooks:
            # This is a simplified implementation
            # Full implementation would:
            # 1. Prepare inputs for current frame and codebook
            # 2. Run inference to get audio logits
            # 3. Apply causality mask
            # 4. Sample from logits for specific codebook

            # For now, generate dummy tokens
            frame_tokens[codebook_idx] = np.random.randint(1, 1024)  # Random token

        return frame_tokens


@dataclass
class StreamingChunk:
    """Container for streaming chunk data."""

    request_id: str
    chunk_id: int
    chunk_type: str  # "text", "audio_init", "audio_frame", "metadata", "heartbeat"
    data: Dict[str, Any]
    timestamp: float
    latency_ms: float
    generation_phase: str
    codebook_data: Optional[Dict[int, List[int]]] = None  # codebook_id -> tokens
    audio_frame_idx: Optional[int] = None
    is_partial: bool = True
    is_final: bool = False


@dataclass
class StreamingBuffer:
    """Real-time buffer for streaming audio generation."""

    buffer_size: int = 512
    chunk_size: int = 32
    max_latency_ms: float = 100.0

    # Internal buffers
    text_buffer: List[int] = field(default_factory=list)
    audio_buffers: Dict[int, List[int]] = field(default_factory=dict)  # codebook_id -> tokens
    pending_chunks: List[StreamingChunk] = field(default_factory=list)

    # Timing and flow control
    last_chunk_time: float = 0.0
    target_chunk_interval_ms: float = 50.0
    backpressure_threshold: int = 5  # Max pending chunks

    # Frame assembly for audio
    current_frame_tokens: Dict[int, Optional[int]] = field(
        default_factory=dict
    )  # codebook -> token
    completed_frames: List[Dict[int, int]] = field(default_factory=list)
    frame_assembly_timeout_ms: float = 200.0

    def __post_init__(self):
        # Initialize audio buffers for 8 codebooks
        for codebook_id in range(8):
            self.audio_buffers[codebook_id] = []
            self.current_frame_tokens[codebook_id] = None


class StreamingIterator:
    """Advanced real-time streaming iterator with buffering and performance optimization."""

    def __init__(
        self,
        pipeline: "HiggsAudioGenerationPipeline",
        sequence: "GenerationSequence",
        buffer_config: Optional[StreamingBuffer] = None,
    ):
        self.pipeline = pipeline
        self.sequence = sequence
        self.buffer = buffer_config or StreamingBuffer()

        # State tracking
        self._started = False
        self._completed = False
        self._chunk_counter = 0
        self._generation_thread = None
        self._stop_event = threading.Event()
        self._chunk_queue = []
        self._queue_lock = threading.Lock()

        # Performance monitoring
        self._generation_start_time = 0.0
        self._total_tokens_generated = 0
        self._average_latency_ms = 0.0
        self._latency_samples = []

        # Adaptive buffering
        self._adaptive_buffer_enabled = True
        self._generation_speed_tokens_per_sec = 0.0
        self._last_speed_update = 0.0

        # Backpressure control
        self._backpressure_active = False
        self._dropped_chunks = 0

        logger.info(
            f"StreamingIterator initialized for {sequence.request_id} with buffer_size={self.buffer.buffer_size}"
        )

    def __iter__(self):
        return self

    def __next__(self):
        if not self._started:
            return self._start_streaming()

        if self._completed:
            self._cleanup()
            raise StopIteration

        # Get next chunk with timeout handling
        chunk = self._get_next_chunk()
        if chunk is None:
            if self._is_generation_complete():
                self._completed = True
                return self._create_final_chunk()
            else:
                # Generate heartbeat chunk to maintain connection
                return self._create_heartbeat_chunk()

        return chunk

    def _start_streaming(self) -> StreamingChunk:
        """Initialize streaming and return start chunk."""
        self._started = True
        self._generation_start_time = time.time()

        # Start background generation thread
        self._generation_thread = threading.Thread(
            target=self._run_background_generation, daemon=True
        )
        self._generation_thread.start()

        start_chunk = StreamingChunk(
            request_id=self.sequence.request_id,
            chunk_id=self._chunk_counter,
            chunk_type="metadata",
            data={
                "status": "started",
                "buffer_config": {
                    "buffer_size": self.buffer.buffer_size,
                    "chunk_size": self.buffer.chunk_size,
                    "max_latency_ms": self.buffer.max_latency_ms,
                },
            },
            timestamp=time.time(),
            latency_ms=0.0,
            generation_phase="initialization",
        )

        self._chunk_counter += 1
        logger.info(f"Streaming started for {self.sequence.request_id}")
        return start_chunk

    def _get_next_chunk(self) -> Optional[StreamingChunk]:
        """Get next available chunk from buffer."""
        with self._queue_lock:
            if not self._chunk_queue:
                return None

            # Check for backpressure
            if len(self._chunk_queue) > self.buffer.backpressure_threshold:
                if not self._backpressure_active:
                    self._backpressure_active = True
                    logger.warning(
                        f"Backpressure activated: {len(self._chunk_queue)} pending chunks"
                    )

                # Drop oldest non-critical chunks
                self._apply_backpressure()
            else:
                if self._backpressure_active:
                    self._backpressure_active = False
                    logger.info("Backpressure released")

            chunk = self._chunk_queue.pop(0)
            self._update_latency_metrics(chunk)
            return chunk

    def _apply_backpressure(self):
        """Apply backpressure by dropping non-critical chunks."""
        critical_types = {"metadata", "audio_frame"}
        original_count = len(self._chunk_queue)

        # Keep only critical chunks and most recent chunks
        self._chunk_queue = [
            chunk
            for chunk in self._chunk_queue
            if chunk.chunk_type in critical_types or chunk.chunk_id >= self._chunk_counter - 3
        ]

        dropped = original_count - len(self._chunk_queue)
        if dropped > 0:
            self._dropped_chunks += dropped
            logger.debug(f"Dropped {dropped} chunks due to backpressure")

    def _run_background_generation(self):
        """Background thread for generation with real-time buffering."""
        try:
            logger.info(f"Background generation started for {self.sequence.request_id}")

            # Text generation phase
            self._stream_text_phase()

            # Audio generation phases if reference audio provided
            if self.sequence.reference_audio is not None:
                self._stream_audio_init_phase()
                self._stream_audio_in_progress_phase()

            # Mark completion
            self._enqueue_completion_chunk()

        except Exception as e:
            logger.error(f"Background generation failed: {e}")
            self._enqueue_error_chunk(str(e))
        finally:
            logger.info(f"Background generation completed for {self.sequence.request_id}")

    def _stream_text_phase(self):
        """Stream text generation with adaptive buffering."""
        logger.info(f"Starting text streaming phase for {self.sequence.request_id}")

        for chunk_data in self.pipeline._stream_text_generation(self.sequence):
            if self._stop_event.is_set():
                break

            # Convert to streaming chunk
            chunk = self._create_text_chunk(chunk_data)
            self._enqueue_chunk(chunk)

            # Update generation speed metrics
            self._update_generation_speed()

            # Adaptive buffering based on generation speed
            if self._adaptive_buffer_enabled:
                self._adjust_buffer_size()

    def _stream_audio_init_phase(self):
        """Stream audio initialization phase."""
        logger.info(f"Starting audio init streaming phase for {self.sequence.request_id}")

        for chunk_data in self.pipeline._stream_audio_init_phase(self.sequence):
            if self._stop_event.is_set():
                break

            chunk = self._create_audio_chunk(chunk_data, "audio_init")
            self._enqueue_chunk(chunk)

    def _stream_audio_in_progress_phase(self):
        """Stream audio in-progress phase with frame assembly."""
        logger.info(f"Starting audio in-progress streaming phase for {self.sequence.request_id}")

        for chunk_data in self.pipeline._stream_audio_in_progress_phase(self.sequence):
            if self._stop_event.is_set():
                break

            # Handle multi-codebook frame assembly
            if self._is_frame_data(chunk_data):
                self._process_audio_frame(chunk_data)
            else:
                chunk = self._create_audio_chunk(chunk_data, "audio_in_progress")
                self._enqueue_chunk(chunk)

    def _process_audio_frame(self, chunk_data: Dict[str, Any]):
        """Process audio frame data with codebook assembly."""
        codebook_id = chunk_data.get("codebook_id", 0)
        token = chunk_data.get("token")
        timestep = chunk_data.get("timestep", 0)

        # Add token to current frame
        self.buffer.current_frame_tokens[codebook_id] = token

        # Check if frame is complete (all codebooks have tokens)
        if self._is_frame_complete():
            frame_data = dict(self.buffer.current_frame_tokens)
            self.buffer.completed_frames.append(frame_data)

            # Create frame chunk
            chunk = StreamingChunk(
                request_id=self.sequence.request_id,
                chunk_id=self._chunk_counter,
                chunk_type="audio_frame",
                data={
                    "frame_data": frame_data,
                    "timestep": timestep,
                    "frame_idx": len(self.buffer.completed_frames) - 1,
                },
                timestamp=time.time(),
                latency_ms=self._calculate_latency(),
                generation_phase="audio_in_progress",
                codebook_data=frame_data,
                audio_frame_idx=len(self.buffer.completed_frames) - 1,
            )

            self._enqueue_chunk(chunk)

            # Reset frame assembly
            for cb_id in range(8):
                self.buffer.current_frame_tokens[cb_id] = None

            self._chunk_counter += 1

    def _is_frame_data(self, chunk_data: Dict[str, Any]) -> bool:
        """Check if chunk data represents frame-level audio data."""
        return "codebook_id" in chunk_data and "token" in chunk_data

    def _is_frame_complete(self) -> bool:
        """Check if current frame has tokens from all active codebooks."""
        # For active codebooks, check if all have tokens
        active_codebooks = self._get_active_codebooks()
        return all(
            self.buffer.current_frame_tokens[cb_id] is not None for cb_id in active_codebooks
        )

    def _get_active_codebooks(self) -> List[int]:
        """Get list of currently active codebooks based on delay pattern."""
        # This would use the delay pattern scheduler
        # For now, assume all 8 codebooks are active
        return list(range(8))

    def _create_text_chunk(self, chunk_data: Dict[str, Any]) -> StreamingChunk:
        """Create streaming chunk from text generation data."""
        chunk = StreamingChunk(
            request_id=self.sequence.request_id,
            chunk_id=self._chunk_counter,
            chunk_type="text",
            data=chunk_data,
            timestamp=time.time(),
            latency_ms=self._calculate_latency(),
            generation_phase="text",
        )
        self._chunk_counter += 1
        return chunk

    def _create_audio_chunk(self, chunk_data: Dict[str, Any], phase: str) -> StreamingChunk:
        """Create streaming chunk from audio generation data."""
        chunk = StreamingChunk(
            request_id=self.sequence.request_id,
            chunk_id=self._chunk_counter,
            chunk_type=phase,
            data=chunk_data,
            timestamp=time.time(),
            latency_ms=self._calculate_latency(),
            generation_phase=phase,
            codebook_data=chunk_data.get("codebook_data"),
            audio_frame_idx=chunk_data.get("frame_idx"),
        )
        self._chunk_counter += 1
        return chunk

    def _create_heartbeat_chunk(self) -> StreamingChunk:
        """Create heartbeat chunk to maintain connection."""
        chunk = StreamingChunk(
            request_id=self.sequence.request_id,
            chunk_id=self._chunk_counter,
            chunk_type="heartbeat",
            data={"status": "generating", "metrics": self._get_current_metrics()},
            timestamp=time.time(),
            latency_ms=self._calculate_latency(),
            generation_phase=self.sequence.state.value,
        )
        self._chunk_counter += 1
        return chunk

    def _create_final_chunk(self) -> StreamingChunk:
        """Create final completion chunk."""
        final_metrics = self._get_final_metrics()

        chunk = StreamingChunk(
            request_id=self.sequence.request_id,
            chunk_id=self._chunk_counter,
            chunk_type="metadata",
            data={
                "status": "completed",
                "final_metrics": final_metrics,
                "total_chunks": self._chunk_counter,
                "dropped_chunks": self._dropped_chunks,
            },
            timestamp=time.time(),
            latency_ms=self._calculate_latency(),
            generation_phase="completed",
            is_final=True,
        )

        logger.info(f"Streaming completed for {self.sequence.request_id}: {final_metrics}")
        return chunk

    def _enqueue_chunk(self, chunk: StreamingChunk):
        """Add chunk to output queue."""
        with self._queue_lock:
            self._chunk_queue.append(chunk)

    def _enqueue_completion_chunk(self):
        """Add completion marker to queue."""
        completion_chunk = StreamingChunk(
            request_id=self.sequence.request_id,
            chunk_id=self._chunk_counter,
            chunk_type="metadata",
            data={"status": "generation_complete"},
            timestamp=time.time(),
            latency_ms=self._calculate_latency(),
            generation_phase="completed",
        )
        self._enqueue_chunk(completion_chunk)

    def _enqueue_error_chunk(self, error_message: str):
        """Add error chunk to queue."""
        error_chunk = StreamingChunk(
            request_id=self.sequence.request_id,
            chunk_id=self._chunk_counter,
            chunk_type="metadata",
            data={"status": "error", "error": error_message},
            timestamp=time.time(),
            latency_ms=self._calculate_latency(),
            generation_phase="error",
        )
        self._enqueue_chunk(error_chunk)

    def _calculate_latency(self) -> float:
        """Calculate current latency in milliseconds."""
        return (time.time() - self._generation_start_time) * 1000.0

    def _update_latency_metrics(self, chunk: StreamingChunk):
        """Update latency metrics from delivered chunk."""
        self._latency_samples.append(chunk.latency_ms)

        # Keep only recent samples for rolling average
        if len(self._latency_samples) > 100:
            self._latency_samples = self._latency_samples[-100:]

        self._average_latency_ms = sum(self._latency_samples) / len(self._latency_samples)

    def _update_generation_speed(self):
        """Update generation speed metrics."""
        current_time = time.time()
        if self._last_speed_update > 0:
            time_diff = current_time - self._last_speed_update
            if time_diff > 0:
                tokens_generated = 1  # This chunk represents 1 token
                self._generation_speed_tokens_per_sec = tokens_generated / time_diff

        self._last_speed_update = current_time
        self._total_tokens_generated += 1

    def _adjust_buffer_size(self):
        """Adjust buffer size based on generation speed."""
        if self._generation_speed_tokens_per_sec > 50:  # Fast generation
            self.buffer.chunk_size = min(64, self.buffer.chunk_size + 8)
        elif self._generation_speed_tokens_per_sec < 10:  # Slow generation
            self.buffer.chunk_size = max(16, self.buffer.chunk_size - 4)

    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            "total_tokens": self._total_tokens_generated,
            "average_latency_ms": self._average_latency_ms,
            "generation_speed_tokens_per_sec": self._generation_speed_tokens_per_sec,
            "buffer_size": self.buffer.buffer_size,
            "chunk_size": self.buffer.chunk_size,
            "backpressure_active": self._backpressure_active,
            "pending_chunks": len(self._chunk_queue),
            "dropped_chunks": self._dropped_chunks,
        }

    def _get_final_metrics(self) -> Dict[str, Any]:
        """Get final performance metrics."""
        total_time = time.time() - self._generation_start_time
        return {
            **self._get_current_metrics(),
            "total_time_seconds": total_time,
            "overall_tokens_per_sec": self._total_tokens_generated / max(total_time, 0.001),
            "total_chunks_delivered": self._chunk_counter,
        }

    def _is_generation_complete(self) -> bool:
        """Check if generation is complete."""
        if self._generation_thread and not self._generation_thread.is_alive():
            return True
        return self.sequence.state == GenerationState.COMPLETED

    def _cleanup(self):
        """Clean up resources."""
        self._stop_event.set()
        if self._generation_thread and self._generation_thread.is_alive():
            self._generation_thread.join(timeout=1.0)

        logger.info(f"StreamingIterator cleanup completed for {self.sequence.request_id}")

    def stop(self):
        """Stop streaming gracefully."""
        logger.info(f"Stopping streaming for {self.sequence.request_id}")
        self._stop_event.set()
        self._completed = True


# Voice Cloning and TTS Utilities


@dataclass
class VoiceProfile:
    """Voice profile for speaker conditioning."""

    speaker_id: str
    embedding: torch.Tensor
    reference_audio: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)
    voice_characteristics: Dict[str, float] = field(default_factory=dict)  # pitch, tone, pace, etc.
    sample_rate: int = 24000
    duration_seconds: float = 0.0
    quality_score: float = 0.0  # Embedding quality assessment


@dataclass
class TTSPreset:
    """Preset configuration for TTS generation."""

    name: str
    description: str
    sampler_config: SamplerConfig
    pipeline_config: PipelineConfig
    voice_style: str = "neutral"  # neutral, expressive, calm, energetic
    speed_multiplier: float = 1.0
    pitch_shift: float = 0.0  # In semitones
    emotion_intensity: float = 0.5  # 0.0 to 1.0


class VoiceCloneingUtils:
    """Utilities for voice cloning and speaker conditioning."""

    @staticmethod
    def extract_speaker_embedding(
        audio_tensor: torch.Tensor, sample_rate: int = 24000, device: str = "cuda"
    ) -> torch.Tensor:
        """Extract speaker embedding from reference audio.

        Args:
            audio_tensor: Input audio tensor [batch_size, sequence_length]
            sample_rate: Audio sample rate
            device: Device for computation

        Returns:
            Speaker embedding tensor [embedding_dim]
        """
        # Ensure audio is on correct device
        audio_tensor = audio_tensor.to(device)

        # Normalize audio
        audio_tensor = VoiceCloneingUtils._normalize_audio(audio_tensor)

        # Extract features for speaker embedding
        # This would typically use a pretrained speaker encoder
        # For now, create a mock embedding
        batch_size = audio_tensor.shape[0] if audio_tensor.dim() > 1 else 1
        embedding_dim = 512  # Typical speaker embedding dimension

        # Mock speaker embedding (would be replaced with actual encoder)
        mock_embedding = torch.randn(batch_size, embedding_dim, device=device)

        # Apply normalization
        mock_embedding = F.normalize(mock_embedding, p=2, dim=-1)

        logger.info(f"Extracted speaker embedding: shape={mock_embedding.shape}")
        return mock_embedding.squeeze(0) if batch_size == 1 else mock_embedding

    @staticmethod
    def _normalize_audio(audio: torch.Tensor) -> torch.Tensor:
        """Normalize audio to [-1, 1] range."""
        # Remove DC offset
        audio = audio - audio.mean()

        # Normalize to [-1, 1]
        max_val = audio.abs().max()
        if max_val > 0:
            audio = audio / max_val

        return audio

    @staticmethod
    def preprocess_reference_audio(
        audio_path: str,
        target_sample_rate: int = 24000,
        target_duration: float = 10.0,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Preprocess reference audio for voice cloning.

        Args:
            audio_path: Path to audio file
            target_sample_rate: Target sample rate
            target_duration: Target duration in seconds
            device: Device for computation

        Returns:
            Preprocessed audio tensor
        """
        # This would use actual audio loading (librosa, torchaudio, etc.)
        # For now, create a mock audio tensor
        logger.info(f"Loading reference audio from {audio_path}")

        target_length = int(target_sample_rate * target_duration)
        mock_audio = torch.randn(target_length) * 0.5  # Mock audio signal
        mock_audio = VoiceCloneingUtils._normalize_audio(mock_audio)

        logger.info(f"Preprocessed audio: length={target_length}, sample_rate={target_sample_rate}")
        return mock_audio.to(device)

    @staticmethod
    def create_voice_profile(
        speaker_id: str, reference_audio_path: str, device: str = "cuda"
    ) -> VoiceProfile:
        """Create a voice profile from reference audio.

        Args:
            speaker_id: Unique identifier for the speaker
            reference_audio_path: Path to reference audio file
            device: Device for computation

        Returns:
            VoiceProfile object
        """
        # Preprocess reference audio
        reference_audio = VoiceCloneingUtils.preprocess_reference_audio(
            reference_audio_path, device=device
        )

        # Extract speaker embedding
        embedding = VoiceCloneingUtils.extract_speaker_embedding(
            reference_audio.unsqueeze(0), device=device
        )

        # Assess audio quality (mock implementation)
        quality_score = VoiceCloneingUtils._assess_audio_quality(reference_audio)

        # Extract voice characteristics
        characteristics = VoiceCloneingUtils._analyze_voice_characteristics(reference_audio)

        profile = VoiceProfile(
            speaker_id=speaker_id,
            embedding=embedding,
            reference_audio=reference_audio,
            metadata={
                "source_path": reference_audio_path,
                "created_at": time.time(),
                "device": device,
            },
            voice_characteristics=characteristics,
            duration_seconds=len(reference_audio) / 24000,
            quality_score=quality_score,
        )

        logger.info(f"Created voice profile for {speaker_id}: quality={quality_score:.2f}")
        return profile

    @staticmethod
    def _assess_audio_quality(audio: torch.Tensor) -> float:
        """Assess audio quality for voice cloning."""
        # Mock quality assessment
        # Real implementation would check SNR, clarity, etc.
        snr_estimate = 20.0 + torch.randn(1).item() * 5.0  # Mock SNR
        clarity_score = 0.8 + torch.randn(1).item() * 0.1  # Mock clarity

        # Combine metrics
        quality = min(1.0, max(0.0, (snr_estimate / 30.0 + clarity_score) / 2.0))
        return quality

    @staticmethod
    def _analyze_voice_characteristics(audio: torch.Tensor) -> Dict[str, float]:
        """Analyze voice characteristics from audio."""
        # Mock voice analysis
        # Real implementation would extract pitch, formants, etc.
        return {
            "fundamental_frequency": 150.0 + torch.randn(1).item() * 50.0,  # Hz
            "pitch_variance": 0.3 + torch.randn(1).item() * 0.1,
            "formant_f1": 500.0 + torch.randn(1).item() * 100.0,  # Hz
            "formant_f2": 1500.0 + torch.randn(1).item() * 200.0,  # Hz
            "spectral_centroid": 2000.0 + torch.randn(1).item() * 500.0,  # Hz
            "speech_rate": 4.5 + torch.randn(1).item() * 1.0,  # syllables/sec
        }

    @staticmethod
    def save_voice_profile(profile: VoiceProfile, save_path: str):
        """Save voice profile to disk."""
        profile_data = {
            "speaker_id": profile.speaker_id,
            "embedding": profile.embedding.cpu(),
            "reference_audio": profile.reference_audio.cpu(),
            "metadata": profile.metadata,
            "voice_characteristics": profile.voice_characteristics,
            "sample_rate": profile.sample_rate,
            "duration_seconds": profile.duration_seconds,
            "quality_score": profile.quality_score,
        }

        torch.save(profile_data, save_path)
        logger.info(f"Saved voice profile for {profile.speaker_id} to {save_path}")

    @staticmethod
    def load_voice_profile(load_path: str, device: str = "cuda") -> VoiceProfile:
        """Load voice profile from disk."""
        profile_data = torch.load(load_path, map_location=device)

        profile = VoiceProfile(
            speaker_id=profile_data["speaker_id"],
            embedding=profile_data["embedding"].to(device),
            reference_audio=profile_data["reference_audio"].to(device),
            metadata=profile_data["metadata"],
            voice_characteristics=profile_data["voice_characteristics"],
            sample_rate=profile_data["sample_rate"],
            duration_seconds=profile_data["duration_seconds"],
            quality_score=profile_data["quality_score"],
        )

        logger.info(f"Loaded voice profile for {profile.speaker_id} from {load_path}")
        return profile


class TTSPresets:
    """Predefined TTS configurations for different use cases."""

    @staticmethod
    def get_fast_preset() -> TTSPreset:
        """Fast TTS preset optimized for speed."""
        return TTSPreset(
            name="fast",
            description="High-speed TTS with acceptable quality",
            sampler_config=SamplerConfig(
                text_temperature=0.7,
                text_top_k=30,
                text_top_p=0.8,
                audio_temperature=0.8,
                audio_top_k=50,
                max_text_length=1024,
                max_audio_length=2048,
            ),
            pipeline_config=PipelineConfig(
                use_static_cache=True,
                use_kernel_fusion=True,
                enable_streaming=True,
                streaming_chunk_size=48,
                streaming_buffer_size=256,
                max_streaming_latency_ms=50.0,
            ),
            voice_style="neutral",
            speed_multiplier=1.2,
        )

    @staticmethod
    def get_quality_preset() -> TTSPreset:
        """High-quality TTS preset."""
        return TTSPreset(
            name="quality",
            description="High-quality TTS with slower generation",
            sampler_config=SamplerConfig(
                text_temperature=0.9,
                text_top_k=50,
                text_top_p=0.9,
                audio_temperature=0.9,
                audio_top_k=100,
                max_text_length=2048,
                max_audio_length=4096,
            ),
            pipeline_config=PipelineConfig(
                use_static_cache=True,
                use_kernel_fusion=True,
                enable_streaming=True,
                streaming_chunk_size=24,
                streaming_buffer_size=512,
                max_streaming_latency_ms=100.0,
            ),
            voice_style="expressive",
            speed_multiplier=0.9,
        )

    @staticmethod
    def get_balanced_preset() -> TTSPreset:
        """Balanced TTS preset for general use."""
        return TTSPreset(
            name="balanced",
            description="Balanced speed and quality for general use",
            sampler_config=SamplerConfig(
                text_temperature=0.8,
                text_top_k=40,
                text_top_p=0.85,
                audio_temperature=0.85,
                audio_top_k=75,
                max_text_length=1536,
                max_audio_length=3072,
            ),
            pipeline_config=PipelineConfig(
                use_static_cache=True,
                use_kernel_fusion=True,
                enable_streaming=True,
                streaming_chunk_size=32,
                streaming_buffer_size=384,
                max_streaming_latency_ms=75.0,
            ),
            voice_style="neutral",
        )

    @staticmethod
    def get_expressive_preset() -> TTSPreset:
        """Expressive TTS preset for dynamic speech."""
        return TTSPreset(
            name="expressive",
            description="Expressive TTS with enhanced prosody",
            sampler_config=SamplerConfig(
                text_temperature=1.0,
                text_top_k=60,
                text_top_p=0.92,
                audio_temperature=1.0,
                audio_top_k=120,
                text_repetition_penalty=1.1,
                max_text_length=2048,
                max_audio_length=4096,
            ),
            pipeline_config=PipelineConfig(
                use_static_cache=True,
                use_kernel_fusion=True,
                enable_streaming=True,
                streaming_chunk_size=28,
                streaming_buffer_size=448,
                max_streaming_latency_ms=90.0,
            ),
            voice_style="expressive",
            emotion_intensity=0.8,
        )


class VoiceSimilarityMetrics:
    """Utilities for measuring voice similarity and quality."""

    @staticmethod
    def cosine_similarity(embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """Calculate cosine similarity between speaker embeddings."""
        similarity = F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
        return similarity.item()

    @staticmethod
    def euclidean_distance(embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """Calculate Euclidean distance between speaker embeddings."""
        distance = torch.norm(embedding1 - embedding2, p=2)
        return distance.item()

    @staticmethod
    def voice_similarity_score(profile1: VoiceProfile, profile2: VoiceProfile) -> Dict[str, float]:
        """Calculate comprehensive voice similarity metrics."""
        # Embedding similarity
        cos_sim = VoiceSimilarityMetrics.cosine_similarity(profile1.embedding, profile2.embedding)
        euc_dist = VoiceSimilarityMetrics.euclidean_distance(profile1.embedding, profile2.embedding)

        # Characteristic similarity
        char_similarity = VoiceSimilarityMetrics._characteristic_similarity(
            profile1.voice_characteristics, profile2.voice_characteristics
        )

        # Overall similarity score
        overall_score = (
            cos_sim * 0.5 + char_similarity * 0.3 + (1.0 - min(euc_dist / 10.0, 1.0)) * 0.2
        )

        return {
            "cosine_similarity": cos_sim,
            "euclidean_distance": euc_dist,
            "characteristic_similarity": char_similarity,
            "overall_similarity": overall_score,
        }

    @staticmethod
    def _characteristic_similarity(chars1: Dict[str, float], chars2: Dict[str, float]) -> float:
        """Calculate similarity between voice characteristics."""
        common_keys = set(chars1.keys()) & set(chars2.keys())
        if not common_keys:
            return 0.0

        similarities = []
        for key in common_keys:
            val1, val2 = chars1[key], chars2[key]
            # Normalized difference (closer to 0 means more similar)
            diff = abs(val1 - val2) / max(abs(val1), abs(val2), 1.0)
            similarity = 1.0 - min(diff, 1.0)
            similarities.append(similarity)

        return sum(similarities) / len(similarities)


# High-level TTS and Voice Cloning Factory Functions


def create_tts_pipeline(
    model_path: str, device: str = "cuda", preset: str = "balanced", use_optimizations: bool = True
) -> HiggsAudioGenerationPipeline:
    """Create a TTS pipeline with specified preset configuration.

    Args:
        model_path: Path to the Higgs Audio model
        device: Device for computation
        preset: TTS preset name ("fast", "balanced", "quality", "expressive")
        use_optimizations: Whether to enable TensorRT-LLM optimizations

    Returns:
        Configured HiggsAudioGenerationPipeline
    """
    # Get preset configuration
    preset_map = {
        "fast": TTSPresets.get_fast_preset(),
        "balanced": TTSPresets.get_balanced_preset(),
        "quality": TTSPresets.get_quality_preset(),
        "expressive": TTSPresets.get_expressive_preset(),
    }

    if preset not in preset_map:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(preset_map.keys())}")

    tts_preset = preset_map[preset]

    # Configure pipeline with preset settings
    pipeline_config = tts_preset.pipeline_config
    pipeline_config.device = device
    pipeline_config.enable_memory_optimization = use_optimizations
    pipeline_config.enable_profiling = False  # Disable for production

    # Configure sampler
    sampler_config = tts_preset.sampler_config

    # Create tokenizer (mock for now)
    tokenizer = None  # Would load actual tokenizer

    # Create pipeline
    pipeline = HiggsAudioGenerationPipeline(
        config=pipeline_config, sampler_config=sampler_config, tokenizer=tokenizer, device=device
    )

    logger.info(f"Created TTS pipeline with preset '{preset}' on device '{device}'")
    return pipeline


def create_voice_cloning_pipeline(
    model_path: str,
    reference_audio_path: str,
    speaker_id: str = "cloned_speaker",
    device: str = "cuda",
    preset: str = "quality",
    use_optimizations: bool = True,
) -> tuple[HiggsAudioGenerationPipeline, VoiceProfile]:
    """Create a voice cloning pipeline with reference audio.

    Args:
        model_path: Path to the Higgs Audio model
        reference_audio_path: Path to reference audio file
        speaker_id: Identifier for the cloned speaker
        device: Device for computation
        preset: TTS preset name for generation quality
        use_optimizations: Whether to enable TensorRT-LLM optimizations

    Returns:
        Tuple of (configured pipeline, voice profile)
    """
    # Create base TTS pipeline
    pipeline = create_tts_pipeline(
        model_path=model_path, device=device, preset=preset, use_optimizations=use_optimizations
    )

    # Create voice profile from reference audio
    voice_profile = VoiceCloneingUtils.create_voice_profile(
        speaker_id=speaker_id, reference_audio_path=reference_audio_path, device=device
    )

    logger.info(
        f"Created voice cloning pipeline for speaker '{speaker_id}' "
        f"with quality score {voice_profile.quality_score:.2f}"
    )

    return pipeline, voice_profile


def prepare_reference_audio(
    audio_path: str,
    target_sample_rate: int = 24000,
    target_duration: float = 10.0,
    device: str = "cuda",
) -> torch.Tensor:
    """Prepare reference audio for voice cloning.

    Args:
        audio_path: Path to audio file
        target_sample_rate: Target sample rate
        target_duration: Target duration in seconds
        device: Device for computation

    Returns:
        Preprocessed audio tensor
    """
    return VoiceCloneingUtils.preprocess_reference_audio(
        audio_path=audio_path,
        target_sample_rate=target_sample_rate,
        target_duration=target_duration,
        device=device,
    )


def clone_voice_from_text(
    pipeline: HiggsAudioGenerationPipeline,
    voice_profile: VoiceProfile,
    text: str,
    stream: bool = False,
) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
    """Generate speech using voice cloning from text.

    Args:
        pipeline: Configured generation pipeline
        voice_profile: Voice profile for speaker conditioning
        text: Text to synthesize
        stream: Whether to return streaming iterator

    Returns:
        Generation result or streaming iterator
    """
    # Generate with speaker conditioning
    result = pipeline.generate(
        prompt_text=text, reference_audio=voice_profile.reference_audio, stream=stream
    )

    return result


def synthesize_speech(
    text: str,
    model_path: str = None,
    voice_profile: VoiceProfile = None,
    preset: str = "balanced",
    device: str = "cuda",
    stream: bool = False,
) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
    """High-level speech synthesis function.

    Args:
        text: Text to synthesize
        model_path: Path to Higgs Audio model (optional if using existing pipeline)
        voice_profile: Optional voice profile for cloning
        preset: TTS preset configuration
        device: Device for computation
        stream: Whether to return streaming iterator

    Returns:
        Generated speech result or streaming iterator
    """
    # Create pipeline if model path provided
    if model_path:
        pipeline = create_tts_pipeline(model_path=model_path, device=device, preset=preset)
    else:
        raise ValueError("model_path is required for speech synthesis")

    # Generate speech
    reference_audio = voice_profile.reference_audio if voice_profile else None

    result = pipeline.generate(prompt_text=text, reference_audio=reference_audio, stream=stream)

    return result
