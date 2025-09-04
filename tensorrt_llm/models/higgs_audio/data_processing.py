"""Data processing and collation utilities for HiggsAudio TensorRT-LLM model.

This module provides the HiggsAudioSampleCollator class for batching heterogeneous
samples containing text, audio/mel features, and metadata with efficient padding
and attention mask generation. Also includes RVQ delay pattern coordination for
streaming multi-codebook audio generation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from .config import HiggsAudioConfig

logger = logging.getLogger(__name__)


class DelayPatternProvider:
    """Provider for RVQ delay patterns to coordinate multi-codebook generation.

    This enables staggered token generation across multiple codebooks while
    maintaining causality constraints for high-quality audio synthesis.
    """

    def __init__(
        self,
        strategy: str = "linear",
        num_codebooks: int = 8,
        max_delay: Optional[int] = None,
        stride: int = 1,
    ):
        """Initialize delay pattern provider.

        Args:
            strategy: Delay pattern strategy ("linear", "exponential", "custom")
            num_codebooks: Number of RVQ codebooks
            max_delay: Maximum allowed delay (defaults to num_codebooks - 1)
            stride: Stride between delays (for custom patterns)
        """
        self.strategy = strategy
        self.num_codebooks = num_codebooks
        self.max_delay = max_delay if max_delay is not None else num_codebooks - 1
        self.stride = stride

        # Validate configuration
        if self.max_delay < 0:
            raise ValueError(f"max_delay must be non-negative, got {self.max_delay}")
        if self.num_codebooks <= 0:
            raise ValueError(f"num_codebooks must be positive, got {self.num_codebooks}")

    def generate_delay_pattern(
        self,
        sequence_length: int,
        n_codebooks: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate delay pattern for the given sequence.

        Args:
            sequence_length: Length of the sequence
            n_codebooks: Number of codebooks (defaults to self.num_codebooks)

        Returns:
            Delay pattern tensor [n_codebooks, sequence_length]
        """
        if n_codebooks is None:
            n_codebooks = self.num_codebooks

        delays = torch.zeros(n_codebooks, sequence_length, dtype=torch.long)

        if self.strategy == "linear":
            # Linear delay: codebook k has delay k * stride
            for k in range(n_codebooks):
                delay = min(k * self.stride, self.max_delay)
                delays[k, :] = delay

        elif self.strategy == "exponential":
            # Exponential delay: codebook k has delay 2^k - 1, clamped to max_delay
            for k in range(n_codebooks):
                delay = min(2**k - 1, self.max_delay) if k > 0 else 0
                delays[k, :] = delay

        elif self.strategy == "custom":
            # Custom delay pattern following Higgs Audio defaults
            # First codebook has no delay, others increase linearly
            for k in range(n_codebooks):
                delay = min(k, self.max_delay)
                delays[k, :] = delay

        else:
            raise ValueError(f"Unknown delay strategy: {self.strategy}")

        return delays

    def apply_delay_pattern(
        self,
        tokens: torch.Tensor,
        delay_pattern: torch.Tensor,
    ) -> torch.Tensor:
        """Apply delay pattern to multi-codebook tokens.

        Args:
            tokens: Multi-codebook tokens [n_codebooks, sequence_length]
            delay_pattern: Delay pattern [n_codebooks, sequence_length]

        Returns:
            Delayed tokens with staggered timing [n_codebooks, delayed_length]
        """
        n_codebooks, seq_len = tokens.shape
        max_delay = delay_pattern.max().item()
        delayed_length = seq_len + max_delay

        # Create delayed token tensor
        delayed_tokens = torch.full(
            (n_codebooks, delayed_length),
            fill_value=-1,  # Use -1 as placeholder for delayed positions
            dtype=tokens.dtype,
            device=tokens.device,
        )

        # Apply delays for each codebook
        for k in range(n_codebooks):
            delay = delay_pattern[k, 0].item()  # Assume uniform delay per codebook
            delayed_tokens[k, delay : delay + seq_len] = tokens[k]

        return delayed_tokens

    def reverse_delay_pattern(
        self,
        delayed_tokens: torch.Tensor,
        delay_pattern: torch.Tensor,
        original_length: int,
    ) -> torch.Tensor:
        """Reverse delay pattern to reconstruct original tokens.

        Args:
            delayed_tokens: Delayed tokens [n_codebooks, delayed_length]
            delay_pattern: Delay pattern used [n_codebooks, sequence_length]
            original_length: Original sequence length

        Returns:
            Reconstructed tokens [n_codebooks, original_length]
        """
        n_codebooks = delayed_tokens.shape[0]

        # Reconstruct original tokens
        tokens = torch.zeros(
            (n_codebooks, original_length),
            dtype=delayed_tokens.dtype,
            device=delayed_tokens.device,
        )

        for k in range(n_codebooks):
            delay = delay_pattern[k, 0].item()  # Assume uniform delay per codebook
            tokens[k] = delayed_tokens[k, delay : delay + original_length]

        return tokens


class AudioTokenUtils:
    """Utilities for handling multi-codebook audio token coordination."""

    def __init__(
        self,
        num_codebooks: int = 8,
        pad_token_id: int = 0,
        audio_stream_bos_id: int = 1024,
        audio_stream_eos_id: int = 1025,
    ):
        """Initialize audio token utilities.

        Args:
            num_codebooks: Number of RVQ codebooks
            pad_token_id: Token ID for padding
            audio_stream_bos_id: Beginning of stream token ID
            audio_stream_eos_id: End of stream token ID
        """
        self.num_codebooks = num_codebooks
        self.pad_token_id = pad_token_id
        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id

    def validate_codebook_sequences(
        self,
        codebook_tokens: List[torch.Tensor],
    ) -> bool:
        """Validate multi-codebook token sequences.

        Args:
            codebook_tokens: List of token tensors, one per codebook

        Returns:
            True if sequences are valid
        """
        if len(codebook_tokens) != self.num_codebooks:
            return False

        # Check that all sequences have the same length
        seq_lengths = [tokens.shape[-1] for tokens in codebook_tokens]
        if len(set(seq_lengths)) > 1:
            return False

        return True

    def interleave_codebook_tokens(
        self,
        codebook_tokens: List[torch.Tensor],
        delay_pattern: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Interleave multi-codebook tokens into unified sequence.

        Args:
            codebook_tokens: List of token tensors [batch_size, seq_len]
            delay_pattern: Optional delay pattern [num_codebooks, seq_len]

        Returns:
            Interleaved tokens [batch_size, total_length]
        """
        if not self.validate_codebook_sequences(codebook_tokens):
            raise ValueError("Invalid codebook token sequences")

        batch_size, seq_len = codebook_tokens[0].shape

        if delay_pattern is not None:
            # Apply delay pattern before interleaving
            delayed_tokens = []
            max_delay = delay_pattern.max().item()

            for k, tokens in enumerate(codebook_tokens):
                delay = delay_pattern[k, 0].item()

                # Pad with stream BOS tokens for delay
                if delay > 0:
                    bos_padding = torch.full(
                        (batch_size, delay),
                        self.audio_stream_bos_id,
                        dtype=tokens.dtype,
                        device=tokens.device,
                    )
                    delayed = torch.cat([bos_padding, tokens], dim=1)
                else:
                    delayed = tokens

                # Pad at end to match maximum delayed length
                total_length = seq_len + max_delay
                if delayed.shape[1] < total_length:
                    end_padding = torch.full(
                        (batch_size, total_length - delayed.shape[1]),
                        self.audio_stream_eos_id,
                        dtype=tokens.dtype,
                        device=tokens.device,
                    )
                    delayed = torch.cat([delayed, end_padding], dim=1)

                delayed_tokens.append(delayed)

            codebook_tokens = delayed_tokens
            seq_len = total_length

        # Interleave tokens: [t0_cb0, t0_cb1, ..., t0_cbN, t1_cb0, t1_cb1, ...]
        interleaved = torch.zeros(
            batch_size,
            seq_len * self.num_codebooks,
            dtype=codebook_tokens[0].dtype,
            device=codebook_tokens[0].device,
        )

        for t in range(seq_len):
            for k in range(self.num_codebooks):
                idx = t * self.num_codebooks + k
                interleaved[:, idx] = codebook_tokens[k][:, t]

        return interleaved

    def extract_codebook_tokens(
        self,
        interleaved_tokens: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Extract individual codebook tokens from interleaved sequence.

        Args:
            interleaved_tokens: Interleaved tokens [batch_size, total_length]

        Returns:
            List of codebook token tensors [batch_size, seq_len]
        """
        batch_size, total_length = interleaved_tokens.shape

        if total_length % self.num_codebooks != 0:
            raise ValueError(
                f"Total length {total_length} not divisible by num_codebooks {self.num_codebooks}"
            )

        codebook_tokens = []

        for k in range(self.num_codebooks):
            # Extract tokens for codebook k: indices k, k+N, k+2N, ...
            indices = torch.arange(
                k, total_length, self.num_codebooks, device=interleaved_tokens.device
            )
            tokens = interleaved_tokens[:, indices]
            codebook_tokens.append(tokens)

        return codebook_tokens


class StreamingCollationState:
    """Manages state for streaming audio collation with delay patterns."""

    def __init__(
        self,
        num_codebooks: int = 8,
        chunk_overlap_frames: int = 100,
    ):
        """Initialize streaming collation state.

        Args:
            num_codebooks: Number of RVQ codebooks
            chunk_overlap_frames: Number of frames to maintain for overlap
        """
        self.num_codebooks = num_codebooks
        self.chunk_overlap_frames = chunk_overlap_frames
        self.tail_context: Optional[torch.Tensor] = None
        self.delay_offsets: Optional[torch.Tensor] = None

    def update_with_chunk(
        self,
        chunk_tokens: torch.Tensor,
        delay_pattern: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update state with new chunk and return processed tokens.

        Args:
            chunk_tokens: New chunk tokens [batch_size, chunk_length]
            delay_pattern: Delay pattern [num_codebooks, seq_len]

        Returns:
            Tuple of (processed_tokens, carry_over_state)
        """
        # For now, implement a simple passthrough
        # Full streaming implementation would maintain delay state across chunks
        return chunk_tokens, chunk_tokens[-self.chunk_overlap_frames :]


class HiggsAudioSampleCollator:
    """Collator for HiggsAudio samples supporting mixed text and audio inputs.

    This collator efficiently batches samples containing:
    - input_ids: Text token sequences
    - mel: Mel-spectrogram features [n_mels, time_frames]
    - audio_meta: Audio metadata (duration, sample_rate, etc.)
    - mode flags: Processing mode indicators

    Features:
    - Right-padding for variable-length sequences
    - Configurable padding multiples for optimal tensor shapes
    - Attention mask generation for padded sequences
    - Special audio token insertion (BOS/EOS)
    - Rich metadata preservation
    - PyTorch DataLoader compatibility
    """

    def __init__(
        self,
        config: HiggsAudioConfig,
        pad_multiple: int = 8,
        text_pad_value: int = None,
        audio_pad_value: float = 0.0,
        return_attention_mask: bool = True,
        return_length_tensors: bool = True,
        max_length: Optional[int] = None,
        max_audio_length: Optional[int] = None,
        enable_delay_pattern: bool = False,
        delay_pattern_strategy: str = "linear",
        num_codebooks: int = 8,
    ):
        """Initialize the collator.

        Args:
            config: HiggsAudio configuration
            pad_multiple: Pad sequence lengths to multiples of this value
            text_pad_value: Padding value for text sequences (defaults to config.pad_token_id)
            audio_pad_value: Padding value for audio/mel features
            return_attention_mask: Whether to generate attention masks
            return_length_tensors: Whether to include original length tensors
            max_length: Maximum sequence length (None for dynamic)
            max_audio_length: Maximum audio sequence length (None for dynamic)
            enable_delay_pattern: Whether to apply RVQ delay patterns
            delay_pattern_strategy: Strategy for delay pattern ("linear", "exponential", "custom")
            num_codebooks: Number of RVQ codebooks for delay patterns
        """
        self.config = config
        self.pad_multiple = pad_multiple
        self.text_pad_value = text_pad_value if text_pad_value is not None else config.pad_token_id
        self.audio_pad_value = audio_pad_value
        self.return_attention_mask = return_attention_mask
        self.return_length_tensors = return_length_tensors
        self.max_length = max_length
        self.max_audio_length = max_audio_length

        # Cache frequently used token IDs
        self.audio_bos_token_id = config.audio_bos_token_id
        self.audio_eos_token_id = config.audio_eos_token_id
        self.audio_out_bos_token_id = config.audio_out_bos_token_id
        self.audio_in_token_idx = config.audio_in_token_idx
        self.audio_out_token_idx = config.audio_out_token_idx

        # Initialize delay pattern components
        self.enable_delay_pattern = enable_delay_pattern
        if self.enable_delay_pattern:
            self.delay_pattern_provider = DelayPatternProvider(
                strategy=delay_pattern_strategy,
                num_codebooks=num_codebooks,
            )
            self.audio_token_utils = AudioTokenUtils(
                num_codebooks=num_codebooks,
                pad_token_id=self.text_pad_value,
                audio_stream_bos_id=config.audio_bos_token_id,
                audio_stream_eos_id=config.audio_eos_token_id,
            )
            self.streaming_state = StreamingCollationState(
                num_codebooks=num_codebooks,
            )
        else:
            self.delay_pattern_provider = None
            self.audio_token_utils = None
            self.streaming_state = None

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of samples.

        Args:
            samples: List of sample dictionaries containing:
                - input_ids: Text token sequence [seq_len] (required)
                - mel: Mel-spectrogram [n_mels, time_frames] (optional)
                - audio_meta: Audio metadata dict (optional)
                - mode: Processing mode string (optional)
                - sample_id: Sample identifier (optional)

        Returns:
            Dictionary with batched tensors:
                - input_ids: [batch_size, max_seq_len]
                - attention_mask: [batch_size, max_seq_len] (if enabled)
                - text_lengths: [batch_size] (if enabled)
                - mel_features: [batch_size, n_mels, max_audio_len] (if present)
                - audio_attention_mask: [batch_size, max_audio_len] (if present)
                - audio_lengths: [batch_size] (if present)
                - batch_metadata: List of per-sample metadata
        """
        batch_size = len(samples)

        if batch_size == 0:
            return {}

        # Separate text and audio data
        text_sequences = []
        mel_features = []
        batch_metadata = []

        # Extract sequences and metadata
        for i, sample in enumerate(samples):
            # Process text input (required)
            if "input_ids" not in sample:
                raise ValueError(f"Sample {i} missing required 'input_ids' field")

            input_ids = sample["input_ids"]
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids, dtype=torch.long)
            elif isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.long()
            else:
                raise ValueError(f"input_ids must be list or tensor, got {type(input_ids)}")

            text_sequences.append(input_ids)

            # Process audio input (optional)
            mel = sample.get("mel", None)
            if mel is not None:
                if isinstance(mel, list):
                    mel = torch.tensor(mel, dtype=torch.float)
                elif isinstance(mel, torch.Tensor):
                    mel = mel.float()
                else:
                    raise ValueError(f"mel must be list or tensor, got {type(mel)}")

                # Ensure correct shape [n_mels, time_frames]
                if mel.dim() == 3 and mel.size(0) == 1:
                    mel = mel.squeeze(0)  # Remove batch dimension if present
                elif mel.dim() != 2:
                    raise ValueError(f"mel must be 2D [n_mels, time], got shape {mel.shape}")

                mel_features.append(mel)
            else:
                mel_features.append(None)

            # Collect metadata
            metadata = {
                "sample_id": sample.get("sample_id", i),
                "original_text_length": len(input_ids),
                "mode": sample.get("mode", "text"),
                "audio_meta": sample.get("audio_meta", {}),
            }
            if mel is not None:
                metadata["original_audio_length"] = mel.size(1)

            batch_metadata.append(metadata)

        # Collate text sequences
        batch_dict = self._collate_text_sequences(text_sequences, batch_metadata)

        # Collate audio features if present
        if any(mel is not None for mel in mel_features):
            audio_dict = self._collate_audio_features(mel_features, batch_metadata)
            batch_dict.update(audio_dict)

        # Add metadata
        batch_dict["batch_metadata"] = batch_metadata

        return batch_dict

    def _collate_text_sequences(
        self, sequences: List[torch.Tensor], metadata: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """Collate text sequences with padding and attention masks.

        Args:
            sequences: List of text token tensors [seq_len]
            metadata: List of metadata dictionaries

        Returns:
            Dictionary with collated text tensors
        """
        batch_size = len(sequences)

        # Find maximum length and apply padding multiple
        lengths = [seq.size(0) for seq in sequences]
        max_length = max(lengths)

        # Apply length constraints
        if self.max_length is not None:
            max_length = min(max_length, self.max_length)

        # Round up to multiple
        if self.pad_multiple > 1:
            max_length = (
                (max_length + self.pad_multiple - 1) // self.pad_multiple
            ) * self.pad_multiple

        # Create padded tensor
        padded_sequences = torch.full(
            (batch_size, max_length), self.text_pad_value, dtype=torch.long
        )

        # Fill sequences and optionally add special tokens
        for i, seq in enumerate(sequences):
            seq_len = min(seq.size(0), max_length)

            # Check if we need to add audio special tokens
            mode = metadata[i].get("mode", "text")

            if mode == "audio_input" and self.audio_bos_token_id is not None:
                # Add audio BOS token at start
                if seq_len < max_length:
                    padded_sequences[i, 0] = self.audio_bos_token_id
                    # Copy remaining sequence
                    copy_len = min(seq_len, max_length - 1)
                    if copy_len > 0:
                        padded_sequences[i, 1 : 1 + copy_len] = seq[:copy_len]
                    # Add EOS token if there's space
                    if copy_len + 1 < max_length and self.audio_eos_token_id is not None:
                        padded_sequences[i, copy_len + 1] = self.audio_eos_token_id
                else:
                    padded_sequences[i, :seq_len] = seq[:seq_len]
            else:
                # Regular text sequence
                padded_sequences[i, :seq_len] = seq[:seq_len]

            # Update metadata with actual used length
            metadata[i]["used_text_length"] = seq_len

        result = {"input_ids": padded_sequences}

        # Add attention mask if requested
        if self.return_attention_mask:
            attention_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
            for i, length in enumerate(lengths):
                actual_length = min(length, max_length)
                attention_mask[i, :actual_length] = True
            result["attention_mask"] = attention_mask

        # Add length tensor if requested
        if self.return_length_tensors:
            text_lengths = torch.tensor(
                [min(length, max_length) for length in lengths], dtype=torch.long
            )
            result["text_lengths"] = text_lengths

        return result

    def _collate_audio_features(
        self, features: List[Optional[torch.Tensor]], metadata: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """Collate audio mel features with padding and attention masks.

        Args:
            features: List of mel feature tensors [n_mels, time_frames] or None
            metadata: List of metadata dictionaries

        Returns:
            Dictionary with collated audio tensors
        """
        # Filter out None features and get dimensions
        valid_features = [f for f in features if f is not None]
        if not valid_features:
            return {}

        batch_size = len(features)
        n_mels = valid_features[0].size(0)

        # Verify all features have same number of mel bins
        for i, feat in enumerate(valid_features):
            if feat.size(0) != n_mels:
                raise ValueError(
                    f"Inconsistent mel dimensions: expected {n_mels}, "
                    f"got {feat.size(0)} for sample {i}"
                )

        # Find maximum time length
        time_lengths = []
        for feat in features:
            if feat is not None:
                time_lengths.append(feat.size(1))
            else:
                time_lengths.append(0)

        max_time = max(time_lengths) if time_lengths else 0

        # Apply length constraints
        if self.max_audio_length is not None:
            max_time = min(max_time, self.max_audio_length)

        # Round up to multiple
        if self.pad_multiple > 1:
            max_time = ((max_time + self.pad_multiple - 1) // self.pad_multiple) * self.pad_multiple

        # Create padded tensor
        padded_features = torch.full(
            (batch_size, n_mels, max_time), self.audio_pad_value, dtype=torch.float
        )

        # Fill features
        for i, feat in enumerate(features):
            if feat is not None:
                time_len = min(feat.size(1), max_time)
                padded_features[i, :, :time_len] = feat[:, :time_len]

                # Update metadata
                metadata[i]["used_audio_length"] = time_len
            else:
                metadata[i]["used_audio_length"] = 0

        result = {"mel_features": padded_features}

        # Add audio attention mask if requested
        if self.return_attention_mask:
            audio_attention_mask = torch.zeros(batch_size, max_time, dtype=torch.bool)
            for i, length in enumerate(time_lengths):
                if length > 0:
                    actual_length = min(length, max_time)
                    audio_attention_mask[i, :actual_length] = True
            result["audio_attention_mask"] = audio_attention_mask

        # Add audio length tensor if requested
        if self.return_length_tensors:
            audio_lengths = torch.tensor(
                [min(length, max_time) for length in time_lengths], dtype=torch.long
            )
            result["audio_lengths"] = audio_lengths

        return result

    def _apply_delay_pattern_to_batch(
        self,
        batch_dict: Dict[str, torch.Tensor],
        audio_tokens: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Apply RVQ delay pattern to audio tokens in batch.

        Args:
            batch_dict: Current batch dictionary
            audio_tokens: Optional list of multi-codebook audio tokens

        Returns:
            Updated batch dictionary with delay pattern applied
        """
        if not self.enable_delay_pattern or audio_tokens is None:
            return batch_dict

        # Apply delay patterns to audio tokens
        delayed_audio_tokens = []
        delay_patterns = []

        for tokens in audio_tokens:
            if tokens is not None and tokens.dim() >= 2:
                # Assume tokens are [n_codebooks, seq_len] or [batch_size, n_codebooks, seq_len]
                if tokens.dim() == 3:
                    batch_size, n_codebooks, seq_len = tokens.shape
                else:
                    n_codebooks, seq_len = tokens.shape
                    batch_size = 1
                    tokens = tokens.unsqueeze(0)

                # Generate delay pattern for this sequence
                delay_pattern = self.delay_pattern_provider.generate_delay_pattern(
                    seq_len, n_codebooks
                )

                # Apply delay pattern to each sample in batch
                delayed_batch = []
                for b in range(batch_size):
                    sample_tokens = tokens[b]  # [n_codebooks, seq_len]
                    delayed_tokens = self.delay_pattern_provider.apply_delay_pattern(
                        sample_tokens, delay_pattern
                    )
                    delayed_batch.append(delayed_tokens)

                delayed_tokens = torch.stack(delayed_batch, dim=0)
                delayed_audio_tokens.append(delayed_tokens)
                delay_patterns.append(delay_pattern)
            else:
                delayed_audio_tokens.append(tokens)
                delay_patterns.append(None)

        # Add delay pattern info to batch
        if any(dp is not None for dp in delay_patterns):
            batch_dict["audio_delay_patterns"] = delay_patterns
            batch_dict["delayed_audio_tokens"] = delayed_audio_tokens

        return batch_dict

    def _validate_delay_pattern_constraints(
        self,
        audio_tokens: torch.Tensor,
        delay_pattern: torch.Tensor,
    ) -> bool:
        """Validate that delay pattern maintains causality constraints.

        Args:
            audio_tokens: Multi-codebook tokens [n_codebooks, seq_len]
            delay_pattern: Delay pattern [n_codebooks, seq_len]

        Returns:
            True if constraints are satisfied
        """
        if not self.enable_delay_pattern:
            return True

        n_codebooks, seq_len = audio_tokens.shape

        # Check that delays are non-negative and non-decreasing
        for k in range(n_codebooks - 1):
            if delay_pattern[k].max() > delay_pattern[k + 1].min():
                logger.warning(
                    f"Delay pattern violates causality: codebook {k} delay exceeds codebook {k + 1}"
                )
                return False

        # Check maximum delay doesn't exceed sequence length
        max_delay = delay_pattern.max().item()
        if max_delay >= seq_len:
            logger.warning(f"Maximum delay {max_delay} exceeds sequence length {seq_len}")
            return False

        return True

    def _pad_to_multiple(self, length: int) -> int:
        """Pad length to the nearest multiple."""
        if self.pad_multiple <= 1:
            return length
        return ((length + self.pad_multiple - 1) // self.pad_multiple) * self.pad_multiple


def create_higgs_audio_collator(
    config: HiggsAudioConfig, pad_multiple: int = 8, **kwargs
) -> HiggsAudioSampleCollator:
    """Create a HiggsAudio sample collator with standard settings.

    Args:
        config: HiggsAudio configuration
        pad_multiple: Padding multiple for sequence lengths
        **kwargs: Additional arguments for HiggsAudioSampleCollator

    Returns:
        Configured HiggsAudioSampleCollator instance
    """
    return HiggsAudioSampleCollator(config=config, pad_multiple=pad_multiple, **kwargs)


# Utility functions for sample preparation
def prepare_text_sample(
    text_tokens: Union[List[int], torch.Tensor],
    sample_id: Optional[Union[str, int]] = None,
    mode: str = "text",
) -> Dict[str, Any]:
    """Prepare a text-only sample for collation.

    Args:
        text_tokens: Text token sequence
        sample_id: Optional sample identifier
        mode: Processing mode

    Returns:
        Sample dictionary ready for collation
    """
    return {
        "input_ids": text_tokens,
        "sample_id": sample_id,
        "mode": mode,
    }


def prepare_audio_sample(
    text_tokens: Union[List[int], torch.Tensor],
    mel_features: torch.Tensor,
    audio_meta: Optional[Dict[str, Any]] = None,
    sample_id: Optional[Union[str, int]] = None,
    mode: str = "audio_input",
) -> Dict[str, Any]:
    """Prepare a sample with both text and audio for collation.

    Args:
        text_tokens: Text token sequence
        mel_features: Mel-spectrogram features [n_mels, time_frames]
        audio_meta: Audio metadata dictionary
        sample_id: Optional sample identifier
        mode: Processing mode

    Returns:
        Sample dictionary ready for collation
    """
    sample = {
        "input_ids": text_tokens,
        "mel": mel_features,
        "sample_id": sample_id,
        "mode": mode,
    }

    if audio_meta is not None:
        sample["audio_meta"] = audio_meta

    return sample


def prepare_mixed_batch(
    text_samples: List[Union[List[int], torch.Tensor]],
    audio_samples: List[Tuple[Union[List[int], torch.Tensor], torch.Tensor]],
    sample_ids: Optional[List[Union[str, int]]] = None,
) -> List[Dict[str, Any]]:
    """Prepare a mixed batch with both text-only and audio samples.

    Args:
        text_samples: List of text token sequences
        audio_samples: List of (text_tokens, mel_features) tuples
        sample_ids: Optional list of sample identifiers

    Returns:
        List of sample dictionaries ready for collation
    """
    samples = []

    # Add text-only samples
    for i, text_tokens in enumerate(text_samples):
        sample_id = sample_ids[i] if sample_ids else f"text_{i}"
        samples.append(prepare_text_sample(text_tokens, sample_id, "text"))

    # Add audio samples
    for i, (text_tokens, mel_features) in enumerate(audio_samples):
        sample_id = sample_ids[len(text_samples) + i] if sample_ids else f"audio_{i}"
        samples.append(prepare_audio_sample(text_tokens, mel_features, sample_id=sample_id))

    return samples


def validate_collated_batch(batch: Dict[str, torch.Tensor]) -> bool:
    """Validate a collated batch for consistency.

    Args:
        batch: Collated batch dictionary

    Returns:
        True if valid, False otherwise
    """
    try:
        # Check required fields
        if "input_ids" not in batch:
            logger.error("Missing required field: input_ids")
            return False

        batch_size = batch["input_ids"].size(0)

        # Check attention mask consistency
        if "attention_mask" in batch:
            if batch["attention_mask"].size(0) != batch_size:
                logger.error("Attention mask batch size mismatch")
                return False

            if batch["attention_mask"].size(1) != batch["input_ids"].size(1):
                logger.error("Attention mask sequence length mismatch")
                return False

        # Check audio feature consistency
        if "mel_features" in batch:
            if batch["mel_features"].size(0) != batch_size:
                logger.error("Mel features batch size mismatch")
                return False

            if "audio_attention_mask" in batch:
                if batch["audio_attention_mask"].size(0) != batch_size:
                    logger.error("Audio attention mask batch size mismatch")
                    return False

                if batch["audio_attention_mask"].size(1) != batch["mel_features"].size(2):
                    logger.error("Audio attention mask time dimension mismatch")
                    return False

        # Check length tensors
        if "text_lengths" in batch:
            if batch["text_lengths"].size(0) != batch_size:
                logger.error("Text lengths batch size mismatch")
                return False

        if "audio_lengths" in batch:
            if batch["audio_lengths"].size(0) != batch_size:
                logger.error("Audio lengths batch size mismatch")
                return False

        # Check metadata
        if "batch_metadata" in batch:
            if len(batch["batch_metadata"]) != batch_size:
                logger.error("Metadata batch size mismatch")
                return False

        return True

    except Exception as e:
        logger.error(f"Batch validation error: {e}")
        return False


# TensorRT-LLM Specific Optimizations
class TensorRTBatchOptimizer:
    """TensorRT-LLM specific batch optimizations for HiggsAudio.

    This class provides optimizations specifically for TensorRT-LLM including:
    - Dynamic sequence length bucketing
    - Efficient padding strategies
    - Packed attention masks
    - Memory-aware batching
    """

    def __init__(
        self,
        bucket_boundaries: Optional[List[int]] = None,
        max_batch_size: int = 32,
        memory_threshold_gb: float = 8.0,
        enable_packed_attention: bool = True,
        enable_pinned_memory: bool = True,
    ):
        """Initialize TensorRT batch optimizer.

        Args:
            bucket_boundaries: Sequence length boundaries for bucketing
            max_batch_size: Maximum batch size
            memory_threshold_gb: Memory threshold for adaptive batching
            enable_packed_attention: Whether to use packed attention masks
            enable_pinned_memory: Whether to use pinned memory allocation
        """
        self.bucket_boundaries = bucket_boundaries or [32, 64, 128, 256, 512, 1024]
        self.max_batch_size = max_batch_size
        self.memory_threshold_gb = memory_threshold_gb
        self.enable_packed_attention = enable_packed_attention
        self.enable_pinned_memory = enable_pinned_memory

        # Cache for bucket assignments
        self._bucket_cache = {}

    def get_bucket_for_length(self, length: int) -> int:
        """Get the appropriate bucket for a given sequence length.

        Args:
            length: Sequence length

        Returns:
            Bucket size (rounded up to next boundary)
        """
        if length in self._bucket_cache:
            return self._bucket_cache[length]

        # Find the smallest bucket that fits this length
        bucket_size = length
        for boundary in self.bucket_boundaries:
            if length <= boundary:
                bucket_size = boundary
                break
        else:
            # If length exceeds all boundaries, use the next power of 2
            bucket_size = 1
            while bucket_size < length:
                bucket_size *= 2

        self._bucket_cache[length] = bucket_size
        return bucket_size

    def organize_samples_by_bucket(
        self,
        samples: List[Dict[str, Any]],
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Organize samples into buckets by sequence length.

        Args:
            samples: List of samples to organize

        Returns:
            Dictionary mapping bucket size to list of samples
        """
        buckets = {}

        for sample in samples:
            # Determine sequence length
            if "input_ids" in sample:
                if isinstance(sample["input_ids"], list):
                    seq_len = len(sample["input_ids"])
                else:
                    seq_len = sample["input_ids"].shape[-1]
            else:
                seq_len = 32  # Default bucket

            bucket_size = self.get_bucket_for_length(seq_len)

            if bucket_size not in buckets:
                buckets[bucket_size] = []
            buckets[bucket_size].append(sample)

        return buckets

    def create_packed_attention_mask(
        self,
        attention_mask: torch.Tensor,
        pack_dim: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create packed attention mask for efficient computation.

        Args:
            attention_mask: Original attention mask [batch_size, seq_len]
            pack_dim: Dimension to pack along

        Returns:
            Tuple of (packed_mask, pack_indices)
        """
        batch_size, seq_len = attention_mask.shape

        # Find actual lengths for each sequence
        actual_lengths = attention_mask.sum(dim=1).long()

        # Calculate total packed length
        total_packed_len = actual_lengths.sum().item()

        # Create packed mask
        packed_mask = torch.ones(
            total_packed_len, dtype=attention_mask.dtype, device=attention_mask.device
        )

        # Create pack indices for reconstruction
        pack_indices = torch.zeros(batch_size + 1, dtype=torch.long, device=attention_mask.device)
        pack_indices[1:] = torch.cumsum(actual_lengths, dim=0)

        return packed_mask, pack_indices

    def allocate_pinned_memory(
        self,
        tensor_shapes: List[Tuple[int, ...]],
        dtype: torch.dtype = torch.float32,
    ) -> List[torch.Tensor]:
        """Allocate pinned memory tensors for faster GPU transfer.

        Args:
            tensor_shapes: List of tensor shapes to allocate
            dtype: Data type for tensors

        Returns:
            List of pinned memory tensors
        """
        if not self.enable_pinned_memory:
            return [torch.empty(shape, dtype=dtype) for shape in tensor_shapes]

        tensors = []
        for shape in tensor_shapes:
            tensor = torch.empty(shape, dtype=dtype, pin_memory=True)
            tensors.append(tensor)

        return tensors

    def estimate_memory_usage(
        self,
        batch_size: int,
        seq_len: int,
        audio_len: Optional[int] = None,
        n_mels: int = 128,
        precision_bytes: int = 4,  # float32
    ) -> float:
        """Estimate memory usage for a batch in GB.

        Args:
            batch_size: Batch size
            seq_len: Text sequence length
            audio_len: Audio sequence length (optional)
            n_mels: Number of mel frequency bins
            precision_bytes: Bytes per element (4 for float32)

        Returns:
            Estimated memory usage in GB
        """
        # Text memory: input_ids + attention_mask
        text_memory = batch_size * seq_len * 2 * precision_bytes

        # Audio memory (if present)
        audio_memory = 0
        if audio_len is not None:
            # mel features + attention mask
            audio_memory = batch_size * n_mels * audio_len * precision_bytes
            audio_memory += batch_size * audio_len * precision_bytes  # attention mask

        # Additional overhead (embeddings, activations, etc.)
        overhead_multiplier = 3.0

        total_bytes = (text_memory + audio_memory) * overhead_multiplier
        return total_bytes / (1024**3)  # Convert to GB

    def get_optimal_batch_size(
        self,
        samples: List[Dict[str, Any]],
        target_memory_gb: Optional[float] = None,
    ) -> int:
        """Determine optimal batch size based on memory constraints.

        Args:
            samples: List of samples to batch
            target_memory_gb: Target memory usage (defaults to threshold)

        Returns:
            Optimal batch size
        """
        if target_memory_gb is None:
            target_memory_gb = self.memory_threshold_gb

        # Estimate memory for different batch sizes
        for batch_size in range(1, min(len(samples), self.max_batch_size) + 1):
            # Take first batch_size samples as representative
            sample_batch = samples[:batch_size]

            # Get average sequence lengths
            text_lens = []
            audio_lens = []

            for sample in sample_batch:
                if "input_ids" in sample:
                    if isinstance(sample["input_ids"], list):
                        text_lens.append(len(sample["input_ids"]))
                    else:
                        text_lens.append(sample["input_ids"].shape[-1])

                if "mel" in sample and sample["mel"] is not None:
                    if isinstance(sample["mel"], torch.Tensor):
                        audio_lens.append(sample["mel"].shape[-1])
                    else:
                        audio_lens.append(50)  # Default

            avg_text_len = int(sum(text_lens) / len(text_lens)) if text_lens else 32
            avg_audio_len = int(sum(audio_lens) / len(audio_lens)) if audio_lens else None

            # Estimate memory usage
            estimated_memory = self.estimate_memory_usage(batch_size, avg_text_len, avg_audio_len)

            if estimated_memory > target_memory_gb:
                return max(1, batch_size - 1)

        return min(len(samples), self.max_batch_size)


class TensorRTOptimizedCollator(HiggsAudioSampleCollator):
    """TensorRT-LLM optimized version of HiggsAudioSampleCollator.

    This collator includes TensorRT-specific optimizations:
    - Sequence length bucketing
    - Memory-aware batching
    - Packed attention masks
    - Pinned memory allocation
    """

    def __init__(
        self, config: HiggsAudioConfig, optimizer: Optional[TensorRTBatchOptimizer] = None, **kwargs
    ):
        """Initialize optimized collator.

        Args:
            config: HiggsAudio configuration
            optimizer: TensorRT batch optimizer (created if None)
            **kwargs: Additional arguments for base collator
        """
        super().__init__(config, **kwargs)

        self.optimizer = optimizer or TensorRTBatchOptimizer()

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate samples with TensorRT optimizations.

        Args:
            samples: List of samples to collate

        Returns:
            Optimized batch dictionary
        """
        # Use bucketing if enabled and we have multiple samples
        if len(samples) > 1:
            # Organize by bucket
            buckets = self.optimizer.organize_samples_by_bucket(samples)

            # Process largest bucket first (usually more efficient)
            largest_bucket = max(buckets.keys())
            if len(buckets[largest_bucket]) == len(samples):
                # All samples fit in one bucket, proceed normally
                batch_dict = super().__call__(samples)
            else:
                # Multiple buckets - process the largest one
                # In a real implementation, you might want to handle this differently
                batch_dict = super().__call__(buckets[largest_bucket])
        else:
            batch_dict = super().__call__(samples)

        # Apply packed attention if enabled
        if (
            self.optimizer.enable_packed_attention
            and "attention_mask" in batch_dict
            and batch_dict["attention_mask"].numel() > 1000
        ):  # Only for larger batches
            packed_mask, pack_indices = self.optimizer.create_packed_attention_mask(
                batch_dict["attention_mask"]
            )
            batch_dict["packed_attention_mask"] = packed_mask
            batch_dict["pack_indices"] = pack_indices

        # Add memory usage info
        if hasattr(self.optimizer, "estimate_memory_usage"):
            batch_size = batch_dict["input_ids"].shape[0]
            seq_len = batch_dict["input_ids"].shape[1]
            audio_len = (
                batch_dict.get("mel_features", torch.empty(0, 0, 0)).shape[-1]
                if "mel_features" in batch_dict
                else None
            )

            estimated_memory = self.optimizer.estimate_memory_usage(batch_size, seq_len, audio_len)
            batch_dict["estimated_memory_gb"] = estimated_memory

        return batch_dict


def create_tensorrt_optimized_collator(
    config: HiggsAudioConfig,
    bucket_boundaries: Optional[List[int]] = None,
    max_batch_size: int = 32,
    **kwargs,
) -> TensorRTOptimizedCollator:
    """Create a TensorRT-optimized HiggsAudio collator.

    Args:
        config: HiggsAudio configuration
        bucket_boundaries: Sequence length boundaries for bucketing
        max_batch_size: Maximum batch size
        **kwargs: Additional arguments for collator

    Returns:
        TensorRT-optimized collator instance
    """
    optimizer = TensorRTBatchOptimizer(
        bucket_boundaries=bucket_boundaries,
        max_batch_size=max_batch_size,
    )

    return TensorRTOptimizedCollator(config=config, optimizer=optimizer, **kwargs)
