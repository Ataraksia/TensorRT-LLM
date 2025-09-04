"""Audio and text preprocessing utilities for HiggsAudio TensorRT-LLM model.

This module provides reusable preprocessing functions for audio and text inputs,
including Whisper-compatible log-mel feature extraction, text normalization,
and long audio chunking with overlap handling.
"""

import logging
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Audio preprocessing utilities for HiggsAudio model."""

    def __init__(
        self,
        target_sr: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        fmin: float = 0.0,
        fmax: float = 8000.0,
        min_db: float = -80.0,
        device: Optional[torch.device] = None,
    ):
        """Initialize audio preprocessor with Whisper-compatible settings.

        Args:
            target_sr: Target sample rate (16kHz for Whisper)
            n_mels: Number of mel frequency bins (80 for Whisper)
            n_fft: FFT window size
            hop_length: Hop length between frames (160 for 10ms @ 16kHz)
            win_length: Window length for STFT
            fmin: Minimum frequency for mel filterbank
            fmax: Maximum frequency for mel filterbank
            min_db: Minimum dB value for log compression
            device: Device for computations
        """
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.fmin = fmin
        self.fmax = fmax
        self.min_db = min_db
        self.device = device or torch.device("cpu")

        # Create mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window_fn=torch.hann_window,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
            power=2.0,  # Power spectrogram
            normalized=False,
            center=True,
            pad_mode="reflect",
        )

        # Move to device
        self.mel_transform = self.mel_transform.to(self.device)

    def load_audio(
        self,
        input_audio: Union[str, Path, np.ndarray, torch.Tensor],
        mono: bool = True,
        normalize: bool = True,
        remove_dc: bool = True,
    ) -> Tuple[torch.Tensor, int]:
        """Load audio from various input formats.

        Args:
            input_audio: Audio input (file path, numpy array, or torch tensor)
            mono: Convert to mono if True
            normalize: Normalize to [-1, 1] range if True
            remove_dc: Remove DC offset if True

        Returns:
            Tuple of (waveform, sample_rate) where waveform is [channels, time]
        """
        if isinstance(input_audio, (str, Path)):
            # Load from file
            waveform, sample_rate = torchaudio.load(str(input_audio))
        elif isinstance(input_audio, np.ndarray):
            # Convert from numpy
            waveform = torch.from_numpy(input_audio)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)  # Add channel dimension
            sample_rate = self.target_sr  # Assume target sample rate
        elif isinstance(input_audio, torch.Tensor):
            # Use tensor directly
            waveform = input_audio.clone()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)  # Add channel dimension
            sample_rate = self.target_sr  # Assume target sample rate
        else:
            raise ValueError(f"Unsupported audio input type: {type(input_audio)}")

        # Ensure float32
        waveform = waveform.float()

        # Convert to mono if requested
        if mono and waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Remove DC offset
        if remove_dc:
            waveform = waveform - torch.mean(waveform, dim=-1, keepdim=True)

        # Resample if needed
        if sample_rate != self.target_sr:
            resampler = T.Resample(
                orig_freq=sample_rate, new_freq=self.target_sr, resampling_method="sinc_interp_hann"
            )
            waveform = resampler(waveform)
            sample_rate = self.target_sr

        # Normalize to [-1, 1]
        if normalize:
            max_val = torch.max(torch.abs(waveform))
            if max_val > 0:
                waveform = torch.clamp(waveform / max_val, -1.0, 1.0)

        return waveform, sample_rate

    def compute_whisper_mel(self, waveform: torch.Tensor, sr: Optional[int] = None) -> torch.Tensor:
        """Compute Whisper-compatible log-mel spectrogram.

        Args:
            waveform: Input waveform [channels, time]
            sr: Sample rate (defaults to target_sr)

        Returns:
            Log-mel spectrogram [n_mels, time_frames]
        """
        if sr is None:
            sr = self.target_sr

        # Ensure correct sample rate
        if sr != self.target_sr:
            resampler = T.Resample(
                orig_freq=sr, new_freq=self.target_sr, resampling_method="sinc_interp_hann"
            )
            waveform = resampler(waveform)

        # Convert to mono if multi-channel
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Move to device
        waveform = waveform.to(self.device)

        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)

        # Take log (base 10) and clamp
        mel_spec = torch.log10(torch.clamp(mel_spec, min=1e-10))
        mel_spec = torch.clamp(mel_spec, min=self.min_db / 10.0)

        # Remove channel dimension and return [n_mels, time]
        mel_spec = mel_spec.squeeze(0)

        return mel_spec

    def get_audio_length(self, waveform: torch.Tensor, sr: int) -> float:
        """Get audio duration in seconds.

        Args:
            waveform: Audio waveform [channels, time]
            sr: Sample rate

        Returns:
            Duration in seconds
        """
        return waveform.size(-1) / sr

    def get_mel_frames(self, audio_length_seconds: float) -> int:
        """Get number of mel frames for given audio duration.

        Args:
            audio_length_seconds: Audio duration in seconds

        Returns:
            Number of mel frames
        """
        audio_samples = int(audio_length_seconds * self.target_sr)
        return (audio_samples + self.hop_length - 1) // self.hop_length

    def validate_audio(self, waveform: torch.Tensor, sr: int) -> bool:
        """Validate audio tensor properties.

        Args:
            waveform: Audio waveform [channels, time]
            sr: Sample rate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(waveform, torch.Tensor):
            return False

        if waveform.ndim != 2:
            return False

        if waveform.size(-1) == 0:
            return False

        if sr <= 0:
            return False

        # Check for NaN or inf values
        if torch.any(torch.isnan(waveform)) or torch.any(torch.isinf(waveform)):
            return False

        return True


class TextPreprocessor:
    """Text preprocessing utilities for HiggsAudio model."""

    def __init__(self, normalize_unicode: bool = True, collapse_whitespace: bool = True):
        """Initialize text preprocessor.

        Args:
            normalize_unicode: Apply Unicode NFKC normalization
            collapse_whitespace: Collapse multiple whitespace characters
        """
        self.normalize_unicode = normalize_unicode
        self.collapse_whitespace = collapse_whitespace

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for tokenization.

        Args:
            text: Input text string

        Returns:
            Preprocessed text string
        """
        if not isinstance(text, str):
            text = str(text)

        # Unicode normalization
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        # Collapse whitespace
        if self.collapse_whitespace:
            # Replace multiple whitespace with single space
            import re

            text = re.sub(r"\s+", " ", text)
            text = text.strip()

        return text

    def get_text_length(self, text: str) -> int:
        """Get text length in characters.

        Args:
            text: Input text

        Returns:
            Number of characters
        """
        return len(text)

    def validate_text(self, text: str) -> bool:
        """Validate text input.

        Args:
            text: Input text

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(text, str):
            return False

        if len(text.strip()) == 0:
            return False

        return True


def create_preprocessors(
    target_sr: int = 16000, device: Optional[torch.device] = None
) -> Tuple[AudioPreprocessor, TextPreprocessor]:
    """Create audio and text preprocessors with default settings.

    Args:
        target_sr: Target sample rate for audio
        device: Device for audio computations

    Returns:
        Tuple of (audio_preprocessor, text_preprocessor)
    """
    audio_preprocessor = AudioPreprocessor(target_sr=target_sr, device=device)
    text_preprocessor = TextPreprocessor()

    return audio_preprocessor, text_preprocessor


# Utility functions for common operations
def load_audio_file(
    file_path: Union[str, Path], target_sr: int = 16000, device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load audio file and compute mel spectrogram.

    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        device: Device for computations

    Returns:
        Tuple of (waveform, mel_spectrogram)
    """
    preprocessor = AudioPreprocessor(target_sr=target_sr, device=device)
    waveform, sr = preprocessor.load_audio(file_path)
    mel_spec = preprocessor.compute_whisper_mel(waveform, sr)

    return waveform, mel_spec


def preprocess_batch_audio(
    audio_list: List[Union[str, Path, np.ndarray, torch.Tensor]],
    target_sr: int = 16000,
    device: Optional[torch.device] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Preprocess a batch of audio inputs.

    Args:
        audio_list: List of audio inputs
        target_sr: Target sample rate
        device: Device for computations

    Returns:
        List of (waveform, mel_spectrogram) tuples
    """
    preprocessor = AudioPreprocessor(target_sr=target_sr, device=device)
    results = []

    for audio_input in audio_list:
        try:
            waveform, sr = preprocessor.load_audio(audio_input)
            mel_spec = preprocessor.compute_whisper_mel(waveform, sr)
            results.append((waveform, mel_spec))
        except Exception as e:
            logger.error(f"Failed to preprocess audio {audio_input}: {e}")
            results.append((None, None))

    return results


def preprocess_batch_text(
    text_list: List[str], normalize_unicode: bool = True, collapse_whitespace: bool = True
) -> List[str]:
    """Preprocess a batch of text inputs.

    Args:
        text_list: List of text strings
        normalize_unicode: Apply Unicode normalization
        collapse_whitespace: Collapse whitespace

    Returns:
        List of preprocessed text strings
    """
    preprocessor = TextPreprocessor(
        normalize_unicode=normalize_unicode, collapse_whitespace=collapse_whitespace
    )

    results = []
    for text in text_list:
        try:
            processed_text = preprocessor.preprocess_text(text)
            results.append(processed_text)
        except Exception as e:
            logger.error(f"Failed to preprocess text '{text}': {e}")
            results.append("")

    return results


class AudioChunker:
    """Utility class for chunking long audio sequences with overlap.

    This class handles automatic chunking of long audio sequences (>30 seconds)
    into smaller, manageable chunks with configurable overlap for continuity.
    """

    def __init__(
        self,
        chunk_duration_seconds: float = 30.0,
        overlap_duration_seconds: float = 1.0,
        hop_length: int = 160,
        sample_rate: int = 16000,
    ):
        """Initialize the audio chunker.

        Args:
            chunk_duration_seconds: Duration of each chunk in seconds
            overlap_duration_seconds: Overlap between chunks in seconds
            hop_length: Hop length for mel spectrogram (affects frame rate)
            sample_rate: Audio sample rate
        """
        self.chunk_duration_seconds = chunk_duration_seconds
        self.overlap_duration_seconds = overlap_duration_seconds
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        # Calculate frame-based parameters
        self.frames_per_second = sample_rate / hop_length  # ~100 fps for default settings
        self.frames_per_chunk = int(chunk_duration_seconds * self.frames_per_second)
        self.overlap_frames = int(overlap_duration_seconds * self.frames_per_second)

        logger.info(
            f"AudioChunker initialized: {chunk_duration_seconds}s chunks, "
            f"{overlap_duration_seconds}s overlap, {self.frames_per_second:.1f} fps, "
            f"{self.frames_per_chunk} frames/chunk, {self.overlap_frames} overlap frames"
        )

    def should_chunk(self, mel_spectrogram: torch.Tensor) -> bool:
        """Determine if audio should be chunked based on duration.

        Args:
            mel_spectrogram: Mel spectrogram tensor [n_mels, time_frames]

        Returns:
            True if audio should be chunked, False otherwise
        """
        time_frames = mel_spectrogram.size(1)
        return time_frames > self.frames_per_chunk

    def chunk_mel_spectrogram(
        self, mel_spectrogram: torch.Tensor, sample_id: Optional[Union[str, int]] = None
    ) -> List[Dict[str, Any]]:
        """Chunk a mel spectrogram into overlapping segments.

        Args:
            mel_spectrogram: Input mel spectrogram [n_mels, time_frames]
            sample_id: Optional sample identifier for metadata

        Returns:
            List of chunk dictionaries with mel data and metadata
        """
        if mel_spectrogram.dim() != 2:
            raise ValueError(f"Expected 2D mel spectrogram, got shape {mel_spectrogram.shape}")

        n_mels, total_frames = mel_spectrogram.shape

        if not self.should_chunk(mel_spectrogram):
            # Return single chunk if not long enough
            return [
                {
                    "mel": mel_spectrogram,
                    "chunk_metadata": {
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "start_frame": 0,
                        "end_frame": total_frames,
                        "start_time": 0.0,
                        "end_time": total_frames / self.frames_per_second,
                        "is_first_chunk": True,
                        "is_last_chunk": True,
                        "original_sample_id": sample_id,
                    },
                }
            ]

        chunks = []
        chunk_index = 0
        start_frame = 0

        while start_frame < total_frames:
            # Calculate chunk boundaries
            end_frame = min(start_frame + self.frames_per_chunk, total_frames)

            # Extract chunk
            chunk_mel = mel_spectrogram[:, start_frame:end_frame]

            # Pad if necessary (only for the last chunk if it's too short)
            actual_frames = end_frame - start_frame
            if actual_frames < self.frames_per_chunk and chunk_index > 0:
                # Pad the last chunk to maintain consistent shapes if needed
                padding = torch.zeros(
                    n_mels, self.frames_per_chunk - actual_frames, dtype=mel_spectrogram.dtype
                )
                chunk_mel = torch.cat([chunk_mel, padding], dim=1)

            # Create metadata
            chunk_metadata = {
                "chunk_index": chunk_index,
                "total_chunks": -1,  # Will be filled after all chunks are processed
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time": start_frame / self.frames_per_second,
                "end_time": end_frame / self.frames_per_second,
                "is_first_chunk": chunk_index == 0,
                "is_last_chunk": False,  # Will be updated later
                "original_sample_id": sample_id,
                "actual_frames": actual_frames,
                "padded_frames": self.frames_per_chunk - actual_frames
                if actual_frames < self.frames_per_chunk
                else 0,
            }

            chunks.append(
                {
                    "mel": chunk_mel,
                    "chunk_metadata": chunk_metadata,
                }
            )

            # Move to next chunk (with overlap consideration)
            if end_frame >= total_frames:
                break

            # Move start position for next chunk (subtract overlap)
            start_frame = end_frame - self.overlap_frames
            chunk_index += 1

        # Update total_chunks and is_last_chunk for all chunks
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            chunk["chunk_metadata"]["total_chunks"] = total_chunks
            chunk["chunk_metadata"]["is_last_chunk"] = i == total_chunks - 1

        logger.debug(
            f"Chunked audio: {total_frames} frames -> {total_chunks} chunks "
            f"({self.frames_per_chunk} frames each, {self.overlap_frames} overlap)"
        )

        return chunks

    def reassemble_chunks(
        self, chunks: List[Dict[str, Any]], blend_overlap: bool = True
    ) -> torch.Tensor:
        """Reassemble chunks back into a single mel spectrogram.

        Args:
            chunks: List of chunk dictionaries from chunk_mel_spectrogram
            blend_overlap: Whether to blend overlapping regions

        Returns:
            Reassembled mel spectrogram [n_mels, time_frames]
        """
        if not chunks:
            raise ValueError("No chunks provided for reassembly")

        if len(chunks) == 1:
            return chunks[0]["mel"]

        # Sort chunks by chunk_index to ensure correct order
        sorted_chunks = sorted(chunks, key=lambda x: x["chunk_metadata"]["chunk_index"])

        # Calculate total output length
        last_chunk_meta = sorted_chunks[-1]["chunk_metadata"]
        total_frames = last_chunk_meta["end_frame"]
        n_mels = sorted_chunks[0]["mel"].size(0)

        # Initialize output tensor
        output = torch.zeros(n_mels, total_frames, dtype=sorted_chunks[0]["mel"].dtype)

        if not blend_overlap:
            # Simple reassembly without blending
            for chunk in sorted_chunks:
                meta = chunk["chunk_metadata"]
                start_frame = meta["start_frame"]
                actual_frames = meta["actual_frames"]

                chunk_mel = chunk["mel"][:, :actual_frames]
                output[:, start_frame : start_frame + actual_frames] = chunk_mel
        else:
            # Reassembly with overlap blending
            for i, chunk in enumerate(sorted_chunks):
                meta = chunk["chunk_metadata"]
                start_frame = meta["start_frame"]
                actual_frames = meta["actual_frames"]

                chunk_mel = chunk["mel"][:, :actual_frames]

                if i == 0:
                    # First chunk: no blending needed
                    output[:, start_frame : start_frame + actual_frames] = chunk_mel
                else:
                    # Subsequent chunks: blend overlap region
                    overlap_start = start_frame
                    overlap_end = min(
                        start_frame + self.overlap_frames, start_frame + actual_frames
                    )

                    if overlap_start < overlap_end:
                        # Linear blending in overlap region
                        blend_frames = overlap_end - overlap_start
                        alpha = torch.linspace(0, 1, blend_frames).unsqueeze(0)  # [1, blend_frames]

                        # Blend existing with new
                        existing = output[:, overlap_start:overlap_end]
                        new_overlap = chunk_mel[:, :blend_frames]
                        blended = (1 - alpha) * existing + alpha * new_overlap
                        output[:, overlap_start:overlap_end] = blended

                        # Add non-overlapping part
                        if overlap_end < start_frame + actual_frames:
                            non_overlap_start = self.overlap_frames
                            output[:, overlap_end : start_frame + actual_frames] = chunk_mel[
                                :, non_overlap_start:actual_frames
                            ]
                    else:
                        # No overlap (shouldn't happen with proper chunking)
                        output[:, start_frame : start_frame + actual_frames] = chunk_mel

        return output

    def get_chunk_boundaries(self, total_frames: int) -> List[Tuple[int, int]]:
        """Get chunk boundary information without actually chunking.

        Args:
            total_frames: Total number of frames in the audio

        Returns:
            List of (start_frame, end_frame) tuples
        """
        if total_frames <= self.frames_per_chunk:
            return [(0, total_frames)]

        boundaries = []
        start_frame = 0

        while start_frame < total_frames:
            end_frame = min(start_frame + self.frames_per_chunk, total_frames)
            boundaries.append((start_frame, end_frame))

            if end_frame >= total_frames:
                break

            start_frame = end_frame - self.overlap_frames

        return boundaries


def create_audio_chunker(
    chunk_duration: float = 30.0, overlap_duration: float = 1.0, **kwargs
) -> AudioChunker:
    """Create an AudioChunker with standard settings.

    Args:
        chunk_duration: Duration of each chunk in seconds
        overlap_duration: Overlap between chunks in seconds
        **kwargs: Additional arguments for AudioChunker

    Returns:
        Configured AudioChunker instance
    """
    return AudioChunker(
        chunk_duration_seconds=chunk_duration, overlap_duration_seconds=overlap_duration, **kwargs
    )


def chunk_long_audio(
    mel_spectrogram: torch.Tensor,
    chunk_duration: float = 30.0,
    overlap_duration: float = 1.0,
    sample_id: Optional[Union[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Convenience function to chunk long audio.

    Args:
        mel_spectrogram: Input mel spectrogram [n_mels, time_frames]
        chunk_duration: Duration of each chunk in seconds
        overlap_duration: Overlap between chunks in seconds
        sample_id: Optional sample identifier

    Returns:
        List of chunk dictionaries
    """
    chunker = create_audio_chunker(chunk_duration, overlap_duration)
    return chunker.chunk_mel_spectrogram(mel_spectrogram, sample_id)
