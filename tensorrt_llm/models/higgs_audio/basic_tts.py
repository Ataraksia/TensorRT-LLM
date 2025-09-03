"""Basic Text-to-Speech Example for Higgs Audio TensorRT-LLM.

This example demonstrates simple text-to-speech generation using the production-ready
Higgs Audio TensorRT-LLM system. It showcases the unified architecture benefits and
provides a straightforward interface for TTS functionality.

Usage:
    python basic_tts.py --text "Hello, this is a test." --output output.wav
    python basic_tts.py --text "Hello, world!" --model_path /path/to/model --engine_path /path/to/engine
"""

import argparse
import os
import queue
import shutil

# TensorRT-LLM imports
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jiwer
import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from boson_multimodal import *  # noqa: F403
from torch.backends.xeon.run_cpu import args
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, pipeline

import tensorrt_llm
from tensorrt_llm import logger
from tensorrt_llm._utils import to_json_file
from tensorrt_llm.builder import Builder
from tensorrt_llm.functional import arange, expand, lt, unsqueeze
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.higgs_audio.chatml_dataset import (
    AudioContent,
    ChatMLDatasetSample,
    ChatMLSample,
    Message,
    TextContent,
    prepare_chatml_sample,
)
from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.models.higgs_audio.higgs_audio_collator import HiggsAudioSampleCollator
from tensorrt_llm.models.higgs_audio.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from tensorrt_llm.models.higgs_audio.model import HiggsAudioForCausalLM
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin import PluginConfig
from tensorrt_llm.runtime import ModelConfig, SamplingConfig, Session
from tensorrt_llm.runtime import ModelRunnerCpp as ModelRunner

logger.set_level("verbose")


class HiggsAudioTTS:
    """Simple TTS interface using the unified Higgs Audio architecture."""

    def __init__(
        self,
        model_path: str,
        engine_path: Optional[str] = None,
        audio_tokenizer_path: str = None,
        device: Union[str, torch.device] = "cuda:0",
        max_len: int = 3072,
        dtype: str = "bfloat16",
    ):
        """Initialize the Higgs Audio TTS system.

        Args:
            model_path: Path to the Higgs Audio model directory
            engine_path: Path to the TensorRT engine (optional, will build if not provided)
            device: Device to run inference on (default: "cuda:0")
        """
        self.model_path = model_path
        self.engine_path = Path(engine_path) if engine_path else None
        self.device = torch.device(device)
        self.max_len = max_len
        # Performance tracking
        self.generation_stats = {
            "total_generations": 0,
            "total_time_ms": 0,
            "avg_latency_ms": 0,
        }

        # Initialize system
        logger.info("Initializing Higgs Audio TTS system...")
        start_time = time.perf_counter()
        # Load configuration first
        self.config = HiggsAudioConfig.from_hugging_face(self.model_path)
        self.dtype = str(self.config.dtype)
        # Initialize TensorRT engine
        # Step 1: Setup directories and validate inputs
        if (
            not self.engine_path
            or not (self.engine_path / "rank0.engine").exists()
            or not (self.engine_path / "config.json").exists()
        ):
            logger.info("Building TensorRT engine for unified Higgs Audio model...")
            self._build_engine()

        logger.info(f"Loading audio engine from {self.engine_path}")
        with open(self.engine_path / "rank0.engine", "rb") as f:
            engine_buffer = f.read()
        logger.info(f"Creating session from engine {self.engine_path}")
        self.session_audio = Session.from_serialized_engine(engine_buffer)

        model_config = ModelConfig(
            max_batch_size=1,
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_kv_heads,
            hidden_size=self.config.hidden_size,
            vocab_size=self.config.vocab_size,
            num_layers=self.config.num_hidden_layers,
            gpt_attention_plugin="auto",
            kv_cache_type="PAGED",
            remove_input_padding=True,
            dtype=self.config.dtype,
            max_beam_width=1,
            gather_context_logits=True,
            gather_generation_logits=True,
        )
        runtime_rank = tensorrt_llm.mpi_rank()
        sampling_config = SamplingConfig(
            end_id=self.config.eos_token_id,
            pad_id=self.config.pad_token_id,
            num_beams=1,
            top_k=50,
            top_p=0.95,
            temperature=1.0,
        )

        runner_kwargs = dict(
            engine_dir=self.engine_dir,
            rank=runtime_rank,
            max_output_len=args.max_new_tokens,
            max_batch_size=model_config.max_batch_size,
            max_input_len=self.max_len // 2,
            max_beam_width=model_config.max_beam_width,
            device_ids=[0],
            cuda_graph_mode=True,
            use_gpu_direct_storage=True,
        )

        logger.info(f"Loading TensorRT engine from {self.engine_path}")
        self.engine_runner = ModelRunner.from_dir(**runner_kwargs)
        self.sampling_config = sampling_config
        # Initialize tokenizers and processors
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Initialize Whisper processor for audio feature extraction
        try:
            from transformers.models.whisper.processing_whisper import WhisperProcessor

            self.whisper_processor = WhisperProcessor.from_pretrained(
                "openai/whisper-large-v3-turbo"
            )
            logger.info("Whisper processor initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Whisper processor: {e}")
            self.whisper_processor = None

        self.audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer_path)
        logger.info("Tokenizers and processors initialized")
        self.collator = HiggsAudioSampleCollator(
            whisper_processor=self.whisper_processor,
            encode_whisper_embed=self.config.encode_whisper_embed,
            audio_in_token_id=self.config.audio_in_token_idx,
            audio_out_token_id=self.config.audio_out_token_idx,
            audio_stream_bos_id=self.config.audio_stream_bos_id,
            audio_stream_eos_id=self.config.audio_stream_eos_id,
            pad_token_id=self.config.pad_token_id,
            return_audio_in_tokens=False,
            use_delay_pattern=self.config.use_delay_pattern,
            audio_num_codebooks=self.config.audio_num_codebooks,
            round_to=1,
        )

        # Warm up the system
        logger.info("Warming up TTS system...")
        warmup_texts = [
            "Test",
            "Short warmup text.",
            "Medium length warmup text for system initialization.",
        ]
        for text in warmup_texts:
            try:
                _ = self._generate_internal(text, warmup=True)
            except Exception as e:
                logger.warning(f"Warmup generation failed: {e}")
        logger.info("System warmup completed")

        init_time = (time.perf_counter() - start_time) * 1000
        logger.info(f"TTS system initialized in {init_time:.1f}ms")

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
            seq_range = arange(0, max_seq_len, dtype="int32")
            seq_range = unsqueeze(seq_range, 0)  # [1, max_seq_len]
            seq_range = expand(seq_range, [batch_size, max_seq_len])  # [batch, max_seq_len]

            # Expand lengths for comparison
            lengths_expand = unsqueeze(audio_feat_out_lengths, 1)  # [batch, 1]
            lengths_expand = expand(
                lengths_expand, [batch_size, max_seq_len]
            )  # [batch, max_seq_len]

            # Create padding mask (True where valid tokens)
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
            check_seq_length=False,  # Skip length validation for flexibility
        )

        # Extract last hidden state
        if isinstance(audio_outputs, dict):
            selected_audio_feature = audio_outputs["last_hidden_state"]
        else:
            # Handle tuple output
            selected_audio_feature = (
                audio_outputs[0] if isinstance(audio_outputs, tuple) else audio_outputs
            )

        # Project audio features to text model dimension
        audio_features_embed = self.audio_encoder_proj(selected_audio_feature)

        return audio_features_embed, audio_feat_out_lengths

    def _build_engine(self):
        """Build TensorRT engine for Higgs Audio TTS model."""
        if not self.engine_path:
            self.engine_path = Path("/home/me/TTS/higgs-audio-engine")
        self.engine_path.mkdir(parents=True, exist_ok=True)
        # # Validate model path exists
        # if not Path(self.model_path).exists():
        #     raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        step_start = time.perf_counter()

        mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
        plugin_config = PluginConfig()
        plugin_config.dtype = self.dtype
        plugin_config.use_fp8_context_fmha = False
        plugin_config.paged_kv_cache = True
        builder = Builder()
        builder_config = builder.create_builder_config(
            precision=self.dtype,
            tensor_parallel=mapping.tp_size,
            pipeline_parallel=mapping.pp_size,
            plugin_config=plugin_config,
            max_multimodal_length=128,
            max_batch_size=1,
            max_input_len=self.max_len // 2,
            max_output_len=self.max_len // 2,
            max_beam_width=1,
            max_num_tokens=self.max_len,
            kv_cache_type="PAGED",
            builder_optimization_level=5,
        )
        engine_config = HiggsAudioConfig.from_hugging_face(
            self.model_path,
            dtype=self.dtype,
            mapping=mapping,
            quant_config=None,
        )
        network = builder.create_network()
        network.plugin_config = plugin_config
        with net_guard(network):
            # Create the complete Higgs Audio model
            higgs_audio_model = HiggsAudioForCausalLM(engine_config)
            # Prepare inputs for network building with cache enabled
            inputs = higgs_audio_model.prepare_inputs(
                max_batch_size=builder_config.max_batch_size,
                max_input_len=builder_config.max_input_len,
                max_seq_len=self.max_len,
                max_num_tokens=builder_config.max_num_tokens,
                use_cache=True,
                max_beam_width=builder_config.max_beam_width,
            )
            outputs = higgs_audio_model.forward(**inputs)
            if outputs is None:
                raise RuntimeError("Model forward pass returned None")

        logger.info("  → Starting engine compilation (this may take several minutes)...")
        config_dict = {
            "version": "1.0",
            "pretrained_config": {
                **engine_config.to_dict(),
                "architecture": "HiggsAudioForCausalLM",
                "dtype": str(self.dtype),
                "logits_dtype": "float32",
                "vocab_size": engine_config.vocab_size,
                "hidden_size": engine_config.hidden_size,
                "num_hidden_layers": engine_config.num_hidden_layers,
                "num_attention_heads": engine_config.num_attention_heads,
                "num_key_value_heads": getattr(
                    engine_config,
                    "num_key_value_heads",
                    engine_config.num_attention_heads,
                ),
                "head_size": engine_config.hidden_size // engine_config.num_attention_heads,
                "intermediate_size": getattr(
                    engine_config, "intermediate_size", engine_config.hidden_size * 4
                ),
                "norm_epsilon": getattr(engine_config, "norm_epsilon", 1e-5),
                "position_embedding_type": "rope_gpt_neox",
                "world_size": mapping.tp_size * mapping.pp_size,
                "tp_size": mapping.tp_size,
                "pp_size": mapping.pp_size,
                "max_position_embeddings": getattr(
                    engine_config, "max_position_embeddings", 131072
                ),
                "use_parallel_embedding": False,
                "embedding_sharding_dim": 0,
                "share_embedding_table": False,
                "quantization": {
                    "quant_algo": None,
                    "kv_cache_quant_algo": None,
                },
                "mapping": {
                    "world_size": mapping.tp_size * mapping.pp_size,
                    "tp_size": mapping.tp_size,
                    "pp_size": mapping.pp_size,
                },
            },
            "build_config": {
                **builder_config.to_dict(),
                "max_num_tokens": builder_config.max_num_tokens,
                "kv_cache_type": "PAGED",
                "plugin_config": {**plugin_config.to_dict()},
                "lora_config": {},
            },
        }
        logger.info(f"  Config_Dict: {config_dict}")
        config_path = self.engine_path / "config.json"
        to_json_file(config_dict, str(config_path))
        logger.info(f"  ✓ Configuration saved: {config_path}")

        logger.info("  ✓ Configuration saved successfully")

        # Build and serialize the TensorRT engine
        logger.info("  → Building TensorRT engine...")
        serialized_engine = builder.build_engine(network, builder_config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # Save the serialized engine (IHostMemory object)
        engine_path = self.engine_path / "rank0.engine"
        with open(engine_path, "wb") as f:
            # Convert IHostMemory to bytes
            engine_bytes = bytes(serialized_engine)
            f.write(engine_bytes)
        logger.info(f"  ✓ Engine saved: {engine_path}")

        step_time = time.perf_counter() - step_start
        logger.info(f"  ✓ Engine built and saved in {step_time:.2f}s")

    def _generate_internal(self, text: str, warmup: bool = False):
        """Internal generation method for warmup."""
        try:
            return self.generate(text, streaming=False, max_new_tokens=256)
        except Exception:
            if not warmup:
                raise
            return np.array([])

    def _split_text_for_streaming(self, text: str, chunk_size: int) -> List[str]:
        """Split text into segments optimized for streaming generation."""
        # Simple word-boundary splitting for better audio quality
        words = text.split()
        segments = []
        current_segment = []
        current_length = 0

        for word in words:
            # Add word if it fits in current chunk
            if current_length + len(word) + 1 <= chunk_size * 4:  # Rough character estimate
                current_segment.append(word)
                current_length += len(word) + 1
            else:
                # Finalize current segment
                if current_segment:
                    segments.append(" ".join(current_segment))

                # Start new segment
                current_segment = [word]
                current_length = len(word)

        # Add final segment
        if current_segment:
            segments.append(" ".join(current_segment))

        return segments

    def generate(
        self,
        text: str,
        voice_sample: Optional[str] = None,
        temperature: Optional[float] = None,
        streaming: bool = False,
        max_new_tokens: int = 1024,
    ) -> np.ndarray:
        """Generate speech from text using multimodal generation with delay pattern.

        Args:
            text: Input text to convert to speech
            voice_sample: Optional path to voice sample for cloning
            temperature: Generation temperature (overrides config default)
            streaming: Enable streaming generation for long texts
            max_new_tokens: Maximum number of new tokens to generate
        """
        if not self.is_initialized():
            raise RuntimeError(
                "TTS system is not properly initialized. Please check the logs for initialization errors."
            )

        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        start_time = time.perf_counter()

        # Prepare inputs with audio features if provided
        # Special tokens for audio generation
        audio_stream_start_token = self.config.audio_stream_bos_id  # <|audio_stream_bos|>
        audio_stream_end_token = self.config.audio_stream_eos_id  # <|audio_stream_eos|>
        audio_start_token = self.config.audio_out_bos_token_id  # <|audio_out_bos|>
        audio_end_token = self.config.audio_eos_token_id  # <|audio_eos|>

        input_audio_features = None
        audio_feature_attention_mask = None

        if voice_sample:
            input_ids, attention_mask, input_audio_features, audio_feature_attention_mask = (
                self.prepare_inputs(audio=voice_sample, text=text)
            )
            logger.info(f"Loaded voice features from {voice_sample}")
        else:
            input_ids, attention_mask, _, _ = self.prepare_inputs(audio=None, text=text)

        # Prepare generation parameters with audio-specific settings
        generation_params = {
            "batch_input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": 50,
            "top_p": 0.95,
            "end_id": self.config.eos_token_id,
            "pad_id": self.config.pad_token_id,
            "return_dict": True,
            "output_sequence_lengths": True,
            "return_generation_logits": True,
            "num_beams": 1,
            "num_return_sequences": 1,
            "length_penalty": 1.0,
            "early_stopping": False,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "stop_words_list": [audio_end_token, audio_stream_end_token],
            "return_dict": False,  # noqa: F601
        }

        # Add audio features if available
        if input_audio_features is not None:
            audio_features, audio_features_length = self._apply_audio_tower(
                input_audio_features, audio_feature_attention_mask
            )

            # merge audio features and input ids
            num_audios, max_audio_tokens, embed_dim = audio_features.shape
            audio_features_mask = torch.arange(max_audio_tokens, device=device).expand(
                num_audios, max_audio_tokens
            ) < num_audio_tokens.unsqueeze(1)
            masked_audio_features = audio_features[audio_features_mask].view(-1, embed_dim)
            batch_size, _ = input_ids.shape

            # 1. Create a mask to know where special audio tokens are
            special_audio_token_mask = input_ids == self.config.audio_token_index
            special_audio_token_num = special_audio_token_mask.sum().item()

            generation_params["encoder_input_features"] = audio_features
            generation_params["audio_feature_attention_mask"] = audio_feature_attention_mask

        if streaming:
            return self._generate_streaming(
                generation_params,
                audio_stream_start_token,
                audio_stream_end_token,
                start_time,
            )
        else:
            return self._generate_batch(
                generation_params, audio_start_token, audio_end_token, start_time
            )

    def _generate_batch(
        self,
        generation_params: Dict[str, Any],
        audio_start_token: int,
        audio_end_token: int,
        start_time: float,
    ) -> np.ndarray:
        """Perform batch generation using multimodal generation with delay pattern."""
        # Use the multimodal generation method from the model
        with torch.no_grad():
            # First, generate the text portion up to the audio start token
            # This ensures we get the proper context before audio generation
            # generation_params['max_new_tokens'] = 150  # Limit initial generation
            outputs = self.engine_runner.generate(**generation_params)

            # Check if we have the audio start token
            generated_ids = outputs["output_ids"][0] if isinstance(outputs, dict) else outputs[0]
            if hasattr(generated_ids, "cpu"):
                generated_ids = generated_ids.cpu().numpy()
            elif hasattr(generated_ids, "numpy"):
                generated_ids = generated_ids.numpy()
            generated_ids = generated_ids.flatten()

            # Find audio start position
            audio_start_pos = None
            for i, token in enumerate(generated_ids):
                if int(token) == audio_start_token:
                    audio_start_pos = i
                    break

            if audio_start_pos is not None:
                logger.info(
                    f"Found audio start token at position {audio_start_pos}, generating audio tokens..."
                )

                # Combine the text portion with audio tokens
                full_sequence = (
                    list(generated_ids[: audio_start_pos + 1]) + audio_tokens + [audio_end_token]
                )
                generated_ids = np.array(full_sequence)
                logger.info(f"Generated sequence with {len(audio_tokens)} audio tokens")

                # Return the outputs with the combined sequence
                outputs = {"output_ids": [generated_ids]}

            # Extract audio tokens from the generated sequence
            generated_ids = outputs["output_ids"][0] if isinstance(outputs, dict) else outputs[0]

            # Convert to numpy array for easier processing
            if hasattr(generated_ids, "cpu"):
                generated_ids = generated_ids.cpu().numpy()
            elif hasattr(generated_ids, "numpy"):
                generated_ids = generated_ids.numpy()

            # Flatten the array to handle multi-dimensional cases
            generated_ids = generated_ids.flatten()

            # Find audio section in generated tokens
            audio_tokens = []
            in_audio_section = False

            # Debug: Print first and last 10 tokens to see what's being generated
            logger.info(f"Generated {len(generated_ids)} tokens")
            logger.info(f"First 10 tokens: {generated_ids[:10].tolist()}")
            logger.info(f"Last 10 tokens: {generated_ids[-10:].tolist()}")
            logger.info(
                f"Looking for audio_start_token: {audio_start_token}, audio_end_token: {audio_end_token}"
            )

            # Check if audio_start_token appears anywhere in the sequence
            if audio_start_token in generated_ids:
                logger.info("Audio start token found in sequence")
            else:
                logger.warning(
                    f"Audio start token {audio_start_token} not found in generated sequence"
                )
                # Try to decode some tokens to see what's being generated
                sample_tokens = generated_ids[60:80]  # Around where we expect audio tokens
                decoded_text = self.tokenizer.decode(sample_tokens, skip_special_tokens=False)
                logger.info(f"Sample decoded text around position 60-80: {decoded_text}")

            # Find start and end positions first
            start_pos = None
            end_pos = None
            for i, token_val in enumerate(generated_ids):
                token_val = int(token_val)
                if token_val == audio_start_token and start_pos is None:
                    start_pos = i
                    logger.info(f"Found audio start token at position {i}")
                elif token_val == audio_end_token and start_pos is not None:
                    end_pos = i
                    logger.info(f"Found audio end token at position {i}")
                    break

            # Extract audio tokens between start and end positions
            if start_pos is not None and end_pos is not None and end_pos > start_pos:
                raw_audio_tokens = [int(token) for token in generated_ids[start_pos + 1 : end_pos]]
                logger.info(
                    f"Extracted {len(raw_audio_tokens)} raw audio tokens between positions {start_pos} and {end_pos}"
                )
            elif start_pos is not None:
                # If we found start but no end, take tokens from start to end of sequence
                raw_audio_tokens = [int(token) for token in generated_ids[start_pos + 1 :]]
                logger.info(
                    f"Extracted {len(raw_audio_tokens)} raw audio tokens from position {start_pos} to end"
                )
            else:
                raw_audio_tokens = []

            # Process the raw audio tokens
            if raw_audio_tokens:
                logger.info(f"Processing {len(raw_audio_tokens)} raw audio tokens")
                logger.info(f"Sample raw tokens: {raw_audio_tokens[:20]}")

            # All tokens should now be in valid audio range (0-1023)
            audio_vocab_size = getattr(self.config, "audio_vocab_size", 1024)
            audio_tokens = [token for token in raw_audio_tokens if 0 <= token < audio_vocab_size]

            if len(audio_tokens) != len(raw_audio_tokens):
                logger.warning(
                    f"Filtered {len(raw_audio_tokens) - len(audio_tokens)} invalid audio tokens (out of range 0-{audio_vocab_size - 1})"
                )
                logger.info(
                    f"Sample invalid tokens: {[t for t in raw_audio_tokens if not (0 <= t < audio_vocab_size)][:10]}"
                )

            logger.info(f"Using {len(audio_tokens)} valid audio tokens for decoding")

            # Decode audio tokens to waveform
            if audio_tokens:
                try:
                    # Reshape audio tokens to match expected format for HiggsAudio tokenizer
                    # Expected shape: (batch_size, num_codebooks, sequence_length)
                    num_codebooks = getattr(self.config, "audio_num_codebooks", 8)
                    sequence_length = len(audio_tokens) // num_codebooks

                    if sequence_length < 1:
                        # Not enough tokens for even one frame
                        logger.warning(
                            f"Not enough audio tokens ({len(audio_tokens)}) for {num_codebooks} codebooks"
                        )
                        # Pad to minimum length
                        audio_tokens.extend([0] * (num_codebooks - len(audio_tokens)))
                        sequence_length = 1

                    if len(audio_tokens) % num_codebooks != 0:
                        # Pad tokens to make it divisible by num_codebooks
                        padding_needed = num_codebooks - (len(audio_tokens) % num_codebooks)
                        audio_tokens.extend([0] * padding_needed)
                        sequence_length = len(audio_tokens) // num_codebooks

                    # Reshape to (num_codebooks, sequence_length) then add batch dimension
                    audio_tokens_reshaped = torch.tensor(audio_tokens, dtype=torch.long).reshape(
                        num_codebooks, sequence_length
                    )
                    audio_tokens_tensor = audio_tokens_reshaped.unsqueeze(0)  # Add batch dimension

                    logger.info(
                        f"Reshaped audio tokens to {audio_tokens_tensor.shape} for decoding"
                    )

                    # Ensure tokens are on the correct device
                    if hasattr(self.audio_tokenizer, "device"):
                        audio_tokens_tensor = audio_tokens_tensor.to(self.audio_tokenizer.device)
                    elif torch.cuda.is_available():
                        audio_tokens_tensor = audio_tokens_tensor.cuda()

                    audio_waveform = self.audio_tokenizer.decode(audio_tokens_tensor)

                    # Convert to numpy array if needed
                    if isinstance(audio_waveform, torch.Tensor):
                        audio_waveform = audio_waveform.cpu().numpy()

                    # Ensure audio is 1D
                    if audio_waveform.ndim > 1:
                        audio_waveform = audio_waveform.squeeze()

                    generation_time = (time.perf_counter() - start_time) * 1000
                    self._update_stats(generation_time)

                    logger.info(
                        f"Generated {len(audio_waveform) / self.audio_tokenizer.sampling_rate:.2f}s of audio"
                    )
                    return audio_waveform

                except Exception as e:
                    logger.error(f"Failed to decode audio tokens: {e}")
                    logger.info(f"Audio tokens shape before reshape: {len(audio_tokens)}")
                    logger.info(f"First 20 audio tokens: {audio_tokens[:20]}")
                    # Return empty array on decode failure
                    return np.array([])
            else:
                logger.warning("No audio tokens generated")
                return np.array([])

    def _generate_streaming(
        self,
        generation_params: Dict[str, Any],
        audio_start_token: int,
        audio_end_token: int,
        start_time: float,
    ) -> np.ndarray:
        """Perform streaming generation with playback."""
        # Audio playback queue
        playback_queue = queue.Queue()
        playback_active = threading.Event()
        playback_active.set()
        audio_chunks = []

        # Playback thread
        def playback_worker():
            while playback_active.is_set() or not playback_queue.empty():
                try:
                    chunk = playback_queue.get(timeout=0.1)
                    if chunk is not None and len(chunk) > 0:
                        sd.play(chunk, self.audio_tokenizer.sampling_rate)
                        sd.wait()
                    playback_queue.task_done()
                except queue.Empty:
                    continue

        playback_thread = threading.Thread(target=playback_worker, daemon=True)
        playback_thread.start()

        # Stream generation
        in_audio_section = False
        audio_token_buffer = []

        with torch.no_grad():
            # Run streaming generation
            generation_params["streaming"] = True
            for chunk in self.engine_runner.generate(**generation_params):
                for token_id in chunk:
                    if token_id == audio_start_token:
                        in_audio_section = True
                        continue
                    elif token_id == audio_end_token:
                        in_audio_section = False
                        # Decode and play remaining buffer
                        if audio_token_buffer:
                            audio_chunk = self._decode_audio_tokens(audio_token_buffer)
                            audio_chunks.append(audio_chunk)
                            playback_queue.put(audio_chunk)
                            audio_token_buffer = []
                        break
                    elif in_audio_section:
                        audio_token_buffer.append(token_id)

                        # Decode and play when buffer is large enough
                        if len(audio_token_buffer) >= 32:  # Chunk size for streaming
                            audio_chunk = self._decode_audio_tokens(audio_token_buffer)
                            audio_chunks.append(audio_chunk)
                            playback_queue.put(audio_chunk)
                            audio_token_buffer = []

        # Wait for playback to finish
        playback_queue.join()
        playback_active.clear()
        playback_thread.join(timeout=5.0)

        logger.info("Streaming playback completed")

        # Concatenate all chunks
        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            generation_time = (time.perf_counter() - start_time) * 1000
            self._update_stats(generation_time)
            return full_audio
        else:
            return np.array([])

    def _decode_audio_tokens(self, tokens: List[int]) -> np.ndarray:
        """Decode audio tokens to waveform."""
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        audio_waveform = self.audio_tokenizer.decode(tokens_tensor)

        if isinstance(audio_waveform, torch.Tensor):
            audio_waveform = audio_waveform.cpu().numpy()

        return audio_waveform

    def prepare_inputs(self, audio=None, text=None):
        assert isinstance(audio, str) or isinstance(text, str), (
            "audio or text must be provided as user input"
        )
        messages = []
        # TODO Since the system text and audio reference never changes, we should simply separate this from the user_message and cache the value after the first call
        system_content = "You are an AI assistant designed to convert text into speech. Generate speech for the following text, using the specified speaker voice.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room\n\n\nSpeaker is an enthusiastic young Australian woman in her early 20s with a bright, clear voice.\n SPEAKER0: "  # noqa: E501
        system_message = Message(
            role="system",
            content=[
                TextContent(system_content),
                AudioContent(audio_url=audio),
                TextContent("\n<|scene_desc_end|>"),
            ],
        )
        user_message = Message(
            role="user",
            content=TextContent(text),
        )
        messages.append(system_message)
        messages.append(user_message)
        chatml_sample = ChatMLSample(messages=messages)
        input_tokens, _, audio_content, _ = prepare_chatml_sample(
            chatml_sample,
            self.tokenizer,
        )
        postfix = "<|start_header_id|>assistant<|end_header_id|>\n\n<|audio_out_bos|>"
        postfix = self.tokenizer.encode(postfix, add_special_tokens=False)
        input_tokens.extend(postfix)
        input_ids = torch.tensor(input_tokens, dtype=torch.int32, device=self.device)
        sample_rate = self.audio_tokenizer.sampling_rate

        # Handle audio loading with proper fallback
        raw_audio = None
        if audio_content[0].audio_url not in ["placeholder", ""]:
            raw_audio, _ = librosa.load(audio_content[0].audio_url, sr=sample_rate)

        # Ensure audio is float32 to match model weights and on CPU for processing
        if isinstance(raw_audio, np.ndarray):
            raw_audio = raw_audio.astype(np.float32)
        elif isinstance(raw_audio, torch.Tensor):
            # Move to CPU first, then convert to numpy
            raw_audio = raw_audio.cpu().numpy().astype(np.float32)

        audio_tokens = self.audio_tokenizer.encode(raw_audio, sample_rate)

        audio_ids = audio_tokens.squeeze(0)
        audio_ids_start = torch.tensor(
            np.cumsum(np.array([0] + [audio_ids.shape[1]])),
            dtype=torch.int32,
            device=self.device,
        )[0:-1]
        # Create audio waveform tensor on CPU first to avoid device conversion issues
        audio_waveforms_concat = torch.tensor(raw_audio, device="cpu")
        audio_waveforms_start = torch.tensor([0], dtype=torch.int32, device="cpu")

        sample = ChatMLDatasetSample(
            input_ids=input_ids,
            label_ids=None,
            audio_ids_concat=audio_ids,
            audio_ids_start=audio_ids_start,
            audio_waveforms_concat=audio_waveforms_concat,
            audio_waveforms_start=audio_waveforms_start,
            audio_sample_rate=torch.tensor([sample_rate], dtype=torch.int32, device="cpu"),
            audio_speaker_indices=torch.tensor(
                [-1], dtype=torch.int32, device="cpu"
            ),  # -1 for unknown speaker
        )
        data = self.collator([sample])
        inputs = asdict(data)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                # Move to device, but handle the case where tensor might already be on device
                if v.device != self.device:
                    inputs[k] = v.to(self.device)
                else:
                    inputs[k] = v
        return (
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["audio_features"],
            inputs["audio_feature_attention_mask"],
        )

    def _update_stats(self, generation_time_ms: float):
        """Update performance statistics."""
        self.generation_stats["total_generations"] += 1
        self.generation_stats["total_time_ms"] += generation_time_ms
        self.generation_stats["avg_latency_ms"] = (
            self.generation_stats["total_time_ms"] / self.generation_stats["total_generations"]
        )

    def save_audio(self, audio: np.ndarray, output_path: str, sample_rate: int = 24000):
        """Save audio array to file.

        Args:
            audio: Audio array to save
            output_path: Output file path
            sample_rate: Audio sample rate (default: 24000)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure audio is in correct format
        audio = np.clip(audio, -1.0, 1.0)  # Clamp to valid range

        sf.write(str(output_path), audio, sample_rate)
        logger.info(f"Audio saved to {output_path}")

    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        return self.generation_stats.copy()

    def is_initialized(self) -> bool:
        """Check if the TTS system is properly initialized."""
        return (
            hasattr(self, "config")
            and hasattr(self, "engine_runner")
            and self.engine_runner is not None
        )

    def get_system_info(self) -> dict:
        """Get system information for debugging."""
        return {
            "model_path": self.model_path,
            "engine_path": str(self.engine_path) if self.engine_path else None,
            "device": str(self.device),
            "is_initialized": self.is_initialized(),
            "has_tokenizer": self.tokenizer is not None,
            "has_audio_tokenizer": self.audio_tokenizer is not None,
            "performance_stats": self.get_performance_stats(),
        }

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, "engine_runner"):
            del self.engine_runner
        torch.cuda.empty_cache()
        logger.info("TTS system resources cleaned up")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Higgs Audio Basic TTS Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a test of the Higgs Audio TTS system.",
        help="Text to convert to speech",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/me/TTS/test_output.wav",
        help="Output audio file path",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="bosonai/higgs-audio-v2-generation-3B-base",
        help="Path to Higgs Audio model directory",
    )
    parser.add_argument(
        "--engine_path",
        type=str,
        default="/home/me/TTS/higgs-audio-engine",
        help="Path to TensorRT engine directory",
    )
    parser.add_argument(
        "--audio_tokenizer_path",
        type=str,
        default="/home/me/TTS/higgs-audio-v2-generation-3B-base-tokenizer",
        help="Path to Higgs Audio audio tokenizer directory",
    )
    parser.add_argument(
        "--voice_sample",
        default="/home/me/TTS/AussieGirl.wav",
        type=str,
        help="Path to voice sample for cloning (optional)",
    )

    parser.add_argument("--temperature", type=float, help="Generation temperature (0.1-1.0)")
    parser.add_argument(
        "--max_len",
        type=int,
        default=1024,
        help="Maximum length of the generated speech",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable streaming generation for long texts",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--test", action="store_true", help="Test the speech output")
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="Force rebuild of TensorRT engine even if it exists",
    )

    args = parser.parse_args()

    # Initialize TTS system
    logger.info("Initializing Higgs Audio TTS system...")
    if args.force_rebuild:
        if args.engine_path and os.path.exists(args.engine_path):
            shutil.rmtree(args.engine_path)

    tts = HiggsAudioTTS(
        model_path=args.model_path,
        engine_path=args.engine_path if args.engine_path else None,
        audio_tokenizer_path=args.audio_tokenizer_path,
        device=args.device,
        max_len=args.max_len,
    )

    # Print system info for debugging
    system_info = tts.get_system_info()
    logger.info(f"System Info: {system_info}")

    # Run benchmark if requested
    if args.benchmark:
        run_benchmark(tts)
        return
    if args.test:
        run_test(tts=tts)
        return

    logger.info(
        f"Generating speech for: '{args.text[:100]}{'...' if len(args.text) > 100 else ''}'"
    )

    # Generate audio using multimodal generation with delay pattern
    audio = tts.generate(
        text=args.text,
        voice_sample=args.voice_sample,
        temperature=args.temperature,
        streaming=args.streaming,
        max_new_tokens=args.max_len // 2,  # Approximate token count
    )

    # Save output
    if audio is not None and len(audio) > 0:
        tts.save_audio(audio, args.output)
    else:
        logger.error("No audio was generated")

    # Print performance stats
    stats = tts.get_performance_stats()
    logger.info(f"Performance: {stats['avg_latency_ms']:.1f}ms average latency")

    # Cleanup
    tts.cleanup()

    logger.info("Speech generation completed successfully!")

    return 0


def run_test(tts: HiggsAudioTTS):
    """Test speech generation with sample inputs."""
    logger.info("Testing speech generation...")
    test_inputs = [
        "Short test.",
        "This is a medium length test sentence for benchmarking.",
        "This is a longer test sentence that will be used to benchmark the performance of the Higgs Audio TTS system with various text lengths and complexity levels.",
    ]
    voice_sample = "/home/me/TTS/AussieGirl.wav"

    model_id = "openai/whisper-large-v3-turbo"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        return_timestamps=True,
    )
    avg_wer = 0.0
    for i, inputs in enumerate(test_inputs):
        logger.info(f"Test {i + 1}/{len(test_inputs)}: {inputs['text']}")
        audio = tts.generate(text=inputs["text"], voice_sample=voice_sample, max_new_tokens=1024)
        actual_transcription = pipe(audio).strip()
        expected_transcription = inputs["text"]
        # Calculate the word error rate
        word_error_rate = jiwer.wer((expected_transcription), (actual_transcription))
        print(f"Expected: {expected_transcription}")
        print(f"Actual: {actual_transcription}")
        avg_wer += word_error_rate

    avg_wer /= len(test_inputs)
    print(f"Average word error rate: {avg_wer}")


# TODO: Add TTFT
def run_benchmark(tts: HiggsAudioTTS):
    """Run performance benchmark."""
    logger.info("Running performance benchmark...")

    test_texts = [
        "Short test.",
        "This is a medium length test sentence for benchmarking.",
        "This is a longer test sentence that will be used to benchmark the performance of the Higgs Audio TTS system with various text lengths and complexity levels.",
        "Very long test text. " * 20,  # Very long text
    ]

    results = []

    for i, text in enumerate(test_texts, 1):
        logger.info(f"Benchmark {i}/{len(test_texts)}: {len(text)} characters")

        # Run multiple iterations
        times = []
        for _ in range(5):
            start = time.perf_counter()
            audio = tts.generate(text, streaming=False)  # Use the correct method name
            duration = (time.perf_counter() - start) * 1000
            times.append(duration)

        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)

        results.append(
            {
                "text_length": len(text),
                "audio_duration": len(audio) / 24000,
                "avg_time_ms": avg_time,
                "min_time_ms": min_time,
                "max_time_ms": max_time,
                "real_time_factor": (len(audio) / 24000) / (avg_time / 1000),
            }
        )

        logger.info(f"  Avg: {avg_time:.1f}ms, RTF: {results[-1]['real_time_factor']:.2f}")

    # Print summary
    logger.info("\nBenchmark Results Summary:")
    logger.info("=" * 60)
    logger.info(f"{'Text Len':<10} {'Audio Dur':<10} {'Avg Time':<10} {'RTF':<10}")
    logger.info("-" * 60)

    for result in results:
        logger.info(
            f"{result['text_length']:<10} "
            f"{result['audio_duration']:<10.1f} "
            f"{result['avg_time_ms']:<10.1f} "
            f"{result['real_time_factor']:<10.2f}"
        )

    overall_stats = tts.get_performance_stats()
    logger.info(f"\nOverall average latency: {overall_stats['avg_latency_ms']:.1f}ms")


if __name__ == "__main__":
    exit(main())
