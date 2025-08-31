"""
Basic Text-to-Speech Example for Higgs Audio TensorRT-LLM

This example demonstrates simple text-to-speech generation using the production-ready
Higgs Audio TensorRT-LLM system. It showcases the unified architecture benefits and
provides a straightforward interface for TTS functionality.

Usage:
    python basic_tts.py --text "Hello, this is a test." --output output.wav
    python basic_tts.py --text "Hello, world!" --model_path /path/to/model --engine_path /path/to/engine
"""
import sounddevice as sd
import os
from transformers import AutoTokenizer, AutoProcessor, AutoConfig, AutoModel, AutoModelForCausalLM
from loguru import logger
import base64
from io import BytesIO
from tensorrt_llm.models.higgs_audio.chatml_dataset import Message, ChatMLSample, AudioContent, TextContent, prepare_chatml_sample, AUDIO_IN_TOKEN, AUDIO_OUT_TOKEN, EOS_TOKEN, ChatMLDatasetSample
from tensorrt_llm.models.higgs_audio.higgs_audio_collator import HiggsAudioSampleCollator
from dataclasses import asdict
import json
import time
from pathlib import Path
from typing import Optional, Union
import torch
import numpy as np
import soundfile as sf
import librosa
import argparse
import shutil
from transformers.models.whisper.processing_whisper import WhisperProcessor
from tensorrt_llm.builder import Builder
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin import PluginConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm._utils import to_json_file
# TensorRT-LLM imports
import sys
sys.path.append("/home/me/TTS/higgs-audio") # Add the directory to the search path
from boson_multimodal.model.higgs_audio import *  
from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.models.higgs_audio.model import HiggsAudioForCausalLM
from tensorrt_llm.models.higgs_audio.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from tensorrt_llm.models.higgs_audio.convert import load_weights_from_hf_model
from tensorrt_llm import LLM, BuildConfig
from tensorrt_llm.llmapi import QuantConfig, QuantAlgo, CalibConfig
from tensorrt_llm.runtime import ModelRunnerCpp as ModelRunner
from tensorrt_llm import logger
from tensorrt_llm.runtime import SamplingConfig
logger.set_level('verbose')


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
            "avg_latency_ms": 0
        }
        
        # Initialize system
        logger.info("Initializing Higgs Audio TTS system...")
        start_time = time.perf_counter()
        # Load configuration first
        self.config = HiggsAudioConfig.from_hugging_face(self.model_path)
        self.dtype = str(self.config.dtype)
        # Initialize TensorRT engine
        # Step 1: Setup directories and validate inputs
        if not self.engine_path or \
            not (self.engine_path / "rank0.engine").exists() or \
            not (self.engine_path / "config.json").exists():
            logger.info("Building TensorRT engine for unified Higgs Audio model...")
            self._build_engine()
            
        logger.info(f"Loading TensorRT engine from {self.engine_path}")
        self.engine_runner = ModelRunner.from_dir(str(self.engine_path), cuda_graph_mode = True, use_gpu_direct_storage = True)

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
            "Medium length warmup text for system initialization."
        ]
        for text in warmup_texts:
            try:
                _ = self._generate_internal(text, warmup=True)
            except Exception as e:
                logger.warning(f"Warmup generation failed: {e}")
        logger.info("System warmup completed")

        init_time = (time.perf_counter() - start_time) * 1000
        logger.info(f"TTS system initialized in {init_time:.1f}ms")
    
    def _build_engine(self) -> ModelRunner:
        """Build TensorRT engine for Higgs Audio TTS model."""
        build_start_time = time.perf_counter()
        self.engine_path.mkdir(parents=True, exist_ok=True)
        # # Validate model path exists
        # if not Path(self.model_path).exists():
        #     raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        step_start = time.perf_counter()
        checkpoint = load_weights_from_hf_model(
            hf_model_dir=self.model_path,
            config=self.config,
            validate_weights=True,
            fallback_strategy='duplicate_text'
        )

        mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
        plugin_config = PluginConfig()
        plugin_config.dtype=self.dtype
        plugin_config.use_fp8_context_fmha = False
        plugin_config.paged_kv_cache = True
        builder = Builder()
        builder_config = builder.create_builder_config(
            precision=self.dtype,
            tensor_parallel=mapping.tp_size,
            pipeline_parallel=mapping.pp_size,
            plugin_config=plugin_config,
            max_multimodal_length = 128,
            max_batch_size = 1,  
            max_input_len = self.max_len //2,  
            max_output_len = self.max_len //2,   
            max_beam_width = 1,
            max_num_tokens = self.max_len,
            kv_cache_type = 'PAGED',
            builder_optimization_level = 5,  
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
            # Prepare inputs for network building
            inputs = higgs_audio_model.prepare_inputs(
                max_batch_size=builder_config.max_batch_size,
                max_input_len=builder_config.max_input_len,
                max_seq_len=self.max_len,
                max_num_tokens=builder_config.max_num_tokens,
                use_cache=False,
            )
            outputs = higgs_audio_model.forward(**inputs)
            if outputs is None:
                raise RuntimeError("Model forward pass returned None")

        logger.info("  → Starting engine compilation (this may take several minutes)...")
        engine_buffer = builder.build_engine(network, builder_config)
        if engine_buffer is None:
            raise RuntimeError("TensorRT builder returned None - engine compilation failed")
        # Save engine
        with open(self.engine_path / "rank0.engine", "wb") as f:
            f.write(engine_buffer)
        logger.info(f"  ✓ Engine saved: {self.engine_path}")
        weights_dict = {}
        for key, tensor in checkpoint['tensors'].items():
            if hasattr(tensor, 'cpu'):
                tensor_cpu = tensor.cpu()
                if hasattr(tensor_cpu, 'dtype') and str(tensor_cpu.dtype) == 'torch.bfloat16':
                    tensor_cpu = tensor_cpu.float()  # Convert BFloat16 to Float32
                weights_dict[key] = tensor_cpu.numpy()
            else:
                weights_dict[key] = tensor
        np.savez(self.engine_path / "weights.npz", **weights_dict)
        logger.info(f"  ✓ Weights saved: {self.engine_path}")
        # Save comprehensive configuration in TensorRT-LLM expected format
        config_path = self.engine_path / "config.json"
        # Create configuration in the expected TensorRT-LLM format
        config_dict = {
            # Root level configuration for ExecutorConfig compatibility
            "version": "1.0",
            # Pretrained model configuration
            "pretrained_config": {
                **engine_config.to_dict(),
            },
            "build_config": {
                **builder_config.to_dict(),
                "plugin_config": {
                    **plugin_config.to_dict()
                },
                "lora_config": {}
            },
        }
        logger.info(f"  Config_Dict: {config_dict}")
        to_json_file(config_dict, str(config_path))
        logger.info(f"  ✓ Configuration saved: {config_path}")
        
        if self._validate_config_format(config_path):
            logger.info("  ✓ Configuration format validation passed")
        else:
            raise RuntimeError("Configuration format validation failed - critical parameters missing or invalid")

        step_time = time.perf_counter() - step_start
        logger.info(f"  ✓ Files saved in {step_time:.2f}s")

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
                    segments.append(' '.join(current_segment))
                
                # Start new segment
                current_segment = [word]
                current_length = len(word)
        
        # Add final segment
        if current_segment:
            segments.append(' '.join(current_segment))
        
        return segments
    
    def generate(
        self,
        text: str,
        voice_sample: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> np.ndarray:
        """Generate speech from text.

        Args:
            text: Input text to convert to speech
            voice_sample: Optional path to voice sample for cloning
            temperature: Generation temperature (overrides config default)
            streaming: Enable streaming generation for long texts
        """
        if not self.is_initialized():
            raise RuntimeError("TTS system is not properly initialized. Please check the logs for initialization errors.")

        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

         # Filter for audio tokens only (skip text tokens)
        audio_start_token = 128013  # <|audio_out_bos|>
        audio_end_token = 128012    # <|audio_eos|>
        inputs = {
            'batch_input_ids': inputs['input_ids'],
            'max_new_tokens': inputs['max_new_tokens'],
            'temperature': inputs['temperature'],
            'top_k': inputs['top_k'],
            'top_p': inputs['top_p'],
            'end_id': inputs['end_id'],
            'pad_id': inputs['pad_id'],
            'return_dict': inputs['return_dict']
        }
        chunk_size = inputs['chunk_size']
        in_audio_section = False

        start_time = time.perf_counter()

        audio_features = None
        if voice_sample:
            input_ids, _, audio_features, audio_feature_attention_mask = self.prepare_inputs(audio=voice_sample, text=text)
            logger.info(f"Loaded voice features from {voice_sample}")
        else:
            input_ids, _, _, _ = self.prepare_inputs(audio=None, text=text)
    
        # if isinstance(input_tokens, torch.Tensor):
        #     input_tokens_cpu = input_tokens.cpu().flatten()
        #     input_ids_list = [input_tokens_cpu]
        # else:
        #     input_tokens_tensor = torch.tensor(input_tokens, dtype=torch.int32)
        #     input_ids_list = [input_tokens_tensor]
        # input_ids = input_tokens.unsqueeze(0)  # Keep tensor version for compatibility

        # Add voice conditioning if available
        if audio_features is not None:
            # Encode voice features
            encoded_features = self.audio_tokenizer.encode(
                input=audio_features,
                sr=self.audio_tokenizer.sampling_rate,
            )
        
        inputs = {  
            batch_input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=1.0,        
            top_k=50,
            top_p=0.95,
            end_id=self.config.eos_token_id,
            pad_id=self.config.pad_token_id,
            chunk_size=32,
            return_dict=True,
        }

        # Audio playback queue
        playback_queue = queue.Queue()
        playback_active = threading.Event()
        playback_active.set()
        
        # Default playback callback (logs audio info)
        if playback_callback is None:
            def default_playback(chunk: np.ndarray):
                sd.play(chunk.audio_data, self.audio_tokenizer.sampling_rate)
                sd.wait()
            
            playback_callback = default_playback
        
        # Playback thread
        def playback_worker():
            while playback_active.is_set() or not playback_queue.empty():
                try:
                    chunk = playback_queue.get(timeout=0.1)
                    playback_callback(chunk)
                    playback_queue.task_done()
                except queue.Empty:
                    continue
        
        playback_thread = threading.Thread(target=playback_worker, daemon=True)
        playback_thread.start()
        # Stream and queue audio chunks
        chunk_start_time = time.perf_counter()

        ##Split text into segments for streaming
        #chunk_size = inputs['chunk_size']
        #text_segments = self._split_text_for_streaming(text, chunk_size)

        for chunk in self.engine_runner.generate(**inputs):
            if not in_audio_section:
                for i, token in enumerate(chunk):
                    if token == audio_start_token:
                        in_audio_section = True
                        chunk = chunk[i+1:]
                        break
            if in_audio_section:
                for i, token in enumerate(chunk):
                    if token == audio_end_token:
                        in_audio_section = False
                        chunk = chunk[:i]
                        break
                    
            chunk_audio = self.audio_tokenizer.decode(chunk)
            # Add to playback queue
            playback_queue.put(chunk_audio)

            # Monitor buffer health
            buffer_size = playback_queue.qsize()
            if buffer_size > self.streaming_config.buffer_size * 0.8:
                logger.warning(f"Playback buffer filling up: {buffer_size} chunks")
            elif buffer_size == 0 and not chunk.is_final:
                logger.warning("Buffer underrun detected")
                self.streaming_stats.buffer_underruns += 1
            
            # Calculate timing
            generation_time = (time.perf_counter() - chunk_start_time) * 1000
            #chunk_duration = len(chunk_audio) / 24000
        

        # Wait for playback to finish
        playback_queue.join()
        playback_active.clear()
        playback_thread.join(timeout=5.0)
        
        logger.info("Live playback completed")
            
        # Concatenate all chunks
        #return np.concatenate(audio_chunks) if audio_chunks else np.array([])

        # Update performance stats
        generation_time = (time.perf_counter() - start_time) * 1000
        self._update_stats(generation_time)

    
    def prepare_inputs(self, audio=None, text=None):
        assert isinstance(audio, str) or isinstance(
            text, str), "audio or text must be provided as user input"
        messages = []
        system_content = f"You are an AI assistant designed to convert text into speech. Generate speech for the following text, using the specified speaker voice.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room\n\n\nSpeaker is an enthusiastic young Australian woman in her early 20s with a bright, clear voice.\n SPEAKER0: "
        system_message = Message(
            role="system",
            content=[TextContent(system_content), AudioContent(audio_url=audio), TextContent("\n<|scene_desc_end|>")],
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
            try:
                raw_audio, _ = librosa.load(audio_content[0].audio_url, sr=sample_rate)
            except Exception as e:
                logger.warning(f"Failed to load audio from {audio_content[0].audio_url}: {e}")
                raw_audio = None
        elif audio_content[0].raw_audio is not None:
            try:
                raw_audio, _ = librosa.load(
                    BytesIO(base64.b64decode(audio_content[0].raw_audio)), sr=sample_rate
                )
            except Exception as e:
                logger.warning(f"Failed to load raw audio: {e}")
                raw_audio = None
        
        # Create dummy audio if no audio is available
        if raw_audio is None:
            # Create a short silence audio (0.1 seconds)
            raw_audio = np.zeros(int(sample_rate * 0.1))
            logger.info("Using dummy audio for text-only generation")
        
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
        audio_waveforms_concat = torch.tensor(raw_audio, device='cpu')
        audio_waveforms_start = torch.tensor([0], dtype=torch.int32, device='cpu')
            
        sample = ChatMLDatasetSample(
            input_ids=input_ids,
            label_ids=None,
            audio_ids_concat=audio_ids,
            audio_ids_start=audio_ids_start,
            audio_waveforms_concat=audio_waveforms_concat,
            audio_waveforms_start=audio_waveforms_start,
            audio_sample_rate=torch.tensor([sample_rate], dtype=torch.int32, device='cpu'),
            audio_speaker_indices=torch.tensor([-1], dtype=torch.int32, device='cpu'),  # -1 for unknown speaker
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
        return inputs["input_ids"], inputs["attention_mask"], inputs["audio_features"], inputs["audio_feature_attention_mask"]
    
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
            hasattr(self, 'config') and
            hasattr(self, 'engine_runner') and
            self.engine_runner is not None
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
            "performance_stats": self.get_performance_stats()
        }
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'engine_runner'):
            del self.engine_runner
        torch.cuda.empty_cache()
        logger.info("TTS system resources cleaned up")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Higgs Audio Basic TTS Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a test of the Higgs Audio TTS system.",
        help="Text to convert to speech"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output audio file path"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="bosonai/higgs-audio-v2-generation-3B-base",
        help="Path to Higgs Audio model directory"
    )
    parser.add_argument(
        "--engine_path",
        type=str,
        default="/home/me/TTS/higgs-audio-engine",
        help="Path to TensorRT engine directory"
    )
    parser.add_argument(
        "--audio_tokenizer_path",
        type=str,
        default="/home/me/TTS/higgs-audio-v2-generation-3B-base-tokenizer",
        help="Path to Higgs Audio audio tokenizer directory"
    )
    parser.add_argument(
        "--voice_sample",
        default="/home/me/TTS/AussieGirl.wav",
        type=str,
        help="Path to voice sample for cloning (optional)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        help="Generation temperature (0.1-1.0)"
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=1024,
        help="Maximum length of the generated speech"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable streaming generation for long texts"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark"
    )
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="Force rebuild of TensorRT engine even if it exists"
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
        max_len=args.max_len
    )

    # Print system info for debugging
    system_info = tts.get_system_info()
    logger.info(f"System Info: {system_info}")


    # Run benchmark if requested
    if args.benchmark:
        run_benchmark(tts)
        return

    logger.info(f"Generating speech for: '{args.text[:100]}{'...' if len(args.text) > 100 else ''}'")

    tts.generate(
        text=args.text,
        voice_sample=args.voice_sample,
        temperature=args.temperature,
        streaming=args.streaming
    )

    # Save output
    tts.save_audio(audio, args.output)

    # Print performance stats
    stats = tts.get_performance_stats()
    logger.info(f"Performance: {stats['avg_latency_ms']:.1f}ms average latency")

    # Cleanup
    tts.cleanup()

    logger.info("Speech generation completed successfully!")

    return 0

#TODO: Add TTFT
def run_benchmark(tts: HiggsAudioTTS):
    """Run performance benchmark."""
    logger.info("Running performance benchmark...")
    
    test_texts = [
        "Short test.",
        "This is a medium length test sentence for benchmarking.",
        "This is a longer test sentence that will be used to benchmark the performance of the Higgs Audio TTS system with various text lengths and complexity levels.",
        "Very long test text. " * 20  # Very long text
    ]
    
    results = []
    
    for i, text in enumerate(test_texts, 1):
        logger.info(f"Benchmark {i}/{len(test_texts)}: {len(text)} characters")
        
        # Run multiple iterations
        times = []
        for _ in range(5):
            start = time.perf_counter()
            audio = tts.generate_speech(text)
            duration = (time.perf_counter() - start) * 1000
            times.append(duration)
        
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        results.append({
            "text_length": len(text),
            "audio_duration": len(audio) / 24000,
            "avg_time_ms": avg_time,
            "min_time_ms": min_time,
            "max_time_ms": max_time,
            "real_time_factor": (len(audio) / 24000) / (avg_time / 1000)
        })
        
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