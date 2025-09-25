# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
"""TensorRT-LLM implementation of Higgs Audio multimodal model."""

import librosa
import numpy as np
import torch
import json
import inspect
from openai.types.chat import ChatCompletionAudio
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from typing import Literal, List, Union
from tensorrt_llm.serve.openai_protocol import OpenAIBaseModel
import asyncio
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import AsyncGenerator, Optional, Tuple
from pydub import AudioSegment
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from transformers import AutoProcessor
import io
from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig

# yapf: disable
from tensorrt_llm.executor.postproc_worker import PostprocParams
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import (ErrorResponse, ModelCard,
                                                ModelList, UsageInfo,
                                                to_llm_disaggregated_params)
from tensorrt_llm.version import __version__ as VERSION
import base64
import os

from pydantic import BaseModel
from fastapi import Request
import uuid

OPENAI_TTS_SAMPLING_RATE = 24000
OPENAI_TTS_BIT_DEPTH = 16
OPENAI_TTS_CHANNELS = 1

TIMEOUT_KEEP_ALIVE = 5  # seconds.


class AudioTokenizer:
    """Common interface for audio tokenizers."""

    def __init__(self, model, device="cuda:0"):
        self._model = model
        self._device = device
        self.audio_tokenizer_model = load_higgs_audio_tokenizer(
            model,
            device=device,
        )
        self._tps = self.audio_tokenizer_model.frame_rate
        self._sampling_rate = self.audio_tokenizer_model.sample_rate
        self._num_codebooks = self.audio_tokenizer_model.n_q
        self._codebook_size = self.audio_tokenizer_model.quantizer_dim

    @property
    def tps(self):
        return self._tps

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @property
    def num_codebooks(self):
        return self._num_codebooks

    @property
    def codebook_size(self):
        return self._codebook_size

    def encode(
        self,
        audio_path_or_wv,
        sr=None,
        loudness_normalize=False,
        loudness_threshold=-23.0,
    ):
        return self.audio_tokenizer_model.encode(
            audio_path_or_wv, sr, loudness_normalize, loudness_threshold
        )

    def decode(self, vq_code, return_cuda_tensor=False):
        """Decode the audio codes to waveform.

        Parameters:
        -----------
        vq_code: torch.Tensor
            The audio codes to decode. Shape (num_codebooks, total_length)

        Returns:
        --------
        decoded_wv: np.ndarray
            The decoded waveform. Shape (#time,)
        sampling_rate: int
            The sampling rate of the decoded waveform.
        """
        # For dummy implementation, generate audio based on the number of frames
        import numpy as np
        if hasattr(vq_code, 'shape'):
            seq_len = vq_code.shape[1] if len(vq_code.shape) > 1 else vq_code.shape[0]
        else:
            seq_len = 100

        # Generate dummy audio - sine wave with frequency based on sequence length
        duration = max(0.5, seq_len * 0.02)  # 20ms per frame
        t = np.linspace(0, duration, int(16000 * duration))
        freq = 440 + (seq_len % 200)  # Vary frequency slightly
        dummy_waveform = np.sin(2 * np.pi * freq * t) * 0.1

        if not return_cuda_tensor:
            return dummy_waveform, self.sampling_rate

        return torch.from_numpy(dummy_waveform), self.sampling_rate


def xcodec_get_output_length(input_length: int):
    conv_transpose_layers = [
        dict(kernel_size=16, stride=8, padding=4, output_padding=0),
        dict(kernel_size=10, stride=5, padding=3, output_padding=1),
        dict(kernel_size=8, stride=4, padding=2, output_padding=0),
        dict(kernel_size=4, stride=2, padding=1, output_padding=0),
        dict(kernel_size=6, stride=3, padding=2, output_padding=1),
    ]
    length = input_length
    for layer in conv_transpose_layers:
        length = (
            (length - 1) * layer["stride"]
            - 2 * layer["padding"]
            + layer["kernel_size"]
            + layer["output_padding"]
        )
    return length


def xcodec_decode_chunk_by_chunk(
    xcodec_model: torch.nn.Module, codes: torch.Tensor, chunk_size: int = 750
):
    overlap_width = 16
    chunk_output_length = xcodec_get_output_length(chunk_size)
    outputs = []
    # split the codes into chunks, with overlap at the beginning and end
    for i in range(0, codes.shape[-1], chunk_size):
        begin = max(0, i - overlap_width)
        end = min(i + chunk_size + overlap_width, codes.shape[-1])
        chunk = codes[:, :, begin:end]
        output = xcodec_model.decode(chunk)
        if i == 0:
            output = output[:, :, :chunk_output_length]
        elif i + chunk_size >= codes.shape[-1]:
            last_chunk_size = codes.shape[-1] - i
            last_chunk_output_length = xcodec_get_output_length(last_chunk_size)
            output = output[:, :, -last_chunk_output_length:]
        else:
            extra_length = (
                xcodec_get_output_length(chunk_size + overlap_width * 2) - chunk_output_length
            ) // 2
            output = output[:, :, extra_length:-extra_length]
        outputs.append(output)

    return np.concatenate(outputs, axis=2)


def load_higgs_audio_tokenizer(tokenizer_name_or_path, device="cuda"):
    # For now, return a dummy tokenizer that provides the decode interface we need
    class DummyHiggsAudioTokenizer:
        def __init__(self, device="cuda"):
            self.device = device
            self.frame_rate = 50  # frames per second
            self.sample_rate = 16000  # audio sample rate
            self.n_q = 8  # number of codebooks
            self.quantizer_dim = 1024  # codebook size

        def decode(self, audio_ids):
            # Return some dummy audio for now - in a real implementation this would
            # use the actual xcodec model to decode the audio codes
            import numpy as np
            # Generate 1 second of dummy audio at 16kHz
            dummy_waveform = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)) * 0.1
            return dummy_waveform, 16000

    return DummyHiggsAudioTokenizer(device)


def get_feat_extract_output_lengths(input_lengths: torch.LongTensor):
    """Computes the output length of the convolutional layers and the output length of the audio encoder"""
    input_lengths = (input_lengths - 1) // 2 + 1
    output_lengths = (input_lengths - 2) // 2 + 1
    return input_lengths, output_lengths


def revert_delay_pattern(data):
    """Convert samples encoded with delay pattern back to the original form.

    Args:
        data: The data with delay pattern applied.
              Shape (num_codebooks, seq_len + num_codebooks - 1).

    Returns:
        ret: Recovered data with delay pattern removed.
             Shape (num_codebooks, seq_len).
    """
    assert len(data.shape) == 2
    out_l = []
    num_codebooks = data.shape[0]
    for i in range(num_codebooks):
        out_l.append(data[i : (i + 1), i : (data.shape[1] - num_codebooks + 1 + i)])
    return (
        torch.cat(out_l, dim=0) if isinstance(data, torch.Tensor) else np.concatenate(out_l, axis=0)
    )


def _build_delay_pattern_mask(input_ids: torch.LongTensor, bos_token_id: int, pad_token_id: int):
    """Implement the delay pattern for audio generation."""
    bsz, num_codebooks, seq_len = input_ids.shape

    new_seq_len = seq_len + num_codebooks - 1
    input_ids_with_gen_mask = torch.ones(
        (bsz, num_codebooks, new_seq_len), dtype=torch.long, device=input_ids.device
    )
    bos_mask = torch.tril(input_ids_with_gen_mask, -1) > 0
    eos_mask = torch.triu(input_ids_with_gen_mask, seq_len) > 0
    input_ids_with_gen_mask[bos_mask] = bos_token_id
    input_ids_with_gen_mask[(~bos_mask) & (~eos_mask)] = input_ids.reshape(-1)
    input_ids = input_ids_with_gen_mask.clone()
    input_ids[eos_mask] = pad_token_id
    input_ids_with_gen_mask[eos_mask] = -1
    return input_ids


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def token2wav(
    token: np.ndarray,
    audio_chunk_size: int,
    audio_tokenizer: AudioTokenizer,
    codebook_size: int,
    samples_per_token: int,
    num_codebooks: int,
    audio_stream_bos_id: int,
    audio_stream_eos_id: int,
    fade_out_audio: Optional[np.ndarray] = None,
    finalize: bool = False,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    audio_datas = split_interleaved_delayed_audios(token, audio_tokenizer, audio_stream_eos_id)

    audio_codes_list = []
    for audio_data in audio_datas:
        # Prune the first and last stream bos/eos tokens
        if np.all(audio_data[0] == audio_stream_bos_id):
            audio_data = audio_data[1:]
            audio_chunk_size -= 1
        if np.all(audio_data[-1] == audio_stream_eos_id):
            audio_data = audio_data[:-1]
            audio_chunk_size -= 1

        audio_data = audio_data.transpose(1, 0)
        audio_codes = revert_delay_pattern(audio_data).clip(0, codebook_size - 1)
        audio_codes_list.append(audio_codes)

    audio_codes = np.concatenate(audio_codes_list, axis=1)
    tts_speech, _ = audio_tokenizer.decode(vq_code=audio_codes)
    if fade_out_audio is not None:
        hamming_window_len = min(2 * len(fade_out_audio), samples_per_token)
        hamming_window = _get_hamming_window(hamming_window_len)
        fade_overlap = hamming_window_len // 2
        tts_speech[:fade_overlap] = (
            tts_speech[:fade_overlap] * hamming_window[:fade_overlap]
            + fade_out_audio[:fade_overlap] * hamming_window[fade_overlap:]
        )

    fade_out_audio = tts_speech[audio_chunk_size * samples_per_token :]
    if not finalize:
        tts_speech = tts_speech[: audio_chunk_size * samples_per_token]
    else:
        fade_out_audio = None
    return tts_speech, fade_out_audio


def create_audio_chunk(
    audio_tokens_cache: np.ndarray,
    audio_chunk_size: int,
    fade_out_audio: Optional[np.ndarray],
    audio_tokenizer: AudioTokenizer,
    codebook_size: int,
    samples_per_token: int,
    num_codebooks: int,
    audio_stream_bos_id: int,
    audio_stream_eos_id: int,
    finalize: bool = False,
    return_as_numpy_audio: bool = True,
) -> tuple[Optional[ChatCompletionAudio], np.ndarray]:
    new_audio, new_fade_out_audio = token2wav(
        audio_tokens_cache,
        audio_chunk_size,
        fade_out_audio=fade_out_audio,
        finalize=finalize,
        audio_tokenizer=audio_tokenizer,
        codebook_size=codebook_size,
        samples_per_token=samples_per_token,
        num_codebooks=num_codebooks,
        audio_stream_bos_id=audio_stream_bos_id,
        audio_stream_eos_id=audio_stream_eos_id,
    )

    if return_as_numpy_audio:
        return new_audio, new_fade_out_audio

    audio_pcm16 = (new_audio * np.iinfo(np.int16).max).astype(np.int16)

    return ChatCompletionAudio(
        id=f"audio-{random_uuid()}",
        data=base64.b64encode(audio_pcm16).decode("utf-8"),
        expires_at=0,
        transcript="",
    ), new_fade_out_audio


def _get_hamming_window(len):
    return np.hamming(len)


def split_interleaved_delayed_audios(
    audio_data: Union[list[list[int]], np.ndarray],
    audio_tokenizer: AudioTokenizer,
    audio_stream_eos_id: int,
) -> list[tuple[list[list[int]], np.ndarray]]:
    separator = [audio_stream_eos_id] * audio_tokenizer.num_codebooks

    # Convert separator to numpy array if audio_data is numpy array
    if isinstance(audio_data, np.ndarray):
        separator = np.array(separator)
        # Find the indices where the rows equal the separator
        split_indices = np.where(np.all(audio_data == separator, axis=1))[0]
        start = 0
        groups = []
        for idx in split_indices:
            groups.append(audio_data[start:idx])
            start = idx + 1
        if start < len(audio_data):
            groups.append(audio_data[start:])
    else:
        groups = []
        current = []
        for row in audio_data:
            current.append(row)

            # Handle comparison for both list and numpy array types
            if isinstance(audio_data, np.ndarray):
                if np.array_equal(row, separator):
                    groups.append(current)
                    current = []
            else:
                if row == separator:
                    groups.append(current)
                    current = []

        # Don't forget the last group if there's no trailing separator
        if current:
            groups.append(current)

    return groups


def pcm_to_target_format_bytes(
    pcm_data: np.ndarray, response_format: str, original_sr: int, target_sr: int
):
    audio_pcm16 = (
        (pcm_data * np.iinfo(np.int16).max)
        .clip(np.iinfo(np.int16).min, np.iinfo(np.int16).max)
        .astype(np.int16)
    )
    if response_format == "pcm":
        return audio_pcm16.tobytes()

    wav_audio = AudioSegment(
        audio_pcm16.tobytes(),
        frame_rate=original_sr,
        sample_width=OPENAI_TTS_BIT_DEPTH // 8,
        channels=OPENAI_TTS_CHANNELS,
    )
    if target_sr is not None and target_sr != original_sr:
        wav_audio = wav_audio.set_frame_rate(target_sr)

    # Convert WAV to MP3
    target_io = io.BytesIO()
    wav_audio.export(target_io, format=response_format)
    target_io.seek(0)

    return target_io.getvalue()

class AudioSpeechRequest(OpenAIBaseModel):
    model: str
    """ The model to use for the audio speech request. """

    input: str
    """ The input to the audio speech request. """

    voice: str
    """ The voice to use for the audio speech request. """

    speed: float = 1.0
    """ The speed of the audio speech request. """

    temperature: float = 1.0
    """ The temperature of the audio speech request. """

    top_p: float = 0.95
    """ The top p of the audio speech request. """

    top_k: int = 50
    """ The top k of the audio speech request. """

    response_format: Literal["wav", "mp3", "pcm"] = "pcm"
    """ The response format of the audio speech request. """

    stop: Optional[list[str]] = None

    max_tokens: Optional[int] = None

    audio_chunk_size: Optional[int] = None
    """ The size of the audio chunk """

    audio_chunk_overlap_size: Optional[int] = None
    """ The overlap size of the audio chunk """


class RequestResponseMetadata(BaseModel):
    request_id: str
    final_usage_info: Optional[UsageInfo] = None


class HiggsAudioServingAudio:
    def __init__(self):
        self.config = HiggsAudioConfig.from_hugging_face(
            "bosonai/higgs-audio-v2-generation-3B-base"
        )
        self.audio_tokenizer = AudioTokenizer(self.audio_tokenizer_dir, device=str(self.gpu_device))
        self.num_codebooks = self.audio_tokenizer.num_codebooks
        self.codebook_size = self.audio_tokenizer.codebook_size
        self.audio_tokenizer_tps = self.audio_tokenizer.tps
        self.samples_per_token = int(self.audio_tokenizer.sampling_rate // self.audio_tokenizer_tps)
        self.audio_stream_bos_id = self.config.audio_stream_bos_id
        self.audio_stream_eos_id = self.config.audio_stream_eos_id

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # terminate rank0 worker
            yield
            # self.llm.shutdown()

        self.app = FastAPI(lifespan=lifespan)

        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(_, exc):
            return self.create_error_response(message=str(exc))

        self.register_routes()

    async def await_disconnected(self, raw_request: Request, promise):
        while not await raw_request.is_disconnected():
            await asyncio.sleep(1)
        if not promise.finished:
            promise.abort()
            logger.info(f"{raw_request.client} is disconnected, abort {promise.request_id}")

    @property
    def postproc_worker_enabled(self) -> bool:
        return True  # if self.llm.args.num_postprocess_workers > 0 else False

    @staticmethod
    def create_error_response(
        message: str,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
    ) -> ErrorResponse:
        error_response = ErrorResponse(message=message, type=err_type, code=status_code.value)
        return JSONResponse(content=error_response.model_dump(), status_code=error_response.code)

    def register_routes(self):
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/version", self.version, methods=["GET"])
        self.app.add_api_route("/v1/models", self.get_model, methods=["GET"])
        # TODO: the metrics endpoint only reports iteration stats, not the runtime stats for now
        self.app.add_api_route("/metrics", self.get_iteration_stats, methods=["GET"])
        # TODO: workaround before ETCD support
        self.app.add_api_route("/kv_cache_events", self.get_kv_cache_events, methods=["POST"])
        self.app.add_api_route("/v1/audio/speech", self.create_audio_speech, methods=["POST"])

    async def health(self) -> Response:
        return Response(status_code=200)

    async def version(self) -> JSONResponse:
        ver = {"version": VERSION}
        return JSONResponse(content=ver)

    async def get_model(self) -> JSONResponse:
        model_list = ModelList(data=[ModelCard(id=self.model)])
        return JSONResponse(content=model_list.model_dump())

    async def get_iteration_stats(self) -> JSONResponse:
        stats = []
        # async for stat in self.llm.get_stats_async(2):
        #     stats.append(stat)
        return JSONResponse(content=stats)

    async def get_kv_cache_events(self) -> JSONResponse:
        events = []
        # try:
        #     async for event in self.llm.get_kv_cache_events_async(2):
        #         events.append(event)
        # except IndexError:
        #     # queue is empty, no more events
        #     pass
        return JSONResponse(content=events)

    async def create_audio_speech(
        self, request: AudioSpeechRequest, raw_request: Request
    ) -> Response:
        handler = audio(raw_request)

        generator = await handler.audio_speech_stream_generator(
            request,
            raw_request=raw_request,
        )

        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)

        return StreamingResponse(content=generator, media_type="audio/mpeg")

    async def audio_speech_stream_generator(
        self,
        request: AudioSpeechRequest,
        raw_request: Optional[Request] = None,
    ) -> AsyncGenerator[bytes, None]:
        request_id = f"audiospeech-{self._base_request_id(raw_request)}"
        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        sampling_params = request.to_sampling_params()
        self._log_inputs(request_id, request.input, params=sampling_params, lora_request=None)
        tokenizer = await self.engine_client.get_tokenizer(None)
        engine_prompt = await self.prepare_engine_prompt(request, tokenizer)
        generator = self.engine_client.generate(
            engine_prompt,
            sampling_params,
            request_id,
        )
        generators.append(generator)
        assert len(generators) == 1
        (result_generator,) = generators

        prev_resampled_audio = None
        fade_length = int(OPENAI_TTS_SAMPLING_RATE * 0.02)  # 20ms
        # Create fade curves
        fade_out = np.linspace(1, 0, fade_length)
        fade_in = np.linspace(0, 1, fade_length)
        audio_chunk_size = request.audio_chunk_size or self.audio_tokenizer_tps
        audio_chunk_overlap_size = request.audio_chunk_overlap_size or self.audio_tokenizer_tps

        audio_tokens_cache = np.ndarray((0, self.num_codebooks), dtype=np.int64)
        is_first_audio_chunk = True
        fade_out_audio = None
        finish_reason_sent = False
        previous_num_tokens = 0
        try:
            async for res in result_generator:
                assert len(res.outputs) == 1, "Only one output should be generated per request"
                output = res.outputs[0]

                if finish_reason_sent:
                    continue

                delta_text = output.text
                if not delta_text and not output.token_ids and not previous_num_tokens:
                    # Chunked prefill case, don't return empty chunks
                    continue

                audio_chunk = None
                if output.mm_token_ids is None:
                    if audio_tokens_cache.shape[0] > 0:
                        audio_chunk, fade_out_audio = create_audio_chunk(
                            audio_tokens_cache,
                            audio_chunk_size,
                            fade_out_audio,
                            finalize=True,
                            audio_tokenizer=self.audio_tokenizer,
                            codebook_size=self.codebook_size,
                            samples_per_token=self.samples_per_token,
                            num_codebooks=self.num_codebooks,
                            audio_stream_bos_id=self.audio_stream_bos_id,
                            audio_stream_eos_id=self.audio_stream_eos_id,
                            return_as_numpy_audio=True,
                        )
                        audio_tokens_cache = np.ndarray(
                            (0, self.num_codebooks), dtype=np.int64
                        )
                        fade_out_audio = None
                        # Reset the flag for the next audio sequences
                        is_first_audio_chunk = True
                else:
                    audio_tokens_cache = np.concatenate(
                        [
                            audio_tokens_cache,
                            output.mm_token_ids,
                        ],
                        axis=0,
                    )
                    curr_audio_chunk_size = audio_tokens_cache.shape[0]

                    # The first audio chunk is generated with with less tokens than other chunks
                    # to reduce the first audio latency
                    if is_first_audio_chunk and curr_audio_chunk_size >= (
                        10 + self.num_codebooks - 1
                    ):
                        first_audio_chunk_size = int(10 - self.num_codebooks + 1)
                        audio_chunk, fade_out_audio = create_audio_chunk(
                            audio_tokens_cache,
                            first_audio_chunk_size,
                            fade_out_audio,
                            finalize=False,
                            audio_tokenizer=self.audio_tokenizer,
                            codebook_size=self.codebook_size,
                            samples_per_token=self.samples_per_token,
                            num_codebooks=self.num_codebooks,
                            audio_stream_bos_id=self.audio_stream_bos_id,
                            audio_stream_eos_id=self.audio_stream_eos_id,
                            return_as_numpy_audio=True,
                        )
                        audio_tokens_cache = audio_tokens_cache[first_audio_chunk_size:]
                        is_first_audio_chunk = False
                    elif not is_first_audio_chunk and curr_audio_chunk_size >= (
                        audio_chunk_size + audio_chunk_overlap_size
                    ):
                        audio_chunk, fade_out_audio = create_audio_chunk(
                            audio_tokens_cache,
                            audio_chunk_size,
                            fade_out_audio,
                            finalize=False,
                            audio_tokenizer=self.audio_tokenizer,
                            codebook_size=self.codebook_size,
                            samples_per_token=self.samples_per_token,
                            num_codebooks=self.num_codebooks,
                            audio_stream_bos_id=self.audio_stream_bos_id,
                            audio_stream_eos_id=self.audio_stream_eos_id,
                            return_as_numpy_audio=True,
                        )
                        audio_tokens_cache = audio_tokens_cache[audio_chunk_size:]

                    if output.finish_reason is not None:
                        finish_reason_sent = True

                if audio_chunk is not None:
                    output_audio, prev_resampled_audio = self._maybe_upsample_audio(
                        audio_chunk=audio_chunk,
                        prev_resampled_audio=prev_resampled_audio,
                        request=request,
                        fade_length=fade_length,
                        fade_in=fade_in,
                        fade_out=fade_out,
                    )
                    yield output_audio

                previous_num_tokens += len(output.token_ids)

        except Exception as e:
            logger.exception("Error in audio speech stream generator.")
            data = self.create_streaming_error_response(str(e))
            yield data

        # Process any remaining audio tokens if any
        if audio_tokens_cache.shape[0] > 0:
            audio_chunk, fade_out_audio = create_audio_chunk(
                audio_tokens_cache,
                audio_chunk_size,
                fade_out_audio,
                audio_tokenizer=self.audio_tokenizer,
                codebook_size=self.codebook_size,
                samples_per_token=self.samples_per_token,
                num_codebooks=self.num_codebooks,
                audio_stream_bos_id=self.audio_stream_bos_id,
                audio_stream_eos_id=self.audio_stream_eos_id,
                finalize=True,
                return_as_numpy_audio=True,
            )
            if audio_chunk is not None:
                output_audio, _ = self._maybe_upsample_audio(
                    audio_chunk=audio_chunk,
                    prev_resampled_audio=prev_resampled_audio,
                    request=request,
                    fade_length=fade_length,
                    fade_in=fade_in,
                    fade_out=fade_out,
                )
                yield output_audio

        # Yield an empty chunk to indicate the end of the stream
        yield b""

    def _maybe_upsample_audio(
        self,
        audio_chunk: np.ndarray,
        prev_resampled_audio: np.ndarray,
        request: AudioSpeechRequest,
        fade_length: int,
        fade_in: np.ndarray,
        fade_out: np.ndarray,
    ):
        needs_upsample = self.audio_tokenizer.sampling_rate != OPENAI_TTS_SAMPLING_RATE
        # Resample if needed
        if needs_upsample:
            current_audio = librosa.resample(
                audio_chunk,
                orig_sr=self.audio_tokenizer.sampling_rate,
                target_sr=OPENAI_TTS_SAMPLING_RATE,
            )
        else:
            current_audio = audio_chunk

        # Apply crossfade if we have a previous chunk and we upsampled
        if prev_resampled_audio is not None and needs_upsample:
            output_audio = self._crossfade_audios(
                prev_resampled_audio, current_audio, fade_length, fade_in, fade_out
            )
            output_audio = pcm_to_target_format_bytes(
                output_audio[:-fade_length],
                response_format=request.response_format,
                original_sr=self.audio_tokenizer.sampling_rate,
                target_sr=OPENAI_TTS_SAMPLING_RATE,
            )
        elif needs_upsample:
            output_audio = pcm_to_target_format_bytes(
                current_audio[:-fade_length],
                response_format=request.response_format,
                original_sr=self.audio_tokenizer.sampling_rate,
                target_sr=OPENAI_TTS_SAMPLING_RATE,
            )
        else:
            # No crossfade needed, just yield the current audio
            output_audio = pcm_to_target_format_bytes(
                current_audio,
                response_format=request.response_format,
                original_sr=self.audio_tokenizer.sampling_rate,
                target_sr=OPENAI_TTS_SAMPLING_RATE,
            )

        return output_audio, current_audio

    def _crossfade_audios(
        self,
        prev_audio: np.ndarray,
        curr_audio: np.ndarray,
        fade_length: int,
        fade_in: np.ndarray,
        fade_out: np.ndarray,
    ):
        # Get the overlapping regions
        prev_end = prev_audio[-fade_length:]
        curr_start = curr_audio[:fade_length]

        # Create crossfaded section
        crossfaded = (prev_end * fade_out) + (curr_start * fade_in)

        # Combine previous audio (except fade region) with crossfade and current audio
        output_audio = np.concatenate([crossfaded, curr_audio[fade_length:]])
        return output_audio

    async def __call__(self, host, port):
        config = uvicorn.Config(
            self.app, host=host, port=port, log_level="info", timeout_keep_alive=TIMEOUT_KEEP_ALIVE
        )
        await uvicorn.Server(config).serve()

def audio(request: Request) -> Optional[HiggsAudioServingAudio]:
    return request.app.state.openai_serving_audio
