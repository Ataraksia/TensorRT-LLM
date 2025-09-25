# SPDX-License-Identifier: Apache-2.0
"""An example showing how to use vLLM to serve multimodal models
and run online inference with OpenAI client.
"""

import argparse
import asyncio
import base64
import time
from io import BytesIO

import jiwer
import numpy as np
import openai
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

OPENAI_AUDIO_SAMPLE_RATE = 24000


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a content from a local file to base64 format."""
    # Read the MP3 file as binary and encode it directly to Base64
    with open(file_path, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
    return audio_base64


async def run_voice_clone_audio(stream: bool = True) -> None:
    request_start_time = time.perf_counter()
    first_audio_latency = None
    audio_bytes_io = BytesIO()

    params = {
        "model": "higgs-audio-v2-generation-3B-base",
        "input": "Chat, stop backseating! I totally know what I'm doing... I think.",
        "voice": "AussieGirl",
        "response_format": "pcm",
        "speed": 1.0,
        "extra_body": {
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.95,
        },
    }
    async with client.audio.speech.with_streaming_response.create(**params) as response:
        async for chunk in response.iter_bytes(chunk_size=1024):
            if first_audio_latency is None:
                first_audio_latency = time.perf_counter()
            audio_bytes_io.write(chunk)
    audio_bytes_io.seek(0)
    audio_data = np.frombuffer(audio_bytes_io.getvalue(), dtype=np.int16)

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
    actual_transcription = pipe(audio_data).strip()
    expected_transcription = params["input"]
    # Calculate the word error rate
    word_error_rate = jiwer.wer((expected_transcription), (actual_transcription))
    print(f"Expected: {expected_transcription}")
    print(f"Actual: {actual_transcription}")

    print(f"Word error rate: {word_error_rate}")

    # print("Saving the audio to file")
    # print(f"First audio latency: {(first_audio_latency - request_start_time) * 1000} ms")
    # sf.write("output_voice_clone.wav", audio_data, OPENAI_AUDIO_SAMPLE_RATE)


def main(args) -> None:
    asyncio.run(run_voice_clone_audio(args.stream))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://192.168.0.60:1338/v1",
        help="API base URL for OpenAI client.",
    )
    parser.add_argument("--api-key", type=str, default="EMPTY", help="API key for OpenAI client.")
    parser.add_argument("--stream", action="store_true", help="Stream the audio.")
    parser.add_argument("--model", type=str, default=None, help="Model to use.")
    args = parser.parse_args()

    client = openai.AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
    )
    model = "higgs-audio-v2-generation-3B-base"

    main(args)
