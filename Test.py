import jiwer
from transformers import AutoProcessor, pipeline

from run_chat_completion import AutoModelForSpeechSeq2Seq
from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.models.higgs_audio.model import (
    HiggsAudioForCausalLM,
    HiggsAudioTRTRunner,  # noqa: F401
)

# Create configuration
config = HiggsAudioConfig()

# Instantiate model
model = HiggsAudioForCausalLM(config)

runner = HiggsAudioTRTRunner(
    config=config,
    engine_dir="/home/me/TTS/TensorRT-LLM/higgs_audio_engine/",
    tokenizer_dir="bosonai/higgs-audio-v2-generation-3B-base",
    audio_tokenizer_dir="bosonai/higgs-audio-v2-tokenizer",
)

input_text = "Chat, stop backseating! I totally know what I'm doing... I think"

audio_path = "/home/me/TTS/TensorRT-LLM/AussieGirl.wav"

# Generate text/audio
audio_output = runner.generate(
    input_text,
    audio_path,
)

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
actual_transcription = pipe(audio_output)["text"]
expected_transcription = input_text
# Calculate the word error rate
word_error_rate = jiwer.wer((expected_transcription), (actual_transcription))
print(f"Expected: {expected_transcription}")
print(f"Actual: {actual_transcription}")

print(f"Word error rate: {word_error_rate}")
