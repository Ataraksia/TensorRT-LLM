import jiwer
from boson_multimodal import *
from transformers import (  # noqa: F403  # noqa: F403
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)

from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.models.higgs_audio.model import HiggsAudioForCausalLM, HiggsAudioTRTRunner

# Create configuration
config = HiggsAudioConfig()

# Instantiate model
model = HiggsAudioForCausalLM(config)

# Set up TensorRT-LLM inference runner
runner = HiggsAudioTRTRunner(
    config=config,
    engine_dir="/home/me/TTS/TensorRT-LLM/higgs_audio_engine/",
    tokenizer_dir="bosonai/higgs-audio-v2-generation-3B-base",
    audio_tokenizer_dir="bosonai/higgs-audio-v2-tokenizer",
)

pre_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an AI assistant designed to convert text into speech. Generate speech for the user's text, using the specified description.<|scene_desc_start|>Audio is recorded from a quiet room. Speaker is an enthusiastic young Australian woman in her early 20s with a bright, high-pitched voice.<|scene_desc_end|><|eot_id|><|start_header_id|>user<|end_header_id|>Can you believe just how realistic this sounds now?<|eot_id|><|start_header_id|>assistant<|end_header_id|><|audio_bos|><|AUDIO|><|audio_eos|><|eot_id|><|start_header_id|>user<|end_header_id|>"  # noqa: E501
post_prompt = "<|eot_id|><|start_header_id|>assistant<|end_header_id|><|audio_out_bos|>"
input_text = "Chat, stop backseating! I totally know what I'm doing... I think"
prompt = pre_prompt + input_text + post_prompt

audio_path = "/home/me/TTS/TensorRT-LLM/AussieGirl.wav"

# Generate text/audio
audio_output = runner.generate(
    prompt,
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
actual_transcription = pipe(audio_output).strip()
expected_transcription = input_text
# Calculate the word error rate
word_error_rate = jiwer.wer((expected_transcription), (actual_transcription))
print(f"Expected: {expected_transcription}")
print(f"Actual: {actual_transcription}")

print(f"Word error rate: {word_error_rate}")
