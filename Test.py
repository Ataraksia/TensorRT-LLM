from boson_multimodal import *  # noqa: F403  # noqa: F403

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
prompt = (
    pre_prompt + "Chat, stop backseating! I totally know what I'm doing... I think" + post_prompt
)

audio_path = "/home/me/TTS/TensorRT-LLM/AussieGirl.wav"

# Generate text/audio
output = runner.generate(
    prompt,
    audio_path,
)
