from boson_multimodal import *  # noqa: F403  # noqa: F403

from tensorrt_llm import logger
from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.models.higgs_audio.model import HiggsAudioForCausalLM, HiggsAudioTRTRunner

logger.set_level("verbose")

# Create configuration
config = HiggsAudioConfig()

# Instantiate model
model = HiggsAudioForCausalLM(config)

# Set up TensorRT-LLM inference runner
runner = HiggsAudioTRTRunner(
    engine_dir="/home/me/TTS/TensorRT-LLM/higgs_audio_engine/",
    tokenizer_dir="bosonai/higgs-audio-v2-generation-3B-base",
    audio_tokenizer_dir="bosonai/higgs-audio-v2-tokenizer",
)

# Generate text/audio
output = runner.generate(
    "Chat, stop backseating! I totally know what I'm doing... I think",
    "/home/me/TTS/TensorRT-LLM/AussieGirl.wav",
)
