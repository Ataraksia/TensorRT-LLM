#!/usr/bin/env python3
"""Test basic generation without audio input to debug the pipeline."""

import torch
import numpy as np
from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.models.higgs_audio.model import HiggsAudioTRTRunner


def test_basic_generation():
    """Test basic text-only generation to debug the pipeline."""
    print("Testing basic generation...")
    
    config = HiggsAudioConfig()
    
    runner = HiggsAudioTRTRunner(
        config=config,
        engine_dir="/home/me/TTS/TensorRT-LLM/higgs_audio_engine/",
        tokenizer_dir="bosonai/higgs-audio-v2-generation-3B-base",
        audio_tokenizer_dir="bosonai/higgs-audio-v2-tokenizer",
    )
    
    # Simple text prompt without audio
    prompt = "Hello world"
    
    print(f"Input prompt: {prompt}")
    
    try:
        # Test generation without audio input
        audio_output = runner.generate(prompt, input_audio=None)
        print(f"Generation successful!")
        print(f"Output type: {type(audio_output)}")
        if hasattr(audio_output, 'shape'):
            print(f"Output shape: {audio_output.shape}")
        elif isinstance(audio_output, (list, tuple)):
            print(f"Output length: {len(audio_output)}")
        else:
            print(f"Output: {audio_output}")
        return True
        
    except Exception as e:
        print(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_basic_generation()
    if success:
        print("✅ Basic generation test passed!")
    else:
        print("❌ Basic generation test failed!")
