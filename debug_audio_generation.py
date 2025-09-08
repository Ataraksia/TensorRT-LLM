#!/usr/bin/env python3
"""Debug audio generation to see what tokens are being generated."""

import torch
import numpy as np
from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.models.higgs_audio.model import HiggsAudioTRTRunner
from transformers import AutoTokenizer


def debug_audio_generation():
    """Debug what tokens are being generated."""
    print("Debugging audio generation...")
    
    config = HiggsAudioConfig()
    
    runner = HiggsAudioTRTRunner(
        config=config,
        engine_dir="/home/me/TTS/TensorRT-LLM/higgs_audio_engine/",
        tokenizer_dir="bosonai/higgs-audio-v2-generation-3B-base",
        audio_tokenizer_dir="bosonai/higgs-audio-v2-tokenizer",
    )
    
    # Use the same prompt as Test.py
    pre_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an AI assistant designed to convert text into speech. Generate speech for the user's text, using the specified description.<|scene_desc_start|>Audio is recorded from a quiet room. Speaker is an enthusiastic young Australian woman in her early 20s with a bright, high-pitched voice.<|scene_desc_end|><|eot_id|><|start_header_id|>user<|end_header_id|>Can you believe just how realistic this sounds now?<|eot_id|><|start_header_id|>assistant<|end_header_id|><|audio_bos|><|AUDIO|><|audio_eos|><|eot_id|><|start_header_id|>user<|end_header_id|>"
    post_prompt = "<|eot_id|><|start_header_id|>assistant<|end_header_id|><|audio_out_bos|>"
    input_text = "Chat, stop backseating! I totally know what I'm doing... I think"
    prompt = pre_prompt + input_text + post_prompt
    
    audio_path = "/home/me/TTS/TensorRT-LLM/AussieGirl.wav"
    
    print(f"Input text: {input_text}")
    print(f"Audio path: {audio_path}")
    print(f"Prompt length: {len(prompt)} characters")
    
    # Encode the prompt to see what tokens we're starting with
    input_ids = runner.tokenizer.encode(prompt, return_tensors="pt").squeeze(0)
    print(f"Input token count: {len(input_ids)}")
    print(f"Last 10 input tokens: {input_ids[-10:]}")
    
    # Decode last few tokens to see the context
    last_tokens_text = runner.tokenizer.decode(input_ids[-10:], skip_special_tokens=False)
    print(f"Last tokens decoded: '{last_tokens_text}'")
    
    try:
        # Generate audio
        print("\nGenerating audio...")
        audio_output = runner.generate(prompt, audio_path)
        
        print(f"Generated audio type: {type(audio_output)}")
        if hasattr(audio_output, 'shape'):
            print(f"Generated audio shape: {audio_output.shape}")
        elif isinstance(audio_output, (list, tuple)):
            print(f"Generated audio length: {len(audio_output)}")
        
        # Check if audio has any meaningful content
        if isinstance(audio_output, np.ndarray):
            print(f"Audio stats: min={audio_output.min():.6f}, max={audio_output.max():.6f}, mean={audio_output.mean():.6f}")
            print(f"Audio RMS: {np.sqrt(np.mean(audio_output**2)):.6f}")
            
            # Check if it's just silence
            if np.abs(audio_output).max() < 1e-6:
                print("⚠️  Generated audio appears to be silence!")
            else:
                print("✅ Generated audio has content")
        
        return audio_output
        
    except Exception as e:
        print(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    audio = debug_audio_generation()
    if audio is not None:
        print("✅ Audio generation completed successfully!")
    else:
        print("❌ Audio generation failed!")
