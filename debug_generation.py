#!/usr/bin/env python3
"""Debug generation to understand what's happening."""

import torch
from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.models.higgs_audio.model import HiggsAudioTRTRunner
from transformers import AutoTokenizer


def debug_generation():
    """Debug the generation process step by step."""
    print("Debugging generation process...")
    
    config = HiggsAudioConfig()
    
    # Check config values
    print(f"Config audio_eos_token_id: {config.audio_eos_token_id}")
    print(f"Config audio_stream_eos_id: {config.audio_stream_eos_id}")
    print(f"Config pad_token_id: {config.pad_token_id}")
    
    # Load tokenizer to check tokens
    tokenizer = AutoTokenizer.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
    print(f"Tokenizer eos_token_id: {tokenizer.eos_token_id}")
    print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
    
    # Test the prompt format
    input_text = "Hello world"
    formatted_text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an AI assistant designed to convert text into speech. Generate speech for the user's text.<|eot_id|><|start_header_id|>user<|end_header_id|>{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|><|audio_out_bos|>"
    
    print(f"Formatted prompt: {formatted_text}")
    
    input_ids = tokenizer.encode(formatted_text, return_tensors="pt").squeeze(0)
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids}")
    print(f"Last few tokens: {input_ids[-5:]}")
    
    # Check what the last token is
    last_token = input_ids[-1].item()
    print(f"Last token ID: {last_token}")
    print(f"Last token decoded: '{tokenizer.decode([last_token])}'")
    
    # Check if last token is an end token
    if last_token == tokenizer.eos_token_id:
        print("⚠️  Last token is EOS - this might cause immediate termination!")
    if last_token == config.audio_eos_token_id:
        print("⚠️  Last token is audio EOS - this might cause immediate termination!")
    
    # Try to create runner and see what happens
    try:
        runner = HiggsAudioTRTRunner(
            config=config,
            engine_dir="/home/me/TTS/TensorRT-LLM/higgs_audio_engine/",
            tokenizer_dir="bosonai/higgs-audio-v2-generation-3B-base",
            audio_tokenizer_dir="bosonai/higgs-audio-v2-tokenizer",
        )
        print("✅ Runner created successfully")
        
        # Check runner configuration
        print(f"Runner audio_eos_token_id: {runner.audio_eos_token_id}")
        print(f"Runner pad_token_id: {runner.pad_token_id}")
        print(f"Runner max_new_tokens: {runner.max_new_tokens}")
        print(f"Runner temperature: {runner.temperature}")
        
    except Exception as e:
        print(f"❌ Failed to create runner: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_generation()
