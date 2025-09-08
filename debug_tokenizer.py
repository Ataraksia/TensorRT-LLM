#!/usr/bin/env python3
"""Debug tokenizer and input format."""

import torch
from transformers import AutoTokenizer


def debug_tokenizer():
    """Debug the tokenizer to understand input format."""
    print("Debugging tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
    
    print(f"Tokenizer type: {type(tokenizer)}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Special tokens: {tokenizer.special_tokens_map}")
    
    # Test encoding
    text = "Hello world"
    encoded = tokenizer.encode(text, return_tensors="pt")
    print(f"Input text: {text}")
    print(f"Encoded: {encoded}")
    print(f"Encoded shape: {encoded.shape}")
    
    # Test decoding
    decoded = tokenizer.decode(encoded[0], skip_special_tokens=True)
    print(f"Decoded: {decoded}")
    
    # Check for special tokens
    if hasattr(tokenizer, 'bos_token_id'):
        print(f"BOS token ID: {tokenizer.bos_token_id}")
    if hasattr(tokenizer, 'eos_token_id'):
        print(f"EOS token ID: {tokenizer.eos_token_id}")
    if hasattr(tokenizer, 'pad_token_id'):
        print(f"PAD token ID: {tokenizer.pad_token_id}")
    
    # Check for audio-specific tokens
    vocab = tokenizer.get_vocab()
    audio_tokens = {k: v for k, v in vocab.items() if 'audio' in k.lower() or 'AUDIO' in k}
    print(f"Audio-related tokens: {audio_tokens}")
    
    # Look for stream tokens
    stream_tokens = {k: v for k, v in vocab.items() if 'stream' in k.lower() or 'STREAM' in k}
    print(f"Stream-related tokens: {stream_tokens}")


if __name__ == "__main__":
    debug_tokenizer()
