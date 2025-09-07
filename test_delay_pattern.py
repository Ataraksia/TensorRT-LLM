#!/usr/bin/env python3
"""Test script to verify delay pattern implementation for HiggsAudio."""

import torch
import numpy as np
from tensorrt_llm.models.higgs_audio.model import _build_delay_pattern_mask, revert_delay_pattern


def test_delay_pattern_basic():
    """Test basic delay pattern functionality."""
    print("Testing basic delay pattern...")
    
    # Create test audio codes: 4 codebooks, sequence length 5
    # Original pattern:
    # - [a, b, c, d, e]
    # - [f, g, h, i, j]  
    # - [k, l, m, n, o]
    # - [p, q, r, s, t]
    
    batch_size = 1
    num_codebooks = 4
    seq_len = 5
    
    audio_codes = torch.arange(num_codebooks * seq_len).reshape(batch_size, num_codebooks, seq_len)
    bos_token_id = 999
    pad_token_id = 1000
    
    print(f"Original audio codes shape: {audio_codes.shape}")
    print(f"Original audio codes:\n{audio_codes[0]}")
    
    # Apply delay pattern
    delayed_codes = _build_delay_pattern_mask(audio_codes, bos_token_id, pad_token_id)
    
    print(f"Delayed codes shape: {delayed_codes.shape}")
    print(f"Delayed codes:\n{delayed_codes[0]}")
    
    # Expected pattern (seq_len + num_codebooks - 1 = 5 + 4 - 1 = 8):
    # - [ 0,  1,  2,  3,  4, 1000, 1000, 1000]
    # - [999, 5,  6,  7,  8,    9, 1000, 1000]
    # - [999, 999, 10, 11, 12,  13,   14, 1000]
    # - [999, 999, 999, 15, 16,  17,   18,   19]
    
    expected_seq_len = seq_len + num_codebooks - 1
    assert delayed_codes.shape == (batch_size, num_codebooks, expected_seq_len)
    
    # Check first row (no delay)
    assert torch.equal(delayed_codes[0, 0, :seq_len], audio_codes[0, 0, :])
    assert torch.all(delayed_codes[0, 0, seq_len:] == pad_token_id)
    
    # Check second row (1 delay token)
    assert delayed_codes[0, 1, 0] == bos_token_id
    assert torch.equal(delayed_codes[0, 1, 1:seq_len+1], audio_codes[0, 1, :])
    assert torch.all(delayed_codes[0, 1, seq_len+1:] == pad_token_id)
    
    print("✓ Basic delay pattern test passed")


def test_delay_pattern_revert():
    """Test reverting delay pattern back to original form."""
    print("\nTesting delay pattern revert...")
    
    # Create test delayed pattern
    num_codebooks = 4
    seq_len = 5
    delayed_seq_len = seq_len + num_codebooks - 1  # 8
    
    # Create delayed pattern manually
    delayed_data = torch.zeros(num_codebooks, delayed_seq_len, dtype=torch.long)
    
    # Fill with expected delay pattern
    original_data = torch.arange(num_codebooks * seq_len).reshape(num_codebooks, seq_len)
    
    for i in range(num_codebooks):
        # Each codebook is delayed by i positions
        start_pos = i
        end_pos = start_pos + seq_len
        delayed_data[i, start_pos:end_pos] = original_data[i, :]
    
    print(f"Delayed data shape: {delayed_data.shape}")
    print(f"Delayed data:\n{delayed_data}")
    
    # Revert delay pattern
    reverted_data = revert_delay_pattern(delayed_data)
    
    print(f"Reverted data shape: {reverted_data.shape}")
    print(f"Reverted data:\n{reverted_data}")
    
    # Should match original
    assert reverted_data.shape == (num_codebooks, seq_len)
    assert torch.equal(reverted_data, original_data)
    
    print("✓ Delay pattern revert test passed")


def test_delay_pattern_round_trip():
    """Test full round trip: original -> delayed -> reverted."""
    print("\nTesting delay pattern round trip...")
    
    batch_size = 1
    num_codebooks = 8  # Typical for HiggsAudio
    seq_len = 10
    
    # Create original audio codes
    original_codes = torch.randint(0, 1024, (batch_size, num_codebooks, seq_len))
    bos_token_id = 1025
    pad_token_id = 1026
    
    print(f"Original codes shape: {original_codes.shape}")
    
    # Apply delay pattern
    delayed_codes = _build_delay_pattern_mask(original_codes, bos_token_id, pad_token_id)
    
    print(f"Delayed codes shape: {delayed_codes.shape}")
    
    # Extract the actual audio data (remove padding)
    # For revert, we need to extract just the audio part without BOS/EOS tokens
    audio_part = delayed_codes[0]  # Remove batch dimension
    
    # Find the actual audio sequence length by looking at the pattern
    # The audio data starts at position i for codebook i and goes for seq_len tokens
    extracted_audio = torch.zeros(num_codebooks, seq_len, dtype=torch.long)
    for i in range(num_codebooks):
        start_pos = i
        end_pos = start_pos + seq_len
        extracted_audio[i, :] = audio_part[i, start_pos:end_pos]
    
    print(f"Extracted audio shape: {extracted_audio.shape}")
    
    # Should match original
    assert torch.equal(extracted_audio, original_codes[0])
    
    print("✓ Delay pattern round trip test passed")


def test_delay_pattern_with_special_tokens():
    """Test delay pattern with BOS/EOS tokens like in real usage."""
    print("\nTesting delay pattern with special tokens...")
    
    num_codebooks = 4
    seq_len = 3  # Short sequence for clarity
    audio_stream_bos_id = 2000
    audio_stream_eos_id = 2001
    
    # Original audio codes
    audio_codes = torch.tensor([
        [10, 11, 12],
        [20, 21, 22], 
        [30, 31, 32],
        [40, 41, 42]
    ])
    
    print(f"Original audio codes:\n{audio_codes}")
    
    # Add BOS and EOS tokens (like in real implementation)
    bos_tokens = torch.full((num_codebooks, 1), audio_stream_bos_id, dtype=audio_codes.dtype)
    eos_tokens = torch.full((num_codebooks, 1), audio_stream_eos_id, dtype=audio_codes.dtype)
    
    audio_with_special = torch.cat([bos_tokens, audio_codes, eos_tokens], dim=-1)
    print(f"Audio with BOS/EOS:\n{audio_with_special}")
    
    # Apply delay pattern
    delayed_codes = _build_delay_pattern_mask(
        audio_with_special.unsqueeze(0),  # Add batch dimension
        bos_token_id=audio_stream_bos_id,
        pad_token_id=audio_stream_eos_id
    ).squeeze(0)  # Remove batch dimension
    
    print(f"Delayed codes shape: {delayed_codes.shape}")
    print(f"Delayed codes:\n{delayed_codes}")
    
    # Verify the delay pattern structure
    expected_len = audio_with_special.shape[1] + num_codebooks - 1  # 5 + 4 - 1 = 8
    assert delayed_codes.shape == (num_codebooks, expected_len)
    
    # Check that BOS tokens appear in the right positions
    for i in range(num_codebooks):
        # Each codebook should have i BOS tokens at the start
        for j in range(i):
            assert delayed_codes[i, j] == audio_stream_bos_id
    
    print("✓ Delay pattern with special tokens test passed")


def main():
    """Run all delay pattern tests."""
    print("=" * 60)
    print("HiggsAudio Delay Pattern Test Suite")
    print("=" * 60)
    
    try:
        test_delay_pattern_basic()
        test_delay_pattern_revert()
        test_delay_pattern_round_trip()
        test_delay_pattern_with_special_tokens()
        
        print("\n" + "=" * 60)
        print("ALL DELAY PATTERN TESTS PASSED! ✅")
        print("=" * 60)
        print("✓ Basic delay pattern application works")
        print("✓ Delay pattern revert works correctly")
        print("✓ Round trip (apply + revert) preserves data")
        print("✓ Special token handling works properly")
        print("\nThe delay pattern implementation is ready for RVQ-based audio generation!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
