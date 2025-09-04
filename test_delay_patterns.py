"""Tests specifically for RVQ delay pattern functionality in HiggsAudio.

This test suite validates:
- DelayPatternProvider for generating patterns
- AudioTokenUtils for multi-codebook coordination
- StreamingCollationState for chunk management
- Integration with HiggsAudioSampleCollator
"""

import torch

# Import the modules we're testing
from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.models.higgs_audio.data_processing import (
    AudioTokenUtils,
    DelayPatternProvider,
    HiggsAudioSampleCollator,
    StreamingCollationState,
)


class TestDelayPatternProvider:
    """Test DelayPatternProvider functionality."""

    def test_linear_delay_pattern(self):
        """Test linear delay pattern generation."""
        print("Testing linear delay pattern...")

        provider = DelayPatternProvider(strategy="linear", num_codebooks=4, stride=1)
        pattern = provider.generate_delay_pattern(sequence_length=10)

        # Check shape
        assert pattern.shape == (4, 10)

        # Check delay values - should be [0, 1, 2, 3] for each position
        expected = torch.tensor(
            [
                [0] * 10,  # Codebook 0: no delay
                [1] * 10,  # Codebook 1: delay 1
                [2] * 10,  # Codebook 2: delay 2
                [3] * 10,  # Codebook 3: delay 3
            ]
        )
        assert torch.equal(pattern, expected)
        print("✓ Linear delay pattern correct")

    def test_exponential_delay_pattern(self):
        """Test exponential delay pattern generation."""
        print("Testing exponential delay pattern...")

        provider = DelayPatternProvider(strategy="exponential", num_codebooks=4)
        pattern = provider.generate_delay_pattern(sequence_length=10)

        # Check shape
        assert pattern.shape == (4, 10)

        # Check delay values - should be [0, 1, 3, 3] (last one clamped to max_delay=3)
        expected = torch.tensor(
            [
                [0] * 10,  # Codebook 0: 2^0 - 1 = 0
                [1] * 10,  # Codebook 1: 2^1 - 1 = 1
                [3] * 10,  # Codebook 2: 2^2 - 1 = 3
                [3] * 10,  # Codebook 3: min(2^3 - 1, 3) = 3
            ]
        )
        assert torch.equal(pattern, expected)
        print("✓ Exponential delay pattern correct")

    def test_custom_delay_pattern(self):
        """Test custom delay pattern generation."""
        print("Testing custom delay pattern...")

        provider = DelayPatternProvider(strategy="custom", num_codebooks=8)
        pattern = provider.generate_delay_pattern(sequence_length=5)

        # Check shape
        assert pattern.shape == (8, 5)

        # Custom strategy: min(k, max_delay) where max_delay = num_codebooks - 1 = 7
        expected_delays = [0, 1, 2, 3, 4, 5, 6, 7]
        for k in range(8):
            assert torch.all(pattern[k] == expected_delays[k])
        print("✓ Custom delay pattern correct")

    def test_apply_delay_pattern(self):
        """Test applying delay patterns to tokens."""
        print("Testing delay pattern application...")

        provider = DelayPatternProvider(strategy="linear", num_codebooks=3)

        # Create test tokens
        tokens = torch.tensor(
            [
                [10, 11, 12],  # Codebook 0
                [20, 21, 22],  # Codebook 1
                [30, 31, 32],  # Codebook 2
            ]
        )

        delay_pattern = provider.generate_delay_pattern(sequence_length=3)
        delayed_tokens = provider.apply_delay_pattern(tokens, delay_pattern)

        # Should be shape [3, 5] (3 + max_delay=2)
        assert delayed_tokens.shape == (3, 5)

        # Check delayed placement
        expected = torch.tensor(
            [
                [10, 11, 12, -1, -1],  # Codebook 0: no delay
                [-1, 20, 21, 22, -1],  # Codebook 1: delay 1
                [-1, -1, 30, 31, 32],  # Codebook 2: delay 2
            ]
        )
        assert torch.equal(delayed_tokens, expected)
        print("✓ Delay pattern application correct")

    def test_reverse_delay_pattern(self):
        """Test reversing delay patterns."""
        print("Testing delay pattern reversal...")

        provider = DelayPatternProvider(strategy="linear", num_codebooks=3)

        # Start with original tokens
        original_tokens = torch.tensor(
            [
                [10, 11, 12],  # Codebook 0
                [20, 21, 22],  # Codebook 1
                [30, 31, 32],  # Codebook 2
            ]
        )

        # Apply delay pattern
        delay_pattern = provider.generate_delay_pattern(sequence_length=3)
        delayed_tokens = provider.apply_delay_pattern(original_tokens, delay_pattern)

        # Reverse it
        reconstructed = provider.reverse_delay_pattern(
            delayed_tokens, delay_pattern, original_length=3
        )

        # Should match original
        assert torch.equal(reconstructed, original_tokens)
        print("✓ Delay pattern reversal correct")


class TestAudioTokenUtils:
    """Test AudioTokenUtils functionality."""

    def test_validate_codebook_sequences(self):
        """Test codebook sequence validation."""
        print("Testing codebook sequence validation...")

        utils = AudioTokenUtils(num_codebooks=3)

        # Valid sequences (all same length)
        valid_sequences = [
            torch.tensor([[1, 2, 3, 4]]),  # Codebook 0
            torch.tensor([[5, 6, 7, 8]]),  # Codebook 1
            torch.tensor([[9, 10, 11, 12]]),  # Codebook 2
        ]
        assert utils.validate_codebook_sequences(valid_sequences)

        # Invalid: wrong number of codebooks
        invalid_num = [
            torch.tensor([[1, 2, 3, 4]]),  # Only 1 codebook
        ]
        assert not utils.validate_codebook_sequences(invalid_num)

        # Invalid: different lengths
        invalid_lengths = [
            torch.tensor([[1, 2, 3, 4]]),  # Length 4
            torch.tensor([[5, 6, 7]]),  # Length 3
            torch.tensor([[9, 10, 11, 12]]),  # Length 4
        ]
        assert not utils.validate_codebook_sequences(invalid_lengths)
        print("✓ Codebook sequence validation correct")

    def test_interleave_codebook_tokens(self):
        """Test interleaving codebook tokens."""
        print("Testing codebook token interleaving...")

        utils = AudioTokenUtils(num_codebooks=2)

        # Create test tokens
        codebook_tokens = [
            torch.tensor([[10, 11, 12]]),  # Codebook 0: [batch_size=1, seq_len=3]
            torch.tensor([[20, 21, 22]]),  # Codebook 1: [batch_size=1, seq_len=3]
        ]

        interleaved = utils.interleave_codebook_tokens(codebook_tokens)

        # Should be [batch_size=1, total_length=6]
        # Pattern: [t0_cb0, t0_cb1, t1_cb0, t1_cb1, t2_cb0, t2_cb1]
        expected = torch.tensor([[10, 20, 11, 21, 12, 22]])
        assert torch.equal(interleaved, expected)
        print("✓ Codebook token interleaving correct")

    def test_extract_codebook_tokens(self):
        """Test extracting codebook tokens from interleaved sequence."""
        print("Testing codebook token extraction...")

        utils = AudioTokenUtils(num_codebooks=2)

        # Interleaved sequence
        interleaved = torch.tensor([[10, 20, 11, 21, 12, 22]])

        extracted = utils.extract_codebook_tokens(interleaved)

        # Should recover original codebooks
        expected_cb0 = torch.tensor([[10, 11, 12]])
        expected_cb1 = torch.tensor([[20, 21, 22]])

        assert len(extracted) == 2
        assert torch.equal(extracted[0], expected_cb0)
        assert torch.equal(extracted[1], expected_cb1)
        print("✓ Codebook token extraction correct")

    def test_interleave_with_delay_pattern(self):
        """Test interleaving with delay patterns."""
        print("Testing interleaving with delay patterns...")

        utils = AudioTokenUtils(
            num_codebooks=2,
            audio_stream_bos_id=999,
            audio_stream_eos_id=888,
        )

        # Create test tokens
        codebook_tokens = [
            torch.tensor([[10, 11]]),  # Codebook 0: [batch_size=1, seq_len=2]
            torch.tensor([[20, 21]]),  # Codebook 1: [batch_size=1, seq_len=2]
        ]

        # Create delay pattern: [0, 1] delays
        delay_pattern = torch.tensor(
            [
                [0, 0],  # Codebook 0: no delay
                [1, 1],  # Codebook 1: delay 1
            ]
        )

        interleaved = utils.interleave_codebook_tokens(codebook_tokens, delay_pattern)

        # Should pad codebook 1 with BOS token at start and EOS at end
        # Codebook 0: [10, 11] -> [10, 11, 888] (padded to length 3)
        # Codebook 1: [20, 21] -> [999, 20, 21] (delayed by 1)
        # Interleaved: [10, 999, 11, 20, 888, 21]
        expected = torch.tensor([[10, 999, 11, 20, 888, 21]])
        assert torch.equal(interleaved, expected)
        print("✓ Interleaving with delay patterns correct")


class TestStreamingCollationState:
    """Test StreamingCollationState functionality."""

    def test_streaming_state_initialization(self):
        """Test streaming state initialization."""
        print("Testing streaming state initialization...")

        state = StreamingCollationState(num_codebooks=4, chunk_overlap_frames=50)

        assert state.num_codebooks == 4
        assert state.chunk_overlap_frames == 50
        assert state.tail_context is None
        assert state.delay_offsets is None
        print("✓ Streaming state initialization correct")

    def test_update_with_chunk(self):
        """Test updating state with new chunks."""
        print("Testing streaming state update...")

        state = StreamingCollationState(num_codebooks=2, chunk_overlap_frames=2)

        # Test chunk
        chunk_tokens = torch.tensor([[1, 2, 3, 4, 5]])
        delay_pattern = torch.zeros(2, 5)

        processed, carry_over = state.update_with_chunk(chunk_tokens, delay_pattern)

        # For now, this is a simple passthrough
        assert torch.equal(processed, chunk_tokens)
        assert torch.equal(carry_over, chunk_tokens[-2:])  # Last 2 frames
        print("✓ Streaming state update correct")


class TestDelayPatternIntegration:
    """Test delay pattern integration with HiggsAudioSampleCollator."""

    def test_collator_with_delay_patterns(self):
        """Test collator with delay pattern functionality enabled."""
        print("Testing collator with delay patterns...")

        config = HiggsAudioConfig()
        collator = HiggsAudioSampleCollator(
            config,
            enable_delay_pattern=True,
            delay_pattern_strategy="linear",
            num_codebooks=3,
        )

        # Check that delay pattern components are initialized
        assert collator.enable_delay_pattern is True
        assert collator.delay_pattern_provider is not None
        assert collator.audio_token_utils is not None
        assert collator.streaming_state is not None
        assert collator.delay_pattern_provider.strategy == "linear"
        assert collator.audio_token_utils.num_codebooks == 3
        print("✓ Collator delay pattern initialization correct")

    def test_delay_pattern_validation(self):
        """Test delay pattern constraint validation."""
        print("Testing delay pattern validation...")

        config = HiggsAudioConfig()
        collator = HiggsAudioSampleCollator(
            config,
            enable_delay_pattern=True,
            num_codebooks=3,
        )

        # Valid delay pattern (non-decreasing delays)
        valid_tokens = torch.tensor(
            [
                [1, 2, 3, 4],  # Codebook 0
                [5, 6, 7, 8],  # Codebook 1
                [9, 10, 11, 12],  # Codebook 2
            ]
        )
        valid_pattern = torch.tensor(
            [
                [0, 0, 0, 0],  # Codebook 0: no delay
                [1, 1, 1, 1],  # Codebook 1: delay 1
                [2, 2, 2, 2],  # Codebook 2: delay 2
            ]
        )

        assert collator._validate_delay_pattern_constraints(valid_tokens, valid_pattern)

        # Invalid delay pattern (excessive delay)
        invalid_pattern = torch.tensor(
            [
                [0, 0, 0, 0],  # Codebook 0: no delay
                [10, 10, 10, 10],  # Codebook 1: delay 10 (exceeds seq_len=4)
                [20, 20, 20, 20],  # Codebook 2: delay 20
            ]
        )

        assert not collator._validate_delay_pattern_constraints(valid_tokens, invalid_pattern)
        print("✓ Delay pattern validation correct")


def test_delay_pattern_end_to_end():
    """End-to-end test of delay pattern functionality."""
    print("\n=== Running End-to-End Delay Pattern Test ===")

    # Initialize components
    provider = DelayPatternProvider(strategy="custom", num_codebooks=4)
    utils = AudioTokenUtils(num_codebooks=4)

    # Create multi-codebook tokens
    codebook_tokens = [
        torch.tensor([[100, 101, 102]]),  # Codebook 0
        torch.tensor([[200, 201, 202]]),  # Codebook 1
        torch.tensor([[300, 301, 302]]),  # Codebook 2
        torch.tensor([[400, 401, 402]]),  # Codebook 3
    ]

    # Generate delay pattern
    delay_pattern = provider.generate_delay_pattern(sequence_length=3, n_codebooks=4)
    print(f"Generated delay pattern: {delay_pattern}")

    # Apply delay pattern to tokens (stack them first to get proper shape)
    stacked_tokens = torch.stack([cb.squeeze(0) for cb in codebook_tokens], dim=0)
    print(f"Stacked tokens shape: {stacked_tokens.shape}")

    delayed_tokens = provider.apply_delay_pattern(stacked_tokens, delay_pattern)
    print(f"Delayed tokens shape: {delayed_tokens.shape}")
    print("✓ Delay patterns applied successfully")

    # Test reversing the delay pattern
    reconstructed = provider.reverse_delay_pattern(delayed_tokens, delay_pattern, original_length=3)
    assert torch.equal(reconstructed, stacked_tokens)
    print("✓ Delay pattern reversal successful")

    # Validate sequences
    assert utils.validate_codebook_sequences(codebook_tokens)
    print("✓ Original sequences validated")

    # Test interleaving
    interleaved = utils.interleave_codebook_tokens(codebook_tokens)
    print(f"Interleaved shape: {interleaved.shape}")

    # Test extraction
    extracted = utils.extract_codebook_tokens(interleaved)
    for i, (original, extracted_tokens) in enumerate(zip(codebook_tokens, extracted)):
        assert torch.equal(original, extracted_tokens)
    print("✓ Token extraction successful")

    print("🎉 End-to-end delay pattern test passed!")


if __name__ == "__main__":
    print("🎵 Running HiggsAudio Delay Pattern Tests 🎵\n")

    # Run all test classes
    test_classes = [
        TestDelayPatternProvider(),
        TestAudioTokenUtils(),
        TestStreamingCollationState(),
        TestDelayPatternIntegration(),
    ]

    for test_class in test_classes:
        print(f"\n--- {test_class.__class__.__name__} ---")
        for method_name in dir(test_class):
            if method_name.startswith("test_"):
                print(f"\nRunning {method_name}...")
                method = getattr(test_class, method_name)
                method()

    # Run end-to-end test
    test_delay_pattern_end_to_end()

    print("\n🎉 All delay pattern tests passed! 🎉")
