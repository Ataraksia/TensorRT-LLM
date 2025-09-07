# HiggsAudio Delay Pattern Implementation

## Overview

The delay pattern is a crucial component of the HiggsAudio model that enables simultaneous code generation across multiple codebooks while supporting streaming audio generation. This implementation is based on Residual Vector Quantization (RVQ) where audio is encoded using multiple codebooks.

## What is the Delay Pattern?

In RVQ-based audio models, audio is represented using multiple codebooks (typically 8 for HiggsAudio). Each codebook captures different aspects of the audio signal. The delay pattern allows the model to generate codes for all codebooks simultaneously while maintaining causal dependencies.

### Visual Example

For 4 codebooks and sequence length 5, the delay pattern transforms:

**Original Pattern:**
```
Codebook 0: [a, b, c, d, e]
Codebook 1: [f, g, h, i, j]  
Codebook 2: [k, l, m, n, o]
Codebook 3: [p, q, r, s, t]
```

**Delayed Pattern (with BOS=B, PAD=P):**
```
Codebook 0: [a, b, c, d, e, P, P, P]
Codebook 1: [B, f, g, h, i, j, P, P]
Codebook 2: [B, B, k, l, m, n, o, P]
Codebook 3: [B, B, B, p, q, r, s, t]
```

## Implementation Details

### Core Functions

#### 1. `_build_delay_pattern_mask()`
```python
def _build_delay_pattern_mask(input_ids, bos_token_id, pad_token_id):
    """Apply delay pattern to RVQ codes for simultaneous generation."""
```

**Purpose**: Transforms original RVQ codes into delayed format for training/inference.

**Process**:
1. Each codebook is offset by its index (codebook 0: no delay, codebook 1: 1 delay, etc.)
2. BOS tokens fill the delay positions at the start
3. PAD tokens fill the end positions after the sequence
4. Output length = original_length + num_codebooks - 1

#### 2. `revert_delay_pattern()`
```python
def revert_delay_pattern(data):
    """Convert delayed codes back to original RVQ format."""
```

**Purpose**: Removes delay pattern from generated codes to recover original audio format.

**Process**:
1. Extract the actual audio codes from each delayed codebook
2. Remove BOS/PAD tokens
3. Reconstruct original (num_codebooks, seq_len) format

### Integration in HiggsAudioTRTRunner

#### New Method: `generate_with_delay_pattern()`

This method provides full delay pattern support for audio generation:

```python
def generate_with_delay_pattern(
    self,
    input_text: str,
    input_audio: str = None,
    use_delay_pattern: bool = True,
    **generation_kwargs,
) -> tuple[str, torch.Tensor]:
```

**Features**:
- Applies delay pattern to input audio codes
- Generates with delay pattern awareness
- Automatically reverts delay pattern from outputs
- Returns both text and audio codes

#### Audio Code Embedding: `_embed_audio_codes()`

Converts RVQ codes to embeddings for the prompt embedding table:

```python
def _embed_audio_codes(self, audio_codes):
    """Convert audio codes to embeddings using codebook embeddings."""
```

**Process**:
1. Apply codebook-specific offsets to codes
2. Look up embeddings from audio codebook embedding table
3. Combine embeddings across codebooks
4. Return embeddings suitable for prompt table

## Usage Examples

### Basic Audio Generation with Delay Pattern

```python
from tensorrt_llm.models.higgs_audio.model import HiggsAudioTRTRunner

# Initialize runner
runner = HiggsAudioTRTRunner(
    engine_dir="./higgs_audio_engine",
    tokenizer_dir="bosonai/higgs-audio-v2-generation-3B-base",
    audio_tokenizer_dir="bosonai/higgs-audio-v2-tokenizer",
)

# Generate with delay pattern (default)
text_output, audio_codes = runner.generate_with_delay_pattern(
    input_text="Generate audio for: Hello world",
    use_delay_pattern=True
)

# audio_codes will be in original RVQ format (num_codebooks, seq_len)
print(f"Generated audio codes shape: {audio_codes.shape}")
```

### Audio Conditioning with Delay Pattern

```python
# Use input audio as conditioning
text_output, audio_codes = runner.generate_with_delay_pattern(
    input_text="Continue this audio: <|AUDIO|>",
    input_audio="path/to/input.wav",
    use_delay_pattern=True
)
```

### Manual Delay Pattern Operations

```python
from tensorrt_llm.models.higgs_audio.model import _build_delay_pattern_mask, revert_delay_pattern

# Apply delay pattern manually
original_codes = torch.randint(0, 1024, (1, 8, 100))  # 8 codebooks, 100 tokens
delayed_codes = _build_delay_pattern_mask(
    original_codes,
    bos_token_id=2000,
    pad_token_id=2001
)

# Revert delay pattern
recovered_codes = revert_delay_pattern(delayed_codes[0])  # Remove batch dim
assert torch.equal(recovered_codes, original_codes[0])
```

## Configuration Parameters

### Audio Stream Tokens

The delay pattern uses special tokens defined in `HiggsAudioConfig`:

```python
class HiggsAudioConfig:
    audio_stream_bos_id: int = 128011  # Beginning of stream token
    audio_stream_eos_id: int = 128012  # End of stream token (also used as PAD)
```

### Codebook Configuration

```python
class HiggsAudioConfig:
    audio_num_codebooks: int = 8       # Number of RVQ codebooks
    audio_codebook_size: int = 1024    # Size of each codebook
```

## Benefits of Delay Pattern

1. **Simultaneous Generation**: All codebooks can be generated in parallel
2. **Streaming Support**: Enables real-time audio generation
3. **Causal Dependencies**: Maintains proper dependencies between codebooks
4. **Efficiency**: Reduces generation time compared to sequential approaches
5. **Quality**: Preserves audio quality through proper RVQ reconstruction

## Testing

The implementation includes comprehensive tests (`test_delay_pattern.py`):

- ✅ Basic delay pattern application
- ✅ Delay pattern reversion  
- ✅ Round-trip consistency
- ✅ Special token handling
- ✅ Multi-codebook support

## Technical Notes

### Memory Considerations

- Delayed sequences are longer: `seq_len + num_codebooks - 1`
- For 8 codebooks and 1000 tokens: 1007 tokens per codebook
- Memory usage scales with number of codebooks

### Performance Implications

- Delay pattern adds minimal computational overhead
- Most processing is tensor reshaping and indexing
- Generation speed depends on sequence length and codebook count

### Compatibility

- Works with existing TensorRT-LLM infrastructure
- Compatible with prompt embedding table mechanism
- Supports both training and inference modes
- Integrates with existing audio tokenizers

## Future Enhancements

- Adaptive delay patterns based on audio content
- Optimized GPU kernels for delay pattern operations
- Support for variable-length sequences
- Integration with streaming audio pipelines
