# HiggsAudio Model Integration Guide

## Overview

The HiggsAudio model has been successfully integrated with TensorRT-LLM to support audio-enhanced text generation. The implementation uses TensorRT-LLM's prompt embedding table mechanism to seamlessly integrate audio features with text processing.

## Architecture

### Core Components

1. **HiggsAudioTransformer**: Main transformer with multimodal support
2. **HiggsAudioDualFFNDecoderLayer**: Decoder layers with dual FFN architecture for text/audio processing
3. **HiggsAudioForCausalLM**: Complete causal language model with audio output projection
4. **HiggsAudioTRTRunner**: Runtime inference wrapper

### Audio Feature Integration

Audio features are integrated through TensorRT-LLM's **prompt embedding table mechanism**:

- Audio features are encoded and passed as `prompt_embedding_table`
- Virtual token IDs (>= vocab_size) replace `<|AUDIO|>` placeholders in input text
- The PromptTuningEmbedding layer handles the mapping between virtual tokens and audio features
- Audio features are seamlessly integrated into the transformer's attention mechanism

## Build Process

### Engine Building

```bash
python3 build_higgs_audio_engine.py --log_level info
```

The build process:
1. Creates a HiggsAudioConfig with default LLaMA-3.2-3B parameters
2. Instantiates HiggsAudioForCausalLM model
3. Enables prompt tuning for multimodal support (max_prompt_embedding_table_size=128)
4. Builds TensorRT engine with audio capabilities

### Key Build Configuration

```python
build_config = BuildConfig(
    max_input_len=max_seq_len,
    max_seq_len=max_seq_len,
    max_batch_size=1,
    max_beam_width=1,
    max_num_tokens=max_seq_len,
    max_prompt_embedding_table_size=num_mul_bins,  # 128 for audio features
    kv_cache_type=KVCacheType.PAGED,
    # ... other configs
)
```

## Runtime Usage

### Basic Audio-Enhanced Generation

```python
from tensorrt_llm.models.higgs_audio.model import HiggsAudioTRTRunner

# Initialize runner
runner = HiggsAudioTRTRunner(
    engine_dir="./higgs_audio_engine",
    tokenizer_dir="bosonai/higgs-audio-v2-generation-3B-base",
    audio_tokenizer_dir="bosonai/higgs-audio-v2-tokenizer",
)

# Generate with audio input
response = runner.generate(
    input_text="Transcribe and summarize this audio: <|AUDIO|>",
    input_audio="path/to/audio.wav"
)
```

### How Audio Processing Works

1. **Audio Encoding**: Audio is encoded using the HiggsAudio tokenizer
2. **Feature Preparation**: Audio features are shaped as `[batch_size, num_audio_tokens, hidden_size]`
3. **Virtual Token Mapping**: `<|AUDIO|>` tokens are replaced with virtual token IDs
4. **Prompt Table Setup**: Audio features become the prompt embedding table
5. **Inference**: The model processes text and audio features together

### Advanced Usage

```python
# Manual audio feature processing
audio_features = audio_tokenizer.encode(audio_path, sr=24000)
prompt_table = audio_features.to(dtype=torch.float16, device="cuda")

# Set up prompt parameters
batch_size, num_audio_tokens = prompt_table.shape[:2]
prompt_tasks = torch.zeros((batch_size, num_audio_tokens), dtype=torch.int32, device="cuda")
prompt_vocab_size = torch.tensor([num_audio_tokens], dtype=torch.int32, device="cuda")

# Run inference with ModelRunnerCpp
outputs = runner.runner.generate(
    batch_input_ids=batch_input_ids,
    sampling_config=sampling_config,
    prompt_table=prompt_table,
    prompt_tasks=prompt_tasks,
    prompt_vocab_size=prompt_vocab_size,
)
```

## Model Configuration

### HiggsAudioConfig Parameters

```python
config = HiggsAudioConfig(
    # Text model configuration (LLaMA-3.2-3B base)
    hidden_size=3072,
    num_hidden_layers=28,
    num_attention_heads=24,
    vocab_size=128256,
    
    # Audio-specific configuration
    audio_num_codebooks=8,
    audio_codebook_size=1024,
    audio_bos_token_id=128011,
    audio_eos_token_id=128012,
    audio_in_token_idx=128015,  # <|AUDIO|> token ID
    
    # Dual FFN configuration
    audio_adapter_type="dual_ffn_fast_forward",
    audio_ffn_intermediate_size=14336,
)
```

## Testing

Run the integration test to verify everything works:

```bash
python3 test_higgs_audio_integration.py
```

This test verifies:
- Model creation and configuration
- Prompt tuning setup
- Audio feature simulation
- Engine build compatibility

## Technical Details

### Prompt Embedding Table Mechanism

The audio integration leverages TensorRT-LLM's prompt tuning infrastructure:

1. **PromptTuningEmbedding**: Handles virtual tokens (ID >= vocab_size)
2. **Virtual Token Mapping**: Audio placeholder tokens → virtual token IDs
3. **Feature Lookup**: Virtual tokens → audio features from prompt_embedding_table
4. **Seamless Integration**: Audio features flow through standard transformer layers

### Dual FFN Architecture

The model includes dual FFN layers for specialized text/audio processing:
- **Text FFN**: Standard transformer FFN for text tokens
- **Audio FFN**: Specialized FFN for audio-derived tokens
- **Runtime Switching**: Uses token masks to route processing (currently simplified for build compatibility)

### Memory and Performance

- **Engine Size**: ~4.7GB for the base model
- **Audio Feature Size**: Configurable, typically 64-256 tokens per audio segment
- **Memory Efficient**: Uses paged KV cache and optimized attention plugins

## Troubleshooting

### Common Issues

1. **Build Failures**: Ensure all dependencies are installed and CUDA is available
2. **Audio Tokenizer**: Verify the audio tokenizer model is accessible
3. **Memory Issues**: Adjust max_prompt_embedding_table_size if needed
4. **Token Mapping**: Ensure <|AUDIO|> token ID matches configuration

### Debug Tips

- Use `--log_level verbose` for detailed build logs
- Check engine files exist in output directory
- Verify audio tokenizer can encode test audio
- Test with simple text-only inputs first

## Future Enhancements

- Runtime dual FFN switching based on audio token masks
- Support for multiple audio segments per input
- Optimized audio feature caching
- Integration with streaming audio processing
