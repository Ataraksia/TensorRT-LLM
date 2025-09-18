# Current Assignment Summary

## Primary Objective

Implement the Higgs Audio model natively in `TensorRT-LLM` to achieve successful text-to-speech generation with Word Error Rate < 0.25 using `test.py` as the acceptance criterion. You are free to rebuild the engine with `build_engine.py` as necessary. You can use `higgs_audio_transformers` and `higgs_audio_vllm` as references, but do not bring in any additional functionality that they might include unless absolutely necessary. For example, this `TensorRT-LLM` implementation sheds the unnecessary `text_lm_head` that is useful for training but not for implementation and only uses the `audio_lm_head`. Additionally, do not attempt to employ other engines in your solution, this should be a 100% `TensorRT-LLM` implementation, although you are free to use them temporarily while testing.

## Success Criteria

- `test.py` runs without errors
- Generated audio transcribes to match input text (or reasonable approximation)
- Word Error Rate < 0.25
- Audio is intelligible and follows input text content

**The assignment is 75% complete** - architecture and token handling are likely correct, but the generation logic needs refinement to produce meaningful speech instead of degenerate patterns.

## Technical Architecture

Generates speech following a text prompt with an optional audio input for voice cloning.

- `TensorRT-LLM` v1-current with custom `HiggsAudio` model implementation
- Dual FFN decoder layers with separate text/audio MLPs and shared attention
- Custom logits processor implementing delay pattern for 8 codebooks
- Audio vocab size: 8208 (8 codebooks × 1026 tokens each)
- Consists of generic Llama-based text tokenizer and custom built `AudioTokenizer` that decodes VQ codes to waveform using delay pattern reversion

The tokens and prompt used in this implementation are all correct and are what were used in training with very slight additions in the case of the prompt that in no way impede the functionality of the model.

### Token Definitions

- audio_bos_token - Designates beginning of audio sequence in the input
    The special `<|audio_bos|>` token. In Higgs-Audio, it is mapped to 128011,
- audio_eos_token Designates end of audio sequence in both the input and output
    The special - `<|audio_eos|>` token. We use 128012 as the default value,
- audio_out_bos_token - Designates beginning of audio sequence in the output
    The special `<|audio_out_bos|>` token. We use 128013 as the default value,
- audio_token - <|AUDIO|> - 128015
    Designates location in the input for audio features extracted by audio tokenizer for training as well as voice cloning during generation
- audio_out_token - <|AUDIO_OUT|> - 128016
    Designates location in the output for audio tokens to be decoded by audio tokenizer for training purposes.
- Audio stream BOS (1024) - Actual audio generation start token
- Local audio tokens (0-1023) - Content tokens
- Audio stream EOS (1025) - Audio generation end token

### Prompt Template

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Generate audio following instruction.<|scene_desc_start|>
Audio is recorded from a quiet room.
Speaker is an enthusiastic young Australian woman in her early 20s.
She has a bright, high-pitched voice.<|scene_desc_end|><|eot_id|>
<|start_header_id|>user<|end_header_id|>
{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|><|audio_out_bos|>
```

## Key Files & Components

- `model.py` - Core model, logits processor, runner
- `config.py` - Model configuration
- `test.py` - End-to-end validation using speech-to-text and WER
- `build_engine.py` - rebuild the engine
- `higgs_audio_transformers.py` - condensed Higgs Audio Transformers implementation
- `higgs_audio_vllm.py` - condensed Higgs Audio vLLM implementation

## General Guidelines

- You should disregard the previous instruction regarding assuming the terminal executed the command successfully and proceeding with the task if you don't see the expected output.  If you don't see the expected output, you should try again until you do.
- When implementing complex features, break them down into logical components but ensure each component is fully implemented and integrated properly with the rest of the system.
- Always use the Context7 tool to validate the functionality you use from other libraries.  Always use the Sequential Thinking tool in place of regular thinking as it has been proven to measurably improve your problem solving ability. Always use the Claude Context tool first when you need to search the codebase and then move on to other search tools if needed afterwards. Always use the `Agents Notes` section of the `AGENTS.md` file to keep track of your current big picture thoughts.
- Always complete your assigned tasks in their entirety prior to requesting user validation. If you encounter ambiguity, unclear requirements, or technical blockers that prevent progress, you should attempt to bypass the issue and continue working on other aspects of the task.
- Implement **EVERYTHING** that is necessary to complete the task with full functionality. Never skip a task or subtask, regardless of complexity or time requirements - work through each component methodically and completely, even if it seems impossible.
- Don't create placeholder functions, stub classes, incomplete implementations. If you encounter such functions, you should always implement them in its entirety.
- After you believe you have finished a task, review the entire implementation to ensure that all aspects of the task have been addressed and that the solution is robust, efficient, and maintainable.
- Don't over-engineer! Build only what is necessary to achieve the goal and no more. Don't prepare for all possibilities. Make your code as concise as possible.

## Agents Notes
