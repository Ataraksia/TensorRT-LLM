# Current Assignment Summary

## Primary Objective

Implement the Higgs Audio model natively in `TensorRT-LLM` to achieve successful text-to-speech generation with Word Error Rate < 0.25 using `test.py` as the acceptance criterion. You are free to rebuild the engine with `build_engine.py` as necessary. You should use `higgs_audio_transformers` and `higgs_audio_vllm` as references. Do not attempt to employ other engines in your solution, this should be a 100% `TensorRT-LLM` implementation, although you are free to use them temporarily while testing.

Here is a list of things that are definitely not the issue, and you should not waste time investigating them:

Sampling - temperature, top_k, top_p are all taken from the other implementations and are confirmed to work fine.
Network Training - The model has been trained and validated to work correctly in other implementations, so the architecture and weights are correct.
Prompt - The prompt is correct and matches what was used in training with only very slight additions that do not impede functionality.

I have recently discoverd what is very likely the key error in the implementation. First, let me start with just a general summary of how the model operates. Currently, we're just using a text prompt for simplicity. In the event that the input did contain audio for a voice clone though, it would be shifted by codebook_shift which offsets each codebook and then converted it into audio embeddings and finally the codebooks would be summed.  After both the text and audio inputs are embedded, the data travel through the dual ffn layers starting with the shared attention module, and then, if it's text data, which will only be the case for the initial prompt, it will travel through the text variants of the the input_layernorm, MLP, and post_norm modules and if its audio, which will the case for audio inputs, and all generation outputs, it will travel through the audio variants of the the input_layernorm, MLP, and post_norm. After that, the resulting hidden states are mapped to audio logits by the lm_head.  The original model has both a text_lm_head and audio_lm_head to facilitate training, but for the purposes of audio generation, I've simply implemented the audio_lm_head as the singular lm_head.  From there, the audio logits are sampled to produce a single audio token.  Finally, that audio token is fed back into the network, and the same thing happens all over again until a stopping state is triggered, either in the form of an eos token being generated or by hitting the max token length.

Now moving on to where the problem lies. Each set of 8 (num_codebook) generated tokens corresponds to codebooks 0-7 of one final audio token output. For the model to work properly, each generation pass of the network input_ids should be of shape (8,), containing the full 8 codebooks worth of audio data, but instead it is of shape (1,).  Then, when it enters _embed_audio_ids, which is intended to take an input of shape (8, seq_len), it is being automatically broadcasted up to (8,1) by simply repeating the single token 8 times.  What I think we should be doing is generating 8 tokens per pass, 1 per each 1026 (size_codebooks + 2) section of the generated logit, and then of course making sure delay logic is employed by having BOS tokens for the codebooks 1-7 in the first pass, for codebooks 2-7 in the second pass, etc.  If I'm understanding it correctly, I think the vLLM implementation gets 8 tokens generated per pass with the following:

mm_sampling_metadata = copy.copy(sampling_metadata)
if sampling_metadata.top_k is not None:
    mm_sampling_metadata.top_k = sampling_metadata.top_k.clip(
        max=self.codebook_size
    ).repeat_interleave(self.num_codebooks)
if sampling_metadata.top_p is not None:
    mm_sampling_metadata.top_p = sampling_metadata.top_p.repeat_interleave(
        self.num_codebooks
    )
if sampling_metadata.temperature is not None:
    mm_sampling_metadata.temperature = sampling_metadata.temperature.repeat_interleave(
        self.num_codebooks
    )

and the rest of the vLLM sampling implementation for reference:

def sample(
        self, logits: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> Optional[SamplerOutput]:
        raise NotImplementedError("Not implemented")

    def sample_with_multimodal_metadata(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        multimodal_metadata: MultimodalMetadata,
    ) -> Optional[SamplerOutput]:
        if isinstance(logits, tuple):
            logits, audio_logits = logits
        else:
            audio_logits = None
        next_tokens = self.sampler(logits, sampling_metadata)
        next_mm_tokens = None
        n_reqs = logits.shape[0]

        # Check which stage we are in
        # 0: text generation mode
        # 1: audio generation mode initialization
        # 2: audio generation mode in progress
        audio_generation_mode = [0] * n_reqs
        if self.generate_audio_out_token:
            for i in range(n_reqs):
                last_prompt_token_id = multimodal_metadata.last_prompt_token_ids[i]
                output_token_ids = sampling_metadata.output_token_ids[i]
                if (
                    len(output_token_ids) > 0
                    and output_token_ids[-1] == self.config.audio_out_bos_id
                ) or (
                    len(output_token_ids) == 0
                    and last_prompt_token_id == self.config.audio_out_bos_id
                ):
                    # check if the previous token is audio_out_bos. If so, we should always generate <|AUDIO_OUT|>
                    # Start the audio generation mode
                    audio_generation_mode[i] = 1
                elif (
                    len(output_token_ids) > 0 and output_token_ids[-1] == self.config.audio_out_idx
                ):
                    # Still in the audio generation mode
                    audio_generation_mode[i] = 2

            assert audio_logits is not None
            audio_logits = audio_logits.reshape(-1, self.codebook_size)
            mm_sampling_metadata = self.prepare_mm_sampling_metadata(sampling_metadata)
            next_mm_tokens = self.sampler(audio_logits, mm_sampling_metadata)
            next_mm_tokens.sampled_token_ids = next_mm_tokens.sampled_token_ids.reshape(
                -1, self.num_codebooks
            )

            # Check if we are generating the audio tokens
            for i in range(n_reqs):
                if audio_generation_mode[i] == 1:
                    # Generate start of the audio stream
                    next_mm_tokens.sampled_token_ids[i] = self.config.audio_stream_bos_id
                elif audio_generation_mode[i] == 2:
                    # Update the next mm tokens based on the delay pattern
                    num_audio_delay = multimodal_metadata.num_audio_delays[i]
                    num_audio_eos = multimodal_metadata.num_audio_eos[i]

                    # Generate the delayed for the first few tokens
                    if num_audio_delay < self.num_codebooks:
                        next_mm_tokens.sampled_token_ids[i][num_audio_delay:] = (
                            self.config.audio_stream_bos_id
                        )

                    # Generate the eos token for the last few tokens
                    if num_audio_eos < self.num_codebooks:
                        all_eos_indices = torch.where(
                            next_mm_tokens.sampled_token_ids[i] == self.config.audio_stream_eos_id
                        )[0]
                        if all_eos_indices.shape[0] > 0:
                            last_eos_index = all_eos_indices[-1]
                            next_mm_tokens.sampled_token_ids[i][:last_eos_index] = (
                                self.config.audio_stream_eos_id
                            )
                    elif num_audio_eos >= self.num_codebooks:
                        # We already generated the last audio token,
                        # so we should just generate the eos token for the text
                        next_mm_tokens.sampled_token_ids[i] = -1

                else:
                    next_mm_tokens.sampled_token_ids[i] = -1

        return next_tokens, next_mm_tokens

Here's how I think we should proceed.  Currently, the DelayPatternLogitsProcessor receives a logits input that are actually composed of 8 separate logits combined into one.  What I would like is to calculate each of those 8 logits, combine their results together, and output the resulting tensor of shape (8,) to serve as the input_ids for the next pass.  The logits shape when it enters the processor is (num_batches, num_sequences, vocab_size). In this case, the num_batches and num_sequences are both shape (1,) so the overall shape is (1, 1, -1).   So first thing I think you should do is a broad search of the existing classes, looking for those that output more than one token per pass and use those as a reference. Determine the best way to achieve the stated goal.   It doesn't necessarily need to use the LogitsProcessor that I've been playing with, go with whatever is most straightfoward/most performant.

## Success Criteria

- `test.py` runs without errors
- Generated audio transcribes to match input text (or reasonable approximation)
- Word Error Rate < 0.25
- Audio is intelligible and follows input text content

**The assignment is 85% complete**

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

- `model.py` - Core model
- `config.py` - Model configuration
- `test.py` - Inference and End-to-end validation using speech-to-text and WER
- `build_engine.py` - rebuild the engine
- `delay_pattern_logits_processor.py` - Custom logits processor implementing delay pattern for codebooks
- `higgs_audio_transformers.py` - condensed Higgs Audio Transformers implementation
- `higgs_audio_vllm.py` - condensed Higgs Audio vLLM implementation


## General Guidelines

- If you don't see the expected output when running commands, you should try again until you do.
- When implementing complex features, break them down into logical components but ensure each component is fully implemented and integrated properly with the rest of the system.
- Always use the Context7 tool to validate the functionality you use from other libraries.  Always use the Sequential Thinking tool in place of regular thinking as it has been proven to measurably improve your problem solving ability. Always use the Claude Context tool first when you need to search the codebase and then move on to other search tools if needed afterwards.
- Always complete your assigned tasks in their entirety prior to requesting user validation. If you encounter ambiguity, unclear requirements, or technical blockers that prevent progress, you should attempt to bypass the issue and continue working on other aspects of the task.
- Implement **EVERYTHING** that is necessary to complete the task with full functionality. Never skip a task or subtask, regardless of complexity or time requirements - work through each component methodically and completely, even if it seems impossible.
- Don't create placeholder functions, stub classes, incomplete implementations. If you encounter such functions, you should always implement them in its entirety.
- After you believe you have finished a task, review the entire implementation to ensure that all aspects of the task have been addressed and that the solution is robust, efficient, and maintainable.
- Don't over-engineer! Build only what is necessary to achieve the goal and no more. Don't prepare for all possibilities. Make your code as concise as possible.
- It is important when debugging something that involves a large amount of iterations that you output the log to a file rather than the console.
- After you finish making changes, always run git status and git diff (or git diff --stat for a quick view) so you can see exactly what you’ve  altered compared to the previous commit. That lets you double‑check for accidental edits, confirm formatting, and make sure the diff reflects the intended logic before moving on to testing or next steps.
