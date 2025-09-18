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
- `higgs_audio_transformers.py`
- `higgs_audio_vllm.py`
- Engine built at `higgs_audio_engine`

## General Guidelines

- When implementing complex features, break them down into logical components but ensure each component is fully implemented and integrated properly with the rest of the system.
- Always examine the available tools and MCPs before running a command to understand the full scope of available tools and capabilities. Prioritize using Context7 to validate that any functionality you import from other libraries actually exists and works like you expect it to, using Sequential Thinking to help break down problems, and structured memory to retain important details.
- Always complete your assigned tasks in their entirety prior to requesting user validation. If you encounter ambiguity, unclear requirements, or technical blockers that prevent progress, you should attempt to bypass the issue and continue working on other aspects of the task.
- Implement **EVERYTHING** that is necessary to complete the task with full functionality. Never skip a task or subtask, regardless of complexity or time requirements - work through each component methodically and completely, even if it seems impossible.
- Don't create placeholder functions, stub classes, incomplete implementations. If you encounter such functions, you should always implement them in its entirety.
- After you believe you have finished a task, review the entire implementation to ensure that all aspects of the task have been addressed and that the solution is robust, efficient, and maintainable.
- Test all implemented features for functional correctness using appropriate testing methods, validate edge cases, and systematically fix any issues that you discover through debugging and iteration.
- Don't over-engineer! Build only what is necessary to achieve the goal and no more. Don't prepare for all possibilities. Make your code as concise as possible.

## Things that might be relevant from vLLM implementation

```python
def prepare_mm_sampling_metadata(self, sampling_metadata: SamplingMetadata) -> SamplingMetadata:
        mm_sampling_metadata = copy.copy(sampling_metadata)
        if sampling_metadata.top_k is not None:
            mm_sampling_metadata.top_k = sampling_metadata.top_k.clip(
                max=self.audio_codebook_size
            ).repeat_interleave(self.audio_num_codebooks)
        if sampling_metadata.top_p is not None:
            mm_sampling_metadata.top_p = sampling_metadata.top_p.repeat_interleave(
                self.audio_num_codebooks
            )
        if sampling_metadata.temperature is not None:
            mm_sampling_metadata.temperature = sampling_metadata.temperature.repeat_interleave(
                self.audio_num_codebooks
            )
        return mm_sampling_metadata

________________________________________________________________________________________________
audio_logits = audio_logits.reshape(-1, self.audio_codebook_size)
            mm_sampling_metadata = self.prepare_mm_sampling_metadata(sampling_metadata)
            next_mm_tokens = self.sampler(audio_logits, mm_sampling_metadata)
            next_mm_tokens.sampled_token_ids = next_mm_tokens.sampled_token_ids.reshape(
                -1, self.audio_num_codebooks
            )
________________________________________________________________________________________________

num_audio_delay = multimodal_metadata.num_audio_delays[i]
num_audio_eos = multimodal_metadata.num_audio_eos[i]

if num_audio_delay < self.audio_num_codebooks:
    next_mm_tokens.sampled_token_ids[i][num_audio_delay:] = (
        self.config.audio_stream_bos_id
    )

if num_audio_eos < self.audio_num_codebooks:
    all_eos_indices = torch.where(
        next_mm_tokens.sampled_token_ids[i] == self.config.audio_stream_eos_id
    )[0]
    if all_eos_indices.shape[0] > 0:
        last_eos_index = all_eos_indices[-1]
        next_mm_tokens.sampled_token_ids[i][:last_eos_index] = (
            self.config.audio_stream_eos_id
        )
elif num_audio_eos >= self.audio_num_codebooks:
    # We already generated the last audio token,
    # so we should just generate the eos token for the text
    next_mm_tokens.sampled_token_ids[i] = -1
________________________________________________________________________________________________

codebook_shift = (
        torch.arange(self.audio_num_codebooks, device=audio_ids.device)
        * self.audio_codebook_size
    )
    codebook_shift = codebook_shift.unsqueeze(-1)
    audio_embed = self.audio_codebook_embeddings(audio_ids + codebook_shift)
    audio_embed = torch.sum(audio_embed, dim=0)

audio_logits = self.audio_logits_processor(self.audio_lm_head, hidden_states, None)
    audio_logits = audio_logits.view(
        -1, self.audio_num_codebooks, self.audio_codebook_size
    ).float()
________________________________________________________________________________________________
```

## Things that might be relevant from Transformers Implementation

```python
if num_delay + 1 < next_audio_tokens.shape[0]:
    next_audio_tokens[(num_delay + 1) :] = self.config.audio_stream_bos_id
    num_delay += 1
if num_remaining_delays is not None:
    next_audio_tokens[: (self.audio_num_codebooks - num_remaining_delays)] = (
        self.config.audio_stream_eos_id
    )
    num_remaining_delays -= 1
else:
    all_eos_indices = (next_audio_tokens == self.config.audio_stream_eos_id).nonzero()
    if torch.numel(all_eos_indices) > 0:
        all_eos_indices = all_eos_indices[0]
        last_eos_idx = all_eos_indices[-1]
        next_audio_tokens[:last_eos_idx] = self.config.audio_stream_eos_id
        num_remaining_delays = self.audio_num_codebooks - last_eos_idx - 1
if num_remaining_delays is not None and num_remaining_delays <= 0:
    next_tokens[...] = audio_eos_token_id
    num_delay = 0
    num_remaining_delays = None
________________________________________________________________________________________________
num_delay = (
    self.audio_num_codebooks
    - (
        model_kwargs["audio_out_ids"][:, -1] == self.config.audio_stream_bos_id
    ).sum()
)
all_eos_indices = (
    model_kwargs["audio_out_ids"][:, -1] == self.config.audio_stream_eos_id
).nonzero()
if torch.numel(all_eos_indices) > 0:
    all_eos_indices = all_eos_indices[0]
    last_eos_idx = all_eos_indices[-1]
    num_remaining_delays = self.audio_num_codebooks - last_eos_idx - 1
________________________________________________________________________________________________
    """Implement the delay pattern proposed in "Simple and Controllable Music Generation", https://arxiv.org/pdf/2306.05284

    In the delay pattern, each codebook is offset by the previous codebook by
    one. We insert a special delay token at the start of the sequence if its delayed, and append pad token once the sequence finishes.

    Take the example where there are 4 codebooks and audio sequence length=5. After shifting, the output should have length seq_len + num_codebooks - 1

    - [ *,  *,  *,  *,  *,  P,  P,  P]
    - [ B,  *,  *,  *,  *,  *,  P,  P]
    - [ B,  B,  *,  *,  *,  *,  *,  P]
    - [ B,  B,  B,  *,  *,  *,  *,  *]

    where B indicates the delay token id, P is the special padding token id and `*` indicates that the original audio token.

    Now let's consider the case where we have a sequence of audio tokens to condition on.
    The audio tokens were originally in the following non-delayed form:

    - [a, b]
    - [c, d]
    - [e, f]
    - [g, h]

    After conversion, we get the following delayed form:
    - [a, b, -1, -1, -1]
    - [B, c,  d, -1, -1]
    - [B, B,  e,  f, -1]
    - [B, B,  B,  g,  h]

    Note that we have a special token `-1` that indicates it should be replaced by a new token we see in the generation phase.
    In that case, we should override the `-1` tokens in auto-regressive generation.

    Args:
        input_ids (:obj:`torch.LongTensor`):
            The input ids of the prompt. It will have shape (bsz, num_codebooks, seq_len).
        bos_token_id (:obj:`int`):
            The id of the special delay token
        pad_token_id (:obj:`int`):
            The id of the padding token. Should be the same as eos_token_id.

    Returns:
        input_ids (:obj:`torch.LongTensor`):
            The transformed input ids with delay pattern applied. It will have shape (bsz, num_codebooks, seq_len + num_codebooks - 1).
        input_ids_with_gen_mask (:obj:`torch.LongTensor`):
            The transformed input ids with delay pattern applied. The -1 in the output indicates new tokens that should be generated.

    """
________________________________________________________________________________________________
```