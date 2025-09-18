"""Higgs-Audio is an end-to-end multimodal model with the capability to understand and generate text / audio."""

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import (
    GenerationConfig,
    GenerationMixin,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from transformers.generation.utils import GenerateNonBeamOutput
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.utils import ModelOutput, logging

from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig

logger = logging.get_logger(__name__)

AUDIO_IN_TOKEN = "<|AUDIO|>"
AUDIO_OUT_TOKEN = "<|AUDIO_OUT|>"
EOS_TOKEN = "<|end_of_text|>"


class HiggsAudioPreTrainedModel(PreTrainedModel):
    config_class = HiggsAudioConfig
    base_model_prefix = "higgs_audio"

    def _init_weights(self, module):
        pass


class GenerationMode(Enum):
    """Enum for different generation modes in HiggsAudio model."""

    TEXT = 0  # Text generation mode
    AUDIO_INIT = 1  # Audio generation mode initialization
    AUDIO_IN_PROGRESS = 2  # Audio generation mode in progress


@dataclass
class AudioContent:
    audio_url: str
    # Base64 encoded audio bytes
    raw_audio: str | None = None
    offset: float | None = None
    duration: float | None = None
    row_id: int | None = None
    type: str = "audio"


@dataclass
class TextContent:
    text: str
    type: str = "text"


@dataclass
class Message:
    role: str
    content: Union[str, AudioContent, TextContent, list[Union[str, AudioContent, TextContent]]]
    recipient: str | None = None


@dataclass
class ChatMLSample:
    """Dataclass to hold multimodal ChatML data."""

    messages: list[Message]
    start_index: int | None = (
        None  # We will mask the messages[:start_index] when finetuning the LLM.
    )
    misc: dict | None = None
    speaker: str | None = None


@dataclass
class HiggsAudioDecoderLayerOutput:
    logits: torch.FloatTensor
    audio_logits: torch.FloatTensor
    attentions: tuple[torch.FloatTensor, ...] | None = None
    past_key_values: tuple[tuple[torch.FloatTensor]] | None = None


class HiggsAudioDecoderProjector(PreTrainedModel):
    """Projection layers that map hidden states from the LLM component to audio / text logits.

    We support two type of audio head:
    - Basic Audio Head:
        Directly map the hidden states to audio logits for all the codebooks.
    """

    def __init__(self, config: HiggsAudioConfig, layer_idx: int | None = None):
        super().__init__(config)
        self.text_lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        self.audio_lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.audio_num_codebooks * (config.audio_codebook_size + 2),
            bias=False,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        hidden_states,
        audio_out_mask,
        label_audio_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        output_audio_hidden_states=False,
        cache_position=None,
    ):
        """Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, seq_len, hidden_size)`):
                Hidden states from the LLM component
            audio_out_mask (`torch.Tensor` of shape `(batch_size, seq_len)`):
                Mask for identifying the audio out tokens.
            label_audio_ids (`torch.Tensor` of shape `(num_codebooks, num_audio_out_tokens)`):
                Label tokens for the audio-out part. This is used for calculating the logits if RQ-Transformer is used.
            attention_mask (`torch.Tensor` of shape `(batch_size, seq_len)`):
                Mask to avoid performing attention on padding token indices


        Returns:
            logits (`torch.Tensor` of shape `(batch_size, seq_len, vocab_size)`):
                Logits for text tokens
            audio_logits (`torch.Tensor` of shape `(num_audio_out_tokens, audio_num_codebooks * audio_codebook_size)`):
                Logits for audio tokens. We ensure `num_text_tokens + num_audio_tokens == batch_size * seq_len`
        """
        logits = self.text_lm_head(hidden_states)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        next_cache = next_decoder_cache if use_cache else None

        audio_logits = self.audio_lm_head(hidden_states[audio_out_mask])

        if output_audio_hidden_states:
            audio_hidden_states = hidden_states[audio_out_mask]
        else:
            audio_hidden_states = None

        return (
            logits,
            audio_logits,
            all_self_attns,
            all_hidden_states,
            audio_hidden_states,
            next_cache,
        )


def build_delay_pattern_mask(
    input_ids: torch.LongTensor,
    bos_token_id: int,
    pad_token_id: int,
):
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
    bsz, num_codebooks, seq_len = input_ids.shape

    new_seq_len = seq_len + num_codebooks - 1
    input_ids_with_gen_mask = torch.ones(
        (bsz, num_codebooks, new_seq_len), dtype=torch.long, device=input_ids.device
    )
    bos_mask = torch.tril(input_ids_with_gen_mask, -1) > 0
    eos_mask = torch.triu(input_ids_with_gen_mask, seq_len) > 0
    input_ids_with_gen_mask[bos_mask] = bos_token_id
    input_ids_with_gen_mask[(~bos_mask) & (~eos_mask)] = input_ids.reshape(-1)
    input_ids = input_ids_with_gen_mask.clone()
    input_ids[eos_mask] = pad_token_id
    input_ids_with_gen_mask[eos_mask] = -1
    return input_ids, input_ids_with_gen_mask


def revert_delay_pattern(data):
    """Convert samples encoded with delay pattern back to the original form.

    Args:
        data (:obj:`torch.Tensor`):
            The data with delay pattern applied. It will have shape (num_codebooks, seq_len + num_codebooks - 1).

    Returns:
        ret (:obj:`torch.Tensor`):
            Recovered data with delay pattern removed. It will have shape (num_codebooks, seq_len).
    """
    assert len(data.shape) == 2
    out_l = []
    num_codebooks = data.shape[0]
    for i in range(num_codebooks):
        out_l.append(data[i : (i + 1), i : (data.shape[1] - num_codebooks + 1 + i)])
    return torch.cat(out_l, dim=0)


def merge_input_ids_with_audio_features(
    audio_features_embed,
    audio_features_length,
    audio_in_embed,
    audio_in_ids_start,
    audio_out_embed,
    audio_out_ids_start,
    audio_in_token_idx,
    audio_out_token_idx,
    inputs_embeds,
    input_ids,
    attention_mask,
    label_ids,
    pad_token_id,
    ignore_index=-100,
    round_to=8,
    left_padding=True,
):
    """Merge input_ids with audio features into final embeddings.

    Args:
        audio_features_embed (`torch.Tensor` of shape `(num_audios, max_audio_tokens, embed_dim)`):
            Encoded vectors of all audios in the batch (obtained from the semantic encoder)
        audio_features_length (`torch.LongTensor` of shape `(num_audios,)`):
            The length of audio embeddings of each audio as stacked in `audio_features_embed`
        audio_in_embed (`torch.Tensor` of shape `(total_num_audio_in_tokens, embed_dim)`):
            The embeddings of audio-in tokens
        audio_in_ids_start (`torch.LongTensor` of shape `(num_audios,)`):
            The start index of the audio-in tokens for each audio
        audio_out_embed (`torch.Tensor` of shape `(total_num_audio_out_tokens, embed_dim)`):
            The embeddings of audio-out tokens
        audio_out_ids_start (`torch.LongTensor` of shape `(num_audios,)`):
            The start index of the audio-out tokens for each audio
        audio_in_token_idx
            The index of the audio-in token in the vocabulary
        audio_out_token_idx
            The index of the audio-out token in the vocabulary
        inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embed_dim)`):
            Token embeddings before merging with audio embeddings
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Input_ids of tokens, possibly filled with audio token
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Mask to avoid performing attention on padding token indices.
        label_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
            labels need to be recalculated to support training (if provided)
        pad_token_id (`int`):
            The index of the pad token in the vocabulary
        ignore_index
            The index to ignore in the loss calculation
        round_to
            The number to round to for padding
        left_padding
            Whether to apply left padding

    Returns:
        final_embedding
            The final embeddings after merging audio embeddings with text embeddings.
        final_attention_mask
            The final attention mask after merging audio embeddings with text embeddings.
        final_labels
            The labels for the text stream

        final_input_ids
            The final input_ids after merging audio embeddings with text embeddings.
        final_audio_in_mask
            Mask for audio-in embeddings
        final_audio_in_discrete_codes_mask
            Mask for audio-in discrete tokens
        final_audio_out_mask
            Mask for audio-out embeddings

    Explanation:
        each audio has variable length embeddings, with length specified by
        - audio_features_length
        - audio_in_ids_start
        - audio_out_ids_start

        Task:
        - fill each <|AUDIO|> with audio embeddings (it can be the combination of embeddings extracted by WhisperEncoder and embeddings from audio codebooks)
        - fill each <|AUDIO_OUT|> with the audio-out embeddings

    Example:
            <|AUDIO_OUT|>: X (5 tokens), Y (3 tokens)
            <|AUDIO|>: Z (8 tokens)

            X, Y are in the same sequence (in-context voice-clone). Z is in a different sequence (audio understanding).
        if right padding
            input_ids: [
                a b c d e f X g h i j k Y l m
                o p q r Z s t u v _ _ _ _ _ _
            ]
            input_ids should be: [
                a b c d e f X X X X X g h i j k Y Y Y l m
                o p q r Z Z Z Z Z Z Z Z s t u v _ _ _ _ _
            ]
            labels should be: [
                a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                o p q r _ _ _ _ _ _ _ _ s t u v _ _ _ _ _
            ]
        elif left padding
            input_ids: [
                a b c d e f X g h i j k Y l m
                _ _ _ _ _ _ o p q r Z s t u v
            ]
            input_ids should be: [
                a b c d e f X X X X X g h i j k Y Y Y l m
                _ _ _ _ _ o p q r Z Z Z Z Z Z Z Z s t u v
            ]
            labels should be: [
                a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                _ _ _ _ _ o p q r _ _ _ _ _ _ _ _ s t u v
            ]

    """
    if label_ids is None:
        skip_labels = True
    else:
        skip_labels = False
    if audio_features_embed is not None and audio_features_embed.shape[0] == 0:
        audio_features_embed = None
    if audio_in_embed is not None and audio_in_embed.shape[0] == 0:
        audio_in_embed = None
    if audio_out_embed is not None and audio_out_embed.shape[0] == 0:
        audio_out_embed = None

    batch_size, sequence_length, embed_dim = inputs_embeds.shape

    target_device = inputs_embeds.device
    if left_padding is None:
        left_padding = torch.any(attention_mask[:, 0] == 0)

    audio_in_token_mask = input_ids == audio_in_token_idx
    audio_out_token_mask = input_ids == audio_out_token_idx
    text_token_mask = (input_ids != audio_in_token_idx) & (input_ids != audio_out_token_idx)

    # 1. Calculate the number of tokens for each placeholder (like [<|AUDIO|>, <|AUDIO_OUT|>]).
    token_placeholder_num = torch.ones_like(input_ids)

    if audio_features_embed is not None:
        num_audios, max_audio_tokens, _ = audio_features_embed.shape
        audio_in_features_mask = torch.arange(max_audio_tokens).expand(
            num_audios, max_audio_tokens
        ).to(audio_features_length.device) < audio_features_length.unsqueeze(1)
        masked_audio_in_features = audio_features_embed[audio_in_features_mask].view(-1, embed_dim)
        token_placeholder_num[audio_in_token_mask] = audio_features_length.long()

    if audio_in_embed is not None:
        audio_in_codes_length = torch.concat(
            [
                audio_in_ids_start[1:] - audio_in_ids_start[:-1],
                torch.tensor(
                    [audio_in_embed.shape[0] - audio_in_ids_start[-1]],
                    device=audio_in_ids_start.device,
                    dtype=torch.long,
                ),
            ],
            dim=0,
        )
        if audio_features_embed is not None:
            token_placeholder_num[audio_in_token_mask] += audio_in_codes_length.long()
        else:
            token_placeholder_num[audio_in_token_mask] = audio_in_codes_length.long()

    if audio_out_embed is not None:
        audio_out_codes_length = torch.concat(
            [
                audio_out_ids_start[1:] - audio_out_ids_start[:-1],
                torch.tensor(
                    [audio_out_embed.shape[0] - audio_out_ids_start[-1]],
                    device=audio_out_ids_start.device,
                    dtype=torch.long,
                ),
            ],
            dim=0,
        )
        token_placeholder_num[audio_out_token_mask] = audio_out_codes_length.long()

    new_token_positions = torch.cumsum(token_placeholder_num, -1) - 1
    max_token_num = _ceil_to_nearest(token_placeholder_num.sum(-1).max(), round_to)
    nb_audio_pad = max_token_num - 1 - new_token_positions[:, -1]

    if left_padding:
        new_token_positions += nb_audio_pad[:, None]  # offset for left padding

    # 2. Create the full embedding, already padded to the maximum position
    final_embedding = torch.zeros(
        (batch_size, max_token_num, embed_dim),
        dtype=inputs_embeds.dtype,
        device=inputs_embeds.device,
    )
    attention_mask = torch.ones_like(input_ids)
    final_attention_mask = torch.zeros(
        (batch_size, max_token_num), dtype=attention_mask.dtype, device=inputs_embeds.device
    )
    final_input_ids = torch.full(
        (batch_size, max_token_num),
        pad_token_id,
        dtype=input_ids.dtype,
        device=inputs_embeds.device,
    )
    if skip_labels:
        final_labels = None
    else:
        final_labels = torch.full(
            (batch_size, max_token_num),
            ignore_index,
            dtype=label_ids.dtype,
            device=inputs_embeds.device,
        )

    final_audio_in_mask = torch.full(
        (batch_size, max_token_num), False, dtype=torch.bool, device=inputs_embeds.device
    )
    final_audio_in_discrete_codes_mask = torch.full(
        (batch_size, max_token_num), False, dtype=torch.bool, device=inputs_embeds.device
    )
    final_audio_out_mask = torch.full(
        (batch_size, max_token_num), False, dtype=torch.bool, device=inputs_embeds.device
    )
    # 3. Get the audio-in token positions and audio-out token positions
    batch_id = (
        torch.arange(batch_size, device=target_device)
        .unsqueeze(1)
        .expand(batch_size, sequence_length)
    )
    audio_in_batch_id = batch_id[audio_in_token_mask]  # Shape (num_audio_in,)
    audio_out_batch_id = batch_id[audio_out_token_mask]  # Shape (num_audio_out,)
    audio_features_token_ends = new_token_positions[audio_in_token_mask]  # Shape (num_audio_in,)
    audio_out_embed_ends = new_token_positions[audio_out_token_mask]  # Shape (num_audio_out,)

    if audio_in_embed is not None:
        # Fill in the audio-in embeddings
        seq_indices = (
            torch.arange(max_token_num, device=target_device)
            .unsqueeze(0)
            .expand(audio_in_ids_start.shape[0], max_token_num)
        )
        audio_in_embed_token_starts = audio_features_token_ends - audio_in_codes_length + 1
        batch_indices, col_indices = torch.where(
            (seq_indices >= audio_in_embed_token_starts.unsqueeze(1))
            & (seq_indices <= audio_features_token_ends.unsqueeze(1))
        )
        batch_indices = audio_in_batch_id[batch_indices]
        final_embedding[batch_indices, col_indices] = audio_in_embed
        final_input_ids[batch_indices, col_indices] = audio_in_token_idx
        if not skip_labels:
            final_labels[batch_indices, col_indices] = ignore_index
        final_audio_in_mask[batch_indices, col_indices] = True
        final_audio_in_discrete_codes_mask[batch_indices, col_indices] = True
        audio_features_token_ends = audio_features_token_ends - audio_in_codes_length

    if audio_features_embed is not None:
        # Fill in the audio features
        seq_indices = (
            torch.arange(max_token_num, device=target_device)
            .unsqueeze(0)
            .expand(audio_features_embed.shape[0], max_token_num)
        )
        audio_features_token_starts = audio_features_token_ends - audio_features_length + 1
        batch_indices, col_indices = torch.where(
            (seq_indices >= audio_features_token_starts.unsqueeze(1))
            & (seq_indices <= audio_features_token_ends.unsqueeze(1))
        )
        batch_indices = audio_in_batch_id[batch_indices]
        final_embedding[batch_indices, col_indices] = masked_audio_in_features
        final_input_ids[batch_indices, col_indices] = audio_in_token_idx
        if not skip_labels:
            final_labels[batch_indices, col_indices] = ignore_index
        final_audio_in_mask[batch_indices, col_indices] = True

    if audio_out_embed is not None:
        # Fill in the audio-out embeddings
        seq_indices = (
            torch.arange(max_token_num, device=target_device)
            .unsqueeze(0)
            .expand(audio_out_ids_start.shape[0], max_token_num)
        )
        audio_out_embed_token_starts = audio_out_embed_ends - audio_out_codes_length + 1
        batch_indices, col_indices = torch.where(
            (seq_indices >= audio_out_embed_token_starts.unsqueeze(1))
            & (seq_indices <= audio_out_embed_ends.unsqueeze(1))
        )
        batch_indices = audio_out_batch_id[batch_indices]
        final_embedding[batch_indices, col_indices] = audio_out_embed
        final_input_ids[batch_indices, col_indices] = audio_out_token_idx
        if not skip_labels:
            final_labels[batch_indices, col_indices] = ignore_index
        final_audio_out_mask[batch_indices, col_indices] = True

    # Fill in the original text embeddings and labels
    batch_indices, non_audio_indices = torch.where(text_token_mask)
    text_to_overwrite = new_token_positions[batch_indices, non_audio_indices]
    final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[
        batch_indices, non_audio_indices
    ]
    if not skip_labels:
        final_labels[batch_indices, text_to_overwrite] = label_ids[batch_indices, non_audio_indices]
    final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_audio_indices]
    final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[
        batch_indices, non_audio_indices
    ]
    final_attention_mask = final_attention_mask | final_audio_in_mask | final_audio_out_mask

    # Trim the tensor if there are redundant padding tokens
    if left_padding:
        first_non_zero_loc = final_attention_mask.sum(0).nonzero()[0]
        first_non_zero_loc = (first_non_zero_loc // round_to) * round_to
        if first_non_zero_loc > 0:
            final_attention_mask = final_attention_mask[:, first_non_zero_loc:]
            final_embedding = final_embedding[:, first_non_zero_loc:]
            if not skip_labels:
                final_labels = final_labels[:, first_non_zero_loc:]
            final_input_ids = final_input_ids[:, first_non_zero_loc:]
            final_audio_in_mask = final_audio_in_mask[:, first_non_zero_loc:]
            final_audio_in_discrete_codes_mask = final_audio_in_discrete_codes_mask[
                :, first_non_zero_loc:
            ]
            final_audio_out_mask = final_audio_out_mask[:, first_non_zero_loc:]
    else:
        # We have done right padding, so we need to trim the mask
        last_non_zero_loc = final_attention_mask.sum(0).nonzero()[-1] + 1
        last_non_zero_loc = ((last_non_zero_loc + round_to - 1) // round_to) * round_to
        if last_non_zero_loc < max_token_num:
            final_attention_mask = final_attention_mask[:, :last_non_zero_loc]
            final_embedding = final_embedding[:, :last_non_zero_loc]
            if not skip_labels:
                final_labels = final_labels[:, :last_non_zero_loc]
            final_input_ids = final_input_ids[:, :last_non_zero_loc]
            final_audio_in_mask = final_audio_in_mask[:, :last_non_zero_loc]
            final_audio_in_discrete_codes_mask = final_audio_in_discrete_codes_mask[
                :, :last_non_zero_loc
            ]
            final_audio_out_mask = final_audio_out_mask[:, :last_non_zero_loc]

    return (
        final_embedding,
        final_attention_mask,
        final_labels,
        final_input_ids,
        final_audio_in_mask,
        final_audio_in_discrete_codes_mask,
        final_audio_out_mask,
    )


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask


class HiggsAudioDualFFNDecoderLayer(nn.Module):
    """We implement a dual-path FFN decoder layer where the audio tokens and text tokens go through separate FFN layers.

    The audio and text tokens share the text-attention layer, but will be encoded with separate feedforward layers.
    In addition, the audio tokens can be configured to go through separate attention layer.

    Following is an illustration:

     t    t    t    a   a     a    t    t    t
                        |
                        | (shared attention layer)
                        v
    h_t  h_t  h_t  h_a  h_a  h_a  h_t  h_t  h_t
                        |
                        | (separate text/audio hidden states)
                        v
    [h_t  h_t  h_t  h_t  h_t  h_t], [h_a, h_a, h_a]
             |                             |
             | (separate FFNs)             |
             v                             v
    [o_t  o_t  o_t  o_t  o_t  o_t], [o_a, o_a, o_a]
                        |
                        | (reorder)
                        v
    o_t  o_t  o_t  o_a  o_a  o_a  o_t  o_t  o_t

    This has a few advantages:
    1) We are able to use a smaller FFN, or even bypass the FFN for audio tokens. This accelerates the inference speed.
    2) The Audio-FFN introduces more trainable parameters to the model.
       This should have the same effect as the mixture-of-expert layer and we may expect better performance due to parameter scaling.
    3) We can replace the original FFN in LLMs with the dual-path FFN without changing the number of FLOPs.


    """

    def __init__(
        self,
        config: HiggsAudioConfig,
        layer_idx: int,
    ):
        super().__init__()
        text_config = config.text_config
        self.hidden_size = text_config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = LlamaAttention(config=text_config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(text_config)

        self.audio_mlp = LlamaMLP(text_config)
        self.audio_input_layernorm = LlamaRMSNorm(
            text_config.hidden_size, eps=text_config.rms_norm_eps
        )
        self.audio_post_attention_layernorm = LlamaRMSNorm(
            text_config.hidden_size, eps=text_config.rms_norm_eps
        )
        self.input_layernorm = LlamaRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            text_config.hidden_size, eps=text_config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        audio_out_mask: torch.BoolTensor | None = None,
        is_decoding_audio_token: bool | None = None,
        past_key_value: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        is_using_cuda_graph: bool | None = False,
        **kwargs,
    ):
        """Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*):
            attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
            query_sequence_length, key_sequence_length)` if default attention is used.

        audio_out_mask
            Mask for identifying the audio tokens. Size (batch_size, sequence_length)
            1 --> location contains audio_out
            0 --> location does not contain audio_out

            When use_cache is True and not in torch compile mode, the audio_out_mask contains audio_out masks for
            all tokens up to the current token.  That means, it has size (batch_size, sequence_length) while
            hidden_states will have size (batch_size, 1). In the torch compile mode, the audio_out_mask will have
            size (batch_size, 1).
        is_decoding_audio_token
            Used in the torch compile mode to determine if the current token is an audio token or not.
        past_key_value (`Cache`, *optional*): cached past key and value projection states. We fetch the corresponding cached key/value via the layer_idx.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence
        position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
            Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
            with `head_dim` being the embedding dimension of each attention head.
        is_using_cuda_graph (`bool`, *optional*):
            Indicates whether the model is running by cuda graph.
        kwargs (`dict`, *optional*):
            Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
            into the model
        """
        residual = hidden_states
        target_length = hidden_states.shape[1]
        use_static_cache = isinstance(past_key_value, StaticCache)
        decode_stage = hidden_states.shape[1] == 1
        if is_using_cuda_graph:
            assert decode_stage and use_static_cache, (
                "The CUDA graph mode should only be used in the decoding stage with static cache."
            )
        has_audio_out = audio_out_mask is not None and audio_out_mask.shape[0] > 0

        if has_audio_out:
            # Apply separate layernorm layers for audio tokens and text tokens
            if use_cache:
                hidden_states = torch.where(
                    audio_out_mask[:, -target_length:].unsqueeze(-1),
                    self.audio_input_layernorm(hidden_states),
                    self.input_layernorm(hidden_states),
                )
            else:
                hidden_states = torch.where(
                    audio_out_mask.unsqueeze(-1),
                    self.audio_input_layernorm(hidden_states),
                    self.input_layernorm(hidden_states),
                )
        else:
            hidden_states = self.input_layernorm(hidden_states)

        # Text Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Apply Dual-path FFN
        residual = hidden_states

        if has_audio_out:
            if use_cache:
                audio_out_mask = audio_out_mask[:, -target_length:]
            else:
                audio_out_mask = audio_out_mask

            # Make whole graph in decode stage
            if decode_stage and is_using_cuda_graph:
                assert is_decoding_audio_token is not None, (
                    "is_decoding_audio_token should be present in the decoding stage."
                )
                if is_decoding_audio_token:
                    hidden_states = self.audio_post_attention_layernorm(hidden_states)
                    hidden_states = self.audio_mlp(hidden_states)
                else:
                    hidden_states = self.post_attention_layernorm(hidden_states)
                    hidden_states = self.mlp(hidden_states)
                residual = residual + hidden_states
            else:
                text_hidden_states = self.post_attention_layernorm(hidden_states[~audio_out_mask])
                audio_hidden_states = self.audio_post_attention_layernorm(
                    hidden_states[audio_out_mask]
                )

                text_hidden_states = self.mlp(text_hidden_states)
                residual[~audio_out_mask] += text_hidden_states

                audio_hidden_states = self.audio_mlp(audio_hidden_states)
                residual[audio_out_mask] += audio_hidden_states

            hidden_states = residual
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if use_cache:
            outputs += (past_key_value,)

        return outputs


@dataclass
class HiggsAudioModelOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    llm_loss: torch.FloatTensor | None = None
    audio_loss: torch.FloatTensor | None = None
    codebook_losses: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    expanded_input_ids: torch.LongTensor | None = None
    expanded_labels: torch.LongTensor | None = None
    audio_in_mask: torch.BoolTensor | None = None
    audio_in_discrete_codes_mask: torch.BoolTensor | None = None
    audio_out_mask: torch.BoolTensor | None = None
    attention_mask: torch.BoolTensor | None = None
    audio_logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    audio_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class HiggsAudioGenerationOutput(ModelOutput):
    """Outputs of HiggsAudio generation models, when using non-beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        audio_sequences (`tuple(torch.LongTensor)` *optional*):
            The generated discrete audio codes. These codes can be used to fill-in related locations of <|AUDIO_OUT|> at input sequences.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token).
            If the generated token is a text token, the tensor will have shape `(batch_size, config.vocab_size)`.
            If the generated token is an audio token, the tensor will have shape `(config.audio_num_codebooks, self.audio_codebook_size)`
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`):
            Unprocessed prediction scores of the language modeling head or the audio head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token).
            If the generated token is a text token, the tensor will have shape `(batch_size, config.vocab_size)`.
            If the generated token is an audio token, the tensor will have shape `(config.audio_num_codebooks, self.audio_codebook_size)`
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True`):
            Returns the model cache, used to speed up decoding. Different models have a different cache format, check
            the model's documentation. Usually, a [`~cache_utils.Cache`] instance.
    """

    sequences: torch.LongTensor = None
    audio_sequences: list[torch.LongTensor] | None = None
    scores: tuple[torch.FloatTensor] | None = None
    logits: tuple[torch.FloatTensor] | None = None
    attentions: tuple[tuple[torch.FloatTensor]] | None = None
    hidden_states: tuple[tuple[torch.FloatTensor]] | None = None
    past_key_values: tuple[tuple[tuple[torch.FloatTensor]]] | None = None


class HiggsAudioModel(HiggsAudioPreTrainedModel, GenerationMixin):
    """Higgs-Audio is an end-to-end multimodal model with the capability to understand and generate text / audio.

    Consider the following example for mixed text/audio understanding / generation:

    - input_tokens: <text_token1><|audio_bos|>[AUDIO]<|audio_eos|><text_token2><|audio_bos|>[AUDIO]<|audio_eos|><text_token4>
    - input_tokens: <text_token1><|audio_bos|>[AUDIO]<|audio_eos|><text_token2><|audio_out_bos|>[AUDIO_OUT]<|audio_eos|><text_token4>

    We will fill [AUDIO] with the audio features extracted by Whisper and fill [AUDIO_OUT] with the audio tokens.

    Consider the following example for mixed text/audio generation:

    text: <|audio_out_bos|>    MASK           MASK           MASK          MASK               MASK         <|audio_eos|> [text_token1]
    audio:     MASK    <|audio_stream_bos|> [audio_token1] [audio_token2] [audio_token3] <|audio_stream_eos|>   MASK           MASK
    token_type: 0               1              1              1             1                  1                 0              0

    """

    _supports_cache_class = True
    _supports_static_cache = True

    def __init__(self, config: HiggsAudioConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.audio_in_token_idx = config.audio_in_token_idx
        self.audio_out_token_idx = config.audio_out_token_idx
        self.audio_out_bos_token_id = (
            config.audio_out_bos_token_id if "audio_out_bos_token_id" in config else None
        )
        self.audio_eos_token_id = (
            config.audio_eos_token_id if "audio_eos_token_id" in config else None
        )
        self.vocab_size = config.text_config.vocab_size
        self.audio_num_codebooks = config.audio_num_codebooks
        self.use_delay_pattern = config.use_delay_pattern

        self.embed_tokens = nn.Embedding(
            self.vocab_size, config.text_config.hidden_size, self.padding_idx
        )

        if config.audio_adapter_type == "dual_ffn":
            layer_idx = 0
            layers = []
            for j in range(config.text_config.num_hidden_layers):
                if j in config.audio_dual_ffn_layers:
                    layers.append(HiggsAudioDualFFNDecoderLayer(config, layer_idx))
                    layer_idx += 2 if self.use_audio_out_self_attention else 1
                else:
                    layers.append(LlamaDecoderLayer(config.text_config, layer_idx))
                    layer_idx += 1
            self.layers = nn.ModuleList(layers)

        self.num_activation_checkpointing_layers = len(self.layers)

        self.decode_graph_runners = defaultdict(dict[bool, CUDAGraphRunner])
        self.norm = LlamaRMSNorm(
            config.text_config.hidden_size, eps=config.text_config.rms_norm_eps
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=config.text_config)
        self.audio_codebook_size = (
            config.audio_codebook_size + 2
        )  # We add 1 for the audio_stream_bos token and 1 for the audio_stream_eos token

        self.audio_codebook_embeddings = nn.Embedding(
            config.audio_num_codebooks * self.audio_codebook_size, config.text_config.hidden_size
        )

        self.audio_codebook_weights = (
            torch.ones(config.audio_num_codebooks) / config.audio_num_codebooks
        )  # default to equal weights
        self.post_init()

    def set_delay_pattern(self):
        self.config.use_delay_pattern = True
        self.use_delay_pattern = True

    def set_audio_special_tokens(self, tokenizer: AutoTokenizer):
        self.audio_out_bos_token_id = tokenizer.convert_tokens_to_ids("<|audio_out_bos|>")
        self.audio_eos_token_id = tokenizer.convert_tokens_to_ids("<|audio_eos|>")

    def _embed_audio_ids(self, audio_ids):
        """Embed the audio ids

        Args:
            audio_ids: torch.LongTensor of shape (num_codebooks, audio_in_total_length)

        Returns:
            audio_embed: torch.LongTensor of shape (audio_in_total_length, hidden_size)
        """
        codebook_shift = (
            torch.arange(self.config.audio_num_codebooks, device=audio_ids.device)
            * self.audio_codebook_size
        )
        audio_embed = self.audio_codebook_embeddings(audio_ids + codebook_shift.unsqueeze(-1))
        if self.config.audio_embed_avg:
            audio_embed = torch.mean(audio_embed, dim=0)
        else:
            audio_embed = torch.sum(audio_embed, dim=0)
        return audio_embed

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.max_cache_len
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    def _forward_core(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        audio_discrete_codes_mask: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Union[Cache, list[torch.FloatTensor]] | None,
        use_cache: bool,
        is_decoding_audio_token: bool | None = None,
        is_using_cuda_graph: bool | None = False,
    ):
        # create position embeddings to be shared across the decoder layers
        # When past_key_values is passed in, we need to offset the position ids when calculating the position embeddings.
        # Therefore, cache_position is used.

        for decoder_layer in self.layers:
            if isinstance(decoder_layer, HiggsAudioDualFFNDecoderLayer):
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    audio_out_mask=audio_discrete_codes_mask,
                    is_decoding_audio_token=is_decoding_audio_token,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    is_using_cuda_graph=is_using_cuda_graph,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

        return hidden_states

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        attention_mask: torch.BoolTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        audio_features: torch.FloatTensor | None = None,
        audio_feature_attention_mask: torch.BoolTensor | None = None,
        audio_in_ids: torch.LongTensor | None = None,
        audio_in_ids_start: torch.LongTensor | None = None,
        audio_out_ids: torch.LongTensor | None = None,
        audio_out_ids_start: torch.LongTensor | None = None,
        audio_out_ids_start_group_loc: torch.LongTensor | None = None,
        label_ids: torch.LongTensor | None = None,
        label_audio_ids: torch.LongTensor | None = None,
        past_key_values: Union[Cache, list[torch.FloatTensor]] | None = None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        cache_audio_discrete_codes_mask: torch.LongTensor | None = None,
        past_key_values_buckets: OrderedDict[int, Cache] | None = None,
        reward: torch.FloatTensor | None = None,
    ):
        """Forward pass for the Higgs-Audio model.

        Args:
            input_ids (:obj:`torch.LongTensor`):
                The input ids of the prompt. It will have shape (bsz, seq_len).
                When use_cache is enabled, the input_ids will have
                shape (bsz, 1) for incremental decode or None
            inputs_embeds:
                Input embeddings. This flag won't be used.
            attention_mask (:obj:`torch.LongTensor`):
                The attention mask of the prompt. It will have shape (bsz, seq_len).
            audio_features (:obj:`torch.FloatTensor`):
                The audio features extracted by Whisper. It will have shape (num_audio_in, feature_dim, max_mel_seq_len).
            audio_feature_attention_mask (:obj:`torch.LongTensor`):
                The attention mask of the audio features. It will have shape (num_audio_in, max_mel_seq_len).
            audio_in_ids (:obj:`torch.LongTensor`):
                The discretized audio tokens. It will have shape (num_codebooks, audio_in_total_length).
            audio_in_ids_start (:obj:`torch.LongTensor`):
                The start indices for each audio in audio_in_ids. It will have shape (num_audio_in,)
            audio_out_ids (:obj:`torch.LongTensor`):
                The discretized audio tokens. It will have shape (num_codebooks, audio_out_total_length).
            audio_out_ids_start (:obj:`torch.LongTensor`):
                The start indices for each audio in audio_out_ids. It will have shape (num_audio_out,)
            audio_out_ids_start_group_loc (:obj:`torch.LongTensor`):
                The sample indices in a batch that map to each element in the audio_out_ids_start. It will have shape (num_audio_out,)
            label_text_ids (:obj:`torch.LongTensor`):
                The labels of the prompt. It will have shape (bsz, seq_len).
            label_audio_ids (:obj:`torch.LongTensor`):
                The labels of the audio tokens. It will have the same shape as audio_out_ids, i.e., (num_codebooks, audio_out_total_length)
            past_key_values (:obj:`Tuple`):
                Tuple of past key values.
            use_cache (:obj:`bool`):
                Whether to use cache.
            return_dict (:obj:`bool`):
                Whether to return a dictionary.
            cache_position (:obj:`torch.LongTensor`):
                The position of the cache.
            cache_audio_discrete_codes_mask (:obj:`torch.LongTensor`):
                The cached audio discrete codes mask. It will only be used when use_cache is turned on.
            past_key_values_buckets (:obj:`OrderedDict`):
                The buckets of past key values.
        """
        target_device = input_ids.device

        # not used
        del inputs_embeds

        if audio_features is not None:
            audio_features = audio_features.to(target_device)
            audio_feature_attention_mask = audio_feature_attention_mask.to(target_device)

        # 1. Extract the input embeddings
        inputs_embeds = self.embed_tokens(input_ids)

        audio_features_embed = audio_features_length = None

        if self.config.encode_audio_in_tokens:
            audio_in_ids = audio_in_ids.to(target_device)
            audio_in_embed = self._embed_audio_ids(audio_in_ids)
        else:
            audio_in_embed = None

        if audio_out_ids is not None and audio_out_ids.shape[-1] > 0:
            audio_out_ids = audio_out_ids.to(target_device)
        else:
            audio_out_ids = torch.zeros(
                (self.audio_num_codebooks, 0), device=target_device, dtype=torch.long
            )
        audio_out_embed = self._embed_audio_ids(audio_out_ids)

        # 3. Merge text, audio-in embeddings, and audio-out embeddings

        # use_cache is turned on during inference time, we should set round_to to 1 to avoid extra padding in the end.
        round_to = 1 if use_cache else 8
        left_padding = True if use_cache or input_ids.shape[0] == 1 else False
        (
            inputs_embeds,
            attention_mask,
            labels,
            input_ids,
            audio_in_mask,
            audio_in_discrete_codes_mask,
            audio_out_mask,
        ) = merge_input_ids_with_audio_features(
            audio_features_embed,
            audio_features_length,
            audio_in_embed,
            audio_in_ids_start,
            audio_out_embed,
            audio_out_ids_start,
            self.audio_in_token_idx,
            self.audio_out_token_idx,
            inputs_embeds,
            input_ids,
            attention_mask,
            label_ids,
            pad_token_id=self.padding_idx,
            round_to=round_to,
            left_padding=left_padding,
        )

        # re-check if we use the correct kv cache bucket after
        # the input_embeds has been merged with audio features
        if (
            past_key_values_buckets is not None
            and inputs_embeds.shape[1] > past_key_values.get_max_cache_shape()
        ):
            past_key_values, self.current_past_key_values_bucket = self._prepare_kv_cache(
                inputs_embeds.shape[1], None, past_key_values_buckets
            )

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
            if (
                isinstance(past_key_values, StaticCache)
                and past_seen_tokens >= past_key_values.get_max_cache_shape()
            ):
                raise ValueError(
                    f"The current sequence length ({past_seen_tokens}) exceeds "
                    f"the maximum cache shape. "
                    f"Please consider increasing the cache size."
                )

        # Use torch compile
        use_static_cache = isinstance(past_key_values, StaticCache)

        # Apply the LLM component
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values
        )

        hidden_states = inputs_embeds

        audio_discrete_codes_mask = audio_in_discrete_codes_mask | audio_out_mask
        if cache_audio_discrete_codes_mask is not None and use_cache:
            audio_discrete_codes_mask = torch.concat(
                [cache_audio_discrete_codes_mask, audio_discrete_codes_mask], dim=1
            )

        # Generate the audio attention mask outside the layer to avoid recompilation
        if use_static_cache:
            # Set the audio out mask to the last token
            if hidden_states.shape[1] == 1:
                audio_discrete_codes_mask = audio_discrete_codes_mask[:, -1:]
                audio_discrete_codes_mask = audio_discrete_codes_mask.reshape((-1, 1)).contiguous()
                is_decoding_audio_token = audio_discrete_codes_mask.item()
            else:
                is_decoding_audio_token = False

        # Use the captured cuda graph runner for decoding
        # if it exists, otherwise use the normal forward pass
        if (
            past_key_values is not None
            and past_key_values.get_max_cache_shape() in self.decode_graph_runners
            and (input_ids.shape[-1] == 1)
        ):
            _forward_core = self.decode_graph_runners[past_key_values.get_max_cache_shape()][
                is_decoding_audio_token
            ]
            is_using_cuda_graph = True
        else:
            _forward_core = self._forward_core
            is_using_cuda_graph = False

        hidden_states, all_hidden_states, all_self_attns = _forward_core(
            hidden_states=hidden_states,
            causal_mask=causal_mask,
            audio_discrete_codes_mask=audio_discrete_codes_mask,
            is_decoding_audio_token=is_decoding_audio_token if use_static_cache else None,
            cache_position=cache_position,
            past_key_values=past_key_values,
            use_cache=use_cache,
            is_using_cuda_graph=is_using_cuda_graph,
        )
        hidden_states = self.norm(hidden_states)

        # Apply the audio decoder projector
        (
            logits,
            audio_logits,
            decoder_all_self_attns,
            decoder_all_hidden_states,
            audio_hidden_states,
            _,
        ) = self.audio_decoder_proj(
            hidden_states,
            audio_out_mask,
            label_audio_ids=label_audio_ids,
            attention_mask=causal_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        if audio_logits is not None:
            audio_logits = audio_logits.view(
                audio_logits.shape[0], self.audio_num_codebooks, self.audio_codebook_size
            ).float()

        next_cache = past_key_values if use_cache else None

        ret = HiggsAudioModelOutputWithPast(
            logits=logits,
            audio_logits=audio_logits,
            expanded_input_ids=input_ids,
            expanded_labels=labels,
            audio_in_mask=audio_in_mask,
            audio_in_discrete_codes_mask=audio_in_discrete_codes_mask,
            audio_out_mask=audio_out_mask,
            attention_mask=attention_mask,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            audio_hidden_states=audio_hidden_states,
            attentions=all_self_attns,
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if not return_dict:
            outputs = ret.to_tuple()
            return outputs

        return ret

    # Overwrite GenerationMixin._update_model_kwargs_for_generation
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
        extend_attention_mask: bool = True,
    ) -> dict[str, Any]:
        """Update the model kwargs for each step."""
        model_kwargs["past_key_values"] = outputs.past_key_values

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            if extend_attention_mask:
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        if "cache_audio_discrete_codes_mask" in model_kwargs:
            if model_kwargs["cache_audio_discrete_codes_mask"] is None:
                model_kwargs["cache_audio_discrete_codes_mask"] = (
                    outputs.audio_in_discrete_codes_mask | outputs.audio_out_mask
                )
            else:
                model_kwargs["cache_audio_discrete_codes_mask"] = torch.concat(
                    [
                        model_kwargs["cache_audio_discrete_codes_mask"],
                        outputs.audio_in_discrete_codes_mask | outputs.audio_out_mask,
                    ],
                    1,
                )

        return model_kwargs

    def _copy_kv_cache(self, from_cache: Cache, to_cache: Cache):
        num_layers = self.config.text_config.num_hidden_layers
        if self.config.audio_dual_ffn_layers is not None:
            num_layers += len(self.config.audio_dual_ffn_layers)
        """ Copy the key-value pairs from one cache to another. """
        for layer_idx in range(num_layers):
            from_cache_size = from_cache.get_max_cache_shape()
            assert to_cache.get_max_cache_shape() >= from_cache_size, (
                f"The target cache size {to_cache.get_max_cache_shape()} is smaller than the source cache size {from_cache_size}."
            )
            to_cache.key_cache[layer_idx][:, :, :from_cache_size, :] = from_cache.key_cache[
                layer_idx
            ]
            to_cache.value_cache[layer_idx][:, :, :from_cache_size, :] = from_cache.value_cache[
                layer_idx
            ]

    def _prepare_kv_cache(
        self,
        current_sequence_length: int,
        current_past_key_values_bucket: int | None,
        past_key_values_buckets: OrderedDict[int, Cache],
    ) -> tuple[Cache | None, int | None]:
        """Prepare the KV cache for the current sequence length."""
        for cache_length in past_key_values_buckets.keys():
            if cache_length >= current_sequence_length:
                # Promote to the next KV cache bucket, copy the current KV cache bucket
                # to the new one.
                if (
                    current_past_key_values_bucket is not None
                    and cache_length != current_past_key_values_bucket
                ):
                    self._copy_kv_cache(
                        past_key_values_buckets[current_past_key_values_bucket],
                        past_key_values_buckets[cache_length],
                    )

                return past_key_values_buckets[cache_length], cache_length

        raise ValueError(
            f"The current sequence length {current_sequence_length} is larger than "
            f"all past key values buckets {past_key_values_buckets.keys()}."
        )

    def _sample_audio_tokens(
        self,
        hidden_states: torch.Tensor,
        audio_logits: torch.Tensor,
        audio_out_ids: torch.Tensor,
        do_sample: bool,
        logits_processor: LogitsProcessorList,
        device: torch.device,
        torch_generator: torch.Generator | None,
        generation_config: GenerationConfig,
        num_delay: int,
        num_remaining_delays: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int | None]:
        """Sample audio tokens and its corresponding text tokens from the logits"""
        # parameters related to repetition aware sampling
        ras_win_len = getattr(generation_config, "ras_win_len", None)
        ras_win_max_num_repeat = getattr(generation_config, "ras_win_max_num_repeat", 2)
        audio_eos_token_id = getattr(generation_config, "audio_eos_token_id", None)
        # In the audio generation mode, we sample from audio_logits and keep updating audio_out_ids.
        next_audio_token_logits = audio_logits.clone()[-1, :, :].float().to(device)
        # TopP, TopK logits processor supports empty input_ids
        next_audio_token_scores = logits_processor(None, next_audio_token_logits)

        # token selection
        if do_sample:
            # next_audio_token_scores has been applied top_p, top_k, and temperature.
            probs = nn.functional.softmax(next_audio_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_audio_tokens = torch.multinomial(
                probs, num_samples=1, generator=torch_generator
            ).squeeze(1)
        else:
            next_audio_tokens = torch.argmax(next_audio_token_scores, dim=-1)

        # next_tokens: (num_codebooks, )
        if ras_win_len is not None:
            # check if there are repetitions over a window of tokens.
            rep_num = (audio_out_ids[:, -ras_win_len:] == next_audio_tokens.unsqueeze(1)).sum(dim=1)

            # if we saw repeated tokens in the most recent window of tokens, resample without temperature.
            row_indices = torch.nonzero(rep_num >= ras_win_max_num_repeat).squeeze(1)
            resampled_next_tokens = (
                next_audio_token_logits[row_indices]
                .softmax(dim=-1)
                .multinomial(1, replacement=True, generator=torch_generator)
                .squeeze(1)
            )
            next_audio_tokens[row_indices] = resampled_next_tokens

        # Force the next text tokens to be <|AUDIO_OUT|> in audio generation mode
        next_tokens = torch.full(
            (audio_logits.shape[0],),
            self.config.audio_out_token_idx,
            dtype=torch.long,
            device=device,
        )

        # Handle delay_pattern
        if self.use_delay_pattern:
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

        return (
            next_tokens,
            next_audio_tokens,
            next_audio_token_logits,
            next_audio_token_scores,
            num_delay,
            num_remaining_delays,
        )

    # Built on top of GenerationMixin._sample.
    # We revise the implementation to support generating both audio / text.
    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        past_key_values_buckets: OrderedDict[int, Cache] | None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""Generates sequences of token ids for joint text/audio models using **multinomial sampling**.

        This function may also be revised to support generating samples from HiggsAudio-like end-to-end text/audio models built on top of LLMs.
        If the input_ids ends with <|audio_out_bos|>, we will switch to the audio-generation mode.

        ```
        ...<|start_header_id|>assistant<|end_header_id|>\n\n<|audio_out_bos|>
        ```

        Otherwise, we will keep generating the text tokens.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        assert input_ids.shape[0] == 1, "Only support batch_size=1 in _sample()"
        audio_out_bos_token_id = getattr(generation_config, "audio_out_bos_token_id", None)

        # torch generator for sampling
        seed = getattr(generation_config, "seed", None)
        if seed is not None:
            torch_generator = torch.Generator(device=input_ids.device).manual_seed(seed)
        else:
            torch_generator = None

        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(
            hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
        )
        do_sample = generation_config.do_sample
        # Used to track which past_key_va
        self.current_past_key_values_bucket = None

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        if generation_config.use_cache:
            model_kwargs["cache_audio_discrete_codes_mask"] = None

        init_model_input = True
        num_delay = 0
        num_remaining_delays = None
        audio_sequences = []
        # A tensor to keep track of all the audio placeholder tokens.
        input_ids_full = input_ids.clone()

        # Initialize the audio variables based on the input prompt.
        if input_ids[0][-1] == self.config.audio_out_token_idx:
            audio_sequences = [
                model_kwargs["audio_out_ids"][:, model_kwargs["audio_out_ids_start"][-1] :]
            ]
            if self.use_delay_pattern:
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

        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device
        ):
            if init_model_input or not generation_config.use_cache:
                model_inputs = {"input_ids": input_ids, **model_kwargs}
            else:
                model_inputs = {"input_ids": input_ids[:, -1:], **model_kwargs}

                if generation_config.use_cache:
                    model_inputs["audio_out_ids"] = model_kwargs["audio_out_ids"][:, -1:]
                    model_inputs["audio_out_ids_start"] = torch.tensor(
                        [0], dtype=torch.long, device=input_ids.device
                    )
                if generation_config.use_cache:
                    if (
                        "audio_features" in model_inputs
                        and model_inputs["audio_features"] is not None
                    ):
                        model_inputs["audio_features"] = model_inputs["audio_features"][:0, ...]
                        model_inputs["audio_feature_attention_mask"] = model_inputs[
                            "audio_feature_attention_mask"
                        ][:0, ...]

                    if "audio_in_ids" in model_inputs and model_inputs["audio_in_ids"] is not None:
                        model_inputs["audio_in_ids"] = None
                        model_inputs["audio_in_ids_start"] = None

            if past_key_values_buckets is not None:
                past_key_values, self.current_past_key_values_bucket = self._prepare_kv_cache(
                    cur_len, self.current_past_key_values_bucket, past_key_values_buckets
                )
                if past_key_values is not None:
                    model_inputs.update({"past_key_values": past_key_values})
                model_inputs["past_key_values_buckets"] = past_key_values_buckets

            # forward pass to get next token
            outputs = self(**model_inputs, return_dict=True)

            # Update the actual sequence length after the first forward pass
            if init_model_input and past_key_values_buckets is not None:
                cur_len = (
                    past_key_values_buckets[self.current_past_key_values_bucket]
                    .get_seq_length()
                    .item()
                )

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
                extend_attention_mask=True,
            )

            # After the first forward pass, we can set init_model_input to False.
            init_model_input = False

            if synced_gpus and this_peer_finished:
                continue

            (
                next_tokens,
                next_audio_tokens,
                next_audio_token_logits,
                next_audio_token_scores,
                num_delay,
                num_remaining_delays,
            ) = self._sample_audio_tokens(
                hidden_states=outputs.audio_hidden_states,
                audio_logits=outputs.audio_logits,
                audio_out_ids=model_kwargs["audio_out_ids"],
                do_sample=do_sample,
                logits_processor=logits_processor,
                device=input_ids.device,
                torch_generator=torch_generator,
                generation_config=generation_config,
                num_delay=num_delay,
                num_remaining_delays=num_remaining_delays,
            )

            # update generated ids, model inputs, and length for next step
            model_kwargs["audio_out_ids"] = torch.cat(
                [model_kwargs["audio_out_ids"], next_audio_tokens[:, None]], dim=-1
            )
            audio_sequences[-1] = torch.cat(
                [audio_sequences[-1], next_audio_tokens[:, None]], dim=-1
            )

            if streamer is not None:
                streamer.put(next_audio_tokens.cpu())

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_audio_token_scores,)
                if output_logits:
                    raw_logits += (next_audio_token_logits,)
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids_full, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            return HiggsAudioGenerationOutput(
                sequences=input_ids,
                audio_sequences=audio_sequences,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return input_ids, audio_sequences

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor | None = None,
        audio_features: torch.FloatTensor | None = None,
        audio_feature_attention_mask: torch.BoolTensor | None = None,
        audio_in_ids: torch.LongTensor | None = None,
        audio_in_ids_start: torch.LongTensor | None = None,
        audio_out_ids: torch.LongTensor | None = None,
        audio_out_ids_start: torch.LongTensor | None = None,
        past_key_values: Union[Cache, list[torch.FloatTensor]] | None = None,
        audio_out_bos_token_id: int = None,
        audio_eos_token_id: int = None,
        past_key_values_buckets: OrderedDict[int, Cache] | None = None,
        seed: int | None = None,
        **kwargs,
    ):
        """The generate function in huggingface generally follows these steps:

        for sample_step in 1, 2, 3, 4, 5, ...
            ...

        """
        # Right now, it's a very simplified version of generate, we should revisit this after our model architecture stabilizes.
        assert input_ids.shape[0] == 1, (
            "Currently HiggsAudioModel.generate() only supports batch_size=1. See the implementation of "
        )
        generation_config, kwargs = self._prepare_generation_config(
            kwargs.pop("generation_config", None), **kwargs
        )
        if audio_out_bos_token_id is not None:
            generation_config.audio_out_bos_token_id = audio_out_bos_token_id
        else:
            try:
                generation_config.audio_out_bos_token_id = self.audio_out_bos_token_id
            except:
                generation_config.audio_out_bos_token_id = None

        if audio_eos_token_id is not None:
            generation_config.audio_eos_token_id = audio_eos_token_id
        else:
            try:
                generation_config.audio_eos_token_id = self.audio_eos_token_id
            except:
                generation_config.audio_eos_token_id = None

        has_default_max_length = (
            kwargs.get("max_length") is None and generation_config.max_length is not None
        )
        has_default_min_length = (
            kwargs.get("min_length") is None and generation_config.min_length is not None
        )

        generation_config.ras_win_len = kwargs.pop("ras_win_len", None)
        generation_config.ras_win_max_num_repeat = kwargs.pop("ras_win_max_num_repeat", 2)
        # Set generation seed if determinstic generation is required
        if seed is not None:
            generation_config.seed = seed

        # Store tokenizer in generation config if it is in kwargs without popping it
        if "tokenizer" in kwargs:
            generation_config.tokenizer_length = len(kwargs["tokenizer"])

        # input_ids: [bsz, seq_len]
        # The merging of audio features happens inside the forward path. The input_ids does not need to change.
        # TODO: prepare the final input embeddings to improve generation performance
        input_ids_length = input_ids.shape[-1]
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=None,
            inputs_tensor=None,
            input_ids_length=input_ids_length,
        )
        assert generation_config.num_beams == 1, (
            "Currently, we only support beam search with num_beams=1"
        )
        return_dict_in_generate = generation_config.return_dict_in_generate
        output_scores = generation_config.output_scores

        # When attn_implement is spda or flash-attention, it will create causal mask automatically.
        attention_mask = kwargs.pop("attention_mask", None)
        return super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_features=audio_features,
            audio_feature_attention_mask=audio_feature_attention_mask,
            audio_in_ids=audio_in_ids,
            audio_in_ids_start=audio_in_ids_start,
            audio_out_ids=audio_out_ids,
            audio_out_ids_start=audio_out_ids_start,
            past_key_values=past_key_values,
            generation_config=generation_config,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            past_key_values_buckets=past_key_values_buckets,
            **kwargs,
        )
