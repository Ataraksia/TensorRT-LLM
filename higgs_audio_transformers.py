"""Higgs-Audio is an end-to-end multimodal model with the capability to understand and generate text / audio."""

import contextlib
import functools
import glob
import math
import os
from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torchaudio
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import (
    GenerationConfig,
    GenerationMixin,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from transformers.generation.utils import GenerateNonBeamOutput
from transformers.integrations import is_deepspeed_available
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer
from transformers.utils import ModelOutput, logging
from vector_quantize_pytorch import ResidualFSQ

logger = logging.get_logger(__name__)

AUDIO_IN_TOKEN = "<|AUDIO|>"
AUDIO_OUT_TOKEN = "<|AUDIO_OUT|>"
EOS_TOKEN = "<|end_of_text|>"


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


class HiggsAudioPreTrainedModel(PreTrainedModel):
    config_class = HiggsAudioConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        std = (
            self.config.init_std
            if hasattr(self.config, "init_std")
            else self.config.audio_encoder_config.init_std
        )

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Conv1d1x1(nn.Conv1d):
    """1x1 Conv1d."""

    def __init__(self, in_channels, out_channels, bias=True):
        super(Conv1d1x1, self).__init__(in_channels, out_channels, kernel_size=1, bias=bias)


class Conv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = -1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if padding < 0:
            padding = (kernel_size - 1) // 2 * dilation
        self.dilation = dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        """Args:
            x (Tensor): Float tensor variable with the shape  (B, C, T).

        Returns:
            Tensor: Float tensor variable with the shape (B, C, T).
        """
        x = self.conv(x)
        return x


class ResidualUnit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        dilation=1,
        bias=False,
        nonlinear_activation="ELU",
        nonlinear_activation_params={},
    ):
        super().__init__()
        self.activation = getattr(nn, nonlinear_activation)(**nonlinear_activation_params)
        self.conv1 = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            bias=bias,
        )
        self.conv2 = Conv1d1x1(out_channels, out_channels, bias)

    def forward(self, x):
        y = self.conv1(self.activation(x))
        y = self.conv2(self.activation(y))
        return x + y


class ConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding=-1,
        output_padding=-1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        if padding < 0:
            padding = (stride + 1) // 2
        if output_padding < 0:
            output_padding = 1 if stride % 2 else 0
        self.deconv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        """Args:
            x (Tensor): Float tensor variable with the shape  (B, C, T).

        Returns:
            Tensor: Float tensor variable with the shape (B, C', T').
        """
        x = self.deconv(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dilations=(1, 1),
        unit_kernel_size=3,
        bias=True,
    ):
        super().__init__()
        self.res_units = torch.nn.ModuleList()
        for dilation in dilations:
            self.res_units += [
                ResidualUnit(
                    in_channels, in_channels, kernel_size=unit_kernel_size, dilation=dilation
                )
            ]
        self.num_res = len(self.res_units)

        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3
            if stride == 1
            else (2 * stride),  # special case: stride=1, do not use kernel=2
            stride=stride,
            bias=bias,
        )

    def forward(self, x):
        for idx in range(self.num_res):
            x = self.res_units[idx](x)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        encode_channels: int,
        channel_ratios=(1, 1),
        strides=(1, 1),
        kernel_size=3,
        bias=True,
        block_dilations=(1, 1),
        unit_kernel_size=3,
    ):
        super().__init__()
        assert len(channel_ratios) == len(strides)

        self.conv = Conv1d(
            in_channels=input_channels,
            out_channels=encode_channels,
            kernel_size=kernel_size,
            stride=1,
            bias=False,
        )
        self.conv_blocks = torch.nn.ModuleList()
        in_channels = encode_channels
        for idx, stride in enumerate(strides):
            out_channels = int(encode_channels * channel_ratios[idx])  # could be float
            self.conv_blocks += [
                EncoderBlock(
                    in_channels,
                    out_channels,
                    stride,
                    dilations=block_dilations,
                    unit_kernel_size=unit_kernel_size,
                    bias=bias,
                )
            ]
            in_channels = out_channels
        self.num_blocks = len(self.conv_blocks)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)
        for i in range(self.num_blocks):
            x = self.conv_blocks[i](x)
        return x


class DecoderBlock(nn.Module):
    """Decoder block (no up-sampling)"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dilations=(1, 1),
        unit_kernel_size=3,
        bias=True,
    ):
        super().__init__()

        if stride == 1:
            self.conv = Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,  # fix kernel=3 when stride=1 for unchanged shape
                stride=stride,
                bias=bias,
            )
        else:
            self.conv = ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(2 * stride),
                stride=stride,
                bias=bias,
            )

        self.res_units = torch.nn.ModuleList()
        for idx, dilation in enumerate(dilations):
            self.res_units += [
                ResidualUnit(
                    out_channels, out_channels, kernel_size=unit_kernel_size, dilation=dilation
                )
            ]
        self.num_res = len(self.res_units)

    def forward(self, x):
        x = self.conv(x)
        for idx in range(self.num_res):
            x = self.res_units[idx](x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        code_dim: int,
        output_channels: int,
        decode_channels: int,
        channel_ratios=(1, 1),
        strides=(1, 1),
        kernel_size=3,
        bias=True,
        block_dilations=(1, 1),
        unit_kernel_size=3,
    ):
        super().__init__()
        assert len(channel_ratios) == len(strides)

        self.conv1 = Conv1d(
            in_channels=code_dim,
            out_channels=int(decode_channels * channel_ratios[0]),
            kernel_size=kernel_size,
            stride=1,
            bias=False,
        )

        self.conv_blocks = torch.nn.ModuleList()
        for idx, stride in enumerate(strides):
            in_channels = int(decode_channels * channel_ratios[idx])
            if idx < (len(channel_ratios) - 1):
                out_channels = int(decode_channels * channel_ratios[idx + 1])
            else:
                out_channels = decode_channels
            self.conv_blocks += [
                DecoderBlock(
                    in_channels,
                    out_channels,
                    stride,
                    dilations=block_dilations,
                    unit_kernel_size=unit_kernel_size,
                    bias=bias,
                )
            ]
        self.num_blocks = len(self.conv_blocks)

        self.conv2 = Conv1d(out_channels, output_channels, kernel_size, 1, bias=False)

    def forward(self, z):
        x = self.conv1(z)
        for i in range(self.num_blocks):
            x = self.conv_blocks[i](x)
        x = self.conv2(x)
        return x


@dataclass
class HiggsAudioDecoderLayerOutput:
    logits: torch.FloatTensor
    audio_logits: torch.FloatTensor
    attentions: tuple[torch.FloatTensor, ...] | None = None
    past_key_values: tuple[tuple[torch.FloatTensor]] | None = None


class HiggsAudioDecoderProjector(HiggsAudioPreTrainedModel):
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


if is_deepspeed_available():
    from deepspeed.sequence.layer import _SeqAllToAll
    from deepspeed.utils import groups as deepspeed_groups
else:
    deepspeed_groups = None
    _SeqAllToAll = None


def _ceil_to_nearest(n, round_to):
    return (n + round_to - 1) // round_to * round_to


def count_parameters(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


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


def is_deepspeed_ulysses_enabled():
    if deepspeed_groups is None:
        return False

    """Check if sequence parallelism is enabled."""
    return deepspeed_groups._get_sequence_parallel_world_size() > 1


def support_deepspeed_ulysses(module):
    """A decorator around Pytorch module. It is needed for the module that needs access to sequence parallel info."""
    module._sp_size = None
    module._sp_rank = None
    module._sp_group = None

    @property
    def sp_size(self):
        if self._sp_size is None:
            self._sp_size = 1
            if is_deepspeed_ulysses_enabled():
                self._sp_size = deepspeed_groups._get_sequence_parallel_group().size()
        return self._sp_size

    @property
    def sp_rank(self):
        if self._sp_rank is None:
            self._sp_rank = 0
            if is_deepspeed_ulysses_enabled():
                self._sp_rank = deepspeed_groups._get_sequence_parallel_rank()
        return self._sp_rank

    @property
    def sp_group(self):
        if self._sp_group is None and is_deepspeed_ulysses_enabled():
            self._sp_group = deepspeed_groups._get_sequence_parallel_group()
        return self._sp_group

    module.sp_size = sp_size
    module.sp_rank = sp_rank
    module.sp_group = sp_group

    return module


def deepspeed_ulysses_attention(seq_dim=1, head_dim=2):
    """Perform all-to-all before and after the attention function."""

    def attention_decorator(attn_func=None):
        def wrapped(*args, **kwargs):
            if is_deepspeed_ulysses_enabled():
                sp_group = deepspeed_groups._get_sequence_parallel_group()
                scatter_idx = head_dim  # Scatter on num_heads dimension
                gather_idx = seq_dim  # Gather on seq_len dimension
                batch_dim_idx = 0
                args = list(args)
                args[0] = _SeqAllToAll.apply(
                    sp_group, args[0], scatter_idx, gather_idx, batch_dim_idx
                )
                args[1] = _SeqAllToAll.apply(
                    sp_group, args[1], scatter_idx, gather_idx, batch_dim_idx
                )
                args[2] = _SeqAllToAll.apply(
                    sp_group, args[2], scatter_idx, gather_idx, batch_dim_idx
                )
                args = tuple(args)

            attn_output = attn_func(*args, **kwargs)

            if is_deepspeed_ulysses_enabled():
                scatter_idx = seq_dim  # Scatter back on seq_len dimension
                gather_idx = head_dim  # Gather on num_heads dimension
                batch_dim_idx = 0
                attn_output = _SeqAllToAll.apply(
                    sp_group, attn_output, scatter_idx, gather_idx, batch_dim_idx
                )

            return attn_output

        return wrapped

    return attention_decorator


def deepspeed_ulysses_rope(state_seq_dim=2, trig_seq_dim=1):
    """Slice the corresponding cos and sin chunks for rope."""

    def rope_decorator(rope_func=None):
        def wrapped(*args, **kwargs):
            if is_deepspeed_ulysses_enabled():
                sp_rank = deepspeed_groups._get_sequence_parallel_rank()
                args = list(args)
                seq_chunk_size = args[0].size(state_seq_dim)
                args[2] = torch.narrow(
                    args[2], trig_seq_dim, sp_rank * seq_chunk_size, seq_chunk_size
                )
                args[3] = torch.narrow(
                    args[3], trig_seq_dim, sp_rank * seq_chunk_size, seq_chunk_size
                )
                args = tuple(args)

            return rope_func(*args, **kwargs)

        return wrapped

    return rope_decorator


def _gather_tensors(input_, group=None):
    """Gather tensors and concatenate them along a dimension."""
    input_ = input_.contiguous()
    world_size = torch.distributed.get_world_size(group)
    if world_size == 1:
        return input_
    tensor_shapes = [
        torch.empty(len(input_.size()), dtype=torch.int64, device=input_.device)
        for _ in range(world_size)
    ]
    input_size = torch.tensor(input_.size(), dtype=torch.int64, device=input_.device)
    torch.distributed.all_gather(tensor_shapes, input_size, group=group)
    gathered_buffers = [
        torch.empty(tensor_shapes[i].tolist(), dtype=input_.dtype, device=input_.device)
        for i in range(world_size)
    ]
    torch.distributed.all_gather(gathered_buffers, input_, group=group)
    return gathered_buffers


def _scatter_tensors(input_, group=None):
    """Scatter tensors."""
    world_size = torch.distributed.get_world_size(group)
    if world_size == 1:
        return input_
    rank = torch.distributed.get_rank(group)
    return input_[rank]


class _GatherTensors(torch.autograd.Function):
    """All gather tensors among the ranks."""

    @staticmethod
    def symbolic(graph, input_, group):
        return _gather_tensors(input_, group)

    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return torch.nested.as_nested_tensor(_gather_tensors(input_, group), layout=torch.jagged)

    @staticmethod
    def backward(ctx, grad_output):
        return _scatter_tensors(grad_output, ctx.group), None


def all_gather_tensors(input_, size=None, dim=0, group=None):
    if torch.distributed.get_world_size(group) == 1:
        # no sequence parallelism
        return input_
    gathered_tensors = _GatherTensors.apply(input_, group)

    if size:
        split_gathered_tensors = []
        for s, gathered_tensor in zip(size, gathered_tensors):
            split_gathered_tensor = torch.split(gathered_tensor, s.tolist())
            split_gathered_tensors.append(split_gathered_tensor)

        gathered_tensors = [y for x in zip(*split_gathered_tensors) for y in x]

    return torch.cat(gathered_tensors, dim).contiguous()


def get_sequence_data_parallel_world_size():
    return torch.distributed.get_world_size()


def get_sequence_data_parallel_rank():
    return torch.distributed.get_rank()


def get_sequence_data_parallel_group():
    return torch.distributed.group.WORLD


if is_deepspeed_available():
    deepspeed_groups._get_sequence_data_parallel_world_size = get_sequence_data_parallel_world_size
    deepspeed_groups._get_sequence_data_parallel_rank = get_sequence_data_parallel_rank
    deepspeed_groups._get_sequence_data_parallel_group = get_sequence_data_parallel_group


def _gather_tokens(input_, dim=0, group=None):
    """Gather tensors and concatenate them along a dimension"""
    input_ = input_.contiguous()
    world_size = torch.distributed.get_world_size(group)
    if world_size == 1:
        return input_

    gather_buffer = torch.empty(
        world_size * input_.numel(), dtype=input_.dtype, device=input_.device
    )
    torch.distributed.all_gather_into_tensor(gather_buffer, input_, group=group)
    if dim == 0:
        shape = list(input_.size())
        shape[0] = shape[0] * world_size
        output = gather_buffer.view(shape)
    else:
        tensor_list = [
            gather_buffer.narrow(0, input_.numel() * i, input_.numel()).view_as(input_)
            for i in range(world_size)
        ]
        # Note: torch.cat already creates a contiguous tensor.
        output = torch.cat(tensor_list, dim=dim).contiguous()

    return output


def _drop_tokens(input_, dim=0, group=None):
    """Divide a tensor among the sequence parallel ranks"""
    world_size = torch.distributed.get_world_size(group)
    if world_size == 1:
        return input_
    this_rank = torch.distributed.get_rank(group)
    assert input_.shape[dim] % world_size == 0, (
        f"input dimension {dim} ({input_.shape[dim]}) is not divisible by sequence parallel world size ({world_size})"
    )
    chunk_size = input_.shape[dim] // world_size

    return torch.narrow(input_, dim, this_rank * chunk_size, chunk_size)


class _DropTokens(torch.autograd.Function):
    """Divide tokens equally among the sequence parallel ranks"""

    @staticmethod
    def symbolic(graph, input_, dim, group, grad_scale):
        return _drop_tokens(input_, dim, group)

    @staticmethod
    def forward(ctx, input_, dim, group, grad_scale):
        ctx.dim = dim
        ctx.group = group
        ctx.grad_scale = grad_scale
        return _drop_tokens(input_, dim, group)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _gather_tokens(grad_output, ctx.dim, ctx.group)
        if ctx.grad_scale != 1:
            grad_input /= ctx.grad_scale
        return grad_input, None, None, None


class _GatherTokens(torch.autograd.Function):
    """Gather tokens among the sequence parallel ranks"""

    @staticmethod
    def symbolic(graph, input_, dim, group, grad_scale):
        return _gather_tokens(input_, dim, group)

    @staticmethod
    def forward(ctx, input_, dim, group, grad_scale):
        ctx.dim = dim
        ctx.group = group
        ctx.grad_scale = grad_scale
        return _gather_tokens(input_, dim, group)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _drop_tokens(grad_output, ctx.dim, ctx.group)
        if ctx.grad_scale != 1:
            grad_input *= ctx.grad_scale
        return grad_input, None, None, None


def drop_tokens(input_, dim=0, group=None, grad_scale=1):
    if torch.distributed.get_world_size(group) == 1:
        # no sequence parallelism
        return input_
    return _DropTokens.apply(input_, dim, group, grad_scale)


def gather_tokens(input_, dim=0, group=None, grad_scale=1):
    if torch.distributed.get_world_size(group) == 1:
        # no sequence parallelism
        return input_
    return _GatherTokens.apply(input_, dim, group, grad_scale)


def sequence_chunking_per_rank(sp_size, sp_rank, *args, dim=1):
    """Slice the inputs to create chuncks per the sequence parallel rank. This is used for the context parallel training.

    Args:
        sp_size (`int`):
            Sequence parallel size.
        sp_rank (`int`):
            Sequence parallel rank for the current process.
        dim (`int`):
           The dimension to slice
    """
    if sp_size == 1:
        return args[0] if len(args) == 1 else args

    seq_length = args[0].size(dim)
    for arg in args[1:]:
        assert arg.size(dim) == seq_length, (
            f"arg={arg} ({arg.shape[dim]}) does not have the same size as args[0] ({seq_length}) in dimension {dim}"
        )
    assert seq_length % sp_size == 0, (
        f"dimension {dim} ({args[0].shape[dim]}) is not divisible by sequence parallel world size ({sp_size})"
    )

    sub_seq_length = seq_length // sp_size
    sub_seq_start = sp_rank * sub_seq_length

    output = []
    for ind in args:
        ind = torch.narrow(ind, dim, sub_seq_start, sub_seq_length)
        output.append(ind)

    return tuple(output) if len(output) > 1 else output[0]


@contextmanager
def disable_deepspeed_ulysses():
    """Disable deepspeed ulysses (sequence parallelism) if it is enabled"""
    if is_deepspeed_ulysses_enabled():
        _old_get_sequence_parallel_world_size = deepspeed_groups._get_sequence_parallel_world_size

        def _get_sequence_parallel_world_size():
            return 1

        deepspeed_groups._get_sequence_parallel_world_size = _get_sequence_parallel_world_size
        try:
            yield
        finally:
            deepspeed_groups._get_sequence_parallel_world_size = (
                _old_get_sequence_parallel_world_size
            )
    else:
        context = contextlib.nullcontext
        with context():
            yield


class PartiallyFrozenEmbedding(nn.Module):
    """Split an existing `nn.Embedding` module that splits the embedding into:

    - A frozen embedding for indices [0..freeze_until_idx].
    - A trainable embedding for indices [freeze_until_idx+1..vocab_size-1].

    This should work with both Zero-2 and Zero-3 seamlessly
    """

    def __init__(self, original_embedding: nn.Embedding, freeze_until_idx: int):
        """:param original_embedding: An instance of nn.Embedding (the original embedding layer).
        :param freeze_until_idx: The index up to which the embedding is frozen (excluding). The freeze_until_idx is not frozen.
        """
        super().__init__()
        self.freeze_until_idx = freeze_until_idx
        self.original_vocab_size = original_embedding.num_embeddings
        self.embedding_dim = original_embedding.embedding_dim

        # Split the original embedding into frozen and trainable parts
        self.embedding_frozen = nn.Embedding(
            freeze_until_idx,
            self.embedding_dim,
            dtype=original_embedding.weight.dtype,
            device=original_embedding.weight.device,
        )
        self.embedding_trainable = nn.Embedding(
            self.original_vocab_size - freeze_until_idx,
            self.embedding_dim,
            dtype=original_embedding.weight.dtype,
            device=original_embedding.weight.device,
        )

        # Copy weights from the original embedding into the frozen and trainable parts
        with torch.no_grad():
            self.embedding_frozen.weight.copy_(original_embedding.weight[:freeze_until_idx])
            self.embedding_trainable.weight.copy_(original_embedding.weight[freeze_until_idx:])

        # Freeze the frozen embedding
        self.embedding_frozen.weight.requires_grad = False

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass for the split embedding wrapper.
        :param input_ids: Tensor of shape [batch_size, seq_len] with indices in [0..original_vocab_size-1].
        """
        # Masks to separate frozen and trainable indices
        # (bsz, seq_len)
        mask_frozen = input_ids < self.freeze_until_idx
        mask_trainable = ~mask_frozen

        # Output tensor for embedding results
        batch_size, seq_len = input_ids.shape
        embeddings = torch.zeros(
            batch_size,
            seq_len,
            self.embedding_dim,
            device=input_ids.device,
            dtype=self.embedding_frozen.weight.dtype,
        )

        # Handle frozen embedding
        if mask_frozen.any():
            frozen_ids = input_ids[mask_frozen]
            frozen_emb = self.embedding_frozen(frozen_ids)
            embeddings[mask_frozen] = frozen_emb

        # Handle trainable embedding
        if mask_trainable.any():
            # Adjust trainable IDs to the local index space of the trainable embedding
            trainable_ids = input_ids[mask_trainable] - (self.freeze_until_idx)
            trainable_emb = self.embedding_trainable(trainable_ids)
            embeddings[mask_trainable] = trainable_emb

        return embeddings

    def to_unsplit(self) -> nn.Embedding:
        unsplit_embedding = nn.Embedding(
            self.original_vocab_size,
            self.embedding_dim,
            dtype=self.embedding_frozen.weight.dtype,
            device=self.embedding_frozen.weight.device,
        )

        with torch.no_grad():
            unsplit_embedding.weight[: self.freeze_until_idx].copy_(self.embedding_frozen.weight)
            unsplit_embedding.weight[self.freeze_until_idx :].copy_(self.embedding_trainable.weight)

        return unsplit_embedding


class PartiallyFrozenLinear(nn.Module):
    """A wrapper around nn.Linear to partially freeze part of the weight matrix."""

    def __init__(self, original_linear: nn.Linear, freeze_until_idx: int):
        """:param original_linear: The original nn.Linear layer.
        :param freeze_until_idx: The index up to which the rows of the weight matrix are frozen.
        """
        super().__init__()
        assert original_linear.bias is None, "Currently only support linear module without bias"

        self.freeze_until_idx = freeze_until_idx
        self.input_dim = original_linear.in_features
        self.output_dim = original_linear.out_features

        # Create frozen and trainable linear layers
        self.linear_frozen = nn.Linear(
            self.input_dim,
            freeze_until_idx,
            bias=False,
            dtype=original_linear.weight.dtype,
            device=original_linear.weight.device,
        )
        self.linear_trainable = nn.Linear(
            self.input_dim,
            self.output_dim - freeze_until_idx,
            bias=False,
            dtype=original_linear.weight.dtype,
            device=original_linear.weight.device,
        )

        # Copy weights from the original linear layer
        with torch.no_grad():
            self.linear_frozen.weight.copy_(original_linear.weight[:freeze_until_idx])
            self.linear_trainable.weight.copy_(original_linear.weight[freeze_until_idx:])

        # Freeze the frozen linear layer
        self.linear_frozen.weight.requires_grad = False

    def forward(self, input_tensor):
        # input_tensor: (bsz, seq_len, hidden_state_dim)
        frozen_output = self.linear_frozen(input_tensor)
        trainable_output = self.linear_trainable(input_tensor)
        return torch.cat((frozen_output, trainable_output), dim=-1)

    def to_unsplit(self) -> nn.Linear:
        unsplit_linear = nn.Linear(
            self.input_dim,
            self.output_dim,
            bias=False,
            dtype=self.linear_frozen.weight.dtype,
            device=self.linear_frozen.weight.device,
        )

        # Copy weights from the frozen and trainable layers into the unsplit linear layer
        with torch.no_grad():
            unsplit_linear.weight[: self.freeze_until_idx].copy_(self.linear_frozen.weight)
            unsplit_linear.weight[self.freeze_until_idx :].copy_(self.linear_trainable.weight)

        return unsplit_linear


def _whisper_encoder_zero_shape_forward(whisper_encoder, *args, **kwargs):
    """The whisper encoder does not support zero-shape tensor by default due to the following implementations

        key_states = self._shape(self.k_proj(current_states), -1, bsz)

    If `bsz` is 0, the "-1" dimension will be ambiguous and triggers error in the shape inference pass.

    See also: https://github.com/huggingface/transformers/blob/30335093276212ce74938bdfd85bfd5df31a668a/src/transformers/models/whisper/modeling_whisper.py#L306-L307

    This function monkey-patches all `_shape` functions in the whisper encoder's self-attention layers to ensure function supports zero-shape tensor.

    #FIXME!!!! This is a temporary workaround and should be removed once the upstream issue is resolved.

    """
    global _higgs_flash_attention_forward

    def _patched_shape(tensor: torch.Tensor, seq_len: int, bsz: int, num_heads: int, head_dim: int):
        if seq_len == -1:
            return (
                tensor.view(bsz, tensor.shape[1], num_heads, head_dim).transpose(1, 2).contiguous()
            )
        else:
            return tensor.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()

    def _patched_scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ) -> torch.Tensor:
        # IMPORTANT! Implementation here is wrong and is only for the purpose of obtaining the correct attn_weight shape
        if enable_gqa:
            key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1)
        return attn_weight @ value

    # Apply monkey-patch
    if whisper_encoder.config._attn_implementation != "flash_attention_2":
        old_shape_functions = []
        for layer in whisper_encoder.layers:
            old_shape_functions.append(layer.self_attn._shape)
            layer.self_attn._shape = functools.partial(
                _patched_shape,
                num_heads=layer.self_attn.num_heads,
                head_dim=layer.self_attn.head_dim,
            )

    original_scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
    torch.nn.functional.scaled_dot_product_attention = _patched_scaled_dot_product_attention

    out = whisper_encoder(*args, **kwargs)
    torch.nn.functional.scaled_dot_product_attention = original_scaled_dot_product_attention

    # Restore the original shape functions
    if whisper_encoder.config._attn_implementation != "flash_attention_2":
        for layer, old_shape_function in zip(whisper_encoder.layers, old_shape_functions):
            layer.self_attn._shape = old_shape_function

    return out


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


class HiggsAudioFeatureProjector(nn.Module):
    """Projector that maps audio features extracted by Whisper to hidden state of the text model."""

    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.linear = nn.Linear(
            config.audio_encoder_config.d_model, config.text_config.hidden_size, bias=True
        )

    def forward(self, audio_features):
        hidden_states = self.linear(audio_features)
        return hidden_states


# Revised on top of transformers.models.qwen2_audio.modeling_qwen2_audio with Qwen2AudioEncoder --> HiggsAudioEncoder
# The code was originally borrowed from WhisperEncoder
class HiggsAudioEncoder(HiggsAudioPreTrainedModel):
    """Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`WhisperEncoderLayer`].

    Args:
        config: HiggsAudioEncoderConfig
    """

    # Ignore copy
    config_class = HiggsAudioEncoderConfig
    main_input_name = "input_features"
    _no_split_modules = ["WhisperEncoderLayer"]

    def __init__(self, config: HiggsAudioEncoderConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)

        # Flash Attention 2 does not support zero shape tensor, so we have to use sdpa implementation for the Whisper component.
        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.d_model)
        # Ignore copy
        self.avg_pooler = nn.AvgPool1d(2, stride=2)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        check_seq_length=True,
    ):
        r"""Args:
        input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
            Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
            obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
            `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
            `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
            and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
        attention_mask (`torch.Tensor`)`, *optional*):
            HiggsAudio does not support masking of the `input_features`, this argument is preserved for compatibility,
            but it is not used. By default the silence in the input log mel spectrogram are ignored.
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        expected_seq_length = (
            self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        )
        if check_seq_length and (input_features.shape[-1] != expected_seq_length):
            raise ValueError(
                f"HiggsAudio expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )

        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Ignore copy
        input_features = input_features.to(
            dtype=self.conv1.weight.dtype, device=self.conv1.weight.device
        )

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (len(self.layers)), (
                f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
            )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            # Ignore copy
            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Ignore copy
        hidden_states = hidden_states.permute(0, 2, 1)
        # If the sequence length after average pooling is not divisible by the sequence parallel size, we would duplicate it across the sequence parallel ranks.
        # In this case, gradients need to be scaled up because the subsequent scaling up in the function _apply_audio_tower is skipped.
        hidden_states = self.avg_pooler(hidden_states)

        hidden_states = hidden_states.permute(0, 2, 1)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, encoder_states, all_attentions] if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

    # Ignore copy
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """Computes the output length of the convolutional layers and the output length of the audio encoder"""
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths


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
        fast_forward: bool = False,
        use_audio_attention: bool = False,
    ):
        super().__init__()
        text_config = config.text_config
        self.hidden_size = text_config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = LlamaAttention(config=text_config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(text_config)

        if not fast_forward:
            if use_audio_attention:
                self.audio_attn = LlamaAttention(config=text_config, layer_idx=layer_idx + 1)
                self.audio_post_audio_attn_layer_norm = LlamaRMSNorm(
                    text_config.hidden_size, eps=text_config.rms_norm_eps
                )

            self.audio_mlp = LlamaMLP(text_config)
            self.audio_input_layernorm = LlamaRMSNorm(
                text_config.hidden_size, eps=text_config.rms_norm_eps
            )
            self.audio_post_attention_layernorm = LlamaRMSNorm(
                text_config.hidden_size, eps=text_config.rms_norm_eps
            )

        self.use_audio_attention = use_audio_attention
        self.fast_forward = fast_forward
        if self.fast_forward:
            assert not self.use_audio_attention, (
                "We cannot use audio_attention if the layer is marked as fast-forward."
            )
        self.input_layernorm = LlamaRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            text_config.hidden_size, eps=text_config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        audio_attention_mask: torch.Tensor | None = None,
        fast_forward_attention_mask: torch.Tensor | None = None,
        audio_out_mask: torch.BoolTensor | None = None,
        is_decoding_audio_token: bool | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = False,
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

        # If we are decoding an audio token and the layer is marked as fast-forward,
        # we can skip it.
        if is_decoding_audio_token and self.fast_forward:
            return (hidden_states,)

        has_audio_out = audio_out_mask is not None and audio_out_mask.shape[0] > 0

        if has_audio_out and not self.fast_forward:
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

        if has_audio_out and not self.fast_forward:
            if use_cache:
                real_audio_out_mask = audio_out_mask_sq[:, -target_length:]
            else:
                real_audio_out_mask = audio_out_mask_sq

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
                text_hidden_states = self.post_attention_layernorm(
                    hidden_states[~real_audio_out_mask]
                )
                audio_hidden_states = self.audio_post_attention_layernorm(
                    hidden_states[real_audio_out_mask]
                )

                text_hidden_states = self.mlp(text_hidden_states)
                residual[~real_audio_out_mask] += text_hidden_states

                audio_hidden_states = self.audio_mlp(audio_hidden_states)
                residual[real_audio_out_mask] += audio_hidden_states

            hidden_states = residual
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        if self.fast_forward and has_audio_out:
            if use_cache:
                hidden_states = torch.where(
                    audio_out_mask_sq[:, -target_length:].unsqueeze(-1),
                    original_hidden_states,
                    hidden_states,
                )
            else:
                hidden_states = torch.where(
                    audio_out_mask_sq.unsqueeze(-1), original_hidden_states, hidden_states
                )

        outputs = (hidden_states,)

        if output_attentions:
            if self.use_audio_attention:
                # The returned attn weights have shape (batch_size, num_heads + num_audio_attn_heads, seq_length, seq_length)
                outputs += (torch.concat([self_attn_weights, audio_self_attn_weights], dim=1),)
            else:
                # The returned attn weights have shape (batch_size, num_heads, seq_length, seq_length)
                outputs += (self_attn_weights,)

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
        self.use_audio_out_embed_projector = config.use_audio_out_embed_projector
        self.use_audio_out_self_attention = config.use_audio_out_self_attention

        self.embed_tokens = nn.Embedding(
            self.vocab_size, config.text_config.hidden_size, self.padding_idx
        )

        if config.audio_adapter_type == "dual_ffn":
            layer_idx = 0
            layers = []
            for j in range(config.text_config.num_hidden_layers):
                if j in config.audio_dual_ffn_layers:
                    layers.append(
                        HiggsAudioDualFFNDecoderLayer(
                            config, layer_idx, use_audio_attention=self.use_audio_out_self_attention
                        )
                    )
                    layer_idx += 2 if self.use_audio_out_self_attention else 1
                else:
                    layers.append(LlamaDecoderLayer(config.text_config, layer_idx))
                    layer_idx += 1
            self.layers = nn.ModuleList(layers)
        elif config.audio_adapter_type == "dual_ffn_fast_forward":
            layer_idx = 0
            layers = []
            for j in range(config.text_config.num_hidden_layers):
                if j in config.audio_dual_ffn_layers:
                    layers.append(
                        HiggsAudioDualFFNDecoderLayer(
                            config,
                            layer_idx,
                            fast_forward=False,
                            use_audio_attention=self.use_audio_out_self_attention,
                        )
                    )
                    layer_idx += 2 if self.use_audio_out_self_attention else 1
                else:
                    layers.append(
                        HiggsAudioDualFFNDecoderLayer(
                            config, layer_idx, fast_forward=True, use_audio_attention=False
                        )
                    )
                    layer_idx += 1
            self.layers = nn.ModuleList(layers)
        elif config.audio_adapter_type == "stack":
            self.layers = nn.ModuleList(
                [
                    LlamaDecoderLayer(config.text_config, layer_idx)
                    for layer_idx in range(config.text_config.num_hidden_layers)
                ]
            )
            layer_idx = config.text_config.num_hidden_layers
        else:
            raise NotImplementedError(
                f"Audio adapter type {config.audio_adapter_type} not implemented."
            )

        self.num_activation_checkpointing_layers = len(self.layers)

        self.decode_graph_runners = defaultdict(dict[bool, CUDAGraphRunner])
        self.norm = LlamaRMSNorm(
            config.text_config.hidden_size, eps=config.text_config.rms_norm_eps
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=config.text_config)

        if not config.skip_audio_tower:
            self.audio_tower = HiggsAudioEncoder(config.audio_encoder_config)
            self.audio_encoder_proj = HiggsAudioFeatureProjector(config)
        else:
            self.audio_tower = None
            self.audio_encoder_proj = None
        self.audio_decoder_proj = HiggsAudioDecoderProjector(config, layer_idx=layer_idx)
        self.audio_codebook_size = (
            config.audio_codebook_size + 2
        )  # We add 1 for the audio_stream_bos token and 1 for the audio_stream_eos token

        if config.use_audio_out_embed_projector:
            self.audio_out_embed_projector = nn.Linear(
                config.text_config.hidden_size, config.text_config.hidden_size, bias=False
            )

        self.audio_codebook_embeddings = nn.Embedding(
            config.audio_num_codebooks * self.audio_codebook_size, config.text_config.hidden_size
        )

        self.audio_codebook_weights = (
            torch.ones(config.audio_num_codebooks) / config.audio_num_codebooks
        )  # default to equal weights
        self.post_init()

    def set_num_activation_checkpointing_layers(self, num_layers):
        self.num_activation_checkpointing_layers = num_layers

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
        if self.use_audio_out_embed_projector:
            audio_embed = self.audio_out_embed_projector(audio_embed)
        return audio_embed

    def _apply_audio_tower(self, audio_features, audio_feature_attention_mask):
        """Apply the audio tower to the audio features"""
        if audio_features.shape[0] == 0:
            if torch.is_grad_enabled():
                # FIXME!!!!!!!!
                # This is a hack to ensure that the forward+backward pass of audio_tower and audio_encoder_proj get triggered.
                # The monkey patch won't overwrite the backward pass of nn.Module.
                audio_outputs = _whisper_encoder_zero_shape_forward(
                    self.audio_tower, audio_features, attention_mask=None, check_seq_length=False
                )
                selected_audio_feature = audio_outputs.last_hidden_state
                audio_features_embed = self.audio_encoder_proj(selected_audio_feature)
                audio_feat_out_lengths = None
                return audio_features_embed, audio_feat_out_lengths
            else:
                return None, None

        audio_feat_lengths, audio_feat_out_lengths = (
            self.audio_tower._get_feat_extract_output_lengths(audio_feature_attention_mask.sum(-1))
        )
        batch_size, _, max_mel_seq_len = audio_features.shape
        max_seq_len = (max_mel_seq_len - 1) // 2 + 1
        # Create a sequence tensor of shape (batch_size, max_seq_len)
        seq_range = (
            torch.arange(
                0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device
            )
            .unsqueeze(0)
            .expand(batch_size, max_seq_len)
        )
        lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
        # Create mask
        padding_mask = seq_range < lengths_expand

        if self.config._attn_implementation != "flash_attention_2":
            audio_attention_mask = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                batch_size, 1, max_seq_len, max_seq_len
            )
        else:
            audio_attention_mask = padding_mask

        audio_outputs = self.audio_tower(audio_features, attention_mask=audio_attention_mask)
        selected_audio_feature = audio_outputs.last_hidden_state
        audio_features_embed = self.audio_encoder_proj(selected_audio_feature)

        return audio_features_embed, audio_feat_out_lengths

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
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
        if (
            self.config._attn_implementation == "sdpa"
            and not using_static_cache
            and not output_attentions
        ):
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
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    def _prepare_all_static_kv_cache_masks(
        self, hidden_states, attention_mask, audio_out_mask, past_key_values
    ):
        target_length = hidden_states.shape[1]
        cur_pos = audio_out_mask.shape[1]
        min_dtype = torch.finfo(hidden_states.dtype).min
        assert len(attention_mask.shape) == 4, "Only support SDPA for now"
        kv_cache_len = past_key_values.get_max_cache_shape()
        audio_out_mask_padded = torch.nn.functional.pad(
            audio_out_mask, (0, kv_cache_len - cur_pos), value=True
        )
        fast_forward_attention_mask = attention_mask.masked_fill(
            audio_out_mask_padded[
                :, audio_out_mask.shape[1] - target_length : audio_out_mask.shape[1]
            ].reshape(audio_out_mask_padded.shape[0], 1, target_length, 1)
            | audio_out_mask_padded.reshape(
                audio_out_mask_padded.shape[0], 1, 1, audio_out_mask_padded.shape[1]
            ),
            min_dtype,
        )

        no_audio_out_mask = ~audio_out_mask
        no_audio_out_mask = torch.nn.functional.pad(
            no_audio_out_mask, (0, kv_cache_len - audio_out_mask.shape[1]), value=False
        )
        no_audio_out_mask = no_audio_out_mask[
            :, audio_out_mask.shape[1] - target_length : audio_out_mask.shape[1]
        ].reshape(audio_out_mask.shape[0], 1, target_length, 1) | no_audio_out_mask.reshape(
            audio_out_mask.shape[0], 1, 1, kv_cache_len
        )
        audio_attention_mask = attention_mask.masked_fill(no_audio_out_mask, min_dtype)
        return fast_forward_attention_mask, audio_attention_mask

    def _forward_core(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        audio_discrete_codes_mask: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Union[Cache, list[torch.FloatTensor]] | None,
        use_cache: bool,
        audio_attention_mask: torch.Tensor,
        fast_forward_attention_mask: torch.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        is_decoding_audio_token: bool | None = None,
        is_using_cuda_graph: bool | None = False,
    ):
        # create position embeddings to be shared across the decoder layers
        # When past_key_values is passed in, we need to offset the position ids when calculating the position embeddings.
        # Therefore, cache_position is used.

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if isinstance(decoder_layer, HiggsAudioDualFFNDecoderLayer):
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    audio_attention_mask=audio_attention_mask,
                    fast_forward_attention_mask=fast_forward_attention_mask,
                    audio_out_mask=audio_discrete_codes_mask,
                    is_decoding_audio_token=is_decoding_audio_token,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    is_using_cuda_graph=is_using_cuda_graph,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        return hidden_states, all_hidden_states, all_self_attns

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
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_audio_hidden_states: bool | None = False,
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
            output_attentions (:obj:`bool`):
                Whether to output attentions.
            output_hidden_states (:obj:`bool`):
                Whether to output hidden states.
            output_audio_hidden_states (:obj:`bool`):
                Whether to output audio hidden states.
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

        # 2. Extract audio embeddings
        if self.config.skip_audio_tower:
            audio_features_embed = audio_features_length = None
        else:
            audio_features_embed, audio_features_length = self._apply_audio_tower(
                audio_features, audio_feature_attention_mask
            )

        if self.config.encode_audio_in_tokens:
            if audio_in_ids is not None and audio_in_ids.shape[-1] > 0:
                audio_in_ids = audio_in_ids.to(target_device)
            else:
                audio_in_ids = torch.zeros(
                    (self.audio_num_codebooks, 0), device=target_device, dtype=torch.long
                )
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
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        audio_discrete_codes_mask = audio_in_discrete_codes_mask | audio_out_mask
        if cache_audio_discrete_codes_mask is not None and use_cache:
            audio_discrete_codes_mask = torch.concat(
                [cache_audio_discrete_codes_mask, audio_discrete_codes_mask], dim=1
            )

        # Generate the audio attention mask outside the layer to avoid recompilation
        if use_static_cache:
            fast_forward_attention_mask, audio_attention_mask = (
                self._prepare_all_static_kv_cache_masks(
                    hidden_states, causal_mask, audio_discrete_codes_mask, past_key_values
                )
            )
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
            audio_attention_mask=audio_attention_mask if use_static_cache else None,
            fast_forward_attention_mask=fast_forward_attention_mask if use_static_cache else None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            is_using_cuda_graph=is_using_cuda_graph,
        )
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

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
            output_attentions=output_attentions,
            output_audio_hidden_states=output_audio_hidden_states,
            cache_position=cache_position,
        )

        if audio_logits is not None:
            audio_logits = audio_logits.view(
                audio_logits.shape[0], self.audio_num_codebooks, self.audio_codebook_size
            ).float()

        if output_hidden_states:
            if decoder_all_hidden_states is not None and len(decoder_all_hidden_states) > 1:
                all_hidden_states += decoder_all_hidden_states[1:]

        if output_attentions:
            all_self_attns += decoder_all_self_attns

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
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
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

        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

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

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update(
                {"output_attentions": output_attentions} if output_attentions else {}
            )
            model_inputs.update(
                {"output_hidden_states": output_hidden_states} if output_hidden_states else {}
            )

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
                if output_attentions:
                    decoder_attentions += (outputs.attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (outputs.hidden_states,)

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

    def parameter_count_per_component(self):
        """Count the number of parameters per component in the model.

        HiggsAudio has the following main components:
            audio_tower: For mapping audio features to hidden states),
            llm_embed: The size of embedding layer of the LLM
            llm_non_embed: The size of non-embedding layer of the LLM
            audio_adapter: The overall size of additional layers for audio generation

        """
        trainable_stats = {
            "audio_tower": 0,
            "llm_embed": 0,
            "llm_non_embed": 0,
            "audio_embed": 0,
            "audio_adapter": 0,
            "overall": 0,
        }
        total_stats = {
            "audio_tower": 0,
            "llm_embed": 0,
            "llm_non_embed": 0,
            "audio_embed": 0,
            "audio_adapter": 0,
            "overall": 0,
        }

        total_stats["overall"] = count_parameters(self, trainable_only=False)
        trainable_stats["overall"] = count_parameters(self, trainable_only=True)

        for mod in [self.audio_tower]:
            if mod is not None:
                total_stats["audio_tower"] += count_parameters(mod, trainable_only=False)
                trainable_stats["audio_tower"] += count_parameters(mod, trainable_only=True)

        total_stats["llm_embed"] = count_parameters(self.embed_tokens, trainable_only=False)
        trainable_stats["llm_embed"] = count_parameters(self.embed_tokens, trainable_only=True)

        total_stats["audio_embed"] = count_parameters(
            self.audio_codebook_embeddings, trainable_only=False
        )
        trainable_stats["audio_embed"] = count_parameters(
            self.audio_codebook_embeddings, trainable_only=True
        )

        # Calculate number of parameters for LLM
        for layer in self.layers:
            if isinstance(layer, HiggsAudioDualFFNDecoderLayer):
                total_param_count = count_parameters(layer, trainable_only=False)
                total_trainable_param_count = count_parameters(layer, trainable_only=True)
                total_stats["llm_non_embed"] += total_param_count
                trainable_stats["llm_non_embed"] += total_trainable_param_count
                if not layer.fast_forward:
                    audio_mlp_param_count = count_parameters(layer.audio_mlp, trainable_only=False)
                    audio_mlp_trainable_param_count = count_parameters(
                        layer.audio_mlp, trainable_only=True
                    )

                    audio_norm_param_count = count_parameters(
                        layer.audio_post_attention_layernorm, trainable_only=False
                    ) + count_parameters(layer.audio_input_layernorm, trainable_only=False)
                    audio_norm_trainable_param_count = count_parameters(
                        layer.audio_post_attention_layernorm, trainable_only=True
                    ) + count_parameters(layer.audio_input_layernorm, trainable_only=True)
                    total_stats["llm_non_embed"] -= audio_mlp_param_count + audio_norm_param_count
                    trainable_stats["llm_non_embed"] -= (
                        audio_mlp_trainable_param_count + audio_norm_trainable_param_count
                    )
                    total_stats["audio_adapter"] += audio_mlp_param_count + audio_norm_param_count
                    trainable_stats["audio_adapter"] += (
                        audio_mlp_trainable_param_count + audio_norm_trainable_param_count
                    )

                    if layer.use_audio_attention:
                        audio_attn_param_count = count_parameters(
                            layer.audio_attn, trainable_only=False
                        ) + count_parameters(
                            layer.audio_post_audio_attn_layer_norm, trainable_only=False
                        )
                        audio_attn_trainable_param_count = count_parameters(
                            layer.audio_attn, trainable_only=True
                        ) + count_parameters(
                            layer.audio_post_audio_attn_layer_norm, trainable_only=True
                        )
                        total_stats["llm_non_embed"] -= audio_attn_param_count
                        trainable_stats["llm_non_embed"] -= audio_attn_trainable_param_count
                        total_stats["audio_adapter"] += audio_attn_param_count
                        trainable_stats["audio_adapter"] += audio_attn_trainable_param_count
            else:
                total_stats["llm_non_embed"] += count_parameters(layer, trainable_only=False)
                trainable_stats["llm_non_embed"] += count_parameters(layer, trainable_only=True)
        total_stats["llm_non_embed"] += count_parameters(self.norm, trainable_only=False)
        trainable_stats["llm_non_embed"] += count_parameters(self.norm, trainable_only=True)

        total_stats["audio_adapter"] += count_parameters(
            self.audio_decoder_proj.audio_lm_head, trainable_only=False
        )
        trainable_stats["audio_adapter"] += count_parameters(
            self.audio_decoder_proj.audio_lm_head, trainable_only=True
        )
        total_stats["llm_embed"] += count_parameters(
            self.audio_decoder_proj.text_lm_head, trainable_only=False
        )
        trainable_stats["llm_embed"] += count_parameters(
            self.audio_decoder_proj.text_lm_head, trainable_only=True
        )

        other_audio_modules = [self.audio_encoder_proj]
        if self.use_audio_out_embed_projector:
            other_audio_modules.append(self.audio_out_embed_projector)

        for mod in other_audio_modules:
            if mod is not None:
                total_stats["audio_adapter"] += count_parameters(mod, trainable_only=False)
                trainable_stats["audio_adapter"] += count_parameters(mod, trainable_only=True)
        return {"trainable": trainable_stats, "total": total_stats}

    def set_skip_audio_tower(self):
        self.config.skip_audio_tower = True
        self.config.encode_whisper_embed = False

    def set_encode_audio_in_tokens(self):
        self.config.encode_audio_in_tokens = True

    def freeze_audio_tower(self):
        if self.audio_tower is not None:
            for param in self.audio_tower.parameters():
                param.requires_grad = False

    def freeze_audio_encoder_proj(self):
        if self.audio_encoder_proj is not None:
            for param in self.audio_encoder_proj.parameters():
                param.requires_grad = False

    def freeze_llm(self, freeze_embed=True, freeze_embed_until_idx: int | None = None):
        for layer in self.layers:
            if isinstance(layer, HiggsAudioDualFFNDecoderLayer):
                for param in layer.self_attn.parameters():
                    param.requires_grad = False
                for param in layer.mlp.parameters():
                    param.requires_grad = False

                for param in layer.post_attention_layernorm.parameters():
                    param.requires_grad = False

                for param in layer.input_layernorm.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = False

        for param in self.norm.parameters():
            param.requires_grad = False

        if freeze_embed:
            if freeze_embed_until_idx is None:
                for param in self.embed_tokens.parameters():
                    param.requires_grad = False
            else:
                assert isinstance(self.embed_tokens, nn.Embedding)
                self.embed_tokens = PartiallyFrozenEmbedding(
                    original_embedding=self.embed_tokens, freeze_until_idx=freeze_embed_until_idx
                )

    def freeze_text_head(self, freeze_text_head_until_idx: int | None = None):
        """Freeze the final text head"""
        if freeze_text_head_until_idx is None:
            for param in self.audio_decoder_proj.text_lm_head.parameters():
                param.requires_grad = False

        else:
            assert isinstance(self.audio_decoder_proj.text_lm_head, nn.Linear)
            self.audio_decoder_proj.text_lm_head = PartiallyFrozenLinear(
                original_linear=self.audio_decoder_proj.text_lm_head,
                freeze_until_idx=freeze_text_head_until_idx,
            )

    @classmethod
    def merge_weights_from_checkpoint(
        cls, checkpoint_dir: str, merged_output_dir: str, *model_args, **kwargs
    ):
        # For users' convenience, we merge back embedding and text_lm_head if they are splitted
        splitted_model = super().from_pretrained(
            checkpoint_dir,
            *model_args,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            **{**kwargs, "state_dict": None},  # Prevent auto-loading state_dict
        )

        # Load all safetensor shards
        state_dict = {}
        shard_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))

        for shard_path in shard_paths:
            shard_dict = load_file(shard_path)  # Load each shard
            state_dict.update(shard_dict)  # Merge into a single dict

        # Merge weights
        if (
            "audio_decoder_proj.text_lm_head.linear_frozen.weight" in state_dict
            and "audio_decoder_proj.text_lm_head.linear_trainable.weight" in state_dict
        ):
            state_dict["audio_decoder_proj.text_lm_head.weight"] = torch.cat(
                [
                    state_dict["audio_decoder_proj.text_lm_head.linear_frozen.weight"],
                    state_dict["audio_decoder_proj.text_lm_head.linear_trainable.weight"],
                ],
                dim=0,
            )

            del state_dict["audio_decoder_proj.text_lm_head.linear_frozen.weight"]
            del state_dict["audio_decoder_proj.text_lm_head.linear_trainable.weight"]

        if (
            "embed_tokens.embedding_frozen.weight" in state_dict
            and "embed_tokens.embedding_trainable.weight" in state_dict
        ):
            state_dict["embed_tokens.weight"] = torch.cat(
                [
                    state_dict["embed_tokens.embedding_frozen.weight"],
                    state_dict["embed_tokens.embedding_trainable.weight"],
                ],
                dim=0,
            )

            del state_dict["embed_tokens.embedding_frozen.weight"]
            del state_dict["embed_tokens.embedding_trainable.weight"]

        # Load the final state_dict
        splitted_model.load_state_dict(state_dict, strict=True)

        if merged_output_dir:
            splitted_model.save_pretrained(
                merged_output_dir, is_main_process=True, state_dict=state_dict
            )

    @torch.inference_mode()
    def capture_model(self, past_key_values: list[Union[Cache, list[torch.FloatTensor]]]) -> None:
        """Capture CUDA graphs for the model's forward pass with different KV cache lengths.

        Args:
            past_key_values: List of KV caches to capture graphs for
        """
        for past_key_value in past_key_values:
            kv_cache_length = past_key_value.get_max_cache_shape()
            # We capture two graphs, one for decoding audio tokens and one for decoding text tokens
            for is_decoding_audio_token in [True, False]:
                runner = CUDAGraphRunner(self._forward_core)

                # Create dummy inputs for graph capture
                batch_size = 1
                hidden_dim = self.config.hidden_size

                hidden_states = torch.zeros(
                    (batch_size, 1, hidden_dim), dtype=self.config.torch_dtype, device=self.device
                )
                causal_mask = torch.ones(
                    (batch_size, 1, 1, kv_cache_length),
                    dtype=self.config.torch_dtype,
                    device=self.device,
                )
                audio_discrete_codes_mask = torch.tensor(
                    [[is_decoding_audio_token]], dtype=torch.bool, device=self.device
                )
                cache_position = torch.tensor(
                    [kv_cache_length - 1], dtype=torch.long, device=self.device
                )
                audio_attention_mask = torch.ones_like(causal_mask)
                fast_forward_attention_mask = torch.ones_like(causal_mask)

                runner.capture(
                    hidden_states=hidden_states,
                    causal_mask=causal_mask,
                    audio_discrete_codes_mask=audio_discrete_codes_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_value,
                    use_cache=True,
                    audio_attention_mask=audio_attention_mask,
                    fast_forward_attention_mask=fast_forward_attention_mask,
                    output_attentions=False,
                    output_hidden_states=False,
                    is_decoding_audio_token=is_decoding_audio_token,
                    is_using_cuda_graph=True,
                    stream=torch.cuda.Stream(device=self.device),
                )

                self.decode_graph_runners[kv_cache_length][is_decoding_audio_token] = runner


# Whisper processor, 30 sec -> 3000 features
# Then we divide 4 in the audio towker, we decrease 3000 features to 750, which gives 25 Hz
WHISPER_EMBED_NUM_HIDDEN_STATE_PER_SEC = 25


@dataclass
class ChatMLDatasetSample:
    input_ids: torch.LongTensor  # Shape (seq_len,): The input text tokens.
    label_ids: torch.LongTensor  # Shape (seq_len,): The label ids.
    audio_ids_concat: (
        torch.LongTensor
    )  # Shape (num_codebooks, audio_seq_len): The audio tokens that are concatenated.
    # Here `audio_seq_len` is the length of the concatenated audio tokens.`
    audio_ids_start: (
        torch.LongTensor
    )  # Shape (num_audios,): The start index of each audio token in the concatenated audio tokens.
    audio_waveforms_concat: (
        torch.Tensor
    )  # Shape (total_wv_length,): The concatenated audio waveforms for audio-in features.
    audio_waveforms_start: torch.LongTensor  # Shape (num_audios,): The start index of each audio waveform in the concatenated audio waveforms.
    audio_sample_rate: (
        torch.Tensor
    )  # Shape (num_audios,): The sampling rate of the audio waveforms.
    audio_speaker_indices: (
        torch.LongTensor
    )  # Shape (num_audios,) -1 means unknown speaker: The speaker indices for each audio.
    audio_label_ids_concat: torch.LongTensor | None = (
        None  # Shape (num_codebooks, audio_seq_len): The audio tokens that are concatenated.
    )
    # Here `audio_seq_len` is the length of the concatenated audio tokens.`
    reward: float | None = None

    def num_audios(self):
        return max(len(self.audio_waveforms_start), len(self.audio_ids_start))

    def get_audio_codes(self, idx):
        code_start = self.audio_ids_start[idx]
        if idx < len(self.audio_ids_start) - 1:
            code_end = self.audio_ids_start[idx + 1]
        else:
            code_end = self.audio_ids_concat.shape[-1]

        return self.audio_ids_concat[:, code_start:code_end]

    def get_audio_codes_labels(self, idx):
        if self.audio_label_ids_concat is None:
            return None
        code_start = self.audio_ids_start[idx]
        if idx < len(self.audio_ids_start) - 1:
            code_end = self.audio_ids_start[idx + 1]
        else:
            code_end = self.audio_ids_concat.shape[-1]

        return self.audio_label_ids_concat[:, code_start:code_end]

    def get_wv(self, idx):
        wv_start = self.audio_waveforms_start[idx]
        sr = self.audio_sample_rate[idx]
        if idx < len(self.audio_waveforms_start) - 1:
            wv_end = self.audio_waveforms_start[idx + 1]
        else:
            wv_end = self.audio_waveforms_concat.shape[-1]
        return self.audio_waveforms_concat[wv_start:wv_end], sr

    def cal_num_tokens(
        self,
        encode_whisper_embed: bool = True,
        encode_audio_in_tokens: bool = False,
        encode_audio_out_tokens: bool = True,
        audio_in_token_id: int = 128015,
        audio_out_token_id: int = 128016,
    ) -> int:
        # we firstly exclude <|AUDIO|> and <|AUDIO_OUT|> because we do late merging and replace those position with actual audio features and audio token ids
        # It's assumed that we always have audio_ids when audio_waveforms are there (but not vice-versa)
        num_tokens = len(self.input_ids) - len(self.audio_ids_start)

        if encode_whisper_embed and len(self.audio_waveforms_concat) > 0:
            audio_lengths = torch.diff(self.audio_waveforms_start)
            if len(audio_lengths):
                # Sum before calling .item()
                num_tokens += (
                    (
                        np.ceil(
                            WHISPER_EMBED_NUM_HIDDEN_STATE_PER_SEC
                            * audio_lengths
                            / self.audio_sample_rate[:-1]
                        )
                    ).sum()
                ).item()
            # add the last audio's token estimation
            num_tokens += (
                np.ceil(
                    WHISPER_EMBED_NUM_HIDDEN_STATE_PER_SEC
                    * (self.audio_waveforms_concat.shape[0] - self.audio_waveforms_start[-1])
                    / self.audio_sample_rate[-1]
                )
            ).item()

        if self.audio_ids_concat.size(1) > 0:
            audio_io_ids = self.input_ids[
                (self.input_ids == audio_in_token_id) | (self.input_ids == audio_out_token_id)
            ]
            audio_io_id_lengths = torch.concat(
                [
                    torch.diff(self.audio_ids_start),
                    torch.tensor([self.audio_ids_concat.shape[-1] - self.audio_ids_start[-1]]),
                ]
            )
            if encode_audio_in_tokens:
                num_tokens += torch.sum(
                    audio_io_id_lengths[audio_io_ids == audio_in_token_id]
                ).item()

            if encode_audio_out_tokens:
                num_tokens += torch.sum(
                    audio_io_id_lengths[audio_io_ids == audio_out_token_id]
                ).item()

        return int(num_tokens)

    @classmethod
    def merge(
        cls,
        samples: list["ChatMLDatasetSample"],
        eos_token_id: int,
        ignore_index: int,
        padding_size: int | None = None,
    ) -> "ChatMLDatasetSample":
        """Merges a list of ChatMLDatasetSample instances, inserting eos_token_id and ignore_index between them, and adjusting offsets for audio_ids_start and audio_waveforms_start.

        Args:
            samples (List[ChatMLDatasetSample]): List of samples to merge.
            eos_token_id (int): Tokens to be inserted into input_ids between samples.
            ignore_index (int): Default label for padding.
            padding_size (Optional[int]): If provided, pad the sequence to with this length.

        Returns:
            ChatMLDatasetSample: Merged and potentially padded sample.
        """
        if not samples:
            logger.fatal("The samples list is empty and cannot be merged.")
            raise ValueError("The samples list is empty and cannot be merged.")

        # Initialize empty lists for concatenation
        input_ids_list = []
        label_ids_list = []
        audio_ids_concat_list = []
        audio_ids_start_list = []
        audio_waveforms_concat_list = []
        audio_waveforms_start_list = []
        audio_sample_rate_list = []
        audio_speaker_indices_list = []

        # Track offsets
        audio_ids_offset = 0
        audio_waveforms_offset = 0

        for sample in samples:
            # Add input_ids and label_ids with padding
            if input_ids_list:
                input_ids_list.append(torch.tensor([eos_token_id], dtype=torch.long))
                label_ids_list.append(torch.tensor([ignore_index], dtype=torch.long))
            input_ids_list.append(sample.input_ids)
            label_ids_list.append(sample.label_ids)

            # Add audio_ids_concat and handle empty audio ids
            if sample.audio_ids_concat.size(1) > 0:
                audio_ids_concat_list.append(sample.audio_ids_concat)

                # Offset and add audio_ids_start
                audio_ids_start_list.append(sample.audio_ids_start + audio_ids_offset)
                audio_ids_offset += sample.audio_ids_concat.size(
                    1
                )  # (num_codebooks, seq_len): Update offset by audio_seq_len

            # Add audio_waveforms_concat
            if sample.audio_waveforms_concat.size(0) > 0:
                # Check dimensions of the audio waveform to ensure consistency
                if (
                    audio_waveforms_concat_list
                    and sample.audio_waveforms_concat.dim() != audio_waveforms_concat_list[0].dim()
                ):
                    logger.warning(
                        f"Skipping audio waveform with inconsistent dimensions: expected {audio_waveforms_concat_list[0].dim()}D, got {sample.audio_waveforms_concat.dim()}D"
                    )
                    continue

                audio_waveforms_concat_list.append(sample.audio_waveforms_concat)
                audio_waveforms_start_list.append(
                    sample.audio_waveforms_start + audio_waveforms_offset
                )
                audio_waveforms_offset += sample.audio_waveforms_concat.size(0)

                # Add audio_sample_rate and audio_speaker_indices
                audio_sample_rate_list.append(sample.audio_sample_rate)

            audio_speaker_indices_list.append(sample.audio_speaker_indices)

        # Concatenate all tensors
        input_ids = torch.cat(input_ids_list, dim=0)
        label_ids = torch.cat(label_ids_list, dim=0)

        # Apply padding if padding_size is specified
        if padding_size is not None and padding_size > 0:
            input_ids = torch.cat(
                [input_ids, torch.full((padding_size,), eos_token_id, dtype=torch.long)], dim=0
            )
            label_ids = torch.cat(
                [label_ids, torch.full((padding_size,), ignore_index, dtype=torch.long)], dim=0
            )

        # Safely concatenate audio tensors with proper error handling
        try:
            audio_ids_concat = (
                torch.cat(audio_ids_concat_list, dim=1)
                if audio_ids_concat_list
                else torch.tensor([[]])
            )
            audio_ids_start = (
                torch.cat(audio_ids_start_list, dim=0) if audio_ids_start_list else torch.tensor([])
            )

            # Check for dimensional consistency in audio waveforms
            if audio_waveforms_concat_list:
                dims = [t.dim() for t in audio_waveforms_concat_list]
                if not all(d == dims[0] for d in dims):
                    # If dimensions don't match, log warning and filter out the problematic tensors
                    logger.warning(
                        f"Inconsistent dimensions in audio waveforms: {dims}. Filtering to keep only consistent ones."
                    )
                    expected_dim = max(set(dims), key=dims.count)  # Most common dimension
                    audio_waveforms_concat_list = [
                        t for t in audio_waveforms_concat_list if t.dim() == expected_dim
                    ]

                    # Recalculate audio_waveforms_start with the filtered list
                    if audio_waveforms_concat_list:
                        audio_waveforms_offset = 0
                        audio_waveforms_start_list = []
                        for waveform in audio_waveforms_concat_list:
                            audio_waveforms_start_list.append(
                                torch.tensor([audio_waveforms_offset])
                            )
                            audio_waveforms_offset += waveform.size(0)

            audio_waveforms_concat = (
                torch.cat(audio_waveforms_concat_list, dim=0)
                if audio_waveforms_concat_list
                else torch.tensor([])
            )
            audio_waveforms_start = (
                torch.cat(audio_waveforms_start_list, dim=0)
                if audio_waveforms_start_list
                else torch.tensor([])
            )
            audio_sample_rate = (
                torch.cat(audio_sample_rate_list, dim=0)
                if audio_sample_rate_list
                else torch.tensor([])
            )
            audio_speaker_indices = (
                torch.cat(audio_speaker_indices_list, dim=0)
                if audio_speaker_indices_list
                else torch.tensor([])
            )

        except RuntimeError as e:
            logger.error(f"Error during tensor concatenation: {str(e)}")
            logger.warning("Falling back to empty audio tensors")
            # Fall back to empty tensors
            audio_ids_concat = torch.tensor([[]])
            audio_ids_start = torch.tensor([])
            audio_waveforms_concat = torch.tensor([])
            audio_waveforms_start = torch.tensor([])
            audio_sample_rate = torch.tensor([])
            audio_speaker_indices = torch.tensor([])

        # Create the merged sample
        merged_sample = cls(
            input_ids=input_ids,
            label_ids=label_ids,
            audio_ids_concat=audio_ids_concat,
            audio_ids_start=audio_ids_start,
            audio_waveforms_concat=audio_waveforms_concat,
            audio_waveforms_start=audio_waveforms_start,
            audio_sample_rate=audio_sample_rate,
            audio_speaker_indices=audio_speaker_indices,
        )

        return merged_sample


@dataclass
class RankedChatMLDatasetSampleTuple:
    samples: list[ChatMLDatasetSample]
    scores: list[float]

    def max_score_sample(self) -> ChatMLDatasetSample:
        idx = self.scores.index(max(self.scores))
        self.samples[idx].reward = self.scores[idx]
        return self.samples[idx]

    def min_score_sample(self) -> ChatMLDatasetSample:
        idx = self.scores.index(min(self.scores))
        self.samples[idx].reward = self.scores[idx]
        return self.samples[idx]


@dataclass
class ChatMLDatasetStorageSample:
    input_tokens: torch.LongTensor
    label_tokens: torch.LongTensor
    audio_bytes_cache_dir_index: int
    audio_codes_cache_dir_index: int
    audio_bytes_indices: torch.LongTensor
    audio_codes_indices: torch.LongTensor
    speaker_indices: torch.LongTensor
    file_index: int
    original_sample_index: int


# TODO(sxjscience): We need to revist the logic about parsing speaker ids.
# Currently, we assume that the speaker id is stored at the "misc" field in ChatMLSample.
def prepare_chatml_sample(sample: Union[ChatMLSample, dict], tokenizer):
    """Preprocess the ChatML sample to get the tokens for the text part.

    Args:
        sample (ChatMLSample): The ChatML sample to preprocess.
        tokenizer: The tokenizer to use for encoding the text.

    """
    try:
        if not isinstance(sample, ChatMLSample):
            # Handle all fields that could be NaN
            if "speaker" in sample and pd.isna(sample["speaker"]):
                sample["speaker"] = None
            if "start_index" in sample and pd.isna(sample["start_index"]):
                sample["start_index"] = None
            if "content" in sample and pd.isna(sample["content"]):
                sample["content"] = ""

            # Convert any other potential NaN values in nested structures
            def convert_nan_to_none(obj):
                import numpy as np

                if isinstance(obj, (pd.Series, np.ndarray)):
                    return obj.tolist()
                elif pd.api.types.is_scalar(obj) and pd.isna(obj):
                    return None
                elif isinstance(obj, dict):
                    return {k: convert_nan_to_none(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):  # Fixed: Handle both list and tuple
                    return [convert_nan_to_none(item) for item in obj]
                return obj

            # Clean the sample data
            clean_sample = convert_nan_to_none(sample)

            val_keys = []
            for field in fields(ChatMLSample):
                if field.name in clean_sample:
                    val_keys.append(field.name)
            clean_sample = {k: clean_sample[k] for k in val_keys}

            try:
                sample = dacite.from_dict(
                    data_class=ChatMLSample,
                    data=clean_sample,
                    config=dacite.Config(strict=True, check_types=True),
                )
            except Exception as e:
                print(f"Failed to convert to ChatMLSample: {e}")
                print(f"Clean sample: {json.dumps(clean_sample, indent=2)}")
                return None, None, None, None

        input_tokens = []
        label_tokens = []
        audio_contents = []
        speaker_id = None
        if sample.speaker is not None:
            speaker_id = sample.speaker
        elif sample.misc is not None:
            if "speaker" in sample.misc:
                speaker_id = sample.misc["speaker"]

        total_m = len(sample.messages)
        for turn_id, message in enumerate(sample.messages):
            role = message.role
            recipient = message.recipient
            content = message.content
            content_l = []

            if isinstance(content, str):
                content_l.append(TextContent(text=content))
            elif isinstance(content, TextContent):
                content_l.append(content)
            elif isinstance(content, AudioContent):
                content_l.append(content)
            elif isinstance(content, list):
                for ele in content:
                    if isinstance(ele, str):
                        content_l.append(TextContent(text=ele))
                    else:
                        content_l.append(ele)
            if turn_id == 0:
                prefix = f"<|begin_of_text|><|start_header_id|>{role}<|end_header_id|>\n\n"
            else:
                prefix = f"<|start_header_id|>{role}<|end_header_id|>\n\n"
            eot_postfix = "<|eot_id|>"
            eom_postfix = "<|eom_id|>"

            prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
            input_tokens.extend(prefix_tokens)
            label_tokens.extend([-100 for _ in prefix_tokens])

            if recipient:
                assert role == "assistant", "Recipient is only available for assistant role."
                recipient_tokens = tokenizer.encode(
                    f"{recipient}<|recipient|>", add_special_tokens=False
                )
                input_tokens.extend(recipient_tokens)
                label_tokens.extend(recipient_tokens)

            for content in content_l:
                if content.type == "text":
                    text_tokens = tokenizer.encode(content.text, add_special_tokens=False)
                    input_tokens.extend(text_tokens)
                    if role == "assistant" and (
                        sample.start_index is None or turn_id >= sample.start_index
                    ):
                        label_tokens.extend(text_tokens)
                    else:
                        label_tokens.extend([-100 for _ in text_tokens])

                elif content.type == "audio":
                    # Generate the text-part of the audio tokens
                    audio_contents.append(content)
                    if role == "user" or role == "system":
                        # Add the text tokens
                        text_tokens = tokenizer.encode(
                            "<|audio_bos|><|AUDIO|><|audio_eos|>",
                            add_special_tokens=False,
                        )
                        input_tokens.extend(text_tokens)
                        label_tokens.extend([-100 for _ in text_tokens])
                    elif role == "assistant":
                        # Add the text tokens for audio-out part.
                        text_tokens = tokenizer.encode(
                            "<|audio_out_bos|><|AUDIO_OUT|><|audio_eos|>",
                            add_special_tokens=False,
                        )
                        input_tokens.extend(text_tokens)
                        if sample.start_index is None or turn_id >= sample.start_index:
                            label_tokens.extend(text_tokens)
                        else:
                            label_tokens.extend([-100 for _ in text_tokens])
            next_id = turn_id + 1
            if (
                role == "assistant"
                and next_id != total_m
                and sample.messages[next_id].role == "assistant"
            ):
                postfix_tokens = tokenizer.encode(eom_postfix, add_special_tokens=False)
                input_tokens.extend(postfix_tokens)
            else:
                postfix_tokens = tokenizer.encode(eot_postfix, add_special_tokens=False)
                input_tokens.extend(postfix_tokens)
            if role == "assistant" and (
                sample.start_index is None or turn_id >= sample.start_index
            ):
                label_tokens.extend(postfix_tokens)
            else:
                label_tokens.extend([-100 for _ in postfix_tokens])

        return input_tokens, label_tokens, audio_contents, speaker_id

    except Exception as e:
        print(f"Error in prepare_chatml_sample: {str(e)}")
        print(f"Sample data: {json.dumps(sample, indent=2)}")
        return None, None, None, None


def extract_generation_prompt_from_input_tokens(input_tokens, tokenizer):
    """Extract the generation prompt and reference answer from the input tokens.

    For example:

    Input Text = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n
    What words do you hear from the provided audio? Write it down for me.<|audio_bos|><|AUDIO|><|audio_eos|><|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>\n\nAt first they went by quick, too quick to even get.<|eot_id|>'

    -->

    Prompt = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n
    What words do you hear from the provided audio? Write it down for me.<|audio_bos|><|AUDIO|><|audio_eos|><|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>\n\n',
    Reference = 'At first they went by quick, too quick to even get.'

    Args:
        input_tokens: The input tokens.
        audio_contents: The audio contents.
        tokenizer: The tokenizer to use for decoding the text.

    Returns:
        prompt_tokens: The tokens for the prompt.
        reference_answer: The reference answer.
        num_audios_in_reference: The number of audios in the reference answer.

    """
    input_text = tokenizer.decode(input_tokens)
    generation_prefix = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    postfix = "<|eot_id|>"
    assert generation_prefix in input_text
    generation_prompt_end_loc = input_text.rfind(generation_prefix) + len(generation_prefix)
    generation_prompt = input_text[:generation_prompt_end_loc]
    reference_answer = input_text[
        generation_prompt_end_loc : input_text.find(postfix, generation_prompt_end_loc)
    ]
    num_audios_in_reference = reference_answer.count(AUDIO_IN_TOKEN) + reference_answer.count(
        AUDIO_OUT_TOKEN
    )
    return (
        tokenizer.encode(generation_prompt, add_special_tokens=False),
        reference_answer,
        num_audios_in_reference,
    )


def prepare_chatml_dataframe_single_process(df, tokenizer):
    """Prepare the ChatML DataFrame."""
    ret = []
    for _, row in df.iterrows():
        input_tokens, label_tokens, audio_contents, speaker_id = prepare_chatml_sample(
            row.to_dict(), tokenizer
        )
        ret.append((input_tokens, label_tokens, audio_contents, speaker_id))
    return ret


def prepare_chatml_dataframe(df, tokenizer, num_process=16):
    if num_process is None:
        return prepare_chatml_dataframe_single_process(df, tokenizer)
    else:
        num_process = max(min(len(df) // 1000, num_process), 1)
        workloads = np.array_split(df, num_process)
        with mp.Pool(num_process) as pool:
            ret = pool.starmap(
                prepare_chatml_dataframe_single_process,
                [(workload, tokenizer) for workload in workloads],
            )
    return sum(ret, [])


class DatasetInterface(ABC):
    @abstractmethod
    def __getitem__(self, idx) -> Union["ChatMLDatasetSample", "RankedChatMLDatasetSampleTuple"]:
        """Retrieve a dataset sample by index."""
        raise NotImplementedError


class IterableDatasetInterface(ABC):
    @abstractmethod
    def __iter__(self) -> Union["ChatMLDatasetSample", "RankedChatMLDatasetSampleTuple"]:
        """Retrieve a sample by iterating through the dataset."""
        raise NotImplementedError


@dataclass
class DatasetInfo:
    dataset_type: str
    group_type: str | None = None
    mask_text: bool | None = None  # Whether to mask the text tokens for pretraining samples.


def _ceil_to_nearest(n, round_to):
    return (n + round_to - 1) // round_to * round_to


def _ceil_to_next_power_of_two(self, x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


@dataclass
class HiggsAudioBatchInput:
    input_ids: torch.LongTensor  # shape (bsz, seq_len).
    attention_mask: torch.Tensor  # shape (bsz, seq_len).
    audio_features: torch.Tensor | None  # shape (num_audio_in, feature_dim, max_mel_seq_len).
    audio_feature_attention_mask: torch.Tensor | None  # shape (num_audio_in, max_mel_seq_len).
    audio_out_ids: torch.LongTensor | None  # shape (num_codebooks, audio_out_total_length)
    audio_out_ids_start: torch.LongTensor | None  # shape (num_audio_out,)
    # The audio_out_ids_start_group_loc has the same length as audio_out_ids_start. It is used to recover group location in a batch for an audio segment
    # Currently, we concatenante audio segments along dim 0 to handle variadic audio segment length. However, in the alignment stage, we need the location information
    # For example,
    #  audio_out_ids_start = [0, 2, 4, 8]; and the first two audio segments come from the same sample in a batch, and other two come from different samples.
    #  This is a batch of 3 samples, then we will have the group location as:
    #  audio_out_ids_start_group_loc = [0, 0, 1, 2]
    audio_out_ids_start_group_loc: (
        torch.LongTensor | None
    )  # shape (num_audio_out,), specify which a sample's group location in the batch
    audio_in_ids: torch.LongTensor | None  # shape (num_codebooks, audio_in_total_length)
    audio_in_ids_start: torch.LongTensor | None  # shape (num_audio_in,)
    label_ids: torch.LongTensor | None  # shape (bsz, seq_len)
    label_audio_ids: torch.LongTensor | None  # shape (num_codebooks, audio_out_total_length)
    reward: float | None = None


class HiggsAudioSampleCollator:
    """Sample collator for Higgs-Audio model.

    Args:
        whisper_processor (WhisperProcessor): The whisper processor.
        audio_in_token_id (int): The token id for audio-in.
        audio_out_token_id (int): The token id for audio-out.
        pad_token_id (int): The token id for padding.
        audio_stream_bos_id (int): The token id for audio-stream beginning of sentence.
        audio_stream_eos_id (int): The token id for audio-stream end of sentence.
        round_to (int): The round-to value.
        pad_left (bool): Whether to pad left.
        return_audio_in_tokens (bool): Whether to return audio-in tokens.
        use_delay_pattern (bool): Whether to use delay pattern.
        disable_audio_codes_transform (bool): Whether to add bos and eos tokens to audio codes.
        chunk_size_seconds (int): The chunk size in seconds.
        add_new_bos_eos_for_long_chunk (bool): Whether to add new bos and eos tokens for long chunks.
        mask_audio_out_token_label (bool): Whether to always mask the label associated with <|AUDIO_OUT|> token. Since we will always have `<|AUDIO_OUT|>` after `<|audio_bos|>`, we can safely mask <|AUDIO_OUT|>.

    """

    def __init__(
        self,
        whisper_processor: WhisperProcessor,
        audio_in_token_id,
        audio_out_token_id,
        pad_token_id,
        audio_stream_bos_id,
        audio_stream_eos_id,
        round_to=8,
        pad_left=False,
        encode_whisper_embed=True,
        return_audio_in_tokens=True,
        audio_num_codebooks=None,
        use_delay_pattern=False,
        disable_audio_codes_transform=False,
        chunk_size_seconds=30,  # Maximum duration for each chunk
        add_new_bos_eos_for_long_chunk=True,
        mask_audio_out_token_label=True,
    ):
        self.whisper_processor = whisper_processor
        self.round_to = round_to
        self.pad_left = pad_left
        self.audio_in_token_id = audio_in_token_id
        self.audio_out_token_id = audio_out_token_id
        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id
        self.pad_token_id = pad_token_id
        self.encode_whisper_embed = encode_whisper_embed
        self.return_audio_in_tokens = return_audio_in_tokens
        self.audio_num_codebooks = audio_num_codebooks
        self.use_delay_pattern = use_delay_pattern
        if encode_whisper_embed:
            self.chunk_size_seconds = chunk_size_seconds
            self.chunk_size_samples = int(
                chunk_size_seconds * whisper_processor.feature_extractor.sampling_rate
            )
        else:
            self.chunk_size_seconds = None
            self.chunk_size_samples = None
        self.disable_audio_codes_transform = disable_audio_codes_transform
        self.add_new_bos_eos_for_long_chunk = add_new_bos_eos_for_long_chunk
        self.mask_audio_out_token_label = mask_audio_out_token_label

    def _process_and_duplicate_audio_tokens(
        self,
        input_ids: torch.Tensor,
        audio_idx: int,
        wv: torch.Tensor,
        sr: int,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Process long audio and duplicate corresponding audio tokens.

        Args:
            input_ids: Input token ids
            audio_idx: Index of the audio token in the sequence
            wv: Audio waveform
            sr: Sample rate
            labels: Optional label ids to be duplicated alongside input ids

        Returns:
            Tuple of:
                - New input ids with duplicated audio tokens
                - New label ids (if labels were provided) or None
                - Number of chunks created
        """
        # Calculate number of chunks needed
        total_samples = len(wv)
        num_chunks = math.ceil(total_samples / self.chunk_size_samples)

        if num_chunks <= 1:
            return input_ids, labels, 1

        # Get the three tokens: <|audio_bos|><|AUDIO|><|audio_eos|>
        audio_token_seq = input_ids[audio_idx - 1 : audio_idx + 2]
        # Duplicate sequence for each chunk
        duplicated_sequence = audio_token_seq.repeat(num_chunks)

        # Create new input_ids with duplicated tokens
        new_input_ids = torch.cat(
            [input_ids[: audio_idx - 1], duplicated_sequence, input_ids[audio_idx + 2 :]]
        )

        # If labels are provided, duplicate them as well
        new_labels = None
        if labels is not None:
            label_seq = labels[audio_idx - 1 : audio_idx + 2]
            duplicated_labels = label_seq.repeat(num_chunks)
            new_labels = torch.cat(
                [labels[: audio_idx - 1], duplicated_labels, labels[audio_idx + 2 :]]
            )

        return new_input_ids, new_labels, num_chunks

    def __call__(self, batch: list[ChatMLDatasetSample]):
        """Collate the input data with support for long audio processing."""
        label_ids = None
        label_audio_ids = None
        if all([ele.label_ids is None for ele in batch]):
            return_labels = False
        else:
            return_labels = True

        if self.encode_whisper_embed:
            # Process each sample in the batch to handle long audio
            # TODO(?) The implementation here can be optimized.
            processed_batch = []
            for i in range(len(batch)):
                sample = batch[i]
                audio_in_mask = sample.input_ids == self.audio_in_token_id
                audio_in_indices = torch.where(audio_in_mask)[0]
                audio_out_mask = sample.input_ids == self.audio_out_token_id

                # Process each audio token and duplicate if needed
                modified_input_ids = sample.input_ids
                modified_labels = sample.label_ids if return_labels else None
                modified_waveforms_concat = []
                modified_waveforms_start = []
                modified_sample_rate = []
                offset = 0  # Track position changes from duplicating tokens
                curr_wv_offset = 0

                # Process input audio tokens
                for idx, audio_idx in enumerate(audio_in_indices):
                    # Get the audio for this token
                    wv, sr = sample.get_wv(idx)  # Use idx since we want the original audio index
                    if sr != self.whisper_processor.feature_extractor.sampling_rate:
                        resampled_wv = librosa.resample(
                            wv.cpu().numpy(),
                            orig_sr=sr,
                            target_sr=self.whisper_processor.feature_extractor.sampling_rate,
                        )
                    else:
                        resampled_wv = wv.cpu().numpy()
                    wv = torch.tensor(resampled_wv, device=wv.device)
                    sr = self.whisper_processor.feature_extractor.sampling_rate

                    # Process and duplicate tokens if necessary
                    token_pos = audio_idx + offset
                    modified_input_ids, modified_labels, num_chunks = (
                        self._process_and_duplicate_audio_tokens(
                            modified_input_ids, token_pos, wv, sr, modified_labels
                        )
                    )

                    # Update audio data
                    for chunk_idx in range(num_chunks):
                        chunk_start = chunk_idx * self.chunk_size_samples
                        chunk_end = min((chunk_idx + 1) * self.chunk_size_samples, len(wv))
                        chunk_wv = wv[chunk_start:chunk_end]
                        modified_waveforms_concat.append(chunk_wv)
                        modified_waveforms_start.append(curr_wv_offset)
                        curr_wv_offset += len(chunk_wv)
                        modified_sample_rate.append(sr)

                    # Update offset for next iteration
                    offset += (num_chunks - 1) * 3  # Each new chunk adds 3 more tokens

                # Create new sample with modified tokens and audio data
                processed_sample = ChatMLDatasetSample(
                    input_ids=modified_input_ids,
                    label_ids=modified_labels if return_labels else sample.label_ids,
                    audio_ids_concat=sample.audio_ids_concat,
                    audio_ids_start=sample.audio_ids_start,
                    audio_waveforms_concat=torch.cat(modified_waveforms_concat)
                    if modified_waveforms_concat
                    else sample.audio_waveforms_concat,
                    audio_waveforms_start=torch.tensor(modified_waveforms_start, dtype=torch.long)
                    if modified_waveforms_start
                    else sample.audio_waveforms_start,
                    audio_sample_rate=torch.tensor(modified_sample_rate)
                    if modified_sample_rate
                    else sample.audio_sample_rate,
                    audio_speaker_indices=torch.tensor([]),
                    # FIXME(sxjscience): The logic here is not correct for audio_label_ids_concat.
                    audio_label_ids_concat=sample.audio_label_ids_concat,
                )
                # audio_in_chunk_len = len(torch.where(modified_input_ids == self.audio_in_token_id)[0])
                # assert audio_in_chunk_len == processed_sample.num_audios(), f"Mismatch: audio_in_chunk_len={audio_in_chunk_len}, processed_sample.num_audios()={processed_sample.num_audios()}"
                processed_batch.append(processed_sample)
        else:
            processed_batch = batch

        # Get the max sequence length based on processed batch
        max_seq_length = _ceil_to_nearest(
            max([len(sample.input_ids) for sample in processed_batch]), self.round_to
        )

        # Get the ids for audio-in and audio-out for each batch
        audio_in_wv_l = []
        audio_in_ids_l = []
        audio_out_ids_l = []
        audio_out_ids_group_loc_l = []
        audio_in_label_ids_l = None
        audio_out_label_ids_l = None
        reward_l = []

        if return_labels:
            audio_out_no_train_flag = []  # Whether the audio-out data should be trained on or not.

        # Process the audio inputs and outputs
        for i in range(len(processed_batch)):
            audio_in_mask = processed_batch[i].input_ids == self.audio_in_token_id
            audio_out_mask = processed_batch[i].input_ids == self.audio_out_token_id
            audio_ids = torch.ones_like(processed_batch[i].input_ids)
            audio_ids[audio_in_mask ^ audio_out_mask] = (
                torch.cumsum(audio_ids[audio_in_mask ^ audio_out_mask], 0) - 1
            )
            audio_in_ids = audio_ids[audio_in_mask]
            audio_out_ids = audio_ids[audio_out_mask]

            if return_labels:
                audio_out_no_train_flag.append(processed_batch[i].label_ids[audio_out_mask] < 0)
                if self.mask_audio_out_token_label:
                    processed_batch[i].label_ids[audio_out_mask] = -100

            # Process audio inputs
            if self.return_audio_in_tokens:
                audio_in_ids_l.extend(
                    [
                        processed_batch[i].get_audio_codes(idx)[: self.audio_num_codebooks, :]
                        for idx in audio_in_ids
                    ]
                )
                if processed_batch[i].audio_label_ids_concat is not None:
                    if audio_in_label_ids_l is None:
                        audio_in_label_ids_l = []
                    audio_in_label_ids_l.extend(
                        [
                            processed_batch[i].get_audio_codes_labels(idx)[
                                : self.audio_num_codebooks, :
                            ]
                            for idx in audio_in_ids
                        ]
                    )

            audio_out_ids_l.extend(
                [
                    processed_batch[i].get_audio_codes(idx)[: self.audio_num_codebooks, :]
                    for idx in audio_out_ids
                ]
            )
            audio_out_ids_group_loc_l.append(i)
            if processed_batch[i].reward is not None:
                reward_l.append(processed_batch[i].reward)

            if processed_batch[i].audio_label_ids_concat is not None:
                if audio_out_label_ids_l is None:
                    audio_out_label_ids_l = []
                audio_out_label_ids_l.extend(
                    [
                        processed_batch[i].get_audio_codes_labels(idx)[
                            : self.audio_num_codebooks, :
                        ]
                        for idx in audio_out_ids
                    ]
                )

            if self.encode_whisper_embed:
                for idx in audio_in_ids:
                    wv, sr = processed_batch[i].get_wv(idx)
                    resampled_wv = wv.cpu().numpy()
                    # Split long audio into chunks
                    total_samples = len(resampled_wv)
                    for chunk_start in range(0, total_samples, self.chunk_size_samples):
                        chunk_end = min(chunk_start + self.chunk_size_samples, total_samples)
                        chunk = resampled_wv[chunk_start:chunk_end]
                        audio_in_wv_l.append(chunk)
            # assert len(audio_in_wv_l) == processed_batch[i].num_audios(), \
            #     f"Assertion failed: Mismatch in number of audios. " \
            #     f"Expected {processed_batch[i].num_audios()}, but got {len(audio_in_wv_l)} at index {i}."

        if return_labels:
            audio_out_no_train_flag = torch.cat(audio_out_no_train_flag, dim=0)

        # Process all audio features
        if len(audio_in_wv_l) > 0:
            feature_ret = self.whisper_processor.feature_extractor(
                audio_in_wv_l,
                sampling_rate=self.whisper_processor.feature_extractor.sampling_rate,
                return_attention_mask=True,
                padding="max_length",
            )
            audio_features = torch.from_numpy(feature_ret["input_features"])
            audio_feature_attention_mask = torch.from_numpy(feature_ret["attention_mask"])
        else:
            if self.encode_whisper_embed:
                audio_features = torch.zeros(
                    (
                        0,
                        self.whisper_processor.feature_extractor.feature_size,
                        self.whisper_processor.feature_extractor.nb_max_frames,
                    ),
                    dtype=torch.float32,
                )
                audio_feature_attention_mask = torch.zeros(
                    (0, self.whisper_processor.feature_extractor.nb_max_frames), dtype=torch.int32
                )
            else:
                audio_features = None
                audio_feature_attention_mask = None

        # Process audio input tokens
        if len(audio_in_ids_l) > 0:
            # Append audio-stream-bos and eos tokens
            new_audio_in_ids_l = []
            for ele in audio_in_ids_l:
                if self.disable_audio_codes_transform:
                    # Do not add audio-stream-bos or eos tokens.
                    # This may indicate that the sample comes from ConstantLengthDatasetWithBuffer.
                    audio_codes = ele
                else:
                    audio_codes = torch.cat(
                        [
                            torch.full(
                                (ele.shape[0], 1), self.audio_stream_bos_id, dtype=torch.long
                            ),
                            ele,
                            torch.full(
                                (ele.shape[0], 1), self.audio_stream_eos_id, dtype=torch.long
                            ),
                        ],
                        dim=1,
                    )
                    if self.use_delay_pattern:
                        audio_codes = build_delay_pattern_mask(
                            audio_codes.unsqueeze(0),
                            bos_token_id=self.audio_stream_bos_id,
                            pad_token_id=self.audio_stream_eos_id,
                        )[0].squeeze(0)
                new_audio_in_ids_l.append(audio_codes)
            audio_in_ids = torch.cat(new_audio_in_ids_l, dim=1).long()
            audio_in_ids_start = torch.cumsum(
                torch.tensor(
                    [0] + [audio_codes.shape[1] for audio_codes in new_audio_in_ids_l[:-1]]
                ),
                dim=0,
            )
        else:
            audio_in_ids = torch.zeros((0, 0), dtype=torch.long)
            audio_in_ids_start = torch.zeros(0, dtype=torch.long)

        # Process audio output tokens
        audio_out_ids_start_group_loc = None
        if len(audio_out_ids_l) > 0:
            new_audio_out_ids_l = []
            label_audio_ids_l = []
            for idx, ele in enumerate(audio_out_ids_l):
                if self.disable_audio_codes_transform:
                    # Do not add audio-stream-bos or eos tokens.
                    # This may indicate that the sample comes from ConstantLengthDatasetWithBuffer.
                    audio_codes = ele
                    if return_labels:
                        label_audio_ids = audio_out_label_ids_l[idx]
                else:
                    audio_codes = torch.cat(
                        [
                            torch.full(
                                (ele.shape[0], 1), self.audio_stream_bos_id, dtype=torch.long
                            ),
                            ele,
                            torch.full(
                                (ele.shape[0], 1), self.audio_stream_eos_id, dtype=torch.long
                            ),
                        ],
                        dim=1,
                    )
                    if return_labels:
                        label_audio_ids = torch.cat(
                            [
                                torch.full((ele.shape[0], 1), -100, dtype=torch.long),
                                ele,
                                torch.full(
                                    (ele.shape[0], 1), self.audio_stream_eos_id, dtype=torch.long
                                ),
                            ],
                            dim=1,
                        )
                    if self.use_delay_pattern:
                        audio_codes = build_delay_pattern_mask(
                            audio_codes.unsqueeze(0),
                            bos_token_id=self.audio_stream_bos_id,
                            pad_token_id=self.audio_stream_eos_id,
                        )[0].squeeze(0)
                        if return_labels:
                            label_audio_ids = build_delay_pattern_mask(
                                label_audio_ids.unsqueeze(0),
                                bos_token_id=-100,
                                pad_token_id=-100,
                            )[0].squeeze(0)
                new_audio_out_ids_l.append(audio_codes)

                if return_labels:
                    if audio_out_no_train_flag[idx]:
                        label_audio_ids[:] = -100
                    label_audio_ids_l.append(label_audio_ids)

            audio_out_ids = torch.cat(new_audio_out_ids_l, dim=1).long()
            if return_labels:
                label_audio_ids = torch.cat(label_audio_ids_l, dim=1).long()
            audio_out_ids_start = torch.cumsum(
                torch.tensor(
                    [0] + [audio_codes.shape[1] for audio_codes in new_audio_out_ids_l[:-1]]
                ),
                dim=0,
            )
            audio_out_ids_start_group_loc = torch.tensor(
                audio_out_ids_group_loc_l, dtype=torch.long
            )
        else:
            audio_out_ids = torch.zeros((0, 0), dtype=torch.long)
            audio_out_ids_start = torch.zeros(0, dtype=torch.long)
            if return_labels:
                label_audio_ids = torch.zeros((0, 0), dtype=torch.long)

        reward = torch.tensor(reward_l, dtype=torch.float32)

        # Handle padding for input ids and attention mask
        if self.pad_left:
            input_ids = torch.stack(
                [
                    F.pad(
                        ele.input_ids,
                        (max_seq_length - len(ele.input_ids), 0),
                        value=self.pad_token_id,
                    )
                    for ele in processed_batch
                ]
            )
            if return_labels:
                label_ids = torch.stack(
                    [
                        F.pad(ele.label_ids, (max_seq_length - len(ele.label_ids), 0), value=-100)
                        for ele in processed_batch
                    ]
                )
            attention_mask = torch.stack(
                [
                    F.pad(
                        torch.ones_like(ele.input_ids),
                        (max_seq_length - len(ele.input_ids), 0),
                        value=0,
                    )
                    for ele in processed_batch
                ]
            )
        else:
            input_ids = torch.stack(
                [
                    F.pad(
                        ele.input_ids,
                        (0, max_seq_length - len(ele.input_ids)),
                        value=self.pad_token_id,
                    )
                    for ele in processed_batch
                ]
            )
            if return_labels:
                label_ids = torch.stack(
                    [
                        F.pad(ele.label_ids, (0, max_seq_length - len(ele.label_ids)), value=-100)
                        for ele in processed_batch
                    ]
                )
            attention_mask = torch.stack(
                [
                    F.pad(
                        torch.ones_like(ele.input_ids),
                        (0, max_seq_length - len(ele.input_ids)),
                        value=0,
                    )
                    for ele in processed_batch
                ]
            )

        if not self.return_audio_in_tokens:
            audio_in_ids = None
            audio_in_ids_start = None

        # Apply audio_num_codebooks limit if specified
        if self.audio_num_codebooks is not None:
            if audio_in_ids is not None:
                audio_in_ids = audio_in_ids[: self.audio_num_codebooks]
            if audio_out_ids is not None:
                audio_out_ids = audio_out_ids[: self.audio_num_codebooks]
            if label_audio_ids is not None:
                label_audio_ids = label_audio_ids[: self.audio_num_codebooks]

        return HiggsAudioBatchInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_features=audio_features,
            audio_feature_attention_mask=audio_feature_attention_mask,
            audio_out_ids=audio_out_ids,
            audio_out_ids_start=audio_out_ids_start,
            audio_out_ids_start_group_loc=audio_out_ids_start_group_loc,
            audio_in_ids=audio_in_ids,
            audio_in_ids_start=audio_in_ids_start,
            label_ids=label_ids,
            label_audio_ids=label_audio_ids,
            reward=reward,
        )


# Based on code from: https://github.com/zhenye234/xcodec
# Licensed under MIT License
# Modifications by BosonAI


class EncodedResult:
    def __init__(self, audio_codes):
        self.audio_codes = audio_codes


class HiggsAudioFeatureExtractor(nn.Module):
    def __init__(self, sampling_rate=16000):
        super().__init__()
        self.sampling_rate = sampling_rate

    def forward(self, raw_audio, sampling_rate=16000, return_tensors="pt"):
        # Convert from librosa to torch
        audio_signal = torch.tensor(raw_audio)
        audio_signal = audio_signal.unsqueeze(0)
        if len(audio_signal.shape) < 3:
            audio_signal = audio_signal.unsqueeze(0)
        return {"input_values": audio_signal}


class HiggsAudioTokenizer(nn.Module):
    def __init__(
        self,
        n_filters: int = 32,
        D: int = 128,
        target_bandwidths: Sequence[Union[int, float]] = [1, 1.5, 2, 4, 6],
        ratios: Sequence[int] = [8, 5, 4, 2],  # downsampling by 320
        sample_rate: int = 16000,
        bins: int = 1024,
        n_q: int = 8,
        codebook_dim: int = None,
        normalize: bool = False,
        causal: bool = False,
        semantic_techer: str = "hubert_base_general",
        last_layer_semantic: bool = True,
        merge_mode: str = "concat",
        downsample_mode: str = "step_down",
        semantic_mode: str = "classic",
        vq_scale: int = 1,
        semantic_sample_rate: int = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.hop_length = np.prod(ratios)
        self.semantic_techer = semantic_techer

        self.frame_rate = math.ceil(sample_rate / np.prod(ratios))  # 50 Hz

        self.target_bandwidths = target_bandwidths
        self.n_q = n_q
        self.sample_rate = sample_rate
        self.encoder = dac2.Encoder(64, ratios, D)

        self.decoder_2 = dac2.Decoder(D, 1024, ratios)
        self.last_layer_semantic = last_layer_semantic
        self.device = device
        if semantic_techer == "hubert_base":
            self.semantic_model = AutoModel.from_pretrained("facebook/hubert-base-ls960")
            self.semantic_sample_rate = 16000
            self.semantic_dim = 768
            self.encoder_semantic_dim = 768

        elif semantic_techer == "wavlm_base_plus":
            self.semantic_model = AutoModel.from_pretrained("microsoft/wavlm-base-plus")
            self.semantic_sample_rate = 16000
            self.semantic_dim = 768
            self.encoder_semantic_dim = 768

        elif semantic_techer == "hubert_base_general":
            self.semantic_model = AutoModel.from_pretrained(
                "bosonai/hubert_base", trust_remote_code=True
            )
            self.semantic_sample_rate = 16000
            self.semantic_dim = 768
            self.encoder_semantic_dim = 768

        # Overwrite semantic model sr to ensure semantic_downsample_factor is an integer
        if semantic_sample_rate is not None:
            self.semantic_sample_rate = semantic_sample_rate

        self.semantic_model.eval()

        # make the semantic model parameters do not need gradient
        for param in self.semantic_model.parameters():
            param.requires_grad = False

        self.semantic_downsample_factor = int(
            self.hop_length / (self.sample_rate / self.semantic_sample_rate) / 320
        )

        self.quantizer_dim = int((D + self.encoder_semantic_dim) // vq_scale)
        self.encoder_semantic = Encoder(
            input_channels=self.semantic_dim, encode_channels=self.encoder_semantic_dim
        )
        self.decoder_semantic = Decoder(
            code_dim=self.encoder_semantic_dim,
            output_channels=self.semantic_dim,
            decode_channels=self.semantic_dim,
        )

        # out_D=D+768
        if isinstance(bins, int):  # RVQ
            self.quantizer = ResidualVectorQuantizer(
                dimension=self.quantizer_dim, codebook_dim=codebook_dim, n_q=n_q, bins=bins
            )
            self.quantizer_type = "RVQ"
        else:  # RFSQ
            self.quantizer = ResidualFSQ(dim=self.quantizer_dim, levels=bins, num_quantizers=n_q)
            self.quantizer_type = "RFSQ"

        self.fc_prior = nn.Linear(D + self.encoder_semantic_dim, self.quantizer_dim)
        self.fc_post1 = nn.Linear(self.quantizer_dim, self.encoder_semantic_dim)
        self.fc_post2 = nn.Linear(self.quantizer_dim, D)

        self.downsample_mode = downsample_mode
        if downsample_mode == "avg":
            self.semantic_pooling = nn.AvgPool1d(
                kernel_size=self.semantic_downsample_factor, stride=self.semantic_downsample_factor
            )

        self.audio_tokenizer_feature_extractor = HiggsAudioFeatureExtractor(
            sampling_rate=self.sample_rate
        )

    @property
    def tps(self):
        return self.frame_rate

    @property
    def sampling_rate(self):
        return self.sample_rate

    @property
    def num_codebooks(self):
        return self.n_q

    @property
    def codebook_size(self):
        return self.quantizer_dim

    def get_last_layer(self):
        return self.decoder.layers[-1].weight

    def calculate_rec_loss(self, rec, target):
        target = target / target.norm(dim=-1, keepdim=True)
        rec = rec / rec.norm(dim=-1, keepdim=True)
        rec_loss = (1 - (target * rec).sum(-1)).mean()

        return rec_loss

    @torch.no_grad()
    def get_regress_target(self, x):
        x = torchaudio.functional.resample(x, self.sample_rate, self.semantic_sample_rate)

        if (
            self.semantic_techer == "hubert_base"
            or self.semantic_techer == "hubert_base_general"
            or self.semantic_techer == "wavlm_base_plus"
        ):
            x = x[:, 0, :]
            x = F.pad(x, (160, 160))
            target = self.semantic_model(x, output_hidden_states=True).hidden_states
            target = torch.stack(
                target, dim=1
            )  # .transpose(-1, -2)#.flatten(start_dim=1, end_dim=2)

            # average for all layers
            target = target.mean(1)
            # target = target[9]
            # if self.hop_length > 320:
            #     target = self.semantic_pooling(target.transpose(1, 2)).transpose(1, 2)

        elif self.semantic_techer == "w2v_bert2":
            target = self.semantic_model(x)

        elif self.semantic_techer.startswith("whisper"):
            if self.last_layer_semantic:
                target = self.semantic_model(x, avg_layers=False)
            else:
                target = self.semantic_model(x, avg_layers=True)

        elif self.semantic_techer.startswith("mert_music"):
            if self.last_layer_semantic:
                target = self.semantic_model(x, avg_layers=False)
            else:
                target = self.semantic_model(x, avg_layers=True)

        elif self.semantic_techer.startswith("qwen_audio_omni"):
            target = self.semantic_model(x)

        if self.downsample_mode == "step_down":
            if self.semantic_downsample_factor > 1:
                target = target[:, :: self.semantic_downsample_factor, :]

        elif self.downsample_mode == "avg":
            target = self.semantic_pooling(target.transpose(1, 2)).transpose(1, 2)
        return target

    def forward(self, x: torch.Tensor, bw: int):
        e_semantic_input = self.get_regress_target(x).detach()

        e_semantic = self.encoder_semantic(e_semantic_input.transpose(1, 2))
        e_acoustic = self.encoder(x)

        e = torch.cat([e_acoustic, e_semantic], dim=1)

        e = self.fc_prior(e.transpose(1, 2))

        if self.quantizer_type == "RVQ":
            e = e.transpose(1, 2)
            quantized, codes, bandwidth, commit_loss = self.quantizer(e, self.frame_rate, bw)
            quantized = quantized.transpose(1, 2)
        else:
            quantized, codes = self.quantizer(e)
            commit_loss = torch.tensor(0.0)

        quantized_semantic = self.fc_post1(quantized).transpose(1, 2)
        quantized_acoustic = self.fc_post2(quantized).transpose(1, 2)

        o = self.decoder_2(quantized_acoustic)

        o_semantic = self.decoder_semantic(quantized_semantic)
        semantic_recon_loss = F.mse_loss(e_semantic_input.transpose(1, 2).detach(), o_semantic)

        return o, commit_loss, semantic_recon_loss, None

    def encode(self, audio_path_or_wv, sr=None, loudness_normalize=False, loudness_threshold=-23.0):
        if isinstance(audio_path_or_wv, str):
            wv, sr = librosa.load(audio_path_or_wv, mono=True, sr=None)
        else:
            wv = audio_path_or_wv
            assert sr is not None
        if loudness_normalize:
            import pyloudnorm as pyln

            meter = pyln.Meter(sr)
            l = meter.integrated_loudness(wv)
            wv = pyln.normalize.loudness(wv, l, loudness_threshold)
        if sr != self.sampling_rate:
            wv = librosa.resample(wv, orig_sr=sr, target_sr=self.sampling_rate)
        if self.audio_tokenizer_feature_extractor is not None:
            inputs = self.audio_tokenizer_feature_extractor(
                raw_audio=wv,
                sampling_rate=self.audio_tokenizer_feature_extractor.sampling_rate,
                return_tensors="pt",
            )
            input_values = inputs["input_values"].to(self.device)
        else:
            input_values = torch.from_numpy(wv).float().unsqueeze(0)
        with torch.no_grad():
            encoder_outputs = self._xcodec_encode(input_values)
            vq_code = encoder_outputs.audio_codes[0]
        return vq_code

    def _xcodec_encode(self, x: torch.Tensor, target_bw: int | None = None) -> torch.Tensor:
        bw = target_bw

        e_semantic_input = self.get_regress_target(x).detach()

        e_semantic = self.encoder_semantic(e_semantic_input.transpose(1, 2))
        e_acoustic = self.encoder(x)

        if e_acoustic.shape[2] != e_semantic.shape[2]:
            pad_size = 160 * self.semantic_downsample_factor
            e_acoustic = self.encoder(F.pad(x[:, 0, :], (pad_size, pad_size)).unsqueeze(0))

        if e_acoustic.shape[2] != e_semantic.shape[2]:
            if e_acoustic.shape[2] > e_semantic.shape[2]:
                e_acoustic = e_acoustic[:, :, : e_semantic.shape[2]]
            else:
                e_semantic = e_semantic[:, :, : e_acoustic.shape[2]]

        e = torch.cat([e_acoustic, e_semantic], dim=1)

        e = self.fc_prior(e.transpose(1, 2))

        if self.quantizer_type == "RVQ":
            e = e.transpose(1, 2)
            quantized, codes, bandwidth, commit_loss = self.quantizer(e, self.frame_rate, bw)
            codes = codes.permute(1, 0, 2)
        else:
            quantized, codes = self.quantizer(e)
            codes = codes.permute(0, 2, 1)

        # return codes
        return EncodedResult(codes)

    def decode(self, vq_code: torch.Tensor) -> torch.Tensor:
        vq_code = vq_code.to(self.device)

        if self.quantizer_type == "RVQ":
            vq_code = vq_code.permute(1, 0, 2)
            quantized = self.quantizer.decode(vq_code)
            quantized = quantized.transpose(1, 2)
        else:
            vq_code = vq_code.permute(0, 2, 1)
            quantized = self.quantizer.get_output_from_indices(vq_code)
        quantized_acoustic = self.fc_post2(quantized).transpose(1, 2)

        o = self.decoder_2(quantized_acoustic)
        return o.detach().cpu().numpy()


def load_higgs_audio_tokenizer(tokenizer_name_or_path, device="cuda"):
    is_local = os.path.exists(tokenizer_name_or_path)
    if not is_local:
        tokenizer_path = snapshot_download(tokenizer_name_or_path)
    else:
        tokenizer_path = tokenizer_name_or_path
    config_path = os.path.join(tokenizer_path, "config.json")
    model_path = os.path.join(tokenizer_path, "model.pth")
    config = json.load(open(config_path))
    model = HiggsAudioTokenizer(
        **config,
        device=device,
    )
    parameter_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(parameter_dict, strict=False)
    model.to(device)
    model.eval()
    return model


@dataclass
class HiggsAudioStreamerDelta:
    """Represents a chunk of generated content, either text or audio tokens."""

    text: str | None = None
    text_tokens: torch.Tensor | None = None
    audio_tokens: torch.Tensor | None = None
    finish_reason: str | None = None


class AsyncHiggsAudioStreamer(BaseStreamer):
    """Async streamer that handles both text and audio token generation from Higgs-Audio model.
    Stores chunks in a queue to be consumed by downstream applications.

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenizer used to decode text tokens.
        skip_prompt (`bool`, *optional*, defaults to `False`):
            Whether to skip the prompt tokens in generation.
        timeout (`float`, *optional*):
            The timeout for the queue. If `None`, the queue will block indefinitely.
        decode_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the tokenizer's `decode` method.

    Examples:
        ```python
        >>> from transformers import AutoTokenizer
        >>> from threading import Thread
        >>> import asyncio

        >>> tokenizer = AutoTokenizer.from_pretrained("path/to/higgs/tokenizer")
        >>> model = HiggsAudioModelTransformers.from_pretrained("path/to/higgs/model")
        >>> inputs = tokenizer(["Generate some text and audio:"], return_tensors="pt")

        >>> async def main():
        ...     streamer = AsyncHiggsAudioStreamer(tokenizer)
        ...     generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=20)
        ...     thread = Thread(target=model.generate, kwargs=generation_kwargs)
        ...     thread.start()
        ...
        ...     async for delta in streamer:
        ...         if delta.text is not None:
        ...             print("Text:", delta.text)
        ...         if delta.audio_tokens is not None:
        ...             print("Audio tokens shape:", delta.audio_tokens.shape)
        >>> asyncio.run(main())
        ```
    """

    def __init__(
        self,
        tokenizer: "AutoTokenizer",
        skip_prompt: bool = False,
        timeout: float | None = None,
        audio_num_codebooks: int = 1,
        **decode_kwargs,
    ):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.timeout = timeout
        self.decode_kwargs = decode_kwargs
        self.audio_num_codebooks = audio_num_codebooks
        # Queue to store generated chunks
        self.queue = asyncio.Queue()
        self.stop_signal = None

        # Get running event loop
        self.loop = asyncio.get_running_loop()
        self.has_asyncio_timeout = hasattr(asyncio, "timeout")

    def put(self, value: torch.Tensor):
        delta = HiggsAudioStreamerDelta(audio_tokens=value)
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.queue.put_nowait, delta)

    def end(self):
        """Flushes any remaining text tokens and signals the end of generation."""
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.queue.put_nowait, self.stop_signal)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            if self.has_asyncio_timeout:
                async with asyncio.timeout(self.timeout):
                    value = await self.queue.get()
            else:
                value = await asyncio.wait_for(self.queue.get(), timeout=self.timeout)
        except asyncio.TimeoutError:
            raise TimeoutError()
        else:
            if value == self.stop_signal:
                raise StopAsyncIteration()
            else:
                return value


class AsyncStoppingCriteria(StoppingCriteria):
    """Stopping criteria that checks for stop signal from a threading event.

    Args:
        stop_signal (threading.Event): Event that will receive stop signals
    """

    def __init__(self, stop_signal: threading.Event):
        self.stop_signal = stop_signal

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if self.stop_signal.is_set():
            logger.info("Stop signal received. Can be caused by client disconnection.")
            return True
        return False


@dataclass
class HiggsAudioResponse:
    audio: np.ndarray | None = None
    generated_audio_tokens: np.ndarray | None = None
    sampling_rate: int | None = None
    generated_text: str = ""
    generated_text_tokens: np.ndarray | None = None
    usage: dict | None = None


class HiggsAudioServeEngine:
    def __init__(
        self,
        model_name_or_path: str,
        audio_tokenizer_name_or_path: str,
        tokenizer_name_or_path: str | None = None,
        device: str = "cuda",
        torch_dtype: Union[torch.dtype, str] = "auto",
        kv_cache_lengths: list[int] = [1024, 4096, 8192],  # Multiple KV cache sizes
    ):
        """Initialize the HiggsAudioServeEngine, a serving wrapper for the HiggsAudioModelTransformers.
        The model, tokenizer, and audio tokenizer will be downloaded from the Hugging Face Hub if they are not local.

        Args:
            model_name_or_path (str):
                The name or path of the model to load.
            audio_tokenizer_name_or_path (str):
                The name or path of the audio tokenizer to load.
            tokenizer_name_or_path (str):
                The name or path of the tokenizer to load.
            device (str):
                The device to use for the model.
            kv_cache_lengths (List[int]):
                The lengths of the KV caches to use for the model. Used for cuda graph capture when device is cuda.
            torch_dtype (Union[torch.dtype, str]):
                The dtype to use for the model.
        """
        self.device = device
        self.model_name_or_path = model_name_or_path
        self.torch_dtype = torch_dtype

        # Initialize model and tokenizer
        self.model = HiggsAudioModel.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype
        ).to(device)
        logger.info(f"Loaded model from {model_name_or_path}, dtype: {self.model.dtype}")

        if tokenizer_name_or_path is None:
            tokenizer_name_or_path = model_name_or_path
        logger.info(f"Loading tokenizer from {tokenizer_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        logger.info("Initializing Higgs Audio Tokenizer")
        self.audio_tokenizer = load_higgs_audio_tokenizer(
            audio_tokenizer_name_or_path, device=device
        )

        self.audio_num_codebooks = self.model.config.audio_num_codebooks
        self.audio_codebook_size = self.model.config.audio_codebook_size
        self.audio_tokenizer_tps = self.audio_tokenizer.tps
        self.samples_per_token = int(self.audio_tokenizer.sampling_rate // self.audio_tokenizer_tps)
        self.hamming_window_len = 2 * self.audio_num_codebooks * self.samples_per_token
        # Set the audio special tokens
        self.model.set_audio_special_tokens(self.tokenizer)

        # Prepare KV caches for different lengths
        cache_config = deepcopy(self.model.config.text_config)
        cache_config.num_hidden_layers = self.model.config.text_config.num_hidden_layers
        if self.model.config.audio_dual_ffn_layers:
            cache_config.num_hidden_layers += len(self.model.config.audio_dual_ffn_layers)
        # A list of KV caches for different lengths
        self.kv_caches = {
            length: StaticCache(
                config=cache_config,
                max_batch_size=1,
                max_cache_len=length,
                device=self.model.device,
                dtype=self.model.dtype,
            )
            for length in sorted(kv_cache_lengths)
        }

        if self.model.config.encode_whisper_embed:
            logger.info("Loading whisper processor")
            whisper_processor = AutoProcessor.from_pretrained(
                "openai/whisper-large-v3-turbo",
                trust_remote=True,
                device=self.device,
            )
        else:
            whisper_processor = None

        # Reuse collator to prepare inference samples
        self.collator = HiggsAudioSampleCollator(
            whisper_processor=whisper_processor,
            encode_whisper_embed=self.model.config.encode_whisper_embed,
            audio_in_token_id=self.model.config.audio_in_token_idx,
            audio_out_token_id=self.model.config.audio_out_token_idx,
            audio_stream_bos_id=self.model.config.audio_stream_bos_id,
            audio_stream_eos_id=self.model.config.audio_stream_eos_id,
            pad_token_id=self.model.config.pad_token_id,
            return_audio_in_tokens=False,
            use_delay_pattern=self.model.config.use_delay_pattern,
            audio_num_codebooks=self.model.config.audio_num_codebooks,
            round_to=1,
        )

        # Capture CUDA graphs for each KV cache length
        if device == "cuda":
            logger.info("Capturing CUDA graphs for each KV cache length")
            self.model.capture_model(self.kv_caches.values())

    def _prepare_inputs(self, chat_ml_sample: ChatMLSample, force_audio_gen: bool = False):
        input_tokens, _, audio_contents, _ = prepare_chatml_sample(
            chat_ml_sample,
            self.tokenizer,
        )

        postfix = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        if force_audio_gen:
            postfix += "<|audio_out_bos|>"
        postfix = self.tokenizer.encode(postfix, add_special_tokens=False)
        input_tokens.extend(postfix)

        # Configure the audio inputs
        audio_ids_l = []
        for audio_content in audio_contents:
            if audio_content.audio_url not in ["placeholder", ""]:
                raw_audio, _ = librosa.load(
                    audio_content.audio_url, sr=self.audio_tokenizer.sampling_rate
                )
            elif audio_content.raw_audio is not None:
                raw_audio, _ = librosa.load(
                    BytesIO(base64.b64decode(audio_content.raw_audio)),
                    sr=self.audio_tokenizer.sampling_rate,
                )
            else:
                raw_audio = None

            if raw_audio is not None:
                audio_ids = self.audio_tokenizer.encode(
                    raw_audio, self.audio_tokenizer.sampling_rate
                )
                audio_ids_l.append(audio_ids.squeeze(0).cpu())

        if len(audio_ids_l) > 0:
            audio_ids_start = torch.tensor(
                np.cumsum(np.array([0] + [audio_ids.shape[1] for audio_ids in audio_ids_l])),
                dtype=torch.long,
                device=self.device,
            )[0:-1]
            audio_ids_concat = torch.cat(audio_ids_l, dim=1)
        else:
            audio_ids_start = None
            audio_ids_concat = None

        sample = ChatMLDatasetSample(
            input_ids=torch.LongTensor(input_tokens),
            label_ids=None,
            audio_ids_concat=audio_ids_concat,
            audio_ids_start=audio_ids_start,
            audio_waveforms_concat=None,
            audio_waveforms_start=None,
            audio_sample_rate=None,
            audio_speaker_indices=None,
        )
        data = self.collator([sample])
        inputs = asdict(data)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.model.device)

        return inputs

    def _prepare_kv_caches(self):
        for kv_cache in self.kv_caches.values():
            kv_cache.reset()

    def generate(
        self,
        chat_ml_sample: ChatMLSample,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_k: int | None = None,
        top_p: float = 0.95,
        stop_strings: list[str] | None = None,
        force_audio_gen: bool = False,
        ras_win_len: int | None = 7,
        ras_win_max_num_repeat: int = 2,
        seed: int | None = None,
    ):
        """Generate audio from a chatml sample.

        Args:
            chat_ml_sample: A chatml sample.
            max_new_tokens: The maximum number of new tokens to generate.
            temperature: The temperature to use for the generation.
            top_p: The top p to use for the generation.
            stop_strings: A list of strings to stop the generation.
            force_audio_gen: Whether to force audio generation. This ensures the model generates audio tokens rather than text tokens.
            ras_win_len: The length of the RAS window. We use 7 by default. You can disable it by setting it to None or <=0.
            ras_win_max_num_repeat: The maximum number of times to repeat the RAS window.

        Returns:
            A dictionary with the following keys:
                audio: The generated audio.
                sampling_rate: The sampling rate of the generated audio.
        """
        # Default stop strings
        if stop_strings is None:
            stop_strings = ["<|end_of_text|>", "<|eot_id|>"]
        if ras_win_len is not None and ras_win_len <= 0:
            ras_win_len = None

        with torch.no_grad():
            inputs = self._prepare_inputs(chat_ml_sample, force_audio_gen=force_audio_gen)
            prompt_token_ids = inputs["input_ids"][0].cpu().numpy()

            self._prepare_kv_caches()

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stop_strings=stop_strings,
                tokenizer=self.tokenizer,
                do_sample=False if temperature == 0.0 else True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                past_key_values_buckets=self.kv_caches,
                ras_win_len=ras_win_len,
                ras_win_max_num_repeat=ras_win_max_num_repeat,
                seed=seed,
            )

            if len(outputs[1]) > 0:
                wv_list = []
                for output_audio in outputs[1]:
                    vq_code = revert_delay_pattern(output_audio).clip(
                        0, self.audio_codebook_size - 1
                    )[:, 1:-1]
                    wv_numpy = self.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]
                    wv_list.append(wv_numpy)
                wv_numpy = np.concatenate(wv_list)
            else:
                wv_numpy = None

            # We only support one request at a time now
            generated_text_tokens = outputs[0][0].cpu().numpy()[len(prompt_token_ids) :]
            generated_text = self.tokenizer.decode(generated_text_tokens)
            generated_audio_tokens = outputs[1][0].cpu().numpy()
            return HiggsAudioResponse(
                audio=wv_numpy,
                generated_audio_tokens=generated_audio_tokens,
                sampling_rate=self.audio_tokenizer.sampling_rate,
                generated_text=generated_text,
                generated_text_tokens=generated_text_tokens,
                usage={
                    "prompt_tokens": prompt_token_ids.shape[0],
                    "completion_tokens": generated_text_tokens.shape[0]
                    + generated_audio_tokens.shape[1],
                    "total_tokens": (
                        prompt_token_ids.shape[0]
                        + generated_text_tokens.shape[0]
                        + generated_audio_tokens.shape[1]
                    ),
                    "cached_tokens": 0,
                },
            )

    async def generate_delta_stream(
        self,
        chat_ml_sample: ChatMLSample,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_k: int | None = None,
        top_p: float = 0.95,
        stop_strings: list[str] | None = None,
        force_audio_gen: bool = False,
        ras_win_len: int | None = 7,
        ras_win_max_num_repeat: int = 2,
        seed: int | None = None,
    ):
        """Generate audio from a chatml sample.

        Args:
            chat_ml_sample: A chatml sample.
            max_new_tokens: The maximum number of new tokens to generate.
            temperature: The temperature to use for the generation.
            top_p: The top p to use for the generation.
            stop_strings: A list of strings to stop the generation.
            force_audio_gen: Whether to force audio generation. This ensures the model generates audio tokens rather than text tokens.
            ras_win_len: The length of the RAS window. We use 7 by default. You can disable it by setting it to None or <=0.
            ras_win_max_num_repeat: The maximum number of times to repeat the RAS window.

        Returns:
             Delta AsyncGenerator
        """
        # Default stop strings
        if stop_strings is None:
            stop_strings = ["<|end_of_text|>", "<|eot_id|>"]
        if ras_win_len is not None and ras_win_len <= 0:
            ras_win_len = None

        with torch.no_grad():
            inputs = self._prepare_inputs(chat_ml_sample, force_audio_gen=force_audio_gen)

            self._prepare_kv_caches()

            streamer = AsyncHiggsAudioStreamer(
                self.tokenizer,
                audio_num_codebooks=self.model.config.audio_num_codebooks,
                skip_prompt=True,
            )
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stop_strings=stop_strings,
                tokenizer=self.tokenizer,
                do_sample=False if temperature == 0.0 else True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                past_key_values_buckets=self.kv_caches,
                ras_win_len=ras_win_len,
                ras_win_max_num_repeat=ras_win_max_num_repeat,
                seed=seed,
                streamer=streamer,
            )
            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            async for delta in streamer:
                yield delta
