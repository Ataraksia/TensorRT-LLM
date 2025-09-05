import functools
import inspect
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from boson_multimodal.audio_processing import HiggsAudioTokenizer
from config import HiggsAudioConfig
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from torch import nn
from transformers import BaseStreamer, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import (
    GenerationConfig,
    GenerationMixin,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from transformers.generation.utils import GenerateNonBeamOutput
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.utils import ModelOutput

from tensorrt_llm._torch.models.modeling_llama import LlamaAttention


def _ceil_to_nearest(n, round_to):
    return (n + round_to - 1) // round_to * round_to


def build_delay_pattern_mask(
    input_ids: torch.LongTensor,
    bos_token_id: int,
    pad_token_id: int,
):
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
    """Merge input_ids with audio features into final embeddings."""
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

    position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_(
        (final_attention_mask == 0), 1
    )
    return (
        final_embedding,
        final_attention_mask,
        final_labels,
        position_ids,
        final_input_ids,
        final_audio_in_mask,
        final_audio_in_discrete_codes_mask,
        final_audio_out_mask,
    )


class AudioTokenizer:
    """Common interface for audio tokenizers."""

    def __init__(self, model, device="cuda:0"):
        self._model = model
        self._device = device
        self.audio_tokenizer_model = load_higgs_audio_tokenizer(
            model,
            device=device,
        )
        self._tps = self.audio_tokenizer_model.frame_rate
        self._sampling_rate = self.audio_tokenizer_model.sample_rate
        self._num_codebooks = self.audio_tokenizer_model.n_q
        self._codebook_size = self.audio_tokenizer_model.quantizer_dim

    @property
    def tps(self):
        return self._tps

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @property
    def num_codebooks(self):
        return self._num_codebooks

    @property
    def codebook_size(self):
        return self._codebook_size

    @property
    def tps(self):
        return self._tps

    def encode(
        self,
        audio_path_or_wv,
        sr=None,
        loudness_normalize=False,
        loudness_threshold=-23.0,
    ):
        return self.audio_tokenizer_model.encode(
            audio_path_or_wv, sr, loudness_normalize, loudness_threshold
        )

    def decode(self, vq_code, return_cuda_tensor=False):
        """Decode the audio codes to waveform.

        Parameters:
        -----------
        vq_code: torch.Tensor
            The audio codes to decode. Shape (num_codebooks, total_length)

        Returns:
        --------
        decoded_wv: np.ndarray
            The decoded waveform. Shape (#time,)
        sampling_rate: int
            The sampling rate of the decoded waveform.
        """
        with torch.no_grad():
            if isinstance(vq_code, torch.Tensor):
                vq_code = vq_code.to(self._device)
            else:
                vq_code = torch.from_numpy(vq_code).to(self._device)
            decoded_wv = xcodec_decode_chunk_by_chunk(
                self.audio_tokenizer_model,
                vq_code.unsqueeze(0),
                chunk_size=60 * self.tps,
            )[0, 0]

            if not return_cuda_tensor:
                return decoded_wv, self.sampling_rate

            sampling_rate = self.sampling_rate
            return torch.from_numpy(decoded_wv), sampling_rate


def xcodec_get_output_length(input_length: int):
    conv_transpose_layers = [
        dict(kernel_size=16, stride=8, padding=4, output_padding=0),
        dict(kernel_size=10, stride=5, padding=3, output_padding=1),
        dict(kernel_size=8, stride=4, padding=2, output_padding=0),
        dict(kernel_size=4, stride=2, padding=1, output_padding=0),
        dict(kernel_size=6, stride=3, padding=2, output_padding=1),
    ]
    length = input_length
    for layer in conv_transpose_layers:
        length = (
            (length - 1) * layer["stride"]
            - 2 * layer["padding"]
            + layer["kernel_size"]
            + layer["output_padding"]
        )
    return length


def xcodec_decode_chunk_by_chunk(
    xcodec_model: torch.nn.Module, codes: torch.Tensor, chunk_size: int = 750
):
    overlap_width = 16
    chunk_output_length = xcodec_get_output_length(chunk_size)
    outputs = []
    # split the codes into chunks, with overlap at the beginning and end
    for i in range(0, codes.shape[-1], chunk_size):
        begin = max(0, i - overlap_width)
        end = min(i + chunk_size + overlap_width, codes.shape[-1])
        chunk = codes[:, :, begin:end]
        output = xcodec_model.decode(chunk)
        if i == 0:
            output = output[:, :, :chunk_output_length]
        elif i + chunk_size >= codes.shape[-1]:
            last_chunk_size = codes.shape[-1] - i
            last_chunk_output_length = xcodec_get_output_length(last_chunk_size)
            output = output[:, :, -last_chunk_output_length:]
        else:
            extra_length = (
                xcodec_get_output_length(chunk_size + overlap_width * 2) - chunk_output_length
            ) // 2
            output = output[:, :, extra_length:-extra_length]
        outputs.append(output)

    return np.concatenate(outputs, axis=2)


def load_higgs_audio_tokenizer(tokenizer_name_or_path, device="cuda"):
    is_local = os.path.exists(tokenizer_name_or_path)
    if not is_local:
        tokenizer_path = snapshot_download(tokenizer_name_or_path)
    else:
        tokenizer_path = tokenizer_name_or_path
    config_path = os.path.join(tokenizer_path, "config.json")
    if os.path.exists(config_path):
        config = json.load(open(config_path))
    elif os.path.exists(os.path.join(tokenizer_path, "config.yaml")):
        # Old version omega config file
        config = OmegaConf.load(os.path.join(tokenizer_path, "config.yaml")).generator.config
    else:
        raise ValueError(f"No config file found in {tokenizer_path}")
    model_path = os.path.join(tokenizer_path, "model.pth")

    # Dynamically get valid parameters from HiggsAudioTokenizer.__init__ method
    init_signature = inspect.signature(HiggsAudioTokenizer.__init__)
    valid_params = set(init_signature.parameters.keys()) - {"self"}  # exclude 'self'
    filtered_config = {k: v for k, v in config.items() if k in valid_params}

    model = HiggsAudioTokenizer(
        **filtered_config,
        device=device,
    )
    parameter_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(parameter_dict, strict=False)
    model.to(device)
    model.eval()
    return model


"""Projector that maps hidden states from the LLM component to multimodal logits."""


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


@dataclass
class HiggsAudioDecoderLayerOutput:
    logits: torch.FloatTensor
    audio_logits: torch.FloatTensor
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


class HiggsAudioDecoderProjector(HiggsAudioPreTrainedModel):
    """Projection layers that map hidden states from the LLM component to audio / text logits.

    We support two type of audio head:
    - Basic Audio Head:
        Directly map the hidden states to audio logits for all the codebooks.
    """

    def __init__(self, config: HiggsAudioConfig, layer_idx: Optional[int] = None):
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
        use_cache=None,
    ):
        logits = self.text_lm_head(hidden_states)
        next_decoder_cache = None

        next_cache = next_decoder_cache if use_cache else None

        audio_logits = self.audio_lm_head(hidden_states[audio_out_mask])

        return (
            logits,
            audio_logits,
            next_cache,
        )


class GenerationMode(Enum):
    """Enum for different generation modes in HiggsAudio model."""

    TEXT = 0  # Text generation mode
    AUDIO_INIT = 1  # Audio generation mode initialization
    AUDIO_IN_PROGRESS = 2  # Audio generation mode in progress


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
            old_shape_functions.append(getattr(layer.self_attn, "_shape"))
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

        self.self_attn.config._attn_implementation = "flash_attention_2"
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
        attention_mask: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        audio_out_mask: Optional[torch.BoolTensor] = None,
        is_decoding_audio_token: Optional[bool] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        **kwargs,
    ):
        residual = hidden_states
        target_length = hidden_states.shape[1]
        use_static_cache = isinstance(past_key_value, StaticCache)

        audio_out_mask_sq = audio_out_mask.unsqueeze(1) & audio_out_mask.unsqueeze(2)

        min_dtype = torch.finfo(hidden_states.dtype).min
        if attention_mask is None:
            attention_mask = ~audio_out_mask

        elif len(attention_mask.shape) == 2:
            # Attention mask has shape (batch_size, sequence_length)
            # We should be using flash attention 2
            attention_mask = attention_mask * ~audio_out_mask
        elif len(attention_mask.shape) == 4:
            # When using static cache, the attention mask was already preprocessed in the previous layer

            if use_cache:
                # Attention mask has shape (batch_size, 1, query_length, key_length)
                # In addition, the attention mask should be inverted, that means "1" (attend_to) --> "0", and "0" --> minimal dtype value.
                attention_mask = attention_mask.masked_fill(
                    audio_out_mask[:, -target_length:].reshape(
                        audio_out_mask.shape[0], 1, target_length, 1
                    )
                    | audio_out_mask.reshape(
                        audio_out_mask.shape[0], 1, 1, audio_out_mask.shape[1]
                    ),
                    min_dtype,
                )
            else:
                attention_mask = attention_mask.masked_fill(
                    audio_out_mask.reshape(audio_out_mask.shape[0], 1, audio_out_mask.shape[1], 1)
                    | audio_out_mask.reshape(
                        audio_out_mask.shape[0], 1, 1, audio_out_mask.shape[1]
                    ),
                    min_dtype,
                )
        else:
            raise NotImplementedError(
                f"Unsupported attention_mask format, attention_mask={attention_mask}"
            )

        # Apply separate layernorm layers for audio tokens and text tokens
        if use_cache:
            hidden_states = torch.where(
                audio_out_mask_sq[:, -target_length:].unsqueeze(-1),
                self.audio_input_layernorm(hidden_states),
                self.input_layernorm(hidden_states),
            )
        else:
            hidden_states = torch.where(
                audio_out_mask_sq.unsqueeze(-1),
                self.audio_input_layernorm(hidden_states),
                self.input_layernorm(hidden_states),
            )

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

        if use_cache:
            real_audio_out_mask = audio_out_mask_sq[:, -target_length:]
        else:
            real_audio_out_mask = audio_out_mask_sq

        text_hidden_states = self.post_attention_layernorm(hidden_states[~real_audio_out_mask])
        audio_hidden_states = self.audio_post_attention_layernorm(
            hidden_states[real_audio_out_mask]
        )

        text_hidden_states = self.mlp(text_hidden_states)
        residual[~real_audio_out_mask] += text_hidden_states

        audio_hidden_states = self.audio_mlp(audio_hidden_states)
        residual[real_audio_out_mask] += audio_hidden_states

        hidden_states = residual

        outputs = (hidden_states,)

        if use_cache:
            outputs += (past_key_value,)

        return outputs


@dataclass
class HiggsAudioModelOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    llm_loss: Optional[torch.FloatTensor] = None
    audio_loss: Optional[torch.FloatTensor] = None
    codebook_losses: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    expanded_input_ids: Optional[torch.LongTensor] = None
    expanded_labels: Optional[torch.LongTensor] = None
    audio_in_mask: Optional[torch.BoolTensor] = None
    audio_in_discrete_codes_mask: Optional[torch.BoolTensor] = None
    audio_out_mask: Optional[torch.BoolTensor] = None
    attention_mask: Optional[torch.BoolTensor] = None
    audio_logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    audio_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class HiggsAudioGenerationOutput(ModelOutput):
    """Outputs of HiggsAudio generation models, when using non-beam methods."""

    sequences: torch.LongTensor = None
    audio_sequences: Optional[List[torch.LongTensor]] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None


class HiggsAudioModel(HiggsAudioPreTrainedModel, GenerationMixin):
    """Higgs-Audio is an end-to-end multimodal model with the capability to understand and generate text / audio."""

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
                    layers.append(HiggsAudioDualFFNDecoderLayer(config, layer_idx))
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
                        )
                    )
                    layer_idx += 2 if self.use_audio_out_self_attention else 1
                else:
                    layers.append(HiggsAudioDualFFNDecoderLayer(config, layer_idx))
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
        self.norm = LlamaRMSNorm(
            config.text_config.hidden_size, eps=config.text_config.rms_norm_eps
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=config.text_config)

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

    def _embed_audio_ids(self, audio_ids):
        """Embed the audio ids."""
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

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
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
        return audio_attention_mask

    def _forward_core(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        position_ids: torch.Tensor,
        audio_discrete_codes_mask: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]],
        use_cache: bool,
        audio_attention_mask: torch.Tensor,
        is_decoding_audio_token: Optional[bool] = None,
        is_using_cuda_graph: Optional[bool] = False,
    ):
        # create position embeddings to be shared across the decoder layers
        # When past_key_values is passed in, we need to offset the position ids when calculating the position embeddings.
        # Therefore, cache_position is used.
        position_id_offset = cache_position[0] if use_cache else 0
        position_embeddings = self.rotary_emb(hidden_states, position_ids + position_id_offset)

        for decoder_layer in self.layers:
            if isinstance(decoder_layer, HiggsAudioDualFFNDecoderLayer):
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    audio_attention_mask=audio_attention_mask,
                    position_ids=position_ids,
                    audio_out_mask=audio_discrete_codes_mask,
                    is_decoding_audio_token=is_decoding_audio_token,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    is_using_cuda_graph=is_using_cuda_graph,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

        return hidden_states

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_feature_attention_mask: Optional[torch.BoolTensor] = None,
        audio_in_ids: Optional[torch.LongTensor] = None,
        audio_in_ids_start: Optional[torch.LongTensor] = None,
        audio_out_ids: Optional[torch.LongTensor] = None,
        audio_out_ids_start: Optional[torch.LongTensor] = None,
        audio_out_ids_start_group_loc: Optional[torch.LongTensor] = None,
        label_ids: Optional[torch.LongTensor] = None,
        label_audio_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cache_audio_discrete_codes_mask: Optional[torch.LongTensor] = None,
        past_key_values_buckets: Optional[OrderedDict[int, Cache]] = None,
        reward: Optional[torch.FloatTensor] = None,
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
            position_ids,
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
            audio_attention_mask = self._prepare_all_static_kv_cache_masks(
                hidden_states, causal_mask, audio_discrete_codes_mask, past_key_values
            )
            # Set the audio out mask to the last token
            if hidden_states.shape[1] == 1:
                audio_discrete_codes_mask = audio_discrete_codes_mask[:, -1:]
                audio_discrete_codes_mask = audio_discrete_codes_mask.reshape((-1, 1)).contiguous()
                is_decoding_audio_token = audio_discrete_codes_mask.item()
            else:
                is_decoding_audio_token = False

        _forward_core = self._forward_core

        hidden_states, all_hidden_states, all_self_attns = _forward_core(
            hidden_states=hidden_states,
            causal_mask=causal_mask,
            position_ids=position_ids,
            audio_discrete_codes_mask=audio_discrete_codes_mask,
            is_decoding_audio_token=is_decoding_audio_token if use_static_cache else None,
            cache_position=cache_position,
            past_key_values=past_key_values,
            use_cache=use_cache,
            audio_attention_mask=audio_attention_mask if use_static_cache else None,
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
            position_ids=position_ids,
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
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
        extend_attention_mask: bool = True,
    ) -> Dict[str, Any]:
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
        current_past_key_values_bucket: Optional[int],
        past_key_values_buckets: OrderedDict[int, Cache],
    ) -> Tuple[Optional[Cache], Optional[int]]:
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
        torch_generator: Optional[torch.Generator],
        generation_config: GenerationConfig,
        num_delay: int,
        num_remaining_delays: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, Optional[int]]:
        """Sample audio tokens and its corresponding text tokens from the logits"""
        # parameters related to repetition aware sampling
        ras_win_len = generation_config.generation_kwargs.get("ras_win_len", None)
        ras_win_max_num_repeat = generation_config.generation_kwargs.get(
            "ras_win_max_num_repeat", 2
        )
        audio_eos_token_id = generation_config.generation_kwargs.get("audio_eos_token_id", None)
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

    def _sample_text_tokens(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        do_sample: bool,
        logits_processor: LogitsProcessorList,
        device: torch.device,
        generation_mode: GenerationMode,
        torch_generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Sample text tokens from the logits"""
        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = logits.clone()[:, -1, :].float()
        next_token_logits = next_token_logits.to(input_ids.device)

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        if generation_mode == GenerationMode.AUDIO_INIT:
            # See the audio bos token, we should start generating audio tokens
            next_tokens = torch.full(
                (input_ids.shape[0],),
                self.audio_out_token_idx,
                dtype=torch.long,
                device=device,
            )
            next_audio_tokens = torch.full(
                (self.config.audio_num_codebooks,),
                self.config.audio_stream_bos_id,
                dtype=torch.long,
                device=device,
            )
        else:
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(
                    probs, num_samples=1, generator=torch_generator
                ).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            next_audio_tokens = None

        return next_tokens, next_audio_tokens, next_token_logits, next_token_scores

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
        past_key_values_buckets: Optional[OrderedDict[int, Cache]],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        assert input_ids.shape[0] == 1, "Only support batch_size=1 in _sample()"
        audio_out_bos_token_id = generation_config.generation_kwargs.get(
            "audio_out_bos_token_id", None
        )

        # torch generator for sampling
        seed = generation_config.generation_kwargs.get("seed", None)
        if seed is not None:
            torch_generator = torch.Generator(device=input_ids.device).manual_seed(seed)
        else:
            torch_generator = None

        # init values
        pad_token_id = generation_config._pad_token_tensor
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(
            hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
        )
        do_sample = generation_config.do_sample
        # Used to track which past_key_va
        self.current_past_key_values_bucket = None

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
            this_peer_finished,
            synced_gpus,
            device=input_ids.device,
            cur_len=cur_len,
            max_length=max_length,
        ):
            # Check which multimodal stage we are in
            # FIXME: Assume single input generation
            if input_ids[0][-1] == audio_out_bos_token_id:
                generation_mode = GenerationMode.AUDIO_INIT
            elif input_ids[0][-1] == self.audio_out_token_idx:
                generation_mode = GenerationMode.AUDIO_IN_PROGRESS
            else:
                generation_mode = GenerationMode.TEXT

            is_audio_generation_mode = generation_mode == GenerationMode.AUDIO_IN_PROGRESS

            if init_model_input or not generation_config.use_cache:
                model_inputs = {"input_ids": input_ids, **model_kwargs}
            else:
                model_inputs = {"input_ids": input_ids[:, -1:], **model_kwargs}

                if is_audio_generation_mode and generation_config.use_cache:
                    model_inputs["audio_out_ids"] = model_kwargs["audio_out_ids"][:, -1:]
                    model_inputs["audio_out_ids_start"] = torch.tensor(
                        [0], dtype=torch.long, device=input_ids.device
                    )
                elif not is_audio_generation_mode:
                    del model_inputs["audio_out_ids"]
                    del model_inputs["audio_out_ids_start"]

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

            if is_audio_generation_mode:
                # In audio generation mode, we sample the audio tokens from audio logits.
                # It might also generate the audio eos token to end the audio generation.
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
            else:
                # In text generation mode, we sample the text tokens from text logits.
                # It might also generate the audio placeholder token to start the audio generation.
                next_tokens, next_audio_tokens, next_token_logits, next_token_scores = (
                    self._sample_text_tokens(
                        input_ids=input_ids,
                        logits=outputs.logits,
                        do_sample=do_sample,
                        logits_processor=logits_processor,
                        device=input_ids.device,
                        generation_mode=generation_mode,
                        torch_generator=torch_generator,
                    )
                )

                if streamer is not None:
                    streamer.put(next_tokens.cpu())

                if next_audio_tokens is not None:
                    # If the token is audio bos token, we will generate the audio placeholder token
                    # and the corrensponding audio stream bos token to start the audio generation.
                    audio_sequences.append(next_audio_tokens[:, None])
                    if streamer is not None:
                        streamer.put(next_audio_tokens.cpu())
                    if (
                        model_kwargs["audio_out_ids"] is None
                        or model_kwargs["audio_out_ids"].shape[0] == 0
                    ):
                        # Initialize audio_out_ids
                        model_kwargs["audio_out_ids"] = next_audio_tokens[:, None]
                        model_kwargs["audio_out_ids_start"] = torch.tensor(
                            [0], dtype=torch.long, device=input_ids.device
                        )
                    else:
                        model_kwargs["audio_out_ids_start"] = torch.concat(
                            [
                                model_kwargs["audio_out_ids_start"],
                                torch.tensor(
                                    [model_kwargs["audio_out_ids"].shape[1]],
                                    dtype=torch.long,
                                    device=input_ids.device,
                                ),
                            ],
                            dim=0,
                        )
                        model_kwargs["audio_out_ids"] = torch.concat(
                            [model_kwargs["audio_out_ids"], next_audio_tokens[:, None]], dim=1
                        )

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            if "tokenizer_length" in generation_config.generation_kwargs:
                tokenizer_length = generation_config.generation_kwargs["tokenizer_length"]
                if torch.max(next_tokens) >= tokenizer_length:
                    raise ValueError(
                        f"Next generated token has max value {torch.max(next_tokens)} which is greater than the tokenizer's vocabulary size {tokenizer_length}, this is undesired behavior."
                    )

            # update generated ids, model inputs, and length for next step
            if not is_audio_generation_mode or next_tokens[0] != self.audio_out_token_idx:
                # We only add one <|AUDIO_OUT|> token to the input_ids for simplicity.
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            input_ids_full = torch.cat([input_ids_full, next_tokens[:, None]], dim=-1)
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids_full, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        return input_ids, audio_sequences

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_feature_attention_mask: Optional[torch.BoolTensor] = None,
        audio_in_ids: Optional[torch.LongTensor] = None,
        audio_in_ids_start: Optional[torch.LongTensor] = None,
        audio_out_ids: Optional[torch.LongTensor] = None,
        audio_out_ids_start: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        audio_out_bos_token_id: int = None,
        audio_eos_token_id: int = None,
        past_key_values_buckets: Optional[OrderedDict[int, Cache]] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        # Right now, it's a very simplified version of generate, we should revisit this after our model architecture stabilizes.
        assert input_ids.shape[0] == 1, (
            "Currently HiggsAudioModel.generate() only supports batch_size=1. See the implementation of "
        )
        generation_config, kwargs = self._prepare_generation_config(
            kwargs.pop("generation_config", None), **kwargs
        )
        if audio_out_bos_token_id is not None:
            generation_config.generation_kwargs["audio_out_bos_token_id"] = audio_out_bos_token_id
        else:
            try:
                generation_config.generation_kwargs["audio_out_bos_token_id"] = (
                    self.audio_out_bos_token_id
                )
            except:
                generation_config.generation_kwargs["audio_out_bos_token_id"] = None

        if audio_eos_token_id is not None:
            generation_config.generation_kwargs["audio_eos_token_id"] = audio_eos_token_id
        else:
            try:
                generation_config.generation_kwargs["audio_eos_token_id"] = self.audio_eos_token_id
            except:
                generation_config.generation_kwargs["audio_eos_token_id"] = None

        has_default_max_length = (
            kwargs.get("max_length") is None and generation_config.max_length is not None
        )
        has_default_min_length = (
            kwargs.get("min_length") is None and generation_config.min_length is not None
        )

        generation_config.generation_kwargs["ras_win_len"] = kwargs.pop("ras_win_len", None)
        generation_config.generation_kwargs["ras_win_max_num_repeat"] = kwargs.pop(
            "ras_win_max_num_repeat", 2
        )
        # Set generation seed if determinstic generation is required
        if seed is not None:
            generation_config.generation_kwargs["seed"] = seed

        # Store tokenizer in generation config if it is in kwargs without popping it
        if "tokenizer" in kwargs:
            generation_config.generation_kwargs["tokenizer_length"] = len(kwargs["tokenizer"])

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
            past_key_values_buckets=past_key_values_buckets,
            **kwargs,
        )
