# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
"""TensorRT-LLM implementation of Higgs Audio multimodal model."""

import copy
import inspect
import json
import math
import os
import sys
import tempfile
import warnings
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from functools import lru_cache
from typing import Any, ClassVar, Dict, List, Literal, Optional, Set, Tuple, TypedDict, Union

import librosa
import numpy as np
import s3fs
import torch
import torch.nn as nn
import torch.nn.functional as F

from boson_multimodal import (
    HiggsAudioFeatureExtractor as BM_HiggsAudioFeatureExtractor,
    HiggsAudioTokenizer,
)
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
import transformers
from transformers import AutoConfig, AutoFeatureExtractor, BatchFeature, ProcessorMixin
from transformers.models.whisper import WhisperFeatureExtractor
from transformers.tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput

# TensorRT-LLM imports
from .config import HiggsAudioConfig, HiggsAudioEncoderConfig
from ..modeling_utils import (
    DecoderModelForCausalLM,
    PretrainedConfig,
    PretrainedModel,
    DecoderLayerList,
)
from ...functional import (
    Tensor,
    ACT2FN,
    PositionEmbeddingType,
    RmsNorm,
    concat,
    constant,
    gather_last_token_logits,
    shape,
    split,
    view,
)
from ...layers import (
    MLP,
    Attention,
    AttentionParams,
    ColumnLinear,
    Embedding,
    GatedMLP,
    KeyValueCacheParams,
    Linear,
    LoraParams,
    RmsNorm,
    RowLinear,
)
from ...mapping import Mapping
from ...module import Module, ModuleList
from ...parameter import Parameter
from ...quantization import QuantMode

# TensorRT-LLM runtime imports
import tensorrt as trt  # type: ignore
import tensorrt_llm
from tensorrt_llm import logger as trtllm_logger
from tensorrt_llm.bindings import KVCacheType
from tensorrt_llm.runtime import (
    PYTHON_BINDINGS,
    ModelConfig,
    ModelRunner,
    SamplingConfig as TRTSamplingConfig,
    Session,
    TensorInfo,
)

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp


def revert_delay_pattern(data):
    """Convert samples encoded with delay pattern back to the original form.

    Args:
        data (:obj:`np.ndarray`):
            The data with delay pattern applied. It will have shape (num_codebooks, seq_len + num_codebooks - 1).

    Returns:
        ret (:obj:`np.ndarray`):
            Recovered data with delay pattern removed. It will have shape (num_codebooks, seq_len).
    """
    assert len(data.shape) == 2
    out_l = []
    num_codebooks = data.shape[0]
    for i in range(num_codebooks):
        out_l.append(data[i : (i + 1), i : (data.shape[1] - num_codebooks + 1 + i)])
    return np.concatenate(out_l, axis=0)


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


# Logger setup
trtllm_logger.info("HiggsAudio TensorRT-LLM model loaded")

_KEYS_TO_MODIFY_MAPPING = {
    "audio_decoder_proj.audio_lm_head": "audio_lm_head",
    "audio_decoder_proj.text_lm_head": "text_lm_head",
    "model.": "",
}

AutoConfig.register("higgs_audio_encoder", HiggsAudioEncoderConfig)
AutoConfig.register("higgs_audio", HiggsAudioConfig)
AutoFeatureExtractor.register(HiggsAudioConfig, AudioTokenizer)
# if transformers.__version__.startswith("4.46"):
transformers._modules.add("AudioTokenizer")
transformers.AudioTokenizer = AudioTokenizer


# # === Audio Inputs === #
class HiggsAudioInputs(TypedDict):
    # (num_audios, num_mel_bins, 3000)`
    audio_features: torch.Tensor

    # (num_audios, 3000)
    audio_feature_attention_mask: torch.Tensor

    # (num_audios, num_codebooks)
    audio_out_ids: torch.Tensor


def _validate_and_reshape_mm_tensor(
    mm_input: object,
    name: str,
    pad_with: Optional[int] = None,
) -> torch.Tensor:
    if not isinstance(mm_input, (torch.Tensor, list)):
        raise ValueError(f"Incorrect type of {name}. Got type: {type(mm_input)}")
    if isinstance(mm_input, torch.Tensor):
        return torch.concat(list(mm_input))
    else:
        if pad_with is not None:
            max_size = max(
                [tensor.size(-1) for tensor in mm_input]
            )  # Find max size along the last dimension
            # Step 2: Pad each tensor to the max size along the last
            # dimension
            padded_tensors = []
            for tensor in mm_input:
                pad_size = max_size - tensor.size(-1)  # Calculate how much padding is needed
                if pad_size > 0:
                    # Pad tensor along the last dimension (right side)
                    padded_tensor = torch.nn.functional.pad(tensor, (0, pad_size))
                else:
                    padded_tensor = tensor
                padded_tensors.append(padded_tensor)
            return torch.concat(padded_tensors)
        else:
            return torch.concat(mm_input)


def _build_delay_pattern_mask(
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
    return input_ids


# TensorRT-LLM implementation of HiggsAudioEncoder
class HiggsAudioEncoderLayer(Module):
    """TensorRT-LLM encoder layer for audio processing."""
    
    def __init__(self, config: HiggsAudioEncoderConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.self_attn = Attention(
            hidden_size=config.d_model,
            num_attention_heads=config.encoder_attention_heads,
            max_position_embeddings=config.max_source_positions,
            num_layers=config.encoder_layers,
            apply_query_key_layer_scaling=False,
            attention_mask_type=None,
            bias=True,
            dtype=config.dtype,
            tp_group=None,
            tp_size=1,
            tp_rank=0,
            quant_mode=QuantMode(0),
        )
        
        self.mlp = MLP(
            hidden_size=config.d_model,
            ffn_hidden_size=config.encoder_ffn_dim,
            hidden_act=config.activation_function,
            dtype=config.dtype,
            bias=True,
            tp_group=None,
            tp_size=1,
            quant_mode=QuantMode(0),
        )
        
        self.self_attn_layer_norm = RmsNorm(
            normalized_shape=config.d_model,
            eps=config.layer_norm_eps,
            dtype=config.dtype
        )
        
        self.final_layer_norm = RmsNorm(
            normalized_shape=config.d_model,
            eps=config.layer_norm_eps,
            dtype=config.dtype
        )

    def forward(self,
                hidden_states: Tensor,
                attention_mask: Optional[Tensor] = None,
                use_cache: bool = False,
                kv_cache_params: Optional[KeyValueCacheParams] = None,
                attention_params: Optional[AttentionParams] = None) -> Tensor:
        
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        
        attention_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
        )
        
        if use_cache:
            attention_output, present_key_value = attention_output
        
        hidden_states = residual + attention_output
        
        # MLP block
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        if use_cache:
            return hidden_states, present_key_value
        return hidden_states


class HiggsAudioEncoder(Module):
    """
    TensorRT-LLM transformer encoder for audio features.
    
    Args:
        config: HiggsAudioEncoderConfig
    """

    def __init__(self, config: HiggsAudioEncoderConfig):
        super().__init__()
        self.config = config
        
        embed_dim = config.d_model
        self.num_mel_bins = config.n_mels
        self.max_source_positions = getattr(config, 'max_source_positions', 1500)
        self.embed_scale = math.sqrt(embed_dim) if getattr(config, 'scale_embedding', True) else 1.0
        
        # Use TensorRT-LLM's conv1d operations via functional API
        # For now, we'll use PyTorch conv1d but will convert to TensorRT-LLM functional later
        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        
        # Positional embeddings
        self.embed_positions = Embedding(
            num_embeddings=self.max_source_positions,
            embedding_dim=embed_dim,
            dtype=config.dtype
        )
        
        # Encoder layers using TensorRT-LLM components
        self.layers = ModuleList([
            HiggsAudioEncoderLayer(config, i) 
            for i in range(getattr(config, 'encoder_layers', 12))
        ])
        
        self.layer_norm = RmsNorm(
            normalized_shape=config.d_model,
            eps=getattr(config, 'layer_norm_eps', 1e-5),
            dtype=config.dtype
        )
        
        # Average pooling - will implement using functional operations
        self.avg_pool_stride = 2

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(self,
                input_features: Tensor,
                attention_mask: Optional[Tensor] = None,
                use_cache: bool = False,
                kv_cache_params: Optional[KeyValueCacheParams] = None,
                attention_params: Optional[AttentionParams] = None) -> Tensor:
        """
        Forward pass for audio encoder.
        
        Args:
            input_features: Audio features of shape (batch_size, n_mels, seq_len)
            attention_mask: Optional attention mask
            use_cache: Whether to use KV cache
            kv_cache_params: KV cache parameters
            attention_params: Attention parameters
            
        Returns:
            Encoded audio features
        """
        
        # Apply convolutions for feature extraction
        # Convert to PyTorch tensors temporarily for conv operations
        if isinstance(input_features, Tensor):
            x = input_features.data
        else:
            x = input_features
            
        x = x.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        
        # Transpose to (batch_size, seq_len, hidden_size)
        x = x.permute(0, 2, 1)
        
        # Convert back to TensorRT-LLM Tensor
        if not isinstance(x, Tensor):
            from ...functional import constant
            hidden_states = constant(x)
        else:
            hidden_states = x
            
        # Add positional embeddings
        seq_len = shape(hidden_states, 1)
        position_ids = constant(torch.arange(seq_len.data.item(), device=hidden_states.data.device))
        position_embeddings = self.embed_positions(position_ids)
        hidden_states = hidden_states + position_embeddings
        
        # Pass through encoder layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                kv_cache_params=kv_cache_params,
                attention_params=attention_params
            )
            
            if use_cache:
                hidden_states, _ = hidden_states  # Ignore cache for now in encoder
        
        # Apply average pooling (stride=2) - implement using TensorRT-LLM operations
        # Convert to torch tensor for pooling, then back to TensorRT-LLM tensor
        x = hidden_states.data.permute(0, 2, 1)  # (batch, hidden, seq)
        x = F.avg_pool1d(x, kernel_size=2, stride=2)
        x = x.permute(0, 2, 1)  # Back to (batch, seq, hidden)
        hidden_states = constant(x)
        
        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        return hidden_states

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and
        the output length of the audio encoder
        """
        # Account for conv2 stride of 2 and avg pooling stride of 2
        input_lengths = (input_lengths - 1) // 2 + 1  # conv2
        input_lengths = (input_lengths - 1) // 2 + 1  # avg pooling
        return input_lengths
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths


class HiggsAudioFeatureProjector(Module):
    """Projector that maps audio features extracted by encoder to
    hidden state of the text model."""

    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.config = config
        
        audio_hidden_size = config.audio_encoder_config.d_model
        text_hidden_size = config.hidden_size
        
        self.linear = Linear(
            in_features=audio_hidden_size,
            out_features=text_hidden_size,
            bias=True,
            dtype=config.dtype,
            tp_group=None,
            tp_size=1,
            gather_output=True,
        )

    def forward(self, audio_features: Tensor) -> Tensor:
        """Project audio features to text hidden size."""
        hidden_states = self.linear(audio_features)
        return hidden_states


class HiggsAudioDecoderProjector(Module):
    """Projection layers that map hidden states from the
    LLM component to audio / text logits."""

    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.config = config
        
        # Audio projection head
        self.audio_lm_head = Linear(
            in_features=config.hidden_size,
            out_features=config.audio_num_codebooks * config.audio_codebook_size,
            bias=False,
            dtype=config.dtype,
            tp_group=None,
            tp_size=1,
            gather_output=True,
        )
        
        # Text projection head  
        self.text_lm_head = Linear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            bias=False,
            dtype=config.dtype,
            tp_group=None,
            tp_size=1,
            gather_output=True,
        )

    def forward(self, 
                hidden_states: Tensor,
                mode: str = "text") -> Tensor:
        """
        Project hidden states to logits.
        
        Args:
            hidden_states: Hidden states from LLM
            mode: "text" or "audio" to select projection head
            
        Returns:
            Logits for text or audio tokens
        """
        if mode == "audio":
            logits = self.audio_lm_head(hidden_states)
            # Reshape to (batch, seq_len, num_codebooks, codebook_size)
            batch_size, seq_len = shape(hidden_states, 0), shape(hidden_states, 1)
            logits = view(logits, [batch_size, seq_len, self.config.audio_num_codebooks, self.config.audio_codebook_size])
        else:  # text mode
            logits = self.text_lm_head(hidden_states)
            
        return logits

    def forward(
        self,
        hidden_states,
        audio_out_mask=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
    ):
        """
        Args:
            hidden_states (`torch.Tensor` of shape
                           `(batch_size, seq_len, hidden_size)`):
                Hidden states from the LLM component
            audio_out_mask (`torch.Tensor` of shape `(batch_size, seq_len)`):
                Mask for identifying the audio out tokens.
            attention_mask (`torch.Tensor` of shape `(batch_size, seq_len)`):
                Mask to avoid performing attention on padding token indices
            position_ids (`torch.Tensor` of shape `(batch_size, seq_len)`):
                Position ids for the input tokens

        Returns:
            logits (`torch.Tensor` of shape
                   `(batch_size, seq_len, vocab_size)`):
                Logits for text tokens
            audio_logits (`torch.Tensor` of shape
                `(num_audio_out_tokens, audio_num_codebooks * audio_codebook_size)`):
                Logits for audio tokens. We ensure
                `num_text_tokens + num_audio_tokens == batch_size * seq_len`.
                If we the model only outputs text logits,
                `audio_logits` will be `None`.

        """
        # TODO(sxjscience) Need to check if DeepSpeed Zero3 supports zero-shape input.
        if self._audio_decoder_proj_num_layers > 0:
            # create position embeddings to be shared across the decoder layers
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            for decoder_layer in self.transformer_layers:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                hidden_states = layer_outputs[0]
            hidden_states = self.norm(hidden_states)

        return hidden_states


def get_processor(
    tokenzier,
    *args,
    trust_remote_code: bool = False,
    **kwargs,
):
    """Gets a processor for the given model name via HuggingFace.

    Derived from `vllm.transformers_utils.image_processor.get_image_processor`.
    """
    # don't put this import at the top level
    # it will call torch.cuda.device_count()
    from transformers import AutoFeatureExtractor

    HIGGS_AUDIO_TOKENIZER = os.getenv("HIGGS_AUDIO_TOKENIZER", "openai/whisper-large-v3-turbo")

    audio_stream_bos_id = kwargs.pop("audio_stream_bos_id", None)
    audio_stream_eos_id = kwargs.pop("audio_stream_eos_id", None)

    if HIGGS_AUDIO_TOKENIZER == "openai/whisper-large-v3-turbo":
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            HIGGS_AUDIO_TOKENIZER,  # TODO: Write into config file
            *args,
            trust_remote_code=trust_remote_code,
            attn_implementation="sdpa",
            **kwargs,
        )
    else:
        HIGGS_AUDIO_TOKENIZER_PATH = os.environ.get(
            "HIGGS_AUDIO_TOKENIZER_PATH",
            None,
        )
        feature_extractor = AudioTokenizer(
            model=HIGGS_AUDIO_TOKENIZER,
            device="cuda",
        )
    processor = HFHiggsAudioProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenzier,
        audio_stream_bos_id=audio_stream_bos_id,
        audio_stream_eos_id=audio_stream_eos_id,
    )
    logger.info("Loaded HFHiggsAudioProcessor")

    return processor


cached_get_processor = lru_cache(get_processor)


def _get_feat_extract_output_lengths(input_lengths: torch.LongTensor):
    """
    Computes the output length of the convolutional layers
    and the output length of the audio encoder
    """
    input_lengths = (input_lengths - 1) // 2 + 1
    output_lengths = (input_lengths - 2) // 2 + 1
    return input_lengths, output_lengths


class HFHiggsAudioProcessor(ProcessorMixin):
    """
    HF Processor class for Higgs audio model. Mostly borrow from
    processing_qwen2_audio.py.
    """

    attributes = ["feature_extractor", "tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        audio_token="<|AUDIO|>",
        audio_bos_token="<|audio_bos|>",
        audio_eos_token="<|audio_eos|>",
        audio_stream_bos_id=None,
        audio_stream_eos_id=None,
        is_audio_out_model=False,
    ):
        self.is_audio_out_model = is_audio_out_model
        if chat_template is None:
            chat_template = self.default_chat_template
        self.audio_token = (
            tokenizer.audio_token if hasattr(tokenizer, "audio_token") else audio_token
        )
        self.audio_bos_token = (
            tokenizer.audio_bos_token if hasattr(tokenizer, "audio_bos_token") else audio_bos_token
        )
        self.audio_eos_token = (
            tokenizer.audio_eos_token if hasattr(tokenizer, "audio_eos_token") else audio_eos_token
        )

        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id
        # HACK: Workaround the class check in the base class
        if feature_extractor is not None:
            self.feature_extractor_class = feature_extractor.__class__.__name__
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audio: Union[np.ndarray, List[np.ndarray]] = None,
        audios=None,  # kept for BC
        padding: Union[bool, str, PaddingStrategy] = False,
        sampling_rate: Optional[int] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and
        audio(s). Borrowed the code from Qwen2 Audio.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence
                can be a string or a list of strings (pretokenized string). If
                the sequences are provided as list of strings (pretokenized),
                you must set `is_split_into_words=True` (to lift the ambiguity
                with a batch of sequences).
            audios (`np.ndarray`, `List[np.ndarray]`):
                The audio or batch of audios to be prepared. Each audio can be
                a NumPy array.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*,
                    defaults to `False`):
                Select a strategy to pad the returned sequences (according to
                the model's padding side and padding index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the
                  batch (or no padding if only a single sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the
                  argument `max_length` or to the maximum acceptable input
                  length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can
                  output a batch with sequences of different lengths).
            sampling_rate (`int`, defaults to 16000):
                The sampling rate at which the audio files should be
                digitalized expressed in hertz (Hz).
        """

        # Handle BC when user passes deprecared keyword argument
        if audios is not None and audio is None:
            audio = audios
            warnings.warn(
                "You may have used the keyword argument for the `audio` inputs. "
                "It is strongly recommended to pass inputs with keyword arguments "
                "with keys `audio` and `text`. From transformers v4.55 `audio` "
                "will be the only acceptable keyword argument.",
                FutureWarning,
            )

        if text is None:
            raise ValueError("You need to specify `text` input to process.")
        elif isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        if audio is not None:
            # ensure we have as much audios as audio tokens
            num_audio_tokens = sum(sample.count(self.audio_token) for sample in text)
            num_audios = 1 if type(audio) is np.ndarray else len(audio)
            if num_audio_tokens != num_audios:
                raise ValueError(
                    f"Found {num_audio_tokens} {self.audio_token} token"
                    f"{'s' if num_audio_tokens > 1 else ''} "
                    f"in provided text but received {num_audios} audio"
                    f"{'s' if num_audios > 1 else ''}"
                )
            # Some kwargs should not be changed so we can expand text with audio tokens below
            use_whisper = False
            if hasattr(self.feature_extractor, "encode"):
                if isinstance(audio, np.ndarray):
                    audio = [audio]
                audio = [a.astype(np.float32) for a in audio]
                audio_ids = [
                    self.feature_extractor.encode(
                        a, self.feature_extractor.sampling_rate
                    ).unsqueeze(0)
                    for a in audio
                ]

                # -2 is the number of codebooks
                num_codebook_dim = -2
                use_delay_pattern = audio_ids[0].shape[num_codebook_dim] > 1
                if use_delay_pattern:
                    for i, audio_id in enumerate(audio_ids):
                        audio_id = torch.cat(
                            [
                                torch.full(
                                    (1, audio_id.shape[num_codebook_dim], 1),
                                    self.audio_stream_bos_id,
                                    dtype=torch.long,
                                    device=audio_id.device,
                                ),
                                audio_id,
                                torch.full(
                                    (1, audio_id.shape[num_codebook_dim], 1),
                                    self.audio_stream_eos_id,
                                    dtype=torch.long,
                                    device=audio_id.device,
                                ),
                            ],
                            dim=-1,
                        )
                        audio_ids[i] = _build_delay_pattern_mask(
                            audio_id,
                            bos_token_id=self.audio_stream_bos_id,
                            pad_token_id=self.audio_stream_eos_id,
                        )

                audio_lengths = [a.shape[-1] for a in audio_ids]
                audio_in_ids_length = torch.tensor(audio_lengths)
                audio_in_ids = _validate_and_reshape_mm_tensor(
                    audio_ids, "audio_in_ids", pad_with=0
                )
                audio_feature_attention_mask = torch.arange(audio_in_ids.shape[-1]).expand(
                    audio_in_ids.shape[0], audio_in_ids.shape[-1]
                ).to(audio_in_ids_length.device) < audio_in_ids_length.unsqueeze(-1)
                audio_inputs = {
                    "input_features": audio_in_ids,
                    "audio_feature_attention_mask": audio_feature_attention_mask,
                }
            else:
                use_whisper = True
                audio_inputs = self.feature_extractor(
                    audio,
                    sampling_rate=sampling_rate,
                    return_attention_mask=True,
                    padding="max_length",
                    **kwargs,
                )
                # Rename to audio_feature_attention_mask to prevent conflicts
                # with text attention mask
                audio_inputs["audio_feature_attention_mask"] = audio_inputs.pop("attention_mask")
                audio_lengths = audio_inputs["audio_feature_attention_mask"].sum(-1).tolist()

            expanded_text = []
            for sample in text:
                replace_str = []
                while self.audio_token in sample:
                    if use_whisper:
                        audio_length = audio_lengths.pop(0)
                        input_length = (audio_length - 1) // 2 + 1
                        num_audio_tokens = (input_length - 2) // 2 + 1
                    else:
                        num_audio_tokens = audio_lengths.pop(0)

                    expanded_audio_token = self.audio_token * num_audio_tokens
                    replace_str.append(expanded_audio_token)
                    sample = sample.replace(self.audio_token, "<placeholder>", 1)

                while "<placeholder>" in sample:
                    sample = sample.replace("<placeholder>", replace_str.pop(0), 1)
                expanded_text.append(sample)
            text = expanded_text

        inputs = self.tokenizer(text, padding=padding, **kwargs)

        if audio is not None:
            inputs.update(audio_inputs)

        return BatchFeature(data={**inputs})

    @property
    def default_chat_template(self):
        # fmt: off
        if self.is_audio_out_model:
            return (
                "{% set loop_messages = messages %}"
                "{% for message in loop_messages %}"
                    "{% set content = '<|start_header_id|>' + message['role'] + "
                    "'<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
                    "{% if loop.index0 == 0 %}"
                        "{% set content = bos_token + content %}"
                    "{% endif %}"
                    "{% if message['role'] == 'assistant' and '<|audio_bos|><|AUDIO|>' in message['content'] %}"
                        "{% set content = content.replace('<|audio_bos|><|AUDIO|>', '<|audio_out_bos|><|AUDIO|>') %}"
                    "{% endif %}"
                    "{{ content }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n<|audio_out_bos|><|AUDIO_OUT|>' }}"
                "{% endif %}"
            )

        return (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
                "{% set content = '<|start_header_id|>' + message['role'] + "
                "'<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
                "{% if loop.index0 == 0 %}"
                "{% set content = bos_token + content %}"
            "{% endif %}"
            "{{ content }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
            "{% endif %}"
        )
        # fmt: on


HiggsAudioFeatureExtractorType = Union[AudioTokenizer, WhisperFeatureExtractor]


# Remove vLLM multimodal processing classes - not needed for TensorRT-LLM implementation


class HiggsAudioForCausalLM(DecoderModelForCausalLM):
    def get_hf_config(self):
        return self.ctx.get_hf_config(HiggsAudioConfig)

    def get_hf_processor(
        self,
        *,
        # Ignored in initialization
        sampling_rate: Optional[int] = None,
        **kwargs: object,
    ) -> HFHiggsAudioProcessor:
        hf_config = self.get_hf_config()
        return cached_get_processor(
            self.ctx.tokenizer,
            audio_stream_bos_id=hf_config.audio_stream_bos_id,
            audio_stream_eos_id=hf_config.audio_stream_eos_id,
        )

    def get_feature_extractor(
        self,
        *,
        # Ignored in initialization
        sampling_rate: Optional[int] = None,
    ) -> HiggsAudioFeatureExtractorType:
        hf_processor = self.get_hf_processor(sampling_rate=sampling_rate)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        hf_config = self.get_hf_config()
        self.audio_tokenizer_type = os.getenv(
            "HIGGS_AUDIO_TOKENIZER", "openai/whisper-large-v3-turbo"
        )
        if self.audio_tokenizer_type == "openai/whisper-large-v3-turbo":
            max_source_position = hf_config.audio_encoder_config.max_source_positions
            max_output_lengths = (max_source_position - 2) // 2 + 1
        else:
            max_output_lengths = (
                30 * self.get_feature_extractor().tps
                + self.get_feature_extractor().num_codebooks
                - 1
                + 2
            )  # bos and eos
        return {"audio": max_output_lengths}


class HiggsAudioForCausalLM(DecoderModelForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_tokenizer_type = os.getenv(
            "HIGGS_AUDIO_TOKENIZER", "openai/whisper-large-v3-turbo"
        )
        self.use_whisper_tokenizer = self.audio_tokenizer_type == "openai/whisper-large-v3-turbo"

    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=self.info.get_feature_extractor().sampling_rate)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Text-only input not supported in composite processor
        if not mm_data.get("audios", []):
            # Set add_special_tokens=False to avoid
            # adding an extra begin of text token
            prompt_ids = self.info.get_tokenizer().encode(prompt, add_special_tokens=False)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            batch_data = BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")
            return batch_data

        feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
        mm_kwargs = dict(
            **mm_kwargs,
            sampling_rate=feature_extractor.sampling_rate,
        )

        batch_data = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        batch_data["audio_features"] = batch_data.pop("input_features")
        return batch_data

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            audio_features=MultiModalFieldConfig.batched("audio"),
            audio_feature_attention_mask=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        # Use getattr with default to be compatible with transformers<4.48
        audio_token = getattr(processor, "audio_token", "<|AUDIO|>")
        audio_token_id = vocab[audio_token]

        audio_feature_attention_mask = out_mm_kwargs.get("audio_feature_attention_mask")
        if audio_feature_attention_mask is None:
            audio_output_lengths = []
        else:
            assert isinstance(audio_feature_attention_mask, torch.Tensor)

            if self.use_whisper_tokenizer:
                _, audio_output_lens = _get_feat_extract_output_lengths(
                    audio_feature_attention_mask.sum(-1)
                )
            else:
                audio_output_lens = audio_feature_attention_mask.sum(-1)
            audio_output_lengths = audio_output_lens.tolist()

        def get_replacement_higgs_audio(item_idx: int):
            num_features = audio_output_lengths[item_idx]
            if num_features == 0:
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio_len = audios.get_audio_length(item_idx)

                raise ValueError(
                    f"The audio (len={audio_len}) is too short to be represented inside the model"
                )

            audio_tokens = [audio_token_id] * num_features

            # New API: PromptUpdateDetails only accepts 'full' and optional 'is_embed'.
            # All tokens are embeddings here, so use from_seq.
            return PromptUpdateDetails.from_seq(audio_tokens)

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_higgs_audio,
            )
        ]


class HiggsAudioDummyInputsBuilder(BaseDummyInputsBuilder[HiggsAudioProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        # Use the processor's placeholder for audio inputs.
        # The processor recognizes '<|AUDIO|>' tokens in prompt text.
        return "<|AUDIO|>" * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, object]:
        feature_extractor = self.info.get_feature_extractor()
        sampling_rate = feature_extractor.sampling_rate
        if hasattr(feature_extractor, "chunk_length"):
            audio_len = feature_extractor.chunk_length * sampling_rate
        else:
            # Default to 30 seconds audio
            audio_len = 30 * sampling_rate
        num_audios = mm_counts.get("audio", 0)

        return {"audio": self._get_dummy_audios(length=audio_len, num_audios=num_audios)}

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        feature_extractor = self.info.get_feature_extractor()

        sampling_rate = feature_extractor.sampling_rate
        if hasattr(feature_extractor, "chunk_length"):
            audio_len = feature_extractor.chunk_length * sampling_rate
        else:
            # Default to 30 seconds audio
            audio_len = 30 * sampling_rate
        num_audios = mm_counts.get("audio", 0)

        mm_data = {"audio": self._get_dummy_audios(length=audio_len, num_audios=num_audios)}

        return ProcessorInputs(
            prompt="<|AUDIO|>" * num_audios,
            mm_data=mm_data,
        )


class HiggsAudioDualFFNDecoderLayer(Module):
    """TensorRT-LLM implementation of dual-path FFN decoder layer.
    
    Audio tokens and text tokens go through separate FFN layers while
    sharing the attention layer. This allows for specialized processing
    of audio vs text tokens.
    """

    def __init__(self, config: HiggsAudioConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # Shared attention layer
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, 'num_key_value_heads', config.num_attention_heads),
            max_position_embeddings=config.max_position_embeddings,
            num_layers=config.num_hidden_layers,
            apply_query_key_layer_scaling=False,
            attention_mask_type=None,
            bias=getattr(config, 'attention_bias', False),
            dtype=config.dtype,
            tp_group=None,
            tp_size=1,
            tp_rank=0,
            quant_mode=QuantMode(0),
        )
        
        # Text MLP
        self.mlp = MLP(
            hidden_size=config.hidden_size,
            ffn_hidden_size=getattr(config, 'intermediate_size', config.hidden_size * 4),
            hidden_act=getattr(config, 'hidden_act', 'silu'),
            dtype=config.dtype,
            bias=getattr(config, 'mlp_bias', False),
            tp_group=None,
            tp_size=1,
            quant_mode=QuantMode(0),
        )
        
        # Audio MLP (potentially smaller)
        self.audio_mlp = MLP(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.audio_ffn_intermediate_size,
            hidden_act=getattr(config, 'hidden_act', 'silu'),
            dtype=config.dtype,
            bias=getattr(config, 'mlp_bias', False),
            tp_group=None,
            tp_size=1,
            quant_mode=QuantMode(0),
        )
        
        # Layer norms
        self.input_layernorm = RmsNorm(
            normalized_shape=config.hidden_size,
            eps=getattr(config, 'rms_norm_eps', 1e-5),
            dtype=config.dtype
        )
        
        self.post_attention_layernorm = RmsNorm(
            normalized_shape=config.hidden_size,
            eps=getattr(config, 'rms_norm_eps', 1e-5),
            dtype=config.dtype
        )
        
        self.audio_input_layernorm = RmsNorm(
            normalized_shape=config.hidden_size,
            eps=getattr(config, 'rms_norm_eps', 1e-5),
            dtype=config.dtype
        )
        
        self.audio_post_attention_layernorm = RmsNorm(
            normalized_shape=config.hidden_size,
            eps=getattr(config, 'rms_norm_eps', 1e-5),
            dtype=config.dtype
        )

    def forward(self,
                hidden_states: Tensor,
                attention_mask: Optional[Tensor] = None,
                use_cache: bool = False,
                kv_cache_params: Optional[KeyValueCacheParams] = None,
                attention_params: Optional[AttentionParams] = None,
                audio_token_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass for dual FFN decoder layer.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            use_cache: Whether to use KV cache
            kv_cache_params: KV cache parameters
            attention_params: Attention parameters
            audio_token_mask: Boolean mask indicating audio tokens
            
        Returns:
            Output hidden states
        """
        
        residual = hidden_states
        
        # Apply appropriate input layer norm based on token type
        if audio_token_mask is not None:
            # Apply different layer norms for audio vs text tokens
            text_states = self.input_layernorm(hidden_states)
            audio_states = self.audio_input_layernorm(hidden_states)
            # Mix based on audio token mask
            hidden_states = audio_token_mask * audio_states + (1 - audio_token_mask) * text_states
        else:
            hidden_states = self.input_layernorm(hidden_states)
        
        # Shared attention layer
        attention_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
        )
        
        if use_cache:
            attention_output, present_key_value = attention_output
            
        hidden_states = residual + attention_output
        
        # Dual-path FFN
        residual = hidden_states
        
        if audio_token_mask is not None:
            # Separate processing for text and audio tokens
            text_states = self.post_attention_layernorm(hidden_states)
            audio_states = self.audio_post_attention_layernorm(hidden_states)
            
            # Apply appropriate MLPs
            text_output = self.mlp(text_states)
            audio_output = self.audio_mlp(audio_states)
            
            # Mix outputs based on token type
            mlp_output = audio_token_mask * audio_output + (1 - audio_token_mask) * text_output
        else:
            # Default to text processing if no audio mask provided
            hidden_states = self.post_attention_layernorm(hidden_states)
            mlp_output = self.mlp(hidden_states)
            
        hidden_states = residual + mlp_output
        
        if use_cache:
            return hidden_states, present_key_value
        return hidden_states
        outputs = (hidden_states, None)

        return outputs


class HiggsAudioForCausalLM(DecoderModelForCausalLM):
    """TensorRT-LLM implementation of Higgs Audio multimodal model."""

    def __init__(self, config: HiggsAudioConfig):
        # Initialize the transformer component
        transformer = HiggsAudioTransformer(config)
        
        # Initialize language model head
        vocab_size_padded = config.vocab_size
        if hasattr(config, 'mapping') and config.mapping is not None:
            if config.mapping.tp_size > 1:
                vocab_size_padded = vocab_size_padded // config.mapping.tp_size
        
        lm_head = ColumnLinear(
            in_features=config.hidden_size,
            out_features=vocab_size_padded,
            bias=False,
            dtype=config.dtype,
            tp_group=config.mapping.tp_group if hasattr(config, 'mapping') and config.mapping else None,
            tp_size=config.mapping.tp_size if hasattr(config, 'mapping') and config.mapping else 1,
            gather_output=True,
        )
        
        super().__init__(config, transformer, lm_head)


class HiggsAudioTransformer(Module):
    """TensorRT-LLM transformer component for Higgs Audio model."""

    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.config = config
        
        # Audio encoder
        self.audio_tower = HiggsAudioEncoder(config.audio_encoder_config)
        
        # Audio feature projector
        self.audio_encoder_proj = HiggsAudioFeatureProjector(config)
        
        # Text embedding
        vocab_size_padded = config.vocab_size
        if hasattr(config, 'mapping') and config.mapping is not None:
            if config.mapping.tp_size > 1:
                vocab_size_padded = vocab_size_padded // config.mapping.tp_size
                
        self.embed_tokens = Embedding(
            num_embeddings=vocab_size_padded,
            embedding_dim=config.hidden_size,
            dtype=config.dtype,
            tp_group=config.mapping.tp_group if hasattr(config, 'mapping') and config.mapping else None,
            tp_size=config.mapping.tp_size if hasattr(config, 'mapping') and config.mapping else 1,
            sharding_dim=0,
            tp_rank=config.mapping.tp_rank if hasattr(config, 'mapping') and config.mapping else 0,
        )
        
        # Audio codebook embeddings for audio generation
        self.audio_codebook_size = config.audio_codebook_size + 2  # +2 for BOS/EOS
        self.audio_num_codebooks = config.audio_num_codebooks
        
        self.audio_codebook_embeddings = Embedding(
            num_embeddings=config.audio_num_codebooks * self.audio_codebook_size,
            embedding_dim=config.hidden_size,
            dtype=config.dtype
        )
        
        # Decoder layers
        if config.audio_adapter_type == "dual_ffn":
            self.layers = DecoderLayerList(HiggsAudioDualFFNDecoderLayer, config)
        else:
            # Use standard decoder layers for other adapter types
            from ..llama.model import LlamaDecoderLayer
            self.layers = DecoderLayerList(LlamaDecoderLayer, config)
            
        # Final layer norm
        self.ln_f = RmsNorm(
            normalized_shape=config.hidden_size,
            eps=getattr(config, 'rms_norm_eps', 1e-5),
            dtype=config.dtype
        )
        
        # Audio output projector
        self.audio_decoder_proj = HiggsAudioDecoderProjector(config)
        
    def forward(self,
                input_ids: Tensor,
                position_ids: Optional[Tensor] = None,
                use_cache: bool = False,
                attention_mask: Optional[Tensor] = None,
                kv_cache_params: Optional[KeyValueCacheParams] = None,
                attention_params: Optional[AttentionParams] = None,
                audio_features: Optional[Tensor] = None,
                audio_feature_attention_mask: Optional[Tensor] = None,
                audio_out_ids: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass for Higgs Audio transformer.
        
        Args:
            input_ids: Input token IDs
            position_ids: Position IDs for tokens
            use_cache: Whether to use KV cache
            attention_mask: Attention mask
            kv_cache_params: KV cache parameters
            attention_params: Attention parameters
            audio_features: Encoded audio features (for comprehension)
            audio_feature_attention_mask: Attention mask for audio features
            audio_out_ids: Audio output token IDs (for generation)
            
        Returns:
            Hidden states
        """
        
        # Process input embeddings
        if audio_features is not None:
            # Audio comprehension mode - process audio features
            audio_embeddings = self.audio_tower(
                input_features=audio_features,
                attention_mask=audio_feature_attention_mask,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None
            )
            
            # Project audio features to text hidden size
            audio_embeddings = self.audio_encoder_proj(audio_embeddings)
            
            # Get text embeddings
            text_embeddings = self.embed_tokens(input_ids)
            
            # Merge audio and text embeddings based on input_ids
            # This would need custom logic to determine where audio tokens are
            hidden_states = text_embeddings  # Simplified for now
            
        elif audio_out_ids is not None:
            # Audio generation mode - use audio codebook embeddings
            batch_size, seq_len, num_codebooks = shape(audio_out_ids, 0), shape(audio_out_ids, 1), shape(audio_out_ids, 2)
            
            # Flatten audio_out_ids to (batch_size, seq_len * num_codebooks)
            flat_audio_ids = view(audio_out_ids, [batch_size, seq_len * num_codebooks])
            
            # Get audio codebook embeddings
            audio_embeddings = self.audio_codebook_embeddings(flat_audio_ids)
            
            # Reshape back to (batch_size, seq_len, num_codebooks, hidden_size)
            audio_embeddings = view(audio_embeddings, [batch_size, seq_len, num_codebooks, self.config.hidden_size])
            
            # Combine codebook embeddings (e.g., sum or average)
            hidden_states = audio_embeddings.sum(dim=2)  # Sum across codebooks
            
        else:
            # Text-only mode
            hidden_states = self.embed_tokens(input_ids)
        
        # Pass through decoder layers
        hidden_states = self.layers(
            hidden_states=hidden_states,
            use_cache=use_cache,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
        )
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        return hidden_states


class HiggsAudioTRTRunner:
    """TensorRT-LLM inference wrapper for HiggsAudio using ModelRunnerCpp.
    
    This class provides an interface to run inference on the HiggsAudio model
    using TensorRT-LLM's optimized runtime.
    """

    def __init__(
        self,
        *,
        engine_dir: str,
        tokenizer_dir: str,
        gpu_id: int = 0,
        audio_engine_path: Optional[str] = None,
        num_beams: int = 1,
        use_py_session: bool = False,
        debug_mode: bool = False,
        lora_dir: Optional[str] = None,
        lora_ckpt_source: Optional[str] = None,
        gpu_weights_percent: Optional[float] = None,
        max_new_tokens: int = 64,
        enable_context_fmha_fp32_acc: bool = False,
    ) -> None:
        """Initialize the TensorRT-LLM runner for HiggsAudio.
        
        Args:
            engine_dir: Directory containing the TensorRT engine
            tokenizer_dir: Directory containing the tokenizer
            gpu_id: GPU device ID
            audio_engine_path: Optional path to audio encoder engine
            num_beams: Number of beams for beam search
            use_py_session: Whether to use Python session
            debug_mode: Enable debug mode
            lora_dir: Directory for LoRA weights
            lora_ckpt_source: Source for LoRA checkpoint
            gpu_weights_percent: Percentage of weights on GPU
            max_new_tokens: Maximum new tokens to generate
            enable_context_fmha_fp32_acc: Enable FP32 accumulation in FMHA
        """
        self.engine_dir = engine_dir
        self.tokenizer_dir = tokenizer_dir
        self.audio_engine_path = audio_engine_path
        self.num_beams = num_beams
        self.use_py_session = use_py_session
        self.debug_mode = debug_mode
        self.lora_dir = lora_dir
        self.lora_ckpt_source = lora_ckpt_source
        self.gpu_weights_percent = gpu_weights_percent
        self.max_new_tokens = max_new_tokens
        self.enable_context_fmha_fp32_acc = enable_context_fmha_fp32_acc

        # Set up device
        self.gpu_device = torch.device("cuda", gpu_id)
        torch.cuda.set_device(self.gpu_device)

        # Initialize runner components
        self.session_audio: Optional[Session] = None
        self.runner: Optional[ModelRunner] = None
        self.tokenizer = None
        self.processor = None
        self.sampling_config: Optional[TRTSamplingConfig] = None
        self.model_config: Optional[ModelConfig] = None
        self.max_seq_len: int = 0
        self.hf_config = None

        # Load components
        self._load_tokenizer()
        self._load_model_config()
        self._setup_runner()
        if audio_engine_path:
            self._setup_audio_session()

    def _load_tokenizer(self):
        """Load the tokenizer from the specified directory."""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_dir,
                trust_remote_code=True,
                use_fast=False
            )
            trtllm_logger.info(f"Loaded tokenizer from {self.tokenizer_dir}")
        except Exception as e:
            trtllm_logger.error(f"Failed to load tokenizer: {e}")
            raise

    def _load_model_config(self):
        """Load the model configuration."""
        try:
            config_path = os.path.join(self.engine_dir, "config.json")
            if os.path.exists(config_path):
                self.model_config = ModelConfig.from_json_file(config_path)
            else:
                # Fallback to default config
                self.model_config = ModelConfig()
            
            # Also load HF config for compatibility
            hf_config_path = os.path.join(self.tokenizer_dir, "config.json")
            if os.path.exists(hf_config_path):
                with open(hf_config_path, 'r') as f:
                    self.hf_config = json.load(f)
                    
            trtllm_logger.info("Loaded model configuration")
        except Exception as e:
            trtllm_logger.error(f"Failed to load model config: {e}")
            raise

    def _setup_runner(self):
        """Set up the TensorRT-LLM model runner."""
        try:
            if PYTHON_BINDINGS and not self.use_py_session:
                self.runner = ModelRunnerCpp.from_dir(
                    engine_dir=self.engine_dir,
                    lora_dir=self.lora_dir,
                    rank=tensorrt_llm.mpi_rank(),
                    debug_mode=self.debug_mode,
                    lora_ckpt_source=self.lora_ckpt_source,
                    gpu_weights_percent=self.gpu_weights_percent,
                    enable_context_fmha_fp32_acc=self.enable_context_fmha_fp32_acc
                )
            else:
                self.runner = ModelRunner.from_dir(
                    engine_dir=self.engine_dir,
                    lora_dir=self.lora_dir,
                    rank=tensorrt_llm.mpi_rank(),
                    debug_mode=self.debug_mode,
                    lora_ckpt_source=self.lora_ckpt_source,
                    gpu_weights_percent=self.gpu_weights_percent
                )
            
            # Set up sampling config
            self.sampling_config = TRTSamplingConfig(
                end_id=self.tokenizer.eos_token_id if self.tokenizer else None,
                pad_id=self.tokenizer.pad_token_id if self.tokenizer else None,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens
            )
            
            trtllm_logger.info("Successfully set up TensorRT-LLM runner")
        except Exception as e:
            trtllm_logger.error(f"Failed to set up runner: {e}")
            raise

    def _setup_audio_session(self):
        """Set up the audio encoder session if path is provided."""
        if not self.audio_engine_path:
            return
            
        try:
            self.session_audio = Session.from_serialized_engine(
                self.audio_engine_path
            )
            trtllm_logger.info(f"Loaded audio encoder from {self.audio_engine_path}")
        except Exception as e:
            trtllm_logger.warning(f"Failed to load audio encoder: {e}")
            self.session_audio = None

    def generate(
        self,
        input_text: str,
        audio_data: Optional[np.ndarray] = None,
        max_new_tokens: Optional[int] = None,
        **generation_kwargs
    ) -> str:
        """Generate text/audio response from input text and optional audio.
        
        Args:
            input_text: Input text prompt
            audio_data: Optional audio input as numpy array
            max_new_tokens: Maximum new tokens to generate
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        if not self.runner:
            raise RuntimeError("Runner not initialized")

        # Tokenize input
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        
        # Process audio if provided
        if audio_data is not None and self.session_audio:
            # Run audio encoder
            audio_features = self._encode_audio(audio_data)
            # TODO: Integrate audio features into prompt
        
        # Set up generation parameters
        if max_new_tokens is not None:
            self.sampling_config.max_new_tokens = max_new_tokens
            
        # Run generation
        with torch.no_grad():
            outputs = self.runner.generate(
                batch_input_ids=[input_ids.squeeze(0).tolist()],
                sampling_config=self.sampling_config,
                **generation_kwargs
            )
        
        # Decode output
        output_ids = outputs['output_ids'][0]
        generated_text = self.tokenizer.decode(
            output_ids, 
            skip_special_tokens=True
        )
        
        return generated_text

    def _encode_audio(self, audio_data: np.ndarray) -> torch.Tensor:
        """Encode audio using the audio encoder session.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Encoded audio features
        """
        if not self.session_audio:
            raise RuntimeError("Audio session not initialized")
            
        # Prepare audio input tensor
        audio_tensor = torch.from_numpy(audio_data).to(self.gpu_device)
        
        # Run audio encoder
        inputs = {
            'input_features': audio_tensor
        }
        
        outputs = self.session_audio.run(inputs)
        return outputs['audio_features']  # Assuming this is the output name

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'session_audio') and self.session_audio:
            del self.session_audio
        if hasattr(self, 'runner') and self.runner:
            del self.runner
        self,
        *,
        engine_dir: str,
        tokenizer_dir: str,
        gpu_id: int = 0,
        audio_engine_path: Optional[str] = None,
        num_beams: int = 1,
        use_py_session: bool = False,
        debug_mode: bool = False,
        lora_dir: Optional[str] = None,
        lora_ckpt_source: Optional[str] = None,
        gpu_weights_percent: Optional[float] = None,
        max_new_tokens: int = 64,
        enable_context_fmha_fp32_acc: bool = False,
    ) -> None:
        """Initialize the TensorRT-LLM runner for HiggsAudio.
        
        Args:
            engine_dir: Directory containing the TensorRT engine
            tokenizer_dir: Directory containing the tokenizer
            gpu_id: GPU device ID
            audio_engine_path: Optional path to audio encoder engine
            num_beams: Number of beams for beam search
            use_py_session: Whether to use Python session
            debug_mode: Enable debug mode
            lora_dir: Directory for LoRA weights
            lora_ckpt_source: Source for LoRA checkpoint
            gpu_weights_percent: Percentage of weights on GPU
            max_new_tokens: Maximum new tokens to generate
            enable_context_fmha_fp32_acc: Enable FP32 accumulation in FMHA
        """
        self.engine_dir = engine_dir
        self.tokenizer_dir = tokenizer_dir
        self.audio_engine_path = audio_engine_path
        self.num_beams = num_beams
        self.use_py_session = use_py_session
        self.debug_mode = debug_mode
        self.lora_dir = lora_dir
        self.lora_ckpt_source = lora_ckpt_source
        self.gpu_weights_percent = gpu_weights_percent
        self.max_new_tokens = max_new_tokens
        self.enable_context_fmha_fp32_acc = enable_context_fmha_fp32_acc

        # Set up device
        self.gpu_device = torch.device("cuda", gpu_id)
        torch.cuda.set_device(self.gpu_device)

        # Initialize runner components
        self.session_audio: Optional[Session] = None
        self.runner: Optional[ModelRunner] = None
        self.tokenizer = None
        self.processor = None
        self.sampling_config: Optional[TRTSamplingConfig] = None
        self.model_config: Optional[ModelConfig] = None
        self.max_seq_len: int = 0
        self.hf_config = None

        # Load components
        self._load_tokenizer()
        self._load_model_config()
        self._setup_runner()
        if audio_engine_path:
            self._setup_audio_session()

    def _load_tokenizer(self):
        """Load the tokenizer from the specified directory."""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_dir,
                trust_remote_code=True,
                use_fast=False
            )
            trtllm_logger.info(f"Loaded tokenizer from {self.tokenizer_dir}")
        except Exception as e:
            trtllm_logger.error(f"Failed to load tokenizer: {e}")
            raise

    def _load_model_config(self):
        """Load the model configuration."""
        try:
            config_path = os.path.join(self.engine_dir, "config.json")
            if os.path.exists(config_path):
                self.model_config = ModelConfig.from_json_file(config_path)
            else:
                # Fallback to default config
                self.model_config = ModelConfig()
            
            # Also load HF config for compatibility
            hf_config_path = os.path.join(self.tokenizer_dir, "config.json")
            if os.path.exists(hf_config_path):
                with open(hf_config_path, 'r') as f:
                    self.hf_config = json.load(f)
                    
            trtllm_logger.info("Loaded model configuration")
        except Exception as e:
            trtllm_logger.error(f"Failed to load model config: {e}")
            raise

    def _setup_runner(self):
        """Set up the TensorRT-LLM model runner."""
        try:
            if PYTHON_BINDINGS and not self.use_py_session:
                self.runner = ModelRunnerCpp.from_dir(
                    engine_dir=self.engine_dir,
                    lora_dir=self.lora_dir,
                    rank=tensorrt_llm.mpi_rank(),
                    debug_mode=self.debug_mode,
                    lora_ckpt_source=self.lora_ckpt_source,
                    gpu_weights_percent=self.gpu_weights_percent,
                    enable_context_fmha_fp32_acc=self.enable_context_fmha_fp32_acc
                )
            else:
                self.runner = ModelRunner.from_dir(
                    engine_dir=self.engine_dir,
                    lora_dir=self.lora_dir,
                    rank=tensorrt_llm.mpi_rank(),
                    debug_mode=self.debug_mode,
                    lora_ckpt_source=self.lora_ckpt_source,
                    gpu_weights_percent=self.gpu_weights_percent
                )
            
            # Set up sampling config
            self.sampling_config = TRTSamplingConfig(
                end_id=self.tokenizer.eos_token_id if self.tokenizer else None,
                pad_id=self.tokenizer.pad_token_id if self.tokenizer else None,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens
            )
            
            trtllm_logger.info("Successfully set up TensorRT-LLM runner")
        except Exception as e:
            trtllm_logger.error(f"Failed to set up runner: {e}")
            raise

    def _setup_audio_session(self):
        """Set up the audio encoder session if path is provided."""
        if not self.audio_engine_path:
            return
            
        try:
            self.session_audio = Session.from_serialized_engine(
                self.audio_engine_path
            )
            trtllm_logger.info(f"Loaded audio encoder from {self.audio_engine_path}")
        except Exception as e:
            trtllm_logger.warning(f"Failed to load audio encoder: {e}")
            self.session_audio = None

    def generate(
        self,
        input_text: str,
        audio_data: Optional[np.ndarray] = None,
        max_new_tokens: Optional[int] = None,
        **generation_kwargs
    ) -> str:
        """Generate text/audio response from input text and optional audio.
        
        Args:
            input_text: Input text prompt
            audio_data: Optional audio input as numpy array
            max_new_tokens: Maximum new tokens to generate
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        if not self.runner:
            raise RuntimeError("Runner not initialized")

        # Tokenize input
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        
        # Process audio if provided
        if audio_data is not None and self.session_audio:
            # Run audio encoder
            audio_features = self._encode_audio(audio_data)
            # TODO: Integrate audio features into prompt
        
        # Set up generation parameters
        if max_new_tokens is not None:
            self.sampling_config.max_new_tokens = max_new_tokens
            
        # Run generation
        with torch.no_grad():
            outputs = self.runner.generate(
                batch_input_ids=[input_ids.squeeze(0).tolist()],
                sampling_config=self.sampling_config,
                **generation_kwargs
            )
        
        # Decode output
        output_ids = outputs['output_ids'][0]
        generated_text = self.tokenizer.decode(
            output_ids, 
            skip_special_tokens=True
        )
        
        return generated_text

    def _encode_audio(self, audio_data: np.ndarray) -> torch.Tensor:
        """Encode audio using the audio encoder session.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Encoded audio features
        """
        if not self.session_audio:
            raise RuntimeError("Audio session not initialized")
            
        # Prepare audio input tensor
        audio_tensor = torch.from_numpy(audio_data).to(self.gpu_device)
        
        # Run audio encoder
        inputs = {
            'input_features': audio_tensor
        }
        
        outputs = self.session_audio.run(inputs)
        return outputs['audio_features']  # Assuming this is the output name

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'session_audio') and self.session_audio:
            del self.session_audio
        if hasattr(self, 'runner') and self.runner:
            del self.runner

    def _parse_and_validate_audio_input(self, **kwargs: object) -> Optional[HiggsAudioInputs]:
        audio_features = kwargs.pop("audio_features", None)
        audio_feature_attention_mask = kwargs.pop("audio_feature_attention_mask", None)
        audio_out_ids = kwargs.pop("audio_out_ids", None)
        if audio_features is None and audio_out_ids is None:
            return None
        if audio_features is not None:
            audio_features = _validate_and_reshape_mm_tensor(
                audio_features,
                "audio_features",
                pad_with=0 if not self.use_whisper_tokenizer else None,
            )
            audio_feature_attention_mask = _validate_and_reshape_mm_tensor(
                audio_feature_attention_mask,
                "audio_feature_attention_mask",
                pad_with=0,
            )
            if not isinstance(audio_features, (torch.Tensor, list)):
                raise ValueError(
                    f"Incorrect type of audio input features. Got type: {type(audio_features)}"
                )
        if audio_out_ids is not None:
            audio_out_ids = _validate_and_reshape_mm_tensor(audio_out_ids, "audio_out_ids")
            # audio_out_ids_length = _validate_and_reshape_mm_tensor(
            #     audio_out_ids_length, "audio_out_ids_length")
        return HiggsAudioInputs(
            audio_features=audio_features,
            audio_feature_attention_mask=audio_feature_attention_mask,
            audio_out_ids=audio_out_ids,
        )

    def _process_whisper_audio_input(
        self, audio_features: torch.Tensor, audio_feature_attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        (
            audio_feat_lengths,
            audio_feat_out_lengths,
        ) = self.audio_tower._get_feat_extract_output_lengths(audio_feature_attention_mask.sum(-1))

        batch_size, _, max_mel_seq_len = audio_features.shape
        max_seq_len = (max_mel_seq_len - 2) // 2 + 1
        # Create a sequence tensor of shape (batch_size, max_seq_len)
        seq_range = (
            torch.arange(
                0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device
            )
            .unsqueeze(0)
            .expand(batch_size, max_seq_len)
        )
        lengths_expand = audio_feat_lengths.unsqueeze(-1).expand(batch_size, max_seq_len)
        # Create mask
        padding_mask = seq_range >= lengths_expand

        audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
            batch_size, 1, max_seq_len, max_seq_len
        )
        audio_attention_mask = audio_attention_mask_.to(
            dtype=self.audio_tower.conv1.weight.dtype, device=self.audio_tower.conv1.weight.device
        )
        audio_attention_mask[audio_attention_mask_] = float("-inf")

        audio_outputs = self.audio_tower(audio_features, attention_mask=audio_attention_mask)
        selected_audio_feature = audio_outputs.last_hidden_state
        audio_features = self.audio_encoder_proj(selected_audio_feature)

        num_audios, max_audio_tokens, embed_dim = audio_features.shape
        audio_feat_out_lengths = audio_feat_out_lengths.unsqueeze(1)
        audio_features_mask = (
            torch.arange(max_audio_tokens)
            .expand(num_audios, max_audio_tokens)
            .to(audio_feat_out_lengths.device)
            < audio_feat_out_lengths
        )
        masked_audio_features = audio_features[audio_features_mask].view(-1, embed_dim)

        # Split to tuple of embeddings for individual audio input.
        return torch.split(masked_audio_features, audio_feat_out_lengths.flatten().tolist())

    def _process_audio_input(self, audio_input: HiggsAudioInputs) -> torch.Tensor:
        audio_features = audio_input["audio_features"]
        audio_feature_attention_mask = audio_input["audio_feature_attention_mask"]

        if self.use_whisper_tokenizer:
            return self._process_whisper_audio_input(audio_features, audio_feature_attention_mask)

        audio_features_flattened = audio_features.transpose(1, 0).reshape(
            audio_features.shape[1], -1
        )
        audio_features_embeddings = self._embed_audio_ids(audio_features_flattened)
        audio_features_attention_mask_flattened = audio_feature_attention_mask.flatten()
        masked_audio_features_embeddings = audio_features_embeddings[
            audio_features_attention_mask_flattened
        ]
        audio_features_lens = audio_feature_attention_mask.sum(-1)
        masked_audio_features_embeddings = torch.split(
            masked_audio_features_embeddings, audio_features_lens.tolist()
        )
        return masked_audio_features_embeddings

    def _embed_audio_ids(self, audio_ids):
        """Embed the audio ids

        Args:
            audio_ids: torch.LongTensor of shape (num_codebooks, audio_in_total_length)

        Returns:
            audio_embed: torch.LongTensor of shape (audio_in_total_length, hidden_size)
        """
        codebook_shift = (
            torch.arange(self.audio_num_codebooks, device=audio_ids.device)
            * self.audio_codebook_size
        )
        codebook_shift = codebook_shift.unsqueeze(-1)
        audio_embed = self.audio_codebook_embeddings(audio_ids + codebook_shift)
        audio_embed = torch.sum(audio_embed, dim=0)
        if self.config.use_audio_out_embed_projector:
            audio_embed = self.audio_out_embed_projector(audio_embed)
        return audio_embed

    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return None
        if audio_input["audio_features"] is not None:
            masked_audio_features = self._process_audio_input(audio_input)
        else:
            masked_audio_features = None
        if kwargs.get("audio_out_ids", None) is not None:
            audio_out_ids = kwargs["audio_out_ids"]
            audio_out_flattened = audio_out_ids.transpose(1, 0)
            audio_out_embeddings = self._embed_audio_ids(audio_out_flattened)
            audio_out_embeddings = torch.chunk(audio_out_embeddings, audio_out_ids.shape[0], dim=0)
            if masked_audio_features is not None:
                masked_audio_features.extend(audio_out_embeddings)
            else:
                masked_audio_features = audio_out_embeddings

        return masked_audio_features

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
        attn_metadata: Optional[object] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                [self.config.audio_in_token_idx, self.config.audio_out_token_idx],
            )

        return inputs_embeds

    def get_language_model(self) -> torch.nn.Module:
        """
        Return the underlying language model used for text generation.
        For this architecture, the current module encapsulates the
        core text model (embedding, decoder layers, norm).
        """
        return self

    def get_input_mm_map(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.isin(
            input_ids,
            torch.tensor(
                [self.config.audio_in_token_idx, self.config.audio_out_token_idx],
                device=input_ids.device,
            ),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            # NOTE: In v1, inputs_embeds is always generated at model runner,
            # this condition is for v0 compatibility.
            if inputs_embeds is None:
                multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)
                inputs_embeds = self.get_input_embeddings(input_ids, multimodal_embeddings)
                input_ids = None
            hidden_states = inputs_embeds
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            if isinstance(layer, HiggsAudioDualFFNDecoderLayer):
                hidden_states, _ = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=None,
                )
            else:
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    residual,
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_logits = self.logits_processor(self.text_lm_head, hidden_states, sampling_metadata)
        if self.generate_audio_out_token:
            audio_logits = self.audio_logits_processor(self.audio_lm_head, hidden_states, None)
            audio_logits = audio_logits.view(
                -1, self.audio_num_codebooks, self.audio_codebook_size
            ).float()
        else:
            audio_logits = None
        return text_logits, audio_logits

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
                    and output_token_ids[-1] == self.config.audio_out_bos_token_id
                ) or (
                    len(output_token_ids) == 0
                    and last_prompt_token_id == self.config.audio_out_bos_token_id
                ):
                    # check if the previous token is audio_out_bos. If so, we should always generate <|AUDIO_OUT|>
                    # Start the audio generation mode
                    audio_generation_mode[i] = 1
                elif (
                    len(output_token_ids) > 0
                    and output_token_ids[-1] == self.config.audio_out_token_idx
                ):
                    # Still in the audio generation mode
                    audio_generation_mode[i] = 2

            assert audio_logits is not None
            audio_logits = audio_logits.reshape(-1, self.audio_codebook_size)
            mm_sampling_metadata = self.prepare_mm_sampling_metadata(sampling_metadata)
            next_mm_tokens = self.sampler(audio_logits, mm_sampling_metadata)
            next_mm_tokens.sampled_token_ids = next_mm_tokens.sampled_token_ids.reshape(
                -1, self.audio_num_codebooks
            )

            # Check if we are generating the audio tokens
            for i in range(n_reqs):
                if audio_generation_mode[i] == 1:
                    # Generate start of the audio stream
                    next_tokens.sampled_token_ids[i] = self.config.audio_out_token_idx
                    next_mm_tokens.sampled_token_ids[i] = self.config.audio_stream_bos_id
                elif audio_generation_mode[i] == 2:
                    next_tokens.sampled_token_ids[i] = self.config.audio_out_token_idx
                    # Update the next mm tokens based on the delay pattern
                    num_audio_delay = multimodal_metadata.num_audio_delays[i]
                    num_audio_eos = multimodal_metadata.num_audio_eos[i]

                    # Generate the delayed for the first few tokens
                    if num_audio_delay < self.audio_num_codebooks:
                        next_mm_tokens.sampled_token_ids[i][num_audio_delay:] = (
                            self.config.audio_stream_bos_id
                        )

                    # Generate the eos token for the last few tokens
                    if num_audio_eos < self.audio_num_codebooks:
                        all_eos_indices = torch.where(
                            next_mm_tokens.sampled_token_ids[i] == self.config.audio_stream_eos_id
                        )[0]
                        if all_eos_indices.shape[0] > 0:
                            last_eos_index = all_eos_indices[-1]
                            next_mm_tokens.sampled_token_ids[i][:last_eos_index] = (
                                self.config.audio_stream_eos_id
                            )
                    elif num_audio_eos == self.audio_num_codebooks:
                        # We already generated the last audio token,
                        # so we should just generate the eos token for the text
                        next_tokens.sampled_token_ids[i] = self.config.audio_eos_token_id
                        next_mm_tokens.sampled_token_ids[i] = -1

                else:
                    next_mm_tokens.sampled_token_ids[i] = -1

        return next_tokens, next_mm_tokens

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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.audio_adapter_type == "stack":
                audio_param_names = [
                    "audio_attn",
                    "audio_input_layernorm",
                    "audio_mlp",
                    "audio_post_attention_layernorm",
                ]
                if any(p in name for p in audio_param_names):
                    continue
            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in name:
                    name = name.replace(key_to_modify, new_key)

            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):  # Loading kv cache scales for compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                continue

            if "audio_tower" in name:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)

                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


class HiggsAudioTRTRunner:
    """TensorRT-LLM inference wrapper for HiggsAudio using ModelRunnerCpp.

    - Loads HF tokenizer and an HF processor to expand <|AUDIO|> placeholders.
    - Optionally loads a separate TRT audio encoder plan via Session.
    - Builds prompt_table from audio embeddings and replaces placeholders with
      fake prompt token IDs, similar to Qwen2-Audio.
    """

    def __init__(
        self,
        *,
        engine_dir: str,
        tokenizer_dir: str,
        gpu_id: int = 0,
        audio_engine_path: Optional[str] = None,
        num_beams: int = 1,
        use_py_session: bool = False,
        debug_mode: bool = False,
        lora_dir: Optional[str] = None,
        lora_ckpt_source: Optional[str] = None,
        gpu_weights_percent: Optional[float] = None,
        max_new_tokens: int = 64,
        enable_context_fmha_fp32_acc: bool = False,
    ) -> None:
        self.engine_dir = engine_dir
        self.tokenizer_dir = tokenizer_dir
        self.audio_engine_path = audio_engine_path
        self.num_beams = num_beams
        self.use_py_session = use_py_session
        self.debug_mode = debug_mode
        self.lora_dir = lora_dir
        self.lora_ckpt_source = lora_ckpt_source
        self.gpu_weights_percent = gpu_weights_percent
        self.max_new_tokens = max_new_tokens
        self.enable_context_fmha_fp32_acc = enable_context_fmha_fp32_acc

        self.gpu_device = torch.device("cuda", gpu_id)
        torch.cuda.set_device(self.gpu_device)

        self.session_audio: Optional[Session] = None
        self.runner = None
        self.tokenizer = None
        self.processor = None
        self.sampling_config: Optional[TRTSamplingConfig] = None
        self.model_config: Optional[ModelConfig] = None
        self.max_seq_len: int = 0
        self.hf_config = None  # HF config

        self._init_audio_session_if_any()
        self._init_tokenizer_and_config()
        self._init_runner()

    def _init_audio_session_if_any(self) -> None:
        if not self.audio_engine_path:
            return
        if not os.path.exists(self.audio_engine_path):
            trtllm_logger.warning(
                f"Audio engine path not found: {self.audio_engine_path}. Skipping audio Session."
            )
            return
        trtllm_logger.info(f"Loading audio engine from {self.audio_engine_path}")
        with open(self.audio_engine_path, "rb") as f:
            engine_buffer = f.read()
        self.session_audio = Session.from_serialized_engine(engine_buffer)

    def _read_engine_config(self) -> dict:
        config_path = os.path.join(self.engine_dir, "config.json")
        with open(config_path, "r") as f:
            return json.load(f)

    def _init_tokenizer_and_config(self) -> None:
        # HF config and tokenizer/processor
        self.hf_config = AutoConfig.from_pretrained(self.tokenizer_dir, trust_remote_code=True)
        from transformers import AutoTokenizer  # local import

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_dir, legacy=False, trust_remote_code=True
        )
        self.processor = cached_get_processor(
            self.tokenizer,
            audio_stream_bos_id=getattr(self.hf_config, "audio_stream_bos_id", None),
            audio_stream_eos_id=getattr(self.hf_config, "audio_stream_eos_id", None),
        )

    def _str_to_torch_dtype(self, s: str) -> torch.dtype:
        s = s.lower()
        if s in ("float16", "fp16", "half"):  # common names
            return torch.float16
        if s in ("bfloat16", "bf16"):
            return torch.bfloat16
        if s in ("float32", "fp32"):
            return torch.float32
        return torch.float16

    def _init_runner(self) -> None:
        cfg = self._read_engine_config()
        self.max_seq_len = cfg["build_config"].get(
            "max_seq_len", cfg["build_config"].get("max_seq_length", 4096)
        )
        assert self.max_seq_len > 0

        gen_config_path = os.path.join(self.tokenizer_dir, "generation_config.json")
        if os.path.exists(gen_config_path):
            with open(gen_config_path, "r") as f:
                gen_cfg = json.load(f)
            top_k = gen_cfg.get("top_k", 0)
            top_p = gen_cfg.get("top_p", 1.0)
        else:
            top_k, top_p = 0, 1.0

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if eos_token_id is None:
            eos_token_id = pad_token_id

        use_gpt_attention_plugin = cfg["build_config"]["plugin_config"].get(
            "gpt_attention_plugin", True
        )
        remove_input_padding = cfg["build_config"]["plugin_config"].get(
            "remove_input_padding", True
        )
        dtype = cfg["pretrained_config"].get("dtype", "float16")
        tp_size = cfg["pretrained_config"]["mapping"].get("tp_size", 1)
        pp_size = cfg["pretrained_config"]["mapping"].get("pp_size", 1)
        world_size = tp_size * pp_size
        if hasattr(tensorrt_llm, "mpi_world_size"):
            assert world_size == tensorrt_llm.mpi_world_size()
        num_heads = cfg["pretrained_config"].get("num_attention_heads", 0)
        hidden_size = cfg["pretrained_config"].get("hidden_size", 0)
        vocab_size = cfg["pretrained_config"].get("vocab_size", 0)
        num_layers = cfg["pretrained_config"].get("num_hidden_layers", 0)
        num_kv_heads = cfg["pretrained_config"].get("num_key_value_heads", num_heads)
        kv_cache_type = KVCacheType.CONTINUOUS
        tokens_per_block = cfg["build_config"]["plugin_config"].get("tokens_per_block", 64)
        max_prompt_embedding_table_size = cfg["build_config"].get(
            "max_prompt_embedding_table_size", 0
        )
        quant_algo = cfg["pretrained_config"].get("quantization", {}).get("quant_algo", None)
        kv_quant_algo = (
            cfg["pretrained_config"].get("quantization", {}).get("kv_cache_quant_algo", None)
        )
        quant_mode = QuantMode.from_quant_algo(quant_algo, kv_quant_algo)

        runtime_rank = getattr(tensorrt_llm, "mpi_rank", lambda: 0)()

        self.model_config = ModelConfig(
            max_batch_size=cfg["build_config"].get("max_batch_size", 1),
            num_heads=num_heads // max(1, world_size),
            num_kv_heads=num_kv_heads,
            hidden_size=hidden_size // max(1, world_size),
            vocab_size=vocab_size,
            num_layers=num_layers,
            gpt_attention_plugin=use_gpt_attention_plugin,
            kv_cache_type=kv_cache_type,
            tokens_per_block=tokens_per_block,
            remove_input_padding=remove_input_padding,
            dtype=dtype,
            quant_mode=quant_mode,
            max_prompt_embedding_table_size=max_prompt_embedding_table_size,
            max_beam_width=self.num_beams,
        )
        self.sampling_config = TRTSamplingConfig(
            end_id=eos_token_id,
            pad_id=pad_token_id,
            num_beams=self.num_beams,
            top_k=top_k,
            top_p=top_p,
            temperature=1.0,
        )

        runner_cls = ModelRunner if self.use_py_session else ModelRunnerCpp
        runner_kwargs = dict(
            engine_dir=self.engine_dir,
            lora_dir=self.lora_dir,
            rank=runtime_rank,
            debug_mode=self.debug_mode,
            lora_ckpt_source=self.lora_ckpt_source or "hf",
            gpu_weights_percent=self.gpu_weights_percent or 1.0,
            max_output_len=self.max_new_tokens,
            enable_context_fmha_fp32_acc=self.enable_context_fmha_fp32_acc,
        )
        if not self.use_py_session:
            runner_kwargs.update(
                is_enc_dec=False,
                max_batch_size=self.model_config.max_batch_size,
                max_input_len=self.max_seq_len - self.max_new_tokens,
                max_beam_width=self.model_config.max_beam_width,
                device_ids=[self.gpu_device.index],
            )
        self.runner = runner_cls.from_dir(**runner_kwargs)

    def _trt_dtype_to_torch(self, dtype: "trt.DataType") -> torch.dtype:
        if dtype == trt.float16:
            return torch.float16
        if dtype == trt.float32:
            return torch.float32
        if dtype == trt.int32:
            return torch.int32
        if hasattr(trt, "bfloat16") and dtype == trt.bfloat16:
            return torch.bfloat16
        return torch.float16

    def _run_audio_encoder(self, input_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert self.session_audio is not None, "Audio Session is not initialized"
        device = self.gpu_device
        stream = torch.cuda.current_stream(device=device)
        # Names here assume an audio encoder with inputs: "input" and "mask" and output: "output"
        inputs_info = [
            TensorInfo("input", trt.DataType.FLOAT, tuple(input_features.shape)),
            TensorInfo("mask", trt.DataType.HALF, tuple(mask.shape)),
        ]
        out_info = self.session_audio.infer_shapes(inputs_info)
        outputs = {
            t.name: torch.empty(
                tuple(t.shape), dtype=self._trt_dtype_to_torch(t.dtype), device=device
            )
            for t in out_info
        }
        ok = self.session_audio.run(
            {"input": input_features.to(device).float(), "mask": mask.to(device)},
            outputs,
            stream.cuda_stream,
        )
        stream.synchronize()
        if not ok:
            raise RuntimeError("Audio session execution failed")
        # Prefer output tensor named 'output' if present
        if "output" in outputs:
            return outputs["output"]
        # Otherwise return the first tensor
        return next(iter(outputs.values()))

    def _ptuning_setup(
        self,
        prompt_table: Optional[torch.Tensor],
        dtype_str: str,
        hidden_size: int,
        tasks: Optional[str],
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.gpu_device
        if prompt_table is not None:
            # Ensure dtype matches engine
            prompt_table = prompt_table.to(device=device, dtype=self._str_to_torch_dtype(dtype_str))
            task_vocab_size = torch.tensor(
                [prompt_table.shape[0]], dtype=torch.int32, device=device
            )
        else:
            prompt_table = torch.empty([1, hidden_size], device=device)
            task_vocab_size = torch.zeros([1], device=device)

        if tasks is not None:
            tasks_tensor = torch.tensor(
                [int(t) for t in tasks.split(",")], dtype=torch.int32, device=device
            )
            assert tasks_tensor.shape[0] == input_ids.shape[0]
        else:
            tasks_tensor = torch.zeros([input_ids.size(0)], dtype=torch.int32, device=device)
        return prompt_table, tasks_tensor, task_vocab_size

    def _feat_lengths(self, input_lengths: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths

    @torch.inference_mode()
    def infer(
        self,
        input_text: str,
        audios: List[np.ndarray],
        audio_ids: Optional[List[int]] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        assert input_text, "input_text must be provided"
        cpu = torch.device("cpu")

        # Process text+audio with HF processor (Whisper path expected)
        feature_extractor = getattr(self.processor, "feature_extractor", None)
        inputs: BatchFeature = self.processor(
            text=input_text,
            audio=audios,
            return_tensors="pt",
            padding=True,
            sampling_rate=getattr(feature_extractor, "sampling_rate", 16000),
        )
        inputs = inputs.to(cpu)
        input_ids = inputs.input_ids

        prompt_table = None
        extra_ids_list: Optional[List[List[int]]] = None

        if hasattr(inputs, "input_features") and inputs.input_features is not None:
            input_features = inputs.input_features
            feature_attention_mask = inputs.audio_feature_attention_mask

            audio_feat_lengths, num_audio_tokens = self._feat_lengths(
                feature_attention_mask.sum(-1)
            )
            bsz, _, max_mel_seq_len = input_features.shape
            max_seq_len = (max_mel_seq_len - 2) // 2 + 1
            seq_range = (
                torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype)
                .unsqueeze(0)
                .expand(bsz, max_seq_len)
            )
            lengths_expand = audio_feat_lengths.unsqueeze(1).expand(bsz, max_seq_len)
            padding_mask = seq_range >= lengths_expand
            audio_attn_mask_ = padding_mask.view(bsz, 1, 1, max_seq_len).expand(
                bsz, 1, max_seq_len, max_seq_len
            )
            audio_attn_mask = audio_attn_mask_.to(dtype=torch.float16)
            audio_attn_mask[audio_attn_mask_] = float("-inf")

            if self.session_audio is None:
                raise RuntimeError("audio_engine_path not provided; TRT audio encoder required")
            audio_features = self._run_audio_encoder(input_features, audio_attn_mask)

            # Collapse to valid frames and build prompt_table
            num_audios, max_audio_tokens, embed_dim = audio_features.shape
            audio_features_mask = torch.arange(max_audio_tokens).expand(
                num_audios, max_audio_tokens
            ) < num_audio_tokens.unsqueeze(1)
            masked_audio_features = audio_features[audio_features_mask].view(-1, embed_dim)

            vocab_size = (
                self.model_config.vocab_size if self.model_config else self.tokenizer.vocab_size
            )
            special_audio_token_idx = getattr(self.hf_config, "audio_in_token_idx", None)
            if special_audio_token_idx is None:
                special_audio_token_idx = self.tokenizer.get_vocab().get("<|AUDIO|>")
            assert special_audio_token_idx is not None, "audio_in_token_idx not found"
            batch_indices, audio_indices = torch.where(input_ids == special_audio_token_idx)
            fake_prompt_ids = torch.arange(vocab_size, vocab_size + masked_audio_features.shape[0])
            input_ids[batch_indices, audio_indices] = fake_prompt_ids

            input_ids_cuda = input_ids.to(dtype=torch.int32, device=self.gpu_device)
            dtype_str = self.model_config.dtype if self.model_config else "float16"
            prompt_table, tasks, task_vocab_size = self._ptuning_setup(
                masked_audio_features, dtype_str, embed_dim, None, input_ids_cuda
            )

            if audio_ids is None:
                audio_ids = list(range(1, num_audio_tokens.size(0) + 1))
            for i in audio_ids:
                assert isinstance(i, int) and i > 0
            extra_ids = torch.zeros_like(input_ids, dtype=torch.int64)
            seq_extra_ids = torch.cat(
                [
                    torch.full((n,), audio_ids[i], dtype=torch.int64)
                    for i, n in enumerate(num_audio_tokens)
                ]
            )
            extra_ids[batch_indices, audio_indices] = seq_extra_ids
            extra_ids_list = extra_ids.tolist()
        else:
            input_ids_cuda = input_ids.to(dtype=torch.int32, device=self.gpu_device)
            dtype_str = self.model_config.dtype if self.model_config else "float16"
            hidden_size = self.model_config.hidden_size if self.model_config else 4096
            prompt_table, tasks, task_vocab_size = self._ptuning_setup(
                None, dtype_str, hidden_size, None, input_ids_cuda
            )
            extra_ids_list = torch.zeros_like(input_ids, dtype=torch.int64).tolist()

        # Sampling overrides
        sc_kwargs = {}
        if top_k is not None:
            sc_kwargs["top_k"] = top_k
        if top_p is not None:
            sc_kwargs["top_p"] = top_p
        if temperature is not None:
            sc_kwargs["temperature"] = temperature

        # Generate
        input_ids_cuda = input_ids.to(dtype=torch.int32, device=self.gpu_device)
        max_input_length = int(input_ids_cuda.size(1))
        max_new_tokens = min(self.max_new_tokens, self.max_seq_len - max_input_length)

        # ModelRunnerCpp expects a list of 1D tensors per request
        batch_input_ids_list = list(input_ids_cuda)

        outputs = self.runner.generate(
            batch_input_ids=batch_input_ids_list,
            max_new_tokens=max_new_tokens,
            end_id=self.sampling_config.end_id
            if self.sampling_config
            else self.tokenizer.pad_token_id,
            pad_id=self.sampling_config.pad_id
            if self.sampling_config
            else self.tokenizer.pad_token_id,
            num_beams=self.num_beams,
            return_dict=True,
            output_sequence_lengths=True,
            prompt_table=prompt_table if prompt_table is not None else None,
            prompt_tasks="0" if prompt_table is not None else None,
            input_token_extra_ids=extra_ids_list if extra_ids_list is not None else None,
            **sc_kwargs,
        )
        output_ids = outputs["output_ids"]

        # Decode first beam for the first sample
        out = output_ids[0][0, max_input_length:].tolist()
        outputs_text = self.tokenizer.decode(out, skip_special_tokens=True)
        return outputs_text
