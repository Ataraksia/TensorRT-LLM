# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main HiggsAudio model implementation for TensorRT-LLM.

HiggsAudio is a multimodal TTS model that combines Llama-3.2-3B text generation
with audio processing through a novel DualFFN architecture. The model processes:
- Text input through standard transformer layers
- Audio input through Whisper-based encoder
- Dual-headed output for simultaneous text and 8-codebook audio generation

The model operates in different generation modes:
- AUDIO_INIT: Audio generation initialization
- AUDIO_IN_PROGRESS: Active audio generation with RVQ coordination
"""

import base64
import inspect
import json
import os
import uuid
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from pydantic import BaseModel
from transformers import AutoTokenizer

from tensorrt_llm._common import default_net
from tensorrt_llm._utils import Union, pad_vocab_size
from tensorrt_llm.functional import Tensor, gather_last_token_logits, gelu, recv
from tensorrt_llm.layers import (
    MLP,
    Attention,
    AttentionMaskType,
    ColumnLinear,
    Conv1d,
    Embedding,
    FusedGatedMLP,
    LayerNorm,
    PositionEmbeddingType,
    RmsNorm,
)
from tensorrt_llm.layers.attention import KeyValueCacheParams
from tensorrt_llm.models.modeling_utils import DecoderLayerList, DecoderModelForCausalLM
from tensorrt_llm.module import Module, ModuleList
from tensorrt_llm.runtime import LogitsProcessor

from .config import HiggsAudioConfig

# Prefer the upstream boson multimodal tokenizer implementation when available.
try:
    from boson_multimodal.higgs_audio_tokenizer import (
        HiggsAudioTokenizer as BosonHiggsAudioTokenizer,
    )
except Exception:  # pragma: no cover - optional dependency
    BosonHiggsAudioTokenizer = None  # type: ignore


class ChatCompletionAudio(BaseModel):
    id: str
    data: str
    expires_at: int
    transcript: str


class HiggsAudioTokenizer(Module):
    """TensorRT-LLM-friendly wrapper around the Higgs Audio tokenizer backend.

    Exposes a stable encode/decode API and basic properties used by the TTS
    pipeline. Internally delegates to the upstream boson_multimodal model.
    """

    def __init__(
        self, tokenizer_dir: str, config: Optional[HiggsAudioConfig] = None, device: str = "cuda:0"
    ):
        super().__init__()
        self.config = config
        self.tokenizer_dir = tokenizer_dir
        self._device = device
        self.audio_tokenizer_model = load_higgs_audio_tokenizer(self.tokenizer_dir, device=device)

        # Lift common properties for quick access
        self._tps = getattr(self.audio_tokenizer_model, "frame_rate")
        self._sampling_rate = getattr(self.audio_tokenizer_model, "sample_rate")
        self._num_codebooks = getattr(self.audio_tokenizer_model, "n_q")
        self._codebook_size = getattr(self.audio_tokenizer_model, "quantizer_dim")

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

    def encode(
        self, audio_path_or_wv, sr=None, loudness_normalize=False, loudness_threshold=-23.0
    ) -> Dict:
        """Encodes audio into a sequence of token codes.

        Args:
            audio_path_or_wv: The audio waveform to encode.
            sr: The sample rate of the audio.
            loudness_normalize: whether to normalize loudness
            loudness_threshold: loudness threshold for normalization

        Returns:
            A dictionary containing at minimum:
            - "codes": np.ndarray or torch.Tensor shaped [num_codebooks, T]
            - "tps": frames-per-second for the codes (e.g., 25)
            - "sample_rate": original sample rate
            - additional backend-specific metadata
        """
        return self.audio_tokenizer_model.encode(
            audio_path_or_wv, sr, loudness_normalize, loudness_threshold
        )

    def decode(self, vq_code, return_cuda_tensor=False):
        """Decodes a sequence of token codes back into an audio waveform.

        Args:
            vq_code: The token codes to decode. Shape (num_codebooks, total_length)
            return_cuda_tensor: whether to return cuda tensor

        Returns:
            The decoded audio waveform.
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


def token2wav(
    token: np.ndarray,
    audio_chunk_size: int,
    audio_tokenizer: HiggsAudioTokenizer,
    audio_codebook_size: int,
    samples_per_token: int,
    audio_num_codebooks: int,
    audio_stream_bos_id: int,
    audio_stream_eos_id: int,
    fade_out_audio: Optional[np.ndarray] = None,
    finalize: bool = False,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if token.shape[0] <= audio_num_codebooks + 2:
        return None, None

    audio_datas = split_interleaved_delayed_audios(token, audio_tokenizer, audio_stream_eos_id)

    audio_codes_list = []
    for audio_data in audio_datas:
        # Prune the first and last stream bos/eos tokens
        if np.all(audio_data[0] == audio_stream_bos_id):
            audio_data = audio_data[1:]
            audio_chunk_size -= 1
        if np.all(audio_data[-1] == audio_stream_eos_id):
            audio_data = audio_data[:-1]
            audio_chunk_size -= 1

        audio_data = audio_data.transpose(1, 0)
        audio_codes = revert_delay_pattern(audio_data).clip(0, audio_codebook_size - 1)
        audio_codes_list.append(audio_codes)

    audio_codes = np.concatenate(audio_codes_list, axis=1)
    tts_speech, _ = audio_tokenizer.decode(vq_code=audio_codes)
    if fade_out_audio is not None:
        hamming_window_len = min(2 * len(fade_out_audio), samples_per_token)
        hamming_window = _get_hamming_window(hamming_window_len)
        fade_overlap = hamming_window_len // 2
        tts_speech[:fade_overlap] = (
            tts_speech[:fade_overlap] * hamming_window[:fade_overlap]
            + fade_out_audio[:fade_overlap] * hamming_window[fade_overlap:]
        )

    fade_out_audio = tts_speech[audio_chunk_size * samples_per_token :]
    if not finalize:
        tts_speech = tts_speech[: audio_chunk_size * samples_per_token]
    else:
        fade_out_audio = None
    return tts_speech, fade_out_audio


def create_audio_chunk(
    audio_tokens_cache: np.ndarray,
    audio_chunk_size: int,
    fade_out_audio: Optional[np.ndarray],
    audio_tokenizer: HiggsAudioTokenizer,
    audio_codebook_size: int,
    samples_per_token: int,
    audio_num_codebooks: int,
    audio_stream_bos_id: int,
    audio_stream_eos_id: int,
    finalize: bool = False,
    return_as_numpy_audio: bool = False,
) -> tuple[Optional[ChatCompletionAudio], np.ndarray]:
    new_audio, new_fade_out_audio = token2wav(
        audio_tokens_cache,
        audio_chunk_size,
        fade_out_audio=fade_out_audio,
        finalize=finalize,
        audio_tokenizer=audio_tokenizer,
        audio_codebook_size=audio_codebook_size,
        samples_per_token=samples_per_token,
        audio_num_codebooks=audio_num_codebooks,
        audio_stream_bos_id=audio_stream_bos_id,
        audio_stream_eos_id=audio_stream_eos_id,
    )

    if return_as_numpy_audio:
        return new_audio, new_fade_out_audio

    audio_pcm16 = (new_audio * np.iinfo(np.int16).max).astype(np.int16)
    return ChatCompletionAudio(
        id=f"audio-{uuid.uuid4().hex}",
        data=base64.b64encode(audio_pcm16).decode("utf-8"),
        expires_at=0,
        transcript="",
    ), new_fade_out_audio


def _get_hamming_window(len):
    return np.hamming(len)


def split_interleaved_delayed_audios(
    audio_data: Union[list[list[int]], np.ndarray],
    audio_tokenizer: HiggsAudioTokenizer,
    audio_stream_eos_id: int,
) -> list[tuple[list[list[int]], np.ndarray]]:
    separator = [audio_stream_eos_id] * audio_tokenizer.num_codebooks

    # Convert separator to numpy array if audio_data is numpy array
    if isinstance(audio_data, np.ndarray):
        separator = np.array(separator)
        # Find the indices where the rows equal the separator
        split_indices = np.where(np.all(audio_data == separator, axis=1))[0]
        start = 0
        groups = []
        for idx in split_indices:
            groups.append(audio_data[start:idx])
            start = idx + 1
        if start < len(audio_data):
            groups.append(audio_data[start:])
    else:
        groups = []
        current = []
        for row in audio_data:
            current.append(row)

            # Handle comparison for both list and numpy array types
            if isinstance(audio_data, np.ndarray):
                if np.array_equal(row, separator):
                    groups.append(current)
                    current = []
            else:
                if row == separator:
                    groups.append(current)
                    current = []

        # Don't forget the last group if there's no trailing separator
        if current:
            groups.append(current)

    return groups


def _build_delay_pattern_mask(
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
    return input_ids


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


def load_higgs_audio_tokenizer(tokenizer_name_or_path: str, device: str = "cuda"):
    """Load the underlying Higgs Audio tokenizer model (DAC+RVQ backend).

    This function prefers the upstream boson_multimodal implementation. The
    tokenizer weights directory must contain a config (json/yaml) and a
    model.pth state dict.
    """
    if BosonHiggsAudioTokenizer is None:
        raise ImportError(
            "boson_multimodal not found. Please install the audio tokenizer backend "
            "(boson-multimodal) or ensure it's on PYTHONPATH to enable encode/decode."
        )

    is_local = os.path.exists(tokenizer_name_or_path)
    tokenizer_path = (
        tokenizer_name_or_path if is_local else snapshot_download(tokenizer_name_or_path)
    )

    config_path = os.path.join(tokenizer_path, "config.json")
    if os.path.exists(config_path):
        config = json.load(open(config_path))
    elif os.path.exists(os.path.join(tokenizer_path, "config.yaml")):
        # Old version omega config file
        config = OmegaConf.load(os.path.join(tokenizer_path, "config.yaml")).generator.config
    else:
        raise ValueError(f"No config file found in {tokenizer_path}")

    model_path = os.path.join(tokenizer_path, "model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Tokenizer checkpoint not found: {model_path}")

    # Only pass init args that are supported by the upstream class
    init_signature = inspect.signature(BosonHiggsAudioTokenizer.__init__)
    valid_params = set(init_signature.parameters.keys()) - {"self"}
    filtered_config = {k: v for k, v in config.items() if k in valid_params}

    model = BosonHiggsAudioTokenizer(
        **filtered_config,
        device=device,
    )
    parameter_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(parameter_dict, strict=False)
    model.to(device)
    model.eval()
    return model


class HiggsAudioDecoderProjector(Module):
    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.text_lm_head = ColumnLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.dtype,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            gather_output=True,
        )

        self.audio_lm_head = ColumnLinear(
            config.hidden_size,
            config.audio_num_codebooks * (config.audio_codebook_size + 2),
            bias=False,
            dtype=config.dtype,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            gather_output=True,
        )

    def forward(self, hidden_states):
        return self.text_lm_head(hidden_states), self.audio_lm_head(hidden_states)


class HiggsAudioFeatureProjector(Module):
    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.linear = ColumnLinear(
            in_features=config.audio_encoder_config.d_model,
            out_features=config.hidden_size,
            bias=True,
            dtype=config.dtype,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            gather_output=True,  # Gather output across TP ranks
        )

    def forward(self, hidden_states):
        return self.linear(hidden_states)


class WhisperEncoderLayer(Module):
    def __init__(self, config: HiggsAudioConfig, dtype: str = "bfloat16"):
        super().__init__()
        self.config = config.audio_encoder_config
        self.dtype = dtype

        self.self_attn = Attention(
            hidden_size=self.config.d_model,
            num_attention_heads=self.config.encoder_attention_heads,
            attention_head_size=self.config.d_model // self.config.encoder_attention_heads,
            dtype=self.dtype,
        )

        self.self_attn_layer_norm = LayerNorm(
            normalized_shape=self.config.d_model, dtype=self.dtype
        )

        self.mlp = MLP(
            hidden_size=self.config.d_model,
            ffn_hidden_size=self.config.encoder_ffn_dim,
            hidden_act=self.config.activation_function,
            dtype=self.dtype,
        )

        self.final_layer_norm = LayerNorm(normalized_shape=self.config.d_model, dtype=self.dtype)

    def forward(self, hidden_states, attention_mask=None):
        def custom_forward(*inputs):
            hidden_states = inputs[0]
            attention_mask = inputs[1]

            residual = hidden_states
            hidden_states = self.self_attn_layer_norm(hidden_states)

            hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)[0]

            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.final_layer_norm(hidden_states)

            hidden_states = self.mlp(hidden_states)

            hidden_states = residual + hidden_states

            return hidden_states

        return custom_forward(hidden_states, attention_mask)


class HiggsAudioEncoder(Module):
    def __init__(self, config: HiggsAudioConfig, dtype: str = "bfloat16"):
        super().__init__()
        self.config = config.audio_encoder_config
        self.dtype = dtype

        self.embed_dim = self.config.d_model
        self.num_mel_bins = self.config.num_mel_bins

        self.conv1 = Conv1d(self.num_mel_bins, self.embed_dim, kernel_size=3, padding=1)
        self.conv2 = Conv1d(self.embed_dim, self.embed_dim, kernel_size=3, stride=2, padding=1)

        # Using functional gelu; no module needed

        self.embed_positions = Embedding(self.config.max_source_positions, self.embed_dim)

        self.layers = ModuleList(
            [WhisperEncoderLayer(config, dtype) for _ in range(self.config.encoder_layers)]
        )

        self.layer_norm = LayerNorm(normalized_shape=self.embed_dim, dtype=self.dtype)

    def forward(self, inputs):
        # TODO: Add zero-shape tensor support

        hidden_states = self.conv1(inputs)
        hidden_states = gelu(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = gelu(hidden_states)

        hidden_states = hidden_states.permute(0, 2, 1)

        positions = self.embed_positions(hidden_states.shape[1])
        hidden_states = hidden_states + positions

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=None,
            )

        hidden_states = self.layer_norm(hidden_states)

        # Optional temporal pooling removed for now to avoid dependency on AvgPool1d

        return hidden_states


class HiggsAudioDualFFNDecoderLayer(Module):
    """Dual FFN decoder layer for Higgs Audio model.

    This layer implements the novel architecture where audio and text tokens
    share a common attention mechanism but are processed through separate
    FFN (feed-forward network) paths. This allows for:

    1. Specialized processing for different modalities
    2. Efficient inference with smaller audio FFN
    3. Additional trainable parameters for audio processing
    4. Maintained compatibility with standard transformer architecture

    Architecture flow:
    1. Shared multi-head attention for all tokens
    2. Split tokens by type (text vs audio)
    3. Apply separate FFNs to each token type
    4. Recombine outputs preserving original order

    Args:
        config: Model configuration containing hidden sizes, attention heads, etc.
        layer_idx: Index of this layer in the model
    """

    def __init__(
        self,
        config,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.quant_config = config.quant_config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.intermediate_size = config.intermediate_size
        self.dtype = config.dtype

        # Shared attention layer
        self.self_attn = Attention(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_kv_heads=self.num_kv_heads,
            max_position_embeddings=getattr(config.text_config, "max_position_embeddings", 8192),
            dtype=self.dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=getattr(config.text_config, "attention_bias", False),
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            rotary_embedding_base=getattr(config.text_config, "rope_theta", 10000.0),
            rotary_embedding_scaling=getattr(config.text_config, "rope_scaling", None),
            tp_group=None,
            tp_size=1,
            quant_mode=getattr(self.quant_config, "quant_mode", None)
            if self.quant_config
            else None,
        )

        # Normalization layers
        norm_eps = getattr(config.text_config, "rms_norm_eps", 1e-6)
        self.input_layernorm = RmsNorm(
            normalized_shape=self.hidden_size,
            eps=norm_eps,
            dtype=self.dtype,
        )
        self.post_attention_layernorm = RmsNorm(
            normalized_shape=self.hidden_size,
            eps=norm_eps,
            dtype=self.dtype,
        )

        # Text FFN (standard MLP)
        # Main text FFN using FusedGatedMLP for optimal fusion

        self.text_mlp = FusedGatedMLP(
            hidden_size=config.hidden_size,
            ffn_hidden_size=getattr(config.text_config, "intermediate_size", 8192),
            hidden_act=getattr(config.text_config, "hidden_act", "silu"),
            bias=getattr(config.text_config, "mlp_bias", False),
            dtype=self.dtype,
            tp_group=None,
            tp_size=1,
            quant_mode=getattr(self.quant_config, "quant_mode", None)
            if self.quant_config
            else None,
        )

        # Audio-specific components (only if using dual FFN)
        # Audio-specific normalization layers
        self.audio_input_layernorm = RmsNorm(
            normalized_shape=self.hidden_size,
            eps=norm_eps,
            dtype=self.dtype,
        )
        self.audio_post_attention_layernorm = RmsNorm(
            normalized_shape=self.hidden_size,
            eps=norm_eps,
            dtype=self.dtype,
        )

        # Audio FFN using FusedGatedMLP for optimal fusion
        # Can be smaller than text FFN for efficiency
        audio_intermediate_size = getattr(config, "audio_intermediate_size", self.intermediate_size)
        self.audio_mlp = FusedGatedMLP(
            hidden_size=self.hidden_size,
            ffn_hidden_size=audio_intermediate_size,
            hidden_act=getattr(
                config.text_config, "hidden_act", "silu"
            ),  # Use same activation as text for consistency
            bias=getattr(config.text_config, "mlp_bias", False),
            dtype=self.dtype,
            tp_group=None,
            tp_size=1,
            quant_mode=getattr(self.quant_config, "quant_mode", None)
            if self.quant_config
            else None,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        use_cache: bool = False,
        kv_cache_params=None,
        attention_params=None,
        audio_out_mask: Optional[Tensor] = None,
        delay_pattern: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass through the dual FFN layer.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask tensor
            kv_cache_params: KV cache parameters for incremental decoding
            attention_params: Additional attention parameters
            mode: Current generation mode (AUDIO_INIT, AUDIO_IN_PROGRESS)
            audio_out_mask: Boolean mask indicating audio token positions [batch_size, seq_len]
            delay_pattern: RVQ codebook delay pattern [num_codebooks, seq_len]

        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        residual = hidden_states

        # Input normalization - use audio-specific norm for audio tokens if dual FFN
        if audio_out_mask is not None:
            # Apply different normalization based on token type
            text_mask = ~audio_out_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
            audio_mask = audio_out_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]

            text_normed = self.input_layernorm(hidden_states) * text_mask
            audio_normed = self.audio_input_layernorm(hidden_states) * audio_mask
            hidden_states = text_normed + audio_normed
        else:
            hidden_states = self.input_layernorm(hidden_states)

        # Apply advanced attention masking for audio generation
        if attention_mask is not None:
            modified_mask = attention_mask
            # Apply RVQ causal constraints during audio generation
            if delay_pattern is not None:
                modified_mask = apply_delay_pattern_masking(
                    modified_mask, audio_out_mask, self.config
                )
            # Block audio->text attention during audio generation to prevent leakage
            audio_positions = audio_out_mask.unsqueeze(1)  # [batch_size, 1, seq_len]
            text_positions = (~audio_out_mask).unsqueeze(2)  # [batch_size, seq_len, 1]

            # Mask audio tokens from attending to text tokens
            audio_to_text_mask = audio_positions & text_positions
            modified_mask = modified_mask.masked_fill(audio_to_text_mask, float("-inf"))

        # Shared attention for all tokens
        attention_output = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            **kwargs,
        )

        # Extract attention output and updated cache
        # TensorRT-LLM attention returns (context, past_key_value) when using cache
        if use_cache:
            attention_output, presents = attention_output

        # Add residual connection after attention
        hidden_states = residual + attention_output
        residual = hidden_states

        # Fusion-friendly dual FFN processing with static graph
        # Always execute both paths and use elementwise masking for routing

        # Create masks for text and audio tokens
        if audio_out_mask is not None:
            text_mask = ~audio_out_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
            audio_mask = audio_out_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        else:
            # Default to text-only processing when no audio mask
            text_mask = torch.ones_like(hidden_states[..., :1])  # [batch_size, seq_len, 1]
            audio_mask = torch.zeros_like(hidden_states[..., :1])  # [batch_size, seq_len, 1]

        # Always process both paths for static graph (fusion-friendly)
        # Text path
        text_norm = self.post_attention_layernorm(hidden_states)
        text_output = self.text_mlp(text_norm)
        text_output = text_output * text_mask  # Apply text mask

        # Audio path (when dual FFN is enabled)
        audio_norm = self.audio_post_attention_layernorm(hidden_states)
        audio_output = self.audio_mlp(audio_norm)
        audio_output = audio_output * audio_mask  # Apply audio mask

        # Combine outputs with elementwise addition (fusion-friendly)
        ffn_output = text_output + audio_output
        hidden_states = residual + ffn_output
        if use_cache:
            # Return both hidden states and updated cache for TensorRT-LLM
            return hidden_states, presents

        return hidden_states


class HiggsAudioLogitsProcessor(LogitsProcessor):
    def __init__(
        self, tokenizer, boost_factor: float, p: int = 2, complete_sentences: bool = False
    ):
        self.eos_token = tokenizer.eos_token_id
        self.boost_factor = boost_factor
        self.p = p
        self.token_count = 0
        self.full_stop_token = text_to_token(tokenizer, "It is a sentence.", last=True)
        self.new_line_token = text_to_token(tokenizer, "It is a new line\n", last=True)
        self.complete_sentences = complete_sentences

    def __call__(
        self,
        req_ids: int,
        logits: torch.Tensor,
        ids: List[List[int]],
        stream_ptr,
        client_id: Optional[int],
    ):
        boost_val = self.boost_factor * (self.token_count**self.p) / (10**self.p)

        stream = None if stream_ptr is None else torch.cuda.ExternalStream(stream_ptr)

        with torch.cuda.stream(stream):
            ids = torch.LongTensor(ids).to(logits.device, non_blocking=True)

            if self.complete_sentences:
                enabled = (ids[:, -1] == self.full_stop_token) | (ids[:, -1] == self.new_line_token)
                logits[:, :, self.eos_token] += enabled * boost_val
            else:
                logits[:, :, self.eos_token] += boost_val

        self.token_count += 1


class HiggsAudioModel(Module):
    """Main HiggsAudio multimodal model for TensorRT-LLM.

    Combines Llama-3.2-3B text generation backbone with audio processing
    through Whisper encoder and novel DualFFN architecture for simultaneous
    text and audio token generation.
    """

    def __init__(self, config: HiggsAudioConfig, dtype: str = "bfloat16"):
        super().__init__()

        # Validate and store configuration
        if not isinstance(config, HiggsAudioConfig):
            raise TypeError(f"Expected HiggsAudioConfig, got {type(config)}")
        config.validate()  # Ensure config is valid
        self.config = config
        self.dtype = dtype

        # Extract key configuration values
        if isinstance(config.text_config, dict):
            self.hidden_size = config.text_config.get("hidden_size", 3072)
            self.num_layers = config.text_config.get("num_hidden_layers", 28)
            self.vocab_size = config.text_config.get("vocab_size", 128256)
            self.num_attention_heads = config.text_config.get("num_attention_heads", 24)
            self.num_key_value_heads = config.text_config.get("num_key_value_heads", 8)
        else:
            self.hidden_size = config.hidden_size
            self.num_layers = getattr(config.text_config, "num_hidden_layers", 28)
            self.vocab_size = getattr(config.text_config, "vocab_size", 128256)
            self.num_attention_heads = getattr(config.text_config, "num_attention_heads", 24)
            self.num_key_value_heads = getattr(config.text_config, "num_key_value_heads", 8)

        # Store special token IDs for easy access
        self.audio_bos_token_id = config.audio_bos_token_id
        self.audio_bos_token = config.audio_bos_token
        self.audio_eos_token_id = config.audio_eos_token_id
        self.audio_eos_token = config.audio_eos_token
        self.audio_out_bos_token_id = config.audio_out_bos_token_id
        self.audio_in_token_idx = config.audio_in_token_idx
        self.audio_in_token = config.audio_in_token
        self.audio_out_token_idx = config.audio_out_token_idx
        self.audio_stream_bos_id = config.audio_stream_bos_id
        self.audio_stream_eos_id = config.audio_stream_eos_id
        self.pad_token_id = config.pad_token_id

        # Audio codebook parameters
        self.audio_num_codebooks = config.audio_num_codebooks
        self.audio_codebook_size = config.audio_codebook_size

        # ========== Model Components ==========

        if self.mapping.is_first_pp_rank():
            self.vocab_embedding = Embedding(
                config.vocab_size, config.hidden_size, dtype=config.dtype
            )

        # 3. Audio Encoder (Whisper-based) - processes mel-spectrograms to features
        self.audio_encoder = HiggsAudioEncoder(config, dtype=dtype)

        # 4. Audio Feature Projector - maps audio features to text model hidden size
        self.audio_feature_projector = HiggsAudioFeatureProjector(config, dtype=dtype)

        # 5. Tokenizers - handles RVQ encoding/decoding
        self.audio_tokenizer = HiggsAudioTokenizer("bosonai/higgs-audio-v2-tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")

        # 6. Decoder Layers - using DualFFN layers for audio-aware processing
        self.layers = DecoderLayerList(HiggsAudioDualFFNDecoderLayer, config=config)

        # 7. Layer Normalization (RMS norm following Llama architecture)
        if isinstance(config.text_config, dict):
            norm_eps = config.text_config.get("rms_norm_eps", 1e-6)
        else:
            norm_eps = getattr(config.text_config, "rms_norm_eps", 1e-6)
        self.norm = RmsNorm(normalized_shape=self.hidden_size, eps=norm_eps, dtype=dtype)

        # 8. Dual-headed output projector
        self.audio_decoder_projector = HiggsAudioDecoderProjector(config)

        # 9. Audio codebook embeddings for input audio tokens
        # Each codebook needs embeddings for its tokens plus stream BOS/EOS
        self.audio_codebook_embeddings = Embedding(
            num_embeddings=config.audio_num_codebooks * (config.audio_codebook_size + 2),
            embedding_dim=self.hidden_size,
            dtype=dtype,
        )

        # ========== State Management ==========

        # Audio feature cache for multimodal processing
        self._audio_feature_cache = None
        self._audio_feature_lengths = None

        # Masks and cache metadata buffers
        self._audio_out_mask = None
        self._attention_mask_cache = None
        self._past_key_values = None

    def forward(
        self,
        input_ids: Tensor,
        kv_cache_params: Optional[KeyValueCacheParams] = None,
        use_cache: bool = True,
        attention_mask: Optional[Tensor] = None,
        hidden_states: Optional[Tensor] = None,
        prompt_embedding_table: Optional[Tensor] = None,
        prompt_tasks: Optional[Tensor] = None,
        prompt_vocab_size: Optional[Tensor] = None,
        last_token_ids: Optional[Tensor] = None,
        # audio_feature_attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass through the HiggsAudio model.

        Args:
            input_ids: Text token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len, seq_len]
            audio_features: Mel-spectrogram features [batch_size, mel_bins, time]
            audio_feature_attention_mask: Audio attention mask [batch_size, time]
            audio_out_ids: Audio output token IDs [batch_size, seq_len, num_codebooks]
            kv_cache_params: KV cache parameters for incremental decoding
            use_cache: Whether to use KV cache
            last_token_ids: Last token IDs for gather operations [batch_size]

        Returns:
            Tuple of (text_logits, audio_logits)
        """
        batch_size, seq_len = input_ids.shape
        audio_features = prompt_embedding_table
        self.setup_inputs(
            input_ids=input_ids,
            audio_features=audio_features,
        )
        # ========== 1. Input Token Embedding ==========

        if self.mapping.is_first_pp_rank():
            hidden_states = self.vocab_embedding(input_ids, prompt_embedding_table)
        else:
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())

        # Get audio output mask for dual processing (needed for later processing)
        audio_out_mask = input_ids == self.audio_out_token_idx

        # ========== 2. Audio Feature Processing ==========
        if self._audio_feature_cache is not None:
            audio_features_embedded = self._audio_feature_cache

        # ========== 3. Audio Output Token Embedding ==========

        if audio_out_ids is not None and audio_out_mask is not None:
            # Embed audio output tokens from codebooks
            # audio_out_ids shape: [batch_size, seq_len, num_codebooks]
            batch_size_ao, seq_len_ao, num_codebooks = audio_out_ids.shape

            # Flatten for embedding lookup: [batch_size * seq_len * num_codebooks]
            audio_flat = audio_out_ids.flatten()
            audio_embedded_flat = self.audio_codebook_embeddings(audio_flat)

            # Reshape back: [batch_size, seq_len, num_codebooks, hidden_size]
            audio_embedded = audio_embedded_flat.view(
                batch_size_ao, seq_len_ao, num_codebooks, self.hidden_size
            )

            # Sum across codebooks: [batch_size, seq_len, hidden_size]
            audio_out_embedded = audio_embedded.sum(dim=2)

            # Merge with text embeddings where audio tokens are present
            audio_mask_expanded = audio_out_mask.unsqueeze(-1).expand_as(hidden_states)
            hidden_states = torch.where(audio_mask_expanded, audio_out_embedded, hidden_states)

        # ========== 2.5. Merge Audio Input Features ==========

        if audio_features_embedded is not None:
            # Find audio input token positions in input_ids.
            # This is a placeholder implementation. A more robust solution would handle
            # sequence length changes and attention mask adjustments by concatenating
            # audio features into the sequence.
            audio_input_mask = input_ids == self.audio_in_token_idx
            if audio_input_mask.any():
                # For simplicity, we replace the embedding of the audio token
                # with the mean-pooled audio features.
                audio_feature_summary = audio_features_embedded.mean(
                    dim=1, keepdim=True
                )  # -> [batch, 1, hidden_size]

                # Use `where` for a clean replacement.
                hidden_states = torch.where(
                    audio_input_mask.unsqueeze(-1), audio_feature_summary, hidden_states
                )

        # ========== 4. Generate Attention Masks ==========

        # Prepare attention mask with audio-specific causality
        if attention_mask is None:
            # Default causal mask
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device)
            )
            attention_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        # ========== 5. Transformer Decoder Layers ==========
        hidden_states = self.layers.forward(
            hidden_states,
            use_cache=use_cache,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
        )

        if use_cache:
            hidden_states, presents = hidden_states

            # Handle layer output (may include updated cache)
            if use_cache:
                hidden_states, layer_present = hidden_states
                presents.append(layer_present)

        # ========== 6. Layer Normalization ==========

        hidden_states = self.norm(hidden_states)

        # ========== 7. Output Projection ==========

        # Compute audio logits for audio generation modes
        if self.generation_mode in [GenerationMode.AUDIO_INIT, GenerationMode.AUDIO_IN_PROGRESS]:
            # Apply optional projection before audio head
            audio_hidden = hidden_states

            # Get raw audio logits
            audio_logits = self.audio_lm_head(audio_hidden)

            # Reshape to [batch, seq_len, num_codebooks, codebook_size+2]
            batch_size, seq_len = audio_logits.shape[:2]
            audio_logits = audio_logits.view(
                batch_size,
                seq_len,
                self.config.audio_num_codebooks,
                self.config.audio_codebook_size + 2,
            )
        else:
            audio_logits = None

        # ========== 8. Gather Last Token Logits (if needed) ==========

        if last_token_ids is not None:
            text_logits = gather_last_token_logits(
                last_token_ids, last_token_ids, default_net().plugin_config.remove_input_padding
            )
            if audio_logits is not None:
                audio_logits = gather_last_token_logits(
                    audio_logits, last_token_ids, default_net().plugin_config.remove_input_padding
                )

        return text_logits, audio_logits


class HiggsForCausalLM(DecoderModelForCausalLM):
    config_class = HiggsAudioConfig

    def __init__(self, config: HiggsAudioConfig):
        transformer = HiggsAudioModel(config)
        vocab_size_padded = pad_vocab_size(config.vocab_size, config.mapping.tp_size)

        lm_head = ColumnLinear(
            config.hidden_size,
            vocab_size_padded,
            bias=False,
            dtype=config.dtype,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            gather_output=True,
        )

        self.quant_mode = config.quant_mode
        self.mapping = config.mapping
        self.trtllm_modules_to_hf_modules = None
        super().__init__(config, transformer, lm_head)

    @classmethod
    def from_hugging_face(
        cls,
        hf_model_or_dir: Union[str, "transformers.PreTrainedModel"],
        dtype: str = "auto",
        mapping: Optional[Mapping] = None,
        quant_config: Optional[QuantConfig] = None,
        **kwargs,
    ):
        """Create a QWenForCausalLM object from give parameters"""
        import transformers

        load_model_on_cpu = kwargs.pop("load_model_on_cpu", False)
        use_autoawq = kwargs.pop("use_autoawq", False)

        assert hf_model_or_dir is not None
        use_preloading = isinstance(hf_model_or_dir, transformers.PreTrainedModel)
        if use_preloading:
            hf_model = hf_model_or_dir
            hf_config_or_dir = hf_model.config
        else:
            hf_model_dir = hf_model_or_dir
            hf_config_or_dir = hf_model_or_dir

        config = QWenConfig.from_hugging_face(
            hf_config_or_dir, dtype=dtype, mapping=mapping, quant_config=quant_config, **kwargs
        )

        if os.environ.get("TRTLLM_DISABLE_UNIFIED_CONVERTER") is None:
            arg_dict = {"use_autoawq": True} if use_autoawq else {}
            custom_dict = {}

            if config.qwen_type == "qwen":
                custom_dict = {
                    "transformer": "transformer",
                    "vocab_embedding": "wte",
                    "ln_f": "ln_f",
                    "layers": "h",
                    "attention": "attn",
                    "qkv": "c_attn",
                    "dense": "c_proj",
                    "gate": "w1",
                    "proj": "c_proj",
                    "fc": "w2",
                    "input_layernorm": "ln_1",
                    "post_layernorm": "ln_2",
                }
            elif config.qwen_type == "qwen2_moe":
                custom_dict = {
                    "mlp.shared_expert": "mlp.shared_expert",
                    "mlp.shared_expert_gate": "mlp.shared_expert_gate",
                    "fc": ["up_proj", "gate_proj"],
                }
            elif config.qwen_type == "qwen3_moe":
                custom_dict = {
                    "fc": ["up_proj", "gate_proj"],
                    "q_layernorm": "q_norm",
                    "k_layernorm": "k_norm",
                }
            elif config.qwen_type in {"qwen2", "qwen2_vl"} and config.tie_word_embeddings:
                custom_dict = {"lm_head": "model.embed_tokens"}
            elif config.architecture == "Qwen2ForSequenceClassification":
                custom_dict = {
                    "lm_head": "score",
                }
            elif config.qwen_type == "qwen2_llava_onevision":
                custom_dict = {
                    "transformer": "language_model.model",
                    "lm_head": "language_model.lm_head",
                }
            elif config.qwen_type == "qwen2_audio":
                custom_dict = {
                    "transformer": "language_model.model",
                    "lm_head": "language_model.lm_head",
                }
            elif config.qwen_type == "qwen3":
                custom_dict = {
                    "q_layernorm": "q_norm",
                    "k_layernorm": "k_norm",
                }
            loader = ModelWeightsLoader(hf_model_dir, custom_dict)
            model = cls(config)
            if config.qwen_type == "qwen" and model.config.mapping.has_tp():

                def reshape_qkv(weights):
                    if weights is None:
                        return weights
                    mapping = model.config.mapping
                    unsqueeze = False
                    if isinstance(weights, torch.Tensor):
                        unsqueeze = True
                        weights = [weights]

                    for idx, w in enumerate(weights):
                        if quant_config.quant_algo == QuantAlgo.W4A16_GPTQ:
                            w = w.reshape(-1, 3, w.shape[-1] // 3)
                            w = w.chunk(mapping.tp_size, 2)[mapping.tp_rank]
                            if w.shape[0] == 1:
                                weights[idx] = w.reshape(-1)
                            else:
                                weights[idx] = w.reshape(w.shape[0], -1)
                        else:
                            w = w.reshape(3, w.shape[0] // 3, -1)
                            w = w.chunk(mapping.tp_size, 1)[mapping.tp_rank]
                            if w.shape[-1] == 1:
                                weights[idx] = w.reshape(-1)
                            else:
                                weights[idx] = w.reshape(-1, w.shape[-1])
                    if unsqueeze:
                        return weights[0]
                    else:
                        return weights

                loader.update_key_mapping(model)
                tllm_weights = {}
                for tllm_key, _ in tqdm(model.named_parameters()):
                    if "qkv" in tllm_key:
                        tllm_weights.update(
                            loader.load(
                                tllm_key,
                                reshape_qkv,
                                skip_tp=True,
                                custom_postprocess_kwargs=arg_dict,
                            )
                        )
                    else:
                        tllm_weights.update(
                            loader.load(tllm_key, custom_postprocess_kwargs=arg_dict)
                        )
                loader.fill(tllm_weights)
            elif config.qwen_type in ("qwen2_moe", "qwen3_moe"):
                for tllm_key, _ in model.named_parameters():
                    sub_module = model
                    for attr in tllm_key.split(".")[:-1]:
                        sub_module = getattr(sub_module, attr)
                    if "router" in tllm_key or isinstance(sub_module, MOEWeightWrapper):
                        sub_module_dic = sub_module.tllm_to_externel_key_dict
                        sub_module_dic["mlp"] = "mlp"
                        if "fc" in sub_module_dic.keys():
                            sub_module_dic["fc"] = [
                                hf_keyword.replace("w1", "gate_proj")
                                for hf_keyword in sub_module_dic["fc"]
                            ]
                            sub_module_dic["fc"] = [
                                hf_keyword.replace("w3", "up_proj")
                                for hf_keyword in sub_module_dic["fc"]
                            ]
                        if "proj" in sub_module_dic.keys():
                            sub_module_dic["proj"] = [
                                hf_keyword.replace("w2", "down_proj")
                                for hf_keyword in sub_module_dic["proj"]
                            ]
                        sub_module.tllm_to_externel_key_dict = sub_module_dic

                def concat_gate_up_proj(weights):
                    return torch.cat(weights, dim=-2)

                loader.update_key_mapping(model)
                tllm_weights = {}
                for tllm_key, _ in tqdm(model.named_parameters()):
                    if tllm_key.endswith("shared_expert.fc.weight"):
                        tllm_weights.update(
                            loader.load(
                                tllm_key, concat_gate_up_proj, custom_postprocess_kwargs=arg_dict
                            )
                        )
                    else:
                        tllm_weights.update(
                            loader.load(tllm_key, custom_postprocess_kwargs=arg_dict)
                        )
                loader.fill(tllm_weights)
            else:
                # For Qwen1 w/o TP, Qwen1.5 and Qwen2 w/o MoE
                loader.generate_tllm_weights(model, arg_dict)
        else:
            if not use_preloading:
                hf_model = load_hf_qwen(hf_model_dir, load_model_on_cpu)

            model = QWenForCausalLM(config)

            if quant_config.quant_algo == QuantAlgo.W4A16_GPTQ:
                weights = load_weights_from_hf_gptq_model(hf_model, config)
            else:
                weights = load_weights_from_hf_model(hf_model, config)
            model.load(weights)
        return model

    def default_plugin_config(self, **kwargs):
        plugin_config = super().default_plugin_config(**kwargs)
        if self.quant_mode.is_int4_weight_only_per_group():
            plugin_config.weight_only_groupwise_quant_matmul_plugin = "auto"
        return plugin_config

    @classmethod
    def quantize(
        cls,
        hf_model_dir: str,
        output_dir: str,
        dtype: str = "auto",
        mapping: Optional[Mapping] = None,
        quant_config: Optional[QuantConfig] = None,
        *,
        calib_dataset="cnn_dailymail",
        calib_batches=512,
        calib_batch_size=1,
        calib_max_seq_length=512,
        random_seed=1234,
        tokenizer_max_seq_length=2048,
        **kwargs,
    ):
        if quant_config._requires_modelopt_quantization:
            # modelopt quantization flow
            super().quantize(
                hf_model_dir,
                output_dir,
                dtype=dtype,
                mapping=mapping,
                quant_config=quant_config,
                calib_dataset=calib_dataset,
                calib_batches=calib_batches,
                calib_batch_size=calib_batch_size,
                calib_max_seq_length=calib_max_seq_length,
                random_seed=random_seed,
                tokenizer_max_seq_length=tokenizer_max_seq_length,
            )
        elif quant_config._requires_calibration:
            # non-modelopt quantization flow
            from . import convert

            config = QWenConfig.from_hugging_face(
                hf_model_dir, dtype=dtype, mapping=mapping, quant_config=quant_config, **kwargs
            )
            convert.quantize(hf_model_dir, output_dir, config=config, calib_dataset=calib_dataset)
        else:
            raise ValueError(
                f"The quant_config ({quant_config}) does not require calibration, try {cls.__name__}.from_hugging_face instead."
            )


class QWenInfer(object):
    def __init__(
        self,
        audio_engine_path,
        tokenizer_dir,
        engine_dir,
        log_level,
        output_csv,
        output_npy,
        num_beams,
        gpu_id=0,
    ):
        self.audio_engine_path = audio_engine_path
        self.tokenizer_dir = tokenizer_dir
        self.engine_dir = engine_dir
        self.log_level = log_level
        self.max_seq_len = 0
        self.runner = None
        self.hf_audio_tower = None
        self.tokenizer = None
        self.config = None
        self.sampling_config = None
        self.output_csv = output_csv
        self.output_npy = output_npy
        self.num_beams = num_beams
        self.model_config = None
        self.gpu_device = torch.device("cuda", gpu_id)

    def get_model(self):
        # --load the tokenizer and engines #
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_dir,
            legacy=False,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(self.tokenizer_dir, trust_remote_code=True)
        config_path = os.path.join(self.engine_dir, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        self.max_seq_len = config["build_config"]["max_seq_len"]
        assert self.max_seq_len > 0, "max_seq_len must be positive"

        gen_config_path = os.path.join(self.tokenizer_dir, "generation_config.json")
        with open(gen_config_path, "r") as f:
            gen_config = json.load(f)
        top_k = gen_config["top_k"]
        top_p = gen_config["top_p"]
        eos_token_id = tokenizer.pad_token_id
        pad_token_id = tokenizer.pad_token_id

        use_gpt_attention_plugin = config["build_config"]["plugin_config"]["gpt_attention_plugin"]
        remove_input_padding = config["build_config"]["plugin_config"]["remove_input_padding"]
        dtype = config["pretrained_config"]["dtype"]
        tp_size = config["pretrained_config"]["mapping"]["tp_size"]
        pp_size = config["pretrained_config"]["mapping"]["pp_size"]
        world_size = tp_size * pp_size
        assert world_size == tensorrt_llm.mpi_world_size(), (
            f"Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})"
        )
        num_heads = config["pretrained_config"]["num_attention_heads"] // world_size
        max_batch_size = config["build_config"]["max_batch_size"]
        hidden_size = config["pretrained_config"]["hidden_size"] // world_size
        vocab_size = config["pretrained_config"]["vocab_size"]
        num_layers = config["pretrained_config"]["num_hidden_layers"]
        num_kv_heads = config["pretrained_config"].get("num_key_value_heads", num_heads)
        if "kv_cache_type" in config["build_config"]:
            kv_cache_type = KVCacheType.from_string(config["build_config"]["kv_cache_type"])
        else:
            kv_cache_type = KVCacheType.CONTINUOUS

        tokens_per_block = config["build_config"]["plugin_config"]["tokens_per_block"]
        max_prompt_embedding_table_size = config["build_config"].get(
            "max_prompt_embedding_table_size", 0
        )
        quant_mode = QuantMode.from_quant_algo(
            config["pretrained_config"]["quantization"]["quant_algo"],
            config["pretrained_config"]["quantization"]["kv_cache_quant_algo"],
        )
        if config["pretrained_config"].get("multi_query_mode", False):
            tensorrt_llm.logger.warning(
                "`multi_query_mode` config is deprecated. Please rebuild the engine."
            )
            num_kv_heads = 1

        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(
            world_size=world_size, rank=runtime_rank, tp_size=tp_size, pp_size=pp_size
        )

        model_config = ModelConfig(
            max_batch_size=max_batch_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            hidden_size=hidden_size,
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
        sampling_config = SamplingConfig(
            end_id=eos_token_id,
            pad_id=pad_token_id,
            num_beams=self.num_beams,
            top_k=top_k,
            top_p=top_p,
            temperature=1.0,
        )

        engine_name = get_engine_name(runtime_rank)
        serialize_path = os.path.join(self.engine_dir, engine_name)
        print(f"Loading engine from {serialize_path}")
        return (
            model_config,
            sampling_config,
            runtime_mapping,
            runtime_rank,
            serialize_path,
            tokenizer,
            processor,
            eos_token_id,
            pad_token_id,
        )

    def qwen_model_init(self, args):
        logger.info(f"Loading audio engine from {self.audio_engine_path}")
        with open(self.audio_engine_path, "rb") as f:
            engine_buffer = f.read()
        logger.info(f"Creating session from engine {self.audio_engine_path}")
        self.session_audio = Session.from_serialized_engine(engine_buffer)

        self.config, _ = AutoConfig.from_pretrained(
            self.tokenizer_dir,
            return_unused_kwargs=True,
            trust_remote_code=True,
        )

        (
            model_config,
            sampling_config,
            runtime_mapping,
            runtime_rank,
            serialize_path,
            tokenizer,
            processor,
            eos_token_id,
            pad_token_id,
        ) = self.get_model()
        runner_cls = ModelRunner if args.use_py_session else ModelRunnerCpp
        runner_kwargs = dict(
            engine_dir=args.engine_dir,
            lora_dir=args.lora_dir,
            rank=runtime_rank,
            debug_mode=args.debug_mode,
            lora_ckpt_source=args.lora_ckpt_source,
            gpu_weights_percent=args.gpu_weights_percent,
            max_output_len=args.max_new_tokens,
        )
        if not args.use_py_session:
            runner_kwargs.update(
                is_enc_dec=False,
                max_batch_size=model_config.max_batch_size,
                max_input_len=self.max_seq_len - args.max_new_tokens,
                max_beam_width=model_config.max_beam_width,
                max_attention_window_size=args.max_attention_window_size,
                sink_token_length=args.sink_token_length,
                max_tokens_in_paged_kv_cache=args.max_tokens_in_paged_kv_cache,
                kv_cache_enable_block_reuse=args.kv_cache_enable_block_reuse,
                kv_cache_free_gpu_memory_fraction=args.kv_cache_free_gpu_memory_fraction,
                cross_kv_cache_fraction=None,
                enable_chunked_context=args.enable_chunked_context,
                multi_block_mode=args.multi_block_mode,
                cuda_graph_mode=args.cuda_graph_mode,
                device_ids=[args.gpu_id],
            )
        runner_kwargs.update(enable_context_fmha_fp32_acc=args.enable_context_fmha_fp32_acc)
        self.runner = runner_cls.from_dir(**runner_kwargs)
        self.tokenizer = tokenizer
        self.processor = processor
        self.sampling_config = sampling_config
        self.model_config = model_config

    def ptuning_setup(self, prompt_table, dtype, hidden_size, tasks, input_ids):
        if prompt_table is not None:
            task_vocab_size = torch.tensor(
                [prompt_table.shape[0]], dtype=torch.int32, device=self.gpu_device
            )
            prompt_table = prompt_table.to(
                dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype), device=self.gpu_device
            )
        else:
            prompt_table = torch.empty([1, hidden_size], device=self.gpu_device)
            task_vocab_size = torch.zeros([1], device=self.gpu_device)

        if tasks is not None:
            tasks = torch.tensor(
                [int(t) for t in tasks.split(",")], dtype=torch.int32, device=self.gpu_device
            )
            assert tasks.shape[0] == input_ids.shape[0], (
                "Number of supplied tasks must match input batch size"
            )
        else:
            tasks = torch.zeros([input_ids.size(0)], dtype=torch.int32, device=self.gpu_device)

        return [prompt_table, tasks, task_vocab_size]

    def build_user_input(self, audio=None, text=None):
        assert isinstance(audio, str) or isinstance(text, str), (
            "audio or text must be provided as user input"
        )
        content = []
        if audio:
            content.append({"type": "audio", "audio_url": audio})
        if text:
            content.append({"type": "text", "text": text})
        user_input = {"role": "user", "content": content}
        return user_input

    def get_raw_audios(self, audio_url):
        audios = []
        for url in audio_url:
            if os.path.isfile(url):
                audio_data, _ = librosa.load(url, sr=self.processor.feature_extractor.sampling_rate)
            else:
                audio_data, _ = librosa.load(
                    BytesIO(urlopen(url).read()), sr=self.processor.feature_extractor.sampling_rate
                )
            audios.append(audio_data)
        return audios

    def audio_tower(self, audios, mask, stream, run_time=1):
        audios = audios.to(self.gpu_device)
        mask = mask.to(self.gpu_device)
        audio_inputs = {"input": audios.float(), "mask": mask}
        audio_output_info = self.session_audio.infer_shapes(
            [
                TensorInfo("input", trt.DataType.FLOAT, audios.shape),
                TensorInfo("mask", trt.DataType.HALF, mask.shape),
            ]
        )
        audio_outputs = {
            t.name: torch.empty(
                tuple(t.shape), dtype=trt_dtype_to_torch(t.dtype), device=self.gpu_device
            )
            for t in audio_output_info
        }
        profiler.start("Audio")
        for _ in range(run_time):
            ok = self.session_audio.run(audio_inputs, audio_outputs, stream.cuda_stream)
        stream.synchronize()
        audio_time = profiler.stop("Audio") / run_time
        logger.info(f"TensorRT-LLM Audio latency: {audio_time:3f} sec ")

        assert ok, "Runtime execution failed for audio session"

        audio_features = audio_outputs["output"]

        return audio_features

    def generate_for_qwen_audio(
        self,
        input_tokens,
        args,
        prompt_table=None,
        extra_ids=None,
        run_time=1,
    ):
        input_ids = torch.as_tensor(input_tokens, device=self.gpu_device, dtype=torch.int32)
        input_lengths = torch.tensor([input_ids.size(1)], device=self.gpu_device, dtype=torch.int32)
        max_input_length = torch.max(input_lengths).item()
        max_new_tokens = min(args.max_new_tokens, self.max_seq_len - max_input_length)

        prompt_table = prompt_table.unsqueeze(0)
        profiler.start("QWen")
        for _ in range(run_time):
            outputs = self.runner.generate(
                batch_input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                max_attention_window_size=args.max_attention_window_size,
                sink_token_length=args.sink_token_length,
                end_id=self.sampling_config.end_id,
                pad_id=self.sampling_config.pad_id,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                num_beams=args.num_beams,
                num_return_sequences=args.num_return_sequences,
                length_penalty=args.length_penalty,
                early_stopping=args.early_stopping,
                repetition_penalty=args.repetition_penalty,
                presence_penalty=args.presence_penalty,
                frequency_penalty=args.frequency_penalty,
                stop_words_list=[[[151643], [151645]]],
                bad_words_list=self.sampling_config.bad_words_list,
                random_seed=args.random_seed,
                lora_uids=args.lora_task_uids,
                prompt_table=prompt_table,
                prompt_tasks="0",
                output_sequence_lengths=True,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                return_dict=True,
                return_all_generated_tokens=False,
                input_token_extra_ids=extra_ids,
            )
            output_ids = outputs["output_ids"]
            torch.cuda.synchronize()
        Qwen_time = profiler.stop("QWen") / run_time

        return output_ids, Qwen_time

    def get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """Computes the output length of the convolutional layers and the output length of the audio encoder"""
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths

    def qwen_infer(
        self,
        input_text,
        audios,
        audio_ids,
        args,
        stream,
        history=None,
        past_audio_features=None,
        run_time=1,
    ):
        assert input_text, "input_text must be provided"
        assert torch.cuda.is_available(), "no gpu available"
        # preprocess on CPU maybe faster
        device = torch.device("cpu")
        if isinstance(history, list):
            history.append(input_text)
            full_text = self.processor.apply_chat_template(
                history, add_generation_prompt=True, tokenize=False
            )
        else:
            full_text = input_text
        inputs = self.processor(
            text=full_text,
            audios=audios,
            return_tensors="pt",
            padding=True,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
        )
        inputs = inputs.to(device)
        input_ids = inputs.input_ids

        if hasattr(inputs, "input_features") and inputs.input_features is not None:
            # audio tower
            batch_size, _, max_mel_seq_len = inputs.input_features.shape
            feature_attention_mask = inputs.feature_attention_mask

            audio_feat_lengths, num_audio_tokens = self.get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1)
            )

            max_seq_len = (max_mel_seq_len - 2) // 2 + 1
            # Create a sequence tensor of shape (batch_size, max_seq_len)
            seq_range = (
                torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=device)
                .unsqueeze(0)
                .expand(batch_size, max_seq_len)
            )
            lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
            # Create mask
            padding_mask = seq_range >= lengths_expand

            audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                batch_size, 1, max_seq_len, max_seq_len
            )
            audio_attention_mask = audio_attention_mask_.to(dtype=torch.float16, device=device)
            audio_attention_mask[audio_attention_mask_] = float("-inf")

            audio_features = self.audio_tower(
                inputs.input_features, audio_attention_mask, stream, run_time
            )

            # merge audio features and input ids
            num_audios, max_audio_tokens, embed_dim = audio_features.shape
            audio_features_mask = torch.arange(max_audio_tokens, device=device).expand(
                num_audios, max_audio_tokens
            ) < num_audio_tokens.unsqueeze(1)
            masked_audio_features = audio_features[audio_features_mask].view(-1, embed_dim)
            batch_size, _ = input_ids.shape

            # 1. Create a mask to know where special audio tokens are
            special_audio_token_mask = input_ids == self.config.audio_token_index
            special_audio_token_num = special_audio_token_mask.sum().item()
            if past_audio_features is not None:
                assert isinstance(past_audio_features, list), "past_audio_features should be a list"
                assert special_audio_token_num == len(past_audio_features) + num_audios, (
                    f"special_audio_token_num {special_audio_token_num} should be equal to len(past_audio_features) + num_audios ({len(past_audio_features)} + {num_audios})"
                )
                # split to get current audio features
                cur_audio_features = torch.split(masked_audio_features, num_audio_tokens.tolist())
                if len(past_audio_features) > 0:
                    # concat past and current audio features
                    masked_audio_features = torch.cat(
                        (
                            torch.cat(past_audio_features).to(masked_audio_features.device),
                            masked_audio_features,
                        )
                    )
                    # get past audio tokens number
                    past_num_audio_tokens = torch.tensor(
                        [past_feat.size(0) for past_feat in past_audio_features]
                    )
                    # concat past and current audio tokens number
                    num_audio_tokens = torch.cat(
                        (past_num_audio_tokens.to(num_audio_tokens.device), num_audio_tokens)
                    )
                # extend past audio features, cache them in CPU memory
                past_audio_features.extend([cur_feat.cpu() for cur_feat in cur_audio_features])

            batch_indices, non_audio_indices = torch.where(
                input_ids != self.config.audio_token_index
            )

            # 2. Fill the final input ids based on the mask.
            batch_indices, audio_indices = torch.where(input_ids == self.config.audio_token_index)

            vocab_size = self.config.vocab_size
            fake_prompt_id = torch.arange(
                vocab_size, vocab_size + num_audio_tokens.sum(), device=device
            )

            input_ids[batch_indices, audio_indices] = fake_prompt_id
            input_lengths = torch.tensor(
                input_ids.size(1), dtype=torch.int32, device=self.gpu_device
            )
            dtype = self.model_config.dtype
            prompt_table, tasks, task_vocab_size = self.ptuning_setup(
                masked_audio_features, dtype, embed_dim, None, input_ids
            )

            # build extra ids
            assert isinstance(audio_ids, list), "audio_ids must be a list"
            assert len(audio_ids) == num_audio_tokens.size(0), (
                f"audio_ids length doesn't match with num_audio_tokens ({len(audio_ids)} != {num_audio_tokens.size(0)})"
            )
            for i in audio_ids:
                assert isinstance(i, int) and i > 0, "audio_id should be an integer greater than 0"
            extra_ids = torch.zeros_like(input_ids, dtype=torch.int64, device=device)
            seq_extra_ids = torch.cat(
                [
                    torch.full((n,), audio_ids[i], dtype=torch.int64)
                    for i, n in enumerate(num_audio_tokens)
                ]
            ).to(device)
            extra_ids[batch_indices, audio_indices] = seq_extra_ids
            extra_ids = extra_ids.tolist()
        else:
            input_ids = input_ids.to(dtype=torch.int32, device=self.gpu_device)
            input_lengths = torch.tensor(
                input_ids.size(1), dtype=torch.int32, device=self.gpu_device
            )
            dtype = self.model_config.dtype
            prompt_table, tasks, task_vocab_size = self.ptuning_setup(
                None, dtype, self.model_config.hidden_size, None, input_ids
            )
            extra_ids = torch.zeros_like(input_ids, dtype=torch.int64).tolist()

        # print(f"extra_ids: {extra_ids}")
        output_ids, Qwen_time = self.generate_for_qwen_audio(
            input_ids, args, prompt_table, extra_ids, run_time
        )

        runtime_rank = tensorrt_llm.mpi_rank()
        input_lengths = torch.tensor([input_ids.size(1)], device=self.gpu_device, dtype=torch.int32)
        effective_output_token = 0
        if runtime_rank == 0:
            if self.output_csv is None and self.output_npy is None:
                for b in range(input_lengths.size(0)):
                    inputs = input_ids[b]
                    if self.num_beams <= 1:
                        outputs = output_ids[b][0, len(inputs) :].tolist()
                        try:
                            effective_output_token = effective_output_token + outputs.index(151643)
                        except:
                            effective_output_token = 1
                        output_text = self.tokenizer.decode(outputs, skip_special_tokens=True)
                        print(f'Output: "{output_text}"')
                    else:
                        for beam in range(self.num_beams):
                            outputs = output_ids[b][beam, len(inputs) :].tolist()
                            output_text = self.tokenizer.decode(outputs, skip_special_tokens=True)
                            print(f'Output(beam: {beam}): "{output_text}"')
        if isinstance(history, list):
            history.append({"role": "assistant", "content": output_text})
        return output_text, past_audio_features


def setup(self):
    pre_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an AI assistant designed to convert text into speech. Generate speech for the user's text, using the specified description.<|scene_desc_start|>Audio is recorded from a quiet room. Speaker is an enthusiastic young Australian woman in her early 20s with a bright, high-pitched voice.<|scene_desc_end|><|eot_id|><|start_header_id|>user<|end_header_id|>Can you believe just how realistic this sounds now?<|eot_id|><|start_header_id|>assistant<|end_header_id|><|audio_bos|><|AUDIO|><|audio_eos|><|eot_id|><|start_header_id|>user<|end_header_id|>"  # noqa: E501
    post_prompt = "<|eot_id|><|start_header_id|>assistant<|end_header_id|><|audio_out_bos|>"
    padding = "max_length"
    sampling_rate = 24000
    text = "Chat, stop backseating! I totally know what I'm doing... I think"
    text = pre_prompt + text + post_prompt

    infer = QWenInfer(
        args.audio_engine_path,
        args.tokenizer_dir,
        args.engine_dir,
    )
    qinfer.qwen_model_init(args)

    audios = qinfer.get_raw_audios(args.audio_url)
    gpu_device = torch.device("cuda", args.gpu_id)
    stream = torch.cuda.current_stream(device=gpu_device)
    qinfer.qwen_infer(args.input_text, audios, [1], args, stream, None, None, 1)

    if audio is not None:
        # ensure we have as much audios as audio tokens
        num_audio_tokens = sum(sample.count(self.audio_in_token) for sample in text)
        num_audios = 1 if type(audio) is np.ndarray else len(audio)
        if num_audio_tokens != num_audios:
            raise ValueError(
                f"Found {num_audio_tokens} {self.audio_in_token_idx} token"
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
                self.feature_extractor.encode(a, self.feature_extractor.sampling_rate).unsqueeze(0)
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
            audio_in_ids = _validate_and_reshape_mm_tensor(audio_ids, "audio_in_ids", pad_with=0)
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
                padding=padding,
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
    return inputs
