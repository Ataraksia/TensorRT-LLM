# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
"""TensorRT-LLM implementation of Higgs Audio multimodal model."""

import librosa
from collections.abc import AsyncGenerator, Sequence
import os
from typing import Any, Optional, List, OrderedDict
import numpy as np
from openai.types.chat import ChatCompletionAudio
import tensorrt
import torch
from boson_multimodal import *
from starlette.datastructures import State
from huggingface_hub import snapshot_download
from torch import nn
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    pipeline,
)
import tensorrt_llm
from tensorrt_llm.bindings import INT32
from tensorrt_llm.mapping import Mapping

from tensorrt_llm.models.model_weights_loader import ModelWeightsLoader
from tensorrt_llm.models.modeling_utils import (
    DecoderLayerList,
    QuantConfig,
    DecoderModelForCausalLM,
)
from tensorrt_llm.module import Module, ModuleList
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.sampling_params import LogitsProcessor
from tensorrt_llm.functional import (
    Tensor,
    arange,
    cumsum,
    expand_dims_like,
    unsqueeze,
    where,
    sum,
    mean,
    concat,
)
from tensorrt_llm.layers import (
    MLP,
    Attention,
    AttentionMaskType,
    AttentionParams,
    ColumnLinear,
    Embedding,
    KeyValueCacheParams,
    RmsNorm,
)
import math
import copy
import inspect
import json
import os
from typing import Optional


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


def get_feat_extract_output_lengths(input_lengths: torch.LongTensor):
    """Computes the output length of the convolutional layers and the output length of the audio encoder"""
    input_lengths = (input_lengths - 1) // 2 + 1
    output_lengths = (input_lengths - 2) // 2 + 1
    return input_lengths, output_lengths


def revert_delay_pattern(data):
    """Convert samples encoded with delay pattern back to the original form.

    Args:
        data: The data with delay pattern applied.
              Shape (num_codebooks, seq_len + num_codebooks - 1).

    Returns:
        ret: Recovered data with delay pattern removed.
             Shape (num_codebooks, seq_len).
    """
    assert len(data.shape) == 2
    out_l = []
    num_codebooks = data.shape[0]
    for i in range(num_codebooks):
        out_l.append(data[i : (i + 1), i : (data.shape[1] - num_codebooks + 1 + i)])
    return (
        torch.cat(out_l, dim=0) if isinstance(data, torch.Tensor) else np.concatenate(out_l, axis=0)
    )


def _build_delay_pattern_mask(input_ids: torch.LongTensor, bos_token_id: int, pad_token_id: int):
    """Implement the delay pattern proposed in "Simple and Controllable Music Generation", https://arxiv.org/pdf/2306.05284

    In the delay pattern, each codebook is offset by the previous codebook by
    one. We insert a special delay token at the start of the sequence if its delayed, and append pad token once the sequence finishes.

    Take the example where there are 4 codebooks and audio sequence length=8. After shifting, the output should have length seq_len + num_codebooks - 1

    - [ *,  *,  *,  *,  *,  *,  *,  *, P, P, P]
    - [ B,  *,  *,  *,  *,  *,  *,  *, *, P, P]
    - [ B,  B,  *,  *,  *,  *,  *,  *, *, *, P]
    - [ B,  B,  B,  *,  *,  *,  *,  *, *, *, *]

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
            The input ids of the prompt. It will have shape (num_codebooks, seq_len).
        bos_token_id (:obj:`int`):
            The id of the special delay token
        pad_token_id (:obj:`int`):
            The id of the padding token. Should be the same as eos_token_id.

    Returns:
        input_ids (:obj:`torch.LongTensor`):
            The transformed input ids with delay pattern applied. It will have shape ( num_codebooks, seq_len + num_codebooks - 1).
        input_ids_with_gen_mask (:obj:`torch.LongTensor`):
            The transformed input ids with delay pattern applied. The -1 in the output indicates new tokens that should be generated.

    """
    num_codebooks, seq_len = input_ids.shape

    new_seq_len = seq_len + num_codebooks - 1
    input_ids_with_gen_mask = torch.ones(
        (num_codebooks, new_seq_len), dtype=torch.long, device=input_ids.device
    )
    bos_mask = torch.tril(input_ids_with_gen_mask, -1) > 0
    eos_mask = torch.triu(input_ids_with_gen_mask, seq_len) > 0
    input_ids_with_gen_mask[bos_mask] = bos_token_id
    input_ids_with_gen_mask[(~bos_mask) & (~eos_mask)] = input_ids.reshape(-1)
    input_ids = input_ids_with_gen_mask.clone()
    input_ids[eos_mask] = pad_token_id
    input_ids_with_gen_mask[eos_mask] = -1
    return input_ids


class HiggsAudioDualFFNDecoderLayer(Module):
    """TensorRT-LLM implementation of dual-path FFN decoder layer."""

    def __init__(self, config: HiggsAudioConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = self.config.hidden_size

        # Shared attention layer
        self.attention = Attention(
            hidden_size=self.config.hidden_size,
            num_attention_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            max_position_embeddings=self.config.max_position_embeddings,
            num_layers=self.config.num_hidden_layers,
            attention_mask_type=AttentionMaskType.causal,
            bias=False,
            dtype=self.config.dtype,
            local_layer_idx=layer_idx,
        )

        # Text MLP
        self.mlp = MLP(
            hidden_size=self.config.hidden_size,
            ffn_hidden_size=self.config.intermediate_size,
            hidden_act=self.config.hidden_act,
            dtype=self.config.dtype,
            bias=False,
        )

        # Audio MLP
        self.audio_mlp = MLP(
            hidden_size=self.config.hidden_size,
            ffn_hidden_size=self.config.intermediate_size,
            hidden_act=self.config.hidden_act,
            dtype=self.config.dtype,
            bias=False,
        )

        # Layer norms
        self.input_layernorm = RmsNorm(
            normalized_shape=self.config.hidden_size,
            eps=self.config.norm_epsilon,
            dtype=self.config.dtype,
        )

        self.audio_input_layernorm = RmsNorm(
            normalized_shape=self.config.hidden_size,
            eps=self.config.norm_epsilon,
            dtype=self.config.dtype,
        )

        self.post_layernorm = RmsNorm(
            normalized_shape=self.config.hidden_size,
            eps=self.config.norm_epsilon,
            dtype=self.config.dtype,
        )

        self.audio_post_layernorm = RmsNorm(
            normalized_shape=self.config.hidden_size,
            eps=self.config.norm_epsilon,
            dtype=self.config.dtype,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        use_cache: bool = False,
        kv_cache_params: Optional[KeyValueCacheParams] = None,
        attention_params: Optional[AttentionParams] = None,
        vision_token_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass for dual FFN decoder layer."""
        residual = hidden_states

        hidden_states = where(
            vision_token_mask,
            self.audio_input_layernorm(hidden_states),
            self.input_layernorm(hidden_states),
        )

        hidden_states = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
        )

        if use_cache:
            hidden_states, presents = hidden_states

        hidden_states = residual + hidden_states

        residual = hidden_states

        residual += where(
            vision_token_mask,
            self.audio_mlp(self.audio_post_layernorm(hidden_states)),
            self.mlp(self.post_layernorm(hidden_states)),
        )

        hidden_states = residual

        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class HiggsAudioTransformer(Module):
    """TensorRT-LLM transformer component for Higgs Audio model."""

    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.config = config

        self.vocab_embedding = Embedding(
            num_embeddings=self.config.text_vocab_size,
            embedding_dim=self.config.hidden_size,
            dtype=self.config.dtype,
        )

        self.audio_codebook_embeddings = Embedding(
            num_embeddings=self.config.audio_vocab_size,
            embedding_dim=self.config.hidden_size,
            dtype=self.config.dtype,
        )

        self.layers = DecoderLayerList(HiggsAudioDualFFNDecoderLayer, config)

        self.ln_f = RmsNorm(
            normalized_shape=self.config.hidden_size,
            eps=self.config.norm_epsilon,
            dtype=self.config.dtype,
        )

    def _embed_audio_ids(self, audio_ids: Tensor):
        """Embed the audio ids"""
        num_codebooks = self.config.audio_num_codebooks
        codebook_size = self.config.audio_codebook_size
        codebook_shift = (arange(0, num_codebooks, "int32") * codebook_size).unsqueeze(-1)
        audio_embed = sum(
            self.audio_codebook_embeddings(audio_ids + codebook_shift),
            dim=0,
        )
        return audio_embed

    def forward(
        self,
        hidden_states: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        use_cache: bool = False,
        attention_mask: Optional[Tensor] = None,
        kv_cache_params: Optional[KeyValueCacheParams] = None,
        attention_params: Optional[AttentionParams] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass for Higgs Audio transformer with multimodal support."""

        bos_mask = input_ids == self.config.audio_stream_bos_id
        # The idea here is that only on the first pass will the input_ids will be larger than audio_num_codebooks and this code is to separate the input audio and text if the a voice clone reference is being used
        if input_ids.shape[0] > self.config.audio_num_codebooks and bos_mask.any():
            start_idx = torch.argmax(bos_mask.float()).item()
            eos_mask = input_ids[start_idx + 1 :] == self.config.audio_stream_eos_id
            eos_idx = torch.argmax(eos_mask.float()).item() + start_idx
            audio_mask = where(position_ids >= start_idx and position_ids <= eos_idx, True, False)
        # This is all subsequent passes where I currently route everything through the audio path.  Is that the right thing to do? I have no idea.
        elif input_ids.shape[0] <= self.config.audio_num_codebooks:
            audio_mask = where(True, True, False)
        # This is the first pass if no voice clone is being  used
        else:
            audio_mask = where(False, True, False)

        audio_ids = where(audio_mask, input_ids, 0)
        audio_embed = self._embed_audio_ids(audio_ids)
        text_ids = where(audio_mask, 0, input_ids)
        text_embed = self.vocab_embedding(text_ids)
        input_embed = text_embed + audio_embed

        hidden_states = self.layers(
            hidden_states=input_embed,
            use_cache=use_cache,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            vision_token_mask=audio_mask.unsqueeze(-1),
        )

        if use_cache:
            hidden_states, presents = hidden_states

        hidden_states = self.ln_f(hidden_states)

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class HiggsAudioForCausalLM(DecoderModelForCausalLM):
    """TensorRT-LLM implementation of Higgs Audio multimodal model."""

    def __init__(self, config: HiggsAudioConfig):
        # Initialize the transformer component
        transformer = HiggsAudioTransformer(config)

        lm_head = ColumnLinear(
            in_features=config.hidden_size,
            out_features=config.audio_vocab_size,
            bias=False,
            dtype=config.dtype,
        )

        super().__init__(config, transformer, lm_head)

    @classmethod
    def from_hugging_face(
        cls,
        hf_config_or_dir: str = "bosonai/higgs-audio-v2-generation-3B-base",
        dtype: str = "bfloat16",
        mapping: Optional[Mapping] = None,
        quant_config: Optional[QuantConfig] = None,
        **kwargs,
    ):
        """Create a HiggsAudioForCausalLM object from HuggingFace model directory.

        Args:
            hf_config_or_dir: Path to the HuggingFace model directory
            dtype: Data type for the model weights
            mapping: Multi-GPU mapping configuration
            quant_config: Quantization configuration
            **kwargs: Additional keyword arguments

        Returns:
            HiggsAudioForCausalLM: The loaded model
        """
        if not os.path.exists(hf_config_or_dir):
            hf_config_or_dir = snapshot_download(repo_id=hf_config_or_dir)

        config = HiggsAudioConfig.from_hugging_face(hf_config_or_dir, **kwargs)
        custom_dict = {
            "transformer": "",
            "lm_head": "audio_decoder_proj.audio_lm_head",
            # "text_lm_head": "audio_decoder_proj.text_lm_head",
            "audio_post_layernorm": "audio_post_attention_layernorm",
            # Ensure audio codebook embeddings are loaded from HF weights
            "transformer.audio_codebook_embeddings": "audio_codebook_embeddings",
            # Ensure text vocab embeddings are loaded (embed_tokens in HF)
            "transformer.vocab_embedding": "embed_tokens",
        }
        loader = ModelWeightsLoader(hf_config_or_dir, custom_dict)
        trtllm_model = cls(config)
        loader.update_key_mapping(trtllm_model)
        loader.generate_tllm_weights(trtllm_model)

        return trtllm_model


class HiggsAudioLogitsProcessor(LogitsProcessor):
    """Custom logits processor for HiggsAudio that applies delay pattern logic during generation."""

    def __init__(self, config: HiggsAudioConfig):
        self.config = config

        self.vocab_size = config.audio_vocab_size
        self.num_codebooks = config.audio_num_codebooks
        self.codebook_size = config.audio_codebook_size
        self.stream_bos_id = config.audio_stream_bos_id
        self.stream_eos_id = config.audio_stream_eos_id
        self.request_states = {}  # Track delay pattern state per request

    def _get_or_create_state(self, req_id: int) -> dict:
        """Get or create state for a request."""
        if req_id not in self.request_states:
            self.request_states[req_id] = {
                "num_delay": 0,  # tracks delay tokens emitted
                "closed": [False] * self.num_codebooks,  # tracks which codebooks finished with EOS
                "step_mod": 0,  # rotating pointer among open codebooks
                "prev_cb": None,  # codebook used in previous step
                "counts": [0] * self.num_codebooks,  # local tokens emitted per codebook
                "step": 0,  # total generation steps
            }
        return self.request_states[req_id]

    def __call__(
        self,
        req_id: int,
        logits: torch.Tensor,
        token_ids: List[List[int]],
        stream_ptr: Optional[int],
        client_id: Optional[int],
    ) -> None:
        """Apply delay pattern logic to audio logits during generation."""

        stream = None if stream_ptr is None else torch.cuda.ExternalStream(stream_ptr)

        with torch.cuda.stream(stream):
            state = self._get_or_create_state(req_id)
            last_token_id: int = token_ids[0][-1]

            # The runtime passes logits as shape [batch=1, beam=1, vocab_size_total].
            # Our lm_head outputs only audio vocab, so slice to that range.
            view_logits = logits[0, 0, : self.vocab_size]
            audio_logits_2d = view_logits.view(self.num_codebooks, -1)

            self._apply_delay_pattern_logic(audio_logits_2d, state, last_token_id)

            # Flatten back; modification is in-place via view

    def _apply_delay_pattern_logic(self, logits: torch.Tensor, state: dict, last_token_id: int):
        """Apply delay pattern logic to audio logits.

        logits shape: (num_codebooks, codebook_size)
        codebook_size includes local tokens [0..local_vocab_size-1] and two specials:
        BOS at index stream_bos_id and EOS at index stream_eos_id.
        """

        local_vocab_size = self.codebook_size - 2

        # Initial warmup: emit BOS once per codebook in order
        if state["num_delay"] < self.num_codebooks:
            current_cb = state["num_delay"]

            # Mask all codebooks except current
            if current_cb + 1 < self.num_codebooks:
                logits[current_cb + 1 :, :] = -float("inf")
            if current_cb > 0:
                logits[:current_cb, :] = -float("inf")

            # In current codebook, allow only BOS
            logits[current_cb, :] = -float("inf")
            logits[current_cb, self.stream_bos_id] = 0.0

            # Remember which cb we targeted this step
            state["prev_cb"] = current_cb
            state["num_delay"] += 1
            return

        # After warmup: rotate among codebooks and generate local tokens, allow EOS
        # If the previous token closed a codebook, mark it
        # Identify EOS based on local id and mark the corresponding codebook closed
        last_local = last_token_id % self.codebook_size
        last_cb = (last_token_id // self.codebook_size) % self.num_codebooks
        if 0 <= last_local < (self.codebook_size - 2):
            # Count local token for that codebook
            state["counts"][last_cb] += 1
        if last_local == self.stream_eos_id:
            state["closed"][last_cb] = True

        # If all codebooks closed: force EOS at last row to terminate
        if all(state["closed"]):
            logits[:, :] = -float("inf")
            logits[-1, self.stream_eos_id] = 0.0
            return

        # Find next open codebook to target
        start = state["step_mod"] % self.num_codebooks
        current_cb = start
        for _ in range(self.num_codebooks):
            if not state["closed"][current_cb]:
                break
            current_cb = (current_cb + 1) % self.num_codebooks

        # Mask all other codebooks
        if current_cb + 1 < self.num_codebooks:
            logits[current_cb + 1 :, :] = -float("inf")
        if current_cb > 0:
            logits[:current_cb, :] = -float("inf")

        # In current codebook: disallow BOS, allow local tokens and optionally EOS
        logits[current_cb, self.stream_bos_id] = -float("inf")

        # Delay EOS until we have enough local tokens for this codebook
        min_tokens_per_cb = 200  # Much higher requirement to delay EOS longer

        # Only allow EOS if:
        # 1. We have enough local tokens for this codebook, AND
        # 2. We have generated a reasonable total amount, AND
        # 3. All codebooks have reasonable content
        enough_local = state["counts"][current_cb] >= min_tokens_per_cb

        # Count total tokens across all codebooks
        total_tokens = 0
        for count in state["counts"]:
            total_tokens += count

        # Check that all codebooks have some minimum content
        min_per_all_cb = 80  # Each codebook should have at least this many tokens
        all_cb_have_content = all(count >= min_per_all_cb for count in state["counts"])

        # Make EOS extremely restrictive - basically never allow it for now
        allow_eos = (
            enough_local  # This codebook has enough tokens
            and total_tokens >= 1000  # Much higher total output requirement
            and all_cb_have_content  # All codebooks have content
            and state["step"] > 1200  # We've generated many more steps
        )

        eos_masked = not allow_eos

        if eos_masked:
            logits[current_cb, self.stream_eos_id] = -float("inf")
        else:
            # Even when EOS is allowed, make local tokens much more attractive
            # Boost local token logits by a small amount to prefer content over termination
            logits[current_cb, :local_vocab_size] += 2.0  # Boost local tokens
            # And slightly penalize EOS to make it less attractive
            logits[current_cb, self.stream_eos_id] -= 1.0  # Slightly penalize EOS

        # mask nothing else within local range
        # But ensure any out-of-range beyond EOS are masked (safety)
        if self.codebook_size > local_vocab_size + 2:
            logits[current_cb, local_vocab_size + 2 :] = -float("inf")

        # Debug: check what's actually happening with the logits
        local_max = logits[current_cb, :local_vocab_size].max().item()
        eos_val = logits[current_cb, self.stream_eos_id].item()
        bos_val = logits[current_cb, self.stream_bos_id].item()

        with open("/home/me/TTS/TensorRT-LLM/lp_debug.txt", "a") as f:
            f.write(f"  current_cb={current_cb} eos_masked={eos_masked}\n")
            f.write(f"  local_max={local_max:.3f} eos_val={eos_val:.3f} bos_val={bos_val:.3f}\n")

            # Show top 5 local token logits
            local_logits = logits[current_cb, :local_vocab_size]
            top_vals, top_idxs = torch.topk(local_logits, min(5, local_vocab_size))
            f.write(
                f"  top_local: {[(idx.item(), val.item()) for idx, val in zip(top_idxs, top_vals)]}\n"
            )

        # Update pointers and step counter
        state["prev_cb"] = current_cb
        state["step_mod"] = (current_cb + 1) % self.num_codebooks
        state["step"] += 1


class HiggsAudioTRTRunner:
    """TensorRT-LLM inference wrapper for HiggsAudio using ModelRunnerCpp."""

    def __init__(
        self,
    ) -> None:
        """Initialize the TensorRT-LLM runner for HiggsAudio."""

        self.engine_dir = "/home/me/TTS/TensorRT-LLM/higgs_audio_engine/"
        self.hf_model_dir = "bosonai/higgs-audio-v2-generation-3B-base"
        self.audio_tokenizer_dir = "bosonai/higgs-audio-v2-tokenizer"
        self.reference_audio = None  # Disable reference audio loading for faster testing
        # self.reference_audio = "/home/me/TTS/TensorRT-LLM/AussieGirl.wav"

        self.config = HiggsAudioConfig.from_hugging_face()

        self.temperature = 1.0
        self.top_k = 50
        self.top_p = 0.95
        self.num_beams = 1
        self.max_num_tokens = self.config.build_config["max_num_tokens"]

        # Set up device
        self.device = torch.device("cuda", 0)
        torch.cuda.set_device(self.device)

        # Load components
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_dir, trust_remote_code=True)
        self.audio_tokenizer = AudioTokenizer(self.audio_tokenizer_dir)

        # Create custom logits processor for delay pattern handling
        self.audio_logits_processor = HiggsAudioLogitsProcessor(self.config)

        from tensorrt_llm.runtime import ModelRunnerCpp

        self.runner = ModelRunnerCpp.from_dir(
            engine_dir=self.engine_dir,
            kv_cache_free_gpu_memory_fraction=0.5,
            # use_gpu_direct_storage=True,
            # cuda_graph_mode=True,
            logits_processor_map={"higgs_audio_logit_processor": self.audio_logits_processor},
        )

        # Preload the part of the input that doesn't change
        if self.reference_audio and self.audio_tokenizer:
            # Load and transcribe reference audio for voice cloning
            whisper_model_id = "openai/whisper-large-v3-turbo"
            whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(whisper_model_id)
            processor = AutoProcessor.from_pretrained(whisper_model_id)
            audio, _ = librosa.load(self.reference_audio, sr=16000)
            pipe = pipeline(
                "automatic-speech-recognition",
                model=whisper_model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                return_timestamps=True,
            )
            transcription = pipe(audio)["text"]

            # Validate audio file exists
            if not os.path.exists(self.reference_audio):
                raise FileNotFoundError(f"Reference audio file not found: {self.reference_audio}")

            audio_ids = self.audio_tokenizer.encode(self.reference_audio, sr=24000)
            # Apply delay pattern if requested and we have multiple codebooks
            # Add BOS and EOS tokens using correct token IDs
            bos_tokens = torch.full(
                (audio_ids.shape[0], 1),
                self.config.audio_stream_bos_id,
                dtype=audio_ids.dtype,
                device=self.device,
            )
            eos_tokens = torch.full(
                (audio_ids.shape[0], 1),
                self.config.audio_stream_eos_id,
                dtype=audio_ids.dtype,
                device=self.device,
            )
            # Concatenate: BOS + audio_ids + EOS
            audio_ids = torch.cat([bos_tokens, audio_ids, eos_tokens], dim=-1)

            # Apply delay pattern
            audio_ids = _build_delay_pattern_mask(
                audio_ids,
                bos_token_id=self.config.audio_stream_bos_id,
                pad_token_id=self.config.audio_stream_eos_id,
            ).flatten()
            import numpy as np

            np.savetxt("log2.txt", audio_ids.cpu().view(-1), delimiter=",", fmt="%d")
            # Format with reference audio (voice cloning) following Higgs Audio expected format
            # The format should include the reference audio transcription and then the target text
            pre_audio_input = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
                f"Generate audio following instruction.<|scene_desc_start|>"
                f"Audio is recorded from a quiet room."
                f"Speaker is an enthusiastic young Australian woman in her early 20s."
                f"She has a bright, high-pitched voice.<|scene_desc_end|><|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>{transcription}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|><|audio_bos|>"
            )
            pre_audio_input_ids = (
                self.tokenizer.encode(pre_audio_input, return_tensors="pt")
                .to(self.device)
                .flatten()
            )
            post_audio_input = f"<|audio_eos|><|eot_id|><|start_header_id|>user<|end_header_id|>"
            post_audio_input_ids = (
                self.tokenizer.encode(post_audio_input, return_tensors="pt")
                .to(self.device)
                .flatten()
            )

            self.saved_input_ids = torch.cat([pre_audio_input_ids, audio_ids, post_audio_input_ids])
        else:
            # Format without reference audio (default voice)
            # Simplified format for direct text-to-speech without voice cloning
            text_input = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
                f"Generate audio following instruction.<|scene_desc_start|>"
                f"Audio is recorded from a quiet room."
                f"Speaker is an enthusiastic young Australian woman in her early 20s."
                f"She has a bright, high-pitched voice.<|scene_desc_end|><|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>"
            )
            self.saved_input_ids = (
                self.tokenizer.encode(text_input, return_tensors="pt").to(self.device).flatten()
            )

    def generate(
        self,
        input_text: str,
        **generation_kwargs,
    ):
        """Generate audio from text input and reference audio (TTS with voice cloning).

        Args:
            input_text: The text prompt to convert to speech
            input_audio: Path to reference audio file for voice cloning

        Returns:
            Generated audio tensor suitable for Whisper transcription"""

        text_input = (
            f"{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|><|audio_out_bos|>"
        )

        input_ids = self.tokenizer.encode(text_input, return_tensors="pt").to(self.device).flatten()
        input_ids = torch.cat([self.saved_input_ids, input_ids])

        max_input_length = torch.tensor([input_ids.size(0)], dtype=torch.int32).max().item()
        max_new_tokens = self.max_num_tokens - max_input_length

        with torch.no_grad():
            outputs = self.runner.generate(
                batch_input_ids=[input_ids],
                max_new_tokens=max_new_tokens,
                end_id=self.config.audio_stream_eos_id,
                temperature=float(self.temperature),
                top_k=int(self.top_k),
                top_p=float(self.top_p),
                logits_processor_names=["higgs_audio_logit_processor"],
            )

        # Extract and process audio tokens with proper delay pattern handling
        try:
            import numpy as np

            vq_code = self._extract_and_process_audio_tokens(outputs[0, 0])
            print(f"Extracted audio tokens shape: {vq_code.shape}")

            # Decode to waveform
            waveform, sr = self.audio_tokenizer.decode(vq_code)

            # Save waveform debug safely for both numpy and torch tensors
            try:
                if isinstance(waveform, torch.Tensor):
                    np.savetxt(
                        "log2.txt",
                        waveform.detach().cpu().view(-1).numpy(),
                        delimiter=",",
                        fmt="%d",
                    )
                else:
                    np.savetxt(
                        "log2.txt", np.asarray(waveform).reshape(-1), delimiter=",", fmt="%d"
                    )
            except Exception:
                pass
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.detach().cpu().numpy()
            if sr != 16000 and isinstance(waveform, np.ndarray):
                waveform = librosa.resample(
                    waveform.astype(np.float32), orig_sr=sr, target_sr=16000
                )
                sr = 16000
            # Normalize amplitude
            if isinstance(waveform, np.ndarray):
                maxv = np.max(np.abs(waveform)) + 1e-8
                waveform = (waveform / maxv) * 0.98
            return waveform.astype(np.float32)
        except Exception as e:
            print(f"Error processing audio tokens: {e}")
            raise

    def _extract_and_process_audio_tokens(self, generated_tokens):
        """Extract and process audio tokens with proper delay pattern handling."""
        if not isinstance(generated_tokens, torch.Tensor):
            generated_tokens = torch.as_tensor(generated_tokens)

        print(f"Input generated_tokens shape: {generated_tokens.shape}")

        # Identify start of audio generation right AFTER all codebooks emitted BOS once
        num_codebooks = self.config.audio_num_codebooks
        codebook_size = self.config.audio_codebook_size  # includes BOS/EOS
        local_vocab_size = codebook_size - 2

        tokens = generated_tokens.flatten()
        local_vals = tokens % codebook_size
        seen_bos = [False] * num_codebooks
        start_idx = None
        for i, (t, lv) in enumerate(zip(tokens.tolist(), local_vals.tolist())):
            if lv == self.config.audio_stream_bos_id:
                cb = (t // codebook_size) % num_codebooks
                if 0 <= cb < num_codebooks:
                    seen_bos[cb] = True
                if all(seen_bos):
                    start_idx = i + 1  # start AFTER the warmup BOS markers
                    break
        if start_idx is None:
            # Fallback to first BOS if we never saw all BOS markers
            bos_any_mask = local_vals == self.config.audio_stream_bos_id
            if not bool(bos_any_mask.any()):
                raise ValueError("No audio_stream_bos_id found in generated tokens")
            start_idx = int(torch.nonzero(bos_any_mask, as_tuple=False)[0])
        stream = tokens[start_idx:]

        print(f"Raw audio tokens length: {len(stream)}")

        # Flattened parsing: attribute tokens to codebooks by value
        per_cb: List[List[int]] = [[] for _ in range(num_codebooks)]
        closed = [False] * num_codebooks
        for t in stream.tolist():
            cb = (t // codebook_size) % num_codebooks
            local = t % codebook_size
            if local == self.config.audio_stream_bos_id:
                continue
            if local == self.config.audio_stream_eos_id:
                closed[cb] = True
                if all(closed):
                    break
                continue
            if closed[cb]:
                continue
            if 0 <= local < local_vocab_size:
                per_cb[cb].append(int(local))

        # Debug: write simple histograms of local token usage per codebook
        try:
            with open("token_hist.txt", "w") as fh:
                for cb in range(num_codebooks):
                    vals = per_cb[cb]
                    if not vals:
                        fh.write(f"cb{cb}: empty\n")
                        continue
                    import collections

                    c = collections.Counter(vals)
                    # top 10 most common
                    common = c.most_common(10)
                    fh.write(f"cb{cb}: {common}\n")
        except Exception:
            pass

        # Reconstruct delay-pattern grid: each row i is padded with i BOS at the start
        # and post-padded with EOS to a common delayed length, then remove delay.
        Q = num_codebooks
        local_vocab_size = local_vocab_size  # alias
        # Determine base content length (max across codebooks)
        content_lengths = [len(x) for x in per_cb]
        if max(content_lengths) == 0:
            raise ValueError("No valid audio tokens after parsing stream")
        base_T = max(content_lengths)
        delayed_len = base_T + Q - 1
        grid = torch.full(
            (Q, delayed_len),
            fill_value=self.config.audio_stream_eos_id,
            dtype=torch.long,
            device=generated_tokens.device,
        )
        for cb in range(Q):
            # prefix BOS repeated cb times
            if cb > 0:
                grid[cb, :cb] = self.config.audio_stream_bos_id
            seq = per_cb[cb]
            # write content starting at offset cb, up to base_T tokens (truncate/pad if needed)
            seq_t = torch.tensor(seq[:base_T], dtype=torch.long, device=generated_tokens.device)
            write_len = seq_t.numel()
            if write_len > 0:
                grid[cb, cb : cb + write_len] = seq_t
            # remaining tail positions already filled with EOS

        # Remove delay pattern to align time frames across codebooks
        vq_delayed = grid
        vq_code_local = revert_delay_pattern(vq_delayed)
        # Ensure we only return local token range [0..local_vocab_size-1]
        vq_code_local = torch.clamp(vq_code_local, 0, local_vocab_size - 1)
        print(f"Reshaped audio tokens: {vq_code_local.shape}")
        return vq_code_local
