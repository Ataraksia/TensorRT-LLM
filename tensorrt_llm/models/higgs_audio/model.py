# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
"""TensorRT-LLM implementation of Higgs Audio multimodal model."""

import builtins
import librosa
from collections import defaultdict, deque, Counter
from collections.abc import AsyncGenerator, Sequence
import os
from pathlib import Path
from typing import Any, Dict, Optional, List, OrderedDict, Union
import numpy as np
from openai.types.chat import ChatCompletionAudio
import tensorrt
import torch
from starlette.datastructures import State
from huggingface_hub import snapshot_download
from torch import nn
from transformers import (
    AutoModel,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    pipeline,
)
from tensorrt_llm._common import default_net
import torchaudio
import tensorrt_llm
import torch.nn.functional as F

from tensorrt_llm.bindings import INT32
from tensorrt_llm.mapping import Mapping

from tensorrt_llm.models.model_weights_loader import ModelWeightsLoader
from tensorrt_llm.models.modeling_utils import (
    DecoderLayerList,
    PretrainedConfig,
    PretrainedModel,
    QuantConfig,
    DecoderModelForCausalLM,
    cp_split_plugin,
)
from tensorrt_llm.module import Module, ModuleList
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.sampling_params import LogitsProcessor
from tensorrt_llm.functional import (
    Tensor,
    allgather,
    arange,
    constant,
    cumsum,
    expand,
    expand_dims_like,
    int32_array,
    nonzero,
    op_or,
    pad,
    shape,
    unsqueeze,
    view,
    where,
    sum,
    mean,
    concat,
    index_select,
    gather_last_token_logits,
    op_and,
    __ge__,
    __le__,
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
from .descriptaudiocodec.dac.model import dac as dac2
from .quantization.vq import ResidualVectorQuantizer
from .semantic.semantic_module import Encoder, Decoder
from vector_quantize_pytorch import ResidualFSQ


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
            self.quantizer = ResidualFSQ(dim=self.quantizer_dim, levels=[bins], num_quantizers=n_q)
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
        if sr != self.sample_rate:
            wv = librosa.resample(wv, orig_sr=sr, target_sr=self.sample_rate)
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

    def decode(self, vq_code):
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
                vq_code = vq_code.to(self.device)
            else:
                vq_code = torch.from_numpy(vq_code).to(self.device)
            codes = vq_code.unsqueeze(0)
            overlap_width = 16
            chunk_size = 60 * self.frame_rate
            chunk_output_length = self.xcodec_get_output_length(chunk_size)
            outputs = []
            # split the codes into chunks, with overlap at the beginning and end
            for i in range(0, codes.shape[-1], chunk_size):
                begin = max(0, i - overlap_width)
                end = min(i + chunk_size + overlap_width, codes.shape[-1])
                chunk = codes[:, :, begin:end]
                output = self._xcodec_decode(chunk)
                if i == 0:
                    output = output[:, :, :chunk_output_length]
                elif i + chunk_size >= codes.shape[-1]:
                    last_chunk_size = codes.shape[-1] - i
                    last_chunk_output_length = self.xcodec_get_output_length(last_chunk_size)
                    output = output[:, :, -last_chunk_output_length:]
                else:
                    extra_length = (
                        self.xcodec_get_output_length(chunk_size + overlap_width * 2)
                        - chunk_output_length
                    ) // 2
                    output = output[:, :, extra_length:-extra_length]
                outputs.append(output)

            decoded_wv = np.concatenate(outputs, axis=2)[0, 0]

            return decoded_wv, self.sample_rate

    def _xcodec_decode(self, vq_code: torch.Tensor) -> torch.Tensor:
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

    def xcodec_get_output_length(self, input_length: int):
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

        self.codebook_embeddings = Embedding(
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
        # Simply repeat the audio_ids for each codebook
        audio_ids = audio_ids.unsqueeze(0)  # Shape: (1, seq_len)

        # Create codebook shifts
        codebook_shift = (
            arange(0, self.config.audio_num_codebooks, dtype="int32")
            * self.config.audio_codebook_size
        ).unsqueeze(-1)  # Shape: (num_codebooks, 1)

        # Broadcast addition will handle the expansion automatically
        shifted_ids = audio_ids + codebook_shift  # Shape: (num_codebooks, seq_len)

        # Get embeddings and sum
        audio_embed = sum(self.codebook_embeddings(shifted_ids), dim=0)
        return audio_embed  # Shape: (seq_len, embedding_dim)

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
        audio_mask = op_or(
            op_and(
                position_ids >= self.config.audio_in_start,
                position_ids <= self.config.audio_in_end,
            ),
            position_ids >= self.config.input_length,
        )

        audio_ids = where(audio_mask, input_ids, 0)
        audio_embed = self._embed_audio_ids(audio_ids)
        text_ids = where(audio_mask, 0, input_ids)
        text_embed = self.vocab_embedding(text_ids)
        input_embed = where(audio_mask.unsqueeze(-1), audio_embed, text_embed)

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


class HiggsAudioForCausalLM(PretrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.transformer = HiggsAudioTransformer(config)

        self.lm_head = ColumnLinear(
            in_features=config.hidden_size,
            out_features=config.audio_vocab_size,
            bias=False,
            dtype=config.dtype,
        )

        self.mup_width_multiplier = getattr(config, "mup_width_multiplier", None)
        Attention.create_attention_const_params(self, config)
        self.position_embedding_type = config.position_embedding_type

    def forward(
        self,
        input_ids: Tensor,
        position_ids=None,
        use_cache=False,
        last_token_ids=None,
        attention_mask=None,
        kv_cache_params=None,
        attention_params=None,
        mrope_params=None,
        hidden_states=None,
        prompt_embedding_table: Optional[Tensor] = None,
        prompt_tasks: Optional[Tensor] = None,
        prompt_vocab_size: Optional[Tensor] = None,
        lora_params=None,
        spec_decoding_params=None,
    ):
        # fill attention params.
        attention_params = Attention.fill_attention_params(self, attention_params)

        # split the sequence for context parallelism
        if self.config.mapping.cp_size > 1:
            if len(input_ids.shape) == 1:
                # input shape is [-1]
                input_ids, cp_join_index = cp_split_plugin(
                    input_ids,
                    attention_params.host_request_types,
                    attention_params.host_context_lengths,
                    self.config.mapping.cp_size,
                    self.config.mapping.cp_rank,
                )
            else:
                assert False, "Context parallelism with non-remove-padding is not supported yet."

        kwargs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
            "kv_cache_params": kv_cache_params,
            "attention_params": attention_params,
        }
        if lora_params is not None:
            kwargs["lora_params"] = lora_params
        if hidden_states is not None:
            kwargs["hidden_states"] = hidden_states
        if prompt_embedding_table is not None:
            kwargs["prompt_embedding_table"] = prompt_embedding_table
        if prompt_tasks is not None:
            kwargs["prompt_tasks"] = prompt_tasks
        if prompt_vocab_size is not None:
            kwargs["prompt_vocab_size"] = prompt_vocab_size

        if spec_decoding_params is not None:
            kwargs["spec_decoding_params"] = spec_decoding_params
        if mrope_params is not None:
            kwargs["mrope_params"] = mrope_params

        hidden_states = self.transformer.forward(**kwargs)

        if use_cache:
            hidden_states, presents = hidden_states

        # All gather and rebuild sequence after transformer layer for context parallelism
        if self.config.mapping.cp_size > 1:
            if len(hidden_states.shape) == 2:
                hidden_states = allgather(hidden_states, self.config.mapping.cp_group, gather_dim=0)
                hidden_states = view(hidden_states, [-1, hidden_states.shape[-1]])
                hidden_states = index_select(hidden_states, 0, cp_join_index)
            else:
                assert False, "Context parallelism with non-remove-padding is not supported yet."

        if self.config.mapping.is_last_pp_rank():
            all_hidden_states = hidden_states
            hidden_states = gather_last_token_logits(
                hidden_states, last_token_ids, default_net().plugin_config.remove_input_padding
            )

            # [batch_size, hidden_size] -> [batch_size, vocab_size]
            lm_logits = self.lm_head(hidden_states)
            if hasattr(self.config, "output_multiplier_scale"):
                lm_logits *= getattr(self.config, "output_multiplier_scale", 1)
            if self.mup_width_multiplier is not None:
                lm_logits = lm_logits / self.mup_width_multiplier
            lm_logits.mark_output("logits", self.config.logits_dtype)

        else:
            hidden_states.mark_output("hidden_states_output", self.config.dtype)

        if use_cache and not default_net().plugin_config.paged_kv_cache:
            for i, present in zip(
                self.config.mapping.pp_layers(self.config.num_hidden_layers), presents
            ):
                present.mark_output(f"present_key_value_{i}", self.config.kv_dtype)
            if self.config.mapping.is_last_pp_rank():
                return (lm_logits, presents, hidden_states)
            return (hidden_states, presents)
        else:
            if self.config.mapping.is_last_pp_rank():
                return lm_logits, hidden_states, all_hidden_states
            return hidden_states

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
            "codebook_embeddings": "audio_codebook_embeddings",
        }
        loader = ModelWeightsLoader(hf_config_or_dir, custom_dict)
        trtllm_model = cls(config)
        loader.update_key_mapping(trtllm_model)
        loader.generate_tllm_weights(trtllm_model)

        return trtllm_model


class HiggsAudioLogitsProcessor(LogitsProcessor):
    """Custom logits processor for Higgs Audio to handle delay pattern and streaming generation."""

    def __init__(self, config):
        self.config = config
        self.num_codebooks = config.audio_num_codebooks
        self.vocab_size = config.audio_vocab_size
        self.local_vocab_size = config.audio_codebook_size
        self.stream_bos_id = config.audio_stream_bos_id  # Local index, e.g., 1024
        self.stream_eos_id = config.audio_stream_eos_id  # Local index, e.g., 1025
        self.audio_eos_token_id = config.audio_eos_token_id  # Global text token for <|audio_eos|>

        self.request_states: Dict[int, Dict] = {}  # Per-request state
        self.lookback_frames = 2  # Inspect last N frames for EOS detection (tune as needed)

    def __call__(
        self,
        req_id: int,
        logits: torch.Tensor,  # Shape: (batch_size=1, sequence_size=1, audio_vocab_size)
        token_ids: List[List[int]],  # List of sequences, assume batch_size=1
        stream_ptr: Optional[int] = None,
        client_id: Optional[int] = None,
    ) -> None:
        if not token_ids or len(token_ids[0]) == 0:
            return

        audio_logits = logits.view(-1)  # Shape: (audio_vocab_size,)
        current_seq = token_ids[0]  # Assume batch_size=1
        seq_len = len(current_seq)

        if req_id not in self.request_states:
            self.request_states[req_id] = {
                "audio_start_len": seq_len - 1,
                "last_detected_eos_frame": -1,  # Track frame where EOS first appeared
                "eos_codebook_idx": None,  # Codebook where EOS was first sampled
            }
        state = self.request_states[req_id]

        audio_start_len = state["audio_start_len"]
        relative_pos = seq_len - 1 - audio_start_len  # 0-based position in audio block
        if relative_pos < 0:
            return

        time_step = relative_pos // self.num_codebooks  # Current frame (0-based)
        cb_idx = relative_pos % self.num_codebooks  # Codebook index in frame (0 to num_codebooks-1)

        # Step 1: Inspect recent history to detect if/where EOS was sampled recently
        lookback_start = max(0, seq_len - (self.lookback_frames * self.num_codebooks))
        recent_audio_rel_pos = [
            i - audio_start_len for i in range(lookback_start, seq_len) if i >= audio_start_len
        ]

        # Find the most recent frame where EOS appeared in any codebook
        eos_detected = False
        for rel_pos in reversed(recent_audio_rel_pos):  # Scan backward from current
            recent_time_step = rel_pos // self.num_codebooks
            recent_cb_idx = rel_pos % self.num_codebooks
            global_idx = audio_start_len + rel_pos  # Absolute position in seq

            # Check if this token was an EOS in its codebook slice
            token_val = current_seq[global_idx]
            cb_start = recent_cb_idx * self.local_vocab_size
            local_token = (token_val - cb_start) if token_val >= cb_start else -1
            if (
                local_token == self.stream_eos_id
                and recent_time_step >= state["last_detected_eos_frame"]
            ):
                # Found a new/recent EOS
                state["last_detected_eos_frame"] = recent_time_step
                state["eos_codebook_idx"] = recent_cb_idx
                eos_detected = True
                break

        # Step 2: Compute current num_delay (for BOS forcing, as before)
        current_num_delay = min(time_step, self.num_codebooks - 1)

        # Step 3: Compute num_remaining_delays for staggered EOS
        num_remaining_delays = None
        if eos_detected and state["eos_codebook_idx"] is not None:
            # Delays remaining = num_codebooks - 1 - eos_codebook_idx (propagate to earlier CBs)
            # But only if we're in a subsequent frame
            if time_step > state["last_detected_eos_frame"]:
                num_remaining_delays = self.num_codebooks - 1 - state["eos_codebook_idx"]
                # Decay it over steps (one less per frame after detection)
                num_remaining_delays = max(
                    0, num_remaining_delays - (time_step - state["last_detected_eos_frame"] - 1)
                )

        # Step 4: Modify logits for current codebook slice
        cb_start = cb_idx * self.local_vocab_size
        cb_end = cb_start + self.local_vocab_size

        # Always mask invalid tokens outside 0 to local_vocab_size-1
        audio_logits[cb_start:cb_end] = float("-inf")
        valid_slice = slice(cb_start, cb_start + self.local_vocab_size)

        # Force BOS for delayed positions
        if cb_idx > current_num_delay:
            bos_global_idx = cb_start + self.stream_bos_id
            audio_logits[valid_slice] = float("-inf")
            audio_logits[bos_global_idx] = 0.0  # Unmask BOS
            return

        # For content positions: Allow content (0 to codebook_size-1), mask BOS
        audio_logits[cb_start + self.stream_bos_id] = float("-inf")  # Mask BOS

        # Staggered EOS forcing
        if num_remaining_delays is not None and cb_idx < num_remaining_delays:
            # Force EOS: mask everything except EOS
            eos_global_idx = cb_start + self.stream_eos_id
            audio_logits[valid_slice] = float("-inf")
            audio_logits[eos_global_idx] = 0.0  # Unmask EOS
            # If this is the last codebook and all are forced, next text step can be <audio_eos>
            if cb_idx == self.num_codebooks - 1 and num_remaining_delays <= 1:
                # Note: Can't force text EOS here; handle in post-gen if needed
                pass
        else:
            # Normal: Allow content + EOS (but bias EOS higher in later codebooks if desired)
            eos_global_idx = cb_start + self.stream_eos_id
            if cb_idx != self.num_codebooks - 1:  # Soften EOS in non-last CBs
                audio_logits[eos_global_idx] = float("-inf") * 0.5  # Or use a lower penalty
            else:
                audio_logits[eos_global_idx] = 0.0  # Unmask EOS
        # Optional: Boost EOS probability in later frames to encourage termination
        if time_step > 10:  # Tune threshold
            audio_logits[cb_start + self.stream_eos_id] += 0.5  # logit boost


class HiggsAudioTRTRunner:
    """TensorRT-LLM inference wrapper for HiggsAudio using ModelRunnerCpp."""

    def __init__(
        self,
    ) -> None:
        """Initialize the TensorRT-LLM runner for HiggsAudio."""

        repo_root = Path(__file__).resolve().parents[3]
        default_engine_dir = repo_root / "higgs_audio_engine"
        engine_dir_env = os.environ.get("HIGGS_AUDIO_ENGINE_DIR")
        engine_path = Path(engine_dir_env) if engine_dir_env else default_engine_dir
        if not engine_path.exists():
            raise FileNotFoundError(
                "Higgs Audio TensorRT engine not found. Build the engine with "
                "build_engine.py or set HIGGS_AUDIO_ENGINE_DIR to the engine directory."
            )
        self.engine_dir = str(engine_path)
        self.hf_model_dir = "bosonai/higgs-audio-v2-generation-3B-base"
        self.audio_tokenizer_dir = "bosonai/higgs-audio-v2-tokenizer"
        self.reference_audio = None  # Disable reference audio loading for faster testing
        # self.reference_audio = "/home/me/TTS/TensorRT-LLM/AussieGirl.wav"
        self.config = HiggsAudioConfig.from_hugging_face(engine_dir=self.engine_dir)
        # self.config.audio_in_start = 0
        # self.config.audio_in_end = 0

        self.temperature = 1.0
        self.top_k = 50
        self.top_p = 0.95
        self.num_beams = 1
        self.max_num_tokens = self.config.max_num_tokens

        self.num_codebooks = self.config.audio_num_codebooks
        self.stream_bos_id = self.config.audio_stream_bos_id
        self.stream_eos_id = self.config.audio_stream_eos_id
        self.audio_codebook_size = self.config.audio_codebook_size

        # Set up device
        self.device = torch.device("cuda", 0)
        torch.cuda.set_device(self.device)

        # Load components
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_dir, trust_remote_code=True)
        self.audio_tokenizer = load_higgs_audio_tokenizer(self.audio_tokenizer_dir)

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

            audio_ids = self.audio_tokenizer.encode(self.reference_audio)

            # Apply delay pattern
            audio_ids = _build_delay_pattern_mask(
                audio_ids,
                bos_token_id=self.stream_bos_id,
                pad_token_id=self.stream_eos_id,
            ).unsqueeze(-1)
            audio_ids = (audio_ids).flatten()

            # Format with reference audio (voice cloning) following Higgs Audio expected format
            # Don't change this!
            pre_audio_input = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"Generate audio following instruction.<|scene_desc_start|>"
                f"Audio is recorded from a quiet room."
                f"Speaker is an enthusiastic young Australian woman in her early 20s."
                f"She has a bright, high-pitched voice.<|scene_desc_end|><|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{transcription}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n<|audio_bos|>"
            )
            pre_audio_input_ids = (
                self.tokenizer.encode(pre_audio_input, return_tensors="pt")
                .to(self.device)
                .flatten()
            )
            post_audio_input = (
                f"<|audio_eos|><|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            )
            post_audio_input_ids = (
                self.tokenizer.encode(post_audio_input, return_tensors="pt")
                .to(self.device)
                .flatten()
            )
            # self.config.audio_in_start = pre_audio_input_ids.size(0)
            # self.config.audio_in_end = pre_audio_input_ids.size(0) + audio_ids.size(0) - 1
            input_ids = torch.cat([pre_audio_input_ids, audio_ids, post_audio_input_ids])
            np.savetxt("audio_ids_in.txt", audio_ids.cpu().view(8, -1), delimiter=",", fmt="%d")
        else:
            # Simplified format for direct text-to-speech without voice cloning
            # Don't change this!
            text_input = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
                f"\n\nGenerate audio following instruction.<|scene_desc_start|>"
                f"Audio is recorded from a quiet room."
                f"Speaker is an enthusiastic young Australian woman in her early 20s."
                f"She has a bright, high-pitched voice.<|scene_desc_end|><|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
            )
            input_ids = (
                self.tokenizer.encode(text_input, return_tensors="pt").to(self.device).flatten()
            )

        self.saved_input_ids = input_ids

    def generate(
        self,
        input_text: str,
        **generation_kwargs,
    ):
        """Generate audio from text input and optional reference audio.

        Args:
            input_text: The text prompt to convert to speech.

        Returns:
            A waveform tensor containing the generated audio suitable for
            Whisper transcription.
        """

        # Don't change this!
        text_input = (
            f"{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"<|audio_out_bos|>"
        )
        next_audio_token = torch.full(
            (self.config.audio_num_codebooks,),
            self.stream_bos_id,
            dtype=torch.long,
            device=self.device,
        ).flatten()

        input_ids = self.tokenizer.encode(text_input, return_tensors="pt").to(self.device).flatten()
        input_ids = torch.cat([self.saved_input_ids, input_ids, next_audio_token])
        self.config.input_length = input_ids.size(0)
        self.config.audio_in_start = 0
        self.config.audio_in_end = self.config.input_length - 1
        max_new_tokens = self.max_num_tokens - self.config.input_length

        with torch.no_grad():
            self.config.input_length = input_ids.size(0)
            self.audio_logits_processor.config = self.config

            outputs = self.runner.generate(
                batch_input_ids=[input_ids],
                max_new_tokens=max_new_tokens,
                end_id=int(self.stream_eos_id),
                temperature=float(self.temperature),
                top_k=int(self.top_k),
                top_p=float(self.top_p),
                logits_processor_names=["higgs_audio_logit_processor"],
            )

        try:
            audio_tokens = outputs[0, self.config.input_length :].cpu().numpy()
            np.savetxt("audio_ids_out.txt", audio_tokens.view(8, -1), delimiter=",", fmt="%d")
            waveform = self.process_audio(audio_tokens)
            print(f"Successfully generated audio: {len(waveform)} samples")
            return waveform
        except Exception as e:
            print(f"Error in audio processing: {e}")
            # Fallback: Generate a short silence or error waveform
            return torch.zeros(16000 * 1)  # 1s silence at 16kHz

    def process_audio(self, generated: np.ndarray):
        """Process generated audio tokens with EOS fixup and decode to waveform."""
        if generated.size == 0:
            raise RuntimeError("No audio tokens were generated by the model.")

        # Trim to complete frames (multiple of num_codebooks)
        remainder = generated.size % self.num_codebooks
        if remainder != 0:
            generated = generated[:-remainder]
        if generated.size == 0:
            raise RuntimeError("Generated tokens did not contain any complete audio frames.")

        # Reshape to frames: (num_frames, num_codebooks)
        frames = generated.reshape(-1, self.num_codebooks).astype(np.int64)
        print(f"Initial frames shape: {frames.shape}")

        # Log initial token stats
        content_tokens = np.sum((frames >= 0) & (frames < self.audio_codebook_size))
        eos_tokens = np.sum(frames == self.stream_eos_id)
        print(
            f"Initial stats - Content (0-{self.audio_codebook_size - 1}): {content_tokens}, "
            f"EOS ({self.stream_eos_id}): {eos_tokens}"
        )

        # BOS and EOS rows for reference
        bos_row = np.full((self.num_codebooks,), self.stream_bos_id, dtype=np.int64)
        eos_row = np.full((self.num_codebooks,), self.stream_eos_id, dtype=np.int64)

        # Trim leading BOS rows (from initial frame)
        while frames.size > 0 and np.all(frames[0] == bos_row):
            frames = frames[1:]

        # Post-generation EOS fixup: Ensure staggered EOS propagation in the last frame(s)
        max_frames_to_fix = 2  # Tune: Check/fix last N frames
        if frames.size > 0:
            last_frames_start = max(0, len(frames) - max_frames_to_fix)
            last_frames = frames[last_frames_start:]

            # Find the earliest codebook with EOS in the last complete frame (or any recent)
            eos_detected_in_last = False
            earliest_eos_cb = self.num_codebooks  # Default: No EOS, force full EOS frame
            for frame_idx in range(len(last_frames) - 1, -1, -1):  # Scan backward
                frame = last_frames[frame_idx]
                eos_cbs = np.where(frame == self.stream_eos_id)[0]
                if len(eos_cbs) > 0:
                    earliest_eos_cb = min(eos_cbs)
                    eos_detected_in_last = True
                    break

            if eos_detected_in_last:
                # Propagate EOS to earlier codebooks in subsequent frames (if any)
                remaining_delays = self.num_codebooks - 1 - earliest_eos_cb
                for fix_idx in range(len(last_frames)):
                    target_frame = last_frames[fix_idx]
                    # Force EOS in codebooks < remaining_delays (staggered)
                    force_eos_up_to = max(0, remaining_delays - fix_idx)
                    for cb in range(force_eos_up_to):
                        if target_frame[cb] != self.stream_eos_id:
                            target_frame[cb] = self.stream_eos_id
                            print(
                                f"Fixed EOS in frame {last_frames_start + fix_idx}, codebook {cb}"
                            )
            else:
                # No EOS detected: Append a full EOS frame to terminate
                print("No EOS detected; appending full EOS frame for safe termination")
                frames = np.vstack([frames, eos_row])

            # Trim trailing full EOS rows (post-fixup)
            while frames.size > 0 and np.all(frames[-1] == eos_row):
                frames = frames[:-1]

        # Update stats after fixup
        content_tokens_after = np.sum((frames >= 0) & (frames < self.audio_codebook_size))
        eos_tokens_after = np.sum(frames == self.stream_eos_id)
        print(f"Post-fixup stats - Content: {content_tokens_after}, EOS: {eos_tokens_after}")

        if frames.size == 0:
            raise RuntimeError("No valid frames after EOS fixup.")

        # Group frames by EOS separators (as before)
        groups: list[np.ndarray] = []
        eos_rows = np.where(np.all(frames == eos_row, axis=1))[0]
        start = 0
        for idx in eos_rows:
            if idx > start:
                groups.append(frames[start:idx])
            start = idx + 1
        if start < len(frames):
            groups.append(frames[start:])

        print(f"Final groups lengths: {[g.shape[0] for g in groups if g.size > 0]}")

        # Decode each group
        waveforms: list[np.ndarray] = []
        sr = self.audio_tokenizer.sampling_rate
        for group_idx, audio_data in enumerate(groups):
            if audio_data.size == 0:
                continue

            print(f"Processing group {group_idx} of length {audio_data.shape[0]}")

            # Revert delay pattern
            if audio_data.shape[0] < self.num_codebooks:
                # Small group: Transpose directly (assume aligned)
                codes = audio_data.T  # Shape: (num_codebooks, seq_len)
            else:
                codes = revert_delay_pattern(audio_data.T)  # Shape: (num_codebooks, seq_len)

            # Clamp to valid range (treat BOS/EOS as 0 for decoding)
            valid_limit = self.audio_codebook_size - 2  # Exclude BOS/EOS
            codes = np.where(codes >= valid_limit, 0, codes)

            print(f"Decoded codes shape for group {group_idx}: {codes.shape}")
            try:
                waveform, sr = self.audio_tokenizer.decode(codes)
                waveforms.append(waveform)
            except Exception as decode_err:
                print(f"Decode failed for group {group_idx} (shape {codes.shape}): {decode_err}")
                continue

        if not waveforms:
            raise RuntimeError("Unable to reconstruct any audio waveform from generated tokens.")

        # Concatenate and resample if needed
        waveform = np.concatenate(waveforms).astype(np.float32)
        if sr != 16000:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)

        print(f"Final waveform length: {len(waveform)} samples at 16000 Hz")
        return waveform
