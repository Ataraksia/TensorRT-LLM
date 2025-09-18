# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
"""TensorRT-LLM implementation of Higgs Audio multimodal model."""

import builtins
import librosa
from collections import defaultdict, deque, Counter
from collections.abc import AsyncGenerator, Sequence
import os
from pathlib import Path
from typing import Any, Optional, List, OrderedDict, Union
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
    """Custom logits processor for HiggsAudio that applies delay pattern logic during generation."""

    def __init__(
        self,
        config: HiggsAudioConfig,
        ras_window: int = 64,
        ras_max_repeats: int = 6,
        debug: bool = False,
    ):
        self.config = config

        self.vocab_size = config.audio_vocab_size
        self.num_codebooks = config.audio_num_codebooks
        self.codebook_size = config.audio_codebook_size
        self.stream_bos_id = config.audio_stream_bos_id
        self.stream_eos_id = config.audio_stream_eos_id
        self.request_states = {}  # Track delay pattern state per request

        # Repetition-Aware Sampling (RAS) configuration
        self.ras_window = int(ras_window)
        self.ras_max_repeats = int(ras_max_repeats)
        self.debug = bool(debug)

        # Debug step counter for tracking generation progress
        self.debug_step_counter = 0

    @staticmethod
    def _sync_stream(stream: Optional[torch.cuda.Stream]) -> None:
        if stream is not None:
            stream.synchronize()
        else:
            torch.cuda.synchronize()

    def _get_or_create_state(self, req_id: int, token_ids: List[List[int]]) -> dict:
        """Get or create generation state for a request."""
        if req_id not in self.request_states:
            prompt_len = min(self.config.input_length, len(token_ids[0]))

            self.request_states[req_id] = {
                "prompt_len": prompt_len,
                "processed_tokens": prompt_len,
                # legacy fields (kept harmlessly)
                "num_delay": 0,
                "num_remaining_delays": None,
                "frames": [],
                "current_frame": [self.stream_bos_id] * self.num_codebooks,
                "frame_counter": 0,
                "countdown_active": False,
                "min_content_frames": self.num_codebooks * 10,  # heuristic gate for EOS
                "last_frame": None,
                "repeat_count": 0,
                "max_repeat_frames": 2,
                "last_column": [self.stream_bos_id] * self.num_codebooks,
                "pending_column": [],
                "history": [[] for _ in range(self.num_codebooks)],
                "max_frames": 160,
                "global_counts": [defaultdict(int) for _ in range(self.num_codebooks)],
                "global_repeat_threshold": 12,
                # aligned delay schedule + EOS propagation state
                "codebook_index": 0,  # 0..7 current active codebook slice
                "frame_step": 0,  # increments only after wrap 7->0
                "started_flags": [False] * self.num_codebooks,  # BOS emitted per codebook
                "eosed_flags": [False] * self.num_codebooks,  # EOS emitted per codebook
                "last_eos_index": -1,  # highest cb index EOSed in current frame progression
                "all_audio_finished": False,  # all CBs EOSed and a wrap completed
                "debug": self.debug,
            }

        # Ensure RAS bookkeeping is present (idempotent)
        state = self.request_states[req_id]
        if "ras_window" not in state:
            state["ras_window"] = self.ras_window
        if "ras_max_repeats" not in state:
            state["ras_max_repeats"] = self.ras_max_repeats
        if "ras_histories" not in state:
            state["ras_histories"] = [
                deque(maxlen=state["ras_window"]) for _ in range(self.num_codebooks)
            ]
        if "ras_counts" not in state:
            state["ras_counts"] = [defaultdict(int) for _ in range(self.num_codebooks)]

        return state

    def _decode_audio_token(self, token: int) -> int:
        return int(token % self.codebook_size)

    def _mask_except(self, vector: torch.Tensor, keep_idx: int) -> None:
        original = vector[keep_idx]
        vector.fill_(torch.finfo(vector.dtype).min)
        floor_value = torch.tensor(0.0, device=vector.device, dtype=vector.dtype)
        if torch.isfinite(original):
            vector[keep_idx] = torch.maximum(original, floor_value)
        else:
            vector[keep_idx] = floor_value

    def _suppress_token(self, vector: torch.Tensor, idx: int) -> None:
        vector[idx] = torch.finfo(vector.dtype).min

    def _ingest_token(self, state: dict, token: int) -> None:
        value = self._decode_audio_token(token)
        state["pending_column"].append(value)
        cb = state["codebook_index"]
        state["current_frame"][cb] = value
        state["history"][cb].append(value)
        history_window = 256
        if len(state["history"][cb]) > history_window:
            state["history"][cb] = state["history"][cb][-history_window:]
        state["global_counts"][cb][value] += 1

        if cb == self.num_codebooks - 1:
            completed_frame = state["current_frame"]
            state["frames"].append(completed_frame.copy())
            state["current_frame"] = [self.stream_bos_id] * self.num_codebooks
            state["codebook_index"] = 0
            state["frame_counter"] += 1
            state["last_column"] = state["pending_column"].copy()
            state["pending_column"] = []
            if state["frame_counter"] >= state["max_frames"]:
                state["all_audio_finished"] = True

            if state["last_frame"] is not None and state["last_frame"] == completed_frame:
                state["repeat_count"] += 1
            else:
                state["repeat_count"] = 0
            state["last_frame"] = completed_frame.copy()
            if state["repeat_count"] >= state["max_repeat_frames"]:
                state["all_audio_finished"] = True

            prev_last_eos = state["last_eos_index"]
            eos_indices = [
                idx for idx, val in enumerate(completed_frame) if val == self.stream_eos_id
            ]
            new_eos_detected = False
            if eos_indices:
                last_eos_idx = eos_indices[-1]
                if last_eos_idx > prev_last_eos:
                    remaining = self.num_codebooks - last_eos_idx - 1
                    state["last_eos_index"] = last_eos_idx
                    state["num_remaining_delays"] = max(remaining, 0)
                    state["countdown_active"] = remaining > 0
                    new_eos_detected = True
                    if remaining == 0:
                        state["countdown_active"] = False
                        state["all_audio_finished"] = True
            if state["countdown_active"] and state["num_remaining_delays"] is not None:
                if not new_eos_detected:
                    state["num_remaining_delays"] = max(state["num_remaining_delays"] - 1, 0)
                if state["num_remaining_delays"] == 0:
                    state["countdown_active"] = False
                    state["all_audio_finished"] = True

            if state["all_audio_finished"]:
                state["num_delay"] = 0

            if (
                state["last_eos_index"] == self.num_codebooks - 1
                and state["num_remaining_delays"] == 0
            ):
                state["all_audio_finished"] = True

            if state["all_audio_finished"]:
                state["num_delay"] = 0
        else:
            state["codebook_index"] += 1

    def __call__(
        self,
        req_id: int,
        logits: torch.Tensor,
        token_ids: List[List[int]],
        stream_ptr: Optional[int],
        client_id: Optional[int],
    ) -> None:
        """Aligned delay-gating + EOS guard: unmask exactly one codebook slice per step and mirror EOS propagation."""
        stream = None if stream_ptr is None else torch.cuda.ExternalStream(stream_ptr)

        with torch.cuda.stream(stream):
            # Debug: increment step counter
            if self.debug:
                self.debug_step_counter += 1

            # Reset state if we detect we're still inside prompt
            if req_id in self.request_states:
                prev_prompt_len = self.request_states[req_id]["prompt_len"]
                if len(token_ids[0]) <= prev_prompt_len:
                    self.request_states.pop(req_id, None)

            state = self._get_or_create_state(req_id, token_ids)

            # Debug: log current state at start of processing
            if self.debug:
                print(f"\n{'=' * 80}")
                print(f"[DEBUG STEP {self.debug_step_counter}] Request ID: {req_id}")
                print(f"{'=' * 80}")
                print(
                    f"[STATE] codebook_index={state['codebook_index']}, frame_step={state['frame_step']}"
                )
                print(f"[STATE] started_flags={state['started_flags']}")
                print(f"[STATE] eosed_flags={state['eosed_flags']}")
                print(f"[STATE] last_eos_index={state['last_eos_index']}")
                print(f"[STATE] num_remaining_delays={state.get('num_remaining_delays', 'None')}")
                print(f"[STATE] all_audio_finished={state['all_audio_finished']}")
                print(
                    f"[STATE] processed_tokens={state['processed_tokens']}, total_len={len(token_ids[0])}"
                )

            # Slice audio logits: [1,1,V] -> [8,1026]
            logits_tensor = logits
            flat = logits_tensor[0, 0, 0 : self.vocab_size]
            logits_2d = flat.view(self.num_codebooks, self.codebook_size)
            min_val = torch.finfo(logits_2d.dtype).min

            # Debug: log active codebook slice info
            if self.debug:
                active_cb = state["codebook_index"]
                allowed_range = (
                    active_cb * self.codebook_size,
                    (active_cb + 1) * self.codebook_size,
                )
                print(
                    f"[CODEBOOK] Active codebook slice: {active_cb} (token range {allowed_range[0]}-{allowed_range[1] - 1})"
                )

            # Ingest any new sampled tokens since last call to advance state
            total_len = len(token_ids[0])
            new_tokens_processed = []
            if total_len > state["processed_tokens"]:
                if self.debug:
                    print(
                        f"[TOKENS] Processing new tokens from index {state['processed_tokens']} to {total_len - 1}"
                    )

                for idx in range(state["processed_tokens"], total_len):
                    if idx < state["prompt_len"]:
                        continue
                    sampled = int(token_ids[0][idx])
                    active_cb = state["codebook_index"]
                    sampled_cb = int(sampled // self.codebook_size)
                    sampled_local = int(sampled % self.codebook_size)

                    # Debug: log token interpretation
                    if self.debug:
                        token_type = (
                            "BOS"
                            if sampled_local == 1024
                            else "EOS"
                            if sampled_local == 1025
                            else f"Local[{sampled_local}]"
                        )
                        print(
                            f"[TOKEN] {sampled} → CB{sampled_cb}:{token_type} (active_cb={active_cb})"
                        )

                    new_tokens_processed.append((sampled, sampled_cb, sampled_local))

                    if state.get("debug", False):
                        assert sampled_cb == active_cb, (
                            f"Sampled cb={sampled_cb} while active={active_cb}"
                        )

                    # Update per-codebook start/EOS flags
                    state_changed = False
                    if sampled_local == self.stream_bos_id:
                        if not state["started_flags"][active_cb]:
                            state["started_flags"][active_cb] = True
                            state_changed = True
                            if self.debug:
                                print(f"[TRANSITION] CB{active_cb}: inactive → BOS")
                    elif sampled_local == self.stream_eos_id:
                        if not state["eosed_flags"][active_cb]:
                            state["eosed_flags"][active_cb] = True
                            state_changed = True
                            if self.debug:
                                print(f"[TRANSITION] CB{active_cb}: content → EOS")
                        if active_cb > state["last_eos_index"]:
                            old_last_eos = state["last_eos_index"]
                            state["last_eos_index"] = active_cb
                            if self.debug:
                                print(
                                    f"[EOS_PROP] last_eos_index updated: {old_last_eos} → {active_cb}"
                                )
                        # Initialize remaining delays countdown for EOS propagation across frames
                        state["num_remaining_delays"] = self.num_codebooks - active_cb - 1
                        # Initialize remaining delays to mirror Transformers countdown semantics
                        state["num_remaining_delays"] = self.num_codebooks - active_cb - 1
                    else:
                        if not state["started_flags"][active_cb]:
                            state["started_flags"][active_cb] = True
                            state_changed = True
                            if self.debug:
                                print(f"[TRANSITION] CB{active_cb}: inactive → content")

                    # RAS bookkeeping: update rolling window per active codebook for local ids [0..1023]
                    if 0 <= sampled_local < 1024:
                        hist = state["ras_histories"][active_cb]
                        cnts = state["ras_counts"][active_cb]
                        old_count = cnts.get(sampled_local, 0)
                        hist.append(sampled_local)
                        cnts[sampled_local] += 1
                        # maintain capacity (deque enforces maxlen, but decrement count if popped)
                        if len(hist) > state["ras_window"]:
                            popped = hist[0]
                            cnts[popped] -= 1
                            if cnts[popped] <= 0:
                                del cnts[popped]

                        if self.debug and cnts[sampled_local] != old_count:
                            print(
                                f"[RAS] CB{active_cb} token {sampled_local}: count {old_count} → {cnts[sampled_local]}"
                            )

                    # Optional delay countdown aligned to Transformers semantics:
                    # Decrease remaining delays on steps that do not introduce a new EOS.
                    if (
                        state.get("num_remaining_delays") is not None
                        and state["num_remaining_delays"] > 0
                        and sampled_local != self.stream_eos_id
                    ):
                        old_delays = state["num_remaining_delays"]
                        state["num_remaining_delays"] -= 1
                        if self.debug:
                            print(
                                f"[DELAY_COUNTDOWN] num_remaining_delays: {old_delays} → {state['num_remaining_delays']}"
                            )

                    # Strict codebook cycling 0->7
                    old_cb_index = state["codebook_index"]
                    state["codebook_index"] = (state["codebook_index"] + 1) % self.num_codebooks

                    # Debug: log codebook wrap and frame advancement
                    if self.debug:
                        if state["codebook_index"] == 0:
                            print(
                                f"[WRAP] Codebook wrap: CB{old_cb_index} → CB0, frame_step: {state['frame_step']} → {state['frame_step'] + 1}"
                            )
                        else:
                            print(
                                f"[ADVANCE] Codebook advance: CB{old_cb_index} → CB{state['codebook_index']}"
                            )

                    if state["codebook_index"] == 0:
                        # Completed one frame/column
                        state["frame_step"] += 1
                        # Reset clamping for new frame progression
                        old_last_eos = state["last_eos_index"]
                        state["last_eos_index"] = -1
                        if self.debug and old_last_eos != -1:
                            print(f"[FRAME_COMPLETE] Reset last_eos_index: {old_last_eos} → -1")
                        # Propagate remaining delays across frames (Transformers semantics)
                        if (
                            state.get("num_remaining_delays") is not None
                            and state["num_remaining_delays"] > 0
                        ):
                            old_delays = state["num_remaining_delays"]
                            state["num_remaining_delays"] -= 1
                            if self.debug:
                                print(
                                    f"[FRAME_DELAY_COUNTDOWN] Cross-frame delay countdown: {old_delays} → {state['num_remaining_delays']}"
                                )
                        # If all codebooks are EOSed, we can stop after wrap
                        if all(state["eosed_flags"]):
                            state["all_audio_finished"] = True
                            if self.debug:
                                print(
                                    f"[END_CONDITION] All codebooks EOSed → all_audio_finished=True"
                                )

                state["processed_tokens"] = total_len

            # Masking: only active codebook row is visible
            active_cb = state["codebook_index"]
            masked_codebooks = 0
            for j in range(self.num_codebooks):
                if j != active_cb:
                    logits_2d[j].fill_(min_val)
                    masked_codebooks += 1

            if self.debug:
                print(
                    f"[MASKING] Masked {masked_codebooks} codebook rows, active row: CB{active_cb}"
                )

            row = logits_2d[active_cb]
            original_row = row.clone()
            final_decision = None

            if state["all_audio_finished"]:
                # Only EOS after completion
                row.fill_(min_val)
                row[self.stream_eos_id] = 0.0
                final_decision = f"Force EOS (all_audio_finished=True)"
                if self.debug:
                    print(f"[DECISION] {final_decision}")
            else:
                # Optional invariant checks
                if state.get("debug", False) and state["last_eos_index"] >= 0:
                    for k in range(0, state["last_eos_index"] + 1):
                        assert state["eosed_flags"][k], (
                            f"Clamp invariant violated: last_eos_index={state['last_eos_index']} "
                            f"but eosed_flags[{k}] is False"
                        )

                started = state["started_flags"][active_cb]
                eosed = state["eosed_flags"][active_cb]
                last_eos_idx = state["last_eos_index"]

                # Clamp: any codebook <= last_eos_index or already EOSed -> EOS only
                if eosed or active_cb <= last_eos_idx:
                    row.fill_(min_val)
                    row[self.stream_eos_id] = 0.0
                    reason = (
                        "already EOSed"
                        if eosed
                        else f"CB{active_cb} <= last_eos_index({last_eos_idx})"
                    )
                    final_decision = f"Force EOS ({reason})"
                    if self.debug:
                        print(f"[DECISION] {final_decision}")
                else:
                    # BOS gating: before a codebook starts, force a single BOS on its first activation
                    if not started:
                        row.fill_(min_val)
                        row[self.stream_bos_id] = 0.0
                        final_decision = f"Force BOS (first activation of CB{active_cb})"
                        if self.debug:
                            print(f"[DECISION] {final_decision}")
                    else:
                        # Normal content phase: keep model distribution while enforcing BOS/EOS schedule
                        row.copy_(original_row)

                        constraints = []

                        # BOS only once per codebook
                        row[self.stream_bos_id] = min_val
                        constraints.append("BOS blocked (already started)")

                        # Replace overly restrictive EOS constraint with more permissive approach
                        min_content_steps = 10  # Allow EOS after some content generation
                        frame_step = state.get("frame_step", 0)

                        if frame_step < min_content_steps:
                            # Block EOS in very early generation to ensure minimum content
                            row[self.stream_eos_id] = min_val
                            constraints.append(
                                f"EOS blocked (frame_step={frame_step} < {min_content_steps})"
                            )
                        elif active_cb > 0 and not state["eosed_flags"][active_cb - 1]:
                            # Prefer sequential EOS but don't completely block it - use penalty instead
                            row[self.stream_eos_id] *= (
                                0.1  # Heavy penalty instead of complete block
                            )
                            constraints.append(f"EOS penalized (CB{active_cb - 1} not EOSed yet)")
                        # Otherwise allow EOS naturally

                        # Final codebook guard: relax restrictions to avoid deadlock
                        if active_cb == (self.num_codebooks - 1):
                            frame_step = state.get("frame_step", 0)
                            if frame_step < min_content_steps:
                                # Still block very early EOS for final codebook
                                row[self.stream_eos_id] = min_val
                                constraints.append(
                                    f"Final CB EOS blocked (frame_step={frame_step} < {min_content_steps})"
                                )
                            else:
                                # Allow final CB EOS after minimum content, with preference for sequential behavior
                                lower_all_eosed = all(
                                    state["eosed_flags"][: self.num_codebooks - 1]
                                )
                                remaining = state.get("num_remaining_delays", None)
                                penalty_applied = False

                                # Apply penalties for non-sequential behavior but don't completely block
                                if not lower_all_eosed:
                                    row[self.stream_eos_id] *= (
                                        0.05  # Strong penalty for non-sequential
                                    )
                                    penalty_applied = True
                                if remaining is not None and remaining > 0:
                                    row[self.stream_eos_id] *= 0.1  # Penalty for remaining delays
                                    penalty_applied = True

                                if penalty_applied:
                                    reasons = []
                                    if not lower_all_eosed:
                                        reasons.append("not all lower CBs EOSed")
                                    if remaining is not None and remaining > 0:
                                        reasons.append(f"remaining_delays={remaining}")
                                    constraints.append(
                                        f"Final CB EOS penalized ({'; '.join(reasons)})"
                                    )

                        # RAS: per-codebook repetition-aware masking on local tokens [0..1023]
                        cnts = state["ras_counts"][active_cb]
                        ras_masked_count = 0
                        if cnts:
                            # gather tokens exceeding repetition threshold
                            threshold = state.get("ras_max_repeats", self.ras_max_repeats)
                            masked_ids = [
                                lid for lid, c in cnts.items() if c >= threshold and 0 <= lid < 1024
                            ]
                            ras_masked_count = len(masked_ids)
                            if masked_ids:
                                neg_inf = torch.finfo(row.dtype).min
                                # optimistic mask
                                for lid in masked_ids:
                                    row[lid] = neg_inf
                                # if all locals masked (and EOS might be masked by schedule), relax to penalization
                                local_valid = torch.isfinite(row[:1024]).any().item()
                                eos_masked = not torch.isfinite(row[self.stream_eos_id]).item()
                                if not local_valid and eos_masked:
                                    # relax: heavy penalty instead of full mask
                                    strong_penalty = row.new_tensor(-10.0)
                                    for lid in masked_ids:
                                        # Only adjust if it was set to -inf above
                                        if not torch.isfinite(original_row[lid]):
                                            # keep as is; but typical case is finite, so just set penalty
                                            row[lid] = strong_penalty
                                        else:
                                            row[lid] = torch.maximum(
                                                original_row[lid] + strong_penalty, strong_penalty
                                            )
                                    constraints.append(
                                        f"RAS relaxed to penalty (all locals+EOS blocked)"
                                    )

                                if self.debug:
                                    # log number masked and top-3 most frequent
                                    sorted_items = sorted(
                                        cnts.items(), key=lambda kv: kv[1], reverse=True
                                    )
                                    top3 = sorted_items[:3]
                                    print(
                                        f"[RAS] CB{active_cb} masked={len(masked_ids)} tokens, top3 frequent={top3}"
                                    )

                        if ras_masked_count > 0:
                            constraints.append(f"RAS masked {ras_masked_count} tokens")

                        # Debug invariant: never include BOS/EOS in RAS counts
                        if state.get("debug", False):
                            assert 1024 not in cnts and 1025 not in cnts

                        # Safety: ensure at least one valid token
                        valid_tokens = torch.isfinite(row).sum().item()
                        if valid_tokens == 0:
                            row.fill_(min_val)
                            # Prefer local content fallback
                            best_local = int(torch.argmax(original_row[:1024]).item())
                            row[best_local] = 0.0
                            constraints.append(f"Safety fallback to token {best_local}")
                            valid_tokens = 1

                        final_decision = f"Content generation (valid tokens: {valid_tokens})"
                        if constraints:
                            final_decision += f" with constraints: {'; '.join(constraints)}"

                        if self.debug:
                            print(f"[DECISION] {final_decision}")

            # Debug: log final decision and next predicted token
            if self.debug:
                # Find the most likely token in the active codebook slice
                active_row = logits_2d[active_cb]
                if torch.isfinite(active_row).any():
                    best_idx = torch.argmax(active_row).item()
                    best_global_id = active_cb * self.codebook_size + best_idx
                    token_type = (
                        "BOS"
                        if best_idx == 1024
                        else "EOS"
                        if best_idx == 1025
                        else f"Local[{best_idx}]"
                    )
                    print(
                        f"[PREDICTION] Most likely next token: {best_global_id} → CB{active_cb}:{token_type}"
                    )
                else:
                    print(f"[PREDICTION] No valid tokens in active codebook slice!")

                print(f"{'=' * 80}\n")

            # Write back
            logits_tensor[0, 0, 0 : self.vocab_size].copy_(logits_2d.view(-1))

        HiggsAudioLogitsProcessor._sync_stream(stream)


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
        self.config.audio_in_start = 0
        self.config.audio_in_end = 0

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
        self.audio_logits_processor = HiggsAudioLogitsProcessor(self.config, debug=False)

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
            )
            codebook_offsets = (
                torch.arange(self.num_codebooks, device=audio_ids.device, dtype=audio_ids.dtype)
                * self.audio_codebook_size
            ).unsqueeze(-1)
            audio_ids = (audio_ids + codebook_offsets).flatten()

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
            self.config.audio_in_start = pre_audio_input_ids.size(0)
            self.config.audio_in_end = pre_audio_input_ids.size(0) + audio_ids.size(0) - 1
            input_ids = torch.cat([pre_audio_input_ids, audio_ids, post_audio_input_ids])
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
            f"<|audio_out_bos|><|AUDIO_OUT|>"
        )

        input_ids = self.tokenizer.encode(text_input, return_tensors="pt").to(self.device).flatten()
        input_ids = torch.cat([self.saved_input_ids, input_ids])
        # np.savetxt("log0.5.txt", input_ids.cpu(), delimiter=",", fmt="%d")
        self.config.input_length = input_ids.size(0)
        max_input_length = int(input_ids.size(0))
        max_new_tokens = self.max_num_tokens - max_input_length

        # Reset logits processor state for a fresh generation run.
        self.audio_logits_processor.request_states.clear()

        # Only terminate once the final codebook emits EOS. The runtime flattens
        # the (codebook, token) pair into a single id, so account for the stride.
        end_token_id = self.stream_eos_id + (self.num_codebooks - 1) * self.audio_codebook_size

        with torch.no_grad():
            # Continue using ModelRunnerCpp!
            outputs = self.runner.generate(
                batch_input_ids=[input_ids],
                max_new_tokens=max_new_tokens,
                end_id=int(end_token_id),
                temperature=float(self.temperature),
                top_k=int(self.top_k),
                top_p=float(self.top_p),
                logits_processor_names=["higgs_audio_logit_processor"],
            )

        # Extract and process audio tokens with proper delay pattern handling
        try:
            return self.process_audio(outputs[0, 0])

        except Exception as e:
            print(f"Error processing audio tokens: {e}")
            raise

    def process_audio(self, data):
        generated = data[self.config.input_length :]
        if generated.numel() == 0:
            raise RuntimeError("No audio tokens were generated by the model.")

        remainder = generated.numel() % self.num_codebooks
        if remainder:
            generated = generated[:-remainder]
        if generated.numel() == 0:
            raise RuntimeError("Generated tokens did not contain any complete audio frames.")

        frames = generated.view(-1, self.num_codebooks).to(torch.int64).cpu().numpy()
        frame_tokens = np.mod(frames, self.audio_codebook_size).astype(np.int64)

        bos_row = np.full((self.num_codebooks,), self.stream_bos_id, dtype=np.int64)
        separator = np.full((self.num_codebooks,), self.stream_eos_id, dtype=np.int64)

        groups: list[np.ndarray] = []
        eos_rows = np.where(np.all(frame_tokens == separator, axis=1))[0]
        start = 0
        for idx in eos_rows:
            if idx > start:
                groups.append(frame_tokens[start:idx])
            start = idx + 1
        if start < len(frame_tokens):
            groups.append(frame_tokens[start:])

        waveforms: list[np.ndarray] = []
        sr = 16000
        for audio_data in groups:
            if audio_data.size == 0:
                continue
            while audio_data.size > 0 and np.all(audio_data[0] == bos_row):
                audio_data = audio_data[1:]
            while audio_data.size > 0 and np.all(audio_data[-1] == separator):
                audio_data = audio_data[:-1]
            if audio_data.size == 0:
                continue

            if np.any(audio_data == self.stream_bos_id) or np.any(audio_data == self.stream_eos_id):
                codes = revert_delay_pattern(audio_data.transpose(1, 0))
            else:
                codes = audio_data.transpose(1, 0)
            codes = np.asarray(codes, dtype=np.int64)
            valid_codebook_limit = self.audio_codebook_size - 2
            # Treat special stream BOS/EOS markers as padding before decoding.
            codes = np.where(codes >= valid_codebook_limit, 0, codes)
            waveform, sr = self.audio_tokenizer.decode(codes)
            waveforms.append(waveform)

        if not waveforms:
            raise RuntimeError("Unable to reconstruct any audio waveform from generated tokens.")

        waveform = np.concatenate(waveforms).astype(np.float32)
        if sr != 16000:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        return waveform
