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

from tensorrt_llm.bindings import INT32, ModelConfig
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
from tensorrt_llm.runtime import GenerationSession, ModelRunnerCpp, SamplingConfig
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
            "codebook_embeddings": "audio_codebook_embeddings",
        }
        loader = ModelWeightsLoader(hf_config_or_dir, custom_dict)
        trtllm_model = cls(config)
        loader.update_key_mapping(trtllm_model)
        loader.generate_tllm_weights(trtllm_model)

        return trtllm_model


class HiggsAudioRunner(ModelRunnerCpp):
    def __init__(self, kwargs) -> None:
        super().__init__(**kwargs)

    def _fill_output(self, kwargs):
        outputs = super()._fill_output(**kwargs)
        # Additional processing for Higgs Audio outputs
        return outputs


class HiggsAudioLogitsProcessor(LogitsProcessor):
    """
    Fixed logits processor for Higgs Audio's delay pattern.

    Key Fixes:
    - Strict delay: No early BOS for higher CBs (mask everything until exact step).
    - Nuanced EOS: Propagate only after content; use remaining_delays for staggered completion.
    - Early EOS penalty: Discourage sampling EOS too soon per CB.
    - Extended generation: Allow all CBs to contribute longer before full end.
    """

    def __init__(
        self,
        config,
        debug_mode: bool = True,
        eos_penalty_steps: int = 3,  # Penalize EOS for first N steps per CB
        eos_penalty: float = -2.0,  # Logit subtraction for early EOS
    ):
        self.config = config
        self.num_codebooks = config.audio_num_codebooks
        self.codebook_size = config.audio_codebook_size
        self.stream_bos_id = config.audio_stream_bos_id
        self.stream_eos_id = config.audio_stream_eos_id
        self.audio_vocab_size = config.audio_vocab_size

        self.debug_mode = debug_mode
        self.eos_penalty_steps = eos_penalty_steps
        self.eos_penalty = eos_penalty
        self.request_states: Dict[int, Dict] = {}

    def __call__(
        self,
        req_id: int,
        logits: torch.Tensor,
        token_ids: List[List[int]],
        stream_ptr: Optional[int] = None,
        client_id: Optional[int] = None,
    ) -> None:
        if logits.shape[-1] != self.audio_vocab_size:
            if self.debug_mode:
                print(
                    f"[DEBUG] Skipping: Vocab size mismatch ({logits.shape[-1]} vs {self.audio_vocab_size})"
                )
            return

        if req_id not in self.request_states:
            self.request_states[req_id] = {
                "audio_start_pos": getattr(self.config, "input_length", 0),
                "eos_codebooks": set(),
                "has_content": [False] * self.num_codebooks,  # Track if CB generated content
                "remaining_delays": None,
                "sequence_ended": False,
                "current_step": -1,
                "cb_step_starts": [None] * self.num_codebooks,  # When each CB started content
            }

        state = self.request_states[req_id]
        current_pos = len(token_ids[0]) if token_ids else 0
        audio_step = current_pos - state["audio_start_pos"]

        if self.debug_mode:
            print(
                f"[DEBUG] Req {req_id}: Pos {current_pos}, Audio step {audio_step}, EOS CBs {state['eos_codebooks']}, Delays {state['remaining_delays']}"
            )

        if audio_step < 0:
            if self.debug_mode:
                print(f"[DEBUG] Skipping: Text phase (step {audio_step})")
            return

        state["current_step"] = audio_step

        # Update from last token
        if current_pos > state["audio_start_pos"] and token_ids:
            self._update_from_last_token(state, token_ids[0][-1], audio_step)

        if state["sequence_ended"]:
            logits.fill_(-float("inf"))
            if self.debug_mode:
                print(f"[DEBUG] Sequence ended at step {audio_step}")
            return

        self._mask_delay_pattern(logits, audio_step, state)

    def _update_from_last_token(self, state: Dict, last_token: int, step: int) -> None:
        """Update EOS/content state from the last sampled token."""
        if 0 <= last_token < self.audio_vocab_size:
            cb_idx = last_token // self.codebook_size
            local_token = last_token % self.codebook_size

            # Mark content if not BOS/EOS
            if 0 <= local_token < self.codebook_size - 2:
                if not state["has_content"][cb_idx]:
                    state["has_content"][cb_idx] = True
                    if state["cb_step_starts"][cb_idx] is None:
                        state["cb_step_starts"][cb_idx] = step
                    if self.debug_mode:
                        print(f"[DEBUG] CB {cb_idx} generated content at step {step}")

            # EOS logic (only if has_content or after start)
            if local_token == self.stream_eos_id and (
                state["has_content"][cb_idx] or state["cb_step_starts"][cb_idx] is not None
            ):
                if cb_idx not in state["eos_codebooks"]:
                    state["eos_codebooks"].add(cb_idx)
                    # Set remaining_delays like HF
                    if state["remaining_delays"] is None:
                        state["remaining_delays"] = self.num_codebooks - cb_idx - 1
                    else:
                        state["remaining_delays"] = max(0, state["remaining_delays"] - 1)
                    # Propagate to higher CBs
                    for j in range(cb_idx + 1, self.num_codebooks):
                        if j not in state["eos_codebooks"]:
                            state["eos_codebooks"].add(j)
                    if self.debug_mode:
                        print(
                            f"[DEBUG] EOS in CB {cb_idx} at step {step}; propagated; delays={state['remaining_delays']}"
                        )

        # Check sequence end
        if (
            len(state["eos_codebooks"]) == self.num_codebooks
            and state["remaining_delays"] is not None
            and state["remaining_delays"] <= 0
        ):
            state["sequence_ended"] = True

    def _mask_delay_pattern(self, logits: torch.Tensor, step: int, state: Dict) -> None:
        """Apply fixed delay pattern masking."""
        inf_value = -float("inf")
        logits.fill_(inf_value)  # Default mask

        remaining_delays = state.get("remaining_delays")
        sequence_ended = state["sequence_ended"]

        for cb_idx in range(self.num_codebooks):
            cb_start = cb_idx * self.codebook_size
            cb_end = min(cb_start + self.codebook_size, self.audio_vocab_size)
            content_end = min(cb_start + self.codebook_size - 2, cb_end - 1)  # Exclude BOS/EOS

            bos_idx = cb_start + self.stream_bos_id
            eos_idx = cb_start + self.stream_eos_id

            if cb_end <= cb_start or sequence_ended:
                continue

            force_eos = cb_idx in state["eos_codebooks"]
            cb_has_started = state["cb_step_starts"][cb_idx] is not None
            early_step = (state["cb_step_starts"][cb_idx] is not None) and (
                step - state["cb_step_starts"][cb_idx] < self.eos_penalty_steps
            )

            if step < cb_idx:
                # Strict delay: Full mask (no early BOS)
                if self.debug_mode:
                    print(f"[DEBUG] Step {step}, CB {cb_idx}: Full mask (delayed)")
                continue  # Everything masked (default)
            elif step == cb_idx:
                # Initiation: BOS only (or EOS if forced)
                if force_eos and 0 <= eos_idx < self.audio_vocab_size:
                    logits[:, :, eos_idx] = 0.0
                    if self.debug_mode:
                        print(f"[DEBUG] Step {step}, CB {cb_idx}: Forced early EOS at {eos_idx}")
                elif 0 <= bos_idx < self.audio_vocab_size:
                    logits[:, :, bos_idx] = 0.0
                    if self.debug_mode:
                        print(f"[DEBUG] Step {step}, CB {cb_idx}: BOS at {bos_idx}")
            else:  # step > cb_idx: Content + EOS
                # Content range
                content_start = cb_start
                if content_start < self.audio_vocab_size:
                    logits[:, :, content_start : content_end + 1] = 0.0  # Unmask content
                    if self.debug_mode:
                        print(
                            f"[DEBUG] Step {step}, CB {cb_idx}: Content {content_start}:{content_end + 1}"
                        )

                # EOS: Unmask, but penalize if early
                if 0 <= eos_idx < self.audio_vocab_size:
                    eos_logit = 0.0 if not early_step else self.eos_penalty
                    logits[:, :, eos_idx] = eos_logit
                    if self.debug_mode and early_step:
                        print(
                            f"[DEBUG] Step {step}, CB {cb_idx}: Penalized early EOS at {eos_idx} ({eos_logit})"
                        )
                    elif self.debug_mode:
                        print(f"[DEBUG] Step {step}, CB {cb_idx}: EOS at {eos_idx}")

            # remaining_delays: For higher CBs post-propagation, limit content
            if remaining_delays is not None and cb_idx > (
                self.num_codebooks - remaining_delays - 1
            ):
                # Allow limited content, then force EOS
                if step > cb_idx + remaining_delays:  # After allowed steps
                    logits[:, :, cb_start:cb_end] = inf_value  # Remask all except EOS
                    if 0 <= eos_idx < self.audio_vocab_size:
                        logits[:, :, eos_idx] = 0.0
                    if self.debug_mode:
                        print(f"[DEBUG] Step {step}, CB {cb_idx}: Delays limit - EOS only")

        # Safety: If all masked (edge case), unmask CB0 minimally
        if torch.all(logits == inf_value) and not sequence_ended:
            if self.debug_mode:
                print(f"[WARNING] All masked at step {step} - fallback unmask CB0")
            cb0_start, cb0_end = 0, min(self.codebook_size - 2, self.audio_vocab_size)
            logits[:, :, cb0_start : cb0_end + 1] = 0.0
            if 0 <= self.stream_eos_id < self.audio_vocab_size:
                logits[:, :, self.stream_eos_id] = 0.0


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
        self.audio_eos_id = self.config.audio_eos_token_id
        self.codebook_size = self.config.audio_codebook_size

        # Set up device
        self.device = torch.device("cuda", 0)
        torch.cuda.set_device(self.device)

        # Load components
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_dir, trust_remote_code=True)
        self.audio_tokenizer = load_higgs_audio_tokenizer(self.audio_tokenizer_dir)

        # Create custom logits processor for delay pattern handling
        self.audio_logits_processor = HiggsAudioLogitsProcessor(self.config)

        self.runner = ModelRunnerCpp.from_dir(
            engine_dir=self.engine_dir,
            kv_cache_free_gpu_memory_fraction=0.5,
            # use_gpu_direct_storage=True,
            # cuda_graph_mode=True,
            logits_processor_map={"higgs_audio_logit_processor": self.audio_logits_processor},
        )
        self.runner.session = HiggsForCausalLMGenerationSession(
            model_config=self.runner.model_config,
            engine_buffer=self.runner.engine_buffer,
            mapping=self.runner.mapping,
            debug_mode=False,
            debug_tensors_to_save=None,
            cuda_graph_mode=False,
            stream=torch.cuda.current_stream(),
            global_max_input_length=2048,
            global_max_output_length=4096,
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
            # self.config.audio_in_start = pre_audio_input_ids.shape[0]
            # self.config.audio_in_end = pre_audio_input_ids.shape[0] + audio_ids.shape[0] - 1
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
        self.config.input_length = input_ids.shape[0]
        self.audio_logits_processor.config = self.config
        self.config.audio_in_start = -1
        self.config.audio_in_end = -1
        max_new_tokens = self.max_num_tokens - self.config.input_length

        with torch.no_grad():
            self.config.input_length = input_ids.shape[0]
            self.audio_logits_processor.config = self.config

            outputs = self.runner.generate(
                batch_input_ids=[input_ids],
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                # logits_processor_names=["higgs_audio_logit_processor"],
            )

        audio_tokens = outputs[0, 0, self.config.input_length :]
        np.savetxt(
            "2.txt",
            audio_tokens.cpu(),
            delimiter=",",
            fmt="%d",
        )
        waveform = self.process_audio(audio_tokens)
        print(f"Successfully generated audio: {len(waveform)} samples")
        return waveform

    def process_audio(self, generated):
        """Process generated audio tokens with EOS fixup and decode to waveform."""
        if generated.numel() == 0:
            raise RuntimeError("No tokens generated.")

        # Trim to complete frames
        remainder = generated.numel() % self.num_codebooks
        if remainder != 0:
            generated = generated[:-remainder]
        if generated.numel() == 0:
            raise RuntimeError("No complete frames.")

        frames = generated.view(-1, self.num_codebooks)  # (num_frames, num_codebooks)
        print(f"Initial frames: {frames.shape}")

        np.savetxt(
            "3.txt",
            frames.cpu(),
            delimiter=",",
            fmt="%d",
        )

        # Log initial token stats
        content_mask = (frames >= 0) & (frames < self.codebook_size)
        eos_mask = frames == self.stream_eos_id
        content_tokens = content_mask.sum().item()
        eos_tokens = eos_mask.sum().item()
        print(
            f"Initial: Content={content_tokens}, EOS={eos_tokens} (ratio EOS/total={eos_tokens / (frames.numel()):.2%})"
        )

        bos_row = torch.full((self.num_codebooks,), self.stream_bos_id, device=generated.device)
        eos_row = torch.full((self.num_codebooks,), self.stream_eos_id, device=generated.device)

        # Trim leading BOS
        while frames.size(0) > 0 and torch.all(frames[0] == bos_row):
            frames = frames[1:]

        # EOS fixup (increased to 3 frames)
        max_frames_to_fix = 3
        if frames.size(0) > 0:
            last_start = max(0, frames.size(0) - max_frames_to_fix)
            last_frames = frames[last_start:]

            eos_detected = False
            earliest_eos_cb = self.num_codebooks
            for f_idx in range(last_frames.size(0) - 1, -1, -1):
                frame = last_frames[f_idx]
                eos_cbs = (frame == self.stream_eos_id).nonzero(as_tuple=True)[0]
                if eos_cbs.numel() > 0:
                    earliest_eos_cb = eos_cbs.min().item()
                    eos_detected = True
                    print(f"EOS detected in frame {last_start + f_idx} at CB {earliest_eos_cb}")
                    break

            if eos_detected:
                remaining = self.num_codebooks - 1 - earliest_eos_cb
                for fix_idx in range(last_frames.size(0)):
                    target = last_frames[fix_idx]
                    force_up_to = max(0, remaining - fix_idx)
                    for cb in range(force_up_to):
                        if target[cb] != self.stream_eos_id:
                            target[cb] = self.stream_eos_id
                            print(f"Fixed EOS: frame {last_start + fix_idx}, CB {cb}")
            else:
                # No EOS: Append 1 full EOS frame
                print("No EOS; appending full EOS frame")
                frames = torch.cat([frames, eos_row.unsqueeze(0)])

            # Trim trailing full EOS
            while frames.size(0) > 0 and torch.all(frames[-1] == eos_row):
                frames = frames[:-1]

        # Post-fixup stats
        content_after = ((frames >= 0) & (frames < self.codebook_size)).sum().item()
        eos_after = (frames == self.stream_eos_id).sum().item()
        print(f"Post-fixup: Content={content_after}, EOS={eos_after}")

        # Validation: Ensure reasonable EOS (e.g., >5% in last 10 frames)
        if frames.size(0) >= 10:
            last_10_eos = (frames[-10:] == self.stream_eos_id).sum().item()
            if last_10_eos / (10 * self.num_codebooks) < 0.05:
                print("Warning: Low EOS density; appending extra EOS frame")
                frames = torch.cat([frames, eos_row.unsqueeze(0)])

        if frames.size(0) == 0:
            raise RuntimeError("No frames after fixup.")

        # Group by full EOS rows
        groups = []
        eos_row_indices = torch.where(torch.all(frames == eos_row, dim=1))[0]
        start = 0
        for idx in eos_row_indices:
            if idx > start:
                groups.append(frames[start:idx])
            start = idx + 1
        if start < frames.size(0):
            groups.append(frames[start:])

        print(f"Groups: {[g.size(0) for g in groups if g.size(0) > 0]}")

        # Decode groups
        waveforms = []
        for g_idx, group in enumerate(groups):
            if group.size(0) == 0:
                continue
            print(f"Group {g_idx}: {group.size(0)} frames")

            # Revert delay
            if group.size(0) < self.num_codebooks:
                codes = group.T  # (num_cb, seq_len)
            else:
                codes = revert_delay_pattern(group.T)  # torch version

            # Clamp BOS/EOS to 0 for decode
            codes = torch.where(codes >= self.codebook_size - 2, torch.zeros_like(codes), codes)

            # Validate first frame has content
            if g_idx == 0 and codes.size(1) > 0:
                first_frame_content = (codes[:, 0] < self.codebook_size).float().mean().item()
                if first_frame_content < 0.5:
                    print(
                        f"Warning: Low content in first frame ({first_frame_content:.2%}); skipping group"
                    )
                    continue

            try:
                # Decode expects np (num_cb, seq_len)
                waveform, sr = self.audio_tokenizer.decode(codes.cpu().numpy())
                waveforms.append(waveform)
            except Exception as e:
                print(f"Decode error group {g_idx}: {e}")
                continue

        if not waveforms:
            raise RuntimeError("No valid waveforms decoded.")

        # Concat and resample
        waveform = np.concatenate(waveforms).astype(np.float32)
        target_sr = 16000
        if sr != target_sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)

        return waveform
