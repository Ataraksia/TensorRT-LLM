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

"""Configuration classes for HiggsAudio multimodal model.

HiggsAudio combines text generation with audio processing through a dual-FFN architecture.
The model includes:
- HiggsAudioConfig: Main configuration for the multimodal model
- HiggsAudioEncoderConfig: Configuration for the Whisper-based audio encoder
"""

import json
from typing import Dict, Optional, Union

from ..modeling_utils import PretrainedConfig, QuantConfig

__all__ = [
    "HiggsAudioConfig",
    "HiggsAudioEncoderConfig",
]


class HiggsAudioEncoderConfig(PretrainedConfig):
    """Configuration for the audio encoder used by HiggsAudio.

    This models the Whisper-style encoder shape parameters that the HiggsAudio
    pipeline expects when computing audio feature lengths and embeddings.

    Attributes:
    - d_model: Hidden size of the encoder outputs (projected to text hidden size later if needed)
    - n_mels: Number of mel bins in input features
    - downsample_factor: Total temporal downsampling factor from raw features to encoder outputs
    - use_whisper_tokenizer: Whether to follow Whisper feature conventions
    """

    def __init__(
        self,
        *,
        d_model: int = 3072,
        n_mels: int = 128,
        downsample_factor: int = 4,
        use_whisper_tokenizer: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            architecture="higgs_audio_encoder",
            dtype=kwargs.pop("dtype", "bfloat16"),
            hidden_size=d_model,
            num_hidden_layers=kwargs.pop("encoder_layers", 12),
            num_attention_heads=kwargs.pop("encoder_attention_heads", 12),
            **kwargs,
        )
        self.d_model = d_model
        self.n_mels = n_mels
        self.downsample_factor = downsample_factor
        self.use_whisper_tokenizer = use_whisper_tokenizer


class HiggsAudioConfig(PretrainedConfig):
    """Configuration class for HiggsAudio multimodal model.

    HiggsAudio is a multimodal model combining text generation with audio processing
    capabilities. It uses a dual-FFN architecture that can process both text and
    audio tokens through separate processing paths.
    """

    def __init__(
        self,
        *,
        # Core configurations
        text_config: Optional[Union[PretrainedConfig, Dict]] = None,
        audio_encoder_config: Optional[Union[HiggsAudioEncoderConfig, Dict]] = None,
        quant_config: Optional[Union[QuantConfig, Dict]] = None,
        audio_tokenizer_config: Optional[Dict] = None,
        dtype: str = "bfloat16",  # Add dtype parameter
        # Adapter configuration
        audio_adapter_type: str = "dual_ffn_fast_forward",
        audio_ffn_hidden_size: int = 4096,
        audio_ffn_intermediate_size: int = 14336,
        # Audio codebook configuration
        audio_num_codebooks: int = 8,
        audio_codebook_size: int = 1024,
        audio_stream_bos_id: int = 1024,
        audio_stream_eos_id: int = 1025,
        # Special tokens
        audio_bos_token: str = "<|audio_bos|>",
        audio_bos_token_id: int = 128011,
        audio_eos_token: str = "<|audio_eos|>",
        audio_eos_token_id: int = 128012,
        audio_out_bos_token: str = "<|audio_out_bos|>",
        audio_out_bos_token_id: int = 128013,
        audio_in_token: str = "<|AUDIO|>",
        audio_out_token: str = "<|AUDIO_OUT|>",
        audio_in_token_idx: int = 128015,
        audio_out_token_idx: int = 128016,
        pad_token_id: int = 128001,
        **kwargs,
    ):
        # Handle nested configurations
        if audio_encoder_config is None:
            audio_encoder_config = HiggsAudioEncoderConfig()
        elif isinstance(audio_encoder_config, dict):
            audio_encoder_config = HiggsAudioEncoderConfig(**audio_encoder_config)

        # For text_config, we'll use a default LLaMA-like config if none provided
        if text_config is None:
            # Default LLaMA-3.2-3B configuration
            text_config = {
                "architecture": "llama",
                "hidden_size": 3072,
                "num_hidden_layers": 28,
                "num_attention_heads": 24,
                "intermediate_size": 8192,
                "vocab_size": 128256,
                "max_position_embeddings": 131072,
                "num_key_value_heads": 8,
            }
        elif isinstance(text_config, dict) and "architecture" not in text_config:
            text_config["architecture"] = "llama"

        # Determine base configuration parameters from text config
        if isinstance(text_config, dict):
            hidden_size = text_config.get("hidden_size", 3072)
            num_layers = text_config.get("num_hidden_layers", 28)
            vocab_size = text_config.get("vocab_size", 128256)
        else:
            hidden_size = getattr(text_config, "hidden_size", 3072)
            num_layers = getattr(text_config, "num_hidden_layers", 28)
            vocab_size = getattr(text_config, "vocab_size", 128256)

        if quant_config is None:
            quant_config = QuantConfig()
        elif isinstance(quant_config, dict):
            quant_config = QuantConfig(**quant_config)

        # Initialize base PretrainedConfig
        super().__init__(
            architecture="higgs_audio",
            dtype=dtype,  # Use the passed dtype parameter
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=24,  # Default for LLaMA-3.2-3B
            vocab_size=vocab_size,
            hidden_act="silu",
            **kwargs,
        )

        # Store nested configurations
        self.text_config = text_config
        self.audio_encoder_config = audio_encoder_config
        self.audio_tokenizer_config = audio_tokenizer_config

        # Additional attributes required by TensorRT-LLM layers
        self.layer_idx_offset = (
            getattr(text_config, "layer_idx_offset", 0)
            if not isinstance(text_config, dict)
            else text_config.get("layer_idx_offset", 0)
        )
        self.max_position_embeddings = (
            getattr(text_config, "max_position_embeddings", 2048)
            if not isinstance(text_config, dict)
            else text_config.get("max_position_embeddings", 2048)
        )

        # LLaMA-specific attributes
        self.use_input_layernorm_in_first_layer = (
            getattr(text_config, "use_input_layernorm_in_first_layer", True)
            if not isinstance(text_config, dict)
            else text_config.get("use_input_layernorm_in_first_layer", True)
        )
        self.rms_norm_eps = (
            getattr(text_config, "rms_norm_eps", 1e-6)
            if not isinstance(text_config, dict)
            else text_config.get("rms_norm_eps", 1e-6)
        )
        self.intermediate_size = (
            getattr(text_config, "intermediate_size", 11008)
            if not isinstance(text_config, dict)
            else text_config.get("intermediate_size", 11008)
        )
        self.num_key_value_heads = (
            getattr(text_config, "num_key_value_heads", self.num_attention_heads)
            if not isinstance(text_config, dict)
            else text_config.get("num_key_value_heads", self.num_attention_heads)
        )
        self.rope_theta = (
            getattr(text_config, "rope_theta", 10000.0)
            if not isinstance(text_config, dict)
            else text_config.get("rope_theta", 10000.0)
        )
        self.rotary_base = (
            getattr(text_config, "rotary_base", 10000.0)
            if not isinstance(text_config, dict)
            else text_config.get("rotary_base", 10000.0)
        )
        self.mlp_bias = (
            getattr(text_config, "mlp_bias", False)
            if not isinstance(text_config, dict)
            else text_config.get("mlp_bias", False)
        )
        self.attention_bias = (
            getattr(text_config, "attention_bias", False)
            if not isinstance(text_config, dict)
            else text_config.get("attention_bias", False)
        )
        self.attn_bias = (
            getattr(text_config, "attn_bias", False)
            if not isinstance(text_config, dict)
            else text_config.get("attn_bias", False)
        )

        # Adapter configuration
        self.audio_adapter_type = audio_adapter_type
        self.audio_ffn_hidden_size = audio_ffn_hidden_size
        self.audio_ffn_intermediate_size = audio_ffn_intermediate_size

        # Audio codebook parameters
        self.audio_num_codebooks = audio_num_codebooks
        self.audio_codebook_size = audio_codebook_size
        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id

        # Special tokens
        self.audio_bos_token = audio_bos_token
        self.audio_bos_token_id = audio_bos_token_id
        self.audio_eos_token = audio_eos_token
        self.audio_eos_token_id = audio_eos_token_id
        self.audio_out_bos_token = audio_out_bos_token
        self.audio_out_bos_token_id = audio_out_bos_token_id
        self.audio_in_token = audio_in_token
        self.audio_out_token = audio_out_token
        self.audio_in_token_idx = audio_in_token_idx
        self.audio_out_token_idx = audio_out_token_idx
        self.pad_token_id = pad_token_id

    def to_dict(self) -> Dict:
        output = super().to_dict()

        # Handle nested configs
        if hasattr(self.audio_encoder_config, "to_dict"):
            output["audio_encoder_config"] = self.audio_encoder_config.to_dict()
        else:
            output["audio_encoder_config"] = self.audio_encoder_config

        # Ensure text_config is JSON-serializable
        if hasattr(self.text_config, "to_dict"):
            output["text_config"] = self.text_config.to_dict()
        else:
            output["text_config"] = self.text_config

        # Add all our custom attributes
        custom_attrs = [
            "audio_tokenizer_config",
            "audio_adapter_type",
            "audio_ffn_hidden_size",
            "audio_ffn_intermediate_size",
            "audio_num_codebooks",
            "audio_codebook_size",
            "audio_stream_bos_id",
            "audio_stream_eos_id",
            "audio_bos_token",
            "audio_bos_token_id",
            "audio_eos_token",
            "audio_eos_token_id",
            "audio_out_bos_token",
            "audio_out_bos_token_id",
            "audio_in_token",
            "audio_out_token",
            "audio_in_token_idx",
            "audio_out_token_idx",
            "pad_token_id",
        ]

        for attr in custom_attrs:
            if hasattr(self, attr):
                output[attr] = getattr(self, attr)

        # Ensure model_type is included
        output["model_type"] = self.model_type

        return output

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "HiggsAudioConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def to_json_string(self, indent: int = 2) -> str:
        """Convert configuration to JSON string."""
        import json

        return json.dumps(self.to_dict(), indent=indent)

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save configuration to a directory."""
        import os

        os.makedirs(save_directory, exist_ok=True)
        config_file = os.path.join(save_directory, "config.json")
        self.to_json_file(config_file)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> "HiggsAudioConfig":
        """Load configuration from a pretrained model path."""
        import os

        if os.path.isdir(model_name_or_path):
            config_file = os.path.join(model_name_or_path, "config.json")
            if os.path.isfile(config_file):
                return cls.from_json_file(config_file)
        # If it's not a directory or config.json doesn't exist,
        # assume it's the model_name_or_path itself is the config file
        if os.path.isfile(model_name_or_path):
            return cls.from_json_file(model_name_or_path)

        raise ValueError(f"Could not find config.json in {model_name_or_path}")

    @property
    def model_type(self) -> str:
        """Return model type for HuggingFace compatibility."""
        return self.architecture

    @classmethod
    def from_json_string(cls, json_string: str):
        """Create a new config from a JSON string."""
        return cls.from_dict(json.loads(json_string))


# Based on code from: https://github.com/zhenye234/xcodec
# Licensed under MIT License
# Modifications by BosonAI

import math
import os
from typing import Optional, Sequence, Union

import dac as dac2
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from huggingface_hub import snapshot_download
from transformers import AutoModel

try:
    from vector_quantize_pytorch import ResidualFSQ, ResidualVectorQuantizer
except Exception:  # pragma: no cover - optional dependency for tokenizer path
    ResidualFSQ = None
    ResidualVectorQuantizer = None


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
        ratios: Sequence[int] = [8, 5, 4, 2],  #  downsampling by 320
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
        # Minimal semantic encoder/decoder to satisfy construction; in real
        # usage, these come from the tokenizer components defined below.
        self.encoder_semantic = Encoder(
            input_channels=self.semantic_dim, encode_channels=self.encoder_semantic_dim
        )
        self.decoder_semantic = Decoder(
            code_dim=self.encoder_semantic_dim,
            output_channels=self.semantic_dim,
            decode_channels=self.semantic_dim,
        )

        # out_D=D+768
        if isinstance(bins, int) and ResidualVectorQuantizer is not None:  # RVQ
            self.quantizer = ResidualVectorQuantizer(
                dimension=self.quantizer_dim, codebook_dim=codebook_dim, n_q=n_q, bins=bins
            )
            self.quantizer_type = "RVQ"
        elif ResidualFSQ is not None:  # RFSQ
            self.quantizer = ResidualFSQ(dim=self.quantizer_dim, levels=bins, num_quantizers=n_q)
            self.quantizer_type = "RFSQ"
        else:
            raise ImportError(
                "vector_quantize_pytorch is required for HiggsAudioTokenizer; "
                "install it or provide compatible quantizer."
            )

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
            loud = meter.integrated_loudness(wv)
            wv = pyln.normalize.loudness(wv, loud, loudness_threshold)
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

    def _xcodec_encode(self, x: torch.Tensor, target_bw: Optional[int] = None) -> torch.Tensor:
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


class Encoder(nn.Module):
    """Lightweight 1D conv encoder used in tokenizer pipeline.

    This is a minimal stub to satisfy construction in environments where the
    full tokenizer submodules are not material to inference via TRT runner.
    """

    def __init__(self, input_channels: int, encode_channels: int):
        super().__init__()
        self.proj = nn.Conv1d(input_channels, encode_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Decoder(nn.Module):
    """Lightweight 1D conv decoder used in tokenizer pipeline."""

    def __init__(self, code_dim: int, output_channels: int, decode_channels: int):
        super().__init__()
        self.proj = nn.Conv1d(code_dim, decode_channels, kernel_size=1)
        self.out = nn.Conv1d(decode_channels, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(torch.relu(self.proj(x)))
