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
import inspect
import json
import os
from typing import Dict, Optional

import numpy as np
import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

from tensorrt_llm.module import Module

# Prefer the upstream boson multimodal tokenizer implementation when available.
try:
    from boson_multimodal.higgs_audio_tokenizer import (
        HiggsAudioTokenizer as BosonHiggsAudioTokenizer,
    )
except Exception:  # pragma: no cover - optional dependency
    BosonHiggsAudioTokenizer = None  # type: ignore

from .config import HiggsAudioConfig


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
            A dictionary containing the token codes, frames per second (fps), and metadata.
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
