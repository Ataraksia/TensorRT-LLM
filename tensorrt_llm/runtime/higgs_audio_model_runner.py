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

import math
import torch
from typing import List, Optional

from .model_runner import ModelRunner
from .generation import SamplingConfig


class HiggsAudioModelRunner(ModelRunner):
    """
    Custom ModelRunner for Higgs Audio that handles multi-codebook generation.

    The key insight is that we need to generate 8 tokens per pass (one per codebook)
    instead of the current approach of generating 1 token and repeating it 8 times.

    This implements a custom generation loop that mimics the vLLM approach.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DelayPatternLogitsProcessor:
    """
    Enhanced version of DelayPatternLogitsProcessor that attempts to generate
    multiple tokens by manipulating the sampling process.
    """

    def __init__(self, config):
        self.config = config
        self.num_bos = 0
        self.num_eos = None

    def __call__(self, req_id: int, logits: torch.Tensor, token_ids):
        """Enhanced processing that tries to implement multi-codebook logic"""
        input_ids = token_ids[0]
        logits = logits.view(1, self.config.num_codebooks, -1)
        for i in range(self.config.num_codebooks):
            if self.num_bos < self.config.num_codebooks:
                # Logits is shape (num_batches, num_sequences, vocab_size) which is (1, 1, vocab_size) I was trying to see if I could maybe just reshape the codebooks into either the num_batches or num_sequences, but I didn't even any luck.
                logits[0, i, : self.config.codebook_size] = -math.inf
                logits[0, i, self.config.audio_stream_bos_id] = 0
        if self.num_eos is not None or input_ids[-1] == self.config.audio_stream_eos_id:
            if self.num_eos is None:
                self.num_eos = 0
            self.num_eos += 1

        self.num_bos += 1
        if self.num_eos and self.num_eos >= self.config.num_codebooks:
            logits[0, ..., : self.config.codebook_size] = -math.inf
            logits[0, ..., self.config.audio_stream_bos_id] = 0
            self.reset()
