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
from typing import List, Optional

import torch

from .._utils import pad_vocab_size
from .generation import LogitsProcessor


class DelayPatternLogitsProcessor(LogitsProcessor):
    """
    A LogitsProcessor that enforces a delay pattern for multi-codebook audio generation,
    inspired by the vLLM implementation for Higgs-Audio.

    This processor is stateful and should be used for a single request at a time.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.reset()

    def reset(self):
        """Resets the internal state of the processor for a new request."""
        # Track the delay state like vLLM
        self.num_bos = 0
        self.num_eos = None

    def __call__(
        self,
        req_id: int,
        logits: torch.Tensor,
        token_ids: List[List[int]],
        stream_ptr: Optional[int],
        client_id: Optional[int],
    ) -> None:
        """
        Apply delay pattern constraints to logits during generation.
        Implements the exact delay pattern from vLLM reference.
        """
        with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
            input_ids = token_ids[0]
            logits = logits.view(1, self.config.num_codebooks, -1)
            for i in range(self.config.num_codebooks):
                if self.num_bos < self.config.num_codebooks:
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
