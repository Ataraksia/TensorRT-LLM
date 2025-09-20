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
        self.audio_generation_active = False
        self.audio_start_pos = -1
        self.seen_eos_in_frame = False
        # Track the delay state like vLLM
        self.num_audio_delay = 0  # How many complete frames have been generated

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
        try:
            if not token_ids or not token_ids[0]:
                self.reset()
                return

            input_ids = token_ids[0]
            vocab_size = logits.shape[-1]

            # 1. Determine if audio generation is active
            if not self.audio_generation_active:
                audio_out_bos_positions = [
                    i for i, token in enumerate(input_ids) if token == self.config.audio_out_bos_id
                ]
                if audio_out_bos_positions:
                    self.audio_generation_active = True
                    self.audio_start_pos = audio_out_bos_positions[-1] + 1
                    print(f"[DELAY] Audio generation activated at position {self.audio_start_pos}")
                else:
                    # Not in audio mode, do nothing
                    return

            if not self.audio_generation_active:
                return

            # 2. Calculate current position in the audio stream
            tokens_since_audio_start = len(input_ids) - self.audio_start_pos
            if tokens_since_audio_start < 0:
                return

            current_codebook_idx = tokens_since_audio_start % self.config.num_codebooks
            current_frame_idx = tokens_since_audio_start // self.config.num_codebooks

            # 3. Mask logits to the current codebook's vocabulary window
            flat_audio_vocab_size = self.config.codebook_size * self.config.num_codebooks

            # Mask out text tokens if the vocab includes them
            if vocab_size > flat_audio_vocab_size:
                logits[:, :, flat_audio_vocab_size:] = -math.inf

            # Calculate the vocabulary window for the current codebook
            window_start = current_codebook_idx * self.config.codebook_size
            window_end = window_start + self.config.codebook_size

            # Mask everything outside the current codebook's window
            logits[:, :, :window_start] = -math.inf
            if window_end < flat_audio_vocab_size:
                logits[:, :, window_end:flat_audio_vocab_size] = -math.inf

            # Define special token indices within the flattened vocab
            bos_token_id_in_window = window_start + self.config.audio_stream_bos_id
            eos_token_id_in_window = window_start + self.config.audio_stream_eos_id

            # 4. Check for EOS propagation within current frame
            start_of_current_frame = self.audio_start_pos + current_frame_idx * self.config.num_codebooks
            end_of_current_frame = min(start_of_current_frame + self.config.num_codebooks, len(input_ids))
            
            # Check if EOS was generated in current frame
            if not self.seen_eos_in_frame:
                for token_pos in range(start_of_current_frame, end_of_current_frame):
                    token = input_ids[token_pos]
                    if (token % self.config.codebook_size) == self.config.audio_stream_eos_id:
                        self.seen_eos_in_frame = True
                        break

            if self.seen_eos_in_frame:
                # Force EOS for the rest of this frame
                logits[:, :, :] = -math.inf
                logits[:, :, eos_token_id_in_window] = 0.0
                # Reset EOS flag when starting a new frame
                if current_codebook_idx == self.config.num_codebooks - 1:
                    self.seen_eos_in_frame = False
                return

            # 5. Apply delay pattern matching vLLM implementation  
            # The key insight: codebook i can only start generating content from frame i
            # num_audio_delay tracks how many codebooks are currently allowed to generate content
            
            # Update num_audio_delay based on current frame
            # In frame 0: only codebook 0 can generate content (num_audio_delay = 1)  
            # In frame 1: codebooks 0,1 can generate content (num_audio_delay = 2)
            # etc.
            self.num_audio_delay = min(current_frame_idx + 1, self.config.num_codebooks)

            if current_codebook_idx >= self.num_audio_delay:
                # This codebook is delayed - force BOS token  
                print(f"[DELAY] Frame {current_frame_idx}, codebook {current_codebook_idx}: DELAYED (num_active={self.num_audio_delay})")
                logits[:, :, :] = -math.inf
                logits[:, :, bos_token_id_in_window] = 0.0
            else:
                # This codebook is active - allow content, disallow BOS
                print(f"[DELAY] Frame {current_frame_idx}, codebook {current_codebook_idx}: ACTIVE (num_active={self.num_audio_delay})")
                logits[:, :, bos_token_id_in_window] = -math.inf

                # 6. Anti-repetition: penalize repeating the same content token
                if tokens_since_audio_start >= self.config.num_codebooks:
                    prev_token_pos = tokens_since_audio_start - self.config.num_codebooks
                    prev_token_id = input_ids[self.audio_start_pos + prev_token_pos]

                    # Penalize exact repetition within the current codebook window
                    if window_start <= prev_token_id < window_end:
                        local_prev_token = prev_token_id % self.config.codebook_size
                        if local_prev_token < self.config.audio_stream_bos_id:  # content token
                            logits[:, :, prev_token_id] -= 2.0  # Soft penalty instead of hard mask

        except Exception as e:
            print(f"[ERROR] DelayPatternLogitsProcessor failed: {e}")
            import traceback
            traceback.print_exc()
            # Allow generation to continue without this processor if it fails
            pass
