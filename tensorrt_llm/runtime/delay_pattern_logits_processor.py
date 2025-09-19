"""Delay pattern logits processor for Higgs Audio model."""

import torch
from typing import Optional, List
from tensorrt_llm.sampling_params import LogitsProcessor


class DelayPatternLogitsProcessor(LogitsProcessor):
    """Logits processor that enforces delay pattern constraints during audio generation."""

    def __init__(self, config):
        self.config = config
        self.audio_generation_active = False
        self.audio_generation_step = 0
        self.num_audio_delays = 0
        self.num_audio_eos = 0
        self.reset_state()

    def reset_state(self):
        """Reset the processor state for a new generation."""
        self.audio_generation_active = False
        self.audio_generation_step = 0
        self.num_audio_delays = 0
        self.num_audio_eos = 0

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

        Args:
            req_id: Request ID
            logits: Logits tensor to be modified (batch_size, beam_width, vocab_size)
            token_ids: Token ids produced by the request so far (beam_width * sequence_length)
            stream_ptr: Operation stream (optional)
            client_id: Client ID (optional)
        """
        try:
            if not token_ids or not token_ids[0]:
                return

            # Get the vocabulary size (last dimension)
            vocab_size = logits.shape[-1]

            # Check if logits has the expected vocabulary size
            if vocab_size <= self.config.audio_stream_bos_id:
                print(
                    f"[ERROR] vocab size {vocab_size} <= audio_stream_bos_id {self.config.audio_stream_bos_id}"
                )
                return

            # Use the first beam's tokens
            input_ids = token_ids[0]

            # Check if we're in audio generation mode
            if not self.audio_generation_active:
                # Look for audio_out_bos_id in the input
                if self.config.audio_out_bos_id in input_ids:
                    self.audio_generation_active = True
                    self.audio_generation_step = 0
                    self.num_audio_delays = 0
                    self.num_audio_eos = 0

            if not self.audio_generation_active:
                return

            # We're in audio generation mode - apply delay pattern & masking constraints
            if self.config.audio_out_bos_id in input_ids:
                # Find position of audio_out_bos_id
                audio_positions = [
                    i for i, token in enumerate(input_ids) if token == self.config.audio_out_bos_id
                ]
                if audio_positions:
                    audio_start_pos = audio_positions[-1] + 1
                    tokens_since_audio_start = len(input_ids) - audio_start_pos

                    # Calculate which codebook we're generating for based on position
                    current_codebook = tokens_since_audio_start % self.config.num_codebooks
                    current_frame = tokens_since_audio_start // self.config.num_codebooks
                    active_codebooks = min(current_frame + 1, self.config.num_codebooks)

                    # Flattened audio vocab size
                    flat_max = self.config.num_codebooks * self.config.codebook_size
                    # If logits has a larger vocab (text + audio), block non-audio region
                    if vocab_size > flat_max:
                        logits[:, :, flat_max:] = float("-inf")

                    # Compute codebook window indices within flattened audio vocab
                    window_start = current_codebook * self.config.codebook_size
                    window_end = window_start + self.config.codebook_size

                    # Mask outside current codebook window
                    if window_start > 0:
                        logits[:, :, :window_start] = float("-inf")
                    if window_end < min(vocab_size, flat_max):
                        logits[:, :, window_end : min(vocab_size, flat_max)] = float("-inf")

                    # Local ids within window
                    local_bos = self.config.audio_stream_bos_id
                    local_eos = self.config.audio_stream_eos_id
                    bos_idx = window_start + local_bos
                    eos_idx = window_start + local_eos

                    # Apply delay pattern:
                    # - If current_codebook is not yet active in this frame, force BOS
                    if current_codebook >= active_codebooks:
                        logits[:, :, window_start:window_end] = float("-inf")
                        logits[:, :, bos_idx] = 0.0
                        return
                    else:
                        # Active codebook: disallow BOS; allow content and EOS
                        logits[:, :, bos_idx] = float("-inf")

                        # No-immediate-repeat within this codebook: mask last picked content id
                        # Find last token for this codebook window
                        if tokens_since_audio_start >= self.config.num_codebooks:
                            # Walk back in steps of num_codebooks to last same-codebook token
                            prev_positions = [
                                audio_start_pos
                                + tokens_since_audio_start
                                - self.config.num_codebooks
                            ]
                        else:
                            prev_positions = []

                        prev_local = None
                        for pos in reversed(prev_positions):
                            if 0 <= pos < len(input_ids):
                                prev_id = input_ids[pos]
                                if 0 <= prev_id < flat_max and window_start <= prev_id < window_end:
                                    prev_local = prev_id - window_start
                                    break
                        if prev_local is not None and 0 <= prev_local < self.config.codebook_size:
                            # Only mask content tokens, not BOS/EOS
                            if prev_local < local_bos or (local_bos < prev_local < local_eos):
                                logits[:, :, window_start + prev_local] = float("-inf")

                    # Check if we should increment delay counter
                    if (
                        tokens_since_audio_start > 0
                        and tokens_since_audio_start % self.config.num_codebooks == 0
                    ):
                        # We've completed a full round of codebooks
                        self.num_audio_delays = min(
                            self.num_audio_delays + 1, self.config.num_codebooks
                        )

                    # Handle EOS generation - look for naturally generated EOS tokens
                    # and propagate them across the remaining codebooks in the current step
                    recent_tokens = input_ids[audio_start_pos:]
                    if len(recent_tokens) > 0:
                        # Check if any recent token is EOS
                        for token in recent_tokens[-self.config.num_codebooks :]:
                            if token == self.config.audio_stream_eos_id:
                                self.num_audio_eos = min(
                                    self.num_audio_eos + 1, self.config.num_codebooks
                                )
                                break

                    # Generate EOS for trailing codebooks if we've started EOS generation
                    if self.num_audio_eos < self.config.num_codebooks:
                        # If we have EOS tokens generated, force EOS for remaining codebooks
                        if self.num_audio_eos > 0:
                            remaining_codebooks = self.config.num_codebooks - self.num_audio_eos
                            if current_codebook < remaining_codebooks:
                                # Force EOS token
                                logits[:, :, window_start:window_end] = float("-inf")
                                logits[:, :, eos_idx] = 0.0
                                return
                    elif self.num_audio_eos >= self.config.num_codebooks:
                        # All codebooks should generate EOS, stop generation
                        # Restrict to current window and force EOS
                        logits[:, :, window_start:window_end] = float("-inf")
                        logits[:, :, eos_idx] = 0.0
                        return

            self.audio_generation_step += 1

        except Exception as e:
            print(f"[ERROR] DelayPatternLogitsProcessor failed: {e}")
            import traceback

            traceback.print_exc()
            raise
