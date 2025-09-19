from dotenv import load_dotenv
import torch
from typing import Optional, List
import copy

from tensorrt_llm.runtime import ModelRunnerCpp

load_dotenv()


class HiggsAudioModelRunner(ModelRunnerCpp):
    def __init__(self, *args, config=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self._reset_higgs_state()

    def _reset_higgs_state(self) -> None:
        self.debug_log: List[str] = []

    @classmethod
    def from_dir(cls, config, **kwargs):
        instance = super().from_dir(**kwargs)
        instance.config = config
        instance._reset_higgs_state()
        return instance

    def _fill_output(self, **kwargs):
        """Override to apply delay pattern to the complete generated sequence."""
        self.debug_log.append("_fill_output called")

        # Apply delay pattern to responses before calling parent
        responses = kwargs.get("responses", [])
        self.debug_log.append(f"Processing {len(responses)} responses")

        for response in responses:
            if hasattr(response.result, "output_token_ids") and response.result.output_token_ids:
                original_tokens = response.result.output_token_ids[0]
                self.debug_log.append(f"Original response has {len(original_tokens)} tokens")

                # Apply delay pattern
                modified_tokens = self._apply_delay_pattern_to_full_sequence(original_tokens)

                if modified_tokens != original_tokens:
                    response.result.output_token_ids[0] = modified_tokens
                    self.debug_log.append(f"Applied delay pattern, modified sequence")
                else:
                    self.debug_log.append(f"No delay pattern modifications needed")

        # Write debug log
        if responses and responses[0].result.is_final:
            with open("/home/me/TTS/TensorRT-LLM/debug_log.txt", "w") as f:
                for line in self.debug_log:
                    f.write(line + "\n")

        # Call parent method with potentially modified responses
        return super()._fill_output(**kwargs)

    def _apply_delay_pattern_to_full_sequence(self, tokens):
        """Apply delay pattern to the complete token sequence."""
        self.debug_log.append(f"Analyzing sequence of {len(tokens)} tokens")

        if self.config.audio_out_bos_id not in tokens:
            self.debug_log.append("No audio_out_bos_id found, returning original sequence")
            return tokens

        # Find the audio section
        audio_start_idx = tokens.index(self.config.audio_out_bos_id)
        self.debug_log.append(f"Found audio_out_bos_id at position {audio_start_idx}")

        # Find where audio ends
        audio_tokens_start = audio_start_idx + 1
        if self.config.audio_eos_id in tokens[audio_tokens_start:]:
            audio_end_idx = tokens.index(self.config.audio_eos_id, audio_tokens_start)
            audio_tokens = tokens[audio_tokens_start:audio_end_idx]
        else:
            # Use rest of sequence
            audio_tokens = tokens[audio_tokens_start:]
            audio_end_idx = len(tokens)

        self.debug_log.append(
            f"Audio section: {len(audio_tokens)} tokens from {audio_tokens_start} to {audio_end_idx}"
        )

        if len(audio_tokens) == 0 or len(audio_tokens) % self.config.num_codebooks != 0:
            # Truncate to complete frames if possible
            if len(audio_tokens) >= self.config.num_codebooks:
                complete_tokens = (
                    len(audio_tokens) // self.config.num_codebooks
                ) * self.config.num_codebooks
                self.debug_log.append(
                    f"Truncating {len(audio_tokens)} audio tokens to {complete_tokens} complete frames"
                )
                audio_tokens = audio_tokens[:complete_tokens]
                audio_end_idx = audio_tokens_start + len(audio_tokens)
            else:
                self.debug_log.append(
                    f"Audio tokens not valid for delay pattern: {len(audio_tokens)} % {self.config.num_codebooks} = {len(audio_tokens) % self.config.num_codebooks}"
                )
                return tokens

        # Apply delay pattern to audio tokens
        num_frames = len(audio_tokens) // self.config.num_codebooks
        audio_tensor = torch.tensor(audio_tokens).view(num_frames, self.config.num_codebooks)

        self.debug_log.append(
            f"Reshaping to {num_frames} frames x {self.config.num_codebooks} codebooks"
        )

        modified = False
        modifications_count = 0
        expected_modifications = 0

        # Calculate expected modifications for diagnostic purposes
        for frame_idx in range(num_frames):
            active_codebooks = min(frame_idx + 1, self.config.num_codebooks)
            expected_modifications += self.config.num_codebooks - active_codebooks

        for frame_idx in range(num_frames):
            # At time step t, only codebooks 0 to min(t, num_codebooks-1) should be active
            active_codebooks = min(frame_idx + 1, self.config.num_codebooks)

            for cb in range(active_codebooks, self.config.num_codebooks):
                if audio_tensor[frame_idx, cb] != self.config.audio_stream_bos_id:
                    audio_tensor[frame_idx, cb] = self.config.audio_stream_bos_id
                    modified = True
                    modifications_count += 1

        self.debug_log.append(
            f"Expected {expected_modifications} modifications, made {modifications_count}"
        )

        # Debug: show several frames in detail
        for frame_idx in [0, 1, 2, 7, 50, 100]:
            if frame_idx < num_frames:
                active_cbs = min(frame_idx + 1, self.config.num_codebooks)
                frame_tokens = audio_tensor[frame_idx].tolist()
                bos_count = sum(1 for t in frame_tokens if t == self.config.audio_stream_bos_id)
                content_count = sum(1 for t in frame_tokens if 0 <= t < self.config.codebook_size)
                self.debug_log.append(
                    f"Frame {frame_idx} (active_cbs={active_cbs}): BOS={bos_count}, content={content_count}, tokens={frame_tokens}"
                )

        if modified:
            # Replace audio section in original tokens
            modified_audio_tokens = audio_tensor.flatten().tolist()
            modified_tokens = (
                tokens[:audio_tokens_start] + modified_audio_tokens + tokens[audio_end_idx:]
            )

            # Final verification: check delay pattern is correct
            final_tensor = torch.tensor(modified_audio_tokens).view(
                num_frames, self.config.num_codebooks
            )
            delay_pattern_correct = True
            for frame_idx in range(min(num_frames, self.config.num_codebooks)):
                active_codebooks = frame_idx + 1
                for cb in range(active_codebooks, self.config.num_codebooks):
                    if final_tensor[frame_idx, cb] != self.config.audio_stream_bos_id:
                        delay_pattern_correct = False
                        break

            self.debug_log.append(
                f"✅ DELAY PATTERN VERIFICATION: {'PASSED' if delay_pattern_correct else 'FAILED'}"
            )

            return modified_tokens
        else:
            return tokens
