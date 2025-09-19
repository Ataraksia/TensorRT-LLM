from dotenv import load_dotenv
import torch
from typing import Optional, List
import copy

from tensorrt_llm.runtime import ModelRunnerCpp
from tensorrt_llm.runtime.delay_pattern_logits_processor import DelayPatternLogitsProcessor

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
        # Create delay pattern logits processor
        delay_processor = DelayPatternLogitsProcessor(config)

        # Add it to the logits processor map
        logits_processor_map = kwargs.get("logits_processor_map", {})
        logits_processor_map["delay_pattern"] = delay_processor
        kwargs["logits_processor_map"] = logits_processor_map

        instance = super().from_dir(**kwargs)
        instance.config = config
        instance.delay_processor = delay_processor
        instance._reset_higgs_state()
        return instance

    def generate(self, batch_input_ids: List[torch.Tensor], **kwargs):
        # Reset delay processor state for new generation
        if hasattr(self, "delay_processor"):
            self.delay_processor.reset_state()

        # Temporarily disable LogitsProcessor and only use post-processing
        # kwargs["logits_processor_names"] = ["delay_pattern"]

        self.debug_log.append("Using post-processing delay pattern implementation ONLY")

        result = super().generate(batch_input_ids, **kwargs)

        # Write debug log
        with open("/home/me/TTS/TensorRT-LLM/debug_log.txt", "w") as f:
            for line in self.debug_log:
                f.write(line + "\n")

        return result

    def _fill_output(
        self,
        *,
        responses,
        output_ids,
        end_id,
        return_dict,
        output_sequence_lengths,
        output_generation_logits,
        output_log_probs,
        output_cum_log_probs,
        batch_input_ids,
        batch_input_ids_list,
        streaming,
        request_ids,
        return_all_generated_tokens,
        sampling_config,
        is_draft_target_model: bool,
    ):
        """Post-process the output to apply delay pattern matching vLLM implementation."""
        # Call parent implementation first
        result = super()._fill_output(
            responses=responses,
            output_ids=output_ids,
            end_id=end_id,
            return_dict=return_dict,
            output_sequence_lengths=output_sequence_lengths,
            output_generation_logits=output_generation_logits,
            output_log_probs=output_log_probs,
            output_cum_log_probs=output_cum_log_probs,
            batch_input_ids=batch_input_ids,
            batch_input_ids_list=batch_input_ids_list,
            streaming=streaming,
            request_ids=request_ids,
            return_all_generated_tokens=return_all_generated_tokens,
            sampling_config=sampling_config,
            is_draft_target_model=is_draft_target_model,
        )

        self.debug_log.append(f"_fill_output called, result type: {type(result)}")

        # Apply vLLM-style delay pattern post-processing to audio tokens
        if isinstance(result, dict):
            # Result is a dict with output_ids key
            output_ids_to_process = result.get("output_ids")
            self.debug_log.append("Result is dict")
        elif hasattr(result, "output_ids"):
            # Result is an object with output_ids attribute
            output_ids_to_process = result.output_ids
            self.debug_log.append("Result has output_ids attribute")
        else:
            # Result is the output_ids tensor directly
            output_ids_to_process = result
            self.debug_log.append("Result is tensor directly")

        if output_ids_to_process is not None:
            self.debug_log.append(
                f"output_ids_to_process type: {type(output_ids_to_process)}, shape: {output_ids_to_process.shape if hasattr(output_ids_to_process, 'shape') else 'no shape'}"
            )

            # Convert tensor to list format for processing
            if isinstance(output_ids_to_process, torch.Tensor):
                if len(output_ids_to_process.shape) == 3:
                    # Batch × beam × sequence format - flatten to get actual sequences
                    output_ids_list = []
                    for i in range(output_ids_to_process.shape[0]):
                        for j in range(output_ids_to_process.shape[1]):
                            seq = output_ids_to_process[i][j].tolist()
                            output_ids_list.append(seq)
                elif len(output_ids_to_process.shape) == 2:
                    # Batch × sequence format
                    output_ids_list = [
                        output_ids_to_process[i].tolist()
                        for i in range(output_ids_to_process.shape[0])
                    ]
                else:
                    # Single sequence format
                    output_ids_list = [output_ids_to_process.tolist()]
            else:
                # Already a list
                output_ids_list = output_ids_to_process

            self.debug_log.append(f"output_ids_list length: {len(output_ids_list)}")
            if output_ids_list:
                self.debug_log.append(f"First sequence length: {len(output_ids_list[0])}")
                self.debug_log.append(f"First 20 tokens: {output_ids_list[0][:20]}")

            # Process each output sequence
            for output_idx in range(len(output_ids_list)):
                output_ids_seq = output_ids_list[output_idx]

                # Find audio generation section
                audio_out_bos_positions = []
                for i, token_id in enumerate(output_ids_seq):
                    if token_id == self.config.audio_out_bos_id:
                        audio_out_bos_positions.append(i)

                self.debug_log.append(f"Found audio_out_bos positions: {audio_out_bos_positions}")

                if not audio_out_bos_positions:
                    continue  # No audio generation in this output

                # Process each audio generation section
                for audio_bos_pos in audio_out_bos_positions:
                    audio_start = audio_bos_pos + 1
                    audio_tokens = []

                    # Show context around audio generation
                    context_start = max(0, audio_bos_pos - 5)
                    context_end = min(len(output_ids_seq), audio_bos_pos + 50)
                    context_tokens = output_ids_seq[context_start:context_end]
                    self.debug_log.append(
                        f"Context around audio_out_bos (pos {audio_bos_pos}): {context_tokens}"
                    )

                    # Collect audio tokens until we hit a non-audio token or end
                    for i in range(audio_start, len(output_ids_seq)):
                        token_id = output_ids_seq[i]
                        if token_id < 1024 or token_id in [1024, 1025]:  # audio content or BOS/EOS
                            audio_tokens.append(token_id)
                        else:
                            self.debug_log.append(
                                f"Audio generation stopped at position {i} with non-audio token: {token_id}"
                            )
                            break  # Hit non-audio token

                    self.debug_log.append(f"Collected {len(audio_tokens)} audio tokens")

                    if not audio_tokens:
                        continue

                    self.debug_log.append(f"Raw audio tokens (first 50): {audio_tokens[:50]}")

                    # Apply vLLM-style delay pattern processing
                    processed_tokens = self._apply_vllm_delay_pattern(audio_tokens)

                    self.debug_log.append(
                        f"Processed audio tokens (first 50): {processed_tokens[:50]}"
                    )

                    # Replace the audio tokens in the output
                    end_pos = audio_start + len(audio_tokens)
                    output_ids_list[output_idx] = (
                        output_ids_seq[:audio_start] + processed_tokens + output_ids_seq[end_pos:]
                    )

            # Update the result with processed tokens, preserving original shape
            if isinstance(result, dict):
                if len(output_ids_to_process.shape) == 3:
                    # Reconstruct 3D tensor shape
                    batch_size, beam_size, _ = output_ids_to_process.shape
                    reconstructed = torch.zeros(
                        (batch_size, beam_size, max(len(seq) for seq in output_ids_list)),
                        dtype=torch.long,
                    )
                    for idx, seq in enumerate(output_ids_list):
                        batch_idx = idx // beam_size
                        beam_idx = idx % beam_size
                        seq_tensor = torch.tensor(seq, dtype=torch.long)
                        reconstructed[batch_idx, beam_idx, : len(seq_tensor)] = seq_tensor
                    result["output_ids"] = reconstructed
                else:
                    result["output_ids"] = torch.tensor(output_ids_list)
            elif hasattr(result, "output_ids"):
                if len(output_ids_to_process.shape) == 3:
                    batch_size, beam_size, _ = output_ids_to_process.shape
                    reconstructed = torch.zeros(
                        (batch_size, beam_size, max(len(seq) for seq in output_ids_list)),
                        dtype=torch.long,
                    )
                    for idx, seq in enumerate(output_ids_list):
                        batch_idx = idx // beam_size
                        beam_idx = idx % beam_size
                        seq_tensor = torch.tensor(seq, dtype=torch.long)
                        reconstructed[batch_idx, beam_idx, : len(seq_tensor)] = seq_tensor
                    result.output_ids = reconstructed
                else:
                    result.output_ids = torch.tensor(output_ids_list)
            else:
                if len(output_ids_to_process.shape) == 3:
                    batch_size, beam_size, _ = output_ids_to_process.shape
                    reconstructed = torch.zeros(
                        (batch_size, beam_size, max(len(seq) for seq in output_ids_list)),
                        dtype=torch.long,
                    )
                    for idx, seq in enumerate(output_ids_list):
                        batch_idx = idx // beam_size
                        beam_idx = idx % beam_size
                        seq_tensor = torch.tensor(seq, dtype=torch.long)
                        reconstructed[batch_idx, beam_idx, : len(seq_tensor)] = seq_tensor
                    result = reconstructed
                else:
                    result = torch.tensor(output_ids_list)
        else:
            self.debug_log.append("output_ids_to_process is None!")

        return result

    def _apply_vllm_delay_pattern(self, audio_tokens):
        """Apply delay pattern post-processing matching vLLM implementation."""
        if not audio_tokens:
            return audio_tokens

        # Convert to tensor for easier manipulation
        tokens_tensor = torch.tensor(audio_tokens, dtype=torch.long)

        # Reshape into frames × codebooks
        total_tokens = len(audio_tokens)
        if total_tokens % self.config.num_codebooks != 0:
            # Pad to complete frames
            pad_length = self.config.num_codebooks - (total_tokens % self.config.num_codebooks)
            tokens_tensor = torch.cat(
                [
                    tokens_tensor,
                    torch.full((pad_length,), self.config.audio_stream_bos_id, dtype=torch.long),
                ]
            )

        num_frames = len(tokens_tensor) // self.config.num_codebooks
        frames = tokens_tensor.view(num_frames, self.config.num_codebooks)

        self.debug_log.append(f"Original frames shape: {frames.shape}")

        # Initialize tracking variables like vLLM
        num_audio_delays = [0] * 1  # Single batch
        num_audio_eos = [0] * 1

        # Process frames sequentially like vLLM does
        for frame_idx in range(num_frames):
            current_frame = frames[frame_idx]

            # Apply delay pattern: force BOS for delayed codebooks
            if num_audio_delays[0] < self.config.num_codebooks:
                for codebook_idx in range(num_audio_delays[0], self.config.num_codebooks):
                    current_frame[codebook_idx] = self.config.audio_stream_bos_id

            # Increment delay counter for next frame
            num_audio_delays[0] = min(num_audio_delays[0] + 1, self.config.num_codebooks)

            # Check for EOS tokens and handle EOS propagation
            eos_found = False
            for codebook_idx in range(self.config.num_codebooks):
                if current_frame[codebook_idx] == self.config.audio_stream_eos_id:
                    eos_found = True
                    break

            if eos_found and num_audio_eos[0] < self.config.num_codebooks:
                num_audio_eos[0] = min(num_audio_eos[0] + 1, self.config.num_codebooks)

                # Propagate EOS to trailing codebooks
                for codebook_idx in range(
                    self.config.num_codebooks - num_audio_eos[0], self.config.num_codebooks
                ):
                    current_frame[codebook_idx] = self.config.audio_stream_eos_id

        self.debug_log.append(f"Post-vLLM-processing frames shape: {frames.shape}")

        # Flatten back to token list
        processed_tokens = frames.flatten().tolist()

        return processed_tokens
