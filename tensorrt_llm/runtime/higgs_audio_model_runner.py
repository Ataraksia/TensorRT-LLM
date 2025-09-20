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
import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from dotenv import load_dotenv

from tensorrt_llm.runtime.generation import SamplingConfig
from tensorrt_llm.runtime.model_runner_cpp import ModelRunnerCpp
from tensorrt_llm.runtime.session import Session, TensorInfo

from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from .delay_pattern_logits_processor import DelayPatternLogitsProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class HiggsAudioModelRunner(ModelRunnerCpp):
    """A wrapper for the HiggsAudio TensorRT-LLM model."""

    def __init__(
        self,
        config: HiggsAudioConfig,
        session: Session,
        max_batch_size: int,
        max_input_len: int,
        max_seq_len: int,
        max_beam_width: int,
        max_num_tokens: int,
        use_lora_plugin: bool,
        lora_manager: Optional[object] = None,
        medusa_choices: torch.Tensor = None,
    ):
        """
        Initializes the HiggsAudioModelRunner.

        Args:
            config: The HiggsAudioConfig object.
            session: The TensorRT-LLM session.
            max_batch_size: The maximum batch size.
            max_input_len: The maximum input length.
            max_seq_len: The maximum sequence length.
            max_beam_width: The maximum beam width.
            max_num_tokens: The maximum number of tokens.
            use_lora_plugin: Whether to use the LoRA plugin.
            lora_manager: The LoRA manager.
            medusa_choices: The Medusa choices.
        """
        logging.info("Initializing HiggsAudioModelRunner...")
        self.config = config
        super().__init__(
            session,
            max_batch_size,
            max_input_len,
            max_seq_len,
            max_beam_width,
            max_num_tokens,
            use_lora_plugin,
            lora_manager,
            medusa_choices,
        )
        logging.info("HiggsAudioModelRunner initialized successfully.")
        """
        Initializes the HiggsAudioModelRunner.

        Args:
            config: The HiggsAudioConfig object.
            session: The TensorRT-LLM session.
            max_batch_size: The maximum batch size.
            max_input_len: The maximum input length.
            max_seq_len: The maximum sequence length.
            max_beam_width: The maximum beam width.
            max_num_tokens: The maximum number of tokens.
            use_lora_plugin: Whether to use the LoRA plugin.
            lora_manager: The LoRA manager.
            medusa_choices: The Medusa choices.
        """
        logging.info("Initializing HiggsAudioModelRunner...")
        self.config = config
        super().__init__(
            session,
            max_batch_size,
            max_input_len,
            max_seq_len,
            max_beam_width,
            max_num_tokens,
            use_lora_plugin,
            lora_manager,
            medusa_choices,
        )
        logging.info("HiggsAudioModelRunner initialized successfully.")

    def generate(
        self,
        batch_input_ids: List[torch.Tensor],
        sampling_config: Optional[Union[dict, "SamplingConfig"]] = None,
        **kwargs,
    ):
        """
        Generates sequences of token ids for a batch of prompts.

        Args:
            batch_input_ids: A list of tensors, each containing the input token ids for a sample.
            sampling_config: The sampling configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            A tensor of shape [batch_size, beam_width, max_seq_len] with the generated token ids.
        """
        logging.info("Starting generation...")
        print(f"[DEBUG] Calling super().generate with kwargs: {list(kwargs.keys())}")
        outputs = super().generate(batch_input_ids, sampling_config=sampling_config, **kwargs)
        logging.info("Generation finished.")
        return outputs

    @classmethod
    def from_dir(
        cls,
        engine_dir: Union[str, Path],
        config: HiggsAudioConfig,
        lora_dir: Optional[str] = None,
        rank: int = 0,
        debug_mode: bool = False,
        lora_ckpt_source: str = "hf",
        medusa_choices: torch.Tensor | None = None,
        **kwargs,
    ) -> "HiggsAudioModelRunner":
        """
        Creates a HiggsAudioModelRunner instance from an engine directory.

        Args:
            engine_dir: The directory containing the TensorRT-LLM engine.
            config: The HiggsAudioConfig object.
            lora_dir: The directory containing the LoRA checkpoints.
            rank: The rank of the current process.
            debug_mode: Whether to enable debug mode.
            lora_ckpt_source: The source of the LoRA checkpoints.
            medusa_choices: The Medusa choices.
            **kwargs: Additional keyword arguments.

        Returns:
            A HiggsAudioModelRunner instance.
        """
        logging.info(f"Creating HiggsAudioModelRunner from directory: {engine_dir}")

        runner = ModelRunnerCpp.from_dir(
            engine_dir=engine_dir,
            lora_dir=lora_dir,
            rank=rank,
            debug_mode=debug_mode,
            lora_ckpt_source=lora_ckpt_source,
            medusa_choices=medusa_choices,
            **kwargs,
        )
        logging.info("ModelRunner.from_dir() completed.")

        runner.__class__ = HiggsAudioModelRunner
        runner.config = config

        logging.info("HiggsAudioModelRunner created and configured.")
        return runner

    def _reset_higgs_state(self) -> None:
        self.debug_log: List[str] = []

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
