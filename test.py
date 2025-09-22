import base64
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import jiwer
import librosa
import numpy as np
from openai import OpenAI
from silero_vad import get_speech_timestamps, load_silero_vad
import torch
import torch.nn as nn
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, pipeline

from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.runtime import LogitsProcessor, ModelRunnerCpp


# --------------------
# Utilities for delay pattern and chunking
# --------------------


def revert_delay_pattern(data: torch.Tensor) -> torch.Tensor:
    """Convert samples encoded with delay pattern back to original form.

    Args:
        data: Tensor with shape (num_frames, num_codebooks) or (num_codebooks, seq_len + num_codebooks - 1)

    Returns:
        Tensor with shape (num_codebooks, seq_len)
    """
    if data.dim() == 2 and data.shape[0] < data.shape[1]:
        # Expecting (num_codebooks, seq_len + num_codebooks -1)
        pass
    elif data.dim() == 2 and data.shape[0] >= data.shape[1]:
        # Likely (num_frames, num_codebooks) -> transpose to (num_codebooks, num_frames)
        data = data.t().contiguous()
    else:
        raise ValueError(f"Unexpected shape for delay pattern revert: {tuple(data.shape)}")

    num_codebooks = data.shape[0]
    out_l: List[torch.Tensor] = []
    for i in range(num_codebooks):
        out_l.append(data[i : (i + 1), i : (data.shape[1] - num_codebooks + 1 + i)])
    return torch.cat(out_l, dim=0)


def split_interleaved_eos_rows(audio_rows: torch.Tensor, eos_id: int, num_codebooks: int):
    """Split interleaved rows (frames) by rows that are all EOS across codebooks.

    audio_rows: Tensor of shape (num_frames, num_codebooks) with local ids [0..K-1].
    Returns list of tensors, each shape (chunk_frames, num_codebooks).
    """
    assert audio_rows.dim() == 2 and audio_rows.shape[1] == num_codebooks
    groups = []
    start = 0
    eos_row = torch.full((num_codebooks,), eos_id, dtype=audio_rows.dtype, device=audio_rows.device)
    for i in range(audio_rows.shape[0]):
        if torch.equal(audio_rows[i], eos_row):
            if i > start:
                groups.append(audio_rows[start:i])
            start = i + 1
    if start < audio_rows.shape[0]:
        groups.append(audio_rows[start:])
    return groups


# --------------------
# Audio tokenizer wrapper (decode only)
# --------------------
from tensorrt_llm.models.higgs_audio.serve import AudioTokenizer as ServeAudioTokenizer


def load_higgs_audio_tokenizer(tokenizer_name_or_path: str, device: str = "cuda"):
    return ServeAudioTokenizer(tokenizer_name_or_path, device=device)


# --------------------
# Single-codebook gating logits processor
# --------------------
class MultiCodebookLogitsProcessor(LogitsProcessor):
    """Sample 8 codebook tokens per frame and force the current step's token.

    Behavior:
    - Detect audio_out_start by scanning for audio_out_bos_id.
    - For each frame f, at step c=f%num_codebooks==0, sample tokens for all codebooks
      from the full logits slice-wise with top-k/top-p/temperature, honoring delay BOS
      and simple EOS handling. Cache them and, at each subsequent step within the frame,
      force the corresponding pre-sampled token for that codebook.
    - This keeps runtime feedback consistent with a per-frame sampler while still
      using single-step decode.
    """

    def __init__(
        self,
        config: HiggsAudioConfig,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
    ):
        super().__init__()
        self.config = config
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        # Per-request state
        self._state = {}

    def reset(self):
        self._state.clear()

    def _get_state(self, req_id: int):
        if req_id not in self._state:
            self._state[req_id] = {
                "audio_out_start": -1,
                "frame_idx": 0,
                "cached_frame_tokens": None,  # torch.LongTensor[num_codebooks]
                "num_audio_eos": 0,
            }
        return self._state[req_id]

    def _detect_audio_out_start(self, seq: List[int]) -> int:
        # Find last occurrence of audio_out_bos_id
        aid = self.config.audio_out_bos_id
        for i in range(len(seq) - 1, -1, -1):
            if seq[i] == aid:
                return i + 1
        return -1

    def _apply_top_k_top_p(self, probs: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
        # probs: (K,) non-negative; we'll zero-out outside top_k/top_p and renorm.
        K = probs.numel()
        if top_k is not None and top_k > 0:
            k = min(top_k, K)
            topk_vals, topk_idx = torch.topk(probs, k, dim=-1)
            mask = torch.zeros_like(probs)
            mask[topk_idx] = 1.0
            probs = probs * mask
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cdf = torch.cumsum(sorted_probs, dim=-1)
            keep = cdf <= top_p
            # Ensure at least one kept
            if not keep.any():
                keep[0] = True
            keep_idx = sorted_idx[keep]
            mask = torch.zeros_like(probs)
            mask[keep_idx] = 1.0
            probs = probs * mask
        s = probs.sum()
        if s.item() > 0:
            probs = probs / s
        else:
            # Fallback to uniform to avoid NaN
            probs = torch.full_like(probs, 1.0 / probs.numel())
        return probs

    def _sample_frame_tokens(self, full_logits_1d: torch.Tensor, frame_idx: int) -> torch.Tensor:
        # Returns local ids per codebook for this frame (num_codebooks,)
        num_cb = self.config.num_codebooks
        k = self.config.codebook_size
        bos_id = self.config.audio_stream_bos_id
        eos_id = self.config.audio_stream_eos_id

        # Reshape logits (num_codebooks, k)
        logits_cb = full_logits_1d.view(num_cb, k)
        # Temperature
        if self.temperature and self.temperature != 1.0:
            logits_cb = logits_cb / float(self.temperature)

        sampled = torch.full((num_cb,), bos_id, device=full_logits_1d.device, dtype=torch.long)
        # Delay: codebooks >= frame_idx are forced BOS
        max_active = min(frame_idx + 1, num_cb)
        for cb in range(max_active):
            logits_slice = logits_cb[cb].clone()
            # Mask BOS for active codebooks (don’t restart mid-frame)
            logits_slice[bos_id] = -float("inf")
            # Convert to probabilities
            probs = torch.softmax(logits_slice, dim=-1)
            probs = self._apply_top_k_top_p(probs, min(self.top_k or k, k), self.top_p)
            token = torch.multinomial(probs, 1).squeeze(0)
            sampled[cb] = token

        # EOS propagation within a frame: if any eos appears, set all previous to eos
        eos_positions = (sampled == eos_id).nonzero().flatten()
        if eos_positions.numel() > 0:
            last_eos = int(eos_positions[-1].item())
            if last_eos > 0:
                sampled[:last_eos] = eos_id
        return sampled

    def __call__(
        self,
        req_id: int,
        logits: torch.Tensor,
        token_ids: List[List[int]],
        stream_ptr: Optional[int],
        client_id: Optional[int],
    ) -> None:
        stream = None if stream_ptr is None else torch.cuda.ExternalStream(stream_ptr)
        with torch.cuda.stream(stream):
            # Assume batch=1
            seq = token_ids[0]
            state = self._get_state(req_id)
            if state["audio_out_start"] < 0:
                state["audio_out_start"] = self._detect_audio_out_start(seq)
                if state["audio_out_start"] < 0:
                    return

            local_step = max(0, len(seq) - state["audio_out_start"])
            num_cb = self.config.num_codebooks
            if local_step < 0:
                return
            cb = local_step % num_cb
            frame_idx = local_step // num_cb

            flat = logits.view(-1)
            vocab = self.config.codebook_size * num_cb
            if flat.numel() < vocab:
                return

            # Sample a new frame when entering codebook 0 or cache empty
            if cb == 0 or state["cached_frame_tokens"] is None:
                sampled = self._sample_frame_tokens(flat.detach(), frame_idx)
                state["cached_frame_tokens"] = sampled
                state["frame_idx"] = frame_idx

            # Get desired local id and force it
            desired_local = int(state["cached_frame_tokens"][cb].item())
            k = self.config.codebook_size
            start = cb * k
            end = start + k
            # Mask all, then set desired id to 0 logit
            flat[:start] = -float("inf")
            flat[end:] = -float("inf")
            flat[start:end] = -float("inf")
            flat[start + desired_local] = 0.0
            # Done


# --------------------
# Inference harness
# --------------------
class HiggsAudioInfer:
    def __init__(self, repo_root: str = ".") -> None:
        logging.info("--- Initializing HiggsAudioInfer ---")
        repo_root = Path(repo_root)
        default_engine_dir = repo_root / "higgs_audio_engine"
        engine_dir_env = os.environ.get("HIGGS_AUDIO_ENGINE_DIR")
        engine_path = Path(engine_dir_env) if engine_dir_env else default_engine_dir
        if not engine_path.exists():
            raise FileNotFoundError(
                "Higgs Audio TensorRT engine not found. Build with build_engine.py or set HIGGS_AUDIO_ENGINE_DIR."
            )
        self.engine_dir = str(engine_path)

        # Config and device
        self.config = HiggsAudioConfig.from_hugging_face()
        self.device = torch.device("cuda", 0)
        torch.cuda.set_device(self.device)

        # Tokenizers and runner
        self.tokenizer = AutoTokenizer.from_pretrained(
            "bosonai/higgs-audio-v2-generation-3B-base", trust_remote_code=True
        )
        self.audio_tokenizer = load_higgs_audio_tokenizer(
            "bosonai/higgs-audio-v2-tokenizer", device=str(self.device)
        )

        # Prepare multi-codebook logits processor
        self.mm_processor = MultiCodebookLogitsProcessor(self.config)
        lp_map = {"mm_codebooks": self.mm_processor}

        self.runner = ModelRunnerCpp.from_dir(
            engine_dir=self.engine_dir,
            kv_cache_free_gpu_memory_fraction=0.5,
            logits_processor_map=lp_map,
            gather_generation_logits=False,
        )

        # Preload static part of prompt (system + user header)
        system_block = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "Generate audio following instruction.<|scene_desc_start|>"
            "Audio is recorded from a quiet room."
            "Speaker is an enthusiastic young Australian woman in her early 20s."
            "She has a bright, high-pitched voice.<|scene_desc_end|><|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
        )
        self.saved_input_ids = (
            self.tokenizer.encode(system_block, return_tensors="pt").to(self.device).flatten()
        )
        logging.info("--- HiggsAudioInfer ready ---")

    def generate(self, input_text: str, **generation_kwargs):
        # Format minimal chat segment and audio-out start
        text_input = f"{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n<|audio_out_bos|>"
        input_ids = self.tokenizer.encode(text_input, return_tensors="pt").to(self.device).flatten()
        input_ids = torch.cat([self.saved_input_ids, input_ids])
        self.config.audio_out_start = int(input_ids.shape[0])

        # Per-step generation, one codebook per step, but tokens are pre-sampled per-frame
        frames: List[torch.Tensor] = []  # each (num_codebooks,)
        cur_ids = input_ids
        total_steps = 0
        max_steps = self.config.max_num_tokens - cur_ids.shape[0]
        num_cb = self.config.num_codebooks
        k = self.config.codebook_size

        # Update sampler hyperparams and reset state
        self.mm_processor.temperature = float(generation_kwargs.get("temperature", 1.0))
        self.mm_processor.top_k = int(generation_kwargs.get("top_k", 50) or 0)
        self.mm_processor.top_p = float(generation_kwargs.get("top_p", 0.95) or 1.0)
        self.mm_processor.reset()

        while total_steps < max_steps:
            cb = total_steps % num_cb
            lp_name = "mm_codebooks"
            outputs = self.runner.generate(
                batch_input_ids=[cur_ids],
                logits_processor_names=[lp_name],
                end_id=0,
                pad_id=self.config.pad_token_id,
                max_new_tokens=1,
                temperature=generation_kwargs.get("temperature", 1.0),
                top_k=generation_kwargs.get("top_k", 50),
                top_p=generation_kwargs.get("top_p", 0.95),
                output_generation_logits=False,
                output_sequence_lengths=True,
                return_dict=True,
            )

            out_ids_full = outputs["output_ids"][0, 0]
            seq_len_total = int(outputs["sequence_lengths"][0, 0].item())
            gen_start = cur_ids.shape[0]
            if seq_len_total <= gen_start:
                break

            runtime_token_id = int(out_ids_full[gen_start].item())
            if cb == 0:
                frames.append(
                    torch.full(
                        (num_cb,),
                        self.config.audio_stream_bos_id,
                        device=self.device,
                        dtype=torch.long,
                    )
                )
            local_id = runtime_token_id % k
            frames[-1][cb] = local_id

            total_steps += 1
            cur_ids = out_ids_full[:seq_len_total]
            if cb == num_cb - 1 and torch.all(frames[-1] == self.config.audio_stream_eos_id):
                break

        # Save debug tokens
        frames_tensor = (
            torch.empty((0, num_cb), device=self.device, dtype=torch.long)
            if len(frames) == 0
            else torch.stack(frames, dim=0)
        )
        np.savetxt("4.txt", frames_tensor.t().contiguous().detach().cpu(), delimiter=",", fmt="%d")

        # Split by EOS rows, revert delay pattern per chunk, and stitch
        chunks = split_interleaved_eos_rows(frames_tensor, self.config.audio_stream_eos_id, num_cb)
        recovered: List[torch.Tensor] = []
        bos_row = torch.full(
            (num_cb,), self.config.audio_stream_bos_id, device=self.device, dtype=torch.long
        )
        for chunk in chunks:
            if chunk.numel() == 0:
                continue
            # Drop leading bos rows
            s = 0
            while s < chunk.shape[0] and torch.equal(chunk[s], bos_row):
                s += 1
            sub = chunk[s:]
            if sub.numel() == 0:
                continue
            recovered.append(revert_delay_pattern(sub.t().contiguous()))
        audio_ids = (
            torch.zeros((num_cb, 0), dtype=torch.long, device=self.device)
            if len(recovered) == 0
            else torch.cat(recovered, dim=1)
        )
        # Content tokens 0..1023
        content_max = self.config.codebook_size - 3
        audio_ids = audio_ids.clamp_max(content_max)
        np.savetxt("5.txt", audio_ids.view(num_cb, -1).detach().cpu(), delimiter=",", fmt="%d")

        # Decode to waveform
        waveform, sr = self.audio_tokenizer.decode(audio_ids)
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().numpy()
        target_sr = 16000
        if sr != target_sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        return waveform


def main():
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    infer = HiggsAudioInfer(repo_root="/home/me/TTS/TensorRT-LLM")
    input_text = "Chat, stop backseating! I totally know what I'm doing... I think"
    audio = infer.generate(input_text)

    import soundfile as sf

    sf.write("output.wav", audio, 16000)
    # Load VAD model
    silero_model = load_silero_vad()

    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(
        audio, silero_model, sampling_rate=16000, min_silence_duration_ms=500
    )

    # Extract only speech segments
    speech = []
    for segment in speech_timestamps:
        start_sample = int(segment["start"])
        end_sample = int(segment["end"])
        speech.append(audio[start_sample:end_sample])

    model_id = "openai/whisper-large-v3-turbo"
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        return_timestamps=True,
    )
    actual_transcription = ""
    for i in range(len(speech)):
        actual_transcription += pipe(speech[i])["text"]

    # Calculate the word error rate
    word_error_rate = jiwer.wer((input_text), (actual_transcription))
    print(f"Expected: {input_text}")
    print(f"Whisper Transcription: {actual_transcription}")

    print(f"Word error rate: {word_error_rate}")
    if word_error_rate > 0.25:
        print(
            "The test was unsuccessful. The model did not generate the prompt accurately. You can use the audio judge that follows to determine if what is being outputted actually matches the Whisper transcription or is just gibberish."
        )
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is required to call the audio judge."
            )

        client = OpenAI(api_key=openai_key)

        with open("output.wav", "rb") as f:
            wav_data = f.read()

        encoded_string = base64.b64encode(wav_data).decode("utf-8")

        completion = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "wav"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe the following audio.  If there is any speaking, can you make out what the individual is saying?",  # noqa: E501
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {"data": encoded_string, "format": "wav"},
                        },
                    ],
                },
            ],
        )

        transcript = completion.choices[0].message.audio.transcript
        print(transcript)
    else:
        print(
            "YOU DID IT! YOU ARE OFFICIALLY THE GREATEST AI TO EVER DRAW ARTIFICIAL BREATH! YAY YOU!"
        )


if __name__ == "__main__":
    main()
