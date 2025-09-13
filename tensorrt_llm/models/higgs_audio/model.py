# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
"""TensorRT-LLM implementation of Higgs Audio multimodal model."""

import librosa
from collections.abc import AsyncGenerator
import os
from typing import Optional, List
import numpy as np
from openai.types.chat import ChatCompletionAudio
import torch
from boson_multimodal import *
from starlette.datastructures import State
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    pipeline,
)
from tensorrt_llm.bindings import INT32
from tensorrt_llm.mapping import Mapping

from tensorrt_llm.models.model_weights_loader import ModelWeightsLoader
from tensorrt_llm.models.modeling_utils import (
    DecoderLayerList,
    QuantConfig,
    DecoderModelForCausalLM,
)
from tensorrt_llm.module import Module, ModuleList
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.sampling_params import LogitsProcessor
from tensorrt_llm.functional import (
    Tensor,
    arange,
    cumsum,
    expand_dims_like,
    unsqueeze,
    where,
    sum,
    mean,
)
from tensorrt_llm.layers import (
    MLP,
    Attention,
    AttentionMaskType,
    AttentionParams,
    ColumnLinear,
    Embedding,
    KeyValueCacheParams,
    RmsNorm,
)
import inspect
import json
import os
from typing import Optional


class AudioTokenizer:
    """Common interface for audio tokenizers."""

    def __init__(self, model, device="cuda:0"):
        self._model = model
        self._device = device
        self.audio_tokenizer_model = load_higgs_audio_tokenizer(
            model,
            device=device,
        )
        self._tps = self.audio_tokenizer_model.frame_rate
        self._sampling_rate = self.audio_tokenizer_model.sample_rate
        self._num_codebooks = self.audio_tokenizer_model.n_q
        self._codebook_size = self.audio_tokenizer_model.quantizer_dim

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
        self,
        audio_path_or_wv,
        sr=None,
        loudness_normalize=False,
        loudness_threshold=-23.0,
    ):
        return self.audio_tokenizer_model.encode(
            audio_path_or_wv, sr, loudness_normalize, loudness_threshold
        )

    def decode(self, vq_code, return_cuda_tensor=False):
        """Decode the audio codes to waveform.

        Parameters:
        -----------
        vq_code: torch.Tensor
            The audio codes to decode. Shape (num_codebooks, total_length)

        Returns:
        --------
        decoded_wv: np.ndarray
            The decoded waveform. Shape (#time,)
        sampling_rate: int
            The sampling rate of the decoded waveform.
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


def load_higgs_audio_tokenizer(tokenizer_name_or_path, device="cuda"):
    is_local = os.path.exists(tokenizer_name_or_path)
    if not is_local:
        tokenizer_path = snapshot_download(tokenizer_name_or_path)
    else:
        tokenizer_path = tokenizer_name_or_path
    config_path = os.path.join(tokenizer_path, "config.json")
    if os.path.exists(config_path):
        config = json.load(open(config_path))
    else:
        raise ValueError(f"No config file found in {tokenizer_path}")
    model_path = os.path.join(tokenizer_path, "model.pth")

    # Dynamically get valid parameters from HiggsAudioTokenizer.__init__ method
    init_signature = inspect.signature(HiggsAudioTokenizer.__init__)
    valid_params = set(init_signature.parameters.keys()) - {"self"}  # exclude 'self'
    filtered_config = {k: v for k, v in config.items() if k in valid_params}

    model = HiggsAudioTokenizer(
        **filtered_config,
        device=device,
    )
    parameter_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(parameter_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def get_feat_extract_output_lengths(input_lengths: torch.LongTensor):
    """Computes the output length of the convolutional layers and the output length of the audio encoder"""
    input_lengths = (input_lengths - 1) // 2 + 1
    output_lengths = (input_lengths - 2) // 2 + 1
    return input_lengths, output_lengths


def revert_delay_pattern(data):
    """Convert samples encoded with delay pattern back to the original form.

    Args:
        data: The data with delay pattern applied.
              Shape (num_codebooks, seq_len + num_codebooks - 1).

    Returns:
        ret: Recovered data with delay pattern removed.
             Shape (num_codebooks, seq_len).
    """
    assert len(data.shape) == 2
    out_l = []
    num_codebooks = data.shape[0]
    for i in range(num_codebooks):
        out_l.append(data[i : (i + 1), i : (data.shape[1] - num_codebooks + 1 + i)])
    return (
        torch.cat(out_l, dim=0) if isinstance(data, torch.Tensor) else np.concatenate(out_l, axis=0)
    )


def _build_delay_pattern_mask(input_ids: torch.LongTensor, bos_token_id: int, pad_token_id: int):
    """Implement the delay pattern proposed in "Simple and Controllable Music Generation".

    In the delay pattern, each codebook is offset by the previous codebook by
    one. We insert a special delay token at the start of the sequence if its delayed,
    and append pad token once the sequence finishes.

    Args:
        input_ids: The input ids of the prompt. Shape (bsz, num_codebooks, seq_len).
        bos_token_id: The id of the special delay token
        pad_token_id: The id of the padding token. Should be the same as eos_token_id.

    Returns:
        input_ids: The transformed input ids with delay pattern applied.
                  Shape (bsz, num_codebooks, seq_len + num_codebooks - 1).
    """
    bsz, num_codebooks, seq_len = input_ids.shape

    new_seq_len = seq_len + num_codebooks - 1
    input_ids_with_gen_mask = torch.ones(
        (bsz, num_codebooks, new_seq_len), dtype=torch.long, device=input_ids.device
    )
    bos_mask = torch.tril(input_ids_with_gen_mask, -1) > 0
    eos_mask = torch.triu(input_ids_with_gen_mask, seq_len) > 0
    input_ids_with_gen_mask[bos_mask] = bos_token_id
    input_ids_with_gen_mask[(~bos_mask) & (~eos_mask)] = input_ids.reshape(-1)
    input_ids = input_ids_with_gen_mask.clone()
    input_ids[eos_mask] = pad_token_id
    input_ids_with_gen_mask[eos_mask] = -1
    return input_ids


class HiggsAudioDualFFNDecoderLayer(Module):
    """TensorRT-LLM implementation of dual-path FFN decoder layer."""

    def __init__(self, config: HiggsAudioConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = self.config.hidden_size

        # Shared attention layer
        self.attention = Attention(
            hidden_size=self.config.hidden_size,
            num_attention_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            max_position_embeddings=self.config.max_position_embeddings,
            num_layers=self.config.num_hidden_layers,
            attention_mask_type=AttentionMaskType.causal,
            bias=False,
            dtype=self.config.dtype,
            local_layer_idx=layer_idx,
        )

        # Text MLP
        self.mlp = MLP(
            hidden_size=self.config.hidden_size,
            ffn_hidden_size=self.config.intermediate_size,
            hidden_act=self.config.hidden_act,
            dtype=self.config.dtype,
            bias=False,
        )

        # Audio MLP (use same intermediate size as text MLP to avoid broadcast issues)
        self.audio_mlp = MLP(
            hidden_size=self.config.hidden_size,
            ffn_hidden_size=self.config.intermediate_size,
            hidden_act=self.config.hidden_act,
            dtype=self.config.dtype,
            bias=False,
        )

        # Layer norms
        self.input_layernorm = RmsNorm(
            normalized_shape=self.config.hidden_size,
            eps=self.config.norm_epsilon,
            dtype=self.config.dtype,
        )

        self.audio_input_layernorm = RmsNorm(
            normalized_shape=self.config.hidden_size,
            eps=self.config.norm_epsilon,
            dtype=self.config.dtype,
        )

        self.post_layernorm = RmsNorm(
            normalized_shape=self.config.hidden_size,
            eps=self.config.norm_epsilon,
            dtype=self.config.dtype,
        )

        self.audio_post_layernorm = RmsNorm(
            normalized_shape=self.config.hidden_size,
            eps=self.config.norm_epsilon,
            dtype=self.config.dtype,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        use_cache: bool = False,
        kv_cache_params: Optional[KeyValueCacheParams] = None,
        attention_params: Optional[AttentionParams] = None,
        vision_token_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass for dual FFN decoder layer."""
        residual = hidden_states

        hidden_states = where(
            vision_token_mask.unsqueeze(-1),
            self.audio_input_layernorm(hidden_states),
            self.input_layernorm(hidden_states),
        )

        hidden_states = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
        )

        if use_cache:
            hidden_states, presents = hidden_states

        hidden_states = residual + hidden_states

        residual = hidden_states

        residual += where(
            vision_token_mask.unsqueeze(-1),
            self.audio_mlp(self.audio_post_layernorm(hidden_states)),
            self.mlp(self.post_layernorm(hidden_states)),
        )

        hidden_states = residual

        if use_cache:
            return (hidden_states, presents)
        return hidden_states, presents


class HiggsAudioTransformer(Module):
    """TensorRT-LLM transformer component for Higgs Audio model."""

    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.config = config

        self.vocab_embedding = Embedding(
            num_embeddings=self.config.text_vocab_size,
            embedding_dim=self.config.hidden_size,
            dtype=self.config.dtype,
        )

        self.audio_codebook_embeddings = Embedding(
            num_embeddings=self.config.audio_vocab_size,
            embedding_dim=self.config.hidden_size,
            dtype=self.config.dtype,
        )

        self.layers = DecoderLayerList(HiggsAudioDualFFNDecoderLayer, config)

        self.ln_f = RmsNorm(
            normalized_shape=self.config.hidden_size,
            eps=self.config.norm_epsilon,
            dtype=self.config.dtype,
        )

    def _embed_audio_ids(self, audio_ids: Tensor):
        """Embed the audio ids.
        The audio embedding table is laid out as a concatenation of codebook slices,
        each of length (codebook_size + 2), where the last two entries are BOS/EOS.
        We therefore shift each codebook by (codebook_size + 2) when indexing.
        """
        num_codebooks = self.config.audio_num_codebooks
        codebook_size = self.config.audio_codebook_size
        slice_size = codebook_size + 2
        codebook_shift = (arange(0, num_codebooks, "int32") * slice_size).unsqueeze(-1)
        audio_embed = sum(
            self.audio_codebook_embeddings(audio_ids + codebook_shift),
            dim=0,
        )
        return audio_embed

    def forward(
        self,
        hidden_states: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        use_cache: bool = False,
        attention_mask: Optional[Tensor] = None,
        kv_cache_params: Optional[KeyValueCacheParams] = None,
        attention_params: Optional[AttentionParams] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass for Higgs Audio transformer with multimodal support."""
        audio_mask = input_ids > (self.config.text_vocab_size - 1)

        # Audio tokens are those with IDs >= text_vocab_size
        # This should include generated audio tokens during inference

        text_ids = where(audio_mask, self.config.text_vocab_size - 1, input_ids)
        text_embed = self.vocab_embedding(text_ids)
        audio_ids = where(audio_mask, input_ids - self.config.text_vocab_size, 0)
        audio_embed = self._embed_audio_ids(audio_ids)
        input_embed = where(unsqueeze(audio_mask, -1), audio_embed, text_embed)

        hidden_states = self.layers(
            hidden_states=input_embed,
            use_cache=use_cache,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            vision_token_mask=audio_mask,
        )

        if use_cache:
            hidden_states, presents = hidden_states

        hidden_states = self.ln_f(hidden_states)

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class HiggsAudioForCausalLM(DecoderModelForCausalLM):
    """TensorRT-LLM implementation of Higgs Audio multimodal model."""

    def __init__(self, config: HiggsAudioConfig):
        # Initialize the transformer component
        transformer = HiggsAudioTransformer(config)

        lm_head = ColumnLinear(
            in_features=config.hidden_size,
            out_features=config.audio_vocab_size,
            bias=False,
            dtype=config.dtype,
            tp_group=None,
            tp_size=1,
            gather_output=True,
        )

        super().__init__(config, transformer, lm_head)

    @classmethod
    def from_hugging_face(
        cls,
        hf_config_or_dir: str,
        dtype: str = "bfloat16",
        mapping: Optional[Mapping] = None,
        quant_config: Optional[QuantConfig] = None,
        **kwargs,
    ):
        """Create a HiggsAudioForCausalLM object from HuggingFace model directory.

        Args:
            hf_config_or_dir: Path to the HuggingFace model directory
            dtype: Data type for the model weights
            mapping: Multi-GPU mapping configuration
            quant_config: Quantization configuration
            **kwargs: Additional keyword arguments

        Returns:
            HiggsAudioForCausalLM: The loaded model
        """
        if not os.path.exists(hf_config_or_dir):
            hf_config_or_dir = snapshot_download(repo_id=hf_config_or_dir)

        config = HiggsAudioConfig.from_hugging_face(hf_config_or_dir, **kwargs)
        custom_dict = {
            "transformer": "",
            "lm_head": "audio_decoder_proj.audio_lm_head",
            "audio_post_layernorm": "audio_post_attention_layernorm",
        }
        loader = ModelWeightsLoader(hf_config_or_dir, custom_dict)
        trtllm_model = cls(config)
        loader.update_key_mapping(trtllm_model)
        loader.generate_tllm_weights(trtllm_model)

        return trtllm_model


class HiggsAudioLogitsProcessor(LogitsProcessor):
    """Custom logits processor for HiggsAudio that applies delay pattern logic during generation."""

    def __init__(self, config: HiggsAudioConfig):
        self.config = config
        self.text_vocab_size = config.text_vocab_size
        self.audio_vocab_size = config.audio_vocab_size
        self.audio_num_codebooks = config.audio_num_codebooks
        self.audio_stream_bos_id = config.audio_stream_bos_id
        self.audio_stream_eos_id = config.audio_stream_eos_id

        # Track delay pattern state per request
        self.request_states = {}

    def _get_or_create_state(self, req_id: int):
        """Get or create delay pattern state for a request."""
        if req_id not in self.request_states:
            self.request_states[req_id] = {
                # When None, we have not observed an EOS yet to start the tail delays
                "num_remaining_delays": None,
                # Number of activated codebooks so far (starts from 0)
                "num_delay": 0,
            }
        return self.request_states[req_id]

    def __call__(
        self,
        req_id: int,
        logits: torch.Tensor,
        token_ids: List[List[int]],
        stream_ptr: Optional[int],
        client_id: Optional[int],
    ) -> None:
        """Apply delay pattern logic to audio logits during generation.

        Enforces the delay pattern by:
        - Activating codebooks cyclically and masking logits to the current codebook slice.
        - Prefilling later codebooks with BOS until they are activated.
        - After seeing an EOS in any codebook, prefill earlier codebooks with EOS for the
          remaining positions in the current frame and tail off in subsequent frames.
        - Finally, when all remaining delays are consumed, force the stream EOS to stop.
        """

        state = self._get_or_create_state(req_id)

        # token_ids is List[beam][generated_tokens]; use the first beam
        past = token_ids[0] if token_ids and token_ids[0] is not None else []
        tokens_generated = len(past)

        num_cbs = self.audio_num_codebooks
        cb_size = self.config.audio_codebook_size
        slice_size = cb_size + 2  # include BOS/EOS per slice

        # Determine current codebook position (cyclic across codebooks)
        current_cb_pos = tokens_generated % num_cbs
        current_base = current_cb_pos * slice_size

        # 1) Mask logits to only allow the current codebook slice
        if logits is not None and logits.numel() > 0:
            # keep only [current_base, current_base + slice_size)
            if current_base > 0:
                logits[..., :current_base] = -1.0e20
            tail_start = current_base + slice_size
            if tail_start < self.audio_vocab_size:
                logits[..., tail_start:] = -1.0e20

        forced_local_id: Optional[int] = None  # id within slice [0..slice_size-1]
        # By default, do not allow sampling BOS/EOS unless explicitly forced
        bos_global = current_base + self.audio_stream_bos_id
        eos_global = current_base + self.audio_stream_eos_id
        if logits is not None and logits.numel() > 0:
            # Disallow BOS by default; add small bias to EOS to encourage termination
            logits[..., bos_global] = -1.0e20
            logits[..., eos_global] += 1.0  # Small positive bias to encourage EOS

        # 2) Initial delays (force BOS for not-yet-activated codebooks)
        if state["num_delay"] + 1 < num_cbs:
            if current_cb_pos > state["num_delay"]:
                forced_local_id = self.audio_stream_bos_id  # equals cb_size
            if current_cb_pos == num_cbs - 1:
                state["num_delay"] += 1

        # 3) Tail delays after encountering any EOS in the past stream
        if state["num_remaining_delays"] is not None:
            # For early positions in the frame, force EOS
            if current_cb_pos < (num_cbs - state["num_remaining_delays"]):
                forced_local_id = self.audio_stream_eos_id
            if current_cb_pos == num_cbs - 1:
                state["num_remaining_delays"] -= 1
        else:
            if tokens_generated > 0:
                past_tensor = torch.as_tensor(past, device=logits.device)
                # Detect EOS irrespective of slice by modulo slice_size
                eos_positions = ((past_tensor % slice_size) == self.audio_stream_eos_id).nonzero(
                    as_tuple=False
                )
                if eos_positions.numel() > 0:
                    last_eos_idx = int(eos_positions[-1, 0].item())
                    last_eos_pos_in_frame = last_eos_idx % num_cbs
                    state["num_remaining_delays"] = num_cbs - last_eos_pos_in_frame - 1
                    if current_cb_pos < last_eos_pos_in_frame:
                        forced_local_id = self.audio_stream_eos_id

        # 4) End generation when all delays are consumed
        if state["num_remaining_delays"] is not None and state["num_remaining_delays"] <= 0:
            forced_local_id = self.audio_stream_eos_id
            state["num_delay"] = 0
            state["num_remaining_delays"] = None

        # 5) Apply forced token inside the current slice (convert to global id)
        if forced_local_id is not None:
            forced_global_id = current_base + forced_local_id
            logits[...] = -1.0e20
            if 0 <= forced_global_id < self.audio_vocab_size:
                logits[..., forced_global_id] = 0.0
            print(
                f"🔒 FORCED token at pos {tokens_generated}, cb {current_cb_pos}: local_id={forced_local_id}, global_id={forced_global_id}"
            )
        else:
            print(
                f"🎲 SAMPLING at pos {tokens_generated}, cb {current_cb_pos}: delay={state['num_delay']}, remaining_delays={state['num_remaining_delays']}"
            )

        # AUDIO-SPACE ANTI-REPETITION: operate within the current codebook slice
        # Penalize repeating the same local id within a codebook across frames,
        # and lightly discourage the most common continuation given the last bigram.
        # TEMPORARILY DISABLED FOR DEBUGGING
        if False and logits is not None and logits.numel() > 0 and tokens_generated > 0:
            slice_size = cb_size + 2

            # Gather local-id history for this codebook (look back up to 64 frames)
            local_hist: List[int] = []
            pos = tokens_generated - num_cbs  # previous time this cb was active
            steps = 0
            while pos >= 0 and steps < 64:
                local_hist.append(int(past[pos] % slice_size))
                pos -= num_cbs
                steps += 1

            if local_hist:
                last_local = local_hist[0]
                # 1) Streak penalty: discourage repeating the same id many frames in a row
                streak = 1
                for x in local_hist[1:]:
                    if x == last_local:
                        streak += 1
                    else:
                        break
                # Apply increasing penalty on the last_local id within current slice
                penalty = 0.6 * (1.25 ** max(0, streak - 2)) if streak >= 2 else 0.0
                if penalty > 0.0:
                    logits[..., current_base + last_local] -= penalty

                # 2) Bigram->next continuation penalty within this codebook
                if len(local_hist) >= 2:
                    bigram = (local_hist[1], local_hist[0])  # (t-2, t-1)
                    cont_counts = {}
                    # Build counts from sliding window over history (oldest to newest is reversed)
                    for i in range(len(local_hist) - 2, -1, -1):
                        a = local_hist[i + 1] if i + 1 < len(local_hist) else None
                        b = local_hist[i]
                        c = local_hist[i - 1] if i - 1 >= 0 else None
                        if a is None or c is None:
                            continue
                        if (a, b) == bigram:
                            cont_counts[c] = cont_counts.get(c, 0) + 1
                    if cont_counts:
                        worst_next = max(cont_counts, key=cont_counts.get)
                        logits[..., current_base + worst_next] -= 0.8

                # 3) If streak is long, encourage EOS to terminate the stream sooner
                if streak >= 6:
                    logits[..., eos_global] += min(4.0, 0.8 * (streak - 5))

                # 4) Frequency-based penalty on overused local IDs in recent window
                # Count frequencies over last up to 48 frames
                freq_counts = {}
                for idx, lid in enumerate(local_hist[:48]):
                    freq_counts[lid] = freq_counts.get(lid, 0) + 1
                if freq_counts:
                    # Sort by frequency (desc) and penalize top few
                    top_locals = sorted(freq_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
                    penalties = [0.6, 0.4, 0.2]
                    for j, (lid, _) in enumerate(top_locals):
                        if j < len(penalties):
                            logits[..., current_base + lid] -= penalties[j]

                # 5) Simple cycle detection to break short loops (period 2..6)
                # Detect repeating pattern at head of history (newest first)
                def detect_cycle(seq, max_period=6, min_repeats=4):
                    for p in range(2, max_period + 1):
                        if len(seq) < p * min_repeats:
                            continue
                        pattern = seq[:p]
                        ok = True
                        for r in range(1, min_repeats):
                            if seq[r * p : (r + 1) * p] != pattern:
                                ok = False
                                break
                        if ok:
                            return pattern
                    return None

                pattern = detect_cycle(local_hist, max_period=6, min_repeats=4)
                if pattern:
                    # Heavily penalize the next element of the detected pattern
                    next_in_pattern = pattern[0]
                    logits[..., current_base + next_in_pattern] -= 2.0
                    # And nudge EOS upwards to allow graceful stop
                    logits[..., eos_global] += 2.0

            # 6) Cross-frame cycle detection across all codebooks
            # Build recent frames of local ids (each frame has num_cbs ids)
            frames = tokens_generated // num_cbs
            if frames >= 6:  # need some history
                max_frames_considered = min(frames, 96)
                start = (frames - max_frames_considered) * num_cbs
                flat = past[start : start + max_frames_considered * num_cbs]
                # Build frames_local newest last
                frames_local: List[List[int]] = []
                for fi in range(0, len(flat), num_cbs):
                    chunk = flat[fi : fi + num_cbs]
                    if len(chunk) < num_cbs:
                        break
                    frames_local.append([int(t % slice_size) for t in chunk])
                # Detect repeated cycle of period p repeated >=3 times
                L = len(frames_local)
                for p in range(2, min(12, L // 3) + 1):
                    if (
                        frames_local[L - p : L]
                        == frames_local[L - 2 * p : L - p]
                        == frames_local[L - 3 * p : L - 2 * p]
                    ):
                        idx_in_cycle = L % p
                        expected_local = frames_local[L - p + idx_in_cycle][current_cb_pos]
                        logits[..., current_base + expected_local] -= 1.2
                        logits[..., eos_global] += 0.6
                        break

            # 7) Add tiny noise to logits to escape flat attractors
            with torch.no_grad():
                logits.add_(torch.empty_like(logits).uniform_(-1e-3, 1e-3))


class HiggsAudioTRTRunner:
    """TensorRT-LLM inference wrapper for HiggsAudio using ModelRunnerCpp."""

    def __init__(
        self,
    ) -> None:
        """Initialize the TensorRT-LLM runner for HiggsAudio."""

        self.config = HiggsAudioConfig.from_hugging_face(
            "bosonai/higgs-audio-v2-generation-3B-base"
        )
        self.engine_dir = "/home/me/TTS/TensorRT-LLM/higgs_audio_engine/"
        self.hf_model_dir = "bosonai/higgs-audio-v2-generation-3B-base"
        self.audio_tokenizer_dir = "bosonai/higgs-audio-v2-tokenizer"
        self.reference_audio = "/home/me/TTS/TensorRT-LLM/AussieGirl.wav"

        self.temperature = 1.0
        self.top_k = 50
        self.top_p = 0.95
        self.num_beams = 1
        self.gpu_weights_percent = 0.5
        self.max_seq_len = 2048

        # Set up device
        self.gpu_device = torch.device("cuda", 0)
        torch.cuda.set_device(self.gpu_device)

        # Load components
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_dir, trust_remote_code=True)
        self.audio_tokenizer = AudioTokenizer(self.audio_tokenizer_dir, device=str(self.gpu_device))

        # Create custom logits processor for delay pattern handling
        self.audio_logits_processor = HiggsAudioLogitsProcessor(self.config)

        self.reference_audio = ""
        # Preload the part of the input that doesn't change
        if self.reference_audio and self.audio_tokenizer:
            # Load and transcribe reference audio for voice cloning
            whisper_model_id = "openai/whisper-large-v3-turbo"
            whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(whisper_model_id)
            processor = AutoProcessor.from_pretrained(whisper_model_id)
            audio, _ = librosa.load(self.reference_audio, sr=16000)
            pipe = pipeline(
                "automatic-speech-recognition",
                model=whisper_model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                return_timestamps=True,
            )
            transcription = pipe(audio)["text"]

            # Validate audio file exists
            if not os.path.exists(self.reference_audio):
                raise FileNotFoundError(f"Reference audio file not found: {self.reference_audio}")

            audio_ids = self.audio_tokenizer.encode(self.reference_audio, sr=24000)

            # Apply delay pattern if requested and we have multiple codebooks
            # Add BOS and EOS tokens using correct token IDs
            bos_tokens = torch.full(
                (audio_ids.shape[0], 1),
                self.config.audio_stream_bos_id,
                dtype=audio_ids.dtype,
                device=audio_ids.device,
            )
            eos_tokens = torch.full(
                (audio_ids.shape[0], 1),
                self.config.audio_stream_eos_id,
                dtype=audio_ids.dtype,
                device=audio_ids.device,
            )
            # Concatenate: BOS + audio_ids + EOS
            audio_ids = torch.cat([bos_tokens, audio_ids, eos_tokens], dim=-1)

            # Apply delay pattern
            audio_ids = _build_delay_pattern_mask(
                audio_ids.unsqueeze(0),  # Add batch dimension
                bos_token_id=self.config.audio_stream_bos_id,
                pad_token_id=self.config.audio_stream_eos_id,
            ).flatten()
            audio_ids += self.config.text_vocab_size
            # Format with reference audio (voice cloning) following Higgs Audio expected format
            # The format should include the reference audio transcription and then the target text
            pre_audio_input = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
                f"Generate audio following instruction.<|scene_desc_start|>"
                f"Audio is recorded from a quiet room."
                f"Speaker is an enthusiastic young Australian woman in her early 20s."
                f"She has a bright, high-pitched voice.<|scene_desc_end|><|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>{transcription}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|><|audio_bos|>"
            )
            pre_audio_input_ids = (
                self.tokenizer.encode(pre_audio_input, return_tensors="pt").squeeze(0).cuda()
            )
            post_audio_input = f"<|audio_eos|><|eot_id|><|start_header_id|>user<|end_header_id|>"
            post_audio_input_ids = (
                self.tokenizer.encode(post_audio_input, return_tensors="pt").squeeze(0).cuda()
            )

            self.saved_input_ids = torch.cat(
                [pre_audio_input_ids, audio_ids, post_audio_input_ids], dim=0
            )
        else:
            # Format without reference audio (default voice)
            # Simplified format for direct text-to-speech without voice cloning
            text_input = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
                f"Generate audio following instruction.<|scene_desc_start|>"
                f"Audio is recorded from a quiet room."
                f"Speaker is an enthusiastic young Australian woman in her early 20s."
                f"She has a bright, high-pitched voice.<|scene_desc_end|><|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>"
            )
            self.saved_input_ids = (
                self.tokenizer.encode(text_input, return_tensors="pt").squeeze(0).cuda()
            )

        from tensorrt_llm.runtime import ModelRunnerCpp

        self.runner = ModelRunnerCpp.from_dir(
            engine_dir=self.engine_dir,
            kv_cache_free_gpu_memory_fraction=0.5,
            use_gpu_direct_storage=True,
            cuda_graph_mode=True,
            logits_processor_map={"higgs_audio_delay_pattern": self.audio_logits_processor},
        )

    def generate(
        self,
        input_text: str,
        **generation_kwargs,
    ):
        """Generate audio with improved sampling parameters and EOS handling."""

        text_input = (
            f"{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|><|audio_out_bos|>"
        )

        text_input_ids = self.tokenizer.encode(text_input, return_tensors="pt").squeeze(0).cuda()
        audio_input_ids = torch.as_tensor(
            [self.config.text_vocab_size + self.config.audio_stream_bos_id]
        ).cuda()
        input_ids = torch.cat(
            [self.saved_input_ids, text_input_ids, audio_input_ids],
            dim=0,
        ).cuda()

        print(f"Input construction:")
        print(f"  saved_input_ids length: {len(self.saved_input_ids)}")
        print(f"  text_input_ids length: {len(text_input_ids)}")
        print(f"  audio_input_ids: {audio_input_ids}")
        print(f"  Total input length: {len(input_ids)}")
        print(f"  Last few input tokens: {input_ids[-10:].tolist()}")
        print(f"  Text vocab size: {self.config.text_vocab_size}")
        print(
            f"  First audio token should be: {self.config.text_vocab_size + self.config.audio_stream_bos_id}"
        )

        # Check if there are any audio tokens in the input
        audio_tokens_in_input = (input_ids >= self.config.text_vocab_size).sum().item()
        print(f"  Audio tokens in input: {audio_tokens_in_input}")

        if audio_tokens_in_input > 0:
            print(
                f"  Audio token positions: {torch.where(input_ids >= self.config.text_vocab_size)[0].tolist()}"
            )
            audio_token_values = input_ids[input_ids >= self.config.text_vocab_size]
            print(f"  Audio token values (global): {audio_token_values.tolist()}")
            print(
                f"  Audio token values (local): {(audio_token_values - self.config.text_vocab_size).tolist()}"
            )

        input_lengths = torch.tensor(
            [input_ids.size(-1)], device=self.gpu_device, dtype=torch.int32
        )
        max_input_length = torch.max(input_lengths).item()
        max_new_tokens = min(
            512, self.max_seq_len - max_input_length
        )  # Reduced for faster iteration

        with torch.no_grad():
            # Prepare stop words: any slice-specific EOS should terminate generation
            slice_size = self.config.audio_codebook_size + 2
            all_eos_ids = [
                i * slice_size + self.config.audio_stream_eos_id
                for i in range(self.config.audio_num_codebooks)
            ]
            stop_words = [[[eid] for eid in all_eos_ids]]

            # Use improved sampling parameters for maximum diversity
            outputs = self.runner.generate(
                batch_input_ids=[input_ids],
                max_new_tokens=max_new_tokens,
                end_id=all_eos_ids[0],
                stop_words_list=stop_words,
                temperature=1.3,  # reduce randomness to avoid loops
                top_k=30,
                top_p=0.6,
                repetition_penalty=1.2,
                logits_processor_names=["higgs_audio_delay_pattern"],
            )

        # Extract and process audio tokens with proper delay pattern handling
        try:
            print(f"Raw generation output length: {len(outputs[0, 0])}")
            print(f"First 20 generated tokens: {outputs[0, 0][:20].tolist()}")
            print(f"Last 20 generated tokens: {outputs[0, 0][-20:].tolist()}")

            vq_code = self._extract_and_process_audio_tokens(outputs[0, 0])
            print(f"Extracted audio tokens shape: {vq_code.shape}")
            print(f"Audio token value ranges per codebook:")
            for i in range(vq_code.shape[0]):
                cb_tokens = vq_code[i]
                print(
                    f"  Codebook {i}: min={cb_tokens.min().item()}, max={cb_tokens.max().item()}, unique_vals={len(torch.unique(cb_tokens))}"
                )

            # Check for reasonable diversity in tokens
            total_unique = len(torch.unique(vq_code))
            print(f"Total unique audio token values across all codebooks: {total_unique}")

            # Decode to waveform
            waveform, sr = self.audio_tokenizer.decode(vq_code)
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.detach().cpu().numpy()
            if sr != 16000 and isinstance(waveform, np.ndarray):
                waveform = librosa.resample(
                    waveform.astype(np.float32), orig_sr=sr, target_sr=16000
                )
                sr = 16000
            return waveform.astype(np.float32)
        except Exception as e:
            print(f"Error processing audio tokens: {e}")
            return None

    def _extract_and_process_audio_tokens(self, generated_tokens):
        """Extract and process audio tokens from a flat interleaved stream.
        The runtime returns a 1D sequence of audio-vocab token IDs, interleaved across
        codebooks in cyclic order. We first de-interleave into shape
        (num_codebooks, seq_len_per_codebook), then revert the delay pattern.
        """
        if not isinstance(generated_tokens, torch.Tensor):
            generated_tokens = torch.as_tensor(generated_tokens)

        print(f"Input generated_tokens shape: {generated_tokens.shape}")
        print(
            f"Text vocab size: {self.config.text_vocab_size}, Audio vocab size: {self.config.audio_vocab_size}"
        )

        # Filter to only audio tokens (>= text_vocab_size)
        audio_mask = generated_tokens >= self.config.text_vocab_size
        audio_only = generated_tokens[audio_mask]
        print(f"Total tokens: {len(generated_tokens)}, Audio tokens: {len(audio_only)}")

        if len(audio_only) == 0:
            raise ValueError("No audio tokens found in generation output")

        # Convert back to local audio vocab space
        audio_local = audio_only - self.config.text_vocab_size
        print(
            f"Audio local token ranges: min={audio_local.min().item()}, max={audio_local.max().item()}"
        )

        num_codebooks = self.config.audio_num_codebooks
        total = int(audio_local.numel())
        if total < num_codebooks:
            raise ValueError(f"Not enough audio tokens: {total} < {num_codebooks}")

        # Keep full frames only
        trim_len = (total // num_codebooks) * num_codebooks
        if trim_len == 0:
            raise ValueError("No valid audio tokens after trimming")
        flat = audio_local[:trim_len]
        seq_len = trim_len // num_codebooks

        print(f"Trimmed to {trim_len} tokens ({seq_len} per codebook)")

        # De-interleave: [t0_cb0, t0_cb1, ..., t0_cbN-1, t1_cb0, ...] -> [N, T]
        audio_tokens = flat.view(seq_len, num_codebooks).transpose(0, 1).contiguous()
        print(f"De-interleaved audio tokens: {audio_tokens.shape}")

        # Check token distributions before delay pattern reversion
        print("Token distributions before delay pattern reversion:")
        for i in range(min(num_codebooks, 4)):  # Show first few codebooks
            cb_tokens = audio_tokens[i]
            unique_vals = torch.unique(cb_tokens)
            print(
                f"  Codebook {i}: unique tokens = {len(unique_vals)}, sample = {cb_tokens[:10].tolist()}"
            )

        # Apply delay pattern reversion to align codebooks in time
        vq_code = revert_delay_pattern(audio_tokens)

        print(f"After delay pattern reversion: {vq_code.shape}")

        # Remove leading BOS and trailing EOS columns if present
        original_width = vq_code.shape[1]
        if vq_code.shape[1] > 0 and torch.all(vq_code[:, 0] == self.config.audio_stream_bos_id):
            vq_code = vq_code[:, 1:]
            print(f"Removed leading BOS column")
        if vq_code.shape[1] > 0 and torch.all(vq_code[:, -1] == self.config.audio_stream_eos_id):
            vq_code = vq_code[:, :-1]
            print(f"Removed trailing EOS column")

        print(f"After BOS/EOS removal: {vq_code.shape} (was {original_width})")

        if vq_code.shape[1] == 0:
            raise ValueError("No audio content after removing BOS/EOS tokens")

        # Clip to valid codebook range
        before_clip_min, before_clip_max = vq_code.min().item(), vq_code.max().item()
        vq_code = vq_code.clip(0, self.config.audio_codebook_size - 1)
        after_clip_min, after_clip_max = vq_code.min().item(), vq_code.max().item()

        print(
            f"Token clipping: [{before_clip_min}, {before_clip_max}] -> [{after_clip_min}, {after_clip_max}]"
        )
        print(f"Final vq_code shape: {vq_code.shape}")

        return vq_code

    def _sample_audio_tokens_multicodebook(
        self, audio_logits, audio_out_ids, num_delay, num_remaining_delays
    ):
        """Sample audio tokens for all codebooks simultaneously with delay pattern."""
        import torch.nn.functional as F

        num_codebooks = self.config.audio_num_codebooks
        device = audio_logits.device

        # Apply temperature
        if self.temperature != 1.0:
            audio_logits = audio_logits / self.temperature

        # Apply top-k and top-p filtering per codebook
        if self.top_k > 0:
            for cb in range(num_codebooks):
                top_k_logits, top_k_indices = torch.topk(
                    audio_logits[cb], min(self.top_k, audio_logits.size(-1))
                )
                audio_logits[cb] = torch.full_like(audio_logits[cb], -float("inf"))
                audio_logits[cb].scatter_(0, top_k_indices, top_k_logits)

        # Sample tokens
        probs = F.softmax(audio_logits, dim=-1)
        next_audio_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Apply delay pattern logic
        if num_delay + 1 < num_codebooks:
            # Force BOS for not-yet-activated codebooks
            next_audio_tokens[num_delay + 1 :] = self.config.audio_stream_bos_id

        if num_remaining_delays is not None:
            # Force EOS for early positions in remaining delay phase
            eos_positions = num_codebooks - num_remaining_delays
            if eos_positions > 0:
                next_audio_tokens[:eos_positions] = self.config.audio_stream_eos_id

        return next_audio_tokens
