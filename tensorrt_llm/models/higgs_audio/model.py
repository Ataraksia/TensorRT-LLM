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
        """Embed the audio ids"""
        num_codebooks = self.config.audio_num_codebooks
        codebook_size = self.config.audio_codebook_size
        codebook_shift = (arange(0, num_codebooks, "int32") * codebook_size).unsqueeze(-1)
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

        This enforces the delay pattern described in the Transformers implementation by:
        - Prefilling later codebooks with BOS until they are activated.
        - After an EOS is generated for some codebook, pre-filling earlier codebooks with EOS
          for a number of subsequent steps equal to the remaining codebooks in the frame.
        - Finally, when all remaining delays are consumed, force the stream EOS to stop generation.
        """

        state = self._get_or_create_state(req_id)

        # token_ids is List[beam][generated_tokens]; use the first beam
        past = token_ids[0] if token_ids and token_ids[0] is not None else []
        tokens_generated = len(past)

        # Determine which codebook position we're currently generating for (cyclic)
        current_codebook_pos = tokens_generated % self.audio_num_codebooks

        forced_token_id = None

        # Phase 1: Apply initial delays (BOS tokens for delayed positions)
        if state["num_delay"] + 1 < self.audio_num_codebooks:
            if current_codebook_pos > state["num_delay"]:
                forced_token_id = self.audio_stream_bos_id
            # Only increment delay counter once per complete cycle
            if current_codebook_pos == self.audio_num_codebooks - 1:
                state["num_delay"] += 1

        # Phase 2: Handle remaining delays after encountering an EOS in any codebook
        if state["num_remaining_delays"] is not None:
            # Apply EOS tokens to positions that should end early
            if current_codebook_pos < (self.audio_num_codebooks - state["num_remaining_delays"]):
                forced_token_id = self.audio_stream_eos_id

            # Decrement remaining delays once per cycle
            if current_codebook_pos == self.audio_num_codebooks - 1:
                state["num_remaining_delays"] -= 1
        else:
            # Initialize remaining delays by finding EOS tokens in current sequence
            if tokens_generated > 0:
                past_tensor = torch.tensor(past, device=logits.device)
                eos_positions = (past_tensor == self.audio_stream_eos_id).nonzero(as_tuple=False)
                if eos_positions.numel() > 0:
                    last_eos_idx = eos_positions[-1, 0].item()
                    last_eos_pos_in_frame = last_eos_idx % self.audio_num_codebooks
                    state["num_remaining_delays"] = (
                        self.audio_num_codebooks - last_eos_pos_in_frame - 1
                    )

                    # Force EOS for positions that should end in the current frame
                    if current_codebook_pos < last_eos_pos_in_frame:
                        forced_token_id = self.audio_stream_eos_id

        # Phase 3: End generation when all delays are complete
        if state["num_remaining_delays"] is not None and state["num_remaining_delays"] <= 0:
            # In TRT-LLM audio-only generation, terminate by forcing the audio stream EOS
            forced_token_id = self.audio_stream_eos_id
            # Reset state for potential future use
            state["num_delay"] = 0
            state["num_remaining_delays"] = None

        # Apply the forced token by manipulating logits in-place (last dim is vocab)
        if forced_token_id is not None:
            logits[...] = -1.0e20
            if 0 <= forced_token_id < self.audio_vocab_size:
                logits[..., forced_token_id] = 0.0


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
        """Generate audio from text input and reference audio (TTS with voice cloning).

        Args:
            input_text: The text prompt to convert to speech
            input_audio: Path to reference audio file for voice cloning

        Returns:
            Generated audio tensor suitable for Whisper transcription"""

        from tensorrt_llm.models.higgs_audio.serve import create_audio_chunk

        text_input = (
            f"{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|><|audio_out_bos|>"
        )

        text_input_ids = self.tokenizer.encode(text_input, return_tensors="pt").squeeze(0).cuda()
        audio_input_ids = torch.as_tensor([self.config.audio_stream_bos_id]).cuda()
        input_ids = torch.cat(
            [self.saved_input_ids, text_input_ids, audio_input_ids],
            dim=0,
        ).cuda()
        input_lengths = torch.tensor(
            [input_ids.size(-1)], device=self.gpu_device, dtype=torch.int32
        )
        max_input_length = torch.max(input_lengths).item()
        max_new_tokens = min(1024, self.max_seq_len - max_input_length)
        with torch.no_grad():
            outputs = self.runner.generate(
                batch_input_ids=[input_ids],
                max_new_tokens=max_new_tokens,
                end_id=self.config.audio_stream_eos_id,
                stop_words_list=[[[self.config.audio_stream_eos_id]]],
                temperature=float(self.temperature),
                top_k=int(self.top_k),
                top_p=float(self.top_p),
                logits_processor_names=["higgs_audio_delay_pattern"],
            )

        # Extract and process audio tokens with proper delay pattern handling
        try:
            vq_code = self._extract_and_process_audio_tokens(outputs[0, 0])
            print(f"Extracted audio tokens shape: {vq_code.shape}")

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

    def _extract_and_process_audio_tokens(self, generated_tokens):
        """Extract and process audio tokens with proper delay pattern handling."""
        # Find the audio stream BOS token (this is where audio generation starts)
        bos_mask = generated_tokens == self.config.audio_stream_bos_id
        if not bos_mask.any():
            raise ValueError("No audio_stream_bos_id found in generated tokens")

        # Find first BOS token position
        start_idx = torch.argmax(bos_mask.float()).item()

        # Find EOS token position (end of audio generation)
        eos_mask = generated_tokens[start_idx:] == self.config.audio_stream_eos_id
        if eos_mask.any():
            eos_idx = torch.argmax(eos_mask.float()).item() + start_idx
            audio_tokens = generated_tokens[start_idx:eos_idx]
        else:
            # No EOS found, take all remaining tokens
            audio_tokens = generated_tokens[start_idx:]

        print(f"Raw audio tokens length: {len(audio_tokens)}")

        # Ensure we have enough tokens for all codebooks
        num_codebooks = self.config.audio_num_codebooks
        if len(audio_tokens) < num_codebooks:
            raise ValueError(f"Not enough audio tokens: {len(audio_tokens)} < {num_codebooks}")

        # Trim to make divisible by num_codebooks
        trim_len = (len(audio_tokens) // num_codebooks) * num_codebooks
        audio_tokens = audio_tokens[:trim_len]

        if trim_len == 0:
            raise ValueError("No valid audio tokens after trimming")

        # Reshape to (num_codebooks, seq_len) for delay pattern processing
        seq_len = trim_len // num_codebooks
        audio_tokens = audio_tokens.view(num_codebooks, seq_len)

        print(f"Reshaped audio tokens: {audio_tokens.shape}")

        # Apply delay pattern reversion
        vq_code = revert_delay_pattern(audio_tokens)

        # Clip to valid codebook range
        vq_code = audio_tokens.clip(0, self.config.audio_codebook_size - 1)

        # Remove BOS/EOS tokens if present
        if vq_code.shape[1] > 0:
            # Check if first column is all BOS tokens
            if torch.all(vq_code[:, 0] == self.config.audio_stream_bos_id):
                vq_code = vq_code[:, 1:]

            # Check if last column is all EOS tokens
            if vq_code.shape[1] > 0 and torch.all(
                vq_code[:, -1] == self.config.audio_stream_eos_id
            ):
                vq_code = vq_code[:, :-1]

        return vq_code
