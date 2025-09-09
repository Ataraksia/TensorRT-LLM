# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
"""TensorRT-LLM implementation of Higgs Audio multimodal model."""

import os
from typing import Optional, List
import numpy as np
import torch
from boson_multimodal import *
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    GenerationMixin,
    PreTrainedModel,
    pipeline,
)
import tensorrt_llm
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.model_weights_loader import ModelWeightsLoader
from tensorrt_llm.models.modeling_utils import (
    DecoderLayerList,
    QuantConfig,
    DecoderModelForCausalLM,
    default_net,
)
from tensorrt_llm.module import Module, ModuleList
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.functional import Tensor, constant, unsqueeze, where
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

# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
"""Wrapper for audio tokenization."""

import inspect
import json
import math
import os
import sys
import tempfile
import warnings
from enum import Enum
from functools import lru_cache
from typing import Optional, Sequence, Tuple, Union

import librosa
import numpy as np
import s3fs
import torch
import torch.nn as nn
import torch.nn.functional as F
from boson_multimodal import HiggsAudioTokenizer

from huggingface_hub import snapshot_download
from omegaconf import OmegaConf


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

    @property
    def tps(self):
        return self._tps

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
    elif os.path.exists(os.path.join(tokenizer_path, "config.yaml")):
        # Old version omega config file
        config = OmegaConf.load(os.path.join(tokenizer_path, "config.yaml")).generator.config
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
    """Implement the delay pattern for audio generation."""
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
        self.hidden_size = self.confighidden_size

        # Shared attention layer
        self.attention = Attention(
            hidden_size=self.confighidden_size,
            num_attention_heads=self.confignum_attention_heads,
            num_kv_heads=self.confignum_key_value_heads,
            max_position_embeddings=self.configmax_position_embeddings,
            num_layers=self.confignum_hidden_layers,
            apply_query_key_layer_scaling=False,
            attention_mask_type=AttentionMaskType.causal,
            bias=False,
            dtype=self.configdtype,
            tp_group=None,
            tp_size=1,
            tp_rank=0,
            quant_mode=QuantMode(0),
            local_layer_idx=layer_idx,
        )

        # Text MLP
        self.mlp = MLP(
            hidden_size=self.confighidden_size,
            ffn_hidden_size=self.configaudio_ffn_intermediate_size,
            hidden_act=self.confighidden_act,
            dtype=self.configdtype,
            bias=False,
            tp_group=None,
            tp_size=1,
            quant_mode=QuantMode(0),
        )

        # Audio MLP (potentially smaller)
        self.audio_mlp = MLP(
            hidden_size=self.confighidden_size,
            ffn_hidden_size=self.configaudio_ffn_intermediate_size,
            hidden_act=self.confighidden_act,
            dtype=self.configdtype,
            bias=False,
            tp_group=None,
            tp_size=1,
            quant_mode=QuantMode(0),
        )

        # Layer norms
        self.input_layernorm = RmsNorm(
            normalized_shape=self.confighidden_size,
            eps=self.confignorm_epsilon,
            dtype=self.configdtype,
        )

        self.audio_input_layernorm = RmsNorm(
            normalized_shape=self.confighidden_size,
            eps=self.confignorm_epsilon,
            dtype=self.configdtype,
        )

        self.post_layernorm = RmsNorm(
            normalized_shape=self.confighidden_size,
            eps=self.confignorm_epsilon,
            dtype=self.configdtype,
        )

        self.audio_post_layernorm = RmsNorm(
            normalized_shape=self.confighidden_size,
            eps=self.confignorm_epsilon,
            dtype=self.configdtype,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        use_cache: bool = False,
        kv_cache_params: Optional[KeyValueCacheParams] = None,
        attention_params: Optional[AttentionParams] = None,
        audio_out_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass for dual FFN decoder layer."""

        residual = hidden_states

        # if use_cache:
        #     audio_out_mask = audio_out_mask[:, -hidden_states.shape[1] :]

        hidden_states = torch.where(
            audio_out_mask.unsqueeze(-1),
            self.audio_input_layernorm(hidden_states),
            self.input_layernorm(hidden_states),
        )

        # Shared attention layer
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

        # # Make whole graph in decode stage
        # if decode_stage and is_using_cuda_graph:
        #     assert is_decoding_audio_token is not None, (
        #         "is_decoding_audio_token should be present in the decoding stage."
        #     )
        #     if is_decoding_audio_token:
        #         hidden_states = self.audio_post_layernorm(hidden_states)
        #         hidden_states = self.audio_mlp(hidden_states)
        #     else:
        #         hidden_states = self.post_layernorm(hidden_states)
        #         hidden_states = self.mlp(hidden_states)
        #     residual = residual + hidden_states
        # else:
        text_hidden_states = self.post_layernorm(hidden_states[~audio_out_mask])
        audio_hidden_states = self.audio_post_layernorm(hidden_states[audio_out_mask])

        text_hidden_states = self.mlp(text_hidden_states)
        audio_hidden_states = self.audio_mlp(audio_hidden_states)

        residual[~audio_out_mask] += text_hidden_states
        residual[audio_out_mask] += audio_hidden_states

        hidden_states = residual

        if use_cache:
            return hidden_states, presents
        return hidden_states


class HiggsAudioTransformer(Module):
    """TensorRT-LLM transformer component for Higgs Audio model."""

    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.config = config

        self.vocab_embedding = Embedding(
            num_embeddings=self.configtext_vocab_size,
            embedding_dim=self.confighidden_size,
            dtype=self.configdtype,
        )

        # Audio codebook embeddings for audio generation
        self.audio_codebook_embeddings = Embedding(
            num_embeddings=self.configaudio_vocab_size,
            embedding_dim=self.confighidden_size,
            dtype=self.configdtype,
        )

        # Decoder layers - use dual FFN for all layers for simplicity
        self.layers = DecoderLayerList(HiggsAudioDualFFNDecoderLayer, config)

        # Final layer norm
        self.ln_f = RmsNorm(
            normalized_shape=self.confighidden_size,
            eps=self.confignorm_epsilon,
            dtype=self.configdtype,
        )

    def forward(
        self,
        hidden_states: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        use_cache: bool = False,
        attention_mask: Optional[Tensor] = None,
        kv_cache_params: Optional[KeyValueCacheParams] = None,
        attention_params: Optional[AttentionParams] = None,
        prompt_embedding_table: Optional[Tensor] = None,
        prompt_tasks: Optional[Tensor] = None,
        prompt_vocab_size: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass for Higgs Audio transformer with multimodal support."""

        audio_tokens_mask = input_ids > (self.config.text_vocab_size - 1)

        text_tokens = where(audio_tokens_mask, self.config.text_vocab_size - 1, input_ids)
        text_embeddings = self.vocab_embeddings(text_tokens)

        audio_tokens = where(audio_tokens_mask, input_ids - self.config.text_vocab_size, 0)
        codebook_shift = torch.arange(self.config.audio_num_codebooks) * self.audio_codebook_size
        audio_embeddings = self.audio_codebook_embeddings(
            audio_tokens + codebook_shift.unsqueeze(-1)
        )
        audio_embeddings = torch.sum(audio_embeddings, dim=1)

        hidden_states = where(unsqueeze(audio_tokens_mask, -1), audio_embeddings, text_embeddings)

        # audio_out_mask = input_ids == self.audio_in_token_idx
        # Decoder layers
        hidden_states = self.layers(
            hidden_states=hidden_states,
            use_cache=use_cache,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            audio_out_mask=audio_tokens_mask,
        )

        if use_cache:
            hidden_states, presents = hidden_states

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        if use_cache:
            return hidden_states, presents
        return hidden_states


# def higgs_audio_logits_processor(
#     req_id: int,
#     logits: torch.Tensor,
#     ids: List[List[int]],
#     stream_ptr: int,
#     client_id: Optional[int],
# ):
#     """Process logits to constrain generation to audio vocabulary.

#     Args:
#         req_id: Request ID
#         logits: Input logits tensor of shape [batch_size, beam_width, vocab_size]
#         ids: Previously generated token IDs
#         stream_ptr: CUDA stream pointer
#         client_id: Optional client ID
#     """
#     # Get vocab size from logits
#     vocab_size = logits.shape[-1]

#     # Only process if we have text tokens to mask out
#     if vocab_size > audio_vocab_size:
#         with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
#             # Create mask for text tokens (>= audio_vocab_size)
#             # Set their logits to negative infinity
#             logits[..., audio_vocab_size:] = float("-inf")

#             # Optional: Small boost to audio tokens to ensure they're preferred
#             logits[..., :audio_vocab_size] += 1.0


class HiggsAudioForCausalLM(DecoderModelForCausalLM):
    """TensorRT-LLM implementation of Higgs Audio multimodal model."""

    def __init__(self, config: HiggsAudioConfig):
        # Initialize the transformer component
        transformer = HiggsAudioTransformer(config)

        # text_lm_head = ColumnLinear(
        #     in_features=self.confighidden_size,
        #     out_features=self.configtext_vocab_size,  # Use full vocab_size to match padded audio_lm_head weights
        #     bias=False,
        #     dtype=self.configdtype,
        #     tp_group=None,
        #     tp_size=1,
        #     gather_output=True,
        # )

        audio_lm_head = ColumnLinear(
            in_features=self.confighidden_size,
            out_features=self.configaudio_vocab_size,  # Use full vocab_size to match padded audio_lm_head weights
            bias=False,
            dtype=self.configdtype,
            tp_group=None,
            tp_size=1,
            gather_output=True,
        )

        super().__init__(config, transformer, audio_lm_head)

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
            "lm_head": "audio_decoder_proj.audio_lm_head.weight",
        }
        loader = ModelWeightsLoader(hf_config_or_dir, custom_dict)

        model = cls(config)

        # tllm_weights = {}
        # for tllm_key, _ in model.named_parameters():
        #     sub_module = model
        #     for attr in tllm_key.split(".")[:-1]:
        #         print(attr)
        #         sub_module = getattr(sub_module, attr)
        #     if "audio" in tllm_key:
        #         sub_module_dic = sub_module.tllm_to_externel_key_dict
        #         sub_module_dic["audio_mlp"] = "audio_mlp"
        #         if "fc" in sub_module_dic.keys():
        #             sub_module_dic["fc"] = [
        #                 hf_keyword.replace("w1", "gate_proj") for hf_keyword in sub_module_dic["fc"]
        #             ]
        #             sub_module_dic["fc"] = [
        #                 hf_keyword.replace("w3", "up_proj") for hf_keyword in sub_module_dic["fc"]
        #             ]
        #         if "proj" in sub_module_dic.keys():
        #             sub_module_dic["proj"] = [
        #                 hf_keyword.replace("w2", "down_proj")
        #                 for hf_keyword in sub_module_dic["proj"]
        #             ]
        #         sub_module.tllm_to_externel_key_dict = sub_module_dic
        for tllm_key, _ in tqdm(model.named_parameters()):
            if "audio" in tllm_key:
                print(tllm_key)
                # tllm_weights.update(loader.load(tllm_key))
            else:
                print(tllm_key)
                # tllm_weights.update(loader.load(tllm_key))
        # loader.fill(tllm_weights)

        return model


class HiggsAudioTRTRunner:
    """TensorRT-LLM inference wrapper for HiggsAudio using ModelRunnerCpp."""

    def __init__(
        self,
        config: HiggsAudioConfig,
        engine_dir: str,
        tokenizer_dir: str,
        audio_tokenizer_dir: str,
    ) -> None:
        """Initialize the TensorRT-LLM runner for HiggsAudio."""

        from tensorrt_llm.runtime import ModelRunnerCpp

        self.config = config
        self.engine_dir = engine_dir
        self.tokenizer_dir = tokenizer_dir
        self.audio_tokenizer_dir = audio_tokenizer_dir

        self.temperature = 1.0
        self.top_k = 50
        self.top_p = 0.95
        self.num_beams = 1
        self.gpu_weights_percent = 0.5
        self.max_new_tokens = 512
        self.max_seq_len = 1500

        # Set up device
        self.gpu_device = torch.device("cuda", 0)
        torch.cuda.set_device(self.gpu_device)

        # Initialize runner components
        self.runner = None
        self.tokenizer = None

        # Load components
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_dir, trust_remote_code=True, use_fast=False
        )
        self.audio_tokenizer = AudioTokenizer(self.audio_tokenizer_dir, device=str(self.gpu_device))

        # logits_processor_map = {"higgs_audio_logits_processor": higgs_audio_logits_processor}

        self.runner = ModelRunnerCpp.from_dir(
            engine_dir=self.engine_dir,
            use_gpu_direct_storage=True,
            cuda_graph_mode=True,
            kv_cache_free_gpu_memory_fraction=self.gpu_weights_percent,
            # logits_processor_map=logits_processor_map,
        )

    def generate(
        self,
        input_text: str,
        input_audio: str = None,
        use_delay_pattern: bool = True,
        **generation_kwargs,
    ):
        """Generate audio from text input and reference audio (TTS with voice cloning).

        Args:
            input_text: The text prompt to convert to speech
            input_audio: Path to reference audio file for voice cloning
            use_delay_pattern: Whether to use delay pattern for RVQ generation

        Returns:
            Generated audio tensor suitable for Whisper transcription
        """
        if not self.runner:
            raise RuntimeError("Runner not initialized")

        # Process text input using the correct format for audio generation
        if input_audio is not None:
            # Load and transcribe reference audio for voice cloning
            model_id = "openai/whisper-large-v3-turbo"
            model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
            processor = AutoProcessor.from_pretrained(model_id)
            audio, _ = librosa.load(input_audio, sr=16000)
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                return_timestamps=True,
            )
            transcription = pipe(audio)["text"]

            # Format with reference audio (voice cloning) following Higgs Audio expected format
            # The format should include the reference audio transcription and then the target text
            formatted_text = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
                f"You are a helpful assistant that can generate speech from text. "
                f"Use the provided audio reference to match the speaker's voice characteristics.<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>{transcription}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|><|audio_bos|><|AUDIO|><|audio_eos|><|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>{input_text}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|><|audio_out_bos|>"
            )
        else:
            # Format without reference audio (default voice)
            # Simplified format for direct text-to-speech without voice cloning
            formatted_text = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
                f"You are a helpful assistant that can generate speech from text.<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>{input_text}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|><|audio_out_bos|>"
            )

        # Encode the formatted text to token IDs
        try:
            input_ids = self.tokenizer.encode(formatted_text, return_tensors="pt").squeeze(0)
            print(f"DEBUG: Encoded input_ids shape: {input_ids.shape}")
            print(f"DEBUG: Last few tokens: {input_ids[-10:].tolist()}")
        except Exception as e:
            raise RuntimeError(f"Failed to encode input text: {e}")

        if input_audio is not None and self.audio_tokenizer:
            print(f"DEBUG: Processing reference audio: {input_audio}")

            # Validate audio file exists
            if not os.path.exists(input_audio):
                raise FileNotFoundError(f"Reference audio file not found: {input_audio}")

            try:
                # Encode audio to RVQ codes
                audio_codes = self.audio_tokenizer.encode(input_audio, sr=24000)
                print(f"DEBUG: Encoded audio codes shape: {audio_codes.shape}")

            except Exception as e:
                print(f"ERROR: Failed to encode reference audio: {e}")
                raise RuntimeError(f"Audio encoding failed: {e}")

            # Apply delay pattern if requested and we have multiple codebooks
            if use_delay_pattern and len(audio_codes.shape) >= 2 and audio_codes.shape[0] > 1:
                # Add BOS and EOS tokens using correct token IDs
                bos_tokens = torch.full(
                    (audio_codes.shape[0], 1),
                    self.config.audio_stream_bos_id,
                    dtype=audio_codes.dtype,
                    device=audio_codes.device,
                )
                eos_tokens = torch.full(
                    (audio_codes.shape[0], 1),
                    self.config.audio_stream_eos_id,
                    dtype=audio_codes.dtype,
                    device=audio_codes.device,
                )
                # Concatenate: BOS + audio_codes + EOS
                audio_codes = torch.cat([bos_tokens, audio_codes, eos_tokens], dim=-1)

                # Apply delay pattern
                audio_codes = _build_delay_pattern_mask(
                    audio_codes.unsqueeze(0),  # Add batch dimension
                    bos_token_id=self.config.audio_bos_token_id,
                    pad_token_id=self.config.audio_eos_token_id,
                )

            # num_audio_tokens = audio_codes.shape[1]

            # # Create task IDs (0 for audio task) - should be a string for ModelRunnerCpp
            # prompt_tasks = "0"  # Single task ID as string

            # # Set vocab size for audio features
            # prompt_vocab_size = torch.tensor(
            #     [num_audio_tokens], dtype=torch.int32, device=self.gpu_device
            # )

        batch_indices, audio_indices = torch.where(input_ids == self.config.audio_in_token_idx)

        fake_prompt_id = torch.arange(
            self.config.text_vocab_size,
            self.config.text_vocab_size + audio_codes.shape[-1],
            device=self.gpu_device,
        )

        input_ids[batch_indices, audio_indices] = fake_prompt_id
        input_lengths = torch.tensor(input_ids.size(-1), dtype=torch.int32, device=self.gpu_device)
        print(input_ids)
        max_input_length = torch.max(input_lengths).item()
        max_new_tokens = min(self.max_new_tokens, self.max_seq_len - max_input_length)

        # Since we're configured for audio-only generation, always use audio mode
        print(
            f"DEBUG: Last token ID: {input_ids[-1].item()}, audio_out_bos_token_id: {self.config.audio_out_bos_token_id}"
        )

        # Run generation with or without audio features
        with torch.no_grad():
            print(f"DEBUG: Input sequence length: {len(input_ids)}")
            print(f"DEBUG: Max new tokens: {max_new_tokens}")

            # outputs = self.runner.generate(
            #     batch_input_ids=[input_ids],
            #     max_new_tokens=max_new_tokens,
            #     beam_width=self.num_beams,
            #     temperature=self.temperature,
            #     top_k=self.top_k,
            #     top_p=self.top_p,
            #     end_id=self.config.audio_eos_token_id,
            #     pad_id=self.configpad_token_id,
            #     prompt_table=audio_codes if audio_codes is not None else None,
            #     prompt_tasks=None,
            #     prompt_vocab_size=None,
            #     # logits_processor_names=["higgs_audio_logits_processor"],
            # )

        # # Process outputs - Convert generated token IDs to audio
        # def _extract_generated_tokens(outputs_tensor: torch.Tensor) -> torch.Tensor:
        #     """Extract 1D generated token ids (without prompt) from ModelRunnerCpp output.
        #     outputs_tensor shape: (batch, num_sequences, seq_len). Returns 1D tensor.
        #     """
        #     if isinstance(outputs_tensor, dict):
        #         outputs_tensor = outputs_tensor["output_ids"]
        #     if not torch.is_tensor(outputs_tensor):
        #         raise RuntimeError("Unexpected outputs type from runner.generate")
        #     # Take first batch, first sequence
        #     seq = outputs_tensor[0, 0].to(device=self.gpu_device)
        #     # Outputs are padded with end_id; truncate at first end_id occurrence
        #     if self.config.audio_eos_token_id is not None:
        #         eos_positions = (seq == self.config.audio_eos_token_id).nonzero(as_tuple=False)
        #         if eos_positions.numel() > 0:
        #             seq = seq[: eos_positions[0].item()]
        #     # Remove any pad_id just in case
        #     if self.configpad_token_id is not None:
        #         if seq.numel() > 0 and seq[-1].item() == self.configpad_token_id:
        #             last_non_pad = (seq != self.configpad_token_id).nonzero(as_tuple=False)
        #             if last_non_pad.numel() > 0:
        #                 seq = seq[: last_non_pad[-1].item() + 1]
        #             else:
        #                 seq = seq[:0]
        #     return seq

        # if outputs is not None:
        #     try:
        #         gen_seq = _extract_generated_tokens(outputs)
        #     except Exception as e:
        #         print(f"DEBUG: Failed to parse outputs: {e}")
        #         gen_seq = torch.empty(0, dtype=torch.int32, device=self.gpu_device)

        #     num_gen = int(gen_seq.numel())
        #     head = gen_seq[:10].tolist() if num_gen > 0 else []
        #     tail = gen_seq[-10:].tolist() if num_gen > 0 else []
        #     print(f"DEBUG: Generated {num_gen} tokens")
        #     print(f"DEBUG: First 10 generated tokens: {head}")
        #     print(f"DEBUG: Last 10 generated tokens: {tail}")

        #     # Check if we're generating text tokens instead of audio tokens
        #     if num_gen > 0:
        #         text_tokens = [t for t in head if t >= 128000]  # Text tokens are typically 128000+
        #         audio_range_tokens = [
        #             t for t in head if t < 10000
        #         ]  # Audio tokens should be < 10000
        #         print(f"DEBUG: Text tokens in first 10: {len(text_tokens)} ({text_tokens})")
        #         print(
        #             f"DEBUG: Audio range tokens in first 10: {len(audio_range_tokens)} ({audio_range_tokens})"
        #         )

        #         if len(text_tokens) > 0:
        #             print("⚠️  WARNING: Model is generating text tokens instead of audio tokens!")
        #             print(
        #                 "This suggests the model is not properly configured for audio generation."
        #             )

        #     if num_gen > 0:
        #         # With logits post processor, all generated tokens should already be in audio vocabulary range
        #         max_audio_token = self.config.audio_num_codebooks * (
        #             self.config.audio_codebook_size + 2
        #         )

        #         # Verify all tokens are in audio vocabulary range
        #         invalid_tokens = gen_seq >= max_audio_token
        #         if invalid_tokens.any():
        #             print(
        #                 f"WARNING: Found {invalid_tokens.sum()} tokens outside audio vocabulary range"
        #             )
        #             # Filter out invalid tokens
        #             audio_tokens = gen_seq[~invalid_tokens]
        #         else:
        #             audio_tokens = gen_seq

        #         # For RVQ format, we need tokens divisible by num_codebooks (8)
        #         num_cbs = self.config.audio_num_codebooks
        #         usable_count = (len(audio_tokens) // num_cbs) * num_cbs
        #         if usable_count < num_cbs:
        #             # If we don't have enough for even one timestep, pad to minimum
        #             usable_count = num_cbs
        #             padding_needed = num_cbs - len(audio_tokens)
        #             silence_token = 0  # Use 0 as silence
        #             padding = torch.full(
        #                 (padding_needed,),
        #                 silence_token,
        #                 dtype=audio_tokens.dtype,
        #                 device=audio_tokens.device,
        #             )
        #             audio_tokens = torch.cat([audio_tokens, padding])

        #         audio_tokens = audio_tokens[:usable_count]

        #         print(
        #             f"DEBUG: Using {len(audio_tokens)} audio tokens (multiple of {num_cbs}) from {num_gen} generated"
        #         )
        #         print(f"DEBUG: First 10 audio tokens: {audio_tokens[:10].tolist()}")
        #         print(f"DEBUG: Last 10 audio tokens: {audio_tokens[-10:].tolist()}")

        #         if len(audio_tokens) > 0:
        #             # Convert flat tokens to (num_codebooks, seq_len) codes
        #             try:
        #                 num_cbs = self.config.audio_num_codebooks
        #                 codebook_size = self.config.audio_codebook_size
        #                 tokens = audio_tokens
        #                 print(f"DEBUG: num_codebooks={num_cbs}, codebook_size={codebook_size}")
        #                 print(f"DEBUG: Token range: min={min(tokens)}, max={max(tokens)}")

        #                 # For Higgs Audio, tokens are typically organized sequentially
        #                 # We need to reshape them into (num_codebooks, seq_len) format
        #                 total_tokens = len(tokens)

        #                 # Calculate sequence length per codebook
        #                 if total_tokens % num_cbs == 0:
        #                     seq_len = total_tokens // num_cbs
        #                     print(
        #                         f"DEBUG: Reshaping {total_tokens} tokens into ({num_cbs}, {seq_len})"
        #                     )

        #                     # Reshape tokens into codebook format
        #                     audio_codes = tokens.view(num_cbs, seq_len)

        #                     # Clamp values to valid codebook range [0, codebook_size)
        #                     audio_codes = torch.clamp(audio_codes, 0, codebook_size - 1)

        #                     print(f"DEBUG: Audio codes shape: {audio_codes.shape}")
        #                     print(f"DEBUG: Audio codes sample: {audio_codes[:, : min(5, seq_len)]}")
        #                 else:
        #                     # Fallback: try to distribute tokens across codebooks
        #                     print(
        #                         f"DEBUG: Token count {total_tokens} not divisible by {num_cbs}, using fallback method"
        #                     )

        #                     # Calculate approximate sequence length
        #                     seq_len = (total_tokens + num_cbs - 1) // num_cbs

        #                     # Create padded tensor
        #                     audio_codes = torch.zeros(
        #                         (num_cbs, seq_len), dtype=tokens.dtype, device=tokens.device
        #                     )

        #                     # Fill in available tokens
        #                     for i in range(num_cbs):
        #                         start_idx = i * seq_len
        #                         end_idx = min(start_idx + seq_len, total_tokens)
        #                         if start_idx < total_tokens:
        #                             actual_len = end_idx - start_idx
        #                             audio_codes[i, :actual_len] = tokens[start_idx:end_idx]

        #                     # Clamp values to valid codebook range
        #                     audio_codes = torch.clamp(audio_codes, 0, codebook_size - 1)

        #                     print(f"DEBUG: Fallback audio codes shape: {audio_codes.shape}")
        #                     print(
        #                         f"DEBUG: Fallback audio codes sample: {audio_codes[:, : min(5, seq_len)]}"
        #                     )

        #                 # Apply delay pattern reversion for better audio quality
        #                 if (
        #                     use_delay_pattern
        #                     and audio_codes.shape[0] > 1
        #                     and audio_codes.shape[1] > 0
        #                 ):
        #                     try:
        #                         print("DEBUG: Reverting delay pattern...")
        #                         original_shape = audio_codes.shape
        #                         audio_codes = revert_delay_pattern(audio_codes)
        #                         print(
        #                             f"DEBUG: After delay pattern revert: {original_shape} -> {audio_codes.shape}"
        #                         )
        #                     except Exception as e:
        #                         print(f"DEBUG: Delay pattern revert failed: {e}")
        #                         print("DEBUG: Using raw codes without delay pattern revert")
        #                 else:
        #                     print("DEBUG: Skipping delay pattern reversion (not applicable)")
        #                 # Decode
        #                 print("DEBUG: Decoding audio codes...")
        #                 decoded_audio = self.audio_tokenizer.decode(
        #                     audio_codes, return_cuda_tensor=False
        #                 )
        #                 # Handle return as (waveform, sampling_rate)
        #                 if isinstance(decoded_audio, tuple) and len(decoded_audio) == 2:
        #                     waveform, sr = decoded_audio
        #                 else:
        #                     waveform, sr = (
        #                         decoded_audio,
        #                         getattr(self.audio_tokenizer, "sampling_rate", 16000),
        #                     )
        #                 if isinstance(waveform, torch.Tensor):
        #                     waveform = waveform.detach().cpu().numpy()
        #                 # Resample to 16kHz for Whisper large-v3-turbo compatibility
        #                 try:
        #                     if sr != 16000 and isinstance(waveform, np.ndarray):
        #                         waveform = librosa.resample(
        #                             waveform.astype(np.float32), orig_sr=sr, target_sr=16000
        #                         )
        #                         sr = 16000
        #                 except Exception:
        #                     pass
        #                 return waveform.astype(np.float32)
        #             except Exception as e:
        #                 print(f"DEBUG: Converting tokens to audio codes failed: {e}")
        #         else:
        #             print("DEBUG: No audio tokens found after filtering")

        # # Return silence if generation failed or produced nothing useful
        # print("DEBUG: No valid output generated, returning silence")
        # return np.zeros(16000, dtype=np.float32)  # 1 second of silence at 16kHz


import torch
import torch.nn as nn
from boson_multimodal.configuration_higgs_audio import HiggsAudioConfig
from transformers import GenerationMixin, PreTrainedModel


class SimpleHiggsAudioModel(PreTrainedModel, GenerationMixin):
    """A simplified HiggsAudio model that only uses the audio_codebook_embeddings layer.
    This model loads the audio codebook embeddings from a pretrained HiggsAudio model
    and applies them to audio tokens.
    """

    config_class = HiggsAudioConfig

    def __init__(self, config: HiggsAudioConfig):
        super().__init__(config)
        self.config = config

        # Store audio codebook size for convenience
        self.audio_codebook_size = config.audio_codebook_size

        # Audio codebook embeddings - this is the main layer we're using
        self.audio_codebook_embeddings = nn.Embedding(
            config.audio_num_codebooks * self.audio_codebook_size, config.text_config.hidden_size
        )

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        audio_in_ids: torch.LongTensor | None = None,
        return_dict: bool = True,
    ):
        """Forward pass that processes audio tokens through the codebook embeddings.

        Args:
            input_ids: Text input ids (not used in this simple model)
            audio_in_ids: Audio token ids of shape (batch_size, num_codebooks, seq_len)
            return_dict: Whether to return a dict or tuple

        Returns:
            Audio embeddings of shape (batch_size, seq_len, hidden_size)
        """
        if audio_in_ids is None:
            raise ValueError("audio_in_ids must be provided")

        # audio_in_ids shape: (batch_size, num_codebooks, seq_len)
        batch_size, num_codebooks, seq_len = audio_in_ids.shape

        # Create codebook shifts: [0, 1024, 2048, ...] for each codebook
        codebook_shift = (
            torch.arange(self.config.audio_num_codebooks, device=audio_in_ids.device)
            * self.audio_codebook_size
        )

        # Add shifts to audio tokens to get correct embedding indices
        # codebook_shift shape: (num_codebooks,) -> (num_codebooks, 1)
        # audio_in_ids shape: (batch_size, num_codebooks, seq_len)
        shifted_audio_tokens = audio_in_ids + codebook_shift.unsqueeze(-1)

        # Look up embeddings for each codebook
        # Shape: (batch_size, num_codebooks, seq_len, hidden_size)
        audio_embeddings = self.audio_codebook_embeddings(shifted_audio_tokens)

        # Sum embeddings across codebooks
        # Shape: (batch_size, seq_len, hidden_size)
        audio_embeddings = torch.sum(audio_embeddings, dim=1)

        if return_dict:
            return {"audio_embeddings": audio_embeddings}
        else:
            return audio_embeddings

    @classmethod
    def load_from_pretrained_higgs_audio(cls, model_name_or_path: str, **kwargs):
        """Load only the audio_codebook_embeddings layer from a pretrained HiggsAudio model.

        Args:
            model_name_or_path: Path to the pretrained HiggsAudio model
            **kwargs: Additional arguments for model loading

        Returns:
            SimpleHiggsAudioModel with loaded audio_codebook_embeddings weights
        """
        from boson_multimodal import HiggsAudioModel

        # Add attention implementation to avoid issues
        kwargs.setdefault("attn_implementation", "eager")

        # Load the full pretrained model
        print(f"Loading pretrained HiggsAudio model from {model_name_or_path}...")
        full_model = HiggsAudioModel.from_pretrained(model_name_or_path, **kwargs)

        # Create our simple model with the same config
        simple_model = cls(full_model.config)

        # Copy the audio_codebook_embeddings weights
        if hasattr(full_model, "audio_codebook_embeddings"):
            print("Copying audio_codebook_embeddings weights...")
            simple_model.audio_codebook_embeddings.load_state_dict(
                full_model.audio_codebook_embeddings.state_dict()
            )
        else:
            print("Warning: audio_codebook_embeddings not found in pretrained model")

        return simple_model


# Example usage
if __name__ == "__main__":
    # Create a simple config for testing
    config = HiggsAudioConfig(
        audio_num_codebooks=4,  # Reduced for testing
        audio_codebook_size=1024,
    )

    # Create the model
    model = SimpleHiggsAudioModel(config)

    # Create some dummy audio tokens
    batch_size = 2
    seq_len = 10
    audio_tokens = torch.randint(
        0, config.audio_codebook_size, (batch_size, config.audio_num_codebooks, seq_len)
    )

    print(f"Input audio tokens shape: {audio_tokens.shape}")

    # Forward pass
    with torch.no_grad():
        output = model(audio_in_ids=audio_tokens)

    print(f"Output embeddings shape: {output['audio_embeddings'].shape}")
    print("Model forward pass completed successfully!")

    model = SimpleHiggsAudioModel.load_from_pretrained_higgs_audio(
        "bosonai/higgs-audio-v2-generation-3B-base"
    )
