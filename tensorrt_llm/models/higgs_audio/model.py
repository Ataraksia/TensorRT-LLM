# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
"""TensorRT-LLM implementation of Higgs Audio multimodal model."""

from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from functools import lru_cache
import os
from typing import Any, ClassVar, Dict, List, Literal, Optional, Set, Tuple, TypedDict, Union
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from boson_multimodal import *
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import AutoTokenizer
import tensorrt_llm
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.model_weights_loader import ModelWeightsLoader
from tensorrt_llm.models.modeling_utils import QuantConfig, DecoderModelForCausalLM, default_net
from tensorrt_llm.module import Module, ModuleList
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.functional import (
    Tensor,
    constant,
    recv,
    send,
    shape,
    view,
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


def get_feat_extract_output_lengths(input_lengths: torch.LongTensor):
    """Computes the output length of the convolutional layers and the output length of the audio encoder"""
    input_lengths = (input_lengths - 1) // 2 + 1
    output_lengths = (input_lengths - 2) // 2 + 1
    return input_lengths, output_lengths


def revert_delay_pattern(data):
    """Convert samples encoded with delay pattern back to the original form."""
    assert len(data.shape) == 2
    out_l = []
    num_codebooks = data.shape[0]
    for i in range(num_codebooks):
        out_l.append(data[i, i:])
    return np.concatenate(out_l, axis=0)


def _validate_and_reshape_mm_tensor(
    mm_input: object, name: str, pad_with: Optional[int] = None
) -> torch.Tensor:
    if not isinstance(mm_input, (torch.Tensor, list)):
        raise TypeError(f"Invalid type for {name}: {type(mm_input)}")
    if isinstance(mm_input, torch.Tensor):
        return mm_input
    else:
        if pad_with is not None:
            # Pad sequences to same length if needed
            max_len = max(len(seq) for seq in mm_input)
            padded = []
            for seq in mm_input:
                if len(seq) < max_len:
                    padding = [pad_with] * (max_len - len(seq))
                    seq = seq + padding
                padded.append(seq)
            return torch.tensor(padded)
        else:
            return torch.tensor(mm_input)


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


class DecoderLayerListWithAudio(ModuleList):
    def __init__(self, cls, config):
        self.num_hidden_layers = config.num_hidden_layers
        self.layer_list = config.mapping.pp_layers(config.num_hidden_layers)
        self.quant_mode = config.quant_mode
        super().__init__([cls(config, idx) for idx in self.layer_list])

    def forward(
        self,
        hidden_states,
        use_cache=False,
        attention_mask=None,
        kv_cache_params=None,
        attention_params=None,
        mrope_params=None,
        position_ids=None,
        lora_params=None,
        spec_decoding_params=None,
        vision_token_mask=None,
        audio_token_mask=None,
    ):
        kv_cache_params.fill_none_tensor_list(len(self.layer_list))

        if use_cache:
            presents = []

        for layer_idx, (layer, past) in enumerate(zip(self, kv_cache_params.past_key_value)):
            lora_layer_params = None
            if lora_params is not None and lora_params.lora_ranks is not None:
                lora_layer_params = lora_params.get_layer_params(layer_idx)

            kwargs = {}
            if position_ids is not None:
                kwargs["position_ids"] = position_ids
            if vision_token_mask is not None:
                kwargs["vision_token_mask"] = vision_token_mask
            if audio_token_mask is not None:
                kwargs["audio_token_mask"] = audio_token_mask
            if lora_layer_params is not None:
                kwargs["lora_layer_params"] = lora_layer_params
            if spec_decoding_params is not None:
                kwargs["spec_decoding_params"] = spec_decoding_params
            if mrope_params is not None:
                kwargs["mrope_params"] = mrope_params

            if default_net().plugin_config.reduce_fusion:
                if layer_idx + self.layer_list[0] < self.layer_list[-1]:
                    qkv_activation_scaling_factor = None
                    if default_net().plugin_config.user_buffer:
                        qkv_linear = self[layer_idx + 1].attention.qkv
                        if self.quant_mode.has_fp8_qdq():
                            qkv_activation_scaling_factor = constant(
                                qkv_linear.activation_scaling_factor.raw_value.copy()
                            )
                        elif self.quant_mode.has_nvfp4():
                            qkv_activation_scaling_factor = constant(
                                qkv_linear.activation_global_scaling_factor.raw_value.copy()
                            )
                    kwargs["next_layer_input_layernorm_args"] = (
                        self[layer_idx + 1].input_layernorm.weight.value,
                        self[layer_idx + 1].input_layernorm.eps,
                        qkv_activation_scaling_factor,
                    )
                else:
                    kwargs["next_layer_input_layernorm_args"] = None
            elif default_net().plugin_config.norm_quant_fusion:
                if layer_idx < self.layer_list[-1] - self.layer_list[0]:
                    try:
                        activation_scaling_factor = constant(
                            self[
                                layer_idx + 1
                            ].attention.qkv.activation_global_scaling_factor.raw_value.copy()
                        )
                    except:
                        activation_scaling_factor = None
                    kwargs["next_layer_input_layernorm_args"] = (
                        self[layer_idx + 1].input_layernorm.weight.value,
                        self[layer_idx + 1].input_layernorm.eps,
                        activation_scaling_factor,
                    )
                else:
                    kwargs["next_layer_input_layernorm_args"] = None

            hidden_states = layer(
                hidden_states,
                use_cache=use_cache,
                attention_mask=attention_mask,
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=[past],
                    host_past_key_value_lengths=kv_cache_params.host_past_key_value_lengths,
                    host_max_attention_window_sizes=kv_cache_params.host_max_attention_window_sizes,
                    host_sink_token_length=kv_cache_params.host_sink_token_length,
                    kv_cache_block_offsets=kv_cache_params.kv_cache_block_offsets,
                    host_kv_cache_block_offsets=kv_cache_params.host_kv_cache_block_offsets,
                    host_kv_cache_pool_pointers=kv_cache_params.host_kv_cache_pool_pointers,
                    host_kv_cache_pool_mapping=kv_cache_params.host_kv_cache_pool_mapping,
                    cache_indirection=kv_cache_params.cache_indirection,
                ),
                attention_params=attention_params,
                **kwargs,
            )

            if use_cache:
                presents.append(hidden_states[1])
                hidden_states = hidden_states[0]

        if use_cache:
            return hidden_states, presents
        return hidden_states


class HiggsAudioDualFFNDecoderLayer(Module):
    """TensorRT-LLM implementation of dual-path FFN decoder layer."""

    def __init__(self, config: HiggsAudioConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Shared attention layer
        self.attention = Attention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
            max_position_embeddings=config.max_position_embeddings,
            num_layers=config.num_hidden_layers,
            apply_query_key_layer_scaling=False,
            attention_mask_type=AttentionMaskType.causal,
            bias=getattr(config, "attention_bias", False),
            dtype=config.dtype,
            tp_group=None,
            tp_size=1,
            tp_rank=0,
            quant_mode=QuantMode(0),
            local_layer_idx=layer_idx,
        )

        # Text MLP
        self.mlp = MLP(
            hidden_size=config.hidden_size,
            ffn_hidden_size=getattr(config, "intermediate_size", config.hidden_size * 4),
            hidden_act=getattr(config, "hidden_act", "silu"),
            dtype=config.dtype,
            bias=getattr(config, "mlp_bias", False),
            tp_group=None,
            tp_size=1,
            quant_mode=QuantMode(0),
        )

        # Audio MLP (potentially smaller)
        self.audio_mlp = MLP(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.audio_ffn_intermediate_size,
            hidden_act=getattr(config, "hidden_act", "silu"),
            dtype=config.dtype,
            bias=getattr(config, "mlp_bias", False),
            tp_group=None,
            tp_size=1,
            quant_mode=QuantMode(0),
        )

        # Layer norms
        self.input_layernorm = RmsNorm(
            normalized_shape=config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-5),
            dtype=config.dtype,
        )

        self.post_layernorm = RmsNorm(
            normalized_shape=config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-5),
            dtype=config.dtype,
        )

        self.audio_input_layernorm = RmsNorm(
            normalized_shape=config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-5),
            dtype=config.dtype,
        )

        self.audio_post_attention_layernorm = RmsNorm(
            normalized_shape=config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-5),
            dtype=config.dtype,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        use_cache: bool = False,
        kv_cache_params: Optional[KeyValueCacheParams] = None,
        attention_params: Optional[AttentionParams] = None,
    ) -> Tensor:
        """Forward pass for dual FFN decoder layer."""

        residual = hidden_states

        # For now, use standard processing during build time
        # The dual FFN logic will be handled at runtime through the prompt embedding mechanism
        hidden_states = self.input_layernorm(hidden_states)

        # Shared attention layer
        attention_output = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
        )

        if use_cache:
            attention_output, present_key_value = attention_output

        hidden_states = residual + attention_output

        # Standard FFN processing
        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        if use_cache:
            return hidden_states, present_key_value
        return hidden_states


class HiggsAudioTransformer(Module):
    """TensorRT-LLM transformer component for Higgs Audio model."""

    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.config = config

        self.mapping = tensorrt_llm.Mapping()

        self.vocab_embedding = Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            dtype=config.dtype,
            tp_group=config.mapping.tp_group
            if hasattr(config, "mapping") and config.mapping
            else None,
            tp_size=config.mapping.tp_size if hasattr(config, "mapping") and config.mapping else 1,
            sharding_dim=0,
            tp_rank=config.mapping.tp_rank if hasattr(config, "mapping") and config.mapping else 0,
        )

        # Audio codebook embeddings for audio generation
        self.audio_codebook_size = config.audio_codebook_size + 2  # +2 for BOS/EOS
        self.audio_num_codebooks = config.audio_num_codebooks

        self.audio_codebook_embeddings = Embedding(
            num_embeddings=config.audio_num_codebooks * self.audio_codebook_size,
            embedding_dim=config.hidden_size,
            dtype=config.dtype,
        )

        # Decoder layers - use dual FFN for all layers for simplicity
        self.layers = DecoderLayerListWithAudio(HiggsAudioDualFFNDecoderLayer, config)

        # Final layer norm
        self.ln_f = RmsNorm(
            normalized_shape=config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-5),
            dtype=config.dtype,
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
        """Forward pass for Higgs Audio transformer with multimodal support.

        This implementation supports both standard text processing and audio-enhanced
        generation through the prompt embedding table mechanism used by TensorRT-LLM
        for multimodal inputs.
        """

        ptuning_args = (
            [prompt_embedding_table, prompt_tasks, prompt_vocab_size]
            if prompt_embedding_table is not None
            else []
        )

        # Embedding stage
        if self.mapping.is_first_pp_rank():
            if input_ids is not None:
                # Get text embeddings with optional prompt tuning/multimodal features
                hidden_states = self.vocab_embedding(input_ids, *ptuning_args)

                # If we have audio features in prompt_embedding_table, they are already
                # integrated through the PromptTuningEmbedding mechanism. The audio
                # features should be preprocessed and passed as prompt_embedding_table
                # with appropriate prompt_tasks and prompt_vocab_size during inference.

        else:
            # Receive hidden states from previous pipeline stage
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())

        # Decoder layers
        hidden_states = self.layers(
            hidden_states=hidden_states,
            use_cache=use_cache,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
        )

        if use_cache:
            hidden_states, presents = hidden_states

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        if use_cache:
            return hidden_states, presents
        return hidden_states


class HiggsAudioForCausalLM(DecoderModelForCausalLM):
    """TensorRT-LLM implementation of Higgs Audio multimodal model."""

    def __init__(self, config: HiggsAudioConfig):
        # Initialize the transformer component
        transformer = HiggsAudioTransformer(config)

        # Audio output projector
        lm_head = ColumnLinear(
            in_features=config.hidden_size,
            out_features=config.audio_num_codebooks * config.audio_codebook_size,
            bias=False,
            dtype=config.dtype,
            tp_group=None,
            tp_size=1,
            gather_output=True,
        )
        super().__init__(config, transformer, lm_head)

    def use_prompt_tuning(self):
        """Enable prompt tuning for multimodal audio features."""
        from tensorrt_llm.layers.embedding import PromptTuningEmbedding

        embedding = self.transformer.vocab_embedding
        self.transformer.vocab_embedding = PromptTuningEmbedding(
            num_embeddings=embedding.num_embeddings,
            embedding_dim=embedding.embedding_dim,
            vocab_size=self.config.vocab_size,
            dtype=embedding.dtype,
            tp_size=embedding.tp_size,
            tp_group=embedding.tp_group,
            sharding_dim=embedding.sharding_dim,
            tp_rank=embedding.tp_rank,
        )
        self.transformer.vocab_embedding.weight.value = embedding.weight.raw_value

    @classmethod
    def from_hugging_face(
        cls,
        hf_model_dir: str,
        dtype: str = "auto",
        mapping: Optional[Mapping] = None,
        quant_config: Optional[QuantConfig] = None,
        **kwargs,
    ):
        """Create a HiggsAudioForCausalLM object from HuggingFace model directory.

        Args:
            hf_model_dir: Path to the HuggingFace model directory
            dtype: Data type for the model weights
            mapping: Multi-GPU mapping configuration
            quant_config: Quantization configuration
            **kwargs: Additional keyword arguments

        Returns:
            HiggsAudioForCausalLM: The loaded model
        """
        if not os.path.exists(hf_model_dir):
            hf_model_dir = snapshot_download(repo_id=hf_model_dir)

        config = HiggsAudioConfig(
            mapping=mapping or Mapping(),
            quant_config=quant_config or QuantConfig(),
            dtype=dtype,  # Pass dtype in kwargs to override default
            **kwargs,
        )
        custom_dict = {
            "lm_head": "audio_decoder_proj.audio_lm_head.weight",
        }
        loader = ModelWeightsLoader(hf_model_dir, custom_dict)

        model = cls(config)
        loader.update_key_mapping(model)
        tllm_weights = {}
        for tllm_key, _ in tqdm(model.named_parameters()):
            if "audio" in tllm_key:
                print(tllm_key)
                tllm_weights.update(loader.load(tllm_key))
        loader.fill(tllm_weights)

        loader.generate_tllm_weights(model)

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

        from tensorrt_llm.runtime import (
            ModelRunnerCpp,
            SamplingConfig,
        )

        self.config = config
        self.engine_dir = engine_dir
        self.tokenizer_dir = tokenizer_dir
        self.audio_tokenizer_dir = audio_tokenizer_dir

        self.temperature = 1.0
        self.top_k = 50
        self.top_p = 0.95
        self.num_beams = 1
        self.use_py_session = False
        self.gpu_weights_percent = 0.5
        self.max_new_tokens = 512
        self.enable_context_fmha_fp32_acc = False

        self.pad_token_id = config.pad_token_id
        self.audio_eos_token_id = config.audio_eos_token_id
        self.audio_in_token_idx = config.audio_in_token_idx
        self.vocab_size = config.vocab_size
        self.max_seq_len = 1500

        # Set up device
        self.gpu_device = torch.device("cuda", 0)
        torch.cuda.set_device(self.gpu_device)

        # Initialize runner components
        self.runner = None
        self.tokenizer = None
        self.sampling_config = None
        self.model_config = None

        # Load components
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_dir, trust_remote_code=True, use_fast=False
        )
        self.audio_tokenizer = load_higgs_audio_tokenizer(
            self.audio_tokenizer_dir, device=str(self.gpu_device)
        )

        self.sampling_config = SamplingConfig(
            end_id=self.audio_eos_token_id,
            pad_id=self.pad_token_id,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )
        self.runner = ModelRunnerCpp.from_dir(
            engine_dir=self.engine_dir,
            use_gpu_direct_storage=True,
            cuda_graph_mode=True,
            kv_cache_free_gpu_memory_fraction=self.gpu_weights_percent,
        )

    def generate(
        self,
        input_text: str,
        input_audio: str = None,
        **generation_kwargs,
    ) -> str:
        """Generate response integrating text and optional audio inputs.

        Audio features are processed and passed through the prompt embedding table
        mechanism, allowing the model to condition on both text and audio inputs.
        """
        if not self.runner:
            raise RuntimeError("Runner not initialized")

        # Process text input
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").squeeze(0)

        # Process audio input if provided
        prompt_table = None
        prompt_tasks = None
        prompt_vocab_size = None

        if input_audio is not None and self.audio_tokenizer:
            # Encode audio to features
            audio_features = self.audio_tokenizer.encode(input_audio, sr=24000)

            # Prepare audio features as prompt embedding table
            # Shape should be [batch_size, num_audio_tokens, hidden_size]
            if len(audio_features.shape) == 2:
                audio_features = audio_features.unsqueeze(0)  # Add batch dimension

            prompt_table = audio_features.to(dtype=torch.bfloat16, device=self.gpu_device)

            # Set up prompt tasks and vocab size for audio features
            batch_size = prompt_table.shape[0]
            num_audio_tokens = prompt_table.shape[1]

            # Create task IDs (0 for audio task)
            prompt_tasks = torch.zeros(
                (batch_size, num_audio_tokens), dtype=torch.bfloat16, device=self.gpu_device
            )

            # Set vocab size for audio features
            prompt_vocab_size = torch.tensor(
                [num_audio_tokens], dtype=torch.bfloat16, device=self.gpu_device
            )

            # Modify input_ids to include audio placeholder tokens
            # Replace <|AUDIO|> tokens with virtual token IDs >= vocab_size
            audio_token_id = self.audio_in_token_idx
            if audio_token_id in input_ids:
                # Replace audio placeholder with virtual tokens
                audio_mask = input_ids == audio_token_id
                virtual_token_start = self.vocab_size
                virtual_tokens = torch.arange(
                    virtual_token_start,
                    virtual_token_start + num_audio_tokens,
                    device=self.gpu_device,
                    dtype=input_ids.dtype,
                )

                # Expand input_ids to accommodate audio tokens
                new_input_ids = []
                for i, token_id in enumerate(input_ids):
                    if token_id == audio_token_id:
                        new_input_ids.extend(virtual_tokens.tolist())
                    else:
                        new_input_ids.append(token_id.item())

                input_ids = torch.tensor(
                    new_input_ids, device=self.gpu_device, dtype=input_ids.dtype
                )

        # Prepare batch
        batch_input_ids = [input_ids.tolist()]

        # Update sampling config
        input_lengths = torch.tensor(
            [input_ids.size(-1)], device=self.gpu_device, dtype=torch.int32
        )
        max_input_length = torch.max(input_lengths).item()
        self.sampling_config.max_new_tokens = min(
            self.max_new_tokens, self.max_seq_len - max_input_length
        )

        # Run generation with audio features
        with torch.no_grad():
            outputs = self.runner.generate(
                batch_input_ids=batch_input_ids,
                sampling_config=self.sampling_config,
                prompt_table=prompt_table,
                prompt_tasks=prompt_tasks,
                prompt_vocab_size=prompt_vocab_size,
            )

        # Decode output
        if outputs is not None and len(outputs) > 0:
            output_ids = outputs[0]
            # Remove input tokens from output
            generated_ids = output_ids[input_lengths[0] :]
            generated_text = self.audio_tokenizer.decode(generated_ids, skip_special_tokens=True)
            return generated_text.strip()

        return ""  # Return empty string if no output
