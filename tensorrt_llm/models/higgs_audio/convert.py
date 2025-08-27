# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Dict, Optional

import torch

from ...logger import logger
from ...mapping import Mapping
from ...quantization import QuantAlgo
from ..convert_utils import (
    infer_dtype,
    iterate_shard_files,
    load_state_dict,
    split,
    split_qkv_tp,
    split_qkv_bias_tp,
    split_matrix_tp,
    dup_kv_weight,
)
from ..modeling_utils import QuantConfig
from .config import HiggsAudioConfig


def build_config_from_hf(
    hf_model_dir: str,
    *,
    dtype: str = "auto",
    mapping: Optional[Mapping] = None,
    quant_config: Optional[QuantConfig] = None,
    trust_remote_code: bool = True,
    **kwargs,
) -> HiggsAudioConfig:
    """Create a `HiggsAudioConfig` from a HuggingFace model directory.

    This reads the HF composition config (text + audio encoder) and flattens
    what we need for TRT-LLM.
    """
    import transformers

    hf_cfg, _unused = transformers.AutoConfig.from_pretrained(
        hf_model_dir, trust_remote_code=trust_remote_code, return_unused_kwargs=True
    )
    return HiggsAudioConfig.from_hugging_face(
        hf_cfg, dtype=dtype, mapping=mapping, quant_config=quant_config, **kwargs
    )


def _backbone_key_template(config: HiggsAudioConfig):
    """Return a list of expected TRT-LLM tensor keys for the LLaMA-like backbone.

    This is a guide for the future HF->TRT-LLM weight mapping implementation.
    """
    keys = []
    # Embedding and final norm/lm_head
    if config.mapping.is_first_pp_rank():
        keys.append("transformer.vocab_embedding.weight")
    for l in range(config.num_hidden_layers):
        # Attention projections
        base = f"transformer.layers.{l}.attention"
        keys.extend([
            f"{base}.qkv.weight",  # or separate q_proj/k_proj/v_proj depending on loader
            f"{base}.qkv.bias" if config.attn_bias else None,
            f"{base}.dense.weight",
            f"{base}.dense.bias" if config.attn_bias else None,
        ])
        # Pre/post norms
        keys.extend([
            f"transformer.layers.{l}.input_layernorm.weight",
            f"transformer.layers.{l}.post_layernorm.weight",
        ])
        # MLP
        mlp_base = f"transformer.layers.{l}.mlp"
        # Follow TRT-LLM LLaMA checkpoint naming
        keys.extend([
            f"{mlp_base}.fc.weight",    # maps from HF gate_proj
            f"{mlp_base}.gate.weight",  # maps from HF up_proj
            f"{mlp_base}.proj.weight",  # maps from HF down_proj
        ])
    if config.mapping.is_last_pp_rank():
        keys.append("transformer.ln_f.weight")
        keys.append("lm_head.weight")
    # Filter Nones
    return [k for k in keys if k is not None]


@torch.no_grad()
def load_weights_from_hf_model(
    hf_model_dir: str,
    config: HiggsAudioConfig,
    *,
    quant_config: Optional[QuantConfig] = None,
):
    """Map HF Higgs-Audio weights into TRT-LLM checkpoint dict (text backbone only).

    Notes:
    - Assumes HF parameter names are LLaMA-like and ROOT-level (no 'model.' prefix), e.g.:
      'embed_tokens.weight', 'layers.0.self_attn.q_proj.weight', ... , 'norm.weight', 'lm_head.weight'.
    - Ignores audio-specific parameters for now (keys prefixed with 'audio_' or containing 'audio_').
    - Handles TP/PP splitting following TRT-LLM helpers.
    """
    dtype = getattr(torch, config.dtype)
    mapping = config.mapping

    # Load shard tensors without instantiating HF model
    model_params: Dict[str, torch.Tensor] = {}
    for shard_path in iterate_shard_files(hf_model_dir, rank=mapping.rank, progress_bar=False):
        sd = load_state_dict(shard_path, dtype=dtype, device='cpu')
        # Merge only relevant text-backbone tensors
        for k, v in sd.items():
            if k.startswith('audio_') or '.audio_' in k:
                continue
            # Keep expected LLaMA-like keys; others are ignored for now
            model_params[k] = v

    weights: Dict[str, torch.Tensor] = {}

    # Embedding (first PP rank) - TP split rows
    if mapping.is_first_pp_rank() and 'embed_tokens.weight' in model_params:
        emb = model_params['embed_tokens.weight']
        emb_split = split(emb, mapping.tp_size, mapping.tp_rank, dim=0)
        weights['transformer.vocab_embedding.weight'] = emb_split

    # Determine layers assigned to this PP rank
    layers_range = mapping.pp_layers(config.num_hidden_layers)

    def _normalize_linear(w: torch.Tensor, expected_out: int, expected_in: int, name: str) -> torch.Tensor:
        # Ensure [out_features, in_features] = [expected_out, expected_in]
        if w is None:
            return None
        if w.dim() != 2:
            return w
        if w.shape[0] == expected_out and w.shape[1] == expected_in:
            return w
        if w.shape[0] == expected_in and w.shape[1] == expected_out:
            return w.t().contiguous()
        # Try to fix based on matching one dimension
        if w.shape[1] == expected_in:
            # already [*, expected_in]; if first dim != expected_out, warn but proceed
            if w.shape[0] != expected_out:
                logger.warning(f"{name}: unexpected out_features {w.shape[0]} != {expected_out}")
            return w
        if w.shape[0] == expected_in:
            t = w.t().contiguous()
            if t.shape[0] != expected_out:
                logger.warning(f"{name}: unexpected out_features after transpose {t.shape[0]} != {expected_out}")
            return t
        logger.warning(f"{name}: could not normalize shape {tuple(w.shape)} to ({expected_out}, {expected_in})")
        return w

    for l in layers_range:
        tllm_prefix = f'transformer.layers.{l - layers_range[0]}.'
        # Q, K, V projections
        q_out = config.num_attention_heads * config.head_size
        kv_out = config.num_key_value_heads * config.head_size
        q_w = _normalize_linear(model_params.get(f'layers.{l}.self_attn.q_proj.weight'), q_out, config.hidden_size, f'layer{l}.q')
        k_w = _normalize_linear(model_params.get(f'layers.{l}.self_attn.k_proj.weight'), kv_out, config.hidden_size, f'layer{l}.k')
        v_w = _normalize_linear(model_params.get(f'layers.{l}.self_attn.v_proj.weight'), kv_out, config.hidden_size, f'layer{l}.v')
        if q_w is None or k_w is None or v_w is None:
            logger.warning(f"Missing Q/K/V for layer {l}; skipping this layer.")
            continue

        # Infer kv heads from shape if config says MHA but shapes indicate GQA
        head_size = config.head_size
        n_heads = config.num_attention_heads
        inferred_kv_heads = k_w.shape[0] // head_size if (k_w.dim() == 2 and head_size > 0) else config.num_key_value_heads
        mha_mode = (inferred_kv_heads == n_heads)

        # Biases (rare; typically absent). If present, pack and split similarly
        q_b = model_params.get(f'layers.{l}.self_attn.q_proj.bias')
        k_b = model_params.get(f'layers.{l}.self_attn.k_proj.bias')
        v_b = model_params.get(f'layers.{l}.self_attn.v_proj.bias')

        # Ensure in_features (dim=1) align across Q/K/V before concat
        if q_w.shape[1] != k_w.shape[1]:
            if k_w.shape[0] == q_w.shape[1]:
                k_w = k_w.t().contiguous()
            else:
                print(f"[convert][layer {l}] adjusting k: q.in={q_w.shape[1]}, k={k_w.shape}")
        if q_w.shape[1] != v_w.shape[1]:
            if v_w.shape[0] == q_w.shape[1]:
                v_w = v_w.t().contiguous()
            else:
                print(f"[convert][layer {l}] adjusting v: q.in={q_w.shape[1]}, v={v_w.shape}")
        print(f"[convert][layer {l}] q={tuple(q_w.shape)} k={tuple(k_w.shape)} v={tuple(v_w.shape)}")

        if not mha_mode:
            # GQA: possibly duplicate KV heads up to tp_size
            if inferred_kv_heads < mapping.tp_size:
                k_w = dup_kv_weight(k_w, inferred_kv_heads, mapping.tp_size)
                v_w = dup_kv_weight(v_w, inferred_kv_heads, mapping.tp_size)
                if k_b is not None and v_b is not None:
                    from ..convert_utils import dup_kv_bias  # local import to avoid clutter at top
                    k_b = dup_kv_bias(k_b, inferred_kv_heads, mapping.tp_size)
                    v_b = dup_kv_bias(v_b, inferred_kv_heads, mapping.tp_size)

            wq = split(q_w, mapping.tp_size, mapping.tp_rank)
            wk = split(k_w, mapping.tp_size, mapping.tp_rank)
            wv = split(v_w, mapping.tp_size, mapping.tp_rank)
            qkv_w_split = torch.cat((wq, wk, wv))

            if q_b is not None and k_b is not None and v_b is not None:
                qkv_b = torch.cat((q_b, k_b, v_b))
                qkv_b_split = split_qkv_bias_tp(qkv_b, config.num_attention_heads, config.hidden_size,
                                                mapping.tp_size, mapping.tp_rank)
            else:
                qkv_b_split = None
        else:
            qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
            qkv_w_split = split_qkv_tp(qkv_w, config.num_attention_heads, config.hidden_size,
                                       mapping.tp_size, mapping.tp_rank)
            if q_b is not None and k_b is not None and v_b is not None:
                qkv_b = torch.cat((q_b, k_b, v_b))
                qkv_b_split = split_qkv_bias_tp(qkv_b, config.num_attention_heads, config.hidden_size,
                                                mapping.tp_size, mapping.tp_rank)
            else:
                qkv_b_split = None

        weights.update({
            tllm_prefix + 'attention.qkv.weight': qkv_w_split,
        })
        if qkv_b_split is not None:
            weights[tllm_prefix + 'attention.qkv.bias'] = qkv_b_split

        # Attention output projection
        o_w = _normalize_linear(model_params.get(f'layers.{l}.self_attn.o_proj.weight'), config.hidden_size, config.hidden_size, f'layer{l}.o')
        if o_w is None:
            logger.warning(f"Missing o_proj for layer {l}")
        else:
            o_w_split = split_matrix_tp(o_w, mapping.tp_size, mapping.tp_rank, dim=1)
            if f'layers.{l}.self_attn.o_proj.bias' in model_params:
                o_b = model_params[f'layers.{l}.self_attn.o_proj.bias']
            else:
                o_b = None
            weights[tllm_prefix + 'attention.dense.weight'] = o_w_split
            if o_b is not None:
                weights[tllm_prefix + 'attention.dense.bias'] = o_b

        # Norms
        in_ln = model_params.get(f'layers.{l}.input_layernorm.weight')
        post_ln = model_params.get(f'layers.{l}.post_attention_layernorm.weight')
        if in_ln is not None:
            weights[tllm_prefix + 'input_layernorm.weight'] = in_ln
        if post_ln is not None:
            weights[tllm_prefix + 'post_layernorm.weight'] = post_ln

        # MLP: gate/up/down map to fc/gate/proj per TRT-LLM naming
        gate = _normalize_linear(model_params.get(f'layers.{l}.mlp.gate_proj.weight'), config.intermediate_size, config.hidden_size, f'layer{l}.mlp.gate')
        up = _normalize_linear(model_params.get(f'layers.{l}.mlp.up_proj.weight'), config.intermediate_size, config.hidden_size, f'layer{l}.mlp.up')
        down = _normalize_linear(model_params.get(f'layers.{l}.mlp.down_proj.weight'), config.hidden_size, config.intermediate_size, f'layer{l}.mlp.down')
        if gate is None or up is None or down is None:
            logger.warning(f"Missing MLP weights for layer {l}")
        else:
            # Column-parallel (fc/gate) split by rows (dim=0). Row-parallel (proj) split by cols (dim=1)
            fc_split = split(gate, mapping.tp_size, mapping.tp_rank, dim=0)
            gate_split = split(up, mapping.tp_size, mapping.tp_rank, dim=0)
            proj_split = split_matrix_tp(down, mapping.tp_size, mapping.tp_rank, dim=1)
            weights[tllm_prefix + 'mlp.fc.weight'] = fc_split
            weights[tllm_prefix + 'mlp.gate.weight'] = gate_split
            weights[tllm_prefix + 'mlp.proj.weight'] = proj_split

    # Final norm and lm_head (last PP rank)
    if mapping.is_last_pp_rank():
        ln_f = model_params.get('norm.weight')
        if ln_f is not None:
            weights['transformer.ln_f.weight'] = ln_f
        lm_head = model_params.get('lm_head.weight')
        if lm_head is not None:
            # For TP, split vocab rows across ranks
            lm_head_split = split_matrix_tp(lm_head, mapping.tp_size, mapping.tp_rank, dim=0)
            weights['lm_head.weight'] = lm_head_split

    ckpt = {
        "metadata": {
            "source": "hf",
            "hf_model_dir": hf_model_dir,
            "dtype": str(config.dtype),
            "vocab_size": int(config.vocab_size),
            "adapter_type": getattr(config, "audio_adapter_type", None),
        },
        "tensors": weights,
        "expected_keys": _backbone_key_template(config),
    }
    return ckpt


def get_default_quant_config() -> QuantConfig:
    """Return a conservative default quantization config for prototyping."""
    return QuantConfig(
        quant_algo=QuantAlgo.W8A16,
        kv_cache_quant_algo=QuantAlgo.FP8,
    )
