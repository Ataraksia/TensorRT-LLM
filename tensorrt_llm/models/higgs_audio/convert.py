# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

import torch

from ...logger import logger
from ...mapping import Mapping
from ...quantization import QuantAlgo
from ..convert_utils import infer_dtype
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
    """Map HF Higgs-Audio weights into TRT-LLM checkpoint dict.

    TODO: implement full mapping for text LLM weights and audio coupling path.
    For now, emit a metadata-only checkpoint with an `expected_keys` template to guide implementation.
    """
    logger.warning("Higgs-Audio conversion: returning metadata + expected key template (no tensors yet).")
    ckpt = {
        "metadata": {
            "source": "hf",
            "hf_model_dir": hf_model_dir,
            "dtype": str(config.dtype),
            "vocab_size": int(config.vocab_size),
            "adapter_type": getattr(config, "audio_adapter_type", None),
        },
        "tensors": {},
        "expected_keys": _backbone_key_template(config),
    }
    return ckpt


def get_default_quant_config() -> QuantConfig:
    """Return a conservative default quantization config for prototyping."""
    return QuantConfig(
        quant_algo=QuantAlgo.W8A16,
        kv_cache_quant_algo=QuantAlgo.FP8,
    )
