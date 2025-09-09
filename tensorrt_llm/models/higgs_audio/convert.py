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

import copy
import functools
import os
import time
from collections import defaultdict
from typing import Optional

import numpy as np
import safetensors
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.pytorch_utils import Conv1D

from ..._utils import pad_vocab_size
from ...mapping import Mapping
from ...quantization import QuantAlgo
from ..convert_utils import (
    dup_kv_bias,
    dup_kv_weight,
    generate_int8,
    get_weight,
    get_weight_and_bias,
    load_calib_dataset,
    smooth_gemm,
    smooth_gemm_fc1_gate,
    split,
    split_matrix_tp,
    split_qkv_bias_tp,
    split_qkv_tp,
)
from .config import HiggsAudioConfig


def make_context_higgs_audio(tokenizer, query, history=None, system=None, max_input_length=512):
    """Create context for HiggsAudio model."""
    # Simple context creation for HiggsAudio - just tokenize the query
    pre_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an AI assistant designed to convert text into speech. Generate speech for the user's text, using the specified description.<|scene_desc_start|>Audio is recorded from a quiet room. Speaker is an enthusiastic young Australian woman in her early 20s with a bright, high-pitched voice.<|scene_desc_end|><|eot_id|><|start_header_id|>user<|end_header_id|>"  # noqa: E501
    post_prompt = "<|eot_id|><|start_header_id|>assistant<|end_header_id|><|audio_out_bos|>"
    query = pre_prompt + query + post_prompt

    tokens = tokenizer.encode(query, max_length=max_input_length, truncation=True)
    return None, tokens


@torch.no_grad()
def smooth_higgs_model(model, scales, alpha, higgs_qkv_para, higgs_smoother):
    # Smooth the activation and weights with smoother = $\diag{s}$
    for name, module in model.named_modules():
        # Import from the correct TensorRT-LLM model location
        from tensorrt_llm.models.higgs_audio.model import HiggsAudioDualFFNDecoderLayer

        if not isinstance(module, HiggsAudioDualFFNDecoderLayer):
            continue
        # qkv_proj
        layer_name_q = name + ".self_attn.q_proj"
        layer_name_k = name + ".self_attn.k_proj"
        layer_name_v = name + ".self_attn.v_proj"
        layer_name_qkv = name + ".self_attn.qkv_proj"

        weight = torch.cat(
            [
                module.self_attn.q_proj.weight,
                module.self_attn.k_proj.weight,
                module.self_attn.v_proj.weight,
            ],
            dim=0,
        )

        smoother = smooth_gemm(
            weight, scales[layer_name_q]["x"], module.input_layernorm.weight, None, alpha
        )

        scales[layer_name_qkv]["x"] = scales[layer_name_q]["x"] / smoother
        scales[layer_name_qkv]["w"] = weight.abs().max(dim=1)[0]
        scales[layer_name_qkv]["y"] = torch.cat(
            [scales[layer_name_q]["y"], scales[layer_name_k]["y"], scales[layer_name_v]["y"]], dim=0
        )

        # see transpose_weights function
        higgs_qkv_para[layer_name_qkv] = weight.transpose(0, 1).contiguous()

        # =================================================================
        layer_name = name + ".self_attn.o_proj"
        smoother = smooth_gemm(
            module.self_attn.o_proj.weight, scales[layer_name]["x"], None, None, alpha
        )
        higgs_smoother[layer_name] = smoother.float()

        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.self_attn.o_proj.weight.abs().max(dim=1)[0]

        # ==================================================================
        fc1_layer_name = name + ".mlp.gate_proj"
        gate_layer_name = name + ".mlp.up_proj"

        smoother = smooth_gemm_fc1_gate(
            module.mlp.gate_proj.weight,
            module.mlp.up_proj.weight,
            scales[fc1_layer_name]["x"],
            module.post_attention_layernorm.weight,
            None,
            alpha,
        )

        scales[fc1_layer_name]["x"] = scales[fc1_layer_name]["x"] / smoother
        scales[fc1_layer_name]["w"] = module.mlp.gate_proj.weight.abs().max(dim=1)[0]

        scales[gate_layer_name]["x"] = scales[gate_layer_name]["x"] / smoother
        scales[gate_layer_name]["w"] = module.mlp.up_proj.weight.abs().max(dim=1)[0]

        # ==================================================================
        layer_name = name + ".mlp.down_proj"
        smoother = smooth_gemm(
            module.mlp.down_proj.weight, scales[layer_name]["x"], None, None, alpha
        )
        higgs_smoother[layer_name] = smoother.float()
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.mlp.down_proj.weight.abs().max(dim=1)[0]


@torch.no_grad()
def capture_activation_range(model, tokenizer, dataset, num_samples=512, seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    act_scales = defaultdict(lambda: {"x": None, "y": None, "w": None})
    tokenizer.pad_token_id = tokenizer.eos_token_id

    def stat_tensor(name, tensor, act_scales, key):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float()

        if act_scales[name][key] is None:
            act_scales[name][key] = comming_max
        else:
            act_scales[name][key] = torch.max(act_scales[name][key], comming_max)

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x, act_scales, "x")
        stat_tensor(name, y, act_scales, "y")

        if act_scales[name]["w"] is None:
            act_scales[name]["w"] = m.weight.abs().clip(1e-8, None).max(dim=1)[0]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) or isinstance(m, Conv1D):
            hooks.append(m.register_forward_hook(functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples), desc="calibrating model"):
        line = dataset[i]
        line = line + " TL;DR: "
        line = line.strip()
        line = line.replace(" n't", "n't")
        _, input_id_list = make_context_higgs_audio(
            tokenizer=tokenizer,
            query=line,
            max_input_length=seq_len,
        )
        line_encoded = (
            torch.from_numpy(np.array(input_id_list, dtype=np.int32)).type(torch.int32).unsqueeze(0)
        )
        line_encoded = line_encoded.to(device)
        model(line_encoded)
    for h in hooks:
        h.remove()
    return act_scales


def get_tllm_linear_weight(
    weight,
    prefix,
    bias=None,
    use_weight_only=False,
    plugin_weight_only_quant_type=torch.int8,
    dtype="float32",
    use_gemm_woq_plugin=True,
    postfix="weight",
    quant_scale_name=None,
):
    results = {}
    if use_weight_only:
        if weight.dim() > 2:
            v = weight.transpose(1, 2).contiguous().clone()
        else:
            v = weight.t().contiguous().clone()
        processed_torch_weights, torch_weight_scales = (
            torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                v.cpu(), plugin_weight_only_quant_type
            )
        )
        if not use_gemm_woq_plugin:
            results[prefix + postfix] = v.to(dtype)
        else:
            results[prefix + postfix] = processed_torch_weights
        if quant_scale_name is not None:
            results[quant_scale_name] = torch_weight_scales
        else:
            results[prefix + "per_channel_scale"] = torch_weight_scales
    else:
        results[prefix + postfix] = weight.clone()

    if bias is not None:
        results[prefix + "bias"] = bias

    return results


def get_tllm_linear_sq_weight(
    vals,
    prefix,
    shape,
    tensor_parallel,
    is_qkv=False,
    per_token=False,
    per_channel=False,
    last_prefix=None,
    bias=None,
    smoother_value=None,
    smoother_shape=None,
    rank=0,
    cat_dim=0,
    multi_query_mode=False,
):
    results = {}

    def multi_query_split(data, local_dim, head_size, tp_size, cur_rank):
        q, k, v = torch.split(data, [local_dim, head_size, head_size], dim=-1)
        q_split = torch.split(q, q.shape[-1] // tp_size, dim=-1)
        k_split = torch.split(k, k.shape[-1] // tp_size, dim=-1)
        v_split = torch.split(v, v.shape[-1] // tp_size, dim=-1)
        return [
            torch.concat((q_split[ii], k_split[ii], v_split[ii]), dim=-1) for ii in range(tp_size)
        ][cur_rank]

    col_shape = shape if (is_qkv or per_channel) else [1, 1]

    if per_token:
        if per_channel:
            original_weights = vals["weight.int8.col"]
        else:
            original_weights = vals["weight.int8"]
        local_dim = original_weights.shape[0]
        head_size = (original_weights.shape[1] - local_dim) // 2

        if multi_query_mode:
            cur_weights = multi_query_split(
                original_weights, local_dim, head_size, tensor_parallel, rank
            )
        else:
            cur_weights = torch.chunk(original_weights, tensor_parallel, dim=cat_dim)[rank]
        if is_qkv:
            hidden_dim = cur_weights.shape[0]
            cur_weights = cur_weights.reshape(hidden_dim, -1)
        results[prefix + "weight"] = cur_weights.t().contiguous()
        if smoother_value is None:
            results[last_prefix] = torch.from_numpy(np.array([1.0], dtype=np.float32))

        if per_channel:
            cur_per_channel_value = vals["scale_w_quant_orig.col"]
            if smoother_value is None:
                if multi_query_mode:
                    cur_per_channel_value = multi_query_split(
                        vals["scale_w_quant_orig.col"], local_dim, head_size, tensor_parallel, rank
                    )
                else:
                    cur_per_channel_value = np.split(
                        vals["scale_w_quant_orig.col"], tensor_parallel, axis=cat_dim
                    )[rank]
        else:
            cur_per_channel_value = vals["scale_w_quant_orig"]
            if is_qkv:
                if multi_query_mode:
                    cur_per_channel_value = multi_query_split(
                        vals["scale_w_quant_orig"], local_dim, head_size, tensor_parallel, rank
                    )
                else:
                    cur_per_channel_value = torch.split(
                        vals["scale_w_quant_orig"], tensor_parallel, axis=cat_dim
                    )[rank]

        results[prefix + "per_channel_scale"] = cur_per_channel_value.reshape(col_shape)
    else:
        if per_channel:
            original_weights = vals["weight.int8.col"]
        else:
            original_weights = vals["weight.int8"]
        local_dim = original_weights.shape[0]
        head_size = (original_weights.shape[1] - local_dim) // 2

        if multi_query_mode:
            cur_weights = multi_query_split(
                original_weights, local_dim, head_size, tensor_parallel, rank
            )
        else:
            cur_weights = torch.chunk(original_weights, tensor_parallel, dim=cat_dim)[rank]
        if is_qkv:
            hidden_dim = cur_weights.shape[0]
            cur_weights = cur_weights.reshape(hidden_dim, -1)
        results[prefix + "weight"] = cur_weights.t().contiguous()

        if per_channel:
            cur_per_channel_value = vals["scale_y_accum_quant.col"]
            if smoother_value is None:
                if multi_query_mode:
                    cur_per_channel_value = multi_query_split(
                        vals["scale_y_accum_quant.col"], local_dim, head_size, tensor_parallel, rank
                    )
                else:
                    cur_per_channel_value = np.split(
                        vals["scale_y_accum_quant.col"], tensor_parallel, axis=cat_dim
                    )[rank]
        else:
            cur_per_channel_value = vals["scale_y_accum_quant"]
            # QKV is always per_channel
            if is_qkv:
                if multi_query_mode:
                    cur_per_channel_value = multi_query_split(
                        vals["scale_y_accum_quant"], local_dim, head_size, tensor_parallel, rank
                    )
                else:
                    cur_per_channel_value = np.split(
                        vals["scale_y_accum_quant"], tensor_parallel, axis=cat_dim
                    )[rank]

        results[prefix + "per_channel_scale"] = cur_per_channel_value.reshape(
            col_shape
        ).contiguous()

        results[last_prefix] = vals["scale_x_orig_quant"].contiguous()

        results[prefix + "act_scale"] = vals["scale_y_quant_orig"].contiguous()

    if smoother_value is not None:
        cur_smoother_value = torch.split(
            smoother_value, smoother_value.shape[-1] // tensor_parallel, dim=cat_dim
        )[rank]

        results[prefix + "smoother"] = (
            cur_smoother_value.reshape(smoother_shape).contiguous().to(torch.float32)
        )

    if bias is not None:
        results[prefix + "bias"] = bias

    return results


def load_hf_higgs_audio_from_safetensors(model_dir: str):
    all


def load_hf_higgs_audio(model_dir: str, load_model_on_cpu: bool = False):
    from transformers import AutoModelForCausalLM as model_cls

    model = model_cls.from_pretrained(
        model_dir,
        device_map="auto" if not load_model_on_cpu else "cpu",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    return model


def convert_hf_higgs(
    hf_model,
    mapping: Mapping,
    vocab_size=128256,
    dtype="bfloat16",
    use_parallel_embedding=False,
    sharding_dim=0,
    use_weight_only=False,
    use_gemm_woq_plugin=False,
    plugin_weight_only_quant_type=torch.int8,
    use_smooth_quant=False,
    per_channel=False,
    per_token=False,
    int8_kv_cache=False,
    act_range=[],
    qkv_para=[],
    smoother=[],
    moe_config=None,
):
    weights = {}
    tik = time.time()
    tensor_parallel = mapping.tp_size
    model_params = dict(hf_model.named_parameters())

    dtype = getattr(torch, dtype)
    hf_config = hf_model.config

    text_config = hf_config.text_config

    if hasattr(hf_config, "llm_config"):
        hf_config = hf_config.llm_config

    text_config = hf_config.text_config
    num_attention_heads = text_config.num_attention_heads
    hidden_size = text_config.hidden_size
    head_size = hidden_size // num_attention_heads

    num_key_value_heads = (
        text_config.num_key_value_heads
        if hasattr(text_config, "num_key_value_heads")
        else num_attention_heads
    )
    mha_mode = num_key_value_heads == num_attention_heads
    layers_range = mapping.pp_layers(text_config.num_hidden_layers)

    layer_prefix = "layers."  # HiggsAudio uses layers. prefix
    intermediate_size = text_config.intermediate_size

    for l in layers_range:
        prefix = layer_prefix + f"{l}."
        tllm_prex = f"transformer.layers.{l - layers_range[0]}."
        q_weight, q_bias = get_weight_and_bias(
            model_params, prefix + "self_attn." + "q_proj", dtype
        )
        k_weight, k_bias = get_weight_and_bias(
            model_params, prefix + "self_attn." + "k_proj", dtype
        )
        v_weight, v_bias = get_weight_and_bias(
            model_params, prefix + "self_attn." + "v_proj", dtype
        )
        if not mha_mode:
            if num_key_value_heads < tensor_parallel:
                # duplicate the KV heads up to tensor_parallel
                k_weight = dup_kv_weight(k_weight, num_key_value_heads, tensor_parallel)
                v_weight = dup_kv_weight(v_weight, num_key_value_heads, tensor_parallel)
                k_bias = dup_kv_bias(k_bias, num_key_value_heads, tensor_parallel)
                v_bias = dup_kv_bias(v_bias, num_key_value_heads, tensor_parallel)
            assert (k_weight.shape[0] % (mapping.tp_size * head_size)) == 0
            assert (v_weight.shape[0] % (mapping.tp_size * head_size)) == 0

            if k_bias is not None and v_bias is not None:
                assert (k_bias.shape[0] % (mapping.tp_size * head_size)) == 0
                assert (v_bias.shape[0] % (mapping.tp_size * head_size)) == 0

            wq = split(q_weight, mapping.tp_size, mapping.tp_rank)
            wk = split(k_weight, mapping.tp_size, mapping.tp_rank)
            wv = split(v_weight, mapping.tp_size, mapping.tp_rank)

            qkv_w = torch.concat((wq, wk, wv))

            if q_bias is not None and k_bias is not None and v_bias is not None:
                bq = split(q_bias, mapping.tp_size, mapping.tp_rank)
                bk = split(k_bias, mapping.tp_size, mapping.tp_rank)
                bv = split(v_bias, mapping.tp_size, mapping.tp_rank)
                qkv_b = torch.concat((bq, bk, bv))
            else:
                qkv_b = None
        else:
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)

            qkv_w = split_qkv_tp(
                qkv_weight, num_attention_heads, hidden_size, tensor_parallel, mapping.tp_rank
            )
            qkv_b = split_qkv_bias_tp(
                qkv_bias, num_attention_heads, hidden_size, tensor_parallel, mapping.tp_rank
            )

        if use_smooth_quant:
            qkv_proj_key = "self_attn.qkv_proj"
            qkv_weight = qkv_para[prefix + qkv_proj_key]
            qkv_out_dim = qkv_weight.shape[1]

            if not mha_mode:
                local_dim = qkv_weight.shape[0]
                kv_hidden_size = (qkv_weight.shape[-1] - local_dim) // 2
                qkv_weight = qkv_weight.reshape(local_dim, local_dim + 2 * kv_hidden_size)
            else:
                qkv_weight = qkv_weight.reshape(hidden_size, 3, hidden_size)

            int8_weights = generate_int8(
                qkv_weight,
                act_range.get(prefix + qkv_proj_key),
                is_qkv=True,
                multi_query_mode=bool(not mha_mode),
            )

            weights.update(
                get_tllm_linear_sq_weight(
                    int8_weights,
                    tllm_prex + "attention.qkv.",
                    [1, qkv_out_dim // tensor_parallel],
                    tensor_parallel,
                    is_qkv=True,
                    per_token=per_token,
                    per_channel=per_channel,
                    last_prefix=tllm_prex + "input_layernorm.scale_to_int",
                    bias=qkv_b,
                    smoother_value=None,
                    smoother_shape=None,
                    rank=mapping.tp_rank,
                    cat_dim=-1,
                    multi_query_mode=bool(not mha_mode),
                )
            )
        else:
            weights.update(
                get_tllm_linear_weight(
                    qkv_w,
                    tllm_prex + "attention.qkv.",
                    qkv_b,
                    use_weight_only,
                    plugin_weight_only_quant_type,
                    dtype,
                    use_gemm_woq_plugin,
                )
            )
        if int8_kv_cache:
            qkv_y = torch.cat(
                [
                    act_range.get(prefix + "self_attn." + "q_proj")["y"],
                    act_range.get(prefix + "self_attn." + "k_proj")["y"],
                    act_range.get(prefix + "self_attn." + "v_proj")["y"],
                ],
                dim=0,
            )

            int8_kv_scales = qkv_y.max() / 127.0

            kv_cache_weights = {}

            kv_cache_weights[tllm_prex + "attention.kv_cache_scaling_factor"] = (
                int8_kv_scales.reshape([1])
            )

            weights.update(kv_cache_weights)

        attn_dense_weight = get_weight(model_params, prefix + "self_attn.o_proj", dtype)
        split_v = split_matrix_tp(attn_dense_weight, tensor_parallel, mapping.tp_rank, dim=1)
        if use_smooth_quant:
            attn_dense_weight = attn_dense_weight.t()
            int8_weights = generate_int8(
                attn_dense_weight, act_range.get(prefix + "self_attn.o_proj")
            )
            weights.update(
                get_tllm_linear_sq_weight(
                    int8_weights,
                    tllm_prex + "attention.dense.",
                    [1, hidden_size],
                    tensor_parallel,
                    is_qkv=False,
                    per_token=per_token,
                    per_channel=per_channel,
                    last_prefix=tllm_prex + "attention.quantization_scaling_factor",
                    smoother_value=smoother[(prefix + "self_attn.o_proj")],
                    smoother_shape=[1, hidden_size // tensor_parallel],
                    rank=mapping.tp_rank,
                    cat_dim=0,
                )
            )
        else:
            weights.update(
                get_tllm_linear_weight(
                    split_v,
                    tllm_prex + "attention.dense.",
                    None,
                    use_weight_only,
                    plugin_weight_only_quant_type,
                    dtype,
                    use_gemm_woq_plugin,
                )
            )
        # Standard MLP fc projection (combined gate and up)
        gate_w = get_weight(model_params, prefix + "mlp.gate_proj", dtype)
        up_w = get_weight(model_params, prefix + "mlp.up_proj", dtype)
        fc_w = torch.cat([gate_w, up_w], dim=0)
        split_v = split_matrix_tp(fc_w, tensor_parallel, mapping.tp_rank, dim=0)

        if use_smooth_quant:
            fc_w_t = fc_w.t()
            int8_weights = generate_int8(fc_w_t, act_range.get(prefix + "mlp.gate_proj"))
            weights.update(
                get_tllm_linear_sq_weight(
                    int8_weights,
                    tllm_prex + "mlp.fc.",
                    [1, intermediate_size // tensor_parallel],
                    tensor_parallel,
                    is_qkv=False,
                    per_token=per_token,
                    per_channel=per_channel,
                    last_prefix=tllm_prex + "post_layernorm.scale_to_int",
                    smoother_value=None,
                    smoother_shape=None,
                    rank=mapping.tp_rank,
                    cat_dim=-1,
                )
            )
        else:
            weights.update(
                get_tllm_linear_weight(
                    split_v,
                    tllm_prex + "mlp.fc.",
                    None,
                    use_weight_only,
                    plugin_weight_only_quant_type,
                    dtype,
                    use_gemm_woq_plugin,
                )
            )

        # Standard MLP proj projection (down)
        mlp_proj_weight = get_weight(model_params, prefix + "mlp.down_proj", dtype)
        split_v = split_matrix_tp(mlp_proj_weight, tensor_parallel, mapping.tp_rank, dim=1)

        if use_smooth_quant:
            mlp_proj_weight = mlp_proj_weight.t()
            int8_weights = generate_int8(mlp_proj_weight, act_range.get(prefix + "mlp.down_proj"))

            weights.update(
                get_tllm_linear_sq_weight(
                    int8_weights,
                    tllm_prex + "mlp.proj.",
                    [1, hidden_size],
                    tensor_parallel,
                    is_qkv=False,
                    per_token=per_token,
                    per_channel=per_channel,
                    last_prefix=tllm_prex + "mlp.quantization_scaling_factor",
                    smoother_value=smoother[prefix + "mlp.down_proj"],
                    smoother_shape=[1, intermediate_size // tensor_parallel],
                    rank=mapping.tp_rank,
                    cat_dim=0,
                )
            )
        else:
            weights.update(
                get_tllm_linear_weight(
                    split_v,
                    tllm_prex + "mlp.proj.",
                    None,
                    use_weight_only,
                    plugin_weight_only_quant_type,
                    dtype,
                    use_gemm_woq_plugin,
                )
            )

        # HiggsAudio dual FFN layers - Handle audio_mlp components separately
        # Check if audio_mlp exists for this layer
        audio_mlp_gate_weight = prefix + "audio_mlp.up_proj.weight"
        if audio_mlp_gate_weight in model_params:
            # Audio MLP fc projection (combined gate and up)
            gate_w = get_weight(model_params, prefix + "audio_mlp.gate_proj", dtype)
            up_w = get_weight(model_params, prefix + "audio_mlp.up_proj", dtype)
            fc_w = torch.cat([gate_w, up_w], dim=0)
            split_v = split_matrix_tp(fc_w, tensor_parallel, mapping.tp_rank, dim=0)

            weights.update(
                get_tllm_linear_weight(
                    split_v,
                    tllm_prex + "audio_mlp.fc.",
                    None,
                    use_weight_only,
                    plugin_weight_only_quant_type,
                    dtype,
                    use_gemm_woq_plugin,
                )
            )

            # Audio MLP proj projection (down)
            audio_mlp_proj_weight = get_weight(model_params, prefix + "audio_mlp.down_proj", dtype)
            split_v = split_matrix_tp(
                audio_mlp_proj_weight, tensor_parallel, mapping.tp_rank, dim=1
            )

            weights.update(
                get_tllm_linear_weight(
                    split_v,
                    tllm_prex + "audio_mlp.proj.",
                    None,
                    use_weight_only,
                    plugin_weight_only_quant_type,
                    dtype,
                    use_gemm_woq_plugin,
                )
            )

        # HiggsAudio audio-specific layer norms (if they exist)
        audio_input_ln_weight = prefix + "audio_input_layernorm.weight"
        if audio_input_ln_weight in model_params:
            audio_input_ln_weight = get_weight(
                model_params, prefix + "audio_input_layernorm", dtype
            )
            weights[tllm_prex + "audio_input_layernorm.weight"] = audio_input_ln_weight

        audio_post_ln_weight = prefix + "audio_post_attention_layernorm.weight"
        if audio_post_ln_weight in model_params:
            audio_post_ln_weight = get_weight(
                model_params, prefix + "audio_post_attention_layernorm", dtype
            )
            weights[tllm_prex + "audio_post_attention_layernorm.weight"] = audio_post_ln_weight

        # Layer norms do not use tensor parallelism
        input_ln_weight = get_weight(model_params, prefix + "input_layernorm", dtype)
        weights[tllm_prex + "input_layernorm.weight"] = input_ln_weight

        post_ln_weight = get_weight(model_params, prefix + "post_attention_layernorm", dtype)
        weights[tllm_prex + "post_layernorm.weight"] = post_ln_weight

    # Load embedding weights
    v = get_weight(model_params, "embed_tokens", dtype)

    if use_parallel_embedding:
        v = split_matrix_tp(v, mapping.tp_size, mapping.tp_rank, dim=sharding_dim)

    if mapping.is_first_pp_rank():
        weights["transformer.vocab_embedding.weight"] = v

    # HiggsAudio specific components
    # Audio codebook embeddings
    audio_codebook_key = "audio_codebook_embeddings.weight"
    if audio_codebook_key in model_params:
        audio_codebook_weight = get_weight(model_params, "audio_codebook_embeddings", dtype)
        weights["transformer.audio_codebook_embeddings.weight"] = audio_codebook_weight

    # Audio decoder projection with dual heads (only load if on last PP rank)
    if mapping.is_last_pp_rank():
        audio_decoder_proj_key = "audio_decoder_proj"

        # Audio LM head (for audio token generation)
        if f"{audio_decoder_proj_key}.audio_lm_head.weight" in model_params:
            audio_lm_head_weight = get_weight(
                model_params, f"{audio_decoder_proj_key}.audio_lm_head", dtype
            )
            print(f"DEBUG: Original audio_lm_head_weight shape: {audio_lm_head_weight.shape}")
            print(f"DEBUG: Expected vocab_size: {vocab_size}")

            # For audio-only generation, we need to pad the audio_lm_head to match vocab_size
            # This allows the model to accept text tokens as input but only generate audio tokens
            if audio_lm_head_weight.shape[0] < vocab_size:
                print(
                    f"DEBUG: Padding audio_lm_head from {audio_lm_head_weight.shape[0]} to {vocab_size}"
                )
                pad_width = vocab_size - audio_lm_head_weight.shape[0]
                # Convert to float32 for padding, then back to original dtype
                original_dtype = audio_lm_head_weight.dtype
                audio_lm_head_weight_np = audio_lm_head_weight.detach().cpu().float().numpy()
                padded_weight_np = np.pad(
                    audio_lm_head_weight_np,
                    ((0, pad_width), (0, 0)),
                    "constant",
                    constant_values=0,
                )
                audio_lm_head_weight = torch.from_numpy(padded_weight_np).to(original_dtype)
                print(f"DEBUG: Padded audio_lm_head_weight shape: {audio_lm_head_weight.shape}")

            if vocab_size % mapping.tp_size != 0:
                vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
                pad_width = vocab_size_padded - vocab_size
                audio_lm_head_weight = torch.from_numpy(
                    np.pad(
                        audio_lm_head_weight.detach().cpu().numpy(),
                        ((0, pad_width), (0, 0)),
                        "constant",
                        constant_values=0,
                    )
                )
            # Use audio_lm_head as the main lm_head for HiggsAudio
            weights["lm_head.weight"] = split_matrix_tp(
                audio_lm_head_weight, tensor_parallel, mapping.tp_rank, dim=0
            )

        # Text LM head (for text token generation, if needed separately)
        if f"{audio_decoder_proj_key}.text_lm_head.weight" in model_params:
            text_lm_head_weight = get_weight(
                model_params, f"{audio_decoder_proj_key}.text_lm_head", dtype
            )
            if vocab_size % mapping.tp_size != 0:
                vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
                pad_width = vocab_size_padded - vocab_size
                text_lm_head_weight = torch.from_numpy(
                    np.pad(
                        text_lm_head_weight.detach().cpu().numpy(),
                        ((0, pad_width), (0, 0)),
                        "constant",
                        constant_values=0,
                    )
                )
            weights["text_lm_head.weight"] = split_matrix_tp(
                text_lm_head_weight, tensor_parallel, mapping.tp_rank, dim=0
            )

    # Final layer norm (only on last PP rank)
    if mapping.is_last_pp_rank():
        ln_f_w = get_weight(model_params, "norm", dtype)
        weights["transformer.ln_f.weight"] = ln_f_w

    tok = time.time()
    t = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
    print(f"Weights loaded. Total time: {t}")
    return weights


def quantize(
    hf_model_dir: str,
    output_dir: str,
    config: HiggsAudioConfig,
    calib_dataset="bosonai/EmergentTTS-Eval",
):
    """Quantize the save the model as TRT-LLM checkpoint to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    config.to_json_file(os.path.join(output_dir, "config.json"))

    mapping = config.mapping
    assert mapping.rank == 0, "quantize should be called at rank 0 only"

    quant_config = config.quantization
    use_smooth_quant = quant_config._use_plugin_sq
    int8_kv_cache = quant_config.kv_cache_quant_algo == "INT8"

    assert use_smooth_quant or int8_kv_cache, "Call from_hugging_face when there is no quantization"
    assert hf_model_dir is not None
    ## only load and call smooth quant routine once for all ranks
    from transformers import AutoModelForCausalLM as model_cls

    hf_model = model_cls.from_pretrained(
        hf_model_dir,
        device_map="auto",
        torch_dtype="auto" if not use_smooth_quant else torch.float16,
        trust_remote_code=True,
    ).half()

    os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get("TOKENIZERS_PARALLELISM", "false")
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_dir, trust_remote_code=True, use_fast=False, padding_side="left"
    )
    dataset = load_calib_dataset(calib_dataset, None, "train", "text_to_synthesize")
    act_range = capture_activation_range(hf_model, tokenizer, dataset)
    qkv_para = {}
    # smoother for inputs of self_attn.o_proj and mlp.down_proj
    smoother = {}
    if use_smooth_quant:
        smooth_higgs_model(hf_model, act_range, quant_config.smoothquant_val, qkv_para, smoother)

    for rank in range(mapping.world_size):
        # To avoid changing the mapping arg in-place, also the given mapping from caller
        # is rank agnostic, since quantize is called from only one rank
        config = copy.deepcopy(config)
        config.set_rank(rank)
        weights = load_weights_from_hf_model(
            hf_model, config=config, act_range=act_range, qkv_para=qkv_para, smoother=smoother
        )
        safetensors.torch.save_file(weights, os.path.join(output_dir, f"rank{rank}.safetensors"))
        del weights


def load_weights_from_hf_model(
    hf_model,
    config: HiggsAudioConfig,
    act_range: Optional[dict] = None,
    qkv_para: Optional[dict] = None,
    smoother: Optional[dict] = None,
):
    # TODO: simplify the parameters here

    assert hf_model is not None
    plugin_weight_only_quant_type = None  # the value does not matter when use_weight_only is False
    quant_algo = config.quantization.quant_algo
    if quant_algo == QuantAlgo.W8A16:
        plugin_weight_only_quant_type = torch.int8
    elif quant_algo == QuantAlgo.W4A16:
        plugin_weight_only_quant_type = torch.quint4x2
    else:
        plugin_weight_only_quant_type = None
    use_gemm_woq_plugin = False

    mapping = config.mapping

    use_weight_only = quant_algo in [QuantAlgo.W8A16, QuantAlgo.W4A16]
    use_smooth_quant = config.quantization._use_plugin_sq
    per_channel = use_smooth_quant and "PER_CHANNEL" in quant_algo
    per_token = use_smooth_quant and "PER_TOKEN" in quant_algo
    int8_kv_cache = config.quantization.kv_cache_quant_algo == QuantAlgo.INT8
    weights = convert_hf_higgs(
        hf_model,
        mapping,
        vocab_size=config.vocab_size,
        dtype=config.dtype,
        use_weight_only=use_weight_only,
        use_gemm_woq_plugin=use_gemm_woq_plugin,
        plugin_weight_only_quant_type=plugin_weight_only_quant_type,
        use_parallel_embedding=config.use_parallel_embedding,
        sharding_dim=config.embedding_sharding_dim,
        use_smooth_quant=use_smooth_quant,
        per_channel=per_channel,
        per_token=per_token,
        int8_kv_cache=int8_kv_cache,
        act_range=act_range,
        qkv_para=qkv_para,
        smoother=smoother,
    )
    return weights
