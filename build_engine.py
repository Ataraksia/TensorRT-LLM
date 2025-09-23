import torch
from tensorrt_llm import Builder
from tensorrt_llm.builder import (
    BuildConfig,
    Engine,
    EngineConfig,
    _init_max_seq_len,
    build,
    optimize_model_with_config,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from tensorrt_llm.models.higgs_audio.model import HiggsAudioForCausalLM
import copy
import json
import math
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import tensorrt as trt

from tensorrt_llm.models.higgs_audio.model import arange

from tensorrt_llm._common import _is_building, check_max_num_tokens, serialize_engine
from tensorrt_llm._utils import (
    get_sm_version,
    np_bfloat16,
    np_float8,
    str_dtype_to_trt,
    to_json_file,
    trt_gte,
)
from tensorrt_llm.auto_parallel import auto_parallel
from tensorrt_llm.auto_parallel.config import AutoParallelConfig
from tensorrt_llm.bindings import KVCacheType
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.graph_rewriting import optimize
from tensorrt_llm.logger import logger
from tensorrt_llm.lora_manager import LoraConfig
from tensorrt_llm.models import PretrainedConfig, PretrainedModel
from tensorrt_llm.models.modeling_utils import SpeculativeDecodingMode, optimize_model
from tensorrt_llm.network import Network, net_guard
from tensorrt_llm.plugin import PluginConfig
from tensorrt_llm.quantization import QuantAlgo, QuantMode
from tensorrt_llm.version import __version__


def main():
    logger.set_level("info")
    gpu_device = torch.device("cuda", 0)
    torch.cuda.set_device(gpu_device)
    trtllm_config = HiggsAudioConfig.from_hugging_face("bosonai/higgs-audio-v2-generation-3B-base")
    trtllm_model = HiggsAudioForCausalLM.from_hugging_face()

    max_num_tokens = trtllm_config.max_num_tokens
    print(f"max_num_tokens: {max_num_tokens}")
    build_config = BuildConfig()
    build_config.max_batch_size = 8
    build_config.opt_batch_size = 8
    build_config.max_input_len = max_num_tokens
    build_config.max_num_tokens = max_num_tokens
    build_config.opt_num_tokens = max_num_tokens // 2
    build_config.max_seq_len = max_num_tokens
    build_config.plugin_config.remove_input_padding = True
    build_config.plugin_config.dtype = "bfloat16"
    build_config.plugin_config.gpt_attention_plugin = "bfloat16"
    build_config.plugin_config.gemm_plugin = "bfloat16"
    build_config.max_beam_width = 1
    # build_config.max_draft_len = 8 - 1
    # build_config.speculative_decoding_mode = SpeculativeDecodingMode.DRAFT_TOKENS_EXTERNAL
    build_config.enable_debug_output = True
    # build_config.plugin_config.use_fp8_context_fmha = True
    # build_config.plugin_config._multiple_profiles = True
    # build_config.strongly_typed = False
    # build_config.plugin_config._gemm_swiglu_plugin = "FP8"
    # build_config.plugin_config._fp8_rowwise_gemm_plugin = "bfloat16"
    # build_config.plugin_config._low_latency_gemm_swiglu_plugin = "FP8"
    # build_config.plugin_config.low_latency_gemm_plugin = FP8
    # build_config.plugin_config.gemm_allreduce_plugin = "bfloat16"
    # build_config.plugin_config.context_fmha = True
    # build_config.plugin_config.norm_quant_fusion = True
    # build_config.plugin_config.user_buffer = True
    # build_config.plugin_config._use_paged_context_fmha = True
    # build_config.plugin_config._use_fp8_context_fmha = True
    # build_config.plugin_config._fuse_fp4_quant = True
    # build_config.plugin_config.paged_state = True
    # build_config.plugin_config._streamingllm = True
    # build_config.plugin_config.use_fused_mlp = True
    # build_config.plugin_config._pp_reduce_scatter = True
    # build_config.plugin_config._use_fused_mlp = True

    trtllm_model.config.max_position_embeddings = max_num_tokens
    trtllm_model.config.build = True
    engine = build(trtllm_model, build_config)
    engine.save("./higgs_audio_engine")


if __name__ == "__main__":
    main()
