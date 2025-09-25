#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#
file(GLOB_RECURSE SRC_CU *.cu)
add_library(ws_layernorm_src STATIC ${SRC_CU})

# Ensure we compile layernorm kernels with arch specific targets for SM90+ to satisfy archCondition.h
if(90 IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)
  # Use arch-specific Hopper target to satisfy archCondition.h (requires specific or family specific target)
  set_cuda_architectures(ws_layernorm_src 90a)
endif()
if(100 IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)
  if(TLLM_EXCLUDE_SM120)
    # Force explicit accel targets; do NOT use family form which could reintroduce 120.
    if(103 IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)
      set_cuda_architectures(ws_layernorm_src 100 103)
    else()
      set_cuda_architectures(ws_layernorm_src 100)
    endif()
  else()
    # Safe to use family form when not excluding 120
    set_cuda_architectures(ws_layernorm_src 100f)
  endif()
endif()
if(120 IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG AND NOT TLLM_EXCLUDE_SM120)
  set_cuda_architectures(ws_layernorm_src 120f)
endif()

if("100" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)

  # for blackwell
  set(WS_LAYERNORM_NVCC_FLAGS)
  list(APPEND WS_LAYERNORM_NVCC_FLAGS --extra-device-vectorization)
  list(APPEND WS_LAYERNORM_NVCC_FLAGS
       --ptxas-options=--warn-on-local-memory-usage,--warn-on-spills)

  target_compile_options(
    ws_layernorm_src
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${WS_LAYERNORM_NVCC_FLAGS}>)
  target_compile_definitions(ws_layernorm_src
                             PRIVATE CUDA_PTX_KNOB_COLD_BLOCK_ENABLED)
endif()

if(NOT WIN32)
  target_compile_options(
    ws_layernorm_src PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-psabi>)
endif()

set_property(TARGET ws_layernorm_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET ws_layernorm_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
