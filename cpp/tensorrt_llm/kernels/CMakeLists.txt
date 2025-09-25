#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
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



file(GLOB_RECURSE SRC_CPP *.cpp)
file(GLOB_RECURSE SRC_CU *.cu)

# Exclude files in the cutlass_kernels, decoderMaskedMultiheadAttention and
# selectiveScan trtllmGenKernels folder
list(FILTER SRC_CPP EXCLUDE REGEX "cutlass_kernels/.*")
list(FILTER SRC_CU EXCLUDE REGEX "cutlass_kernels/.*")
list(FILTER SRC_CPP EXCLUDE REGEX "flashMLA/.*")
list(FILTER SRC_CU EXCLUDE REGEX "flashMLA/.*")
list(FILTER SRC_CPP EXCLUDE REGEX "contextFusedMultiHeadAttention/.*")
list(FILTER SRC_CU EXCLUDE REGEX "contextFusedMultiHeadAttention/.*")
list(FILTER SRC_CPP EXCLUDE REGEX "decoderMaskedMultiheadAttention/.*")
list(FILTER SRC_CU EXCLUDE REGEX "decoderMaskedMultiheadAttention/.*")
list(FILTER SRC_CPP EXCLUDE REGEX "trtllmGenKernels/.*")
list(FILTER SRC_CU EXCLUDE REGEX "trtllmGenKernels/.*")
list(FILTER SRC_CPP EXCLUDE REGEX "selectiveScan/.*")
list(FILTER SRC_CU EXCLUDE REGEX "selectiveScan/.*")
list(FILTER SRC_CPP EXCLUDE REGEX "userbuffers/.*")
list(FILTER SRC_CU EXCLUDE REGEX "userbuffers/.*")
list(FILTER SRC_CU EXCLUDE REGEX "fusedLayernormKernels/.*")

# When explicitly excluding SM120, also drop communicationKernels entirely to avoid
# generating compute_120 PTX from those very large all-reduce fusion kernels which
# currently ignore per-target architecture filtering and always pick up the global
# architecture list (leading to ptxas errors on unsupported instructions).
if(TLLM_EXCLUDE_SM120)
  list(LENGTH SRC_CPP _SRC_CPP_LEN_BEFORE)
  list(LENGTH SRC_CU _SRC_CU_LEN_BEFORE)
  list(FILTER SRC_CPP EXCLUDE REGEX "communicationKernels/.*")
  list(FILTER SRC_CU EXCLUDE REGEX "communicationKernels/.*")
  # Extra defensive removal in case some paths did not match regex
  set(_TMP_CPP)
  foreach(_f IN LISTS SRC_CPP)
    if(_f MATCHES "communicationKernels")
      continue()
    endif()
    list(APPEND _TMP_CPP ${_f})
  endforeach()
  set(SRC_CPP ${_TMP_CPP})
  set(_TMP_CU)
  foreach(_f IN LISTS SRC_CU)
    # Exclude files that use SM120+ features (FP8, FP4, advanced communication, etc.)
    if(_f MATCHES "communicationKernels" OR
       _f MATCHES "fusedMoeCommKernels" OR
       _f MATCHES "quantization" OR
       _f MATCHES "unfusedAttentionKernels.*fp4" OR
       _f MATCHES ".*fp8.*" OR
       _f MATCHES ".*fp4.*")
      continue()
    endif()
    list(APPEND _TMP_CU ${_f})
  endforeach()
  set(SRC_CU ${_TMP_CU})
  list(LENGTH SRC_CPP _SRC_CPP_LEN_AFTER)
  list(LENGTH SRC_CU _SRC_CU_LEN_AFTER)
  math(EXPR _CPP_REMOVED "${_SRC_CPP_LEN_BEFORE} - ${_SRC_CPP_LEN_AFTER}")
  math(EXPR _CU_REMOVED "${_SRC_CU_LEN_BEFORE} - ${_SRC_CU_LEN_AFTER}")
  message(STATUS "TLLM_EXCLUDE_SM120: removed ${_CPP_REMOVED} C++ and ${_CU_REMOVED} CUDA communicationKernels sources from kernels_src build")

  # Fatal check: verify no SM120-related sources remain
  foreach(_f IN LISTS SRC_CU)
    if(_f MATCHES "communicationKernels" OR
       _f MATCHES "fusedMoeCommKernels" OR
       _f MATCHES "quantization")
      message(FATAL_ERROR "TLLM_EXCLUDE_SM120 enabled but SM120-related source still present: ${_f}")
    endif()
  endforeach()
endif()

function(filter_cuda_archs ARCH SOURCES_VAR)
  if(NOT "${ARCH}" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)
    set(FILTER_REGEX
        ".*_sm(_)?${ARCH}[.]cubin[.]cpp|^.*Sm(_)?${ARCH}.*cubin.cpp$|.*_sm(_)?${ARCH}[.]cu|^.*Sm(_)?${ARCH}.*cu$"
    )
    list(APPEND SOURCES ${${SOURCES_VAR}})
    list(APPEND SOURCES_FILTERED ${SOURCES})
    list(FILTER SOURCES_FILTERED INCLUDE REGEX "${FILTER_REGEX}")
    list(LENGTH SOURCES_FILTERED SOURCES_FILTERED_LEN)
    message(
      STATUS
        "Excluding ${SOURCES_FILTERED_LEN} cubins for SM ${ARCH} from ${CMAKE_CURRENT_SOURCE_DIR}"
    )
    foreach(filtered_item ${SOURCES_FILTERED})
      message(VERBOSE "- ${filtered_item}")
    endforeach()
    list(FILTER SOURCES EXCLUDE REGEX "${FILTER_REGEX}")
    set(${SOURCES_VAR}
        "${SOURCES}"
        PARENT_SCOPE)
    add_compile_definitions("EXCLUDE_SM_${ARCH}")
  endif()
endfunction()

if(NOT ENABLE_MULTI_DEVICE)
  list(FILTER SRC_CU EXCLUDE REGEX "customAllReduceKernels*.*cu$")
endif()

add_library(kernels_src STATIC ${SRC_CPP} ${SRC_CU})
set_property(TARGET kernels_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET kernels_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(
  kernels_src
  PUBLIC
    $<TARGET_PROPERTY:${INTERNAL_CUTLASS_KERNELS_TARGET},INTERFACE_INCLUDE_DIRECTORIES>
)
if(TLLM_EXCLUDE_SM120)
  # Restrict kernels_src to architectures explicitly excluding 120 to prevent compute_120 PTX generation
  # Keep existing minimum arch (89) unless user supplied others via CMAKE_CUDA_ARCHITECTURES
  # Build explicit architecture list from the original numeric set, omitting 120 and avoiding *f family entries.
  set(_KERNELS_NUMERIC_ARCHS)
  foreach(ORIG IN LISTS CMAKE_CUDA_ARCHITECTURES_ORIG)
    if(ORIG STREQUAL "120")
      continue()
    endif()
    list(APPEND _KERNELS_NUMERIC_ARCHS ${ORIG})
  endforeach()
  if(_KERNELS_NUMERIC_ARCHS)
    set_cuda_architectures(kernels_src ${_KERNELS_NUMERIC_ARCHS})
  else()
    add_cuda_architectures(kernels_src 89)
  endif()
else()
  add_cuda_architectures(kernels_src 89)
endif()

# Provide arch specific compilation for Hopper (90) to satisfy archCondition.h which enforces
# __CUDA_ARCH_SPECIFIC__ or family specific for SM90+ when using CUDA 12.9 toolchain.
if("90" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)
  # Force arch specific 90 build for this aggregate target.
  set_cuda_architectures(kernels_src 90a)
endif()

add_subdirectory(cutlass_kernels)
add_subdirectory(flashMLA)
add_subdirectory(decoderMaskedMultiheadAttention)
add_subdirectory(selectiveScan)
add_subdirectory(userbuffers)
add_subdirectory(contextFusedMultiHeadAttention)
add_subdirectory(trtllmGenKernels)
add_subdirectory(fusedLayernormKernels)
add_subdirectory(groupRmsNormKernels)
add_subdirectory(llama4MinLatencyKernels)
add_subdirectory(dsv3MinLatencyKernels)
add_subdirectory(causalConv1d)
