#!/bin/bash
<<<<<<< HEAD

echo "Please find the \`disaggr_torch.slurm\` script in the \`examples/disaggregated/slurm/benchmark/\` directory."

partition=<partition>
account=<account>
job_name=<job_name>
container_image=<container_image>
mounts=<mounts>  # e.g. /mnt/data:/mnt/data
workdir=<workdir>  # Path to disaggr_torch.slurm
model_dir=<model_dir>  # Path to the model checkpoint
repo_dir=<repo_dir>  # Path to the repo to install TensorRT-LLM, if this is empty, the pre-installed version will be used

mtp_size=0
ntasks_per_node=4 # 4 GPUs per GB200 node, 8 GPUs per B200 node

isl=1024
osl=1024
multi_round=10
streaming=true
benchmark_mode=e2e

# dep16 eplb0, 256, 288
for b in 1 64 1024; do
    for eplb_num_slots in 0 256 288; do
        concurrency=$((b * 16))
        ctx_node_num=$(((concurrency + 5499)/5500)) # $(((concurrency + 10999)/11000)) for B200
        ctx_num=${ctx_node_num} # $((ctx_node_num * 2)) for B200
        total_node_num=$((ctx_node_num + 4)) # $((ctx_node_num + 2)) for B200
        ntasks=$((total_node_num * ntasks_per_node))

        args=(
            ${ctx_num} 4 4 4480 true "0.85"   # Context servers arguments
            1 16 1024 1024 true "0.7"       # Generation servers arguments
            $eplb_num_slots $mtp_size  # Other arguments
            $concurrency               # Benchmarking arguments
            $isl
            $osl
            $multi_round
            $streaming
            $container_image           # User specific arguments
            $mounts
            $workdir
            $model_dir
            $benchmark_mode
            $repo_dir
        )

        sbatch --nodes=${total_node_num} \
            --ntasks=${ntasks} \
            --ntasks-per-node=${ntasks_per_node} \
            --partition=${partition} \
            --account=${account} \
            --job-name=${job_name} \
            --gres=gpu:${ntasks_per_node} \
            --segment=${total_node_num} \
            ${workdir}/disaggr_torch.slurm "${args[@]}"
    done
done

# dep32 eplb288
for b in 512; do
    concurrency=$((b * 32))
    ctx_node_num=$(((concurrency + 5499)/5500)) # $(((concurrency + 10999)/11000)) for B200
    ctx_num=${ctx_node_num} # $((ctx_node_num * 2)) for B200
    total_node_num=$((ctx_node_num + 8)) # $((ctx_node_num + 4)) for B200
    ntasks=$((total_node_num * ntasks_per_node))
    eplb_num_slots=288

    args=(
        ${ctx_num} 4 4 4480 true "0.85"   # Context servers arguments
        1 32 1024 1024 true "0.7"  # Generation servers arguments
        $eplb_num_slots $mtp_size  # Other arguments
        $concurrency               # Benchmarking arguments
        $isl
        $osl
        $multi_round
        $streaming
        $container_image           # User specific arguments
        $mounts
        $workdir
        $model_dir
        $benchmark_mode
        $repo_dir
    )

    sbatch --nodes=${total_node_num} \
        --ntasks=${ntasks} \
        --ntasks-per-node=${ntasks_per_node} \
        --partition=${partition} \
        --account=${account} \
        --job-name=${job_name} \
        --gres=gpu:${ntasks_per_node} \
        --segment=${total_node_num} \
        ${workdir}/disaggr_torch.slurm "${args[@]}"
done
=======
set -euo pipefail

echo "Please find the \`disaggr_torch.slurm\` script in the \`examples/disaggregated/slurm/benchmark/\` directory."

# Configuration
slurm_file="disaggr_torch.slurm"

# SLURM Configuration
partition="<partition>"
account="<account>"
job_time="02:00:00"
job_name="<job_name>"

##############################################################
# User Configuration - Review and edit the following variables

numa_bind=true
benchmark_mode="e2e" # e2e or gen_only

# Hardware Configuration
gpus_per_node=4  # Modify this with your hardware configuration

# Benchmark Configuration
use_nv_sa_benchmark=false   # Whether to use NVIDIA SA benchmark script instead of default one
isl=1024                   # Input sequence length
osl=1024                   # Output sequence length
multi_round=10              # Number of benchmark rounds
benchmark_ratio=0.8        # Benchmark ratio
streaming=true             # Enable streaming mode
cache_max_tokens=4608     # Cache transceiver max tokens
seq_offset=203  # Offset added to sequence lengths
# Dataset file for benchmarking
dataset_file="<dataset_file>"

# Environment Configuration
# Directories mount to the container
container_mount="<container_mount>" # path1:path1,path2:path2
# Container image
container_image="<container_image>"
# Path to the model directory
model_path="<model_path>"
# Path to the TensorRT-LLM repository
trtllm_repo="<trtllm_repo>"
# Set to true to do a clean build of TensorRT-LLM from source
build_wheel=false

# Workspace Configuration
work_dir=$(pwd) # path to the work directory containing the scripts

# Profiling Configuration
nsys_on=false  # Set to true to enable profiling

##############################################################

# Check if SLURM file exists
if [[ ! -f "${slurm_file}" ]]; then
    echo "Error: SLURM script '${slurm_file}' not found" >&2
    exit 1
fi

# Validate required paths
[[ ! -d "${model_path}" ]] && { echo "Error: model_path not found: ${model_path}" >&2; exit 1; }
[[ ! -d "${work_dir}" ]] && { echo "Error: work_dir '${work_dir}' not found" >&2; exit 1; }
[[ ! -f "${dataset_file}" ]] && { echo "Error: dataset_file '${dataset_file}' not found" >&2; exit 1; }

# Calculate required nodes based on tensor parallel size and server count
calc_nodes() {
    local tp_size=$1
    local num_servers=$2
    echo $(( (tp_size + gpus_per_node - 1) / gpus_per_node * num_servers ))
}

# Submit a single benchmark job
run_single() {
    # Context server params
    local ctx_num=$1
    local ctx_tp_size=$2
    local ctx_pp_size=$3
    local ctx_batch_size=$4
    local ctx_max_num_tokens=$5
    local ctx_enable_attention_dp=$6
    local ctx_gpu_frac=$7
    # Generation server params
    local gen_num=$8
    local gen_tp_size=$9
    local gen_pp_size=${10}
    local gen_batch_size=${11}
    local gen_max_num_tokens=${12}
    local gen_enable_attention_dp=${13}
    local gen_gpu_frac=${14}
    local gen_eplb_num_slots=${15}
    local mtp_size=${16}
    local gen_concurrency_list=${17}

    # Calculate total nodes needed
    local gen_nodes=$(calc_nodes "$gen_tp_size" "$gen_num")
    local ctx_nodes=$(calc_nodes "$ctx_tp_size" "$ctx_num")
    local total_nodes=$((ctx_nodes + gen_nodes))
    local total_tasks=$((total_nodes * gpus_per_node))

    # Handle SLURM reservation if needed
    local reservation_str=""
    [[ $gen_eplb_num_slots -gt 0 ]] && reservation_str="--reservation=sla_res_fw_11"

    # Submit job
    set -x
    sbatch \
        --partition="${partition}" \
        --gres=gpu:${gpus_per_node} \
        --account="${account}" \
        --time="${job_time}" \
        --job-name="${job_name}" \
        --nodes="${total_nodes}" \
        --ntasks="${total_tasks}" \
        --ntasks-per-node="${gpus_per_node}" \
        --segment="${total_nodes}" \
        ${reservation_str} \
        "${slurm_file}" \
        "${ctx_num}" "${ctx_tp_size}" "${ctx_pp_size}" "${ctx_batch_size}" "${ctx_max_num_tokens}" "${ctx_enable_attention_dp}" "${ctx_gpu_frac}" \
        "${gen_num}" "${gen_tp_size}" "${gen_pp_size}" "${gen_batch_size}" "${gen_max_num_tokens}" "${gen_enable_attention_dp}" "${gen_gpu_frac}" \
        "${gen_eplb_num_slots}" "${mtp_size}" "${gen_concurrency_list}" \
        "${gpus_per_node}" "${use_nv_sa_benchmark}" "${isl}" "${osl}" "${multi_round}" "${benchmark_ratio}" \
        "${streaming}" "${cache_max_tokens}" "${dataset_file}" "${container_mount}" "${container_image}" \
        "${model_path}" "${trtllm_repo}" "${build_wheel}" "${work_dir}" "${nsys_on}" "${seq_offset}" "${numa_bind}" "${benchmark_mode}"
    set +x
}

# Example benchmark configuration
#          |------------------- context -----------------|  |---------------------- generation ----------------------|
#           num  tp  pp  batch  tokens  attn_dp  gpu_frac    num  tp  pp  batch  tokens  attn_dp  gpu_frac  eplb  mtp  concurrency
# 1k-1k
run_single  1    4   1   4      4608    true     0.85        1    16  1   64     256     true     "0.7"     0    3    "512 1075"
run_single  2    4   1   4      4608    true     0.85        1    16  1   128    256     true     "0.7"     0    1    "2150"
run_single  1    4   1   4      4608    true     0.85        1    32  1   16     64      true     "0.6"     0    3    "512"
run_single  1    4   1   4      4608    true     0.85        1    32  1   32     32      true     "0.7"     0    0    "1075"
run_single  1    4   1   4      4608    true     0.85        1    16  1   64     64      true     "0.75"    0    0    "1075"
run_single  2    4   1   4      4608    true     0.85        1    16  1   256    256     true     "0.75"    0    0    "2048 4300"
>>>>>>> upstream/main
