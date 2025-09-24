(product-overview)=

# Overview

<<<<<<< HEAD
## About TensorRT-LLM

[TensorRT-LLM](https://developer.nvidia.com/tensorrt) accelerates and optimizes inference performance for the latest large language models (LLMs) on NVIDIA GPUs. This open-source library is available for free on the [TensorRT-LLM GitHub repo](https://github.com/NVIDIA/TensorRT-LLM) and as part of the [NVIDIA NeMo framework](https://www.nvidia.com/en-us/ai-data-science/generative-ai/nemo-framework/).

LLMs have revolutionized the field of artificial intelligence and created entirely new ways of interacting with the digital world. But, as organizations and application developers around the world look to incorporate LLMs into their work, some of the challenges with running these models become apparent. Put simply, LLMs are large. That fact can make them expensive and slow to run without the right techniques.

TensorRT-LLM offers a comprehensive library for compiling and optimizing LLMs for inference. TensorRT-LLM incorporates all of the optimizations (that is, kernel fusion and quantization, runtime optimizations like C++ implementations, KV caching, continuous in-flight batching, and paged attention) and more, while providing an intuitive Model Definition API for defining and building new models.

Some of the major benefits that TensorRT-LLM provides are:

### Common LLM Support

TensorRT-LLM supports the latest LLMs. Refer to the {ref}`support-matrix-software` for the full list.

### In-Flight Batching and Paged Attention

{ref}`inflight-batching` takes advantage of the overall text generation process for an LLM can be broken down into multiple iterations of execution on the model. Rather than waiting for the whole batch to finish before moving on to the next set of requests, the TensorRT-LLM runtime immediately evicts finished sequences from the batch. It then begins executing new requests while other requests are still in flight. It's a {ref}`executor` that aims at reducing wait times in queues, eliminating the need for padding requests, and allowing for higher GPU utilization.

### Multi-GPU Multi-Node Inference

TensorRT-LLM consists of pre– and post-processing steps and multi-GPU multi-node communication primitives in a simple, open-source Model Definition API for groundbreaking LLM inference performance on GPUs. Refer to the {ref}`multi-gpu-multi-node` section for more information.

### FP8 Support

[NVIDIA H100 GPUs](https://www.nvidia.com/en-us/data-center/dgx-h100/) with TensorRT-LLM give you the ability to convert model weights into a new FP8 format easily and compile models to take advantage of optimized FP8 kernels automatically. This is made possible through [NVIDIA Hopper](https://blogs.nvidia.com/blog/h100-transformer-engine/) and done without having to change any model code.

### Latest GPU Support

TensorRT-LLM supports GPUs based on the NVIDIA Hopper, NVIDIA Ada Lovelace, and NVIDIA Ampere architectures.
Certain limitations might apply. Refer to the {ref}`support-matrix` for more information.

### Native Windows Support

Windows platform support is deprecated as of v0.18.0. All Windows-related code and functionality will be completely removed in future releases.

## What Can You Do With TensorRT-LLM?

Let TensorRT-LLM accelerate inference performance on the latest LLMs on NVIDIA GPUs. Use TensorRT-LLM as an optimization backbone for LLM inference in NVIDIA NeMo, an end-to-end framework to build, customize, and deploy generative AI applications into production. NeMo provides complete containers, including TensorRT-LLM and NVIDIA Triton, for generative AI deployments.

TensorRT-LLM improves ease of use and extensibility through an open-source modular Model Definition API for defining, optimizing, and executing new architectures and enhancements as LLMs evolve, and can be customized easily.

If you’re eager to dive into the world of LLMs, now is the time to get started with TensorRT-LLM. Explore its capabilities, experiment with different models and optimizations, and embark on your journey to unlock the incredible power of AI-driven language models. To get started, refer to the {ref}`quick-start-guide`.
=======
## About TensorRT LLM

[TensorRT LLM](https://developer.nvidia.com/tensorrt) is NVIDIA's comprehensive open-source library for accelerating and optimizing inference performance of the latest large language models (LLMs) on NVIDIA GPUs. 

## Key Capabilities

### 🔥 **Architected on Pytorch**

TensorRT LLM provides a high-level Python [LLM API](./quick-start-guide.md#run-offline-inference-with-llm-api) that supports a wide range of inference setups - from single-GPU to multi-GPU or multi-node deployments. It includes built-in support for various parallelism strategies and advanced features. The LLM API integrates seamlessly with the broader inference ecosystem, including NVIDIA [Dynamo](https://github.com/ai-dynamo/dynamo) and the [Triton Inference Server](https://github.com/triton-inference-server/server).

TensorRT LLM is designed to be modular and easy to modify. Its PyTorch-native architecture allows developers to experiment with the runtime or extend functionality. Several popular models are also pre-defined and can be customized using [native PyTorch code](source:tensorrt_llm/_torch/models/modeling_deepseekv3.py), making it easy to adapt the system to specific needs.

### ⚡ **State-of-the-Art Performance**

TensorRT LLM delivers breakthrough performance on the latest NVIDIA GPUs:

- **DeepSeek R1**: [World-record inference performance on Blackwell GPUs](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-world-record-deepseek-r1-inference-performance/)
- **Llama 4 Maverick**: [Breaks the 1,000 TPS/User Barrier on B200 GPUs](https://developer.nvidia.com/blog/blackwell-breaks-the-1000-tps-user-barrier-with-metas-llama-4-maverick/)

### 🎯 **Comprehensive Model Support**

TensorRT LLM supports the latest and most popular LLM architectures:

### FP4 Support
[NVIDIA B200 GPUs](https://www.nvidia.com/en-us/data-center/dgx-b200/) , when used with TensorRT LLM, enable seamless loading of model weights in the new [FP4 format](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/#what_is_nvfp4), allowing you to automatically leverage optimized FP4 kernels for efficient and accurate low-precision inference.

### FP8 Support

TensorRT LLM strives to support the most popular models on **Day 0**.

### 🚀 **Advanced Optimization & Production Features**
- **In-Flight Batching & Paged Attention**: {ref}`inflight-batching` eliminates wait times by dynamically managing request execution, processing context and generation phases together for maximum GPU utilization and reduced latency.
- **Multi-GPU Multi-Node Inference**: Seamless distributed inference with tensor, pipeline, and expert parallelism across multiple GPUs and nodes through the Model Definition API.
- **Advanced Quantization**: 
  - **FP4 Quantization**: Native support on NVIDIA B200 GPUs with optimized FP4 kernels
  - **FP8 Quantization**: Automatic conversion on NVIDIA H100 GPUs leveraging Hopper architecture
- **Speculative Decoding**: Multiple algorithms including EAGLE, MTP and NGram
- **KV Cache Management**: Paged KV cache with intelligent block reuse and memory optimization
- **Chunked Prefill**: Efficient handling of long sequences by splitting context into manageable chunks
- **LoRA Support**: Multi-adapter support with HuggingFace and NeMo formats, efficient fine-tuning and adaptation
- **Checkpoint Loading**: Flexible model loading from various formats (HuggingFace, NeMo, custom)
- **Guided Decoding**: Advanced sampling with stop words, bad words, and custom constraints
- **Disaggregated Serving (Beta)**: Separate context and generation phases across different GPUs for optimal resource utilization

### 🔧 **Latest GPU Architecture Support**

TensorRT LLM supports the full spectrum of NVIDIA GPU architectures:
- **NVIDIA Blackwell**: B200, GB200, RTX Pro 6000 SE with FP4 optimization
- **NVIDIA Hopper**: H100, H200,GH200 with FP8 acceleration
- **NVIDIA Ada Lovelace**: L40/L40S, RTX 40 series with FP8 acceleration
- **NVIDIA Ampere**: A100, RTX 30 series for production workloads

## What Can You Do With TensorRT LLM?

Whether you're building the next generation of AI applications, optimizing existing LLM deployments, or exploring the frontiers of large language model technology, TensorRT LLM provides the tools, performance, and flexibility you need to succeed in the era of generative AI.To get started, refer to the {ref}`quick-start-guide`.
>>>>>>> upstream/main
