<<<<<<< HEAD
.. TensorRT-LLM documentation master file, created by
=======
.. TensorRT LLM documentation master file, created by
>>>>>>> upstream/main
   sphinx-quickstart on Wed Sep 20 08:35:21 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

<<<<<<< HEAD
Welcome to TensorRT-LLM's Documentation!
=======
Welcome to TensorRT LLM's Documentation!
>>>>>>> upstream/main
========================================

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :name: Getting Started

   overview.md
   quick-start-guide.md
<<<<<<< HEAD
   key-features.md
   torch.md
   release-notes.md


.. toctree::
   :maxdepth: 2
   :caption: Installation
   :name: Installation

   .. installation/overview.md

   installation/containers.md
   installation/linux.md
   installation/build-from-source-linux.md
=======
   installation/index.rst
>>>>>>> upstream/main


.. toctree::
   :maxdepth: 2
   :caption: Deployment Guide
   :name: Deployment Guide

<<<<<<< HEAD
   deployment-guide/quick-start-recipe-for-llama4-scout-on-trtllm.md
   deployment-guide/quick-start-recipe-for-deepseek-r1-on-trtllm.md
   deployment-guide/quick-start-recipe-for-llama3.3-70b-on-trtllm.md


.. toctree::
   :maxdepth: 2
   :caption: LLM API
   :hidden:
   :glob:

   llm-api/*


.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   examples/index.rst
   examples/customization.md
   examples/llm_api_examples
   examples/trtllm_serve_examples
=======
   examples/llm_api_examples.rst
   examples/trtllm_serve_examples
   examples/dynamo_k8s_example.rst
   deployment-guide/index.rst

.. toctree::
   :maxdepth: 2
   :caption: Models
   :name: Models

   models/supported-models.md
   models/adding-new-model.md

>>>>>>> upstream/main


.. toctree::
   :maxdepth: 2
<<<<<<< HEAD
   :caption: Model Definition API
   :hidden:

   python-api/tensorrt_llm.layers.rst
   python-api/tensorrt_llm.functional.rst
   python-api/tensorrt_llm.models.rst
   python-api/tensorrt_llm.plugin.rst
   python-api/tensorrt_llm.quantization.rst
   python-api/tensorrt_llm.runtime.rst


.. toctree::
   :maxdepth: 2
   :caption: C++ API
   :hidden:

   _cpp_gen/executor.rst
   _cpp_gen/runtime.rst


.. toctree::
   :maxdepth: 2
   :caption: Command-Line Reference
   :name: Command-Line Reference

   commands/trtllm-bench
   commands/trtllm-build
=======
   :caption: CLI Reference
   :name: CLI Reference

   commands/trtllm-bench
   commands/trtllm-eval
>>>>>>> upstream/main
   commands/trtllm-serve/index


.. toctree::
   :maxdepth: 2
<<<<<<< HEAD
   :caption: Architecture
   :name: Architecture

   architecture/overview.md
   architecture/core-concepts.md
   architecture/checkpoint.md
   architecture/workflow.md
   architecture/add-model.md

.. toctree::
   :maxdepth: 2
   :caption: Advanced
   :name: Advanced

   advanced/gpt-attention.md
   advanced/gpt-runtime.md
   advanced/executor.md
   advanced/graph-rewriting.md
   advanced/inference-request.md
   advanced/lora.md
   advanced/expert-parallelism.md
   advanced/kv-cache-management.md
   advanced/kv-cache-reuse.md
   advanced/speculative-decoding.md
   advanced/disaggregated-service.md

.. toctree::
   :maxdepth: 2
   :caption: Performance
   :name: Performance

   performance/perf-overview.md
   Benchmarking <performance/perf-benchmarking.md>
   performance/performance-tuning-guide/index
   performance/perf-analysis.md
=======
   :caption: API Reference

   llm-api/index.md
   llm-api/reference.rst
>>>>>>> upstream/main


.. toctree::
   :maxdepth: 2
<<<<<<< HEAD
   :caption: Reference
   :name: Reference

   reference/troubleshooting.md
   reference/support-matrix.md

   .. reference/upgrading.md

   reference/precision.md
   reference/memory.md
   reference/ci-overview.md
   reference/dev-containers.md
=======
   :caption: Features

   features/feature-combination-matrix.md
   features/attention.md
   features/disagg-serving.md
   features/kvcache.md
   features/long-sequence.md
   features/lora.md
   features/multi-modality.md
   features/overlap-scheduler.md
   features/paged-attention-ifb-scheduler.md
   features/parallel-strategy.md
   features/quantization.md
   features/sampling.md
   features/speculative-decoding.md
   features/checkpoint-loading.md
   features/auto_deploy/auto-deploy.md

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer-guide/overview.md
   developer-guide/perf-analysis.md
   developer-guide/perf-benchmarking.md
   developer-guide/ci-overview.md
   developer-guide/dev-containers.md
>>>>>>> upstream/main


.. toctree::
   :maxdepth: 2
   :caption: Blogs
<<<<<<< HEAD
   :hidden:

   blogs/H100vsA100.md
   blogs/H200launch.md
   blogs/Falcon180B-H200.md
   blogs/quantization-in-TRT-LLM.md
   blogs/XQA-kernel.md
   blogs/tech_blog/blog1_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.md
   blogs/tech_blog/blog2_DeepSeek_R1_MTP_Implementation_and_Optimization.md

=======
   :glob:

   blogs/tech_blog/*
   blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md
   blogs/H200launch.md
   blogs/XQA-kernel.md
   blogs/H100vsA100.md


.. toctree::
   :maxdepth: 2
   :caption: Quick Links

   Releases <https://github.com/NVIDIA/TensorRT-LLM/releases>
   Github Code <https://github.com/NVIDIA/TensorRT-LLM>
   Roadmap <https://github.com/NVIDIA/TensorRT-LLM/issues?q=is%3Aissue%20state%3Aopen%20label%3Aroadmap>

.. toctree::
   :maxdepth: 2
   :caption: Use TensorRT Engine
   :hidden:

   legacy/tensorrt_quickstart.md
>>>>>>> upstream/main

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
