import os
import subprocess
import sys

import pytest
<<<<<<< HEAD
from utils.util import skip_gpu_memory_less_than_80gb
=======
from utils.util import skip_gpu_memory_less_than_80gb, skip_pre_hopper
>>>>>>> upstream/main

from .openai_server import RemoteOpenAIServer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_llm import get_model_path


@pytest.fixture(scope="module")
<<<<<<< HEAD
def model_name():
    return "llama-3.1-model/Meta-Llama-3.1-8B"
=======
def model_name(request):
    return request.param
>>>>>>> upstream/main


@pytest.fixture(scope="module")
def model_path(model_name: str):
    return get_model_path(model_name)


@pytest.fixture(scope="module")
def server(model_path: str):
<<<<<<< HEAD
    # fix port to facilitate concise trtllm-serve examples
    with RemoteOpenAIServer(model_path, port=8000) as remote_server:
=======
    args = ["--kv_cache_free_gpu_memory_fraction", "0.8"]
    # fix port to facilitate concise trtllm-serve examples
    with RemoteOpenAIServer(model_path, cli_args=args,
                            port=8000) as remote_server:
>>>>>>> upstream/main
        yield remote_server


@pytest.fixture(scope="module")
def benchmark_root():
    llm_root = os.getenv("LLM_ROOT")
    return os.path.join(llm_root, "tensorrt_llm", "serve", "scripts")


def dataset_path(dataset_name: str):
    if dataset_name == "sharegpt":
        return get_model_path(
            "datasets/ShareGPT_V3_unfiltered_cleaned_split.json")
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")


@skip_gpu_memory_less_than_80gb
<<<<<<< HEAD
def test_trtllm_serve_benchmark(server: RemoteOpenAIServer, benchmark_root: str,
                                model_path: str):
    client_script = os.path.join(benchmark_root, "benchmark_serving.py")
    dataset = dataset_path("sharegpt")
    benchmark_cmd = [
        "python3", client_script, "--dataset-name", "sharegpt", "--model",
        "llama", "--dataset-path", dataset, "--tokenizer", model_path
    ]

    # CalledProcessError will be raised if any errors occur
    subprocess.run(benchmark_cmd,
                   stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE,
                   text=True,
                   check=True)
=======
@pytest.mark.parametrize("model_name", [
    "llama-3.1-model/Meta-Llama-3.1-8B",
    pytest.param("gpt_oss/gpt-oss-20b", marks=skip_pre_hopper)
],
                         indirect=True)
def test_trtllm_serve_benchmark(server: RemoteOpenAIServer, benchmark_root: str,
                                model_path: str):
    model_name = model_path.split("/")[-1]
    client_script = os.path.join(benchmark_root, "benchmark_serving.py")
    dataset = dataset_path("sharegpt")
    benchmark_cmd = [
        "python3",
        client_script,
        "--dataset-name",
        "sharegpt",
        "--model",
        model_name,
        "--dataset-path",
        dataset,
        "--tokenizer",
        model_path,
        "--temperature",
        "1.0",
        "--top-p",
        "1.0",
    ]

    # CalledProcessError will be raised if any errors occur
    result = subprocess.run(benchmark_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            check=True)
    assert result.returncode == 0
    assert "Serving Benchmark Result" in result.stdout
>>>>>>> upstream/main
