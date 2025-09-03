"""Test fixtures and utilities for Higgs Audio TTS model testing.

This module provides comprehensive fixtures, mock utilities, and test infrastructure
for validating the complete Higgs Audio TTS implementation. It supports testing
of all major components including generation mode management, DualFFN architecture,
CUDA graph optimizations, and performance validation.
"""

import json
import os
import sys
import tempfile
import time
import warnings
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import pytest

# Add tensorrt_llm to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

try:
    from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig

    TENSORRT_LLM_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"TensorRT-LLM imports not available: {e}")
    TENSORRT_LLM_AVAILABLE = False
    HiggsAudioConfig = None


class MockTensor:
    """Enhanced mock tensor class for comprehensive TTS testing."""

    def __init__(self, data, dtype=None, device="cpu", requires_grad=False):
        if isinstance(data, (list, tuple)):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data.copy()
        else:
            self.data = np.array([data])

        self._shape = self.data.shape
        self.dtype = dtype or "float32"
        self.device = device
        self.requires_grad = requires_grad
        self._grad = None

    @property
    def shape(self):
        return self._shape

    @property
    def grad(self):
        return self._grad

    def dim(self):
        return len(self._shape)

    def size(self, dim=None):
        if dim is None:
            return self.data.size
        return self._shape[dim] if dim < len(self._shape) else 1

    def item(self):
        return self.data.item() if self.data.size == 1 else self.data

    def numpy(self):
        return self.data.copy()

    def detach(self):
        return MockTensor(self.data.copy(), self.dtype, self.device, False)

    def cpu(self):
        return MockTensor(self.data.copy(), self.dtype, "cpu", self.requires_grad)

    def cuda(self, device=None):
        device_str = f"cuda:{device}" if device is not None else "cuda"
        return MockTensor(self.data.copy(), self.dtype, device_str, self.requires_grad)

    def to(self, device_or_dtype):
        if isinstance(device_or_dtype, str):
            return MockTensor(self.data.copy(), self.dtype, device_or_dtype, self.requires_grad)
        else:
            return MockTensor(self.data.copy(), device_or_dtype, self.device, self.requires_grad)

    def unsqueeze(self, dim):
        new_data = np.expand_dims(self.data, dim)
        return MockTensor(new_data, self.dtype, self.device, self.requires_grad)

    def squeeze(self, dim=None):
        if dim is None:
            new_data = np.squeeze(self.data)
        else:
            new_data = np.squeeze(self.data, axis=dim)
        return MockTensor(new_data, self.dtype, self.device, self.requires_grad)

    def view(self, *shape):
        new_data = self.data.reshape(shape)
        return MockTensor(new_data, self.dtype, self.device, self.requires_grad)

    def reshape(self, *shape):
        return self.view(*shape)

    def transpose(self, dim0, dim1):
        new_data = np.swapaxes(self.data, dim0, dim1)
        return MockTensor(new_data, self.dtype, self.device, self.requires_grad)

    def permute(self, *dims):
        new_data = np.transpose(self.data, dims)
        return MockTensor(new_data, self.dtype, self.device, self.requires_grad)

    def expand(self, *sizes):
        new_data = np.broadcast_to(self.data, sizes)
        return MockTensor(new_data, self.dtype, self.device, self.requires_grad)

    def repeat(self, *sizes):
        new_data = np.tile(self.data, sizes)
        return MockTensor(new_data, self.dtype, self.device, self.requires_grad)

    def __getitem__(self, key):
        return MockTensor(self.data[key], self.dtype, self.device, self.requires_grad)

    def __setitem__(self, key, value):
        if isinstance(value, MockTensor):
            self.data[key] = value.data
        else:
            self.data[key] = value

    def __add__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data + other.data, self.dtype, self.device)
        return MockTensor(self.data + other, self.dtype, self.device)

    def __sub__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data - other.data, self.dtype, self.device)
        return MockTensor(self.data - other, self.dtype, self.device)

    def __mul__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data * other.data, self.dtype, self.device)
        return MockTensor(self.data * other, self.dtype, self.device)

    def __truediv__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data / other.data, self.dtype, self.device)
        return MockTensor(self.data / other, self.dtype, self.device)

    def __eq__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data == other.data, "bool", self.device)
        return MockTensor(self.data == other, "bool", self.device)

    def __ne__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data != other.data, "bool", self.device)
        return MockTensor(self.data != other, "bool", self.device)

    def sum(self, dim=None, keepdim=False):
        result = np.sum(self.data, axis=dim, keepdims=keepdim)
        return MockTensor(result, self.dtype, self.device)

    def mean(self, dim=None, keepdim=False):
        result = np.mean(self.data, axis=dim, keepdims=keepdim)
        return MockTensor(result, self.dtype, self.device)

    def max(self, dim=None, keepdim=False):
        if dim is None:
            result = np.max(self.data)
            return MockTensor(result, self.dtype, self.device)
        else:
            result = np.max(self.data, axis=dim, keepdims=keepdim)
            return MockTensor(result, self.dtype, self.device)

    def min(self, dim=None, keepdim=False):
        if dim is None:
            result = np.min(self.data)
            return MockTensor(result, self.dtype, self.device)
        else:
            result = np.min(self.data, axis=dim, keepdims=keepdim)
            return MockTensor(result, self.dtype, self.device)

    def all(self, dim=None, keepdim=False):
        result = np.all(self.data, axis=dim, keepdims=keepdim)
        return MockTensor(result, "bool", self.device)

    def any(self, dim=None, keepdim=False):
        result = np.any(self.data, axis=dim, keepdims=keepdim)
        return MockTensor(result, "bool", self.device)

    def clone(self):
        return MockTensor(self.data.copy(), self.dtype, self.device, self.requires_grad)

    def contiguous(self):
        return MockTensor(
            np.ascontiguousarray(self.data), self.dtype, self.device, self.requires_grad
        )


@pytest.fixture
def mock_tensor():
    """Factory function for creating MockTensor instances."""
    return MockTensor


@pytest.fixture
def sample_higgs_audio_config():
    """Create a sample HiggsAudioConfig for testing."""
    if not TENSORRT_LLM_AVAILABLE:
        pytest.skip("TensorRT-LLM not available")

    # Create a mock mapping object
    mock_mapping = MagicMock()
    mock_mapping.tp_size = 1
    mock_mapping.tp_rank = 0
    mock_mapping.tp_group = None
    mock_mapping.pp_size = 1
    mock_mapping.pp_rank = 0
    mock_mapping.cp_size = 1
    mock_mapping.cp_rank = 0
    mock_mapping.cp_group = None
    mock_mapping.is_first_pp_rank.return_value = True
    mock_mapping.is_last_pp_rank.return_value = True
    mock_mapping.prev_pp_rank.return_value = 0
    mock_mapping.next_pp_rank.return_value = 0

    config = HiggsAudioConfig(
        # Core architecture
        num_hidden_layers=4,  # Small for testing
        num_attention_heads=8,
        hidden_size=512,
        intermediate_size=1536,
        vocab_size=32000,
        max_position_embeddings=2048,
        # Audio configuration
        audio_num_codebooks=4,
        audio_codebook_size=1024,
        audio_delay_pattern_strategy="linear",
        audio_delay_pattern_stride=1,
        audio_dual_ffn_layers=[1, 3],  # Enable DualFFN on layers 1 and 3
        # TTS-specific
        audio_realtime_mode=True,
        use_delay_pattern=True,
        # Mock dependencies
        mapping=mock_mapping,
        dtype="float16",
    )

    return config


@pytest.fixture
def sample_text_tokens(mock_tensor):
    """Sample text token sequences for testing."""
    return {
        "single_batch": mock_tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype="int64"),
        "multi_batch": mock_tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype="int64"),
        "long_sequence": mock_tensor([[i for i in range(1, 33)]], dtype="int64"),
        "with_padding": mock_tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]], dtype="int64"),
    }


@pytest.fixture
def sample_audio_tokens(mock_tensor):
    """Sample audio token sequences for multi-codebook testing."""
    return {
        "unified_format": mock_tensor(
            [
                [10, 20, 30, 40, 11, 21, 31, 41, 12, 22, 32, 42],  # 4 codebooks, 3 time steps
                [50, 60, 70, 80, 51, 61, 71, 81, 52, 62, 72, 82],
            ],
            dtype="int64",
        ),
        "per_codebook_format": [
            mock_tensor([[10, 11, 12], [50, 51, 52]], dtype="int64"),  # Codebook 0
            mock_tensor([[20, 21, 22], [60, 61, 62]], dtype="int64"),  # Codebook 1
            mock_tensor([[30, 31, 32], [70, 71, 72]], dtype="int64"),  # Codebook 2
            mock_tensor([[40, 41, 42], [80, 81, 82]], dtype="int64"),  # Codebook 3
        ],
        "streaming_chunks": [
            mock_tensor([[100, 110, 120, 130]], dtype="int64"),  # Chunk 0
            mock_tensor([[140, 150, 160, 170]], dtype="int64"),  # Chunk 1
            mock_tensor([[180, 190, 200, 210]], dtype="int64"),  # Chunk 2
        ],
    }


@pytest.fixture
def sample_attention_masks(mock_tensor):
    """Sample attention masks for testing."""
    return {
        "full_attention": mock_tensor([[1, 1, 1, 1, 1]], dtype="bool"),
        "with_padding": mock_tensor([[1, 1, 1, 0, 0]], dtype="bool"),
        "multi_batch": mock_tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype="bool"),
        "causal_mask": mock_tensor(
            [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]], dtype="bool"
        ),
    }


@pytest.fixture
def sample_audio_features(mock_tensor):
    """Sample audio feature embeddings for testing."""
    return {
        "whisper_features": mock_tensor(np.random.randn(2, 1500, 1280), dtype="float16"),
        "short_audio": mock_tensor(np.random.randn(1, 100, 1280), dtype="float16"),
        "batch_audio": mock_tensor(np.random.randn(4, 500, 1280), dtype="float16"),
    }


@pytest.fixture
def mock_tensorrt_functions():
    """Mock TensorRT-LLM functional operations."""
    mocks = {}

    # Mock tensor creation functions
    def mock_constant(data):
        return MockTensor(data)

    def mock_zeros(*size, dtype=None, device=None):
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            size = size[0]
        return MockTensor(np.zeros(size), dtype, device)

    def mock_ones(*size, dtype=None, device=None):
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            size = size[0]
        return MockTensor(np.ones(size), dtype, device)

    def mock_full(size, fill_value, dtype=None, device=None):
        return MockTensor(np.full(size, fill_value), dtype, device)

    # Mock tensor operations
    def mock_cat(tensors, dim=0):
        arrays = [t.data if isinstance(t, MockTensor) else t for t in tensors]
        result = np.concatenate(arrays, axis=dim)
        return MockTensor(result)

    def mock_stack(tensors, dim=0):
        arrays = [t.data if isinstance(t, MockTensor) else t for t in tensors]
        result = np.stack(arrays, axis=dim)
        return MockTensor(result)

    def mock_max(tensor, dim=None):
        data = tensor.data if isinstance(tensor, MockTensor) else tensor
        if dim is None:
            return MockTensor(np.max(data))
        return MockTensor(np.max(data, axis=dim))

    def mock_where(condition, x, y):
        cond_data = condition.data if isinstance(condition, MockTensor) else condition
        x_data = x.data if isinstance(x, MockTensor) else x
        y_data = y.data if isinstance(y, MockTensor) else y
        result = np.where(cond_data, x_data, y_data)
        return MockTensor(result)

    def mock_cast(tensor, dtype):
        data = tensor.data if isinstance(tensor, MockTensor) else tensor
        return MockTensor(data.astype(np.float32 if "float" in str(dtype) else np.int64), dtype)

    mocks.update(
        {
            "constant": mock_constant,
            "zeros": mock_zeros,
            "ones": mock_ones,
            "full": mock_full,
            "cat": mock_cat,
            "stack": mock_stack,
            "max": mock_max,
            "where": mock_where,
            "cast": mock_cast,
        }
    )

    return mocks


@pytest.fixture
def mock_layers():
    """Mock TensorRT-LLM layer classes for testing."""

    class MockEmbedding:
        def __init__(self, num_embeddings, embedding_dim, **kwargs):
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.weight = MockTensor(np.random.randn(num_embeddings, embedding_dim))

        def forward(self, input_ids):
            # Simple embedding lookup simulation
            batch_size, seq_len = input_ids.shape
            return MockTensor(np.random.randn(batch_size, seq_len, self.embedding_dim))

    class MockRmsNorm:
        def __init__(self, normalized_shape, eps=1e-5, **kwargs):
            self.normalized_shape = normalized_shape
            self.eps = eps

        def forward(self, x):
            # Identity operation for testing
            return x

        def __call__(self, x):
            return self.forward(x)

    class MockAttention:
        def __init__(self, **kwargs):
            self.config = kwargs

        def forward(self, hidden_states, **kwargs):
            # Return same shape with slight modification
            return hidden_states

        def __call__(self, hidden_states, **kwargs):
            result = self.forward(hidden_states, **kwargs)
            if kwargs.get("use_cache", False):
                # Return (output, kv_cache) tuple
                mock_cache = MagicMock()
                return (result, mock_cache)
            return result

    class MockGatedMLP:
        def __init__(self, hidden_size, ffn_hidden_size, **kwargs):
            self.hidden_size = hidden_size
            self.ffn_hidden_size = ffn_hidden_size

        def forward(self, x, **kwargs):
            # Identity operation for testing
            return x

        def __call__(self, x, **kwargs):
            return self.forward(x, **kwargs)

    class MockColumnLinear:
        def __init__(self, in_features, out_features, **kwargs):
            self.in_features = in_features
            self.out_features = out_features

        def forward(self, x):
            batch_size, seq_len, _ = x.shape
            return MockTensor(np.random.randn(batch_size, seq_len, self.out_features))

        def __call__(self, x):
            return self.forward(x)

    return {
        "Embedding": MockEmbedding,
        "RmsNorm": MockRmsNorm,
        "Attention": MockAttention,
        "GatedMLP": MockGatedMLP,
        "ColumnLinear": MockColumnLinear,
    }


@pytest.fixture
def performance_benchmarker():
    """Utility for measuring and validating performance improvements."""

    class PerformanceBenchmarker:
        def __init__(self):
            self.measurements = {}

        def measure_latency(self, func, *args, **kwargs):
            """Measure function execution latency."""
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            return result, latency_ms

        def measure_memory_usage(self, func, *args, **kwargs):
            """Mock memory usage measurement."""
            # In real implementation, this would use memory profiling
            result = func(*args, **kwargs)
            mock_memory_mb = np.random.randint(100, 500)  # Mock memory usage
            return result, mock_memory_mb

        def measure_throughput(self, func, batch_sizes, *args, **kwargs):
            """Measure throughput across different batch sizes."""
            throughputs = {}
            for batch_size in batch_sizes:
                # Mock throughput measurement
                throughputs[batch_size] = batch_size * np.random.randint(50, 100)
            return throughputs

        def validate_improvement(
            self, baseline_metric, optimized_metric, expected_improvement_pct, tolerance_pct=5
        ):
            """Validate that optimized metric shows expected improvement."""
            actual_improvement_pct = ((baseline_metric - optimized_metric) / baseline_metric) * 100
            lower_bound = expected_improvement_pct - tolerance_pct
            upper_bound = expected_improvement_pct + tolerance_pct

            return lower_bound <= actual_improvement_pct <= upper_bound, actual_improvement_pct

        def record_measurement(self, name: str, value: float, metadata: Dict[str, Any] = None):
            """Record a measurement for later analysis."""
            self.measurements[name] = {
                "value": value,
                "timestamp": time.time(),
                "metadata": metadata or {},
            }

        def get_summary(self) -> Dict[str, Any]:
            """Get summary of all measurements."""
            return {
                "total_measurements": len(self.measurements),
                "measurements": self.measurements.copy(),
            }

    return PerformanceBenchmarker()


@pytest.fixture
def mock_cuda_graphs():
    """Mock CUDA graph functionality for testing."""

    class MockCudaGraphManager:
        def __init__(self, config, **kwargs):
            self.config = config
            self.graphs = {}
            self.is_initialized = True

        def prewarm_graphs(self, batch_sizes, sequence_lengths):
            """Mock graph prewarming."""
            for bs in batch_sizes:
                for seq_len in sequence_lengths:
                    key = f"bs_{bs}_seq_{seq_len}"
                    self.graphs[key] = f"mock_graph_{key}"
            return True

        def get_graph(self, batch_size, sequence_length):
            """Get mock graph for execution."""
            key = f"bs_{batch_size}_seq_{sequence_length}"
            return self.graphs.get(key, None)

        def execute_graph(self, graph_key, inputs):
            """Mock graph execution."""
            return inputs  # Identity operation for testing

        def cleanup(self):
            """Mock cleanup."""
            self.graphs.clear()

    return MockCudaGraphManager


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock model files
        config_file = os.path.join(temp_dir, "config.json")
        with open(config_file, "w") as f:
            json.dump(
                {
                    "model_type": "higgs_audio",
                    "text_config": {
                        "num_hidden_layers": 4,
                        "num_attention_heads": 8,
                        "hidden_size": 512,
                        "intermediate_size": 1536,
                        "vocab_size": 32000,
                        "max_position_embeddings": 2048,
                    },
                    "audio_encoder_config": {
                        "num_mel_bins": 128,
                        "encoder_layers": 4,
                        "encoder_attention_heads": 8,
                        "encoder_ffn_dim": 1024,
                        "d_model": 512,
                        "max_source_positions": 1500,
                    },
                },
                f,
            )

        yield temp_dir


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Automatically setup test environment."""
    # Suppress warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # Set random seeds for reproducibility
    np.random.seed(42)

    # Mock TensorRT-LLM imports if not available
    if not TENSORRT_LLM_AVAILABLE:
        mock_modules = [
            "tensorrt_llm",
            "tensorrt_llm.functional",
            "tensorrt_llm.layers",
            "tensorrt_llm.module",
            "tensorrt_llm.parameter",
            "tensorrt_llm.models",
            "tensorrt_llm.models.modeling_utils",
            "tensorrt_llm.runtime",
        ]

        for module_name in mock_modules:
            if module_name not in sys.modules:
                sys.modules[module_name] = MagicMock()

    yield

    # Cleanup after tests
    pass


def pytest_configure(config):
    """Configure pytest with custom markers for TTS testing."""
    config.addinivalue_line("markers", "integration: integration tests for TTS components")
    config.addinivalue_line("markers", "performance: performance validation tests")
    config.addinivalue_line("markers", "end_to_end: end-to-end TTS pipeline tests")
    config.addinivalue_line("markers", "slow: slow-running tests")
    config.addinivalue_line("markers", "cuda_graphs: CUDA graph optimization tests")
    config.addinivalue_line("markers", "dualffn: DualFFN architecture tests")
    config.addinivalue_line("markers", "generation_modes: generation mode management tests")
    config.addinivalue_line("markers", "streaming: streaming TTS tests")
    config.addinivalue_line("markers", "config: configuration and compatibility tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add markers based on test class/function names
        test_name = item.name.lower()

        if "integration" in test_name or "Integration" in (item.cls.__name__ if item.cls else ""):
            item.add_marker(pytest.mark.integration)

        if "performance" in test_name or "Performance" in (item.cls.__name__ if item.cls else ""):
            item.add_marker(pytest.mark.performance)

        if "end_to_end" in test_name or "EndToEnd" in (item.cls.__name__ if item.cls else ""):
            item.add_marker(pytest.mark.end_to_end)

        if "cuda_graph" in test_name or "CudaGraph" in (item.cls.__name__ if item.cls else ""):
            item.add_marker(pytest.mark.cuda_graphs)

        if "dualffn" in test_name or "DualFFN" in (item.cls.__name__ if item.cls else ""):
            item.add_marker(pytest.mark.dualffn)

        if "generation_mode" in test_name or "GenerationMode" in (
            item.cls.__name__ if item.cls else ""
        ):
            item.add_marker(pytest.mark.generation_modes)

        if "streaming" in test_name or "Streaming" in (item.cls.__name__ if item.cls else ""):
            item.add_marker(pytest.mark.streaming)

        if "config" in test_name or "Config" in (item.cls.__name__ if item.cls else ""):
            item.add_marker(pytest.mark.config)
