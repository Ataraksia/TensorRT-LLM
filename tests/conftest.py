"""
Configuration and fixtures for Higgs Audio delay pattern tests.

This file provides common test configuration, fixtures, and utilities
used across the delay pattern test suite.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock
import numpy as np

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def mock_tensorrt_imports():
    """Mock TensorRT-LLM imports that might not be available in test environment."""
    
    # Mock the tensorrt_llm modules
    with patch.dict('sys.modules', {
        'tensorrt_llm': MagicMock(),
        'tensorrt_llm.functional': MagicMock(),
        'tensorrt_llm.layers': MagicMock(),
        'tensorrt_llm.module': MagicMock(),
        'tensorrt_llm.runtime': MagicMock(),
        'tensorrt_llm.models': MagicMock(),
    }):
        yield


@pytest.fixture
def sample_config():
    """Provide a sample HiggsAudioConfig for testing."""
    from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
    
    config = MagicMock(spec=HiggsAudioConfig)
    config.audio_num_codebooks = 4
    config.audio_delay_pattern_strategy = 'linear'
    config.audio_delay_pattern_stride = 1
    config.audio_delay_pattern_max_delay = None
    config.audio_delay_pattern_custom_delays = None
    config.hidden_size = 4096
    config.num_hidden_layers = 32
    config.vocab_size = 32000
    config.pad_token_id = 0
    config.eos_token_id = 2
    config.audio_start_token_id = 128010
    config.audio_end_token_id = 128011
    
    return config


@pytest.fixture
def sample_audio_tokens():
    """Provide sample audio tokens for testing."""
    # Create multi-codebook audio tokens
    return {
        'unified': np.array([
            [10, 20, 30, 40, 11, 21, 31, 41],  # Batch 0: interleaved tokens
            [50, 60, 70, 80, 51, 61, 71, 81]   # Batch 1: interleaved tokens
        ]),
        'per_codebook': [
            np.array([[10, 11], [50, 51]]),  # Codebook 0
            np.array([[20, 21], [60, 61]]),  # Codebook 1
            np.array([[30, 31], [70, 71]]),  # Codebook 2
            np.array([[40, 41], [80, 81]])   # Codebook 3
        ]
    }


@pytest.fixture
def sample_delay_patterns():
    """Provide sample delay patterns for testing."""
    return {
        'linear_4x8': np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],  # Codebook 0: no delay
            [1, 1, 1, 1, 1, 1, 1, 1],  # Codebook 1: 1-step delay
            [2, 2, 2, 2, 2, 2, 2, 2],  # Codebook 2: 2-step delay
            [3, 3, 3, 3, 3, 3, 3, 3]   # Codebook 3: 3-step delay
        ]),
        'exponential_4x8': np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],  # Codebook 0: 0 delay
            [1, 1, 1, 1, 1, 1, 1, 1],  # Codebook 1: 1 delay
            [3, 3, 3, 3, 3, 3, 3, 3],  # Codebook 2: 3 delay
            [7, 7, 7, 7, 7, 7, 7, 7]   # Codebook 3: 7 delay
        ]),
        'custom_4x8': np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],  # Codebook 0: custom 0
            [2, 2, 2, 2, 2, 2, 2, 2],  # Codebook 1: custom 2
            [1, 1, 1, 1, 1, 1, 1, 1],  # Codebook 2: custom 1
            [4, 4, 4, 4, 4, 4, 4, 4]   # Codebook 3: custom 4
        ])
    }


class MockTensor:
    """Enhanced mock tensor class for comprehensive testing."""
    
    def __init__(self, data, dtype=None, device='cpu'):
        if isinstance(data, (list, tuple)):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array([data])
        
        self._shape = self.data.shape
        self.dtype = dtype or 'float32'
        self.device = device
    
    @property
    def shape(self):
        return self._shape
    
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
        return MockTensor(self.data.copy(), self.dtype, self.device)
    
    def cpu(self):
        return MockTensor(self.data.copy(), self.dtype, 'cpu')
    
    def cuda(self):
        return MockTensor(self.data.copy(), self.dtype, 'cuda')
    
    def to(self, device):
        return MockTensor(self.data.copy(), self.dtype, device)
    
    def unsqueeze(self, dim):
        new_data = np.expand_dims(self.data, dim)
        return MockTensor(new_data, self.dtype, self.device)
    
    def squeeze(self, dim=None):
        if dim is None:
            new_data = np.squeeze(self.data)
        else:
            new_data = np.squeeze(self.data, axis=dim)
        return MockTensor(new_data, self.dtype, self.device)
    
    def view(self, *shape):
        new_data = self.data.reshape(shape)
        return MockTensor(new_data, self.dtype, self.device)
    
    def reshape(self, *shape):
        return self.view(*shape)
    
    def transpose(self, dim0, dim1):
        new_data = np.swapaxes(self.data, dim0, dim1)
        return MockTensor(new_data, self.dtype, self.device)
    
    def permute(self, *dims):
        new_data = np.transpose(self.data, dims)
        return MockTensor(new_data, self.dtype, self.device)
    
    def __getitem__(self, key):
        return MockTensor(self.data[key], self.dtype, self.device)
    
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
    
    def __eq__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data == other.data, 'bool', self.device)
        return MockTensor(self.data == other, 'bool', self.device)
    
    def __ne__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data != other.data, 'bool', self.device)
        return MockTensor(self.data != other, 'bool', self.device)
    
    def __lt__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data < other.data, 'bool', self.device)
        return MockTensor(self.data < other, 'bool', self.device)
    
    def __le__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data <= other.data, 'bool', self.device)
        return MockTensor(self.data <= other, 'bool', self.device)
    
    def __gt__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data > other.data, 'bool', self.device)
        return MockTensor(self.data > other, 'bool', self.device)
    
    def __ge__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data >= other.data, 'bool', self.device)
        return MockTensor(self.data >= other, 'bool', self.device)
    
    def sum(self, dim=None, keepdim=False):
        result = np.sum(self.data, axis=dim, keepdims=keepdim)
        return MockTensor(result, self.dtype, self.device)
    
    def mean(self, dim=None, keepdim=False):
        result = np.mean(self.data, axis=dim, keepdims=keepdim)
        return MockTensor(result, self.dtype, self.device)
    
    def max(self, dim=None, keepdim=False):
        if dim is None:
            result = np.max(self.data)
        else:
            result = np.max(self.data, axis=dim, keepdims=keepdim)
        return MockTensor(result, self.dtype, self.device)
    
    def min(self, dim=None, keepdim=False):
        if dim is None:
            result = np.min(self.data)
        else:
            result = np.min(self.data, axis=dim, keepdims=keepdim)
        return MockTensor(result, self.dtype, self.device)
    
    def all(self, dim=None, keepdim=False):
        result = np.all(self.data, axis=dim, keepdims=keepdim)
        return MockTensor(result, 'bool', self.device)
    
    def any(self, dim=None, keepdim=False):
        result = np.any(self.data, axis=dim, keepdims=keepdim)
        return MockTensor(result, 'bool', self.device)


def create_mock_tensor(data, dtype=None, device='cpu'):
    """Helper function to create MockTensor instances."""
    return MockTensor(data, dtype, device)


def create_mock_constant(data):
    """Helper function to create mock constant tensors."""
    return MockTensor(data)


@pytest.fixture
def mock_torch_functions():
    """Mock common torch functions used in the implementation."""
    
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
    
    def mock_cat(tensors, dim=0):
        arrays = [t.data if isinstance(t, MockTensor) else t for t in tensors]
        result = np.concatenate(arrays, axis=dim)
        return MockTensor(result)
    
    def mock_stack(tensors, dim=0):
        arrays = [t.data if isinstance(t, MockTensor) else t for t in tensors]
        result = np.stack(arrays, axis=dim)
        return MockTensor(result)
    
    def mock_tril(input_tensor, diagonal=0):
        data = input_tensor.data if isinstance(input_tensor, MockTensor) else input_tensor
        result = np.tril(data, k=diagonal)
        return MockTensor(result)
    
    def mock_triu(input_tensor, diagonal=0):
        data = input_tensor.data if isinstance(input_tensor, MockTensor) else input_tensor
        result = np.triu(data, k=diagonal)
        return MockTensor(result)
    
    mock_funcs = {
        'torch.zeros': mock_zeros,
        'torch.ones': mock_ones,
        'torch.full': mock_full,
        'torch.cat': mock_cat,
        'torch.stack': mock_stack,
        'torch.tril': mock_tril,
        'torch.triu': mock_triu,
    }
    
    return mock_funcs


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Automatically set up the test environment for all tests."""
    
    # Suppress warnings during testing
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Set numpy random seed for reproducible tests
    np.random.seed(42)
    
    yield
    
    # Cleanup after tests
    pass


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may take several seconds)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "real_world: marks tests as real-world scenario tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add markers based on test class names
        if "Integration" in item.cls.__name__ if item.cls else "":
            item.add_marker(pytest.mark.integration)
        elif "RealWorld" in (item.cls.__name__ if item.cls else ""):
            item.add_marker(pytest.mark.real_world)
        elif "Complex" in (item.cls.__name__ if item.cls else ""):
            item.add_marker(pytest.mark.slow)
        else:
            item.add_marker(pytest.mark.unit)


# Pytest command line options
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow", action="store_true", default=False,
        help="run slow tests"
    )
    parser.addoption(
        "--run-integration", action="store_true", default=False,
        help="run integration tests"
    )
    parser.addoption(
        "--run-real-world", action="store_true", default=False,
        help="run real-world scenario tests"
    )


def pytest_runtest_setup(item):
    """Skip tests based on command line options."""
    if "slow" in item.keywords and not item.config.getoption("--run-slow"):
        pytest.skip("need --run-slow option to run slow tests")
    
    if "integration" in item.keywords and not item.config.getoption("--run-integration"):
        pytest.skip("need --run-integration option to run integration tests")
    
    if "real_world" in item.keywords and not item.config.getoption("--run-real-world"):
        pytest.skip("need --run-real-world option to run real-world tests")