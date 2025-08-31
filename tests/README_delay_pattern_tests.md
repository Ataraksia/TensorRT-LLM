# Higgs Audio Delay Pattern Tests

This directory contains comprehensive tests for the Higgs Audio delay pattern functionality, which is critical for RVQ (Residual Vector Quantization) codebook generation in TTS models.

## Overview

The delay pattern system enables:
- **Simultaneous Generation**: Multiple audio codebooks generate tokens simultaneously
- **Streaming Support**: Real-time audio generation with controlled latency
- **Token Coordination**: Proper synchronization across multiple codebooks
- **Memory Efficiency**: Optimized memory usage for large-scale generation

## Test Structure

### Core Test Files

- **`test_higgs_audio_delay_patterns.py`**: Main test suite with comprehensive coverage
- **`conftest.py`**: Shared fixtures, utilities, and test configuration
- **`README_delay_pattern_tests.md`**: This documentation file

### Test Classes

#### 1. `TestDelayPatternProvider`
Tests the core DelayPatternProvider class functionality:
- Pattern generation strategies (linear, exponential, custom, none)
- Pattern application and reversal
- Input validation and error handling
- Fallback mechanisms

#### 2. `TestAudioTokenUtils`
Tests audio token management utilities:
- Multi-codebook token splitting and merging
- Token validation and statistics
- Cross-codebook synchronization analysis
- Various input format support

#### 3. `TestDelayAwareAttentionUtils`
Tests delay-aware attention mechanisms:
- Attention mask generation
- Codebook routing masks
- Position ID adjustment for delays
- Streaming attention support

#### 4. `TestIntegrationScenarios`
Integration tests for complete workflows:
- End-to-end delay pattern workflows
- Streaming coordination scenarios
- Error recovery and fallback testing
- Performance with large configurations

#### 5. `TestComplexScenarios`
Complex real-world testing:
- Multi-codebook streaming coordination
- Memory constraint validation
- Cross-codebook synchronization quality
- Mixed strategy fallback scenarios

#### 6. `TestRealWorldUseCases`
Realistic TTS scenario testing:
- Voice cloning scenarios
- Real-time streaming TTS
- Batch processing scenarios
- Production-like configurations

## Test Categories

### Unit Tests (Default)
Basic functionality tests for individual components:
```bash
pytest tests/test_higgs_audio_delay_patterns.py -m unit
```

### Integration Tests
Tests that validate component interaction:
```bash
pytest tests/test_higgs_audio_delay_patterns.py -m integration --run-integration
```

### Real-World Scenario Tests
Tests with realistic TTS configurations:
```bash
pytest tests/test_higgs_audio_delay_patterns.py -m real_world --run-real-world
```

### Slow Tests
Tests that may take significant time:
```bash
pytest tests/test_higgs_audio_delay_patterns.py -m slow --run-slow
```

## Running Tests

### Quick Tests (Basic Functionality)
```bash
cd tests
python test_higgs_audio_delay_patterns.py quick
```

### Integration Tests
```bash
cd tests
python test_higgs_audio_delay_patterns.py integration
```

### Real-World Scenario Tests
```bash
cd tests
python test_higgs_audio_delay_patterns.py real-world
```

### Full Test Suite
```bash
cd tests
python test_higgs_audio_delay_patterns.py
```

### Using pytest directly
```bash
# Basic tests only
pytest tests/test_higgs_audio_delay_patterns.py -v

# Include slow tests
pytest tests/test_higgs_audio_delay_patterns.py -v --run-slow

# Include integration tests
pytest tests/test_higgs_audio_delay_patterns.py -v --run-integration

# Include all test types
pytest tests/test_higgs_audio_delay_patterns.py -v --run-slow --run-integration --run-real-world
```

## Test Configuration

### Delay Pattern Strategies Tested

1. **Linear Strategy**: `[0, 1, 2, 3, ...]`
   - Simple incremental delays
   - Predictable latency patterns
   - Good for real-time applications

2. **Exponential Strategy**: `[0, 1, 3, 7, ...]` (`2^n - 1`)
   - Exponentially increasing delays
   - Better quality for complex audio
   - Used in high-quality TTS scenarios

3. **Custom Strategy**: User-defined delays
   - Flexible delay patterns
   - Optimized for specific use cases
   - Requires careful configuration

4. **None Strategy**: All zeros
   - No delay between codebooks
   - Fastest generation
   - May reduce audio quality

### Codebook Configurations Tested

- **Small**: 2-4 codebooks (basic audio quality)
- **Medium**: 6-8 codebooks (good audio quality)
- **Large**: 16+ codebooks (high audio quality)

### Sequence Length Scenarios

- **Short**: 16-64 tokens (real-time streaming)
- **Medium**: 128-512 tokens (typical TTS)
- **Long**: 1024+ tokens (long-form audio)

### Batch Size Scenarios

- **Single**: Batch size 1 (individual requests)
- **Small Batch**: 2-8 (typical inference)
- **Large Batch**: 16+ (high-throughput scenarios)

## Mock Framework

The tests use a comprehensive mock framework to avoid TensorRT-LLM dependencies:

### MockTensor Class
- Mimics PyTorch tensor behavior
- Supports common operations (view, reshape, transpose, etc.)
- Provides numpy-compatible data access
- Includes arithmetic operations and comparisons

### Mock Fixtures
- `sample_config`: Standard HiggsAudioConfig
- `sample_audio_tokens`: Multi-codebook token examples
- `sample_delay_patterns`: Pre-generated delay patterns
- `mock_torch_functions`: Mock torch operations

## Test Data

### Audio Token Format
```python
# Unified format: [batch_size, total_sequence_length]
# Interleaved: [t0_cb0, t0_cb1, t0_cb2, t0_cb3, t1_cb0, t1_cb1, ...]
unified_tokens = [[10, 20, 30, 40, 11, 21, 31, 41]]

# Per-codebook format: List[Tensor[batch_size, time_steps]]
per_codebook_tokens = [
    [[10, 11]],  # Codebook 0
    [[20, 21]],  # Codebook 1
    [[30, 31]],  # Codebook 2
    [[40, 41]]   # Codebook 3
]
```

### Delay Pattern Format
```python
# Pattern: [n_codebooks, sequence_length]
delay_pattern = [
    [0, 0, 0, 0, 0],  # Codebook 0: no delay
    [1, 1, 1, 1, 1],  # Codebook 1: 1-step delay
    [2, 2, 2, 2, 2],  # Codebook 2: 2-step delay
    [3, 3, 3, 3, 3]   # Codebook 3: 3-step delay
]
```

## Error Scenarios Tested

### DelayPatternProvider Errors
- Invalid strategy names
- Invalid parameter combinations
- Memory constraint violations
- Pattern consistency failures

### AudioTokenUtils Errors
- Invalid token formats
- Codebook count mismatches
- Shape inconsistencies
- Token range violations

### AttentionUtils Errors
- Invalid attention configurations
- Delay pattern incompatibilities
- Memory allocation failures
- Streaming constraint violations

## Performance Testing

### Memory Usage Validation
- Reasonable memory limits for production
- Warning generation for large configurations
- Automatic fallback for extreme cases

### Latency Considerations
- Real-time streaming constraints (max delay ≤ 4)
- Batch processing efficiency
- Cross-codebook synchronization overhead

### Throughput Scenarios
- Single request processing
- Concurrent batch processing
- Resource utilization patterns

## Validation Metrics

### Pattern Quality Metrics
- Pattern consistency across sequences
- Delay distribution analysis
- Memory efficiency measurements

### Token Synchronization Metrics
- Cross-codebook alignment quality
- Padding distribution analysis
- Special token handling validation

### Error Recovery Metrics
- Fallback success rates
- Warning generation accuracy
- Graceful degradation behavior

## Expected Test Results

### Unit Test Success Criteria
- All basic functionality tests pass
- Error handling works correctly
- Input validation catches invalid cases
- Mock framework operates correctly

### Integration Test Success Criteria
- End-to-end workflows complete successfully
- Component interactions work as expected
- Performance meets reasonable benchmarks
- Error recovery functions properly

### Real-World Test Success Criteria
- Realistic scenarios execute without errors
- Memory usage stays within reasonable bounds
- Latency meets real-time requirements
- Quality metrics meet TTS standards

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the parent directory is in the Python path
2. **Mock Framework Issues**: Check that all TensorRT-LLM dependencies are properly mocked
3. **Test Timeouts**: Use `--run-slow` flag for tests that may take longer
4. **Memory Issues**: Large test configurations may require more RAM

### Debug Mode
```bash
pytest tests/test_higgs_audio_delay_patterns.py -v -s --tb=long
```

### Test Coverage
```bash
pytest tests/test_higgs_audio_delay_patterns.py --cov=tensorrt_llm.models.higgs_audio
```

## Contributing

When adding new tests:

1. Follow the existing test structure and naming conventions
2. Add appropriate markers (`@pytest.mark.unit`, `@pytest.mark.integration`, etc.)
3. Include comprehensive error case testing
4. Document expected behavior and edge cases
5. Update this README if adding new test categories

## Dependencies

### Required Packages
- `pytest >= 7.0.0`
- `numpy >= 1.20.0`
- `unittest.mock` (built-in)

### Optional Packages
- `pytest-cov` (for coverage reports)
- `pytest-xdist` (for parallel testing)
- `pytest-benchmark` (for performance testing)

## Test Maintenance

### Regular Tasks
- Update test data when model architecture changes
- Verify mock framework compatibility with new TensorRT-LLM versions
- Review and update performance benchmarks
- Validate test coverage for new functionality

### Version Compatibility
- Tests are designed to work with Python 3.8+
- Mock framework should be updated if TensorRT-LLM API changes
- Consider backwards compatibility when adding new test cases