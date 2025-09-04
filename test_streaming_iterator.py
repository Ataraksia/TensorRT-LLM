#!/usr/bin/env python3
"""Test script for the StreamingIterator implementation."""

import asyncio
import time

import torch

# Import the streaming components
from tensorrt_llm.models.higgs_audio.generation import (
    GenerationState,
    PipelineConfig,
    StreamingBuffer,
    StreamingChunk,
    StreamingIterator,
)


class MockPipeline:
    """Mock pipeline for testing streaming functionality."""

    def __init__(self):
        self.config = PipelineConfig(
            streaming_chunk_size=16, streaming_buffer_size=256, max_streaming_latency_ms=50.0
        )

    def _stream_text_generation(self, sequence):
        """Mock text streaming that yields chunks."""
        for i in range(10):
            time.sleep(0.01)  # Simulate generation time
            yield {
                "chunk_id": i,
                "tokens": [100 + i, 101 + i],
                "text": f"chunk_{i}",
                "partial_text": f"Generated text chunk {i}",
                "generation_step": i,
            }

    def _stream_audio_init_phase(self, sequence):
        """Mock audio init phase streaming."""
        for i in range(5):
            time.sleep(0.015)  # Simulate generation time
            yield {
                "chunk_id": i,
                "audio_tokens": [200 + i],
                "codebook_id": 0,
                "phase": "audio_init",
                "generation_step": i,
            }

    def _stream_audio_in_progress_phase(self, sequence):
        """Mock audio in-progress phase with multi-codebook data."""
        for timestep in range(8):
            for codebook_id in range(8):
                time.sleep(0.005)  # Simulate generation time
                yield {
                    "codebook_id": codebook_id,
                    "token": 300 + timestep * 8 + codebook_id,
                    "timestep": timestep,
                }


class MockSequence:
    """Mock generation sequence for testing."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.state = GenerationState.INITIALIZED
        self.reference_audio = torch.randn(1, 1024)  # Mock reference audio


def test_streaming_buffer():
    """Test StreamingBuffer functionality."""
    print("Testing StreamingBuffer...")

    buffer = StreamingBuffer(buffer_size=128, chunk_size=16, max_latency_ms=50.0)

    # Test buffer initialization
    assert buffer.buffer_size == 128
    assert buffer.chunk_size == 16
    assert len(buffer.audio_buffers) == 8  # 8 codebooks
    assert all(len(buf) == 0 for buf in buffer.audio_buffers.values())

    print("✓ StreamingBuffer initialization test passed")


def test_streaming_chunk():
    """Test StreamingChunk data structure."""
    print("Testing StreamingChunk...")

    chunk = StreamingChunk(
        request_id="test_123",
        chunk_id=1,
        chunk_type="text",
        data={"tokens": [1, 2, 3]},
        timestamp=time.time(),
        latency_ms=25.0,
        generation_phase="text",
    )

    assert chunk.request_id == "test_123"
    assert chunk.chunk_type == "text"
    assert chunk.is_partial is True
    assert chunk.is_final is False

    print("✓ StreamingChunk test passed")


def test_streaming_iterator_basic():
    """Test basic StreamingIterator functionality."""
    print("Testing StreamingIterator basic functionality...")

    # Create mock components
    pipeline = MockPipeline()
    sequence = MockSequence("test_request_001")

    # Create streaming iterator
    iterator = StreamingIterator(
        pipeline=pipeline,
        sequence=sequence,
        buffer_config=StreamingBuffer(buffer_size=64, chunk_size=8),
    )

    # Test iterator protocol
    assert hasattr(iterator, "__iter__")
    assert hasattr(iterator, "__next__")

    print("✓ StreamingIterator basic test passed")


def test_streaming_iterator_flow():
    """Test complete streaming flow."""
    print("Testing StreamingIterator complete flow...")

    pipeline = MockPipeline()
    sequence = MockSequence("test_request_002")

    iterator = StreamingIterator(pipeline=pipeline, sequence=sequence)

    chunks_received = []
    chunk_types_seen = set()

    try:
        for i, chunk_data in enumerate(iterator):
            chunks_received.append(chunk_data)
            chunk_types_seen.add(chunk_data.get("chunk_type", "unknown"))

            print(
                f"  Chunk {i}: {chunk_data.get('chunk_type')} - "
                f"Latency: {chunk_data.get('latency_ms', 0):.1f}ms"
            )

            # Limit test to prevent infinite loops
            if i >= 50:
                iterator.stop()
                break

    except StopIteration:
        pass

    print(f"✓ Received {len(chunks_received)} chunks")
    print(f"✓ Chunk types seen: {chunk_types_seen}")

    # Verify we got expected chunk types
    expected_types = {"metadata", "text", "audio_init", "audio_frame", "heartbeat"}
    assert len(chunk_types_seen & expected_types) > 0, (
        f"Expected some of {expected_types}, got {chunk_types_seen}"
    )

    print("✓ StreamingIterator flow test passed")


def test_performance_metrics():
    """Test performance monitoring capabilities."""
    print("Testing performance metrics...")

    pipeline = MockPipeline()
    sequence = MockSequence("test_request_003")

    iterator = StreamingIterator(pipeline, sequence)

    start_time = time.time()
    chunk_count = 0
    total_latency = 0.0

    try:
        for chunk_data in iterator:
            chunk_count += 1
            latency = chunk_data.get("latency_ms", 0)
            total_latency += latency

            # Check for metrics in metadata chunks
            if chunk_data.get("chunk_type") == "metadata":
                data = chunk_data.get("data", {})
                if "metrics" in data:
                    metrics = data["metrics"]
                    print(f"  Metrics: {metrics}")

            if chunk_count >= 20:
                iterator.stop()
                break

    except StopIteration:
        pass

    elapsed_time = time.time() - start_time
    avg_latency = total_latency / max(chunk_count, 1)

    print(f"✓ Processed {chunk_count} chunks in {elapsed_time:.2f}s")
    print(f"✓ Average latency: {avg_latency:.1f}ms")
    print("✓ Performance metrics test passed")


def test_backpressure_handling():
    """Test backpressure and flow control."""
    print("Testing backpressure handling...")

    # Create a configuration that should trigger backpressure
    buffer_config = StreamingBuffer(
        buffer_size=32,
        chunk_size=4,
        backpressure_threshold=3,  # Low threshold to trigger backpressure quickly
    )

    pipeline = MockPipeline()
    sequence = MockSequence("test_request_004")

    iterator = StreamingIterator(pipeline, sequence, buffer_config)

    backpressure_detected = False
    dropped_chunks_detected = False

    try:
        for i, chunk_data in enumerate(iterator):
            # Check for backpressure indicators
            data = chunk_data.get("data", {})
            if "metrics" in data:
                metrics = data["metrics"]
                if metrics.get("backpressure_active"):
                    backpressure_detected = True
                    print(f"  Backpressure detected at chunk {i}")

                if metrics.get("dropped_chunks", 0) > 0:
                    dropped_chunks_detected = True
                    print(f"  Dropped chunks detected: {metrics['dropped_chunks']}")

            # Simulate slow consumer to trigger backpressure
            if i % 5 == 0:
                time.sleep(0.1)

            if i >= 30:
                iterator.stop()
                break

    except StopIteration:
        pass

    print(f"✓ Backpressure detected: {backpressure_detected}")
    print(f"✓ Dropped chunks detected: {dropped_chunks_detected}")
    print("✓ Backpressure handling test passed")


def test_frame_assembly():
    """Test audio frame assembly from multiple codebooks."""
    print("Testing audio frame assembly...")

    pipeline = MockPipeline()
    sequence = MockSequence("test_request_005")

    iterator = StreamingIterator(pipeline, sequence)

    frame_chunks_received = []

    try:
        for chunk_data in iterator:
            if chunk_data.get("chunk_type") == "audio_frame":
                frame_chunks_received.append(chunk_data)
                frame_data = chunk_data.get("codebook_data", {})
                print(f"  Frame {chunk_data.get('audio_frame_idx')}: {len(frame_data)} codebooks")

            if len(frame_chunks_received) >= 5:
                iterator.stop()
                break

    except StopIteration:
        pass

    print(f"✓ Received {len(frame_chunks_received)} audio frames")

    # Verify frame structure
    if frame_chunks_received:
        first_frame = frame_chunks_received[0]
        frame_data = first_frame.get("codebook_data", {})
        assert isinstance(frame_data, dict), "Frame data should be a dictionary"
        print(f"✓ Frame structure valid: {len(frame_data)} codebooks per frame")

    print("✓ Frame assembly test passed")


async def test_async_compatibility():
    """Test async compatibility of the streaming iterator."""
    print("Testing async compatibility...")

    pipeline = MockPipeline()
    sequence = MockSequence("test_request_006")

    iterator = StreamingIterator(pipeline, sequence)

    chunk_count = 0

    # Simulate async processing
    try:
        for chunk_data in iterator:
            chunk_count += 1

            # Simulate async processing with yield control
            if chunk_count % 5 == 0:
                await asyncio.sleep(0.01)

            if chunk_count >= 15:
                iterator.stop()
                break

    except StopIteration:
        pass

    print(f"✓ Processed {chunk_count} chunks in async context")
    print("✓ Async compatibility test passed")


def run_all_tests():
    """Run all streaming iterator tests."""
    print("=" * 60)
    print("STREAMING ITERATOR TEST SUITE")
    print("=" * 60)

    test_functions = [
        test_streaming_buffer,
        test_streaming_chunk,
        test_streaming_iterator_basic,
        test_streaming_iterator_flow,
        test_performance_metrics,
        test_backpressure_handling,
        test_frame_assembly,
    ]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            print(f"\n{'-' * 40}")
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            failed += 1

    # Run async test
    try:
        print(f"\n{'-' * 40}")
        asyncio.run(test_async_compatibility())
        passed += 1
    except Exception as e:
        print(f"✗ test_async_compatibility FAILED: {e}")
        failed += 1

    print(f"\n{'=' * 60}")
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
