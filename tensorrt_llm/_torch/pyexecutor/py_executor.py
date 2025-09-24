import dataclasses
import datetime
import functools
import gc
<<<<<<< HEAD
import heapq
import os
import queue
=======
import os
>>>>>>> upstream/main
import threading
import time
import traceback
import weakref
<<<<<<< HEAD
from collections import namedtuple
from contextlib import contextmanager
from itertools import chain
from typing import Dict, List, Optional, Tuple, Union

import dill  # nosec B403
import numpy as np
import torch

from tensorrt_llm._utils import (global_mpi_rank, is_trace_enabled, nvtx_range,
                                 trace_func)
from tensorrt_llm.bindings.executor import (DisServingRequestStats,
                                            FinishReason, InflightBatchingStats,
                                            IterationStats, KvCacheStats,
                                            RequestStage, RequestStats,
                                            RequestType, StaticBatchingStats)
from tensorrt_llm.bindings.internal.batch_manager import (LlmRequestType,
                                                          ReqIdsSet)
from tensorrt_llm.logger import logger

from ..distributed import Distributed
from .kv_cache_transceiver import KvCacheTransceiver
from .llm_request import (ExecutorRequest, ExecutorResponse, LlmRequest,
                          LlmRequestState, executor_request_to_llm_request)
from .model_engine import ModelEngine
from .sampler import Sampler, SampleState, SampleStateTensors
from .scheduler import ScheduledRequests
=======
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

from tensorrt_llm._torch.pyexecutor.resource_manager import (
    ResourceManagerType, request_context)
from tensorrt_llm._torch.pyexecutor.seq_slot_manager import SeqSlotManager
from tensorrt_llm._utils import (customized_gc_thresholds, global_mpi_rank,
                                 is_trace_enabled, nvtx_range, trace_func)
from tensorrt_llm.bindings.executor import (DisServingRequestStats,
                                            FinishReason, InflightBatchingStats,
                                            IterationStats, KvCacheStats,
                                            PeftCacheConfig, RequestStage,
                                            RequestStats, SpecDecodingStats,
                                            StaticBatchingStats)
from tensorrt_llm.bindings.internal.batch_manager import (LlmRequestType,
                                                          ReqIdsSet)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import CpType
from tensorrt_llm.runtime.generation import CUASSERT

from ..distributed import Distributed
from ..models.modeling_utils import DecoderModelForCausalLM
from ..modules.decoder_layer import DecoderLayer
from ..speculative.drafter import Drafter
from .executor_request_queue import ExecutorRequestQueue, RequestQueueItem
from .guided_decoder import GuidedDecoder
from .handle_logits import HandleLogits
from .kv_cache_connector import KvCacheConnectorManager
from .kv_cache_transceiver import KvCacheTransceiver
from .llm_request import (ExecutorRequest, LlmRequest, LlmRequestState,
                          LlmResponse)
from .model_engine import ModelEngine
from .sampler import Sampler, SampleState, SampleStateTensors
from .scheduler import RequestScheduler, ScheduledRequests
>>>>>>> upstream/main

# Environment variable to specify iteration ranges for profiling start/stop.
# Format: "start1-stop1,start2-stop2,..." or single iterations "iter1,iter2,..."
PROFILE_START_STOP_ENV_VAR_NAME = "TLLM_PROFILE_START_STOP"

# Environment variable to enable garbage collection profiling.
# Set to "1" to enable recording of garbage collection events during profiling.
PROFILE_RECORD_GC_ENV_VAR_NAME = "TLLM_PROFILE_RECORD_GC"

# Environment variable to enable PyTorch profiler tracing.
# Set to a path to save detailed tracing of PyTorch operations.
PROFILE_TRACE_ENV_VAR_NAME = "TLLM_TORCH_PROFILE_TRACE"

<<<<<<< HEAD

def _is_executor_request(req_queue_item) -> bool:
    return isinstance(req_queue_item, tuple)


def _is_cancel_request(req_queue_item) -> bool:
    return isinstance(req_queue_item, int)


def _get_from_request_queue(request_queue, timeout: datetime.timedelta,
                            max_req_count: int):
    items = []
    timeout = timeout.total_seconds() if timeout is not None else None
    req_count = 0
    try:
        if request_queue.empty() and (timeout is None or timeout > 0):
            # if queue is empty and want to wait, wait
            items.append(request_queue.get(timeout=timeout))
        else:
            # if not empty or don't want to wait, just return all items in queue
            while req_count < max_req_count:
                queue_item = request_queue.get_nowait()
                items.append(queue_item)
                if _is_executor_request(queue_item):
                    # if it is request, (Not finish signal or cancel signal)
                    req_count += 1
    except queue.Empty:
        pass
    return items
=======
# Unique tag base to avoid collisions with token/logits comms
TERMINATION_COMM_TAG_BASE = 20000
>>>>>>> upstream/main


@functools.cache
def _load_iteration_indexes(env_var: str):
    spans = os.environ.get(env_var, None)
    starts, stops = [], []

    if spans:
        spans = spans.split(',')

        for span in spans:
            try:
                if '-' in span:
                    start, stop = span.strip().split('-')
                    starts.append(int(start))
                    stops.append(int(stop))
                else:
                    it = int(span.strip())
                    starts.append(it)
                    stops.append(it)
            except ValueError as e:
                raise ValueError(
                    f"Cannot parse span in environment variable `{env_var}`: {e}"
                ) from None

    return frozenset(starts), frozenset(stops)


class _GCNvtxHandle:
    pass


def _gc_nvtx_watcher():
    enabled = os.environ.get(PROFILE_RECORD_GC_ENV_VAR_NAME, None)
    if not enabled:
        return None

    range_id: Optional[int] = None

    def gc_callback(phase, _):
        nonlocal range_id
        if phase == "start":
            assert range_id is None, "Unexpected state in GC callback: another GC while last GC not finished?"
            range_id = torch.cuda.nvtx.range_start("Python GC")
        elif phase == "stop":
            assert range_id is not None, "Unexpected state in GC callback: no active GC but got GC finished?"
            torch.cuda.nvtx.range_end(range_id)
            range_id = None

    gc.callbacks.append(gc_callback)

    def gc_cleanup(callback):
        try:
            gc.callbacks.remove(callback)
        except ValueError:
            pass

    handle = _GCNvtxHandle()
    weakref.finalize(handle, gc_cleanup, gc_callback)
    return handle


@dataclasses.dataclass
class BatchState:
    sample_state: SampleState

    iter_start_time: float = 0
    iter_stats: IterationStats = None
    ctx_transmission_reqs: list[LlmRequest] = None


@dataclasses.dataclass
class BatchStatePP(BatchState):
    microbatch_id: int = -1
<<<<<<< HEAD
=======
    scheduled_ctx_reqs: list[LlmRequest] = None
>>>>>>> upstream/main


class PyExecutor:

    def __init__(self,
                 resource_manager,
<<<<<<< HEAD
                 scheduler,
                 model_engine: ModelEngine,
                 sampler: Sampler,
                 dist: Distributed,
                 disable_overlap_scheduler: bool = False,
                 max_input_len: int = 2048,
                 max_batch_size: int = 8,
                 max_draft_tokens: int = 0,
                 kv_cache_transceiver: KvCacheTransceiver = None,
                 draft_model_engine: Optional[ModelEngine] = None,
                 start_worker: bool = True):
        super(PyExecutor, self).__init__()
        self.device_id = torch.cuda.current_device()
        self.global_rank = global_mpi_rank()
        self.request_queue = queue.Queue()
=======
                 scheduler: RequestScheduler,
                 model_engine: ModelEngine,
                 sampler: Sampler,
                 dist: Distributed,
                 max_num_sequences: int,
                 drafter: Optional[Drafter] = None,
                 disable_overlap_scheduler: bool = False,
                 max_input_len: int = 2048,
                 max_batch_size: int = 8,
                 max_beam_width: int = 1,
                 max_draft_len: int = 0,
                 kv_cache_transceiver: Optional[KvCacheTransceiver] = None,
                 guided_decoder: Optional[GuidedDecoder] = None,
                 garbage_collection_gen0_threshold: Optional[int] = None,
                 start_worker: bool = True,
                 kv_connector_manager: Optional[KvCacheConnectorManager] = None,
                 max_seq_len: Optional[int] = None,
                 peft_cache_config: Optional[PeftCacheConfig] = None):
        super(PyExecutor, self).__init__()
        self.device_id = torch.cuda.current_device()
        self.global_rank = global_mpi_rank()

        self.peft_cache_config = peft_cache_config
>>>>>>> upstream/main

        # profile config
        self.profile_start_iters, self.profile_stop_iters = _load_iteration_indexes(
            PROFILE_START_STOP_ENV_VAR_NAME)
        self.gc_nvtx_watcher_handle = _gc_nvtx_watcher()
<<<<<<< HEAD
        self.is_warmup = False  # During warmup, we don't enable the profiler
=======
>>>>>>> upstream/main

        # related modules
        self.resource_manager = resource_manager
        self.scheduler = scheduler
        self.model_engine = model_engine
        self.enable_attention_dp = model_engine.enable_attention_dp
        self.sampler = sampler
<<<<<<< HEAD
        self.dist = dist
        self.disable_overlap_scheduler = disable_overlap_scheduler

        # Draft model for certain spec decode algorithms, e.g. EAGLE3
        self.draft_model_engine = draft_model_engine

        # enqueue and _fetch_new_requests used data
        self.enqueue_lock = threading.Lock()
        self.active = True
        self.next_req_id = max_batch_size  # The first max_batch_size request IDs are reserved for dummy requests
        self.max_draft_tokens = max_draft_tokens
        self.print_log = model_engine.pytorch_backend_config.print_iter_log
        self.enable_iter_perf_stats = model_engine.pytorch_backend_config.enable_iter_perf_stats
        self.enable_iter_req_stats = model_engine.pytorch_backend_config.enable_iter_req_stats
=======
        self.drafter = drafter
        self.draft_model_engine = getattr(self.drafter, "draft_model_engine",
                                          None)
        self.guided_decoder = guided_decoder
        self.dist = dist
        self.disable_overlap_scheduler = disable_overlap_scheduler

        # enqueue and _fetch_new_requests used data
        self.active = True
        self.max_beam_width = max_beam_width
        self.max_draft_len = max_draft_len
        self.max_num_tokens = model_engine.pytorch_backend_config.max_num_tokens
        self.print_log = model_engine.pytorch_backend_config.print_iter_log
        self.enable_iter_perf_stats = model_engine.pytorch_backend_config.enable_iter_perf_stats
        self.enable_iter_req_stats = model_engine.pytorch_backend_config.enable_iter_req_stats
        self.stream_interval = model_engine.pytorch_backend_config.stream_interval
        self.attention_dp_enable_balance = model_engine.pytorch_backend_config.attention_dp_enable_balance
        self.attention_dp_time_out_iters = model_engine.pytorch_backend_config.attention_dp_time_out_iters
        self.attention_dp_batching_wait_iters = model_engine.pytorch_backend_config.attention_dp_batching_wait_iters
        self.batch_wait_timeout_ms = model_engine.pytorch_backend_config.batch_wait_timeout_ms
        self.batch_wait_timeout_iters = model_engine.pytorch_backend_config.batch_wait_timeout_iters
        self.batch_wait_max_tokens_ratio = model_engine.pytorch_backend_config.batch_wait_max_tokens_ratio
        self.enable_batch_waiting = self.batch_wait_timeout_iters > 0 or self.batch_wait_max_tokens_ratio > 0

>>>>>>> upstream/main
        self.num_fetch_requests_cur_rank = 0
        self.num_fetch_requests = 0
        self.shutdown_event = threading.Event()

        # response used data
        self.response_lock = threading.Lock()
        self.response_cv = threading.Condition(self.response_lock)
        self.responses = {}

        # kv cache events
        self.kv_cache_manager = self.resource_manager.resource_managers.get(
<<<<<<< HEAD
            "kv_cache_manager")
        self.enable_kv_cache_events = self.kv_cache_manager is not None and self.kv_cache_manager.event_buffer_max_size > 0

        if self.draft_model_engine is not None and self.kv_cache_manager is not None:
            if self.kv_cache_manager.enable_block_reuse:
                raise NotImplementedError(
                    "Draft model engine + KV cache reuse is not supported yet. "
                    "This will be fixed in the near future!")
=======
            ResourceManagerType.KV_CACHE_MANAGER)
        self.enable_kv_cache_events = self.kv_cache_manager is not None and self.kv_cache_manager.event_buffer_max_size > 0
        self.enable_kv_cache_reuse = self.kv_cache_manager is not None and self.kv_cache_manager.enable_block_reuse
>>>>>>> upstream/main

        self.max_input_len = max_input_len
        # _executor_loop private data
        self.max_num_active_requests = model_engine.get_max_num_sequences()
        self.active_requests: List[LlmRequest] = []
<<<<<<< HEAD
        self.all_ranks_num_active_requests = [
            0
        ] * self.dist.tp_size if self.enable_attention_dp else []
        self.expected_num_active_requests = 0
        self.has_context_request = False
        self.ctx_in_transmission_requests = []
        self.previous_batch: Optional[BatchState] = None
        self.num_scheduled_requests: int = 0
=======
        self.expected_num_active_requests = 0
        self.ctx_in_transmission_requests = []
        self.previous_batch: Optional[BatchState] = None
        self.has_previous_draft_tokens = False
        self.num_scheduled_requests: int = 0
        self.benchmark_req_queues_size = int(
            os.environ.get("TLLM_BENCHMARK_REQ_QUEUES_SIZE", 0))
>>>>>>> upstream/main

        # list of requests in each PP micro batch
        self.num_micro_batches = self.dist.pp_size
        self.micro_batches: List[BatchStatePP
                                 | None] = [None] * self.num_micro_batches
        self.send_handles = [None] * self.num_micro_batches

        self.inflight_req_ids = ReqIdsSet()
<<<<<<< HEAD
        self.canceled_req_ids = ReqIdsSet()

        self.model_engine.warmup(self.resource_manager)
        if self.draft_model_engine is not None:
            self.draft_model_engine.warmup(self.resource_manager)

        self.is_shutdown = False

        self.stats_lock = threading.Lock()
        self.stats = []
        self.start_times = {}
        self.new_active_requests_queue_latency_ms = 0
        self.gather_all_responses = False

        self.kv_cache_transceiver = kv_cache_transceiver
=======

        # During warmup, we don't enable the profiler
        self.is_warmup = True
        self.model_engine.warmup(self.resource_manager)
        if self.draft_model_engine is not None:
            self.draft_model_engine.warmup(self.resource_manager)
        self.is_warmup = False

        self.is_shutdown = False
        self.max_batch_size = max_batch_size
        self.adp_ctx_waiting_iters_count = 0
        self.adp_ctx_batching_wait_iters_count = 0
        self.batch_wait_iters_count = 0

        # request fetcher initialization
        self.executor_request_queue = ExecutorRequestQueue(
            dist=self.dist,
            enable_attention_dp=self.enable_attention_dp,
            max_batch_size=max_batch_size,
            max_beam_width=self.max_beam_width,
            max_num_active_requests=self.max_num_active_requests,
            enable_iter_perf_stats=self.enable_iter_perf_stats,
            batch_wait_timeout_ms=self.batch_wait_timeout_ms,
            is_disaggregated=kv_cache_transceiver is not None,
        )
        self.executor_request_queue.set_exclude_last_generation_logits(
            self.disable_overlap_scheduler, self.dist.pp_size)

        self.stats_lock = threading.Lock()
        self.stats = []
        self.gather_all_responses = False

        self.kv_cache_transceiver = kv_cache_transceiver

        # Initialize disagg PP termination handler if needed
        self._disagg_pp_termination_handler = None
        if self.dist.pp_size > 1 and self.enable_kv_cache_reuse and self.kv_cache_transceiver:
            self._disagg_pp_termination_handler = DisaggPPTerminationHandler(
                self.num_micro_batches, self.dist)

>>>>>>> upstream/main
        if self.dist.pp_size > 1:
            self.event_loop = self._executor_loop_pp
        else:
            self.event_loop = self._executor_loop if disable_overlap_scheduler else self._executor_loop_overlap
<<<<<<< HEAD

        if is_trace_enabled("TLLM_TRACE_EXECUTOR_LOOP"):
            self.event_loop = trace_func(self.event_loop)

        if self.draft_model_engine is not None and self.event_loop.__name__ != self._executor_loop.__name__:
            raise NotImplementedError(
                "Drafting is not supported for selected executor loop. "
                "Please disable disagg/pipeline parallelism/overlap scheduler.")

        self.worker_started = False
        self.worker_lock = threading.Lock()
        if start_worker:
            self.start_worker()

    def start_worker(self):
        self.worker_lock.acquire()
        try:
            if self.worker_started == False:
                self.worker_thread = threading.Thread(target=self.event_loop,
                                                      daemon=True)
                self.worker_thread.start()
                self.worker_started = True
        finally:
            self.worker_lock.release()
=======
        if is_trace_enabled("TLLM_TRACE_EXECUTOR_LOOP"):
            self.event_loop = trace_func(self.event_loop)

        if self.drafter is not None:
            if self.event_loop.__name__ == self._executor_loop_pp.__name__:
                raise NotImplementedError(
                    "Drafting is not supported for selected executor loop. "
                    "Please disable disagg/pipeline parallelism scheduler.")
            self.draft_seq_slot_manager = SeqSlotManager(max_num_sequences)
        self.garbage_collection_gen0_threshold = garbage_collection_gen0_threshold
        self.max_seq_len = max_seq_len

        self.worker_started = False
        self.worker_lock = threading.Lock()

        self.kv_connector_manager = kv_connector_manager

        self._maybe_init_kv_connector_manager()

        if start_worker:
            self.start_worker()

    def _maybe_init_kv_connector_manager(self):
        if self.kv_connector_manager is not None:
            if self.kv_cache_transceiver is not None:
                raise NotImplementedError(
                    "KV Cache Connector is not supported with KvCacheTransceiver."
                )

            if self.dist.pp_size > 1:
                raise NotImplementedError(
                    "KV Cache Connector is not supported with pipeline parallelism."
                )

            if self.kv_cache_manager is None:
                raise ValueError(
                    "KV Cache Connector requires a KV Cache Manager.")

            kv_tensor = self.kv_cache_manager.get_unique_primary_pool()
            self.kv_connector_manager.worker.register_kv_caches(kv_tensor)

            # For each of our layers, we need to register the pre/post hooks.
            # These are used for methods like `wait_for_layer_load` and `save_kv_layer`.
            for _name, module in self.model_engine.model.named_modules():
                if isinstance(module, DecoderLayer):
                    module.register_forward_pre_hook(
                        self.kv_connector_manager.layer_pre_hook)
                    module.register_forward_hook(
                        self.kv_connector_manager.layer_post_hook)

    def _event_loop_wrapper(self):
        try:
            with customized_gc_thresholds(
                    self.garbage_collection_gen0_threshold):
                self.event_loop()
        except Exception as e:
            logger.error(f"Error in event loop: {e}")
            logger.error(traceback.format_exc())
            raise e
        finally:
            self._executor_loop_cleanup()

    @property
    def is_warmup(self) -> bool:
        return getattr(self, "_is_warmup", False)

    @is_warmup.setter
    def is_warmup(self, value: bool):
        self._is_warmup = value
        # Set warmup flag in model engine to trigger torch compile and avoid moe load balancer statistics update
        self.model_engine.is_warmup = value
        if self.draft_model_engine is not None:
            self.draft_model_engine.is_warmup = value

    def start_worker(self):
        with self.worker_lock:
            if self.worker_started == False:
                self.worker_thread = threading.Thread(
                    target=self._event_loop_wrapper, daemon=True)
                self.worker_thread.start()
                self.worker_started = True
>>>>>>> upstream/main

    def __enter__(self):
        return self

    def __exit__(self):
        self.shutdown()

<<<<<<< HEAD
    def enqueue_requests(self, requests: List[ExecutorRequest]):
        """
        Enqueue new requests
        """
        req_ids = []
        try:
            self.enqueue_lock.acquire()
            assert self.active, "PyExecutor has already been shutdown."
            start_time = time.time()
            for request in requests:
                self.start_times[self.next_req_id] = start_time
                self.request_queue.put((self.next_req_id, request))
                req_ids.append(self.next_req_id)
                self.next_req_id += 1
        finally:
            self.enqueue_lock.release()
=======
    def enqueue_requests(self, requests: List[ExecutorRequest]) -> List[int]:
        """
        Enqueue new requests
        """
        req_ids = self.executor_request_queue.enqueue_requests(requests)
>>>>>>> upstream/main
        return req_ids

    def await_responses(
        self,
        id: Optional[Union[List[int], int]] = None,
        timeout: Optional[datetime.timedelta] = None,
<<<<<<< HEAD
    ) -> Union[List[List[ExecutorResponse]], List[ExecutorResponse]]:
=======
    ) -> Union[List[List[LlmResponse]], List[LlmResponse]]:
>>>>>>> upstream/main
        """
        Await for ready responses
        Args:
            id (Optional[Union[List[int], int]]): Request id
            timeout (Optional[datetime.timedelta]): The maximum time to wait for new responses
        Returns:
<<<<<<< HEAD
            Union[List[tensorrt_llm.bindings.executor.Response], List[List[tensorrt_llm.bindings.executor.Response]]]: Responses
=======
            Union[List[LlmResponse], List[List[LlmResponse]]]: Responses
>>>>>>> upstream/main
        """
        timeout = timeout.total_seconds() if timeout is not None else None
        if id is None:
            return self._await_any_response(timeout=timeout)
        if isinstance(id, int):
            return self._await_single_response(id=id, timeout=timeout)
        responses = []
        for req_id in id:
            responses.append(
                self._await_single_response(id=req_id, timeout=timeout))
        return responses

    def cancel_request(self, id: int):
        """
        Cancel the request with provided request id
        Args:
            id (int): The request id for which to cancel the response
        """
<<<<<<< HEAD
        self.canceled_req_ids.insert(id)
=======
        self.executor_request_queue.enqueue_cancel_request(id)
>>>>>>> upstream/main

    def shutdown(self):
        """
        Signals the server to shutdown.
        """
<<<<<<< HEAD
        try:
            self.enqueue_lock.acquire()
            self.request_queue.put(None)
            self.active = False
        finally:
            self.enqueue_lock.release()
=======
        self.executor_request_queue.enqueue_shutdown_request()
>>>>>>> upstream/main
        self.shutdown_event.wait()
        self.worker_thread.join()
        self.worker_started = False
        for manager in self.resource_manager.resource_managers.values():
            if manager:
                manager.shutdown()
        del self.model_engine
        if self.draft_model_engine is not None:
            del self.draft_model_engine

    def can_enqueue_requests(self) -> bool:
        """
        Indicates if the current process is allowed to enqueue requests
        """
<<<<<<< HEAD
        self.enqueue_lock.acquire()
        can_enqueue = self.active
        self.enqueue_lock.release()
        return can_enqueue and self.dist.rank == 0
=======
        return self.executor_request_queue.can_enqueue_request()
>>>>>>> upstream/main

    def get_latest_iteration_stats(self):
        """
        Returns the per-iterations statistics computed since last call to this method.
        Contains at most iter_stats_max_iterations iterations.
        """
        if self.enable_iter_perf_stats == False:
            return []

        latest_stats = (IterationStats(), None)
<<<<<<< HEAD
        try:
            self.stats_lock.acquire()
            latest_stats = self.stats
            self.stats = []
        finally:
            self.stats_lock.release()

=======
        with self.stats_lock:
            latest_stats = self.stats
            self.stats = []
>>>>>>> upstream/main
        return latest_stats

    def get_latest_kv_cache_events(self):
        kv_cache_manager = self.resource_manager.resource_managers.get(
<<<<<<< HEAD
            "kv_cache_manager")
=======
            ResourceManagerType.KV_CACHE_MANAGER)
>>>>>>> upstream/main
        if not kv_cache_manager or not self.enable_kv_cache_events:
            return []

        events = kv_cache_manager.get_latest_events(0)
        return events

    def wait_shutdown(self):
        self.shutdown_event.wait()

    def enqueue_request(self,
                        request: ExecutorRequest,
<<<<<<< HEAD
                        query: Optional[List] = None):
        """
        Enqueue a new request, only used in `StarAttention`.
        """
        try:
            self.enqueue_lock.acquire()
            assert self.active, "PyExecutor has already been shutdown."
            req_id = self.next_req_id
            if self.enable_iter_perf_stats:
                self.start_times[req_id] = time.time()

            if query is not None:
                self.request_queue.put((req_id, request, query))
            else:
                self.request_queue.put((req_id, request))
            self.next_req_id += 1
        finally:
            self.enqueue_lock.release()
=======
                        query: Optional[List] = None) -> int:
        """
        Enqueue a new request, query is only used in `StarAttention`.
        """
        req_id = self.executor_request_queue.enqueue_request(request, query)

>>>>>>> upstream/main
        return req_id

    def set_gather_responses(self, gather_all_responses):
        self.gather_all_responses = gather_all_responses

<<<<<<< HEAD
=======
    @property
    def should_stop_processing(self):
        return self.is_shutdown and len(self.active_requests) == 0 and \
            self.executor_request_queue.get_waiting_queue_size() == 0

>>>>>>> upstream/main
    @contextmanager
    def _profiler(self):
        it = -1
        enabled = False
        start_time = None
<<<<<<< HEAD
=======

        # These events are used to record the time of the previous batch.
        # We need two set of the start-end events to record the time through
        # a ping-pong way so that it works with overlap scheduler.
        start_event_1 = None
        end_event_1 = torch.cuda.Event(enable_timing=True)
        start_event_2 = None
        end_event_2 = torch.cuda.Event(enable_timing=True)
        prev_device_step_time = None

>>>>>>> upstream/main
        torch_trace_path = os.environ.get(PROFILE_TRACE_ENV_VAR_NAME, None)
        profile_start_stop = os.environ.get(PROFILE_START_STOP_ENV_VAR_NAME,
                                            None)
        enable_torch_trace = bool(torch_trace_path and profile_start_stop)
        if torch_trace_path and profile_start_stop is None:
            logger.warning(
                f"{PROFILE_START_STOP_ENV_VAR_NAME} environment variable "
                "needs to be set to enable the torch trace. Example to profile "
                f"iteration 10-20: export {PROFILE_START_STOP_ENV_VAR_NAME}=10-20"
            )

        if enable_torch_trace:
            activities = [
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
                torch.profiler.ProfilerActivity.XPU,
            ]
            torch_profiler = torch.profiler.profile(activities=activities,
                                                    record_shapes=True,
                                                    with_modules=True)

        def profile_step():
<<<<<<< HEAD
            nonlocal it, enabled, start_time
=======
            nonlocal it, enabled, start_time, start_event_1, end_event_1, start_event_2, end_event_2, prev_device_step_time
>>>>>>> upstream/main
            if it in self.profile_stop_iters and not self.is_warmup:
                assert enabled, "Inconsistent CUDA profiling state"
                if enable_torch_trace:
                    torch_profiler.stop()
                    torch_profiler.export_chrome_trace(torch_trace_path)
                    logger.info(f"Profiling stopped at iteration {it}, "
                                f"trace saved to {torch_trace_path}")
                torch.cuda.cudart().cudaProfilerStop()
                enabled = False

            if start_time is not None and self.print_log and self.dist.rank == 0:
                end_time = time.time()
<<<<<<< HEAD

=======
                if it % 2 == 0:
                    end_event_1.record()
                    if start_event_2 is not None:
                        end_event_2.synchronize()
                        prev_device_step_time = start_event_2.elapsed_time(
                            end_event_2)
                else:
                    end_event_2.record()
                    if start_event_1 is not None:
                        end_event_1.synchronize()
                        prev_device_step_time = start_event_1.elapsed_time(
                            end_event_1)

                if prev_device_step_time is None:
                    prev_device_step_time = "N/A"  # Handle first iteration
                else:
                    prev_device_step_time = f"{prev_device_step_time}ms"
                host_step_time = (end_time - start_time) * 1000  # milliseconds
>>>>>>> upstream/main
                formatted_timestamp = datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S")
                logger.info(
                    f"iter = {self.model_engine.iter_counter}, "
                    f"global_rank = {self.global_rank}, "
                    f"rank = {self.dist.rank}, "
<<<<<<< HEAD
                    f"currank_total_requests = {self.num_fetch_requests_cur_rank}/{self.num_fetch_requests}, "
                    f"elapsed_time = {end_time - start_time}s, "
=======
                    f"currank_total_requests = {self.executor_request_queue.num_fetch_requests_cur_rank}/"
                    f"{self.executor_request_queue.num_fetch_requests}, "
                    f"host_step_time = {host_step_time}ms, "
                    f"prev_device_step_time = {prev_device_step_time}, "
>>>>>>> upstream/main
                    f"timestamp = {formatted_timestamp}, "
                    f"num_scheduled_requests: {self.num_scheduled_requests}, "
                    f"states = {self.model_engine.iter_states}")

            it += 1

            if it in self.profile_start_iters and not self.is_warmup:
                assert not enabled, "Inconsistent CUDA profiling state"
                torch.cuda.cudart().cudaProfilerStart()
                if enable_torch_trace:
                    torch_profiler.start()
                logger.info(f"Profiling started at iteration {it}.")
                enabled = True
            start_time = time.time()
<<<<<<< HEAD
=======
            if it % 2 == 0:
                if start_event_1 is None:
                    start_event_1 = torch.cuda.Event(enable_timing=True)
                start_event_1.record()
            else:
                if start_event_2 is None:
                    start_event_2 = torch.cuda.Event(enable_timing=True)
                start_event_2.record()
>>>>>>> upstream/main

        try:
            yield profile_step
        finally:
            if enabled:
                # Stop on early exit / exception
                if enable_torch_trace:
                    torch_profiler.stop()
                    torch_profiler.export_chrome_trace(torch_trace_path)
                    logger.info(f"Profiling stopped at iteration {it}, "
                                f"trace saved to {torch_trace_path}")
                torch.cuda.cudart().cudaProfilerStop()

    def _get_init_iter_stats(self, num_new_active_requests,
                             new_active_requests_queue_latency_ms):
        stats = IterationStats()
        stats.timestamp = datetime.datetime.now().strftime(
            "%m-%d-%Y %H:%M:%S.%f")

        stats.num_new_active_requests = num_new_active_requests
        stats.num_active_requests = len(self.active_requests)
        stats.new_active_requests_queue_latency_ms = new_active_requests_queue_latency_ms
        stats.inflight_batching_stats = InflightBatchingStats()
        # staticBatchingStats is not used in pytorch path
        stats.static_batching_stats = StaticBatchingStats()
<<<<<<< HEAD
=======
        spec_resource_manager = self.resource_manager.resource_managers.get(
            ResourceManagerType.SPEC_RESOURCE_MANAGER)
        if spec_resource_manager is not None:
            stats.specdec_stats = SpecDecodingStats()
>>>>>>> upstream/main
        return stats

    def _populate_req_stats(
            self, finished_requests: List[LlmRequest],
            active_requests: List[LlmRequest],
            scheduled_requests: ScheduledRequests
    ) -> Optional[List[RequestStats]]:

        def get_req_stats(req: LlmRequest) -> RequestStats:
            req_stat = RequestStats()
            req_stat.id = req.request_id
            req_stat.context_prefill_position = req.context_current_position
            req_stat.num_generated_tokens = req.max_beam_num_tokens - req.orig_prompt_len
            req_stat.avg_num_decoded_tokens_per_iter = req.avg_decoded_tokens_per_iter
            req_stat.alloc_total_blocks_per_request = req.alloc_total_blocks
            req_stat.alloc_new_blocks_per_request = req.alloc_new_blocks
            req_stat.reused_blocks_per_request = req.reused_blocks
            req_stat.missed_blocks_per_request = req.missed_blocks
            req_stat.kv_cache_hit_rate_per_request = req.kv_cache_hit_rate
            req_stat.scheduled = req in scheduled_requests.context_requests or req in scheduled_requests.generation_requests
            if req.llm_request_type == LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY or req.llm_request_type == LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY:
                req_stat.dis_serving_stats = DisServingRequestStats()
                req_stat.dis_serving_stats.kv_cache_transfer_ms = req.kv_cache_transfer_time_ms
                req_stat.dis_serving_stats.kv_cache_size = req.kv_cache_size
            return req_stat

        def get_queued_req_stats(request_id: int) -> RequestStats:
            req_stat = RequestStats()
            req_stat.id = request_id
            req_stat.context_prefill_position = 0
            req_stat.num_generated_tokens = 0
            req_stat.avg_num_decoded_tokens_per_iter = 0
            req_stat.alloc_total_blocks_per_request = 0
            req_stat.alloc_new_blocks_per_request = 0
            req_stat.reused_blocks_per_request = 0
            req_stat.missed_blocks_per_request = 0
            req_stat.kv_cache_hit_rate_per_request = 0
            return req_stat

        req_stats = []
        for req in active_requests:
            req_stat = get_req_stats(req)
            req_stat.stage = req.stage
            req_stats.append(req_stat)

<<<<<<< HEAD
        for req in list(self.request_queue.queue):
            if isinstance(req, Tuple):
                req_stat = get_queued_req_stats(req[0])
=======
        for req in list(self.executor_request_queue.get_request_queue().queue):
            if isinstance(req, RequestQueueItem):
                req_stat = get_queued_req_stats(req.id)
>>>>>>> upstream/main
                req_stat.stage = RequestStage.QUEUED
                req_stats.append(req_stat)

        for req in finished_requests:
            req_stat = get_req_stats(req)
            req_stat.stage = RequestStage.GENERATION_COMPLETE
            req_stats.append(req_stat)

        return req_stats

    def _update_iter_stats(self, stats, iter_latency_ms, num_completed_requests,
                           scheduled_batch) -> IterationStats:
        stats.iter_latency_ms = iter_latency_ms

<<<<<<< HEAD
        stats.num_queued_requests = self.request_queue.qsize()
=======
        stats.num_queued_requests = self.executor_request_queue.get_request_queue_size(
        )
>>>>>>> upstream/main
        stats.num_completed_requests = num_completed_requests
        stats.max_num_active_requests = self.max_num_active_requests

        end, total_gpu_memory = torch.cuda.mem_get_info()
        stats.gpu_mem_usage = total_gpu_memory - end
        stats.cpu_mem_usage = 0
        stats.pinned_mem_usage = 0

        stats.iter = self.model_engine.iter_counter

        kv_cache_manager = self.resource_manager.resource_managers.get(
<<<<<<< HEAD
            "kv_cache_manager")
=======
            ResourceManagerType.KV_CACHE_MANAGER)
>>>>>>> upstream/main
        if kv_cache_manager is not None:
            kv_stats = kv_cache_manager.get_kv_cache_stats()
            kv_stats_to_save = KvCacheStats()
            kv_stats_to_save.max_num_blocks = kv_stats.max_num_blocks
            kv_stats_to_save.free_num_blocks = kv_stats.free_num_blocks
            kv_stats_to_save.used_num_blocks = kv_stats.used_num_blocks
            kv_stats_to_save.tokens_per_block = kv_stats.tokens_per_block
            kv_stats_to_save.alloc_total_blocks = kv_stats.alloc_total_blocks
            kv_stats_to_save.alloc_new_blocks = kv_stats.alloc_new_blocks
            kv_stats_to_save.reused_blocks = kv_stats.reused_blocks
            kv_stats_to_save.missed_blocks = kv_stats.missed_blocks
            kv_stats_to_save.cache_hit_rate = kv_stats.cache_hit_rate
            stats.kv_cache_stats = kv_stats_to_save

        stats.inflight_batching_stats.num_scheduled_requests = len(
            scheduled_batch.context_requests) + len(
                scheduled_batch.generation_requests)
        stats.inflight_batching_stats.num_context_requests = len(
            scheduled_batch.context_requests)
        stats.inflight_batching_stats.num_gen_requests = len(
            scheduled_batch.generation_requests)
        stats.inflight_batching_stats.num_paused_requests = len(
            scheduled_batch.paused_requests)
        stats.inflight_batching_stats.avg_num_decoded_tokens_per_iter = 0
        stats.inflight_batching_stats.micro_batch_id = 0
<<<<<<< HEAD
=======
        if stats.specdec_stats is not None:
            stats.specdec_stats.draft_overhead = 0.0 if iter_latency_ms <= 0.0 else float(
                stats.specdec_stats.iter_latency_ms) / float(iter_latency_ms)
>>>>>>> upstream/main
        return stats

    def _append_iter_stats(self,
                           stats: IterationStats,
                           req_stats: Optional[List[RequestStats]] = None):

<<<<<<< HEAD
        try:
            self.stats_lock.acquire()
            self.stats.append((stats, req_stats))
        finally:
            self.stats_lock.release()
=======
        with self.stats_lock:
            self.stats.append((stats, req_stats))
>>>>>>> upstream/main

    def _process_iter_stats(self, finished_requests: list[LlmRequest],
                            active_requests: List[LlmRequest],
                            batch_state: BatchState):
        iter_end_time = time.time()
<<<<<<< HEAD
        iter_latency_ms = iter_end_time - batch_state.iter_start_time
=======
        iter_latency_ms = (iter_end_time - batch_state.iter_start_time) * 1e3
>>>>>>> upstream/main
        if batch_state.iter_stats is None:
            return

        req_stats = self._populate_req_stats(
            finished_requests, active_requests,
            batch_state.sample_state.scheduled_requests) if (
                self.enable_iter_req_stats
                and self.enable_iter_perf_stats) else None

        self._append_iter_stats(
            self._update_iter_stats(
                batch_state.iter_stats, iter_latency_ms, len(finished_requests),
                batch_state.sample_state.scheduled_requests), req_stats)

    def _executor_loop_cleanup(self):
<<<<<<< HEAD
=======

        for h in self.send_handles:
            if h is not None:
                h.wait()

        if self._disagg_pp_termination_handler is not None:
            self._disagg_pp_termination_handler.cleanup()

>>>>>>> upstream/main
        with self.response_cv:
            self.is_shutdown = True
            self.response_cv.notify_all()
        self.shutdown_event.set()

    def _executor_loop_pp(self):
<<<<<<< HEAD
        torch.cuda.set_device(self.device_id)
        got_finish_signal = False
        num_dummy_request = 0
=======
        logger.debug(f"Starting executor loop for pp_rank {self.dist.pp_rank}")
        torch.cuda.set_device(self.device_id)
        # ensure the context is created, otherwise, some MPI calls will fail.
        CUASSERT(cudart.cudaSetDevice(self.device_id))
>>>>>>> upstream/main
        microbatch_id = 0
        with self._profiler() as profile_step:
            iter_start_time = time.time()
            iter_stats = None
<<<<<<< HEAD
            while not got_finish_signal or len(self.active_requests) > 0:
                profile_step()
                if self.enable_iter_perf_stats:
                    iter_start_time = time.time()
                new_requests = self._fetch_new_requests()
                got_finish_signal = self._merge_requests(
                    new_requests) or got_finish_signal
                if got_finish_signal and len(self.active_requests) == 0:
                    break

                if self.enable_iter_perf_stats:
                    iter_stats = self._get_init_iter_stats(
                        len(new_requests),
                        self.new_active_requests_queue_latency_ms)

                if not got_finish_signal:
                    num_dummy_request = self._get_num_dummy_request()
                if num_dummy_request > 0:
                    self._merge_dummy_request(num_dummy_request)
                scheduled_batch, _, _ = self._schedule()

                self.num_scheduled_requests = scheduled_batch.batch_size
=======
            while True:
                profile_step()
                if self.enable_iter_perf_stats:
                    iter_start_time = time.time()
                new_requests = self._fetch_and_activate_new_requests()
                if self.should_stop_processing:
                    break

                if self.kv_cache_transceiver:
                    self._check_disagg_gen_transfer_status()

                if self.enable_iter_perf_stats:
                    iter_stats = self._get_init_iter_stats(
                        len(new_requests),
                        self.executor_request_queue.
                        get_new_active_requests_queue_latency())

                self._pad_attention_dp_dummy_request()

                scheduled_batch, fitting_disagg_gen_init_requests, num_fitting_reqs = self._schedule(
                )

                if self.kv_cache_transceiver:
                    # For requests that are fitting disagg gen init, also prepare resources for KV cache manager
                    self._prepare_disagg_gen_init(
                        fitting_disagg_gen_init_requests)

                    if num_fitting_reqs == 0 and not fitting_disagg_gen_init_requests:
                        logger.warning(
                            "num_fitting_reqs=0 and fitting_disagg_gen_init_requests is empty, may not have enough kvCache"
                        )
                        self._check_disagg_ctx_cache_transfer_status(1)

                self.num_scheduled_requests = scheduled_batch.batch_size

>>>>>>> upstream/main
                logger.debug(
                    f'has {len(self.active_requests)} active_request, '
                    f'scheduled {len(scheduled_batch.context_requests)} context requests and '
                    f'{len(scheduled_batch.generation_requests)} generation requests'
                )

                if self.enable_attention_dp:
                    tp_batch_sizes = self.dist.tp_allgather(
                        scheduled_batch.batch_size)
                    can_queue = 0 not in tp_batch_sizes
                else:
                    can_queue = scheduled_batch.batch_size > 0
<<<<<<< HEAD
                    if not can_queue:
                        assert len(self.inflight_req_ids) > 0, (
                            "fail to schedule any pending request, probably run out of resource"
                        )
=======
>>>>>>> upstream/main

                if not can_queue:
                    self.micro_batches[microbatch_id] = None
                else:
                    self._add_inflight_ids(scheduled_batch)
<<<<<<< HEAD
                    self.resource_manager.prepare_resources(scheduled_batch)

=======

                    if self.kv_cache_transceiver:
                        # For generation requests which have completed KV cache transfer
                        self._prepare_disagg_gen_transmission_complete(
                            scheduled_batch)

                    self.resource_manager.prepare_resources(scheduled_batch)

                    # The generation requests that are do not have batch_idx,
                    # needs to be in front of the batch due to the assumptions
                    # made in model_engine.py::_forward_step. This is only important
                    # for disaggregated serving. For non-disaggregated serving,
                    # the generation requests always have batch_idx.
                    scheduled_batch.generation_requests = sorted(  # stable sort
                        scheduled_batch.generation_requests,
                        key=lambda req: int(req.py_batch_idx is not None),
                    )

                    if self.kv_cache_transceiver:
                        # Return the first token to the client
                        self._handle_first_token_response(scheduled_batch)

>>>>>>> upstream/main
                    # Stage 1: Async forward (all ranks) and decoding pass (last rank only)
                    if not self.dist.is_last_pp_rank:
                        sample_state = self._forward_step_inter_pp(
                            scheduled_batch)
                    else:
                        with torch.cuda.nvtx.range("_forward_step_last_pp"):
<<<<<<< HEAD
                            batch_outputs = self._forward_step(scheduled_batch)
                            sample_state = self._sample_async(
                                scheduled_batch, batch_outputs)
=======
                            # init_disagg_gen_requests must be before engine forward, where the prev_seq_slot is updated.
                            if self.guided_decoder is not None and self.kv_cache_transceiver:
                                self.guided_decoder.add_batch(scheduled_batch)
                                self.guided_decoder.init_disagg_gen_requests()

                            batch_outputs = self._forward_step(scheduled_batch)

                            if self.guided_decoder is not None:
                                self.guided_decoder.add_batch(scheduled_batch)
                                self.guided_decoder.execute(
                                    batch_outputs['logits'])

                            sample_state = self._sample_async(
                                scheduled_batch, batch_outputs)
                            assert sample_state is not None, "Sampling failed"
>>>>>>> upstream/main
                            self._update_request_states(scheduled_batch)

                    if self.enable_iter_perf_stats:
                        iter_stats.inflight_batching_stats.num_ctx_tokens = self.model_engine.iter_states[
                            'num_ctx_tokens']
                    batch_state = BatchStatePP(
                        sample_state=sample_state,
                        iter_start_time=iter_start_time,
                        iter_stats=iter_stats,
                        microbatch_id=microbatch_id,
<<<<<<< HEAD
                    )

                    if num_dummy_request > 0:
                        self._finish_dummy_request(
                            sample_state.scheduled_requests)
=======
                        scheduled_ctx_reqs=scheduled_batch.context_requests,
                    )

>>>>>>> upstream/main
                    self.micro_batches[microbatch_id] = batch_state

                # Stage 2: Communicate new tokens for previous batch between ranks
                # send/recv chain: (pp_size - 1) -> 0 -> 1 -> ... -> (pp_size - 2)
<<<<<<< HEAD
                # last rank: sync decoder for previous microbatch to start new tokens comm chain.
=======
                # last rank: sync sampler for previous microbatch to start new tokens comm chain.
>>>>>>> upstream/main
                # other ranks: send/recv tokens for next microbatch to allow overlap
                offset = -1 if self.dist.is_last_pp_rank else 1
                prev_microbatch_id = (microbatch_id +
                                      offset) % self.num_micro_batches
                previous_batch = self.micro_batches[prev_microbatch_id]
                if previous_batch is not None:
<<<<<<< HEAD
=======
                    sample_state = previous_batch.sample_state
>>>>>>> upstream/main
                    if not self.dist.is_last_pp_rank:
                        torch.cuda.nvtx.range_push(
                            "_handle_new_tokens_inter_pp")
                        # Receive tokens from previous pp rank (w.r.t model forward direction)
<<<<<<< HEAD
                        self.dist.recv_tensor_list(
                            previous_batch.sample_state.host.values(),
                            src=self.dist.prev_pp_rank,
                            tag=prev_microbatch_id)
                    else:
                        torch.cuda.nvtx.range_push("_handle_new_tokens_last_pp")
                        previous_batch.sample_state.sampler_event.synchronize()
=======
                        sample_state.host = self.dist.recv_object(
                            src=self.dist.prev_pp_rank,
                            tag=prev_microbatch_id,
                        )
                    else:
                        torch.cuda.nvtx.range_push("_handle_new_tokens_last_pp")
                        sample_state.sampler_event.synchronize()
>>>>>>> upstream/main

                    # Send tokens to next pp rank (w.r.t model forward direction)
                    # Second last rank does not need to since last rank has original decoded tokens
                    if not self.dist.is_second_last_pp_rank:
<<<<<<< HEAD
                        if self.send_handles[prev_microbatch_id] is not None:
                            self.send_handles[prev_microbatch_id].Wait()
                        self.send_handles[
                            prev_microbatch_id] = self.dist.isend_tensor_list(
                                previous_batch.sample_state.host.values(),
=======
                        self.wait_on_pp_send_handles(prev_microbatch_id)
                        self.send_handles[
                            prev_microbatch_id] = self.dist.isend_object(
                                sample_state.host,
>>>>>>> upstream/main
                                dest=self.dist.next_pp_rank,
                                tag=prev_microbatch_id)
                    torch.cuda.nvtx.range_pop()

                # Stage 3: Finalize previous batch that finished tokens communication
                # In last pp rank, stage 2 and 3 process different previous batches
                prev_microbatch_id = (microbatch_id +
                                      1) % self.num_micro_batches
                previous_batch = self.micro_batches[prev_microbatch_id]
                finished_requests = []
                if previous_batch is not None:
                    with torch.cuda.nvtx.range("_handle_previous_batch_pp"):
                        self._update_requests(previous_batch.sample_state)
<<<<<<< HEAD
                        self._handle_cancelled_requests()
=======

                        if self.kv_cache_transceiver and previous_batch.scheduled_ctx_reqs:
                            self._send_disagg_ctx_cache(
                                previous_batch.scheduled_ctx_reqs)

                        self._handle_canceled_requests()

                        self._handle_logits_communication(
                            previous_batch, prev_microbatch_id)

>>>>>>> upstream/main
                        finished_requests = self._handle_responses()
                        previous_scheduled_batch = previous_batch.sample_state.scheduled_requests
                        self.resource_manager.update_resources(
                            previous_scheduled_batch)
                        self._remove_inflight_ids(previous_scheduled_batch)
<<<<<<< HEAD
                    self.micro_batches[prev_microbatch_id] = None

                # march forward in microbatch slots
                microbatch_id = (microbatch_id + 1) % self.num_micro_batches
                self._gather_dp_requests_num()
=======

                    self.wait_on_pp_send_handles(prev_microbatch_id)
                    self.micro_batches[prev_microbatch_id] = None

                if self.kv_cache_transceiver and self.ctx_in_transmission_requests:
                    self._terminate_ctx_finished_requests()

                if self._disagg_pp_termination_handler is not None:
                    requests_to_terminate = self._disagg_pp_termination_handler.sync(
                        prev_microbatch_id)
                    for req in requests_to_terminate:
                        self._do_terminate_request(req)

                # march forward in microbatch slots
                microbatch_id = (microbatch_id + 1) % self.num_micro_batches
>>>>>>> upstream/main

                if self.enable_iter_perf_stats and previous_batch is not None:
                    self._process_iter_stats(finished_requests,
                                             self.active_requests,
                                             previous_batch)
<<<<<<< HEAD
        self._executor_loop_cleanup()

    def _executor_loop(self):
        torch.cuda.set_device(self.device_id)
        got_finish_signal = False
        num_dummy_request = 0
        with self._profiler() as profile_step:
            iter_start_time = time.time()
            iter_stats = None
            while not got_finish_signal or len(self.active_requests) > 0:
                profile_step()
                if self.enable_iter_perf_stats:
                    iter_start_time = time.time()
                new_requests = self._fetch_new_requests()
                got_finish_signal = self._merge_requests(
                    new_requests) or got_finish_signal
                if got_finish_signal and len(self.active_requests) == 0:
                    break
                if self.enable_iter_perf_stats:
                    iter_stats = self._get_init_iter_stats(
                        len(new_requests),
                        self.new_active_requests_queue_latency_ms)

                if self.kv_cache_transceiver:
                    self._check_disagg_gen_transfer_status()

                if not got_finish_signal:
                    num_dummy_request = self._get_num_dummy_request()
                if num_dummy_request > 0:
                    self._merge_dummy_request(num_dummy_request)

                if self.draft_model_engine is not None:
                    self._prepare_draft_requests()

                scheduled_batch, fitting_disagg_gen_init_requests, num_fitting_reqs = self._schedule(
                )

                if self.kv_cache_transceiver:
                    self._prepare_disagg_gen_init(
                        fitting_disagg_gen_init_requests)
                    if num_fitting_reqs == 0 and not fitting_disagg_gen_init_requests:
                        logger.warning(
                            "num_fitting_reqs=0 and fitting_disagg_gen_init_requests is empty, may not have enough kvCache"
                        )
                        self.kv_cache_transceiver.check_context_transfer_status(
                            1)
                else:
                    assert scheduled_batch.batch_size > 0, (
                        "fail to schedule any pending request, "
                        "probably run out of resource.")

                self.num_scheduled_requests = scheduled_batch.batch_size
                logger.debug(
                    f'has {len(self.active_requests)} active_request, '
                    f'scheduled {len(scheduled_batch.context_requests)} context requests and '
                    f'{len(scheduled_batch.generation_requests)} generation requests'
                )
=======

    def wait_on_pp_send_handles(self, microbatch_id):
        if self.send_handles[microbatch_id] is not None:
            self.send_handles[microbatch_id].wait()
            self.send_handles[microbatch_id] = None

    def _prepare_and_schedule_batch(self):
        new_requests = self._fetch_and_activate_new_requests()
        if self.should_stop_processing:
            return None, None

        if self.kv_cache_transceiver:
            self._check_disagg_gen_transfer_status()

        iter_stats = None
        if self.enable_iter_perf_stats:
            iter_stats = self._get_init_iter_stats(
                len(new_requests),
                self.executor_request_queue.
                get_new_active_requests_queue_latency())

        self._pad_attention_dp_dummy_request()

        if self.drafter is not None:
            self.use_spec_decode = self.drafter.should_use_spec_decode(
                self.active_requests, self.max_batch_size,
                self.model_engine.max_num_tokens,
                self.model_engine.spec_config.max_draft_len)
            self.model_engine.enable_spec_decode = self.use_spec_decode

            # When overlap scheduler is enabled, and we already prepared the draft tokens in the previous batch,
            # we don't need to initialize py_draft_tokens at this stage because we haven't append the accepted tokens to the request yet.
            if not self.has_previous_draft_tokens:
                # If speculation is off, this function sets py_draft_tokens to []
                # for all active requests. If it's on, we initialize py_draft_tokens
                # with dummy draft tokens to make the scheduler aware of the fact
                # that speculation is about to happen.
                self._prepare_draft_requests()

        scheduled_batch, fitting_disagg_gen_init_requests, num_fitting_reqs = self._schedule(
        )

        if self.kv_cache_transceiver:
            # For requests that are fitting disagg gen init, also prepare resources for KV cache manager
            self._prepare_disagg_gen_init(fitting_disagg_gen_init_requests)

            if num_fitting_reqs == 0 and not fitting_disagg_gen_init_requests:
                logger.warning(
                    "num_fitting_reqs=0 and fitting_disagg_gen_init_requests is empty, may not have enough kvCache"
                )
                self._check_disagg_ctx_cache_transfer_status(1)

        self.num_scheduled_requests = scheduled_batch.batch_size
        logger.debug(
            f'has {len(self.active_requests)} active_request, '
            f'scheduled {len(scheduled_batch.context_requests)} context requests and '
            f'{len(scheduled_batch.generation_requests)} generation requests')
        return scheduled_batch, iter_stats

    def _kv_connector_start_batch(self, scheduled_batch):
        if self.kv_connector_manager:
            self.kv_connector_manager.take_scheduled_requests_pending_load(
                scheduled_batch)
            self.kv_connector_manager.handle_metadata()
            self.kv_connector_manager.worker.start_load_kv(
                torch.cuda.current_stream())

    def _kv_connector_terminate_requests(self):
        if self.kv_connector_manager:
            reqs_to_terminate = self.kv_connector_manager.get_finished()
            for req in reqs_to_terminate:
                self.resource_manager.free_resources(req)

    def _kv_connector_wait_for_save(self):
        if self.kv_connector_manager is not None:
            self.kv_connector_manager.worker.wait_for_save(
                torch.cuda.current_stream())

    def _executor_loop(self):
        torch.cuda.set_device(self.device_id)
        # ensure the context is created, otherwise, some MPI calls will fail.
        CUASSERT(cudart.cudaSetDevice(self.device_id))
        with self._profiler() as profile_step:
            sample_state = None
            iter_start_time = time.time()
            iter_stats = None
            while True:
                profile_step()
                if self.enable_iter_perf_stats:
                    iter_start_time = time.time()

                scheduled_batch, iter_stats = self._prepare_and_schedule_batch()
                if scheduled_batch is None:
                    break
>>>>>>> upstream/main

                self._pause_requests(scheduled_batch.paused_requests)

                finished_requests = []

<<<<<<< HEAD
                if scheduled_batch.batch_size > 0:
=======
                if scheduled_batch.batch_size > 0 or (
                        self.enable_attention_dp and self.dist.tp_size > 1):
>>>>>>> upstream/main
                    if self.kv_cache_transceiver:
                        # For generation requests which have completed KV cache transfer
                        self._prepare_disagg_gen_transmission_complete(
                            scheduled_batch)

<<<<<<< HEAD
                    self.resource_manager.prepare_resources(scheduled_batch)
                    if self.draft_model_engine is not None:
                        self._prepare_draft_tokens(scheduled_batch)

                    batch_outputs = self._forward_step(scheduled_batch)
=======
                        # Return the first token to the client
                        self._handle_first_token_response(scheduled_batch)
                    self.resource_manager.prepare_resources(scheduled_batch)

                    self._kv_connector_start_batch(scheduled_batch)

                if scheduled_batch.batch_size > 0 or (
                        self.enable_attention_dp and self.dist.tp_size > 1):
                    # init_disagg_gen_requests must be before drafter loop, otherwise draft requests do not have initialized matchers.
                    # init_disagg_gen_requests must be before engine forward, where the prev_seq_slot is updated.
                    if self.guided_decoder is not None:
                        self.guided_decoder.add_batch(scheduled_batch)
                        if self.kv_cache_transceiver:
                            self.guided_decoder.init_disagg_gen_requests()

                    if self.drafter is not None and self.use_spec_decode:
                        if self.guided_decoder is not None:
                            self.guided_decoder.rollback_rejected_tokens()
                        with request_context(
                                is_draft=self.draft_model_engine is not None,
                                scheduled_requests=scheduled_batch):
                            self.drafter.prepare_draft_tokens(
                                scheduled_batch, self.resource_manager)
                            # Pad draft tokens to the max draft length. This is for CUDA graph compatibility.
                            self.drafter.pad_draft_tokens_for_cuda_graph(
                                scheduled_batch)
                        # add_batch must be called again to restore to target requests with updated draft tokens.
                        if self.guided_decoder is not None:
                            self.guided_decoder.add_batch(scheduled_batch)
                            if hasattr(self.drafter, "guided_decoder"):
                                self.guided_decoder.rollback_draft_tokens()

                    batch_outputs = self._forward_step(scheduled_batch)
                    if self.guided_decoder is not None:
                        self.guided_decoder.execute(batch_outputs['logits'])
>>>>>>> upstream/main

                    sample_state = self._sample_async(scheduled_batch,
                                                      batch_outputs)

                    self._update_request_states(scheduled_batch)
<<<<<<< HEAD

                    ctx_transmission_reqs = self._send_disagg_ctx_cache(
                        scheduled_batch.context_requests
                    ) if self.kv_cache_transceiver else []

                    self._update_requests(sample_state)

                    if self.kv_cache_transceiver:
                        # For context only req in transmission, we reset the state since decoder might have changed it
                        for req in ctx_transmission_reqs:
                            req.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

                    self._handle_cancelled_requests()
=======
                    self._update_requests(sample_state)

                    if self.kv_cache_transceiver:
                        ctx_transmission_reqs = self._send_disagg_ctx_cache(
                            scheduled_batch.context_requests)
                        # For context only req in transmission, we reset the state since sampler might have changed it
                        for req in ctx_transmission_reqs:
                            req.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

                    self._handle_canceled_requests()
>>>>>>> upstream/main
                    finished_requests = self._handle_responses()
                    self.resource_manager.update_resources(scheduled_batch)
                    if self.enable_kv_cache_events:
                        self._add_kv_cache_events()

                if self.kv_cache_transceiver and self.ctx_in_transmission_requests:
                    self._terminate_ctx_finished_requests()

<<<<<<< HEAD
                self._gather_dp_requests_num()
=======
                self._kv_connector_terminate_requests()
>>>>>>> upstream/main

                if self.enable_iter_perf_stats:
                    iter_stats.inflight_batching_stats.num_ctx_tokens = self.model_engine.iter_states[
                        'num_ctx_tokens']
                    self._process_iter_stats(
                        finished_requests, self.active_requests,
                        BatchState(sample_state=SampleState(
                            scheduled_requests=scheduled_batch),
                                   iter_stats=iter_stats,
                                   iter_start_time=iter_start_time))

<<<<<<< HEAD
        self._executor_loop_cleanup()

=======
>>>>>>> upstream/main
    def _prepare_draft_requests(self):
        try:
            # Set draft tokens here to make the KV cache manager
            # and scheduler aware of them.
            for req in self.active_requests:
<<<<<<< HEAD
                if req.state != LlmRequestState.GENERATION_IN_PROGRESS:
                    continue
                req.py_last_draft_tokens = req.py_draft_tokens
                max_draft_len = self.model_engine.spec_config.max_draft_tokens

                if max_draft_len > 0:
                    req.py_draft_tokens = [0] * max_draft_len
                    req.py_draft_pages_allocated = max_draft_len
                else:
                    req.py_draft_tokens = None
=======
                if req.state not in (LlmRequestState.GENERATION_IN_PROGRESS,
                                     LlmRequestState.DISAGG_GENERATION_INIT):
                    continue

                req.py_last_draft_tokens = req.py_draft_tokens
                max_draft_len = self.model_engine.spec_config.max_draft_len

                if max_draft_len > 0 and self.use_spec_decode:
                    req.py_draft_tokens = [0] * max_draft_len
                    req.py_draft_pages_allocated = max_draft_len
                else:
                    req.py_draft_tokens = []
>>>>>>> upstream/main
                    req.py_draft_pages_allocated = 0

        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(f"Encountered an error in decode: {error_msg}")
            self._handle_errors(error_msg)

    def _executor_loop_overlap(self):
        torch.cuda.set_device(self.device_id)
<<<<<<< HEAD
        got_finish_signal = False
        num_dummy_request = 0
        with self._profiler() as profile_step:
            iter_start_time = time.time()
            iter_stats = None
            while not got_finish_signal or len(self.active_requests) > 0:
                profile_step()
                if self.enable_iter_perf_stats:
                    iter_start_time = time.time()
                new_requests = self._fetch_new_requests()
                got_finish_signal = self._merge_requests(
                    new_requests) or got_finish_signal
                if got_finish_signal and len(self.active_requests) == 0:
                    break

                if self.kv_cache_transceiver:
                    self._check_disagg_gen_transfer_status()

                if self.enable_iter_perf_stats:
                    iter_stats = self._get_init_iter_stats(
                        len(new_requests),
                        self.new_active_requests_queue_latency_ms)

                if not got_finish_signal:
                    num_dummy_request = self._get_num_dummy_request()
                if num_dummy_request > 0:
                    self._merge_dummy_request(num_dummy_request)
                scheduled_batch, fitting_disagg_gen_init_requests, num_fitting_reqs = self._schedule(
                )

                if self.kv_cache_transceiver:

                    # For requests that are fitting disagg gen init, also prepare resources for KV cache manager
                    self._prepare_disagg_gen_init(
                        fitting_disagg_gen_init_requests)

                    if num_fitting_reqs == 0 and not fitting_disagg_gen_init_requests:
                        logger.warning(
                            "num_fitting_reqs =0 and fitting_disagg_gen_init_requests is empty , may not have enough kvCache"
                        )
                        self.kv_cache_transceiver.check_context_transfer_status(
                            1)
                else:
                    assert scheduled_batch.batch_size > 0, (
                        "fail to schedule any pending request, "
                        "probably run out of resource.")

                self.num_scheduled_requests = scheduled_batch.batch_size
                logger.debug(
                    f'has {len(self.active_requests)} active_request, '
                    f'scheduled {len(scheduled_batch.context_requests)} context requests and '
                    f'{len(scheduled_batch.generation_requests)} generation requests'
                )

=======
        # ensure the context is created, otherwise, some MPI calls will fail.
        CUASSERT(cudart.cudaSetDevice(self.device_id))
        if self.dist.rank == 0 and not self.is_warmup and self.benchmark_req_queues_size > 0 and self.kv_cache_transceiver:
            while self.executor_request_queue.get_request_queue_size(
            ) < self.benchmark_req_queues_size:
                logger.info(
                    f"sleep 5 seconds, num_request_queue: {self.executor_request_queue.get_request_queue_size()}"
                )
                time.sleep(5)

        with self._profiler() as profile_step:
            iter_start_time = time.time()
            iter_stats = None
            while True:
                profile_step()
                if self.enable_iter_perf_stats:
                    iter_start_time = time.time()

                scheduled_batch, iter_stats = self._prepare_and_schedule_batch()
                if scheduled_batch is None:
                    break

>>>>>>> upstream/main
                self._pause_requests(scheduled_batch.paused_requests)

                if scheduled_batch.batch_size > 0:
                    if self.kv_cache_transceiver:
                        # For generation requests which have completed KV cache transfer
                        self._prepare_disagg_gen_transmission_complete(
                            scheduled_batch)
<<<<<<< HEAD

                    self.resource_manager.prepare_resources(scheduled_batch)

                    generation_requests = scheduled_batch.generation_requests
=======
                    self.resource_manager.prepare_resources(scheduled_batch)

                    self._kv_connector_start_batch(scheduled_batch)

                if scheduled_batch.batch_size > 0:
>>>>>>> upstream/main

                    # The generation requests that are do not have batch_idx,
                    # needs to be in front of the batch due to the assumptions
                    # made in model_engine.py::_forward_step. This is only important
                    # for disaggregated serving. For non-disaggregated serving,
                    # the generation requests always have batch_idx.
<<<<<<< HEAD
                    new_generation_requests = []
                    for req in generation_requests:
                        if req.py_batch_idx is None:
                            new_generation_requests.append(req)

                    for req in generation_requests:
                        if req.py_batch_idx is not None:
                            new_generation_requests.append(req)
                    scheduled_batch.generation_requests = new_generation_requests

                    previous_tensors_device = self.previous_batch and self.previous_batch.sample_state.device
=======
                    scheduled_batch.generation_requests = sorted(  # stable sort
                        scheduled_batch.generation_requests,
                        key=lambda req: int(req.py_batch_idx is not None),
                    )

                    if self.kv_cache_transceiver:
                        # Return the first token to the client
                        self._handle_first_token_response(scheduled_batch)

                    # init_disagg_gen_requests must be before engine forward, where the prev_seq_slot is updated.
                    if self.guided_decoder is not None and self.kv_cache_transceiver:
                        self.guided_decoder.add_batch(scheduled_batch)
                        self.guided_decoder.init_disagg_gen_requests()

                    previous_tensors = self.previous_batch and self.previous_batch.sample_state
                    target_inputs = None
                    draft_outputs = None
                    # If there are previous draft tokens, we need to update the target requests to accept some draft tokens.
                    # When there's any accepted tokens, we can't directly use the previous batch's outputs in this iteration for the target model,
                    # so we'll set the target model's input to None and skip updating the target requests after target model forward.
                    use_previous_draft_tokens = self.has_previous_draft_tokens
                    if self.drafter is not None and (self.use_spec_decode or
                                                     use_previous_draft_tokens):
                        target_inputs, draft_outputs, draft_batch = self._handle_speculative_decoding(
                            scheduled_batch, previous_tensors)

                    # Use the draft_model's outputs if we've launched the draft model.
                    # Otherwise, use the previous batch's outputs.
                    if target_inputs is not None or use_previous_draft_tokens:
                        previous_tensors_device = target_inputs
                    else:
                        previous_tensors_device = self.previous_batch and self.previous_batch.sample_state and self.previous_batch.sample_state.device
>>>>>>> upstream/main

                    batch_outputs = self._forward_step(scheduled_batch,
                                                       previous_tensors_device)

<<<<<<< HEAD
                    sample_state = self._sample_async(scheduled_batch,
                                                      batch_outputs)
=======
                    if target_inputs is not None:
                        self._process_draft_results(scheduled_batch,
                                                    draft_outputs, draft_batch)
                    elif self.previous_batch is not None and not use_previous_draft_tokens:
                        self._update_requests(self.previous_batch.sample_state)

                    if self.guided_decoder is not None:
                        # add_batch must be called again to have updated new tokens.
                        self.guided_decoder.add_batch(scheduled_batch)
                        self.guided_decoder.execute(batch_outputs['logits'])

                    sample_state = self._sample_async(scheduled_batch,
                                                      batch_outputs)
                    assert sample_state is not None, "Sampling failed"
>>>>>>> upstream/main

                    self._update_request_states(scheduled_batch)

                    ctx_transmission_reqs = self._send_disagg_ctx_cache(
                        scheduled_batch.context_requests
                    ) if self.kv_cache_transceiver else []

<<<<<<< HEAD
                    if num_dummy_request > 0:
                        self._finish_dummy_request(scheduled_batch)

                    has_previous_batch = self.previous_batch is not None
                    if has_previous_batch:
                        previous_batch_size = self.previous_batch.sample_state.scheduled_requests.batch_size
                        if previous_batch_size > 0:  # first previous batch size is 0
                            self._process_previous_batch()
                        self.previous_batch: Optional[BatchState] = None

                    # Separate chunked requests so we can handle them in _update_requests w/o relying on the request state.
                    # This is necessary because _forward_step updates the state before _update_requests is executed.
                    scheduled_batch.chunked_requests = [
                        r for r in scheduled_batch.context_requests
                        if r.get_context_remaining_length() != 0
                    ]
                    scheduled_batch.context_requests = [
                        r for r in scheduled_batch.context_requests
                        if r.get_context_remaining_length() == 0
                    ]
=======
                    if self.previous_batch is not None:
                        self._process_previous_batch()
>>>>>>> upstream/main

                    if self.enable_iter_perf_stats:
                        iter_stats.inflight_batching_stats.num_ctx_tokens = self.model_engine.iter_states[
                            'num_ctx_tokens']

                    self.previous_batch = BatchState(
                        sample_state=sample_state,
                        iter_start_time=iter_start_time,
                        iter_stats=iter_stats,
                        ctx_transmission_reqs=ctx_transmission_reqs)
<<<<<<< HEAD
                    self._gather_dp_requests_num()
=======
>>>>>>> upstream/main

                if self.kv_cache_transceiver and self.ctx_in_transmission_requests:
                    self._terminate_ctx_finished_requests()

<<<<<<< HEAD
        self._executor_loop_cleanup()

    def _process_previous_batch(self):
        self._update_requests(self.previous_batch.sample_state)

=======
                self._kv_connector_terminate_requests()

    def _process_previous_batch(self):
>>>>>>> upstream/main
        if self.kv_cache_transceiver and self.previous_batch.ctx_transmission_reqs:
            for req in self.previous_batch.ctx_transmission_reqs:
                req.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

<<<<<<< HEAD
        self._handle_cancelled_requests()
=======
        self._handle_canceled_requests()
>>>>>>> upstream/main
        finished_requests = self._handle_responses()
        scheduled_requests = self.previous_batch.sample_state.scheduled_requests
        self.resource_manager.update_resources(scheduled_requests)
        if self.enable_kv_cache_events:
            self._add_kv_cache_events()

        if self.enable_iter_perf_stats:
            self._process_iter_stats(finished_requests, self.active_requests,
                                     self.previous_batch)

    @nvtx_range("_forward_step_inter_pp")
    def _forward_step_inter_pp(self, scheduled_batch) -> SampleState:
<<<<<<< HEAD
        batch_outputs = self._forward_step(scheduled_batch)
        sample_state = self._sample_async(scheduled_batch, batch_outputs)
        self._update_request_states(scheduled_batch)
        sample_state.sampler_event.synchronize()
        return sample_state

    def _update_new_active_requests_queue_latency(self, new_requests):
        if self.enable_iter_perf_stats and self.dist.rank == 0:
            now = time.time()
            for req in new_requests:
                if isinstance(req, tuple):
                    req_id = req[0]
                    if req_id in self.start_times:
                        self.new_active_requests_queue_latency_ms += now - self.start_times.pop(
                            req_id)

    @nvtx_range("_broadcast_new_requests")
    def _broadcast_new_requests(
        self,
        new_requests: List[ExecutorRequest],
        py_request_objects: tuple[str, dict] = None
    ) -> tuple[List[ExecutorRequest], Optional[tuple[str, dict]]]:
        """Broadcasts new_requests and optional Python-only metadata (`py_request_objects`) across pipeline stages.
           `py_request_objects` is a tuple of (attribute_name, {request_id: object}).
        """
        payloads = (new_requests, py_request_objects
                    ) if py_request_objects is not None else new_requests

        if not self.dist.has_pp:
            result = self.dist.broadcast(payloads, root=0)
            return result if isinstance(result, tuple) else (result, None)

        # broadcast within first tp group before send/recv chain to other tp groups
        if self.dist.tp_size > 1 and self.dist.is_first_pp_rank:
            payloads = self.dist.tp_broadcast(payloads, root=0)

        # tag = [0, num_micro_batches - 1] used for new_tokens send/recv
        tag = self.num_micro_batches

        # 1. send metadata: len(num_requests) and serialized buffer size
        new_requests = payloads[0] if isinstance(payloads, tuple) else payloads
        if self.dist.is_first_pp_rank and len(new_requests) > 0:
            buf = np.array(bytearray(dill.dumps(payloads)))
            buf_size = len(buf)
        else:
            buf, buf_size = None, 0
        metadata_arr = np.array([len(new_requests), buf_size])

        if not self.dist.is_first_pp_rank:
            self.dist.recv(metadata_arr, self.dist.prev_pp_rank, tag)

        if not self.dist.is_last_pp_rank:
            self.dist.send(metadata_arr, self.dist.next_pp_rank, tag)

        # 2. send serialized buffer when new requests is not empty
        num_new_requests = metadata_arr[0]
        if num_new_requests > 0:
            buf_size = metadata_arr[1]
            if not self.dist.is_first_pp_rank:
                buf = np.array(bytearray(buf_size))
                self.dist.recv(buf, self.dist.prev_pp_rank, tag)

            if not self.dist.is_last_pp_rank:
                self.dist.send(buf, self.dist.next_pp_rank, tag)

            if not self.dist.is_first_pp_rank:
                buf_data = dill.loads(buf.tobytes())  # nosec B301
                if isinstance(buf_data, tuple):
                    new_requests, py_request_objects = buf_data
                else:
                    new_requests = buf_data

                assert len(new_requests) == num_new_requests

        return new_requests, py_request_objects

    @nvtx_range("_fetch_new_requests")
    def _fetch_new_requests(self):
        if self.enable_attention_dp:
            total_num_active_requests = sum(self.all_ranks_num_active_requests)
            total_max_num_active_requests = self.dist.tp_size * self.max_num_active_requests
        else:
            total_num_active_requests = len(self.active_requests)
            total_max_num_active_requests = self.max_num_active_requests

        timeout = None if total_num_active_requests == 0 else datetime.timedelta(
            0)
        new_requests = []
        if self.dist.rank == 0:
            new_requests = _get_from_request_queue(
                self.request_queue, timeout,
                total_max_num_active_requests - total_num_active_requests)

        if self.dist.rank == 0:
            py_request_objects = self._collect_py_objects_from_requests(
                new_requests, "py_logits_post_processors")
        else:
            py_request_objects = None

        if self.dist.rank == 0:
            # Preserve original `new_requests` on rank 0 since it may contain
            # Python-only objects (e.g., custom logits processors) not serializable by pybind.
            _ = self._broadcast_new_requests(new_requests, py_request_objects)
        else:
            new_requests, py_request_objects = self._broadcast_new_requests(
                new_requests, py_request_objects)

        if py_request_objects and (self.dist.tp_size > 1
                                   or self.dist.has_pp) and self.dist.rank > 0:
            attr_name, req_obj_dict = py_request_objects
            self._attach_py_objects_to_requests(new_requests, attr_name,
                                                req_obj_dict)

        if not self.enable_attention_dp:
            self._update_new_active_requests_queue_latency(new_requests)
            return new_requests

        num_new_requests_all_ranks = len(new_requests)
        self.expected_num_active_requests = max(
            (total_num_active_requests + num_new_requests_all_ranks +
             self.dist.tp_size - 1) // self.dist.tp_size,
            max(self.all_ranks_num_active_requests),
        )

        self.has_context_request = False
        new_requests_cur_rank = []
        if new_requests != [] and new_requests[
                0] != None and self.expected_num_active_requests > self.all_ranks_num_active_requests[
                    self.dist.tp_rank]:
            # Balance context tokens across ranks
            HeapVal = namedtuple(
                'HeapVal',
                [
                    'num_tokens',  # number of context tokens that have been added
                    'num_requests',  # number of requests to be added
                    'rank',  # rank
                    'request_list',  # new requests that have been added
                ],
            )
            all_ranks_new_requests_heap = [
                HeapVal(0, self.expected_num_active_requests - val, tp_rank, [])
                for tp_rank, val in enumerate(
                    self.all_ranks_num_active_requests)
            ]
            new_requests_cur_rank = all_ranks_new_requests_heap[
                self.dist.tp_rank].request_list
            all_ranks_new_requests_heap = [
                val for val in all_ranks_new_requests_heap
                if val.num_requests > 0
            ]
            heapq.heapify(all_ranks_new_requests_heap)
            new_requests = sorted(new_requests,
                                  key=lambda x: len(x[1].input_token_ids),
                                  reverse=True)
            for request in new_requests:
                val = heapq.heappop(all_ranks_new_requests_heap)
                val = val._replace(
                    num_tokens=val.num_tokens + len(request[1].input_token_ids),
                    num_requests=val.num_requests - 1,
                )
                val.request_list.append(request)
                if val.num_requests > 0:
                    heapq.heappush(all_ranks_new_requests_heap, val)
                elif val.rank == self.dist.tp_rank:
                    break

            # In disaggregated serving, we might get either context request or
            # generation request. In IFB, we only get context request from request queue
            if self.kv_cache_transceiver:
                for req in new_requests_cur_rank:
                    if req[1].request_type == RequestType.REQUEST_TYPE_CONTEXT_ONLY:
                        self.has_context_request = True
                        break
            else:
                self.has_context_request = len(new_requests_cur_rank) > 0
            self._update_new_active_requests_queue_latency(
                new_requests_cur_rank)

        self.num_fetch_requests = self.num_fetch_requests + num_new_requests_all_ranks
        self.num_fetch_requests_cur_rank = self.num_fetch_requests_cur_rank + len(
            new_requests_cur_rank)

        if len(new_requests) == 1 and new_requests[0] is None:
            new_requests_cur_rank = new_requests
        return new_requests_cur_rank

    @nvtx_range("_gather_dp_requests_num")
    def _gather_dp_requests_num(self):
        if self.enable_attention_dp:
            gather_active_requests = []
            responses_list = self.dist.tp_allgather(len(self.active_requests))
            for num_active_requests in responses_list:
                gather_active_requests.append(num_active_requests)
            self.all_ranks_num_active_requests = gather_active_requests

    def _add_kv_cache_events(self):
        kv_cache_manager = self.resource_manager.resource_managers.get(
            "kv_cache_manager")
=======
        self._forward_step(scheduled_batch)
        sampler_event = torch.cuda.Event()
        sampler_event.record()
        self._update_request_states(scheduled_batch)
        sampler_event.synchronize()
        return self.sampler.SampleState(
            scheduled_requests=scheduled_batch,
            sampler_event=sampler_event,
        )

    def _validate_request(self, request: LlmRequest):
        if isinstance(self.model_engine.model, DecoderModelForCausalLM):
            # Only skip token‐range checks for Llama4 when the request has multimodal data
            from ..models.modeling_llama import Llama4ForConditionalGeneration
            if isinstance(self.model_engine.model,
                          Llama4ForConditionalGeneration):
                has_mm = bool(request.py_multimodal_data)
                if has_mm:
                    logger.debug(
                        f"Skipping token-range validation for {type(self.model_engine.model).__name__} "
                        "(multimodal request)")
                    return

            # FIXME: This check is necessary because of how Qwen2ForProcessRewardModel
            #        subclasses DecoderModelForCausalLM. Perhaps the functionality
            #        of DecoderModelForCausalLM reused by Qwen2ForProcessRewardModel
            #        should be factored out into a separate class instead.
            if not hasattr(self.model_engine.model, "lm_head"):
                return

            if not request.check_token_id_range(
                    self.model_engine.model.lm_head.num_embeddings):
                raise ValueError("Token ID out of range")

    @nvtx_range("_fetch_and_activate_new_requests")
    def _fetch_and_activate_new_requests(self) -> List[LlmRequest]:

        def _respond_if_invalid(request: LlmRequest) -> bool:
            """Immediately fail invalid request.

            Return True if invalid request was encountered and
            handled.
            """
            try:
                self._validate_request(request)
                return False
            except Exception as e:
                self._handle_errors(str(e), requests=[request])
                return True

        new_requests_cur_rank = self.executor_request_queue.fetch_new_requests(
            self.active_requests)
        self.is_shutdown = self.executor_request_queue.is_shutdown
        self.expected_num_active_requests = self.executor_request_queue.get_expected_num_active_requests(
        )

        validated_requests = [
            request for request in new_requests_cur_rank
            if not _respond_if_invalid(request)
        ]

        self.active_requests.extend(validated_requests)
        return validated_requests

    def _add_kv_cache_events(self):
        kv_cache_manager = self.resource_manager.resource_managers.get(
            ResourceManagerType.KV_CACHE_MANAGER)
>>>>>>> upstream/main
        if not kv_cache_manager:
            return
        # Flush iteration events at each iteration to ensure that events have enough time
        # to be transferred to main thread when user needs them.
        kv_cache_manager.flush_iteration_events()

<<<<<<< HEAD
    def _merge_tp_requests(self, new_requests: List[ExecutorRequest]):
        for request in new_requests:
            if request is None:
                return True
        for req_item in new_requests:
            if _is_executor_request(req_item):
                req_id, exe_req = req_item
                req = executor_request_to_llm_request(req_id, exe_req)
                self.active_requests.append(req)
            elif _is_cancel_request(req_item):
                self.canceled_req_ids.insert(req_item)

        return False

    def _merge_dummy_request(self, num_dummy_request: int):
        llm_request_list = self.kv_cache_manager.add_dummy_requests(
            request_ids=list(range(num_dummy_request)),
            is_gen=not self.has_context_request,
            prepare_resource=not self.has_context_request,
            max_num_draft_tokens=self.max_draft_tokens,
        )
        for llm_request in llm_request_list:
            llm_request.is_attention_dp_dummy = True
        self.active_requests += llm_request_list

    def _finish_dummy_request(self, scheduled_requests: ScheduledRequests):
        for req in scheduled_requests.context_requests:
            if req.is_attention_dp_dummy:
                req.state = LlmRequestState.GENERATION_COMPLETE
        for req in scheduled_requests.generation_requests:
            if req.is_attention_dp_dummy:
                req.state = LlmRequestState.GENERATION_COMPLETE
        for req in self.active_requests[:]:
            if req.is_attention_dp_dummy:
                self.inflight_req_ids.erase(req.request_id)
                self._terminate_request(req)
                self.active_requests.remove(req)

    def _collect_py_objects_from_requests(
            self, requests, attribute_name: str) -> Optional[tuple[str, dict]]:
        """WAR to gather dynamic Python-only attributes (e.g., custom logits processors)
        that cannot be handled by pybind serialization during MP communication.

        Returns:
            A tuple of (attribute_name, {request_id: object}) or None.
        """
        req_id_to_obj = {}
        for item in requests:
            if item is None:
                continue
            req_id, req = item[:2]
            obj = getattr(req, attribute_name, None)
            if obj is not None:
                req_id_to_obj[req_id] = obj
        return None if not req_id_to_obj else (attribute_name, req_id_to_obj)

    def _attach_py_objects_to_requests(self, requests, attribute_name: str,
                                       py_request_objects: dict):
        """Attaches Python-only objects (e.g., dynamic attributes not handled by pybind)
        to each request.
        """
        for item in requests:
            if item is None:
                continue
            req_id, req = item[:2]
            py_obj = py_request_objects.get(req_id)
            if py_obj is not None:
                setattr(req, attribute_name, py_obj)

    def _partition_context(self, ctx_ids_list):
        ctx_ids = torch.tensor(ctx_ids_list).unsqueeze(0)
        ctx_len = ctx_ids.shape[-1]
        block_size = self.dist.cp_config['block_size']
        if block_size is None:
            block_size = ctx_len // self.dist.cp_size
        anchor_block_size = self.dist.cp_config['cp_anchor_size']
        if anchor_block_size is None:
            anchor_block_size = block_size

        assert anchor_block_size <= block_size, f'cp_anchor_size {anchor_block_size} should be smaller than block_size {block_size}'
        padding = 0
        if ctx_len % block_size != 0:
            padding = block_size - (ctx_len % block_size)
            assert padding <= ctx_len, f'block size is too large for context, please set it smaller'
            ctx_ids = torch.cat(
                (ctx_ids, torch.zeros_like(ctx_ids)[:, :padding]), dim=-1)
        position_ids = torch.arange(0, ctx_ids.shape[-1]).unsqueeze(0)

        ctx_ids_blocks = torch.tensor_split(
            torch.stack(ctx_ids.split(block_size, dim=-1)), self.dist.cp_size)
        position_ids_blocks = torch.tensor_split(
            torch.stack(position_ids.split(block_size, dim=-1)),
            self.dist.cp_size)
        if self.dist.cp_rank != 0:
            ctx_blocks, position_blocks = [
                ctx_ids_blocks[0][0].tolist()[0][:anchor_block_size]
            ], [position_ids_blocks[0][0].tolist()[0][:anchor_block_size]]
        else:
            ctx_blocks, position_blocks = [], []

        for idx in range(len(ctx_ids_blocks[self.dist.cp_rank])):
            ctx_block = ctx_ids_blocks[self.dist.cp_rank][idx]
            position_block = position_ids_blocks[self.dist.cp_rank][idx]
            ctx_blocks.append(ctx_block.tolist()[0])
            position_blocks.append(position_block.tolist()[0])
        return ctx_blocks, position_blocks, padding

    def _merge_star_attention_requests(self,
                                       new_requests: List[ExecutorRequest]):
        for request in new_requests:
            if request is None:
                return True
        for req_item in new_requests:
            if _is_executor_request(req_item):
                req_id, exe_req, query_token_ids = req_item
                ctx_len0 = len(exe_req.input_token_ids)
                ctx_blocks, position_blocks, last_block_padding_num = [
                    exe_req.input_token_ids
                ], [[i for i in range(ctx_len0)]], 0
                ctx_blocks, position_blocks, last_block_padding_num = self._partition_context(
                    exe_req.input_token_ids)
                if self.dist.cp_rank == self.dist.cp_size - 1 and last_block_padding_num > 0:
                    ctx_blocks[-1] = ctx_blocks[-1][:-last_block_padding_num]
                    position_blocks[-1] = position_blocks[
                        -1][:-last_block_padding_num]
                #if has query
                if query_token_ids:
                    ctx_blocks.append(query_token_ids)
                    position_blocks.append([
                        i for i in range(ctx_len0, ctx_len0 +
                                         len(query_token_ids))
                    ])

                # insert the dummy block to align the number of ctx iterations of each rank
                block_size = self.dist.cp_config['block_size']
                total_blocks = (ctx_len0 + block_size - 1) // block_size
                num_blocks_per_rank = (
                    total_blocks + self.dist.cp_size -
                    1) // self.dist.cp_size + 1  # 1 for query block
                if len(ctx_blocks) == num_blocks_per_rank:
                    ctx_blocks.insert(1, [])
                    position_blocks.insert(1, [])
                elif len(ctx_blocks) == num_blocks_per_rank + 1:
                    # anchor + ctx_blocks + qry_block
                    pass
                else:
                    print(
                        f'rank = {self.dist.cp_rank}, len(ctx_blocks)  = {len(ctx_blocks) }, num_blocks_per_rank = {num_blocks_per_rank}'
                    )
                    assert False, f'invalid context partition'

                # fake data for scheduler
                ctx_blocks_list = [0] * (block_size +
                                         self.dist.cp_config['cp_anchor_size'])

                req = executor_request_to_llm_request(req_id, exe_req,
                                                      ctx_blocks_list)
                req.gen_iters = 0
                req.ctx_iters = 0
                req.ctx_blocks = ctx_blocks
                req.ctx_position_blocks = position_blocks
                req.query_id = query_token_ids
                self.active_requests.append(req)
            elif _is_cancel_request(req_item):
                self.canceled_req_ids.insert(req_item)

        return False

    @nvtx_range("_merge_requests")
    def _merge_requests(self, new_requests: List[ExecutorRequest]):
        cp_config = self.dist.cp_config
        if 'cp_type' in cp_config:
            cp_type = cp_config['cp_type']
            if cp_type == 'star_attention':
                ret = self._merge_star_attention_requests(new_requests)
            elif cp_type == 'ring_attention':
                raise NotImplementedError("ring attention not implemented yet")
            else:
                raise NotImplementedError(f'unsupport cp type {cp_type}')
        else:
            ret = self._merge_tp_requests(new_requests)
        return ret
=======
    def _balance_adp_requests(self, context_requests: list[LlmRequest],
                              generation_requests: list[LlmRequest]):
        balanced_context_requests = context_requests
        num_scheduled_context_requests = len(context_requests)
        num_scheduled_generation_requests = len(generation_requests)
        num_scheduled_tokens = sum(
            [len(req.get_tokens(0))
             for req in context_requests]) + num_scheduled_generation_requests
        responses_list = self.dist.tp_allgather([
            num_scheduled_context_requests, num_scheduled_generation_requests,
            num_scheduled_tokens
        ])
        all_ranks_num_scheduled_context_requests = [
            response[0] for response in responses_list
        ]
        all_ranks_num_scheduled_generation_requests = [
            response[1] for response in responses_list
        ]
        all_ranks_have_free_ctx_slots = all([
            num_gen < self.max_batch_size
            for num_gen in all_ranks_num_scheduled_generation_requests
        ])
        all_ranks_have_ctx_requests = all([
            num_ctx > 0 for num_ctx in all_ranks_num_scheduled_context_requests
        ])
        all_ranks_have_gen_requests = all([
            num_gen > 0
            for num_gen in all_ranks_num_scheduled_generation_requests
        ])

        if self.attention_dp_enable_balance:
            # wait for all ranks have context requests
            if all_ranks_have_free_ctx_slots and all_ranks_have_ctx_requests:
                self.adp_ctx_waiting_iters_count = 0
                # balance number of context requests across ranks
                if all_ranks_have_gen_requests:
                    if self.adp_ctx_batching_wait_iters_count < self.attention_dp_batching_wait_iters:
                        self.adp_ctx_batching_wait_iters_count += 1
                        balanced_context_requests = []
                    else:
                        self.adp_ctx_batching_wait_iters_count = 0
            else:
                self.adp_ctx_waiting_iters_count += 1
                balanced_context_requests = []
                timeout_reached = self.adp_ctx_waiting_iters_count >= self.attention_dp_time_out_iters
                if timeout_reached or not all_ranks_have_gen_requests:
                    self.adp_ctx_waiting_iters_count = 0
                    balanced_context_requests = context_requests
        return balanced_context_requests

    def _waiting_requests(self, context_requests: list[LlmRequest],
                          generation_requests: list[LlmRequest]):
        if not self.enable_batch_waiting:
            return context_requests

        waited_context_requests = []
        stop_waiting = False
        num_scheduled_ctx_tokens = sum(
            len(ctx_req.get_tokens(0)) for ctx_req in context_requests)
        num_scheduled_gen_tokens = sum(1 + gen_req.num_draft_tokens
                                       for gen_req in generation_requests)
        num_scheduled_tokens = num_scheduled_ctx_tokens + num_scheduled_gen_tokens

        stop_waiting = self.batch_wait_iters_count >= self.batch_wait_timeout_iters or num_scheduled_tokens >= self.batch_wait_max_tokens_ratio * self.max_num_tokens
        if stop_waiting:
            waited_context_requests = context_requests
            self.batch_wait_iters_count = 0
        else:
            self.batch_wait_iters_count += 1
        return waited_context_requests
>>>>>>> upstream/main

    @nvtx_range("_schedule")
    def _schedule(self):
        scheduler_output = self.scheduler.schedule_request(
            self.active_requests, self.inflight_req_ids)
<<<<<<< HEAD
        scheduled_requests = ScheduledRequests()

        scheduled_requests.context_requests = scheduler_output.context_requests
=======
        scheduled_context_requests = scheduler_output.context_requests
        if self.enable_attention_dp and self.attention_dp_enable_balance:
            scheduled_context_requests = self._balance_adp_requests(
                scheduler_output.context_requests,
                scheduler_output.generation_requests)

        # if no generation requests, no need to wait, to avoid dead waiting
        if not self.enable_attention_dp and self.enable_batch_waiting and len(
                scheduler_output.context_requests) > 0 and len(
                    scheduler_output.generation_requests) > 0:
            scheduled_context_requests = self._waiting_requests(
                scheduler_output.context_requests,
                scheduler_output.generation_requests)

        scheduled_requests = ScheduledRequests()
        scheduled_requests.context_requests = scheduled_context_requests
>>>>>>> upstream/main
        scheduled_requests.generation_requests = scheduler_output.generation_requests
        scheduled_requests.paused_requests = scheduler_output.paused_requests
        return scheduled_requests, scheduler_output.fitting_disagg_gen_init_requests, scheduler_output.num_fitting_requests

    @nvtx_range("_check_disagg_gen_transfer_status")
    def _check_disagg_gen_transfer_status(self):

        need_check = any([
            req.is_disagg_generation_transmission_in_progress
            for req in self.active_requests
        ])
        need_check_one = all([
            req.is_disagg_generation_transmission_in_progress
            for req in self.active_requests
        ])

        if need_check:
            at_least_num = 1 if need_check_one else 0
<<<<<<< HEAD
            self.kv_cache_transceiver.check_gen_transfer_status(at_least_num)

        return

    @nvtx_range("_get_num_dummy_request")
    def _get_num_dummy_request(self):
        if self.enable_attention_dp:
            assert self.expected_num_active_requests >= len(
                self.active_requests)
            if self.kv_cache_transceiver is None:
                num_active_request = len(self.active_requests)
            else:
                num_active_request = sum([
                    0 if req.is_disagg_generation_init_state
                    or req.is_disagg_generation_transmission_in_progress else 1
                    for req in self.active_requests
                ])
            num_dummy_request = self.expected_num_active_requests - num_active_request
        else:
            num_dummy_request = 0
        return num_dummy_request
=======
            self._check_disagg_gen_cache_transfer_status(at_least_num)

        return

    @nvtx_range("_pad_attention_dp_dummy_request")
    def _pad_attention_dp_dummy_request(self):
        """
        Pad with a generation dummy request, if required, to ensure every attention_dp rank has at least one active request.
        """
        if not self.enable_attention_dp:
            return

        assert self.expected_num_active_requests >= len(self.active_requests)
        if self.kv_cache_transceiver is None:
            num_active_request = len(self.active_requests)
        else:
            num_active_request = sum([
                0 if req.is_disagg_generation_init_state
                or req.is_disagg_generation_transmission_in_progress else 1
                for req in self.active_requests
            ])

        if self.expected_num_active_requests - num_active_request > 0 and num_active_request == 0:
            llm_request = self.kv_cache_manager.add_dummy_requests(
                request_ids=[0],
                is_gen=True,
                prepare_resource=True,
                max_num_draft_tokens=self.max_draft_len,
            )[0]
            llm_request.is_attention_dp_dummy = True
            spec_resource_manager = self.resource_manager.get_resource_manager(
                ResourceManagerType.SPEC_RESOURCE_MANAGER)
            if spec_resource_manager is not None:
                spec_resource_manager.add_dummy_requests([0])
            self.active_requests.append(llm_request)
>>>>>>> upstream/main

    @nvtx_range("_prepare_disagg_gen_init")
    def _prepare_disagg_gen_init(self, fitting_disagg_gen_init_requests):
        if fitting_disagg_gen_init_requests:
            disagg_gen_init_to_prepare = ScheduledRequests()
            disagg_gen_init_to_prepare.context_requests = fitting_disagg_gen_init_requests
            disagg_gen_init_to_prepare.generation_requests = []
            disagg_gen_init_to_prepare.paused_requests = []

<<<<<<< HEAD
            self.resource_manager.resource_managers[
                'kv_cache_manager'].prepare_resources(
                    disagg_gen_init_to_prepare)
=======
            for resource_mgr_type in (
                    ResourceManagerType.KV_CACHE_MANAGER,
                    ResourceManagerType.SPEC_RESOURCE_MANAGER,
                    ResourceManagerType.DRAFT_KV_CACHE_MANAGER):
                if (resource_mgr_type in self.resource_manager.resource_managers
                        and self.resource_manager.
                        resource_managers[resource_mgr_type] is not None):
                    self.resource_manager.resource_managers[
                        resource_mgr_type].prepare_resources(
                            disagg_gen_init_to_prepare)
>>>>>>> upstream/main

            # Trigger KV cache exchange for new disagg_gen_init_requests
            self._recv_disagg_gen_cache(fitting_disagg_gen_init_requests)

    @nvtx_range("_prepare_disagg_gen_transmission_complete")
    def _prepare_disagg_gen_transmission_complete(self, scheduled_batch):
        cache_trans_complete_requests = []
        for req in scheduled_batch.generation_requests:
            if req.is_disagg_generation_transmission_complete:
                cache_trans_complete_requests.append(req)
        if len(cache_trans_complete_requests) > 0:
<<<<<<< HEAD
            self._setup_sampler_step(cache_trans_complete_requests)
=======
            requests = ScheduledRequests()
            requests.context_requests = cache_trans_complete_requests
            self.resource_manager.resource_managers[
                ResourceManagerType.SEQ_SLOT_MANAGER].prepare_resources(
                    requests)
            self._setup_sampler_step(requests)
>>>>>>> upstream/main

        for req in scheduled_batch.generation_requests:
            if req.is_disagg_generation_transmission_complete:
                req.state = LlmRequestState.GENERATION_IN_PROGRESS
                req.context_current_position = req.prompt_len
                req.decoding_iter = 1
                req.py_decoding_iter = 1
                first_gen_tokens = req.context_phase_params.first_gen_tokens
                ctx_draft_tokens = req.context_phase_params.draft_tokens
                req.py_draft_tokens = [] if ctx_draft_tokens is None else ctx_draft_tokens
                beam_width = req.sampling_config.beam_width
                for beam in range(0, beam_width):
                    req.add_new_token(first_gen_tokens[beam], beam)

    @nvtx_range("_recv_disagg_gen_cache")
    def _recv_disagg_gen_cache(self, new_gen_reqs):

        # For gen-only benchmarking, mark new gen request as transmission complete right away
        if os.getenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY") == "1":
            for req in new_gen_reqs:
                req.state = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
            return

        if os.getenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP") == "1":
            for req in new_gen_reqs:
                self.kv_cache_transceiver.request_and_receive_sync(req)
        else:
            for req in new_gen_reqs:
                self.kv_cache_transceiver.request_and_receive_async(req)

        block_transfer = all([
            req.is_disagg_generation_transmission_in_progress
            for req in self.active_requests
        ])
<<<<<<< HEAD
        self.kv_cache_transceiver.check_gen_transfer_status(
            1 if block_transfer else 0)
=======
        self._check_disagg_gen_cache_transfer_status(1 if block_transfer else 0)
>>>>>>> upstream/main

        return

    @nvtx_range("_send_disagg_ctx_cache")
    def _send_disagg_ctx_cache(self, scheduled_ctx_requests):
        if (scheduled_ctx_requests is None or len(scheduled_ctx_requests) == 0):
            return []
        for req in scheduled_ctx_requests:
<<<<<<< HEAD
            if req.is_context_only_request and req.is_context_finished:
                self.kv_cache_transceiver.respond_and_send_async(req)
                self.resource_manager.resource_managers[
                    "seq_slot_manager"].free_resources(req)

        self.kv_cache_transceiver.check_context_transfer_status(0)
=======
            if req.is_context_only_request and (req.is_context_finished or
                                                req.is_finished_due_to_length):
                self.kv_cache_transceiver.respond_and_send_async(req)
                for resource_mgr_type in (
                        ResourceManagerType.SEQ_SLOT_MANAGER,
                        ResourceManagerType.SPEC_RESOURCE_MANAGER):
                    if resource_mgr_type in self.resource_manager.resource_managers and self.resource_manager.resource_managers[
                            resource_mgr_type] is not None:
                        self.resource_manager.resource_managers[
                            resource_mgr_type].free_resources(req)

        self._check_disagg_ctx_cache_transfer_status(0)
>>>>>>> upstream/main

        # Keep track of ctx requests that are in transmission
        ctx_transmission_reqs = [
            req for req in scheduled_ctx_requests
            if req.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
        ]

        return ctx_transmission_reqs

<<<<<<< HEAD
=======
    def _check_request_error_state(self):
        error_requests = []
        for req in self.active_requests:
            if req.state == LlmRequestState.DISAGG_TRANS_ERROR:
                error_requests.append(req)

        return error_requests

    @nvtx_range("_check_disagg_ctx_cache_transfer_status")
    def _check_disagg_ctx_cache_transfer_status(self, atLeastNum: int = 0):
        self.kv_cache_transceiver.check_context_transfer_status(atLeastNum)

        # Check if any request is in error state
        error_requests = self._check_request_error_state()
        if len(error_requests) > 0:
            self._handle_errors(
                f"Error in kv cache transfer for context requests",
                requests=error_requests)

    @nvtx_range("_check_disagg_gen_cache_transfer_status")
    def _check_disagg_gen_cache_transfer_status(self, atLeastNum: int = 0):
        self.kv_cache_transceiver.check_gen_transfer_status(atLeastNum)

        # Check if any request is in error state
        error_requests = self._check_request_error_state()
        if len(error_requests) > 0:
            self._handle_errors(
                f"Error in kv cache transfer for generation requests",
                requests=error_requests)

>>>>>>> upstream/main
    def _forward_step(self,
                      scheduled_requests,
                      new_tensors_device: Optional[SampleStateTensors] = None):

        @nvtx_range(
<<<<<<< HEAD
            f"[Executor] _forward_step: {len(scheduled_requests.context_requests)} ctx reqs, {len(scheduled_requests.generation_requests)} gen reqs"
        )
        def forward(scheduled_requests, resource_manager, new_tensors_device):
            return self.model_engine.forward(scheduled_requests,
                                             resource_manager,
                                             new_tensors_device)

        try:
            outputs = forward(scheduled_requests, self.resource_manager,
                              new_tensors_device)
=======
            f"[Executor] _forward_step {self.model_engine.iter_counter + 1}: {len(scheduled_requests.context_requests)} ctx reqs, {len(scheduled_requests.generation_requests)} gen reqs"
        )
        def forward(scheduled_requests, resource_manager, new_tensors_device,
                    gather_context_logits, cache_indirection_buffer):
            return self.model_engine.forward(
                scheduled_requests,
                resource_manager,
                new_tensors_device,
                gather_context_logits=gather_context_logits,
                cache_indirection_buffer=cache_indirection_buffer)

        try:
            gather_context_logits = any(
                a.py_return_context_logits
                for a in scheduled_requests.context_requests)
            cache_indirection_buffer = self.sampler.get_cache_indirection()
            outputs = forward(scheduled_requests, self.resource_manager,
                              new_tensors_device, gather_context_logits,
                              cache_indirection_buffer)

            self._kv_connector_wait_for_save()

>>>>>>> upstream/main
            return outputs
        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(
                f"Encountered an error in forward function: {error_msg}")
            self._handle_errors(error_msg)
            return None

    def _update_request_states_tp(self, scheduled_requests: ScheduledRequests):
<<<<<<< HEAD
        for request in scheduled_requests.context_requests:
            request.move_to_next_context_chunk()
            if request.get_context_remaining_length() == 0:
                request.state = LlmRequestState.GENERATION_IN_PROGRESS
            if request.is_attention_dp_dummy:
                request.state = LlmRequestState.GENERATION_COMPLETE
=======
        # handle potential attention dp dummy request
        if self.active_requests and self.active_requests[
                -1].is_attention_dp_dummy:
            request = self.active_requests[-1]
            request.state = LlmRequestState.GENERATION_COMPLETE
            self.inflight_req_ids.erase(request.py_request_id)
            self._terminate_request(request)
            self.active_requests.remove(request)

        for request in scheduled_requests.context_requests:
            if request.state != LlmRequestState.GENERATION_COMPLETE:  # skip failed requests
                request.py_last_context_chunk = (
                    request.context_current_position,
                    request.context_current_position +
                    request.context_chunk_size)
                request.move_to_next_context_chunk()
            if request.context_remaining_length == 0:
                request.state = LlmRequestState.GENERATION_IN_PROGRESS
>>>>>>> upstream/main

    def _update_request_states_star_attention(
            self, scheduled_requests: ScheduledRequests):
        for request in scheduled_requests.context_requests:
            if request.ctx_iters >= len(request.ctx_blocks) - 2:
                request.state = LlmRequestState.GENERATION_IN_PROGRESS
            request.ctx_iters += 1

        for request in scheduled_requests.generation_requests:
            request.gen_iters += 1

    @nvtx_range("_update_request_states")
    def _update_request_states(self, scheduled_requests: ScheduledRequests):
        cp_config = self.dist.cp_config
        if 'cp_type' in cp_config:
            cp_type = cp_config['cp_type']
<<<<<<< HEAD
            if cp_type == 'star_attention':
=======
            if cp_type == CpType.STAR:
>>>>>>> upstream/main
                self._update_request_states_star_attention(scheduled_requests)
            else:
                assert False, f'Unsupport cp_type {cp_type}'
        else:
            self._update_request_states_tp(scheduled_requests)

    @nvtx_range("_sample_async")
    def _sample_async(self, scheduled_batch,
                      batch_outputs) -> SampleState | None:
        try:
            if batch_outputs is not None:
<<<<<<< HEAD
                return self.sampler.sample_async(scheduled_batch, batch_outputs)
=======
                num_context_logits_prefix_sum = [0]
                prefix_sum = 0
                for request in scheduled_batch.context_requests:
                    prefix_sum += request.context_chunk_size if request.py_return_context_logits else 1
                    num_context_logits_prefix_sum.append(prefix_sum)

                HandleLogits()(scheduled_batch.context_requests,
                               scheduled_batch.generation_requests,
                               batch_outputs["logits"],
                               self.sampler.beam_width(
                                   scheduled_batch.all_requests()),
                               num_context_logits_prefix_sum,
                               self.sampler.is_generation_model())

                return self.sampler.sample_async(scheduled_batch, batch_outputs,
                                                 num_context_logits_prefix_sum)
>>>>>>> upstream/main
        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(f"Encountered an error in sampling: {error_msg}")
            self._handle_errors(error_msg)

    @nvtx_range("_setup_sampler_step")
<<<<<<< HEAD
    def _setup_sampler_step(self, requests):
=======
    def _setup_sampler_step(self, requests: ScheduledRequests):
>>>>>>> upstream/main
        try:
            return self.sampler.setup_sampler_step(requests)
        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(f"Encountered an error in sampling: {error_msg}")
            self._handle_errors(error_msg)

    @nvtx_range("_update_requests")
    def _update_requests(self, sample_state: SampleState):
        try:
            self.sampler.update_requests(sample_state)
        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(f"Encountered an error in sampling: {error_msg}")
            self._handle_errors(error_msg)

<<<<<<< HEAD
    @nvtx_range("_prepare_draft_batch")
    def _prepare_draft_batch(
        self, scheduled_requests: ScheduledRequests
    ) -> Tuple[ScheduledRequests, Dict[int, LlmRequest]]:
        """
        Prepares a batch for the draft model engine. Draft tokens are only produced
        for generation requests.

        The requests are prepared as follows:
        1. The first time the draft engine sees a request, it's a context request.
        2. Otherwise, if draft tokens were accepted on the last target model decoding
        step, it's a chunked context request (we process all the accepted tokens together).
        3. Otherwise, it's a generation request.
        """
        try:
            draft_batch = ScheduledRequests()
            req_id_to_num_rejected_tokens = {}

            for request in scheduled_requests.generation_requests:
                if request.py_draft_pages_allocated == 0:
                    # No space for draft tokens.
                    continue

                num_draft_tokens = len(
                    request.py_last_draft_tokens
                ) if request.py_last_draft_tokens is not None else 0
                request.py_draft_tokens = []

                num_accepted_tokens = getattr(request,
                                              "py_num_accepted_draft_tokens", 0)
                num_rejected_tokens = num_draft_tokens - num_accepted_tokens
                assert num_rejected_tokens >= 0
                req_id_to_num_rejected_tokens[
                    request.py_request_id] = num_rejected_tokens

                spec_config = self.model_engine.spec_config
                beam_idx = 0
                input_tokens = spec_config.get_draft_model_prompt(
                    request.get_tokens()[beam_idx])

                if request.max_beam_num_tokens - 1 == request.py_prompt_len:
                    # This is the first time the draft model is seeing this request.
                    # Prepare a context request. We discard the first token and take
                    # the newly decoded one - this is the convention for EAGLE 2 and 3.
                    assert num_draft_tokens == 0
                    new_request = LlmRequest(
                        request_id=request.py_request_id,
                        max_new_tokens=request.py_max_new_tokens,
                        input_tokens=input_tokens,
                        sampling_config=request.sampling_config,
                        is_streaming=False)

                    draft_batch.context_requests.append(new_request)
                elif getattr(request, "py_num_accepted_draft_tokens", 0) == 0:
                    new_request = LlmRequest(
                        request_id=request.py_request_id,
                        max_new_tokens=request.py_max_new_tokens,
                        input_tokens=input_tokens[:-1],
                        sampling_config=request.sampling_config,
                        is_streaming=False)
                    # Explicitly add the last token so get_last_tokens() returns
                    # the right value
                    new_request.add_new_token(input_tokens[-1], beam_idx)
                    new_request.state = LlmRequestState.GENERATION_IN_PROGRESS
                    draft_batch.generation_requests.append(new_request)
                else:
                    new_request = LlmRequest(
                        request_id=request.py_request_id,
                        max_new_tokens=request.py_max_new_tokens,
                        input_tokens=input_tokens,
                        sampling_config=request.sampling_config,
                        is_streaming=False)
                    new_request.context_chunk_size = num_accepted_tokens + 1
                    new_request.context_current_position = len(
                        input_tokens) - num_accepted_tokens - 1

                    draft_batch.context_requests.append(new_request)

                new_request.py_stop_words_list = request.py_stop_words_list

            return draft_batch, req_id_to_num_rejected_tokens

        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(f"Encountered an error in decode: {error_msg}")
            self._handle_errors(error_msg)

    @nvtx_range("_prepare_draft_tokens")
    def _prepare_draft_tokens(self, scheduled_requests: ScheduledRequests):
        try:
            draft_batch, num_rejected_tokens = self._prepare_draft_batch(
                scheduled_requests)

            if draft_batch.batch_size == 0:
                return

            req_id_to_old_request = {
                req.py_request_id: req
                for req in chain(scheduled_requests.context_requests,
                                 scheduled_requests.generation_requests)
            }

            spec_metadata = self.model_engine.last_spec_metadata

            hidden_states = spec_metadata.get_hidden_states(
                draft_batch, num_rejected_tokens)

            if spec_metadata.spec_dec_mode.is_eagle3():
                # Hack for eagle3. We might need to run a matmul to reduce
                # the dimensionality of the hidden states on the first pass
                # through the draft model. Shape dependent control flow will
                # not work with CUDA graphs. So we just do it here.
                hidden_states = self.draft_model_engine.model.apply_eagle3_fc(
                    hidden_states)

            extra_model_inputs = {'hidden_states': hidden_states}

            outputs = self.draft_model_engine.forward(
                draft_batch,
                self.resource_manager,
                extra_model_inputs=extra_model_inputs)

            if spec_metadata.spec_dec_mode.is_eagle3():
                outputs['d2t'] = self.draft_model_engine.model.model.d2t.data

            sample_state = self._sample_async(draft_batch, outputs)

            self._update_request_states(draft_batch)

            self._update_requests(sample_state)

            def _process_decoded_tokens():
                new_requests = []
                for req in chain(draft_batch.context_requests,
                                 draft_batch.generation_requests):
                    target_model_req = req_id_to_old_request[req.py_request_id]
                    target_model_req.py_draft_tokens.append(
                        req.get_last_tokens(0))
                    if req.state != LlmRequestState.GENERATION_COMPLETE and len(
                            target_model_req.py_draft_tokens
                    ) < target_model_req.py_draft_pages_allocated:
                        new_requests.append(req)

                return new_requests

            # The TRTLLM attention kernels cannot handle generation requests with
            # different seqlens. No issues with flashinfer, should we look into removing
            # this? Just needs proper kernel support.
            def _pad_to_max_draft_tokens():
                for req in scheduled_requests.generation_requests:
                    max_draft_tokens = spec_metadata.max_draft_tokens
                    num_draft_tokens = len(req.py_draft_tokens)
                    req.py_draft_tokens.extend(
                        0 for _ in range(max_draft_tokens - num_draft_tokens))

            new_requests = _process_decoded_tokens()
            if not new_requests:
                _pad_to_max_draft_tokens()
                return

            draft_batch.generation_requests = new_requests
            draft_batch.context_requests = []

            for _ in range(spec_metadata.max_draft_tokens - 1):
                draft_spec_metadata = self.draft_model_engine.last_spec_metadata
                hidden_states = draft_spec_metadata.get_hidden_states(
                    draft_batch)
                extra_model_inputs = {'hidden_states': hidden_states}

                outputs = self.draft_model_engine.forward(
                    draft_batch,
                    self.resource_manager,
                    extra_model_inputs=extra_model_inputs)

                if spec_metadata.spec_dec_mode.is_eagle3():
                    outputs[
                        'd2t'] = self.draft_model_engine.model.model.d2t.data
                sample_state = self._sample_async(draft_batch, outputs)
                self._update_request_states(draft_batch)
                self._update_requests(sample_state)

                new_requests = _process_decoded_tokens()
                if not new_requests:
                    break
                draft_batch.generation_requests = new_requests

            _pad_to_max_draft_tokens()

        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(f"Encountered an error in decode: {error_msg}")
            self._handle_errors(error_msg)

    def _handle_errors(self, error_msg: Optional[str] = None):
        error_responses = {}
        error_msg = error_msg or "error"
        for request in self.active_requests:
            req_id = request.py_request_id
            request.state = LlmRequestState.GENERATION_COMPLETE
            self._terminate_request(request)
            error_responses[req_id] = ExecutorResponse(
                req_id, error_msg, client_id=request.py_client_id)
        self.active_requests.clear()
        self._enqueue_responses(error_responses)

    def _terminate_request(self, request: LlmRequest):
        self.resource_manager.free_resources(request)

    @nvtx_range("_handle_cancelled_requests")
    def _handle_cancelled_requests(self):
        #TODO: properly handle canceled ids in pp case
        if self.dist.has_tp:
            self.canceled_req_ids = self.dist.broadcast(self.canceled_req_ids,
                                                        root=0)

        if len(self.canceled_req_ids) == 0:
            return

        cancelled_responses = {}
        left_requests = []
        # Tracks canceled requests for proper handling in overlap mode during `sampler.update_requests`.
        self.canceled_requests = []
        for request in self.active_requests:
            req_id = request.py_request_id
            if req_id in self.canceled_req_ids:
                self._terminate_request(request)
                request.finish_by_reason(FinishReason.CANCELLED)
                request.decoding_iter = request.py_decoding_iter
                cancelled_responses[req_id] = request.create_response(
                    False, self.dist.rank)
                self.canceled_requests.append(request)
                self.canceled_req_ids.erase(req_id)
            else:
                left_requests.append(request)
        self.active_requests = left_requests

        # enqueue the cancelled requests' responses as they are not
        # active_requests and be discarded in the decoder loop.
        self._enqueue_responses(cancelled_responses)

    @nvtx_range("_enqueue_responses")
    def _enqueue_responses(self, responses: Dict[int, ExecutorResponse]):
=======
    def _handle_errors(self,
                       error_msg: Optional[str] = None,
                       *,
                       requests: Optional[List[LlmRequest]] = None):
        error_responses: Dict[int, LlmResponse] = {}
        error_msg = error_msg or "error"
        failed_requests = requests if requests is not None else self.active_requests
        for request in failed_requests:
            req_id = request.py_request_id
            request.state = LlmRequestState.GENERATION_COMPLETE
            self._terminate_request(request)
            error_responses[req_id] = LlmResponse(
                request_id=req_id,
                error_msg=error_msg,
                client_id=request.py_client_id)
        if requests is None:
            self.active_requests.clear()
        else:
            self.active_requests = [
                request for request in self.active_requests
                if request not in requests
            ]
        self._enqueue_responses(error_responses.items())

    def _terminate_request(self, request: LlmRequest):
        if self._disagg_pp_termination_handler is not None:
            self._disagg_pp_termination_handler.terminate(request)
        else:
            self._do_terminate_request(request)

    def _do_terminate_request(self, request: LlmRequest):
        if self.kv_connector_manager is not None:
            # Only call request_finished on the connector if the request has already been added to the kv cache manager.
            try:
                cache_block_ids = self.kv_cache_manager.get_cache_indices(
                    request)
            except IndexError:
                # If the request has not yet been added to the kv cache manager,
                # we still need to free resources corresponding to other resource managers.
                self.resource_manager.free_resources(request)
            else:
                if not self.kv_connector_manager.request_finished(
                        request, cache_block_ids):
                    self.resource_manager.free_resources(request)
        else:
            self.resource_manager.free_resources(request)

    @nvtx_range("_handle_canceled_requests")
    def _handle_canceled_requests(self):
        if self.executor_request_queue.get_canceled_req_ids_size() == 0:
            return

        # Remove cancel request in the waiting queue
        self.executor_request_queue.update_waiting_queue()

        for request in self.active_requests:
            req_id = request.py_request_id if not request.is_child else request.parent_request_id
            if req_id in self.executor_request_queue.get_canceled_req_ids():
                # Mark requests as finished, then, we reuse all existing code
                # to clean up the KV cache resources.
                request.finish_by_reason(FinishReason.CANCELLED)
                request.decoding_iter = request.py_decoding_iter

        if self.enable_attention_dp:
            # TODO: revisit the cancel logic of attention dp
            # When enable attention dp, each rank does not have full copy of requests
            # so we need to remove the cancel requests not in the local rank
            self.executor_request_queue.clear_canceled_req_ids()

    @nvtx_range("_enqueue_responses")
    def _enqueue_responses(self, responses: Iterable[Tuple[int, LlmResponse]]):
>>>>>>> upstream/main
        if 0 not in self.dist.mapping.tp_group and not self.gather_all_responses:
            return

        logger.debug(
            f'before gather, rank = {self.dist.rank}, responses = {responses}')
<<<<<<< HEAD
        if self.enable_attention_dp:
=======
        if self.enable_attention_dp and self.dist.world_size != 1:
>>>>>>> upstream/main
            if not self.gather_all_responses:
                responses_list = self.dist.tp_gather(responses)
            else:
                responses_list = self.dist.allgather(responses)
            if self.dist.rank == 0 or self.gather_all_responses:
<<<<<<< HEAD
                gather_responses = {}
                if responses_list is not None:
                    for resp in responses_list:
                        gather_responses.update(resp)
=======
                gather_responses = []
                if responses_list is not None:
                    for resp in responses_list:
                        if resp is not None:
                            gather_responses.extend(resp)
>>>>>>> upstream/main
                    responses = gather_responses
        logger.debug(
            f'after gather, rank = {self.dist.rank}, responses = {responses}')

        if self.dist.rank == 0 or self.gather_all_responses:
            with self.response_cv:
<<<<<<< HEAD
                for req_id, resp in responses.items():
=======
                for req_id, resp in responses:
>>>>>>> upstream/main
                    if req_id in self.responses.keys():
                        self.responses[req_id].append(resp)
                    else:
                        self.responses.update({req_id: [resp]})
                self.response_cv.notify_all()

<<<<<<< HEAD
    @nvtx_range("_handle_responses")
    def _handle_responses(self):
        new_responses = {}
=======
    @nvtx_range("_handle_first_token_response")
    def _handle_first_token_response(self, scheduled_batch):
        new_responses = []
        for req in scheduled_batch.generation_requests:
            if req.py_decoding_iter == 1:
                logger.debug(
                    f'Send first token response for request {req.py_request_id}'
                )
                response = req.create_response(False, self.dist.rank)
                new_responses.append((req.py_request_id, response))

        self._enqueue_responses(new_responses)

    @nvtx_range("_handle_responses")
    def _handle_responses(self):
        new_responses = []
>>>>>>> upstream/main
        requests_to_terminate = []
        new_active_requests = []
        logger.debug(
            f'------before _handle_responses, rank = {self.dist.rank}, output = {self.active_requests}'
        )
        for request in self.active_requests:
            req_id = request.py_request_id
            # no responses for dummy request, and finish it
            if request.is_attention_dp_dummy:
                requests_to_terminate.append(request)
                continue

<<<<<<< HEAD
            if request.is_generation_only_request:
                # If request is in transmission, so we don't need to emit a response
                # Also, for the first iteration with overlap, we should skip since first token has already been emitted by context server
=======
            if request.is_generation_only_request():
                # If request is in transmission, so we don't need to emit a response
                # Also, for the first iteration with overlap, we should skip since first
                # token has already been emitted previously
>>>>>>> upstream/main
                if request.is_disagg_generation_transmission_in_progress or (
                        not self.disable_overlap_scheduler
                        and request.py_decoding_iter <= 1):
                    new_active_requests.append(request)
                    continue

            request.draft_tokens = request.py_draft_tokens
            request.decoding_iter = request.py_decoding_iter
<<<<<<< HEAD
            response: Response = request.create_response(False, self.dist.rank)
            request_done = False
            if response:
                request_done = response.result.is_final
                new_responses.update({req_id: response})
=======

            if request.return_perf_metrics:
                request.update_perf_metrics(self.model_engine.iter_counter)

            request_done = False
            if request.py_decoding_iter == 1 or request.is_finished or \
                    request.py_decoding_iter % self.stream_interval == 0:
                response = request.create_response(False, self.dist.rank)
                if response:
                    request_done = request.is_finished
                    new_responses.append((req_id, response))

>>>>>>> upstream/main
            if request_done:
                if request.is_disagg_context_transmission_state:
                    self.ctx_in_transmission_requests.append(request)
                else:
                    requests_to_terminate.append(request)
            else:
                new_active_requests.append(request)
<<<<<<< HEAD
        self.active_requests = new_active_requests
=======
        self.active_requests.clear()
        self.active_requests.extend(new_active_requests)
>>>>>>> upstream/main
        self._enqueue_responses(new_responses)
        for request in requests_to_terminate:
            self._terminate_request(request)
        return requests_to_terminate

    @nvtx_range("_terminate_ctx_finished_requests")
    def _terminate_ctx_finished_requests(self):
        for request in self.ctx_in_transmission_requests[:]:
            if request.is_disagg_context_complete_state:
                self._terminate_request(request)
                self.ctx_in_transmission_requests.remove(request)

<<<<<<< HEAD
    def _await_any_response(self,
                            timeout: Union[float, None] = None
                            ) -> List[ExecutorResponse]:
=======
    def _handle_logits_communication(self, previous_batch, prev_microbatch_id):
        """Handle logits communication between pipeline parallel ranks.

        If logits were requested, the last PP rank sends to the first PP rank (who sends responses)
        the logits of the requests that have finished.

        Args:
            previous_batch: The previous batch state
            prev_microbatch_id: The microbatch ID for the previous batch
        """
        # NOTE: If the rank processing the logits ever becomes the same as
        # the rank sending the responses, this code can be removed.
        finished_reqs = [
            r for r in
            previous_batch.sample_state.scheduled_requests.all_requests()
            if r.state == LlmRequestState.GENERATION_COMPLETE and (
                r.py_return_context_logits or r.py_return_generation_logits)
        ]
        if self.dist.is_first_pp_rank and len(finished_reqs):
            finished_reqs_py_results = [r.py_result for r in finished_reqs]
            finished_reqs_py_results = self.dist.recv_object(
                src=self.dist.prev_pp_rank,
                tag=prev_microbatch_id,
            )
            for req, py_result in zip(finished_reqs, finished_reqs_py_results):
                req.py_result = py_result

        elif self.dist.is_last_pp_rank and len(finished_reqs):
            self.wait_on_pp_send_handles(prev_microbatch_id)
            self.send_handles[prev_microbatch_id] = self.dist.isend_object(
                [r.py_result for r in finished_reqs],
                dest=self.dist.next_pp_rank,
                tag=prev_microbatch_id)

    def _await_any_response(self,
                            timeout: Optional[float] = None
                            ) -> List[LlmResponse]:
>>>>>>> upstream/main

        def any_responses_ready():
            return len(self.responses) > 0 or self.is_shutdown

        responses = []
        with self.response_cv:
            self.response_cv.wait_for(any_responses_ready, timeout=timeout)
            for req_id, response in self.responses.items():
                responses += response
            self.responses = {}

        return responses

    def _await_single_response(
            self,
            id: int,
<<<<<<< HEAD
            timeout: Union[float, None] = None) -> List[ExecutorResponse]:
=======
            timeout: Optional[float] = None) -> List[LlmResponse]:
>>>>>>> upstream/main
        with self.response_cv:

            def key_has_response():
                return id in self.responses.keys()

            self.response_cv.wait_for(key_has_response, timeout=timeout)
            response = self.responses[id]
            self.responses.pop(id)
            return response

    def _pause_requests(self, requests_to_pause):
        # todo: support work with self.inflight_req_ids.
        #       Currently, self.inflight_req_ids is not.
        max_input_len = self.max_input_len
        for req in requests_to_pause:
            req.pause(max_input_len)
            self._terminate_request(req)

    def _add_inflight_ids(self, scheduled_requests):
        """Add reqids of current requests to self.inflight_req_ids."""
<<<<<<< HEAD
        for req in chain(scheduled_requests.context_requests,
                         scheduled_requests.generation_requests):
=======
        for req in scheduled_requests.all_requests():
>>>>>>> upstream/main
            self.inflight_req_ids.insert(req.request_id)

    def _remove_inflight_ids(self, scheduled_requests):
        """Remove reqids of current requests from self.inflight_req_ids."""
<<<<<<< HEAD
        for req in chain(scheduled_requests.context_requests,
                         scheduled_requests.generation_requests):
            self.inflight_req_ids.erase(req.request_id)
=======
        for req in scheduled_requests.all_requests():
            self.inflight_req_ids.erase(req.request_id)

    def _handle_speculative_decoding(self, scheduled_batch, previous_tensors):
        with request_context(is_draft=self.draft_model_engine is not None,
                             scheduled_requests=scheduled_batch):
            # Do an early checking to see if we need to forward the draft model.
            # If needed, the overlap should happen between the target requests and the draft requests.
            # Otherwise, we can still do overlap between the previous target requests and the current target requests.
            has_draft_batch = (
                self.previous_batch is not None and self.use_spec_decode
                and self.drafter.should_forward_draft_model(scheduled_batch))

            if has_draft_batch or self.has_previous_draft_tokens:
                self._update_requests(self.previous_batch.sample_state)
                if self.has_previous_draft_tokens:
                    self._prepare_draft_requests()

            if has_draft_batch:
                target_inputs, draft_outputs, draft_batch = self.drafter.generate_draft_tokens_with_overlap(
                    scheduled_batch, self.resource_manager,
                    previous_tensors.device if previous_tensors else None)

                self.has_previous_draft_tokens = target_inputs is not None and target_inputs.next_draft_tokens is not None
            else:
                self.has_previous_draft_tokens = False
                target_inputs, draft_outputs, draft_batch = None, None, None

        return target_inputs, draft_outputs, draft_batch

    def _process_draft_results(self, scheduled_batch, draft_outputs,
                               draft_batch):
        """
        Append the draft tokens to the target requests, and clean up the draft resources.
        """
        with request_context(is_draft=self.draft_model_engine is not None,
                             scheduled_requests=scheduled_batch):
            req_id_to_old_request = {
                req.py_request_id: req
                for req in scheduled_batch.all_requests()
            }

            if self.drafter.use_static_draft_loop:
                self.drafter.process_static_draft_outputs(
                    draft_outputs, draft_batch, req_id_to_old_request)
            elif draft_outputs is not None:
                self.drafter.process_dynamic_draft_outputs(
                    draft_outputs, req_id_to_old_request)

            # Pad draft tokens to the max draft length. This is for CUDA graph compatibility.
            self.drafter.pad_draft_tokens_for_cuda_graph(scheduled_batch)
            # add_batch must be called again to restore to target requests with updated draft tokens.
            if self.guided_decoder is not None:
                self.guided_decoder.add_batch(scheduled_batch)
                if hasattr(self.drafter, "guided_decoder"):
                    self.guided_decoder.rollback_draft_tokens()


class DisaggPPTerminationHandler:
    """Handles termination synchronization across pipeline parallel ranks under disaggregated serving.

    We require synchronization when terminating requests in disaggregated PP when
    KV cache reuse is enabled. All PP ranks need to reach consensus before freeing
    resources to avoid a NCCL hang.
    """

    def __init__(self, num_micro_batches: int, dist):
        self.dist = dist
        # Request termination synchronization across PP ranks
        # {request_id: {'ready_to_terminate': set{ranks}, 'terminated': {ranks}}}
        self.pending_termination = {}
        self.termination_handles = [None] * num_micro_batches
        # Local map from request_id -> local LlmRequest awaiting consensus termination
        self.local_termination = {}

    def terminate(self, request: LlmRequest) -> bool:
        req_key = request.py_request_id
        self.local_termination[req_key] = request
        state = self.pending_termination.get(req_key, None)
        if state is None:
            state = {'ready_to_terminate': set(), 'terminated': set()}
            self.pending_termination[req_key] = state
        if self.dist.rank not in state['ready_to_terminate']:
            state['ready_to_terminate'].add(self.dist.rank)
        return False

    def sync(self, microbatch_id: int) -> List[LlmRequest]:
        """Ring-communicate pending termination state and apply local terminations upon consensus.

        Each rank sends its current pending_termination snapshot to the next PP rank
        and receives the previous rank's snapshot. After merging, apply any terminations
        that have reached consensus (i.e., all PP ranks are ready).
        """
        snapshot = {
            req_id: {
                'ready_to_terminate': state.get('ready_to_terminate', set()),
                'terminated': state.get('terminated', set()),
            }
            for req_id, state in self.pending_termination.items()
        }

        if self.termination_handles[microbatch_id] is not None:
            self.termination_handles[microbatch_id].wait()

        term_tag = TERMINATION_COMM_TAG_BASE + microbatch_id
        self.termination_handles[microbatch_id] = self.dist.isend_object(
            snapshot,
            dest=self.dist.next_pp_rank,
            tag=term_tag,
        )
        remote_state = self.dist.recv_object(
            src=self.dist.prev_pp_rank,
            tag=term_tag,
        )
        logger.debug(
            f"received remote state for microbatch {microbatch_id}, prev pp rank: {self.dist.prev_pp_rank} state {remote_state}"
        )

        if remote_state:
            for req_id, state in remote_state.items():
                local = self.pending_termination.get(req_id)
                if local is None:
                    self.pending_termination[req_id] = {
                        'ready_to_terminate': state.get('ready_to_terminate',
                                                        set()),
                        'terminated': state.get('terminated', set()),
                    }
                else:
                    for key in ('ready_to_terminate', 'terminated'):
                        for r in state.get(key, []):
                            if r not in local[key]:
                                local[key].add(r)

        requests_to_terminate = []
        to_delete = []
        for req_id, state in self.pending_termination.items():
            ready = state.get('ready_to_terminate', set())
            done = state.get('terminated', set())
            # If all PP ranks are ready to terminate the request, we can free the resources
            if len(ready) >= self.dist.pp_size and self.dist.rank not in done:
                local_req = self.local_termination.get(req_id)
                if local_req is not None:
                    requests_to_terminate.append(local_req)
                done.add(self.dist.rank)
            if len(done) >= self.dist.pp_size:
                to_delete.append(req_id)
                if req_id in self.local_termination:
                    self.local_termination.pop(req_id, None)
        for req_id in to_delete:
            self.pending_termination.pop(req_id, None)

        return requests_to_terminate

    def cleanup(self):
        for h in self.termination_handles:
            if h is not None:
                h.wait()
>>>>>>> upstream/main
