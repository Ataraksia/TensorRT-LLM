import asyncio
import concurrent.futures
import os
<<<<<<< HEAD
=======
import sys
import threading
import traceback
>>>>>>> upstream/main
from concurrent.futures import ProcessPoolExecutor
from queue import Empty, Queue
from typing import Any, Callable, List, NamedTuple, Optional

<<<<<<< HEAD
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.bindings.executor import Response
from tensorrt_llm.llmapi.utils import print_colored_debug
from tensorrt_llm.logger import logger
=======
from strenum import StrEnum

from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.llmapi.utils import enable_llm_debug, print_colored_debug
>>>>>>> upstream/main

from ..llmapi.mpi_session import (MpiCommSession, MpiPoolSession, MpiSession,
                                  RemoteMpiCommSessionClient)
from ..llmapi.utils import print_colored_debug
<<<<<<< HEAD

PERIODICAL_RESP_IN_AWAIT = os.getenv(
    "TLLM_EXECUTOR_PERIODICAL_RESP_IN_AWAIT") == "1"
=======
from ..logger import logger


class LlmLauncherEnvs(StrEnum):
    # Spawn a process for the LLM-API Proxy
    TLLM_SPAWN_PROXY_PROCESS = "TLLM_SPAWN_PROXY_PROCESS"
    TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR = "TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR"
    TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY = "TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY"

    # Whether to use periodical responses handler in await_responses
    TLLM_EXECUTOR_PERIODICAL_RESP_IN_AWAIT = "TLLM_EXECUTOR_PERIODICAL_RESP_IN_AWAIT"
>>>>>>> upstream/main


def get_spawn_proxy_process_ipc_addr_env() -> str | None:
    ''' Get the IPC address for the spawn proxy process dynamically. '''
<<<<<<< HEAD
    return os.getenv("TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR")
=======
    return os.getenv(LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR)
>>>>>>> upstream/main


def get_spawn_proxy_process_ipc_hmac_key_env() -> bytes | None:
    ''' Get the HMAC key for the spawn proxy process dynamically. '''
    if key := os.getenv("TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY"):
        return bytes.fromhex(key)


def get_spawn_proxy_process_env() -> bool:
    ''' Get the environment variable for the spawn proxy process dynamically. '''
<<<<<<< HEAD
    return os.getenv("TLLM_SPAWN_PROXY_PROCESS") == "1"


if PERIODICAL_RESP_IN_AWAIT:
    logger.info("Using periodical responses in await_responses")
=======
    return os.getenv(LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS) == "1"
>>>>>>> upstream/main


def create_mpi_comm_session(
        n_workers: int) -> RemoteMpiCommSessionClient | MpiPoolSession:
    assert mpi_rank(
    ) == 0, f"create_mpi_comm_session must be called by rank 0, but it was called by rank {mpi_rank()}"
    if get_spawn_proxy_process_env():
        assert get_spawn_proxy_process_ipc_addr_env(
<<<<<<< HEAD
        ), "TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR is not set."
        print_colored_debug(
            f"Using RemoteMpiPoolSessionClient to bind to external MPI processes at {get_spawn_proxy_process_ipc_addr_env()}\n",
            "yellow")
        hmac_key = get_spawn_proxy_process_ipc_hmac_key_env()
        return RemoteMpiCommSessionClient(
            addr=get_spawn_proxy_process_ipc_addr_env(), hmac_key=hmac_key)
=======
        ), f"{LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR} is not set."
        print_colored_debug(
            f"Using RemoteMpiPoolSessionClient to bind to external MPI processes at {get_spawn_proxy_process_ipc_addr_env()}\n",
            "yellow")
        get_spawn_proxy_process_ipc_hmac_key_env()
        return RemoteMpiCommSessionClient(
            addr=get_spawn_proxy_process_ipc_addr_env())
>>>>>>> upstream/main
    else:
        print_colored_debug(
            f"Using MpiCommSession to bind to external MPI processes\n",
            "yellow")
        return MpiCommSession(n_workers=n_workers)


def has_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class RequestError(RuntimeError):
    ''' The error raised when the request is failed. '''


class ProcessPoolExecutorSession(MpiSession):
    # This process pool is introduced for better recoverable exceptions handling.
    # It replaces MpiPoolExecutor for single-gpu case.

    def __init__(self, n_workers: int, **kwargs):
        self.n_workers = n_workers
        self.mpi_pool = ProcessPoolExecutor(max_workers=self.n_workers,
                                            **kwargs)

    def submit(self, task: Callable, *args,
               **kwargs) -> List[concurrent.futures.Future]:
        return [
            self.mpi_pool.submit(task, *args, **kwargs)
            for i in range(self.n_workers)
        ]

    def submit_sync(self, task: Callable, *args, **kwargs) -> List[Any]:
        futures = [
            self.mpi_pool.submit(task, *args, **kwargs)
            for i in range(self.n_workers)
        ]
        return [future.result() for future in futures]

    def shutdown(self):
        self.mpi_pool.shutdown(wait=True)


class ErrorResponse(NamedTuple):
    client_id: int
    error_msg: str
    request_id: int


class IntraProcessQueue:
    ''' A Queue-like container for IPC within the same process. '''

    def __init__(self):
        self.queue = Queue()

    def put(self, obj: Any):
        self.queue.put(obj)

    def get(self, timeout=None) -> Any:
        return self.queue.get(timeout=timeout)

    def close(self):
        pass

    def poll(self, timeout=None) -> bool:
        try:
            # Try to get an item from the queue without blocking
            item = self.queue.get(timeout=timeout)
            # If successful, put the item back to not alter the state
            self.queue.put(item)
            return True
        except Empty:
            # If the queue thread is empty, return False
            return False


class WorkerCommIpcAddrs(NamedTuple):
    ''' IPC addresses (str) and HMAC keys (bytes) for communication with the worker processes. '''
    request_queue_addr: tuple[str, Optional[bytes]]
<<<<<<< HEAD
    request_error_queue_addr: tuple[str, Optional[bytes]]
=======
    worker_init_status_queue_addr: tuple[str, Optional[bytes]]
>>>>>>> upstream/main
    result_queue_addr: tuple[str, Optional[bytes]]
    stats_queue_addr: tuple[str, Optional[bytes]]
    kv_cache_events_queue_addr: tuple[str, Optional[bytes]]


def is_llm_response(instance):
<<<<<<< HEAD
    from tensorrt_llm._torch.pyexecutor.llm_request import \
        LlmResponse as PyLlmResponse

    from .result import ResponseWrapper
    return isinstance(instance, (Response, PyLlmResponse, ResponseWrapper))
=======
    return hasattr(instance, "result")


def print_alive_threads():
    assert enable_llm_debug(
    ), "print_alive_threads must be called with enable_llm_debug() enabled"

    # Print all alive threads for debugging
    alive_threads = [t for t in threading.enumerate() if t.is_alive()]
    logger.info(
        f'All alive threads after shutdown: {[t.name for t in alive_threads]}\n',
        "red")
    for t in alive_threads:
        logger.info(f'Thread {t.name} (daemon={t.daemon}) is still alive')
        # Get the stack trace for this thread
        stack = sys._current_frames().get(t.ident)
        if stack is not None:
            logger.info(f'Stack trace for thread {t.name}:')
            traceback.print_stack(stack, file=sys.stdout)
            logger.info('')
>>>>>>> upstream/main
