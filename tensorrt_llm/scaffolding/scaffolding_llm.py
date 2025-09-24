import asyncio
import threading
import traceback
from collections import deque
from dataclasses import dataclass
from typing import Any, Generator, List, Mapping, Union

<<<<<<< HEAD
from .controller import Controller, ParallelProcess, ScaffoldingOutput
=======
from .controller import Controller, ParallelProcess
from .result import ScaffoldingResult
from .task import Task
>>>>>>> upstream/main
from .worker import Worker


@dataclass(frozen=True)
class ScaffoldingRequest:
    prompt: str
    kwargs: Mapping[str, Any]
    controller: Controller
    result: "ScaffoldingResult"


<<<<<<< HEAD
class ScaffoldingResult:

    def __init__(self):
        self._done = False
        self.aqueue = asyncio.Queue()
        self.output = None
        self.task_collections = None

    def set_output(self, output: ScaffoldingOutput):
        self.aqueue.put_nowait(output)

    def set_task_collections(self, task_collections: Mapping[str,
                                                             "TaskCollection"]):
        self.task_collections = task_collections

    async def aresult_step(self):
        # TODO: error handling or raise exception?
        self.output = await self.aqueue.get()
        if self.output is None:
            raise Exception("ScaffoldingLlm execution failed")
        self._done = True

    def result(self) -> "ScaffoldingResult":
        if not self._done:
            loop = asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(self.aresult(), loop).result()
        return self

    async def aresult(self) -> "ScaffoldingResult":
        while not self._done:
            await self.aresult_step()

        return self

    def __await__(self):
        return self.aresult().__await__()


=======
>>>>>>> upstream/main
class ScaffoldingLlm:

    def __init__(
            self,
            prototype_controller: Controller,
<<<<<<< HEAD
            workers: Mapping[
                str, Worker],  # map of role of Crontroller to a worker instance
=======
            workers: Mapping[str, Worker],  # map of role to worker instance
>>>>>>> upstream/main
    ):
        self.prototype_controller = prototype_controller
        self.workers = workers

        self.loop = self._get_loop()
        asyncio.set_event_loop(self.loop)
        self.task_queue = asyncio.Queue()
        self.main_loop_stop_event = asyncio.Event()
        self.shutdown_event = asyncio.Event()
<<<<<<< HEAD
=======
        self.streaming_event = asyncio.Event()
>>>>>>> upstream/main
        if self.own_loop:
            self._run_main_loop_thread()
        else:
            self._run_main_loop_coroutine()

        # For top scheduler
        self.running_req_count = 0
        self.max_parallel_requests = 64
        self.pending_queue = deque()

        self.output_task_collection = False

    def __enter__(self):
        return self

    def __exit__(self):
        self.shutdown()

    def _get_loop(self):
        try:
            self.own_loop = False
            return asyncio.get_running_loop()
        except RuntimeError:
            self.own_loop = True
            return asyncio.new_event_loop()
        return None

<<<<<<< HEAD
    async def _main_loop_async_func(self):

        async def handle_controller_generator(gen: Generator):
            for obj in gen:
                if isinstance(obj, ParallelProcess):
                    await handle_parallel_process(obj)
                else:
                    task_list = obj
                    async_tasks = []
                    for task in task_list:
                        task_worker_tag = task.worker_tag
                        assert task_worker_tag in self.workers.keys()
                        worker = self.workers[task_worker_tag]
                        async_tasks.append(
                            asyncio.create_task(worker.run_task(task)))
                    await asyncio.gather(*async_tasks)

        async def handle_parallel_process(request: ParallelProcess):
            async_tasks = []
            for sub_gen in request.sub_gens:
                async_task = asyncio.create_task(
                    handle_controller_generator(sub_gen))
                async_tasks.append(async_task)
            await asyncio.gather(*async_tasks)

        async def handle_single_request(request: ScaffoldingRequest):
            # warp to a generator without return value
            def controller_generator_wrapper(request: ScaffoldingRequest):
                scaffolding_output = yield from request.controller.generate(
                    request.prompt, **request.kwargs)
                if self.output_task_collection:
                    request.result.set_task_collections(
                        request.controller.task_collections)
                request.result.set_output(scaffolding_output)

            try:
                gen = controller_generator_wrapper(request)
                await handle_controller_generator(gen)
            except Exception as e:
                # Catch the exception and set output to avoid the user thread to be hang
                print('scaffoldingLlm handle request exception:', str(e))
                traceback.print_exc()
                request.result.set_output(None)
                raise e
            finally:
                self.running_req_count -= 1
                maybe_schedule()

        def schedule_request(request: ScaffoldingRequest):
            asyncio.create_task(handle_single_request(request))
            self.running_req_count += 1

        def maybe_schedule(request: ScaffoldingRequest = None):
            if self.shutdown_event.is_set():
                return

            if request is not None:
                self.pending_queue.append(request)

            while self.running_req_count < self.max_parallel_requests and len(
                    self.pending_queue) > 0:
                first_request = self.pending_queue.popleft()
                schedule_request(first_request)

        async def handle_event():
            while True:
                item = await self.task_queue.get()
                if item is None:
                    return
                elif isinstance(item, ScaffoldingRequest):
                    maybe_schedule(item)
                else:
                    raise ValueError(
                        f'type of task_queue item ({type(item)}) is not supported'
                    )

        handle_event_task = asyncio.create_task(handle_event())
=======
    async def _handle_controller_generator(self,
                                           gen: Generator,
                                           request: ScaffoldingRequest = None):
        """Handle a controller generator, processing tasks and parallel processes."""
        for obj in gen:
            if isinstance(obj, ParallelProcess):
                await self._handle_parallel_process(obj, request)
            else:
                await self._handle_task_list(obj, request)

    async def _handle_task_list(self,
                                tasks: List[Task],
                                request: ScaffoldingRequest = None):
        """Execute a list of tasks concurrently."""
        async_tasks = [
            asyncio.create_task(self.workers[task.worker_tag].run_task(task))
            for task in tasks
        ]
        await asyncio.gather(*async_tasks)
        for task in tasks:
            if getattr(task, 'streaming', False):
                await request.result.set_output_async(task.result)
                self.streaming_event.clear()
                await self.streaming_event.wait()

    async def _handle_parallel_process(self,
                                       tasks: ParallelProcess,
                                       request: ScaffoldingRequest = None):
        """Handle parallel execution of multiple generators."""
        async_tasks = [
            asyncio.create_task(
                self._handle_controller_generator(sub_gen, request))
            for sub_gen in tasks.sub_gens
        ]
        await asyncio.gather(*async_tasks)

    async def _handle_single_request(self, request: ScaffoldingRequest):
        """Process a single scaffolding request."""
        try:
            gen = self._create_controller_generator(request)
            await self._handle_controller_generator(gen, request)
        except Exception as e:
            print(f'ScaffoldingLLM request exception: {e}')
            traceback.print_exc()
            request.result.set_output(None)
            raise
        finally:
            self.running_req_count -= 1
            self._maybe_schedule()

    def _create_controller_generator(self, request: ScaffoldingRequest):
        """Create a generator wrapper for the controller."""
        scaffolding_output = yield from request.controller.generate(
            request.prompt, **request.kwargs)

        if self.output_task_collection:
            request.result.set_task_collections(
                request.controller.task_collections)
        request.result.set_output(scaffolding_output)

    def _schedule_request(self, request: ScaffoldingRequest):
        """Schedule a single request for execution."""
        asyncio.create_task(self._handle_single_request(request))
        self.running_req_count += 1

    def _maybe_schedule(self, request: ScaffoldingRequest = None):
        """Schedule pending requests if capacity allows."""
        if self.shutdown_event.is_set():
            return

        if request is not None:
            self.pending_queue.append(request)

        while (self.running_req_count < self.max_parallel_requests
               and self.pending_queue):
            next_request = self.pending_queue.popleft()
            self._schedule_request(next_request)

    async def _handle_event_loop(self):
        """Main event handling loop."""
        while True:
            item = await self.task_queue.get()

            if item is None:
                return
            elif isinstance(item, ScaffoldingRequest):
                self._maybe_schedule(item)
            else:
                raise ValueError(
                    f'Unsupported task_queue item type: {type(item)}')

    async def _main_loop_async_func(self):
        """Main async loop function with reduced complexity."""
        handle_event_task = asyncio.create_task(self._handle_event_loop())
>>>>>>> upstream/main
        await handle_event_task
        self.main_loop_stop_event.set()

    def _run_main_loop_coroutine(self):
        asyncio.run_coroutine_threadsafe(self._main_loop_async_func(),
                                         self.loop)

    def _run_main_loop_thread(self):

        def main_loop_thread():
            self.loop.run_until_complete(self._main_loop_async_func())

        self.main_loop_thread = threading.Thread(target=main_loop_thread)
        self.main_loop_thread.start()

    def generate_async(self, prompt: str) -> ScaffoldingResult:
<<<<<<< HEAD
        result = ScaffoldingResult()
=======
        result = ScaffoldingResult(self.streaming_event)
>>>>>>> upstream/main

        async def put_request():
            request = ScaffoldingRequest(
                prompt=prompt,
                kwargs={},
                result=result,
                controller=self.prototype_controller.clone())

            await self.task_queue.put(request)

        asyncio.run_coroutine_threadsafe(put_request(), self.loop)

        return result

    def generate(
        self, prompts: Union[str, List[str]]
    ) -> Union[ScaffoldingResult, List[ScaffoldingResult]]:

        unbatched = not isinstance(prompts, list)
        batched_prompts = [prompts] if unbatched else prompts

        scaffolding_results = []
        for prompt in batched_prompts:
            scaffolding_results.append(self.generate_async(prompt))

        for scaffolding_result in scaffolding_results:
            scaffolding_result.result()

        return scaffolding_results[0] if unbatched else scaffolding_results

    def enable_output_task_collection(self):
        self.output_task_collection = True

    def shutdown(self, shutdown_workers=False):

        def shutdown_workers():
            for worker in self.workers.values():
                worker.shutdown()

        # Let the merge thread break
        async def stop_task_on_loop():
            await self.task_queue.put(None)
            await self.main_loop_stop_event.wait()

        asyncio.run_coroutine_threadsafe(stop_task_on_loop(), self.loop)

        if self.own_loop:
            self.main_loop_thread.join()
        else:
            # if we don't own the loop, we can't ensure the "stop_task_on_loop"
            # is finished, so we need to set the shutdown event to make sure the main loop
            # will not submit new tasks to workers.
            self.shutdown_event.set()

        if shutdown_workers:
            shutdown_workers()
