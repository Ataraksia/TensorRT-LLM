from __future__ import annotations

import asyncio
<<<<<<< HEAD
import json
=======
import sys
from functools import partial
>>>>>>> upstream/main
from pathlib import Path

import click
from click_option_group import (MutuallyExclusiveOptionGroup, OptionGroup,
                                optgroup)
from huggingface_hub import snapshot_download

<<<<<<< HEAD
from tensorrt_llm.bench.benchmark.utils.asynchronous import async_benchmark
from tensorrt_llm.bench.benchmark.utils.processes import IterationWriter
from tensorrt_llm.bench.build.build import get_model_config

# isort: off
from tensorrt_llm.bench.benchmark.utils.general import (
    get_settings_from_engine, get_settings)
# isort: on
from tensorrt_llm._torch.llm import LLM as PyTorchLLM
from tensorrt_llm.bench.benchmark.utils.general import generate_warmup_dataset
=======
from tensorrt_llm.bench.benchmark import (GeneralExecSettings,
                                          generate_json_report,
                                          get_general_cli_options, get_llm)
from tensorrt_llm.bench.benchmark.utils.asynchronous import async_benchmark
from tensorrt_llm.tools.importlib_utils import import_custom_module_from_dir

# isort: off
from tensorrt_llm.bench.benchmark.utils.general import (
    get_settings_from_engine, get_settings, ALL_SUPPORTED_BACKENDS)
# isort: on
from tensorrt_llm.bench.benchmark.utils.general import (
    generate_warmup_dataset, update_sampler_args_with_extra_options)
>>>>>>> upstream/main
from tensorrt_llm.bench.dataclasses.configuration import RuntimeConfig
from tensorrt_llm.bench.dataclasses.general import BenchmarkEnvironment
from tensorrt_llm.bench.dataclasses.reporting import ReportUtility
from tensorrt_llm.bench.utils.data import (create_dataset_from_stream,
                                           initialize_tokenizer,
                                           update_metadata_for_multimodal)
<<<<<<< HEAD
from tensorrt_llm.llmapi import LLM, CapacitySchedulerPolicy
=======
from tensorrt_llm.llmapi import CapacitySchedulerPolicy
>>>>>>> upstream/main
from tensorrt_llm.logger import logger
from tensorrt_llm.sampling_params import SamplingParams


@click.command(name="throughput")
@optgroup.group("Engine run configuration.",
                help="Runtime settings for executing a TensorRT-LLM engine.")
@optgroup.option(
    "--engine_dir",
    type=click.Path(exists=True,
                    readable=True,
                    path_type=Path,
                    resolve_path=True),
    default=None,
    help="Path to a serialized TRT-LLM engine.",
)
@optgroup.option("--backend",
<<<<<<< HEAD
                 type=click.Choice(["pytorch", "autodeploy"]),
                 default=None,
                 help="Set to 'pytorch' for pytorch path. Default is cpp path.")
=======
                 type=click.Choice(ALL_SUPPORTED_BACKENDS),
                 default="pytorch",
                 help="The backend to use when running benchmarking.")
@optgroup.option(
    "--custom_module_dirs",
    type=click.Path(exists=True,
                    readable=True,
                    path_type=Path,
                    resolve_path=True),
    default=None,
    multiple=True,
    help="Paths to custom module directories to import.",
)
>>>>>>> upstream/main
@optgroup.option(
    "--extra_llm_api_options",
    type=str,
    default=None,
    help=
    "Path to a YAML file that overwrites the parameters specified by trtllm-bench."
)
<<<<<<< HEAD
=======
@optgroup.option("--sampler_options",
                 type=click.Path(exists=True,
                                 readable=True,
                                 path_type=Path,
                                 resolve_path=True),
                 default=None,
                 help="Path to a YAML file that sets sampler options.")
>>>>>>> upstream/main
@optgroup.option(
    "--max_batch_size",
    type=int,
    help="Maximum runtime batch size to run the engine with.",
)
@optgroup.option(
    "--max_num_tokens",
    type=int,
    help="Maximum runtime tokens that an engine can accept.",
)
@optgroup.option(
    "--max_seq_len",
    type=int,
    default=None,
    help="Maximum sequence length.",
)
@optgroup.option(
    "--beam_width",
    type=int,
    default=1,
    help="Number of search beams.",
)
@optgroup.option(
    "--kv_cache_free_gpu_mem_fraction",
    type=float,
    default=.90,
    help="The percentage of memory to use for KV Cache after model load.",
)
@optgroup.group(
    "Engine Input Configuration",
    help="Input configuration for driving the engine.",
)
@optgroup.option(
    "--dataset",
    type=click.Path(exists=True,
                    readable=True,
                    path_type=Path,
                    resolve_path=True),
    default=None,
    required=False,
    help="Pass in a dataset file for parsing instead of stdin.",
)
<<<<<<< HEAD
=======
# For text models, tokenizer initialization is not needed when loading the model since the dataset is already tokenized.
# For this reason, we skip tokenizer initialization by default.
# However, for VLM models, tokenizer initialization is needed inside the model since the dataset contains texts and
# raw media data. We cannot skip tokenizer initialization in this case.
@optgroup.option(
    "--no_skip_tokenizer_init",
    is_flag=True,
    default=False,
    help="Do not skip tokenizer initialization when loading the model.",
)
>>>>>>> upstream/main
@optgroup.option(
    "--eos_id",
    type=int,
    default=-1,
    required=False,
    help=
    "Set the end-of-sequence token for the benchmark. Set to -1 to disable EOS.",
)
@optgroup.option(
    "--modality",
    type=click.Choice(["image", "video"]),
    default=None,
    help="Modality of the multimodal requests.",
)
@optgroup.option(
<<<<<<< HEAD
=======
    "--image_data_format",
    type=click.Choice(["pt", "pil"]),
    default="pt",
    help="Format of the image data for multimodal models.",
)
@optgroup.option(
    "--data_device",
    type=click.Choice(["cuda", "cpu"]),
    default="cuda",
    help="Device to load the multimodal data on.",
)
@optgroup.option(
>>>>>>> upstream/main
    "--max_input_len",
    type=int,
    default=4096,
    help=
    "Maximum input sequence length to use for multimodal models. This is used only when --modality "
    "is specified since the actual number of vision tokens is unknown before the model is run.",
)
@optgroup.option(
    "--num_requests",
    type=int,
    default=0,
    help=
    "Number of requests to cap benchmark run at. If not specified or set to 0, it will be the "
    "length of dataset.",
)
@optgroup.option(
    "--warmup",
    type=int,
    default=2,
    help="Number of requests warm up benchmark.",
)
@optgroup.option(
    "--target_input_len",
    default=None,
    type=click.IntRange(min=1),
    help="Target (average) input length for tuning heuristics.",
)
@optgroup.option(
    "--target_output_len",
    default=None,
    type=click.IntRange(min=1),
    help="Target (average) sequence length for tuning heuristics.",
)
@optgroup.group(
    "World Configuration",
    help="Options for configuring the backend multi-GPU world.",
)
@optgroup.option(
    "--tp",
    type=int,
    default=1,
    help="tensor parallelism size",
)
@optgroup.option(
    "--pp",
    type=int,
    default=1,
    help="pipeline parallelism size",
)
@optgroup.option(
    "--ep",
    type=int,
    default=None,
    help="expert parallelism size",
)
@optgroup.option(
    "--cluster_size",
    type=int,
    default=None,
    help="expert cluster parallelism size",
)
@optgroup.group("Request Load Control Options",
                cls=MutuallyExclusiveOptionGroup,
                help="Limits how requests are loaded.")
@optgroup.option(
    "--concurrency",
    type=int,
    default=-1,
    help=
    "Desired concurrency rate (number of requests processing at the same time), <=0 for no concurrency limit.",
)
@click.option(
    "--streaming",
    is_flag=True,
    default=False,
    help="Enable streaming mode for requests.",
)
@optgroup.group("Reporting Options",
                help="Options for reporting benchmark results.",
                cls=OptionGroup)
@optgroup.option(
    "--report_json",
    type=click.Path(dir_okay=False,
                    writable=True,
                    readable=False,
                    path_type=Path,
                    resolve_path=True),
    required=False,
    help="Path where report is written to.",
)
@optgroup.option(
    "--iteration_log",
    type=click.Path(dir_okay=False,
                    writable=True,
                    readable=False,
                    path_type=Path,
                    resolve_path=True),
    required=False,
    help="Path where iteration logging is written to.",
)
@optgroup.option(
    "--output_json",
    type=click.Path(dir_okay=False,
                    writable=True,
                    readable=False,
                    path_type=Path,
                    resolve_path=True),
    required=False,
    help="Path where output should be written to.",
)
<<<<<<< HEAD
=======
@optgroup.option(
    "--request_json",
    type=click.Path(dir_okay=False,
                    writable=True,
                    readable=False,
                    path_type=Path,
                    resolve_path=True),
    required=False,
    help="Path where per request information is written to.",
)
@optgroup.option(
    "--enable_chunked_context/--disable_chunked_context",
    default=True,
    help=
    "Enable/disable chunking in prefill stage for enhanced throughput benchmark. "
)
@optgroup.option(
    "--scheduler_policy",
    type=click.Choice(["guaranteed_no_evict", "max_utilization"]),
    default="guaranteed_no_evict",
    help=
    "KV cache scheduler policy: guaranteed_no_evict prevents request eviction, max_utilization optimizes for throughput.",
)
>>>>>>> upstream/main
@click.pass_obj
def throughput_command(
    bench_env: BenchmarkEnvironment,
    **params,
) -> None:
    """Run a throughput test on a TRT-LLM engine."""
<<<<<<< HEAD

    logger.info("Preparing to run throughput benchmark...")
    # Parameters from CLI
    # Model, experiment, and engine params
    dataset_path: Path = params.pop("dataset")
    eos_id: int = params.pop("eos_id")
    warmup: int = params.get("warmup")
    num_requests: int = params.pop("num_requests")
    max_seq_len: int = params.pop("max_seq_len")
    model: str = bench_env.model
    checkpoint_path: Path = bench_env.checkpoint_path or bench_env.model
    engine_dir: Path = params.pop("engine_dir")
    concurrency: int = params.pop("concurrency")
    backend: str = params.get("backend")
    modality: str = params.pop("modality")
    max_input_len: int = params.pop("max_input_len")
    model_type = get_model_config(model, checkpoint_path).model_type

    # Reporting options
    report_json: Path = params.pop("report_json")
    output_json: Path = params.pop("output_json")
    iteration_log: Path = params.pop("iteration_log")
    iteration_writer = IterationWriter(iteration_log)
=======
    logger.info("Preparing to run throughput benchmark...")
    # Parameters from CLI
    image_data_format: str = params.get("image_data_format", "pt")
    data_device: str = params.get("data_device", "cpu")
    no_skip_tokenizer_init: bool = params.get("no_skip_tokenizer_init", False)

    # Get general CLI options using the centralized function
    options: GeneralExecSettings = get_general_cli_options(params, bench_env)
    tokenizer = initialize_tokenizer(options.checkpoint_path)

    # Extract throughput-specific options not handled by GeneralExecSettings
    max_batch_size = params.get("max_batch_size")
    max_num_tokens = params.get("max_num_tokens")
    enable_chunked_context: bool = params.get("enable_chunked_context")
    scheduler_policy: str = params.get("scheduler_policy")

    custom_module_dirs: list[Path] = params.pop("custom_module_dirs", [])
    for custom_module_dir in custom_module_dirs:
        try:
            import_custom_module_from_dir(custom_module_dir)
        except Exception as e:
            logger.error(
                f"Failed to import custom module from {custom_module_dir}: {e}")
            raise e
>>>>>>> upstream/main

    # Runtime kwargs and option tracking.
    kwargs = {}

<<<<<<< HEAD
    # Initialize the HF tokenizer for the specified model.
    tokenizer = initialize_tokenizer(checkpoint_path)

    # Dataset Loading and Preparation
    with open(dataset_path, "r") as dataset:
        metadata, requests = create_dataset_from_stream(
            tokenizer,
            dataset,
            num_requests=num_requests,
            model_dir=checkpoint_path,
            model_type=model_type,
            modality=modality,
            max_input_seq_len_for_multimodal=max_input_len)
        metadata.dataset_path = dataset_path
=======
    # Dataset Loading and Preparation
    with open(options.dataset_path, "r") as dataset:
        metadata, requests = create_dataset_from_stream(
            tokenizer,
            dataset,
            num_requests=options.num_requests,
            model_dir=options.checkpoint_path,
            model_type=options.model_type,
            modality=options.modality,
            image_data_format=image_data_format,
            data_device=data_device,
            max_input_seq_len_for_multimodal=options.max_input_len)
        metadata.dataset_path = options.dataset_path
>>>>>>> upstream/main
        params["target_input_len"] = params.get(
            "target_input_len") or metadata.avg_isl
        params["target_output_len"] = params.get(
            "target_output_len") or metadata.avg_osl

<<<<<<< HEAD
    if modality is None:
=======
    if options.modality is None:
>>>>>>> upstream/main
        # Log dataset info
        # NOTE: This table is only accurate for non-multimodal models.
        #       The accurate table for multimodal models will be logged after the benchmark is done.
        logger.info(metadata.get_summary_for_print())

    # Engine configuration parsing
<<<<<<< HEAD
    if backend and backend.lower() in ["pytorch", "autodeploy"]:
        # If we're dealing with a model name, perform a snapshot download to
        # make sure we have a local copy of the model.
        if bench_env.checkpoint_path is None:
            snapshot_download(model)

        exec_settings = get_settings(params, metadata, bench_env.model,
                                     bench_env.checkpoint_path)
        kwargs_max_sql = max_seq_len or metadata.max_sequence_length
        logger.info(f"Setting PyTorch max sequence length to {kwargs_max_sql}")
        kwargs["max_seq_len"] = kwargs_max_sql
    else:
        assert max_seq_len is None, (
            "max_seq_len is not a runtime parameter for C++ backend")
        exec_settings, build_cfg = get_settings_from_engine(engine_dir)
=======
    if options.backend and options.backend.lower(
    ) in ALL_SUPPORTED_BACKENDS and options.backend.lower() != "tensorrt":
        # If we're dealing with a model name, perform a snapshot download to
        # make sure we have a local copy of the model.
        if bench_env.checkpoint_path is None:
            snapshot_download(options.model)

        exec_settings = get_settings(params, metadata, bench_env.model,
                                     bench_env.checkpoint_path)
        kwargs_max_sql = options.max_seq_len or metadata.max_sequence_length
        logger.info(f"Setting PyTorch max sequence length to {kwargs_max_sql}")
        kwargs["max_seq_len"] = kwargs_max_sql
    elif options.backend.lower() == "tensorrt":
        assert options.max_seq_len is None, (
            "max_seq_len is not a runtime parameter for C++ backend")
        exec_settings, build_cfg = get_settings_from_engine(options.engine_dir)
>>>>>>> upstream/main
        engine_max_seq_len = build_cfg["max_seq_len"]

        # TODO: Verify that the engine can handle the max/min ISL/OSL.
        if metadata.max_sequence_length > engine_max_seq_len:
            raise RuntimeError(
                f"Engine supports a max sequence of {engine_max_seq_len}. "
                "Provided dataset contains a maximum sequence of "
                f"{metadata.max_sequence_length}. Please rebuild a new engine "
                "to support this dataset.")
<<<<<<< HEAD

    exec_settings["model"] = model
=======
    else:
        raise RuntimeError(
            f"Invalid backend: {options.backend}, please use one of the following: "
            "pytorch, tensorrt, _autodeploy.")

    exec_settings["model"] = options.model
>>>>>>> upstream/main
    engine_bs = exec_settings["settings_config"]["max_batch_size"]
    engine_tokens = exec_settings["settings_config"]["max_num_tokens"]

    # Runtime Options
<<<<<<< HEAD
    runtime_max_bs = params.pop("max_batch_size")
    runtime_max_tokens = params.pop("max_num_tokens")
    runtime_max_bs = runtime_max_bs or engine_bs
    runtime_max_tokens = runtime_max_tokens or engine_tokens
    kv_cache_percent = params.pop("kv_cache_free_gpu_mem_fraction")
    beam_width = params.pop("beam_width")
    streaming: bool = params.pop("streaming")

    # Update configuration with runtime options
    exec_settings["settings_config"]["kv_cache_percent"] = kv_cache_percent
    exec_settings["settings_config"]["max_batch_size"] = runtime_max_bs
    exec_settings["settings_config"]["max_num_tokens"] = runtime_max_tokens
    exec_settings["settings_config"]["beam_width"] = beam_width
    exec_settings["settings_config"][
        "scheduler_policy"] = CapacitySchedulerPolicy.GUARANTEED_NO_EVICT
=======
    runtime_max_bs = max_batch_size or engine_bs
    runtime_max_tokens = max_num_tokens or engine_tokens

    # Update configuration with runtime options
    exec_settings["settings_config"][
        "kv_cache_percent"] = options.kv_cache_percent
    exec_settings["settings_config"]["max_batch_size"] = runtime_max_bs
    exec_settings["settings_config"]["max_num_tokens"] = runtime_max_tokens
    exec_settings["settings_config"]["beam_width"] = options.beam_width
    exec_settings["settings_config"][
        "scheduler_policy"] = CapacitySchedulerPolicy.GUARANTEED_NO_EVICT if scheduler_policy == "guaranteed_no_evict" else CapacitySchedulerPolicy.MAX_UTILIZATION
    exec_settings["settings_config"]["chunking"] = enable_chunked_context
>>>>>>> upstream/main

    # Dynamic runtime features.
    exec_settings["settings_config"]["dynamic_max_batch_size"] = True

    # LlmArgs
    exec_settings["extra_llm_api_options"] = params.pop("extra_llm_api_options")
<<<<<<< HEAD
    exec_settings["iteration_log"] = iteration_log
=======
    exec_settings["iteration_log"] = options.iteration_log
>>>>>>> upstream/main

    # Construct the runtime configuration dataclass.
    runtime_config = RuntimeConfig(**exec_settings)
    llm = None
<<<<<<< HEAD
    try:
        logger.info("Setting up throughput benchmark.")
        kwargs = kwargs | runtime_config.get_llm_args()
        kwargs['backend'] = backend

        if "pytorch_backend_config" in kwargs and iteration_log is not None:
            kwargs["pytorch_backend_config"].enable_iter_perf_stats = True

        if runtime_config.backend == 'pytorch':
            llm = PyTorchLLM(**kwargs)
        else:
            llm = LLM(**kwargs)

        sampling_params = SamplingParams(end_id=eos_id,
                                         pad_id=eos_id,
                                         n=beam_width,
                                         use_beam_search=beam_width > 1)

        # Perform warmup if requested.
        if warmup > 0:
            logger.info("Setting up for warmup...")
            warmup_dataset = generate_warmup_dataset(requests, warmup)
=======

    try:
        logger.info("Setting up throughput benchmark.")
        kwargs = kwargs | runtime_config.get_llm_args()
        kwargs['skip_tokenizer_init'] = not no_skip_tokenizer_init
        kwargs['backend'] = options.backend

        llm = get_llm(runtime_config, kwargs)

        sampler_args = {
            "end_id": options.eos_id,
            "pad_id": options.eos_id,
            "n": options.beam_width,
            "use_beam_search": options.beam_width > 1
        }
        sampler_args = update_sampler_args_with_extra_options(
            sampler_args, params.pop("sampler_options"))
        sampling_params = SamplingParams(**sampler_args)

        post_proc_params = None  # No detokenization

        # Perform warmup if requested.
        if options.warmup > 0:
            logger.info("Setting up for warmup...")
            warmup_dataset = generate_warmup_dataset(requests, options.warmup)
>>>>>>> upstream/main
            logger.info("Running warmup.")
            asyncio.run(
                async_benchmark(llm,
                                sampling_params,
<<<<<<< HEAD
                                warmup_dataset,
                                False,
                                concurrency,
                                modality=modality))
=======
                                post_proc_params,
                                warmup_dataset,
                                False,
                                options.concurrency,
                                modality=options.modality))
>>>>>>> upstream/main
            # WAR: IterationResult is a singleton tied to the executor.
            # Since the benchmark calls asyncio.run() multiple times (e.g., during warmup),
            # we must reset it to ensure it attaches to the correct event loop.
            llm._executor._iter_stats_result = None
            logger.info("Warmup done.")

<<<<<<< HEAD
=======
        iteration_writer = options.iteration_writer
>>>>>>> upstream/main
        with iteration_writer.capture():
            statistics = asyncio.run(
                async_benchmark(llm,
                                sampling_params,
<<<<<<< HEAD
                                requests,
                                streaming,
                                concurrency,
                                iteration_writer.full_address,
                                modality=modality))

        logger.info(f"Benchmark done. Reporting results...")
        if modality is not None:
=======
                                post_proc_params,
                                requests,
                                options.streaming,
                                options.concurrency,
                                iteration_writer.full_address,
                                modality=options.modality))

        logger.info("Benchmark done. Reporting results...")
        if options.modality is not None:
>>>>>>> upstream/main
            # For multimodal models, we need to update the metadata with the correct input lengths
            metadata = update_metadata_for_multimodal(metadata, statistics)

        report_utility = ReportUtility(statistics, metadata, runtime_config,
<<<<<<< HEAD
                                       logger, kwargs, streaming)
        if report_json:
            logger.info(f"Writing report to '{report_json}'.")
            with open(report_json, "w") as f:
                f.write(
                    json.dumps(report_utility.get_statistics_dict(), indent=4))
        if output_json:
            logger.info(f"Writing output to {output_json}.")
            with open(output_json, "w") as f:
                output_token_info = report_utility.get_output_tokens(tokenizer)
                f.write(json.dumps(output_token_info, indent=4))
        report_utility.report_statistics()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt, exiting benchmark...")
=======
                                       logger, kwargs, options.streaming)
        # Generate reports for statistics, output tokens, and request info.
        generate_json_report(options.report_json,
                             report_utility.get_statistics_dict)
        generate_json_report(
            options.output_json,
            partial(report_utility.get_output_tokens, tokenizer))
        generate_json_report(
            options.request_json,
            partial(report_utility.get_request_info, tokenizer))
        report_utility.report_statistics()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt, exiting benchmark...")
    except Exception:
        import traceback
        logger.error(f"Error during benchmarking:\n{traceback.format_exc()}")
        sys.exit(1)
>>>>>>> upstream/main
    finally:
        if llm is not None:
            llm.shutdown()
