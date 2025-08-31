#!/usr/bin/env python3
# SPDX-License-Identifier: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Higgs Audio TensorRT-LLM Benchmark Runner

This script provides a comprehensive command-line interface for running
performance benchmarks on the Higgs Audio TensorRT-LLM implementation.

Usage:
    python benchmark_runner.py --unified-engine /path/to/unified/engine \
                              --separate-text-engine /path/to/text/engine \
                              --separate-audio-engine /path/to/audio/engine \
                              --model-config /path/to/config.json \
                              --benchmark-types latency memory throughput \
                              --output-dir ./benchmark_results

Key Features:
- Comprehensive performance validation against quantified claims
- Statistical rigor with 50+ runs per benchmark for significance
- TTS-specific benchmark categories (generation modes, delay patterns, streaming)
- Unified vs separate architecture comparison
- CUDA graph performance analysis
- Automated report generation with visualizations
- Performance regression detection
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from benchmark_suite import (
    BenchmarkConfiguration, BenchmarkType, ArchitectureType,
    run_higgs_audio_benchmarks, create_latency_benchmark_configs,
    create_memory_benchmark_configs, create_throughput_benchmark_configs,
    create_tts_specific_benchmark_configs, create_cuda_graph_benchmark_configs
)
from tts_benchmarks import TTSBenchmarkOrchestrator
from analysis_reporting import BenchmarkAnalysisOrchestrator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark_runner.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Main benchmark runner class."""
    
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize orchestrators
        self.tts_orchestrator = None
        self.analysis_orchestrator = BenchmarkAnalysisOrchestrator(str(self.output_dir))
        
        # Store execution metadata
        self.execution_metadata = {
            'start_time': time.time(),
            'command_line_args': vars(args),
            'hostname': self._get_hostname(),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'working_directory': str(Path.cwd())
        }
    
    def _get_hostname(self) -> str:
        """Get system hostname."""
        try:
            import socket
            return socket.gethostname()
        except Exception:
            return "unknown"
    
    def validate_paths(self) -> bool:
        """Validate that all required paths exist."""
        required_paths = [
            ('unified_engine', self.args.unified_engine),
            ('separate_text_engine', self.args.separate_text_engine),
            ('separate_audio_engine', self.args.separate_audio_engine),
            ('model_config', self.args.model_config)
        ]
        
        missing_paths = []
        for name, path in required_paths:
            if path and not Path(path).exists():
                missing_paths.append(f"{name}: {path}")
        
        if missing_paths:
            logger.error("Missing required paths:")
            for path_info in missing_paths:
                logger.error(f"  - {path_info}")
            return False
        
        logger.info("All required paths validated successfully")
        return True
    
    def create_benchmark_configs(self) -> List[BenchmarkConfiguration]:
        """Create benchmark configurations based on command line arguments."""
        
        configs = []
        benchmark_types = self._parse_benchmark_types()
        
        logger.info(f"Creating configurations for benchmark types: {[t.value for t in benchmark_types]}")
        
        for benchmark_type in benchmark_types:
            if benchmark_type == BenchmarkType.LATENCY:
                configs.extend(create_latency_benchmark_configs())
            elif benchmark_type == BenchmarkType.MEMORY:
                configs.extend(create_memory_benchmark_configs())
            elif benchmark_type == BenchmarkType.THROUGHPUT:
                configs.extend(create_throughput_benchmark_configs())
            elif benchmark_type in [BenchmarkType.GENERATION_MODES, BenchmarkType.DELAY_PATTERNS, 
                                   BenchmarkType.STREAMING, BenchmarkType.DUALFFN]:
                # TTS-specific benchmarks
                tts_configs = create_tts_specific_benchmark_configs()
                # Filter for the specific type requested
                configs.extend([c for c in tts_configs if c.benchmark_type == benchmark_type])
            elif benchmark_type == BenchmarkType.CUDA_GRAPHS:
                configs.extend(create_cuda_graph_benchmark_configs())
        
        # Apply customizations from command line arguments
        configs = self._customize_configs(configs)
        
        logger.info(f"Created {len(configs)} benchmark configurations")
        return configs
    
    def _parse_benchmark_types(self) -> List[BenchmarkType]:
        """Parse benchmark types from command line arguments."""
        if not self.args.benchmark_types:
            # Default to all benchmark types
            return list(BenchmarkType)
        
        benchmark_types = []
        for type_str in self.args.benchmark_types:
            try:
                benchmark_type = BenchmarkType(type_str.lower())
                benchmark_types.append(benchmark_type)
            except ValueError:
                logger.warning(f"Unknown benchmark type: {type_str}")
                continue
        
        return benchmark_types
    
    def _customize_configs(self, configs: List[BenchmarkConfiguration]) -> List[BenchmarkConfiguration]:
        """Apply customizations from command line arguments."""
        
        for config in configs:
            # Apply custom run counts
            if self.args.num_runs:
                config.num_runs = self.args.num_runs
            
            if self.args.warmup_runs:
                config.warmup_runs = self.args.warmup_runs
            
            # Apply custom batch sizes
            if self.args.batch_sizes:
                config.batch_sizes = self.args.batch_sizes
            
            # Apply custom sequence lengths
            if self.args.sequence_lengths:
                config.sequence_lengths = self.args.sequence_lengths
            
            # Apply CUDA graph settings
            if self.args.enable_cuda_graphs is not None:
                config.enable_cuda_graphs = self.args.enable_cuda_graphs
            
            # Apply concurrent testing settings
            if self.args.enable_concurrent is not None:
                config.enable_concurrent_testing = self.args.enable_concurrent
            
            if self.args.max_concurrent_requests:
                config.max_concurrent_requests = self.args.max_concurrent_requests
        
        return configs
    
    async def run_benchmarks(self) -> dict:
        """Run the complete benchmark suite."""
        
        logger.info("Starting Higgs Audio TensorRT-LLM Benchmark Suite")
        logger.info("=" * 60)
        
        # Validate paths
        if not self.validate_paths():
            raise RuntimeError("Path validation failed")
        
        # Create benchmark configurations
        configs = self.create_benchmark_configs()
        if not configs:
            raise RuntimeError("No valid benchmark configurations created")
        
        # Run main benchmarks
        logger.info("Running main benchmark suite...")
        benchmark_results = await run_higgs_audio_benchmarks(
            unified_engine_path=self.args.unified_engine,
            separate_text_engine_path=self.args.separate_text_engine,
            separate_audio_engine_path=self.args.separate_audio_engine,
            model_config_path=self.args.model_config,
            output_dir=str(self.output_dir),
            benchmark_types=self._parse_benchmark_types()
        )
        
        # Run TTS-specific benchmarks if requested
        tts_benchmark_types = [BenchmarkType.GENERATION_MODES, BenchmarkType.DELAY_PATTERNS, 
                              BenchmarkType.STREAMING, BenchmarkType.DUALFFN]
        requested_tts_types = [t for t in self._parse_benchmark_types() if t in tts_benchmark_types]
        
        if requested_tts_types:
            logger.info("Running TTS-specific benchmarks...")
            await self._run_tts_specific_benchmarks(requested_tts_types)
        
        # Perform comprehensive analysis
        logger.info("Performing comprehensive analysis...")
        analysis_results = await self.analysis_orchestrator.analyze_benchmark_results(
            benchmark_results=benchmark_results.get('benchmark_results', {}),
            comparison_results=benchmark_results.get('comparison_results', {}),
            performance_claims_validation=benchmark_results.get('performance_claims_validation', {}),
            system_info=self._get_system_info(),
            execution_metadata=self.execution_metadata
        )
        
        # Update execution metadata
        self.execution_metadata['end_time'] = time.time()
        self.execution_metadata['total_duration_seconds'] = (
            self.execution_metadata['end_time'] - self.execution_metadata['start_time']
        )
        
        # Generate final summary
        summary = self._generate_execution_summary(benchmark_results, analysis_results)
        
        logger.info("Benchmark suite completed successfully!")
        logger.info(f"Total execution time: {summary['total_duration_minutes']:.1f} minutes")
        logger.info(f"Results saved to: {self.output_dir}")
        
        return {
            'benchmark_results': benchmark_results,
            'analysis_results': analysis_results,
            'execution_summary': summary
        }
    
    async def _run_tts_specific_benchmarks(self, benchmark_types: List[BenchmarkType]) -> None:
        """Run TTS-specific benchmarks."""
        
        # Initialize TTS orchestrator if needed
        if self.tts_orchestrator is None:
            # Import and initialize model components (simplified for this example)
            try:
                from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
                from tensorrt_llm.models.higgs_audio.model import HiggsAudioForCausalLM
                
                # Load configuration
                config = HiggsAudioConfig.from_hugging_face(self.args.model_config)
                
                # Initialize model (simplified - would need actual model loading)
                model = None  # Placeholder
                
                self.tts_orchestrator = TTSBenchmarkOrchestrator(model, config)
                
            except Exception as e:
                logger.warning(f"Failed to initialize TTS orchestrator: {e}")
                return
        
        # Create TTS-specific configurations
        tts_configs = []
        for benchmark_type in benchmark_types:
            config = BenchmarkConfiguration(
                benchmark_type=benchmark_type,
                architecture_type=ArchitectureType.UNIFIED,
                num_runs=self.args.num_runs or 30,
                batch_sizes=self.args.batch_sizes or [1, 2],
                sequence_lengths=self.args.sequence_lengths or [256, 512]
            )
            tts_configs.append(config)
        
        # Run TTS benchmarks
        for config in tts_configs:
            logger.info(f"Running TTS benchmark: {config.benchmark_type.value}")
            
            try:
                tts_results = await self.tts_orchestrator.run_tts_benchmarks(config)
                
                # Save TTS results
                tts_output_path = self.output_dir / f"tts_{config.benchmark_type.value}_results.json"
                with open(tts_output_path, 'w') as f:
                    json.dump(tts_results, f, indent=2, default=str)
                
                logger.info(f"TTS {config.benchmark_type.value} results saved to {tts_output_path}")
                
            except Exception as e:
                logger.error(f"TTS benchmark failed for {config.benchmark_type.value}: {e}")
    
    def _get_system_info(self) -> dict:
        """Get comprehensive system information."""
        try:
            import psutil
            import torch
            
            system_info = {
                'cpu_count': psutil.cpu_count(),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'memory_total_gb': psutil.virtual_memory().total / 1e9,
                'memory_available_gb': psutil.virtual_memory().available / 1e9,
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
            
            # Add GPU information
            if torch.cuda.is_available():
                gpu_info = []
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpu_info.append({
                        'name': props.name,
                        'memory_total_gb': props.total_memory / 1e9,
                        'compute_capability': f"{props.major}.{props.minor}",
                        'multiprocessor_count': props.multi_processor_count
                    })
                system_info['gpu_info'] = gpu_info
            
            return system_info
            
        except Exception as e:
            logger.warning(f"Failed to collect system info: {e}")
            return {'error': str(e)}
    
    def _generate_execution_summary(self, benchmark_results: dict, analysis_results: dict) -> dict:
        """Generate execution summary."""
        
        duration_seconds = self.execution_metadata['total_duration_seconds']
        
        summary = {
            'total_duration_seconds': duration_seconds,
            'total_duration_minutes': duration_seconds / 60,
            'benchmark_types_run': len(benchmark_results.get('benchmark_results', {})),
            'comparisons_performed': len(benchmark_results.get('comparison_results', {})),
            'performance_claims_validated': sum(benchmark_results.get('performance_claims_validation', {}).values()),
            'performance_claims_total': len(benchmark_results.get('performance_claims_validation', {})),
            'output_directory': str(self.output_dir),
            'execution_timestamp': self.execution_metadata['start_time']
        }
        
        # Add validation summary
        claims_validation = benchmark_results.get('performance_claims_validation', {})
        if claims_validation:
            summary['validation_success_rate'] = (
                claims_validation.get('validation_ratio', 0) * 100
            )
            summary['overall_validation'] = claims_validation.get('overall_validation', False)
        
        return summary


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    
    parser = argparse.ArgumentParser(
        description="Higgs Audio TensorRT-LLM Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks with default settings
  python benchmark_runner.py --unified-engine /path/to/unified \\
                            --separate-text-engine /path/to/text \\
                            --separate-audio-engine /path/to/audio \\
                            --model-config /path/to/config.json

  # Run specific benchmark types
  python benchmark_runner.py --unified-engine /path/to/unified \\
                            --separate-text-engine /path/to/text \\
                            --separate-audio-engine /path/to/audio \\
                            --model-config /path/to/config.json \\
                            --benchmark-types latency memory throughput

  # Run TTS-specific benchmarks with custom settings
  python benchmark_runner.py --unified-engine /path/to/unified \\
                            --separate-text-engine /path/to/text \\
                            --separate-audio-engine /path/to/audio \\
                            --model-config /path/to/config.json \\
                            --benchmark-types generation_modes delay_patterns \\
                            --num-runs 50 --batch-sizes 1 2 4 \\
                            --enable-cuda-graphs
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--unified-engine',
        required=True,
        help='Path to unified Higgs Audio TensorRT engine'
    )
    
    parser.add_argument(
        '--separate-text-engine',
        required=True,
        help='Path to separate text engine (for comparison)'
    )
    
    parser.add_argument(
        '--separate-audio-engine',
        required=True,
        help='Path to separate audio engine (for comparison)'
    )
    
    parser.add_argument(
        '--model-config',
        required=True,
        help='Path to Higgs Audio model configuration'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output-dir',
        default='./benchmark_results',
        help='Output directory for results (default: ./benchmark_results)'
    )
    
    parser.add_argument(
        '--benchmark-types',
        nargs='+',
        choices=[t.value for t in BenchmarkType],
        help='Specific benchmark types to run (default: all)'
    )
    
    parser.add_argument(
        '--num-runs',
        type=int,
        help='Number of runs per benchmark (default: 50 for statistical significance)'
    )
    
    parser.add_argument(
        '--warmup-runs',
        type=int,
        default=5,
        help='Number of warmup runs per benchmark (default: 5)'
    )
    
    parser.add_argument(
        '--batch-sizes',
        type=int,
        nargs='+',
        help='Batch sizes to test (default: [1, 2, 4, 8])'
    )
    
    parser.add_argument(
        '--sequence-lengths',
        type=int,
        nargs='+',
        help='Sequence lengths to test (default: [128, 256, 512, 1024])'
    )
    
    parser.add_argument(
        '--enable-cuda-graphs',
        action='store_true',
        default=None,
        help='Enable CUDA graph optimizations'
    )
    
    parser.add_argument(
        '--disable-cuda-graphs',
        action='store_false',
        dest='enable_cuda_graphs',
        help='Disable CUDA graph optimizations'
    )
    
    parser.add_argument(
        '--enable-concurrent',
        action='store_true',
        default=None,
        help='Enable concurrent request testing'
    )
    
    parser.add_argument(
        '--max-concurrent-requests',
        type=int,
        default=32,
        help='Maximum concurrent requests for throughput testing (default: 32)'
    )
    
    parser.add_argument(
        '--confidence-level',
        type=float,
        default=0.95,
        choices=[0.90, 0.95, 0.99],
        help='Confidence level for statistical analysis (default: 0.95)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform dry run without executing benchmarks'
    )
    
    return parser


async def main():
    """Main entry point."""
    
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Higgs Audio TensorRT-LLM Benchmark Runner")
    logger.info("=" * 50)
    
    # Create benchmark runner
    runner = BenchmarkRunner(args)
    
    try:
        if args.dry_run:
            logger.info("Performing dry run...")
            
            # Validate paths and configurations
            if not runner.validate_paths():
                sys.exit(1)
            
            configs = runner.create_benchmark_configs()
            logger.info(f"Dry run successful - would run {len(configs)} benchmark configurations")
            
            # Print configuration summary
            for i, config in enumerate(configs[:5]):  # Show first 5
                logger.info(f"  {i+1}. {config.benchmark_type.value} ({config.architecture_type.value})")
            
            if len(configs) > 5:
                logger.info(f"  ... and {len(configs) - 5} more configurations")
            
            return
        
        # Run benchmarks
        results = await runner.run_benchmarks()
        
        # Print final summary
        summary = results['execution_summary']
        logger.info("\n" + "=" * 60)
        logger.info("EXECUTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Duration: {summary['total_duration_minutes']:.1f} minutes")
        logger.info(f"Benchmark Types: {summary['benchmark_types_run']}")
        logger.info(f"Comparisons: {summary['comparisons_performed']}")
        logger.info(f"Claims Validated: {summary['performance_claims_validated']}/{summary['performance_claims_total']}")
        logger.info(f"Overall Validation: {'PASSED' if summary.get('overall_validation', False) else 'FAILED'}")
        logger.info(f"Results Directory: {summary['output_directory']}")
        logger.info("=" * 60)
        
        # Exit with appropriate code
        if summary.get('overall_validation', False):
            logger.info("✅ All performance claims validated successfully!")
            sys.exit(0)
        else:
            logger.warning("⚠️ Some performance claims failed validation")
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Benchmark execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())