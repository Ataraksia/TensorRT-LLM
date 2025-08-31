    async def _run_single_unified_benchmark(self, 
                                          config: BenchmarkConfiguration,
                                          batch_size: int,
                                          seq_len: int,
                                          generation_mode: Optional[GenerationMode] = None) -> BenchmarkResult:
        """Run a single benchmark iteration with unified architecture."""
        
        benchmark_id = f"unified_{config.benchmark_type.value}_{batch_size}_{seq_len}"
        if generation_mode:
            benchmark_id += f"_{generation_mode.value}"
        
        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            benchmark_type=config.benchmark_type,
            architecture_type=ArchitectureType.UNIFIED,
            config=config,
            test_parameters={
                'batch_size': batch_size,
                'sequence_length': seq_len,
                'generation_mode': generation_mode.value if generation_mode else None
            }
        )
        
        # Warmup runs
        logger.info(f"Warmup for {benchmark_id}")
        for _ in range(config.warmup_runs):
            try:
                await self._execute_unified_inference(
                    batch_size, seq_len, generation_mode, warmup=True
                )
            except Exception as e:
                logger.warning(f"Warmup failed: {e}")
        
        # Benchmark runs
        logger.info(f"Running {config.num_runs} benchmark iterations for {benchmark_id}")
        measurements = []
        
        for run_idx in range(config.num_runs):
            try:
                start_time = time.time()
                
                # Execute inference
                await self._execute_unified_inference(
                    batch_size, seq_len, generation_mode, warmup=False
                )
                
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                measurements.append(latency_ms)
                
                # Memory profiling if enabled
                if config.enable_memory_profiling and run_idx % 10 == 0:
                    memory_info = self._get_memory_info()
                    result.gpu_memory_used_mb = max(result.gpu_memory_used_mb, memory_info['gpu_used_mb'])
                    result.gpu_memory_reserved_mb = max(result.gpu_memory_reserved_mb, memory_info['gpu_reserved_mb'])
                
            except Exception as e:
                logger.error(f"Benchmark run {run_idx} failed: {e}")
                result.failed_runs += 1
                result.error_messages.append(str(e))
        
        result.raw_measurements = measurements
        result.measurement_unit = "ms"
        result.compute_statistics()
        
        # Add system info
        system_info = self._get_memory_info()
        result.cpu_usage_percent = system_info['cpu_percent']
        result.system_memory_used_mb = system_info['system_memory_mb']
        
        logger.info(f"Completed {benchmark_id}: {result.mean:.2f} ± {result.std_dev:.2f} ms")
        
        return result
    
    async def _run_single_separate_benchmark(self, 
                                           config: BenchmarkConfiguration,
                                           batch_size: int,
                                           seq_len: int) -> BenchmarkResult:
        """Run a single benchmark iteration with separate engines."""
        
        benchmark_id = f"separate_{config.benchmark_type.value}_{batch_size}_{seq_len}"
        
        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            benchmark_type=config.benchmark_type,
            architecture_type=ArchitectureType.SEPARATE_ENGINES,
            config=config,
            test_parameters={
                'batch_size': batch_size,
                'sequence_length': seq_len
            }
        )
        
        # Warmup runs
        logger.info(f"Warmup for {benchmark_id}")
        for _ in range(config.warmup_runs):
            try:
                await self._execute_separate_inference(
                    batch_size, seq_len, warmup=True
                )
            except Exception as e:
                logger.warning(f"Warmup failed: {e}")
        
        # Benchmark runs
        logger.info(f"Running {config.num_runs} benchmark iterations for {benchmark_id}")
        measurements = []
        
        for run_idx in range(config.num_runs):
            try:
                start_time = time.time()
                
                # Execute inference with separate engines
                await self._execute_separate_inference(
                    batch_size, seq_len, warmup=False
                )
                
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                measurements.append(latency_ms)
                
                # Memory profiling if enabled
                if config.enable_memory_profiling and run_idx % 10 == 0:
                    memory_info = self._get_memory_info()
                    result.gpu_memory_used_mb = max(result.gpu_memory_used_mb, memory_info['gpu_used_mb'])
                    result.gpu_memory_reserved_mb = max(result.gpu_memory_reserved_mb, memory_info['gpu_reserved_mb'])
                
            except Exception as e:
                logger.error(f"Benchmark run {run_idx} failed: {e}")
                result.failed_runs += 1
                result.error_messages.append(str(e))
        
        result.raw_measurements = measurements
        result.measurement_unit = "ms"
        result.compute_statistics()
        
        # Add system info
        system_info = self._get_memory_info()
        result.cpu_usage_percent = system_info['cpu_percent']
        result.system_memory_used_mb = system_info['system_memory_mb']
        
        logger.info(f"Completed {benchmark_id}: {result.mean:.2f} ± {result.std_dev:.2f} ms")
        
        return result
    
    async def _execute_unified_inference(self, 
                                       batch_size: int, 
                                       seq_len: int, 
                                       generation_mode: Optional[GenerationMode] = None,
                                       warmup: bool = False) -> None:
        """Execute inference using unified Higgs Audio model."""
        if self._unified_runner is None:
            raise RuntimeError("Unified runner not loaded")
        
        # Create sample input
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len), dtype=torch.int32)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        
        # Create sampling config
        sampling_config = SamplingConfig(
            end_id=self.config.eos_token_id or 2,
            pad_id=self.config.pad_token_id or 0,
            max_new_tokens=50,
            temperature=1.0,
            top_k=50,
            top_p=0.95
        )
        
        # Execute inference
        if warmup:
            # For warmup, just run without collecting results
            self._unified_runner.generate(
                batch_input_ids=input_ids,
                sampling_config=sampling_config,
                max_new_tokens=10  # Shorter for warmup
            )
        else:
            # Full inference for benchmarking
            outputs = self._unified_runner.generate(
                batch_input_ids=input_ids,
                sampling_config=sampling_config,
                max_new_tokens=50
            )
            
            # Ensure outputs are computed
            if hasattr(outputs, 'output_ids'):
                _ = outputs.output_ids.cpu()
    
    async def _execute_separate_inference(self, 
                                        batch_size: int, 
                                        seq_len: int,
                                        warmup: bool = False) -> None:
        """Execute inference using separate text and audio engines."""
        if self._separate_text_runner is None or self._separate_audio_runner is None:
            raise RuntimeError("Separate runners not loaded")
        
        # Create sample inputs
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len), dtype=torch.int32)
        
        # Create sampling config
        sampling_config = SamplingConfig(
            end_id=self.config.eos_token_id or 2,
            pad_id=self.config.pad_token_id or 0,
            max_new_tokens=50,
            temperature=1.0,
            top_k=50,
            top_p=0.95
        )
        
        # Execute text inference
        if warmup:
            self._separate_text_runner.generate(
                batch_input_ids=input_ids,
                sampling_config=sampling_config,
                max_new_tokens=10
            )
        else:
            text_outputs = self._separate_text_runner.generate(
                batch_input_ids=input_ids,
                sampling_config=sampling_config,
                max_new_tokens=50
            )
            
            # Simulate audio processing (simplified for benchmarking)
            # In real implementation, this would involve audio token processing
            await asyncio.sleep(0.001)  # Minimal delay to simulate audio processing
            
            # Ensure outputs are computed
            if hasattr(text_outputs, 'output_ids'):
                _ = text_outputs.output_ids.cpu()
    
    def _load_unified_runner(self) -> ModelRunner:
        """Load unified Higgs Audio model runner."""
        try:
            logger.info(f"Loading unified runner from {self.unified_engine_path}")
            
            # This is a simplified loading - in practice, you'd need to handle
            # the specific TensorRT-LLM loading mechanism for Higgs Audio
            runner = ModelRunnerCpp.from_dir(str(self.unified_engine_path))
            
            return runner
            
        except Exception as e:
            logger.error(f"Failed to load unified runner: {e}")
            raise
    
    def _load_separate_text_runner(self) -> ModelRunner:
        """Load separate text engine runner."""
        try:
            logger.info(f"Loading separate text runner from {self.separate_text_engine_path}")
            
            runner = ModelRunnerCpp.from_dir(str(self.separate_text_engine_path))
            
            return runner
            
        except Exception as e:
            logger.error(f"Failed to load separate text runner: {e}")
            raise
    
    def _load_separate_audio_runner(self) -> ModelRunner:
        """Load separate audio engine runner."""
        try:
            logger.info(f"Loading separate audio runner from {self.separate_audio_engine_path}")
            
            # This would need to be implemented based on the specific audio engine format
            # For now, return a placeholder
            logger.warning("Separate audio runner loading not fully implemented")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load separate audio runner: {e}")
            raise
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information."""
        try:
            gpu_memory = torch.cuda.mem_get_info()
            gpu_used_mb = (gpu_memory[1] - gpu_memory[0]) / 1024 / 1024
            gpu_reserved_mb = gpu_memory[1] / 1024 / 1024
            
            return {
                'gpu_used_mb': gpu_used_mb,
                'gpu_reserved_mb': gpu_reserved_mb,
                'cpu_percent': psutil.cpu_percent(),
                'system_memory_mb': psutil.virtual_memory().used / 1024 / 1024
            }
        except Exception as e:
            logger.warning(f"Failed to get memory info: {e}")
            return {
                'gpu_used_mb': 0.0,
                'gpu_reserved_mb': 0.0,
                'cpu_percent': 0.0,
                'system_memory_mb': 0.0
            }
    
    async def _run_comparative_analysis(self) -> None:
        """Run comparative analysis between unified and separate architectures."""
        logger.info("Running comparative analysis")
        
        # Compare each benchmark type
        for benchmark_type in BenchmarkType:
            unified_key = f"{benchmark_type.value}_unified"
            separate_key = f"{benchmark_type.value}_separate_engines"
            
            if unified_key in self.benchmark_results and separate_key in self.benchmark_results:
                unified_results = self.benchmark_results[unified_key]
                separate_results = self.benchmark_results[separate_key]
                
                comparison = ComparisonResult(
                    comparison_id=f"comparison_{benchmark_type.value}",
                    baseline_name="separate_engines",
                    comparison_name="unified_engine",
                    metric_name=benchmark_type.value
                )
                
                comparison.baseline_results = separate_results
                comparison.comparison_results = unified_results
                comparison.compute_comparison()
                
                self.comparison_results[benchmark_type.value] = comparison
                
                logger.info(f"Comparison for {benchmark_type.value}:")
                logger.info(f"  Improvement: {comparison.improvement_percentage:.2f}%")
                logger.info(f"  P-value: {comparison.p_value:.6f}")
                logger.info(f"  Significant: {comparison.statistical_significance}")
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        report = {
            'timestamp': time.time(),
            'benchmark_suite_version': '1.0.0',
            'system_info': self.system_info,
            'gpu_info': self.gpu_info,
            'benchmark_results': {},
            'comparison_results': {},
            'performance_claims_validation': {},
            'summary': {}
        }
        
        # Add benchmark results
        for key, results in self.benchmark_results.items():
            report['benchmark_results'][key] = [
                {
                    'benchmark_id': r.benchmark_id,
                    'mean': r.mean,
                    'std_dev': r.std_dev,
                    'min': r.min_value,
                    'max': r.max_value,
                    'percentile_95': r.percentile_95,
                    'percentile_99': r.percentile_99,
                    'outliers_removed': r.outliers_removed,
                    'failed_runs': r.failed_runs
                }
                for r in results
            ]
        
        # Add comparison results
        for key, comparison in self.comparison_results.items():
            report['comparison_results'][key] = {
                'improvement_mean': comparison.improvement_mean,
                'improvement_percentage': comparison.improvement_percentage,
                'statistical_significance': comparison.statistical_significance,
                'p_value': comparison.p_value,
                'effect_size': comparison.effect_size,
                'validates_claim': comparison.validates_claim,
                'validation_confidence': comparison.validation_confidence
            }
        
        # Generate summary
        report['summary'] = self._generate_summary(report)
        
        return report
    
    def _generate_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from the report."""
        summary = {
            'total_benchmarks_run': 0,
            'total_comparisons': len(report['comparison_results']),
            'significant_improvements': 0,
            'validated_claims': 0,
            'average_improvement': 0.0,
            'performance_highlights': [],
            'concerns': []
        }
        
        # Count benchmarks and analyze results
        total_improvement = 0.0
        improvement_count = 0
        
        for key, comparison_data in report['comparison_results'].items():
            summary['total_benchmarks_run'] += 1
            
            if comparison_data['statistical_significance']:
                summary['significant_improvements'] += 1
            
            if comparison_data['validates_claim']:
                summary['validated_claims'] += 1
            
            total_improvement += comparison_data['improvement_percentage']
            improvement_count += 1
        
        if improvement_count > 0:
            summary['average_improvement'] = total_improvement / improvement_count
        
        # Generate highlights and concerns
        for key, comparison_data in report['comparison_results'].items():
            improvement = comparison_data['improvement_percentage']
            
            if improvement > 20:
                summary['performance_highlights'].append(
                    f"Strong improvement in {key}: {improvement:.1f}%"
                )
            elif improvement < 0:
                summary['concerns'].append(
                    f"Performance regression in {key}: {improvement:.1f}%"
                )
        
        return summary
    
    async def _save_results(self, report: Dict[str, Any]) -> None:
        """Save benchmark results to files."""
        timestamp = int(report['timestamp'])
        
        # Save JSON report
        json_path = self.output_dir / f"benchmark_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save summary as text
        summary_path = self.output_dir / f"benchmark_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write("Higgs Audio TensorRT-LLM Benchmark Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {time.ctime(report['timestamp'])}\n")
            f.write(f"GPU: {self.gpu_info['name']} ({self.gpu_info['memory_gb']:.1f}GB)\n\n")
            
            summary = report['summary']
            f.write("SUMMARY:\n")
            f.write(f"- Total benchmarks run: {summary['total_benchmarks_run']}\n")
            f.write(f"- Significant improvements: {summary['significant_improvements']}\n")
            f.write(f"- Validated claims: {summary['validated_claims']}\n")
            f.write(f"- Average improvement: {summary['average_improvement']:.2f}%\n\n")
            
            if summary['performance_highlights']:
                f.write("PERFORMANCE HIGHLIGHTS:\n")
                for highlight in summary['performance_highlights']:
                    f.write(f"- {highlight}\n")
                f.write("\n")
            
            if summary['concerns']:
                f.write("CONCERNS:\n")
                for concern in summary['concerns']:
                    f.write(f"- {concern}\n")
                f.write("\n")
            
            f.write("DETAILED COMPARISONS:\n")
            for key, comparison_data in report['comparison_results'].items():
                f.write(f"\n{key.upper()}:\n")
                f.write(f"  Improvement: {comparison_data['improvement_percentage']:.2f}%\n")
                f.write(f"  P-value: {comparison_data['p_value']:.6f}\n")
                f.write(f"  Significant: {'Yes' if comparison_data['statistical_significance'] else 'No'}\n")
                f.write(f"  Validates claim: {'Yes' if comparison_data['validates_claim'] else 'No'}\n")
        
        logger.info(f"Benchmark results saved to {self.output_dir}")


# Utility functions for creating benchmark configurations

def create_latency_benchmark_configs() -> List[BenchmarkConfiguration]:
    """Create benchmark configurations for latency testing."""
    configs = []
    
    for arch in [ArchitectureType.UNIFIED, ArchitectureType.SEPARATE_ENGINES]:
        config = BenchmarkConfiguration(
            benchmark_type=BenchmarkType.LATENCY,
            architecture_type=arch,
            num_runs=50,
            batch_sizes=[1, 2, 4],
            sequence_lengths=[128, 256, 512],
            enable_memory_profiling=True
        )
        configs.append(config)
    
    return configs


def create_memory_benchmark_configs() -> List[BenchmarkConfiguration]:
    """Create benchmark configurations for memory usage testing."""
    configs = []
    
    for arch in [ArchitectureType.UNIFIED, ArchitectureType.SEPARATE_ENGINES]:
        config = BenchmarkConfiguration(
            benchmark_type=BenchmarkType.MEMORY,
            architecture_type=arch,
            num_runs=30,  # Fewer runs for memory testing
            batch_sizes=[1, 4, 8, 16],
            sequence_lengths=[512, 1024],
            enable_memory_profiling=True
        )
        configs.append(config)
    
    return configs


def create_throughput_benchmark_configs() -> List[BenchmarkConfiguration]:
    """Create benchmark configurations for throughput testing."""
    configs = []
    
    for arch in [ArchitectureType.UNIFIED, ArchitectureType.SEPARATE_ENGINES]:
        config = BenchmarkConfiguration(
            benchmark_type=BenchmarkType.THROUGHPUT,
            architecture_type=arch,
            num_runs=20,  # Fewer runs for throughput testing
            batch_sizes=[4, 8, 16],
            sequence_lengths=[256, 512],
            enable_concurrent_testing=True,
            max_concurrent_requests=16
        )
        configs.append(config)
    
    return configs


def create_tts_specific_benchmark_configs() -> List[BenchmarkConfiguration]:
    """Create benchmark configurations for TTS-specific features."""
    configs = []
    
    # Generation modes benchmark
    config = BenchmarkConfiguration(
        benchmark_type=BenchmarkType.GENERATION_MODES,
        architecture_type=ArchitectureType.UNIFIED,
        num_runs=30,
        batch_sizes=[1, 2],
        sequence_lengths=[256, 512],
        generation_modes=[
            GenerationMode.TEXT,
            GenerationMode.AUDIO_INIT,
            GenerationMode.AUDIO_IN_PROGRESS
        ]
    )
    configs.append(config)
    
    # Delay patterns benchmark
    config = BenchmarkConfiguration(
        benchmark_type=BenchmarkType.DELAY_PATTERNS,
        architecture_type=ArchitectureType.UNIFIED,
        num_runs=30,
        batch_sizes=[1, 2],
        sequence_lengths=[256, 512],
        num_codebooks=12,
        delay_pattern_strategies=["linear", "exponential"]
    )
    configs.append(config)
    
    # Streaming benchmark
    config = BenchmarkConfiguration(
        benchmark_type=BenchmarkType.STREAMING,
        architecture_type=ArchitectureType.UNIFIED,
        num_runs=25,
        batch_sizes=[1],
        sequence_lengths=[128, 256],
        streaming_chunk_sizes=[16, 32, 64]
    )
    configs.append(config)
    
    return configs


def create_cuda_graph_benchmark_configs() -> List[BenchmarkConfiguration]:
    """Create benchmark configurations for CUDA graph testing."""
    configs = []
    
    # CUDA graphs enabled
    config = BenchmarkConfiguration(
        benchmark_type=BenchmarkType.CUDA_GRAPHS,
        architecture_type=ArchitectureType.UNIFIED,
        num_runs=40,
        batch_sizes=[1, 2, 4],
        sequence_lengths=[256, 512],
        enable_cuda_graphs=True
    )
    configs.append(config)
    
    # CUDA graphs disabled (baseline)
    config = BenchmarkConfiguration(
        benchmark_type=BenchmarkType.CUDA_GRAPHS,
        architecture_type=ArchitectureType.UNIFIED,
        num_runs=40,
        batch_sizes=[1, 2, 4],
        sequence_lengths=[256, 512],
        enable_cuda_graphs=False
    )
    configs.append(config)
    
    return configs


# Main execution function

async def run_higgs_audio_benchmarks(
    unified_engine_path: str,
    separate_text_engine_path: str,
    separate_audio_engine_path: str,
    model_config_path: str,
    output_dir: str = "benchmark_results",
    benchmark_types: Optional[List[BenchmarkType]] = None
) -> Dict[str, Any]:
    """
    Run comprehensive Higgs Audio benchmarking suite.
    
    Args:
        unified_engine_path: Path to unified Higgs Audio engine
        separate_text_engine_path: Path to separate text engine
        separate_audio_engine_path: Path to separate audio engine
        model_config_path: Path to model configuration
        output_dir: Output directory for results
        benchmark_types: Specific benchmark types to run (None for all)
        
    Returns:
        Comprehensive benchmark report
    """
    
    # Initialize benchmark suite
    suite = HiggsAudioBenchmarkSuite(
        unified_engine_path=unified_engine_path,
        separate_text_engine_path=separate_text_engine_path,
        separate_audio_engine_path=separate_audio_engine_path,
        model_config_path=model_config_path,
        output_dir=output_dir
    )
    
    # Create benchmark configurations
    configs = []
    
    if benchmark_types is None:
        benchmark_types = list(BenchmarkType)
    
    for benchmark_type in benchmark_types:
        if benchmark_type == BenchmarkType.LATENCY:
            configs.extend(create_latency_benchmark_configs())
        elif benchmark_type == BenchmarkType.MEMORY:
            configs.extend(create_memory_benchmark_configs())
        elif benchmark_type == BenchmarkType.THROUGHPUT:
            configs.extend(create_throughput_benchmark_configs())
        elif benchmark_type in [BenchmarkType.GENERATION_MODES, BenchmarkType.DELAY_PATTERNS, BenchmarkType.STREAMING]:
            configs.extend(create_tts_specific_benchmark_configs())
        elif benchmark_type == BenchmarkType.CUDA_GRAPHS:
            configs.extend(create_cuda_graph_benchmark_configs())
    
    # Run comprehensive benchmarks
    report = await suite.run_comprehensive_benchmark(configs, enable_comparison=True)
    
    # Validate performance claims
    validation_results = await suite.validate_performance_claims()
    report['performance_claims_validation'] = validation_results
    
    logger.info("Benchmark suite completed successfully")
    logger.info(f"Results saved to {output_dir}")
    
    return report


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Higgs Audio TensorRT-LLM Benchmark Suite")
    parser.add_argument("--unified-engine", required=True, help="Path to unified Higgs Audio engine")
    parser.add_argument("--separate-text-engine", required=True, help="Path to separate text engine")
    parser.add_argument("--separate-audio-engine", required=True, help="Path to separate audio engine")
    parser.add_argument("--model-config", required=True, help="Path to model configuration")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output directory")
    parser.add_argument("--benchmark-types", nargs="+", choices=[t.value for t in BenchmarkType], 
                       help="Specific benchmark types to run")
    
    args = parser.parse_args()
    
    # Convert benchmark types
    benchmark_types = None
    if args.benchmark_types:
        benchmark_types = [BenchmarkType(t) for t in args.benchmark_types]
    
    # Run benchmarks
    asyncio.run(run_higgs_audio_benchmarks(
        unified_engine_path=args.unified_engine,
        separate_text_engine_path=args.separate_text_engine,
        separate_audio_engine_path=args.separate_audio_engine,
        model_config_path=args.model_config,
        output_dir=args.output_dir,
        benchmark_types=benchmark_types
    ))