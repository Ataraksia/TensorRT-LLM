
#!/usr/bin/env python3
"""
Advanced Configuration Example for Higgs Audio TensorRT-LLM

This example demonstrates sophisticated configuration and customization capabilities
of the production-ready Higgs Audio TensorRT-LLM system. It showcases advanced
features including DualFFN layer configuration, CUDA graph optimization, delay
pattern strategies, generation mode management, and performance tuning.

Features:
- Advanced DualFFN architecture configuration
- CUDA graph optimization with TTS-specific patterns
- Delay pattern strategies and customization
- Generation mode management and transitions
- Memory optimization and resource management
- Performance profiling and monitoring
- Custom configuration templates and presets

Usage:
    python advanced_config.py --config_preset production --show_config
    python advanced_config.py --optimize_for latency --export_config config.json
    python advanced_config.py --custom_config custom.json --benchmark
"""

import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
import copy

import torch
import numpy as np

from tensorrt_llm.models.higgs_audio.config import HiggsAudioConfig
from basic_tts import HiggsAudioTTS
from tensorrt_llm import logger

# Set up logging



@dataclass
class ConfigurationProfile:
    """Configuration profile for different use cases."""
    name: str
    description: str
    optimization_target: str  # "latency", "throughput", "quality", "memory"
    config_overrides: Dict[str, Any]
    performance_expectations: Dict[str, str]
    use_cases: List[str]


class AdvancedConfigurationManager:
    """Advanced configuration management for Higgs Audio TTS.
    
    This class provides sophisticated configuration capabilities including
    preset optimization profiles, custom configuration templates, performance
    tuning, and comprehensive configuration validation.
    
    Example:
        >>> config_manager = AdvancedConfigurationManager()
        >>> config = config_manager.get_optimized_config("production", "latency")
        >>> config_manager.validate_configuration(config)
        >>> tts = HiggsAudioTTS(config=config)
    """
    
    def __init__(self, base_model_path: Optional[str] = None):
        """Initialize the configuration manager.
        
        Args:
            base_model_path: Path to base model for configuration initialization
        """
        self.base_model_path = base_model_path
        self.configuration_profiles = self._load_configuration_profiles()
        self.performance_benchmarks = {}
        
        # Load base configuration if model path provided
        if base_model_path:
            self.base_config = HiggsAudioConfig.from_hugging_face(base_model_path)
        else:
            self.base_config = HiggsAudioConfig()
            
        logger.info("Advanced configuration manager initialized")
    
    def _load_configuration_profiles(self) -> Dict[str, ConfigurationProfile]:
        """Load predefined configuration profiles."""
        profiles = {}
        
        # Development Profile
        profiles["development"] = ConfigurationProfile(
            name="development",
            description="Development and experimentation profile",
            optimization_target="flexibility",
            config_overrides={
                # Basic settings for development
                "cuda_graph_enable": False,  # Disable for easier debugging
                "audio_generation_temperature": 0.8,
                "cuda_graph_debug_mode": True,
                "cuda_graph_validation_enabled": True,
                "generation_mode_auto_transitions": True,
                "generation_mode_validation_enabled": True,
                # Memory settings
                "cuda_graph_memory_pool_size_gb": 2.0,
                "cuda_graph_max_cache_size": 8,
            },
            performance_expectations={
                "latency": "200-300ms (not optimized for speed)",
                "memory": "4-6GB (debugging overhead)",
                "throughput": "2-4 req/sec (single request focus)"
            },
            use_cases=["model development", "feature testing", "debugging", "experimentation"]
        )
        
        # Production Profile
        profiles["production"] = ConfigurationProfile(
            name="production",
            description="Production deployment profile with balanced optimization",
            optimization_target="balanced",
            config_overrides={
                # CUDA graph optimization
                "cuda_graph_enable": True,
                "cuda_graph_enable_streaming": True,
                "cuda_graph_enable_dualffn": True,
                "cuda_graph_enable_delay_patterns": True,
                
                # Batch size optimization
                "cuda_graph_tts_batch_sizes": [1, 2, 4, 8],
                "cuda_graph_tts_sequence_lengths": [128, 256, 512, 1024],
                "cuda_graph_streaming_chunk_sizes": [16, 32, 64],
                
                # Memory optimization
                "cuda_graph_memory_pool_size_gb": 4.0,
                "cuda_graph_max_cache_size": 32,
                "cuda_graph_cleanup_threshold": 0.8,
                
                # Generation settings
                "audio_generation_temperature": 0.7,
                "use_delay_pattern": True,
                "audio_streaming_chunk_size": 48,
                
                # DualFFN optimization
                "audio_dual_ffn_layers": [8, 16, 24],  # Strategic layer placement
                "cuda_graph_dualffn_separate_graphs": True,
                
                # Performance monitoring
                "cuda_graph_enable_performance_monitoring": True,
                "cuda_graph_export_metrics": True,
                
                # Reliability features
                "cuda_graph_fallback_enabled": True,
                "generation_mode_recovery_enabled": True,
            },
            performance_expectations={
                "latency": "155-195ms (15-25ms improvement)",
                "memory": "8-12GB (20-30% reduction)",
                "throughput": "10-17 req/sec (25-40% increase)"
            },
            use_cases=["production deployment", "web services", "applications", "enterprise"]
        )
        
        # High-Performance Profile
        profiles["high_performance"] = ConfigurationProfile(
            name="high_performance",
            description="Maximum performance profile optimized for throughput",
            optimization_target="throughput",
            config_overrides={
                # Aggressive CUDA graph settings
                "cuda_graph_enable": True,
                "cuda_graph_enable_streaming": True,
                "cuda_graph_enable_dualffn": True,
                "cuda_graph_enable_delay_patterns": True,
                
                # Large batch optimization
                "cuda_graph_tts_batch_sizes": [1, 2, 4, 8, 16, 32],
                "cuda_graph_tts_sequence_lengths": [128, 256, 512, 1024, 2048],
                "cuda_graph_streaming_chunk_sizes": [32, 64, 128],
                
                # Maximum memory utilization
                "cuda_graph_memory_pool_size_gb": 8.0,
                "cuda_graph_max_cache_size": 64,
                "cuda_graph_cleanup_threshold": 0.9,
                
                # Performance-focused generation
                "audio_generation_temperature": 0.8,  # Slightly higher for speed
                "audio_streaming_chunk_size": 64,  # Larger chunks
                
                # Aggressive DualFFN
                "audio_dual_ffn_layers": [4, 8, 12, 16, 20, 24, 28],  # More layers
                "cuda_graph_dualffn_separate_graphs": True,
                "cuda_graph_dualffn_audio_text_ratio_threshold": 0.4,
                
                # Maximum optimization
                "cuda_graph_warmup_iterations": 5,
                "cuda_graph_enable_performance_monitoring": True,
            },
            performance_expectations={
                "latency": "120-160ms (maximum speed)",
                "memory": "12-20GB (high utilization)",
                "throughput": "20-30 req/sec (maximum concurrent)"
            },
            use_cases=["high-load services", "batch processing", "real-time applications"]
        )
        
        # Low-Latency Profile
        profiles["low_latency"] = ConfigurationProfile(
            name="low_latency",
            description="Ultra-low latency profile for real-time applications",
            optimization_target="latency",
            config_overrides={
                # Latency-optimized CUDA graphs
                "cuda_graph_enable": True,
                "cuda_graph_enable_streaming": True,
                "cuda_graph_streaming_latency_target_ms": 50.0,
                
                # Small batch focus
                "cuda_graph_tts_batch_sizes": [1, 2],
                "cuda_graph_tts_sequence_lengths": [128, 256, 512],
                "cuda_graph_streaming_chunk_sizes": [16, 32],
                
                # Fast memory management
                "cuda_graph_memory_pool_size_gb": 2.0,
                "cuda_graph_max_cache_size": 16,
                
                # Fast generation settings
                "audio_generation_temperature": 0.9,  # Higher temp for speed
                "audio_streaming_chunk_size": 32,  # Smaller chunks
                "use_delay_pattern": True,
                "audio_delay_pattern_strategy": "linear",
                "audio_delay_pattern_stride": 1,
                
                # Minimal DualFFN for speed
                "audio_dual_ffn_layers": [16],  # Single strategic layer
                "cuda_graph_dualffn_separate_graphs": False,
                
                # Real-time optimizations
                "audio_realtime_mode": True,
                "cuda_graph_capture_mode": "automatic",
                "cuda_graph_warmup_iterations": 3,
            },
            performance_expectations={
                "latency": "80-120ms (ultra-fast)",
                "memory": "4-8GB (optimized)",
                "throughput": "8-15 req/sec (latency-focused)"
            },
            use_cases=["real-time chat", "live streaming", "interactive applications"]
        )
        
        # Memory-Optimized Profile
        profiles["memory_optimized"] = ConfigurationProfile(
            name="memory_optimized",
            description="Memory-efficient profile for resource-constrained environments",
            optimization_target="memory",
            config_overrides={
                # Conservative CUDA graph settings
                "cuda_graph_enable": True,
                "cuda_graph_memory_pool_size_gb": 1.5,
                "cuda_graph_max_cache_size": 8,
                "cuda_graph_cleanup_threshold": 0.7,
                
                # Small batch sizes
                "cuda_graph_tts_batch_sizes": [1, 2, 4],
                "cuda_graph_tts_sequence_lengths": [128, 256, 512],
                "cuda_graph_streaming_chunk_sizes": [16, 32],
                
                # Memory-efficient generation
                "audio_generation_temperature": 0.7,
                "audio_streaming_chunk_size": 32,
                
                # Minimal DualFFN
                "audio_dual_ffn_layers": [12, 20],  # Fewer layers
                "cuda_graph_dualffn_separate_graphs": False,
                
                # Memory management
                "kv_cache_free_gpu_memory_fraction": 0.85,
                "kv_cache_enable_block_reuse": True,
                "state_compression_enabled": True,
            },
            performance_expectations={
                "latency": "180-220ms (memory-focused)",
                "memory": "3-6GB (minimal usage)",
                "throughput": "4-8 req/sec (resource-constrained)"
            },
            use_cases=["edge deployment", "resource-limited servers", "development machines"]
        )
        
        # Quality-Focused Profile
        profiles["high_quality"] = ConfigurationProfile(
            name="high_quality",
            description="Maximum quality profile for premium audio generation",
            optimization_target="quality",
            config_overrides={
                # Quality-focused settings
                "cuda_graph_enable": True,
                "cuda_graph_enable_streaming": True,
                "cuda_graph_enable_dualffn": True,
                
                # Moderate batch sizes for stability
                "cuda_graph_tts_batch_sizes": [1, 2, 4],
                "cuda_graph_tts_sequence_lengths": [256, 512, 1024, 2048],
                "cuda_graph_streaming_chunk_sizes": [32, 64],
                
                # Quality generation settings
                "audio_generation_temperature": 0.6,  # Lower temp for stability
                "audio_streaming_chunk_size": 32,  # Smaller for quality
                "audio_voice_stability": 0.8,
                "audio_voice_similarity": 0.9,
                
                # Advanced delay patterns
                "use_delay_pattern": True,
                "audio_delay_pattern_strategy": "exponential",
                "audio_delay_pattern_max_delay": 32,
                
                # Comprehensive DualFFN
                "audio_dual_ffn_layers": [6, 12, 18, 24, 30],
                "cuda_graph_dualffn_separate_graphs": True,
                
                # Quality monitoring
                "cuda_graph_enable_performance_monitoring": True,
                "generation_mode_validation_enabled": True,
            },
            performance_expectations={
                "latency": "200-280ms (quality-focused)",
                "memory": "10-16GB (quality processing)",
                "throughput": "4-8 req/sec (quality over speed)"
            },
            use_cases=["audiobook production", "professional TTS", "high-end applications"]
        )
        
        return profiles
    
    def get_configuration_profile(self, profile_name: str) -> ConfigurationProfile:
        """Get a configuration profile by name."""
        if profile_name not in self.configuration_profiles:
            available = ", ".join(self.configuration_profiles.keys())
            raise ValueError(f"Unknown profile '{profile_name}'. Available: {available}")
        
        return self.configuration_profiles[profile_name]
    
    def get_optimized_config(
        self,
        profile_name: str,
        additional_overrides: Optional[Dict[str, Any]] = None
    ) -> HiggsAudioConfig:
        """Get optimized configuration based on profile and overrides.
        
        Args:
            profile_name: Name of configuration profile
            additional_overrides: Additional configuration overrides
            
        Returns:
            Optimized HiggsAudioConfig instance
        """
        profile = self.get_configuration_profile(profile_name)
        
        # Start with base configuration
        config = copy.deepcopy(self.base_config)
        
        # Apply profile overrides
        for key, value in profile.config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")
        
        # Apply additional overrides
        if additional_overrides:
            for key, value in additional_overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logger.warning(f"Unknown configuration key: {key}")
        
        # Validate configuration
        self.validate_configuration(config)
        
        logger.info(f"Created optimized configuration with profile '{profile_name}'")
        return config
    
    def validate_configuration(self, config: HiggsAudioConfig) -> List[str]:
        """Validate configuration and return any warnings or issues.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation warnings/issues
        """
        warnings = []
        
        # Validate CUDA graph settings
        if config.cuda_graph_enable:
            if not config.cuda_graph_tts_batch_sizes:
                warnings.append("CUDA graphs enabled but no batch sizes specified")
            
            if config.cuda_graph_memory_pool_size_gb <= 0:
                warnings.append("Invalid memory pool size")
        
        # Validate DualFFN settings
        if config.audio_dual_ffn_layers:
            max_layer = max(config.audio_dual_ffn_layers)
            if max_layer >= config.num_hidden_layers:
                warnings.append(f"DualFFN layer {max_layer} exceeds model layers {config.num_hidden_layers}")
        
        # Validate delay pattern settings
        if config.use_delay_pattern:
            if config.audio_delay_pattern_strategy == "custom":
                if not config.audio_delay_pattern_custom_delays:
                    warnings.append("Custom delay pattern strategy requires custom delays")
        
        # Validate generation parameters
        if not 0.1 <= config.audio_generation_temperature <= 2.0:
            warnings.append(f"Temperature {config.audio_generation_temperature} outside recommended range [0.1, 2.0]")
        
        # Performance consistency checks
        if config.cuda_graph_streaming_latency_target_ms < 50:
            warnings.append("Very low latency target may cause performance issues")
        
        if config.cuda_graph_memory_pool_size_gb > 16:
            warnings.append("Large memory pool may cause allocation issues")
        
        # Log warnings
        for warning in warnings:
            logger.warning(f"Configuration validation: {warning}")
        
        return warnings
    
    def export_configuration(self, config: HiggsAudioConfig, output_path: str):
        """Export configuration to JSON file.
        
        Args:
            config: Configuration to export
            output_path: Output file path
        """
        config_dict = config.to_dict()
        
        # Add metadata
        config_dict["_metadata"] = {
            "export_time": time.time(),
            "tensorrt_llm_version": "0.8.0",
            "config_manager_version": "1.0.0"
        }
        
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        logger.info(f"Configuration exported to {output_path}")
    
    def import_configuration(self, config_path: str) -> HiggsAudioConfig:
        """Import configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded HiggsAudioConfig instance
        """
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Remove metadata if present
        config_dict.pop("_metadata", None)
        
        # Create configuration
        config = HiggsAudioConfig(**config_dict)
        
        # Validate imported configuration
        self.validate_configuration(config)
        
        logger.info(f"Configuration imported from {config_path}")
        return config
    
    def create_custom_profile(
        self,
        name: str,
        description: str,
        base_profile: str,
        overrides: Dict[str, Any]
    ) -> ConfigurationProfile:
        """Create a custom configuration profile.
        
        Args:
            name: Profile name
            description: Profile description
            base_profile: Base profile to extend
            overrides: Configuration overrides
            
        Returns:
            New ConfigurationProfile instance
        """
        base = self.get_configuration_profile(base_profile)
        
        # Merge overrides
        merged_overrides = base.config_overrides.copy()
        merged_overrides.update(overrides)
        
        custom_profile = ConfigurationProfile(
            name=name,
            description=description,
            optimization_target="custom",
            config_overrides=merged_overrides,
            performance_expectations={"note": "Custom profile - performance varies"},
            use_cases=["custom use case"]
        )
        
        # Add to profiles
        self.configuration_profiles[name] = custom_profile
        
        logger.info(f"Created custom profile '{name}' based on '{base_profile}'")
        return custom_profile
    
    def benchmark_configuration(
        self,
        config: HiggsAudioConfig,
        test_texts: Optional[List[str]] = None,
        num_iterations: int = 10
    ) -> Dict[str, Any]:
        """Benchmark a configuration's performance.
        
        Args:
            config: Configuration to benchmark
            test_texts: Test texts (default sample texts used if None)
            num_iterations: Number of benchmark iterations
            
        Returns:
            Benchmark results dictionary
        """
        if test_texts is None:
            test_texts = [
                "Short test.",
                "This is a medium length test sentence.",
                "This is a longer test sentence that will be used to benchmark the performance and capabilities of the system.",
            ]
        
        logger.info(f"Benchmarking configuration with {num_iterations} iterations")
        
        # Initialize TTS system with config
        tts = HiggsAudioTTS(
            model_path=str(self.base_model_path) if self.base_model_path else "/models/higgs-audio",
            device="cuda:0",
            optimization_level="custom"
        )
        # Apply custom config (this would need to be implemented in HiggsAudioTTS)
        tts.config = config
        
        benchmark_results = {
            "config_summary": self._get_config_summary(config),
            "test_results": [],
            "overall_metrics": {}
        }
        
        all_times = []
        all_rtf = []
        
        for text in test_texts:
            text_results = []
            
            for iteration in range(num_iterations):
                start_time = time.perf_counter()
                
                # Generate speech
                audio = tts.generate_speech(text)
                
                end_time = time.perf_counter()
                generation_time = (end_time - start_time) * 1000  # ms
                audio_duration = len(audio) / 22050  # seconds
                rtf = audio_duration / (generation_time / 1000)
                
                text_results.append({
                    "iteration": iteration,
                    "generation_time_ms": generation_time,
                    "audio_duration_seconds": audio_duration,
                    "real_time_factor": rtf
                })
                
                all_times.append(generation_time)
                all_rtf.append(rtf)
            
            # Calculate statistics for this text
            times = [r["generation_time_ms"] for r in text_results]
            rtfs = [r["real_time_factor"] for r in text_results]
            
            benchmark_results["test_results"].append({
                "text": text,
                "text_length": len(text),
                "avg_generation_time_ms": np.mean(times),
                "std_generation_time_ms": np.std(times),
                "min_generation_time_ms": np.min(times),
                "max_generation_time_ms": np.max(times),
                "avg_real_time_factor": np.mean(rtfs),
                "iterations": text_results
            })
        
        # Overall metrics
        benchmark_results["overall_metrics"] = {
            "total_iterations": len(all_times),
            "avg_generation_time_ms": np.mean(all_times),
            "std_generation_time_ms": np.std(all_times),
            "avg_real_time_factor": np.mean(all_rtf),
            "throughput_req_per_sec": 1000 / np.mean(all_times) if all_times else 0
        }
        
        # Cleanup
        tts.cleanup()
        
        # Store benchmark results
        config_hash = hash(str(config.to_dict()))
        self.performance_benchmarks[config_hash] = benchmark_results
        
        logger.info(f"Benchmark completed - Avg: {benchmark_results['overall_metrics']['avg_generation_time_ms']:.1f}ms, RTF: {benchmark_results['overall_metrics']['avg_real_time_factor']:.2f}")
        
        return benchmark_results
    
    def _get_config_summary(self, config: HiggsAudioConfig) -> Dict[str, Any]:
        """Get configuration summary for reporting."""
        return {
            "cuda_graph_enabled": config.cuda_graph_enable,
            "batch_sizes": config.cuda_graph_tts_batch_sizes,
            "memory_pool_gb": config.cuda_graph_memory_pool_size_gb,
            "dualffn_layers": config.audio_dual_ffn_layers,
            "delay_patterns": config.use_delay_pattern,
            "streaming_chunk_size": config.audio_streaming_chunk_size,
            "temperature": config.audio_generation_temperature
        }
    
    def compare_configurations(
        self,
        configs: Dict[str, HiggsAudioConfig],
        test_texts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare multiple configurations side by side.
        
        Args:
            configs: Dictionary mapping config names to configurations
            test_texts: Test texts for comparison
            
        Returns:
            Comparison results dictionary
        """
        logger.info(f"Comparing {len(configs)} configurations")
        
        comparison_results = {
            "configurations": {},
            "comparison_matrix": {},
            "recommendations": []
        }
        
        # Benchmark each configuration
        for config_name, config in configs.items():
            logger.info(f"Benchmarking configuration: {config_name}")
            benchmark = self.benchmark_configuration(config, test_texts)
            comparison_results["configurations"][config_name] = benchmark
        
        # Create comparison matrix
        metrics = ["avg_generation_time_ms", "avg_real_time_factor", "throughput_req_per_sec"]
        
        for metric in metrics:
            comparison_results["comparison_matrix"][metric] = {}
            for config_name in configs.keys():
                value = comparison_results["configurations"][config_name]["overall_metrics"][metric]
                comparison_results["comparison_matrix"][metric][config_name] = value
        
        # Generate recommendations
        recommendations = []
        
        # Fastest configuration
        fastest_config = min(
            configs.keys(),
            key=lambda x: comparison_results["configurations"][x]["overall_metrics"]["avg_generation_time_ms"]
        )
        fastest_time = comparison_results["configurations"][fastest_config]["overall_metrics"]["avg_generation_time_ms"]
        recommendations.append(f"Fastest: {fastest_config} ({fastest_time:.1f}ms)")
        
        # Highest throughput
        highest_throughput_config = max(
            configs.keys(),
            key=lambda x: comparison_results["configurations"][x]["overall_metrics"]["throughput_req_per_sec"]
        )
        highest_throughput = comparison_results["configurations"][highest_throughput_config]["overall_metrics"]["throughput_req_per_sec"]
        recommendations.append(f"Highest throughput: {highest_throughput_config} ({highest_throughput:.1f} req/sec)")
        
        comparison_results["recommendations"] = recommendations
        
        return comparison_results
    
    def generate_config_report(self, config: HiggsAudioConfig, profile_name: str = "custom") -> str:
        """Generate comprehensive configuration report.
        
        Args:
            config: Configuration to report on
            profile_name: Profile name for context
            
        Returns:
            Configuration report string
        """
        profile = self.configuration_profiles.get(profile_name, None)
        
        report = f"""
HIGGS AUDIO TTS CONFIGURATION REPORT
===================================

Profile: {profile_name}
{profile.description if profile else 'Custom configuration'}

CUDA Graph Configuration:
- Enabled: {config.cuda_graph_enable}
- Streaming: {config.cuda_graph_enable_streaming}
- DualFFN Graphs: {config.cuda_graph_enable_dualffn}
- Delay Pattern Graphs: {config.cuda_graph_enable_delay_patterns}

Optimization Targets:
- Batch Sizes: {config.cuda_graph_tts_batch_sizes}
- Sequence Lengths: {config.cuda_graph_tts_sequence_lengths}
- Streaming Chunks: {config.cuda_graph_streaming_chunk_sizes}

Memory Management:
- Pool Size: {config.cuda_graph_memory_pool_size_gb}GB
- Max Cache: {config.cuda_graph_max_cache_size} graphs
- Cleanup Threshold: {config.cuda_graph_cleanup_threshold:.1%}

DualFFN Architecture:
- Expert Layers: {config.audio_dual_ffn_layers}
- Separate Graphs: {config.cuda_graph_dualffn_separate_graphs}

Generation Settings:
- Temperature: {config.audio_generation_temperature}
- Chunk Size: {config.audio_streaming_chunk_size}
- Delay Patterns: {config.use_delay_pattern}
- Strategy: {config.audio_delay_pattern_strategy if config.use_delay_pattern else 'N/A'}

Performance Targets:
- Latency Target: {config.cuda_graph_streaming_latency_target_ms}ms
- Warmup Iterations: {config.cuda_graph_warmup_iterations}
"""
        
        if profile:
            report += f"""
Expected Performance ({profile.optimization_target} optimized):
- Latency: {profile.performance_expectations.get('latency', 'N/A')}
- Memory: {profile.performance_expectations.get('memory', 'N/A')}
- Throughput: {profile.performance_expectations.get('throughput', 'N/A')}

Recommended Use Cases:
{chr(10).join(f'- {use_case}' for use_case in profile.use_cases)}
"""
        
        return report
    
    def list_profiles(self) -> List[str]:
        """List all available configuration profiles."""
        return list(self.configuration_profiles.keys())
    
    def get_profile_info(self, profile_name: str) -> Dict[str, Any]:
        """Get detailed information about a profile."""
        profile = self.get_configuration_profile(profile_name)
        return {
            "name": profile.name,
            "description": profile.description,
            "optimization_target": profile.optimization_target,
            "performance_expectations": profile.performance_expectations,
            "use_cases": profile.use_cases,
            "config_overrides_count": len(profile.config_overrides)
        }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Higgs Audio Advanced Configuration Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration options
    parser.add_argument(
        "--config_preset",
        type=str,
        choices=["development", "production", "high_performance", "low_