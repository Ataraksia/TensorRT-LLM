"""
Comprehensive test execution framework for Higgs Audio TTS testing suite.

This module provides coordinated test execution, detailed reporting, and 
coverage analysis for the complete TTS testing framework.
"""

import pytest
import sys
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import argparse

# Test execution configuration
@dataclass
class TestSuiteConfig:
    """Configuration for test suite execution."""
    test_categories: List[str] = field(default_factory=lambda: [
        'delay_patterns',
        'integration', 
        'end_to_end',
        'performance',
        'configuration'
    ])
    
    performance_benchmarks: bool = True
    detailed_reporting: bool = True
    coverage_analysis: bool = True
    parallel_execution: bool = False
    verbose_output: bool = True
    
    # Test filtering
    include_slow_tests: bool = False
    include_gpu_tests: bool = False
    include_integration_tests: bool = True
    
    # Reporting configuration
    report_formats: List[str] = field(default_factory=lambda: ['console', 'json', 'html'])
    report_output_dir: str = 'test_reports'
    
    # Performance validation
    validate_performance_claims: bool = True
    benchmark_iterations: int = 3
    performance_tolerance_pct: float = 10.0


@dataclass
class TestResult:
    """Container for individual test results."""
    test_name: str
    category: str
    status: str  # 'passed', 'failed', 'skipped', 'error'
    duration_ms: float
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuiteResults:
    """Container for complete test suite results."""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    error_tests: int = 0
    total_duration_ms: float = 0.0
    
    # Category-specific results
    category_results: Dict[str, Dict[str, int]] = field(default_factory=dict)
    test_results: List[TestResult] = field(default_factory=list)
    
    # Performance metrics
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    coverage_report: Dict[str, Any] = field(default_factory=dict)


class HiggsAudioTestRunner:
    """Comprehensive test runner for Higgs Audio TTS test suite."""
    
    def __init__(self, config: TestSuiteConfig):
        self.config = config
        self.results = TestSuiteResults()
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('higgs_audio_test_runner')
        logger.setLevel(logging.DEBUG if self.config.verbose_output else logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        os.makedirs(self.config.report_output_dir, exist_ok=True)
        file_handler = logging.FileHandler(f'{self.config.report_output_dir}/test_execution.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def run_comprehensive_test_suite(self) -> TestSuiteResults:
        """Run the complete TTS test suite with comprehensive reporting."""
        self.logger.info("="*80)
        self.logger.info("STARTING HIGGS AUDIO TTS COMPREHENSIVE TEST SUITE")
        self.logger.info("="*80)
        
        start_time = time.time()
        
        try:
            # Initialize test environment
            self._setup_test_environment()
            
            # Run test categories
            for category in self.config.test_categories:
                self.logger.info(f"\n--- Running {category.upper()} Tests ---")
                category_results = self._run_test_category(category)
                self._process_category_results(category, category_results)
            
            # Performance validation
            if self.config.validate_performance_claims:
                self.logger.info("\n--- Validating Performance Claims ---")
                self._validate_performance_improvements()
            
            # Coverage analysis
            if self.config.coverage_analysis:
                self.logger.info("\n--- Analyzing Test Coverage ---")
                self._analyze_test_coverage()
            
            # Generate comprehensive reports
            self._generate_comprehensive_reports()
            
        except Exception as e:
            self.logger.error(f"Test suite execution failed: {e}")
            self.results.error_tests += 1
        
        finally:
            self.results.total_duration_ms = (time.time() - start_time) * 1000
            
        self.logger.info("="*80)
        self.logger.info("HIGGS AUDIO TTS TEST SUITE COMPLETED")
        self.logger.info("="*80)
        
        return self.results
    
    def _setup_test_environment(self):
        """Setup test environment and validate dependencies."""
        self.logger.info("Setting up test environment...")
        
        # Validate TensorRT-LLM availability
        try:
            import tensorrt_llm
            self.logger.info(f"TensorRT-LLM version: {tensorrt_llm.__version__}")
        except ImportError:
            self.logger.warning("TensorRT-LLM not available - some tests will be skipped")
        
        # Validate CUDA availability if GPU tests enabled
        if self.config.include_gpu_tests:
            try:
                import torch
                if torch.cuda.is_available():
                    self.logger.info(f"CUDA available: {torch.cuda.device_count()} GPUs")
                else:
                    self.logger.warning("CUDA not available - GPU tests will be skipped")
            except ImportError:
                self.logger.warning("PyTorch not available - GPU tests will be skipped")
        
        # Setup test data directories
        test_data_dir = Path("test_data")
        test_data_dir.mkdir(exist_ok=True)
        
        self.logger.info("Test environment setup completed")
    
    def _run_test_category(self, category: str) -> Dict[str, Any]:
        """Run tests for a specific category."""
        self.logger.info(f"Executing {category} tests...")
        
        # Map categories to test modules
        test_modules = {
            'delay_patterns': 'tests/test_higgs_audio_delay_patterns.py',
            'integration': 'tests/models/higgs_audio/test_tts_integration.py',
            'end_to_end': 'tests/models/higgs_audio/test_tts_end_to_end.py',
            'performance': 'tests/models/higgs_audio/test_tts_performance.py',
            'configuration': 'tests/models/higgs_audio/test_tts_configuration.py'
        }
        
        test_module = test_modules.get(category)
        if not test_module or not Path(test_module).exists():
            self.logger.warning(f"Test module for {category} not found: {test_module}")
            return {'status': 'skipped', 'reason': 'module_not_found'}
        
        # Build pytest arguments
        pytest_args = self._build_pytest_args(category, test_module)
        
        # Run tests with pytest
        start_time = time.time()
        exit_code = pytest.main(pytest_args)
        duration = (time.time() - start_time) * 1000
        
        # Process results
        results = {
            'category': category,
            'module': test_module,
            'exit_code': exit_code,
            'duration_ms': duration,
            'status': 'passed' if exit_code == 0 else 'failed'
        }
        
        self.logger.info(f"{category} tests completed in {duration:.2f}ms - Status: {results['status']}")
        
        return results
    
    def _build_pytest_args(self, category: str, test_module: str) -> List[str]:
        """Build pytest command-line arguments for test execution."""
        args = [test_module]
        
        # Verbosity
        if self.config.verbose_output:
            args.extend(['-v', '-s'])
        
        # Parallel execution
        if self.config.parallel_execution:
            args.extend(['-n', 'auto'])  # Requires pytest-xdist
        
        # Test markers based on category
        category_markers = {
            'delay_patterns': 'delay_patterns',
            'integration': 'integration',
            'end_to_end': 'end_to_end', 
            'performance': 'performance',
            'configuration': 'configuration'
        }
        
        marker = category_markers.get(category)
        if marker:
            args.extend(['-m', marker])
        
        # Conditional markers
        if not self.config.include_slow_tests:
            args.extend(['-m', 'not slow'])
        
        if not self.config.include_gpu_tests:
            args.extend(['-m', 'not gpu'])
        
        # Output format
        report_dir = Path(self.config.report_output_dir)
        report_dir.mkdir(exist_ok=True)
        
        # JUnit XML report
        args.extend(['--junit-xml', f'{report_dir}/{category}_results.xml'])
        
        # Coverage reporting
        if self.config.coverage_analysis:
            args.extend([
                '--cov=tensorrt_llm.models.higgs_audio',
                f'--cov-report=html:{report_dir}/coverage_{category}',
                f'--cov-report=json:{report_dir}/coverage_{category}.json'
            ])
        
        return args
    
    def _process_category_results(self, category: str, results: Dict[str, Any]):
        """Process and aggregate category test results."""
        # Update overall statistics
        if results['status'] == 'passed':
            self.results.passed_tests += 1
        elif results['status'] == 'failed':
            self.results.failed_tests += 1
        else:
            self.results.skipped_tests += 1
        
        self.results.total_tests += 1
        self.results.total_duration_ms += results['duration_ms']
        
        # Store category-specific results
        self.results.category_results[category] = {
            'status': results['status'],
            'duration_ms': results['duration_ms'],
            'exit_code': results.get('exit_code', -1)
        }
        
        # Parse detailed results from JUnit XML if available
        xml_file = Path(self.config.report_output_dir) / f'{category}_results.xml'
        if xml_file.exists():
            detailed_results = self._parse_junit_xml(xml_file, category)
            self.results.test_results.extend(detailed_results)
    
    def _parse_junit_xml(self, xml_file: Path, category: str) -> List[TestResult]:
        """Parse JUnit XML output to extract detailed test results."""
        try:
            import xml.etree.ElementTree as ET
            
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            test_results = []
            
            for testcase in root.findall('.//testcase'):
                test_name = testcase.get('name', 'unknown')
                duration = float(testcase.get('time', '0')) * 1000  # Convert to ms
                
                # Determine status
                status = 'passed'
                error_message = None
                
                if testcase.find('failure') is not None:
                    status = 'failed'
                    failure = testcase.find('failure')
                    error_message = failure.text if failure is not None else None
                elif testcase.find('error') is not None:
                    status = 'error'
                    error = testcase.find('error')
                    error_message = error.text if error is not None else None
                elif testcase.find('skipped') is not None:
                    status = 'skipped'
                
                result = TestResult(
                    test_name=test_name,
                    category=category,
                    status=status,
                    duration_ms=duration,
                    error_message=error_message
                )
                
                test_results.append(result)
            
            return test_results
            
        except Exception as e:
            self.logger.error(f"Failed to parse JUnit XML {xml_file}: {e}")
            return []
    
    def _validate_performance_improvements(self):
        """Validate claimed performance improvements from test results."""
        self.logger.info("Validating performance improvement claims...")
        
        performance_validation = {
            'latency_improvements': {'target': '15-25ms', 'status': 'unknown'},
            'memory_efficiency': {'target': '20-30% reduction', 'status': 'unknown'},
            'throughput_gains': {'target': '25-40% increase', 'status': 'unknown'},
            'architecture_score': {'target': '50/50 vs 19/50', 'status': 'unknown'}
        }
        
        # Check if performance tests were run
        perf_category = self.results.category_results.get('performance')
        if not perf_category or perf_category['status'] != 'passed':
            self.logger.warning("Performance tests not run or failed - cannot validate claims")
            for claim in performance_validation:
                performance_validation[claim]['status'] = 'not_validated'
        else:
            # Look for performance benchmarker results
            perf_results_file = Path(self.config.report_output_dir) / 'performance_benchmarks.json'
            if perf_results_file.exists():
                try:
                    with open(perf_results_file) as f:
                        benchmark_data = json.load(f)
                    
                    # Validate each claim
                    self._validate_latency_claims(performance_validation, benchmark_data)
                    self._validate_memory_claims(performance_validation, benchmark_data)
                    self._validate_throughput_claims(performance_validation, benchmark_data)
                    self._validate_architecture_claims(performance_validation, benchmark_data)
                    
                except Exception as e:
                    self.logger.error(f"Failed to load performance benchmark data: {e}")
            else:
                self.logger.warning("Performance benchmark data not found")
        
        self.results.performance_metrics['improvement_validation'] = performance_validation
        
        # Log validation summary
        passed_validations = sum(1 for v in performance_validation.values() if v['status'] == 'validated')
        total_validations = len(performance_validation)
        
        self.logger.info(f"Performance validation complete: {passed_validations}/{total_validations} claims validated")
    
    def _validate_latency_claims(self, validation: Dict, data: Dict):
        """Validate latency improvement claims."""
        try:
            unified_latencies = [m for m in data.get('measurements', []) 
                               if 'unified' in m.get('metadata', {}).get('architecture', '')]
            separate_latencies = [m for m in data.get('measurements', [])
                                if 'separate' in m.get('metadata', {}).get('architecture', '')]
            
            if unified_latencies and separate_latencies:
                avg_unified = sum(m['value'] for m in unified_latencies) / len(unified_latencies)
                avg_separate = sum(m['value'] for m in separate_latencies) / len(separate_latencies)
                improvement_ms = avg_separate - avg_unified
                
                if 15 <= improvement_ms <= 25:
                    validation['latency_improvements']['status'] = 'validated'
                    validation['latency_improvements']['actual'] = f'{improvement_ms:.1f}ms'
                else:
                    validation['latency_improvements']['status'] = 'failed'
                    validation['latency_improvements']['actual'] = f'{improvement_ms:.1f}ms'
        except Exception as e:
            self.logger.error(f"Latency validation failed: {e}")
    
    def _validate_memory_claims(self, validation: Dict, data: Dict):
        """Validate memory efficiency claims."""
        try:
            unified_memory = next((m for m in data.get('measurements', [])
                                 if 'unified_architecture_memory' in m.get('metric_name', '')), None)
            separate_memory = next((m for m in data.get('measurements', [])
                                  if 'separate_engines_memory' in m.get('metric_name', '')), None)
            
            if unified_memory and separate_memory:
                reduction_pct = ((separate_memory['value'] - unified_memory['value']) / 
                               separate_memory['value']) * 100
                
                if 20 <= reduction_pct <= 30:
                    validation['memory_efficiency']['status'] = 'validated'
                    validation['memory_efficiency']['actual'] = f'{reduction_pct:.1f}%'
                else:
                    validation['memory_efficiency']['status'] = 'failed'
                    validation['memory_efficiency']['actual'] = f'{reduction_pct:.1f}%'
        except Exception as e:
            self.logger.error(f"Memory validation failed: {e}")
    
    def _validate_throughput_claims(self, validation: Dict, data: Dict):
        """Validate throughput improvement claims."""
        try:
            unified_throughputs = [m for m in data.get('measurements', [])
                                 if 'unified_throughput' in m.get('metric_name', '')]
            separate_throughputs = [m for m in data.get('measurements', [])
                                  if 'separate_throughput' in m.get('metric_name', '')]
            
            if unified_throughputs and separate_throughputs:
                avg_unified = sum(m['value'] for m in unified_throughputs) / len(unified_throughputs)
                avg_separate = sum(m['value'] for m in separate_throughputs) / len(separate_throughputs)
                improvement_pct = ((avg_unified - avg_separate) / avg_