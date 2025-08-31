#!/usr/bin/env python3
# SPDX-License-Identifier: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Analysis and Reporting Infrastructure for Higgs Audio TensorRT-LLM Benchmarks

This module provides comprehensive analysis, visualization, and reporting capabilities
for benchmark results, including statistical analysis, performance regression detection,
and automated report generation.
"""

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_ind, ks_2samp

from .benchmark_suite import BenchmarkResult, ComparisonResult, BenchmarkType, ArchitectureType


logger = logging.getLogger(__name__)


@dataclass
class StatisticalAnalysis:
    """Comprehensive statistical analysis of benchmark results."""
    
    # Basic statistics
    mean: float = 0.0
    median: float = 0.0
    std_dev: float = 0.0
    variance: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    
    # Distribution analysis
    percentile_25: float = 0.0
    percentile_75: float = 0.0
    percentile_95: float = 0.0
    percentile_99: float = 0.0
    iqr: float = 0.0
    
    # Normality tests
    shapiro_wilk_stat: float = 0.0
    shapiro_wilk_p_value: float = 0.0
    is_normal_distribution: bool = False
    
    # Outlier analysis
    outlier_count: int = 0
    outlier_percentage: float = 0.0
    outlier_method_used: str = ""
    
    # Confidence intervals
    confidence_interval_95: Tuple[float, float] = (0.0, 0.0)
    confidence_interval_99: Tuple[float, float] = (0.0, 0.0)
    
    # Performance stability
    coefficient_of_variation: float = 0.0
    performance_stability_score: float = 0.0  # 0-1 scale, higher is better
    
    def compute_from_measurements(self, measurements: List[float], confidence_level: float = 0.95) -> None:
        """Compute all statistical measures from raw measurements."""
        if not measurements:
            return
        
        data = np.array(measurements)
        
        # Basic statistics
        self.mean = float(np.mean(data))
        self.median = float(np.median(data))
        self.std_dev = float(np.std(data, ddof=1))
        self.variance = float(np.var(data, ddof=1))
        
        # Distribution shape
        self.skewness = float(stats.skew(data))
        self.kurtosis = float(stats.kurtosis(data))
        
        # Percentiles
        self.percentile_25 = float(np.percentile(data, 25))
        self.percentile_75 = float(np.percentile(data, 75))
        self.percentile_95 = float(np.percentile(data, 95))
        self.percentile_99 = float(np.percentile(data, 99))
        self.iqr = self.percentile_75 - self.percentile_25
        
        # Normality test
        try:
            if len(data) >= 3:
                stat, p_value = stats.shapiro(data)
                self.shapiro_wilk_stat = float(stat)
                self.shapiro_wilk_p_value = float(p_value)
                self.is_normal_distribution = p_value > 0.05
        except Exception as e:
            logger.warning(f"Normality test failed: {e}")
            self.is_normal_distribution = False
        
        # Confidence intervals
        if len(data) > 1:
            try:
                sem = stats.sem(data)
                ci_95 = stats.t.interval(confidence_level, len(data)-1, loc=self.mean, scale=sem)
                self.confidence_interval_95 = (float(ci_95[0]), float(ci_95[1]))
                
                ci_99 = stats.t.interval(0.99, len(data)-1, loc=self.mean, scale=sem)
                self.confidence_interval_99 = (float(ci_99[0]), float(ci_99[1]))
            except Exception as e:
                logger.warning(f"Confidence interval computation failed: {e}")
        
        # Performance stability
        if self.mean > 0:
            self.coefficient_of_variation = self.std_dev / self.mean
            # Stability score: lower CV = higher stability
            self.performance_stability_score = max(0.0, 1.0 - min(self.coefficient_of_variation, 1.0))


@dataclass
class PerformanceRegression:
    """Performance regression detection and analysis."""
    
    # Regression identification
    regression_detected: bool = False
    regression_magnitude: float = 0.0
    regression_percentage: float = 0.0
    regression_confidence: float = 0.0
    
    # Regression details
    baseline_mean: float = 0.0
    current_mean: float = 0.0
    statistical_significance: bool = False
    p_value: float = 1.0
    
    # Trend analysis
    trend_direction: str = "stable"  # "improving", "degrading", "stable"
    trend_slope: float = 0.0
    trend_r_squared: float = 0.0
    
    # Recommendations
    recommended_actions: List[str] = field(default_factory=list)
    severity_level: str = "low"  # "low", "medium", "high", "critical"
    
    def analyze_regression(self, 
                          baseline_results: List[BenchmarkResult],
                          current_results: List[BenchmarkResult],
                          threshold_percentage: float = 0.05) -> None:
        """Analyze performance regression between baseline and current results."""
        
        if not baseline_results or not current_results:
            return
        
        # Extract measurements
        baseline_measurements = []
        current_measurements = []
        
        for result in baseline_results:
            baseline_measurements.extend(result.raw_measurements)
        
        for result in current_results:
            current_measurements.extend(result.raw_measurements)
        
        if not baseline_measurements or not current_measurements:
            return
        
        # Compute means
        self.baseline_mean = np.mean(baseline_measurements)
        self.current_mean = np.mean(current_measurements)
        
        # For latency metrics, increase is regression (degradation)
        # For throughput metrics, decrease is regression
        if "latency" in str(baseline_results[0].benchmark_type).lower():
            self.regression_magnitude = self.current_mean - self.baseline_mean
            self.regression_percentage = ((self.current_mean - self.baseline_mean) / self.baseline_mean) * 100
            self.regression_detected = self.regression_percentage > threshold_percentage
        else:
            self.regression_magnitude = self.baseline_mean - self.current_mean
            self.regression_percentage = ((self.baseline_mean - self.current_mean) / self.baseline_mean) * 100
            self.regression_detected = self.regression_percentage > threshold_percentage
        
        # Statistical significance
        try:
            stat, p_value = ttest_ind(baseline_measurements, current_measurements, equal_var=False)
            self.p_value = float(p_value)
            self.statistical_significance = p_value < 0.05
        except Exception as e:
            logger.warning(f"Statistical significance test failed: {e}")
        
        # Determine severity
        self._determine_severity()
        
        # Generate recommendations
        self._generate_recommendations()
    
    def _determine_severity(self) -> None:
        """Determine regression severity level."""
        abs_percentage = abs(self.regression_percentage)
        
        if abs_percentage < 5:
            self.severity_level = "low"
        elif abs_percentage < 15:
            self.severity_level = "medium"
        elif abs_percentage < 30:
            self.severity_level = "high"
        else:
            self.severity_level = "critical"
    
    def _generate_recommendations(self) -> None:
        """Generate recommendations based on regression analysis."""
        if not self.regression_detected:
            self.recommended_actions.append("Performance is stable - no action required")
            return
        
        if self.severity_level == "low":
            self.recommended_actions.append("Monitor performance in next benchmark cycle")
        elif self.severity_level == "medium":
            self.recommended_actions.append("Investigate potential causes of performance change")
            self.recommended_actions.append("Review recent code changes for performance impact")
        elif self.severity_level == "high":
            self.recommended_actions.append("Immediate investigation required")
            self.recommended_actions.append("Profile application to identify bottlenecks")
            self.recommended_actions.append("Consider reverting recent changes if performance-critical")
        else:  # critical
            self.recommended_actions.append("CRITICAL: Performance regression requires immediate attention")
            self.recommended_actions.append("Stop deployment until root cause is identified")
            self.recommended_actions.append("Engage performance engineering team")


class BenchmarkVisualizer:
    """Visualization utilities for benchmark results."""
    
    def __init__(self, output_dir: str = "benchmark_plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_comprehensive_dashboard(self, 
                                     results: Dict[str, List[BenchmarkResult]],
                                     comparisons: Dict[str, ComparisonResult],
                                     filename: str = "benchmark_dashboard.png") -> str:
        """Create comprehensive benchmark dashboard visualization."""
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Performance overview
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_performance_overview(ax1, results)
        
        # 2. Latency distribution
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_latency_distribution(ax2, results)
        
        # 3. Architecture comparison
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_architecture_comparison(ax3, comparisons)
        
        # 4. Performance stability
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_performance_stability(ax4, results)
        
        # 5. Memory usage
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_memory_usage(ax5, results)
        
        # 6. Throughput analysis
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_throughput_analysis(ax6, results)
        
        # 7. Error analysis
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_error_analysis(ax7, results)
        
        # 8. Trend analysis
        ax8 = fig.add_subplot(gs[3, :])
        self._plot_trend_analysis(ax8, results)
        
        plt.suptitle('Higgs Audio TensorRT-LLM Benchmark Dashboard', fontsize=16, fontweight='bold')
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def create_detailed_report_plots(self, 
                                   results: Dict[str, List[BenchmarkResult]],
                                   output_prefix: str = "detailed") -> List[str]:
        """Create detailed report plots for each benchmark type."""
        
        plot_files = []
        
        for benchmark_type in BenchmarkType:
            type_results = results.get(f"{benchmark_type.value}_unified", [])
            if not type_results:
                continue
            
            # Create detailed plot for this benchmark type
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Detailed {benchmark_type.value.upper()} Analysis', fontsize=14)
            
            # Latency vs batch size
            self._plot_latency_vs_batch_size(axes[0, 0], type_results)
            
            # Latency distribution
            self._plot_detailed_distribution(axes[0, 1], type_results)
            
            # Performance stability
            self._plot_stability_analysis(axes[1, 0], type_results)
            
            # Memory vs latency correlation
            self._plot_memory_latency_correlation(axes[1, 1], type_results)
            
            filename = f"{output_prefix}_{benchmark_type.value}.png"
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plot_files.append(str(output_path))
        
        return plot_files
    
    def _plot_performance_overview(self, ax, results: Dict[str, List[BenchmarkResult]]) -> None:
        """Plot performance overview across all benchmark types."""
        benchmark_types = []
        means = []
        stds = []
        
        for benchmark_type in BenchmarkType:
            type_results = results.get(f"{benchmark_type.value}_unified", [])
            if type_results:
                latencies = []
                for result in type_results:
                    latencies.extend(result.raw_measurements)
                
                if latencies:
                    benchmark_types.append(benchmark_type.value.upper())
                    means.append(np.mean(latencies))
                    stds.append(np.std(latencies))
        
        if benchmark_types:
            x_pos = np.arange(len(benchmark_types))
            ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(benchmark_types, rotation=45, ha='right')
            ax.set_ylabel('Latency (ms)')
            ax.set_title('Performance Overview by Benchmark Type')
            ax.grid(True, alpha=0.3)
    
    def _plot_latency_distribution(self, ax, results: Dict[str, List[BenchmarkResult]]) -> None:
        """Plot latency distribution for all results."""
        all_latencies = []
        
        for benchmark_results in results.values():
            for result in benchmark_results:
                all_latencies.extend(result.raw_measurements)
        
        if all_latencies:
            ax.hist(all_latencies, bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(all_latencies), color='red', linestyle='--', label=f'Mean: {np.mean(all_latencies):.2f}ms')
            ax.axvline(np.median(all_latencies), color='green', linestyle='--', label=f'Median: {np.median(all_latencies):.2f}ms')
            ax.set_xlabel('Latency (ms)')
            ax.set_ylabel('Frequency')
            ax.set_title('Latency Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_architecture_comparison(self, ax, comparisons: Dict[str, ComparisonResult]) -> None:
        """Plot unified vs separate architecture comparison."""
        if not comparisons:
            ax.text(0.5, 0.5, 'No comparison data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        categories = []
        improvements = []
        errors = []
        
        for key, comparison in comparisons.items():
            if comparison.baseline_results and comparison.comparison_results:
                categories.append(key.upper())
                improvements.append(comparison.improvement_percentage)
                
                # Calculate error bars from confidence intervals
                ci_lower, ci_upper = comparison.confidence_interval
                error_margin = max(abs(ci_upper - comparison.improvement_percentage),
                                 abs(ci_lower - comparison.improvement_percentage))
                errors.append(error_margin)
        
        if categories:
            x_pos = np.arange(len(categories))
            colors = ['green' if x > 0 else 'red' for x in improvements]
            ax.bar(x_pos, improvements, yerr=errors, capsize=5, alpha=0.7, color=colors)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(categories, rotation=45, ha='right')
            ax.set_ylabel('Improvement (%)')
            ax.set_title('Unified vs Separate Architecture')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax.grid(True, alpha=0.3)
    
    def _plot_performance_stability(self, ax, results: Dict[str, List[BenchmarkResult]]) -> None:
        """Plot performance stability analysis."""
        stabilities = []
        labels = []
        
        for benchmark_type in BenchmarkType:
            type_results = results.get(f"{benchmark_type.value}_unified", [])
            if type_results:
                for result in type_results:
                    if result.raw_measurements:
                        cv = np.std(result.raw_measurements) / np.mean(result.raw_measurements)
                        stabilities.append(cv)
                        labels.append(f"{benchmark_type.value[:8]}...")
        
        if stabilities:
            ax.boxplot(stabilities, labels=labels)
            ax.set_ylabel('Coefficient of Variation')
            ax.set_title('Performance Stability')
            ax.grid(True, alpha=0.3)
    
    def _plot_memory_usage(self, ax, results: Dict[str, List[BenchmarkResult]]) -> None:
        """Plot memory usage analysis."""
        memory_usage = []
        labels = []
        
        for benchmark_type in BenchmarkType:
            type_results = results.get(f"{benchmark_type.value}_unified", [])
            if type_results:
                for result in type_results:
                    if result.gpu_memory_used_mb > 0:
                        memory_usage.append(result.gpu_memory_used_mb)
                        labels.append(f"{benchmark_type.value[:6]}...")
        
        if memory_usage:
            ax.bar(range(len(memory_usage)), memory_usage, alpha=0.7)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('GPU Memory (MB)')
            ax.set_title('Memory Usage by Benchmark')
            ax.grid(True, alpha=0.3)
    
    def _plot_throughput_analysis(self, ax, results: Dict[str, List[BenchmarkResult]]) -> None:
        """Plot throughput analysis."""
        # This would analyze throughput metrics if available
        ax.text(0.5, 0.5, 'Throughput analysis\n(implementation pending)', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Throughput Analysis')
    
    def _plot_error_analysis(self, ax, results: Dict[str, List[BenchmarkResult]]) -> None:
        """Plot error analysis."""
        failed_runs = []
        labels = []
        
        for benchmark_type in BenchmarkType:
            type_results = results.get(f"{benchmark_type.value}_unified", [])
            if type_results:
                total_failed = sum(result.failed_runs for result in type_results)
                failed_runs.append(total_failed)
                labels.append(f"{benchmark_type.value[:6]}...")
        
        if failed_runs:
            ax.bar(range(len(failed_runs)), failed_runs, alpha=0.7, color='red')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Failed Runs')
            ax.set_title('Error Analysis')
            ax.grid(True, alpha=0.3)
    
    def _plot_trend_analysis(self, ax, results: Dict[str, List[BenchmarkResult]]) -> None:
        """Plot trend analysis over time."""
        # This would show performance trends if historical data is available
        ax.text(0.5, 0.5, 'Trend analysis\n(historical data required)', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Performance Trends')
    
    def _plot_latency_vs_batch_size(self, ax, results: List[BenchmarkResult]) -> None:
        """Plot latency vs batch size."""
        batch_sizes = []
        latencies = []
        
        for result in results:
            if 'batch_size' in result.test_parameters:
                batch_sizes.append(result.test_parameters['batch_size'])
                latencies.append(result.mean)
        
        if batch_sizes and latencies:
            ax.scatter(batch_sizes, latencies, alpha=0.7)
            ax.plot(batch_sizes, latencies, 'b-', alpha=0.5)
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Latency (ms)')
            ax.set_title('Latency vs Batch Size')
            ax.grid(True, alpha=0.3)
    
    def _plot_detailed_distribution(self, ax, results: List[BenchmarkResult]) -> None:
        """Plot detailed latency distribution."""
        all_latencies = []
        for result in results:
            all_latencies.extend(result.raw_measurements)
        
        if all_latencies:
            ax.hist(all_latencies, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(all_latencies), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(all_latencies):.2f}')
            ax.axvline(np.median(all_latencies), color='green', linestyle='--',
                      label=f'Median: {np.median(all_latencies):.2f}')
            ax.set_xlabel('Latency (ms)')
            ax.set_ylabel('Frequency')
            ax.set_title('Latency Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_stability_analysis(self, ax, results: List[BenchmarkResult]) -> None:
        """Plot stability analysis."""
        stability_scores = []
        labels = []
        
        for result in results:
            if result.raw_measurements:
                cv = np.std(result.raw_measurements) / np.mean(result.raw_measurements)
                stability_score = max(0, 1 - cv)  # Convert CV to stability score
                stability_scores.append(stability_score)
                labels.append(f"BS{result.test_parameters.get('batch_size', '?')}")
        
        if stability_scores:
            ax.bar(range(len(stability_scores)), stability_scores, alpha=0.7)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Stability Score')
            ax.set_title('Performance Stability')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
    
    def _plot_memory_latency_correlation(self, ax, results: List[BenchmarkResult]) -> None:
        """Plot memory vs latency correlation."""
        latencies = []
        memory_usage = []
        
        for result in results:
            if result.gpu_memory_used_mb > 0:
                latencies.append(result.mean)
                memory_usage.append(result.gpu_memory_used_mb)
        
        if latencies and memory_usage:
            ax.scatter(memory_usage, latencies, alpha=0.7)
            
            # Add trend line
            if len(latencies) > 1:
                z = np.polyfit(memory_usage, latencies, 1)
                p = np.poly1d(z)
                ax.plot(memory_usage, p(memory_usage), "r--", alpha=0.7)
            
            ax.set_xlabel('GPU Memory (MB)')
            ax.set_ylabel('Latency (ms)')
            ax.set_title('Memory vs Latency Correlation')
            ax.grid(True, alpha=0.3)


class BenchmarkReportGenerator:
    """Comprehensive benchmark report generator."""
    
    def __init__(self, output_dir: str = "benchmark_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualizer = BenchmarkVisualizer(str(self.output_dir / "plots"))
    
    def generate_comprehensive_report(self, 
                                    benchmark_results: Dict[str, List[BenchmarkResult]],
                                    comparison_results: Dict[str, ComparisonResult],
                                    performance_claims_validation: Dict[str, bool],
                                    system_info: Dict[str, Any],
                                    execution_metadata: Dict[str, Any]) -> str:
        """Generate comprehensive benchmark report."""
        
        timestamp = int(time.time())
        report_data = {
            'timestamp': timestamp,
            'execution_metadata': execution_metadata,
            'system_info': system_info,
            'benchmark_results': {},
            'comparison_results': {},
            'performance_claims_validation': performance_claims_validation,
            'analysis': {},
            'recommendations': []
        }
        
        # Process benchmark results
        for key, results in benchmark_results.items():
            report_data['benchmark_results'][key] = self._process_benchmark_results(results)
        
        # Process comparison results
        for key, comparison in comparison_results.items():
            report_data['comparison_results'][key] = {
                'improvement_percentage': comparison.improvement_percentage,
                'statistical_significance': comparison.statistical_significance,
                'p_value': comparison.p_value,
                'effect_size': comparison.effect_size,
                'validates_claim': comparison.validates_claim,
                'validation_confidence': comparison.validation_confidence
            }
        
        # Generate analysis
        report_data['analysis'] = self._generate_comprehensive_analysis(
            benchmark_results, comparison_results, performance_claims_validation
        )
        
        # Generate recommendations
        report_data['recommendations'] = self._generate_recommendations(
            benchmark_results, comparison_results, performance_claims_validation
        )
        
        # Generate visualizations
        plot_files = []
        try:
            dashboard_plot = self.visualizer.create_comprehensive_dashboard(
                benchmark_results, comparison_results, f"dashboard_{timestamp}.png"
            )
            plot_files.append(dashboard_plot)
            
            detailed_plots = self.visualizer.create_detailed_report_plots(
                benchmark_results, f"detailed_{timestamp}"
            )
            plot_files.extend(detailed_plots)
            
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")
        
        report_data['visualizations'] = plot_files
        
        # Save report
        report_path = self.output_dir / f"benchmark_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate HTML report
        html_path = self._generate_html_report(report_data, timestamp)
        
        # Generate summary text report
        summary_path = self._generate_summary_report(report_data, timestamp)
        
        logger.info(f"Comprehensive report generated: {report_path}")
        logger.info(f"HTML report: {html_path}")
        logger.info(f"Summary report: {summary_path}")
        
        return str(report_path)
    
    def _process_benchmark_results(self, results: List[BenchmarkResult]) -> List[Dict[str, Any]]:
        """Process benchmark results for reporting."""
        processed_results = []
        
        for result in results:
            # Perform statistical analysis
            analysis = StatisticalAnalysis()
            analysis.compute_from_measurements(result.raw_measurements)
            
            processed_result = {
                'benchmark_id': result.benchmark_id,
                'benchmark_type': result.benchmark_type.value,
                'architecture_type': result.architecture_type.value,
                'test_parameters': result.test_parameters,
                'statistics': {
                    'mean': analysis.mean,
                    'median': analysis.median,
                    'std_dev': analysis.std_dev,
                    'percentile_95': analysis.percentile_95,
                    'percentile_99': analysis.percentile_99,
                    'coefficient_of_variation': analysis.coefficient_of_variation,
                    'performance_stability_score': analysis.performance_stability_score,
                    'is_normal_distribution': analysis.is_normal_distribution
                },
                'system_metrics': {
                    'gpu_memory_used_mb': result.gpu_memory_used_mb,
                    'gpu_memory_reserved_mb': result.gpu_memory_reserved_mb,
                    'cpu_usage_percent': result.cpu_usage_percent,
                    'system_memory_used_mb': result.system_memory_used_mb
                },
                'quality_metrics': {
                    'failed_runs': result.failed_runs,
                    'outliers_removed': result.outliers_removed,
                    'total_measurements': len(result.raw_measurements)
                }
            }
            
            processed_results.append(processed_result)
        
        return processed_results
    
    def _generate_comprehensive_analysis(self, 
                                       benchmark_results: Dict[str, List[BenchmarkResult]],
                                       comparison_results: Dict[str, ComparisonResult],
                                       performance_claims_validation: Dict[str, bool]) -> Dict[str, Any]:
        """Generate comprehensive analysis of all results."""
        
        analysis = {
            'overall_performance_score': 0.0,
            'performance_highlights': [],
            'performance_concerns': [],
            'architecture_comparison_summary': {},
            'claims_validation_summary': {},
            'performance_trends': {},
            'system_performance_analysis': {}
        }
        
        # Calculate overall performance score
        total_score = 0
        score_components = 0
        
        # Architecture comparison analysis
        significant_improvements = 0
        total_comparisons = len(comparison_results)
        
        for key, comparison in comparison_results.items():
            if comparison.statistical_significance and comparison.improvement_percentage > 0:
                significant_improvements += 1
                analysis['performance_highlights'].append(
                    f"Significant improvement in {key}: {comparison.improvement_percentage:.2f}%"
                )
            elif comparison.improvement_percentage < 0:
                analysis['performance_concerns'].append(
                    f"Performance degradation in {key}: {comparison.improvement_percentage:.2f}%"
                )
        
        if total_comparisons > 0:
            improvement_ratio = significant_improvements / total_comparisons
            total_score += improvement_ratio * 40  # 40% weight
            score_components += 1
        
        # Claims validation analysis
        validated_claims = sum(performance_claims_validation.values())
        total_claims = len(performance_claims_validation)
        
        if total_claims > 0:
            validation_ratio = validated_claims / total_claims
            total_score += validation_ratio * 30  # 30% weight
            score_components += 1
            
            analysis['claims_validation_summary'] = {
                'validated_claims': validated_claims,
                'total_claims': total_claims,
                'validation_ratio': validation_ratio
            }
        
        # Performance stability analysis
        stability_scores = []
        for results in benchmark_results.values():
            for result in results:
                if result.raw_measurements:
                    cv = np.std(result.raw_measurements) / np.mean(result.raw_measurements)
                    stability_score = max(0, 1 - cv)
                    stability_scores.append(stability_score)
        
        if stability_scores:
            avg_stability = np.mean(stability_scores)
            total_score += avg_stability * 30  # 30% weight
            score_components += 1
        
        # Calculate final score
        if score_components > 0:
            analysis['overall_performance_score'] = total_score / score_components
        
        return analysis
    
    def _generate_recommendations(self, 
                                benchmark_results: Dict[str, List[BenchmarkResult]],
                                comparison_results: Dict[str, ComparisonResult],
                                performance_claims_validation: Dict[str, bool]) -> List[str]:
        """Generate recommendations based on analysis."""
        
        recommendations = []
        
        # Architecture recommendations
        unified_better = 0
        separate_better = 0
        
        for comparison in comparison_results.values():
            if comparison.improvement_percentage > 5:
                unified_better += 1
            elif comparison.improvement_percentage < -5:
                separate_better += 1
        
        if unified_better > separate_better:
            recommendations.append("Unified architecture shows superior performance - recommended for production")
        elif separate_better > unified_better:
            recommendations.append("Separate engines may be preferable for specific use cases - investigate further")
        
        # Claims validation recommendations
        validated_ratio = sum(performance_claims_validation.values()) / len(performance_claims_validation)
        
        if validated_ratio >= 0.8:
            recommendations.append("Performance claims largely validated - high confidence in implementation")
        elif validated_ratio >= 0.6:
            recommendations.append("Most performance claims validated - minor discrepancies to investigate")
        else:
            recommendations.append("Significant discrepancies in performance claims - detailed investigation required")
        
        # Performance optimization recommendations
        for key, results in benchmark_results.items():
            for result in results:
                if result.mean > 100:  # High latency
                    recommendations.append(f"High latency detected in {key} - consider optimization")
                
                if result.failed_runs > 0:
                    recommendations.append(f"Errors detected in {key} - investigate stability issues")
        
        return recommendations
    
    def _generate_html_report(self, report_data: Dict[str, Any], timestamp: int) -> str:
        """Generate HTML report."""
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Higgs Audio TensorRT-LLM Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .section {{ margin-bottom: 30px; }}
        .metric {{ background: #e8f4f8; padding: 10px; margin: 5px 0; border-radius: 5px; }}
        .success {{ color: green; }}
        .warning {{ color: orange; }}
        .error {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Higgs Audio TensorRT-LLM Benchmark Report</h1>
        <p><strong>Generated:</strong> {time.ctime(timestamp)}</p>
        <p><strong>Overall Performance Score:</strong> {report_data['analysis']['overall_performance_score']:.1f}/100</p>
    </div>
    
    <div class="section">
        <h2>Performance Claims Validation</h2>
        <table>
            <tr><th>Claim</th><th>Status</th></tr>
"""
        
        for claim, validated in report_data['performance_claims_validation'].items():
            if claim not in ['overall_validation', 'validation_ratio']:
                status_class = "success" if validated else "error"
                status_text = "VALIDATED" if validated else "FAILED"
                html_content += f"<tr><td>{claim}</td><td class='{status_class}'>{status_text}</td></tr>"
        
        html_content += """
        </table>
    </div>
    
    <div class="section">
        <h2>Architecture Comparison</h2>
        <table>
            <tr><th>Benchmark Type</th><th>Improvement (%)</th><th>Statistical Significance</th></tr>
"""
        
        for key, comparison in report_data['comparison_results'].items():
            sig_text = "Yes" if comparison['statistical_significance'] else "No"
            html_content += f"<tr><td>{key}</td><td>{comparison['improvement_percentage']:.2f}%</td><td>{sig_text}</td></tr>"
        
        html_content += """
        </table>
    </div>
    
    <div class="section">
        <h2>Key Findings</h2>
"""
        
        for highlight in report_data['analysis']['performance_highlights']:
            html_content += f"<div class='metric success'>✓ {highlight}</div>"
        
        for concern in report_data['analysis']['performance_concerns']:
            html_content += f"<div class='metric error'>⚠ {concern}</div>"
        
        html_content += """
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
"""
        
        for recommendation in report_data['recommendations']:
            html_content += f"<div class='metric'>• {recommendation}</div>"
        
        html_content += """
    </div>
</body>
</html>
"""
        
        html_path = self.output_dir / f"benchmark_report_{timestamp}.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return str(html_path)
    
    def _generate_summary_report(self, report_data: Dict[str, Any], timestamp: int) -> str:
        """Generate summary text report."""
        
        summary_content = f"""
HIGGS AUDIO TENSORRT-LLM BENCHMARK REPORT
{'='*50}

Generated: {time.ctime(timestamp)}
Overall Performance Score: {report_data['analysis']['overall_performance_score']:.1f}/100

PERFORMANCE CLAIMS VALIDATION:
{'-'*30}
"""
        
        for claim, validated in report_data['performance_claims_validation'].items():
            if claim not in ['overall_validation', 'validation_ratio']:
                status = "✓ VALIDATED" if validated else "✗ FAILED"
                summary_content += f"{claim}: {status}\n"
        
        summary_content += f"\nOverall Validation: {'✓ PASSED' if report_data['performance_claims_validation'].get('overall_validation', False) else '✗ FAILED'}\n"
        
        summary_content += f"\nARCHITECTURE COMPARISON SUMMARY:\n{'-'*30}\n"
        
        for key, comparison in report_data['comparison_results'].items():
            summary_content += f"{key}:\n"
            summary_content += f"  Improvement: {comparison['improvement_percentage']:.2f}%\n"
            summary_content += f"  Statistical Significance: {'Yes' if comparison['statistical_significance'] else 'No'}\n"
            summary_content += f"  P-value: {comparison['p_value']:.6f}\n\n"
        
        summary_content += f"KEY FINDINGS:\n{'-'*30}\n"
        
        for highlight in report_data['analysis']['performance_highlights']:
            summary_content += f"✓ {highlight}\n"
        
        for concern in report_data['analysis']['performance_concerns']:
            summary_content += f"⚠ {concern}\n"
        
        summary_content += f"\nRECOMMENDATIONS:\n{'-'*30}\n"
        
        for recommendation in report_data['recommendations']:
            summary_content += f"• {recommendation}\n"
        
        summary_path = self.output_dir / f"benchmark_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write(summary_content)
        
        return str(summary_path)


# Main analysis orchestrator

class BenchmarkAnalysisOrchestrator:
    """Main orchestrator for benchmark analysis and reporting."""
    
    def __init__(self, output_dir: str = "benchmark_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_generator = BenchmarkReportGenerator(str(self.output_dir))
    
    async def analyze_benchmark_results(self, 
                                      benchmark_results: Dict[str, List[BenchmarkResult]],
                                      comparison_results: Dict[str, ComparisonResult],
                                      performance_claims_validation: Dict[str, bool],
                                      system_info: Dict[str, Any],
                                      execution_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis of benchmark results."""
        
        logger.info("Starting comprehensive benchmark analysis")
        
        # Generate comprehensive report
        report_path = self.report_generator.generate_comprehensive_report(
            benchmark_results=benchmark_results,
            comparison_results=comparison_results,
            performance_claims_validation=performance_claims_validation,
            system_info=system_info,
            execution_metadata=execution_metadata
        )
        
        # Perform regression analysis if baseline data is available
        regression_analysis = await self._perform_regression_analysis(benchmark_results)
        
        # Generate performance insights
        insights = self._generate_performance_insights(
            benchmark_results, comparison_results, performance_claims_validation
        )
        
        analysis_results = {
            'report_path': report_path,
            'regression_analysis': regression_analysis,
            'performance_insights': insights,
            'analysis_timestamp': time.time()
        }
        
        # Save analysis results
        analysis_path = self.output_dir / f"analysis_results_{int(time.time())}.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        logger.info(f"Analysis completed: {analysis_path}")
        
        return analysis_results
    
    async def _perform_regression_analysis(self, 
                                         benchmark_results: Dict[str, List[BenchmarkResult]]) -> Dict[str, Any]:
        """Perform performance regression analysis."""
        # This would compare current results with historical baseline
        # For now, return placeholder
        return {
            'regression_analysis_available': False,
            'message': 'Historical baseline data required for regression analysis'
        }
    
    def _generate_performance_insights(self, 
                                     benchmark_results: Dict[str, List[BenchmarkResult]],
                                     comparison_results: Dict[str, ComparisonResult],
                                     performance_claims_validation: Dict[str, bool]) -> Dict[str, Any]:
        """Generate performance insights and recommendations."""
        
        insights = {
            'top_performers': [],
            'bottlenecks': [],
            'optimization_opportunities': [],
            'architecture_recommendations': [],
            'confidence_assessment': {}
        }
        
        # Identify top performers
        performance_scores = {}
        for key, results in benchmark_results.items():
            if results:
                avg_latency = np.mean([r.mean for r in results if r.raw_measurements])
                performance_scores[key] = avg_latency
        
        if performance_scores:
            sorted_scores = sorted(performance_scores.items(), key=lambda x: x[1])
            insights['top_performers'] = [name for name, _ in sorted_scores[:3]]
            insights['bottlenecks'] = [name for name, _ in sorted_scores[-3:]]
        
        # Architecture recommendations
        unified_benefits = []
        separate_benefits = []
        
        for key, comparison in comparison_results.items():
            if comparison.improvement_percentage > 10:
                unified_benefits.append(key)
            elif comparison.improvement_percentage < -10:
                separate_benefits.append(key)
        
        if unified_benefits:
            insights['architecture_recommendations'].append(
                f"Unified architecture excels in: {', '.join(unified_benefits)}"
            )
        
        if separate_benefits:
            insights['architecture_recommendations'].append(
                f"Separate engines may be better for: {', '.join(separate_benefits)}"
            )
        
        # Confidence assessment
        validation_ratio = sum(performance_claims_validation.values()) / len(performance_claims_validation)
        if validation_ratio >= 0.9:
            insights['confidence_assessment']['overall'] = 'High confidence in performance claims'
        elif validation_ratio >= 0.7:
            insights['confidence_assessment']['overall'] = 'Moderate confidence with minor discrepancies'
        else:
            insights['confidence_assessment']['overall'] = 'Low confidence - significant investigation required'
        
        return insights