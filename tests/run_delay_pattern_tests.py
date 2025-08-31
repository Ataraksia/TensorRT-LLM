#!/usr/bin/env python3
"""
Test runner script for Higgs Audio delay pattern tests.

This script provides an easy way to run different test suites with appropriate
configuration and error handling.
"""

import sys
import os
import subprocess
import argparse
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def run_command(cmd: List[str], capture_output: bool = False) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            cwd=os.path.dirname(__file__)
        )
        return result
    except FileNotFoundError:
        print(f"Error: Command not found: {cmd[0]}")
        print("Make sure pytest is installed: pip install pytest")
        sys.exit(1)


def check_dependencies() -> bool:
    """Check if required dependencies are available."""
    try:
        import pytest
        import numpy
        print("✓ Required dependencies available")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Install requirements: pip install pytest numpy")
        return False


def run_quick_tests() -> int:
    """Run quick functionality tests."""
    print("\n" + "="*60)
    print("RUNNING QUICK TESTS (Basic Functionality)")
    print("="*60)
    
    cmd = [
        "python", "-m", "pytest",
        "test_higgs_audio_delay_patterns.py::TestDelayPatternProvider::test_initialization_valid_params",
        "test_higgs_audio_delay_patterns.py::TestDelayPatternProvider::test_generate_delay_pattern_linear",
        "test_higgs_audio_delay_patterns.py::TestDelayPatternProvider::test_generate_delay_pattern_exponential",
        "test_higgs_audio_delay_patterns.py::TestAudioTokenUtils::test_initialization",
        "test_higgs_audio_delay_patterns.py::TestAudioTokenUtils::test_validate_audio_tokens_single_tensor",
        "test_higgs_audio_delay_patterns.py::TestDelayAwareAttentionUtils::test_initialization",
        "-v", "--tb=short"
    ]
    
    result = run_command(cmd)
    return result.returncode


def run_unit_tests() -> int:
    """Run all unit tests."""
    print("\n" + "="*60)
    print("RUNNING UNIT TESTS")
    print("="*60)
    
    cmd = [
        "python", "-m", "pytest",
        "test_higgs_audio_delay_patterns.py",
        "-m", "unit",
        "-v", "--tb=short"
    ]
    
    result = run_command(cmd)
    return result.returncode


def run_integration_tests() -> int:
    """Run integration tests."""
    print("\n" + "="*60)
    print("RUNNING INTEGRATION TESTS")
    print("="*60)
    
    cmd = [
        "python", "-m", "pytest",
        "test_higgs_audio_delay_patterns.py",
        "-m", "integration",
        "--run-integration",
        "-v", "--tb=short"
    ]
    
    result = run_command(cmd)
    return result.returncode


def run_real_world_tests() -> int:
    """Run real-world scenario tests."""
    print("\n" + "="*60)
    print("RUNNING REAL-WORLD SCENARIO TESTS")
    print("="*60)
    
    cmd = [
        "python", "-m", "pytest",
        "test_higgs_audio_delay_patterns.py",
        "-m", "real_world",
        "--run-real-world",
        "-v", "--tb=short"
    ]
    
    result = run_command(cmd)
    return result.returncode


def run_slow_tests() -> int:
    """Run slow/comprehensive tests."""
    print("\n" + "="*60)
    print("RUNNING SLOW/COMPREHENSIVE TESTS")
    print("="*60)
    
    cmd = [
        "python", "-m", "pytest",
        "test_higgs_audio_delay_patterns.py",
        "-m", "slow",
        "--run-slow",
        "-v", "--tb=short"
    ]
    
    result = run_command(cmd)
    return result.returncode


def run_all_tests() -> int:
    """Run the complete test suite."""
    print("\n" + "="*60)
    print("RUNNING COMPLETE TEST SUITE")
    print("="*60)
    
    cmd = [
        "python", "-m", "pytest",
        "test_higgs_audio_delay_patterns.py",
        "--run-slow", "--run-integration", "--run-real-world",
        "-v", "--tb=short"
    ]
    
    result = run_command(cmd)
    return result.returncode


def run_with_coverage() -> int:
    """Run tests with coverage reporting."""
    print("\n" + "="*60)
    print("RUNNING TESTS WITH COVERAGE")
    print("="*60)
    
    try:
        import pytest_cov
        coverage_available = True
    except ImportError:
        print("Warning: pytest-cov not available. Install with: pip install pytest-cov")
        coverage_available = False
    
    if coverage_available:
        cmd = [
            "python", "-m", "pytest",
            "test_higgs_audio_delay_patterns.py",
            "--cov=tensorrt_llm.models.higgs_audio",
            "--cov-report=html",
            "--cov-report=term",
            "-v"
        ]
    else:
        cmd = [
            "python", "-m", "pytest",
            "test_higgs_audio_delay_patterns.py",
            "-v"
        ]
    
    result = run_command(cmd)
    
    if coverage_available and result.returncode == 0:
        print("\n" + "="*60)
        print("Coverage report generated in htmlcov/index.html")
        print("="*60)
    
    return result.returncode


def run_specific_test(test_name: str) -> int:
    """Run a specific test by name."""
    print(f"\n" + "="*60)
    print(f"RUNNING SPECIFIC TEST: {test_name}")
    print("="*60)
    
    cmd = [
        "python", "-m", "pytest",
        f"test_higgs_audio_delay_patterns.py::{test_name}",
        "-v", "--tb=long", "-s"
    ]
    
    result = run_command(cmd)
    return result.returncode


def list_available_tests():
    """List all available tests."""
    print("\n" + "="*60)
    print("AVAILABLE TESTS")
    print("="*60)
    
    cmd = [
        "python", "-m", "pytest",
        "test_higgs_audio_delay_patterns.py",
        "--collect-only", "-q"
    ]
    
    result = run_command(cmd)
    return result.returncode


def validate_test_environment() -> bool:
    """Validate the test environment setup."""
    print("\n" + "="*60)
    print("VALIDATING TEST ENVIRONMENT")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Check test files exist
    test_file = os.path.join(os.path.dirname(__file__), 'test_higgs_audio_delay_patterns.py')
    conftest_file = os.path.join(os.path.dirname(__file__), 'conftest.py')
    
    if not os.path.exists(test_file):
        print(f"✗ Test file not found: {test_file}")
        return False
    else:
        print(f"✓ Test file found: {test_file}")
    
    if not os.path.exists(conftest_file):
        print(f"✗ Conftest file not found: {conftest_file}")
        return False
    else:
        print(f"✓ Conftest file found: {conftest_file}")
    
    # Check if model files exist
    model_file = os.path.join(os.path.dirname(__file__), '..', 'tensorrt_llm', 'models', 'higgs_audio', 'model.py')
    config_file = os.path.join(os.path.dirname(__file__), '..', 'tensorrt_llm', 'models', 'higgs_audio', 'config.py')
    
    if not os.path.exists(model_file):
        print(f"✗ Model file not found: {model_file}")
        return False
    else:
        print(f"✓ Model file found: {model_file}")
    
    if not os.path.exists(config_file):
        print(f"✗ Config file not found: {config_file}")
        return False
    else:
        print(f"✓ Config file found: {config_file}")
    
    print("✓ Test environment validation complete")
    return True


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Run Higgs Audio delay pattern tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s quick           # Run basic functionality tests
  %(prog)s unit            # Run all unit tests
  %(prog)s integration     # Run integration tests
  %(prog)s real-world      # Run real-world scenario tests
  %(prog)s all             # Run complete test suite
  %(prog)s coverage        # Run tests with coverage
  %(prog)s list            # List available tests
  %(prog)s validate        # Validate test environment
  %(prog)s specific TestDelayPatternProvider::test_initialization_valid_params
        """
    )
    
    parser.add_argument(
        'test_type',
        choices=['quick', 'unit', 'integration', 'real-world', 'slow', 'all', 'coverage', 'list', 'validate', 'specific'],
        help='Type of tests to run'
    )
    
    parser.add_argument(
        'test_name',
        nargs='?',
        help='Specific test name to run (when test_type is "specific")'
    )
    
    parser.add_argument(
        '--validate-first',
        action='store_true',
        help='Validate environment before running tests'
    )
    
    args = parser.parse_args()
    
    # Validate environment if requested
    if args.validate_first or args.test_type == 'validate':
        if not validate_test_environment():
            print("\n✗ Environment validation failed!")
            sys.exit(1)
        
        if args.test_type == 'validate':
            print("\n✓ Environment validation successful!")
            return
    
    # Run the requested tests
    exit_code = 0
    
    try:
        if args.test_type == 'quick':
            exit_code = run_quick_tests()
        elif args.test_type == 'unit':
            exit_code = run_unit_tests()
        elif args.test_type == 'integration':
            exit_code = run_integration_tests()
        elif args.test_type == 'real-world':
            exit_code = run_real_world_tests()
        elif args.test_type == 'slow':
            exit_code = run_slow_tests()
        elif args.test_type == 'all':
            exit_code = run_all_tests()
        elif args.test_type == 'coverage':
            exit_code = run_with_coverage()
        elif args.test_type == 'list':
            exit_code = list_available_tests()
        elif args.test_type == 'specific':
            if not args.test_name:
                print("Error: test_name is required when using 'specific'")
                parser.print_help()
                sys.exit(1)
            exit_code = run_specific_test(args.test_name)
        
        # Print summary
        if exit_code == 0:
            print("\n" + "="*60)
            print("✓ ALL TESTS PASSED!")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("✗ SOME TESTS FAILED!")
            print(f"Exit code: {exit_code}")
            print("="*60)
    
    except KeyboardInterrupt:
        print("\n\nTest run interrupted by user")
        exit_code = 130
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        exit_code = 1
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()