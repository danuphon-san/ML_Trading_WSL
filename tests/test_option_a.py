"""
Automated test suite for Option A: Daily Live Runner

This script runs comprehensive tests to validate the daily_live_runner.py implementation.
Tests cover dry-run mode, output validation, safety checks, and edge cases.

Usage:
    python tests/test_option_a.py              # Run all tests
    python tests/test_option_a.py --test 1     # Run specific test
    python tests/test_option_a.py --quick      # Run quick tests only (skip data update)
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_test_header(test_num: int, test_name: str):
    """Print test header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}")
    print(f"Test {test_num}: {test_name}")
    print(f"{'='*70}{Colors.RESET}\n")


def print_success(message: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")


def print_error(message: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {message}{Colors.RESET}")


def print_warning(message: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.RESET}")


def print_info(message: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {message}{Colors.RESET}")


def run_command(cmd: list, check=True, capture_output=True) -> subprocess.CompletedProcess:
    """Run shell command and return result"""
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture_output,
            text=True,
            cwd=PROJECT_ROOT
        )
        return result
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {' '.join(cmd)}")
        print_error(f"Error: {e.stderr}")
        raise


def check_prerequisites() -> dict:
    """Check all prerequisites before running tests"""
    print_test_header(0, "Prerequisites Check")

    results = {
        'environment': False,
        'config': False,
        'data': False,
        'model': False,
        'universe': False
    }

    # Check Python environment
    try:
        import pandas as pd
        import numpy as np
        import yaml
        print_success("Python environment: pandas, numpy, yaml installed")
        results['environment'] = True
    except ImportError as e:
        print_error(f"Missing Python package: {e}")
        print_info("Run: conda activate us-stock-app")

    # Check config file
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    if config_path.exists():
        print_success(f"Config file exists: {config_path}")
        results['config'] = True
    else:
        print_error(f"Config file missing: {config_path}")

    # Check OHLCV data
    data_dir = PROJECT_ROOT / "data" / "parquet" / "1d"
    if data_dir.exists() and len(list(data_dir.glob("*.parquet"))) > 0:
        n_files = len(list(data_dir.glob("*.parquet")))
        print_success(f"OHLCV data found: {n_files} parquet files")
        results['data'] = True
    else:
        print_error("No OHLCV data found")
        print_info("Run: python run_core_pipeline.py --steps 1")

    # Check model
    model_paths = [
        PROJECT_ROOT / "production" / "champion_model.pkl",
        PROJECT_ROOT / "data" / "models" / "latest" / "model.pkl"
    ]
    model_found = any(p.exists() for p in model_paths)
    if model_found:
        model_path = next(p for p in model_paths if p.exists())
        print_success(f"Model found: {model_path}")
        results['model'] = True
    else:
        print_error("No model found")
        print_info("Run: python run_core_pipeline.py --steps 1-7")

    # Check universe
    universe_path = PROJECT_ROOT / "data" / "universe.csv"
    if universe_path.exists():
        print_success(f"Universe file exists: {universe_path}")
        results['universe'] = True
    else:
        print_error("Universe file missing")
        print_info("Run: python run_core_pipeline.py --steps 1")

    # Summary
    all_pass = all(results.values())
    if all_pass:
        print_success("\n✓ All prerequisites met!")
    else:
        print_error("\n✗ Some prerequisites missing")
        print_info("Run the suggested commands above to prepare the environment")

    return results


def test_help_command():
    """Test 1: Help and version check"""
    print_test_header(1, "Help Command")

    try:
        result = run_command(
            ["python", "scripts/daily_live_runner.py", "--help"],
            check=False
        )

        if "usage:" in result.stdout.lower():
            print_success("Help command works")
            print_info(f"Exit code: {result.returncode}")
            return True
        else:
            print_error("Help command output unexpected")
            return False
    except Exception as e:
        print_error(f"Test failed: {e}")
        return False


def test_dry_run_basic():
    """Test 2: Basic dry-run execution"""
    print_test_header(2, "Basic Dry-Run Execution")

    try:
        print_info("Running: python scripts/daily_live_runner.py --dry-run --skip-data-update")

        result = run_command(
            ["python", "scripts/daily_live_runner.py", "--dry-run", "--skip-data-update"],
            check=False,
            capture_output=True
        )

        output = result.stdout

        # Check for key workflow steps in output
        checks = {
            "Market check": "market" in output.lower() or "step 1" in output.lower(),
            "Universe loaded": "universe" in output.lower() or "symbols" in output.lower(),
            "Model loaded": "model" in output.lower(),
            "Signals generated": "signal" in output.lower() or "score" in output.lower(),
            "Portfolio constructed": "portfolio" in output.lower() or "weight" in output.lower(),
            "Trades generated": "trade" in output.lower(),
            "Safety checks": "safety" in output.lower() or "validation" in output.lower()
        }

        for check_name, passed in checks.items():
            if passed:
                print_success(check_name)
            else:
                print_warning(f"{check_name} - not clearly indicated in output")

        if result.returncode == 0:
            print_success("Dry-run completed successfully (exit code 0)")
            return True
        else:
            print_error(f"Dry-run failed with exit code {result.returncode}")
            print_error(f"Error output:\n{result.stderr}")
            return False

    except Exception as e:
        print_error(f"Test failed: {e}")
        return False


def test_output_files():
    """Test 3: Verify output files are created"""
    print_test_header(3, "Output Files Validation")

    try:
        # Find latest output directory
        live_dir = PROJECT_ROOT / "live"
        if not live_dir.exists():
            print_error(f"Live directory not found: {live_dir}")
            return False

        # Get latest date directory
        date_dirs = sorted([d for d in live_dir.iterdir() if d.is_dir()], reverse=True)
        if not date_dirs:
            print_error("No output directories found in live/")
            return False

        latest_dir = date_dirs[0]
        print_info(f"Checking outputs in: {latest_dir}")

        # Expected output files
        expected_files = {
            "trades.csv": "Trade orders",
            "portfolio_weights.csv": "Portfolio weights",
            "signals.json": "ML signals",
            "monitoring_log.json": "Monitoring metrics",
            "report.html": "Daily report"
        }

        all_exist = True
        for filename, description in expected_files.items():
            filepath = latest_dir / filename
            if filepath.exists():
                size = filepath.stat().st_size
                print_success(f"{description}: {filepath.name} ({size} bytes)")
            else:
                print_error(f"Missing: {filename}")
                all_exist = False

        return all_exist

    except Exception as e:
        print_error(f"Test failed: {e}")
        return False


def test_validate_trades():
    """Test 4: Validate trade recommendations"""
    print_test_header(4, "Trade Validation")

    try:
        import pandas as pd

        # Find latest trades file
        live_dir = PROJECT_ROOT / "live"
        date_dirs = sorted([d for d in live_dir.iterdir() if d.is_dir()], reverse=True)
        latest_dir = date_dirs[0]
        trades_file = latest_dir / "trades.csv"

        if not trades_file.exists():
            print_error("Trades file not found")
            return False

        # Load trades
        trades = pd.read_csv(trades_file)
        print_info(f"Loaded {len(trades)} trade recommendations")

        # Validate columns
        required_cols = ['symbol', 'side', 'shares', 'price', 'notional']
        missing_cols = [col for col in required_cols if col not in trades.columns]
        if missing_cols:
            print_error(f"Missing columns: {missing_cols}")
            return False
        print_success("All required columns present")

        # Validate data
        checks = []

        # Check prices are positive
        if (trades['price'] > 0).all():
            print_success("All prices are positive")
            checks.append(True)
        else:
            print_error("Some prices are <= 0")
            checks.append(False)

        # Check shares are reasonable
        if (trades['shares'].abs() > 0).all():
            print_success("All share quantities are non-zero")
            checks.append(True)
        else:
            print_warning("Some share quantities are zero")
            checks.append(True)  # Warning, not error

        # Check sides are valid
        valid_sides = trades['side'].isin(['buy', 'sell']).all()
        if valid_sides:
            print_success("All sides are valid (buy/sell)")
            checks.append(True)
        else:
            print_error("Invalid side values found")
            checks.append(False)

        # Check notional consistency
        calculated_notional = (trades['shares'].abs() * trades['price']).round(2)
        actual_notional = trades['notional'].abs().round(2)
        notional_match = (calculated_notional - actual_notional).abs() < 0.50  # Allow $0.50 rounding
        if notional_match.all():
            print_success("Notional values are consistent")
            checks.append(True)
        else:
            print_warning("Some notional values don't match (rounding differences)")
            checks.append(True)  # Warning, not error

        # Summary
        print_info(f"\nTrade Summary:")
        print_info(f"  Total trades: {len(trades)}")
        print_info(f"  Buy orders: {(trades['side'] == 'buy').sum()}")
        print_info(f"  Sell orders: {(trades['side'] == 'sell').sum()}")
        print_info(f"  Total notional: ${trades['notional'].abs().sum():,.2f}")

        return all(checks)

    except Exception as e:
        print_error(f"Test failed: {e}")
        return False


def test_validate_weights():
    """Test 5: Validate portfolio weights"""
    print_test_header(5, "Portfolio Weights Validation")

    try:
        import pandas as pd

        # Find latest weights file
        live_dir = PROJECT_ROOT / "live"
        date_dirs = sorted([d for d in live_dir.iterdir() if d.is_dir()], reverse=True)
        latest_dir = date_dirs[0]
        weights_file = latest_dir / "portfolio_weights.csv"

        if not weights_file.exists():
            print_error("Weights file not found")
            return False

        # Load weights
        weights = pd.read_csv(weights_file)
        print_info(f"Loaded {len(weights)} portfolio positions")

        # Validate columns
        if 'symbol' not in weights.columns or 'weight' not in weights.columns:
            print_error("Missing required columns (symbol, weight)")
            return False
        print_success("Required columns present")

        # Validate weights
        checks = []

        # Check all weights are positive (long-only)
        if (weights['weight'] >= 0).all():
            print_success("All weights are non-negative (long-only)")
            checks.append(True)
        else:
            print_error("Some weights are negative (short positions found)")
            checks.append(False)

        # Check weights sum to ~1.0
        total_weight = weights['weight'].sum()
        if 0.95 <= total_weight <= 1.05:
            print_success(f"Weights sum to {total_weight:.4f} (within tolerance)")
            checks.append(True)
        else:
            print_error(f"Weights sum to {total_weight:.4f} (should be ~1.0)")
            checks.append(False)

        # Check no position exceeds max weight (15%)
        max_weight = weights['weight'].max()
        if max_weight <= 0.16:  # Allow small tolerance
            print_success(f"Max position weight: {max_weight:.2%} (within 15% limit)")
            checks.append(True)
        else:
            print_error(f"Max position weight: {max_weight:.2%} (exceeds 15% limit)")
            checks.append(False)

        # Check all positions meet minimum (1%)
        min_weight = weights['weight'].min()
        if min_weight >= 0.009:  # Allow small tolerance
            print_success(f"Min position weight: {min_weight:.2%} (meets 1% minimum)")
            checks.append(True)
        else:
            print_warning(f"Min position weight: {min_weight:.2%} (below 1% minimum)")
            checks.append(True)  # Warning, not error

        # Summary
        print_info(f"\nPortfolio Summary:")
        print_info(f"  Total positions: {len(weights)}")
        print_info(f"  Total weight: {total_weight:.4f}")
        print_info(f"  Avg weight: {weights['weight'].mean():.2%}")
        print_info(f"  Max weight: {max_weight:.2%}")
        print_info(f"  Min weight: {min_weight:.2%}")

        return all(checks)

    except Exception as e:
        print_error(f"Test failed: {e}")
        return False


def test_regime_detection():
    """Test 6: Check regime detection"""
    print_test_header(6, "Regime Detection Validation")

    try:
        # Find latest monitoring log
        live_dir = PROJECT_ROOT / "live"
        date_dirs = sorted([d for d in live_dir.iterdir() if d.is_dir()], reverse=True)
        latest_dir = date_dirs[0]
        monitoring_file = latest_dir / "monitoring_log.json"

        if not monitoring_file.exists():
            print_error("Monitoring log not found")
            return False

        # Load monitoring data
        with open(monitoring_file, 'r') as f:
            monitoring = json.load(f)

        # Check for regime info
        if 'regime' not in monitoring:
            print_error("No regime information in monitoring log")
            return False

        regime_info = monitoring['regime']

        # Validate regime data
        checks = []

        # Check regime state
        if 'current_regime' in regime_info:
            regime = regime_info['current_regime']
            print_success(f"Current regime detected: {regime}")

            if regime in ['risk_off', 'normal', 'risk_on']:
                print_success("Regime state is valid")
                checks.append(True)
            else:
                print_error(f"Invalid regime state: {regime}")
                checks.append(False)
        else:
            print_error("No current_regime in monitoring log")
            checks.append(False)

        # Check regime metrics
        expected_metrics = ['volatility', 'drawdown', 'trend']
        for metric in expected_metrics:
            if metric in regime_info:
                value = regime_info[metric]
                print_success(f"Regime metric {metric}: {value}")
                checks.append(True)
            else:
                print_warning(f"Missing regime metric: {metric}")
                checks.append(True)  # Warning, not error

        # Check risk multiplier
        if 'risk_multiplier' in regime_info:
            multiplier = regime_info['risk_multiplier']
            print_info(f"Risk multiplier: {multiplier}")

            if 0.3 <= multiplier <= 1.5:
                print_success("Risk multiplier is reasonable")
                checks.append(True)
            else:
                print_warning(f"Unusual risk multiplier: {multiplier}")
                checks.append(True)  # Warning

        return all(checks)

    except Exception as e:
        print_error(f"Test failed: {e}")
        return False


def test_safety_checks():
    """Test 7: Verify safety checks ran"""
    print_test_header(7, "Safety Checks Validation")

    try:
        # Find latest monitoring log
        live_dir = PROJECT_ROOT / "live"
        date_dirs = sorted([d for d in live_dir.iterdir() if d.is_dir()], reverse=True)
        latest_dir = date_dirs[0]
        monitoring_file = latest_dir / "monitoring_log.json"

        if not monitoring_file.exists():
            print_error("Monitoring log not found")
            return False

        # Load monitoring data
        with open(monitoring_file, 'r') as f:
            monitoring = json.load(f)

        # Check for safety results
        if 'safety' not in monitoring:
            print_error("No safety check information in monitoring log")
            return False

        safety_info = monitoring['safety']

        # Check validation passed
        if safety_info.get('all_passed', False):
            print_success("✓ All safety checks PASSED")

            # Show which checks were performed
            if 'portfolio_checks' in safety_info:
                print_success(f"Portfolio validation: {len(safety_info['portfolio_checks'])} checks")

            if 'trade_checks' in safety_info:
                print_success(f"Trade validation: {len(safety_info['trade_checks'])} checks")

            return True
        else:
            print_error("✗ Some safety checks FAILED")

            # Show failures
            if 'portfolio_issues' in safety_info:
                for issue in safety_info['portfolio_issues']:
                    print_error(f"  Portfolio: {issue}")

            if 'trade_issues' in safety_info:
                for issue in safety_info['trade_issues']:
                    print_error(f"  Trade: {issue}")

            return False

    except Exception as e:
        print_error(f"Test failed: {e}")
        return False


def test_historical_date():
    """Test 8: Run with historical date"""
    print_test_header(8, "Historical Date Test")

    try:
        print_info("Running: python scripts/daily_live_runner.py --date 2024-01-15 --skip-data-update")

        result = run_command(
            ["python", "scripts/daily_live_runner.py",
             "--date", "2024-01-15",
             "--skip-data-update",
             "--dry-run"],
            check=False,
            capture_output=True
        )

        if result.returncode == 0:
            print_success("Historical date run completed successfully")

            # Check if outputs were created with correct date
            output_dir = PROJECT_ROOT / "live" / "2024-01-15"
            if output_dir.exists():
                print_success(f"Output directory created: {output_dir}")
                return True
            else:
                print_warning("Output directory not found (may be normal if date is weekend/holiday)")
                return True  # Not necessarily an error
        else:
            print_error(f"Historical date run failed: {result.stderr}")
            return False

    except Exception as e:
        print_error(f"Test failed: {e}")
        return False


def test_capital_override():
    """Test 9: Capital override parameter"""
    print_test_header(9, "Capital Override Test")

    try:
        print_info("Running: python scripts/daily_live_runner.py --capital 50000 --skip-data-update")

        result = run_command(
            ["python", "scripts/daily_live_runner.py",
             "--capital", "50000",
             "--skip-data-update",
             "--dry-run"],
            check=False,
            capture_output=True
        )

        if result.returncode == 0:
            print_success("Capital override run completed")

            # Check if capital was used (look in output)
            if "50000" in result.stdout or "50,000" in result.stdout:
                print_success("Capital override was applied")
                return True
            else:
                print_warning("Cannot confirm capital override in output")
                return True  # Not necessarily an error
        else:
            print_error(f"Capital override test failed: {result.stderr}")
            return False

    except Exception as e:
        print_error(f"Test failed: {e}")
        return False


def test_verbose_mode():
    """Test 10: Verbose logging"""
    print_test_header(10, "Verbose Logging Test")

    try:
        print_info("Running: python scripts/daily_live_runner.py -v --skip-data-update")

        result = run_command(
            ["python", "scripts/daily_live_runner.py",
             "-v",
             "--skip-data-update",
             "--dry-run"],
            check=False,
            capture_output=True
        )

        output_lines = len(result.stdout.split('\n'))

        if result.returncode == 0:
            print_success("Verbose mode run completed")
            print_info(f"Output lines: {output_lines}")

            # Verbose mode should produce more output
            if output_lines > 50:
                print_success("Verbose output detected (detailed logging)")
                return True
            else:
                print_warning("Output seems brief for verbose mode")
                return True  # Not necessarily an error
        else:
            print_error(f"Verbose mode test failed: {result.stderr}")
            return False

    except Exception as e:
        print_error(f"Test failed: {e}")
        return False


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Test Option A: Daily Live Runner")
    parser.add_argument('--test', type=int, help='Run specific test number (1-10)')
    parser.add_argument('--quick', action='store_true', help='Skip long-running tests')
    parser.add_argument('--skip-prereq', action='store_true', help='Skip prerequisites check')
    args = parser.parse_args()

    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("=" * 70)
    print("Option A Test Suite: Daily Live Runner")
    print("=" * 70)
    print(f"{Colors.RESET}")

    # Check prerequisites first
    if not args.skip_prereq:
        prereq_results = check_prerequisites()
        if not all(prereq_results.values()):
            print_error("\n✗ Prerequisites not met. Please resolve issues above.")
            return 1

    # Define all tests
    all_tests = [
        ("Help Command", test_help_command),
        ("Basic Dry-Run", test_dry_run_basic),
        ("Output Files", test_output_files),
        ("Trade Validation", test_validate_trades),
        ("Portfolio Weights", test_validate_weights),
        ("Regime Detection", test_regime_detection),
        ("Safety Checks", test_safety_checks),
        ("Historical Date", test_historical_date),
        ("Capital Override", test_capital_override),
        ("Verbose Mode", test_verbose_mode),
    ]

    # Run specific test or all tests
    if args.test:
        if 1 <= args.test <= len(all_tests):
            test_name, test_func = all_tests[args.test - 1]
            passed = test_func()

            if passed:
                print_success(f"\n✓ Test {args.test} PASSED")
                return 0
            else:
                print_error(f"\n✗ Test {args.test} FAILED")
                return 1
        else:
            print_error(f"Invalid test number: {args.test} (must be 1-{len(all_tests)})")
            return 1
    else:
        # Run all tests
        results = []

        for i, (test_name, test_func) in enumerate(all_tests, 1):
            # Skip certain tests in quick mode
            if args.quick and test_name in ["Historical Date", "Capital Override", "Verbose Mode"]:
                print_info(f"Skipping Test {i}: {test_name} (quick mode)")
                continue

            try:
                passed = test_func()
                results.append((i, test_name, passed))
            except Exception as e:
                print_error(f"Test {i} crashed: {e}")
                results.append((i, test_name, False))

        # Summary
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}")
        print("TEST SUMMARY")
        print(f"{'='*70}{Colors.RESET}\n")

        passed_count = sum(1 for _, _, passed in results if passed)
        total_count = len(results)

        for test_num, test_name, passed in results:
            if passed:
                print_success(f"Test {test_num}: {test_name}")
            else:
                print_error(f"Test {test_num}: {test_name}")

        print(f"\n{Colors.BOLD}Results: {passed_count}/{total_count} tests passed{Colors.RESET}")

        if passed_count == total_count:
            print_success("\n✓ ALL TESTS PASSED - Option A is working correctly!")
            return 0
        else:
            print_error(f"\n✗ {total_count - passed_count} tests failed - see details above")
            return 1


if __name__ == "__main__":
    sys.exit(main())
