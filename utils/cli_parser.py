"""
CLI argument parsing for pipeline orchestrators

Shared argument parsing logic for core and enhancement pipelines
"""
import argparse
from typing import List, Optional


def create_core_pipeline_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for core pipeline (steps 1-11)

    Returns:
        ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="ML Trading System - Core Pipeline (Steps 1-11)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python run_core_pipeline.py

  # Run specific steps
  python run_core_pipeline.py --steps 1-7

  # Run with custom symbols
  python run_core_pipeline.py --symbols 50 --start-date 2020-01-01

  # Skip steps
  python run_core_pipeline.py --skip 1,2

  # Continue on error
  python run_core_pipeline.py --continue-on-error
        """
    )

    parser.add_argument(
        '--steps',
        type=str,
        default='1-11',
        help='Steps to run (e.g., "1-11", "1,3,5", "1-7"). Default: 1-11'
    )

    parser.add_argument(
        '--skip',
        type=str,
        default=None,
        help='Steps to skip (e.g., "1,2" to skip data ingestion). Default: None'
    )

    parser.add_argument(
        '--symbols',
        type=int,
        default=500,
        help='Number of symbols to use from universe. Default: 500'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date for data ingestion (YYYY-MM-DD). Default: from config'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file. Default: config/config.yaml'
    )

    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue pipeline even if a step fails. Default: False'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print steps without executing. Default: False'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate artifacts, do not run pipeline. Default: False'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging. Default: False'
    )

    return parser


def create_enhancements_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for enhancements pipeline (steps 12-15)

    Returns:
        ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="ML Trading System - Enhancements Pipeline (Steps 12-15)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all enhancements
  python run_enhancements.py

  # Run specific steps
  python run_enhancements.py --steps 12,14

  # Run with email summary
  python run_enhancements.py --email-summary

  # Validate core artifacts first
  python run_enhancements.py --validate-core
        """
    )

    parser.add_argument(
        '--steps',
        type=str,
        default='12-15',
        help='Steps to run (e.g., "12-15", "12,14"). Default: 12-15'
    )

    parser.add_argument(
        '--skip',
        type=str,
        default=None,
        help='Steps to skip. Default: None'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file. Default: config/config.yaml'
    )

    parser.add_argument(
        '--email-summary',
        action='store_true',
        help='Send email summary of results. Default: False'
    )

    parser.add_argument(
        '--validate-core',
        action='store_true',
        help='Validate that core pipeline artifacts exist before running. Default: False'
    )

    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue even if a step fails. Default: False'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print steps without executing. Default: False'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging. Default: False'
    )

    return parser


def parse_step_ranges(step_arg: str, skip_arg: Optional[str] = None) -> List[int]:
    """
    Parse step range argument and apply skip filter

    Examples:
        parse_step_ranges("1-5") -> [1, 2, 3, 4, 5]
        parse_step_ranges("1,3,5") -> [1, 3, 5]
        parse_step_ranges("1-5", "2,4") -> [1, 3, 5]

    Args:
        step_arg: Step range string
        skip_arg: Optional skip string

    Returns:
        List of step numbers to execute
    """
    steps = []

    # Parse main step argument
    for part in step_arg.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            steps.extend(range(int(start), int(end) + 1))
        else:
            steps.append(int(part))

    # Remove duplicates
    steps = sorted(set(steps))

    # Apply skip filter
    if skip_arg:
        skip_steps = []
        for part in skip_arg.split(','):
            part = part.strip()
            if '-' in part:
                start, end = part.split('-')
                skip_steps.extend(range(int(start), int(end) + 1))
            else:
                skip_steps.append(int(part))

        steps = [s for s in steps if s not in skip_steps]

    return steps


def validate_step_ranges(steps: List[int], valid_range: tuple = (1, 15)):
    """
    Validate that steps are in valid range

    Args:
        steps: List of step numbers
        valid_range: Tuple of (min, max) valid step numbers

    Raises:
        ValueError if any step is out of range
    """
    min_step, max_step = valid_range

    for step in steps:
        if step < min_step or step > max_step:
            raise ValueError(f"Step {step} is out of valid range [{min_step}, {max_step}]")


def print_execution_plan(steps: List[int], step_names: dict):
    """
    Print execution plan

    Args:
        steps: List of step numbers to execute
        step_names: Dict mapping step number to name
    """
    print("\n" + "="*60)
    print("EXECUTION PLAN")
    print("="*60)

    for step in steps:
        name = step_names.get(step, f"Step {step}")
        print(f"  {step:2d}. {name}")

    print("="*60 + "\n")
