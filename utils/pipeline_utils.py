"""
Pipeline utility functions

Helper functions for error handling, artifact validation, progress tracking, etc.
"""
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from loguru import logger


class PipelineTracker:
    """Track pipeline execution progress and errors"""

    def __init__(self):
        self.steps = {}
        self.start_time = None
        self.end_time = None
        self.errors = []

    def start(self):
        """Mark pipeline start"""
        self.start_time = datetime.now()
        logger.info("="*60)
        logger.info("PIPELINE STARTED")
        logger.info(f"Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)

    def mark_step_start(self, step_num: int, step_name: str):
        """Mark step start"""
        self.steps[step_num] = {
            'name': step_name,
            'status': 'running',
            'start_time': datetime.now(),
            'end_time': None,
            'error': None
        }
        logger.info(f"\n{'='*60}")
        logger.info(f"Step {step_num}: {step_name}")
        logger.info(f"{'='*60}")

    def mark_step_complete(self, step_num: int):
        """Mark step complete"""
        if step_num in self.steps:
            self.steps[step_num]['status'] = 'completed'
            self.steps[step_num]['end_time'] = datetime.now()

            duration = (self.steps[step_num]['end_time'] -
                       self.steps[step_num]['start_time']).total_seconds()

            logger.info(f"✓ Step {step_num} completed in {duration:.1f}s")

    def mark_step_error(self, step_num: int, error: Exception):
        """Mark step error"""
        if step_num in self.steps:
            self.steps[step_num]['status'] = 'failed'
            self.steps[step_num]['end_time'] = datetime.now()
            self.steps[step_num]['error'] = str(error)

        self.errors.append({
            'step': step_num,
            'step_name': self.steps[step_num]['name'],
            'error': str(error),
            'traceback': traceback.format_exc()
        })

        logger.error(f"✗ Step {step_num} failed: {error}")

    def finish(self):
        """Mark pipeline finish"""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        completed = sum(1 for s in self.steps.values() if s['status'] == 'completed')
        failed = sum(1 for s in self.steps.values() if s['status'] == 'failed')
        total = len(self.steps)

        logger.info("\n" + "="*60)
        logger.info("PIPELINE FINISHED")
        logger.info(f"Duration: {duration:.1f}s ({duration/60:.1f} minutes)")
        logger.info(f"Steps completed: {completed}/{total}")
        if failed > 0:
            logger.error(f"Steps failed: {failed}/{total}")
        logger.info("="*60)

    def get_summary(self) -> Dict:
        """Get execution summary"""
        return {
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': (self.end_time - self.start_time).total_seconds() if self.end_time else None,
            'steps': self.steps,
            'errors': self.errors
        }


def validate_artifacts(artifact_paths: Dict[str, str], required: bool = True) -> Dict[str, bool]:
    """
    Validate that required artifacts exist

    Args:
        artifact_paths: Dict mapping artifact name to file path
        required: If True, raise error on missing artifacts

    Returns:
        Dict mapping artifact name to existence status
    """
    status = {}
    missing = []

    for name, path in artifact_paths.items():
        path_obj = Path(path)

        # Handle glob patterns (e.g., ops_report_*.html)
        if '*' in str(path):
            matching_files = list(path_obj.parent.glob(path_obj.name))
            exists = len(matching_files) > 0
        else:
            exists = path_obj.exists()

        status[name] = exists

        if not exists:
            missing.append(name)

    logger.info("Artifact validation:")
    for name, exists in status.items():
        symbol = "✓" if exists else "✗"
        logger.info(f"  {symbol} {name}")

    if required and missing:
        raise FileNotFoundError(f"Missing required artifacts: {', '.join(missing)}")

    return status


def run_with_error_handling(
    func: Callable,
    step_num: int,
    step_name: str,
    tracker: PipelineTracker,
    continue_on_error: bool = False,
    **kwargs
) -> Optional[Any]:
    """
    Run a pipeline step with error handling

    Args:
        func: Function to execute
        step_num: Step number
        step_name: Step name
        tracker: PipelineTracker instance
        continue_on_error: If True, continue pipeline on error
        **kwargs: Arguments to pass to func

    Returns:
        Function result or None if error
    """
    tracker.mark_step_start(step_num, step_name)

    try:
        result = func(**kwargs)
        tracker.mark_step_complete(step_num)
        return result

    except Exception as e:
        tracker.mark_step_error(step_num, e)

        if not continue_on_error:
            logger.error("Pipeline halted due to error")
            raise

        logger.warning(f"Continuing despite error in step {step_num}")
        return None


def send_email_summary(
    config: Dict,
    subject: str,
    body: str,
    html_body: Optional[str] = None
) -> bool:
    """
    Send email summary (simplified implementation)

    Args:
        config: Configuration dict
        subject: Email subject
        body: Plain text body
        html_body: Optional HTML body

    Returns:
        True if sent successfully
    """
    ops_config = config.get('ops', {})
    recipients = ops_config.get('email_recipients', [])

    if not recipients or ops_config.get('alerts') != 'email':
        logger.info("Email alerts not configured, skipping")
        return False

    logger.info(f"Email summary would be sent to: {recipients}")
    logger.info(f"Subject: {subject}")
    logger.info(f"Body preview: {body[:200]}...")

    # TODO: Implement actual SMTP sending
    # In production, add SMTP server config to config.yaml

    return True


def parse_step_ranges(step_arg: str) -> List[int]:
    """
    Parse step range argument

    Examples:
        "1-5" -> [1, 2, 3, 4, 5]
        "1,3,5" -> [1, 3, 5]
        "1-3,7,9-11" -> [1, 2, 3, 7, 9, 10, 11]

    Args:
        step_arg: Step range string

    Returns:
        List of step numbers
    """
    steps = []

    for part in step_arg.split(','):
        if '-' in part:
            start, end = part.split('-')
            steps.extend(range(int(start), int(end) + 1))
        else:
            steps.append(int(part))

    return sorted(set(steps))


def ensure_directories(paths: List[str]):
    """
    Ensure directories exist

    Args:
        paths: List of directory paths to create
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {path}")


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "2m 30s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def get_step_name(step_num: int) -> str:
    """
    Get step name by number

    Args:
        step_num: Step number (1-15)

    Returns:
        Step name
    """
    step_names = {
        # Core pipeline (1-11)
        1: "Data Ingestion (OHLCV + Fundamentals)",
        2: "Data Preprocessing & Alignment",
        3: "Technical Feature Engineering",
        4: "Fundamental Feature Engineering (PIT)",
        5: "Label Generation",
        6: "Feature Selection & Normalization",
        7: "Model Training",
        8: "Signal Generation (Scoring)",
        9: "Portfolio Construction",
        10: "Backtesting",
        11: "Model Evaluation & Reporting",

        # Enhancements (12-15)
        12: "Regime Detection",
        13: "Sleeve Allocation",
        14: "Turnover Management",
        15: "Operations Monitoring & Reporting"
    }

    return step_names.get(step_num, f"Step {step_num}")


def print_banner(text: str, char: str = "="):
    """
    Print a banner

    Args:
        text: Banner text
        char: Border character
    """
    width = 60
    logger.info(char * width)
    logger.info(text.center(width))
    logger.info(char * width)
