from typing import Any
from ..core import ExperimentTracker

def log_error(error: Exception, tracker: ExperimentTracker) -> None:
    error_msg = str(error)
    print(f"ERROR: {error_msg}")
    tracker.log({"error": error_msg})