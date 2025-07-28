from typing import Any
from src.core.experiment_tracker import ExperimentTracker

def log_error(error: Exception, tracker: ExperimentTracker) -> None:
    error_msg = str(error)
    print(f"ERROR: {error_msg}")
    tracker.log({"error": error_msg})