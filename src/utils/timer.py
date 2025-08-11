import time
from typing import Optional, Union

class Timer:
    """Static timer class with context manager support."""
    
    _start_time: Optional[float] = None
    _end_time: Optional[float] = None

    @classmethod
    def start(cls) -> None:
        """Start the timer."""
        cls._start_time = time.perf_counter()
        cls._end_time = None
        return cls._start_time

    @classmethod
    def end(cls) -> None:
        """Stop the timer."""
        if cls._start_time is None:
            raise RuntimeError("Timer not started!")
        cls._end_time = time.perf_counter()
        return cls._end_time

    @classmethod
    def duration(cls, human_readable: bool = True) -> Union[float, str]:
        """
        Get duration in seconds or human-readable format.
        
        Args:
            human_readable: If True, returns formatted string (e.g., "1h 2m 3.45s")
        
        Returns:
            float: Duration in seconds if human_readable=False
            str: Formatted duration if human_readable=True
        """
        if cls._start_time is None:
            raise RuntimeError("Timer not started!")
        end = cls._end_time or time.perf_counter()
        total_seconds = end - cls._start_time

        if not human_readable:
            return total_seconds

        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

    @classmethod
    def reset(cls) -> None:
        """Reset the timer."""
        cls._start_time = None
        cls._end_time = None

    # Context manager support
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()

# Usage examples
if __name__ == "__main__":
    # Method 1: Manual control
    Timer.start()
    time.sleep(1.5)
    Timer.end()
    print(f"Duration: {Timer.duration()}")  # 0h 0m 1.50s
    print(f"Raw seconds: {Timer.duration(human_readable=False)}")  # 1.500123...

    # Method 2: Context manager (recommended)
    with Timer():
        time.sleep(0.75)
    print(f"Context duration: {Timer.duration()}")