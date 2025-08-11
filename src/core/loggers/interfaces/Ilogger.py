

from abc import ABC, abstractmethod
import os
from datetime import datetime

class ILogger(ABC):
    """Interface for Logger implementations."""
    
    @abstractmethod
    def log(self, level: str, message: str) -> None:
        """Log a message with specified level.
        
        Args:
            level: Log level (e.g., 'info', 'warning', 'error')
            message: Message to log
        """
        pass
    
    @abstractmethod
    def info(self, message: str) -> None:
        """Log an info message."""
        pass
    
    @abstractmethod
    def warning(self, message: str) -> None:
        """Log a warning message."""
        pass
    
    @abstractmethod
    def error(self, message: str) -> None:
        """Log an error message."""
        pass
    
    @abstractmethod
    def debug(self, message: str) -> None:
        """Log a debug message."""
        pass
