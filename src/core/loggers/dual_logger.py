import os
from datetime import datetime
from src.core.loggers.interfaces.Ilogger import ILogger
class DualLogger(ILogger):
    def __init__(self, log_file_path='app.log', max_file_size=1048576, backup_count=5):
        """
        Initialize the DualLogger.
        
        Args:
            log_file_path (str): Path to the log file. Defaults to 'app.log'.
            max_file_size (int): Maximum size of log file before rotation (in bytes). Default 1MB.
            backup_count (int): Number of backup logs to keep. Default 5.
        """
        self.log_file_path = log_file_path
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self._ensure_log_directory_exists()
        
    def _ensure_log_directory_exists(self):
        """Ensure the directory for the log file exists."""
        log_dir = os.path.dirname(self.log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    def _rotate_logs(self):
        """Rotate logs if current log file exceeds max size."""
        if not os.path.exists(self.log_file_path):
            return
            
        if os.path.getsize(self.log_file_path) >= self.max_file_size:
            # Remove the oldest backup if it exists
            oldest_backup = f"{self.log_file_path}.{self.backup_count}"
            if os.path.exists(oldest_backup):
                os.remove(oldest_backup)
                
            # Shift all other backups
            for i in range(self.backup_count - 1, 0, -1):
                current_backup = f"{self.log_file_path}.{i}"
                next_backup = f"{self.log_file_path}.{i + 1}"
                if os.path.exists(current_backup):
                    os.rename(current_backup, next_backup)
                    
            # Rotate current log to backup.1
            os.rename(self.log_file_path, f"{self.log_file_path}.1")
    
    def _write_to_file(self, level, message):
        """Write log message to file with timestamp."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] [{level.upper()}] {message}\n"
        
        self._rotate_logs()
        
        with open(self.log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write(log_entry)
    
    def _print_to_terminal(self, level, message):
        """Print colored log message to terminal."""
        colors = {
            'info': '\033[94m',    # Blue
            'warning': '\033[93m', # Yellow
            'error': '\033[91m',   # Red
            'debug': '\033[90m',    # Gray
            'reset': '\033[0m'      # Reset color
        }
        
        color = colors.get(level.lower(), colors['reset'])
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"{color}[{timestamp}] [{level.upper()}] {message}{colors['reset']}")
    
    def log(self, level, message):
        """
        Log a message with specified level.
        
        Args:
            level (str): Log level (info, warning, error, debug, etc.)
            message (str): Message to log
        """
        level = level.lower()
        self._write_to_file(level, message)
        self._print_to_terminal(level, message)
    
    # Convenience methods
    def info(self, message):
        """Log an info message."""
        self.log('info', message)
    
    def warning(self, message):
        """Log a warning message."""
        self.log('warning', message)
    
    def error(self, message):
        """Log an error message."""
        self.log('error', message)
    
    def debug(self, message):
        """Log a debug message."""
        self.log('debug', message)
        
        



