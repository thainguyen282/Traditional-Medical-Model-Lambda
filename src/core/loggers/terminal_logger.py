from datetime import datetime
from src.core.loggers.interfaces.Ilogger import ILogger
class TerminalLogger(ILogger):
   
        
    def log(self, level, message):
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