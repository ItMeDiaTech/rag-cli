"""Structured logging infrastructure for RAG-CLI core library.

This module provides comprehensive logging with JSON format support,
file rotation, and different log levels for debugging and monitoring.
Platform-agnostic version for use in the core rag_cli library.
"""

import os
import sys
import json
import logging
import logging.handlers
import platform
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from functools import wraps
import time
import structlog
from structlog.processors import JSONRenderer, TimeStamper, add_log_level


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for standard logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON formatted string
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "created", "filename", "funcName",
                           "levelname", "levelno", "lineno", "module", "exc_info",
                           "exc_text", "stack_info", "pathname", "processName",
                           "process", "relativeCreated", "thread", "threadName",
                           "getMessage", "message"]:
                log_data[key] = value

        return json.dumps(log_data)


class TextFormatter(logging.Formatter):
    """Enhanced text formatter with colors for console output."""

    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'      # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors for console.

        Args:
            record: Log record to format

        Returns:
            Formatted string with colors
        """
        if sys.stdout.isatty():  # Only add colors if outputting to terminal
            levelname = f"{self.COLORS.get(record.levelname, '')}{record.levelname}{self.COLORS['RESET']}"
        else:
            levelname = record.levelname

        # Create formatted message
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        message = f"[{timestamp}] {levelname:8} | {record.name:20} | {record.getMessage()}"

        # Add exception info if present
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"

        return message


def create_file_handler(
    log_file: Path,
    max_bytes: int = 10485760,
    backup_count: int = 5,
    level: int = logging.INFO,
    formatter: Optional[logging.Formatter] = None
) -> logging.Handler:
    """Create a Windows-safe file handler for logging.

    On Windows, uses TimedRotatingFileHandler with process-specific log files
    to avoid file locking issues. On other platforms, uses RotatingFileHandler.

    Args:
        log_file: Path to log file
        max_bytes: Maximum bytes per log file (for RotatingFileHandler)
        backup_count: Number of backup files to keep
        level: Logging level
        formatter: Optional formatter to use

    Returns:
        Configured file handler
    """
    is_windows = platform.system() == 'Windows'

    if is_windows:
        # On Windows, use TimedRotatingFileHandler to avoid file locking issues
        # Also add process ID to filename for multi-process safety
        log_dir = log_file.parent
        log_name = log_file.stem
        log_ext = log_file.suffix

        # Add process ID to log filename for multi-process safety
        pid = os.getpid()
        process_log_file = log_dir / f"{log_name}.{pid}{log_ext}"

        # Use daily rotation which is less prone to locking issues
        file_handler = logging.handlers.TimedRotatingFileHandler(
            str(process_log_file),
            when='midnight',
            interval=1,
            backupCount=backup_count,
            encoding='utf-8'
        )
    else:
        # On non-Windows, use standard RotatingFileHandler
        file_handler = logging.handlers.RotatingFileHandler(
            str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )

    file_handler.setLevel(level)
    if formatter:
        file_handler.setFormatter(formatter)

    return file_handler


class Logger:
    """Main logger class for RAG-CLI core library.

    Platform-agnostic implementation that doesn't depend on plugin-specific code.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize logger with configuration."""
        if not self._initialized:
            self._setup_logging()
            self._initialized = True

    def _get_log_config(self) -> Dict[str, Any]:
        """Get logging configuration with fallback defaults.

        Returns:
            Dictionary with logging configuration
        """
        # Try to get config from rag_cli.core.config if available
        try:
            from rag_cli.core.config import get_config
            config = get_config()
            log_config = config.monitoring

            # Convert to dict format
            if hasattr(log_config, '__dict__'):
                return {
                    'log_level': getattr(log_config, 'log_level', 'INFO'),
                    'log_format': getattr(log_config, 'log_format', 'json'),
                    'log_file': getattr(log_config, 'log_file', None),
                    'log_rotation': {
                        'max_bytes': getattr(log_config.log_rotation, 'max_bytes', 10485760) if hasattr(log_config, 'log_rotation') else 10485760,
                        'backup_count': getattr(log_config.log_rotation, 'backup_count', 5) if hasattr(log_config, 'log_rotation') else 5
                    }
                }
            elif isinstance(log_config, dict):
                return log_config
        except Exception:
            pass

        # Fallback to environment variables or defaults
        return {
            'log_level': os.environ.get('RAG_CLI_LOG_LEVEL', 'INFO'),
            'log_format': os.environ.get('RAG_CLI_LOG_FORMAT', 'json'),
            'log_file': os.environ.get('RAG_CLI_LOG_FILE', None),
            'log_rotation': {
                'max_bytes': int(os.environ.get('RAG_CLI_LOG_MAX_BYTES', '10485760')),
                'backup_count': int(os.environ.get('RAG_CLI_LOG_BACKUP_COUNT', '5'))
            }
        }

    def _get_log_file_path(self) -> Path:
        """Determine log file path.

        Priority: config > environment variable > project directory

        Returns:
            Path to log file
        """
        log_config = self._get_log_config()

        # Check config
        if log_config.get('log_file'):
            return Path(log_config['log_file'])

        # Check environment variable
        if os.environ.get('RAG_CLI_LOG_FILE'):
            return Path(os.environ['RAG_CLI_LOG_FILE'])

        # Default to project logs directory
        # Attempt to find project root (go up from utils -> src -> project root)
        project_root = Path(__file__).resolve().parents[2]
        logs_dir = project_root / 'logs'
        return logs_dir / 'rag-cli.log'

    def _setup_logging(self):
        """Set up logging configuration with platform-agnostic settings."""
        log_config = self._get_log_config()
        log_file = self._get_log_file_path()

        # Create logs directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Get log level
        log_level = log_config.get('log_level', 'INFO')
        level = getattr(logging, log_level.upper(), logging.INFO)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Check if console output should be suppressed
        is_hook_context = os.environ.get('CLAUDE_HOOK_CONTEXT') == '1'
        suppress_console = os.environ.get('RAG_CLI_SUPPRESS_CONSOLE') == '1'

        # Console handler (only add if not suppressed)
        if not is_hook_context and not suppress_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_formatter = TextFormatter()
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)

        # File handler with rotation (Windows-safe)
        log_format = log_config.get('log_format', 'json')
        rotation = log_config.get('log_rotation', {})
        max_bytes = rotation.get('max_bytes', 10485760)
        backup_count = rotation.get('backup_count', 5)

        if log_format == "json":
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

        # Use Windows-safe file handler factory
        file_handler = create_file_handler(
            log_file=log_file,
            max_bytes=max_bytes,
            backup_count=backup_count,
            level=level,
            formatter=file_formatter
        )
        root_logger.addHandler(file_handler)

        # Configure structlog
        structlog.configure(
            processors=[
                TimeStamper(fmt="iso"),
                add_log_level,
                JSONRenderer() if log_format == "json" else structlog.dev.ConsoleRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        self.logger = structlog.get_logger()
        self.standard_logger = logging.getLogger(__name__)

    def get_logger(self, name: Optional[str] = None) -> structlog.BoundLogger:
        """Get a logger instance.

        Args:
            name: Logger name, defaults to caller's module

        Returns:
            Configured logger instance
        """
        if name is None:
            import inspect
            frame = inspect.currentframe()
            caller_frame = frame.f_back
            name = caller_frame.f_globals.get('__name__', 'rag-cli')

        return structlog.get_logger(name)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message, **kwargs)

    def exception(self, message: str, exc_info=True, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(message, exc_info=exc_info, **kwargs)


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """Get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return Logger().get_logger(name)


def log_execution_time(func):
    """Decorator to log function execution time.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function
    """
    logger = get_logger(func.__module__)

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"{func.__name__} executed",
                         function=func.__name__,
                         elapsed_seconds=elapsed,
                         status="success")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func.__name__} failed",
                         function=func.__name__,
                         elapsed_seconds=elapsed,
                         status="error",
                         error=str(e))
            raise

    return wrapper


def log_api_call(service: str):
    """Decorator to log API calls.

    Args:
        service: Name of the service being called

    Returns:
        Decorator function
    """
    def decorator(func):
        logger = get_logger(func.__module__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            request_id = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')

            logger.info("API call started",
                        service=service,
                        function=func.__name__,
                        request_id=request_id)

            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info("API call completed",
                            service=service,
                            function=func.__name__,
                            request_id=request_id,
                            elapsed_seconds=elapsed,
                            status="success")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error("API call failed",
                             service=service,
                             function=func.__name__,
                             request_id=request_id,
                             elapsed_seconds=elapsed,
                             status="error",
                             error=str(e))
                raise

        return wrapper
    return decorator


class MetricsLogger:
    """Logger for metrics and performance tracking."""

    def __init__(self):
        """Initialize metrics logger."""
        self.logger = get_logger("metrics")
        self.metrics = {}

    def record_latency(self, operation: str, latency_ms: float):
        """Record operation latency.

        Args:
            operation: Name of the operation
            latency_ms: Latency in milliseconds
        """
        self.logger.info("latency_recorded",
                         operation=operation,
                         latency_ms=latency_ms,
                         metric_type="latency")

    def record_success(self, operation: str):
        """Record successful operation.

        Args:
            operation: Name of the operation
        """
        self.logger.info("operation_success",
                         operation=operation,
                         metric_type="success")

    def record_failure(self, operation: str, error: str):
        """Record failed operation.

        Args:
            operation: Name of the operation
            error: Error message
        """
        self.logger.error("operation_failure",
                          operation=operation,
                          error=error,
                          metric_type="failure")

    def record_count(self, metric: str, count: int):
        """Record a count metric.

        Args:
            metric: Metric name
            count: Count value
        """
        self.logger.info("count_recorded",
                         metric=metric,
                         count=count,
                         metric_type="count")

    def record_gauge(self, metric: str, value: float):
        """Record a gauge metric.

        Args:
            metric: Metric name
            value: Gauge value
        """
        self.logger.info("gauge_recorded",
                         metric=metric,
                         value=value,
                         metric_type="gauge")


# Create global instances
_logger = Logger()
_metrics_logger = MetricsLogger()


def get_metrics_logger() -> MetricsLogger:
    """Get global metrics logger instance.

    Returns:
        MetricsLogger instance
    """
    return _metrics_logger


# Convenience functions
def debug(message: str, **kwargs):
    """Log debug message."""
    _logger.debug(message, **kwargs)


def info(message: str, **kwargs):
    """Log info message."""
    _logger.info(message, **kwargs)


def warning(message: str, **kwargs):
    """Log warning message."""
    _logger.warning(message, **kwargs)


def error(message: str, **kwargs):
    """Log error message."""
    _logger.error(message, **kwargs)


def critical(message: str, **kwargs):
    """Log critical message."""
    _logger.critical(message, **kwargs)


def exception(message: str, **kwargs):
    """Log exception with traceback."""
    _logger.exception(message, **kwargs)


if __name__ == "__main__":
    # Test logging
    logger = get_logger("test")

    logger.debug("Debug message", extra_field="test")
    logger.info("Info message", user="admin", action="login")
    logger.warning("Warning message", threshold=0.8)
    logger.error("Error message", error_code=500)

    # Test metrics
    metrics = get_metrics_logger()
    metrics.record_latency("vector_search", 45.2)
    metrics.record_success("document_indexing")
    metrics.record_failure("api_call", "Rate limit exceeded")
    metrics.record_count("documents_processed", 150)
    metrics.record_gauge("memory_usage_mb", 512.3)

    print("Logging test completed. Check logs/rag-cli.log")
