"""Structured logging infrastructure for RAG-CLI.

This module provides comprehensive logging with JSON format support,
file rotation, and different log levels for debugging and monitoring.
"""

import os
import sys
import json
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional
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


class MetricsCollectorHandler(logging.Handler):
    """Custom logging handler that forwards logs to metrics collector for dashboard display.

    This handler enables real-time log streaming to the web dashboard by capturing
    log records and sending them via HTTP to the TCP server for cross-process support.
    """

    TCP_SERVER_URL = "http://localhost:9999"

    def emit(self, record: logging.LogRecord):
        """Process a log record and send it to TCP server.

        Args:
            record: Log record to process
        """
        try:
            import urllib.request
            import urllib.error

            # Format the message
            message = self.format(record)

            # Send log as an event to TCP server for cross-process support
            log_event = {
                "event_type": "log",
                "data": {
                    "level": record.levelname,
                    "message": message
                }
            }

            req = urllib.request.Request(
                f"{self.TCP_SERVER_URL}/api/events/submit",
                data=json.dumps(log_event).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=1):
                pass  # Successfully sent

        except Exception:
            # Silently fail - don't break logging if TCP server unavailable
            # Only log in debug mode to avoid log spam
            if __debug__:
                pass  # Intentionally silent: TCP server may not be running


class Logger:
    """Main logger class for RAG-CLI."""

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

    def _setup_logging(self):
        """Set up logging configuration."""
        from src.core.config import get_config

        # Determine log file path
        # Priority: plugin directory > project directory
        try:
            # Use expanduser to properly handle ~ on all platforms
            claude_plugin_dir = Path.home() / '.claude' / 'plugins' / 'rag-cli'

            # Verify the path is valid and accessible on Windows
            # Check if we're in a valid RAG-CLI installation
            is_valid_plugin_dir = (
                claude_plugin_dir.exists() and
                (claude_plugin_dir / 'src' / 'core').exists()
            )
        except Exception:
            is_valid_plugin_dir = False
            claude_plugin_dir = None

        project_root = Path(__file__).resolve().parents[2]

        # Use plugin directory for logs if it exists and is valid, otherwise use project directory
        if is_valid_plugin_dir:
            logs_dir = claude_plugin_dir / 'logs'
        else:
            logs_dir = project_root / 'logs'

        log_file = logs_dir / 'rag-cli.log'

        # Get configuration
        try:
            config = get_config()
            log_config = config.monitoring
        except Exception as e:
            # Fallback to defaults if config not available
            print(f"Warning: Could not load config, using defaults: {e}")
            log_config = {
                "log_level": "INFO",
                "log_format": "json",
                "log_file": str(log_file),
                "log_rotation": {"max_bytes": 10485760, "backup_count": 5}
            }

        # Override log file path with plugin directory
        if hasattr(log_config, 'log_file'):
            # Use the resolved log file path
            log_config.log_file = str(log_file)
        elif isinstance(log_config, dict):
            log_config['log_file'] = str(log_file)

        # Create logs directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Get log level
        if hasattr(log_config, 'log_level'):
            log_level = log_config.log_level
        else:
            log_level = log_config.get('log_level', 'INFO')
        level = getattr(logging, log_level.upper(), logging.INFO)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Check if we're running in a Claude Code hook context
        # Hooks should NOT output logs to console (only to file)
        is_hook_context = os.environ.get('CLAUDE_HOOK_CONTEXT') == '1'
        suppress_console = os.environ.get('RAG_CLI_SUPPRESS_CONSOLE') == '1'

        # Console handler (only add if not in hook context)
        if not is_hook_context and not suppress_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_formatter = TextFormatter()
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)

        # File handler with rotation
        if hasattr(log_config, 'log_format'):
            log_format = log_config.log_format
        else:
            log_format = log_config.get('log_format', 'json')

        if hasattr(log_config, 'log_rotation'):
            rotation = log_config.log_rotation
        else:
            rotation = log_config.get('log_rotation', {})

        if isinstance(rotation, dict):
            max_bytes = rotation.get('max_bytes', 10485760)
            backup_count = rotation.get('backup_count', 5)
        else:
            max_bytes = getattr(rotation, 'max_bytes', 10485760)
            backup_count = getattr(rotation, 'backup_count', 5)

        file_handler = logging.handlers.RotatingFileHandler(
            str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(level)

        if log_format == "json":
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        # Add metrics collector handler for dashboard log streaming
        try:
            metrics_handler = MetricsCollectorHandler()
            metrics_handler.setLevel(level)
            metrics_handler.setFormatter(file_formatter)
            root_logger.addHandler(metrics_handler)
        except Exception:
            # Silently fail if metrics collector not available
            pass

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
