"""
RAG-CLI Utilities

Shared utilities for configuration, logging, and path management.
"""

from rag_cli.utils.logger import (
    get_logger,
    get_metrics_logger,
    log_execution_time,
    log_api_call,
    Logger,
    MetricsLogger,
    JSONFormatter,
    TextFormatter,
    debug,
    info,
    warning,
    error,
    critical,
    exception
)

from rag_cli.utils.latency_tracker import (
    get_latency_tracker,
    record_latency,
    time_operation,
    get_latency_stats,
    get_all_latency_stats,
    LatencyTracker,
    LatencyTimer,
    LatencyStats
)

from rag_cli.utils.error_tracker import (
    get_error_tracker,
    ErrorTracker,
    ErrorOccurrence
)

__all__ = [
    # Logger
    'get_logger',
    'get_metrics_logger',
    'log_execution_time',
    'log_api_call',
    'Logger',
    'MetricsLogger',
    'JSONFormatter',
    'TextFormatter',
    'debug',
    'info',
    'warning',
    'error',
    'critical',
    'exception',
    # Latency Tracker
    'get_latency_tracker',
    'record_latency',
    'time_operation',
    'get_latency_stats',
    'get_all_latency_stats',
    'LatencyTracker',
    'LatencyTimer',
    'LatencyStats',
    # Error Tracker
    'get_error_tracker',
    'ErrorTracker',
    'ErrorOccurrence'
]
