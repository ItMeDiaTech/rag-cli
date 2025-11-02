"""Percentile latency tracking for performance monitoring.

This module provides detailed latency tracking with percentile calculations
(p50, p95, p99) for different operations in the RAG pipeline.
Platform-agnostic version for use in the core rag_cli library.
"""

import time
import threading
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
import statistics

from rag_cli.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LatencyStats:
    """Latency statistics for an operation."""
    operation: str
    count: int
    p50: float  # Median
    p75: float
    p90: float
    p95: float
    p99: float
    min: float
    max: float
    mean: float
    std_dev: float
    total: float
    timestamp: datetime


class LatencyTracker:
    """Tracks latency percentiles for different operations."""

    def __init__(self, window_size: int = 1000, cleanup_interval: int = 300):
        """Initialize latency tracker.

        Args:
            window_size: Number of recent measurements to keep per operation
            cleanup_interval: Seconds between cleanup of old data
        """
        self.window_size = window_size
        self.cleanup_interval = cleanup_interval

        # Store latencies per operation (rolling window)
        self.latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))

        # Total counts and sums for all-time stats
        self.total_counts: Dict[str, int] = defaultdict(int)
        self.total_sums: Dict[str, float] = defaultdict(float)

        # Thread safety
        self.lock = threading.Lock()

        # Last cleanup time
        self.last_cleanup = time.time()

        logger.info("Latency tracker initialized",
                    window_size=window_size,
                    cleanup_interval=cleanup_interval)

    def record(self, operation: str, latency_ms: float):
        """Record a latency measurement.

        Args:
            operation: Name of the operation
            latency_ms: Latency in milliseconds
        """
        with self.lock:
            self.latencies[operation].append(latency_ms)
            self.total_counts[operation] += 1
            self.total_sums[operation] += latency_ms

            # Periodic cleanup
            if time.time() - self.last_cleanup > self.cleanup_interval:
                self._cleanup()
                self.last_cleanup = time.time()

    def _cleanup(self):
        """Clean up old data (already handled by deque maxlen)."""
        # With deque(maxlen=window_size), old entries are automatically removed
        # This method is kept for future enhancements
        pass

    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile from sorted values.

        Args:
            values: Sorted list of values
            percentile: Percentile to calculate (0-100)

        Returns:
            Percentile value
        """
        if not values:
            return 0.0

        k = (len(values) - 1) * (percentile / 100)
        f = int(k)
        c = f + 1 if f < len(values) - 1 else f

        if f == c:
            return values[f]

        # Linear interpolation
        return values[f] + (k - f) * (values[c] - values[f])

    def get_stats(self, operation: str) -> Optional[LatencyStats]:
        """Get latency statistics for an operation.

        Args:
            operation: Name of the operation

        Returns:
            Latency statistics or None if no data
        """
        with self.lock:
            if operation not in self.latencies or len(self.latencies[operation]) == 0:
                return None

            values = list(self.latencies[operation])
            sorted_values = sorted(values)

            # Calculate percentiles
            p50 = self._calculate_percentile(sorted_values, 50)
            p75 = self._calculate_percentile(sorted_values, 75)
            p90 = self._calculate_percentile(sorted_values, 90)
            p95 = self._calculate_percentile(sorted_values, 95)
            p99 = self._calculate_percentile(sorted_values, 99)

            # Calculate basic stats
            min_val = min(sorted_values)
            max_val = max(sorted_values)
            mean_val = statistics.mean(values)
            std_dev = statistics.stdev(values) if len(values) > 1 else 0.0

            return LatencyStats(
                operation=operation,
                count=len(values),
                p50=p50,
                p75=p75,
                p90=p90,
                p95=p95,
                p99=p99,
                min=min_val,
                max=max_val,
                mean=mean_val,
                std_dev=std_dev,
                total=self.total_sums[operation],
                timestamp=datetime.now()
            )

    def get_all_stats(self) -> Dict[str, LatencyStats]:
        """Get latency statistics for all operations.

        Returns:
            Dictionary of operation name to statistics
        """
        with self.lock:
            stats = {}
            for operation in self.latencies.keys():
                stat = self.get_stats(operation)
                if stat:
                    stats[operation] = stat
            return stats

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all latency statistics.

        Returns:
            Summary dictionary
        """
        all_stats = self.get_all_stats()

        summary = {
            "timestamp": datetime.now().isoformat(),
            "operations": {},
            "totals": {
                "operations_tracked": len(all_stats),
                "total_measurements": sum(self.total_counts.values()),
            }
        }

        for operation, stats in all_stats.items():
            summary["operations"][operation] = {
                "count": stats.count,
                "p50_ms": round(stats.p50, 2),
                "p95_ms": round(stats.p95, 2),
                "p99_ms": round(stats.p99, 2),
                "mean_ms": round(stats.mean, 2),
                "min_ms": round(stats.min, 2),
                "max_ms": round(stats.max, 2),
            }

        return summary

    def format_stats(self, operation: str) -> str:
        """Format statistics for display.

        Args:
            operation: Name of the operation

        Returns:
            Formatted string
        """
        stats = self.get_stats(operation)
        if not stats:
            return f"No latency data for operation: {operation}"

        return f"""{stats.operation} Latency Statistics:
  Count:  {stats.count}
  p50:    {stats.p50:.2f}ms (median)
  p75:    {stats.p75:.2f}ms
  p90:    {stats.p90:.2f}ms
  p95:    {stats.p95:.2f}ms
  p99:    {stats.p99:.2f}ms
  Min:    {stats.min:.2f}ms
  Max:    {stats.max:.2f}ms
  Mean:   {stats.mean:.2f}ms
  StdDev: {stats.std_dev:.2f}ms"""

    def reset(self, operation: Optional[str] = None):
        """Reset statistics for an operation or all operations.

        Args:
            operation: Specific operation to reset, or None for all
        """
        with self.lock:
            if operation:
                if operation in self.latencies:
                    self.latencies[operation].clear()
                    self.total_counts[operation] = 0
                    self.total_sums[operation] = 0.0
                    logger.info(f"Reset latency stats for {operation}")
            else:
                self.latencies.clear()
                self.total_counts.clear()
                self.total_sums.clear()
                logger.info("Reset all latency stats")


class LatencyTimer:
    """Context manager for measuring and recording latencies."""

    def __init__(self, tracker: LatencyTracker, operation: str):
        """Initialize latency timer.

        Args:
            tracker: Latency tracker instance
            operation: Name of the operation being timed
        """
        self.tracker = tracker
        self.operation = operation
        self.start_time = None
        self.latency_ms = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record latency."""
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.latency_ms = elapsed * 1000  # Convert to ms
            self.tracker.record(self.operation, self.latency_ms)

        # Don't suppress exceptions
        return False

    def get_latency_ms(self) -> Optional[float]:
        """Get measured latency in milliseconds.

        Returns:
            Latency in ms or None if not measured yet
        """
        return self.latency_ms


# Global tracker instance
_latency_tracker: Optional[LatencyTracker] = None


def get_latency_tracker() -> LatencyTracker:
    """Get or create the global latency tracker.

    Returns:
        Latency tracker instance
    """
    global _latency_tracker
    if _latency_tracker is None:
        _latency_tracker = LatencyTracker()
    return _latency_tracker


def record_latency(operation: str, latency_ms: float):
    """Convenience function to record latency.

    Args:
        operation: Name of the operation
        latency_ms: Latency in milliseconds
    """
    tracker = get_latency_tracker()
    tracker.record(operation, latency_ms)


def time_operation(operation: str) -> LatencyTimer:
    """Convenience function to create a latency timer.

    Args:
        operation: Name of the operation

    Returns:
        Latency timer context manager

    Example:
        with time_operation("vector_search"):
            # do search
            pass
    """
    tracker = get_latency_tracker()
    return LatencyTimer(tracker, operation)


def get_latency_stats(operation: str) -> Optional[LatencyStats]:
    """Convenience function to get latency stats.

    Args:
        operation: Name of the operation

    Returns:
        Latency statistics
    """
    tracker = get_latency_tracker()
    return tracker.get_stats(operation)


def get_all_latency_stats() -> Dict[str, LatencyStats]:
    """Convenience function to get all latency stats.

    Returns:
        Dictionary of all statistics
    """
    tracker = get_latency_tracker()
    return tracker.get_all_stats()
