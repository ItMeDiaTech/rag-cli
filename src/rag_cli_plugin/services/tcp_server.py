"""TCP monitoring server for RAG-CLI.

This module provides a TCP server that exposes monitoring endpoints
for PowerShell and other clients to query system status and metrics.
"""

import os
import json
import time
import threading
import socket
import signal
import weakref
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import deque
from queue import Queue, Empty, Full

from flask import Flask, jsonify, request
import psutil

from rag_cli.core.config import get_config
from rag_cli_plugin.services.logger import get_logger, get_metrics_logger


logger = get_logger(__name__)
metrics_logger = get_metrics_logger()

# Global shutdown flag for graceful termination
_shutdown_event = threading.Event()

# Global metrics storage


class MetricsCollector:
    """Collects and stores system metrics."""

    def __init__(self, max_history: int = 1000):
        """Initialize metrics collector.

        Args:
            max_history: Maximum number of historical entries to keep
        """
        self.max_history = max_history
        self.start_time = time.time()

        # Metrics storage
        self.latency_metrics = deque(maxlen=max_history)
        self.throughput_metrics = deque(maxlen=max_history)
        self.resource_metrics = deque(maxlen=max_history)
        self.query_count = 0
        self.error_count = 0
        self.cache_hits = 0
        self.cache_misses = 0

        # Log buffer
        self.log_buffer = deque(maxlen=100)

        # Component status
        self.component_status = {
            "vector_store": "unknown",
            "embeddings": "unknown",
            "retriever": "unknown",
            "claude": "unknown"
        }

        # Event streaming support (SSE) with weak references to prevent memory leaks
        self.event_subscribers = []  # List of weak references to queues for SSE clients
        self.event_history = deque(maxlen=100)  # Recent events for new subscribers

        # New event categories
        self.activity_events = deque(maxlen=100)
        self.reasoning_events = deque(maxlen=100)
        self.query_enhancement_events = deque(maxlen=100)

    def subscribe_to_events(self) -> Queue:
        """Subscribe to real-time events. Returns a queue for SSE streaming.

        Uses weak references to automatically clean up dead subscribers.

        Returns:
            Queue that will receive events
        """
        queue = Queue(maxsize=100)
        # Store weak reference to prevent memory leaks
        self.event_subscribers.append(weakref.ref(queue))

        # Send recent event history to new subscriber
        for event in list(self.event_history)[-20:]:
            try:
                queue.put_nowait(event)
            except Full:
                pass

        return queue

    def unsubscribe_from_events(self, queue: Queue):
        """Unsubscribe from events.

        Args:
            queue: Queue to remove from subscribers
        """
        # Find and remove weak reference to this queue
        self.event_subscribers = [ref for ref in self.event_subscribers
                                  if ref() is not None and ref() != queue]

    def emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event to all subscribers.

        Args:
            event_type: Type of event (activity, reasoning, metric, log)
            data: Event data
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }

        # Store in history
        self.event_history.append(event)

        # Store in appropriate category
        if event_type.startswith("activity"):
            self.activity_events.append(event)
        elif event_type.startswith("reasoning"):
            self.reasoning_events.append(event)
        elif event_type.startswith("query_enhancement"):
            self.query_enhancement_events.append(event)
        elif event_type == "log":
            # Add log events to log buffer for /api/logs endpoint
            log_entry = {
                "timestamp": event["timestamp"],
                "level": data.get("level", "INFO"),
                "message": data.get("message", "")
            }
            self.log_buffer.append(log_entry)

        # Send to all subscribers and clean up dead ones
        active_subscribers = []
        for queue_ref in self.event_subscribers:
            queue = queue_ref()  # Dereference weak reference
            if queue is not None:
                try:
                    queue.put_nowait(event)
                    active_subscribers.append(queue_ref)  # Keep alive subscribers
                except (Full, AttributeError):
                    # Queue is full or broken, don't add to active list
                    pass
            # If queue is None, weak reference is dead - don't add to active list

        # Update subscriber list with only active ones (automatic cleanup)
        self.event_subscribers = active_subscribers

    def record_activity_event(self, activity: str, component: str, metadata: Optional[Dict] = None):
        """Record a plugin activity event.

        Args:
            activity: Activity description (e.g., 'query_received', 'documents_retrieved')
            component: Component name (e.g., 'user_prompt_hook', 'retrieval_pipeline')
            metadata: Optional additional metadata
        """
        self.emit_event("activity", {
            "activity": activity,
            "component": component,
            "metadata": metadata or {}
        })

    def record_reasoning_event(self, reasoning: str, component: str, context: Optional[Dict] = None):
        """Record a reasoning/decision event.

        Args:
            reasoning: Explanation of decision/reasoning
            component: Component that made the decision
            context: Optional context data
        """
        self.emit_event("reasoning", {
            "reasoning": reasoning,
            "component": component,
            "context": context or {}
        })

    def record_query_enhancement(self, original_query: str, enhanced_query: str,
                                 documents: List[Dict], reasoning: str):
        """Record query enhancement details.

        Args:
            original_query: Original user query
            enhanced_query: Enhanced query with context
            documents: Retrieved documents
            reasoning: Explanation of enhancement strategy
        """
        self.emit_event("query_enhancement", {
            "original_query": original_query,
            "enhanced_query": enhanced_query,
            "documents_count": len(documents),
            "documents": documents[:3],  # First 3 docs for preview
            "reasoning": reasoning
        })

    def record_latency(self, operation: str, latency_ms: float):
        """Record latency metric."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "latency_ms": latency_ms
        }
        self.latency_metrics.append(entry)

        # Emit event for real-time monitoring
        self.emit_event("metric", {
            "metric_type": "latency",
            "operation": operation,
            "value": latency_ms
        })

    def record_query(self):
        """Record a query."""
        self.query_count += 1

    def record_error(self):
        """Record an error."""
        self.error_count += 1

    def record_cache(self, hit: bool):
        """Record cache hit or miss."""
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def add_log(self, level: str, message: str):
        """Add a log entry to the buffer."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        }
        self.log_buffer.append(entry)

        # Emit event for real-time log streaming
        self.emit_event("log", {
            "level": level,
            "message": message
        })

    def get_uptime(self) -> float:
        """Get server uptime in seconds."""
        return time.time() - self.start_time

    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate percentage."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return (self.cache_hits / total) * 100

    def get_latest_latencies(self) -> Dict[str, float]:
        """Get average latencies from recent operations."""
        latencies = {
            "vector_search": [],
            "keyword_search": [],
            "reranking": [],
            "claude_api": [],
            "end_to_end": []
        }

        # Collect recent latencies
        for entry in list(self.latency_metrics)[-100:]:  # Last 100 entries
            op = entry["operation"]
            if op in latencies:
                latencies[op].append(entry["latency_ms"])

        # Calculate averages
        avg_latencies = {}
        for op, values in latencies.items():
            if values:
                avg_latencies[op] = sum(values) / len(values)
            else:
                avg_latencies[op] = 0.0

        return avg_latencies

    def get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        process = psutil.Process(os.getpid())

        return {
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "threads": process.num_threads()
        }

    def update_component_status(self, component: str, status: str):
        """Update component status."""
        if component in self.component_status:
            self.component_status[component] = status


# Global metrics collector
metrics_collector = MetricsCollector()


class MonitoringServer:
    """TCP monitoring server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9999):
        """Initialize monitoring server.

        Args:
            host: Host to bind to
            port: Port to bind to
        """
        self.host = host
        self.port = port
        self.running = False
        self.server_socket = None
        self.thread = None

        logger.info("Monitoring server initialized", host=host, port=port)

    def start(self):
        """Start the monitoring server."""
        if self.running:
            logger.warning("Monitoring server already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()

        logger.info(f"Monitoring server started on {self.host}:{self.port}")

    def stop(self):
        """Stop the monitoring server."""
        self.running = False

        if self.server_socket:
            self.server_socket.close()

        if self.thread:
            self.thread.join(timeout=5)

        logger.info("Monitoring server stopped")

    def _run_server(self):
        """Run the TCP server loop."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.server_socket.settimeout(1.0)  # Timeout for checking running flag

            logger.info(f"Monitoring server listening on {self.host}:{self.port}")

            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    logger.debug(f"Client connected from {address}")

                    # Handle client in a separate thread
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, address),
                        daemon=True
                    )
                    client_thread.start()

                except socket.timeout:
                    continue  # Check running flag
                except Exception as e:
                    if self.running:
                        logger.error(f"Server error: {e}")

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
        finally:
            if self.server_socket:
                self.server_socket.close()

    def _handle_client(self, client_socket: socket.socket, address: tuple):
        """Handle a client connection.

        Args:
            client_socket: Client socket
            address: Client address
        """
        try:
            # Receive request
            request = client_socket.recv(1024).decode().strip()
            logger.debug(f"Request from {address}: {request}")

            # Process request
            response = self._process_request(request)

            # Send response
            client_socket.sendall(response.encode())

        except Exception as e:
            logger.error(f"Error handling client {address}: {e}")
            error_response = json.dumps({"error": str(e)})
            try:
                client_socket.sendall(error_response.encode())
            except (socket.error, BrokenPipeError, ConnectionResetError) as send_error:
                logger.debug(f"Failed to send error response to {address}: {send_error}")
        finally:
            client_socket.close()

    def _process_request(self, request: str) -> str:
        """Process a monitoring request.

        Args:
            request: Request command

        Returns:
            JSON response
        """
        command = request.upper()

        if command == "STATUS":
            return self._get_status()
        elif command == "LOGS":
            return self._get_logs()
        elif command == "METRICS":
            return self._get_metrics()
        elif command == "HEALTH":
            return self._get_health()
        else:
            return json.dumps({"error": f"Unknown command: {command}"})

    def _get_status(self) -> str:
        """Get system status."""
        from core.vector_store import get_vector_store
        from datetime import datetime

        try:
            vector_store = get_vector_store()
            total_vectors = vector_store.index.ntotal
        except (AttributeError, RuntimeError, Exception) as e:
            logger.debug(f"Could not retrieve vector count: {e}")
            total_vectors = 0

        uptime_seconds = metrics_collector.get_uptime()

        # Format uptime as human-readable string
        hours = int(uptime_seconds // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        seconds = int(uptime_seconds % 60)

        if hours > 0:
            uptime_formatted = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            uptime_formatted = f"{minutes}m {seconds}s"
        else:
            uptime_formatted = f"{seconds}s"

        status = {
            "version": "0.1.0",
            "uptime": {
                "seconds": uptime_seconds,
                "formatted": uptime_formatted
            },
            "status": "operational",
            "last_updated": datetime.now().isoformat(),
            "components": metrics_collector.component_status,
            "statistics": {
                "total_documents": 0,  # Would need to track this
                "total_vectors": total_vectors,
                "total_queries": metrics_collector.query_count,
                "total_errors": metrics_collector.error_count,
                "cache_hit_rate": metrics_collector.get_cache_hit_rate()
            }
        }

        return json.dumps(status, indent=2)

    def _get_logs(self) -> list:
        """Get recent logs as list of dictionaries for dashboard display.

        Returns:
            List of log entries with timestamp, level, and message fields
        """
        # Return last 20 logs from buffer in reverse order (newest first)
        return list(metrics_collector.log_buffer)[-20:]

    def _get_metrics(self) -> str:
        """Get performance metrics."""
        # Calculate cache metrics
        cache_hits = metrics_collector.cache_hits
        cache_misses = metrics_collector.cache_misses
        cache_total = cache_hits + cache_misses
        cache_hit_rate = metrics_collector.get_cache_hit_rate()

        metrics = {
            "latency": metrics_collector.get_latest_latencies(),
            "throughput": {
                "queries_per_minute": metrics_collector.query_count * 60 / max(1, metrics_collector.get_uptime()),
                "docs_per_minute": 0  # Would need to track this
            },
            "cache": {
                "hits": cache_hits,
                "misses": cache_misses,
                "total": cache_total,
                "hit_rate": cache_hit_rate
            },
            "cache_hit_rate": cache_hit_rate,  # Keep for backward compatibility
            "resources": metrics_collector.get_resource_usage()
        }

        return json.dumps(metrics, indent=2)

    def _get_health(self) -> str:
        """Get health check status."""
        issues = []

        # Check components
        for component, status in metrics_collector.component_status.items():
            if status not in ["operational", "ready", "healthy"]:
                issues.append(f"{component} is {status}")

        # Check error rate
        if metrics_collector.error_count > 0:
            error_rate = metrics_collector.error_count / max(1, metrics_collector.query_count)
            if error_rate > 0.1:  # More than 10% errors
                issues.append(f"High error rate: {error_rate:.1%}")

        # Check memory usage
        resources = metrics_collector.get_resource_usage()
        if resources["memory_mb"] > 2048:  # More than 2GB
            issues.append(f"High memory usage: {resources['memory_mb']:.0f} MB")

        if issues:
            return json.dumps({
                "status": "unhealthy",
                "issues": issues
            }, indent=2)
        else:
            return json.dumps({"status": "healthy"}, indent=2)


# Flask app for HTTP monitoring (optional alternative to TCP)
app = Flask(__name__)


@app.route('/api/status')
def flask_status():
    """Flask endpoint for status."""
    server = get_monitoring_server()
    return jsonify(json.loads(server._get_status()))


@app.route('/api/metrics')
def flask_metrics():
    """Flask endpoint for metrics."""
    server = get_monitoring_server()
    return jsonify(json.loads(server._get_metrics()))


@app.route('/api/health')
def flask_health():
    """Flask endpoint for health."""
    server = get_monitoring_server()
    return jsonify(json.loads(server._get_health()))


@app.route('/api/logs')
def flask_logs():
    """Flask endpoint for logs."""
    server = get_monitoring_server()
    logs = server._get_logs()
    return jsonify({"logs": logs})


@app.route('/api/events/submit', methods=['POST'])
def submit_event():
    """Accept event submissions from external processes (hooks, etc.)."""
    try:
        # SECURITY: Limit JSON size to prevent DoS
        content_length = request.content_length
        if content_length and content_length > 10240:  # 10KB limit
            return jsonify({"error": "Request too large (max 10KB)"}), 413

        event_data = request.get_json(force=False, cache=False)

        if not event_data:
            return jsonify({"error": "No event data provided"}), 400

        # SECURITY: Validate event_type length and characters
        event_type = event_data.get('event_type')
        if not event_type:
            return jsonify({"error": "event_type is required"}), 400

        # Ensure event_type is a string and reasonable length
        if not isinstance(event_type, str) or len(event_type) > 100:
            return jsonify({"error": "Invalid event_type"}), 400

        # SECURITY: Validate data is a dictionary and limit properties
        data = event_data.get('data', {})
        if not isinstance(data, dict):
            return jsonify({"error": "data must be a dictionary"}), 400

        # Limit number of properties to prevent memory exhaustion
        if len(data) > 50:
            return jsonify({"error": "Too many data properties (max 50)"}), 400

        # Emit the event to all subscribers
        metrics_collector.emit_event(event_type, data)

        logger.debug("Event submitted via HTTP", event_type=event_type)

        return jsonify({
            "status": "success",
            "event_type": event_type,
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error submitting event: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/events/stream')
def stream_events():
    """Server-Sent Events endpoint for real-time event streaming."""
    from flask import Response, stream_with_context

    def generate():
        # Subscribe to events
        queue = metrics_collector.subscribe_to_events()

        try:
            # Send initial connection confirmation
            yield f"data: {json.dumps({'event_type': 'connected', 'message': 'SSE stream connected'})}\n\n"

            # Add loop exit condition for graceful shutdown
            while not _shutdown_event.is_set():
                try:
                    # Wait for events with timeout for keepalive
                    event = queue.get(timeout=15)
                    yield f"data: {json.dumps(event)}\n\n"

                except Empty:
                    # Timeout - send keepalive
                    yield ": keepalive\n\n"
                    # Check for shutdown during keepalive
                    if _shutdown_event.is_set():
                        yield f"data: {json.dumps({'event_type': 'shutdown', 'message': 'Server shutting down'})}\n\n"
                        break

        except GeneratorExit:
            # Client disconnected
            metrics_collector.unsubscribe_from_events(queue)
            logger.debug("SSE client disconnected")

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )


@app.route('/api/events/history')
def get_event_history():
    """Get recent event history."""
    # SECURITY: Validate category against allowed values
    allowed_categories = ['all', 'activity', 'reasoning', 'query_enhancement']
    category = request.args.get('category', 'all')
    if category not in allowed_categories:
        return jsonify({"error": "Invalid category parameter"}), 400

    # SECURITY: Validate and bound limit parameter to prevent DoS
    try:
        limit = int(request.args.get('limit', 50))
        # Bound limit between 1 and 1000 to prevent memory exhaustion
        limit = max(1, min(limit, 1000))
    except (ValueError, TypeError):
        limit = 50

    if category == 'activity':
        events = list(metrics_collector.activity_events)[-limit:]
    elif category == 'reasoning':
        events = list(metrics_collector.reasoning_events)[-limit:]
    elif category == 'query_enhancement':
        events = list(metrics_collector.query_enhancement_events)[-limit:]
    else:
        events = list(metrics_collector.event_history)[-limit:]

    return jsonify({
        "category": category,
        "count": len(events),
        "events": events
    })


@app.route('/api/latency')
def get_latency_stats():
    """Get latency statistics with percentiles."""
    try:
        from monitoring.latency_tracker import get_latency_tracker

        tracker = get_latency_tracker()
        summary = tracker.get_summary()

        return jsonify(summary)

    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Failed to get latency statistics"
        }), 500


@app.route('/api/latency/<operation>')
def get_operation_latency(operation: str):
    """Get latency statistics for a specific operation."""
    # SECURITY: Validate operation parameter
    if not operation or len(operation) > 100:
        return jsonify({"error": "Invalid operation parameter"}), 400

    # Sanitize operation string - only allow alphanumeric, underscore, dash
    import re
    if not re.match(r'^[a-zA-Z0-9_\-]+$', operation):
        return jsonify({"error": "Invalid operation format"}), 400

    try:
        from monitoring.latency_tracker import get_latency_tracker

        tracker = get_latency_tracker()
        stats = tracker.get_stats(operation)

        if stats:
            return jsonify({
                "operation": stats.operation,
                "count": stats.count,
                "percentiles": {
                    "p50": round(stats.p50, 2),
                    "p75": round(stats.p75, 2),
                    "p90": round(stats.p90, 2),
                    "p95": round(stats.p95, 2),
                    "p99": round(stats.p99, 2)
                },
                "stats": {
                    "min": round(stats.min, 2),
                    "max": round(stats.max, 2),
                    "mean": round(stats.mean, 2),
                    "std_dev": round(stats.std_dev, 2)
                },
                "timestamp": stats.timestamp.isoformat()
            })
        else:
            return jsonify({
                "error": "Not found",
                "message": f"No latency data for operation: {operation}"
            }), 404

    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Failed to get operation latency"
        }), 500


# Singleton server instance
_monitoring_server: Optional[MonitoringServer] = None


def get_monitoring_server(host: str = "127.0.0.1", port: int = 9999) -> MonitoringServer:
    """Get or create monitoring server.

    Args:
        host: Host to bind to
        port: Port to bind to

    Returns:
        Monitoring server instance
    """
    global _monitoring_server

    if _monitoring_server is None:
        _monitoring_server = MonitoringServer(host, port)

    return _monitoring_server


def start_monitoring_server():
    """Start the monitoring server with environment variable support."""
    config = get_config()

    if config.monitoring.tcp_server.get("enabled", True):
        host = config.monitoring.tcp_server.get("host", "127.0.0.1")
        # Allow port override from environment variable
        port = int(os.environ.get("RAG_TCP_PORT", config.monitoring.tcp_server.get("port", 9999)))

        server = get_monitoring_server(host, port)
        server.start()

        logger.info(f"Monitoring server started on {host}:{port}")
        return server
    else:
        logger.info("Monitoring server disabled in configuration")
        return None


def shutdown_handler(signum=None, frame=None):
    """Handle shutdown signals gracefully."""
    logger.info("Shutdown signal received")
    _shutdown_event.set()
    # Give threads time to clean up
    time.sleep(0.5)


if __name__ == "__main__":
    import sys

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Use Flask development server for SSE streaming support
    # NOTE: Waitress and other production WSGI servers buffer responses
    # and cannot stream Server-Sent Events (SSE) in real-time
    print("Starting RAG-CLI Monitoring Server (HTTP - SSE Streaming Mode)...")
    print("Server running on http://localhost:9999")
    print("Endpoints: /api/status, /api/metrics, /api/health, /api/logs, /api/events/stream")
    print("Using Flask development server (required for SSE streaming)...")
    print("Press Ctrl+C to stop...")

    try:
        # Use Flask dev server with threading for SSE support
        # SECURITY: Bind to localhost only to prevent network access
        app.run(host="127.0.0.1", port=9999, debug=False, threaded=True)
    except KeyboardInterrupt:
        shutdown_handler()
        print("\nShutting down gracefully...")
        sys.exit(0)
