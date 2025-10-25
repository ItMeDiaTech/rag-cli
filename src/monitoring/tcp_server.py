"""TCP monitoring server for RAG-CLI.

This module provides a TCP server that exposes monitoring endpoints
for PowerShell and other clients to query system status and metrics.
"""

import os
import json
import time
import threading
import socket
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import deque
from io import StringIO

from flask import Flask, jsonify, request
import psutil

from src.core.config import get_config
from src.monitoring.logger import get_logger, get_metrics_logger


logger = get_logger(__name__)
metrics_logger = get_metrics_logger()

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

    def record_latency(self, operation: str, latency_ms: float):
        """Record latency metric."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "latency_ms": latency_ms
        }
        self.latency_metrics.append(entry)

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

        logger.info(f"Monitoring server initialized", host=host, port=port)

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
            except:
                pass
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
        from src.core.vector_store import get_vector_store

        try:
            vector_store = get_vector_store()
            total_vectors = vector_store.index.ntotal
        except:
            total_vectors = 0

        status = {
            "version": "0.1.0",
            "uptime": f"{metrics_collector.get_uptime():.0f} seconds",
            "status": "operational",
            "components": metrics_collector.component_status,
            "statistics": {
                "total_documents": 0,  # Would need to track this
                "total_vectors": total_vectors,
                "total_queries": metrics_collector.query_count,
                "total_errors": metrics_collector.error_count,
                "cache_hit_rate": f"{metrics_collector.get_cache_hit_rate():.1f}"
            }
        }

        return json.dumps(status, indent=2)

    def _get_logs(self) -> str:
        """Get recent logs."""
        logs = []

        for entry in list(metrics_collector.log_buffer)[-20:]:  # Last 20 logs
            log_line = f"[{entry['timestamp']}] {entry['level']:8} | {entry['message']}"
            logs.append(log_line)

        return "\n".join(logs)

    def _get_metrics(self) -> str:
        """Get performance metrics."""
        metrics = {
            "latency": metrics_collector.get_latest_latencies(),
            "throughput": {
                "queries_per_minute": metrics_collector.query_count * 60 / max(1, metrics_collector.get_uptime()),
                "docs_per_minute": 0  # Would need to track this
            },
            "cache_hit_rate": metrics_collector.get_cache_hit_rate(),
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


@app.route('/status')
def flask_status():
    """Flask endpoint for status."""
    server = get_monitoring_server()
    return jsonify(json.loads(server._get_status()))


@app.route('/metrics')
def flask_metrics():
    """Flask endpoint for metrics."""
    server = get_monitoring_server()
    return jsonify(json.loads(server._get_metrics()))


@app.route('/health')
def flask_health():
    """Flask endpoint for health."""
    server = get_monitoring_server()
    return jsonify(json.loads(server._get_health()))


@app.route('/logs')
def flask_logs():
    """Flask endpoint for logs."""
    server = get_monitoring_server()
    logs = server._get_logs()
    return jsonify({"logs": logs})


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
    """Start the monitoring server."""
    config = get_config()

    if config.monitoring.tcp_server.get("enabled", True):
        host = config.monitoring.tcp_server.get("host", "127.0.0.1")
        port = config.monitoring.tcp_server.get("port", 9999)

        server = get_monitoring_server(host, port)
        server.start()

        return server
    else:
        logger.info("Monitoring server disabled in configuration")
        return None


if __name__ == "__main__":
    import sys

    # Start Flask HTTP server instead of raw TCP
    print("Starting RAG-CLI Monitoring Server (HTTP)...")
    print("Server running on http://localhost:9999")
    print("Endpoints: /status, /metrics, /health, /logs")
    print("Press Ctrl+C to stop...")

    try:
        app.run(host="127.0.0.1", port=9999, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)