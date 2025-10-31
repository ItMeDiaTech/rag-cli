"""Real-time Performance Monitoring Dashboard for RAG-CLI.

This module provides a web-based dashboard for monitoring RAG-CLI performance,
including real-time metrics, historical trends, and system health indicators.
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import deque
import threading
import time

from flask import Flask, render_template_string, jsonify, request
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and aggregates metrics for the dashboard."""

    def __init__(self, history_size: int = 1000):
        """Initialize metrics collector.

        Args:
            history_size: Maximum number of data points to keep
        """
        self.history_size = history_size
        self._lock = threading.RLock()

        # Metric storage
        self.latency_history: deque = deque(maxlen=history_size)
        self.throughput_history: deque = deque(maxlen=history_size)
        self.error_count: int = 0
        self.success_count: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0

        # Component-specific metrics
        self.component_latencies: Dict[str, deque] = {}
        self.component_errors: Dict[str, int] = {}

        # System metrics
        self.start_time = datetime.now()
        self.last_update = datetime.now()

    def record_latency(self, component: str, latency_ms: float) -> None:
        """Record latency for a component.

        Args:
            component: Component name
            latency_ms: Latency in milliseconds
        """
        with self._lock:
            timestamp = datetime.now()

            # Overall latency
            self.latency_history.append({
                'timestamp': timestamp.isoformat(),
                'value': latency_ms,
                'component': component
            })

            # Component-specific latency
            if component not in self.component_latencies:
                self.component_latencies[component] = deque(maxlen=self.history_size)

            self.component_latencies[component].append({
                'timestamp': timestamp.isoformat(),
                'value': latency_ms
            })

            self.last_update = timestamp

    def record_success(self) -> None:
        """Record a successful operation."""
        with self._lock:
            self.success_count += 1
            self.last_update = datetime.now()

    def record_error(self, component: str) -> None:
        """Record an error.

        Args:
            component: Component where error occurred
        """
        with self._lock:
            self.error_count += 1

            if component not in self.component_errors:
                self.component_errors[component] = 0

            self.component_errors[component] += 1
            self.last_update = datetime.now()

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        with self._lock:
            self.cache_hits += 1
            self.last_update = datetime.now()

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        with self._lock:
            self.cache_misses += 1
            self.last_update = datetime.now()

    def record_throughput(self, operations_per_second: float) -> None:
        """Record throughput measurement.

        Args:
            operations_per_second: Current throughput
        """
        with self._lock:
            timestamp = datetime.now()
            self.throughput_history.append({
                'timestamp': timestamp.isoformat(),
                'value': operations_per_second
            })
            self.last_update = timestamp

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics.

        Returns:
            Dictionary containing metric summary
        """
        with self._lock:
            uptime = (datetime.now() - self.start_time).total_seconds()
            total_operations = self.success_count + self.error_count
            success_rate = (self.success_count / total_operations * 100) if total_operations > 0 else 0
            cache_total = self.cache_hits + self.cache_misses
            cache_hit_rate = (self.cache_hits / cache_total * 100) if cache_total > 0 else 0

            # Calculate average latencies
            avg_latencies = {}
            for component, latencies in self.component_latencies.items():
                if latencies:
                    avg = sum(l['value'] for l in latencies) / len(latencies)
                    avg_latencies[component] = round(avg, 2)

            return {
                'uptime_seconds': round(uptime, 2),
                'total_operations': total_operations,
                'success_count': self.success_count,
                'error_count': self.error_count,
                'success_rate': round(success_rate, 2),
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cache_hit_rate': round(cache_hit_rate, 2),
                'component_latencies': avg_latencies,
                'component_errors': self.component_errors,
                'last_update': self.last_update.isoformat()
            }

    def get_time_series(self, metric: str, minutes: int = 60) -> List[Dict]:
        """Get time series data for a metric.

        Args:
            metric: Metric name ('latency', 'throughput')
            minutes: Number of minutes of history to return

        Returns:
            List of data points
        """
        with self._lock:
            cutoff = datetime.now() - timedelta(minutes=minutes)

            if metric == 'latency':
                history = list(self.latency_history)
            elif metric == 'throughput':
                history = list(self.throughput_history)
            else:
                return []

            # Filter by time range
            filtered = [
                point for point in history
                if datetime.fromisoformat(point['timestamp']) >= cutoff
            ]

            return filtered


class PerformanceDashboard:
    """Web-based performance monitoring dashboard."""

    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 9998,
        collector: Optional[MetricsCollector] = None
    ):
        """Initialize dashboard.

        Args:
            host: Host to bind to
            port: Port to bind to
            collector: Metrics collector instance
        """
        self.host = host
        self.port = port
        self.collector = collector or MetricsCollector()

        self.app = Flask(__name__)
        self._setup_routes()
        self._running = False

    def _setup_routes(self):
        """Set up Flask routes."""

        @self.app.route('/')
        def index():
            """Serve dashboard homepage."""
            return render_template_string(DASHBOARD_TEMPLATE)

        @self.app.route('/api/metrics/summary')
        def get_summary():
            """Get metrics summary."""
            return jsonify(self.collector.get_summary())

        @self.app.route('/api/metrics/timeseries')
        def get_timeseries():
            """Get time series data."""
            metric = request.args.get('metric', 'latency')
            minutes = int(request.args.get('minutes', 60))

            data = self.collector.get_time_series(metric, minutes)
            return jsonify({
                'metric': metric,
                'data': data
            })

        @self.app.route('/api/health')
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat()
            })

    def start(self, debug: bool = False):
        """Start the dashboard server.

        Args:
            debug: Enable debug mode
        """
        logger.info(f"Starting dashboard on http://{self.host}:{self.port}")
        self._running = True

        try:
            self.app.run(host=self.host, port=self.port, debug=debug, use_reloader=False)
        except Exception as e:
            logger.error(f"Dashboard server error: {e}")
            raise

    def start_background(self):
        """Start dashboard in background thread."""
        thread = threading.Thread(target=self.start, daemon=True)
        thread.start()
        logger.info("Dashboard started in background")
        return thread

    def stop(self):
        """Stop the dashboard server."""
        self._running = False
        logger.info("Dashboard stopped")


# Dashboard HTML template with embedded JavaScript
DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG-CLI Performance Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        header {
            margin-bottom: 30px;
        }

        h1 {
            font-size: 2.5rem;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 10px;
        }

        .subtitle {
            color: #94a3b8;
            font-size: 1.1rem;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .metric-card {
            background: #1e293b;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #334155;
        }

        .metric-label {
            color: #94a3b8;
            font-size: 0.9rem;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #f8fafc;
        }

        .metric-unit {
            font-size: 1rem;
            color: #64748b;
            margin-left: 5px;
        }

        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .chart-card {
            background: #1e293b;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #334155;
        }

        .chart-title {
            font-size: 1.3rem;
            margin-bottom: 15px;
            color: #f8fafc;
        }

        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }

        .status-healthy {
            background: #10b981;
        }

        .status-warning {
            background: #f59e0b;
        }

        .status-error {
            background: #ef4444;
        }

        .component-list {
            list-style: none;
        }

        .component-item {
            padding: 12px;
            margin-bottom: 8px;
            background: #0f172a;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .component-name {
            font-weight: 500;
        }

        .component-value {
            color: #60a5fa;
            font-weight: 600;
        }

        @keyframes pulse {
            0%, 100% {
                opacity: 1;
            }
            50% {
                opacity: 0.5;
            }
        }

        .updating {
            animation: pulse 1.5s ease-in-out infinite;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>RAG-CLI Performance Dashboard</h1>
            <p class="subtitle">
                <span class="status-indicator status-healthy"></span>
                Real-time monitoring | Last update: <span id="last-update">-</span>
            </p>
        </header>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Success Rate</div>
                <div class="metric-value">
                    <span id="success-rate">0</span><span class="metric-unit">%</span>
                </div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Total Operations</div>
                <div class="metric-value">
                    <span id="total-operations">0</span>
                </div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Cache Hit Rate</div>
                <div class="metric-value">
                    <span id="cache-hit-rate">0</span><span class="metric-unit">%</span>
                </div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Uptime</div>
                <div class="metric-value">
                    <span id="uptime">0</span><span class="metric-unit">s</span>
                </div>
            </div>
        </div>

        <div class="charts-grid">
            <div class="chart-card">
                <h3 class="chart-title">Latency Over Time</h3>
                <canvas id="latency-chart"></canvas>
            </div>

            <div class="chart-card">
                <h3 class="chart-title">Component Latencies</h3>
                <ul class="component-list" id="component-latencies"></ul>
            </div>
        </div>

        <div class="chart-card">
            <h3 class="chart-title">Error Distribution</h3>
            <canvas id="error-chart"></canvas>
        </div>
    </div>

    <script>
        // Initialize charts
        const latencyCtx = document.getElementById('latency-chart').getContext('2d');
        const errorCtx = document.getElementById('error-chart').getContext('2d');

        const latencyChart = new Chart(latencyCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Latency (ms)',
                    data: [],
                    borderColor: '#60a5fa',
                    backgroundColor: 'rgba(96, 165, 250, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { color: '#94a3b8' },
                        grid: { color: '#334155' }
                    },
                    x: {
                        ticks: { color: '#94a3b8' },
                        grid: { color: '#334155' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#e2e8f0' }
                    }
                }
            }
        });

        const errorChart = new Chart(errorCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Errors',
                    data: [],
                    backgroundColor: '#ef4444'
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { color: '#94a3b8' },
                        grid: { color: '#334155' }
                    },
                    x: {
                        ticks: { color: '#94a3b8' },
                        grid: { color: '#334155' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#e2e8f0' }
                    }
                }
            }
        });

        // Update dashboard
        async function updateDashboard() {
            try {
                const response = await fetch('/api/metrics/summary');
                const data = await response.json();

                // Update metrics
                document.getElementById('success-rate').textContent = data.success_rate;
                document.getElementById('total-operations').textContent = data.total_operations;
                document.getElementById('cache-hit-rate').textContent = data.cache_hit_rate;
                document.getElementById('uptime').textContent = data.uptime_seconds;
                document.getElementById('last-update').textContent = new Date(data.last_update).toLocaleTimeString();

                // Update component latencies
                const componentList = document.getElementById('component-latencies');
                componentList.innerHTML = '';
                for (const [component, latency] of Object.entries(data.component_latencies)) {
                    const li = document.createElement('li');
                    li.className = 'component-item';
                    li.innerHTML = `
                        <span class="component-name">${component}</span>
                        <span class="component-value">${latency} ms</span>
                    `;
                    componentList.appendChild(li);
                }

                // Update error chart
                errorChart.data.labels = Object.keys(data.component_errors);
                errorChart.data.datasets[0].data = Object.values(data.component_errors);
                errorChart.update();

            } catch (error) {
                console.error('Failed to update dashboard:', error);
            }
        }

        async function updateLatencyChart() {
            try {
                const response = await fetch('/api/metrics/timeseries?metric=latency&minutes=60');
                const data = await response.json();

                if (data.data.length > 0) {
                    const times = data.data.map(p => new Date(p.timestamp).toLocaleTimeString());
                    const values = data.data.map(p => p.value);

                    latencyChart.data.labels = times;
                    latencyChart.data.datasets[0].data = values;
                    latencyChart.update();
                }
            } catch (error) {
                console.error('Failed to update latency chart:', error);
            }
        }

        // Update every 2 seconds
        setInterval(updateDashboard, 2000);
        setInterval(updateLatencyChart, 5000);

        // Initial update
        updateDashboard();
        updateLatencyChart();
    </script>
</body>
</html>
'''


# Example usage and testing
if __name__ == '__main__':
    import random

    # Create dashboard
    collector = MetricsCollector()
    dashboard = PerformanceDashboard(collector=collector)

    # Simulate metrics in background
    def simulate_metrics():
        components = ['embeddings', 'vector_store', 'retrieval', 'claude_api']

        while True:
            # Random metrics
            component = random.choice(components)
            latency = random.uniform(10, 200)
            collector.record_latency(component, latency)

            if random.random() > 0.9:
                collector.record_error(component)
            else:
                collector.record_success()

            if random.random() > 0.3:
                collector.record_cache_hit()
            else:
                collector.record_cache_miss()

            time.sleep(0.5)

    # Start simulation thread
    sim_thread = threading.Thread(target=simulate_metrics, daemon=True)
    sim_thread.start()

    # Start dashboard
    # SECURITY: Only enable debug mode in development
    import os
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    dashboard.start(debug=debug_mode)
