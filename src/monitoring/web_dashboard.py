"""Web dashboard server for RAG-CLI monitoring.

Provides a Flask web server with a modern dashboard interface for
viewing real-time RAG-CLI metrics and system status.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, jsonify
from flask_cors import CORS

from src.monitoring.tcp_server import metrics_collector

# Get the templates directory
templates_dir = Path(__file__).parent / 'templates'

app = Flask(__name__, template_folder=str(templates_dir))
CORS(app)


@app.route('/')
def dashboard():
    """Serve the dashboard HTML."""
    return render_template('dashboard.html')


@app.route('/api/status')
def api_status():
    """Get current system status."""
    try:
        from src.core.vector_store import get_vector_store
        vector_store = get_vector_store()
        total_vectors = vector_store.index.ntotal
    except:
        total_vectors = 0

    uptime_seconds = metrics_collector.get_uptime()
    uptime_minutes = int(uptime_seconds / 60)
    uptime_hours = int(uptime_minutes / 60)
    uptime_mins = uptime_minutes % 60

    return jsonify({
        "version": "0.1.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "uptime": {
            "seconds": uptime_seconds,
            "formatted": f"{uptime_hours}h {uptime_mins}m {int(uptime_seconds % 60)}s"
        },
        "components": metrics_collector.component_status,
        "statistics": {
            "total_documents": 0,
            "total_vectors": total_vectors,
            "total_queries": metrics_collector.query_count,
            "total_errors": metrics_collector.error_count,
            "cache_hit_rate": round(metrics_collector.get_cache_hit_rate(), 1)
        }
    })


@app.route('/api/metrics')
def api_metrics():
    """Get performance metrics."""
    latencies = metrics_collector.get_latest_latencies()
    uptime = max(1, metrics_collector.get_uptime())

    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "latency": {
            "vector_search": round(latencies.get("vector_search", 0), 2),
            "keyword_search": round(latencies.get("keyword_search", 0), 2),
            "reranking": round(latencies.get("reranking", 0), 2),
            "claude_api": round(latencies.get("claude_api", 0), 2),
            "end_to_end": round(latencies.get("end_to_end", 0), 2)
        },
        "throughput": {
            "queries_per_minute": round(metrics_collector.query_count * 60 / uptime, 2),
            "docs_per_minute": 0
        },
        "cache": {
            "hit_rate": round(metrics_collector.get_cache_hit_rate(), 1),
            "hits": metrics_collector.cache_hits,
            "misses": metrics_collector.cache_misses
        },
        "resources": metrics_collector.get_resource_usage()
    })


@app.route('/api/logs')
def api_logs():
    """Get recent logs."""
    logs = list(metrics_collector.log_buffer)[-50:]
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "logs": logs
    })


@app.route('/api/health')
def api_health():
    """Get health check status."""
    issues = []

    for component, status in metrics_collector.component_status.items():
        if status not in ["operational", "ready", "healthy"]:
            issues.append({
                "component": component,
                "status": status
            })

    if metrics_collector.error_count > 0:
        error_rate = metrics_collector.error_count / max(1, metrics_collector.query_count)
        if error_rate > 0.1:
            issues.append({
                "component": "error_rate",
                "status": f"{error_rate:.1%}"
            })

    resources = metrics_collector.get_resource_usage()
    if resources["memory_mb"] > 2048:
        issues.append({
            "component": "memory",
            "status": f"{resources['memory_mb']:.0f} MB"
        })

    return jsonify({
        "status": "unhealthy" if issues else "healthy",
        "timestamp": datetime.now().isoformat(),
        "issues": issues
    })


if __name__ == '__main__':
    import sys
    from pathlib import Path

    # Get port from command line or use default
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000

    print(f"RAG-CLI Web Dashboard starting on http://localhost:{port}")
    app.run(host='127.0.0.1', port=port, debug=False)
