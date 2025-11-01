"""Web dashboard server for RAG-CLI monitoring.

Provides a Flask web server with a modern dashboard interface for
viewing real-time RAG-CLI metrics and system status.
"""

import json
import time
import threading
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, jsonify, Response, stream_with_context
from flask_cors import CORS

# Global shutdown event for graceful termination
_shutdown_event = threading.Event()

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
    """Proxy to TCP monitoring server."""
    import requests
    try:
        response = requests.get('http://127.0.0.1:9999/api/status', timeout=2)
        return response.json()
    except Exception as e:
        return jsonify({
            "error": "Unable to connect to monitoring server",
            "details": str(e),
            "status": "offline"
        }), 503


@app.route('/api/metrics')
def api_metrics():
    """Proxy to TCP monitoring server."""
    import requests
    try:
        response = requests.get('http://127.0.0.1:9999/api/metrics', timeout=2)
        return response.json()
    except Exception as e:
        return jsonify({
            "error": "Unable to connect to monitoring server",
            "details": str(e)
        }), 503


@app.route('/api/logs')
def api_logs():
    """Proxy to TCP monitoring server."""
    import requests
    try:
        response = requests.get('http://127.0.0.1:9999/api/logs', timeout=2)
        return response.json()
    except Exception as e:
        return jsonify({
            "error": "Unable to connect to monitoring server",
            "details": str(e),
            "logs": []
        }), 503


@app.route('/api/health')
def api_health():
    """Proxy to TCP monitoring server."""
    import requests
    try:
        response = requests.get('http://127.0.0.1:9999/api/health', timeout=2)
        return response.json()
    except Exception:
        return jsonify({
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "issues": [{
                "component": "monitoring_server",
                "status": "disconnected"
            }]
        }), 503


@app.route('/api/events')
def api_events():
    """Server-Sent Events endpoint for real-time updates.

    Proxies the SSE stream from the TCP server to enable cross-process
    event streaming from hooks to the web dashboard.
    """
    def generate_events():
        import requests

        try:
            # Connect to TCP server's SSE stream
            # Use stream=True to get real-time events, not buffered
            response = requests.get(
                'http://127.0.0.1:9999/api/events/stream',
                stream=True,
                timeout=None,  # No timeout for SSE
                headers={'Accept': 'text/event-stream'}
            )

            # Stream events line by line from TCP server
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    # Forward the SSE line to client
                    yield f"{line}\n"
                else:
                    # Empty line (event separator)
                    yield "\n"

        except requests.exceptions.RequestException as e:
            # TCP server not available - send error event
            error_event = {
                "event_type": "error",
                "message": f"Cannot connect to TCP server: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(error_event)}\n\n"

            # Send keepalive every 15 seconds while disconnected
            while not _shutdown_event.is_set():
                if _shutdown_event.wait(timeout=15):
                    yield f"data: {json.dumps({'event': 'shutdown', 'message': 'Server shutting down'})}\n\n"
                    break
                yield ": keepalive (disconnected)\n\n"

    return Response(
        stream_with_context(generate_events()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )


if __name__ == '__main__':
    import sys
    import os
    from pathlib import Path

    # Get port from environment variable, command line, or use default
    port = int(os.environ.get('RAG_DASHBOARD_PORT',
                              sys.argv[1] if len(sys.argv) > 1 else 5000))

    print(f"RAG-CLI Web Dashboard starting on http://localhost:{port}")

    # Use Flask development server for SSE streaming support
    # Waitress doesn't support streaming responses properly
    print("Using Flask development server (required for SSE streaming)...")
    app.run(host='127.0.0.1', port=port, debug=False, threaded=True)
