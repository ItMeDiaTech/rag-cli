"""Service manager for RAG-CLI monitoring services.

Manages the lifecycle of monitoring services (TCP server, web dashboard)
including auto-start, health checks, and graceful shutdown.
"""

import subprocess
import time
import socket
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from src.monitoring.logger import get_logger

logger = get_logger(__name__)

# Service configuration
SERVICES_CONFIG = {
    'tcp_server': {
        'name': 'TCP Monitoring Server',
        'module': 'src.monitoring.tcp_server',
        'port': 9999,
        'required': True,
    },
    'web_dashboard': {
        'name': 'Web Dashboard',
        'module': 'src.monitoring.web_dashboard',
        'port': 5000,
        'args': '5000',
        'required': False,
    }
}

# Status file to track running services
STATUS_DIR = Path(__file__).parent.parent.parent / 'config'
STATUS_FILE = STATUS_DIR / 'services_status.json'


def load_services_status() -> Dict[str, Any]:
    """Load services status from file.

    Returns:
        Dictionary with services status
    """
    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.debug(f"Failed to load services status: {e}")

    return {
        'tcp_server': {'running': False, 'pid': None, 'started_at': None},
        'web_dashboard': {'running': False, 'pid': None, 'started_at': None}
    }


def save_services_status(status: Dict[str, Any]):
    """Save services status to file.

    Args:
        status: Dictionary with services status
    """
    try:
        STATUS_DIR.mkdir(parents=True, exist_ok=True)
        with open(STATUS_FILE, 'w') as f:
            json.dump(status, f, indent=2)
    except Exception as e:
        logger.debug(f"Failed to save services status: {e}")


def is_port_open(host: str = '127.0.0.1', port: int = 9999, timeout: float = 1.0) -> bool:
    """Check if a port is open and accepting connections.

    Args:
        host: Host to check
        port: Port to check
        timeout: Connection timeout in seconds

    Returns:
        True if port is open
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            return result == 0
    except Exception as e:
        logger.debug(f"Port check failed for {host}:{port}: {e}")
        return False


def start_tcp_server() -> Optional[subprocess.Popen]:
    """Start the TCP monitoring server.

    Returns:
        Popen object if successful, None otherwise
    """
    if is_port_open(port=9999):
        logger.info("TCP server already running on port 9999")
        return None

    try:
        logger.info("Starting TCP monitoring server...")

        # Get the project root
        project_root = Path(__file__).resolve().parents[2]

        # Start the process
        process = subprocess.Popen(
            ['python', '-m', 'src.monitoring.tcp_server'],
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy()
        )

        # Wait for server to start
        max_attempts = 30
        for attempt in range(max_attempts):
            if is_port_open(port=9999):
                logger.info(f"TCP server started successfully (PID: {process.pid})")
                return process
            time.sleep(0.1)

        logger.error("TCP server failed to start - timeout waiting for port 9999")
        process.terminate()
        return None

    except Exception as e:
        logger.error(f"Failed to start TCP server: {e}")
        return None


def start_web_dashboard() -> Optional[subprocess.Popen]:
    """Start the web dashboard server.

    Returns:
        Popen object if successful, None otherwise
    """
    if is_port_open(port=5000):
        logger.info("Web dashboard already running on port 5000")
        return None

    try:
        logger.info("Starting web dashboard server...")

        # Get the project root
        project_root = Path(__file__).resolve().parents[2]

        # Start the process
        process = subprocess.Popen(
            ['python', '-m', 'src.monitoring.web_dashboard', '5000'],
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy()
        )

        # Wait for server to start
        max_attempts = 30
        for attempt in range(max_attempts):
            if is_port_open(port=5000):
                logger.info(f"Web dashboard started successfully (PID: {process.pid})")
                return process
            time.sleep(0.1)

        logger.error("Web dashboard failed to start - timeout waiting for port 5000")
        process.terminate()
        return None

    except Exception as e:
        logger.error(f"Failed to start web dashboard: {e}")
        return None


def ensure_services_running() -> Dict[str, bool]:
    """Ensure all required services are running.

    Starts services if they're not already running.

    Returns:
        Dictionary with service name -> running status
    """
    status = load_services_status()
    results = {}

    try:
        # Check and start TCP server (required)
        if not is_port_open(port=9999):
            tcp_process = start_tcp_server()
            if tcp_process:
                status['tcp_server']['running'] = True
                status['tcp_server']['pid'] = tcp_process.pid
                status['tcp_server']['started_at'] = datetime.now().isoformat()
                results['tcp_server'] = True
            else:
                logger.error("Failed to start TCP server")
                results['tcp_server'] = False
        else:
            results['tcp_server'] = True
            status['tcp_server']['running'] = True

        # Check and start web dashboard (optional)
        if not is_port_open(port=5000):
            dashboard_process = start_web_dashboard()
            if dashboard_process:
                status['web_dashboard']['running'] = True
                status['web_dashboard']['pid'] = dashboard_process.pid
                status['web_dashboard']['started_at'] = datetime.now().isoformat()
                results['web_dashboard'] = True
            else:
                logger.warning("Failed to start web dashboard")
                results['web_dashboard'] = False
        else:
            results['web_dashboard'] = True
            status['web_dashboard']['running'] = True

        # Save status
        save_services_status(status)

        # Log summary
        all_running = all(results.values())
        if all_running:
            logger.info("All monitoring services running successfully")
        else:
            logger.warning(f"Some services failed to start: {results}")

        return results

    except Exception as e:
        logger.error(f"Service startup failed: {e}")
        return {
            'tcp_server': False,
            'web_dashboard': False,
            'error': str(e)
        }


def get_services_status() -> Dict[str, Any]:
    """Get current status of all services.

    Returns:
        Dictionary with service status information
    """
    status = load_services_status()

    for service_name, config in SERVICES_CONFIG.items():
        port = config['port']
        is_running = is_port_open(port=port)

        status[service_name]['running'] = is_running
        status[service_name]['port'] = port
        status[service_name]['name'] = config['name']

        if is_running:
            if service_name == 'tcp_server':
                status[service_name]['url'] = f"tcp://127.0.0.1:{port}"
            elif service_name == 'web_dashboard':
                status[service_name]['url'] = f"http://localhost:{port}"

    return status


def open_dashboard_in_browser():
    """Open the web dashboard in the default browser."""
    import webbrowser

    if is_port_open(port=5000):
        url = "http://localhost:5000"
        try:
            webbrowser.open(url)
            logger.info(f"Opened dashboard in browser: {url}")
        except Exception as e:
            logger.error(f"Failed to open browser: {e}")
    else:
        logger.error("Web dashboard is not running")


if __name__ == '__main__':
    # Test the service manager
    print("RAG-CLI Service Manager")
    print("=" * 50)

    # Ensure services are running
    print("\nStarting services...")
    results = ensure_services_running()
    print(f"Results: {results}")

    # Get status
    print("\nService Status:")
    status = get_services_status()
    for service, info in status.items():
        if isinstance(info, dict):
            running = "✓ Running" if info.get('running') else "✗ Stopped"
            print(f"  {info.get('name', service)}: {running}")
            if info.get('url'):
                print(f"    URL: {info['url']}")
