"""Service manager for RAG-CLI monitoring services.

Manages the lifecycle of monitoring services (TCP server, web dashboard)
including auto-start, health checks, and graceful shutdown.

Also implements MCP (Model Context Protocol) server interface for Claude Code integration.
"""

import subprocess
import time
import socket
import json
import os
import sys
import atexit
import signal
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

# PID directory for tracking process IDs
PID_DIR = STATUS_DIR / 'pids'
PID_DIR.mkdir(parents=True, exist_ok=True)

# Global registry for running processes and resources
_running_processes: Dict[str, Dict[str, Any]] = {
    'tcp_server': {'process': None, 'log_file': None, 'pid': None},
    'web_dashboard': {'process': None, 'log_file': None, 'pid': None}
}

_shutdown_registered = False


def write_pid_file(service_name: str, pid: int):
    """Write PID to file for later cleanup.

    Args:
        service_name: Name of the service
        pid: Process ID
    """
    try:
        pid_file = PID_DIR / f"{service_name}.pid"
        pid_file.write_text(str(pid))
        logger.debug(f"Wrote PID file for {service_name}: {pid}")
    except Exception as e:
        logger.error(f"Failed to write PID file for {service_name}: {e}")


def remove_pid_file(service_name: str):
    """Remove PID file.

    Args:
        service_name: Name of the service
    """
    try:
        pid_file = PID_DIR / f"{service_name}.pid"
        if pid_file.exists():
            pid_file.unlink()
            logger.debug(f"Removed PID file for {service_name}")
    except Exception as e:
        logger.error(f"Failed to remove PID file for {service_name}: {e}")


def cleanup_stale_processes():
    """Kill processes from previous runs that may have been orphaned."""
    if not PID_DIR.exists():
        return

    import psutil

    logger.info("Checking for stale processes from previous runs...")
    cleaned = 0

    for pid_file in PID_DIR.glob("*.pid"):
        try:
            pid = int(pid_file.read_text().strip())
            service_name = pid_file.stem

            # Check if process exists
            try:
                process = psutil.Process(pid)

                # Verify it's a Python process (our service)
                if 'python' in process.name().lower():
                    logger.warning(f"Found stale {service_name} process (PID: {pid}), terminating...")
                    process.terminate()

                    # Wait up to 5 seconds for graceful termination
                    try:
                        process.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        logger.warning(f"Process {pid} did not terminate, killing...")
                        process.kill()

                    cleaned += 1
                else:
                    # PID reused by another process, just remove the file
                    logger.debug(f"PID {pid} reused by {process.name()}, removing stale PID file")

            except psutil.NoSuchProcess:
                # Process already dead, just clean up PID file
                logger.debug(f"Process {pid} already terminated")

            # Remove the PID file
            pid_file.unlink()

        except (ValueError, FileNotFoundError) as e:
            logger.warning(f"Invalid PID file {pid_file}: {e}")
            try:
                pid_file.unlink()
            except:
                pass
        except Exception as e:
            logger.error(f"Error cleaning up PID file {pid_file}: {e}")

    if cleaned > 0:
        logger.info(f"Cleaned up {cleaned} stale process(es)")


def shutdown_services():
    """Gracefully shutdown all monitoring services.

    This function is called automatically on program exit via atexit
    or signal handlers. It ensures all subprocess are terminated and
    resources are cleaned up properly.
    """
    logger.info("Shutting down monitoring services...")

    for service_name, info in _running_processes.items():
        process = info.get('process')
        log_file = info.get('log_file')
        pid = info.get('pid')

        # Remove PID file
        remove_pid_file(service_name)

        # Terminate subprocess if running
        if process and pid:
            try:
                logger.info(f"Terminating {service_name} (PID: {pid})")

                # Try graceful termination first
                process.terminate()

                try:
                    # Wait up to 5 seconds for graceful shutdown
                    process.wait(timeout=5)
                    logger.info(f"{service_name} terminated gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if still running
                    logger.warning(f"{service_name} did not terminate gracefully, forcing...")
                    process.kill()
                    process.wait(timeout=2)
                    logger.info(f"{service_name} force killed")

            except Exception as e:
                logger.error(f"Error terminating {service_name}: {e}")

        # Close log file handle if open
        if log_file:
            try:
                if not log_file.closed:
                    log_file.close()
                    logger.debug(f"Closed log file for {service_name}")
            except Exception as e:
                logger.error(f"Error closing log file for {service_name}: {e}")

    # Clear the registry
    for service in _running_processes:
        _running_processes[service] = {'process': None, 'log_file': None, 'pid': None}

    logger.info("All monitoring services shut down successfully")


def _signal_handler(signum, frame):
    """Handle termination signals.

    Args:
        signum: Signal number
        frame: Current stack frame
    """
    logger.info(f"Received signal {signum}, initiating shutdown...")
    shutdown_services()
    sys.exit(0)


def register_shutdown_handlers():
    """Register shutdown handlers for graceful cleanup.

    Registers both atexit handler for normal exit and signal handlers
    for forced termination (SIGTERM, SIGINT).
    """
    global _shutdown_registered

    if _shutdown_registered:
        return

    # Register atexit handler for normal program exit
    atexit.register(shutdown_services)
    logger.debug("Registered atexit shutdown handler")

    # Register signal handlers for forced termination
    try:
        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)
        logger.debug("Registered signal handlers (SIGTERM, SIGINT)")
    except (AttributeError, ValueError) as e:
        # Some signals may not be available on all platforms
        logger.debug(f"Could not register all signal handlers: {e}")

    _shutdown_registered = True


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

        # Create logs directory if it doesn't exist
        log_dir = project_root / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        # Open log file for subprocess output
        tcp_log_file = log_dir / 'tcp_server.log'
        tcp_log = open(tcp_log_file, 'a', buffering=1)  # Line buffered
        logger.info(f"TCP server output will be logged to {tcp_log_file}")

        # Start the process with output redirected to log file
        process = subprocess.Popen(
            ['python', '-m', 'src.monitoring.tcp_server'],
            cwd=str(project_root),
            stdout=tcp_log,
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            env=os.environ.copy()
        )

        # Wait for server to start with exponential backoff
        max_attempts = 20
        wait_time = 0.05  # Start with 50ms
        for attempt in range(max_attempts):
            if is_port_open(port=9999, timeout=0.5):
                logger.info(f"TCP server started successfully (PID: {process.pid})")

                # Register process and log file in global registry
                _running_processes['tcp_server'] = {
                    'process': process,
                    'log_file': tcp_log,
                    'pid': process.pid
                }

                # Write PID file for cleanup on next start
                write_pid_file('tcp_server', process.pid)

                return process
            time.sleep(wait_time)
            wait_time = min(wait_time * 1.5, 0.5)  # Exponential backoff, max 500ms

        logger.error("TCP server failed to start - timeout waiting for port 9999")
        process.terminate()
        tcp_log.close()
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

        # Create logs directory if it doesn't exist
        log_dir = project_root / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        # Open log file for subprocess output
        dashboard_log_file = log_dir / 'web_dashboard.log'
        dashboard_log = open(dashboard_log_file, 'a', buffering=1)  # Line buffered
        logger.info(f"Web dashboard output will be logged to {dashboard_log_file}")

        # Start the process with output redirected to log file
        process = subprocess.Popen(
            ['python', '-m', 'src.monitoring.web_dashboard', '5000'],
            cwd=str(project_root),
            stdout=dashboard_log,
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            env=os.environ.copy()
        )

        # Wait for server to start with exponential backoff
        max_attempts = 20
        wait_time = 0.05  # Start with 50ms
        for attempt in range(max_attempts):
            if is_port_open(port=5000, timeout=0.5):
                logger.info(f"Web dashboard started successfully (PID: {process.pid})")

                # Register process and log file in global registry
                _running_processes['web_dashboard'] = {
                    'process': process,
                    'log_file': dashboard_log,
                    'pid': process.pid
                }

                # Write PID file for cleanup on next start
                write_pid_file('web_dashboard', process.pid)

                return process
            time.sleep(wait_time)
            wait_time = min(wait_time * 1.5, 0.5)  # Exponential backoff, max 500ms

        logger.error("Web dashboard failed to start - timeout waiting for port 5000")
        process.terminate()
        dashboard_log.close()
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
    # Clean up any stale processes from previous runs
    cleanup_stale_processes()

    # Register shutdown handlers for graceful cleanup on exit
    register_shutdown_handlers()

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


def find_chrome_path() -> Optional[str]:
    """Find Chrome executable path on different platforms.

    Returns:
        Path to Chrome executable or None if not found
    """
    import platform

    system = platform.system()

    # Windows paths
    if system == 'Windows':
        possible_paths = [
            os.path.expandvars(r'%ProgramFiles%\Google\Chrome\Application\chrome.exe'),
            os.path.expandvars(r'%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe'),
            os.path.expandvars(r'%LocalAppData%\Google\Chrome\Application\chrome.exe'),
        ]
    # Mac paths
    elif system == 'Darwin':
        possible_paths = [
            '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
            os.path.expanduser('~/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'),
        ]
    # Linux paths
    else:
        possible_paths = [
            '/usr/bin/google-chrome',
            '/usr/bin/google-chrome-stable',
            '/usr/bin/chromium-browser',
            '/usr/bin/chromium',
        ]

    logger.debug(f"Searching for Chrome on {system} in {len(possible_paths)} locations")

    for path in possible_paths:
        logger.debug(f"Checking Chrome path: {path}")
        if os.path.exists(path):
            # Verify it's executable (not just a file)
            if os.access(path, os.X_OK) or system == 'Windows':
                logger.info(f"Chrome found at: {path}")
                return path
            else:
                logger.debug(f"Path exists but not executable: {path}")

    logger.warning(f"Chrome not found on {system}. Checked paths: {possible_paths}")
    return None


def open_dashboard_in_browser(use_chrome: bool = True, wait_for_ready: bool = True):
    """Open the web dashboard in Chrome browser with dev flags.

    Args:
        use_chrome: Try to use Chrome specifically (falls back to default browser if False or not found)
        wait_for_ready: Wait for dashboard to be ready before opening
    """
    import webbrowser

    url = "http://localhost:5000"

    # Wait for dashboard to be ready (with timeout)
    if wait_for_ready:
        max_attempts = 10
        for attempt in range(max_attempts):
            if is_port_open(port=5000):
                logger.info("Dashboard is ready")
                break
            if attempt < max_attempts - 1:
                logger.debug(f"Waiting for dashboard... ({attempt + 1}/{max_attempts})")
                time.sleep(0.5)
        else:
            logger.error("Web dashboard is not ready after waiting")
            return False

    # Try Chrome first with security flags for local development
    if use_chrome:
        chrome_path = find_chrome_path()

        if chrome_path:
            # Validate Chrome path before attempting to launch
            if not os.path.exists(chrome_path):
                logger.warning(f"Chrome path returned but doesn't exist: {chrome_path}")
                logger.info("Falling back to default browser")
            else:
                try:
                    # Chrome flags for local development
                    chrome_flags = [
                        chrome_path,
                        '--new-window',
                        '--disable-web-security',  # Allow CORS for local dev
                        '--allow-file-access-from-files',
                        '--disable-features=IsolateOrigins,site-per-process',
                        f'--user-data-dir={Path.home() / ".rag-cli-chrome-temp"}',  # Temp profile
                        url
                    ]

                    logger.debug(f"Launching Chrome with command: {chrome_flags}")

                    # Start Chrome
                    subprocess.Popen(
                        chrome_flags,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                    )

                    logger.info(f"Successfully opened dashboard in Chrome at {chrome_path}: {url}")
                    return True

                except subprocess.SubprocessError as e:
                    logger.error(f"Subprocess error launching Chrome at {chrome_path}: {type(e).__name__}: {e}")
                    logger.info("Falling back to default browser")
                except Exception as e:
                    logger.error(f"Unexpected error launching Chrome at {chrome_path}: {type(e).__name__}: {e}")
                    logger.info("Falling back to default browser")
        else:
            logger.info("Chrome not found on system, using default browser")

    # Fallback to default browser
    try:
        webbrowser.open(url)
        logger.info(f"Opened dashboard in default browser: {url}")
        return True
    except Exception as e:
        logger.error(f"Failed to open browser: {e}")
        return False


# MCP Server Implementation
def send_mcp_response(response: Dict[str, Any]):
    """Send an MCP protocol response to stdout."""
    json_response = json.dumps(response)
    sys.stdout.write(json_response + '\n')
    sys.stdout.flush()


def handle_mcp_initialize(request_id: int) -> Dict[str, Any]:
    """Handle MCP initialize message."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {
                    "listChanged": True
                }
            },
            "serverInfo": {
                "name": "RAG-CLI",
                "version": "0.1.0"
            }
        }
    }


def handle_mcp_list_tools(request_id: int) -> Dict[str, Any]:
    """Handle MCP list tools request."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "tools": [
                {
                    "name": "start_services",
                    "description": "Start RAG-CLI monitoring services (TCP server and web dashboard)",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                },
                {
                    "name": "get_services_status_tool",
                    "description": "Get the current status of RAG-CLI services",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                },
                {
                    "name": "open_dashboard",
                    "description": "Open the RAG-CLI web dashboard in the default browser",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            ]
        }
    }


def handle_mcp_call_tool(request_id: int, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle MCP tool call request."""
    try:
        if tool_name == "start_services":
            results = ensure_services_running()
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Services started: {json.dumps(results, indent=2)}"
                        }
                    ]
                }
            }
        elif tool_name == "get_services_status_tool":
            status = get_services_status()
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Service status:\n{json.dumps(status, indent=2)}"
                        }
                    ]
                }
            }
        elif tool_name == "open_dashboard":
            open_dashboard_in_browser()
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": "Web dashboard opened in default browser at http://localhost:5000"
                        }
                    ]
                }
            }
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown tool: {tool_name}"
                }
            }
    except Exception as e:
        logger.error(f"Error calling tool {tool_name}: {e}")
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        }


def run_mcp_server():
    """Run the MCP server, reading from stdin and writing to stdout."""
    logger.info("RAG-CLI MCP server starting")

    # Auto-start services when MCP server starts
    logger.info("Auto-starting services...")
    ensure_services_running()

    # Read and process MCP messages from stdin
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                message = json.loads(line)
                request_id = message.get("id", 0)
                method = message.get("method")
                params = message.get("params", {})

                logger.debug(f"MCP request: {method}")

                if method == "initialize":
                    response = handle_mcp_initialize(request_id)
                elif method == "tools/list":
                    response = handle_mcp_list_tools(request_id)
                elif method == "tools/call":
                    tool_name = params.get("name")
                    arguments = params.get("arguments", {})
                    response = handle_mcp_call_tool(request_id, tool_name, arguments)
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32601,
                            "message": f"Unknown method: {method}"
                        }
                    }

                send_mcp_response(response)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
            except Exception as e:
                logger.error(f"Error processing MCP message: {e}")
    except KeyboardInterrupt:
        logger.info("MCP server interrupted")
    except Exception as e:
        logger.error(f"MCP server error: {e}")


def main():
    """Main entry point - runs MCP server."""
    # Check if running as module from Claude Code (MCP mode)
    # When run from Claude Code, stdin is not a TTY
    if not sys.stdin.isatty():
        # Run as MCP server
        run_mcp_server()
    else:
        # Run as CLI tool
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


if __name__ == '__main__':
    main()
