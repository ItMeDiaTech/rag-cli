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
