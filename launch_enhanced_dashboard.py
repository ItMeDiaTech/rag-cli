#!/usr/bin/env python3
"""
Enhanced Dashboard Launcher
Quick launch script for the RAG-CLI Enhanced Monitoring Dashboard
"""

import os
import sys
import subprocess
import webbrowser
import time
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def check_dependencies():
    """Check if required dependencies are installed"""
    required = ['flask', 'flask_cors', 'requests']
    missing = []

    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"pip install {' '.join(missing)}")
        return False

    return True


def check_port_available(port):
    """Check if a port is available"""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('localhost', port))
        sock.close()
        return True
    except OSError:
        return False


def start_dashboard(port=5000, simulation=True, open_browser=True, debug=False):
    """Start the enhanced dashboard"""

    if not check_dependencies():
        return False

    if not check_port_available(port):
        print(f"Port {port} is already in use!")
        print(f"Stop the process using port {port} or use --port to specify a different port")
        return False

    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  RAG-CLI Enhanced Dashboard                                   ║
║  Multi-Agent Orchestration & RAG Pipeline Monitor             ║
╚═══════════════════════════════════════════════════════════════╝

Starting dashboard on http://localhost:{port}

Features:
  • Real-time agent orchestration visualization
  • Cost and token tracking per agent
  • RAG pipeline performance monitoring
  • Decision tree and reasoning display
  • Interactive agent graph (D3.js)
  • Message flow visualization
  • Comprehensive metrics and analytics

Mode: {'Simulation (Demo Data)' if simulation else 'Production'}
""")

    # Set environment variables
    os.environ['RAG_DASHBOARD_PORT'] = str(port)

    # Import and run dashboard
    try:
        from src.monitoring.enhanced_web_dashboard import app, simulate_data, metrics_collector

        # Start simulation if requested
        if simulation:
            print("Starting data simulation...")
            simulate_data()

        # Open browser
        if open_browser:
            print(f"\nOpening browser...")
            time.sleep(1)
            webbrowser.open(f'http://localhost:{port}')

        print(f"\nDashboard is running on http://localhost:{port}")
        print("Press Ctrl+C to stop\n")

        # Run Flask app
        app.run(
            host='0.0.0.0',
            port=port,
            debug=debug,
            threaded=True
        )

    except KeyboardInterrupt:
        print("\n\nShutting down dashboard...")
        return True
    except Exception as e:
        print(f"\nError starting dashboard: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Launch RAG-CLI Enhanced Monitoring Dashboard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Start with simulation (default)
  python launch_enhanced_dashboard.py

  # Start in production mode (no simulation)
  python launch_enhanced_dashboard.py --no-simulation

  # Custom port
  python launch_enhanced_dashboard.py --port 8080

  # Don't open browser automatically
  python launch_enhanced_dashboard.py --no-browser

  # Enable Flask debug mode
  python launch_enhanced_dashboard.py --debug
        '''
    )

    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to run dashboard on (default: 5000)'
    )

    parser.add_argument(
        '--no-simulation',
        action='store_true',
        help='Disable data simulation (production mode)'
    )

    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not open browser automatically'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable Flask debug mode'
    )

    args = parser.parse_args()

    success = start_dashboard(
        port=args.port,
        simulation=not args.no_simulation,
        open_browser=not args.no_browser,
        debug=args.debug
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
