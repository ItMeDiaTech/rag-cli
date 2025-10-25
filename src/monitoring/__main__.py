"""Main entry point for monitoring services.

This module can be run as: python -m src.monitoring
Ensures all monitoring services are running and stays alive.
"""

import sys
import time
import signal
from pathlib import Path

from src.monitoring.logger import get_logger
from src.monitoring.service_manager import ensure_services_running, get_services_status

logger = get_logger(__name__)


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    logger.info("Shutdown signal received")
    sys.exit(0)


def main():
    """Main entry point for monitoring services."""
    logger.info("RAG-CLI Monitoring Services Startup")
    logger.info("=" * 50)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Ensure services are running
    try:
        logger.info("Starting monitoring services...")
        results = ensure_services_running()

        # Display results
        logger.info("Service startup results:")
        for service_name, status in results.items():
            if isinstance(status, bool):
                status_str = "Running" if status else "Failed"
                logger.info(f"  {service_name}: {status_str}")

        # Display service status
        logger.info("Current service status:")
        services_status = get_services_status()
        for service_name, info in services_status.items():
            if isinstance(info, dict):
                running = info.get('running', False)
                status_str = "Online" if running else "Offline"
                logger.info(f"  {info.get('name', service_name)}: {status_str}")
                if info.get('url'):
                    logger.info(f"    URL: {info['url']}")

        logger.info("Monitoring services ready!")
        logger.info("Press Ctrl+C to shutdown...")

        # Keep running
        while True:
            time.sleep(60)
            # Periodically check services are still running
            try:
                services = get_services_status()
                for service, info in services.items():
                    if isinstance(info, dict) and not info.get('running'):
                        logger.warning(f"Service {service} is not running, attempting restart...")
                        ensure_services_running()
            except Exception as e:
                logger.debug(f"Health check error: {e}")

    except KeyboardInterrupt:
        logger.info("Shutdown initiated by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
