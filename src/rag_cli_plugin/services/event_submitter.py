"""Event submission service for TCP server communication.

This module handles communication between hooks and the TCP monitoring server,
with intelligent caching and exponential backoff to minimize performance impact.
"""

import time
import json
from typing import Dict, Any, Optional
from rag_cli.core.constants import TCP_CHECK_CACHE_SECONDS


class EventSubmitter:
    """Handles event submission to TCP server with intelligent backoff."""

    def __init__(self, tcp_server_url: str = "http://localhost:9999"):
        """Initialize event submitter.

        Args:
            tcp_server_url: URL of the TCP monitoring server
        """
        self.tcp_server_url = tcp_server_url

        # Cache state
        self._server_available: Optional[bool] = None
        self._check_time: float = 0
        self._consecutive_failures: int = 0
        self._backoff_until: float = 0

    def is_server_available(self) -> bool:
        """Check if TCP server is available with exponential backoff.

        Implements exponential backoff: after consecutive failures, wait progressively
        longer before retrying (30s, 60s, 120s, max 240s).

        Returns:
            True if server is reachable, False otherwise
        """
        current_time = time.time()

        # Check if in backoff period
        if current_time < self._backoff_until:
            return False

        # Use cached result if check was recent
        if self._server_available is not None and (current_time - self._check_time) < TCP_CHECK_CACHE_SECONDS:
            return self._server_available

        # Try to connect to TCP server
        try:
            import urllib.request
            import urllib.error

            req = urllib.request.Request(
                f"{self.tcp_server_url}/api/health",
                method='GET'
            )

            with urllib.request.urlopen(req, timeout=0.5) as response:
                # Success - reset failure count
                self._server_available = (response.status == 200)
                self._check_time = current_time
                self._consecutive_failures = 0
                self._backoff_until = 0
                return self._server_available

        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError, OSError):
            # Network/connection errors are expected when server is not running
            self._server_available = False
            self._check_time = current_time

            # Increment failure count and calculate backoff
            self._consecutive_failures += 1
            backoff_seconds = min(TCP_CHECK_CACHE_SECONDS * (2 ** (self._consecutive_failures - 1)), 240)
            self._backoff_until = current_time + backoff_seconds

            return False

    def submit_event(self, event_type: str, data: Dict[str, Any]) -> bool:
        """Submit an event to the TCP server.

        Args:
            event_type: Type of event (activity, reasoning, query_enhancement, etc.)
            data: Event data dictionary

        Returns:
            True if successful, False otherwise
        """
        # Check if server is available before attempting connection
        if not self.is_server_available():
            return False

        try:
            import urllib.request
            import urllib.error

            event_payload = {
                "event_type": event_type,
                "data": data
            }

            req = urllib.request.Request(
                f"{self.tcp_server_url}/api/events/submit",
                data=json.dumps(event_payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=1) as response:
                return response.status == 200

        except (urllib.error.URLError, urllib.error.HTTPError, Exception):
            # Mark server as unavailable on error
            self._server_available = False
            return False

    def reset_backoff(self):
        """Reset backoff state (useful for testing or manual retry)."""
        self._consecutive_failures = 0
        self._backoff_until = 0
        self._server_available = None


# Singleton instance
_event_submitter: Optional[EventSubmitter] = None


def get_event_submitter(tcp_server_url: str = "http://localhost:9999") -> EventSubmitter:
    """Get or create the global event submitter instance.

    Args:
        tcp_server_url: URL of the TCP monitoring server

    Returns:
        EventSubmitter instance
    """
    global _event_submitter

    if _event_submitter is None:
        _event_submitter = EventSubmitter(tcp_server_url)

    return _event_submitter
