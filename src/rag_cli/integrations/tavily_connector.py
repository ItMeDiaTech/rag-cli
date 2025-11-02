"""Tavily Search API connector for web search.

This connector provides access to Tavily's AI-optimized search API
with quota tracking for the free tier (1,000 searches/month) and
graceful fallback when quota is exhausted.
"""

import os
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import requests

from rag_cli.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TavilyResult:
    """Represents a Tavily search result."""
    title: str
    url: str
    content: str
    score: float
    published_date: Optional[str] = None


class TavilyConnector:
    """Connector for Tavily Search API with quota tracking."""

    API_URL = "https://api.tavily.com/search"
    FREE_TIER_LIMIT = 1000  # Monthly limit for free tier
    WARN_THRESHOLD = 900     # Warn at 90%

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Tavily connector with quota tracking.

        Args:
            api_key: Tavily API key (reads from env TAVILY_API_KEY if not provided)
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.quota_file = Path("config/tavily_usage.json")
        self.rate_limit_delay = 6.0  # 10 requests/minute = 6s between requests

        self.last_request_time = 0.0
        self.enabled = bool(self.api_key)

        if not self.enabled:
            logger.warning("Tavily API key not found - connector disabled. "
                           "Set TAVILY_API_KEY environment variable to enable.")
        else:
            logger.info("Tavily connector initialized",
                        quota_limit=self.FREE_TIER_LIMIT,
                        warn_threshold=self.WARN_THRESHOLD)

        # Ensure quota file exists
        self._init_quota_file()

    def _init_quota_file(self):
        """Initialize quota tracking file if it doesn't exist."""
        self.quota_file.parent.mkdir(parents=True, exist_ok=True)

        if not self.quota_file.exists():
            initial_data = {
                "month": datetime.now().strftime("%Y-%m"),
                "searches": 0,
                "last_reset": datetime.now().isoformat()
            }
            self.quota_file.write_text(json.dumps(initial_data, indent=2))
            logger.info("Created Tavily quota tracking file")

    def _get_usage(self) -> Dict[str, Any]:
        """Get current usage statistics.

        Returns:
            Dictionary with month, searches, last_reset
        """
        try:
            data = json.loads(self.quota_file.read_text())

            # Check if we need to reset for new month
            current_month = datetime.now().strftime("%Y-%m")
            if data.get("month") != current_month:
                data = {
                    "month": current_month,
                    "searches": 0,
                    "last_reset": datetime.now().isoformat()
                }
                self.quota_file.write_text(json.dumps(data, indent=2))
                logger.info("Tavily quota reset for new month", month=current_month)

            return data

        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Failed to read quota file: {e}")
            self._init_quota_file()
            return self._get_usage()

    def _increment_usage(self):
        """Increment usage counter."""
        data = self._get_usage()
        data["searches"] += 1
        self.quota_file.write_text(json.dumps(data, indent=2))

        # Log warning if approaching limit
        if data["searches"] == self.WARN_THRESHOLD:
            remaining = self.FREE_TIER_LIMIT - data["searches"]
            logger.warning(f"Tavily quota warning: {data['searches']}/{self.FREE_TIER_LIMIT} "
                           f"searches used. {remaining} remaining this month.")

    def _rate_limit(self):
        """Enforce rate limiting (10 requests/minute)."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed
            logger.debug(f"Tavily rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def is_quota_available(self) -> bool:
        """Check if quota is available for this month.

        Returns:
            True if under limit, False otherwise
        """
        if not self.enabled:
            return False

        usage = self._get_usage()
        return usage["searches"] < self.FREE_TIER_LIMIT

    def get_remaining_quota(self) -> int:
        """Get remaining searches for current month.

        Returns:
            Number of searches remaining
        """
        usage = self._get_usage()
        return max(0, self.FREE_TIER_LIMIT - usage["searches"])

    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None
    ) -> List[TavilyResult]:
        """Search using Tavily API.

        Args:
            query: Search query
            max_results: Maximum number of results (default: 5)
            search_depth: Search depth - "basic" or "advanced" (default: basic)
            include_domains: List of domains to include
            exclude_domains: List of domains to exclude

        Returns:
            List of TavilyResult objects (empty if quota exceeded or disabled)
        """
        # Check if enabled
        if not self.enabled:
            logger.debug("Tavily disabled (no API key)")
            return []

        # Check quota
        if not self.is_quota_available():
            usage = self._get_usage()
            logger.warning(f"Tavily quota exceeded: {usage['searches']}/{self.FREE_TIER_LIMIT} "
                           "searches used this month. Falling back to other sources.")
            return []

        # Rate limit
        self._rate_limit()

        try:
            # Build request
            payload = {
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                "include_answer": False,  # We use our own LLM for answers
                "include_raw_content": False  # Don't need full HTML
            }

            if include_domains:
                payload["include_domains"] = include_domains
            if exclude_domains:
                payload["exclude_domains"] = exclude_domains

            logger.info("Searching Tavily",
                        query=query,
                        max_results=max_results,
                        remaining_quota=self.get_remaining_quota())

            # Make request
            response = requests.post(
                self.API_URL,
                json=payload,
                timeout=10
            )
            response.raise_for_status()

            # Parse response
            data = response.json()
            results = self._parse_response(data)

            # Increment usage counter
            self._increment_usage()

            logger.info("Tavily search completed",
                        query=query,
                        results=len(results),
                        remaining_quota=self.get_remaining_quota())

            return results

        except requests.RequestException as e:
            logger.error(f"Tavily API request failed: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Tavily response parsing failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Tavily search failed: {e}", exc_info=True)
            return []

    def _parse_response(self, data: Dict[str, Any]) -> List[TavilyResult]:
        """Parse Tavily API response.

        Args:
            data: JSON response from Tavily API

        Returns:
            List of TavilyResult objects
        """
        results = []

        try:
            for item in data.get("results", []):
                result = TavilyResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                    score=item.get("score", 0.0),
                    published_date=item.get("published_date")
                )
                results.append(result)

        except (KeyError, TypeError) as e:
            logger.error(f"Failed to parse Tavily result: {e}")

        return results

    def to_retrieval_results(self, results: List[TavilyResult]) -> List[Dict[str, Any]]:
        """Convert Tavily results to standard retrieval result format.

        Args:
            results: List of TavilyResult objects

        Returns:
            List of dictionaries in RAG retrieval format
        """
        retrieval_results = []

        for result in results:
            retrieval_result = {
                "source": f"Web: {result.url}",
                "title": result.title,
                "content": result.content,
                "url": result.url,
                "score": result.score,
                "metadata": {
                    "source_type": "web_search",
                    "search_engine": "tavily",
                    "published_date": result.published_date
                }
            }
            retrieval_results.append(retrieval_result)

        return retrieval_results


# Singleton instance
_tavily_connector: Optional[TavilyConnector] = None


def get_tavily_connector() -> TavilyConnector:
    """Get or create the global Tavily connector instance.

    Returns:
        Tavily connector instance
    """
    global _tavily_connector

    if _tavily_connector is None:
        _tavily_connector = TavilyConnector()

    return _tavily_connector
