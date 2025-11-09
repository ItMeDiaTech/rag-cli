"""ArXiv API connector for academic paper retrieval.

This connector provides access to ArXiv's free academic paper database
with rate limiting to comply with API guidelines (3 requests/second).
"""

import time
import threading
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import OrderedDict
import requests

from rag_cli.core.constants import DEFAULT_HTTP_TIMEOUT
from rag_cli.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ArXivPaper:
    """Represents an ArXiv paper."""
    id: str
    title: str
    authors: List[str]
    abstract: str
    pdf_url: str
    published: datetime
    categories: List[str]
    relevance_score: float = 0.0


class ArXivConnector:
    """Connector for ArXiv API with rate limiting."""

    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(self, rate_limit_delay: float = 0.35):
        """Initialize ArXiv connector with rate limiting.

        Args:
            rate_limit_delay: Delay between requests in seconds (default: 0.35s = 2.86 req/s)
        """
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0
        # Bounded cache with LRU eviction to prevent memory leaks
        self.cache = OrderedDict()
        self.max_cache_size = 100  # Keep last 100 search results
        self.cache_ttl = timedelta(days=30)  # Cache papers for 30 days

        logger.info("ArXiv connector initialized",
                    rate_limit=f"{1 / rate_limit_delay:.1f} req/sec",
                    cache_ttl=f"{self.cache_ttl.days} days",
                    max_cache_size=self.max_cache_size)

    def _rate_limit(self):
        """Enforce rate limiting to stay under 3 requests/second."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.3f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _build_query_string(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance",
        categories: Optional[List[str]] = None
    ) -> str:
        """Build ArXiv API query string.

        Args:
            query: Search query
            max_results: Maximum number of results
            sort_by: Sort order (relevance, lastUpdatedDate, submittedDate)
            categories: Filter by categories (e.g., ['cs.AI', 'cs.LG'])

        Returns:
            Query string for ArXiv API
        """
        # Build search query
        search_query = f"all:{query}"

        # Add category filters if specified
        if categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
            search_query = f"({search_query}) AND ({cat_query})"

        return search_query

    def search(
        self,
        query: str,
        max_results: int = 5,
        categories: Optional[List[str]] = None,
        sort_by: str = "relevance"
    ) -> List[ArXivPaper]:
        """Search ArXiv for papers.

        Args:
            query: Search query
            max_results: Maximum number of results (default: 5)
            categories: Filter by categories (e.g., ['cs.AI', 'cs.LG'])
            sort_by: Sort order (relevance, lastUpdatedDate, submittedDate)

        Returns:
            List of ArXivPaper objects
        """
        # Check cache first with LRU tracking
        cache_key = f"{query}_{max_results}_{categories}_{sort_by}"
        if cache_key in self.cache:
            cached_papers, cache_time = self.cache[cache_key]
            if datetime.now() - cache_time < self.cache_ttl:
                # Move to end to mark as recently used (O(1) with OrderedDict)
                self.cache.move_to_end(cache_key)
                logger.debug("ArXiv cache hit", query=query, count=len(cached_papers))
                return cached_papers
            else:
                # Expired, remove from cache
                del self.cache[cache_key]

        # Rate limit
        self._rate_limit()

        try:
            # Build query
            search_query = self._build_query_string(query, max_results, sort_by, categories)

            # Make request
            params = {
                "search_query": search_query,
                "max_results": max_results,
                "sortBy": sort_by,
                "sortOrder": "descending"
            }

            logger.info("Searching ArXiv", query=query, max_results=max_results)

            response = requests.get(
                self.BASE_URL,
                params=params,
                timeout=DEFAULT_HTTP_TIMEOUT
            )
            response.raise_for_status()

            # Parse XML response
            papers = self._parse_response(response.text)

            # Cache results with LRU eviction
            self.cache[cache_key] = (papers, datetime.now())
            self.cache.move_to_end(cache_key)  # Mark as recently used

            # Evict oldest if over size - O(1) with OrderedDict
            if len(self.cache) > self.max_cache_size:
                oldest_key, _ = self.cache.popitem(last=False)
                logger.debug("ArXiv cache eviction", evicted_key=oldest_key[:50])

            logger.info("ArXiv search completed",
                        query=query,
                        results=len(papers))

            return papers

        except requests.RequestException as e:
            logger.error(f"ArXiv API request failed: {e}")
            return []
        except ET.ParseError as e:
            logger.error(f"ArXiv response parsing failed: {e}")
            return []
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}", exc_info=True)
            return []

    def _parse_response(self, xml_text: str) -> List[ArXivPaper]:
        """Parse ArXiv API XML response.

        Args:
            xml_text: XML response from ArXiv API

        Returns:
            List of ArXivPaper objects
        """
        papers = []

        try:
            root = ET.fromstring(xml_text)

            # Define namespaces
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }

            # Parse each entry
            for entry in root.findall('atom:entry', ns):
                try:
                    # Extract paper ID
                    paper_id = entry.find('atom:id', ns).text.split('/')[-1]

                    # Extract title
                    title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')

                    # Extract authors
                    authors = [
                        author.find('atom:name', ns).text
                        for author in entry.findall('atom:author', ns)
                    ]

                    # Extract abstract
                    abstract = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')

                    # Extract PDF URL
                    pdf_link = None
                    for link in entry.findall('atom:link', ns):
                        if link.get('title') == 'pdf':
                            pdf_link = link.get('href')
                            break
                    if not pdf_link:
                        pdf_link = f"http://arxiv.org/pdf/{paper_id}"

                    # Extract published date
                    published_str = entry.find('atom:published', ns).text
                    published = datetime.fromisoformat(published_str.replace('Z', '+00:00'))

                    # Extract categories
                    categories = [
                        cat.get('term')
                        for cat in entry.findall('atom:category', ns)
                    ]

                    paper = ArXivPaper(
                        id=paper_id,
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        pdf_url=pdf_link,
                        published=published,
                        categories=categories
                    )

                    papers.append(paper)

                except (AttributeError, ValueError) as e:
                    logger.warning(f"Failed to parse ArXiv entry: {e}")
                    continue

        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")

        return papers

    def to_retrieval_results(self, papers: List[ArXivPaper]) -> List[Dict[str, Any]]:
        """Convert ArXiv papers to standard retrieval result format.

        Args:
            papers: List of ArXivPaper objects

        Returns:
            List of dictionaries in RAG retrieval format
        """
        results = []

        for paper in papers:
            result = {
                "source": f"ArXiv: {paper.id}",
                "title": paper.title,
                "content": paper.abstract,
                "url": paper.pdf_url,
                "score": paper.relevance_score,
                "metadata": {
                    "authors": ", ".join(paper.authors),
                    "published": paper.published.strftime("%Y-%m-%d"),
                    "categories": ", ".join(paper.categories),
                    "source_type": "academic_paper",
                    "paper_id": paper.id
                }
            }
            results.append(result)

        return results


# Singleton instance
_arxiv_connector: Optional[ArXivConnector] = None
_arxiv_lock = threading.Lock()


def get_arxiv_connector() -> ArXivConnector:
    """Get or create the global ArXiv connector instance with thread-safe initialization.

    Returns:
        ArXiv connector instance
    """
    global _arxiv_connector

    if _arxiv_connector is None:
        with _arxiv_lock:
            if _arxiv_connector is None:
                _arxiv_connector = ArXivConnector()

    return _arxiv_connector
