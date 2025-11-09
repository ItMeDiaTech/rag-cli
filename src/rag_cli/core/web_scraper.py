"""Web scraper for documentation sites.

This module provides functionality to crawl and extract content from
documentation websites for offline indexing.
"""

import re
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, urlunparse
from collections import deque
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from rag_cli.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ScrapedDocument:
    """Represents a scraped document."""
    url: str
    title: str
    content: str
    metadata: Dict[str, str]
    doc_type: str  # 'official', 'tutorial', 'reference', 'examples'


class WebScraper:
    """Web scraper for documentation sites."""

    def __init__(self, max_depth: int = 3, max_pages: int = 100, delay: float = 1.0):
        """Initialize web scraper.

        Args:
            max_depth: Maximum crawl depth from starting URL
            max_pages: Maximum number of pages to scrape
            delay: Delay between requests in seconds (politeness)
        """
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.delay = delay
        self.session = self._create_session()
        self.visited_urls: Set[str] = set()

    def _create_session(self) -> requests.Session:
        """Create requests session with retry logic.

        Returns:
            Configured session
        """
        session = requests.Session()

        # Retry strategy
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )

        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # User agent
        session.headers.update({
            'User-Agent': 'RAG-CLI Documentation Indexer (Educational Tool)'
        })

        return session

    def scrape_documentation(
        self,
        start_url: str,
        doc_name: str,
        doc_type: str = 'official'
    ) -> List[ScrapedDocument]:
        """Scrape documentation from a starting URL.

        Args:
            start_url: Starting URL for crawling
            doc_name: Name of documentation (e.g., "Python", "Flask")
            doc_type: Type of documentation

        Returns:
            List of scraped documents
        """
        logger.info(f"Starting scrape of {doc_name} from {start_url}")

        # Try sitemap first for efficiency
        documents = self._try_sitemap_scrape(start_url, doc_name, doc_type)

        if documents:
            logger.info(f"Scraped {len(documents)} documents from sitemap")
            return documents

        # Fall back to BFS crawling
        logger.info("No sitemap found, falling back to BFS crawl")
        documents = self._bfs_crawl(start_url, doc_name, doc_type)

        logger.info(f"Scraped {len(documents)} documents via crawling")
        return documents

    def _try_sitemap_scrape(
        self,
        base_url: str,
        doc_name: str,
        doc_type: str
    ) -> List[ScrapedDocument]:
        """Try to scrape using sitemap.xml.

        Args:
            base_url: Base URL of documentation
            doc_name: Name of documentation
            doc_type: Type of documentation

        Returns:
            List of scraped documents, empty if sitemap not found
        """
        parsed = urlparse(base_url)
        sitemap_urls = [
            f"{parsed.scheme}://{parsed.netloc}/sitemap.xml",
            f"{parsed.scheme}://{parsed.netloc}/sitemap_index.xml",
            urljoin(base_url, "sitemap.xml"),
        ]

        for sitemap_url in sitemap_urls:
            try:
                response = self.session.get(sitemap_url, timeout=10)
                if response.status_code == 200:
                    urls = self._parse_sitemap(response.content)

                    # Filter URLs to only those under the documentation path
                    filtered_urls = [
                        url for url in urls
                        if url.startswith(base_url)
                    ][:self.max_pages]

                    logger.info(f"Found {len(filtered_urls)} URLs in sitemap")

                    # Scrape pages from sitemap
                    documents = []
                    for url in filtered_urls:
                        if len(documents) >= self.max_pages:
                            break

                        result = self._scrape_page(url, doc_name, doc_type)
                        if result:
                            doc, soup = result
                            documents.append(doc)

                        time.sleep(self.delay)

                    return documents

            except Exception as e:
                logger.debug(f"Failed to fetch sitemap {sitemap_url}: {e}")
                continue

        return []

    def _parse_sitemap(self, content: bytes) -> List[str]:
        """Parse sitemap XML and extract URLs.

        Args:
            content: Sitemap XML content

        Returns:
            List of URLs
        """
        urls = []

        try:
            root = ET.fromstring(content)

            # Handle sitemap index (links to other sitemaps)
            for sitemap in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap'):
                loc = sitemap.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                if loc is not None and loc.text:
                    # Recursively fetch nested sitemap
                    try:
                        response = self.session.get(loc.text, timeout=10)
                        if response.status_code == 200:
                            urls.extend(self._parse_sitemap(response.content))
                    except Exception as e:
                        logger.warning(f"Failed to fetch nested sitemap {loc.text}: {e}")

            # Handle regular sitemap (URLs)
            for url_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                loc = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                if loc is not None and loc.text:
                    urls.append(loc.text)

        except Exception as e:
            logger.error(f"Failed to parse sitemap: {e}")

        return urls

    def _bfs_crawl(
        self,
        start_url: str,
        doc_name: str,
        doc_type: str
    ) -> List[ScrapedDocument]:
        """Crawl documentation using BFS.

        Args:
            start_url: Starting URL
            doc_name: Name of documentation
            doc_type: Type of documentation

        Returns:
            List of scraped documents
        """
        documents = []
        queue = deque([(start_url, 0)])  # (url, depth)
        self.visited_urls = {start_url}
        base_domain = urlparse(start_url).netloc
        # Get base path - keep only up to version or main docs directory
        # E.g., /en/3.0.x/quickstart/ -> /en/3.0.x/
        parsed_start = urlparse(start_url)
        path_parts = [p for p in parsed_start.path.split('/') if p]
        # Keep first 2-3 path components (e.g., ['en', '3.0.x'])
        base_path = '/' + '/'.join(path_parts[:min(2, len(path_parts))]) + '/' if path_parts else '/'

        while queue and len(documents) < self.max_pages:
            url, depth = queue.popleft()

            if depth > self.max_depth:
                continue

            # Scrape page
            result = self._scrape_page(url, doc_name, doc_type)
            if result:
                doc, soup = result
                documents.append(doc)

                # Extract and queue links
                if depth < self.max_depth:
                    links = self._extract_links(soup, url, base_domain, base_path)
                    for link in links:
                        if link not in self.visited_urls and len(self.visited_urls) < self.max_pages * 2:
                            self.visited_urls.add(link)
                            queue.append((link, depth + 1))

            time.sleep(self.delay)

        return documents

    def _scrape_page(
        self,
        url: str,
        doc_name: str,
        doc_type: str
    ) -> Optional[Tuple[ScrapedDocument, BeautifulSoup]]:
        """Scrape a single page.

        Args:
            url: URL to scrape
            doc_name: Name of documentation
            doc_type: Type of documentation

        Returns:
            Tuple of (scraped document, soup) or None if failed
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract title
            title = self._extract_title(soup, url)

            # Extract main content
            content = self._extract_content(soup)

            if not content or len(content) < 100:
                logger.debug(f"Skipping {url}: insufficient content")
                return None

            # Create metadata
            metadata = {
                'source': doc_name,
                'url': url,
                'doc_type': doc_type,
                'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            doc = ScrapedDocument(
                url=url,
                title=title,
                content=content,
                metadata=metadata,
                doc_type=doc_type
            )

            return (doc, soup)

        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")
            return None

    def _extract_title(self, soup: BeautifulSoup, url: str) -> str:
        """Extract page title.

        Args:
            soup: BeautifulSoup object
            url: Page URL

        Returns:
            Page title
        """
        # Try common title locations
        title_candidates = [
            soup.find('h1'),
            soup.find('title'),
            soup.find('meta', {'property': 'og:title'}),
            soup.find('meta', {'name': 'title'})
        ]

        for candidate in title_candidates:
            if candidate:
                if candidate.name == 'meta':
                    title = candidate.get('content', '')
                else:
                    title = candidate.get_text(strip=True)

                if title:
                    return title

        # Fallback to URL path
        path = urlparse(url).path.strip('/').split('/')[-1]
        return path.replace('-', ' ').replace('_', ' ').title()

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from page.

        Args:
            soup: BeautifulSoup object

        Returns:
            Extracted text content
        """
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()

        # Try to find main content container
        content_selectors = [
            {'role': 'main'},
            {'id': 'main'},
            {'id': 'content'},
            {'class': 'content'},
            {'class': 'main'},
            {'class': 'documentation'},
            {'class': 'docs'},
            {'class': 'article'},
            {'tag': 'main'},
            {'tag': 'article'},
        ]

        main_content = None
        for selector in content_selectors:
            if 'tag' in selector:
                main_content = soup.find(selector['tag'])
            else:
                main_content = soup.find(**selector)

            if main_content:
                break

        # Fall back to body if no main content found
        if not main_content:
            main_content = soup.find('body')

        if not main_content:
            return ""

        # Extract text, preserving structure
        text = main_content.get_text(separator='\n', strip=True)

        # Clean up excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        return text

    def _extract_links(
        self,
        soup: BeautifulSoup,
        base_url: str,
        base_domain: str,
        base_path: str
    ) -> List[str]:
        """Extract links from page.

        Args:
            soup: BeautifulSoup object
            base_url: Base URL for resolving relative links
            base_domain: Base domain to restrict crawling
            base_path: Base path to restrict crawling to documentation section

        Returns:
            List of absolute URLs
        """
        links = []

        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']

            # Skip anchors, javascript, and mailto links
            if href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
                continue

            # Convert to absolute URL
            absolute_url = urljoin(base_url, href)

            # Parse URL
            parsed = urlparse(absolute_url)

            # Only include links from same domain
            if parsed.netloc != base_domain:
                continue

            # Only include links under the same documentation path
            if base_path and not parsed.path.startswith(base_path):
                continue

            # Remove fragment
            clean_url = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                ''  # Remove fragment
            ))

            # Skip common non-documentation pages
            skip_patterns = [
                '/search', '/login', '/signup', '/download',
                '.pdf', '.zip', '.tar', '.gz',
                '/api/v', '/api-', '/blog', '/news'
            ]

            if any(pattern in clean_url.lower() for pattern in skip_patterns):
                continue

            if clean_url not in links:
                links.append(clean_url)

        return links


class DocumentationScraperFactory:
    """Factory for creating site-specific scrapers."""

    @staticmethod
    def create_scraper(doc_name: str, url: str) -> WebScraper:
        """Create a scraper configured for specific documentation site.

        Args:
            doc_name: Name of documentation
            url: Documentation URL

        Returns:
            Configured WebScraper
        """
        # Site-specific configurations
        configs = {
            'Python': {'max_depth': 2, 'max_pages': 50, 'delay': 0.5},
            'Flask': {'max_depth': 2, 'max_pages': 30, 'delay': 0.5},
            'Django': {'max_depth': 2, 'max_pages': 50, 'delay': 0.5},
            'FastAPI': {'max_depth': 2, 'max_pages': 30, 'delay': 0.5},
            'React': {'max_depth': 2, 'max_pages': 40, 'delay': 0.5},
            'Vue.js': {'max_depth': 2, 'max_pages': 40, 'delay': 0.5},
            'Angular': {'max_depth': 2, 'max_pages': 40, 'delay': 0.5},
            'LangChain': {'max_depth': 2, 'max_pages': 50, 'delay': 0.5},
            'Anthropic SDK': {'max_depth': 2, 'max_pages': 30, 'delay': 0.5},
            'NumPy': {'max_depth': 2, 'max_pages': 40, 'delay': 0.5},
            'Pandas': {'max_depth': 2, 'max_pages': 40, 'delay': 0.5},
        }

        config = configs.get(doc_name, {'max_depth': 2, 'max_pages': 30, 'delay': 1.0})

        return WebScraper(**config)


def scrape_and_save(
    url: str,
    doc_name: str,
    doc_type: str,
    output_dir: Path
) -> List[ScrapedDocument]:
    """Scrape documentation and save to disk.

    Args:
        url: Starting URL
        doc_name: Name of documentation
        doc_type: Type of documentation
        output_dir: Output directory for scraped content

    Returns:
        List of scraped documents
    """
    scraper = DocumentationScraperFactory.create_scraper(doc_name, url)
    documents = scraper.scrape_documentation(url, doc_name, doc_type)

    # Save documents to disk
    doc_dir = output_dir / doc_name.lower().replace(' ', '_')
    doc_dir.mkdir(parents=True, exist_ok=True)

    for i, doc in enumerate(documents):
        # Create filename from URL
        url_hash = hashlib.blake2b(doc.url.encode(), digest_size=8).hexdigest()
        filename = f"{i:04d}_{url_hash}.txt"

        filepath = doc_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Title: {doc.title}\n")
            f.write(f"URL: {doc.url}\n")
            f.write(f"Type: {doc.doc_type}\n")
            f.write(f"Source: {doc.metadata['source']}\n")
            f.write(f"Scraped: {doc.metadata['scraped_at']}\n")
            f.write("\n" + "=" * 80 + "\n\n")
            f.write(doc.content)

    logger.info(f"Saved {len(documents)} documents to {doc_dir}")

    return documents
