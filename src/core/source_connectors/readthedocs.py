"""ReadTheDocs and official documentation scraper.

Scrapes documentation from ReadTheDocs, DevDocs, and official language documentation sites.
"""

import requests
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import time
from urllib.parse import urljoin, urlparse
import trafilatura
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class Documentation:
    """Represents a documentation page."""
    title: str
    content: str
    url: str
    source: str  # 'readthedocs', 'devdocs', 'official'
    framework: Optional[str] = None
    version: Optional[str] = None
    section: Optional[str] = None
    language: Optional[str] = None
    last_updated: Optional[datetime] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ReadTheDocsConnector:
    """Connector for ReadTheDocs and official documentation sites."""

    def __init__(self, rate_limit: int = 100, timeout: int = 15):
        """Initialize ReadTheDocs connector.

        Args:
            rate_limit: Maximum requests per hour
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.rate_limit = rate_limit
        self.requests_made = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; RAG-CLI-Documentation-Bot/1.0)'
        })

    def _wait_if_needed(self):
        """Simple rate limiting."""
        now = time.time()
        cutoff = now - 3600  # 1 hour window

        # Remove old requests
        self.requests_made = [t for t in self.requests_made if t > cutoff]

        if len(self.requests_made) >= self.rate_limit:
            wait_time = 3600 - (now - self.requests_made[0])
            if wait_time > 0:
                logger.warning(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)
                self.requests_made = []

        self.requests_made.append(now)

    def fetch_page(self, url: str) -> Optional[Documentation]:
        """Fetch and extract content from a documentation page.

        Args:
            url: URL of documentation page

        Returns:
            Documentation object or None if error
        """
        self._wait_if_needed()

        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            html_content = response.text

            # Use trafilatura for main content extraction
            extracted = trafilatura.extract(
                html_content,
                include_links=True,
                include_tables=True,
                include_comments=False,
                output_format='markdown'
            )

            if not extracted:
                logger.warning(f"Could not extract content from {url}")
                return None

            # Extract title
            soup = BeautifulSoup(html_content, 'html.parser')
            title_tag = soup.find('title')
            title = title_tag.get_text() if title_tag else url

            # Determine source type
            source = self._determine_source(url)

            # Extract metadata
            metadata = self._extract_metadata(soup, url)

            return Documentation(
                title=title.strip(),
                content=extracted,
                url=url,
                source=source,
                framework=metadata.get('framework'),
                version=metadata.get('version'),
                section=metadata.get('section'),
                language=metadata.get('language'),
                metadata=metadata
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching page {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing page {url}: {e}")
            return None

    def search_readthedocs(self, project: str, query: str, version: str = "latest") -> List[str]:
        """Search ReadTheDocs project for pages matching query.

        Args:
            project: ReadTheDocs project name
            query: Search query
            version: Documentation version (default: latest)

        Returns:
            List of URLs matching query
        """
        base_url = f"https://{project}.readthedocs.io/en/{version}/"

        # Try to fetch search results
        search_url = f"{base_url}search.html?q={query}"

        try:
            response = self.session.get(search_url, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract result links
            urls = []
            for link in soup.find_all('a', class_='reference'):
                href = link.get('href')
                if href and not href.startswith('#'):
                    full_url = urljoin(base_url, href)
                    if full_url not in urls:
                        urls.append(full_url)

            return urls[:10]  # Limit results

        except Exception as e:
            logger.error(f"Error searching ReadTheDocs: {e}")
            return []

    def get_table_of_contents(self, project: str, version: str = "latest") -> List[str]:
        """Get all pages from ReadTheDocs project table of contents.

        Args:
            project: ReadTheDocs project name
            version: Documentation version

        Returns:
            List of documentation page URLs
        """
        base_url = f"https://{project}.readthedocs.io/en/{version}/"

        try:
            response = self.session.get(base_url, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find TOC (usually in sidebar)
            toc = soup.find('div', class_='toctree-wrapper') or soup.find('nav', class_='wy-nav-side')

            urls = []
            if toc:
                for link in toc.find_all('a'):
                    href = link.get('href')
                    if href and not href.startswith('#'):
                        full_url = urljoin(base_url, href)
                        if full_url not in urls:
                            urls.append(full_url)

            return urls

        except Exception as e:
            logger.error(f"Error getting TOC for {project}: {e}")
            return []

    def fetch_official_docs(self, language: str, topic: str) -> List[Documentation]:
        """Fetch official documentation for a language/framework.

        Args:
            language: Programming language (python, javascript, etc.)
            topic: Topic or module to search for

        Returns:
            List of Documentation objects
        """
        # Map of languages to their official docs base URLs
        official_docs = {
            'python': f'https://docs.python.org/3/search.html?q={topic}',
            'javascript': f'https://developer.mozilla.org/en-US/search?q={topic}',
            'typescript': f'https://www.typescriptlang.org/docs/handbook/{topic}',
            'rust': f'https://doc.rust-lang.org/std/?search={topic}',
            'go': f'https://golang.org/search?q={topic}',
            'java': f'https://docs.oracle.com/en/java/javase/17/docs/api/java.base/module-summary.html'
        }

        language = language.lower()
        if language not in official_docs:
            logger.warning(f"No official docs configured for {language}")
            return []

        # For now, just fetch the main page
        # In a full implementation, would parse search results and fetch multiple pages
        docs = []
        url = official_docs[language]

        doc = self.fetch_page(url)
        if doc:
            doc.language = language
            docs.append(doc)

        return docs

    def _determine_source(self, url: str) -> str:
        """Determine the source type from URL.

        Args:
            url: Page URL

        Returns:
            Source type (readthedocs, devdocs, official)
        """
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        if 'readthedocs.io' in domain or 'readthedocs.org' in domain:
            return 'readthedocs'
        elif 'devdocs.io' in domain:
            return 'devdocs'
        elif any(official in domain for official in ['docs.python.org', 'developer.mozilla.org',
                                                       'doc.rust-lang.org', 'golang.org',
                                                       'docs.oracle.com']):
            return 'official'
        else:
            return 'web'

    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract metadata from documentation page.

        Args:
            soup: BeautifulSoup object of the page
            url: Page URL

        Returns:
            Dictionary of metadata
        """
        metadata = {}

        # Try to extract version
        version_meta = soup.find('meta', attrs={'name': 'version'})
        if version_meta:
            metadata['version'] = version_meta.get('content')
        else:
            # Try to extract from URL
            parts = url.split('/')
            for i, part in enumerate(parts):
                if part in ['en', 'latest', 'stable', 'v1', 'v2'] and i + 1 < len(parts):
                    metadata['version'] = parts[i + 1]
                    break

        # Extract framework/project name
        parsed = urlparse(url)
        if 'readthedocs' in parsed.netloc:
            # Project name is subdomain
            project = parsed.netloc.split('.')[0]
            metadata['framework'] = project

        # Try to extract section
        breadcrumbs = soup.find('ul', class_='breadcrumb') or soup.find('nav', attrs={'aria-label': 'breadcrumb'})
        if breadcrumbs:
            sections = [link.get_text().strip() for link in breadcrumbs.find_all('a')]
            if sections:
                metadata['section'] = ' > '.join(sections)

        # Extract language from meta tags
        lang_meta = soup.find('html', attrs={'lang': True})
        if lang_meta:
            metadata['doc_language'] = lang_meta.get('lang')

        return metadata

    def fetch_devdocs(self, technology: str, topic: str) -> List[Documentation]:
        """Fetch documentation from DevDocs.io.

        Args:
            technology: Technology name (e.g., 'python', 'javascript')
            topic: Topic to search for

        Returns:
            List of Documentation objects
        """
        # DevDocs.io has a specific URL structure
        base_url = f"https://devdocs.io/{technology}/"

        # Search for topic
        search_url = f"{base_url}?q={topic}"

        doc = self.fetch_page(search_url)
        if doc:
            doc.source = 'devdocs'
            doc.framework = technology
            return [doc]

        return []

    def bulk_fetch(self, urls: List[str], max_pages: int = 50) -> List[Documentation]:
        """Fetch multiple documentation pages.

        Args:
            urls: List of URLs to fetch
            max_pages: Maximum number of pages to fetch

        Returns:
            List of Documentation objects
        """
        docs = []

        for url in urls[:max_pages]:
            doc = self.fetch_page(url)
            if doc:
                docs.append(doc)

            # Be polite with rate limiting
            time.sleep(0.5)

        return docs


class OfficialDocsConnector(ReadTheDocsConnector):
    """Specialized connector for official language documentation.

    Inherits from ReadTheDocsConnector but adds language-specific logic.
    """

    def __init__(self, allowed_domains: List[str], **kwargs):
        """Initialize connector.

        Args:
            allowed_domains: List of allowed documentation domains
            **kwargs: Additional arguments for parent class
        """
        super().__init__(**kwargs)
        self.allowed_domains = allowed_domains

    def is_allowed_domain(self, url: str) -> bool:
        """Check if URL is from an allowed domain.

        Args:
            url: URL to check

        Returns:
            True if domain is allowed
        """
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        return any(allowed in domain for allowed in self.allowed_domains)

    def fetch_page(self, url: str) -> Optional[Documentation]:
        """Fetch page only if from allowed domain.

        Args:
            url: URL to fetch

        Returns:
            Documentation object or None
        """
        if not self.is_allowed_domain(url):
            logger.warning(f"URL not from allowed domain: {url}")
            return None

        return super().fetch_page(url)
