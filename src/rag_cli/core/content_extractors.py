"""Content extraction and cleaning utilities for online documentation.

Provides tools to extract clean, structured content from HTML pages and convert to markdown.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from bs4 import BeautifulSoup
import trafilatura

logger = logging.getLogger(__name__)


class ContentExtractor:
    """Extract and clean content from HTML pages."""

    def __init__(self, max_page_size_kb: int = 500, extract_code_blocks: bool = True,
                 preserve_links: bool = True, clean_html: bool = True):
        """Initialize content extractor.

        Args:
            max_page_size_kb: Maximum page size to process in kilobytes
            extract_code_blocks: Whether to extract and preserve code blocks
            preserve_links: Whether to keep links in output
            clean_html: Whether to clean HTML before processing
        """
        self.max_page_size_kb = max_page_size_kb
        self.extract_code_blocks = extract_code_blocks
        self.preserve_links = preserve_links
        self.clean_html = clean_html

    def extract_from_html(self, html: str, url: Optional[str] = None) -> Optional[str]:
        """Extract clean content from HTML.

        Args:
            html: HTML content
            url: Optional URL for context

        Returns:
            Extracted content in markdown format, or None if error
        """
        # Check size
        size_kb = len(html.encode('utf-8')) / 1024
        if size_kb > self.max_page_size_kb:
            logger.warning(f"Page too large ({size_kb:.1f}KB), skipping")
            return None

        try:
            # Use trafilatura for main extraction
            content = trafilatura.extract(
                html,
                include_links=self.preserve_links,
                include_tables=True,
                include_comments=False,
                output_format='markdown',
                url=url
            )

            if content and self.extract_code_blocks:
                # Ensure code blocks are properly formatted
                content = self._fix_code_blocks(content)

            return content

        except (IOError, OSError, FileNotFoundError) as e:
            # Expected errors - file not found, permission issues
            logger.error(f"Error reading file: {e}")
            return None
        except (ValueError, TypeError, UnicodeDecodeError) as e:
            # Expected errors - invalid content, encoding issues
            logger.error(f"Error extracting content: {e}")
            return None
        except Exception as e:
            # Unexpected errors - log with traceback
            logger.exception("Unexpected error in content extraction", exc_info=True)
            return None

    def extract_code_blocks(self, content: str) -> Tuple[str, List[Dict[str, str]]]:
        """Extract code blocks from markdown content.

        Args:
            content: Markdown content

        Returns:
            Tuple of (content without code blocks, list of code blocks with metadata)
        """
        code_blocks = []
        pattern = r'```(\w+)?\n(.*?)\n```'

        def replace_code_block(match):
            language = match.group(1) or 'unknown'
            code = match.group(2)

            # Store code block
            block_id = f"CODE_BLOCK_{len(code_blocks)}"
            code_blocks.append({
                'id': block_id,
                'language': language,
                'code': code
            })

            # Replace with placeholder
            return f"\n[{block_id}]\n"

        content_without_code = re.sub(pattern, replace_code_block, content, flags=re.DOTALL)

        return content_without_code, code_blocks

    def extract_metadata_from_html(self, html: str) -> Dict[str, Any]:
        """Extract metadata from HTML.

        Args:
            html: HTML content

        Returns:
            Dictionary of metadata
        """
        soup = BeautifulSoup(html, 'html.parser')
        metadata = {}

        # Extract title
        title = soup.find('title')
        if title:
            metadata['title'] = title.get_text().strip()

        # Extract meta tags
        meta_tags = {
            'description': soup.find('meta', attrs={'name': 'description'}),
            'keywords': soup.find('meta', attrs={'name': 'keywords'}),
            'author': soup.find('meta', attrs={'name': 'author'}),
            'version': soup.find('meta', attrs={'name': 'version'}),
        }

        for key, tag in meta_tags.items():
            if tag and tag.get('content'):
                metadata[key] = tag.get('content')

        # Extract Open Graph tags
        og_tags = soup.find_all('meta', property=re.compile(r'^og:'))
        for tag in og_tags:
            property_name = tag.get('property', '').replace('og:', '')
            content = tag.get('content')
            if property_name and content:
                metadata[f'og_{property_name}'] = content

        # Extract language
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang'):
            metadata['language'] = html_tag.get('lang')

        return metadata

    def clean_documentation_content(self, content: str) -> str:
        """Clean and normalize documentation content.

        Args:
            content: Raw content

        Returns:
            Cleaned content
        """
        if not content:
            return ""

        # Remove excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' +', ' ', content)

        # Remove navigation markers
        nav_markers = [
            'Table of Contents',
            'Navigation',
            'Next:',
            'Previous:',
            'On this page',
            'Quick search'
        ]
        for marker in nav_markers:
            content = content.replace(marker, '')

        # Clean up markdown links that are broken
        content = re.sub(r'\[([^\]]+)\]\(\)', r'\1', content)  # Remove empty links

        # Remove edit/view source links
        content = re.sub(r'\[Edit on GitHub\]\([^)]+\)', '', content)
        content = re.sub(r'\[View source\]\([^)]+\)', '', content)

        # Normalize headers
        content = re.sub(r'^#{7,}', '######', content, flags=re.MULTILINE)

        # Remove common footer/header patterns
        content = re.sub(r'Copyright \d{4}.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'(c).*$', '', content, flags=re.MULTILINE)

        # Final cleanup
        content = content.strip()

        return content

    def _fix_code_blocks(self, content: str) -> str:
        """Fix and normalize code blocks in markdown.

        Args:
            content: Markdown content

        Returns:
            Content with fixed code blocks
        """
        # Ensure code blocks have proper spacing
        content = re.sub(r'```(\w+)?\n', r'\n```\1\n', content)
        content = re.sub(r'\n```\n', r'\n```\n\n', content)

        return content

    def extract_tables(self, html: str) -> List[Dict[str, Any]]:
        """Extract tables from HTML.

        Args:
            html: HTML content

        Returns:
            List of tables with headers and rows
        """
        soup = BeautifulSoup(html, 'html.parser')
        tables = []

        for table in soup.find_all('table'):
            table_data = {
                'headers': [],
                'rows': []
            }

            # Extract headers
            thead = table.find('thead')
            if thead:
                header_row = thead.find('tr')
                if header_row:
                    table_data['headers'] = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]

            # Extract rows
            tbody = table.find('tbody') or table
            for row in tbody.find_all('tr'):
                cells = [td.get_text().strip() for td in row.find_all(['td', 'th'])]
                if cells and cells != table_data['headers']:  # Avoid duplicate headers
                    table_data['rows'].append(cells)

            if table_data['rows']:
                tables.append(table_data)

        return tables

    def extract_links(self, html: str, base_url: Optional[str] = None) -> List[Dict[str, str]]:
        """Extract all links from HTML.

        Args:
            html: HTML content
            base_url: Base URL for resolving relative links

        Returns:
            List of links with text and URL
        """
        from urllib.parse import urljoin

        soup = BeautifulSoup(html, 'html.parser')
        links = []

        for link in soup.find_all('a', href=True):
            href = link.get('href')
            text = link.get_text().strip()

            if href and not href.startswith('#'):
                # Resolve relative URLs if base_url provided
                if base_url:
                    href = urljoin(base_url, href)

                links.append({
                    'text': text,
                    'url': href
                })

        return links

    def is_documentation_page(self, html: str, url: str) -> bool:
        """Determine if a page is likely documentation.

        Args:
            html: HTML content
            url: Page URL

        Returns:
            True if page appears to be documentation
        """
        # Check URL patterns
        doc_url_patterns = [
            r'/docs?/',
            r'/documentation/',
            r'/guide/',
            r'/tutorial/',
            r'/api/',
            r'/reference/',
            r'readthedocs',
            r'devdocs'
        ]

        for pattern in doc_url_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return True

        # Check content
        soup = BeautifulSoup(html, 'html.parser')

        # Look for documentation indicators
        doc_indicators = [
            soup.find(class_=re.compile(r'documentation', re.I)),
            soup.find(class_=re.compile(r'docs', re.I)),
            soup.find(id=re.compile(r'documentation', re.I)),
            soup.find('nav', class_=re.compile(r'toc', re.I))  # Table of contents
        ]

        return any(indicator for indicator in doc_indicators)

    def extract_code_examples(self, content: str) -> List[Dict[str, str]]:
        """Extract code examples with context.

        Args:
            content: Markdown or HTML content

        Returns:
            List of code examples with descriptions
        """
        examples = []

        # Pattern: text followed by code block
        pattern = r'(.*?)\n```(\w+)?\n(.*?)\n```'

        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            description = match.group(1).strip().split('\n')[-1]  # Get last line before code
            language = match.group(2) or 'unknown'
            code = match.group(3)

            # Only include if there's meaningful description
            if len(description) > 20 or any(word in description.lower() for word in
                                            ['example', 'usage', 'sample', 'demo']):
                examples.append({
                    'description': description,
                    'language': language,
                    'code': code
                })

        return examples


def extract_error_signature(error_text: str) -> Dict[str, str]:
    """Extract structured information from error messages.

    Args:
        error_text: Full error message/traceback

    Returns:
        Dictionary with error type, message, and context
    """
    result = {
        'error_type': '',
        'error_message': '',
        'file': '',
        'line': '',
        'context': ''
    }

    lines = error_text.strip().split('\n')

    # Find error type (usually last line or near it)
    for line in reversed(lines):
        if ':' in line:
            parts = line.split(':', 1)
            if parts[0].strip().endswith('Error') or parts[0].strip().endswith('Exception'):
                result['error_type'] = parts[0].strip()
                result['error_message'] = parts[1].strip() if len(parts) > 1 else ''
                break

    # Extract file and line number
    file_pattern = r'File "([^"]+)", line (\d+)'
    match = re.search(file_pattern, error_text)
    if match:
        result['file'] = match.group(1)
        result['line'] = match.group(2)

    # Get context (a few lines around error)
    if len(lines) > 3:
        result['context'] = '\n'.join(lines[-5:])

    return result
