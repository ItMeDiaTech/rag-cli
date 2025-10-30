"""GitHub API connector for documentation retrieval.

Fetches documentation from GitHub repositories, README files, wikis, and issues.
"""

import requests
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import base64

logger = logging.getLogger(__name__)


@dataclass
class GitHubDocument:
    """Represents a document fetched from GitHub."""
    title: str
    content: str
    url: str
    repo: str
    path: str
    language: Optional[str] = None
    last_modified: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RateLimiter:
    """Simple rate limiter for API requests."""

    def __init__(self, requests_per_hour: int):
        """Initialize rate limiter.

        Args:
            requests_per_hour: Maximum number of requests allowed per hour
        """
        self.requests_per_hour = requests_per_hour
        self.requests = []
        self.window_seconds = 3600  # 1 hour

    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.window_seconds)

        # Remove old requests outside window
        self.requests = [req_time for req_time in self.requests if req_time > cutoff]

        if len(self.requests) >= self.requests_per_hour:
            # Need to wait
            oldest = self.requests[0]
            wait_until = oldest + timedelta(seconds=self.window_seconds)
            wait_seconds = (wait_until - now).total_seconds()

            if wait_seconds > 0:
                logger.warning(f"Rate limit reached, waiting {wait_seconds:.1f} seconds")
                time.sleep(wait_seconds)
                self.requests = []

        self.requests.append(now)


class GitHubConnector:
    """Connector for GitHub API to fetch documentation."""

    def __init__(self, token: str, rate_limit: int = 5000, timeout: int = 10):
        """Initialize GitHub connector.

        Args:
            token: GitHub personal access token
            rate_limit: Maximum requests per hour (default: 5000 for authenticated)
            timeout: Request timeout in seconds
        """
        self.token = token
        self.timeout = timeout
        self.rate_limiter = RateLimiter(rate_limit)
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'RAG-CLI-Documentation-Retriever'
        })

    def search_repositories(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for repositories matching query.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of repository information
        """
        self.rate_limiter.wait_if_needed()

        url = "https://api.github.com/search/repositories"
        params = {
            'q': query,
            'sort': 'stars',
            'order': 'desc',
            'per_page': max_results
        }

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get('items', [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching repositories: {e}")
            return []

    def search_code(self, query: str, repo: Optional[str] = None,
                   language: Optional[str] = None, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for code matching query.

        Args:
            query: Search query
            repo: Optional repository to search (format: owner/name)
            language: Optional programming language filter
            max_results: Maximum number of results

        Returns:
            List of code search results
        """
        self.rate_limiter.wait_if_needed()

        # Build search query
        search_query = query
        if repo:
            search_query += f" repo:{repo}"
        if language:
            search_query += f" language:{language}"

        url = "https://api.github.com/search/code"
        params = {
            'q': search_query,
            'per_page': max_results
        }

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get('items', [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching code: {e}")
            return []

    def get_file_content(self, repo: str, path: str, ref: str = "main") -> Optional[str]:
        """Get content of a file from a repository.

        Args:
            repo: Repository in format owner/name
            path: Path to file in repository
            ref: Git reference (branch, tag, or commit). Default: main

        Returns:
            File content as string, or None if error
        """
        self.rate_limiter.wait_if_needed()

        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        params = {'ref': ref}

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            # Decode base64 content
            if 'content' in data:
                content = base64.b64decode(data['content']).decode('utf-8')
                return content
            else:
                logger.warning(f"No content field in response for {repo}/{path}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching file content: {e}")
            return None
        except Exception as e:
            logger.error(f"Error decoding file content: {e}")
            return None

    def get_readme(self, repo: str) -> Optional[GitHubDocument]:
        """Get README file from repository.

        Args:
            repo: Repository in format owner/name

        Returns:
            GitHubDocument with README content, or None if not found
        """
        self.rate_limiter.wait_if_needed()

        url = f"https://api.github.com/repos/{repo}/readme"

        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            # Decode content
            content = base64.b64decode(data['content']).decode('utf-8')

            return GitHubDocument(
                title=f"README - {repo}",
                content=content,
                url=data.get('html_url', f"https://github.com/{repo}"),
                repo=repo,
                path=data.get('path', 'README.md'),
                metadata={
                    'type': 'readme',
                    'size': data.get('size', 0),
                    'sha': data.get('sha', '')
                }
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching README: {e}")
            return None

    def get_documentation_files(self, repo: str, paths: List[str] = None) -> List[GitHubDocument]:
        """Get documentation files from repository.

        Args:
            repo: Repository in format owner/name
            paths: List of paths to search (default: docs/, documentation/)

        Returns:
            List of GitHubDocuments with documentation content
        """
        if paths is None:
            paths = ['docs', 'documentation', 'doc']

        documents = []

        for base_path in paths:
            docs = self._get_directory_contents(repo, base_path)
            documents.extend(docs)

        return documents

    def _get_directory_contents(self, repo: str, path: str,
                                recursive: bool = True) -> List[GitHubDocument]:
        """Get all files in a directory.

        Args:
            repo: Repository in format owner/name
            path: Directory path
            recursive: Whether to recursively get subdirectories

        Returns:
            List of GitHubDocuments
        """
        self.rate_limiter.wait_if_needed()

        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        documents = []

        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            items = response.json()

            if not isinstance(items, list):
                # Single file
                items = [items]

            for item in items:
                if item['type'] == 'file':
                    # Check if it's a documentation file
                    if self._is_documentation_file(item['name']):
                        content = self.get_file_content(repo, item['path'])
                        if content:
                            documents.append(GitHubDocument(
                                title=item['name'],
                                content=content,
                                url=item['html_url'],
                                repo=repo,
                                path=item['path'],
                                metadata={
                                    'type': 'documentation',
                                    'size': item.get('size', 0),
                                    'sha': item.get('sha', '')
                                }
                            ))

                elif item['type'] == 'dir' and recursive:
                    # Recursively get subdirectory
                    subdocs = self._get_directory_contents(repo, item['path'], recursive=True)
                    documents.extend(subdocs)

            return documents

        except requests.exceptions.RequestException as e:
            logger.debug(f"Directory {path} not found or error: {e}")
            return documents

    def _is_documentation_file(self, filename: str) -> bool:
        """Check if file is likely a documentation file.

        Args:
            filename: Name of the file

        Returns:
            True if file appears to be documentation
        """
        doc_extensions = ['.md', '.rst', '.txt', '.adoc', '.asciidoc']
        doc_names = ['readme', 'contributing', 'changelog', 'license', 'guide', 'tutorial']

        filename_lower = filename.lower()

        # Check extension
        for ext in doc_extensions:
            if filename_lower.endswith(ext):
                return True

        # Check common doc file names
        for name in doc_names:
            if name in filename_lower:
                return True

        return False

    def search_issues(self, repo: str, query: str, state: str = "closed",
                     max_results: int = 10) -> List[Dict[str, Any]]:
        """Search issues/discussions in a repository.

        Args:
            repo: Repository in format owner/name
            query: Search query
            state: Issue state (open, closed, all)
            max_results: Maximum results to return

        Returns:
            List of issues matching query
        """
        self.rate_limiter.wait_if_needed()

        search_query = f"repo:{repo} {query} type:issue state:{state}"
        url = "https://api.github.com/search/issues"
        params = {
            'q': search_query,
            'sort': 'reactions',
            'order': 'desc',
            'per_page': max_results
        }

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get('items', [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching issues: {e}")
            return []

    def get_repository_info(self, repo: str) -> Optional[Dict[str, Any]]:
        """Get repository information.

        Args:
            repo: Repository in format owner/name

        Returns:
            Repository metadata
        """
        self.rate_limiter.wait_if_needed()

        url = f"https://api.github.com/repos/{repo}"

        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching repository info: {e}")
            return None

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status.

        Returns:
            Rate limit information
        """
        url = "https://api.github.com/rate_limit"

        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching rate limit: {e}")
            return {}
