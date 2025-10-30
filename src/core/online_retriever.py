"""Online documentation retriever orchestrator.

Coordinates multiple documentation sources and manages caching for online retrieval.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import requests_cache

from .source_connectors.github import GitHubConnector, GitHubDocument
from .source_connectors.stackoverflow import StackOverflowConnector, StackOverflowAnswer
from .source_connectors.readthedocs import ReadTheDocsConnector, OfficialDocsConnector, Documentation
from .content_extractors import ContentExtractor, extract_error_signature
from .config import Config, get_config
from src.integrations.arxiv_connector import get_arxiv_connector
from src.integrations.tavily_connector import get_tavily_connector
from src.core.query_classifier import QueryIntent

logger = logging.getLogger(__name__)


@dataclass
class OnlineRetrievalResult:
    """Result from online documentation retrieval."""
    content: str
    title: str
    url: str
    source: str  # 'github', 'stackoverflow', 'readthedocs', 'official'
    score: float  # Relevance score
    metadata: Dict[str, Any]
    fetch_date: datetime


class OnlineRetriever:
    """Orchestrates online documentation retrieval from multiple sources."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize online retriever.

        Args:
            config: Configuration object (loads default if None)
        """
        self.config = config or get_config()
        self.online_config = self.config.online_docs

        # Initialize cache
        if self.online_config.cache.get('enabled', True):
            cache_path = self.online_config.cache.get('path', './data/cache/online_docs.db')
            ttl_hours = self.online_config.cache.get('ttl_hours', 24)
            requests_cache.install_cache(
                cache_path,
                backend='sqlite',
                expire_after=ttl_hours * 3600
            )

        # Initialize content extractor
        content_config = self.online_config.content
        self.extractor = ContentExtractor(
            max_page_size_kb=content_config.get('max_page_size_kb', 500),
            extract_code_blocks=content_config.get('extract_code_blocks', True),
            preserve_links=content_config.get('preserve_links', True),
            clean_html=content_config.get('clean_html', True)
        )

        # Initialize source connectors
        self._init_connectors()

    def _init_connectors(self):
        """Initialize source connectors based on configuration."""
        self.connectors = {}

        api_keys = self.online_config.api_keys
        sources = self.online_config.sources

        # GitHub connector
        if sources.get('github', {}).get('enabled', True):
            github_token = api_keys.get('github_token', '')
            if github_token:
                self.connectors['github'] = GitHubConnector(
                    token=github_token,
                    rate_limit=sources.get('github', {}).get('rate_limit', 5000),
                    timeout=sources.get('github', {}).get('timeout', 10)
                )
                logger.info("GitHub connector initialized")
            else:
                logger.warning("GitHub token not configured, skipping GitHub connector")

        # Stack Overflow connector
        if sources.get('stackoverflow', {}).get('enabled', True):
            so_key = api_keys.get('stackoverflow_key')
            so_config = sources.get('stackoverflow', {})
            self.connectors['stackoverflow'] = StackOverflowConnector(
                api_key=so_key,
                rate_limit=so_config.get('rate_limit', 300),
                timeout=so_config.get('timeout', 10),
                min_score=so_config.get('min_score', 5)
            )
            logger.info("Stack Overflow connector initialized")

        # ReadTheDocs connector
        if sources.get('readthedocs', {}).get('enabled', True):
            rtd_config = sources.get('readthedocs', {})
            self.connectors['readthedocs'] = ReadTheDocsConnector(
                rate_limit=rtd_config.get('rate_limit', 100),
                timeout=rtd_config.get('timeout', 15)
            )
            logger.info("ReadTheDocs connector initialized")

        # Official docs connector
        if sources.get('official_docs', {}).get('enabled', True):
            official_config = sources.get('official_docs', {})
            allowed_domains = official_config.get('allowed_domains', [])
            self.connectors['official'] = OfficialDocsConnector(
                allowed_domains=allowed_domains,
                rate_limit=official_config.get('rate_limit', 100),
                timeout=official_config.get('timeout', 15)
            )
            logger.info("Official docs connector initialized")

        # ArXiv connector (always enabled - free with no API key needed)
        self.connectors['arxiv'] = get_arxiv_connector()
        logger.info("ArXiv connector initialized (academic papers)")

        # Tavily connector (enabled if API key is set, graceful fallback otherwise)
        self.connectors['tavily'] = get_tavily_connector()
        if self.connectors['tavily'].enabled:
            logger.info("Tavily connector initialized (AI-optimized web search)")
        else:
            logger.info("Tavily connector initialized but disabled (no API key)")

    def should_fetch_online(self, local_results: List[Any], query: str) -> bool:
        """Determine if online fetch is needed based on triggers.

        Args:
            local_results: Results from local vector store
            query: User query

        Returns:
            True if should fetch from online sources
        """
        if not self.online_config.enabled:
            return False

        triggers = self.online_config.triggers

        # Check confidence score
        if local_results:
            max_score = max(getattr(r, 'score', 0) for r in local_results)
            min_confidence = triggers.get('min_confidence_score', 0.65)
            if max_score < min_confidence:
                logger.info(f"Max score {max_score:.3f} below threshold {min_confidence}, triggering online fetch")
                return True

        # Check result count
        min_results = triggers.get('min_result_count', 3)
        if len(local_results) < min_results:
            logger.info(f"Only {len(local_results)} results, below threshold {min_results}, triggering online fetch")
            return True

        # Check for error messages
        if triggers.get('detect_error_messages', True):
            if self._contains_error_pattern(query):
                logger.info("Error pattern detected in query, triggering online fetch")
                return True

        # Check for version/recency keywords
        if triggers.get('detect_version_keywords', True):
            if self._contains_version_keywords(query):
                logger.info("Version/recency keywords detected, triggering online fetch")
                return True

        return False

    def retrieve(self, query: str, max_results: int = 5, query_intent: Optional[str] = None) -> List[OnlineRetrievalResult]:
        """Retrieve documentation from online sources with MCP integration.

        Args:
            query: Search query
            max_results: Maximum results to return
            query_intent: Optional query intent (RESEARCH, RECENT_NEWS, etc.)

        Returns:
            List of OnlineRetrievalResult objects
        """
        all_results = []

        # Detect query type
        is_error = self._contains_error_pattern(query)
        is_research = query_intent == "research" or query_intent == QueryIntent.RESEARCH
        is_recent_news = query_intent == "recent_news" or query_intent == QueryIntent.RECENT_NEWS
        language = self._detect_language(query)

        # ArXiv search for research queries (NEW - MCP)
        if is_research and 'arxiv' in self.connectors:
            logger.info("Using ArXiv for research query")
            arxiv_results = self._search_arxiv(query, max_results=3)
            all_results.extend(arxiv_results)

        # Tavily search for recent/version queries (NEW - MCP)
        if is_recent_news and 'tavily' in self.connectors:
            logger.info("Using Tavily for recent news query")
            tavily_results = self._search_tavily(query, max_results=3)
            all_results.extend(tavily_results)

        # GitHub search
        if 'github' in self.connectors:
            github_results = self._search_github(query, language)
            all_results.extend(github_results)

        # Stack Overflow search (prioritize for errors)
        if 'stackoverflow' in self.connectors:
            if is_error:
                so_results = self._search_stackoverflow_error(query, language)
            else:
                so_results = self._search_stackoverflow(query, language)
            all_results.extend(so_results)

        # ReadTheDocs search
        if 'readthedocs' in self.connectors:
            rtd_results = self._search_readthedocs(query)
            all_results.extend(rtd_results)

        # Official docs search
        if 'official' in self.connectors and language:
            official_results = self._search_official_docs(query, language)
            all_results.extend(official_results)

        # Fallback to Tavily for general queries if other sources yield few results
        if len(all_results) < 3 and 'tavily' in self.connectors and not is_recent_news:
            logger.info("Fallback to Tavily - few results from primary sources")
            tavily_results = self._search_tavily(query, max_results=2)
            all_results.extend(tavily_results)

        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)

        # Sort by score and return top results
        unique_results.sort(key=lambda r: r.score, reverse=True)
        return unique_results[:max_results]

    def _search_github(self, query: str, language: Optional[str] = None) -> List[OnlineRetrievalResult]:
        """Search GitHub for documentation.

        Args:
            query: Search query
            language: Optional programming language

        Returns:
            List of results
        """
        results = []
        connector = self.connectors['github']

        try:
            # Search code
            code_results = connector.search_code(query, language=language, max_results=3)

            for item in code_results:
                repo = item.get('repository', {}).get('full_name', '')
                path = item.get('path', '')

                # Get file content
                content = connector.get_file_content(repo, path)
                if content:
                    results.append(OnlineRetrievalResult(
                        content=content,
                        title=f"{repo}/{path}",
                        url=item.get('html_url', ''),
                        source='github',
                        score=0.8,  # Fixed score for now
                        metadata={
                            'repo': repo,
                            'path': path,
                            'language': language
                        },
                        fetch_date=datetime.now()
                    ))

        except Exception as e:
            logger.error(f"Error searching GitHub: {e}")

        return results

    def _search_stackoverflow(self, query: str, language: Optional[str] = None) -> List[OnlineRetrievalResult]:
        """Search Stack Overflow for Q&A.

        Args:
            query: Search query
            language: Optional programming language

        Returns:
            List of results
        """
        results = []
        connector = self.connectors['stackoverflow']

        try:
            tags = [language] if language else None
            answers = connector.search_with_answers(query, tags=tags, max_results=3)

            for answer in answers:
                content = answer.to_document()
                # Score based on answer score and whether it's accepted
                score = min(0.9, 0.6 + (answer.score / 100) + (0.2 if answer.is_accepted else 0))

                results.append(OnlineRetrievalResult(
                    content=content,
                    title=answer.question_title,
                    url=answer.question_url,
                    source='stackoverflow',
                    score=score,
                    metadata={
                        'question_id': answer.question_id,
                        'answer_id': answer.answer_id,
                        'tags': answer.tags,
                        'answer_score': answer.score,
                        'is_accepted': answer.is_accepted
                    },
                    fetch_date=datetime.now()
                ))

        except Exception as e:
            logger.error(f"Error searching Stack Overflow: {e}")

        return results

    def _search_stackoverflow_error(self, error_query: str, language: Optional[str] = None) -> List[OnlineRetrievalResult]:
        """Search Stack Overflow specifically for error solutions.

        Args:
            error_query: Error message
            language: Optional programming language

        Returns:
            List of results
        """
        results = []
        connector = self.connectors['stackoverflow']

        try:
            answers = connector.search_by_error(error_query, language=language, max_results=5)

            for answer in answers:
                content = answer.to_document()
                # Higher base score for error-specific searches
                score = min(0.95, 0.7 + (answer.score / 100) + (0.2 if answer.is_accepted else 0))

                results.append(OnlineRetrievalResult(
                    content=content,
                    title=answer.question_title,
                    url=answer.question_url,
                    source='stackoverflow',
                    score=score,
                    metadata={
                        'question_id': answer.question_id,
                        'answer_id': answer.answer_id,
                        'tags': answer.tags,
                        'answer_score': answer.score,
                        'is_accepted': answer.is_accepted,
                        'error_match': True
                    },
                    fetch_date=datetime.now()
                ))

        except Exception as e:
            logger.error(f"Error searching Stack Overflow for error: {e}")

        return results

    def _search_readthedocs(self, query: str) -> List[OnlineRetrievalResult]:
        """Search ReadTheDocs.

        Args:
            query: Search query

        Returns:
            List of results
        """
        results = []
        connector = self.connectors['readthedocs']

        # Try to detect project name from query
        # This is simplified - in production would have better project detection
        common_projects = ['django', 'flask', 'requests', 'numpy', 'pandas', 'pytest']

        for project in common_projects:
            if project in query.lower():
                try:
                    urls = connector.search_readthedocs(project, query, version='latest')
                    for url in urls[:2]:  # Limit to 2 per project
                        doc = connector.fetch_page(url)
                        if doc:
                            results.append(OnlineRetrievalResult(
                                content=doc.content,
                                title=doc.title,
                                url=doc.url,
                                source='readthedocs',
                                score=0.75,
                                metadata={
                                    'project': project,
                                    'framework': doc.framework,
                                    'version': doc.version
                                },
                                fetch_date=datetime.now()
                            ))
                except Exception as e:
                    logger.debug(f"Error searching ReadTheDocs for {project}: {e}")
                break  # Only search one project

        return results

    def _search_official_docs(self, query: str, language: str) -> List[OnlineRetrievalResult]:
        """Search official language documentation.

        Args:
            query: Search query
            language: Programming language

        Returns:
            List of results
        """
        results = []
        connector = self.connectors['official']

        try:
            docs = connector.fetch_official_docs(language, query)

            for doc in docs:
                results.append(OnlineRetrievalResult(
                    content=doc.content,
                    title=doc.title,
                    url=doc.url,
                    source='official',
                    score=0.85,  # High score for official docs
                    metadata={
                        'language': language,
                        'framework': doc.framework
                    },
                    fetch_date=datetime.now()
                ))

        except Exception as e:
            logger.error(f"Error searching official docs: {e}")

        return results

    def _contains_error_pattern(self, text: str) -> bool:
        """Check if text contains error patterns.

        Args:
            text: Text to check

        Returns:
            True if error pattern detected
        """
        error_patterns = [
            r'Error:',
            r'Exception:',
            r'Traceback',
            r'File ".*", line \d+',
            r'raise \w+Error',
            r'undefined.*not.*function',
            r'cannot find module',
            r'No such file'
        ]

        for pattern in error_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def _contains_version_keywords(self, text: str) -> bool:
        """Check if text contains version/recency keywords.

        Args:
            text: Text to check

        Returns:
            True if version keywords detected
        """
        keywords = ['latest', 'new', 'recent', '2024', '2025', 'updated', 'current']

        text_lower = text.lower()
        return any(keyword in text_lower for keyword in keywords)

    def _detect_language(self, query: str) -> Optional[str]:
        """Detect programming language from query.

        Args:
            query: Search query

        Returns:
            Detected language or None
        """
        languages = ['python', 'javascript', 'typescript', 'java', 'go', 'rust',
                    'c++', 'cpp', 'c#', 'csharp', 'ruby', 'php', 'swift', 'kotlin']

        query_lower = query.lower()

        for lang in languages:
            if lang in query_lower:
                # Normalize language name
                if lang == 'cpp':
                    return 'c++'
                elif lang == 'csharp':
                    return 'c#'
                return lang

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about online retrieval.

        Returns:
            Dictionary of statistics
        """
        stats = {
            'enabled': self.online_config.enabled,
            'connectors': list(self.connectors.keys()),
            'cache_enabled': self.online_config.cache.get('enabled', True)
        }

        # Get connector-specific stats
        if 'stackoverflow' in self.connectors:
            stats['stackoverflow'] = self.connectors['stackoverflow'].get_statistics()

        if 'github' in self.connectors:
            try:
                stats['github'] = self.connectors['github'].get_rate_limit_status()
            except Exception as e:
                logger.warning(f"Failed to get GitHub rate limit status: {e}")
                stats['github'] = {'error': str(e)}

        return stats

    def _search_arxiv(self, query: str, max_results: int = 3) -> List[OnlineRetrievalResult]:
        """Search ArXiv for academic papers (NEW - MCP).

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of results
        """
        results = []
        connector = self.connectors['arxiv']

        try:
            # Search ArXiv with CS/AI categories
            papers = connector.search(
                query,
                max_results=max_results,
                categories=['cs.AI', 'cs.LG', 'cs.CL', 'cs.NE'],
                sort_by='relevance'
            )

            # Convert to standard format
            arxiv_results = connector.to_retrieval_results(papers)

            for item in arxiv_results:
                result = OnlineRetrievalResult(
                    content=item['content'],
                    title=item['title'],
                    url=item['url'],
                    source='arxiv',
                    score=0.85,  # High score for academic papers
                    metadata=item['metadata'],
                    fetch_date=datetime.now()
                )
                results.append(result)

            logger.info(f"ArXiv search returned {len(results)} papers")

        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")

        return results

    def _search_tavily(self, query: str, max_results: int = 3) -> List[OnlineRetrievalResult]:
        """Search using Tavily AI-optimized web search (NEW - MCP).

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of results
        """
        results = []
        connector = self.connectors['tavily']

        # Check if connector is enabled and has quota
        if not connector.enabled or not connector.is_quota_available():
            logger.debug("Tavily not available - disabled or quota exceeded")
            return results

        try:
            # Search Tavily
            tavily_results = connector.search(
                query,
                max_results=max_results,
                search_depth='basic'
            )

            # Convert to standard format
            formatted_results = connector.to_retrieval_results(tavily_results)

            for item in formatted_results:
                result = OnlineRetrievalResult(
                    content=item['content'],
                    title=item['title'],
                    url=item['url'],
                    source='tavily',
                    score=item['score'],
                    metadata=item['metadata'],
                    fetch_date=datetime.now()
                )
                results.append(result)

            logger.info(f"Tavily search returned {len(results)} results, "
                       f"remaining quota: {connector.get_remaining_quota()}")

        except Exception as e:
            logger.error(f"Tavily search failed: {e}")

        return results
