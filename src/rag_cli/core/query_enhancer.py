"""Query enhancement for improved retrieval.

This module provides query expansion, entity extraction, acronym resolution,
and other enhancements to improve retrieval quality.
"""

import re
from typing import List, Dict
from dataclasses import dataclass

from rag_cli_plugin.services.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EnhancedQuery:
    """Enhanced query with expansions and metadata."""
    original_query: str
    enhanced_query: str
    expansions: List[str]
    resolved_acronyms: Dict[str, str]
    extracted_entities: List[str]
    keywords: List[str]


class QueryEnhancer:
    """Enhances queries for better retrieval."""

    # Common technical acronyms
    ACRONYM_GLOSSARY = {
        'rag': 'Retrieval-Augmented Generation',
        'api': 'Application Programming Interface',
        'rest': 'Representational State Transfer',
        'http': 'Hypertext Transfer Protocol',
        'https': 'Hypertext Transfer Protocol Secure',
        'crud': 'Create Read Update Delete',
        'orm': 'Object-Relational Mapping',
        'sql': 'Structured Query Language',
        'nosql': 'Not Only SQL',
        'json': 'JavaScript Object Notation',
        'xml': 'Extensible Markup Language',
        'yaml': 'YAML Ain\'t Markup Language',
        'cli': 'Command Line Interface',
        'gui': 'Graphical User Interface',
        'ui': 'User Interface',
        'ux': 'User Experience',
        'jwt': 'JSON Web Token',
        'oauth': 'Open Authorization',
        'ssl': 'Secure Sockets Layer',
        'tls': 'Transport Layer Security',
        'cdn': 'Content Delivery Network',
        'dns': 'Domain Name System',
        'ip': 'Internet Protocol',
        'tcp': 'Transmission Control Protocol',
        'udp': 'User Datagram Protocol',
        'url': 'Uniform Resource Locator',
        'uri': 'Uniform Resource Identifier',
        'ide': 'Integrated Development Environment',
        'sdk': 'Software Development Kit',
        'npm': 'Node Package Manager',
        'pip': 'Package Installer for Python',
        'ml': 'Machine Learning',
        'ai': 'Artificial Intelligence',
        'llm': 'Large Language Model',
        'nlp': 'Natural Language Processing',
        'cv': 'Computer Vision',
        'gpu': 'Graphics Processing Unit',
        'cpu': 'Central Processing Unit',
        'ram': 'Random Access Memory',
        'ssd': 'Solid State Drive',
        'hdd': 'Hard Disk Drive',
        'os': 'Operating System',
        'vm': 'Virtual Machine',
        'docker': 'containerization platform',
        'k8s': 'Kubernetes',
        'ci': 'Continuous Integration',
        'cd': 'Continuous Deployment',
        'git': 'version control system',
        'svg': 'Scalable Vector Graphics',
        'html': 'Hypertext Markup Language',
        'css': 'Cascading Style Sheets',
        'js': 'JavaScript',
        'ts': 'TypeScript',
        'py': 'Python',
        'db': 'Database',
        'async': 'Asynchronous',
        'sync': 'Synchronous',
        'regex': 'Regular Expression',
    }

    # Synonym mappings for common technical terms
    SYNONYM_MAP = {
        'function': ['method', 'procedure', 'routine', 'callable'],
        'error': ['exception', 'failure', 'issue', 'problem'],
        'fix': ['solve', 'resolve', 'repair', 'correct'],
        'create': ['build', 'make', 'generate', 'construct'],
        'remove': ['delete', 'erase', 'drop', 'eliminate'],
        'change': ['modify', 'update', 'alter', 'edit'],
        'get': ['retrieve', 'fetch', 'obtain', 'acquire'],
        'send': ['transmit', 'post', 'dispatch', 'emit'],
        'receive': ['get', 'accept', 'obtain', 'acquire'],
        'configure': ['setup', 'initialize', 'set up', 'configure'],
        'install': ['setup', 'deploy', 'set up'],
        'run': ['execute', 'start', 'launch', 'invoke'],
        'stop': ['terminate', 'kill', 'halt', 'end'],
        'debug': ['troubleshoot', 'diagnose', 'fix', 'trace'],
        'test': ['verify', 'validate', 'check', 'examine'],
        'optimize': ['improve', 'enhance', 'tune', 'refine'],
        'connect': ['link', 'join', 'attach', 'bind'],
        'disconnect': ['unlink', 'detach', 'separate'],
    }

    def __init__(self, enable_expansion: bool = True, enable_acronym_resolution: bool = True):
        """Initialize query enhancer.

        Args:
            enable_expansion: Whether to expand with synonyms
            enable_acronym_resolution: Whether to resolve acronyms
        """
        self.enable_expansion = enable_expansion
        self.enable_acronym_resolution = enable_acronym_resolution

    def enhance(self, query: str, max_expansions: int = 3) -> EnhancedQuery:
        """Enhance a query for better retrieval.

        Args:
            query: Original user query
            max_expansions: Maximum number of synonym expansions per term

        Returns:
            EnhancedQuery with expansions and metadata
        """
        # Extract keywords
        keywords = self._extract_keywords(query)

        # Resolve acronyms
        resolved_acronyms = {}
        if self.enable_acronym_resolution:
            resolved_acronyms = self._resolve_acronyms(query)

        # Extract entities
        entities = self._extract_entities(query)

        # Generate expansions
        expansions = []
        if self.enable_expansion:
            expansions = self._generate_expansions(query, keywords, max_expansions)

        # Build enhanced query
        enhanced_query = self._build_enhanced_query(
            query, expansions, resolved_acronyms
        )

        logger.debug(
            "Enhanced query",
            original_length=len(query),
            enhanced_length=len(enhanced_query),
            expansions=len(expansions),
            acronyms_resolved=len(resolved_acronyms)
        )

        return EnhancedQuery(
            original_query=query,
            enhanced_query=enhanced_query,
            expansions=expansions,
            resolved_acronyms=resolved_acronyms,
            extracted_entities=entities,
            keywords=keywords
        )

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query.

        Args:
            query: User query

        Returns:
            List of keywords
        """
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'o', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }

        # Tokenize
        words = re.findall(r'\b\w+\b', query.lower())

        # Filter stop words and short words
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords

    def _resolve_acronyms(self, query: str) -> Dict[str, str]:
        """Resolve acronyms in query.

        Args:
            query: User query

        Returns:
            Dictionary of acronym -> full form
        """
        resolved = {}
        query_lower = query.lower()

        for acronym, full_form in self.ACRONYM_GLOSSARY.items():
            # Look for acronym as whole word
            pattern = r'\b' + re.escape(acronym) + r'\b'
            if re.search(pattern, query_lower):
                resolved[acronym] = full_form

        return resolved

    def _extract_entities(self, query: str) -> List[str]:
        """Extract technical entities (programming languages, frameworks, etc.).

        Args:
            query: User query

        Returns:
            List of entities
        """
        entities = []

        # Common programming languages
        languages = ['python', 'javascript', 'java', 'typescript', 'rust', 'go', 'c++', 'c#', 'ruby', 'php']
        for lang in languages:
            if lang.lower() in query.lower():
                entities.append(lang)

        # Common frameworks
        frameworks = ['django', 'flask', 'fastapi', 'react', 'vue', 'angular', 'express', 'nextjs', 'langchain']
        for fw in frameworks:
            if fw.lower() in query.lower():
                entities.append(fw)

        return entities

    def _generate_expansions(self, query: str, keywords: List[str], max_per_term: int) -> List[str]:
        """Generate query expansions using synonyms.

        Args:
            query: Original query
            keywords: Extracted keywords
            max_per_term: Max expansions per keyword

        Returns:
            List of expansion phrases
        """
        expansions = []

        for keyword in keywords:
            if keyword in self.SYNONYM_MAP:
                synonyms = self.SYNONYM_MAP[keyword][:max_per_term]
                for synonym in synonyms:
                    # Create expansion phrase
                    expanded = query.lower().replace(keyword, synonym)
                    if expanded != query.lower() and expanded not in expansions:
                        expansions.append(synonym)

        return expansions[:10]  # Limit total expansions

    def _build_enhanced_query(
        self,
        original: str,
        expansions: List[str],
        acronyms: Dict[str, str]
    ) -> str:
        """Build enhanced query string.

        Args:
            original: Original query
            expansions: Synonym expansions
            acronyms: Resolved acronyms

        Returns:
            Enhanced query string
        """
        # Start with original
        parts = [original]

        # Add acronym expansions
        if acronyms:
            acronym_text = " ".join(f"({full})" for full in acronyms.values())
            parts.append(acronym_text)

        # Add synonym expansions
        if expansions:
            expansion_text = " OR ".join(expansions)
            parts.append(expansion_text)

        return " ".join(parts)


def get_query_enhancer(enable_expansion: bool = True, enable_acronym_resolution: bool = True) -> QueryEnhancer:
    """Get or create query enhancer instance.

    Args:
        enable_expansion: Whether to expand with synonyms
        enable_acronym_resolution: Whether to resolve acronyms

    Returns:
        QueryEnhancer instance
    """
    return QueryEnhancer(
        enable_expansion=enable_expansion,
        enable_acronym_resolution=enable_acronym_resolution
    )
