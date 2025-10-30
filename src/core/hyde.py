"""HyDE (Hypothetical Document Embeddings) for improved retrieval.

This module implements HyDE, a technique that generates hypothetical answers
to queries before retrieval, improving search quality especially for "how-to"
and technical questions.
"""

import re
import time
import hashlib
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import OrderedDict

from src.monitoring.logger import get_logger
from src.core.claude_code_adapter import get_adapter, OperationMode

logger = get_logger(__name__)


@dataclass
class HyDEResult:
    """Result from HyDE processing."""
    original_query: str
    hypothetical_document: str
    enhanced_query: str
    method: str  # 'llm' or 'heuristic'
    confidence: float


class HyDEGenerator:
    """Generates hypothetical documents for improved retrieval with caching."""

    def __init__(self, cache_size: int = 1000, cache_ttl: int = 604800):
        """Initialize HyDE generator with caching.

        Args:
            cache_size: Maximum number of cached results (default: 1000)
            cache_ttl: Cache time-to-live in seconds (default: 7 days)
        """
        self.adapter = get_adapter()
        self.claude_client = None

        # Initialize cache (LRU cache with TTL)
        self.cache: OrderedDict[str, Tuple[HyDEResult, float]] = OrderedDict()
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self.cache_hits = 0
        self.cache_misses = 0

        # Initialize Claude client if in standalone mode
        if self.adapter.should_use_api():
            try:
                from anthropic import Anthropic
                self.claude_client = Anthropic()
                logger.info("HyDE initialized with LLM generation and caching",
                           cache_size=cache_size, cache_ttl=cache_ttl)
            except Exception as e:
                logger.warning(f"Failed to initialize Claude client for HyDE: {e}")
                self.claude_client = None
        else:
            logger.info("HyDE initialized with heuristic generation and caching (Claude Code mode)",
                       cache_size=cache_size, cache_ttl=cache_ttl)

    def _detect_query_type(self, query: str) -> str:
        """Detect the type of query for appropriate HyDE strategy.

        Args:
            query: User query

        Returns:
            Query type: 'how_to', 'what_is', 'why', 'error', 'general'
        """
        query_lower = query.lower()

        if any(q in query_lower for q in ['how to', 'how do i', 'how can i', 'how does']):
            return 'how_to'
        elif any(q in query_lower for q in ['what is', 'what are', 'what does']):
            return 'what_is'
        elif any(q in query_lower for q in ['why does', 'why is', 'why do']):
            return 'why'
        elif any(q in query_lower for q in ['error', 'exception', 'failed', 'not working']):
            return 'error'
        else:
            return 'general'

    def _generate_heuristic_document(self, query: str, query_type: str) -> str:
        """Generate hypothetical document using heuristics (no LLM).

        Args:
            query: User query
            query_type: Detected query type

        Returns:
            Hypothetical document text
        """
        # Remove question words to create declarative statements
        clean_query = re.sub(r'^(how|what|why|when|where|who|which)\s+(to|is|are|does|do|can|could|would|should)\s+', '', query.lower())
        clean_query = clean_query.rstrip('?')

        if query_type == 'how_to':
            # Generate step-by-step format
            hypothetical = f"""To {clean_query}, follow these steps:

1. First, ensure you have the necessary requirements and dependencies installed.
2. Configure the relevant settings and parameters for your specific use case.
3. Implement the core functionality following best practices.
4. Test the implementation to verify it works correctly.
5. Handle edge cases and error conditions appropriately.

The recommended approach is to {clean_query} by using the standard methods and tools."""

        elif query_type == 'what_is':
            # Generate definition format
            subject = clean_query.replace('what is ', '').replace('what are ', '')
            hypothetical = f"""{subject.title()} is a concept/tool/method used in software development.

Key characteristics:
- It provides functionality for specific use cases
- It follows established patterns and best practices
- It integrates with related systems and tools

Common use cases for {subject} include configuration, implementation, and integration scenarios."""

        elif query_type == 'why':
            # Generate explanation format
            hypothetical = f"""The reason for this behavior is related to {clean_query}.

This occurs because:
- The system follows specific design principles
- There are performance or security considerations
- It maintains compatibility with established standards

Understanding this helps in proper configuration and usage."""

        elif query_type == 'error':
            # Generate troubleshooting format
            hypothetical = f"""When encountering this error, the typical solution involves:

1. Check the configuration files and settings
2. Verify all dependencies are correctly installed
3. Review the error message and stack trace
4. Ensure permissions and access rights are correct
5. Consult the official documentation for known issues

Common causes include misconfiguration, missing dependencies, or version incompatibilities."""

        else:  # general
            # Generate general informational format
            hypothetical = f"""Regarding {clean_query}:

This involves understanding the relevant concepts, configurations, and implementations.
The standard approach includes reviewing documentation, following best practices, and
ensuring proper setup. Key considerations include compatibility, performance, and
maintainability."""

        return hypothetical

    def _generate_llm_document(self, query: str, query_type: str) -> Optional[str]:
        """Generate hypothetical document using LLM.

        Args:
            query: User query
            query_type: Detected query type

        Returns:
            Generated hypothetical document or None if failed
        """
        if not self.claude_client:
            return None

        try:
            # Create prompt based on query type
            prompt = f"""Generate a brief, technical answer (2-3 sentences) to this question: {query}

Write as if you're providing the actual answer from documentation. Be concise and technical."""

            response = self.claude_client.messages.create(
                model="claude-haiku-4-5-20251001",  # Fast model for HyDE
                max_tokens=100,  # Optimized: 100 tokens sufficient, 33% faster than 150
                temperature=0.1,  # Optimized: Lower for speed and consistency
                messages=[{"role": "user", "content": prompt}]
            )

            hypothetical = response.content[0].text.strip()

            logger.debug("Generated LLM hypothetical document",
                        query_length=len(query),
                        doc_length=len(hypothetical))

            return hypothetical

        except Exception as e:
            logger.warning(f"LLM hypothetical generation failed: {e}")
            return None

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query.

        Args:
            query: User query

        Returns:
            Cache key (hash of normalized query)
        """
        # Normalize query: lowercase, strip whitespace, remove punctuation
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[HyDEResult]:
        """Get result from cache if available and not expired.

        Args:
            cache_key: Cache key

        Returns:
            Cached HyDE result or None
        """
        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            age = time.time() - timestamp

            if age < self.cache_ttl:
                # Move to end (LRU)
                self.cache.move_to_end(cache_key)
                self.cache_hits += 1
                logger.debug("HyDE cache hit", cache_key=cache_key[:8], age_seconds=age)
                return result
            else:
                # Expired, remove
                del self.cache[cache_key]
                logger.debug("HyDE cache expired", cache_key=cache_key[:8], age_seconds=age)

        self.cache_misses += 1
        return None

    def _cache_result(self, cache_key: str, result: HyDEResult):
        """Store result in cache.

        Args:
            cache_key: Cache key
            result: HyDE result to cache
        """
        # Enforce cache size limit (LRU eviction)
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            self.cache.popitem(last=False)
            logger.debug("HyDE cache eviction", cache_size=len(self.cache))

        self.cache[cache_key] = (result, time.time())
        logger.debug("HyDE result cached", cache_key=cache_key[:8],
                    cache_size=len(self.cache),
                    hit_rate=self.cache_hits / max(1, self.cache_hits + self.cache_misses))

    def generate(self, query: str, use_llm: bool = None) -> HyDEResult:
        """Generate hypothetical document for query with caching.

        Args:
            query: User query
            use_llm: Whether to use LLM (auto-detected if None)

        Returns:
            HyDE result with enhanced query
        """
        # Check cache first
        cache_key = self._get_cache_key(query)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result

        # Auto-detect if we should use LLM
        if use_llm is None:
            use_llm = self.claude_client is not None and self.adapter.should_use_api()

        # Detect query type
        query_type = self._detect_query_type(query)

        # Generate hypothetical document
        if use_llm:
            hypothetical = self._generate_llm_document(query, query_type)
            if hypothetical is None:
                # Fallback to heuristic
                hypothetical = self._generate_heuristic_document(query, query_type)
                method = 'heuristic_fallback'
                confidence = 0.6
            else:
                method = 'llm'
                confidence = 0.9
        else:
            hypothetical = self._generate_heuristic_document(query, query_type)
            method = 'heuristic'
            confidence = 0.7

        # Combine query with hypothetical document
        # Weight the original query more heavily
        enhanced_query = f"{query}\n\n{hypothetical}"

        logger.info("HyDE generation completed",
                   query_type=query_type,
                   method=method,
                   confidence=confidence,
                   original_length=len(query),
                   enhanced_length=len(enhanced_query))

        result = HyDEResult(
            original_query=query,
            hypothetical_document=hypothetical,
            enhanced_query=enhanced_query,
            method=method,
            confidence=confidence
        )

        # Cache the result for future queries
        self._cache_result(cache_key, result)

        return result

    def should_use_hyde(self, query: str) -> bool:
        """Determine if HyDE should be used for this query.

        Args:
            query: User query

        Returns:
            True if HyDE should be used
        """
        # OPTIMIZATION: Skip HyDE for queries that don't benefit from it

        # Skip very short queries (likely exact lookups)
        if len(query.split()) < 3:
            return False

        # Skip exact technical lookups (code, configs, API references)
        # These queries work better with direct keyword matching
        if any(char in query for char in ['()', '{}', '[]', '::']):
            return False

        # Skip all-caps queries (likely acronyms or exact matches)
        if query.isupper() and len(query) > 2:
            return False

        # Skip queries that look like file paths or URLs
        if any(pattern in query for pattern in ['/', '\\', 'http://', 'https://']):
            return False

        # Use HyDE for:
        # 1. Questions (contains question marks or question words)
        # 2. "How to" queries
        # 3. Error-related queries
        # 4. Conceptual queries (5+ words without special chars)

        query_lower = query.lower()

        indicators = [
            '?' in query,
            any(q in query_lower for q in ['how', 'what', 'why', 'when', 'where']),
            any(q in query_lower for q in ['error', 'exception', 'failed', 'issue', 'problem']),
            len(query.split()) >= 5  # Longer queries benefit more from HyDE
        ]

        return any(indicators)


# Global instance
_hyde_generator: Optional[HyDEGenerator] = None


def get_hyde_generator() -> HyDEGenerator:
    """Get or create the global HyDE generator.

    Returns:
        HyDE generator instance
    """
    global _hyde_generator
    if _hyde_generator is None:
        _hyde_generator = HyDEGenerator()
    return _hyde_generator


def generate_hypothetical_document(query: str) -> HyDEResult:
    """Convenience function to generate hypothetical document.

    Args:
        query: User query

    Returns:
        HyDE result
    """
    generator = get_hyde_generator()
    return generator.generate(query)
