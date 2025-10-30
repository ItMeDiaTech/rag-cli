"""Semantic caching for RAG queries.

This module implements intelligent caching that matches similar queries
using cosine similarity rather than exact string matching. This significantly
improves cache hit rates for semantically equivalent queries.
"""

import time
import threading
import json
import numpy as np
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import OrderedDict
from pathlib import Path

from monitoring.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Cached query result."""
    query: str
    query_embedding: np.ndarray
    result: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    hit_count: int = 0

    def update_access(self):
        """Update access statistics."""
        self.last_accessed = datetime.now()
        self.access_count += 1


class SemanticCache:
    """Semantic cache for query results using embedding similarity."""

    def __init__(self,
                 embedding_generator,
                 similarity_threshold: float = 0.95,
                 max_size: int = 1000,
                 ttl_seconds: int = 3600):
        """Initialize semantic cache.

        Args:
            embedding_generator: Embedding generator for query encoding
            similarity_threshold: Minimum cosine similarity for cache hit (0-1)
            max_size: Maximum number of cache entries
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.embedding_generator = embedding_generator
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)

        # Use OrderedDict for LRU eviction
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

        logger.info("Semantic cache initialized",
                    similarity_threshold=similarity_threshold,
                    max_size=max_size,
                    ttl_seconds=ttl_seconds)

    def _compute_cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Cosine similarity (0-1)
        """
        # Normalize embeddings
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)

        # Compute dot product (cosine similarity for normalized vectors)
        similarity = np.dot(emb1_norm, emb2_norm)

        return float(similarity)

    def _find_similar_entry(self, query_embedding: np.ndarray) -> Optional[Tuple[str, CacheEntry, float]]:
        """Find most similar cache entry above threshold.

        Args:
            query_embedding: Query embedding to match

        Returns:
            Tuple of (cache_key, entry, similarity) or None if no match
        """
        best_match = None
        best_similarity = 0.0
        best_key = None

        for key, entry in self.cache.items():
            # Check if entry is expired
            if datetime.now() - entry.created_at > self.ttl:
                continue

            # Compute similarity
            similarity = self._compute_cosine_similarity(query_embedding, entry.query_embedding)

            # Track best match
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = entry
                best_key = key

        if best_match:
            return (best_key, best_match, best_similarity)

        return None

    def _evict_lru(self):
        """Evict least recently used entry."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (LRU)
            evicted_key, evicted_entry = self.cache.popitem(last=False)
            self.evictions += 1
            logger.debug("Evicted LRU cache entry",
                         evicted_query=evicted_entry.query[:50],
                         access_count=evicted_entry.access_count)

    def _clean_expired(self):
        """Remove expired entries."""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.cache.items()
            if now - entry.created_at > self.ttl
        ]

        for key in expired_keys:
            del self.cache[key]
            self.evictions += 1

        if expired_keys:
            logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")

    def get(self, query: str) -> Optional[Tuple[Any, float]]:
        """Get cached result for query.

        Args:
            query: Query string

        Returns:
            Tuple of (cached_result, similarity) or None if no match
        """
        start_time = time.time()

        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.encode(query)

            # Find similar entry
            match = self._find_similar_entry(query_embedding)

            if match:
                key, entry, similarity = match

                # Update access statistics
                entry.update_access()
                entry.hit_count += 1

                # Move to end (most recently used)
                self.cache.move_to_end(key)

                self.hits += 1
                elapsed_ms = (time.time() - start_time) * 1000

                logger.debug("Cache hit",
                             query=query[:50],
                             similarity=similarity,
                             cached_query=entry.query[:50],
                             cache_age_seconds=(datetime.now() - entry.created_at).total_seconds(),
                             lookup_time_ms=elapsed_ms)

                return (entry.result, similarity)

            else:
                self.misses += 1
                elapsed_ms = (time.time() - start_time) * 1000

                logger.debug("Cache miss",
                             query=query[:50],
                             lookup_time_ms=elapsed_ms)

                return None

        except Exception as e:
            logger.error(f"Cache lookup failed: {e}")
            self.misses += 1
            return None

    def put(self, query: str, result: Any):
        """Store result in cache.

        Args:
            query: Query string
            result: Result to cache
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.encode(query)

            # Create cache entry
            entry = CacheEntry(
                query=query,
                query_embedding=query_embedding,
                result=result,
                created_at=datetime.now(),
                last_accessed=datetime.now()
            )

            # Evict LRU if at capacity
            self._evict_lru()

            # Clean expired entries periodically
            if len(self.cache) % 100 == 0:
                self._clean_expired()

            # Store entry
            cache_key = f"{hash(query)}"
            self.cache[cache_key] = entry

            logger.debug("Cache entry added",
                         query=query[:50],
                         cache_size=len(self.cache))

        except Exception as e:
            logger.error(f"Cache put failed: {e}")

    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "similarity_threshold": self.similarity_threshold,
            "ttl_seconds": self.ttl.total_seconds()
        }

    def save(self, filepath: Path):
        """Save cache to disk using JSON (secure alternative to pickle).

        Implements automatic rotation when cache file exceeds size limit.

        Args:
            filepath: Path to save cache
        """
        try:
            MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB limit

            # Check if rotation is needed before saving
            if filepath.exists():
                file_size = filepath.stat().st_size
                if file_size > MAX_FILE_SIZE:
                    # Rotate old cache
                    backup_path = filepath.with_suffix('.json.old')
                    if backup_path.exists():
                        backup_path.unlink()  # Remove oldest backup
                    filepath.rename(backup_path)
                    logger.info(f"Rotated cache file (was {file_size / 1024 / 1024:.1f}MB)")

            # Only save entries that haven't expired
            current_time = time.time()
            active_cache = OrderedDict()
            for key, entry in self.cache.items():
                age = current_time - entry['timestamp']
                if age < self.ttl.total_seconds():
                    active_cache[key] = entry

            # Convert cache to JSON-serializable format
            serializable_cache = {}
            for key, entry in active_cache.items():
                serializable_cache[key] = {
                    'query': entry['query'],
                    'response': entry['response'],
                    'embedding': entry['embedding'].tolist(),  # Convert numpy array to list
                    'timestamp': entry['timestamp'],
                    'metadata': entry.get('metadata', {})
                }

            cache_data = {
                'cache': serializable_cache,
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'version': '1.0',  # For future compatibility
                'saved_at': datetime.now().isoformat()
            }

            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)

            logger.info(f"Cache saved to {filepath}",
                        entries=len(active_cache),
                        expired_removed=len(self.cache) - len(active_cache))

        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def load(self, filepath: Path):
        """Load cache from disk (JSON format for security).

        Args:
            filepath: Path to load cache from
        """
        try:
            if not filepath.exists():
                logger.warning(f"Cache file not found: {filepath}")
                return

            with open(filepath, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            # Convert back from JSON format
            loaded_cache = OrderedDict()
            for key, entry in cache_data['cache'].items():
                loaded_cache[key] = {
                    'query': entry['query'],
                    'response': entry['response'],
                    'embedding': np.array(entry['embedding'], dtype=np.float32),  # Convert list back to numpy
                    'timestamp': entry['timestamp'],
                    'metadata': entry.get('metadata', {})
                }

            self.cache = loaded_cache
            self.hits = cache_data.get('hits', 0)
            self.misses = cache_data.get('misses', 0)
            self.evictions = cache_data.get('evictions', 0)

            # Clean expired entries
            self._clean_expired()

            logger.info(f"Cache loaded from {filepath}",
                        entries=len(self.cache),
                        version=cache_data.get('version', 'unknown'))

        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            self.cache.clear()


# Global cache instance
_semantic_cache: Optional[SemanticCache] = None


_semantic_cache_lock = threading.Lock()


def get_semantic_cache(embedding_generator=None,
                       similarity_threshold: float = 0.95,
                       max_size: int = 1000,
                       ttl_seconds: int = 3600) -> SemanticCache:
    """Get or create the global semantic cache instance (thread-safe).

    Args:
        embedding_generator: Embedding generator (required on first call)
        similarity_threshold: Minimum similarity for cache hit
        max_size: Maximum cache size
        ttl_seconds: Time-to-live for cache entries

    Returns:
        Semantic cache instance
    """
    global _semantic_cache

    # Double-check locking pattern for thread safety
    if _semantic_cache is None:
        with _semantic_cache_lock:
            if _semantic_cache is None:
                if embedding_generator is None:
                    raise ValueError("embedding_generator required for cache initialization")

                _semantic_cache = SemanticCache(
                    embedding_generator=embedding_generator,
                    similarity_threshold=similarity_threshold,
                    max_size=max_size,
                    ttl_seconds=ttl_seconds
                )

    return _semantic_cache


def clear_cache():
    """Clear the global semantic cache."""
    if _semantic_cache:
        _semantic_cache.clear()
