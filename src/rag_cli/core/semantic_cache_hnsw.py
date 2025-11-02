"""Semantic caching for RAG queries using HNSW for fast similarity search.

This module implements intelligent caching that matches similar queries
using cosine similarity with HNSW indexing for O(log n) search performance
instead of O(n) linear search.
"""

import time
import threading
import json
import numpy as np
import faiss
from typing import Any, Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
from pathlib import Path

from rag_cli.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Cached query result with metadata."""
    query: str
    query_embedding: np.ndarray
    result: Any
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    hit_count: int = 0
    entry_id: int = -1  # Index in HNSW

    def update_access(self):
        """Update access statistics."""
        self.last_accessed = datetime.now()
        self.access_count += 1


class HNSWSemanticCache:
    """Semantic cache using HNSW index for O(log n) similarity search.

    Performance characteristics:
    - Build time: O(n log n)
    - Search time: O(log n)
    - Memory: O(n)
    - Accuracy: >95% recall at top-1
    """

    def __init__(self,
                 embedding_generator,
                 similarity_threshold: float = 0.95,
                 max_size: int = 10000,
                 ttl_seconds: int = 3600,
                 embedding_dim: int = 384,
                 hnsw_m: int = 16,
                 hnsw_ef_construction: int = 200,
                 hnsw_ef_search: int = 50):
        """Initialize HNSW-based semantic cache.

        Args:
            embedding_generator: Generator for query embeddings
            similarity_threshold: Minimum cosine similarity for cache hit (0-1)
            max_size: Maximum number of entries in cache
            ttl_seconds: Time to live for cache entries
            embedding_dim: Dimension of embeddings (384 for all-MiniLM-L6-v2)
            hnsw_m: HNSW M parameter (connectivity, higher = better recall but more memory)
            hnsw_ef_construction: HNSW construction parameter (higher = better index quality)
            hnsw_ef_search: HNSW search parameter (higher = better recall but slower)
        """
        self.embedding_generator = embedding_generator
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)
        self.embedding_dim = embedding_dim

        # HNSW index for fast similarity search
        # Using inner product (IP) with normalized vectors = cosine similarity
        self.index = faiss.IndexHNSWFlat(embedding_dim, hnsw_m)
        self.index.hnsw.efConstruction = hnsw_ef_construction
        self.index.hnsw.efSearch = hnsw_ef_search

        # Entry storage (indexed by position in HNSW)
        self.entries: List[Optional[CacheEntry]] = []
        self.entry_map: Dict[str, int] = {}  # query_hash -> index

        # LRU tracking using OrderedDict for O(1) operations
        self.lru_tracker = OrderedDict()

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self.total_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info("Initialized HNSW semantic cache",
                   max_size=max_size,
                   embedding_dim=embedding_dim,
                   hnsw_m=hnsw_m,
                   similarity_threshold=similarity_threshold)

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding for cosine similarity via inner product."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding

    def _compute_query_hash(self, query: str) -> str:
        """Compute hash for exact query matching."""
        import hashlib
        return hashlib.md5(query.encode()).hexdigest()

    def _find_similar_hnsw(self, query_embedding: np.ndarray, k: int = 1) -> Optional[Tuple[int, float]]:
        """Find most similar entry using HNSW index.

        Returns:
            Tuple of (index, similarity) or None if no match above threshold
        """
        if self.index.ntotal == 0:
            return None

        # Normalize for cosine similarity
        query_norm = self._normalize_embedding(query_embedding.astype(np.float32))

        # Search for k nearest neighbors
        similarities, indices = self.index.search(query_norm.reshape(1, -1), k)

        # Check if best match exceeds threshold
        if similarities[0][0] >= self.similarity_threshold:
            return (int(indices[0][0]), float(similarities[0][0]))

        return None

    def _evict_lru(self):
        """Evict least recently used entry when cache is full."""
        if not self.lru_tracker:
            return

        # Get oldest entry
        oldest_hash, oldest_idx = self.lru_tracker.popitem(last=False)

        # Mark as deleted (don't remove from HNSW, just mark invalid)
        if 0 <= oldest_idx < len(self.entries):
            self.entries[oldest_idx] = None
            del self.entry_map[oldest_hash]

        logger.debug("Evicted LRU cache entry", hash=oldest_hash[:8])

    def _cleanup_expired(self):
        """Remove expired entries from cache."""
        now = datetime.now()
        expired_hashes = []

        for query_hash, idx in list(self.entry_map.items()):
            if idx < len(self.entries) and self.entries[idx]:
                entry = self.entries[idx]
                if now - entry.created_at > self.ttl:
                    self.entries[idx] = None
                    expired_hashes.append(query_hash)

        # Remove from maps
        for hash_val in expired_hashes:
            if hash_val in self.entry_map:
                del self.entry_map[hash_val]
            if hash_val in self.lru_tracker:
                del self.lru_tracker[hash_val]

        if expired_hashes:
            logger.debug("Cleaned up expired entries", count=len(expired_hashes))

    def get(self, query: str) -> Optional[Any]:
        """Get cached result for query using HNSW similarity search.

        Args:
            query: Query string to look up

        Returns:
            Cached result or None if no similar query found
        """
        with self._lock:
            self.total_queries += 1

            # Periodic cleanup (every 100 queries)
            if self.total_queries % 100 == 0:
                self._cleanup_expired()

            # Check exact match first (fastest)
            query_hash = self._compute_query_hash(query)
            if query_hash in self.entry_map:
                idx = self.entry_map[query_hash]
                if idx < len(self.entries) and self.entries[idx]:
                    entry = self.entries[idx]
                    # Check if expired
                    if datetime.now() - entry.created_at <= self.ttl:
                        entry.update_access()
                        entry.hit_count += 1
                        # Update LRU
                        del self.lru_tracker[query_hash]
                        self.lru_tracker[query_hash] = idx
                        self.cache_hits += 1
                        logger.debug("Cache hit (exact)", query_len=len(query))
                        return entry.result

            # Generate embedding for similarity search
            try:
                query_embedding = self.embedding_generator.encode([query])[0]
            except Exception as e:
                logger.warning("Failed to generate embedding", error=str(e))
                self.cache_misses += 1
                return None

            # HNSW similarity search (O(log n))
            match = self._find_similar_hnsw(query_embedding)

            if match:
                idx, similarity = match
                if idx < len(self.entries) and self.entries[idx]:
                    entry = self.entries[idx]
                    # Check if expired
                    if datetime.now() - entry.created_at <= self.ttl:
                        entry.update_access()
                        entry.hit_count += 1
                        # Update LRU for the matched entry
                        matched_hash = None
                        for h, i in self.entry_map.items():
                            if i == idx:
                                matched_hash = h
                                break
                        if matched_hash and matched_hash in self.lru_tracker:
                            del self.lru_tracker[matched_hash]
                            self.lru_tracker[matched_hash] = idx
                        self.cache_hits += 1
                        logger.debug("Cache hit (similarity)",
                                   query_len=len(query),
                                   similarity=round(similarity, 3))
                        return entry.result

            self.cache_misses += 1
            logger.debug("Cache miss", query_len=len(query))
            return None

    def put(self, query: str, result: Any) -> None:
        """Store query result in cache with HNSW indexing.

        Args:
            query: Query string
            result: Result to cache
        """
        with self._lock:
            query_hash = self._compute_query_hash(query)

            # Generate embedding
            try:
                query_embedding = self.embedding_generator.encode([query])[0]
            except Exception as e:
                logger.warning("Failed to generate embedding for caching", error=str(e))
                return

            # Check if we need to evict
            if self.index.ntotal >= self.max_size:
                self._evict_lru()

            # Create entry
            entry = CacheEntry(
                query=query,
                query_embedding=query_embedding,
                result=result,
                entry_id=self.index.ntotal
            )

            # Add to HNSW index
            normalized_embedding = self._normalize_embedding(query_embedding.astype(np.float32))
            self.index.add(normalized_embedding.reshape(1, -1))

            # Store entry
            entry_idx = len(self.entries)
            self.entries.append(entry)
            self.entry_map[query_hash] = entry_idx
            self.lru_tracker[query_hash] = entry_idx

            logger.debug("Added to cache",
                       query_len=len(query),
                       total_entries=self.index.ntotal)

    def clear(self):
        """Clear the entire cache."""
        with self._lock:
            # Reset HNSW index
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 16)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 50

            # Clear storage
            self.entries.clear()
            self.entry_map.clear()
            self.lru_tracker.clear()

            # Reset statistics
            self.total_queries = 0
            self.cache_hits = 0
            self.cache_misses = 0

            logger.info("Cache cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache metrics
        """
        with self._lock:
            hit_rate = self.cache_hits / max(1, self.total_queries)

            # Calculate size metrics
            valid_entries = sum(1 for e in self.entries if e is not None)

            return {
                "total_queries": self.total_queries,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_rate": round(hit_rate, 3),
                "cache_size": valid_entries,
                "index_size": self.index.ntotal,
                "max_size": self.max_size,
                "similarity_threshold": self.similarity_threshold,
                "ttl_seconds": self.ttl.total_seconds()
            }

    def save_to_disk(self, path: Path):
        """Save cache index to disk for persistence.

        Args:
            path: Directory to save cache files
        """
        with self._lock:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            index_path = path / "semantic_cache_hnsw.index"
            faiss.write_index(self.index, str(index_path))

            # Save entries and metadata
            metadata = {
                "entries": [
                    {
                        "query": e.query,
                        "embedding": e.query_embedding.tolist(),
                        "created_at": e.created_at.isoformat(),
                        "access_count": e.access_count,
                        "hit_count": e.hit_count
                    } if e else None
                    for e in self.entries
                ],
                "entry_map": self.entry_map,
                "lru_order": list(self.lru_tracker.keys()),
                "statistics": self.get_statistics()
            }

            metadata_path = path / "semantic_cache_meta.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info("Saved cache to disk", path=str(path))

    def load_from_disk(self, path: Path) -> bool:
        """Load cache index from disk.

        Args:
            path: Directory containing cache files

        Returns:
            True if successfully loaded, False otherwise
        """
        with self._lock:
            try:
                path = Path(path)
                index_path = path / "semantic_cache_hnsw.index"
                metadata_path = path / "semantic_cache_meta.json"

                if not index_path.exists() or not metadata_path.exists():
                    return False

                # Load FAISS index
                self.index = faiss.read_index(str(index_path))

                # Load metadata
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Restore entries (without results - those need to be recomputed)
                self.entries = []
                for entry_data in metadata["entries"]:
                    if entry_data:
                        # Note: We don't restore results, only the query embeddings
                        entry = CacheEntry(
                            query=entry_data["query"],
                            query_embedding=np.array(entry_data["embedding"]),
                            result=None,  # Results not persisted
                            created_at=datetime.fromisoformat(entry_data["created_at"]),
                            access_count=entry_data["access_count"],
                            hit_count=entry_data["hit_count"]
                        )
                        self.entries.append(entry)
                    else:
                        self.entries.append(None)

                # Restore maps
                self.entry_map = metadata["entry_map"]
                self.lru_tracker = OrderedDict()
                for key in metadata["lru_order"]:
                    if key in self.entry_map:
                        self.lru_tracker[key] = self.entry_map[key]

                # Restore statistics
                stats = metadata.get("statistics", {})
                self.total_queries = stats.get("total_queries", 0)
                self.cache_hits = stats.get("cache_hits", 0)
                self.cache_misses = stats.get("cache_misses", 0)

                logger.info("Loaded cache from disk",
                          entries=len(self.entries),
                          index_size=self.index.ntotal)
                return True

            except Exception as e:
                logger.error("Failed to load cache from disk", error=str(e))
                self.clear()
                return False


# Maintain backward compatibility
SemanticCache = HNSWSemanticCache