"""Embedding generation module for RAG-CLI.

This module handles text embedding generation using sentence-transformers,
with batch processing support and LRU caching for efficiency.
"""

import time
from typing import List, Union, Optional, Tuple
from functools import lru_cache
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.core.config import get_config
from src.monitoring.logger import get_logger, get_metrics_logger, log_execution_time


logger = get_logger(__name__)
metrics = get_metrics_logger()


class EmbeddingCache:
    """Cache for embedding results to avoid recomputation."""

    def __init__(self, cache_size: int = 1000):
        """Initialize embedding cache.

        Args:
            cache_size: Maximum number of cached embeddings
        """
        self.cache_size = cache_size
        self._cache = {}
        self._access_order = []

    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache.

        Args:
            text: Text to get embedding for

        Returns:
            Cached embedding or None if not found
        """
        if text in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(text)
            self._access_order.append(text)
            logger.debug("Cache hit for embedding", text_length=len(text))
            return self._cache[text]
        return None

    def put(self, text: str, embedding: np.ndarray):
        """Store embedding in cache.

        Args:
            text: Original text
            embedding: Computed embedding
        """
        if text in self._cache:
            # Already in cache, just update access order
            self._access_order.remove(text)
            self._access_order.append(text)
            return

        # Add to cache
        if len(self._cache) >= self.cache_size:
            # Remove least recently used
            lru_text = self._access_order.pop(0)
            del self._cache[lru_text]
            logger.debug("Evicted from cache", text_length=len(lru_text))

        self._cache[text] = embedding
        self._access_order.append(text)

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()
        logger.info("Embedding cache cleared")


class EmbeddingGenerator:
    """Generates embeddings for text using sentence-transformers."""

    def __init__(self, model_name: Optional[str] = None):
        """Initialize embedding generator.

        Args:
            model_name: Name of the sentence-transformer model to use
        """
        config = get_config()
        self.model_name = model_name or config.embeddings.model_name
        self.dimensions = config.embeddings.dimensions
        self.batch_size = config.embeddings.batch_size
        self.normalize = config.embeddings.normalize
        self.device = config.embeddings.device
        self.max_seq_length = config.embeddings.max_seq_length
        self.cache_size = config.embeddings.cache_size

        # Initialize model
        logger.info(f"Loading embedding model: {self.model_name}")
        start_time = time.time()

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device
        )
        self.model.max_seq_length = self.max_seq_length

        load_time = time.time() - start_time
        logger.info(f"Model loaded", model=self.model_name, load_time_seconds=load_time)
        metrics.record_latency("model_load", load_time * 1000)

        # Initialize cache
        self.cache = EmbeddingCache(self.cache_size)

        # Verify dimensions
        test_embedding = self.model.encode("test")
        actual_dims = len(test_embedding)
        if actual_dims != self.dimensions:
            logger.warning(
                f"Model dimensions mismatch",
                expected=self.dimensions,
                actual=actual_dims
            )
            self.dimensions = actual_dims

    @log_execution_time
    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
        use_cache: bool = True
    ) -> np.ndarray:
        """Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts to encode
            show_progress: Whether to show progress bar
            use_cache: Whether to use caching

        Returns:
            Embeddings as numpy array
        """
        # Handle single text input
        if isinstance(texts, str):
            if use_cache:
                cached = self.cache.get(texts)
                if cached is not None:
                    metrics.record_success("cache_hit")
                    return cached

            embedding = self._encode_batch([texts], show_progress=False)[0]

            if use_cache:
                self.cache.put(texts, embedding)

            return embedding

        # Handle batch input
        embeddings = []
        texts_to_encode = []
        cached_indices = []

        # Check cache for each text
        for i, text in enumerate(texts):
            if use_cache:
                cached = self.cache.get(text)
                if cached is not None:
                    embeddings.append(cached)
                    cached_indices.append(i)
                    continue
            texts_to_encode.append(text)

        # Log cache statistics
        if use_cache and cached_indices:
            cache_ratio = len(cached_indices) / len(texts)
            logger.debug(
                f"Cache hit ratio",
                cached=len(cached_indices),
                total=len(texts),
                ratio=cache_ratio
            )
            metrics.record_gauge("cache_hit_ratio", cache_ratio)

        # Encode uncached texts
        if texts_to_encode:
            new_embeddings = self._encode_batch(texts_to_encode, show_progress)

            # Cache new embeddings
            if use_cache:
                for text, embedding in zip(texts_to_encode, new_embeddings):
                    self.cache.put(text, embedding)

            # Merge cached and new embeddings in correct order
            new_idx = 0
            final_embeddings = []
            for i in range(len(texts)):
                if i in cached_indices:
                    # Get from cached results
                    cached_idx = cached_indices.index(i)
                    final_embeddings.append(embeddings[cached_idx])
                else:
                    # Get from newly encoded
                    final_embeddings.append(new_embeddings[new_idx])
                    new_idx += 1

            embeddings = final_embeddings
        else:
            # All texts were cached
            final_embeddings = [None] * len(texts)
            for cached_idx, original_idx in enumerate(cached_indices):
                final_embeddings[original_idx] = embeddings[cached_idx]
            embeddings = final_embeddings

        return np.array(embeddings)

    def _encode_batch(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> np.ndarray:
        """Encode a batch of texts.

        Args:
            texts: List of texts to encode
            show_progress: Whether to show progress bar

        Returns:
            Embeddings as numpy array
        """
        if not texts:
            return np.array([])

        logger.debug(f"Encoding batch", batch_size=len(texts))
        start_time = time.time()

        # Process in batches
        all_embeddings = []
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        # Create progress bar if requested
        if show_progress:
            pbar = tqdm(
                total=len(texts),
                desc="Generating embeddings",
                unit="texts"
            )
        else:
            pbar = None

        for i in range(num_batches):
            batch_start = i * self.batch_size
            batch_end = min((i + 1) * self.batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]

            # Generate embeddings
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False
            )

            all_embeddings.append(batch_embeddings)

            # Update progress bar
            if pbar:
                pbar.update(len(batch_texts))

        if pbar:
            pbar.close()

        # Combine all batches
        embeddings = np.vstack(all_embeddings)

        # Record metrics
        elapsed_time = time.time() - start_time
        texts_per_second = len(texts) / elapsed_time
        logger.info(
            f"Batch encoding completed",
            num_texts=len(texts),
            elapsed_seconds=elapsed_time,
            texts_per_second=texts_per_second
        )
        metrics.record_latency("batch_encoding", elapsed_time * 1000)
        metrics.record_gauge("encoding_speed", texts_per_second)

        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query with optimizations.

        Args:
            query: Query text to encode

        Returns:
            Query embedding
        """
        # Always use cache for queries
        return self.encode(query, use_cache=True)

    def encode_documents(
        self,
        documents: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """Encode multiple documents efficiently.

        Args:
            documents: List of document texts
            show_progress: Whether to show progress bar

        Returns:
            Document embeddings
        """
        logger.info(f"Encoding documents", count=len(documents))

        # For large document sets, disable cache to avoid memory issues
        use_cache = len(documents) < 1000

        embeddings = self.encode(
            documents,
            show_progress=show_progress,
            use_cache=use_cache
        )

        logger.info(f"Documents encoded", count=len(documents))
        metrics.record_count("documents_encoded", len(documents))

        return embeddings

    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity between query and documents.

        Args:
            query_embedding: Query embedding vector
            document_embeddings: Document embedding matrix

        Returns:
            Similarity scores
        """
        # Ensure inputs are numpy arrays
        query_embedding = np.array(query_embedding)
        document_embeddings = np.array(document_embeddings)

        # Normalize if not already normalized
        if self.normalize:
            # Already normalized during encoding
            pass
        else:
            # Normalize for cosine similarity
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            doc_norms = document_embeddings / np.linalg.norm(
                document_embeddings, axis=1, keepdims=True
            )
            query_embedding = query_norm
            document_embeddings = doc_norms

        # Compute cosine similarity
        similarities = np.dot(document_embeddings, query_embedding)

        return similarities

    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache.clear()

    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings.

        Returns:
            Embedding dimensions
        """
        return self.dimensions

    def warmup(self):
        """Warm up the model with a test encoding."""
        logger.debug("Warming up embedding model")
        _ = self.encode("warmup test")
        logger.debug("Model warmed up")


# Singleton instance
_embedding_generator: Optional[EmbeddingGenerator] = None


def get_embedding_generator(
    model_name: Optional[str] = None
) -> EmbeddingGenerator:
    """Get or create the global embedding generator.

    Args:
        model_name: Optional model name to override config

    Returns:
        Embedding generator instance
    """
    global _embedding_generator

    if _embedding_generator is None or (
        model_name and model_name != _embedding_generator.model_name
    ):
        _embedding_generator = EmbeddingGenerator(model_name)

    return _embedding_generator


if __name__ == "__main__":
    # Test embedding generation
    print("Testing Embedding Generator...")

    # Initialize generator
    generator = get_embedding_generator()

    # Test single text
    text = "This is a test document for embedding generation."
    embedding = generator.encode(text)
    print(f"Single text embedding shape: {embedding.shape}")
    print(f"Embedding dimensions: {len(embedding)}")

    # Test batch encoding
    texts = [
        "First document about RAG systems.",
        "Second document about embeddings.",
        "Third document about vector search.",
        "Fourth document about Claude integration.",
        "Fifth document about monitoring."
    ]

    embeddings = generator.encode_documents(texts, show_progress=True)
    print(f"\nBatch embeddings shape: {embeddings.shape}")

    # Test similarity computation
    query = "How do RAG systems work?"
    query_embedding = generator.encode_query(query)
    similarities = generator.compute_similarity(query_embedding, embeddings)

    print(f"\nSimilarity scores for query: '{query}'")
    for i, (text, score) in enumerate(zip(texts, similarities)):
        print(f"  {i+1}. {text[:50]}... - Score: {score:.4f}")

    # Test caching
    print("\nTesting cache (should be fast):")
    start = time.time()
    _ = generator.encode(texts[0])  # Should hit cache
    cache_time = time.time() - start
    print(f"Cached encoding time: {cache_time:.4f}s")

    print("\nEmbedding tests completed successfully!")