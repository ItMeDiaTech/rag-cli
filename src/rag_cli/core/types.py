"""Type definitions and protocols for RAG-CLI.

This module provides Protocol definitions for better type safety with duck typing.
"""

from typing import Protocol, Any, Dict, List, Optional, Tuple
import numpy as np


class VectorStoreProtocol(Protocol):
    """Protocol for vector store implementations."""

    def add(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        sources: Optional[List[str]] = None
    ) -> None:
        """Add vectors to the store."""
        ...

    def upsert(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        sources: Optional[List[str]] = None
    ) -> List[str]:
        """Update or insert vectors."""
        ...

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors."""
        ...

    def get_vector_count(self) -> int:
        """Get total number of vectors in the store."""
        ...

    def get_dimension(self) -> int:
        """Get the dimension of vectors in this store."""
        ...

    def is_empty(self) -> bool:
        """Check if the vector store is empty."""
        ...

    def delete_by_source(self, source: str) -> int:
        """Delete all vectors from a source."""
        ...


class EmbeddingGeneratorProtocol(Protocol):
    """Protocol for embedding generator implementations."""

    def encode(
        self,
        text: str,
        normalize: bool = True
    ) -> np.ndarray:
        """Generate embedding for a single text."""
        ...

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """Generate embeddings for a batch of texts."""
        ...

    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings."""
        ...

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        ...


class CacheProtocol(Protocol):
    """Protocol for cache implementations."""

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        ...

    def put(self, key: str, value: Any) -> None:
        """Store value in cache."""
        ...

    def clear(self) -> None:
        """Clear all cache entries."""
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        ...


class DocumentProcessorProtocol(Protocol):
    """Protocol for document processor implementations."""

    def process_file(
        self,
        file_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ) -> List[Dict[str, Any]]:
        """Process a single file into chunks."""
        ...

    def process_directory(
        self,
        directory: str,
        recursive: bool = False,
        pattern: str = "*.*"
    ) -> List[Dict[str, Any]]:
        """Process all files in a directory."""
        ...


class RetrievalPipelineProtocol(Protocol):
    """Protocol for retrieval pipeline implementations."""

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.7,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        ...

    def build_bm25_index(self) -> None:
        """Build BM25 index for keyword search."""
        ...


class ClaudeIntegrationProtocol(Protocol):
    """Protocol for Claude integration implementations."""

    def generate_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        stream: bool = True
    ) -> Any:
        """Generate a response using Claude."""
        ...

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        ...
