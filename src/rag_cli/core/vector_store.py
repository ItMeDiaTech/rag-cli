"""ChromaDB vector store implementation for RAG-CLI.

This module provides efficient vector storage and similarity search
using ChromaDB with native persistence and metadata management.

Performance Features:
- Native persistence (no manual save/load needed)
- Built-in metadata filtering
- Thread-safe operations
- NumPy 2.0 compatible
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import threading

import numpy as np
import chromadb
# ChromaDB 1.0+ API compatibility
try:
    from chromadb.config import Settings
except ImportError:
    # For ChromaDB 1.3.0+, Settings might not be needed or in different location
    Settings = None

from rag_cli.core.config import get_config
from rag_cli.utils.logger import get_logger, get_metrics_logger, log_execution_time


logger = get_logger(__name__)
metrics = get_metrics_logger()


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return super().default(obj)


@dataclass
class VectorMetadata:
    """Metadata for a stored vector."""
    id: str
    text: str
    source: str
    timestamp: datetime
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""

        def serialize_value(value):
            """Recursively serialize values, converting datetime to ISO format."""
            if isinstance(value, datetime):
                return value.isoformat()
            elif isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                return [serialize_value(item) for item in value]
            else:
                return value

        serialized_metadata = serialize_value(self.metadata)

        return {
            'id': self.id,
            'text': self.text,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'metadata': serialized_metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VectorMetadata':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            text=data['text'],
            source=data['source'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )


class ChromaVectorStore:
    """ChromaDB-based vector store with native persistence."""

    def __init__(
        self,
        dimension: int = 384,
        collection_name: str = "rag_documents",
        persist_directory: Optional[str] = None
    ):
        """Initialize ChromaDB vector store.

        Args:
            dimension: Dimension of vectors (informational only, ChromaDB handles automatically)
            collection_name: Name of the collection
            persist_directory: Directory to persist data (defaults to config value)
        """
        config = get_config()
        self.dimension = dimension
        self.collection_name = collection_name

        # Set persist directory
        if persist_directory:
            self.persist_directory = persist_directory
        else:
            # Use the directory from vector_store save_path config
            save_path = Path(config.vector_store.save_path)
            self.persist_directory = str(save_path.parent / "chroma_db")

        # Ensure directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistence
        # ChromaDB 1.3.0+ API: Settings handling may differ
        try:
            if Settings:
                # Older API with Settings object
                self.client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
            else:
                # Newer API: settings passed as kwargs or use defaults
                self.client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    anonymized_telemetry=False
                )
        except Exception as e:
            # Fallback: try with minimal configuration
            logger.warning(f"ChromaDB client initialization with settings failed: {e}, trying minimal config")
            self.client = chromadb.PersistentClient(path=self.persist_directory)

        # Get or create collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"dimension": dimension, "hnsw:space": "l2"}
            )
        except Exception as e:
            logger.error(f"Failed to create/get collection: {e}")
            raise

        self.id_counter = self._initialize_counter()
        self._lock = threading.RLock()

        logger.info(
            "ChromaDB vector store initialized",
            dimension=dimension,
            collection=self.collection_name,
            persist_directory=self.persist_directory,
            existing_vectors=self.collection.count()
        )

    def _initialize_counter(self) -> int:
        """Initialize ID counter from existing vectors."""
        try:
            count = self.collection.count()
            if count > 0:
                # Get highest ID to continue numbering
                results = self.collection.get(limit=1, include=[])
                if results and results.get('ids'):
                    # Try to extract number from ID format vec_00000001
                    last_id = max(results['ids'], key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
                    if '_' in last_id:
                        return int(last_id.split('_')[1]) + 1
            return 0
        except Exception as e:
            logger.warning(f"Could not initialize counter from existing data: {e}")
            return 0

    def _validate_metadata(self, metadata: List[Dict[str, Any]], texts: List[str], sources: Optional[List[str]] = None) -> None:
        """Validate metadata before adding to vector store.

        Args:
            metadata: List of metadata dictionaries to validate
            texts: List of text content (for length validation)
            sources: Optional list of source identifiers

        Raises:
            ValueError: If metadata is invalid
        """
        if not metadata:
            return

        # Reserved ChromaDB keys that shouldn't be in custom metadata
        reserved_keys = {'id', 'ids', 'embedding', 'embeddings', 'document', 'documents',
                        'metadata', 'metadatas', 'distance', 'distances'}

        for i, meta in enumerate(metadata):
            if not isinstance(meta, dict):
                raise ValueError(f"Metadata at index {i} must be a dictionary, got {type(meta)}")

            # Check for reserved keys
            for key in meta.keys():
                if key in reserved_keys:
                    raise ValueError(f"Metadata contains reserved key '{key}' at index {i}")

                # Warn about internal prefixes
                if key.startswith('custom_'):
                    logger.warning(f"Metadata key '{key}' at index {i} starts with 'custom_' prefix - this will be double-prefixed")

            # Validate metadata size (ChromaDB has limits)
            try:
                serialized = json.dumps(meta)
                if len(serialized) > 10000:  # 10KB limit per metadata entry
                    logger.warning(f"Metadata at index {i} is large ({len(serialized)} bytes) - consider reducing size")
            except (TypeError, ValueError) as e:
                raise ValueError(f"Metadata at index {i} contains non-serializable data: {e}")

        logger.debug(f"Validated {len(metadata)} metadata entries")

    @log_execution_time
    def add(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        sources: Optional[List[str]] = None
    ) -> List[str]:
        """Add vectors with metadata to the store (thread-safe).

        Args:
            embeddings: Vector embeddings to add
            texts: Text content for each vector
            metadata: Optional metadata for each vector
            sources: Optional source identifiers

        Returns:
            List of generated IDs
        """
        with self._lock:
            if len(embeddings) != len(texts):
                raise ValueError("Number of embeddings must match number of texts")

            # Validate metadata
            if metadata:
                self._validate_metadata(metadata, texts, sources)

            # Ensure embeddings are float32 and 2D
            embeddings = np.array(embeddings, dtype=np.float32)
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)

            num_vectors = len(embeddings)
            start_time = time.time()

            # Generate IDs
            ids = []
            for i in range(num_vectors):
                vector_id = f"vec_{self.id_counter:08d}"
                ids.append(vector_id)
                self.id_counter += 1

            # Prepare metadata for ChromaDB
            chroma_metadata = []
            for i in range(num_vectors):
                meta = {
                    "text": texts[i],
                    "source": sources[i] if sources else "unknown",
                    "timestamp": datetime.now().isoformat()
                }
                # Add custom metadata if provided
                if metadata and i < len(metadata):
                    # Flatten nested metadata - ChromaDB requires flat dict
                    custom_meta = metadata[i]
                    for key, value in custom_meta.items():
                        # Convert complex types to JSON strings
                        if isinstance(value, (dict, list)):
                            meta[f"custom_{key}"] = json.dumps(value)
                        else:
                            meta[f"custom_{key}"] = str(value)

                chroma_metadata.append(meta)

            # Add to ChromaDB (automatically persisted)
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=chroma_metadata
            )

            # Record metrics
            elapsed = time.time() - start_time
            vectors_per_second = num_vectors / elapsed if elapsed > 0 else 0
            total_vectors = self.collection.count()

            logger.info(
                "Added vectors to ChromaDB",
                count=num_vectors,
                elapsed_seconds=elapsed,
                vectors_per_second=vectors_per_second,
                total_vectors=total_vectors
            )
            metrics.record_count("vectors_added", num_vectors)
            metrics.record_gauge("total_vectors", total_vectors)

            return ids

    @log_execution_time
    def upsert(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        sources: Optional[List[str]] = None
    ) -> List[str]:
        """Upsert vectors with metadata (update if exists, insert if new).

        This method is preferred over add() when re-indexing documents or
        updating existing content to avoid creating duplicates.

        Args:
            embeddings: Vector embeddings to upsert
            texts: Text content for each vector
            ids: Optional IDs to use (if None, auto-generates)
            metadata: Optional metadata for each vector
            sources: Optional source identifiers

        Returns:
            List of IDs that were upserted
        """
        with self._lock:
            if len(embeddings) != len(texts):
                raise ValueError("Number of embeddings must match number of texts")

            # Validate metadata
            if metadata:
                self._validate_metadata(metadata, texts, sources)

            # Ensure embeddings are float32 and 2D
            embeddings = np.array(embeddings, dtype=np.float32)
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)

            num_vectors = len(embeddings)
            start_time = time.time()

            # Generate or validate IDs
            if ids is None:
                ids = []
                for i in range(num_vectors):
                    vector_id = f"vec_{self.id_counter:08d}"
                    ids.append(vector_id)
                    self.id_counter += 1
            else:
                if len(ids) != num_vectors:
                    raise ValueError("Number of IDs must match number of vectors")

            # Prepare metadata for ChromaDB
            chroma_metadata = []
            for i in range(num_vectors):
                meta = {
                    "text": texts[i],
                    "source": sources[i] if sources else "unknown",
                    "timestamp": datetime.now().isoformat()
                }
                # Add custom metadata if provided
                if metadata and i < len(metadata):
                    # Flatten nested metadata - ChromaDB requires flat dict
                    custom_meta = metadata[i]
                    for key, value in custom_meta.items():
                        # Convert complex types to JSON strings
                        if isinstance(value, (dict, list)):
                            meta[f"custom_{key}"] = json.dumps(value)
                        else:
                            meta[f"custom_{key}"] = str(value)

                chroma_metadata.append(meta)

            # Upsert to ChromaDB (automatically persisted)
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=chroma_metadata
            )

            # Record metrics
            elapsed = time.time() - start_time
            vectors_per_second = num_vectors / elapsed if elapsed > 0 else 0
            total_vectors = self.collection.count()

            logger.info(
                "Upserted vectors to ChromaDB",
                count=num_vectors,
                elapsed_seconds=elapsed,
                vectors_per_second=vectors_per_second,
                total_vectors=total_vectors
            )
            metrics.record_count("vectors_upserted", num_vectors)
            metrics.record_gauge("total_vectors", total_vectors)

            return ids

    @log_execution_time
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: Optional[float] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[VectorMetadata, float]]:
        """Search for similar vectors (thread-safe).

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            threshold: Optional similarity threshold
            where: Optional metadata filter

        Returns:
            List of (metadata, score) tuples
        """
        with self._lock:
            if self.collection.count() == 0:
                logger.warning("Search called on empty collection")
                return []

            # Ensure query is float32 and 1D for ChromaDB
            query_embedding = np.array(query_embedding, dtype=np.float32)
            if len(query_embedding.shape) > 1:
                query_embedding = query_embedding.flatten()

            start_time = time.time()

            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where,
                include=["metadatas", "documents", "distances"]
            )

            # Convert results to VectorMetadata
            output = []
            if results and results['ids'] and len(results['ids']) > 0:
                for i, (id_, distance, metadata_dict, document) in enumerate(zip(
                    results['ids'][0],
                    results['distances'][0],
                    results['metadatas'][0],
                    results['documents'][0]
                )):
                    # Convert L2 distance to similarity score
                    score = 1.0 / (1.0 + distance) if distance >= 0 else 0.0

                    # Apply threshold if specified
                    if threshold is not None and score < threshold:
                        continue

                    # Extract custom metadata
                    custom_metadata = {}
                    for key, value in metadata_dict.items():
                        if key.startswith("custom_"):
                            original_key = key[7:]  # Remove "custom_" prefix
                            try:
                                # Try to parse as JSON
                                custom_metadata[original_key] = json.loads(value)
                            except (json.JSONDecodeError, TypeError):
                                custom_metadata[original_key] = value

                    # Create VectorMetadata
                    meta = VectorMetadata(
                        id=id_,
                        text=metadata_dict.get("text", document),
                        source=metadata_dict.get("source", "unknown"),
                        timestamp=datetime.fromisoformat(metadata_dict.get("timestamp", datetime.now().isoformat())),
                        metadata=custom_metadata
                    )
                    output.append((meta, score))

            # Record metrics
            elapsed = time.time() - start_time
            logger.debug(
                "Vector search completed",
                top_k=top_k,
                results=len(output),
                elapsed_seconds=elapsed
            )
            metrics.record_latency("vector_search", elapsed * 1000)

            return output

    def get_by_id(self, vector_id: str) -> Optional[VectorMetadata]:
        """Get metadata by vector ID.

        Args:
            vector_id: ID of the vector

        Returns:
            Vector metadata or None if not found
        """
        try:
            results = self.collection.get(
                ids=[vector_id],
                include=["metadatas", "documents"]
            )

            if results and results['ids'] and len(results['ids']) > 0:
                metadata_dict = results['metadatas'][0]
                document = results['documents'][0]

                # Extract custom metadata
                custom_metadata = {}
                for key, value in metadata_dict.items():
                    if key.startswith("custom_"):
                        original_key = key[7:]
                        try:
                            custom_metadata[original_key] = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            custom_metadata[original_key] = value

                return VectorMetadata(
                    id=vector_id,
                    text=metadata_dict.get("text", document),
                    source=metadata_dict.get("source", "unknown"),
                    timestamp=datetime.fromisoformat(metadata_dict.get("timestamp", datetime.now().isoformat())),
                    metadata=custom_metadata
                )
            return None
        except Exception as e:
            logger.error(f"Error getting vector by ID: {e}")
            return None

    def get_by_source(self, source: str) -> List[VectorMetadata]:
        """Get all vectors from a specific source.

        Useful for finding all chunks from a particular document
        or checking if a source has been indexed.

        Args:
            source: Source identifier to search for

        Returns:
            List of vector metadata from the source
        """
        try:
            results = self.collection.get(
                where={"source": source},
                include=["metadatas", "documents"]
            )

            vectors = []
            if results and results['ids'] and len(results['ids']) > 0:
                for i, (id_, metadata_dict, document) in enumerate(zip(
                    results['ids'],
                    results['metadatas'],
                    results['documents']
                )):
                    # Extract custom metadata
                    custom_metadata = {}
                    for key, value in metadata_dict.items():
                        if key.startswith("custom_"):
                            original_key = key[7:]
                            try:
                                custom_metadata[original_key] = json.loads(value)
                            except (json.JSONDecodeError, TypeError):
                                custom_metadata[original_key] = value

                    vectors.append(VectorMetadata(
                        id=id_,
                        text=metadata_dict.get("text", document),
                        source=metadata_dict.get("source", "unknown"),
                        timestamp=datetime.fromisoformat(metadata_dict.get("timestamp", datetime.now().isoformat())),
                        metadata=custom_metadata
                    ))

            logger.debug(f"Found {len(vectors)} vectors from source: {source}")
            return vectors

        except Exception as e:
            logger.error(f"Error getting vectors by source: {e}")
            return []

    def delete_by_source(self, source: str) -> int:
        """Delete all vectors from a specific source.

        Useful for removing all chunks from a document before re-indexing
        or cleaning up deleted files.

        Args:
            source: Source identifier to delete

        Returns:
            Number of vectors deleted
        """
        try:
            with self._lock:
                # Get all IDs for this source
                results = self.collection.get(
                    where={"source": source},
                    include=[]
                )

                if not results or not results['ids'] or len(results['ids']) == 0:
                    logger.debug(f"No vectors found for source: {source}")
                    return 0

                vector_ids = results['ids']
                self.collection.delete(ids=vector_ids)

                deleted_count = len(vector_ids)
                logger.info(f"Deleted {deleted_count} vectors from source: {source}")
                metrics.record_count("vectors_deleted_by_source", deleted_count)

                return deleted_count

        except Exception as e:
            logger.error(f"Error deleting vectors by source: {e}")
            return 0

    def update_by_source(
        self,
        source: str,
        embeddings: np.ndarray,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Replace all vectors from a source with new content.

        This is a convenience method that combines delete_by_source()
        and add() in a single operation. Useful for re-indexing updated documents.

        Args:
            source: Source identifier to update
            embeddings: New vector embeddings
            texts: New text content
            metadata: Optional metadata for new vectors

        Returns:
            List of new vector IDs
        """
        with self._lock:
            # Delete existing vectors from this source
            deleted_count = self.delete_by_source(source)
            logger.info(f"Replaced {deleted_count} existing vectors for source: {source}")

            # Add new vectors
            sources = [source] * len(texts)
            new_ids = self.add(embeddings, texts, metadata=metadata, sources=sources)

            logger.info(f"Added {len(new_ids)} new vectors for source: {source}")
            metrics.record_count("vectors_updated_by_source", len(new_ids))

            return new_ids

    def delete(self, vector_ids: List[str]) -> int:
        """Delete vectors by ID.

        Args:
            vector_ids: List of vector IDs to delete

        Returns:
            Number of vectors deleted
        """
        if not vector_ids:
            return 0

        try:
            with self._lock:
                self.collection.delete(ids=vector_ids)

                deleted_count = len(vector_ids)
                logger.info("Deleted vectors", count=deleted_count)
                metrics.record_count("vectors_deleted", deleted_count)

                return deleted_count
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            return 0

    def save(self, path: Optional[str] = None, metadata_path: Optional[str] = None):
        """Save is automatic with ChromaDB persistence.

        This method is kept for API compatibility but does nothing
        since ChromaDB automatically persists all changes.

        Args:
            path: Ignored (kept for compatibility)
            metadata_path: Ignored (kept for compatibility)
        """
        logger.debug("ChromaDB save called (persistence is automatic)")
        metrics.record_success("vector_store_save")

    async def save_async(self, path: Optional[str] = None, metadata_path: Optional[str] = None):
        """Async save (automatic with ChromaDB).

        This method is kept for API compatibility.

        Args:
            path: Ignored
            metadata_path: Ignored
        """
        logger.debug("ChromaDB async save called (persistence is automatic)")
        metrics.record_success("vector_store_save_async")

    def load(self, path: Optional[str] = None, metadata_path: Optional[str] = None):
        """Load is automatic with ChromaDB persistence.

        This method is kept for API compatibility but does nothing
        since ChromaDB automatically loads from persist_directory.

        Args:
            path: Ignored (kept for compatibility)
            metadata_path: Ignored (kept for compatibility)
        """
        count = self.collection.count()
        logger.info(f"ChromaDB collection loaded with {count} vectors (automatic)")
        metrics.record_success("vector_store_load")

    async def load_async(self, path: Optional[str] = None, metadata_path: Optional[str] = None):
        """Async load (automatic with ChromaDB).

        This method is kept for API compatibility.

        Args:
            path: Ignored
            metadata_path: Ignored
        """
        count = self.collection.count()
        logger.info(f"ChromaDB collection loaded with {count} vectors (automatic, async)")
        metrics.record_success("vector_store_load_async")

    def clear(self):
        """Clear all vectors and metadata."""
        try:
            with self._lock:
                # Delete the collection and recreate it
                self.client.delete_collection(name=self.collection_name)
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"dimension": self.dimension, "hnsw:space": "l2"}
                )
                self.id_counter = 0
                logger.info("ChromaDB collection cleared")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store.

        Returns:
            Dictionary with statistics
        """
        count = self.collection.count()

        stats = {
            "total_vectors": count,
            "dimension": self.dimension,
            "index_type": "chromadb_hnsw",
            "metadata_count": count,
            "persist_directory": self.persist_directory,
            "collection_name": self.collection_name
        }

        if count > 0:
            # Get sample to analyze sources
            try:
                results = self.collection.get(
                    limit=min(count, 1000),
                    include=["metadatas"]
                )

                if results and results.get('metadatas'):
                    sources = {}
                    timestamps = []

                    for meta in results['metadatas']:
                        source = meta.get('source', 'unknown')
                        sources[source] = sources.get(source, 0) + 1

                        if 'timestamp' in meta:
                            try:
                                timestamps.append(datetime.fromisoformat(meta['timestamp']))
                            except Exception:
                                pass

                    stats["sources"] = sources
                    if timestamps:
                        stats["oldest_timestamp"] = min(timestamps)
                        stats["newest_timestamp"] = max(timestamps)

            except Exception as e:
                logger.warning(f"Could not gather detailed statistics: {e}")

        return stats

    def get_index_size_mb(self) -> float:
        """Get estimated collection size in MB.

        Returns:
            Estimated size in megabytes
        """
        try:
            # Estimate based on count and dimension
            count = self.collection.count()
            if count == 0:
                return 0.0

            # ChromaDB overhead is roughly:
            # vectors: count * dimension * 4 bytes
            # metadata: ~500 bytes per document (average)
            # HNSW index: ~20% overhead

            vector_size_bytes = count * self.dimension * 4
            metadata_size_bytes = count * 500
            hnsw_overhead = vector_size_bytes * 0.2

            total_bytes = vector_size_bytes + metadata_size_bytes + hnsw_overhead
            size_mb = total_bytes / (1024 * 1024)

            return size_mb
        except Exception as e:
            logger.warning(f"Could not estimate size: {e}")
            return 0.0

    def save_async_sync_wrapper(self, path: Optional[str] = None, metadata_path: Optional[str] = None):
        """Compatibility wrapper (no-op for ChromaDB).

        Args:
            path: Ignored
            metadata_path: Ignored
        """
        self.save(path, metadata_path)

    def load_async_sync_wrapper(self, path: Optional[str] = None, metadata_path: Optional[str] = None):
        """Compatibility wrapper (no-op for ChromaDB).

        Args:
            path: Ignored
            metadata_path: Ignored
        """
        self.load(path, metadata_path)


# Singleton instance
_vector_store: Optional[ChromaVectorStore] = None
_vector_store_lock = threading.Lock()


def get_vector_store(
    dimension: int = 384,
    collection_name: str = "rag_documents",
    persist_directory: Optional[str] = None
) -> ChromaVectorStore:
    """Get or create the global vector store (thread-safe).

    Args:
        dimension: Vector dimension
        collection_name: Name of the ChromaDB collection
        persist_directory: Directory to persist data

    Returns:
        Vector store instance
    """
    global _vector_store

    if _vector_store is None:
        with _vector_store_lock:
            if _vector_store is None:
                _vector_store = ChromaVectorStore(
                    dimension=dimension,
                    collection_name=collection_name,
                    persist_directory=persist_directory
                )

    return _vector_store


# Maintain backward compatibility alias
FAISSVectorStore = ChromaVectorStore


if __name__ == "__main__":
    # Test vector store
    print("Testing ChromaDB Vector Store...")

    # Initialize store
    store = get_vector_store(dimension=384)

    # Create sample embeddings
    num_samples = 100
    dimension = 384
    embeddings = np.random.randn(num_samples, dimension).astype(np.float32)

    texts = [f"Sample document {i}: This is test content." for i in range(num_samples)]
    sources = [f"source_{i % 5}.txt" for i in range(num_samples)]

    # Add vectors
    print(f"\nAdding {num_samples} vectors...")
    ids = store.add(embeddings, texts, sources=sources)
    print(f"Added vectors with IDs: {ids[:5]}... (showing first 5)")

    # Test search
    query = np.random.randn(dimension).astype(np.float32)
    print("\nSearching for similar vectors...")
    results = store.search(query, top_k=5)

    print(f"Top {len(results)} results:")
    for meta, score in results:
        print(f"  ID: {meta.id}, Score: {score:.4f}, Source: {meta.source}")
        print(f"    Text: {meta.text[:50]}...")

    # Get statistics
    stats = store.get_statistics()
    print("\nVector Store Statistics:")
    for key, value in stats.items():
        if key != "sources":
            print(f"  {key}: {value}")

    # Test persistence
    print("\nTesting persistence...")
    print("ChromaDB persists automatically - no manual save needed")

    # Delete test
    print("\nTesting deletion...")
    delete_ids = ids[:10]
    deleted = store.delete(delete_ids)
    print(f"Deleted {deleted} vectors")
    print(f"Remaining vectors: {store.collection.count()}")

    print("\nChromaDB vector store tests completed successfully!")
