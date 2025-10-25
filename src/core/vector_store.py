"""FAISS vector store implementation for RAG-CLI.

This module provides efficient vector storage and similarity search
using Facebook's FAISS library with metadata management.
"""

import os
import pickle
import time
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import faiss

from src.core.config import get_config
from src.monitoring.logger import get_logger, get_metrics_logger, log_execution_time


logger = get_logger(__name__)
metrics = get_metrics_logger()


@dataclass
class VectorMetadata:
    """Metadata for a stored vector."""
    id: str
    text: str
    source: str
    timestamp: datetime
    metadata: Dict[str, Any]


class FAISSVectorStore:
    """FAISS-based vector store with metadata management."""

    def __init__(
        self,
        dimension: int = 384,
        index_type: str = "auto",
        save_path: Optional[str] = None,
        metadata_path: Optional[str] = None
    ):
        """Initialize FAISS vector store.

        Args:
            dimension: Dimension of vectors
            index_type: Type of index (auto, flat, hnsw, ivf)
            save_path: Path to save index
            metadata_path: Path to save metadata
        """
        config = get_config()
        self.dimension = dimension
        self.index_type = index_type if index_type != "auto" else self._determine_index_type()
        self.save_path = save_path or config.vector_store.save_path
        self.metadata_path = metadata_path or config.vector_store.metadata_path
        self.auto_save = config.vector_store.auto_save
        self.backup_enabled = config.vector_store.backup_enabled
        self.backup_count = config.vector_store.backup_count

        # Initialize index and metadata
        self.index = self._create_index()
        self.metadata: List[VectorMetadata] = []
        self.id_counter = 0

        logger.info(
            "Vector store initialized",
            dimension=dimension,
            index_type=self.index_type
        )

    def _determine_index_type(self, estimated_vectors: int = 10000) -> str:
        """Automatically determine best index type based on estimated size.

        Args:
            estimated_vectors: Estimated number of vectors

        Returns:
            Recommended index type
        """
        if estimated_vectors < 100000:
            return "flat"
        elif estimated_vectors < 1000000:
            return "hnsw"
        else:
            return "ivf"

    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on configuration.

        Returns:
            FAISS index
        """
        config = get_config()
        params = config.vector_store.index_params

        logger.debug(f"Creating FAISS index", index_type=self.index_type)

        if self.index_type == "flat":
            # Exact search with L2 distance
            metric = params.get("flat", {}).get("metric", "l2")
            if metric == "l2":
                index = faiss.IndexFlatL2(self.dimension)
            else:  # inner product
                index = faiss.IndexFlatIP(self.dimension)

        elif self.index_type == "hnsw":
            # Hierarchical Navigable Small World graph
            hnsw_params = params.get("hnsw", {})
            M = hnsw_params.get("M", 32)
            index = faiss.IndexHNSWFlat(self.dimension, M)
            index.hnsw.efConstruction = hnsw_params.get("efConstruction", 200)
            index.hnsw.efSearch = hnsw_params.get("efSearch", 100)

        elif self.index_type == "ivf":
            # Inverted file index with clustering
            ivf_params = params.get("ivf", {})
            nlist = ivf_params.get("nlist", 100)
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            index.nprobe = ivf_params.get("nprobe", 10)

        else:
            # Default to flat index
            logger.warning(f"Unknown index type: {self.index_type}, using flat")
            index = faiss.IndexFlatL2(self.dimension)

        logger.info(f"Created {self.index_type} index")
        return index

    @log_execution_time
    def add(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        sources: Optional[List[str]] = None
    ) -> List[str]:
        """Add vectors with metadata to the store.

        Args:
            embeddings: Vector embeddings to add
            texts: Text content for each vector
            metadata: Optional metadata for each vector
            sources: Optional source identifiers

        Returns:
            List of generated IDs
        """
        if len(embeddings) != len(texts):
            raise ValueError("Number of embeddings must match number of texts")

        # Ensure embeddings are float32
        embeddings = np.array(embeddings, dtype=np.float32)

        # Handle single vector
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

        # Add to index
        if self.index_type == "ivf" and not self.index.is_trained:
            logger.info("Training IVF index")
            self.index.train(embeddings)

        self.index.add(embeddings)

        # Store metadata
        for i in range(num_vectors):
            meta = VectorMetadata(
                id=ids[i],
                text=texts[i],
                source=sources[i] if sources else "unknown",
                timestamp=datetime.now(),
                metadata=metadata[i] if metadata else {}
            )
            self.metadata.append(meta)

        # Record metrics
        elapsed = time.time() - start_time
        vectors_per_second = num_vectors / elapsed
        logger.info(
            f"Added vectors to store",
            count=num_vectors,
            elapsed_seconds=elapsed,
            vectors_per_second=vectors_per_second
        )
        metrics.record_count("vectors_added", num_vectors)
        metrics.record_gauge("total_vectors", self.index.ntotal)

        # Auto-save if enabled
        if self.auto_save:
            self.save()

        return ids

    @log_execution_time
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Tuple[VectorMetadata, float]]:
        """Search for similar vectors.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            threshold: Optional similarity threshold

        Returns:
            List of (metadata, score) tuples
        """
        if self.index.ntotal == 0:
            logger.warning("Search called on empty index")
            return []

        # Ensure query is float32 and 2D
        query_embedding = np.array(query_embedding, dtype=np.float32)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        start_time = time.time()

        # Search
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        # Get results with metadata
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                # Convert L2 distance to similarity score
                score = 1.0 / (1.0 + dist) if dist >= 0 else 0.0

                # Apply threshold if specified
                if threshold is None or score >= threshold:
                    results.append((self.metadata[idx], score))

        # Record metrics
        elapsed = time.time() - start_time
        logger.debug(
            f"Vector search completed",
            top_k=top_k,
            results=len(results),
            elapsed_seconds=elapsed
        )
        metrics.record_latency("vector_search", elapsed * 1000)

        return results

    def get_by_id(self, vector_id: str) -> Optional[VectorMetadata]:
        """Get metadata by vector ID.

        Args:
            vector_id: ID of the vector

        Returns:
            Vector metadata or None if not found
        """
        for meta in self.metadata:
            if meta.id == vector_id:
                return meta
        return None

    def delete(self, vector_ids: List[str]) -> int:
        """Delete vectors by ID (rebuilds index).

        Args:
            vector_ids: List of vector IDs to delete

        Returns:
            Number of vectors deleted
        """
        if not vector_ids:
            return 0

        # Find indices to keep
        id_set = set(vector_ids)
        indices_to_keep = []
        new_metadata = []

        for i, meta in enumerate(self.metadata):
            if meta.id not in id_set:
                indices_to_keep.append(i)
                new_metadata.append(meta)

        deleted_count = len(self.metadata) - len(new_metadata)

        if deleted_count > 0:
            # Rebuild index with remaining vectors
            if indices_to_keep:
                # Get vectors to keep
                vectors_to_keep = []
                for idx in indices_to_keep:
                    vector = self.index.reconstruct(idx)
                    vectors_to_keep.append(vector)

                vectors_array = np.array(vectors_to_keep, dtype=np.float32)

                # Create new index
                self.index = self._create_index()
                if self.index_type == "ivf":
                    self.index.train(vectors_array)
                self.index.add(vectors_array)
            else:
                # All vectors deleted
                self.index = self._create_index()

            self.metadata = new_metadata

            logger.info(f"Deleted vectors", count=deleted_count)
            metrics.record_count("vectors_deleted", deleted_count)

            if self.auto_save:
                self.save()

        return deleted_count

    def save(self, path: Optional[str] = None, metadata_path: Optional[str] = None):
        """Save index and metadata to disk.

        Args:
            path: Optional path to save index
            metadata_path: Optional path to save metadata
        """
        index_path = path or self.save_path
        meta_path = metadata_path or self.metadata_path

        # Create directories if needed
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(meta_path).parent.mkdir(parents=True, exist_ok=True)

        # Backup if enabled
        if self.backup_enabled:
            self._create_backup(index_path, meta_path)

        try:
            # Save index
            faiss.write_index(self.index, index_path)

            # Save metadata
            with open(meta_path, 'wb') as f:
                pickle.dump(self.metadata, f)

            logger.info(
                "Vector store saved",
                index_path=index_path,
                metadata_path=meta_path,
                vectors=self.index.ntotal
            )
            metrics.record_success("vector_store_save")

        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            metrics.record_failure("vector_store_save", str(e))
            raise

    def load(self, path: Optional[str] = None, metadata_path: Optional[str] = None):
        """Load index and metadata from disk.

        Args:
            path: Optional path to load index from
            metadata_path: Optional path to load metadata from
        """
        index_path = path or self.save_path
        meta_path = metadata_path or self.metadata_path

        if not Path(index_path).exists():
            logger.warning(f"Index file not found: {index_path}")
            return

        if not Path(meta_path).exists():
            logger.warning(f"Metadata file not found: {meta_path}")
            return

        try:
            # Load index
            self.index = faiss.read_index(index_path)

            # Load metadata
            with open(meta_path, 'rb') as f:
                self.metadata = pickle.load(f)

            # Update ID counter
            if self.metadata:
                last_id = max(int(m.id.split('_')[1]) for m in self.metadata)
                self.id_counter = last_id + 1

            logger.info(
                "Vector store loaded",
                index_path=index_path,
                metadata_path=meta_path,
                vectors=self.index.ntotal
            )
            metrics.record_success("vector_store_load")

        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            metrics.record_failure("vector_store_load", str(e))
            raise

    def _create_backup(self, index_path: str, metadata_path: str):
        """Create backup of existing files.

        Args:
            index_path: Path to index file
            metadata_path: Path to metadata file
        """
        if not Path(index_path).exists():
            return

        # Create backup directory
        backup_dir = Path(index_path).parent / "backups"
        backup_dir.mkdir(exist_ok=True)

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Backup index
        index_backup = backup_dir / f"vectors_{timestamp}.index"
        shutil.copy2(index_path, index_backup)

        # Backup metadata
        if Path(metadata_path).exists():
            meta_backup = backup_dir / f"metadata_{timestamp}.pkl"
            shutil.copy2(metadata_path, meta_backup)

        # Clean old backups
        self._cleanup_backups(backup_dir)

        logger.debug(f"Created backup", timestamp=timestamp)

    def _cleanup_backups(self, backup_dir: Path):
        """Remove old backups keeping only the most recent ones.

        Args:
            backup_dir: Directory containing backups
        """
        # Get all backup files
        index_backups = sorted(backup_dir.glob("vectors_*.index"))
        meta_backups = sorted(backup_dir.glob("metadata_*.pkl"))

        # Keep only the most recent backups
        if len(index_backups) > self.backup_count:
            for backup in index_backups[:-self.backup_count]:
                backup.unlink()
                logger.debug(f"Removed old backup: {backup.name}")

        if len(meta_backups) > self.backup_count:
            for backup in meta_backups[:-self.backup_count]:
                backup.unlink()

    def clear(self):
        """Clear all vectors and metadata."""
        self.index = self._create_index()
        self.metadata = []
        self.id_counter = 0
        logger.info("Vector store cleared")

        if self.auto_save:
            self.save()

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metadata_count": len(self.metadata),
            "memory_usage_bytes": self.index.ntotal * self.dimension * 4,  # float32
        }

        if self.metadata:
            # Add metadata statistics
            sources = {}
            for meta in self.metadata:
                sources[meta.source] = sources.get(meta.source, 0) + 1

            stats["sources"] = sources
            stats["oldest_timestamp"] = min(m.timestamp for m in self.metadata)
            stats["newest_timestamp"] = max(m.timestamp for m in self.metadata)

        return stats


# Singleton instance
_vector_store: Optional[FAISSVectorStore] = None


def get_vector_store(
    dimension: int = 384,
    index_type: str = "auto"
) -> FAISSVectorStore:
    """Get or create the global vector store.

    Args:
        dimension: Vector dimension
        index_type: Type of index to use

    Returns:
        Vector store instance
    """
    global _vector_store

    if _vector_store is None:
        _vector_store = FAISSVectorStore(dimension, index_type)

        # Try to load existing index
        if Path(_vector_store.save_path).exists():
            _vector_store.load()

    return _vector_store


if __name__ == "__main__":
    # Test vector store
    print("Testing FAISS Vector Store...")

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
    print(f"\nVector Store Statistics:")
    for key, value in stats.items():
        if key != "sources":
            print(f"  {key}: {value}")

    # Save and load test
    print("\nTesting save/load...")
    store.save()
    print("Saved successfully")

    # Create new store and load
    new_store = FAISSVectorStore(dimension=384)
    new_store.load()
    print(f"Loaded {new_store.index.ntotal} vectors")

    # Delete test
    print("\nTesting deletion...")
    delete_ids = ids[:10]
    deleted = store.delete(delete_ids)
    print(f"Deleted {deleted} vectors")
    print(f"Remaining vectors: {store.index.ntotal}")

    print("\nVector store tests completed successfully!")