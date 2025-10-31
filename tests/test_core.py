"""Core component tests for RAG-CLI - Simplified test suite."""

import pytest
import pytest_asyncio
import asyncio
import numpy as np
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
import sys
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from core.embeddings import EmbeddingGenerator, EmbeddingCache
from core.vector_store import FAISSVectorStore
from core.document_processor import DocumentProcessor
from core.retrieval_pipeline import HybridRetriever
from core.claude_integration import ClaudeAssistant


class TestEmbeddingCache:
    """Test embedding cache functionality."""

    def test_cache_get_put(self):
        """Test cache put and get operations."""
        cache = EmbeddingCache(cache_size=10)

        text = "test text"
        embedding = np.array([0.1, 0.2, 0.3])

        # Put in cache
        cache.put(text, embedding)

        # Get from cache
        retrieved = cache.get(text)

        assert retrieved is not None
        assert np.array_equal(retrieved, embedding)

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache exceeds size."""
        cache = EmbeddingCache(cache_size=2)

        # Fill cache
        cache.put("text1", np.array([0.1]))
        cache.put("text2", np.array([0.2]))

        # Add third item, should evict text1
        cache.put("text3", np.array([0.3]))

        assert cache.get("text1") is None  # Evicted
        assert cache.get("text2") is not None  # Still there
        assert cache.get("text3") is not None  # New item

    def test_cache_size_info(self):
        """Test cache info reporting."""
        cache = EmbeddingCache(cache_size=100)
        cache.put("text1", np.array([0.1]))

        info = cache.info()
        assert info["max_size"] == 100
        assert info["current_size"] == 1
        assert info["utilization"] == 0.01


class TestVectorStore:
    """Test FAISS vector store."""

    @patch('src.core.vector_store.get_config')
    def test_index_creation(self, mock_get_config):
        """Test FAISS index creation."""
        config = Mock()
        config.vector_store.save_path = "./data/vectors/vectors.index"
        config.vector_store.metadata_path = "./data/vectors/metadata.json"
        config.vector_store.auto_save = False
        config.vector_store.backup_enabled = False
        config.vector_store.backup_count = 0
        mock_get_config.return_value = config

        store = FAISSVectorStore(dimension=384, index_type="flat")

        assert store.index is not None
        assert store.index.d == 384
        assert store.index.ntotal == 0

    @patch('src.core.vector_store.get_config')
    def test_add_and_search(self, mock_get_config):
        """Test adding documents and searching."""
        config = Mock()
        config.vector_store.save_path = "./data/vectors/vectors.index"
        config.vector_store.metadata_path = "./data/vectors/metadata.json"
        config.vector_store.auto_save = False
        config.vector_store.backup_enabled = False
        config.vector_store.backup_count = 0
        mock_get_config.return_value = config

        store = FAISSVectorStore(dimension=3, index_type="flat")

        # Add documents
        embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        # add() method requires source text
        store.add(embeddings, texts=["text1", "text2"])

        assert store.index.ntotal == 2

        # Search
        query_embedding = np.array([[0.1, 0.2, 0.3]])
        distances, indices = store.index.search(query_embedding, 1)
        assert len(indices[0]) == 1


class TestVectorStoreAsync:
    """Test async file I/O operations for vector store."""

    @pytest.fixture
    def temp_paths(self):
        """Create temporary paths for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        index_path = str(temp_dir / "test_vectors.index")
        metadata_path = str(temp_dir / "test_metadata.json")

        yield index_path, metadata_path

        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()
        config.vector_store.save_path = "./data/vectors/vectors.index"
        config.vector_store.metadata_path = "./data/vectors/metadata.json"
        config.vector_store.auto_save = False
        config.vector_store.backup_enabled = False
        config.vector_store.backup_count = 0
        return config

    @patch('core.vector_store.get_config')
    def test_async_save_and_load(self, mock_get_config, temp_paths, mock_config):
        """Test async save and load operations."""
        mock_get_config.return_value = mock_config
        index_path, metadata_path = temp_paths

        # Create store and add data
        store = FAISSVectorStore(dimension=3, index_type="flat")
        embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        store.add(embeddings, texts=["text1", "text2"])

        assert store.index.ntotal == 2

        # Save using async method (wrapper for sync context)
        store.save_async_sync_wrapper(path=index_path, metadata_path=metadata_path)

        # Verify files were created
        assert Path(index_path).exists()
        assert Path(metadata_path).exists()

        # Create new store and load
        new_store = FAISSVectorStore(dimension=3, index_type="flat")
        new_store.load_async_sync_wrapper(path=index_path, metadata_path=metadata_path)

        assert new_store.index.ntotal == 2
        assert len(new_store.metadata) == 2

    @patch('core.vector_store.get_config')
    def test_async_save_performance(self, mock_get_config, temp_paths, mock_config):
        """Test async save is faster than sync save for large datasets."""
        mock_get_config.return_value = mock_config
        index_path, metadata_path = temp_paths

        # Create store with moderate dataset
        store = FAISSVectorStore(dimension=384, index_type="flat")

        # Generate 1000 vectors
        embeddings = np.random.rand(1000, 384).astype(np.float32)
        texts = [f"text_{i}" for i in range(1000)]
        store.add(embeddings, texts=texts)

        # Measure sync save time
        sync_start = time.perf_counter()
        store.save(path=index_path, metadata_path=metadata_path)
        sync_time = time.perf_counter() - sync_start

        # Measure async save time
        async_start = time.perf_counter()
        store.save_async_sync_wrapper(path=index_path, metadata_path=metadata_path)
        async_time = time.perf_counter() - async_start

        # Async should be comparable or faster
        # (Note: For small datasets, overhead might make it similar)
        assert async_time <= sync_time * 1.5  # Allow 50% margin

        print(f"\nSync save: {sync_time:.4f}s, Async save: {async_time:.4f}s")
        print(f"Speedup: {sync_time / async_time:.2f}x")

    @patch('core.vector_store.get_config')
    def test_async_load_performance(self, mock_get_config, temp_paths, mock_config):
        """Test async load is faster than sync load for large datasets."""
        mock_get_config.return_value = mock_config
        index_path, metadata_path = temp_paths

        # Create and save dataset
        store = FAISSVectorStore(dimension=384, index_type="flat")
        embeddings = np.random.rand(1000, 384).astype(np.float32)
        texts = [f"text_{i}" for i in range(1000)]
        store.add(embeddings, texts=texts)
        store.save(path=index_path, metadata_path=metadata_path)

        # Measure sync load time
        sync_store = FAISSVectorStore(dimension=384, index_type="flat")
        sync_start = time.perf_counter()
        sync_store.load(path=index_path, metadata_path=metadata_path)
        sync_time = time.perf_counter() - sync_start

        # Measure async load time
        async_store = FAISSVectorStore(dimension=384, index_type="flat")
        async_start = time.perf_counter()
        async_store.load_async_sync_wrapper(path=index_path, metadata_path=metadata_path)
        async_time = time.perf_counter() - async_start

        # Both stores should have same data
        assert async_store.index.ntotal == sync_store.index.ntotal
        assert len(async_store.metadata) == len(sync_store.metadata)

        # Async should be comparable or faster
        assert async_time <= sync_time * 1.5  # Allow 50% margin

        print(f"\nSync load: {sync_time:.4f}s, Async load: {async_time:.4f}s")
        print(f"Speedup: {sync_time / async_time:.2f}x")

    @pytest.mark.asyncio
    @patch('core.vector_store.get_config')
    async def test_native_async_save(self, mock_get_config, temp_paths, mock_config):
        """Test native async save in async context."""
        mock_get_config.return_value = mock_config
        index_path, metadata_path = temp_paths

        # Create store and add data
        store = FAISSVectorStore(dimension=3, index_type="flat")
        embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        store.add(embeddings, texts=["text1", "text2"])

        # Save using native async method
        await store.save_async(path=index_path, metadata_path=metadata_path)

        # Verify files were created
        assert Path(index_path).exists()
        assert Path(metadata_path).exists()

    @pytest.mark.asyncio
    @patch('core.vector_store.get_config')
    async def test_native_async_load(self, mock_get_config, temp_paths, mock_config):
        """Test native async load in async context."""
        mock_get_config.return_value = mock_config
        index_path, metadata_path = temp_paths

        # Create and save dataset
        store = FAISSVectorStore(dimension=3, index_type="flat")
        embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        store.add(embeddings, texts=["text1", "text2"])
        await store.save_async(path=index_path, metadata_path=metadata_path)

        # Load using native async method
        new_store = FAISSVectorStore(dimension=3, index_type="flat")
        await new_store.load_async(path=index_path, metadata_path=metadata_path)

        assert new_store.index.ntotal == 2
        assert len(new_store.metadata) == 2


class TestDocumentProcessor:
    """Test document processing."""

    def test_text_chunking(self):
        """Test document chunking."""
        processor = DocumentProcessor()

        # Create long text (need >500 tokens to get multiple chunks)
        text = " ".join(["word"] * 500)  # This will create ~500 tokens
        # process_text returns list of DocumentChunk objects
        chunks = processor.process_text(text, source="test.txt")

        assert len(chunks) >= 1  # At least one chunk
        assert all(hasattr(chunk, 'content') for chunk in chunks)
        assert all(hasattr(chunk, 'metadata') for chunk in chunks)
        # Verify chunk metadata structure
        assert chunks[0].chunk_id is not None
        assert chunks[0].source == "test.txt"


class TestQueryClassifier:
    """Test query classification functionality."""

    def test_intent_detection(self):
        """Test query intent detection."""
        from core.query_classifier import QueryClassifier, QueryIntent, get_query_classifier

        classifier = get_query_classifier()

        # Test different intent types
        result = classifier.classify("How to fix ImportError?")
        assert result is not None


class TestClaudeIntegration:
    """Test Claude API integration."""

    def test_assistant_creation(self):
        """Test Claude assistant initialization."""
        config = Mock()
        config.claude.api_key = "test-key"
        config.claude.model = "claude-haiku"
        config.claude.max_tokens = 1000
        config.claude.temperature = 0.7

        assistant = ClaudeAssistant(config)

        # Just verify it was created successfully
        assert assistant is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
