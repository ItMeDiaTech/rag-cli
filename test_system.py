#!/usr/bin/env python3
"""Quick test script for RAG-CLI system."""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all core imports work."""
    print("Testing imports...")
    try:
        from src.core.config import get_config
        print("[OK] Config module")

        from src.core.embeddings import EmbeddingModel
        print("[OK] Embeddings module")

        from src.core.vector_store import VectorStore
        print("[OK] Vector store module")

        from src.core.document_processor import DocumentProcessor
        print("[OK] Document processor module")

        from src.core.retrieval_pipeline import HybridRetriever
        print("[OK] Retrieval pipeline module")

        from src.monitoring.logger import get_logger
        print("[OK] Logging module")

        return True
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        return False


def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")
    try:
        from src.core.config import get_config
        config = get_config()

        print(f"[OK] Config loaded")
        print(f"  - Embedding model: {config.embeddings.model_name}")
        print(f"  - Vector dimensions: {config.embeddings.model_dim}")
        print(f"  - Retrieval top_k: {config.retrieval.top_k}")

        return True
    except Exception as e:
        print(f"[FAIL] Config failed: {e}")
        return False


def test_document_processing():
    """Test document processing with sample files."""
    print("\nTesting document processing...")
    try:
        from src.core.document_processor import DocumentProcessor
        from src.core.config import get_config

        config = get_config()
        processor = DocumentProcessor(config)

        # Test with markdown files in data/documents
        docs_path = Path("data/documents")
        if not docs_path.exists():
            print("[FAIL] No documents folder found")
            return False

        md_files = list(docs_path.glob("*.md"))
        if not md_files:
            print("[FAIL] No markdown files found")
            return False

        # Process first markdown file
        test_file = md_files[0]
        print(f"  Processing: {test_file.name}")

        chunks = processor.process_file(test_file)
        print(f"[OK] Processed into {len(chunks)} chunks")

        if chunks:
            print(f"  - First chunk: {len(chunks[0]['content'])} chars")
            print(f"  - Metadata: {chunks[0].get('metadata', {})}")

        return True
    except Exception as e:
        print(f"[FAIL] Document processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embeddings():
    """Test embedding generation."""
    print("\nTesting embeddings...")
    try:
        from src.core.embeddings import EmbeddingModel
        from src.core.config import get_config

        config = get_config()
        model = EmbeddingModel(config)

        # Test single encoding
        test_text = "This is a test query about API authentication"
        embedding = model.encode(test_text)

        print(f"[OK] Generated embedding")
        print(f"  - Shape: {embedding.shape}")
        print(f"  - Dimension: {embedding.shape[0]}")
        print(f"  - Type: {embedding.dtype}")

        # Test batch encoding
        texts = ["Query 1", "Query 2", "Query 3"]
        embeddings = model.encode_batch(texts)

        print(f"[OK] Batch encoding")
        print(f"  - Batch shape: {embeddings.shape}")

        return True
    except Exception as e:
        print(f"[FAIL] Embedding generation failed: {e}")
        return False


def test_vector_store():
    """Test vector store operations."""
    print("\nTesting vector store...")
    try:
        from src.core.vector_store import VectorStore
        from src.core.config import get_config
        import numpy as np

        config = get_config()
        store = VectorStore(config)

        # Create test documents
        test_docs = [
            {"id": "1", "content": "API authentication methods", "metadata": {"source": "test1.md"}},
            {"id": "2", "content": "Error handling best practices", "metadata": {"source": "test2.md"}},
            {"id": "3", "content": "Performance optimization tips", "metadata": {"source": "test3.md"}}
        ]

        # Create random embeddings for testing (normally from embedding model)
        dim = config.embeddings.model_dim
        test_embeddings = np.random.randn(3, dim).astype(np.float32)

        # Add to store
        store.add_documents(test_docs, test_embeddings)
        print(f"[OK] Added {len(test_docs)} documents")
        print(f"  - Index size: {store.index.ntotal} vectors")

        # Test search
        query_embedding = np.random.randn(dim).astype(np.float32)
        results = store.search(query_embedding, top_k=2)

        print(f"[OK] Search completed")
        print(f"  - Found {len(results)} results")
        if results:
            print(f"  - Top result: {results[0].get('content', 'N/A')[:50]}...")

        return True
    except Exception as e:
        print(f"[FAIL] Vector store failed: {e}")
        return False


def test_retrieval_pipeline():
    """Test the full retrieval pipeline."""
    print("\nTesting retrieval pipeline...")
    try:
        from src.core.vector_store import VectorStore
        from src.core.embeddings import EmbeddingModel
        from src.core.retrieval_pipeline import HybridRetriever
        from src.core.config import get_config

        config = get_config()

        # Initialize components
        embedding_model = EmbeddingModel(config)
        vector_store = VectorStore(config)

        # Add test data
        test_docs = [
            {"content": "Authentication using API keys is the simplest method", "metadata": {"source": "auth.md"}},
            {"content": "OAuth 2.0 provides more secure authentication", "metadata": {"source": "oauth.md"}},
            {"content": "Error handling should use proper status codes", "metadata": {"source": "errors.md"}}
        ]

        embeddings = embedding_model.encode_batch([doc["content"] for doc in test_docs])
        vector_store.add_documents(test_docs, embeddings)

        # Create retriever
        retriever = HybridRetriever(vector_store, embedding_model, config)

        # Test retrieval
        query = "How to authenticate API requests?"
        results = retriever.vector_search(query, top_k=2)

        print(f"[OK] Retrieval completed")
        print(f"  - Query: '{query}'")
        print(f"  - Found {len(results)} results")

        for i, result in enumerate(results[:2], 1):
            print(f"  - Result {i}: {result.text[:50]}...")
            print(f"    Score: {result.score:.3f}")

        return True
    except Exception as e:
        print(f"[FAIL] Retrieval pipeline failed: {e}")
        return False


def test_monitoring():
    """Test monitoring components."""
    print("\nTesting monitoring...")
    try:
        from src.monitoring.tcp_server import MetricsCollector

        collector = MetricsCollector()

        # Record some test metrics
        collector.record_latency("test_operation", 123.45)
        collector.record_query()
        collector.record_cache(hit=True)

        print(f"[OK] Metrics collection")
        print(f"  - Queries: {collector.query_count}")
        print(f"  - Cache hits: {collector.cache_hits}")
        print(f"  - Uptime: {collector.get_uptime():.2f}s")

        return True
    except Exception as e:
        print(f"[FAIL] Monitoring failed: {e}")
        return False


def run_performance_test():
    """Run basic performance benchmarks."""
    print("\nRunning performance benchmarks...")

    results = {
        "vector_search": None,
        "embedding_generation": None,
        "document_processing": None
    }

    try:
        from src.core.embeddings import EmbeddingModel
        from src.core.vector_store import VectorStore
        from src.core.config import get_config
        import time

        config = get_config()

        # Test embedding speed
        model = EmbeddingModel(config)
        test_text = "This is a test query for performance measurement"

        start = time.time()
        for _ in range(10):
            _ = model.encode(test_text)
        embedding_time = (time.time() - start) / 10 * 1000
        results["embedding_generation"] = embedding_time
        print(f"  - Embedding generation: {embedding_time:.2f}ms")

        # Test vector search speed
        store = VectorStore(config)
        dim = config.embeddings.model_dim

        # Add 1000 random vectors
        docs = [{"id": str(i), "content": f"Doc {i}", "metadata": {}} for i in range(1000)]
        embeddings = np.random.randn(1000, dim).astype(np.float32)
        store.add_documents(docs, embeddings)

        query_embedding = np.random.randn(dim).astype(np.float32)

        start = time.time()
        for _ in range(10):
            _ = store.search(query_embedding, top_k=5)
        search_time = (time.time() - start) / 10 * 1000
        results["vector_search"] = search_time
        print(f"  - Vector search (1000 docs): {search_time:.2f}ms")

        # Check against targets
        print("\n  Performance vs Targets:")
        if search_time < 100:
            print(f"  [OK] Vector search <100ms target (actual: {search_time:.2f}ms)")
        else:
            print(f"  [FAIL] Vector search >100ms target (actual: {search_time:.2f}ms)")

        return results
    except Exception as e:
        print(f"[FAIL] Performance test failed: {e}")
        return results


def main():
    """Run all tests."""
    print("=" * 60)
    print("RAG-CLI System Test Suite")
    print("=" * 60)

    test_results = {
        "imports": test_imports(),
        "configuration": test_configuration(),
        "document_processing": test_document_processing(),
        "embeddings": test_embeddings(),
        "vector_store": test_vector_store(),
        "retrieval_pipeline": test_retrieval_pipeline(),
        "monitoring": test_monitoring()
    }

    # Performance tests
    print("\n" + "=" * 60)
    perf_results = run_performance_test()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for v in test_results.values() if v)
    total = len(test_results)

    for test_name, passed in test_results.items():
        status = "[OK] PASSED" if passed else "[FAIL] FAILED"
        print(f"{test_name:20} : {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if perf_results["vector_search"] is not None:
        print(f"\nPerformance Metrics:")
        print(f"  Vector Search: {perf_results['vector_search']:.2f}ms")
        print(f"  Embedding Gen: {perf_results['embedding_generation']:.2f}ms")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())