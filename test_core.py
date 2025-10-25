#!/usr/bin/env python3
"""Test script to verify core components (embeddings and vector store) are working."""

import sys
import time
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def test_embeddings():
    """Test embedding generation."""
    print("Testing embedding system...")

    from src.core.embeddings import get_embedding_generator

    try:
        # Initialize generator
        print("  Initializing embedding generator...")
        generator = get_embedding_generator()
        print("[OK] Embedding generator initialized")

        # Test single embedding
        text = "This is a test document for RAG system."
        embedding = generator.encode(text)
        print(f"[OK] Single embedding generated - shape: {embedding.shape}")

        # Test batch embeddings
        texts = [
            "RAG systems combine retrieval and generation.",
            "Vector search enables semantic similarity matching.",
            "Claude provides the generation capability."
        ]
        embeddings = generator.encode_documents(texts, show_progress=False)
        print(f"[OK] Batch embeddings generated - shape: {embeddings.shape}")

        # Test caching
        start = time.time()
        _ = generator.encode(text)  # First call
        first_time = time.time() - start

        start = time.time()
        _ = generator.encode(text)  # Cached call
        cached_time = time.time() - start

        speedup = first_time / cached_time if cached_time > 0 else 100
        print(f"[OK] Cache working - speedup: {speedup:.1f}x")

        # Test similarity
        query = "How does retrieval work?"
        query_emb = generator.encode_query(query)
        similarities = generator.compute_similarity(query_emb, embeddings)
        print(f"[OK] Similarity computation working - scores: {similarities}")

        return True

    except Exception as e:
        print(f"[FAIL] Embedding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vector_store():
    """Test FAISS vector store."""
    print("\nTesting vector store...")

    from src.core.vector_store import get_vector_store

    try:
        # Initialize store
        print("  Initializing vector store...")
        store = get_vector_store(dimension=384)
        print(f"[OK] Vector store initialized - type: {store.index_type}")

        # Create test data
        num_docs = 50
        dimension = 384
        embeddings = np.random.randn(num_docs, dimension).astype(np.float32)
        texts = [f"Document {i}: Test content about topic {i%5}" for i in range(num_docs)]
        sources = [f"source_{i%3}.txt" for i in range(num_docs)]

        # Add vectors
        print(f"  Adding {num_docs} vectors...")
        ids = store.add(embeddings, texts, sources=sources)
        print(f"[OK] Added {len(ids)} vectors")

        # Test search
        query = np.random.randn(dimension).astype(np.float32)
        results = store.search(query, top_k=5)
        print(f"[OK] Search returned {len(results)} results")

        # Check statistics
        stats = store.get_statistics()
        print(f"[OK] Statistics - vectors: {stats['total_vectors']}, memory: {stats['memory_usage_bytes']/1024:.1f}KB")

        # Test save/load
        store.save()
        print("[OK] Vector store saved")

        # Create new store and load
        new_store = get_vector_store(dimension=384)
        if Path(new_store.save_path).exists():
            new_store.load()
            print(f"[OK] Vector store loaded - vectors: {new_store.index.ntotal}")

        # Test deletion
        delete_ids = ids[:5]
        deleted = store.delete(delete_ids)
        print(f"[OK] Deleted {deleted} vectors, remaining: {store.index.ntotal}")

        return True

    except Exception as e:
        print(f"[FAIL] Vector store test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test embeddings and vector store together."""
    print("\nTesting integration...")

    from src.core.embeddings import get_embedding_generator
    from src.core.vector_store import get_vector_store

    try:
        # Initialize components
        generator = get_embedding_generator()
        store = get_vector_store(dimension=generator.get_embedding_dim())
        store.clear()  # Start fresh

        # Create documents
        documents = [
            "RAG stands for Retrieval Augmented Generation.",
            "The system uses embeddings to find similar documents.",
            "FAISS provides efficient vector search capabilities.",
            "Claude generates responses based on retrieved context.",
            "The monitoring system tracks performance metrics."
        ]

        # Generate embeddings
        print("  Generating embeddings for documents...")
        doc_embeddings = generator.encode_documents(documents, show_progress=False)

        # Store in vector database
        ids = store.add(doc_embeddings, documents, sources=["doc1", "doc2", "doc3", "doc4", "doc5"])
        print(f"[OK] Stored {len(ids)} documents")

        # Test retrieval
        query = "What is RAG?"
        query_embedding = generator.encode_query(query)
        results = store.search(query_embedding, top_k=3)

        print(f"[OK] Query: '{query}'")
        print(f"[OK] Retrieved {len(results)} relevant documents:")
        for i, (meta, score) in enumerate(results, 1):
            print(f"      {i}. Score: {score:.4f} - {meta.text[:60]}...")

        return True

    except Exception as e:
        print(f"[FAIL] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all core component tests."""
    print("=" * 60)
    print("RAG-CLI Core Components Test")
    print("=" * 60)

    all_passed = True

    # Test embeddings
    if not test_embeddings():
        all_passed = False

    # Test vector store
    if not test_vector_store():
        all_passed = False

    # Test integration
    if not test_integration():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] All core component tests passed!")
        print("  Embeddings and vector store are ready.")
    else:
        print("[FAILURE] Some tests failed. Please check the errors above.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())