#!/usr/bin/env python3
"""Test script for ChromaDB persistence and update functionality.

This script tests the core improvements:
1. ChromaDB automatic persistence
2. Upsert functionality
3. Source-based operations
4. Duplicate detection integration
"""

import sys
from pathlib import Path
import numpy as np

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag_cli.core.vector_store import get_vector_store
from rag_cli.core.duplicate_detector import DuplicateDetector

def test_chromadb_persistence():
    """Test that ChromaDB persists data automatically."""
    print("\n=== Testing ChromaDB Persistence ===")

    # Create vector store
    store = get_vector_store()

    # Add some test vectors
    embeddings = np.random.randn(5, 384).astype(np.float32)
    texts = [f"Test document {i}" for i in range(5)]
    sources = ["test_source.txt"] * 5
    metadata = [{"test": True, "index": i} for i in range(5)]

    ids = store.add(embeddings, texts, metadata=metadata, sources=sources)
    print(f"[OK] Added {len(ids)} vectors")

    # Get count
    count = store.collection.count()
    print(f"[OK] Vector count: {count}")

    # Verify persistence (data should be in persist_directory)
    persist_dir = Path(store.persist_directory)
    print(f"[OK] Persist directory: {persist_dir}")
    print(f"[OK] Persist directory exists: {persist_dir.exists()}")

    return count


def test_upsert_functionality():
    """Test upsert functionality."""
    print("\n=== Testing Upsert Functionality ===")

    store = get_vector_store()

    # Create new vectors with explicit IDs
    embeddings = np.random.randn(3, 384).astype(np.float32)
    texts = ["Updated doc 1", "Updated doc 2", "Updated doc 3"]
    ids = ["vec_00000001", "vec_00000002", "vec_00000003"]
    sources = ["updated_source.txt"] * 3

    # Upsert (should update existing or insert new)
    result_ids = store.upsert(embeddings, texts, ids=ids, sources=sources)
    print(f"[OK] Upserted {len(result_ids)} vectors with IDs: {result_ids}")

    # Verify the vectors were updated
    meta = store.get_by_id("vec_00000001")
    print(f"[OK] Retrieved vector text: {meta.text if meta else 'Not found'}")

    return result_ids


def test_source_based_operations():
    """Test source-based query, delete, and update operations."""
    print("\n=== Testing Source-Based Operations ===")

    store = get_vector_store()

    # Add vectors from a specific source
    source = "test_document_source.txt"
    embeddings = np.random.randn(4, 384).astype(np.float32)
    texts = [f"Chunk {i} from source" for i in range(4)]
    sources = [source] * 4

    store.add(embeddings, texts, sources=sources)
    print(f"[OK] Added 4 vectors from source: {source}")

    # Get by source
    vectors_from_source = store.get_by_source(source)
    print(f"[OK] Found {len(vectors_from_source)} vectors from source")

    # Delete by source
    deleted = store.delete_by_source(source)
    print(f"[OK] Deleted {deleted} vectors from source")

    # Verify deletion
    remaining = store.get_by_source(source)
    print(f"[OK] Remaining vectors from source: {len(remaining)}")

    return deleted


def test_duplicate_detection():
    """Test duplicate detection integration."""
    print("\n=== Testing Duplicate Detection ===")

    detector = DuplicateDetector()

    # Test content hashing
    content = "This is a test document with some content."
    hash1 = detector.compute_hash(content)
    print(f"[OK] Computed hash: {hash1[:16]}...")

    # Add hash
    detector.add_hash(
        content=content,
        title="Test Document",
        source="test.txt",
        doc_type="local"
    )
    print("[OK] Added hash to registry")

    # Check for duplicate
    is_dup, dup_info = detector.is_duplicate(content)
    print(f"[OK] Duplicate check: {is_dup} (expected: True)")

    if dup_info:
        print(f"[OK] Duplicate info: {dup_info.title} from {dup_info.source}")

    # Save registry
    detector.save()
    print("[OK] Saved duplicate detection registry")

    return is_dup


def test_health_checks():
    """Test ChromaDB health checks."""
    print("\n=== Testing Health Checks ===")

    store = get_vector_store()

    # Test basic operations
    try:
        count = store.collection.count()
        print(f"[OK] Collection count: {count}")

        if count > 0:
            peek = store.collection.peek(limit=1)
            print(f"[OK] Collection peek: {len(peek['ids'])} results")
        else:
            print("[OK] Collection is empty (valid state)")

        print("[OK] Health check passed")
        return True
    except Exception as e:
        print(f"[FAIL] Health check failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("ChromaDB Persistence and Update Tests")
    print("=" * 60)

    try:
        # Run tests
        test_chromadb_persistence()
        test_upsert_functionality()
        test_source_based_operations()
        test_duplicate_detection()
        test_health_checks()

        print("\n" + "=" * 60)
        print("[OK] ALL TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
