#!/usr/bin/env python3
"""
Test script to verify ChromaDB health and indexing status.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_cli.core.vector_store import get_vector_store
from rag_cli.core.config import get_config

def main():
    print("=" * 60)
    print("ChromaDB Health Check")
    print("=" * 60)

    # Get configuration
    config = get_config()
    print(f"\nConfiguration:")
    print(f"  Save path: {config.vector_store.save_path}")

    # Get vector store
    print("\nInitializing vector store...")
    try:
        vector_store = get_vector_store()
        print(f"  Persist directory: {vector_store.persist_directory}")
        print(f"  Collection name: {vector_store.collection_name}")

        # Check vector count
        count = vector_store.collection.count()
        print(f"\n  Total vectors: {count}")

        if count > 0:
            # Get sample vectors
            print("\nSample vectors (first 5):")
            try:
                results = vector_store.collection.peek(limit=5)
                if results and 'ids' in results:
                    for i, (vec_id, metadata) in enumerate(zip(results['ids'], results['metadatas']), 1):
                        source = metadata.get('source', 'unknown')
                        title = metadata.get('title', 'untitled')
                        print(f"  {i}. {title} (source: {source})")
                else:
                    print("  No sample data available")
            except Exception as e:
                print(f"  Error getting sample: {e}")

            # Test retrieval
            print("\nTesting retrieval with sample query...")
            try:
                from rag_cli.core.embeddings import EmbeddingGenerator
                embed_gen = EmbeddingGenerator()

                query = "test query"
                query_embedding = embed_gen.encode(query)

                results = vector_store.search(
                    query_embedding=query_embedding,
                    top_k=3
                )

                print(f"  Retrieved {len(results)} results")
                for i, result in enumerate(results[:3], 1):
                    print(f"  {i}. Score: {result.score:.4f}, Source: {result.metadata.get('source', 'unknown')}")

            except Exception as e:
                print(f"  Retrieval test failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\n  No vectors indexed yet!")
            print("  Run indexing with: rag-index ./data/documents --recursive")

        # Check duplicate registry
        print("\nDuplicate Registry:")
        duplicate_registry_path = Path(vector_store.persist_directory).parent / "content_hashes.json"
        if duplicate_registry_path.exists():
            import json
            with open(duplicate_registry_path, 'r') as f:
                registry = json.load(f)
            hash_count = len(registry.get('hashes', {}))
            print(f"  Location: {duplicate_registry_path}")
            print(f"  Tracked documents: {hash_count}")
        else:
            print(f"  Registry not found at: {duplicate_registry_path}")

        print("\n" + "=" * 60)
        print("Health check completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
