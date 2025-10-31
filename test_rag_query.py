#!/usr/bin/env python3
"""Quick test script to trigger RAG retrieval and populate dashboard metrics."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.config import get_config
from core.vector_store import get_vector_store
from core.embeddings import get_embedding_generator
from core.retrieval_pipeline import HybridRetriever
from core.claude_code_adapter import get_adapter

def test_retrieval():
    """Test RAG retrieval with a sample query."""
    print("Testing RAG-CLI system...")
    print("=" * 50)

    # Initialize components
    print("\n1. Loading configuration...")
    config = get_config()

    print("2. Loading vector store...")
    vector_store = get_vector_store()
    print(f"   Vectors loaded: {vector_store.index.ntotal}")

    print("3. Initializing retrieval pipeline...")
    retriever = HybridRetriever()

    # Test query - use command line argument if provided
    test_query = sys.argv[1] if len(sys.argv) > 1 else "What is the document processor chunk size?"
    print(f"\n4. Testing retrieval with query: '{test_query}'")

    results = retriever.retrieve(test_query, top_k=3)

    print(f"\n5. Retrieved {len(results)} documents:")
    for i, doc in enumerate(results, 1):
        print(f"\n   [{i}] Score: {doc.score:.2%}")
        print(f"       Source: {doc.source}")
        content_preview = doc.text[:150]
        print(f"       Content: {content_preview}...")

    # Format for Claude
    print("\n6. Formatting context for Claude Code...")
    adapter = get_adapter()
    response = adapter.format_context_for_claude(results, test_query)

    print(f"\n   Mode: {response.mode}")
    print(f"   Sources: {len(response.sources)}")
    print(f"   Documents found: {response.metadata['documents_found']}")

    print("\n" + "=" * 50)
    print("Test complete! Check the dashboard for updated metrics.")
    print("Dashboard: http://127.0.0.1:5000")
    print("=" * 50)

if __name__ == "__main__":
    try:
        test_retrieval()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
