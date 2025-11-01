"""Benchmark script to test async performance improvements.

This script tests:
1. Async retrieval (parallel vector + keyword search) vs sync
2. Parallel embedding encoding vs sequential
3. Parallel document processing vs sequential

Expected improvements:
- Retrieval: 30-40% latency reduction
- Embeddings: 2-3x speedup for large batches
- Document processing: 3-5x speedup for directories
"""

import time
import asyncio
from pathlib import Path
from typing import List
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from core.retrieval_pipeline import get_retriever
from core.embeddings import get_embedding_generator, get_embedding_pool
from core.document_processor import get_document_processor
from core.vector_store import get_vector_store
from monitoring.logger import get_logger


logger = get_logger(__name__)


def benchmark_retrieval_sync_vs_async():
    """Benchmark sync vs async retrieval performance."""
    print("\n" + "="*70)
    print("BENCHMARK 1: Retrieval Pipeline (Sync vs Async)")
    print("="*70)

    retriever = get_retriever()

    # Test queries
    queries = [
        "How to implement RAG systems?",
        "What is vector search?",
        "Explain embeddings and semantic similarity",
        "Best practices for document chunking",
        "How to optimize retrieval performance?"
    ]

    print(f"\nTesting with {len(queries)} queries...")

    # Benchmark sync retrieval
    print("\n[SYNC] Running sequential retrieval...")
    sync_times = []
    sync_start = time.time()

    for query in queries:
        start = time.time()
        results = retriever.retrieve(query, top_k=5, use_cache=False)
        elapsed = time.time() - start
        sync_times.append(elapsed)
        print(f"  Query: '{query[:40]}...' - {elapsed:.3f}s ({len(results)} results)")

    sync_total = time.time() - sync_start
    sync_avg = sum(sync_times) / len(sync_times)

    print(f"\nSync Total: {sync_total:.3f}s | Avg per query: {sync_avg:.3f}s")

    # Benchmark async retrieval
    print("\n[ASYNC] Running parallel retrieval...")
    async_times = []
    async_start = time.time()

    async def run_async_queries():
        times = []
        for query in queries:
            start = time.time()
            results = await retriever.retrieve_async(query, top_k=5, use_cache=False)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Query: '{query[:40]}...' - {elapsed:.3f}s ({len(results)} results)")
        return times

    async_times = asyncio.run(run_async_queries())
    async_total = time.time() - async_start
    async_avg = sum(async_times) / len(async_times)

    print(f"\nAsync Total: {async_total:.3f}s | Avg per query: {async_avg:.3f}s")

    # Calculate improvement
    improvement = ((sync_avg - async_avg) / sync_avg) * 100
    speedup = sync_avg / async_avg

    print(f"\n{'RESULTS':^70}")
    print("-"*70)
    print(f"Average Latency Reduction: {improvement:+.1f}%")
    print(f"Speedup Factor: {speedup:.2f}x")
    print(f"Target Achievement: {' PASS' if improvement >= 30 else ' BELOW TARGET'} (target: 30-40%)")

    return improvement


def benchmark_embedding_parallel():
    """Benchmark parallel vs sequential embedding encoding."""
    print("\n" + "="*70)
    print("BENCHMARK 2: Embedding Encoding (Sequential vs Parallel)")
    print("="*70)

    generator = get_embedding_generator()
    pool = get_embedding_pool()

    # Create test texts (100 medium-length documents)
    test_texts = [
        f"This is test document number {i}. It contains information about RAG systems, "
        f"embeddings, vector search, and various natural language processing topics. "
        f"The document discusses semantic similarity, chunking strategies, and retrieval optimization. "
        * 3  # Make documents longer
        for i in range(100)
    ]

    print(f"\nTesting with {len(test_texts)} documents...")

    # Sequential encoding
    print("\n[SEQUENTIAL] Encoding documents one batch at a time...")
    seq_start = time.time()
    seq_embeddings = generator.encode(test_texts, show_progress=False, use_cache=False)
    seq_elapsed = time.time() - seq_start
    seq_speed = len(test_texts) / seq_elapsed

    print(f"Sequential: {seq_elapsed:.3f}s | {seq_speed:.1f} docs/sec")

    # Parallel encoding
    print("\n[PARALLEL] Encoding documents with thread pool...")
    parallel_start = time.time()
    parallel_embeddings = pool.encode_parallel(test_texts, show_progress=False)
    parallel_elapsed = time.time() - parallel_start
    parallel_speed = len(test_texts) / parallel_elapsed

    print(f"Parallel: {parallel_elapsed:.3f}s | {parallel_speed:.1f} docs/sec")

    # Calculate improvement
    speedup = seq_elapsed / parallel_elapsed
    improvement = ((seq_elapsed - parallel_elapsed) / seq_elapsed) * 100

    print(f"\n{'RESULTS':^70}")
    print("-"*70)
    print(f"Speedup Factor: {speedup:.2f}x")
    print(f"Time Reduction: {improvement:.1f}%")
    print(f"Target Achievement: {' PASS' if speedup >= 2.0 else ' BELOW TARGET'} (target: 2-3x)")

    return speedup


def benchmark_document_processing():
    """Benchmark parallel vs sequential document processing."""
    print("\n" + "="*70)
    print("BENCHMARK 3: Document Processing (Sequential vs Parallel)")
    print("="*70)

    processor = get_document_processor()

    # Create a temporary test directory with documents
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 20 test markdown files
        num_files = 20
        print(f"\nCreating {num_files} test documents...")

        for i in range(num_files):
            file_path = Path(tmpdir) / f"test_doc_{i}.md"
            content = f"""# Test Document {i}

This is a test document for benchmarking parallel document processing.

## Section 1
Content about RAG systems and embeddings. This section discusses various aspects
of retrieval-augmented generation including vector search, semantic similarity,
and document chunking strategies.

## Section 2
More content about natural language processing, transformers, and language models.
This section explores different techniques for text processing and analysis.

## Section 3
Additional information about performance optimization, caching strategies, and
best practices for building efficient RAG systems at scale.

{'Extended content paragraph. ' * 20}
"""
            file_path.write_text(content, encoding='utf-8')

        print(f"Created {num_files} test files in {tmpdir}")

        # Sequential processing
        print("\n[SEQUENTIAL] Processing documents one at a time...")
        seq_start = time.time()
        seq_docs = processor.process_directory(tmpdir, recursive=False)
        seq_elapsed = time.time() - seq_start
        seq_speed = len(seq_docs) / seq_elapsed if seq_elapsed > 0 else 0

        print(f"Sequential: {seq_elapsed:.3f}s | {seq_speed:.1f} docs/sec | {len(seq_docs)} docs processed")

        # Parallel processing
        print("\n[PARALLEL] Processing documents with thread pool...")
        parallel_start = time.time()
        parallel_docs = processor.process_directory_parallel(tmpdir, recursive=False)
        parallel_elapsed = time.time() - parallel_start
        parallel_speed = len(parallel_docs) / parallel_elapsed if parallel_elapsed > 0 else 0

        print(f"Parallel: {parallel_elapsed:.3f}s | {parallel_speed:.1f} docs/sec | {len(parallel_docs)} docs processed")

        # Calculate improvement
        if parallel_elapsed > 0:
            speedup = seq_elapsed / parallel_elapsed
            improvement = ((seq_elapsed - parallel_elapsed) / seq_elapsed) * 100
        else:
            speedup = 1.0
            improvement = 0.0

        print(f"\n{'RESULTS':^70}")
        print("-"*70)
        print(f"Speedup Factor: {speedup:.2f}x")
        print(f"Time Reduction: {improvement:.1f}%")
        print(f"Target Achievement: {' PASS' if speedup >= 3.0 else ' BELOW TARGET'} (target: 3-5x)")

        return speedup


def main():
    """Run all benchmarks."""
    print("\n" + "="*70)
    print("RAG-CLI ASYNC PERFORMANCE BENCHMARK SUITE")
    print("Testing Phase 1 parallel processing improvements")
    print("="*70)

    # Initialize components first (cold start)
    print("\nInitializing components (cold start)...")
    from core.embeddings import get_embedding_generator
    from core.vector_store import get_vector_store
    from core.retrieval_pipeline import get_retriever

    generator = get_embedding_generator()
    store = get_vector_store()
    retriever = get_retriever()

    # Add some test documents to vector store for retrieval tests
    print("Setting up test data...")
    test_docs = [
        "RAG systems combine retrieval and generation for better responses.",
        "Vector search enables semantic similarity matching in documents.",
        "Embeddings capture semantic meaning in numerical form.",
        "Document chunking is crucial for effective RAG implementations.",
        "Hybrid search combines vector and keyword search methods.",
        "Cross-encoder models can rerank documents for better accuracy.",
        "Semantic caching improves performance by reducing redundant computations.",
        "HyDE (Hypothetical Document Embeddings) improves retrieval accuracy."
    ] * 5  # Duplicate to have more data

    embeddings = generator.encode_documents(test_docs, show_progress=False)
    store.clear()
    store.add(embeddings, test_docs, sources=[f"doc_{i}" for i in range(len(test_docs))])

    # Build BM25 index
    retriever._auto_build_bm25_index()

    print("Setup complete!\n")

    # Run benchmarks
    results = {}

    try:
        results['retrieval_improvement'] = benchmark_retrieval_sync_vs_async()
    except Exception as e:
        print(f"\n Retrieval benchmark failed: {e}")
        results['retrieval_improvement'] = 0

    try:
        results['embedding_speedup'] = benchmark_embedding_parallel()
    except Exception as e:
        print(f"\n Embedding benchmark failed: {e}")
        results['embedding_speedup'] = 1.0

    try:
        results['doc_processing_speedup'] = benchmark_document_processing()
    except Exception as e:
        print(f"\n Document processing benchmark failed: {e}")
        results['doc_processing_speedup'] = 1.0

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    print(f"\n1. Retrieval Pipeline:")
    print(f"   - Latency Reduction: {results['retrieval_improvement']:+.1f}%")
    print(f"   - Status: {' PASS' if results['retrieval_improvement'] >= 30 else ' BELOW TARGET'}")

    print(f"\n2. Embedding Encoding:")
    print(f"   - Speedup: {results['embedding_speedup']:.2f}x")
    print(f"   - Status: {' PASS' if results['embedding_speedup'] >= 2.0 else ' BELOW TARGET'}")

    print(f"\n3. Document Processing:")
    print(f"   - Speedup: {results['doc_processing_speedup']:.2f}x")
    print(f"   - Status: {' PASS' if results['doc_processing_speedup'] >= 3.0 else ' BELOW TARGET'}")

    # Overall assessment
    all_pass = (
        results['retrieval_improvement'] >= 30 and
        results['embedding_speedup'] >= 2.0 and
        results['doc_processing_speedup'] >= 3.0
    )

    print(f"\n{'='*70}")
    print(f"Overall Phase 1 Assessment: {' ALL TARGETS MET' if all_pass else ' SOME TARGETS MISSED'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
