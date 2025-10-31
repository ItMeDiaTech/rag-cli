"""Benchmark script for testing Phase 4 indexing throughput improvements.

This script tests the performance improvements from multi-process architecture:
1. Process-parallel embedding generation (target: 3-4x speedup)
2. Process-parallel document chunking (target: 5-7x speedup)
3. Full indexing pipeline (target: 50% throughput improvement)

USAGE:
    python test_indexing_throughput.py

REQUIREMENTS:
    - Sample documents in data/documents/ directory
    - At least 50+ documents for meaningful benchmarks
"""

import time
import os
import sys
from pathlib import Path
from typing import List, Tuple
import tempfile
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from core.document_processor import get_document_processor, DocumentChunk
from core.embeddings import (
    get_embedding_generator,
    get_embedding_pool,
    get_process_embedding_pool
)
from core.vector_store import get_vector_store


class IndexingBenchmark:
    """Benchmark suite for indexing throughput."""

    def __init__(self, document_dir: str = "data/documents"):
        """Initialize benchmark.

        Args:
            document_dir: Directory containing test documents
        """
        self.document_dir = Path(document_dir)
        self.processor = get_document_processor()
        self.embedding_generator = get_embedding_generator()
        self.thread_pool = None  # Lazy init
        self.process_pool = None  # Lazy init

        # Performance targets
        self.targets = {
            'embedding_speedup': 3.0,  # 3-4x speedup
            'chunking_speedup': 5.0,   # 5-7x speedup
            'indexing_improvement': 1.5  # 50% faster (1.5x)
        }

    def benchmark_document_chunking(self) -> dict:
        """Benchmark document chunking: sequential vs thread-parallel vs process-parallel.

        Returns:
            Dictionary with timing results
        """
        print("\n" + "=" * 70)
        print("BENCHMARK 1: Document Chunking")
        print("=" * 70)

        # Test 1: Sequential chunking
        print("\n[1/3] Sequential chunking...")
        start_time = time.time()
        documents_seq, chunks_seq = self.processor.process_and_chunk_directory(
            self.document_dir,
            recursive=True
        )
        time_sequential = time.time() - start_time
        print(f"  Time: {time_sequential:.2f}s")
        print(f"  Docs: {len(documents_seq)}, Chunks: {len(chunks_seq)}")

        if len(documents_seq) == 0:
            print("ERROR: No documents found. Add documents to data/documents/")
            return {'error': 'No documents found'}

        # Test 2: Thread-parallel chunking
        print("\n[2/3] Thread-parallel chunking...")
        start_time = time.time()
        documents_thread, chunks_thread = self.processor.process_and_chunk_directory_parallel(
            self.document_dir,
            recursive=True,
            max_workers=4
        )
        time_thread = time.time() - start_time
        print(f"  Time: {time_thread:.2f}s")
        print(f"  Docs: {len(documents_thread)}, Chunks: {len(chunks_thread)}")

        # Test 3: Process-parallel chunking
        print("\n[3/3] Process-parallel chunking...")
        start_time = time.time()
        documents_process, chunks_process = self.processor.process_and_chunk_directory_process_parallel(
            self.document_dir,
            recursive=True,
            max_workers=4
        )
        time_process = time.time() - start_time
        print(f"  Time: {time_process:.2f}s")
        print(f"  Docs: {len(documents_process)}, Chunks: {len(chunks_process)}")

        # Calculate speedups
        thread_speedup = time_sequential / time_thread if time_thread > 0 else 0
        process_speedup = time_sequential / time_process if time_process > 0 else 0

        print("\n--- Chunking Results ---")
        print(f"Sequential:       {time_sequential:.2f}s (baseline)")
        print(f"Thread-parallel:  {time_thread:.2f}s ({thread_speedup:.2f}x speedup)")
        print(f"Process-parallel: {time_process:.2f}s ({process_speedup:.2f}x speedup)")

        target_met = process_speedup >= self.targets['chunking_speedup']
        status = "PASS" if target_met else "FAIL"
        print(f"\nTarget: {self.targets['chunking_speedup']}x speedup")
        print(f"Result: {process_speedup:.2f}x speedup [{status}]")

        return {
            'sequential_time': time_sequential,
            'thread_time': time_thread,
            'process_time': time_process,
            'thread_speedup': thread_speedup,
            'process_speedup': process_speedup,
            'chunks': len(chunks_process),
            'target_met': target_met,
            'chunks_data': chunks_process  # For next benchmark
        }

    def benchmark_embedding_generation(self, chunks: List[DocumentChunk]) -> dict:
        """Benchmark embedding generation: sequential vs thread-pool vs process-pool.

        Args:
            chunks: Document chunks to encode

        Returns:
            Dictionary with timing results
        """
        print("\n" + "=" * 70)
        print("BENCHMARK 2: Embedding Generation")
        print("=" * 70)

        # Extract texts from chunks
        texts = [chunk.content for chunk in chunks[:500]]  # Limit to 500 for speed
        print(f"\nEncoding {len(texts)} chunks...")

        # Test 1: Sequential encoding
        print("\n[1/3] Sequential encoding...")
        start_time = time.time()
        embeddings_seq = self.embedding_generator.encode(texts, show_progress=False, use_cache=False)
        time_sequential = time.time() - start_time
        print(f"  Time: {time_sequential:.2f}s")
        print(f"  Embeddings shape: {embeddings_seq.shape}")

        # Test 2: Thread-pool encoding
        print("\n[2/3] Thread-pool parallel encoding...")
        if self.thread_pool is None:
            from core.embeddings import EmbeddingPool
            self.thread_pool = EmbeddingPool(self.embedding_generator, max_workers=4)

        start_time = time.time()
        embeddings_thread = self.thread_pool.encode_parallel(texts, show_progress=False)
        time_thread = time.time() - start_time
        print(f"  Time: {time_thread:.2f}s")
        print(f"  Embeddings shape: {embeddings_thread.shape}")

        # Test 3: Process-pool encoding
        print("\n[3/3] Process-pool parallel encoding...")
        if self.process_pool is None:
            self.process_pool = get_process_embedding_pool(max_workers=4)

        start_time = time.time()
        embeddings_process = self.process_pool.encode_parallel(texts, show_progress=False)
        time_process = time.time() - start_time
        print(f"  Time: {time_process:.2f}s")
        print(f"  Embeddings shape: {embeddings_process.shape}")

        # Calculate speedups
        thread_speedup = time_sequential / time_thread if time_thread > 0 else 0
        process_speedup = time_sequential / time_process if time_process > 0 else 0

        print("\n--- Embedding Results ---")
        print(f"Sequential:       {time_sequential:.2f}s (baseline)")
        print(f"Thread-pool:      {time_thread:.2f}s ({thread_speedup:.2f}x speedup)")
        print(f"Process-pool:     {time_process:.2f}s ({process_speedup:.2f}x speedup)")

        target_met = process_speedup >= self.targets['embedding_speedup']
        status = "PASS" if target_met else "FAIL"
        print(f"\nTarget: {self.targets['embedding_speedup']}x speedup")
        print(f"Result: {process_speedup:.2f}x speedup [{status}]")

        return {
            'sequential_time': time_sequential,
            'thread_time': time_thread,
            'process_time': time_process,
            'thread_speedup': thread_speedup,
            'process_speedup': process_speedup,
            'embeddings_count': len(texts),
            'target_met': target_met
        }

    def benchmark_full_indexing_pipeline(self) -> dict:
        """Benchmark full end-to-end indexing pipeline.

        Returns:
            Dictionary with timing results
        """
        print("\n" + "=" * 70)
        print("BENCHMARK 3: Full Indexing Pipeline")
        print("=" * 70)

        # Test 1: Traditional pipeline (sequential + thread-parallel)
        print("\n[1/2] Traditional pipeline (thread-parallel)...")
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "index_thread.faiss"
            meta_path = Path(tmpdir) / "metadata_thread.json"

            start_time = time.time()

            # Load and chunk documents (thread-parallel)
            docs, chunks = self.processor.process_and_chunk_directory_parallel(
                self.document_dir, recursive=True, max_workers=4
            )

            # Generate embeddings (thread-parallel)
            texts = [chunk.content for chunk in chunks]
            if self.thread_pool is None:
                from core.embeddings import EmbeddingPool
                self.thread_pool = EmbeddingPool(self.embedding_generator, max_workers=4)
            embeddings = self.thread_pool.encode_parallel(texts, show_progress=False)

            # Create and populate vector store
            store = get_vector_store(dimension=384, index_type="flat")
            store.clear()  # Clear existing data
            sources = [chunk.source for chunk in chunks]
            store.add(embeddings, texts, sources=sources)
            store.save(str(store_path), str(meta_path))

            time_thread = time.time() - start_time
            print(f"  Time: {time_thread:.2f}s")
            print(f"  Indexed: {store.index.ntotal} vectors")

        # Test 2: Optimized pipeline (process-parallel)
        print("\n[2/2] Optimized pipeline (process-parallel)...")
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "index_process.faiss"
            meta_path = Path(tmpdir) / "metadata_process.json"

            start_time = time.time()

            # Load and chunk documents (process-parallel)
            docs, chunks = self.processor.process_and_chunk_directory_process_parallel(
                self.document_dir, recursive=True, max_workers=4
            )

            # Generate embeddings (process-parallel)
            texts = [chunk.content for chunk in chunks]
            if self.process_pool is None:
                self.process_pool = get_process_embedding_pool(max_workers=4)
            embeddings = self.process_pool.encode_parallel(texts, show_progress=False)

            # Create and populate vector store
            store = get_vector_store(dimension=384, index_type="flat")
            store.clear()  # Clear existing data
            sources = [chunk.source for chunk in chunks]
            store.add(embeddings, texts, sources=sources)
            store.save(str(store_path), str(meta_path))

            time_process = time.time() - start_time
            print(f"  Time: {time_process:.2f}s")
            print(f"  Indexed: {store.index.ntotal} vectors")

        # Calculate improvement
        improvement = time_thread / time_process if time_process > 0 else 0
        time_saved = time_thread - time_process
        percent_faster = ((time_thread - time_process) / time_thread * 100) if time_thread > 0 else 0

        print("\n--- Full Pipeline Results ---")
        print(f"Thread-parallel:  {time_thread:.2f}s (baseline)")
        print(f"Process-parallel: {time_process:.2f}s ({improvement:.2f}x speedup)")
        print(f"Time saved:       {time_saved:.2f}s ({percent_faster:.0f}% faster)")

        target_met = improvement >= self.targets['indexing_improvement']
        status = "PASS" if target_met else "FAIL"
        print(f"\nTarget: {self.targets['indexing_improvement']}x improvement (50% faster)")
        print(f"Result: {improvement:.2f}x improvement [{status}]")

        return {
            'thread_time': time_thread,
            'process_time': time_process,
            'improvement': improvement,
            'time_saved': time_saved,
            'percent_faster': percent_faster,
            'vectors_indexed': len(chunks),
            'target_met': target_met
        }

    def cleanup(self):
        """Clean up resources."""
        if self.thread_pool:
            self.thread_pool.shutdown()
        if self.process_pool:
            self.process_pool.shutdown()


def print_summary(results: dict):
    """Print summary of all benchmark results.

    Args:
        results: Dictionary with all benchmark results
    """
    print("\n" + "=" * 70)
    print("PHASE 4 BENCHMARK SUMMARY")
    print("=" * 70)

    # Chunking
    print("\n1. Document Chunking:")
    if 'error' not in results['chunking']:
        print(f"   Process-parallel speedup: {results['chunking']['process_speedup']:.2f}x")
        print(f"   Target: {results['chunking'].get('target', 5.0)}x")
        print(f"   Status: {'PASS' if results['chunking']['target_met'] else 'FAIL'}")
    else:
        print(f"   ERROR: {results['chunking']['error']}")

    # Embeddings
    print("\n2. Embedding Generation:")
    print(f"   Process-pool speedup: {results['embeddings']['process_speedup']:.2f}x")
    print(f"   Target: 3.0x")
    print(f"   Status: {'PASS' if results['embeddings']['target_met'] else 'FAIL'}")

    # Full pipeline
    print("\n3. Full Indexing Pipeline:")
    print(f"   Improvement: {results['pipeline']['improvement']:.2f}x ({results['pipeline']['percent_faster']:.0f}% faster)")
    print(f"   Target: 1.5x (50% faster)")
    print(f"   Status: {'PASS' if results['pipeline']['target_met'] else 'FAIL'}")

    # Overall status
    all_passed = (
        results['chunking'].get('target_met', False) and
        results['embeddings']['target_met'] and
        results['pipeline']['target_met']
    )

    print("\n" + "=" * 70)
    print(f"OVERALL STATUS: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 70)

    if all_passed:
        print("\nPhase 4 multi-process architecture is performing as expected!")
        print("Ready to proceed to Phase 5.")
    else:
        print("\nSome performance targets not met. Consider:")
        print("- Increasing worker count")
        print("- Using larger document sets")
        print("- Checking CPU utilization")
        print("- Profiling bottlenecks")


def main():
    """Run all indexing throughput benchmarks."""
    print("=" * 70)
    print("RAG-CLI PHASE 4: INDEXING THROUGHPUT BENCHMARK")
    print("=" * 70)
    print("\nThis benchmark tests multi-process performance improvements:")
    print("- Process-parallel document chunking")
    print("- Process-parallel embedding generation")
    print("- Full end-to-end indexing pipeline")
    print("\nNOTE: Requires at least 50+ documents in data/documents/ for accurate results")
    print("=" * 70)

    # Initialize benchmark
    benchmark = IndexingBenchmark()

    try:
        # Run benchmarks
        results = {}

        # 1. Document chunking
        results['chunking'] = benchmark.benchmark_document_chunking()
        if 'error' in results['chunking']:
            print("\nSkipping remaining benchmarks due to missing documents.")
            return

        # 2. Embedding generation (using chunks from previous benchmark)
        results['embeddings'] = benchmark.benchmark_embedding_generation(
            results['chunking']['chunks_data']
        )

        # 3. Full pipeline
        results['pipeline'] = benchmark.benchmark_full_indexing_pipeline()

        # Print summary
        print_summary(results)

    finally:
        # Cleanup
        benchmark.cleanup()


if __name__ == "__main__":
    main()
