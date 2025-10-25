"""Hybrid retrieval pipeline for RAG-CLI.

This module implements a two-stage retrieval system combining vector search
with keyword-based BM25 search and cross-encoder reranking for optimal accuracy.
"""

import time
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict

# BM25 for keyword search
from rank_bm25 import BM25Okapi

# Cross-encoder for reranking
from sentence_transformers import CrossEncoder

from src.core.config import get_config
from src.core.embeddings import get_embedding_generator
from src.core.vector_store import get_vector_store, VectorMetadata
from src.core.document_processor import DocumentChunk, get_document_processor
from src.monitoring.logger import get_logger, get_metrics_logger, log_execution_time


logger = get_logger(__name__)
metrics = get_metrics_logger()


@dataclass
class RetrievalResult:
    """Result from retrieval pipeline."""
    chunk_id: str
    text: str
    score: float
    source: str
    metadata: Dict[str, Any]
    retrieval_method: str  # 'vector', 'keyword', or 'hybrid'
    rank_position: int


class RetrievalCache:
    """Cache for retrieval results."""

    def __init__(self, ttl_seconds: int = 3600):
        """Initialize retrieval cache.

        Args:
            ttl_seconds: Time to live for cache entries
        """
        self.cache = {}
        self.timestamps = {}
        self.ttl = ttl_seconds

    def get(self, query: str, top_k: int) -> Optional[List[RetrievalResult]]:
        """Get cached results for query.

        Args:
            query: Query string
            top_k: Number of results requested

        Returns:
            Cached results or None
        """
        cache_key = f"{query}:{top_k}"

        if cache_key in self.cache:
            # Check if expired
            if time.time() - self.timestamps[cache_key] < self.ttl:
                logger.debug("Retrieval cache hit", query_length=len(query))
                metrics.record_success("retrieval_cache_hit")
                return self.cache[cache_key]
            else:
                # Expired, remove from cache
                del self.cache[cache_key]
                del self.timestamps[cache_key]

        return None

    def put(self, query: str, top_k: int, results: List[RetrievalResult]):
        """Store results in cache.

        Args:
            query: Query string
            top_k: Number of results
            results: Retrieval results
        """
        cache_key = f"{query}:{top_k}"
        self.cache[cache_key] = results
        self.timestamps[cache_key] = time.time()

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.timestamps.clear()


class HybridRetriever:
    """Hybrid retrieval system combining vector and keyword search."""

    def __init__(self):
        """Initialize hybrid retriever."""
        config = get_config()

        # Weights for combining scores
        self.vector_weight = config.retrieval.vector_weight
        self.keyword_weight = config.retrieval.keyword_weight

        # Retrieval settings
        self.initial_candidates = config.retrieval.initial_candidates
        self.final_results = config.retrieval.final_results
        self.use_reranker = config.retrieval.use_reranker
        self.reranker_model_name = config.retrieval.reranker_model
        self.min_score_threshold = config.retrieval.min_score_threshold

        # Initialize components
        self.embedding_generator = get_embedding_generator()
        self.vector_store = get_vector_store()
        self.document_processor = get_document_processor()

        # BM25 index (will be built when documents are added)
        self.bm25_index = None
        self.bm25_documents = []
        self.bm25_doc_ids = []

        # Cross-encoder for reranking
        if self.use_reranker:
            logger.info(f"Loading cross-encoder model: {self.reranker_model_name}")
            self.cross_encoder = CrossEncoder(self.reranker_model_name)
        else:
            self.cross_encoder = None

        # Cache
        cache_enabled = config.retrieval.cache_enabled
        cache_ttl = config.retrieval.cache_ttl_seconds
        self.cache = RetrievalCache(cache_ttl) if cache_enabled else None

        logger.info(
            "Hybrid retriever initialized",
            vector_weight=self.vector_weight,
            keyword_weight=self.keyword_weight,
            use_reranker=self.use_reranker
        )

    def build_bm25_index(self, documents: List[str], doc_ids: List[str]):
        """Build BM25 index from documents.

        Args:
            documents: List of document texts
            doc_ids: List of document IDs
        """
        logger.info(f"Building BM25 index", documents=len(documents))

        # Tokenize documents for BM25
        tokenized_docs = [doc.lower().split() for doc in documents]

        # Create BM25 index
        self.bm25_index = BM25Okapi(tokenized_docs)
        self.bm25_documents = documents
        self.bm25_doc_ids = doc_ids

        logger.info("BM25 index built")
        metrics.record_count("bm25_documents", len(documents))

    @log_execution_time
    def vector_search(
        self,
        query_embedding: np.ndarray,
        top_k: int
    ) -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """Perform vector similarity search.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of (id, text, score, metadata) tuples
        """
        start_time = time.time()

        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k)

        # Format results
        formatted_results = []
        for metadata, score in results:
            formatted_results.append((
                metadata.id,
                metadata.text,
                score,
                metadata.metadata
            ))

        elapsed = time.time() - start_time
        logger.debug(f"Vector search completed", results=len(results), elapsed=elapsed)
        metrics.record_latency("vector_search", elapsed * 1000)

        return formatted_results

    @log_execution_time
    def keyword_search(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """Perform BM25 keyword search.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of (id, text, score, metadata) tuples
        """
        if self.bm25_index is None:
            logger.warning("BM25 index not built, returning empty results")
            return []

        start_time = time.time()

        # Tokenize query
        query_tokens = query.lower().split()

        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]

        # Format results
        results = []
        for idx in top_indices:
            if idx < len(self.bm25_documents):
                # Normalize BM25 score to 0-1 range
                normalized_score = scores[idx] / (scores[idx] + 1)

                results.append((
                    self.bm25_doc_ids[idx],
                    self.bm25_documents[idx],
                    normalized_score,
                    {}  # Metadata would be retrieved from vector store
                ))

        elapsed = time.time() - start_time
        logger.debug(f"Keyword search completed", results=len(results), elapsed=elapsed)
        metrics.record_latency("keyword_search", elapsed * 1000)

        return results

    def reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[str, str, float, Dict[str, Any]]],
        keyword_results: List[Tuple[str, str, float, Dict[str, Any]]],
        k: int = 60
    ) -> List[Tuple[str, str, float, Dict[str, Any], str]]:
        """Merge results using Reciprocal Rank Fusion.

        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            k: RRF constant (typically 60)

        Returns:
            Merged results with combined scores and retrieval method
        """
        # Calculate RRF scores
        rrf_scores = defaultdict(float)
        text_map = {}
        metadata_map = {}
        method_map = defaultdict(list)

        # Process vector results
        for rank, (doc_id, text, score, metadata) in enumerate(vector_results):
            rrf_scores[doc_id] += self.vector_weight * (1.0 / (k + rank + 1))
            text_map[doc_id] = text
            metadata_map[doc_id] = metadata
            method_map[doc_id].append("vector")

        # Process keyword results
        for rank, (doc_id, text, score, metadata) in enumerate(keyword_results):
            rrf_scores[doc_id] += self.keyword_weight * (1.0 / (k + rank + 1))
            if doc_id not in text_map:
                text_map[doc_id] = text
                metadata_map[doc_id] = metadata
            method_map[doc_id].append("keyword")

        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Format results
        merged_results = []
        for doc_id, rrf_score in sorted_results:
            retrieval_method = "hybrid" if len(method_map[doc_id]) > 1 else method_map[doc_id][0]
            merged_results.append((
                doc_id,
                text_map[doc_id],
                rrf_score,
                metadata_map[doc_id],
                retrieval_method
            ))

        logger.debug(f"RRF fusion completed", input_count=len(vector_results) + len(keyword_results), output_count=len(merged_results))

        return merged_results

    @log_execution_time
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str, float, Dict[str, Any], str]],
        top_k: int
    ) -> List[RetrievalResult]:
        """Rerank candidates using cross-encoder.

        Args:
            query: Query string
            candidates: Candidate results
            top_k: Number of results to return after reranking

        Returns:
            Reranked results
        """
        if not self.cross_encoder:
            # No reranking, just convert to RetrievalResult
            results = []
            for rank, (doc_id, text, score, metadata, method) in enumerate(candidates[:top_k]):
                results.append(RetrievalResult(
                    chunk_id=doc_id,
                    text=text,
                    score=score,
                    source=metadata.get("source", "unknown"),
                    metadata=metadata,
                    retrieval_method=method,
                    rank_position=rank + 1
                ))
            return results

        start_time = time.time()

        # Prepare query-document pairs
        pairs = [[query, doc[1]] for doc in candidates]

        # Get cross-encoder scores
        ce_scores = self.cross_encoder.predict(pairs)

        # Combine with original scores (weighted average)
        combined_scores = []
        for i, (doc_id, text, orig_score, metadata, method) in enumerate(candidates):
            # Weight: 70% cross-encoder, 30% original
            combined_score = 0.7 * ce_scores[i] + 0.3 * orig_score
            combined_scores.append((
                doc_id, text, combined_score, metadata, method, ce_scores[i]
            ))

        # Sort by combined score
        combined_scores.sort(key=lambda x: x[2], reverse=True)

        # Create RetrievalResult objects
        results = []
        for rank, (doc_id, text, score, metadata, method, ce_score) in enumerate(combined_scores[:top_k]):
            if score >= self.min_score_threshold:
                result = RetrievalResult(
                    chunk_id=doc_id,
                    text=text,
                    score=score,
                    source=metadata.get("source", "unknown"),
                    metadata={**metadata, "cross_encoder_score": ce_score},
                    retrieval_method=method,
                    rank_position=rank + 1
                )
                results.append(result)

        elapsed = time.time() - start_time
        logger.info(f"Reranking completed", candidates=len(candidates), results=len(results), elapsed=elapsed)
        metrics.record_latency("reranking", elapsed * 1000)

        return results

    @log_execution_time
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_cache: bool = True
    ) -> List[RetrievalResult]:
        """Perform hybrid retrieval with optional reranking.

        Args:
            query: Query string
            top_k: Number of final results (defaults to config)
            use_cache: Whether to use cache

        Returns:
            Retrieved and ranked results
        """
        # Use configured value if not specified
        if top_k is None:
            top_k = self.final_results

        # Check cache
        if use_cache and self.cache:
            cached_results = self.cache.get(query, top_k)
            if cached_results:
                return cached_results

        logger.info(f"Retrieving for query", query_length=len(query), top_k=top_k)
        start_time = time.time()

        # Generate query embedding
        query_embedding = self.embedding_generator.encode_query(query)

        # Perform vector search
        vector_results = self.vector_search(query_embedding, self.initial_candidates)

        # Perform keyword search
        keyword_results = self.keyword_search(query, self.initial_candidates)

        # Merge results with RRF
        merged_results = self.reciprocal_rank_fusion(vector_results, keyword_results)

        # Rerank if enabled
        if self.use_reranker and merged_results:
            final_results = self.rerank(query, merged_results, top_k)
        else:
            # Convert to RetrievalResult without reranking
            final_results = []
            for rank, (doc_id, text, score, metadata, method) in enumerate(merged_results[:top_k]):
                if score >= self.min_score_threshold:
                    final_results.append(RetrievalResult(
                        chunk_id=doc_id,
                        text=text,
                        score=score,
                        source=metadata.get("source", "unknown"),
                        metadata=metadata,
                        retrieval_method=method,
                        rank_position=rank + 1
                    ))

        # Cache results
        if use_cache and self.cache and final_results:
            self.cache.put(query, top_k, final_results)

        # Record metrics
        elapsed = time.time() - start_time
        logger.info(
            f"Retrieval completed",
            query_length=len(query),
            results=len(final_results),
            elapsed_seconds=elapsed
        )
        metrics.record_latency("total_retrieval", elapsed * 1000)
        metrics.record_gauge("retrieval_results", len(final_results))

        return final_results

    def index_documents(self, chunks: List[DocumentChunk]):
        """Index document chunks for retrieval.

        Args:
            chunks: List of document chunks to index
        """
        logger.info(f"Indexing documents for retrieval", chunks=len(chunks))

        # Extract texts and metadata
        texts = [chunk.content for chunk in chunks]
        doc_ids = [chunk.chunk_id for chunk in chunks]

        # Build BM25 index
        self.build_bm25_index(texts, doc_ids)

        logger.info("Document indexing completed")

    def clear_cache(self):
        """Clear the retrieval cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Retrieval cache cleared")


# Singleton instance
_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    """Get or create the global retriever.

    Returns:
        Hybrid retriever instance
    """
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


if __name__ == "__main__":
    # Test retrieval pipeline
    print("Testing Retrieval Pipeline...")

    # Initialize components
    from src.core.embeddings import get_embedding_generator
    from src.core.vector_store import get_vector_store
    from src.core.document_processor import get_document_processor

    generator = get_embedding_generator()
    store = get_vector_store()
    processor = get_document_processor()
    retriever = get_retriever()

    # Create test documents
    test_docs = [
        "RAG systems combine retrieval and generation for better responses.",
        "Vector search enables semantic similarity matching in documents.",
        "Claude is an AI assistant that can generate human-like text.",
        "FAISS is a library for efficient similarity search.",
        "BM25 is a probabilistic ranking function for keyword search.",
        "Cross-encoder models can rerank documents for better accuracy.",
        "The hybrid approach combines vector and keyword search methods.",
        "Embeddings capture semantic meaning in numerical form."
    ]

    # Clear vector store
    store.clear()

    # Create chunks
    chunks = []
    for i, text in enumerate(test_docs):
        chunk = DocumentChunk(
            content=text,
            metadata={"source": f"doc_{i}.txt"},
            chunk_index=0,
            total_chunks=1,
            char_count=len(text),
            token_count=len(text) // 4,
            source=f"doc_{i}.txt",
            doc_id=f"doc_{i}",
            chunk_id=f"doc_{i}_chunk_0"
        )
        chunks.append(chunk)

    # Generate embeddings and add to vector store
    print("\nIndexing documents...")
    embeddings = generator.encode_documents([c.content for c in chunks], show_progress=False)
    ids = store.add(
        embeddings,
        [c.content for c in chunks],
        sources=[c.source for c in chunks]
    )

    # Build BM25 index
    retriever.index_documents(chunks)

    # Test retrieval
    queries = [
        "How does RAG work?",
        "What is vector search?",
        "Tell me about Claude",
        "keyword matching algorithms"
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        results = retriever.retrieve(query, top_k=3)

        for result in results:
            print(f"  [{result.rank_position}] Score: {result.score:.4f} | Method: {result.retrieval_method}")
            print(f"      {result.text[:80]}...")

    print("\nRetrieval pipeline tests completed!")