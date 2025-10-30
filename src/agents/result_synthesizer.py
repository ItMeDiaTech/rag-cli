"""Result Synthesizer Agent for RAG-CLI.

Merges and synthesizes results from multiple sub-query retrievals.

SYNTHESIS STRATEGIES:
1. Deduplication: Remove overlapping content based on similarity
2. Relevance Ranking: Re-rank merged results by relevance
3. Source Attribution: Track which sub-query each result came from
4. Coherent Ordering: Organize results in logical sequence

EXAMPLE:
Sub-query 1: "How to implement FastAPI" → 5 results
Sub-query 2: "FastAPI async database" → 5 results
Sub-query 3: "FastAPI CORS" → 5 results
→ Synthesized: 10 unique results (5 duplicates removed, reranked)
"""

import asyncio
from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import hashlib

from core.retrieval_pipeline import RetrievalResult
from agents.query_decomposer import SubQuery
from monitoring.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SynthesisResult:
    """Result of multi-query synthesis."""
    merged_results: List[RetrievalResult]
    total_input_results: int
    duplicates_removed: int
    sub_query_map: Dict[int, List[int]]  # SubQuery index → result indices
    confidence: float
    metadata: Dict[str, Any]


class ResultSynthesizer:
    """Synthesizes results from multiple sub-query retrievals."""

    def __init__(self):
        """Initialize result synthesizer."""
        self.similarity_threshold = 0.85  # For deduplication
        self.max_merged_results = 15  # Limit final result set

        logger.info(
            "Result synthesizer initialized",
            similarity_threshold=self.similarity_threshold,
            max_results=self.max_merged_results
        )

    async def synthesize(
        self,
        sub_queries: List[SubQuery],
        sub_query_results: List[List[RetrievalResult]],
        top_k: int = 10
    ) -> SynthesisResult:
        """Synthesize results from multiple sub-queries.

        Args:
            sub_queries: Original sub-queries
            sub_query_results: Results for each sub-query (parallel lists)
            top_k: Number of final results to return

        Returns:
            SynthesisResult with merged and deduplicated results
        """
        logger.info(
            f"Synthesizing results from {len(sub_queries)} sub-queries",
            total_results=sum(len(results) for results in sub_query_results)
        )

        # Step 1: Collect all results with source tracking
        all_results_with_source = []
        sub_query_map = defaultdict(list)

        for sq_idx, (sub_query, results) in enumerate(zip(sub_queries, sub_query_results)):
            for result in results:
                # Track which sub-query this result came from
                result_id = len(all_results_with_source)
                all_results_with_source.append((sq_idx, result))
                sub_query_map[sq_idx].append(result_id)

        logger.debug(
            "Collected results",
            total=len(all_results_with_source),
            by_query=[len(results) for results in sub_query_results]
        )

        # Step 2: Deduplicate based on content similarity
        unique_results, duplicates_removed = self._deduplicate_results(
            all_results_with_source
        )

        logger.info(
            "Deduplication complete",
            unique=len(unique_results),
            duplicates_removed=duplicates_removed
        )

        # Step 3: Re-rank merged results
        ranked_results = self._rerank_results(unique_results, sub_queries)

        logger.debug("Reranking complete", top_score=ranked_results[0].score if ranked_results else 0)

        # Step 4: Select top-k results
        final_results = ranked_results[:min(top_k, self.max_merged_results)]

        # Step 5: Update result positions
        for i, result in enumerate(final_results):
            result.rank_position = i + 1

        # Calculate overall confidence
        confidence = self._calculate_confidence(
            final_results,
            len(all_results_with_source),
            duplicates_removed
        )

        logger.info(
            "Synthesis complete",
            final_results=len(final_results),
            confidence=confidence
        )

        return SynthesisResult(
            merged_results=final_results,
            total_input_results=len(all_results_with_source),
            duplicates_removed=duplicates_removed,
            sub_query_map=dict(sub_query_map),
            confidence=confidence,
            metadata={
                'sub_query_count': len(sub_queries),
                'avg_results_per_query': len(all_results_with_source) / len(sub_queries) if sub_queries else 0,
                'deduplication_rate': duplicates_removed / len(all_results_with_source) if all_results_with_source else 0
            }
        )

    def _deduplicate_results(
        self,
        results_with_source: List[Tuple[int, RetrievalResult]]
    ) -> Tuple[List[RetrievalResult], int]:
        """Remove duplicate results based on content similarity.

        Uses text hashing for exact duplicates and content comparison for near-duplicates.

        Args:
            results_with_source: List of (sub_query_idx, result) tuples

        Returns:
            Tuple of (unique_results, num_duplicates_removed)
        """
        seen_hashes: Set[str] = set()
        seen_texts: List[str] = []
        unique_results: List[RetrievalResult] = []
        duplicates = 0

        for sq_idx, result in results_with_source:
            # Create hash of text content
            text_hash = hashlib.md5(result.text.encode()).hexdigest()

            # Check exact duplicate
            if text_hash in seen_hashes:
                duplicates += 1
                logger.debug(f"Exact duplicate detected: {result.chunk_id}")
                continue

            # Check near-duplicate (high similarity)
            is_near_duplicate = False
            for seen_text in seen_texts:
                if self._text_similarity(result.text, seen_text) >= self.similarity_threshold:
                    is_near_duplicate = True
                    duplicates += 1
                    logger.debug(f"Near-duplicate detected: {result.chunk_id}")
                    break

            if not is_near_duplicate:
                seen_hashes.add(text_hash)
                seen_texts.append(result.text)
                unique_results.append(result)

        return unique_results, duplicates

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (Jaccard).

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        # Simple word-based Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def _rerank_results(
        self,
        results: List[RetrievalResult],
        sub_queries: List[SubQuery]
    ) -> List[RetrievalResult]:
        """Re-rank merged results by relevance.

        Scoring factors:
        - Original retrieval score (0.6 weight)
        - Source diversity bonus (0.2 weight)
        - Retrieval method quality (0.2 weight)

        Args:
            results: Deduplicated results
            sub_queries: Original sub-queries

        Returns:
            Reranked results
        """
        # Calculate new scores
        scored_results = []

        source_counts = defaultdict(int)
        for result in results:
            source_counts[result.source] += 1

        for result in results:
            # Base score from retrieval
            score = result.score * 0.6

            # Diversity bonus (sources that appear less frequently get bonus)
            diversity_bonus = 1.0 / (source_counts[result.source] + 1)
            score += diversity_bonus * 0.2

            # Method quality bonus
            method_scores = {
                'hybrid': 1.0,
                'vector': 0.8,
                'keyword': 0.6,
                'online': 0.9
            }
            method_bonus = method_scores.get(result.retrieval_method, 0.5)
            score += method_bonus * 0.2

            scored_results.append((score, result))

        # Sort by new score
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Update scores in results
        reranked = []
        for new_score, result in scored_results:
            # Create a copy with updated score
            reranked.append(RetrievalResult(
                chunk_id=result.chunk_id,
                text=result.text,
                score=new_score,
                source=result.source,
                metadata={**result.metadata, 'original_score': result.score, 'reranked': True},
                retrieval_method=result.retrieval_method,
                rank_position=result.rank_position
            ))

        return reranked

    def _calculate_confidence(
        self,
        final_results: List[RetrievalResult],
        total_input: int,
        duplicates: int
    ) -> float:
        """Calculate overall synthesis confidence.

        Args:
            final_results: Final synthesized results
            total_input: Total input results
            duplicates: Number of duplicates removed

        Returns:
            Confidence score (0-1)
        """
        if not final_results:
            return 0.0

        # Factor 1: Average result scores
        avg_score = sum(r.score for r in final_results) / len(final_results)

        # Factor 2: Coverage (more unique results = higher confidence)
        coverage = len(final_results) / total_input if total_input > 0 else 0

        # Factor 3: Deduplication quality (finding duplicates is good)
        dedup_quality = min(duplicates / (total_input * 0.3), 1.0) if total_input > 0 else 0

        # Weighted combination
        confidence = (
            avg_score * 0.6 +
            coverage * 0.2 +
            dedup_quality * 0.2
        )

        return min(confidence, 1.0)

    def format_synthesis_summary(
        self,
        synthesis_result: SynthesisResult,
        sub_queries: List[SubQuery]
    ) -> str:
        """Format a human-readable synthesis summary.

        Args:
            synthesis_result: Synthesis result
            sub_queries: Original sub-queries

        Returns:
            Formatted summary string
        """
        lines = []
        lines.append("=== Query Decomposition Results ===")
        lines.append(f"Sub-queries executed: {len(sub_queries)}")
        lines.append("")

        for sq in sub_queries:
            result_count = len(synthesis_result.sub_query_map.get(sq.index, []))
            lines.append(f"  [{sq.index + 1}] {sq.text} → {result_count} results")

        lines.append("")
        lines.append(f"Total results collected: {synthesis_result.total_input_results}")
        lines.append(f"Duplicates removed: {synthesis_result.duplicates_removed}")
        lines.append(f"Unique results: {len(synthesis_result.merged_results)}")
        lines.append(f"Confidence: {synthesis_result.confidence:.0%}")
        lines.append("")
        lines.append("=== Top Synthesized Results ===")

        for i, result in enumerate(synthesis_result.merged_results[:5], 1):
            lines.append(f"{i}. [{result.source}] (score: {result.score:.2f})")
            lines.append(f"   {result.text[:150]}...")
            lines.append("")

        return "\n".join(lines)


# Singleton instance
_synthesizer: Optional[ResultSynthesizer] = None


def get_result_synthesizer() -> ResultSynthesizer:
    """Get or create the global result synthesizer instance.

    Returns:
        Result synthesizer instance
    """
    global _synthesizer

    if _synthesizer is None:
        _synthesizer = ResultSynthesizer()

    return _synthesizer


async def test_synthesizer():
    """Test result synthesizer functionality."""
    print("Testing Result Synthesizer...")
    print("=" * 70)

    from agents.query_decomposer import SubQuery

    synthesizer = get_result_synthesizer()

    # Create mock sub-queries
    sub_queries = [
        SubQuery(text="How to implement FastAPI", index=0, original_context="...", dependencies=[], priority=0, metadata={}),
        SubQuery(text="FastAPI async database", index=1, original_context="...", dependencies=[], priority=1, metadata={}),
        SubQuery(text="FastAPI CORS", index=2, original_context="...", dependencies=[], priority=2, metadata={})
    ]

    # Create mock results
    mock_results = [
        [  # Results for sub-query 0
            RetrievalResult("id1", "FastAPI is a modern web framework...", 0.9, "doc1.md", {}, "hybrid", 1),
            RetrievalResult("id2", "To implement FastAPI, install it first...", 0.85, "doc2.md", {}, "vector", 2),
            RetrievalResult("id3", "FastAPI provides automatic API docs...", 0.8, "doc3.md", {}, "hybrid", 3),
        ],
        [  # Results for sub-query 1
            RetrievalResult("id4", "Async database connections in FastAPI use SQLAlchemy...", 0.88, "doc4.md", {}, "hybrid", 1),
            RetrievalResult("id1", "FastAPI is a modern web framework...", 0.87, "doc1.md", {}, "vector", 2),  # Duplicate
            RetrievalResult("id5", "Database migrations with Alembic...", 0.82, "doc5.md", {}, "keyword", 3),
        ],
        [  # Results for sub-query 2
            RetrievalResult("id6", "CORS middleware in FastAPI is configured...", 0.91, "doc6.md", {}, "hybrid", 1),
            RetrievalResult("id7", "Enable CORS with CORSMiddleware...", 0.86, "doc7.md", {}, "vector", 2),
            RetrievalResult("id4", "Async database connections in FastAPI...", 0.84, "doc4.md", {}, "hybrid", 3),  # Duplicate
        ]
    ]

    # Synthesize
    result = await synthesizer.synthesize(sub_queries, mock_results, top_k=10)

    # Print summary
    summary = synthesizer.format_synthesis_summary(result, sub_queries)
    print(summary)

    print("\n" + "=" * 70)
    print("Result synthesizer test complete!")


if __name__ == "__main__":
    asyncio.run(test_synthesizer())
