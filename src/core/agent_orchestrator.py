"""Agent Orchestrator for RAG-CLI.

Coordinates between RAG retrieval and MAF agents for enhanced query processing.
Implements intelligent routing based on query intent and parallel execution.

ARCHITECTURE:
┌──────────────────────┐
│  Agent Orchestrator  │
└──────┬───────────────┘
       │
       ├─────► Intent Classification
       │
       ├─────► Route Decision
       │       ├─ Simple query → RAG only
       │       ├─ Error query → RAG || MAF Debugger
       │       └─ Complex query → Query Decomposer
       │
       ├─────► Parallel Execution
       │       ├─ RAG retrieval (async)
       │       └─ MAF agent (async)
       │
       └─────► Result Synthesis
               └─ Hybrid RAG + MAF response
"""

import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.core.retrieval_pipeline import get_retriever
from src.core.query_classifier import get_query_classifier, QueryIntent, QueryClassification, TechnicalDepth
from src.integrations.maf_connector import get_maf_connector, MAFResult
from src.agents.query_decomposer import get_query_decomposer, DecompositionResult
from src.agents.result_synthesizer import get_result_synthesizer, SynthesisResult
from src.monitoring.logger import get_logger, get_metrics_logger
from src.monitoring.output_formatter import OutputFormatter, format_rag_pipeline, format_maf_pipeline, format_hybrid_pipeline

logger = get_logger(__name__)
metrics = get_metrics_logger()


class RoutingStrategy(Enum):
    """Query routing strategies."""
    RAG_ONLY = "rag_only"  # Simple queries
    MAF_ONLY = "maf_only"  # Pure code/debugging tasks
    PARALLEL_RAG_MAF = "parallel"  # Error queries, complex analysis
    DECOMPOSED = "decomposed"  # Complex multi-part queries


@dataclass
class OrchestrationResult:
    """Result from agent orchestration."""
    content: str
    sources: List[Dict[str, Any]]
    confidence: float
    strategy_used: RoutingStrategy
    rag_results: Optional[List] = None
    maf_result: Optional[MAFResult] = None
    decomposition_result: Optional[DecompositionResult] = None
    synthesis_result: Optional[SynthesisResult] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None


class AgentOrchestrator:
    """Orchestrates queries between RAG and MAF agents with intent-based routing."""

    def __init__(self):
        """Initialize agent orchestrator."""
        self.retriever = get_retriever()
        self.classifier = get_query_classifier()
        self.maf_connector = get_maf_connector()
        self.query_decomposer = get_query_decomposer()
        self.result_synthesizer = get_result_synthesizer()

        # Configuration
        self.enable_maf = self.maf_connector.is_available()
        self.parallel_threshold_confidence = 0.7
        self.maf_timeout = 30.0
        self.decomposition_complexity_threshold = 0.6  # Trigger decomposition above this

        logger.info(
            "Agent orchestrator initialized",
            maf_enabled=self.enable_maf,
            decomposition_enabled=True,
            maf_agents=self.maf_connector.get_available_agents() if self.enable_maf else []
        )

    async def orchestrate(
        self,
        query: str,
        top_k: int = 5,
        use_cache: bool = True
    ) -> OrchestrationResult:
        """Orchestrate query processing with intelligent routing.

        ROUTING LOGIC:
        1. TROUBLESHOOTING intent → Parallel RAG + MAF Debugger
        2. Complex query → Query decomposition (future Phase 3)
        3. Simple query → RAG only
        4. MAF unavailable → RAG fallback

        Args:
            query: User query
            top_k: Number of results to return
            use_cache: Whether to use caching

        Returns:
            OrchestrationResult with synthesized response
        """
        start_time = time.time()

        logger.info(f"Orchestrating query", query_length=len(query))

        # Step 1: Classify query
        classification = self.classifier.classify(query)

        logger.info(
            f"Query classified",
            intent=classification.primary_intent.value,
            confidence=classification.confidence
        )

        # Step 2: Determine routing strategy
        strategy = self._determine_strategy(classification)

        logger.info(f"Routing strategy selected", strategy=strategy.value)

        # Emit reasoning event
        from src.monitoring.tcp_server import metrics_collector
        metrics_collector.record_reasoning_event(
            reasoning=f"Agent orchestration: Classified as {classification.primary_intent.value} "
                     f"with {classification.confidence:.0%} confidence. "
                     f"Strategy: {strategy.value}. "
                     f"MAF {'enabled' if self.enable_maf else 'disabled'}.",
            component="agent_orchestrator",
            context={
                'intent': classification.primary_intent.value,
                'confidence': classification.confidence,
                'strategy': strategy.value,
                'maf_available': self.enable_maf
            }
        )

        # Step 3: Execute based on strategy
        if strategy == RoutingStrategy.RAG_ONLY:
            result = await self._execute_rag_only(query, top_k, use_cache, classification)

        elif strategy == RoutingStrategy.PARALLEL_RAG_MAF:
            result = await self._execute_parallel_rag_maf(query, top_k, use_cache, classification)

        elif strategy == RoutingStrategy.DECOMPOSED:
            result = await self._execute_decomposed(query, top_k, use_cache, classification)

        else:
            # Fallback to RAG
            result = await self._execute_rag_only(query, top_k, use_cache, classification)

        # Record execution time
        result.execution_time = time.time() - start_time

        logger.info(
            f"Orchestration complete",
            strategy=strategy.value,
            confidence=result.confidence,
            elapsed=result.execution_time
        )
        metrics.record_latency("orchestration_total", result.execution_time * 1000)

        return result

    def _determine_strategy(self, classification: QueryClassification) -> RoutingStrategy:
        """Determine optimal routing strategy based on classification.

        Args:
            classification: Query classification

        Returns:
            Routing strategy to use
        """
        intent = classification.primary_intent

        # TROUBLESHOOTING queries benefit from MAF Debugger
        if intent == QueryIntent.TROUBLESHOOTING and self.enable_maf:
            if classification.confidence >= self.parallel_threshold_confidence:
                logger.debug("High-confidence troubleshooting query → Parallel RAG + MAF")
                return RoutingStrategy.PARALLEL_RAG_MAF
            else:
                logger.debug("Low-confidence troubleshooting → RAG only")
                return RoutingStrategy.RAG_ONLY

        # Complex queries use decomposition (Phase 3 - IMPLEMENTED)
        # Detect complexity based on multiple intents or advanced technical depth
        is_complex = (
            len(classification.all_intents) > 2 or  # Multiple intents
            classification.technical_depth == TechnicalDepth.ADVANCED or  # Advanced topic
            len(classification.entities) > 3  # Many technical entities
        )

        if is_complex and self.enable_maf:
            logger.debug("Complex query detected → Query decomposition")
            return RoutingStrategy.DECOMPOSED

        # Default to RAG only
        return RoutingStrategy.RAG_ONLY

    async def _execute_rag_only(
        self,
        query: str,
        top_k: int,
        use_cache: bool,
        classification: QueryClassification
    ) -> OrchestrationResult:
        """Execute RAG retrieval only.

        Args:
            query: User query
            top_k: Number of results
            use_cache: Use caching
            classification: Query classification

        Returns:
            OrchestrationResult with RAG results
        """
        logger.debug("Executing RAG-only strategy")

        # Execute async retrieval
        rag_results = await self.retriever.retrieve_async(
            query,
            top_k=top_k,
            use_cache=use_cache,
            classification=classification
        )

        # Build response
        if rag_results:
            content = self._format_rag_response(rag_results)
            sources = [
                {
                    'text': r.text[:200],
                    'source': r.source,
                    'score': r.score,
                    'method': r.retrieval_method
                }
                for r in rag_results[:top_k]
            ]
            confidence = sum(r.score for r in rag_results) / len(rag_results) if rag_results else 0.0
        else:
            content = "No relevant results found."
            sources = []
            confidence = 0.0

        return OrchestrationResult(
            content=content,
            sources=sources,
            confidence=confidence,
            strategy_used=RoutingStrategy.RAG_ONLY,
            rag_results=rag_results,
            maf_result=None,
            metadata={'rag_result_count': len(rag_results)}
        )

    async def _execute_parallel_rag_maf(
        self,
        query: str,
        top_k: int,
        use_cache: bool,
        classification: QueryClassification
    ) -> OrchestrationResult:
        """Execute RAG and MAF in parallel for enhanced results.

        PARALLEL EXECUTION:
        ┌─────────────┐
        │    Query    │
        └──────┬──────┘
               │
        ┌──────┴───────┐
        │   asyncio    │
        │   .gather()  │
        └──────┬───────┘
               │
        ┌──────┴──────┬──────────────┐
        │             │              │
        ▼             ▼              ▼
    RAG Retrieval  MAF Debugger  (timeout)
    (2s timeout)   (30s timeout)
        │             │              │
        └──────┬──────┴──────────────┘
               │
               ▼
       Result Synthesis
               │
               ▼
         Hybrid Response

        Args:
            query: User query
            top_k: Number of results
            use_cache: Use caching
            classification: Query classification

        Returns:
            OrchestrationResult with synthesized RAG + MAF results
        """
        logger.debug("Executing parallel RAG + MAF strategy")

        # Emit activity event
        from src.monitoring.tcp_server import metrics_collector
        metrics_collector.record_activity_event(
            activity="parallel_rag_maf_started",
            component="agent_orchestrator",
            metadata={
                'query': query[:100],
                'intent': classification.primary_intent.value
            }
        )

        # Execute in parallel with individual timeouts
        rag_task = self.retriever.retrieve_async(
            query,
            top_k=top_k,
            use_cache=use_cache,
            classification=classification,
            vector_timeout=2.0,
            keyword_timeout=2.0,
            rerank_timeout=3.0
        )

        # For troubleshooting, route to MAF Debugger
        maf_task = self.maf_connector.execute_debugger(
            error_message=query,
            context="User query analysis"
        )

        # Wait for both with timeout protection
        try:
            rag_results, maf_result = await asyncio.gather(
                rag_task,
                maf_task,
                return_exceptions=True
            )

            # Handle exceptions
            if isinstance(rag_results, Exception):
                logger.error(f"RAG retrieval failed: {rag_results}")
                rag_results = []

            if isinstance(maf_result, Exception):
                logger.error(f"MAF execution failed: {maf_result}")
                maf_result = None

        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            rag_results = []
            maf_result = None

        # Synthesize results
        return self._synthesize_hybrid_results(
            query,
            rag_results if not isinstance(rag_results, Exception) else [],
            maf_result if not isinstance(maf_result, Exception) else None,
            top_k
        )

    def _synthesize_hybrid_results(
        self,
        query: str,
        rag_results: List,
        maf_result: Optional[MAFResult],
        top_k: int
    ) -> OrchestrationResult:
        """Synthesize hybrid response from RAG + MAF results.

        SYNTHESIS STRATEGY:
        - If both available: Combine with MAF insights first, RAG context second
        - If MAF failed: Use RAG only
        - If RAG failed: Use MAF only (if available)
        - Confidence: Weighted average based on availability

        Args:
            query: Original query
            rag_results: Results from RAG retrieval
            maf_result: Result from MAF agent
            top_k: Number of top results to include

        Returns:
            Synthesized OrchestrationResult
        """
        logger.debug("Synthesizing hybrid results", rag_count=len(rag_results), maf_available=maf_result is not None)

        content_parts = []
        sources = []
        confidence_scores = []

        # Add MAF analysis if available
        if maf_result and maf_result.status == 'completed':
            content_parts.append(f"[MAF {maf_result.agent_name.title()} Analysis]")
            content_parts.append(maf_result.content[:500])  # Limit length
            confidence_scores.append(maf_result.confidence)
            sources.append({
                'text': f"MAF {maf_result.agent_name} analysis",
                'source': 'maf_agent',
                'score': maf_result.confidence,
                'method': 'maf_agent'
            })
            logger.debug(f"MAF result included", agent=maf_result.agent_name, confidence=maf_result.confidence)

        # Add RAG results
        if rag_results:
            content_parts.append("\n[RAG Context]")
            rag_content = self._format_rag_response(rag_results[:top_k])
            content_parts.append(rag_content)

            rag_confidence = sum(r.score for r in rag_results) / len(rag_results)
            confidence_scores.append(rag_confidence)

            sources.extend([
                {
                    'text': r.text[:200],
                    'source': r.source,
                    'score': r.score,
                    'method': r.retrieval_method
                }
                for r in rag_results[:top_k]
            ])
            logger.debug(f"RAG results included", count=len(rag_results), avg_confidence=rag_confidence)

        # Calculate overall confidence (weighted average)
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

        # Build final content
        if not content_parts:
            content = "No results available from RAG or MAF agents."
            overall_confidence = 0.0
        else:
            content = "\n\n".join(content_parts)

        # Emit reasoning event
        from src.monitoring.tcp_server import metrics_collector
        metrics_collector.record_reasoning_event(
            reasoning=f"Hybrid synthesis: Combined MAF {'analysis' if maf_result else 'unavailable'} "
                     f"with {len(rag_results)} RAG results. "
                     f"Overall confidence: {overall_confidence:.0%}. "
                     f"Sources: {len(sources)} total.",
            component="agent_orchestrator",
            context={
                'maf_used': maf_result is not None,
                'rag_count': len(rag_results),
                'source_count': len(sources),
                'confidence': overall_confidence
            }
        )

        return OrchestrationResult(
            content=content,
            sources=sources,
            confidence=overall_confidence,
            strategy_used=RoutingStrategy.PARALLEL_RAG_MAF,
            rag_results=rag_results,
            maf_result=maf_result,
            metadata={
                'rag_count': len(rag_results),
                'maf_available': maf_result is not None,
                'maf_status': maf_result.status if maf_result else None
            }
        )

    async def _execute_decomposed(
        self,
        query: str,
        top_k: int,
        use_cache: bool,
        classification: QueryClassification
    ) -> OrchestrationResult:
        """Execute query decomposition strategy with parallel sub-query retrieval.

        DECOMPOSED STRATEGY:
        1. Decompose query into sub-queries
        2. Execute all sub-queries in parallel (async retrieval)
        3. Deduplicate and synthesize results
        4. Format unified response

        Args:
            query: User query
            top_k: Number of results
            use_cache: Use caching
            classification: Query classification

        Returns:
            OrchestrationResult with synthesized results
        """
        logger.debug("Executing decomposed strategy")

        # Emit activity event
        from src.monitoring.tcp_server import metrics_collector
        metrics_collector.record_activity_event(
            activity="query_decomposition_started",
            component="agent_orchestrator",
            metadata={'query': query[:100]}
        )

        # Step 1: Decompose query
        decomposition = await self.query_decomposer.decompose(query)

        if not decomposition.is_complex or len(decomposition.sub_queries) == 1:
            # Not actually complex, fallback to RAG only
            logger.info("Query deemed not complex enough for decomposition, using RAG only")
            return await self._execute_rag_only(query, top_k, use_cache, classification)

        logger.info(
            f"Query decomposed",
            sub_queries=len(decomposition.sub_queries),
            strategy=decomposition.strategy_used.value,
            confidence=decomposition.confidence
        )

        # Emit reasoning event
        metrics_collector.record_reasoning_event(
            reasoning=f"Complex query decomposed into {len(decomposition.sub_queries)} sub-queries "
                     f"using {decomposition.strategy_used.value} strategy. "
                     f"Sub-queries will execute in parallel for 2-4x faster results. "
                     f"Confidence: {decomposition.confidence:.0%}.",
            component="agent_orchestrator",
            context={
                'num_sub_queries': len(decomposition.sub_queries),
                'strategy': decomposition.strategy_used.value,
                'confidence': decomposition.confidence
            }
        )

        # Step 2: Execute all sub-queries in parallel
        logger.info(f"Executing {len(decomposition.sub_queries)} sub-queries in parallel")

        # Create retrieval tasks for each sub-query
        retrieval_tasks = []
        for sub_query in decomposition.sub_queries:
            task = self.retriever.retrieve_async(
                sub_query.text,
                top_k=min(top_k, 5),  # Limit per sub-query to avoid overload
                use_cache=use_cache,
                classification=classification
            )
            retrieval_tasks.append(task)

        # Execute all in parallel
        try:
            sub_query_results = await asyncio.gather(
                *retrieval_tasks,
                return_exceptions=True
            )

            # Handle exceptions
            valid_results = []
            for i, result in enumerate(sub_query_results):
                if isinstance(result, Exception):
                    logger.error(f"Sub-query {i} failed: {result}")
                    valid_results.append([])  # Empty results for failed query
                else:
                    valid_results.append(result)

            sub_query_results = valid_results

        except Exception as e:
            logger.error(f"Parallel sub-query execution failed: {e}")
            # Fallback to RAG only
            return await self._execute_rag_only(query, top_k, use_cache, classification)

        # Log sub-query results
        for i, (sq, results) in enumerate(zip(decomposition.sub_queries, sub_query_results)):
            logger.debug(f"Sub-query {i}: '{sq.text}' → {len(results)} results")

        # Step 3: Synthesize results
        logger.info("Synthesizing results from sub-queries")
        synthesis = await self.result_synthesizer.synthesize(
            decomposition.sub_queries,
            sub_query_results,
            top_k=top_k
        )

        logger.info(
            f"Synthesis complete",
            merged_results=len(synthesis.merged_results),
            duplicates_removed=synthesis.duplicates_removed,
            confidence=synthesis.confidence
        )

        # Emit reasoning for synthesis
        metrics_collector.record_reasoning_event(
            reasoning=f"Synthesized {synthesis.total_input_results} results from {len(decomposition.sub_queries)} sub-queries. "
                     f"Removed {synthesis.duplicates_removed} duplicates. "
                     f"Final set: {len(synthesis.merged_results)} unique results. "
                     f"Confidence: {synthesis.confidence:.0%}.",
            component="agent_orchestrator",
            context={
                'total_input': synthesis.total_input_results,
                'duplicates': synthesis.duplicates_removed,
                'final_count': len(synthesis.merged_results),
                'confidence': synthesis.confidence
            }
        )

        # Step 4: Format response
        content_parts = []
        content_parts.append("[Query Decomposition Results]")
        content_parts.append(f"Sub-queries executed: {len(decomposition.sub_queries)}")
        content_parts.append("")

        for sq in decomposition.sub_queries:
            result_count = len([r for results in sub_query_results for r in results])
            content_parts.append(f"  [{sq.index + 1}] {sq.text}")

        content_parts.append("")
        content_parts.append(f"Unique results: {len(synthesis.merged_results)} (from {synthesis.total_input_results} total, {synthesis.duplicates_removed} duplicates removed)")
        content_parts.append("")
        content_parts.append("[Synthesized Results]")

        # Add top results
        for i, result in enumerate(synthesis.merged_results[:top_k], 1):
            content_parts.append(f"{i}. [{result.source}] (score: {result.score:.2f})")
            content_parts.append(f"   {result.text[:250]}...")
            content_parts.append("")

        content = "\n".join(content_parts)

        # Build sources
        sources = [
            {
                'text': r.text[:200],
                'source': r.source,
                'score': r.score,
                'method': r.retrieval_method
            }
            for r in synthesis.merged_results[:top_k]
        ]

        return OrchestrationResult(
            content=content,
            sources=sources,
            confidence=synthesis.confidence,
            strategy_used=RoutingStrategy.DECOMPOSED,
            rag_results=synthesis.merged_results,
            maf_result=None,
            decomposition_result=decomposition,
            synthesis_result=synthesis,
            metadata={
                'sub_query_count': len(decomposition.sub_queries),
                'total_results_collected': synthesis.total_input_results,
                'duplicates_removed': synthesis.duplicates_removed,
                'deduplication_rate': synthesis.metadata.get('deduplication_rate', 0),
                'decomposition_strategy': decomposition.strategy_used.value
            }
        )

    def _format_rag_response(self, results: List) -> str:
        """Format RAG retrieval results into readable text with clean formatting.

        Args:
            results: List of RetrievalResult objects

        Returns:
            Formatted response string
        """
        if not results:
            return "No relevant information found."

        formatter = OutputFormatter(verbose=False)
        output = formatter.format_header("Retrieved Documents", 2)

        for i, result in enumerate(results[:5], 1):  # Top 5
            output += formatter.format_document_preview(
                title=f"{i}. {result.source} (score: {result.score:.2f})",
                content=result.text,
                max_length=300
            )

        return output


# Singleton instance
_orchestrator: Optional[AgentOrchestrator] = None


def get_agent_orchestrator() -> AgentOrchestrator:
    """Get or create the global agent orchestrator instance.

    Returns:
        Agent orchestrator instance
    """
    global _orchestrator

    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()

    return _orchestrator


async def test_orchestrator():
    """Test agent orchestrator functionality."""
    print("Testing Agent Orchestrator...")
    print("=" * 70)

    orchestrator = get_agent_orchestrator()

    # Test queries with different intents
    test_queries = [
        ("How to implement vector search in Python?", "Simple query → RAG only"),
        ("ValueError: invalid literal for int() with base 10: 'abc'", "Error query → Parallel RAG + MAF"),
        ("What are best practices for RAG systems?", "Best practices → RAG only")
    ]

    for query, description in test_queries:
        print(f"\nQuery: {query}")
        print(f"Expected: {description}")
        print("-" * 70)

        result = await orchestrator.orchestrate(query, top_k=3)

        print(f"Strategy: {result.strategy_used.value}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Sources: {len(result.sources)}")
        print(f"Execution Time: {result.execution_time:.3f}s")
        print(f"MAF Used: {'Yes' if result.maf_result else 'No'}")

        if result.metadata:
            print(f"Metadata: {result.metadata}")

    print("\n" + "=" * 70)
    print("Orchestrator test complete!")


if __name__ == "__main__":
    asyncio.run(test_orchestrator())
