#!/usr/bin/env python3
"""RAG Retrieval Skill for Claude Code.

This skill provides semantic search capabilities over locally indexed documents
and generates AI-powered answers using Claude Haiku.
"""

import sys
import argparse
import time
from typing import Dict, Any
from pathlib import Path

from rag_cli.core.config import get_config
from rag_cli.core.vector_store import get_vector_store
from rag_cli.core.embeddings import get_embedding_model
from rag_cli.core.retrieval_pipeline import HybridRetriever
from rag_cli.core.claude_integration import ClaudeAssistant
from rag_cli.core.claude_code_adapter import get_adapter, is_claude_code_mode
from rag_cli_plugin.services.logger import get_logger
from rag_cli_plugin.services.tcp_server import metrics_collector

logger = get_logger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RAG Retrieval Skill - Query your document knowledge base"
    )

    parser.add_argument(
        "query",
        type=str,
        help="Your question or search query"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum similarity score (default: 0.7)"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["hybrid", "vector", "keyword"],
        default="hybrid",
        help="Search mode (default: hybrid)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output"
    )

    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM generation, only show retrieved documents"
    )

    return parser.parse_args()

def format_output(result: Dict[str, Any], verbose: bool = False) -> str:
    """Format the result for CLI output.

    Args:
        result: Result dictionary from retrieval
        verbose: Whether to show detailed output

    Returns:
        Formatted output string
    """
    output = []

    # Add answer if available
    if "answer" in result:
        output.append("## Answer\n")
        output.append(result["answer"])
        output.append("\n")

    # Add sources
    if "sources" in result and result["sources"]:
        output.append("\n## Sources\n")
        for i, doc in enumerate(result["sources"], 1):
            output.append(f"\n### [{i}] {doc.source}")
            if verbose:
                output.append(f"**Score**: {doc.score:.3f}")
                output.append(f"**Content**: {doc.text[:200]}...")
            else:
                output.append(f"*Relevance: {doc.score:.1%}*")

    # Add metrics if verbose
    if verbose and "metrics" in result:
        output.append("\n## Performance Metrics\n")
        metrics = result["metrics"]
        output.append(f"- Vector Search: {metrics.get('vector_search_ms', 0):.0f}ms")
        output.append(f"- Reranking: {metrics.get('reranking_ms', 0):.0f}ms")
        if "claude_api_ms" in metrics:
            output.append(f"- Claude API: {metrics.get('claude_api_ms', 0):.0f}ms")
        output.append(f"- Total: {metrics.get('total_ms', 0):.0f}ms")

    return "\n".join(output)

def perform_retrieval(
    query: str,
    top_k: int = 5,
    threshold: float = 0.7,
    mode: str = "hybrid",
    use_llm: bool = True
) -> Dict[str, Any]:
    """Perform RAG retrieval and generation.

    Args:
        query: User query
        top_k: Number of documents to retrieve
        threshold: Minimum similarity threshold
        mode: Search mode (hybrid, vector, keyword)
        use_llm: Whether to use LLM for answer generation

    Returns:
        Result dictionary with answer and sources
    """
    start_time = time.time()
    result = {
        "query": query,
        "sources": [],
        "metrics": {}
    }

    try:
        # Initialize components
        logger.info(f"Processing query: {query}", mode=mode, top_k=top_k)

        config = get_config()
        vector_store = get_vector_store()
        embedding_model = get_embedding_model()

        # Create retriever
        retriever = HybridRetriever(
            vector_store=vector_store,
            embedding_model=embedding_model,
            config=config
        )

        # Perform retrieval
        retrieval_start = time.time()

        if mode == "vector":
            documents = retriever.vector_search(query, top_k=top_k)
        elif mode == "keyword":
            documents = retriever.keyword_search(query, top_k=top_k)
        else:  # hybrid
            documents = retriever.search(query, top_k=top_k)

        retrieval_time = (time.time() - retrieval_start) * 1000
        result["metrics"]["retrieval_ms"] = retrieval_time

        # Filter by threshold
        filtered_docs = [
            doc for doc in documents
            if doc.score >= threshold
        ]

        result["sources"] = filtered_docs

        # Record metrics
        metrics_collector.record_query()
        metrics_collector.record_latency("retrieval", retrieval_time)

        if not filtered_docs:
            logger.warning("No documents found above threshold",
                           threshold=threshold,
                           max_score=max([d.score for d in documents]) if documents else 0)
            result["answer"] = "No relevant documents found for your query. Try lowering the threshold or using different keywords."
            return result

        # Generate answer based on mode
        if use_llm:
            # Check if we're in Claude Code mode
            if is_claude_code_mode():
                logger.info("Claude Code mode - formatting context for Claude")

                # Use adapter to format response for Claude Code
                adapter = get_adapter()
                formatted_response = adapter.format_skill_response(filtered_docs, query)

                result["answer"] = formatted_response.get("context", "")
                result["mode"] = "claude_code"
                result["message"] = formatted_response.get("message", "")

                logger.info("Context formatted for Claude Code",
                            docs_count=len(filtered_docs))
            else:
                # Standalone mode - use Claude API
                claude_start = time.time()

                assistant = ClaudeAssistant(config)
                response = assistant.generate_response(query, filtered_docs)

                claude_time = (time.time() - claude_start) * 1000
                result["metrics"]["claude_api_ms"] = claude_time
                result["answer"] = response["answer"]

                metrics_collector.record_latency("claude_api", claude_time)

                logger.info("Answer generated successfully",
                            answer_length=len(response["answer"]),
                            sources_used=len(filtered_docs))

        # Calculate total time
        total_time = (time.time() - start_time) * 1000
        result["metrics"]["total_ms"] = total_time

        metrics_collector.record_latency("end_to_end", total_time)

        # Update component status
        metrics_collector.update_component_status("vector_store", "operational")
        metrics_collector.update_component_status("retriever", "operational")
        if use_llm:
            metrics_collector.update_component_status("claude", "operational")

        return result

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        metrics_collector.record_error()

        result["error"] = str(e)
        result["answer"] = f"An error occurred during retrieval: {e}"

        # Update component status
        metrics_collector.update_component_status("retriever", "error")

        return result

def main():
    """Main function for the RAG retrieval skill."""
    args = parse_arguments()

    # Check if vector store exists
    # Get project root (4 levels up from this file)
    project_root = Path(__file__).resolve().parents[4]
    vector_store_path = project_root / "data" / "vectors" / "chroma_db"
    if not vector_store_path.exists():
        print("Error: No vector index found. Please index documents first:")
        print("  rag-index ./data/documents --recursive")
        sys.exit(1)

    # Perform retrieval
    result = perform_retrieval(
        query=args.query,
        top_k=args.top_k,
        threshold=args.threshold,
        mode=args.mode,
        use_llm=not args.no_llm
    )

    # Format and print output
    output = format_output(result, verbose=args.verbose)
    print(output)

    # Return error code if retrieval failed
    if "error" in result:
        sys.exit(1)

if __name__ == "__main__":
    main()
