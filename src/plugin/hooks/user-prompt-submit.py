#!/usr/bin/env python3
"""UserPromptSubmit hook for RAG enhancement.

This hook intercepts user queries and enhances them with relevant context
from the document knowledge base when RAG is enabled.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.core.config import get_config
from src.core.vector_store import get_vector_store
from src.core.embeddings import get_embedding_model
from src.core.retrieval_pipeline import HybridRetriever
from src.monitoring.logger import get_logger

logger = get_logger(__name__)

# RAG settings file
SETTINGS_FILE = project_root / "config" / "rag_settings.json"


def load_rag_settings() -> Dict[str, Any]:
    """Load RAG settings from file.

    Returns:
        Dictionary with RAG settings
    """
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load RAG settings: {e}")

    # Default settings
    return {
        "enabled": False,
        "auto_trigger_threshold": 5,  # Minimum words to trigger
        "context_limit": 3,  # Maximum documents to include
        "relevance_threshold": 0.6,  # Minimum similarity score
        "cache_queries": True,
        "exclude_patterns": []  # Patterns to exclude from enhancement
    }


def save_rag_settings(settings: Dict[str, Any]):
    """Save RAG settings to file.

    Args:
        settings: Settings dictionary to save
    """
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save RAG settings: {e}")


def should_enhance_query(query: str, settings: Dict[str, Any]) -> bool:
    """Determine if a query should be enhanced with RAG.

    Args:
        query: User query
        settings: RAG settings

    Returns:
        True if query should be enhanced
    """
    # Check if RAG is enabled
    if not settings.get("enabled", False):
        return False

    # Check minimum word count
    word_count = len(query.split())
    if word_count < settings.get("auto_trigger_threshold", 5):
        return False

    # Check exclusion patterns
    exclude_patterns = settings.get("exclude_patterns", [])
    query_lower = query.lower()
    for pattern in exclude_patterns:
        if pattern.lower() in query_lower:
            return False

    # Check if it's a command
    if query.strip().startswith("/"):
        return False

    return True


def retrieve_context(query: str, settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Retrieve relevant context for a query.

    Args:
        query: User query
        settings: RAG settings

    Returns:
        List of relevant documents
    """
    try:
        # Check if vector store exists
        vector_store_path = project_root / "data" / "vectors" / "faiss_index"
        if not vector_store_path.exists():
            logger.warning("No vector index found, skipping RAG enhancement")
            return []

        # Initialize components
        config = get_config()
        vector_store = get_vector_store()
        embedding_model = get_embedding_model()

        # Create retriever
        retriever = HybridRetriever(
            vector_store=vector_store,
            embedding_model=embedding_model,
            config=config
        )

        # Retrieve documents
        context_limit = settings.get("context_limit", 3)
        relevance_threshold = settings.get("relevance_threshold", 0.6)

        documents = retriever.search(query, top_k=context_limit * 2)

        # Filter by threshold and limit
        filtered_docs = []
        for doc in documents:
            if doc.get("score", 0) >= relevance_threshold:
                filtered_docs.append(doc)
                if len(filtered_docs) >= context_limit:
                    break

        logger.info(f"Retrieved {len(filtered_docs)} documents for query enhancement",
                   query_length=len(query),
                   max_score=max([d.get("score", 0) for d in filtered_docs]) if filtered_docs else 0)

        return filtered_docs

    except Exception as e:
        logger.error(f"Failed to retrieve context: {e}")
        return []


def format_enhanced_query(query: str, documents: List[Dict[str, Any]]) -> str:
    """Format the enhanced query with retrieved context.

    Args:
        query: Original user query
        documents: Retrieved documents

    Returns:
        Enhanced query with context
    """
    if not documents:
        return query

    enhanced = []
    enhanced.append("### Context from Knowledge Base\n")

    for i, doc in enumerate(documents, 1):
        source = doc.get("source", "Unknown")
        content = doc.get("content", "")
        score = doc.get("score", 0)

        # Truncate long content
        if len(content) > 500:
            content = content[:500] + "..."

        enhanced.append(f"**[{i}] {source}** (Relevance: {score:.1%})")
        enhanced.append(f"{content}\n")

    enhanced.append("### User Query\n")
    enhanced.append(query)

    return "\n".join(enhanced)


def process_hook(event: Dict[str, Any]) -> Dict[str, Any]:
    """Process the UserPromptSubmit hook event.

    Args:
        event: Hook event data

    Returns:
        Modified event data
    """
    try:
        # Extract query from event
        query = event.get("prompt", "")
        if not query:
            return event

        # Load settings
        settings = load_rag_settings()

        # Check if we should enhance
        if not should_enhance_query(query, settings):
            logger.debug("Query enhancement skipped",
                        reason="Criteria not met",
                        rag_enabled=settings.get("enabled", False))
            return event

        # Retrieve context
        start_time = time.time()
        documents = retrieve_context(query, settings)
        retrieval_time = (time.time() - start_time) * 1000

        if documents:
            # Format enhanced query
            enhanced_query = format_enhanced_query(query, documents)

            # Update event
            event["prompt"] = enhanced_query
            event["metadata"] = event.get("metadata", {})
            event["metadata"]["rag_enhanced"] = True
            event["metadata"]["documents_used"] = len(documents)
            event["metadata"]["retrieval_time_ms"] = retrieval_time

            logger.info("Query enhanced with RAG",
                       original_length=len(query),
                       enhanced_length=len(enhanced_query),
                       documents=len(documents),
                       time_ms=retrieval_time)

    except Exception as e:
        logger.error(f"Hook processing failed: {e}")
        # Return original event on error

    return event


def main():
    """Main function for the hook."""
    try:
        # Read event from stdin
        event_json = sys.stdin.read()
        event = json.loads(event_json)

        # Process the event
        result = process_hook(event)

        # Write result to stdout
        print(json.dumps(result))

    except Exception as e:
        logger.error(f"Hook failed: {e}")
        # On error, pass through the original event
        print(event_json if 'event_json' in locals() else "{}")
        sys.exit(1)


if __name__ == "__main__":
    main()