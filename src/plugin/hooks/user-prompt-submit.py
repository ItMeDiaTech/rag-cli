#!/usr/bin/env python3
"""UserPromptSubmit hook for RAG enhancement.

This hook intercepts user queries and enhances them with relevant context
from the document knowledge base when RAG is enabled.
"""

import sys
import os
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Set environment variable to suppress console logging in hooks
os.environ['CLAUDE_HOOK_CONTEXT'] = '1'
os.environ['RAG_CLI_SUPPRESS_CONSOLE'] = '1'

# Add project root to path - handle multiple possible locations
# Could be in .claude/plugins/rag-cli (when synced to Claude Code)
# or in development directory
hook_file = Path(__file__).resolve()

# Strategy 1: Check environment variable (most explicit)
project_root = None
if 'RAG_CLI_ROOT' in os.environ:
    env_path = Path(os.environ['RAG_CLI_ROOT'])
    if env_path.exists() and (env_path / 'src' / 'core').exists():
        project_root = env_path

# Strategy 2: Try to find project root by walking up from hook location
if project_root is None:
    current = hook_file.parent
    for _ in range(10):  # Search up to 10 levels
        # Check if this is the RAG-CLI root (has src/core and src/monitoring)
        if (current / 'src' / 'core').exists() and (current / 'src' / 'monitoring').exists():
            project_root = current
            break
        current = current.parent

# Strategy 3: Check common installation locations
if project_root is None:
    potential_paths = [
        # User's home directory plugin location
        Path.home() / '.claude' / 'plugins' / 'rag-cli',
        # Relative to current working directory
        Path.cwd(),
        # Development path (if exists)
        Path.home() / 'Pictures' / 'DiaTech' / 'Programs' / 'DocHub' / 'development' / 'RAG-CLI',
    ]

    for path in potential_paths:
        if path.exists() and (path / 'src' / 'core').exists():
            project_root = path
            break

# Strategy 4: Last resort - relative to hook file location
if project_root is None:
    # Assume hook is in src/plugin/hooks/, so project root is 3 levels up
    project_root = hook_file.parents[3]
    # Validate this actually looks like project root
    if not (project_root / 'src' / 'core').exists():
        # If validation fails, raise clear error
        raise RuntimeError(
            f"Failed to locate RAG-CLI project root. Searched from: {hook_file}\n"
            f"Please set RAG_CLI_ROOT environment variable to the project directory.\n"
            f"Example: export RAG_CLI_ROOT=/path/to/RAG-CLI"
        )

sys.path.insert(0, str(project_root))

from src.core.config import get_config
from src.core.vector_store import get_vector_store
from src.core.embeddings import get_embedding_generator
from src.core.retrieval_pipeline import HybridRetriever
from src.core.claude_code_adapter import get_adapter
from src.monitoring.logger import get_logger
from src.monitoring.service_manager import ensure_services_running

logger = get_logger(__name__)

# RAG settings file
SETTINGS_FILE = project_root / "config" / "rag_settings.json"

# TCP Server URL for event submission
TCP_SERVER_URL = "http://localhost:9999"

# Cache TCP server availability to avoid repeated checks
_tcp_server_available = None
_tcp_check_time = 0


def check_tcp_server_available() -> bool:
    """Check if TCP server is available.

    Uses caching to avoid repeated connection attempts within a short time window.

    Returns:
        True if server is reachable, False otherwise
    """
    global _tcp_server_available, _tcp_check_time

    current_time = time.time()

    # Use cached result if check was recent (within 30 seconds)
    if _tcp_server_available is not None and (current_time - _tcp_check_time) < 30:
        return _tcp_server_available

    # Try to connect to TCP server
    try:
        import urllib.request
        import urllib.error

        req = urllib.request.Request(
            f"{TCP_SERVER_URL}/api/health",
            method='GET'
        )

        with urllib.request.urlopen(req, timeout=0.5) as response:
            _tcp_server_available = (response.status == 200)
            _tcp_check_time = current_time
            return _tcp_server_available

    except Exception:
        _tcp_server_available = False
        _tcp_check_time = current_time
        return False


def submit_event_to_server(event_type: str, data: Dict[str, Any]) -> bool:
    """Submit an event to the TCP server via HTTP POST.

    This enables cross-process event streaming from hooks to the web dashboard.

    Args:
        event_type: Type of event (activity, reasoning, query_enhancement, etc.)
        data: Event data dictionary

    Returns:
        True if successful, False otherwise
    """
    # Check if server is available before attempting connection
    if not check_tcp_server_available():
        logger.debug("TCP server not available, skipping event submission")
        return False

    try:
        import urllib.request
        import urllib.error

        event_payload = {
            "event_type": event_type,
            "data": data
        }

        req = urllib.request.Request(
            f"{TCP_SERVER_URL}/api/events/submit",
            data=json.dumps(event_payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=1) as response:
            return response.status == 200

    except (urllib.error.URLError, urllib.error.HTTPError, Exception) as e:
        # Mark server as unavailable on error
        global _tcp_server_available
        _tcp_server_available = False
        logger.debug(f"Failed to submit event to TCP server: {e}")
        return False


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


def should_enhance_query(query: str, settings: Dict[str, Any]) -> Tuple[bool, Optional['QueryClassification']]:
    """Determine if a query should be enhanced with RAG using intelligent classification.

    Args:
        query: User query
        settings: RAG settings

    Returns:
        Tuple of (should_enhance, classification)
    """
    # Check if RAG is enabled
    if not settings.get("enabled", False):
        return (False, None)

    # Check if it's a command
    if query.strip().startswith("/"):
        return (False, None)

    # Import query classifier
    try:
        from src.core.query_classifier import get_query_classifier, QueryIntent
    except ImportError:
        logger.warning("Query classifier not available, falling back to basic filtering")
        # Fallback to basic word count check
        word_count = len(query.split())
        if word_count < settings.get("auto_trigger_threshold", 5):
            return (False, None)
        return (True, None)

    # Classify query
    classifier = get_query_classifier(
        confidence_threshold=settings.get("classification_confidence_threshold", 0.3)
    )
    classification = classifier.classify(query)

    # Check if query is technical
    if not classification.is_technical:
        logger.debug(f"Skipping non-technical query: {query[:50]}...")
        return (False, classification)

    # Check minimum word count (relaxed with classification)
    word_count = len(query.split())
    min_words = settings.get("auto_trigger_threshold", 5)
    if word_count < min_words:
        # Allow shorter queries if they have high confidence technical intent
        if classification.confidence < 0.7:
            logger.debug(f"Query too short ({word_count} words) and low confidence ({classification.confidence:.2f})")
            return (False, classification)

    # Check exclusion patterns
    exclude_patterns = settings.get("exclude_patterns", [])
    query_lower = query.lower()
    for pattern in exclude_patterns:
        if pattern.lower() in query_lower:
            return (False, classification)

    # Check confidence threshold
    min_confidence = settings.get("min_classification_confidence", 0.5)
    if classification.confidence < min_confidence:
        logger.debug(
            f"Query confidence {classification.confidence:.2f} below threshold {min_confidence}",
            intent=classification.primary_intent.value
        )
        return (False, classification)

    # Log classification results
    logger.info(
        f"Query classified for RAG enhancement",
        intent=classification.primary_intent.value,
        confidence=classification.confidence,
        depth=classification.technical_depth.value,
        entities=len(classification.entities)
    )

    return (True, classification)


def retrieve_context(query: str, settings: Dict[str, Any], classification: Optional['QueryClassification'] = None) -> List[Dict[str, Any]]:
    """Retrieve relevant context for a query.

    Args:
        query: User query
        settings: RAG settings
        classification: Optional query classification for adaptive retrieval

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
        embedding_generator = get_embedding_generator()

        # Create retriever
        retriever = HybridRetriever(
            vector_store=vector_store,
            embedding_generator=embedding_generator,
            config=config
        )

        # Retrieve documents
        context_limit = settings.get("context_limit", 3)
        relevance_threshold = settings.get("relevance_threshold", 0.6)

        documents = retriever.search(query, top_k=context_limit * 2)

        # Filter by threshold and limit
        filtered_docs = []
        rejected_docs = []
        for doc in documents:
            score = doc.score
            if score >= relevance_threshold:
                filtered_docs.append(doc)

                # Emit reasoning for document selection
                submit_event_to_server("reasoning", {
                    "reasoning": f"Selected document '{doc.source}' with score {score:.2f} (threshold: {relevance_threshold}). "
                                f"Document matches query semantically and meets relevance threshold.",
                    "component": "user_prompt_hook",
                    "context": {
                        "document_source": doc.source,
                        "score": score,
                        "threshold": relevance_threshold,
                        "content_preview": doc.text[:100]
                    }
                })

                if len(filtered_docs) >= context_limit:
                    break
            else:
                rejected_docs.append(doc)

                # Emit reasoning for rejection
                if len(rejected_docs) <= 2:  # Only log first 2 rejections
                    submit_event_to_server("reasoning", {
                        "reasoning": f"Rejected document '{doc.source}' with score {score:.2f} "
                                    f"(below threshold: {relevance_threshold}).",
                        "component": "user_prompt_hook",
                        "context": {"document_source": doc.source, "score": score}
                    })

        logger.info(f"Retrieved {len(filtered_docs)} documents for query enhancement",
                   query_length=len(query),
                   max_score=max([d.score for d in filtered_docs]) if filtered_docs else 0)

        # Emit activity event
        submit_event_to_server("activity", {
            "activity": "documents_retrieved",
            "component": "user_prompt_hook",
            "metadata": {
                "query_length": len(query),
                "total_candidates": len(documents),
                "selected": len(filtered_docs),
                "rejected": len(rejected_docs),
                "max_score": max([d.score for d in filtered_docs]) if filtered_docs else 0
            }
        })

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

    # Use adapter for consistent formatting
    adapter = get_adapter()
    return adapter.format_hook_enhancement(documents, query)


def process_hook(event: Dict[str, Any]) -> Dict[str, Any]:
    """Process the UserPromptSubmit hook event.

    Args:
        event: Hook event data

    Returns:
        Modified event data
    """
    start_time = time.time()
    logger.info("Hook execution started", hook="UserPromptSubmit")

    try:
        # Ensure monitoring services are running (auto-start if needed)
        try:
            ensure_services_running()
        except Exception as e:
            logger.debug(f"Service startup check failed: {e}")

        # Extract query from event
        query = event.get("prompt", "")
        if not query:
            return event

        # Emit activity event: query received
        submit_event_to_server("activity", {
            "activity": "query_received",
            "component": "user_prompt_hook",
            "metadata": {
                "query_length": len(query),
                "word_count": len(query.split())
            }
        })

        # Load settings
        settings = load_rag_settings()

        # Check if we should enhance (now returns tuple)
        should_enhance, classification = should_enhance_query(query, settings)

        if not should_enhance:
            # Build skip reason
            skip_reason = "Criteria not met"
            if classification:
                skip_reason = f"Classification: {classification.primary_intent.value} (conf: {classification.confidence:.2f})"
                if not classification.is_technical:
                    skip_reason = "Non-technical query detected"

            logger.debug("Query enhancement skipped",
                        reason=skip_reason,
                        rag_enabled=settings.get("enabled", False))

            # Emit reasoning for skipping
            reasoning_context = {
                "rag_enabled": settings.get("enabled"),
                "query_word_count": len(query.split())
            }
            if classification:
                reasoning_context.update({
                    "intent": classification.primary_intent.value,
                    "confidence": classification.confidence,
                    "is_technical": classification.is_technical
                })

            submit_event_to_server("reasoning", {
                "reasoning": f"Query enhancement skipped: {skip_reason}",
                "component": "user_prompt_hook",
                "context": reasoning_context
            })

            return event

        # Retrieve context with agent orchestration (falls back to simple RAG if orchestrator unavailable)
        start_time = time.time()

        # Try orchestrated retrieval with multi-agent support
        use_orchestrator = settings.get("enable_agent_orchestration", True)
        documents = []
        orchestration_result = None
        strategy_used = "retrieve_context_fallback"

        if use_orchestrator:
            try:
                from src.core.agent_orchestrator import AgentOrchestrator

                # Initialize orchestrator
                orchestrator = AgentOrchestrator()

                # Run async orchestration
                orchestration_result = asyncio.run(orchestrator.orchestrate(
                    query=query,
                    top_k=settings.get("context_limit", 3),
                    use_cache=True
                ))

                # Extract documents from orchestration result
                if orchestration_result.rag_results:
                    documents = orchestration_result.rag_results
                    strategy_used = orchestration_result.strategy_used.value

                    # Emit orchestration reasoning
                    submit_event_to_server("reasoning", {
                        "reasoning": f"Agent orchestration used strategy: {strategy_used}. "
                                    f"Classification: {classification.primary_intent.value if classification else 'unknown'}. "
                                    f"Confidence: {orchestration_result.confidence:.2f}. "
                                    f"Retrieved {len(documents)} documents.",
                        "component": "agent_orchestrator",
                        "context": {
                            "strategy": strategy_used,
                            "intent": classification.primary_intent.value if classification else None,
                            "confidence": orchestration_result.confidence,
                            "documents_count": len(documents),
                            "maf_used": orchestration_result.maf_result is not None
                        }
                    })

                logger.info(f"Orchestrated retrieval complete",
                           strategy=strategy_used,
                           documents_count=len(documents),
                           confidence=orchestration_result.confidence)

            except Exception as e:
                logger.warning(f"Agent orchestration failed, falling back to simple retrieval: {e}")
                use_orchestrator = False

        # Fallback to simple retrieve_context if orchestrator not used or failed
        if not use_orchestrator or not documents:
            documents = retrieve_context(query, settings, classification=classification)
            strategy_used = "rag_only_fallback"

        retrieval_time = (time.time() - start_time) * 1000

        if documents:
            # Format enhanced query
            enhanced_query = format_enhanced_query(query, documents)

            # Emit query enhancement event with full details
            doc_summaries = [{
                "source": doc.source,
                "score": doc.score,
                "content_preview": doc.text[:100]
            } for doc in documents[:3]]

            submit_event_to_server("query_enhancement", {
                "original_query": query,
                "enhanced_query": enhanced_query,
                "documents_count": len(doc_summaries),
                "documents": doc_summaries,
                "reasoning": f"Enhanced query with {len(documents)} documents. "
                            f"Orchestration Strategy: {strategy_used}. "
                            f"Retrieved using {'agent orchestration' if use_orchestrator else 'fallback RAG'}. "
                            f"Context injected as markdown-formatted knowledge base references."
            })

            # Emit activity event: context assembled
            submit_event_to_server("activity", {
                "activity": "context_assembled",
                "component": "user_prompt_hook",
                "metadata": {
                    "original_query_length": len(query),
                    "enhanced_query_length": len(enhanced_query),
                    "documents_count": len(documents),
                    "retrieval_time_ms": retrieval_time
                }
            })

            # Cache retrieval results for ResponsePost hook
            try:
                import hashlib
                session_id = event.get("session_id", "unknown")
                prompt_hash = hashlib.md5(query.encode()).hexdigest()[:16]
                cache_key = f"{session_id}_{prompt_hash}"

                cache_dir = project_root / "data" / "cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file = cache_dir / f"{cache_key}.json"

                # Save retrieval results
                cache_data = {
                    "documents": [{
                        "source": doc.source,
                        "score": doc.score,
                        "text": doc.text,
                        "metadata": doc.metadata
                    } for doc in documents],
                    "timestamp": time.time()
                }

                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, indent=2)

                logger.debug(f"Cached retrieval results: {cache_key}")

            except Exception as cache_error:
                logger.warning(f"Failed to cache retrieval results: {cache_error}")
                # Continue execution even if caching fails

            # Update event
            event["prompt"] = enhanced_query
            event["metadata"] = event.get("metadata", {})
            event["metadata"]["rag_enhanced"] = True
            event["metadata"]["documents_used"] = len(documents)
            event["metadata"]["retrieval_time_ms"] = retrieval_time
            event["metadata"]["original_prompt"] = query  # Store for ResponsePost hook

            logger.info("Query enhanced with RAG",
                       original_length=len(query),
                       enhanced_length=len(enhanced_query),
                       documents=len(documents),
                       time_ms=retrieval_time)

    except Exception as e:
        logger.error(f"Hook processing failed: {e}")
        # Return original event on error
    finally:
        execution_time = (time.time() - start_time) * 1000
        logger.info("Hook execution completed",
                   hook="UserPromptSubmit",
                   execution_time_ms=execution_time,
                   rag_enhanced=event.get("metadata", {}).get("rag_enhanced", False))

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