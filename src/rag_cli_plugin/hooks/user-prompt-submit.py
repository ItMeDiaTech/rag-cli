#!/usr/bin/env python3
"""UserPromptSubmit hook for RAG enhancement.

This hook intercepts user queries and enhances them with relevant context
from the document knowledge base when RAG is enabled.
"""

import sys
import os
import json
import time
import threading
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Set environment variable to suppress console logging in hooks
os.environ['CLAUDE_HOOK_CONTEXT'] = '1'
os.environ['RAG_CLI_SUPPRESS_CONSOLE'] = '1'

# Set up project paths using centralized utility
from rag_cli_plugin.hooks.path_utils import setup_sys_path
project_root = setup_sys_path(__file__)

from rag_cli.core.config import get_config
from rag_cli.core.vector_store import get_vector_store
from rag_cli.core.embeddings import get_embedding_generator
from rag_cli.core.retrieval_pipeline import HybridRetriever
from rag_cli.core.claude_code_adapter import get_adapter
from rag_cli.core.query_classifier import QueryClassification
from rag_cli_plugin.services.logger import get_logger
from rag_cli_plugin.services.service_manager import ensure_services_running
from rag_cli.core.constants import TCP_CHECK_CACHE_SECONDS, MAX_BACKOFF_SECONDS

logger = get_logger(__name__)

# RAG settings file
SETTINGS_FILE = project_root / "config" / "rag_settings.json"

# TCP Server URL for event submission
TCP_SERVER_URL = "http://localhost:9999"

# Cache TCP server availability to avoid repeated checks with exponential backoff
_tcp_state_lock = threading.Lock()
_tcp_server_available = None
_tcp_check_time = 0
_tcp_consecutive_failures = 0
_tcp_backoff_until = 0

def check_tcp_server_available() -> bool:
    """Check if TCP server is available with exponential backoff on failures.

    Uses caching to avoid repeated connection attempts within a short time window.
    Implements exponential backoff: after consecutive failures, wait progressively
    longer before retrying (30s, 60s, 120s, max 240s).

    Returns:
        True if server is reachable, False otherwise
    """
    global _tcp_server_available, _tcp_check_time, _tcp_consecutive_failures, _tcp_backoff_until

    with _tcp_state_lock:
        current_time = time.time()

        # Check if in backoff period
        if current_time < _tcp_backoff_until:
            logger.debug(f"TCP server in backoff period (until {_tcp_backoff_until - current_time:.1f}s)")
            return False

        # Use cached result if check was recent
        if _tcp_server_available is not None and (current_time - _tcp_check_time) < TCP_CHECK_CACHE_SECONDS:
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
                # Success - reset failure count
                _tcp_server_available = (response.status == 200)
                _tcp_check_time = current_time
                _tcp_consecutive_failures = 0
                _tcp_backoff_until = 0
                return _tcp_server_available

        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError, OSError) as e:
            # Network/connection errors are expected when server is not running
            logger.debug(f"TCP server not reachable: {type(e).__name__}")
            _tcp_server_available = False
            _tcp_check_time = current_time

            # Increment failure count and calculate backoff
            _tcp_consecutive_failures += 1
            backoff_seconds = min(TCP_CHECK_CACHE_SECONDS * (2 ** (_tcp_consecutive_failures - 1)), MAX_BACKOFF_SECONDS)
            _tcp_backoff_until = current_time + backoff_seconds

            if _tcp_consecutive_failures > 1:
                logger.debug(f"TCP server check failed {_tcp_consecutive_failures} times, backing off for {backoff_seconds}s")

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
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, IOError, OSError) as e:
            logger.error(f"Failed to read RAG settings file: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in RAG settings file: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading RAG settings: {e}", exc_info=True)

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
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)
    except (FileNotFoundError, IOError, OSError) as e:
        logger.error(f"Failed to write RAG settings file: {e}")
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid settings data structure: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving RAG settings: {e}", exc_info=True)

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
        from rag_cli.core.query_classifier import get_query_classifier
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
        "Query classified for RAG enhancement",
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
        vector_store_path = project_root / "data" / "vectors" / "chroma_db"
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
                    "Document matches query semantically and meets relevance threshold.",
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

    except (ConnectionError, TimeoutError) as e:
        logger.error(f"Network error during context retrieval: {e}")
        return []
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Vector store file error: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to retrieve context: {e}", exc_info=True)
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


# ==================== Process Hook Helper Functions ====================
# These functions break down the complex process_hook() logic into
# manageable, single-responsibility components for better maintainability.

def _start_monitoring_services(logger) -> None:
    """Ensure monitoring services are running.

    Args:
        logger: Logger instance

    Note:
        Failures are logged but don't stop execution.
    """
    try:
        ensure_services_running()
    except Exception as e:
        logger.debug(f"Service startup check failed: {e}")


def _validate_and_extract_query(event: Dict[str, Any]) -> Optional[str]:
    """Extract and validate query from event.

    Args:
        event: Hook event data

    Returns:
        Query string if valid, None otherwise
    """
    # Check if already processed - prevent infinite loop
    metadata = event.get("metadata", {})
    if metadata.get("rag_enhanced"):
        logger.debug("Event already processed by RAG, skipping")
        return None
    
    # Check if slash command was blocked - skip RAG enhancement
    if metadata.get("slash_command_blocked"):
        logger.debug("Slash command blocked, skipping RAG enhancement")
        return None
    
    query = event.get("prompt", "")
    if not query:
        logger.debug("Empty prompt, skipping")
        return None
    
    return query


def _emit_query_received_event(query: str) -> None:
    """Emit activity event for query received.

    Args:
        query: User query string
    """
    submit_event_to_server("activity", {
        "activity": "query_received",
        "component": "user_prompt_hook",
        "metadata": {
            "query_length": len(query),
            "word_count": len(query.split())
        }
    })


def _emit_skip_reasoning(skip_reason: str, settings: Dict[str, Any],
                         classification: Optional[QueryClassification],
                         query: str) -> None:
    """Emit reasoning event when query enhancement is skipped.

    Args:
        skip_reason: Human-readable reason for skipping
        settings: RAG settings dictionary
        classification: Query classification result
        query: Original query string
    """
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


def _orchestrate_retrieval(query: str, settings: Dict[str, Any],
                           classification: Optional[QueryClassification]) -> Tuple[List, str, Any]:
    """Attempt orchestrated retrieval with multi-agent support.

    Args:
        query: User query string
        settings: RAG settings dictionary
        classification: Query classification result

    Returns:
        Tuple of (documents, strategy_used, orchestration_result)

    Raises:
        Exception: If orchestration fails (caller should handle)
    """
    from rag_cli.core.agent_orchestrator import AgentOrchestrator

    orchestrator = AgentOrchestrator()

    # Run async orchestration
    orchestration_result = asyncio.run(orchestrator.orchestrate(
        query=query,
        top_k=settings.get("context_limit", 3),
        use_cache=True
    ))

    # Extract documents
    documents = []
    strategy_used = "retrieve_context_fallback"

    if orchestration_result.rag_results:
        documents = orchestration_result.rag_results
        strategy_used = orchestration_result.strategy_used.value

    return documents, strategy_used, orchestration_result


def _format_orchestration_summary(strategy_used: str, classification: Optional[QueryClassification],
                                  orchestration_result: Any, documents: List) -> str:
    """Format orchestration output for display.

    Args:
        strategy_used: Strategy name used
        classification: Query classification
        orchestration_result: Orchestration result object
        documents: Retrieved documents list

    Returns:
        Formatted markdown summary string
    """
    from rag_cli_plugin.services.output_formatter import OutputFormatter

    formatter = OutputFormatter(verbose=False)
    formatted_summary = formatter.format_header("Query Processing", 2)
    formatted_summary += f"**Strategy:** {strategy_used}\n"
    formatted_summary += f"**Intent:** {classification.primary_intent.value if classification else 'unknown'}\n"
    formatted_summary += f"**Confidence:** {orchestration_result.confidence:.1%}\n"
    formatted_summary += f"**Documents:** {len(documents)}\n"

    if orchestration_result.maf_result:
        formatted_summary += f"**MAF Agent:** {orchestration_result.maf_result.agent_name}\n"

    return formatted_summary


def _emit_orchestration_reasoning(strategy_used: str, classification: Optional[QueryClassification],
                                  orchestration_result: Any, documents: List,
                                  formatted_summary: str) -> None:
    """Emit reasoning event for orchestration.

    Args:
        strategy_used: Strategy name
        classification: Query classification
        orchestration_result: Orchestration result
        documents: Retrieved documents
        formatted_summary: Formatted summary string
    """
    submit_event_to_server("reasoning", {
        "reasoning": f"Agent orchestration used strategy: {strategy_used}. "
        f"Classification: {classification.primary_intent.value if classification else 'unknown'}. "
        f"Confidence: {orchestration_result.confidence:.2f}. "
        f"Retrieved {len(documents)} documents.",
        "component": "agent_orchestrator",
        "formatted_output": formatted_summary,
        "context": {
            "strategy": strategy_used,
            "intent": classification.primary_intent.value if classification else None,
            "confidence": orchestration_result.confidence,
            "documents_count": len(documents),
            "maf_used": orchestration_result.maf_result is not None
        }
    })


def _emit_query_enhancement_event(query: str, enhanced_query: str, documents: List,
                                  strategy_used: str, use_orchestrator: bool) -> None:
    """Emit query enhancement event with document details.

    Args:
        query: Original query
        enhanced_query: Enhanced query with context
        documents: Retrieved documents
        strategy_used: Strategy name
        use_orchestrator: Whether orchestrator was used
    """
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
        "Context injected as markdown-formatted knowledge base references."
    })


def _emit_context_assembled_event(query: str, enhanced_query: str,
                                  documents: List, retrieval_time: float) -> None:
    """Emit activity event for context assembly.

    Args:
        query: Original query
        enhanced_query: Enhanced query
        documents: Retrieved documents
        retrieval_time: Retrieval time in milliseconds
    """
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


def _cache_retrieval_results(documents: List, query: str, event: Dict[str, Any],
                             project_root: Path, logger) -> None:
    """Cache retrieval results for ResponsePost hook.

    Args:
        documents: Retrieved documents
        query: Original query
        event: Hook event
        project_root: Project root path
        logger: Logger instance

    Note:
        Failures are logged but don't stop execution.
    """
    try:
        import hashlib

        session_id = event.get("session_id", "unknown")
        prompt_hash = hashlib.blake2b(query.encode(), digest_size=16).hexdigest()
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

    except (FileNotFoundError, IOError, OSError) as e:
        logger.warning(f"Failed to write cache file: {e}")
    except (TypeError, ValueError) as e:
        logger.warning(f"Invalid cache data structure: {e}")


def _update_event_metadata(event: Dict[str, Any], enhanced_query: str,
                           query: str, documents: List, retrieval_time: float) -> None:
    """Update event with enhancement metadata.

    Args:
        event: Hook event (modified in place)
        enhanced_query: Enhanced query string
        query: Original query
        documents: Retrieved documents
        retrieval_time: Retrieval time in milliseconds
    """
    event["prompt"] = enhanced_query
    event["metadata"] = event.get("metadata", {})
    event["metadata"]["rag_enhanced"] = True
    event["metadata"]["documents_used"] = len(documents)
    event["metadata"]["retrieval_time_ms"] = retrieval_time
    event["metadata"]["original_prompt"] = query


# ==================== Main Process Hook Function ====================

def process_hook(event: Dict[str, Any]) -> Dict[str, Any]:
    """Process the UserPromptSubmit hook event.

    This function has been refactored into smaller helper functions for better
    maintainability. Each helper function handles a single responsibility.

    Args:
        event: Hook event data

    Returns:
        Modified event data with RAG enhancement if applicable
    """
    start_time = time.time()
    logger.info("Hook execution started", hook="UserPromptSubmit")

    try:
        # Early return: Check if already processed to prevent infinite loop
        metadata = event.get("metadata", {})
        if metadata.get("rag_enhanced"):
            logger.debug("Event already processed by RAG, skipping re-processing")
            return event
        
        # Early return: Check if slash command was blocked
        if metadata.get("slash_command_blocked"):
            logger.debug("Slash command blocked, skipping RAG enhancement")
            return event
        
        # Step 1: Start monitoring services
        _start_monitoring_services(logger)

        # Step 2: Extract and validate query
        query = _validate_and_extract_query(event)
        if not query:
            return event

        # Step 3: Emit query received event
        _emit_query_received_event(query)

        # Step 4: Load settings and check if enhancement should proceed
        settings = load_rag_settings()
        should_enhance, classification = should_enhance_query(query, settings)

        if not should_enhance:
            # Build skip reason and emit
            skip_reason = "Criteria not met"
            if classification:
                skip_reason = f"Classification: {classification.primary_intent.value} (conf: {classification.confidence:.2f})"
                if not classification.is_technical:
                    skip_reason = "Non-technical query detected"

            logger.debug("Query enhancement skipped",
                         reason=skip_reason,
                         rag_enabled=settings.get("enabled", False))

            _emit_skip_reasoning(skip_reason, settings, classification, query)
            return event

        # Step 5: Retrieve context with orchestration or fallback
        retrieval_start = time.time()
        use_orchestrator = settings.get("enable_agent_orchestration", True)
        documents = []
        orchestration_result = None
        strategy_used = "retrieve_context_fallback"

        if use_orchestrator:
            try:
                documents, strategy_used, orchestration_result = _orchestrate_retrieval(
                    query, settings, classification
                )

                # Format and emit orchestration summary
                formatted_summary = _format_orchestration_summary(
                    strategy_used, classification, orchestration_result, documents
                )
                _emit_orchestration_reasoning(
                    strategy_used, classification, orchestration_result,
                    documents, formatted_summary
                )

                logger.info("Orchestrated retrieval complete",
                            strategy=strategy_used,
                            documents_count=len(documents),
                            confidence=orchestration_result.confidence)

            except ImportError as e:
                logger.debug(f"Agent orchestrator not available: {e}")
                use_orchestrator = False
            except (ConnectionError, TimeoutError) as e:
                logger.warning(f"Network error during orchestration: {e}")
                use_orchestrator = False
            except Exception as e:
                logger.warning(f"Agent orchestration failed, falling back to simple retrieval: {e}", exc_info=True)
                use_orchestrator = False

        # Fallback to simple retrieve_context if orchestrator not used or failed
        if not use_orchestrator or not documents:
            documents = retrieve_context(query, settings, classification=classification)
            strategy_used = "rag_only_fallback"

        retrieval_time = (time.time() - retrieval_start) * 1000

        # Step 6: Process retrieved documents
        if documents:
            # Format enhanced query
            enhanced_query = format_enhanced_query(query, documents)

            # Emit query enhancement and context assembly events
            _emit_query_enhancement_event(query, enhanced_query, documents,
                                         strategy_used, use_orchestrator)
            _emit_context_assembled_event(query, enhanced_query, documents, retrieval_time)

            # Cache results for ResponsePost hook
            _cache_retrieval_results(documents, query, event, project_root, logger)

            # Update event metadata
            _update_event_metadata(event, enhanced_query, query, documents, retrieval_time)

            logger.info("Query enhanced with RAG",
                        original_length=len(query),
                        enhanced_length=len(enhanced_query),
                        documents=len(documents),
                        time_ms=retrieval_time)

    except KeyError as e:
        logger.error(f"Missing required event field: {e}")
        # Return original event on error
    except (ConnectionError, TimeoutError) as e:
        logger.error(f"Network error during hook processing: {e}")
        # Return original event on error
    except Exception as e:
        logger.error(f"Hook processing failed: {e}", exc_info=True)
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
