#!/usr/bin/env python3
"""ResponsePost hook for RAG citation injection.

This hook intercepts Claude's responses and adds inline citations [1][2]
when the response was enhanced with RAG context.

Metadata:
  priority: 80
  enabled: true
  triggers: ["response_generated"]
"""

import sys
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional

# Set environment variable to suppress console logging in hooks
os.environ['CLAUDE_HOOK_CONTEXT'] = '1'
os.environ['RAG_CLI_SUPPRESS_CONSOLE'] = '1'

# Import path resolution utilities
from path_utils import setup_sys_path

from monitoring.logger import get_logger

logger = get_logger(__name__)

# Cache file for retrieval results
CACHE_DIR = project_root / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_cache_key(session_id: str, prompt: str) -> str:
    """Generate cache key from session ID and prompt.

    Args:
        session_id: Session identifier
        prompt: User prompt text

    Returns:
        Cache key string
    """
    # Use hash of prompt to create deterministic key
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:16]
    return f"{session_id}_{prompt_hash}"

def load_cached_results(cache_key: str) -> Optional[List[Dict[str, Any]]]:
    """Load retrieval results from cache.

    Args:
        cache_key: Cache key

    Returns:
        List of documents or None if not found
    """
    cache_file = CACHE_DIR / f"{cache_key}.json"

    try:
        if not cache_file.exists():
            return None

        # Check if cache is stale (older than 5 minutes)
        import time
        if time.time() - cache_file.stat().st_mtime > 300:
            cache_file.unlink()  # Delete stale cache
            return None

        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('documents', [])

    except Exception as e:
        logger.error(f"Failed to load cache: {e}")
        return None

def format_citations(documents: List[Dict[str, Any]], max_citations: int = 3) -> str:
    """Format document sources as citations.

    Args:
        documents: List of retrieved documents
        max_citations: Maximum number of citations to include

    Returns:
        Formatted citation text
    """
    if not documents:
        return ""

    # Limit to max citations
    docs = documents[:max_citations]

    # Build citation list
    citations = ["\n\nSources:"]

    for i, doc in enumerate(docs, 1):
        source = doc.get('source', 'Unknown')
        score = doc.get('score', 0.0)

        # Extract location info if available
        metadata = doc.get('metadata', {})
        location = ""

        if 'line_start' in metadata and 'line_end' in metadata:
            location = f" (line {metadata['line_start']}-{metadata['line_end']})"
        elif 'section' in metadata:
            location = f" (section: {metadata['section']})"
        elif 'page' in metadata:
            location = f" (page {metadata['page']})"

        # Format citation
        citation = f"[{i}] {source}{location} - relevance: {score:.2f}"
        citations.append(citation)

    return "\n".join(citations)

def inject_inline_citations(response: str, num_citations: int) -> str:
    """Inject inline citation markers [1][2] into response text.

    This is a simple heuristic-based approach. For production use,
    more sophisticated NLP techniques could be used to identify
    which parts of the response correspond to which sources.

    Args:
        response: Original response text
        num_citations: Number of available citations

    Returns:
        Response with inline citations
    """
    # For now, return response without inline markers
    # In production, this would use NLP to map response segments to sources
    # Example advanced implementation:
    # - Use sentence embeddings to match response sentences to source chunks
    # - Insert [N] markers after sentences that match sources
    # - Avoid over-citing (max 1 citation per sentence)

    return response

def process_hook(event: Dict[str, Any]) -> Dict[str, Any]:
    """Process ResponsePost hook event.

    Args:
        event: Hook event data

    Returns:
        Modified event with citations
    """
    try:
        # Extract event data
        response = event.get('response', '')
        metadata = event.get('metadata', {})
        session_id = event.get('session_id', 'unknown')
        original_prompt = metadata.get('original_prompt', '')

        # Check if response was RAG-enhanced
        rag_enhanced = metadata.get('rag_enhanced', False)

        if not rag_enhanced:
            logger.debug("Response was not RAG-enhanced, skipping citations")
            return event

        # Try to get cached retrieval results
        cache_key = get_cache_key(session_id, original_prompt)
        documents = load_cached_results(cache_key)

        if not documents:
            logger.warning("No cached retrieval results found, skipping citations")
            return event

        logger.info(f"Adding citations from {len(documents)} sources")

        # Format citations
        citation_text = format_citations(documents, max_citations=3)

        # Inject inline citations (future enhancement)
        # response = inject_inline_citations(response, len(documents))

        # Append citations to response
        enhanced_response = response + citation_text

        # Update event
        event['response'] = enhanced_response
        metadata['citations_added'] = len(documents[:3])
        event['metadata'] = metadata

        logger.info("Citations added successfully", count=len(documents[:3]))

    except Exception as e:
        logger.error(f"Citation injection failed: {e}")
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
