#!/usr/bin/env python3
"""ErrorHandler hook for RAG-CLI.

This hook provides graceful degradation when RAG operations fail,
showing inline warnings with fix instructions (no emojis).

Metadata:
  priority: 70
  enabled: true
  triggers: ["error_occurred"]
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any

# Set environment variable to suppress console logging in hooks
os.environ['CLAUDE_HOOK_CONTEXT'] = '1'
os.environ['RAG_CLI_SUPPRESS_CONSOLE'] = '1'

# Import path resolution utilities
from path_utils import setup_sys_path

# Add project root to path - handle multiple possible locations
hook_file = Path(__file__).resolve()
project_root = setup_sys_path(hook_file)

from src.monitoring.logger import get_logger

logger = get_logger(__name__)


# Error type classification
RAG_ERROR_TYPES = {
    'VectorStoreNotFound': {
        'message': 'RAG Enhancement Unavailable - Vector store not found',
        'fix': 'Run /rag-project to index documents',
        'severity': 'warning'
    },
    'ServiceUnavailable': {
        'message': 'RAG Enhancement Unavailable - Service not running',
        'fix': 'Check if RAG services are running with /rag-status',
        'severity': 'warning'
    },
    'TimeoutError': {
        'message': 'RAG Enhancement Timeout - Retrieval took too long',
        'fix': 'Try reducing context_limit in configuration',
        'severity': 'warning'
    },
    'EmbeddingError': {
        'message': 'RAG Enhancement Unavailable - Embedding generation failed',
        'fix': 'Check embedding model configuration',
        'severity': 'error'
    },
    'QueryClassificationError': {
        'message': 'Query classification failed',
        'fix': 'Query will proceed without classification',
        'severity': 'info'
    },
    'IndexingError': {
        'message': 'Document indexing failed',
        'fix': 'Check document format and try again',
        'severity': 'error'
    },
    'ConfigurationError': {
        'message': 'RAG configuration invalid',
        'fix': 'Check config/rag_settings.json for errors',
        'severity': 'error'
    }
}


def classify_error(error: Dict[str, Any]) -> str:
    """Classify error type from error object.

    Args:
        error: Error dictionary with type and message

    Returns:
        Error type classification
    """
    error_type = error.get('type', '')
    error_message = str(error.get('message', '')).lower()

    # Check explicit type
    if error_type in RAG_ERROR_TYPES:
        return error_type

    # Pattern matching on error message
    if 'vector store' in error_message or 'faiss' in error_message:
        return 'VectorStoreNotFound'
    elif 'service' in error_message or 'connection' in error_message:
        return 'ServiceUnavailable'
    elif 'timeout' in error_message:
        return 'TimeoutError'
    elif 'embedding' in error_message:
        return 'EmbeddingError'
    elif 'classification' in error_message or 'classifier' in error_message:
        return 'QueryClassificationError'
    elif 'index' in error_message:
        return 'IndexingError'
    elif 'config' in error_message:
        return 'ConfigurationError'

    # Default to generic service error
    return 'ServiceUnavailable'


def format_error_message(error_type: str, context: Dict[str, Any]) -> str:
    """Format error message for display.

    Args:
        error_type: Classified error type
        context: Error context information

    Returns:
        Formatted error message (no emojis)
    """
    error_info = RAG_ERROR_TYPES.get(error_type, {
        'message': 'RAG Enhancement Error',
        'fix': 'Please check RAG configuration',
        'severity': 'error'
    })

    # Build message
    lines = [
        "",
        "=" * 60,
        f"RAG NOTICE: {error_info['message']}",
        "-" * 60,
    ]

    # Add context if available
    hook_name = context.get('hook', 'Unknown')
    if hook_name:
        lines.append(f"Hook: {hook_name}")

    query = context.get('query', '')
    if query:
        lines.append(f"Query: {query[:100]}...")

    # Add fix instruction
    lines.append("")
    lines.append(f"How to fix: {error_info['fix']}")

    # Add footer
    lines.append("-" * 60)
    lines.append("Your query will proceed without RAG enhancement.")
    lines.append("=" * 60)
    lines.append("")

    return "\n".join(lines)


def process_hook(event: Dict[str, Any]) -> Dict[str, Any]:
    """Process ErrorHandler hook event.

    Args:
        event: Hook event data

    Returns:
        Modified event with error handling
    """
    try:
        error = event.get('error', {})
        context = event.get('context', {})

        # Classify error
        error_type = classify_error(error)
        severity = RAG_ERROR_TYPES.get(error_type, {}).get('severity', 'error')

        # Log error
        logger.error(
            f"RAG error occurred: {error_type}",
            error_message=error.get('message'),
            hook=context.get('hook'),
            severity=severity
        )

        # Format error message
        error_message = format_error_message(error_type, context)

        # Add warning to event
        # Different hooks handle warnings differently
        if context.get('hook') == 'UserPromptSubmit':
            # Prepend warning to prompt
            original_prompt = event.get('prompt', '')
            event['prompt'] = error_message + "\n" + original_prompt

        # Store error info in metadata
        metadata = event.get('metadata', {})
        metadata['rag_error'] = {
            'type': error_type,
            'severity': severity,
            'handled': True
        }
        event['metadata'] = metadata

        # Mark that error was handled
        event['error_handled'] = True

        logger.info(f"Error handled gracefully: {error_type}")

    except Exception as e:
        logger.error(f"Error handler failed: {e}")
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
