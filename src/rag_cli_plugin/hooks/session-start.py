#!/usr/bin/env python3
"""SessionStart hook for RAG-CLI initialization.

This hook initializes RAG-CLI resources when a Claude Code session starts,
including loading settings, checking vector store availability, and
starting monitoring services.
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any

# Set environment variable to suppress console logging in hooks
os.environ['CLAUDE_HOOK_CONTEXT'] = '1'
os.environ['RAG_CLI_SUPPRESS_CONSOLE'] = '1'

# Initialize path resolution variables
hook_file = Path(__file__).resolve()
project_root = os.environ.get('RAG_CLI_ROOT')
if project_root:
    project_root = Path(project_root)
else:
    project_root = None

# Strategy 2: Try to find project root by walking up from hook location
if project_root is None:
    current = hook_file.parent
    for _ in range(10):  # Search up to 10 levels
        if (current / 'src' / 'rag_cli' / 'core').exists() and (current / 'src' / 'rag_cli_plugin' / 'services').exists():
            project_root = current
            break
        current = current.parent

# Strategy 3: Check common installation locations
if project_root is None:
    potential_paths = [
        Path.home() / '.claude' / 'plugins' / 'marketplaces' / 'rag-cli',
        Path.home() / '.claude' / 'plugins' / 'rag-cli',
        Path.cwd(),
    ]

    for path in potential_paths:
        if path.exists() and (path / 'src' / 'rag_cli' / 'core').exists():
            project_root = path
            break

# Strategy 4: Last resort - relative to hook file location
if project_root is None:
    project_root = hook_file.parents[3]
    if not (project_root / 'src' / 'rag_cli' / 'core').exists():
        raise RuntimeError(
            f"Failed to locate RAG-CLI project root. Searched from: {hook_file}\n"
            "Please set RAG_CLI_ROOT environment variable to the project directory."
        )

sys.path.insert(0, str(project_root / 'src'))

from rag_cli_plugin.services.logger import get_logger

logger = get_logger(__name__)

# Settings file
SETTINGS_FILE = project_root / "config" / "rag_settings.json"

def load_settings() -> Dict[str, Any]:
    """Load RAG settings from file.

    Returns:
        Settings dictionary
    """
    try:
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        else:
            # Return defaults
            return {
                "enabled": False,
                "auto_trigger_threshold": 5,
                "context_limit": 3,
                "relevance_threshold": 0.6,
                "exclude_patterns": [],
                "enable_agent_orchestration": True,
                "classification_confidence_threshold": 0.3,
                "min_classification_confidence": 0.5
            }
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        return {}

def health_check_chromadb() -> Dict[str, Any]:
    """Perform comprehensive ChromaDB health check.

    Returns:
        Dictionary with health check results
    """
    health_status = {
        "healthy": False,
        "vector_count": 0,
        "collection_exists": False,
        "can_query": False,
        "persist_directory": None,
        "errors": []
    }

    try:
        from rag_cli.core.vector_store import get_vector_store

        # Initialize vector store (creates if doesn't exist)
        vector_store = get_vector_store()

        # Check collection exists
        if vector_store.collection:
            health_status["collection_exists"] = True
            health_status["persist_directory"] = vector_store.persist_directory

            # Get vector count
            try:
                count = vector_store.collection.count()
                health_status["vector_count"] = count
                logger.info(f"ChromaDB health check: {count} vectors in collection")

                # Test query capability (if vectors exist)
                if count > 0:
                    try:
                        # Peek at first few vectors to verify read access
                        peek_result = vector_store.collection.peek(limit=1)
                        if peek_result and peek_result.get('ids'):
                            health_status["can_query"] = True
                            logger.debug("ChromaDB query test passed")
                        else:
                            health_status["errors"].append("Collection peek returned no results")
                    except Exception as e:
                        health_status["errors"].append(f"Query test failed: {str(e)}")
                else:
                    # Empty collection is valid, just no vectors yet
                    health_status["can_query"] = True
                    logger.info("ChromaDB collection is empty - ready for indexing")

            except Exception as e:
                health_status["errors"].append(f"Could not get vector count: {str(e)}")

            # Overall health status
            health_status["healthy"] = (
                health_status["collection_exists"] and
                (health_status["can_query"] or health_status["vector_count"] == 0)
            )

        else:
            health_status["errors"].append("Collection does not exist")

    except Exception as e:
        health_status["errors"].append(f"Vector store initialization failed: {str(e)}")
        logger.error(f"ChromaDB health check failed: {e}")

    return health_status


def initialize_resources() -> bool:
    """Initialize RAG resources (vector store, services, etc.).

    Returns:
        True if successful, False otherwise
    """
    try:
        # Perform ChromaDB health check
        health = health_check_chromadb()

        if health["healthy"]:
            logger.info(
                "ChromaDB health check passed",
                vectors=health["vector_count"],
                persist_dir=health["persist_directory"]
            )
        else:
            logger.warning(
                "ChromaDB health check issues",
                errors=health["errors"]
            )
            # Still return True - collection will be created on first use
            if not health["collection_exists"]:
                logger.info("ChromaDB collection will be created on first indexing")

        # Try to start monitoring services
        try:
            from rag_cli_plugin.services.service_manager import ensure_services_running
            ensure_services_running()
            logger.info("Monitoring services started for session")
        except Exception as e:
            logger.debug(f"Monitoring services not started: {e}")
            # Not critical for session start

        return True

    except Exception as e:
        logger.error(f"Resource initialization failed: {e}")
        return False

def process_hook(event: Dict[str, Any]) -> Dict[str, Any]:
    """Process SessionStart hook event.

    Args:
        event: Hook event data

    Returns:
        Modified event
    """
    try:
        session_id = event.get('session_id', 'unknown')
        logger.info("RAG-CLI session started", session_id=session_id)

        # Load settings
        settings = load_settings()
        logger.debug("RAG settings loaded", enabled=settings.get('enabled', False))

        # Initialize resources
        if initialize_resources():
            logger.info("RAG-CLI session initialization completed successfully")
            event['initialization_status'] = 'success'
        else:
            logger.warning("RAG-CLI session initialization completed with warnings")
            event['initialization_status'] = 'partial'

        # Store settings in event metadata for downstream hooks
        event['rag_settings'] = settings

    except Exception as e:
        logger.error(f"Session start hook failed: {e}")
        event['initialization_status'] = 'error'

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
