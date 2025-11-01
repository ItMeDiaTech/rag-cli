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
        if (current / 'src' / 'core').exists() and (current / 'src' / 'monitoring').exists():
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
        if path.exists() and (path / 'src' / 'core').exists():
            project_root = path
            break

# Strategy 4: Last resort - relative to hook file location
if project_root is None:
    project_root = hook_file.parents[3]
    if not (project_root / 'src' / 'core').exists():
        raise RuntimeError(
            f"Failed to locate RAG-CLI project root. Searched from: {hook_file}\n"
            "Please set RAG_CLI_ROOT environment variable to the project directory."
        )

sys.path.insert(0, str(project_root))

from monitoring.logger import get_logger

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

def initialize_resources() -> bool:
    """Initialize RAG resources (vector store, services, etc.).

    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if vector store exists
        from core.config import get_config

        config = get_config()
        index_path = Path(config.vector_store.save_path)

        if not index_path.exists():
            logger.info("Vector store not yet initialized - will be created on first indexing")
            return True

        # Try to load vector store to verify it's accessible
        try:
            from core.vector_store import get_vector_store
            vector_store = get_vector_store()
            doc_count = vector_store.count()
            logger.info(f"Vector store loaded successfully: {doc_count} documents available")
        except Exception as e:
            logger.warning(f"Could not load vector store: {e}")
            return True  # Not critical for session start

        # Try to start monitoring services
        try:
            from monitoring.service_manager import ensure_services_running
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
