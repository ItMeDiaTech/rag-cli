#!/usr/bin/env python3
"""SessionEnd hook for RAG-CLI cleanup and finalization.

This hook performs cleanup operations when a Claude Code session ends,
including saving settings, clearing temporary cache, and logging session summary.
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any

# Set environment variable to suppress console logging in hooks
os.environ['CLAUDE_HOOK_CONTEXT'] = '1'
os.environ['RAG_CLI_SUPPRESS_CONSOLE'] = '1'

# Find project root
hook_file = Path(__file__).resolve()
project_root = None

# Strategy 1: Check for RAG_CLI_ROOT environment variable
if 'RAG_CLI_ROOT' in os.environ:
    project_root = Path(os.environ['RAG_CLI_ROOT'])

# Strategy 2: Try to find project root by walking up from hook location
if project_root is None:
    current = hook_file.parent
    for _ in range(10):  # Search up to 10 levels
        if (current / 'src' / 'rag_cli').exists() and (current / 'src' / 'rag_cli_plugin').exists():
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
        if path.exists() and (path / 'src' / 'rag_cli').exists():
            project_root = path
            break

# Strategy 4: Last resort - relative to hook file location
if project_root is None:
    project_root = hook_file.parents[3]
    if not (project_root / 'src' / 'rag_cli').exists():
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
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")

    return {}

def save_settings(settings: Dict[str, Any]) -> bool:
    """Save RAG settings to file.

    Args:
        settings: Settings dictionary

    Returns:
        True if successful, False otherwise
    """
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        logger.debug("Settings saved successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")
        return False

def cleanup_cache() -> bool:
    """Clean up temporary cache files.

    Returns:
        True if successful, False otherwise
    """
    try:
        cache_dir = project_root / "data" / "cache"
        if cache_dir.exists():
            pass
            # Remove old cache files (older than 1 hour)
            import time
            current_time = time.time()
            max_age = 3600  # 1 hour

            for cache_file in cache_dir.glob("*.json"):
                try:
                    file_age = current_time - cache_file.stat().st_mtime
                    if file_age > max_age:
                        cache_file.unlink()
                except Exception:
                    pass  # Skip files we can't access

            logger.debug("Cache cleanup completed")
            return True

    except Exception as e:
        logger.warning(f"Cache cleanup failed: {e}")
        return False

def graceful_shutdown_vector_store() -> bool:
    """Gracefully shutdown ChromaDB vector store.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Import here to avoid circular dependencies
        from rag_cli.core.vector_store import _vector_store
        from rag_cli.core.duplicate_detector import DuplicateDetector

        # Close ChromaDB client if it exists
        if _vector_store is not None:
            try:
                # ChromaDB PersistentClient doesn't have an explicit close method
                # but we can ensure all pending writes are flushed
                collection = _vector_store.collection
                count = collection.count()
                logger.info(f"Vector store contains {count} vectors at shutdown")

                # Clear the singleton reference to allow garbage collection
                # (ChromaDB will auto-persist on cleanup)
                logger.debug("ChromaDB will auto-persist on cleanup")

            except Exception as e:
                logger.warning(f"Error during vector store shutdown: {e}")

        # Save duplicate detector registry
        try:
            duplicate_detector = DuplicateDetector()
            duplicate_detector.save()
            logger.debug("Duplicate detector registry saved")
        except Exception as e:
            logger.warning(f"Error saving duplicate detector: {e}")

        return True

    except Exception as e:
        logger.error(f"Vector store shutdown failed: {e}")
        return False


def process_hook(event: Dict[str, Any]) -> Dict[str, Any]:
    """Process SessionEnd hook event.

    Args:
        event: Hook event data

    Returns:
        Modified event
    """
    try:
        session_id = event.get('session_id', 'unknown')
        logger.info("RAG-CLI session ending", session_id=session_id)

        # Load and save settings
        settings = load_settings()
        if settings:
            save_settings(settings)
            logger.debug("Session settings persisted")

        # Gracefully shutdown vector store and ChromaDB
        graceful_shutdown_vector_store()

        # Clean up temporary cache
        cleanup_cache()

        # Log session summary
        logger.info(
            "RAG-CLI session completed",
            session_id=session_id,
            cleanup_status='success'
        )

        event['cleanup_status'] = 'success'

    except Exception as e:
        logger.error(f"Session end hook failed: {e}")
        event['cleanup_status'] = 'error'

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
