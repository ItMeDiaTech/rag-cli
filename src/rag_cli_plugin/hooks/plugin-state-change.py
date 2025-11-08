#!/usr/bin/env python3
"""PluginStateChange hook for RAG-CLI.

This hook handles plugin enable/disable events and persists settings
across Claude Code restarts.

Metadata:
  priority: 60
  enabled: True
  triggers: ["plugin_enabled", "plugin_disabled"]
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
        Path.home() / '.claude' / 'plugins' / 'rag-cli',
        Path.home() / '.claude' / 'plugins' / 'marketplaces' / 'rag-cli',
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
                "enabled": True,
                "auto_trigger_threshold": 5,
                "context_limit": 3,
                "relevance_threshold": 0.6,
                "exclude_patterns": []
            }
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
        logger.info("Settings saved successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")
        return False

def initialize_resources() -> bool:
    """Initialize RAG resources (vector store, services, etc.).

    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if vector store exists
        from rag_cli.core.config import get_config

        config = get_config()
        index_path = Path(config.vector_store.save_path)

        if not index_path.exists():
            logger.warning("Vector store not found - will be created on first use")
            return True

        # Try to load vector store to verify it's accessible
        try:
            from rag_cli.core.vector_store import get_vector_store
            vector_store = get_vector_store()
            doc_count = vector_store.count()
            logger.info(f"Vector store loaded: {doc_count} documents")
        except Exception as e:
            logger.warning(f"Could not load vector store: {e}")

        # Try to start monitoring services
        try:
            from rag_cli_plugin.services.service_manager import ensure_services_running
            ensure_services_running()
            logger.info("Monitoring services started")
        except Exception as e:
            logger.warning(f"Could not start monitoring services: {e}")

        return True

    except Exception as e:
        logger.error(f"Resource initialization failed: {e}")
        return False

def cleanup_resources() -> bool:
    """Cleanup RAG resources on plugin disable.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Clear cache
        cache_dir = project_root / "data" / "cache"
        if cache_dir.exists():
            import shutil
            try:
                shutil.rmtree(cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cache cleared")
            except OSError as e:
                logger.warning(f"Failed to clear cache directory: {e}")
                # Don't fail entirely if cache cleanup fails

        # Note: Don't stop monitoring services as they may be used by other sessions

        return True

    except Exception as e:
        logger.error(f"Resource cleanup failed: {e}")
        return False

def process_hook(event: Dict[str, Any]) -> Dict[str, Any]:
    """Process PluginStateChange hook event.

    Args:
        event: Hook event data

    Returns:
        Modified event
    """
    try:
        state_change = event.get('state_change', '')
        plugin_name = event.get('plugin', '')
        timestamp = event.get('timestamp', '')

        logger.info(
            f"Plugin state change: {state_change}",
            plugin=plugin_name,
            timestamp=timestamp
        )

        # Only process RAG-CLI events
        if plugin_name != 'rag-cli':
            return event

        if state_change == 'enabled':
            # Load settings
            settings = load_settings()
            logger.info("Settings loaded", enabled=settings.get('enabled'))

            # Initialize resources
            if initialize_resources():
                logger.info("RAG-CLI plugin enabled and initialized")
                event['initialization_status'] = 'success'
            else:
                logger.warning("RAG-CLI plugin enabled but initialization failed")
                event['initialization_status'] = 'partial'

            # Store settings in event metadata
            event['settings'] = settings

        elif state_change == 'disabled':
            # Save current settings
            settings = load_settings()
            save_settings(settings)

            # Cleanup resources
            if cleanup_resources():
                logger.info("RAG-CLI plugin disabled and cleaned up")
                event['cleanup_status'] = 'success'
            else:
                logger.warning("RAG-CLI plugin disabled but cleanup failed")
                event['cleanup_status'] = 'partial'

        # Mark event as processed
        event['state_change_processed'] = True

    except Exception as e:
        logger.error(f"State change hook failed: {e}")
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
