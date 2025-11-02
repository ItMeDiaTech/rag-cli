#!/usr/bin/env python3
"""DocumentIndexing hook for RAG-CLI.

This hook watches for file changes and automatically indexes new/modified documents
into the RAG knowledge base.

Metadata:
  priority: 50
  enabled: false  # Disabled by default, enable via configuration
  triggers: ["file_created", "file_modified"]
"""

import sys
import os
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Set environment variable to suppress console logging in hooks
os.environ['CLAUDE_HOOK_CONTEXT'] = '1'
os.environ['RAG_CLI_SUPPRESS_CONSOLE'] = '1'

# Import path resolution utilities (relative import for hook context)
try:
    from path_utils import setup_sys_path
except ImportError:
    # Fallback for absolute import if relative fails
    import importlib.util
    hook_dir = Path(__file__).parent
    path_utils_path = hook_dir / 'path_utils.py'
    spec = importlib.util.spec_from_file_location('path_utils', path_utils_path)
    path_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(path_utils)
    setup_sys_path = path_utils.setup_sys_path

# Set up sys.path before importing other modules
project_root = setup_sys_path()

from rag_cli.core.config import get_config
from rag_cli.core.document_processor import DocumentProcessor
from rag_cli.core.vector_store import get_vector_store
from rag_cli.core.embeddings import get_embedding_generator
from rag_cli_plugin.services.logger import get_logger

logger = get_logger(__name__)

# Configuration file
CONFIG_FILE = project_root / "config" / "auto_indexing.json"

# Debounce tracking
_pending_files: Dict[str, float] = {}  # file_path -> last_modified_time
_debounce_interval = 5.0  # seconds

def load_auto_indexing_config() -> Dict[str, Any]:
    """Load auto-indexing configuration.

    Returns:
        Configuration dictionary
    """
    default_config = {
        "enabled": False,
        "watch_patterns": [
            "docs/**/*.md",
            "README.md",
            "*.txt",
            "*.rst"
        ],
        "exclude_patterns": [
            "node_modules/**",
            ".git/**",
            "venv/**",
            "__pycache__/**",
            "*.pyc",
            ".env"
        ],
        "debounce_ms": 5000,
        "supported_formats": [".md", ".txt", ".rst", ".pd", ".docx"],
        "max_file_size_mb": 10
    }

    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                user_config = json.load(f)
                # Merge with defaults
                return {**default_config, **user_config}
        return default_config
    except Exception as e:
        logger.error(f"Failed to load auto-indexing config: {e}")
        return default_config

def should_index_file(file_path: Path, config: Dict[str, Any]) -> bool:
    """Check if file should be indexed.

    Args:
        file_path: Path to file
        config: Auto-indexing configuration

    Returns:
        True if file should be indexed
    """
    # Check file extension
    supported_formats = config.get("supported_formats", [])
    if file_path.suffix not in supported_formats:
        return False

    # Check file size
    max_size = config.get("max_file_size_mb", 10) * 1024 * 1024  # Convert to bytes
    try:
        if file_path.stat().st_size > max_size:
            logger.warning(f"File too large to index: {file_path} ({file_path.stat().st_size / 1024 / 1024:.2f} MB)")
            return False
    except (OSError, PermissionError):
        return False

    # Check exclude patterns
    exclude_patterns = config.get("exclude_patterns", [])
    file_str = str(file_path)
    for pattern in exclude_patterns:
        import fnmatch
        if fnmatch.fnmatch(file_str, pattern):
            logger.debug(f"File excluded by pattern {pattern}: {file_path}")
            return False

    return True

async def index_file(file_path: Path) -> bool:
    """Index a single file into the knowledge base.

    Args:
        file_path: Path to file to index

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Indexing file: {file_path}")

        # Initialize components
        config = get_config()
        processor = DocumentProcessor(config)
        vector_store = get_vector_store()
        embedding_generator = get_embedding_generator()

        # Read and process document
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Process into chunks
        chunks = processor.process_text(
            text=content,
            source=str(file_path),
            metadata={
                'filename': file_path.name,
                'file_type': file_path.suffix,
                'indexed_at': datetime.now().isoformat()
            }
        )

        if not chunks:
            logger.warning(f"No chunks generated from file: {file_path}")
            return False

        # Generate embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = embedding_generator.generate_batch(texts)

        # Add to vector store
        vector_store.add_documents(chunks, embeddings)

        logger.info(f"Successfully indexed {len(chunks)} chunks from {file_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to index file {file_path}: {e}")
        return False

def process_hook(event: Dict[str, Any]) -> Dict[str, Any]:
    """Process DocumentIndexing hook event.

    Args:
        event: Hook event data

    Returns:
        Modified event
    """
    try:
        # Load configuration
        config = load_auto_indexing_config()

        # Check if auto-indexing is enabled
        if not config.get("enabled", False):
            logger.debug("Auto-indexing disabled, skipping")
            return event

        # Extract event data
        event_type = event.get('event_type', '')
        file_path_str = event.get('file_path', '')
        event.get('project_path', '')

        if not file_path_str:
            logger.warning("No file path in event")
            return event

        file_path = Path(file_path_str)

        # Check if file should be indexed
        if not should_index_file(file_path, config):
            logger.debug(f"File not eligible for indexing: {file_path}")
            return event

        # Debounce: Check if file was recently modified
        current_time = time.time()
        debounce_interval = config.get("debounce_ms", 5000) / 1000.0  # Convert to seconds

        if file_path_str in _pending_files:
            last_time = _pending_files[file_path_str]
            if current_time - last_time < debounce_interval:
                logger.debug(f"File change debounced: {file_path}")
                return event

        # Update debounce tracking
        _pending_files[file_path_str] = current_time

        # Index the file asynchronously
        logger.info(f"Queueing file for indexing: {file_path}")

        # Run indexing (this is synchronous in the hook)
        # For async, we'd need to use asyncio.create_task and return immediately
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        success = loop.run_until_complete(index_file(file_path))

        # Update event metadata
        metadata = event.get('metadata', {})
        metadata['auto_indexed'] = success
        metadata['indexed_at'] = datetime.now().isoformat() if success else None
        event['metadata'] = metadata

        # Notify user
        if success:
            event['notification'] = f"Indexed: {file_path.name}"
        else:
            event['notification'] = f"Failed to index: {file_path.name}"

        logger.info(
            f"Auto-indexing {'successful' if success else 'failed'}: {file_path}",
            event_type=event_type
        )

    except Exception as e:
        logger.error(f"Document indexing hook failed: {e}")
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
