#!/usr/bin/env python3
"""File watcher for automatic document indexing.

This module provides file system monitoring capabilities using the watchdog
library. It automatically indexes documents when they are created or modified
in watched directories.

Uses best practice pattern: run in background thread, debounced events.
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Callable
from datetime import datetime, timedelta
import fnmatch

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.monitoring.logger import get_logger

logger = get_logger(__name__)


class DocumentFileHandler(FileSystemEventHandler):
    """Handles file system events for document indexing."""

    def __init__(
        self,
        config: Dict,
        index_callback: Callable,
        debounce_interval: float = 5.0
    ):
        """Initialize the file handler.

        Args:
            config: Configuration dictionary with file patterns
            index_callback: Async callback function for indexing
            debounce_interval: Debounce interval in seconds
        """
        super().__init__()
        self.config = config
        self.index_callback = index_callback
        self.debounce_interval = debounce_interval
        self.pending_files: Dict[str, float] = {}
        self.supported_formats = set(config.get("supported_formats", [".md", ".txt", ".rst"]))
        self.exclude_patterns = config.get("exclude_patterns", [])
        self.max_file_size_mb = config.get("max_file_size_mb", 10)

    def should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed.

        Args:
            file_path: Path to the file

        Returns:
            True if file should be indexed
        """
        # Check file extension
        if file_path.suffix not in self.supported_formats:
            return False

        # Check file size
        try:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                logger.debug(f"File too large: {file_path} ({file_size_mb:.2f} MB)")
                return False
        except Exception:
            return False

        # Check exclusion patterns
        file_str = str(file_path)
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(file_str, pattern):
                logger.debug(f"File excluded: {file_path} (pattern: {pattern})")
                return False

        return True

    def on_modified(self, event: FileModifiedEvent):
        """Handle file modified event."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if not self.should_process_file(file_path):
            return

        logger.debug(f"File modified: {file_path}")
        self._queue_indexing(file_path)

    def on_created(self, event: FileCreatedEvent):
        """Handle file created event."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if not self.should_process_file(file_path):
            return

        logger.debug(f"File created: {file_path}")
        self._queue_indexing(file_path)

    def _queue_indexing(self, file_path: Path):
        """Queue a file for indexing with debouncing.

        Args:
            file_path: Path to file to index
        """
        file_str = str(file_path)
        current_time = time.time()

        # Check if file is already pending
        if file_str in self.pending_files:
            last_time = self.pending_files[file_str]
            if current_time - last_time < self.debounce_interval:
                logger.debug(f"File change debounced: {file_path}")
                return

        # Update pending files
        self.pending_files[file_str] = current_time

        # Schedule indexing callback
        try:
            asyncio.create_task(self.index_callback(file_path))
        except RuntimeError:
            # No event loop running, try with new loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.index_callback(file_path))
            finally:
                loop.close()


class FileWatcher:
    """Manages file system watching for document indexing."""

    def __init__(self, config: Dict, index_callback: Callable):
        """Initialize the file watcher.

        Args:
            config: Configuration dictionary
            index_callback: Async callback function for indexing
        """
        if not WATCHDOG_AVAILABLE:
            logger.warning("watchdog library not available - file watching disabled")
            self.observer = None
            return

        self.config = config
        self.index_callback = index_callback
        self.observer = Observer()
        self.watch_paths: Set[str] = set()
        self.enabled = config.get("enabled", False)

        logger.info(f"File watcher initialized (enabled={self.enabled})")

    def add_watch_path(self, path: Path) -> bool:
        """Add a directory to watch.

        Args:
            path: Directory path to watch

        Returns:
            True if successfully added
        """
        if not WATCHDOG_AVAILABLE:
            logger.warning("watchdog not available, cannot add watch path")
            return False

        if not self.enabled:
            logger.debug("File watching disabled, skipping path registration")
            return False

        if not path.exists() or not path.is_dir():
            logger.warning(f"Path does not exist or is not a directory: {path}")
            return False

        path_str = str(path)
        if path_str in self.watch_paths:
            logger.debug(f"Path already being watched: {path}")
            return True

        try:
            handler = DocumentFileHandler(
                config=self.config,
                index_callback=self.index_callback,
                debounce_interval=self.config.get("debounce_ms", 5000) / 1000.0
            )

            self.observer.schedule(handler, str(path), recursive=True)
            self.watch_paths.add(path_str)

            logger.info(f"Added watch path: {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to add watch path {path}: {e}")
            return False

    def start(self) -> bool:
        """Start the file watcher.

        Returns:
            True if successfully started
        """
        if not WATCHDOG_AVAILABLE:
            logger.warning("watchdog not available, cannot start file watcher")
            return False

        if not self.enabled:
            logger.debug("File watching disabled, not starting observer")
            return True

        if self.observer.is_alive():
            logger.debug("File watcher already running")
            return True

        try:
            self.observer.start()
            logger.info("File watcher started")
            return True
        except Exception as e:
            logger.error(f"Failed to start file watcher: {e}")
            return False

    def stop(self) -> bool:
        """Stop the file watcher.

        Returns:
            True if successfully stopped
        """
        if not WATCHDOG_AVAILABLE or not self.observer:
            return True

        if not self.observer.is_alive():
            logger.debug("File watcher not running")
            return True

        try:
            self.observer.stop()
            self.observer.join(timeout=5)
            logger.info("File watcher stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop file watcher: {e}")
            return False

    def is_running(self) -> bool:
        """Check if file watcher is running.

        Returns:
            True if running
        """
        return WATCHDOG_AVAILABLE and self.observer and self.observer.is_alive()


async def index_document_callback(file_path: Path) -> bool:
    """Default callback for indexing a document.

    Args:
        file_path: Path to document to index

    Returns:
        True if successful
    """
    try:
        logger.info(f"Indexing document: {file_path}")

        from src.core.config import get_config
        from src.core.document_processor import DocumentProcessor
        from src.core.vector_store import get_vector_store
        from src.core.embeddings import get_embedding_generator

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

        logger.info(f"Successfully indexed {len(chunks)} chunks from {file_path.name}")
        return True

    except Exception as e:
        logger.error(f"Failed to index document {file_path}: {e}")
        return False
