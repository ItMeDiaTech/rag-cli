"""Centralized path resolution for RAG-CLI.

This module provides a singleton PathResolver for consistent path handling
across the RAG-CLI codebase, eliminating duplicate path resolution logic.
"""

from pathlib import Path
from typing import Optional, List
import os
import threading


class PathResolver:
    """Centralized path resolution service for RAG-CLI.

    This class provides consistent path resolution across different deployment
    scenarios (development, Claude Code plugin, standalone).
    """

    _instance: Optional['PathResolver'] = None
    _lock = threading.Lock()

    def __init__(self):
        """Initialize path resolver with project root detection."""
        self.project_root = self._resolve_project_root()
        self.config_root = self.project_root / "config"
        self.data_root = self.project_root / "data"
        self.src_root = self.project_root / "src"
        self.plugin_root = self.project_root / "src" / "rag_cli_plugin"

    @classmethod
    def get_instance(cls) -> 'PathResolver':
        """Get singleton instance of PathResolver (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = PathResolver()
        return cls._instance

    def _resolve_project_root(self) -> Path:
        """Resolve project root using multiple strategies.

        Returns:
            Resolved project root path

        Raises:
            RuntimeError: If project root cannot be determined
        """
        # Strategy 1: Environment variable (highest priority)
        # This is used during marketplace installation (CLAUDE_PLUGIN_ROOT)
        if root_env := os.environ.get('RAG_CLI_ROOT'):
            root = Path(root_env)
            if root.exists():
                return root

        # Also check CLAUDE_PLUGIN_ROOT (set by Claude Code during lifecycle hooks)
        if claude_root := os.environ.get('CLAUDE_PLUGIN_ROOT'):
            root = Path(claude_root)
            if root.exists() and (root / 'src' / 'rag_cli' / 'core').exists():
                return root

        # Strategy 2: Claude plugin directory (v2.0 structure)
        plugin_dir = Path.home() / '.claude' / 'plugins' / 'rag-cli'
        if plugin_dir.exists() and (plugin_dir / 'src' / 'rag_cli' / 'core').exists():
            return plugin_dir

        # Strategy 3: Marketplace cache directory (v2.0 structure)
        # IMPORTANT: Skip this during lifecycle hooks to prevent file locks
        # The marketplace cache is temporary and should not be used during installation
        skip_marketplace = os.environ.get('CLAUDE_LIFECYCLE_HOOK') == 'true'
        if not skip_marketplace:
            marketplace_dir = Path.home() / '.claude' / 'plugins' / 'marketplaces' / 'rag-cli'
            if marketplace_dir.exists() and (marketplace_dir / 'src' / 'rag_cli' / 'core').exists():
                return marketplace_dir

        # Strategy 4: Walk up from current file (v2.0 structure)
        current = Path(__file__).resolve().parent
        for _ in range(5):  # Search up to 5 levels
            if (current / 'src' / 'rag_cli').exists() and (current / 'src' / 'rag_cli_plugin').exists():
                return current
            current = current.parent

        # Strategy 5: Current working directory (v2.0 structure)
        cwd = Path.cwd()
        if (cwd / 'src' / 'rag_cli').exists():
            return cwd

        # If all strategies fail, raise error
        raise RuntimeError(
            "Could not resolve RAG-CLI project root. "
            "Please set RAG_CLI_ROOT environment variable to the project directory."
        )

    def get_config_path(self, filename: str) -> Path:
        """Get path to configuration file.

        Args:
            filename: Configuration filename (e.g., 'rag_settings.json')

        Returns:
            Absolute path to configuration file
        """
        return self.config_root / filename

    def get_data_path(self, *parts: str) -> Path:
        """Get path within data directory.

        Args:
            *parts: Path components within data directory

        Returns:
            Absolute path within data directory

        Example:
            >>> resolver = get_path_resolver()
            >>> vector_path = resolver.get_data_path("vectors", "chroma_db")
        """
        return self.data_root.joinpath(*parts)

    def get_plugin_path(self, *parts: str) -> Path:
        """Get path within plugin directory.

        Args:
            *parts: Path components within plugin directory

        Returns:
            Absolute path within plugin directory
        """
        return self.plugin_root.joinpath(*parts)

    def ensure_directory(self, path: Path) -> Path:
        """Ensure directory exists, creating it if necessary.

        Args:
            path: Directory path to ensure

        Returns:
            The path (for chaining)
        """
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_vector_store_path(self) -> Path:
        """Get path to vector store directory."""
        return self.get_data_path("vectors")

    def get_documents_path(self) -> Path:
        """Get path to documents directory."""
        return self.get_data_path("documents")

    def get_logs_path(self) -> Path:
        """Get path to logs directory."""
        logs_path = self.project_root / "logs"
        self.ensure_directory(logs_path)
        return logs_path


# Singleton accessor function
_path_resolver: Optional[PathResolver] = None
_path_resolver_lock = threading.Lock()


def get_path_resolver() -> PathResolver:
    """Get singleton PathResolver instance (thread-safe).

    Returns:
        PathResolver singleton instance
    """
    global _path_resolver

    if _path_resolver is None:
        with _path_resolver_lock:
            if _path_resolver is None:
                _path_resolver = PathResolver()

    return _path_resolver


def get_project_root() -> Path:
    """Quick accessor for project root path.

    Returns:
        Project root path
    """
    return get_path_resolver().project_root


def get_config_path(filename: str) -> Path:
    """Quick accessor for configuration file path.

    Args:
        filename: Configuration filename

    Returns:
        Absolute path to configuration file
    """
    return get_path_resolver().get_config_path(filename)


def get_data_path(*parts: str) -> Path:
    """Quick accessor for data directory path.

    Args:
        *parts: Path components within data directory

    Returns:
        Absolute path within data directory
    """
    return get_path_resolver().get_data_path(*parts)
