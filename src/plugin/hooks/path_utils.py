#!/usr/bin/env python3
"""Path resolution utilities for RAG-CLI hooks.

This module provides a shared path resolution strategy for all hooks,
eliminating code duplication and ensuring consistent behavior.
"""

import os
from pathlib import Path
from typing import Optional


def find_project_root(hook_file: Optional[Path] = None, marker_file: str = 'src/core') -> Path:
    """Find RAG-CLI project root using multiple strategies.

    Strategies (in order of priority):
    1. RAG_CLI_ROOT environment variable (most explicit)
    2. Walking up from hook file location (if provided)
    3. Common installation locations (.claude/plugins/rag-cli, current directory)
    4. Relative to hook file location (fallback)

    Args:
        hook_file: Path to the hook file calling this function.
                  If None, uses __file__ from caller context.
        marker_file: File/directory to look for to identify project root.
                    Defaults to 'src/core'. Can be 'sync_plugin.py' for update hooks.

    Returns:
        Path object pointing to RAG-CLI project root

    Raises:
        RuntimeError: If project root cannot be located
    """
    if hook_file is None:
        # If not provided, we can't do walk-up strategy
        # Will fall back to env var and common locations
        hook_file = Path.cwd()
    else:
        hook_file = Path(hook_file).resolve()

    # Convert marker_file string to Path check
    def has_marker(path: Path) -> bool:
        marker_path = path / marker_file if '/' in marker_file else path / marker_file
        return (path / marker_file).exists() if '/' not in marker_file else marker_path.exists()

    # Simple marker check
    def check_marker(path: Path) -> bool:
        if marker_file == 'src/core':
            return (path / 'src' / 'core').exists()
        elif marker_file == 'sync_plugin.py':
            return (path / 'sync_plugin.py').exists()
        else:
            return (path / marker_file).exists()

    # Strategy 1: Check environment variable (most explicit)
    if 'RAG_CLI_ROOT' in os.environ:
        env_path = Path(os.environ['RAG_CLI_ROOT']).resolve()
        if env_path.exists() and check_marker(env_path):
            return env_path

    # Strategy 2: Try to find project root by walking up from hook location
    current = hook_file.parent if hook_file.is_file() else hook_file
    for _ in range(10):  # Search up to 10 levels
        # Check if this is the RAG-CLI root
        if check_marker(current):
            # For src/core marker, also check src/monitoring exists
            if marker_file == 'src/core' and not (current / 'src' / 'monitoring').exists():
                current = current.parent
                continue
            return current
        current = current.parent

    # Strategy 3: Check common installation locations
    potential_paths = [
        # User's home directory plugin location
        Path.home() / '.claude' / 'plugins' / 'rag-cli',
        # Relative to current working directory
        Path.cwd(),
    ]

    for path in potential_paths:
        if path.exists() and check_marker(path):
            return path

    # Strategy 4: Last resort - relative to hook file location
    if hook_file.is_file():
        # Assume hook is in src/plugin/hooks/, so project root is 3 levels up
        project_root = hook_file.parents[3]
        # Validate this actually looks like project root
        if check_marker(project_root):
            return project_root

    # If all strategies failed, raise clear error
    raise RuntimeError(
        f"Failed to locate RAG-CLI project root. Searched from: {hook_file}\n"
        f"Please set RAG_CLI_ROOT environment variable to the project directory.\n"
        f"Example: export RAG_CLI_ROOT=/path/to/RAG-CLI"
    )


def setup_sys_path(hook_file: Optional[Path] = None, marker_file: str = 'src/core') -> Path:
    """Find project root and add it to sys.path.

    This is a convenience function for hooks that need to import from
    the project root.

    Args:
        hook_file: Path to the hook file calling this function.
                  If None, uses current working directory.
        marker_file: File/directory to look for to identify project root.
                    Defaults to 'src/core'. Can be 'sync_plugin.py' for update hooks.

    Returns:
        Path object pointing to RAG-CLI project root
    """
    import sys

    project_root = find_project_root(hook_file, marker_file)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    return project_root
