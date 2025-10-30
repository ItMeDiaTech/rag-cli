#!/usr/bin/env python3
"""Utility module for resolving RAG-CLI project root path.

This module provides a single source of truth for project root resolution
across all hook files, eliminating ~70 lines of duplicate code per hook.
"""

import sys
import os
from pathlib import Path
from typing import Optional


def get_rag_cli_root() -> Path:
    """Get RAG-CLI project root using multiple fallback strategies.

    Tries the following in order:
    1. RAG_CLI_ROOT environment variable (most explicit)
    2. Walking up from current hook file location
    3. Common installation locations
    4. Relative to hook file location (last resort)

    Returns:
        Path to RAG-CLI project root

    Raises:
        RuntimeError: If project root cannot be found after all strategies
    """
    hook_file = Path(__file__).resolve()

    # Strategy 1: Check environment variable (most explicit)
    project_root = None
    if 'RAG_CLI_ROOT' in os.environ:
        env_path = Path(os.environ['RAG_CLI_ROOT'])
        if env_path.exists() and (env_path / 'src' / 'core').exists():
            return env_path

    # Strategy 2: Try to find project root by walking up from hook location
    current = hook_file.parent
    for _ in range(10):  # Search up to 10 levels
        # Check if this is the RAG-CLI root (has src/core and src/monitoring)
        if (current / 'src' / 'core').exists() and (current / 'src' / 'monitoring').exists():
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
        if path.exists() and (path / 'src' / 'core').exists():
            return path

    # Strategy 4: Last resort - relative to hook file location
    # Assume hook is in src/plugin/hooks/, so project root is 3 levels up
    project_root = hook_file.parents[3]

    # Validate this actually looks like project root
    if not (project_root / 'src' / 'core').exists():
        raise RuntimeError(
            f"Failed to locate RAG-CLI project root. Searched from: {hook_file}\n"
            "Please set RAG_CLI_ROOT environment variable to the project directory.\n"
            "Example: export RAG_CLI_ROOT=/path/to/RAG-CLI"
        )

    return project_root


def setup_path(project_root: Optional[Path] = None) -> Path:
    """Setup Python path and return project root.

    This is a convenience function that resolves the project root and
    adds it to sys.path for imports to work correctly.

    Args:
        project_root: Optional explicit project root. If not provided,
                     it will be resolved automatically.

    Returns:
        Path to RAG-CLI project root
    """
    if project_root is None:
        project_root = get_rag_cli_root()

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    return project_root
