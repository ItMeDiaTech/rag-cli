#!/usr/bin/env python3
"""Utility module for resolving RAG-CLI project root path.

This module provides a single source of truth for project root resolution
across all hook files, eliminating ~70 lines of duplicate code per hook.
"""

import sys
import os
from pathlib import Path
from typing import Optional

def get_rag_cli_root(hook_file: Optional[Path] = None) -> Path:
    """Get RAG-CLI project root using multiple fallback strategies.

    Tries the following in order:
    1. RAG_CLI_ROOT environment variable (most explicit)
    2. Walking up from current hook file location
    3. Common installation locations
    4. Relative to hook file location (last resort)

    Args:
        hook_file: Optional path to the hook file. If not provided, will try to detect from call stack.

    Returns:
        Path to RAG-CLI project root

    Raises:
        RuntimeError: If project root cannot be found after all strategies
    """
    # Get the calling hook file location
    if hook_file is None:
        import inspect
        try:
            # Try to get from call stack (setup_sys_path -> hook)
            stack = inspect.stack()
            # Look for the hook file (not path_utils.py itself)
            for frame_info in stack[1:]:
                frame_file = Path(frame_info.filename).resolve()
                if frame_file.name != 'path_utils.py' and 'hooks' in str(frame_file):
                    hook_file = frame_file
                    break
            if hook_file is None:
                # Fallback to caller's file
                hook_file = Path(stack[2].filename).resolve()
        except (IndexError, AttributeError):
            # If inspection fails, try __file__ from path_utils itself
            hook_file = Path(__file__).resolve()
    
    if not isinstance(hook_file, Path):
        hook_file = Path(hook_file).resolve()
    
    project_root: Optional[Path] = None
    
    # Strategy 1: RAG_CLI_ROOT environment variable (most explicit)
    rag_cli_root = os.environ.get('RAG_CLI_ROOT')
    if rag_cli_root:
        project_root = Path(rag_cli_root)
        if project_root.exists() and (project_root / 'src' / 'rag_cli').exists():
            return project_root
    
    # Strategy 2: Try to find project root by walking up from hook location
    current = hook_file.parent
    for _ in range(10):  # Search up to 10 levels
        # Check if this is the RAG-CLI root (has src/rag_cli and src/rag_cli_plugin)
        if (current / 'src' / 'rag_cli').exists() and (current / 'src' / 'rag_cli_plugin').exists():
            project_root = current
            break
        # Also check legacy structure (src/core and src/monitoring)
        if (current / 'src' / 'core').exists() and (current / 'src' / 'monitoring').exists():
            project_root = current
            break
        current = current.parent
    
    # Strategy 3: Check common installation locations
    if project_root is None:
        # IMPORTANT: Skip marketplace cache during lifecycle hooks to prevent file locks
        skip_marketplace = os.environ.get('CLAUDE_LIFECYCLE_HOOK') == 'true'

        potential_paths = [
            # Manual plugin installation location (check first)
            Path.home() / '.claude' / 'plugins' / 'rag-cli',
            # Relative to current working directory
            Path.cwd(),
        ]

        # Only check marketplace cache if NOT in lifecycle hook
        if not skip_marketplace:
            # GitHub marketplace installation location (temporary cache)
            potential_paths.insert(0, Path.home() / '.claude' / 'plugins' / 'marketplaces' / 'rag-cli')

        for path in potential_paths:
            if path.exists():
                # Check for new structure
                if (path / 'src' / 'rag_cli').exists() and (path / 'src' / 'rag_cli_plugin').exists():
                    project_root = path
                    break
                # Check for legacy structure
                if (path / 'src' / 'core').exists() and (path / 'src' / 'monitoring').exists():
                    project_root = path
                    break
    
    # Strategy 4: Last resort - relative to hook file location
    if project_root is None:
        # Try different parent levels based on hook location
        # hooks are typically in src/rag_cli_plugin/hooks/
        for levels_up in [3, 4, 5]:
            try:
                potential_root = hook_file.parents[levels_up - 1]
                if (potential_root / 'src' / 'rag_cli').exists():
                    project_root = potential_root
                    break
                if (potential_root / 'src' / 'core').exists():
                    project_root = potential_root
                    break
            except IndexError:
                continue
    
    # Validate that we found a valid project root
    if project_root is None:
        raise RuntimeError(
            f"Failed to locate RAG-CLI project root. Searched from: {hook_file}\n"
            "Please set RAG_CLI_ROOT environment variable to the project directory.\n"
            "Example: export RAG_CLI_ROOT=/path/to/RAG-CLI"
        )
    
    # Final validation
    if not project_root.exists():
        raise RuntimeError(
            f"RAG-CLI project root does not exist: {project_root}\n"
            "Please verify RAG_CLI_ROOT environment variable or installation."
        )
    
    return project_root


def setup_sys_path(hook_file: Optional[Path] = None) -> Path:
    """Set up sys.path for RAG-CLI module imports.
    
    Finds the RAG-CLI project root and adds it (and src/) to sys.path
    so that rag_cli and rag_cli_plugin modules can be imported.
    
    Args:
        hook_file: Optional path to the hook file calling this function.
                   If not provided, will try to detect from call stack.
    
    Returns:
        Path to RAG-CLI project root
        
    Raises:
        RuntimeError: If project root cannot be found
    """
    # Get hook file from caller if not provided
    if hook_file is None:
        import inspect
        try:
            # Get the file that called setup_sys_path
            caller_frame = inspect.stack()[1]
            hook_file = Path(caller_frame.filename).resolve()
        except (IndexError, AttributeError):
            hook_file = None
    
    project_root = get_rag_cli_root(hook_file)
    
    # Add project root to sys.path (for editable installs)
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    
    # Add src directory to sys.path (for package imports)
    src_dir = project_root / 'src'
    if src_dir.exists():
        src_str = str(src_dir)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)
    
    return project_root