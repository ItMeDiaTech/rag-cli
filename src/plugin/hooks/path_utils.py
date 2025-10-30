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