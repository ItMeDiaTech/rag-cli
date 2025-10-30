#!/usr/bin/env python3
"""Dependency validation for RAG-CLI.

This module provides utilities to validate that all required dependencies
are installed before RAG-CLI attempts to use them.
"""

import sys
import importlib
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class DependencyValidator:
    """Validates that all required dependencies are available."""

    REQUIRED_DEPENDENCIES = {
        'sentence_transformers': 'SentenceTransformer model loading',
        'faiss': 'Vector store operations',
        'anthropic': 'Claude API integration',
        'pydantic': 'Configuration management',
        'yaml': 'Configuration file parsing',
        'flask': 'Web dashboard (optional, but recommended)',
        'pytest': 'Testing framework',
    }

    OPTIONAL_DEPENDENCIES = {
        'psutil': 'Process management utilities',
        'beautifulsoup4': 'HTML/XML parsing for document processing',
        'pdf2image': 'PDF processing',
        'python-docx': 'DOCX file processing',
        'selenium': 'Web scraping for online docs',
    }

    @classmethod
    def check_dependency(cls, module_name: str) -> Tuple[bool, Optional[str]]:
        """Check if a single dependency is installed.

        Args:
            module_name: Name of the module to check (e.g., 'sentence_transformers')

        Returns:
            Tuple of (is_installed, error_message)
            - is_installed: True if module is importable, False otherwise
            - error_message: Detailed error message if not installed, None if ok
        """
        try:
            importlib.import_module(module_name)
            return True, None
        except ImportError as e:
            return False, str(e)

    @classmethod
    def validate_required(cls) -> Tuple[bool, List[str]]:
        """Validate all required dependencies are installed.

        Returns:
            Tuple of (all_valid, error_messages)
            - all_valid: True if all required dependencies are installed
            - error_messages: List of detailed error messages
        """
        errors = []

        for module_name, description in cls.REQUIRED_DEPENDENCIES.items():
            installed, error = cls.check_dependency(module_name)
            if not installed:
                errors.append(
                    f"MISSING REQUIRED: {module_name} ({description})\n"
                    f"  Error: {error}\n"
                    f"  Install with: pip install {module_name}"
                )

        return len(errors) == 0, errors

    @classmethod
    def validate_optional(cls) -> Dict[str, str]:
        """Check optional dependencies and warn about missing ones.

        Returns:
            Dictionary of module_name -> warning_message for missing optional deps
        """
        warnings = {}

        for module_name, description in cls.OPTIONAL_DEPENDENCIES.items():
            installed, error = cls.check_dependency(module_name)
            if not installed:
                warnings[module_name] = (
                    f"OPTIONAL: {module_name} not found ({description})\n"
                    f"  Install with: pip install {module_name} "
                    f"(optional, for enhanced functionality)"
                )

        return warnings

    @classmethod
    def fail_fast(cls) -> None:
        """Validate required dependencies and fail fast if any are missing.

        Raises:
            RuntimeError: If any required dependencies are missing
        """
        all_valid, errors = cls.validate_required()

        if not all_valid:
            error_msg = "RAG-CLI Dependency Validation Failed:\n\n"
            error_msg += "\n".join(errors)
            error_msg += "\n\nPlease install missing dependencies before proceeding."

            raise RuntimeError(error_msg)

    @classmethod
    def validate_and_warn(cls) -> None:
        """Validate required dependencies and warn about optional ones.

        Raises:
            RuntimeError: If any required dependencies are missing
        """
        # First, validate required dependencies
        all_valid, errors = cls.validate_required()

        if not all_valid:
            error_msg = "RAG-CLI Dependency Validation Failed:\n\n"
            error_msg += "\n".join(errors)
            error_msg += "\n\nPlease install missing dependencies before proceeding."
            raise RuntimeError(error_msg)

        # Then check optional dependencies
        optional_warnings = cls.validate_optional()
        if optional_warnings:
            import logging
            logger = logging.getLogger(__name__)
            for module_name, warning in optional_warnings.items():
                logger.warning(f"Optional dependency missing: {module_name}")


def validate_rag_cli_dependencies() -> None:
    """Main entry point for dependency validation.

    This function should be called early in RAG-CLI initialization
    to ensure all dependencies are available before attempting to use them.

    Raises:
        RuntimeError: If any required dependencies are missing
    """
    DependencyValidator.fail_fast()
