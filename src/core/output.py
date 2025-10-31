"""Clean output module for user-facing CLI messages.

This module provides a clean separation between user-facing output and
internal logging. It supports verbosity levels and formatting.
"""

import sys
from enum import IntEnum
from typing import Optional, Any, List


class Verbosity(IntEnum):
    """Verbosity levels for CLI output."""
    QUIET = 0      # No output except errors and final results
    NORMAL = 1     # Clean, minimal output (default)
    VERBOSE = 2    # Detailed output with progress info
    DEBUG = 3      # Full debug output with technical details


class Output:
    """Manages clean user-facing output."""

    def __init__(self, verbosity: Verbosity = Verbosity.NORMAL):
        """Initialize output manager.

        Args:
            verbosity: Output verbosity level
        """
        self.verbosity = verbosity
        self._suppress_logs = False

    def set_verbosity(self, level: Verbosity):
        """Set verbosity level.

        Args:
            level: New verbosity level
        """
        self.verbosity = level

    def suppress_logs(self, suppress: bool = True):
        """Control whether to suppress internal logs.

        Args:
            suppress: True to suppress logs, False to show them
        """
        self._suppress_logs = suppress

    def info(self, message: str, verbose_only: bool = False):
        """Output info message.

        Args:
            message: Message to output
            verbose_only: Only show in verbose mode
        """
        if verbose_only and self.verbosity < Verbosity.VERBOSE:
            return

        if self.verbosity >= Verbosity.NORMAL:
            print(message, file=sys.stdout)

    def success(self, message: str):
        """Output success message.

        Args:
            message: Success message
        """
        if self.verbosity >= Verbosity.NORMAL:
            print(f"[*] {message}", file=sys.stdout)

    def error(self, message: str):
        """Output error message.

        Args:
            message: Error message
        """
        # Always show errors
        print(f"[*] Error: {message}", file=sys.stderr)

    def warning(self, message: str):
        """Output warning message.

        Args:
            message: Warning message
        """
        if self.verbosity >= Verbosity.NORMAL:
            print(f"[WARNING] Warning: {message}", file=sys.stderr)

    def debug(self, message: str, **kwargs):
        """Output debug message.

        Args:
            message: Debug message
            **kwargs: Additional context to display
        """
        if self.verbosity >= Verbosity.DEBUG:
            context = " | ".join(f"{k}={v}" for k, v in kwargs.items())
            if context:
                print(f"[DEBUG] {message} | {context}", file=sys.stdout)
            else:
                print(f"[DEBUG] {message}", file=sys.stdout)

    def verbose(self, message: str):
        """Output verbose message.

        Args:
            message: Verbose message
        """
        if self.verbosity >= Verbosity.VERBOSE:
            print(f"  {message}", file=sys.stdout)

    def result(self, message: str):
        """Output final result (always shown except in quiet mode).

        Args:
            message: Result message
        """
        if self.verbosity > Verbosity.QUIET:
            print(message, file=sys.stdout)

    def format_retrieval_results(self, documents: List[Any], time_ms: float) -> str:
        """Format retrieval results based on verbosity.

        Args:
            documents: Retrieved documents
            time_ms: Retrieval time in milliseconds

        Returns:
            Formatted message
        """
        count = len(documents)

        if self.verbosity == Verbosity.QUIET:
            return ""  # No output
        elif self.verbosity == Verbosity.NORMAL:
            return f"Retrieved {count} result{'s' if count != 1 else ''} ({time_ms:.0f}ms)"
        elif self.verbosity == Verbosity.VERBOSE:
            sources = list(set(doc.source for doc in documents))
            return (f"Retrieved {count} result{'s' if count != 1 else ''} ({time_ms:.0f}ms)\n"
                    f"  Sources: {', '.join(sources)}")
        else:  # DEBUG
            details = []
            for i, doc in enumerate(documents, 1):
                details.append(f"  [{i}] {doc.source} (score: {doc.score:.3f})")
            return (f"Retrieved {count} result{'s' if count != 1 else ''} ({time_ms:.0f}ms)\n" +
                    "\n".join(details))


# Global instance
_output: Optional[Output] = None


def get_output() -> Output:
    """Get or create the global output instance.

    Returns:
        Output instance
    """
    global _output
    if _output is None:
        # Default to QUIET mode in Claude Code environment
        # Users won't see internal logs in the CLI
        verbosity = Verbosity.QUIET

        # Check environment variable for override
        import os
        verbosity_env = os.environ.get('RAG_CLI_VERBOSITY', '').upper()
        if verbosity_env == 'NORMAL':
            verbosity = Verbosity.NORMAL
        elif verbosity_env == 'VERBOSE':
            verbosity = Verbosity.VERBOSE
        elif verbosity_env == 'DEBUG':
            verbosity = Verbosity.DEBUG

        _output = Output(verbosity)
    return _output


def set_verbosity(level: Verbosity):
    """Set global verbosity level.

    Args:
        level: Verbosity level
    """
    output = get_output()
    output.set_verbosity(level)


# Convenience functions
def info(message: str, verbose_only: bool = False):
    """Output info message."""
    get_output().info(message, verbose_only)


def success(message: str):
    """Output success message."""
    get_output().success(message)


def error(message: str):
    """Output error message."""
    get_output().error(message)


def warning(message: str):
    """Output warning message."""
    get_output().warning(message)


def debug(message: str, **kwargs):
    """Output debug message."""
    get_output().debug(message, **kwargs)


def verbose(message: str):
    """Output verbose message."""
    get_output().verbose(message)


def result(message: str):
    """Output final result."""
    get_output().result(message)
