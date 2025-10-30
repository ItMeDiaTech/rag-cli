"""Command-line interface tools for RAG-CLI.

This module contains CLI commands for indexing documents and retrieving information.
"""

from cli.index import main as index_main
from cli.retrieve import main as retrieve_main

__all__ = ['index_main', 'retrieve_main']
