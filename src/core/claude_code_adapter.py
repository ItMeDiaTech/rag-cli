"""Claude Code adapter for RAG-CLI.

This module provides seamless integration with Claude Code, allowing the system
to work as a plugin without requiring external API keys.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.monitoring.logger import get_logger

logger = get_logger(__name__)


class OperationMode(Enum):
    """Operation modes for RAG-CLI."""
    CLAUDE_CODE = "claude_code"  # Running as Claude Code plugin
    STANDALONE = "standalone"     # Running independently with API key
    HYBRID = "hybrid"            # Can switch between modes


@dataclass
class ContextResponse:
    """Response containing formatted context for Claude."""
    context: str
    sources: List[str]
    metadata: Dict[str, Any]
    mode: str


class ClaudeCodeAdapter:
    """Adapter for Claude Code integration."""

    def __init__(self):
        """Initialize the Claude Code adapter."""
        self.mode = self._detect_mode()
        self.claude_code_env = self._is_claude_code_environment()

        logger.info(f"Claude Code adapter initialized",
                   mode=self.mode.value,
                   is_claude_code=self.claude_code_env)

    def _is_claude_code_environment(self) -> bool:
        """Detect if running in Claude Code environment.

        Returns:
            True if running in Claude Code, False otherwise
        """
        # Check for Claude-specific indicators
        indicators = [
            # Check for .claude directory
            os.path.exists('.claude'),
            # Check for Claude Code specific env vars
            os.environ.get('CLAUDE_CODE_ENV') is not None,
            os.environ.get('CLAUDE_CLI') is not None,
            # Check if running through Claude Code skills
            'claude-code' in os.environ.get('PATH', '').lower(),
            # Check parent process
            os.environ.get('CLAUDE_PARENT_PROCESS') is not None
        ]

        is_claude = any(indicators)

        if is_claude:
            logger.debug("Claude Code environment detected")

        return is_claude

    def _detect_mode(self) -> OperationMode:
        """Detect the operation mode based on environment.

        Returns:
            Detected operation mode
        """
        # First check if explicitly set
        mode_env = os.environ.get('RAG_CLI_MODE', '').lower()

        if mode_env == 'standalone':
            return OperationMode.STANDALONE
        elif mode_env == 'claude_code':
            return OperationMode.CLAUDE_CODE
        elif mode_env == 'hybrid':
            return OperationMode.HYBRID

        # Auto-detect based on environment
        if self._is_claude_code_environment():
            # Check if API key is available
            has_api_key = bool(os.environ.get('ANTHROPIC_API_KEY'))
            if has_api_key:
                return OperationMode.HYBRID
            else:
                return OperationMode.CLAUDE_CODE
        else:
            return OperationMode.STANDALONE

    def format_context_for_claude(self,
                                 documents: List[Dict[str, Any]],
                                 query: str,
                                 include_metadata: bool = True) -> ContextResponse:
        """Format retrieved documents as context for Claude.

        Args:
            documents: Retrieved documents with content and metadata
            query: Original user query
            include_metadata: Whether to include source metadata

        Returns:
            Formatted context response
        """
        if not documents:
            return ContextResponse(
                context="No relevant documents found for your query.",
                sources=[],
                metadata={"query": query, "documents_found": 0},
                mode=self.mode.value
            )

        # Build formatted context
        context_parts = []
        sources = set()

        # Add header
        context_parts.append("### Retrieved Context from Knowledge Base\n")

        for i, doc in enumerate(documents, 1):
            # Extract document info - handle both dict and object formats
            if isinstance(doc, dict):
                content = doc.get('content') or doc.get('text', '')
                source = doc.get('source', f'Document {i}')
                score = doc.get('score', 0.0)
            else:
                content = doc.text
                source = doc.source or doc.metadata.get('source', f'Document {i}')
                score = doc.score

            sources.add(source)

            # Format document
            if include_metadata:
                context_parts.append(f"**[{i}] Source: {source}** (Relevance: {score:.2%})")
            else:
                context_parts.append(f"**[{i}]**")

            context_parts.append(content)
            context_parts.append("")  # Empty line between documents

        # Add query context
        context_parts.append(f"\n### User Query\n{query}")

        # Build final context
        formatted_context = "\n".join(context_parts)

        # Prepare metadata
        def get_score(d):
            return d.get('score', 0.0) if isinstance(d, dict) else d.score

        metadata = {
            "query": query,
            "documents_found": len(documents),
            "unique_sources": len(sources),
            "mode": self.mode.value,
            "average_score": sum(get_score(d) for d in documents) / len(documents) if documents else 0
        }

        return ContextResponse(
            context=formatted_context,
            sources=list(sources),
            metadata=metadata,
            mode=self.mode.value
        )

    def should_use_api(self) -> bool:
        """Determine if API calls should be made.

        Returns:
            True if API should be used, False for context-only mode
        """
        if self.mode == OperationMode.CLAUDE_CODE:
            return False
        elif self.mode == OperationMode.STANDALONE:
            return True
        else:  # HYBRID mode
            # Use API if available and not in active Claude Code session
            has_api_key = bool(os.environ.get('ANTHROPIC_API_KEY'))
            in_claude_session = self._is_active_claude_session()
            return has_api_key and not in_claude_session

    def _is_active_claude_session(self) -> bool:
        """Check if currently in an active Claude Code session.

        Returns:
            True if in active session, False otherwise
        """
        # Check for session indicators
        session_file = Path('.claude/session.lock')
        return session_file.exists()

    def get_mode_info(self) -> Dict[str, Any]:
        """Get information about current operation mode.

        Returns:
            Dictionary with mode information
        """
        return {
            "mode": self.mode.value,
            "is_claude_code": self.claude_code_env,
            "api_available": bool(os.environ.get('ANTHROPIC_API_KEY')),
            "should_use_api": self.should_use_api(),
            "features": {
                "context_retrieval": True,
                "direct_api_calls": self.should_use_api(),
                "query_enhancement": True,
                "monitoring": True
            }
        }

    def format_skill_response(self,
                             retrieved_docs: List[Dict[str, Any]],
                             query: str) -> Dict[str, Any]:
        """Format response for Claude Code skill execution.

        Args:
            retrieved_docs: Retrieved documents
            query: User query

        Returns:
            Formatted response for skill
        """
        context_response = self.format_context_for_claude(retrieved_docs, query)

        # Return structured response for Claude Code
        return {
            "status": "success",
            "mode": self.mode.value,
            "context": context_response.context,
            "sources": context_response.sources,
            "metadata": context_response.metadata,
            "message": "Context retrieved successfully. Claude will generate a response based on this information."
        }

    def format_hook_enhancement(self,
                               documents: List[Dict[str, Any]],
                               original_query: str) -> str:
        """Format document context for query enhancement in hooks.

        Args:
            documents: Retrieved documents
            original_query: Original user query

        Returns:
            Enhanced query with context
        """
        if not documents:
            return original_query

        enhancement_parts = []

        # Add context header
        enhancement_parts.append("### Relevant Context from Knowledge Base")
        enhancement_parts.append("")

        # Add documents
        for i, doc in enumerate(documents, 1):
            content = doc.text
            source = doc.source or doc.metadata.get('source', 'Unknown')
            score = doc.score

            # Truncate long content for enhancement
            if len(content) > 500:
                content = content[:500] + "..."

            enhancement_parts.append(f"[{i}] From {source} (Relevance: {score:.1%}):")
            enhancement_parts.append(content)
            enhancement_parts.append("")

        # Add original query
        enhancement_parts.append("### Query")
        enhancement_parts.append(original_query)

        return "\n".join(enhancement_parts)


# Singleton instance
_adapter_instance: Optional[ClaudeCodeAdapter] = None


def get_adapter() -> ClaudeCodeAdapter:
    """Get or create the Claude Code adapter instance.

    Returns:
        Claude Code adapter instance
    """
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = ClaudeCodeAdapter()
    return _adapter_instance


def is_claude_code_mode() -> bool:
    """Quick check if running in Claude Code mode.

    Returns:
        True if in Claude Code mode
    """
    adapter = get_adapter()
    return adapter.mode == OperationMode.CLAUDE_CODE


def format_for_claude(documents: List[Dict[str, Any]],
                     query: str) -> Dict[str, Any]:
    """Convenience function to format documents for Claude.

    Args:
        documents: Retrieved documents
        query: User query

    Returns:
        Formatted response
    """
    adapter = get_adapter()

    if is_claude_code_mode():
        return adapter.format_skill_response(documents, query)
    else:
        # Return raw data for API mode
        return {
            "documents": documents,
            "query": query,
            "mode": "standalone"
        }