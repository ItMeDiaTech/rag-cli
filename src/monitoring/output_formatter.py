#!/usr/bin/env python3
"""Output formatting utilities for clean MCP/MAF orchestration display.

This module provides formatted output for RAG-CLI operations, making the
orchestration process clear and readable instead of displaying raw implementation details.
"""

import time
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass


class StageStatus(Enum):
    """Status of an orchestration stage."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """Result of an orchestration stage."""
    name: str
    status: StageStatus
    duration_ms: Optional[float] = None
    items_processed: Optional[int] = None
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class OutputFormatter:
    """Formats orchestration output for clean display."""

    def __init__(self, verbose: bool = False):
        """Initialize formatter.

        Args:
            verbose: Show detailed information
        """
        self.verbose = verbose
        self.start_time = time.time()

    @staticmethod
    def format_header(text: str, level: int = 1) -> str:
        """Format a markdown header.

        Args:
            text: Header text
            level: Header level (1-3)

        Returns:
            Formatted header string
        """
        prefix = "#" * level
        return f"\n{prefix} {text}\n"

    @staticmethod
    def format_stage(stage: StageResult) -> str:
        """Format a stage result.

        Args:
            stage: Stage result to format

        Returns:
            Formatted stage string
        """
        # Status indicator
        indicators = {
            StageStatus.PENDING: "⏳",
            StageStatus.IN_PROGRESS: "🔄",
            StageStatus.COMPLETED: "✓",
            StageStatus.FAILED: "✗",
            StageStatus.SKIPPED: "→"
        }
        indicator = indicators.get(stage.status, "•")

        # Format base status line
        parts = [f"{indicator} **{stage.name}**"]

        # Add duration if available
        if stage.duration_ms is not None:
            parts.append(f"({stage.duration_ms:.0f}ms)")

        # Add item count if available
        if stage.items_processed is not None:
            parts.append(f"- {stage.items_processed} items")

        status_line = " ".join(parts)

        # Add error if present
        if stage.error:
            status_line += f"\n  Error: {stage.error}"

        # Add details in verbose mode
        if stage.details:
            details_str = OutputFormatter._format_details(stage.details)
            if details_str:
                status_line += f"\n{details_str}"

        return status_line

    @staticmethod
    def _format_details(details: Dict[str, Any], indent: int = 2) -> str:
        """Format details dictionary.

        Args:
            details: Details to format
            indent: Indentation spaces

        Returns:
            Formatted details string
        """
        if not details:
            return ""

        lines = []
        indent_str = " " * indent

        for key, value in details.items():
            if isinstance(value, (list, tuple)):
                lines.append(f"{indent_str}- {key}: {len(value)} items")
            elif isinstance(value, dict):
                lines.append(f"{indent_str}- {key}:")
                nested = OutputFormatter._format_details(value, indent + 2)
                lines.append(nested)
            else:
                lines.append(f"{indent_str}- {key}: {value}")

        return "\n".join(lines)

    def format_query_analysis(self, query: str, intent: str, strategy: str) -> str:
        """Format query analysis output.

        Args:
            query: User query
            intent: Detected intent
            strategy: Selected routing strategy

        Returns:
            Formatted analysis string
        """
        output = self.format_header("Query Analysis", 2)
        output += f"**Query:** {query}\n"
        output += f"**Intent:** {intent}\n"
        output += f"**Strategy:** {strategy}\n"
        return output

    def format_retrieval_progress(self, current: int, total: int, source: str) -> str:
        """Format retrieval progress.

        Args:
            current: Current item number
            total: Total items
            source: Source being searched

        Returns:
            Formatted progress string
        """
        return f"Searching {source}... ({current}/{total})"

    def format_search_results(
        self,
        num_results: int,
        search_time_ms: float,
        sources: Optional[List[str]] = None
    ) -> str:
        """Format search results summary.

        Args:
            num_results: Number of results found
            search_time_ms: Search duration in milliseconds
            sources: List of sources searched

        Returns:
            Formatted results string
        """
        output = self.format_header("Retrieval Results", 2)
        output += f"**Found:** {num_results} relevant documents\n"
        output += f"**Time:** {search_time_ms:.0f}ms\n"

        if sources and self.verbose:
            output += f"**Sources:** {', '.join(sources)}\n"

        return output

    def format_maf_execution(
        self,
        agent_name: str,
        task: str,
        duration_ms: Optional[float] = None
    ) -> str:
        """Format MAF agent execution.

        Args:
            agent_name: Name of agent
            task: Task being performed
            duration_ms: Execution duration

        Returns:
            Formatted execution string
        """
        output = self.format_header(f"MAF Agent: {agent_name}", 2)
        output += f"**Task:** {task}\n"

        if duration_ms is not None:
            output += f"**Duration:** {duration_ms:.0f}ms\n"

        return output

    def format_synthesis(
        self,
        num_sources: int,
        confidence: Optional[float] = None
    ) -> str:
        """Format response synthesis.

        Args:
            num_sources: Number of sources synthesized
            confidence: Confidence score (0-1)

        Returns:
            Formatted synthesis string
        """
        output = self.format_header("Response Synthesis", 2)
        output += f"**Sources Integrated:** {num_sources}\n"

        if confidence is not None:
            output += f"**Confidence:** {confidence:.1%}\n"

        return output

    def format_orchestration_summary(self, stages: List[StageResult]) -> str:
        """Format complete orchestration summary.

        Args:
            stages: List of stage results

        Returns:
            Formatted summary string
        """
        output = self.format_header("Orchestration Pipeline", 1)

        # Add each stage
        for stage in stages:
            output += self.format_stage(stage) + "\n"

        # Add total time
        total_time = (time.time() - self.start_time) * 1000
        output += f"\n**Total Time:** {total_time:.0f}ms\n"

        return output

    def format_error(self, error: str, details: Optional[str] = None) -> str:
        """Format error message.

        Args:
            error: Error message
            details: Additional error details

        Returns:
            Formatted error string
        """
        output = "\n✗ **Error**\n"
        output += f"{error}\n"

        if details and self.verbose:
            output += f"\nDetails: {details}\n"

        return output

    def format_collapsible_section(
        self,
        title: str,
        content: str,
        collapsed: bool = True
    ) -> str:
        """Format a collapsible section (for verbose output).

        Args:
            title: Section title
            content: Section content
            collapsed: Whether section starts collapsed

        Returns:
            Formatted collapsible section
        """
        # Using HTML details tag for markdown
        state = "" if collapsed else " open"
        return f"\n<details{state}>\n<summary>{title}</summary>\n\n{content}\n\n</details>\n"

    def format_progress_bar(
        self,
        current: int,
        total: int,
        width: int = 30,
        prefix: str = ""
    ) -> str:
        """Format a text progress bar.

        Args:
            current: Current progress
            total: Total items
            width: Bar width in characters
            prefix: Prefix text

        Returns:
            Formatted progress bar string
        """
        if total == 0:
            return f"{prefix}[{'=' * width}] 100%"

        percent = current / total
        filled = int(width * percent)
        bar = "=" * filled + "-" * (width - filled)

        return f"{prefix}[{bar}] {percent:.0%} ({current}/{total})"

    def format_metrics_table(self, metrics: Dict[str, Any]) -> str:
        """Format metrics as a table.

        Args:
            metrics: Metrics dictionary

        Returns:
            Formatted table string
        """
        if not metrics:
            return ""

        output = "\n| Metric | Value |\n|--------|-------|\n"

        for key, value in metrics.items():
            # Format different value types
            if isinstance(value, float):
                formatted_value = f"{value:.2f}"
            elif isinstance(value, int):
                formatted_value = str(value)
            else:
                formatted_value = str(value)

            # Clean up key for display
            display_key = key.replace("_", " ").title()

            output += f"| {display_key} | {formatted_value} |\n"

        return output

    @staticmethod
    def format_document_preview(
        title: str,
        content: str,
        max_length: int = 200
    ) -> str:
        """Format a document preview.

        Args:
            title: Document title
            content: Document content
            max_length: Maximum preview length

        Returns:
            Formatted preview string
        """
        # Truncate content if needed
        preview = content[:max_length]
        if len(content) > max_length:
            preview += "..."

        return f"**{title}**\n> {preview}\n"

    def format_citation(self, source: str, index: int) -> str:
        """Format a citation reference.

        Args:
            source: Source identifier
            index: Citation index

        Returns:
            Formatted citation string
        """
        return f"[{index}] {source}"

    def format_timestamp(self, timestamp: Optional[float] = None) -> str:
        """Format a timestamp.

        Args:
            timestamp: Unix timestamp (defaults to now)

        Returns:
            Formatted timestamp string
        """
        if timestamp is None:
            timestamp = time.time()

        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


# Convenience functions for common formatting tasks

def format_rag_pipeline(
    query: str,
    num_docs: int,
    search_time_ms: float,
    synthesize_time_ms: float,
    verbose: bool = False
) -> str:
    """Format complete RAG pipeline output.

    Args:
        query: User query
        num_docs: Number of documents retrieved
        search_time_ms: Search duration
        synthesize_time_ms: Synthesis duration
        verbose: Show detailed information

    Returns:
        Formatted pipeline output
    """
    formatter = OutputFormatter(verbose=verbose)

    stages = [
        StageResult(
            name="Query Processing",
            status=StageStatus.COMPLETED,
            duration_ms=10,
            details={"query_length": len(query)} if verbose else None
        ),
        StageResult(
            name="Document Retrieval",
            status=StageStatus.COMPLETED,
            duration_ms=search_time_ms,
            items_processed=num_docs
        ),
        StageResult(
            name="Response Synthesis",
            status=StageStatus.COMPLETED,
            duration_ms=synthesize_time_ms
        )
    ]

    return formatter.format_orchestration_summary(stages)


def format_maf_pipeline(
    agent: str,
    task: str,
    execution_time_ms: float,
    success: bool = True,
    error: Optional[str] = None
) -> str:
    """Format MAF agent pipeline output.

    Args:
        agent: Agent name
        task: Task description
        execution_time_ms: Execution duration
        success: Whether execution succeeded
        error: Error message if failed

    Returns:
        Formatted pipeline output
    """
    formatter = OutputFormatter()

    status = StageStatus.COMPLETED if success else StageStatus.FAILED

    stages = [
        StageResult(
            name=f"MAF {agent} Agent",
            status=status,
            duration_ms=execution_time_ms,
            details={"task": task},
            error=error
        )
    ]

    return formatter.format_orchestration_summary(stages)


def format_hybrid_pipeline(
    rag_docs: int,
    rag_time_ms: float,
    maf_agent: str,
    maf_time_ms: float,
    synthesis_time_ms: float
) -> str:
    """Format hybrid RAG+MAF pipeline output.

    Args:
        rag_docs: Number of RAG documents
        rag_time_ms: RAG retrieval time
        maf_agent: MAF agent name
        maf_time_ms: MAF execution time
        synthesis_time_ms: Final synthesis time

    Returns:
        Formatted pipeline output
    """
    formatter = OutputFormatter()

    stages = [
        StageResult(
            name="RAG Retrieval",
            status=StageStatus.COMPLETED,
            duration_ms=rag_time_ms,
            items_processed=rag_docs
        ),
        StageResult(
            name=f"MAF {maf_agent}",
            status=StageStatus.COMPLETED,
            duration_ms=maf_time_ms
        ),
        StageResult(
            name="Hybrid Synthesis",
            status=StageStatus.COMPLETED,
            duration_ms=synthesis_time_ms
        )
    ]

    return formatter.format_orchestration_summary(stages)
