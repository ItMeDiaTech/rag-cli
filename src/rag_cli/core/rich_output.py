"""Rich-formatted output for RAG-CLI.

This module provides beautiful, formatted terminal output using the Rich library
for tables, progress bars, syntax highlighting, and structured displays.
"""

from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.tree import Tree
from rich import box

from rag_cli.core.output import Verbosity, get_output


class RichOutput:
    """Rich-formatted output manager."""

    def __init__(self):
        """Initialize Rich output."""
        self.console = Console()
        self.output = get_output()

    def print_retrieval_results(self, documents: List[Any], query: str, time_ms: float):
        """Print formatted retrieval results.

        Args:
            documents: Retrieved documents
            query: User query
            time_ms: Retrieval time in milliseconds
        """
        if self.output.verbosity == Verbosity.QUIET:
            return

        count = len(documents)

        # Normal mode: simple output
        if self.output.verbosity == Verbosity.NORMAL:
            self.console.print(f"[green][*][/green] Found {count} result{'s' if count != 1 else ''} ({time_ms:.0f}ms)")
            return

        # Verbose/Debug mode: table output
        table = Table(title=f"Retrieval Results for: '{query}'", box=box.ROUNDED)
        table.add_column("#", style="cyan", width=4)
        table.add_column("Source", style="magenta", width=30)
        table.add_column("Score", style="green", width=10)

        if self.output.verbosity >= Verbosity.DEBUG:
            table.add_column("Preview", style="white", width=60)

        for i, doc in enumerate(documents, 1):
            source = doc.source if hasattr(doc, 'source') else "Unknown"
            score = f"{doc.score:.3f}" if hasattr(doc, 'score') else "N/A"

            if self.output.verbosity >= Verbosity.DEBUG:
                preview = doc.text[:60] + "..." if hasattr(doc, 'text') and len(doc.text) > 60 else doc.text if hasattr(doc, 'text') else ""
                table.add_row(str(i), source, score, preview)
            else:
                table.add_row(str(i), source, score)

        self.console.print(table)
        self.console.print(f"[dim]Retrieval time: {time_ms:.2f}ms[/dim]")

    def print_query_enhancement(self, original_query: str, enhanced: bool, doc_count: int = 0):
        """Print query enhancement status.

        Args:
            original_query: Original query
            enhanced: Whether query was enhanced
            doc_count: Number of documents used for enhancement
        """
        if self.output.verbosity == Verbosity.QUIET:
            return

        if enhanced:
            if self.output.verbosity >= Verbosity.VERBOSE:
                self.console.print(Panel(
                    f"[green][*][/green] Query enhanced with {doc_count} document{'s' if doc_count != 1 else ''}\n"
                    f"[dim]Original:[/dim] {original_query[:80]}...",
                    title="RAG Enhancement",
                    border_style="green"
                ))
            else:
                self.console.print(f"[green][*][/green] Enhanced with {doc_count} doc{'s' if doc_count != 1 else ''}")
        else:
            if self.output.verbosity >= Verbosity.VERBOSE:
                self.console.print("[dim]RAG enhancement skipped[/dim]")

    def print_indexing_progress(self, total_docs: int) -> Progress:
        """Create and return a progress bar for indexing.

        Args:
            total_docs: Total number of documents to process

        Returns:
            Progress object
        """
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        )

        return progress

    def print_code_block(self, code: str, language: str = "python"):
        """Print syntax-highlighted code block.

        Args:
            code: Code to display
            language: Programming language for syntax highlighting
        """
        if self.output.verbosity == Verbosity.QUIET:
            return

        syntax = Syntax(code, language, theme="monokai", line_numbers=True)
        self.console.print(syntax)

    def print_document_tree(self, document_structure: Dict[str, Any]):
        """Print document structure as a tree.

        Args:
            document_structure: Dictionary representing document structure
        """
        if self.output.verbosity < Verbosity.VERBOSE:
            return

        tree = Tree("[bold]Document Collection[/bold]")

        for source, docs in document_structure.items():
            source_node = tree.add(f"[magenta]{source}[/magenta]")
            if isinstance(docs, list):
                for doc in docs:
                    if isinstance(doc, dict):
                        score = doc.get('score', 'N/A')
                        source_node.add(f"[green]Score: {score}[/green]")

        self.console.print(tree)

    def print_context_summary(self, context: str, max_length: int = 200):
        """Print formatted context summary.

        Args:
            context: Context string
            max_length: Maximum length to display
        """
        if self.output.verbosity == Verbosity.QUIET:
            return

        if len(context) > max_length:
            context_preview = context[:max_length] + "..."
        else:
            context_preview = context

        if self.output.verbosity >= Verbosity.VERBOSE:
            self.console.print(Panel(
                context_preview,
                title="Retrieved Context",
                border_style="blue"
            ))
        else:
            self.console.print(f"[blue]Context:[/blue] {len(context)} chars")

    def print_error(self, error_msg: str, details: Optional[str] = None):
        """Print formatted error message.

        Args:
            error_msg: Error message
            details: Optional error details
        """
        if details and self.output.verbosity >= Verbosity.VERBOSE:
            self.console.print(Panel(
                f"[red]{error_msg}[/red]\n\n[dim]{details}[/dim]",
                title="Error",
                border_style="red"
            ))
        else:
            self.console.print(f"[red][*] Error:[/red] {error_msg}")

    def print_warning(self, warning_msg: str):
        """Print formatted warning message.

        Args:
            warning_msg: Warning message
        """
        if self.output.verbosity >= Verbosity.NORMAL:
            self.console.print(f"[yellow][WARNING] Warning:[/yellow] {warning_msg}")

    def print_success(self, success_msg: str):
        """Print formatted success message.

        Args:
            success_msg: Success message
        """
        if self.output.verbosity >= Verbosity.NORMAL:
            self.console.print(f"[green][*][/green] {success_msg}")

    def print_service_status(self, services: Dict[str, Any]):
        """Print formatted service status table.

        Args:
            services: Dictionary of service statuses
        """
        if self.output.verbosity == Verbosity.QUIET:
            return

        table = Table(title="Service Status", box=box.SIMPLE)
        table.add_column("Service", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Port", style="magenta")
        table.add_column("URL", style="blue")

        for service_name, status in services.items():
            running = status.get('running', False)
            status_icon = "[green][/green] Running" if running else "[red][/red] Stopped"
            port = str(status.get('port', 'N/A'))
            url = status.get('url', '-')

            table.add_row(
                status.get('name', service_name),
                status_icon,
                port,
                url
            )

        self.console.print(table)

    def print_metrics_summary(self, metrics: Dict[str, Any]):
        """Print formatted metrics summary.

        Args:
            metrics: Dictionary of metrics
        """
        if self.output.verbosity < Verbosity.VERBOSE:
            return

        table = Table(title="Performance Metrics", box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for metric_name, value in metrics.items():
            # Format value based on type
            if isinstance(value, float):
                if 'time' in metric_name.lower() or 'latency' in metric_name.lower():
                    formatted_value = f"{value:.2f}ms"
                elif 'score' in metric_name.lower():
                    formatted_value = f"{value:.3f}"
                else:
                    formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)

            table.add_row(metric_name, formatted_value)

        self.console.print(table)

    def print_markdown(self, markdown_text: str):
        """Print formatted markdown.

        Args:
            markdown_text: Markdown text to display
        """
        if self.output.verbosity >= Verbosity.NORMAL:
            md = Markdown(markdown_text)
            self.console.print(md)


# Global instance
_rich_output: Optional[RichOutput] = None


def get_rich_output() -> RichOutput:
    """Get or create the global Rich output instance.

    Returns:
        RichOutput instance
    """
    global _rich_output
    if _rich_output is None:
        _rich_output = RichOutput()
    return _rich_output


# Convenience functions
def print_retrieval_results(documents: List[Any], query: str, time_ms: float):
    """Print formatted retrieval results."""
    get_rich_output().print_retrieval_results(documents, query, time_ms)


def print_query_enhancement(original_query: str, enhanced: bool, doc_count: int = 0):
    """Print query enhancement status."""
    get_rich_output().print_query_enhancement(original_query, enhanced, doc_count)


def print_service_status(services: Dict[str, Any]):
    """Print formatted service status."""
    get_rich_output().print_service_status(services)


def print_error(error_msg: str, details: Optional[str] = None):
    """Print formatted error."""
    get_rich_output().print_error(error_msg, details)


def print_success(success_msg: str):
    """Print formatted success message."""
    get_rich_output().print_success(success_msg)
