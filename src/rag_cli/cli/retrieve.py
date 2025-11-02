#!/usr/bin/env python3
"""Document retrieval and Q&A script for RAG-CLI.

This script performs retrieval-augmented generation to answer questions
using the indexed document collection.
"""

import sys
import io
import time
from pathlib import Path
import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn

from rag_cli.core.config import load_config, get_config
from rag_cli.core.embeddings import get_embedding_generator
from rag_cli.core.vector_store import get_vector_store
from rag_cli.core.retrieval_pipeline import get_retriever
from rag_cli.core.claude_integration import get_claude_integration
from rag_cli_plugin.services.logger import get_logger

# Fix Windows console encoding issues
if sys.platform == 'win32':
    # Reconfigure stdout/stderr to use UTF-8 encoding
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

console = Console(force_terminal=True, legacy_windows=False)
logger = get_logger(__name__)


def format_response(response_text: str, sources: list, show_sources: bool = True) -> str:
    """Format the response with proper markdown and sources.

    Args:
        response_text: The response text
        sources: List of source citations
        show_sources: Whether to show sources

    Returns:
        Formatted response
    """
    formatted = response_text

    if show_sources and sources:
        formatted += "\n\n---\n### Sources:\n"
        for source in sources:
            formatted += f"- {source}\n"

    return formatted


@click.command()
@click.option('--query', '-q', help='Query to search for (interactive if not provided)')
@click.option('--top-k', '-k', type=int, default=5, help='Number of chunks to retrieve')
@click.option('--no-generate', is_flag=True, help='Only retrieve, don\'t generate response')
@click.option('--show-chunks', is_flag=True, help='Display retrieved chunks')
@click.option('--no-sources', is_flag=True, help='Hide source citations')
@click.option('--stream', is_flag=True, help='Stream the response')
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode (multiple queries)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def main(
    query: str,
    top_k: int,
    no_generate: bool,
    show_chunks: bool,
    no_sources: bool,
    stream: bool,
    interactive: bool,
    verbose: bool
):
    """Retrieve relevant documents and generate responses using RAG.

    This command searches the indexed documents for relevant information
    and optionally generates a response using Claude.

    Examples:
        rag-retrieve --query "What is RAG?"
        rag-retrieve --interactive
        rag-retrieve -q "How does it work?" --show-chunks
    """
    # Load configuration
    config = load_config()

    # Header
    console.print(Panel.fit(
        "[bold cyan]RAG-CLI Retrieval System[/bold cyan]\n"
        f"Retrieval: Hybrid (Vector + BM25)\n"
        f"Generation: {config.claude.model if not no_generate else 'Disabled'}",
        title="Configuration"
    ))

    # Initialize components
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Initializing components...", total=None)

        generator = get_embedding_generator()
        vector_store = get_vector_store()
        retriever = get_retriever()

        if not no_generate:
            claude = get_claude_integration()
        else:
            claude = None

        progress.update(task, completed=100)

    # Check if index exists
    if vector_store.index.ntotal == 0:
        console.print("[red]No documents indexed! Please run rag-index first.[/red]")
        sys.exit(1)

    console.print(f"[green]Index loaded: {vector_store.index.ntotal} vectors[/green]\n")

    def process_query(q: str):
        """Process a single query."""
        start_time = time.time()

        # Retrieve relevant chunks
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Retrieving relevant documents...", total=None)

            try:
                results = retriever.retrieve(q, top_k=top_k)
                retrieval_time = time.time() - start_time
                progress.update(task, completed=100)
            except Exception as e:
                console.print(f"[red]Retrieval failed: {e}[/red]")
                return

        if not results:
            console.print("[yellow]No relevant documents found.[/yellow]")
            return

        # Display retrieved chunks if requested
        if show_chunks:
            console.print("\n[cyan]Retrieved Chunks:[/cyan]")
            for i, result in enumerate(results, 1):
                chunk_panel = Panel(
                    result.text[:500] + ("..." if len(result.text) > 500 else ""),
                    title=f"[{i}] {Path(result.source).name} (Score: {result.score:.3f})",
                    border_style="dim"
                )
                console.print(chunk_panel)

        # Generate response if requested
        if not no_generate and claude:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task("Generating response with Claude...", total=None)

                try:
                    response = claude.generate_response(
                        q,
                        results,
                        stream=stream,
                        use_cache=True
                    )
                    generation_time = time.time() - start_time - retrieval_time
                    progress.update(task, completed=100)

                    # Format and display response
                    formatted_response = format_response(
                        response.answer,
                        response.sources,
                        show_sources=not no_sources
                    )

                    console.print("\n[green]Response:[/green]")
                    console.print(Markdown(formatted_response))

                    # Display metrics
                    if verbose:
                        metrics_table = Table(show_header=False)
                        metrics_table.add_column("Metric", style="cyan")
                        metrics_table.add_column("Value", justify="right", style="white")

                        metrics_table.add_row("Retrieval Time", f"{retrieval_time:.2f}s")
                        metrics_table.add_row("Generation Time", f"{generation_time:.2f}s")
                        metrics_table.add_row("Total Time", f"{response.latency_seconds:.2f}s")
                        metrics_table.add_row("Chunks Retrieved", str(len(results)))
                        metrics_table.add_row("Input Tokens", str(response.token_usage.get('input', 0)))
                        metrics_table.add_row("Output Tokens", str(response.token_usage.get('output', 0)))

                        console.print("\n")
                        console.print(metrics_table)

                except Exception as e:
                    console.print(f"[red]Generation failed: {e}[/red]")
                    logger.exception("Generation failed", error=str(e))
        else:
            # Just show retrieval results summary
            console.print(f"\n[green]Retrieved {len(results)} relevant chunks in {retrieval_time:.2f}s[/green]")

            sources = list(set(Path(r.source).name for r in results))
            console.print("\n[cyan]Sources:[/cyan]")
            for source in sources:
                console.print(f"  * {source}")

    # Handle interactive or single query mode
    if interactive or (not query):
        console.print("[yellow]Interactive mode. Type 'quit' or 'exit' to stop.[/yellow]\n")

        while True:
            try:
                q = console.input("[bold cyan]Query:[/bold cyan] ")

                if q.lower() in ['quit', 'exit', 'q']:
                    console.print("[yellow]Goodbye![/yellow]")
                    break

                if q.strip():
                    process_query(q)
                    console.print("\n" + "="*60 + "\n")

            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                logger.exception("Query processing failed", error=str(e))

    else:
        # Single query mode
        console.print(f"[cyan]Query:[/cyan] {query}\n")
        process_query(query)

    # Show usage statistics if using Claude
    if not no_generate and claude and verbose:
        stats = claude.get_usage_stats()
        console.print("\n[dim]Session Statistics:[/dim]")
        console.print(f"[dim]  Total tokens used: {stats['total_tokens']}[/dim]")
        console.print(f"[dim]  Estimated cost: ${stats['total_cost']:.4f}[/dim]")
        console.print(f"[dim]  Cache hit rate: {stats['cache_hit_rate']:.1%}[/dim]")


if __name__ == "__main__":
    main()
