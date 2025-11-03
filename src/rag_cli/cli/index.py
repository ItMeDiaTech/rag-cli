#!/usr/bin/env python3
"""Document indexing script for RAG-CLI.

This script processes documents from a directory, generates embeddings,
and stores them in the FAISS vector index for retrieval.
"""

import sys
import time
from pathlib import Path
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from rag_cli.core.config import load_config, get_config
from rag_cli.core.document_processor import get_document_processor
from rag_cli.core.embeddings import get_embedding_generator
from rag_cli.core.vector_store import get_vector_store
from rag_cli.core.retrieval_pipeline import get_retriever
from rag_cli.core.duplicate_detector import DuplicateDetector
from rag_cli.utils.logger import get_logger

console = Console()
logger = get_logger(__name__)


@click.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--recursive', '-r', is_flag=True, help='Process subdirectories recursively')
@click.option('--pattern', '-p', help='File pattern to match (e.g., "*.md")')
@click.option('--chunk-size', type=int, help='Override chunk size (tokens)')
@click.option('--chunk-overlap', type=int, help='Override chunk overlap (tokens)')
@click.option('--clear', is_flag=True, help='Clear existing index before indexing')
@click.option('--incremental', '-i', is_flag=True, help='Skip unchanged documents (checks content hash)')
@click.option('--update', '-u', is_flag=True, help='Update existing documents (uses upsert instead of add)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def main(
    directory: str,
    recursive: bool,
    pattern: str,
    chunk_size: int,
    chunk_overlap: int,
    clear: bool,
    incremental: bool,
    update: bool,
    verbose: bool
):
    """Index documents from DIRECTORY into the RAG vector store.

    This command processes all supported documents in the specified directory,
    chunks them appropriately, generates embeddings, and stores them in the
    FAISS vector index for later retrieval.

    Example:
        rag-index ./docs --recursive --pattern "*.md"
    """
    # Load configuration
    config = load_config()

    # Header
    console.print(Panel.fit(
        "[bold cyan]RAG-CLI Document Indexer[/bold cyan]\n"
        f"Model: {config.embeddings.model_name}\n"
        f"Chunk Size: {chunk_size or config.document_processing.chunk_size} tokens",
        title="Configuration"
    ))

    # Initialize components
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Initializing components...", total=None)

        processor = get_document_processor()
        generator = get_embedding_generator()
        vector_store = get_vector_store()
        retriever = get_retriever()
        duplicate_detector = DuplicateDetector() if incremental or update else None

        progress.update(task, completed=100)

    # Clear index if requested
    if clear:
        console.print("[yellow]Clearing existing index...[/yellow]")
        vector_store.clear()
        retriever.clear_cache()

    # Process documents
    console.print(f"\n[cyan]Processing documents from:[/cyan] {directory}")
    console.print(f"[cyan]Recursive:[/cyan] {recursive}")
    if pattern:
        console.print(f"[cyan]Pattern:[/cyan] {pattern}")

    start_time = time.time()

    try:
        # Load and process documents
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            transient=False,
        ) as progress:
            # Process documents
            task1 = progress.add_task("Loading documents...", total=None)
            documents = processor.process_directory(directory, recursive, pattern)
            progress.update(task1, completed=100, description=f"Loaded {len(documents)} documents")

            if not documents:
                console.print("[red]No documents found to index![/red]")
                return

            # Filter duplicates if incremental mode
            documents_to_process = documents
            skipped_count = 0
            updated_sources = set()

            if duplicate_detector and incremental:
                task_filter = progress.add_task("Checking for duplicates...", total=len(documents))
                filtered_docs = []

                for doc in documents:
                    # Convert document to dict format for duplicate detector
                    doc_dict = {
                        'content': doc.content,
                        'title': doc.metadata.get('title', Path(doc.source).name),
                        'source': doc.source,
                        'doc_type': 'local'
                    }

                    is_dup, dup_info = duplicate_detector.is_duplicate(doc_dict['content'])

                    if not is_dup:
                        filtered_docs.append(doc)
                        # Add to hash registry for future checks
                        duplicate_detector.add_hash(
                            content=doc_dict['content'],
                            title=doc_dict['title'],
                            source=doc_dict['source'],
                            doc_type='local'
                        )
                    else:
                        skipped_count += 1
                        logger.debug(f"Skipping duplicate: {doc.source}")

                    progress.update(task_filter, advance=1)

                documents_to_process = filtered_docs
                console.print(f"[yellow]Skipped {skipped_count} duplicate documents[/yellow]")

            elif duplicate_detector and update:
                # In update mode, track sources that need updating
                for doc in documents:
                    doc_dict = {
                        'content': doc.content,
                        'title': doc.metadata.get('title', Path(doc.source).name),
                        'source': doc.source,
                        'doc_type': 'local'
                    }

                    is_dup, dup_info = duplicate_detector.is_duplicate(doc_dict['content'])

                    if is_dup:
                        # Mark for update (delete old version first)
                        updated_sources.add(doc.source)
                        console.print(f"[cyan]Will update: {Path(doc.source).name}[/cyan]")
                    else:
                        # Add new hash
                        duplicate_detector.add_hash(
                            content=doc_dict['content'],
                            title=doc_dict['title'],
                            source=doc_dict['source'],
                            doc_type='local'
                        )

            if not documents_to_process:
                console.print("[yellow]No new documents to index (all duplicates)[/yellow]")
                return

            # Chunk documents
            all_chunks = []
            task2 = progress.add_task("Chunking documents...", total=len(documents_to_process))

            for doc in documents_to_process:
                chunks = processor.chunk_document(
                    doc,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                all_chunks.extend(chunks)
                progress.update(task2, advance=1)

            console.print(f"\n[green]Created {len(all_chunks)} chunks from {len(documents_to_process)} documents[/green]")

            # Generate embeddings
            task3 = progress.add_task("Generating embeddings...", total=len(all_chunks))
            chunk_texts = [chunk.content for chunk in all_chunks]

            # Process in batches to show progress
            batch_size = 32
            all_embeddings = []

            for i in range(0, len(chunk_texts), batch_size):
                batch = chunk_texts[i:i + batch_size]
                embeddings = generator.encode(batch, show_progress=False, use_cache=False)
                all_embeddings.append(embeddings)
                progress.update(task3, advance=len(batch))

            # Combine embeddings
            import numpy as np
            final_embeddings = np.vstack(all_embeddings)

            # Store in vector database
            task4 = progress.add_task("Storing in vector index...", total=1)

            metadata_list = []
            sources = []
            for chunk in all_chunks:
                metadata_list.append(chunk.metadata)
                sources.append(chunk.source)

            # In update mode, delete old chunks from sources being updated
            if update and updated_sources:
                for source in updated_sources:
                    deleted = vector_store.delete_by_source(source)
                    if deleted > 0:
                        console.print(f"[yellow]Deleted {deleted} old chunks from {Path(source).name}[/yellow]")

            # Use upsert if update mode, otherwise add
            if update:
                ids = vector_store.upsert(
                    final_embeddings,
                    chunk_texts,
                    metadata=metadata_list,
                    sources=sources
                )
                console.print(f"[cyan]Upserted {len(ids)} chunks to vector store[/cyan]")
            else:
                ids = vector_store.add(
                    final_embeddings,
                    chunk_texts,
                    metadata=metadata_list,
                    sources=sources
                )

            progress.update(task4, completed=1)

            # Build BM25 index for hybrid retrieval
            task5 = progress.add_task("Building keyword index...", total=1)
            retriever.index_documents(all_chunks)
            progress.update(task5, completed=1)

        # Save the index and duplicate registry
        console.print("\n[cyan]Saving index to disk...[/cyan]")
        vector_store.save()

        if duplicate_detector:
            duplicate_detector.save()
            console.print("[cyan]Saved duplicate detection registry[/cyan]")

        elapsed = time.time() - start_time

        # Display summary
        stats_table = Table(title="Indexing Summary", show_header=True, header_style="bold magenta")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", justify="right", style="green")

        stats_table.add_row("Documents Processed", str(len(documents_to_process)))
        stats_table.add_row("Total Chunks", str(len(all_chunks)))
        stats_table.add_row("Average Chunks/Document", f"{len(all_chunks) / len(documents_to_process):.1f}")
        stats_table.add_row("Total Vectors", str(vector_store.collection.count()))
        if skipped_count > 0:
            stats_table.add_row("Skipped Duplicates", str(skipped_count))
        if updated_sources:
            stats_table.add_row("Updated Sources", str(len(updated_sources)))
        stats_table.add_row("Index Dimension", str(generator.get_embedding_dim()))
        stats_table.add_row("Processing Time", f"{elapsed:.2f} seconds")
        stats_table.add_row("Throughput", f"{len(all_chunks) / elapsed:.1f} chunks/second")

        console.print(stats_table)

        # Show indexed sources
        sources_dict = {}
        for chunk in all_chunks:
            source = Path(chunk.source).name
            sources_dict[source] = sources_dict.get(source, 0) + 1

        sources_table = Table(title="Indexed Sources", show_header=True, header_style="bold cyan")
        sources_table.add_column("File", style="yellow")
        sources_table.add_column("Chunks", justify="right", style="white")

        for source, count in sorted(sources_dict.items()):
            sources_table.add_row(source, str(count))

        console.print(sources_table)

        console.print("\n[bold green][OK] Indexing completed successfully![/bold green]")

        if verbose:
            # Show vector store statistics
            stats = vector_store.get_statistics()
            console.print(f"\n[dim]Vector store statistics:[/dim]")
            console.print(f"[dim]  Memory usage: {stats['memory_usage_bytes'] / 1024 / 1024:.2f} MB[/dim]")

    except Exception as e:
        console.print(f"\n[bold red][FAIL] Indexing failed:[/bold red] {str(e)}")
        logger.exception("Indexing failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
