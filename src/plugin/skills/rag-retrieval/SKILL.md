# RAG Retrieval Skill

Query your local document knowledge base using semantic search and get AI-powered answers.

## Overview

This skill enables RAG (Retrieval-Augmented Generation) queries against your locally indexed documents. It uses semantic search to find relevant documents and generates answers using Claude Haiku.

## Usage

```
/skill rag-retrieval "How to configure the API?"
```

## Features

- **Semantic Search**: Uses vector similarity to find relevant documents
- **Hybrid Retrieval**: Combines vector search with keyword matching for better accuracy
- **Context-Aware Answers**: Uses claude-haiku-4-5-20251001 to generate responses
- **Citation Support**: Shows sources for generated answers
- **Performance Monitoring**: Tracks query latency and accuracy

## Arguments

- `query` (required): Your question or search query
- `--top-k` (optional): Number of documents to retrieve (default: 5)
- `--threshold` (optional): Minimum similarity score (default: 0.7)
- `--mode` (optional): Search mode - "hybrid", "vector", or "keyword" (default: "hybrid")

## Examples

### Basic Query
```
/skill rag-retrieval "What is the authentication process?"
```

### Retrieve More Context
```
/skill rag-retrieval "How to handle errors?" --top-k 10
```

### Vector-Only Search
```
/skill rag-retrieval "API rate limits" --mode vector
```

## Configuration

The skill uses the following configuration from `config/default.yaml`:

- `retrieval.top_k`: Default number of documents to retrieve
- `retrieval.hybrid_ratio`: Balance between vector and keyword search (0.7 = 70% vector)
- `claude.model`: LLM model for response generation
- `claude.max_tokens`: Maximum response length

## Performance

Typical latencies:
- Vector search: <100ms
- End-to-end response: <5 seconds
- Indexing: ~0.5s per 100 documents

## Requirements

- Indexed documents in `data/vectors/`
- Valid Anthropic API key in environment
- At least 2GB RAM for vector operations

## Troubleshooting

### No Results Found
- Ensure documents are indexed: `python scripts/index.py --input data/documents`
- Lower the similarity threshold: `--threshold 0.5`
- Try keyword mode if vector search fails

### Slow Responses
- Reduce top_k value for faster retrieval
- Check if vector index is optimized for your document count
- Monitor memory usage with `python -m src.monitoring.tcp_server`

### API Errors
- Verify ANTHROPIC_API_KEY environment variable
- Check API rate limits and quota
- Review logs in `logs/rag_cli.log`