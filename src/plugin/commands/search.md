# /search - RAG Document Search

Search your indexed documents and get AI-powered answers.

## Usage

```
/search [query]
```

## Description

The `/search` command allows you to query your locally indexed document knowledge base using semantic search. It finds the most relevant documents and generates a comprehensive answer using Claude Haiku.

## Arguments

- `query` (required): Your question or search terms

## Examples

### Basic Search
```
/search how to configure API authentication
```

### Complex Questions
```
/search what are the best practices for error handling in Python
```

### Troubleshooting Queries
```
/search why is my database connection timing out
```

## Features

- **Semantic Understanding**: Understands the meaning behind your query, not just keywords
- **Context-Aware**: Uses surrounding document context for better answers
- **Source Citations**: Shows which documents were used to generate the answer
- **Fast Response**: Typically returns results in under 5 seconds

## Configuration

The command uses settings from your RAG-CLI configuration:

- Number of documents to retrieve (default: 5)
- Similarity threshold (default: 0.7)
- Hybrid search ratio (70% vector, 30% keyword)

## Requirements

Before using this command, ensure:

1. Documents are indexed in `data/vectors/`
2. ANTHROPIC_API_KEY is set in your environment
3. The monitoring server is running (optional but recommended)

To index documents:
```bash
python scripts/index.py --input data/documents
```

## Troubleshooting

### No Results
- Check if documents are indexed
- Try different keywords or phrasing
- Lower the similarity threshold in config

### Slow Performance
- Reduce the number of retrieved documents
- Ensure vector index is optimized
- Check system memory usage

### API Errors
- Verify your Anthropic API key
- Check rate limits and quotas
- Review logs for detailed errors