# /rag:enable - Enable RAG Enhancement

Enable automatic RAG enhancement for all your queries.

## Usage

```
/rag:enable
```

## Description

The `/rag:enable` command activates automatic RAG (Retrieval-Augmented Generation) for all your interactions with Claude Code. When enabled, your queries will automatically search the document knowledge base and include relevant context in responses.

## How It Works

When RAG is enabled:

1. **Query Interception**: Your queries are automatically analyzed
2. **Context Retrieval**: Relevant documents are retrieved from your knowledge base
3. **Enhanced Response**: Claude uses both its training and your documents to answer
4. **Seamless Integration**: Works transparently without changing how you interact

## Benefits

- **More Accurate Answers**: Responses are grounded in your actual documentation
- **Project-Specific Knowledge**: Uses your codebase and documentation
- **Reduced Hallucinations**: Answers are based on real documents, not assumptions
- **Automatic Context**: No need to manually provide context for every query

## Example Workflow

```bash
# Enable RAG enhancement
/rag:enable

# Now all your queries automatically use RAG
How do I authenticate users in this project?
# Claude will search your docs and provide project-specific answers

What's the database schema for products?
# Claude will find and reference your actual schema documentation
```

## Configuration

RAG enhancement uses these settings:

- **Auto-trigger threshold**: Minimum query length to activate (default: 5 words)
- **Context limit**: Maximum documents to include (default: 3)
- **Relevance threshold**: Minimum similarity score (default: 0.6)

## Performance Impact

- Adds ~1-2 seconds to response time
- Uses local vector search (no external API calls for retrieval)
- Caches frequent queries for faster responses

## Related Commands

- `/rag:disable` - Disable automatic RAG enhancement
- `/rag:status` - Check if RAG is enabled
- `/search` - Manually search documents (works regardless of auto-RAG setting)

## Notes

- RAG enhancement persists across sessions
- Settings are stored in `config/rag_settings.json`
- You can still use `/search` for explicit document queries even when auto-RAG is disabled