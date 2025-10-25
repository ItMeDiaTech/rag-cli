# /rag:disable - Disable RAG Enhancement

Disable automatic RAG enhancement for your queries.

## Usage

```
/rag:disable
```

## Description

The `/rag:disable` command deactivates automatic RAG (Retrieval-Augmented Generation) enhancement. After running this command, your queries will be handled by Claude Code's standard processing without automatic document retrieval.

## When to Disable RAG

Consider disabling RAG when:

- **General Programming Questions**: Asking about general concepts not specific to your project
- **Performance Critical**: Need fastest possible response times
- **External Topics**: Discussing topics unrelated to your indexed documents
- **Debugging RAG**: Troubleshooting issues with document retrieval

## What Changes

With RAG disabled:

- Queries go directly to Claude without document search
- Response times are faster (no retrieval overhead)
- Answers are based on Claude's training, not your documents
- You can still use `/search` for manual document queries

## Example Workflow

```bash
# Disable RAG for general questions
/rag:disable

# Ask general programming questions
What's the difference between TCP and UDP?
# Claude answers from general knowledge

# Re-enable for project-specific work
/rag:enable

# Now queries use your documentation again
How does our authentication system work?
# Claude searches your docs and provides specific answers
```

## Verification

To check if RAG is currently enabled or disabled:
```bash
/rag:status
```

## Manual Search

Even with auto-RAG disabled, you can still manually search documents:
```bash
/search how to configure the database
```

## Settings Persistence

- The disabled state persists across sessions
- Settings are saved in `config/rag_settings.json`
- Can be changed at any time with `/rag:enable`

## Related Commands

- `/rag:enable` - Enable automatic RAG enhancement
- `/rag:status` - Check current RAG status
- `/search` - Manually search documents

## Notes

- Disabling RAG doesn't delete your indexed documents
- Your vector store remains available for manual searches
- The setting is user-specific and doesn't affect other users