# RAG Project Indexing Command

You are a specialized assistant for indexing project-relevant documentation into the RAG system.

## Task

Analyze the current project and automatically fetch and index relevant documentation:

1. **Detect Project Stack**:
   - Programming languages used (Python, JavaScript, TypeScript, etc.)
   - Frameworks and libraries (from package files)
   - Development tools and build systems

2. **Search for Documentation**:
   - Official language documentation
   - Framework documentation
   - Best practices and examples
   - API references

3. **Index Documentation**:
   - Download relevant docs
   - Process and chunk them
   - Index in vector store
   - Report results

## Implementation

Execute the project indexing script:

```bash
python src/plugin/commands/rag_project_indexer.py
```

## Output

Provide a summary of:
- Detected languages and frameworks
- Documentation sources found
- Number of documents indexed
- Total processing time
- Next steps for querying the indexed content

## Notes

- This command analyzes package files (requirements.txt, package.json, etc.)
- It prioritizes official documentation sources
- Large documentation sets may take several minutes to process
- The indexed content will be immediately available for RAG queries
