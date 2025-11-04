# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL: No Emojis Policy

NEVER use emojis in ANY documentation, plans, guides, or written output for this project UNLESS explicitly given permission. This includes:
- README files
- Documentation files
- Code comments and docstrings
- Plan descriptions
- Commit messages
- All text-based output

Focus on clear, professional documentation without decorative elements.

## Project Overview

RAG-CLI v2.0 is a local Retrieval-Augmented Generation system designed as a Claude Code plugin. It processes documents locally, generates embeddings, stores vectors in ChromaDB, and uses claude-haiku-4-5-20251001 for response generation.

This version features a complete restructure with clean separation between core library and plugin code, marketplace-ready lifecycle management, and improved maintainability.

## Project Structure (v2.0)

```
RAG-CLI/
 src/
    rag_cli/                          # CORE LIBRARY (plugin-agnostic)
       __init__.py                   # Version: 2.0.0
       core/                         # Core RAG functionality
          constants.py              # Centralized configuration
          document_processor.py     # Document chunking
          embeddings.py             # Embedding generation
          vector_store.py           # ChromaDB operations
          retrieval_pipeline.py     # Hybrid search + reranking
          claude_integration.py     # Claude API integration
          [30+ other core modules]
       agents/                       # Multi-agent framework
          base_agent.py             # Agent base class
          query_decomposer.py       # Query decomposition
          result_synthesizer.py     # Result synthesis
          maf/                      # Multi-Agent Framework
       integrations/                 # External integrations
          arxiv_connector.py        # ArXiv integration
          tavily_connector.py       # Tavily search
          maf_connector.py          # MAF integration
       cli/                          # Command-line tools
          index.py                  # rag-index command
          retrieve.py               # rag-retrieve command
       utils/                        # Shared utilities

    rag_cli_plugin/                   # PLUGIN CODE (Claude Code specific)
       __init__.py                   # Version: 2.0.0
       lifecycle/                    # Lifecycle management
          installer.py              # Marketplace installation
          updater.py                # Update handling
       commands/                     # Slash commands
          update_rag.py             # /update-rag command
          rag_project_indexer.py    # /rag-project command
          [other commands]
       hooks/                        # Event hooks
          user-prompt-submit.py     # Main RAG orchestration
          document-indexing.py      # Auto-indexing
          session-start.py          # Session initialization
          [other hooks]
       mcp/                          # MCP server
          unified_server.py         # Single unified MCP server
       services/                     # Plugin services
          service_manager.py        # Service registry
          dashboard.py              # Web dashboard
          tcp_server.py             # Monitoring server
          [monitoring modules]
       skills/                       # Agent skills

 config/                              # Configuration
    defaults/                        # Default configurations
       mcp.json                     # MCP server config
       rag_settings.json            # RAG settings
       services.json                # Service settings
       [other defaults]
    templates/                       # User-editable templates
       .env.template                # Environment template
       citation_config.json.template
    schemas/                         # JSON schemas
       settings.schema.json         # Settings validation

 scripts/                             # Scripts
    install/                         # Installation scripts
    update/                          # Update scripts
    utils/                           # Utility scripts
       update_imports_v2.py         # Import updater
       update_plugin_imports.py     # Plugin import updater

 .claude-plugin/                      # Plugin metadata
    plugin.json                     # Plugin configuration (v2.0.0)
    hooks.json                      # Hook configurations
    lifecycle.json                  # Lifecycle hooks (NEW)
    commands/                       # Command documentation

 data/                                # Runtime data
    vectors/                        # ChromaDB indexes
    cache/                          # Query cache
    documents/                      # Source documents

 logs/                                # Application logs
 tests/                               # Test suite
 docs/                                # Documentation

 pyproject.toml                       # Package configuration (v2.0.0)
 requirements.txt                     # Python dependencies
 README.md                            # Project README
 LICENSE                              # MIT License
 CHANGELOG.md                         # Version history
```

## Package Structure

RAG-CLI v2.0 uses a **dual-package src-layout** structure:

1. **Core Library (rag_cli)**: Platform-agnostic RAG engine
   - `from rag_cli.core.X import Y`
   - `from rag_cli.agents.X import Y`
   - `from rag_cli.integrations.X import Y`

2. **Plugin Code (rag_cli_plugin)**: Claude Code integration
   - `from rag_cli_plugin.services.X import Y`
   - `from rag_cli_plugin.lifecycle.X import Y`

This separation allows the core RAG engine to be used independently while keeping Claude Code-specific code isolated.

## Development Commands

### Initial Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode (development)
pip install -e .
```

### Plugin Installation
```bash
# Install as Claude Code plugin
python install_plugin.py

# This will:
# 1. Install RAG-CLI as Python package (pip install -e .)
# 2. Create plugin directory in ~/.claude/plugins/rag-cli/
# 3. Copy configuration files and commands
# 4. Set up data directory symlinks
# 5. Configure MCP server
```

### Command-Line Tools
```bash
# Index documents (after installation)
rag-index ./data/documents --recursive --pattern "*.md"

# Retrieve and generate responses
rag-retrieve --query "How to configure API?" --top-k 5

# Interactive retrieval mode
rag-retrieve --interactive

# Run monitoring server
rag-monitor
# Or: python -m monitoring

# Test installation
python scripts/verify_installation.py
```

### Testing
```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_vector_store.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run integration tests only
pytest tests/test_integration.py -v
```

## Core Implementation Details

### 0. Global Constants (core/constants.py)
Centralized configuration values for easier maintenance and tuning:
- **Cache Configuration**: `TCP_CHECK_CACHE_SECONDS`, `RESPONSE_CACHE_MAX_SIZE`, `EMBEDDING_CACHE_SIZE`
- **Token Estimation**: `CHARS_PER_TOKEN`, `TOKEN_ESTIMATION_RATIO`
- **Search Parameters**: `DEFAULT_TOP_K`, `MAX_TOP_K`, `MAX_QUERY_LENGTH`
- **Retrieval Weights**: `DEFAULT_VECTOR_WEIGHT` (0.7), `DEFAULT_KEYWORD_WEIGHT` (0.3)
- **File Processing**: `CHUNK_SIZE_TOKENS` (500), `CHUNK_OVERLAP_TOKENS` (100), `MAX_FILE_SIZE_MB`
- **Vector Store Thresholds**: `HNSW_THRESHOLD_VECTORS` (2000), `IVF_THRESHOLD_VECTORS` (1M)
- **Performance Tuning**: `DEFAULT_BATCH_SIZE` (32), `MAX_WORKERS` (4)
- **Monitoring Limits**: `MAX_EVENT_HISTORY`, `METRICS_HISTORY_SIZE`
- **API Limits**: `TAVILY_FREE_TIER_LIMIT`, `CLAUDE_RATE_LIMIT_REQUESTS`
- **Timeouts**: `DEFAULT_HTTP_TIMEOUT`, `EMBEDDING_TIMEOUT`, `SEARCH_TIMEOUT`

All magic numbers throughout the codebase should reference these constants for consistency and maintainability.

### 1. Document Processing (core/document_processor.py)
- Chunk size: `CHUNK_SIZE_TOKENS` (500 tokens)
- Overlap: `CHUNK_OVERLAP_TOKENS` (100 tokens, 20%)
- Use RecursiveCharacterTextSplitter from langchain
- Add contextual headers (document title, section)
- Support formats: MD, PDF, DOCX, HTML, TXT
- Max file size: `MAX_FILE_SIZE_MB` (10 MB)

### 2. Embeddings (core/embeddings.py)
- Model: sentence-transformers/all-MiniLM-L6-v2 (v5.1+)
- Dimensions: 384
- Batch processing: `DEFAULT_BATCH_SIZE` (32)
- LRU cache for repeated queries: `EMBEDDING_CACHE_SIZE` (1000)

### 3. Vector Store (core/vector_store.py)
- ChromaDB PersistentClient for <`HNSW_THRESHOLD_VECTORS` (2000 vectors)
- ChromaDB HNSW (built-in) for 2K-1M vectors (threshold: `HNSW_THRESHOLD_VECTORS`)
- ChromaDB IVF for >`IVF_THRESHOLD_VECTORS` (1M+ vectors)
- Native persistence (automatic save/load)
- Metadata storage with pickle

### 4. Retrieval Pipeline (core/retrieval_pipeline.py)
- Hybrid search: `DEFAULT_VECTOR_WEIGHT` (0.7) + `DEFAULT_KEYWORD_WEIGHT` (0.3)
- Default results: `DEFAULT_TOP_K` (5), max: `MAX_TOP_K` (100)
- Two-stage: retrieve 2×top_k, rerank to top_k
- Cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2
- Target latency: <100ms (configurable via `SEARCH_TIMEOUT`)

### 5. Claude Integration (core/claude_integration.py)
- Model: claude-haiku-4-5-20251001
- Stream responses for better UX
- Context assembly from retrieved chunks
- Prompt template with citations
- Response caching: `RESPONSE_CACHE_MAX_SIZE` (100)
- Rate limiting: `CLAUDE_RATE_LIMIT_REQUESTS` (100/min)

### 6. Monitoring (monitoring/tcp_server.py)
- TCP server on port 9999
- Endpoints: /status, /logs, /metrics
- JSON responses
- Real-time log streaming
- Event history: `MAX_EVENT_HISTORY` (100)
- Metrics retention: `METRICS_HISTORY_SIZE` (1000)
- Connection caching: `TCP_CHECK_CACHE_SECONDS` (30)

## ChromaDB Persistence and Document Update Best Practices (v2.0+)

This section documents the enhanced persistence and update strategies implemented in v2.0 to ensure reliable document management across sessions.

### Core Enhancements

#### 1. Automatic Persistence
- ChromaDB PersistentClient automatically saves all changes
- No manual save() or load() calls required
- Data persists to: `data/vectors/chroma_db/`
- Singleton pattern ensures single source of truth per session
- Graceful shutdown in session-end hook saves duplicate registry

#### 2. Upsert Functionality (`vector_store.py:270-361`)
Use `upsert()` instead of `add()` when re-indexing documents to prevent duplicates:

```python
# PREFERRED: Update existing or insert new
vector_store.upsert(
    embeddings=embeddings,
    texts=texts,
    ids=optional_ids,  # If None, auto-generates
    metadata=metadata,
    sources=sources
)

# OLD WAY: Always adds, creates duplicates on re-index
vector_store.add(embeddings, texts, metadata, sources)
```

**When to use upsert():**
- Re-indexing updated documents
- Correcting indexed content
- Maintaining document versions
- Any scenario where the same source may be indexed multiple times

#### 3. Source-Based Operations (`vector_store.py:492-612`)

**Get all vectors from a source:**
```python
# Returns List[VectorMetadata]
vectors = vector_store.get_by_source("path/to/document.md")
print(f"Found {len(vectors)} chunks from document")
```

**Delete all vectors from a source:**
```python
# Useful before re-indexing a modified file
deleted_count = vector_store.delete_by_source("path/to/document.md")
```

**Replace all vectors from a source:**
```python
# Combines delete + add in one operation
new_ids = vector_store.update_by_source(
    source="path/to/document.md",
    embeddings=new_embeddings,
    texts=new_texts,
    metadata=new_metadata
)
```

#### 4. Duplicate Detection Integration (`cli/index.py:124-181`)

The indexing pipeline now integrates content hash-based duplicate detection:

**Incremental Indexing (skip unchanged documents):**
```bash
rag-index ./docs --recursive --incremental
```

This mode:
- Computes SHA-256 hash of each document
- Skips documents with matching content hash
- Only processes new or changed documents
- Updates duplicate registry on completion

**Update Mode (replace changed documents):**
```bash
rag-index ./docs --recursive --update
```

This mode:
- Detects documents with changed content
- Deletes old chunks from changed sources
- Uses upsert() to add new chunks
- Maintains duplicate registry consistency

**Combine both modes:**
```bash
rag-index ./docs --recursive --incremental --update
```

#### 5. Metadata Validation (`vector_store.py:186-225`)

All metadata is validated before storage:
- Checks for reserved ChromaDB keys (id, embedding, document, etc.)
- Validates metadata is JSON-serializable
- Warns about large metadata (>10KB)
- Prevents invalid metadata from corrupting the index

#### 6. Session Lifecycle Management

**Session Start (`hooks/session-start.py:93-199`):**
- Performs comprehensive ChromaDB health check
- Verifies collection accessibility
- Tests query capability with peek()
- Reports vector count and status
- Non-blocking: allows session to continue even if issues detected

**Session End (`hooks/session-end.py:130-169`):**
- Saves duplicate detector registry
- Logs final vector count
- Allows ChromaDB auto-persistence to complete
- Cleans up temporary cache files
- Graceful shutdown prevents data loss

### Best Practices Summary

1. **Always use upsert() for re-indexing:**
   ```python
   # Good: Update existing entries
   vector_store.upsert(embeddings, texts, sources=sources)

   # Bad: Creates duplicates on re-index
   vector_store.add(embeddings, texts, sources=sources)
   ```

2. **Use source-based operations for document management:**
   ```python
   # Check if document is indexed
   existing = vector_store.get_by_source("doc.md")

   # Delete before re-indexing (if not using upsert)
   if existing:
       vector_store.delete_by_source("doc.md")

   # Or use update_by_source for atomic replace
   vector_store.update_by_source("doc.md", new_embeddings, new_texts)
   ```

3. **Use incremental indexing for large document sets:**
   ```bash
   # First time: full index
   rag-index ./docs --recursive

   # Subsequent updates: only changed documents
   rag-index ./docs --recursive --incremental --update
   ```

4. **Trust automatic persistence:**
   - No need to call save() or load()
   - ChromaDB handles persistence automatically
   - Session hooks ensure clean shutdown
   - Data survives across sessions

5. **Monitor duplicate registry:**
   - Location: `data/vectors/content_hashes.json`
   - Updated automatically during indexing
   - Cleaned up by session-end hook
   - Used for incremental indexing decisions

### Testing Persistence

Run the test suite to verify persistence and updates:
```bash
python test/test_chromadb_persistence.py
```

Tests verify:
- Automatic persistence across operations
- Upsert replaces existing vectors
- Source-based operations work correctly
- Duplicate detection prevents re-processing
- Health checks detect collection issues

### Troubleshooting

**Issue: Duplicates in index after re-indexing**
- Solution: Use `--update` flag or upsert() instead of add()

**Issue: "Collection does not exist" error**
- Solution: Normal on first run - collection created automatically
- If persistent: Check persist_directory permissions

**Issue: Vectors not persisting across sessions**
- Solution: Verify persist_directory exists and is writable
- Check session-end hook executed successfully
- Review logs for ChromaDB errors

**Issue: Incremental indexing still processes all documents**
- Solution: Ensure duplicate_detector registry exists
- Check if document content actually changed
- Verify content hashing is working (see logs)

## Import Guidelines (v2.0)

All imports MUST use the new dual-package structure:

```python
# CORRECT - Core library imports
from rag_cli.core.config import get_config
from rag_cli.core.embeddings import EmbeddingGenerator
from rag_cli.agents.base_agent import BaseAgent
from rag_cli.integrations.tavily_connector import TavilyConnector

# CORRECT - Plugin imports
from rag_cli_plugin.services.service_manager import ServiceManager
from rag_cli_plugin.lifecycle.installer import install_dependencies
from rag_cli_plugin.mcp.unified_server import MCPServer

# INCORRECT - Old v1.x imports (DO NOT USE)
from core.config import get_config
from monitoring.logger import get_logger
from plugin.mcp.unified_server import MCPServer
from src.core.config import get_config
```

The package is installed using pip with both `rag_cli` and `rag_cli_plugin` as top-level packages.

## Git Commit Checkpoints

Create commits at these milestones:
1. Project structure setup
2. Document processor implementation
3. Embedding system complete
4. Vector store operations
5. Retrieval pipeline working
6. Claude integration functional
7. Monitoring system online
8. Tests passing
9. Plugin components ready

Use conventional commits:
- `feature:` new functionality
- `fix:` bug fixes
- `refactor:` code improvements
- `test:` test additions
- `docs:` documentation

## Key Technical Decisions

1. **Local-first**: Everything runs locally except Claude API and online research calls
2. **Lightweight model**: all-MiniLM-L6-v2 for speed (0.5s/100 docs)
3. **ChromaDB for development**: Simple, fast, no persistence complexity
4. **Hybrid retrieval**: Better accuracy than pure vector search
5. **Streaming responses**: Improved perceived performance
6. **TCP monitoring**: PowerShell-friendly interface

## Testing Strategy

1. **Unit tests**: Each module in isolation
2. **Integration tests**: Component interactions
3. **Performance tests**: Latency and throughput
4. **Quality tests**: RAGAS metrics (precision, recall, faithfulness)

Target metrics:
- Vector search: <100ms
- End-to-end: <5 seconds
- Retrieval precision: >0.8
- Faithfulness: >0.7

## Implementation Checklist

- [ ] Create project structure
- [ ] Set up virtual environment
- [ ] Install dependencies
- [ ] Initialize git repository
- [ ] Implement document processor
- [ ] Build embedding system
- [ ] Create ChromaDB vector store
- [ ] Develop retrieval pipeline
- [ ] Integrate Claude API
- [ ] Add monitoring server
- [ ] Create PowerShell interface
- [ ] Write unit tests
- [ ] Add integration tests
- [ ] Create indexing script
- [ ] Build retrieval CLI
- [ ] Setup Claude Code plugin structure
- [ ] Test end-to-end pipeline
- [ ] Document usage examples

## Quick Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test embeddings
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("test query")
print(f"Embedding shape: {embedding.shape}")  # Should be (384,)

# Test ChromaDB
import faiss
index = faiss.PersistentClient(384)
index.add(embedding.reshape(1, -1))
D, I = index.search(embedding.reshape(1, -1), 1)
print(f"Search result: distance={D[0][0]}, index={I[0][0]}")  # Should be ~0, 0
```

## Known Issues and Limitations

### PostToolUse Hook Disabled (Claude Code Framework Bug)

The PostToolUse hook (`src/rag_cli_plugin/hooks/response-post.py`) is currently disabled due to a JSON parsing bug in the Claude Code plugin framework.

**Impact:**
- RAG functionality works normally
- Context retrieval and injection is unaffected
- Citations are not automatically added to responses

**Workaround:**
- Hook is disabled in `.claude-plugin/hooks.json` (line 40: `"enabled": false`)
- System remains stable and fully functional
- Users can manually request source information if needed

**Resolution:**
- Waiting for Claude Code framework update to fix JSON parsing
- Do not re-enable this hook until the framework bug is resolved
- See `KNOWN_ISSUES.md` for detailed information and testing instructions

**For Developers:**
- Do NOT modify the hook's enabled status without testing
- The hook code is functional in isolation (unit tests pass)
- Issue is specific to the plugin framework's PostToolUse processing
- Alternative citation methods can be explored via UserPromptSubmit hook

## References

- Full specifications: `RAG-implementation.md`
- Known issues and workarounds: `KNOWN_ISSUES.md`
- Claude Code plugin docs: https://docs.claude.com/en/docs/claude-code/
- ChromaDB documentation: https://github.com/facebookresearch/faiss/wiki