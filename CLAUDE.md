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

RAG-CLI v2.0 is a local Retrieval-Augmented Generation system designed as a Claude Code plugin. It processes documents locally, generates embeddings, stores vectors in FAISS, and uses claude-haiku-4-5-20251001 for response generation.

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
          vector_store.py           # FAISS operations
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
    vectors/                        # FAISS indexes
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
- Model: sentence-transformers/all-MiniLM-L6-v2
- Dimensions: 384
- Batch processing: `DEFAULT_BATCH_SIZE` (32)
- LRU cache for repeated queries: `EMBEDDING_CACHE_SIZE` (1000)

### 3. Vector Store (core/vector_store.py)
- FAISS IndexFlatL2 for <`HNSW_THRESHOLD_VECTORS` (2000 vectors)
- FAISS IndexHNSWFlat for 2K-1M vectors (threshold: `HNSW_THRESHOLD_VECTORS`)
- FAISS IVF for >`IVF_THRESHOLD_VECTORS` (1M+ vectors)
- Save/load functionality with automatic metadata_dict rebuilding
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
3. **FAISS for development**: Simple, fast, no persistence complexity
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
- [ ] Create FAISS vector store
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

# Test FAISS
import faiss
index = faiss.IndexFlatL2(384)
index.add(embedding.reshape(1, -1))
D, I = index.search(embedding.reshape(1, -1), 1)
print(f"Search result: distance={D[0][0]}, index={I[0][0]}")  # Should be ~0, 0
```

## References

- Full specifications: `RAG-implementation.md`
- Claude Code plugin docs: https://docs.claude.com/en/docs/claude-code/
- FAISS documentation: https://github.com/facebookresearch/faiss/wiki