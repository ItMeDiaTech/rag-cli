# RAG-CLI v2.0

A production-ready local Retrieval-Augmented Generation (RAG) system designed as a Claude Code plugin. RAG-CLI processes documents locally, generates embeddings, stores vectors in FAISS, and provides AI-powered responses using Claude Haiku with intelligent context retrieval.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-2.0.0-green.svg)](https://github.com/ItMeDiaTech/rag-cli/releases)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Goals](#project-goals)
- [Project Scope](#project-scope)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

RAG-CLI is a **local-first Retrieval-Augmented Generation system** that brings the power of semantic search and contextual AI assistance directly into your development workflow. Unlike cloud-based solutions, RAG-CLI processes all your documents locally, ensuring complete privacy and eliminating API costs for document indexing.

### What is RAG-CLI?

RAG-CLI combines three core technologies:
1. **Document Processing**: Intelligent chunking and extraction from multiple formats (Markdown, PDF, DOCX, HTML, TXT)
2. **Vector Search**: Lightning-fast semantic similarity search using FAISS with hybrid ranking
3. **AI Enhancement**: Context-aware responses from Claude AI using only relevant document chunks

### Why RAG-CLI?

**Privacy-First**: Your codebase and documentation never leave your machine during indexing. Only final queries go to Claude AI.

**Cost-Effective**: Zero API costs for document processing. Pay only for Claude AI responses, not for indexing thousands of documents.

**Developer-Focused**: Built specifically for development workflows with Claude Code integration, supporting technical documentation, code context, and real-time project indexing.

**Performance**: Sub-100ms vector search, sub-5s end-to-end responses, and intelligent caching ensure minimal impact on your workflow.

---

## Project Goals

### Primary Goals

1. **Enable Local Document Intelligence**
   - Process and index documentation without external API calls
   - Provide instant access to relevant context from large document sets
   - Support real-time updates as documentation changes

2. **Seamless Claude Code Integration**
   - Native plugin architecture for zero-configuration setup
   - Automatic context enhancement for user queries
   - Slash commands for manual control and project indexing

3. **Production-Grade Performance**
   - Vector search under 100ms for instant results
   - End-to-end query processing under 5 seconds
   - Efficient memory usage supporting 100K+ document chunks

4. **Developer Experience**
   - Simple installation via Claude Code marketplace
   - Minimal configuration required for common use cases
   - Clear documentation and troubleshooting guides

### Secondary Goals

1. **Multi-Agent Orchestration**: Intelligent routing between specialized agents for complex queries
2. **Online Retrieval**: Integration with ArXiv, Tavily, and web search for up-to-date information
3. **Extensibility**: Plugin architecture supporting custom data sources and retrieval strategies
4. **Monitoring**: Real-time observability with TCP server and web dashboard

---

## Project Scope

### In Scope

#### Core Functionality
- Local document processing and chunking (400-500 tokens per chunk)
- Embedding generation using sentence-transformers/all-MiniLM-L6-v2
- Vector storage and retrieval using FAISS (IndexFlatL2, IndexHNSWFlat, IndexIVF)
- Hybrid search combining vector similarity (0.7) and keyword matching (0.3)
- Two-stage retrieval with cross-encoder reranking
- Claude API integration (claude-haiku-4-5-20251001) for response generation

#### Document Support
- Markdown (.md)
- PDF documents (.pdf)
- Microsoft Word (.docx)
- HTML files (.html)
- Plain text (.txt)
- Code files (via text extraction)

#### Integration Features
- Claude Code plugin with lifecycle hooks (installation, updates)
- Slash commands: /rag-project, /update-rag, /rag-enable, /rag-disable
- Event hooks: UserPromptSubmit, ResponsePost, SessionStart, SessionEnd
- MCP (Model Context Protocol) server with 14+ tools
- File watching with automatic re-indexing

#### Advanced Features
- Multi-Agent Framework (MAF) with 7 specialized agents
- Query classification (10 intent types)
- HyDE (Hypothetical Document Embeddings) optimization
- Semantic caching for repeated queries
- Online retrieval (ArXiv, Tavily, web scraping)
- Duplicate detection and deduplication

#### Monitoring & Observability
- TCP server (port 9999) with REST API
- Web dashboard for real-time monitoring
- Comprehensive logging with rotating file handlers
- Performance metrics tracking (latency, throughput, cache hit rates)
- Error tracking with automatic recovery

### Out of Scope

#### Explicitly Excluded
- **Cloud Storage**: No cloud-based vector databases (Pinecone, Weaviate, etc.)
- **User Authentication**: Single-user local system only
- **Distributed Processing**: No multi-machine indexing or search
- **Custom LLMs**: Only Claude API supported (no OpenAI, local models, etc.)
- **GUI Application**: Command-line and plugin interface only
- **Real-Time Collaboration**: Single-user workflow only
- **Document Editing**: Read-only document processing
- **Version Control Integration**: No automatic git tracking (manual indexing only)

#### Future Considerations (Not in v2.0)
- Support for additional LLM providers (OpenAI, Gemini, local models)
- Distributed vector search for enterprise deployments
- Built-in document version tracking
- Advanced query language (filters, facets, date ranges)
- Vector database migrations (FAISS to Qdrant, ChromaDB, etc.)
- Multi-user support with access control

### Technical Boundaries

**Supported Platforms**:
- Windows 10+
- macOS 10.14+
- Linux (Ubuntu 18.04+, Debian 10+, Fedora 30+)

**Python Versions**:
- Python 3.8 - 3.13 (tested)
- No Python 2.x support

**Resource Limits**:
- Max file size: 10MB per document
- Recommended max chunks: 1,000,000 (scales with RAM)
- Embedding dimensions: 384 (fixed by model)
- Max concurrent queries: Limited by Python GIL (single-threaded)

---

## Architecture

### Dual-Package Structure

RAG-CLI v2.0 uses a dual-package architecture for clean separation of concerns:

```
RAG-CLI/
├── src/
│   ├── rag_cli/              # Core Library (platform-agnostic)
│   │   ├── core/             # RAG engine
│   │   ├── agents/           # Multi-agent framework
│   │   ├── integrations/     # External services
│   │   ├── cli/              # Command-line tools
│   │   └── utils/            # Shared utilities
│   │
│   └── rag_cli_plugin/       # Plugin Code (Claude Code specific)
│       ├── lifecycle/        # Installation & updates
│       ├── commands/         # Slash commands
│       ├── hooks/            # Event handlers
│       ├── mcp/              # MCP server
│       └── services/         # Monitoring & dashboard
│
├── config/
│   ├── defaults/             # Default configurations
│   ├── templates/            # User-editable templates
│   └── schemas/              # JSON schemas
│
├── scripts/                  # Installation & utilities
├── tests/                    # Test suite
├── data/                     # Runtime data (vectors, cache)
└── docs/                     # Documentation
```

### Core Library (rag_cli)

**Platform-agnostic RAG engine** - Can be used independently of Claude Code

**Key Components**:
- `core/vector_store.py` - FAISS operations with auto-scaling indexes
- `core/embeddings.py` - Embedding generation with caching
- `core/retrieval_pipeline.py` - Hybrid search with reranking
- `core/document_processor.py` - Multi-format document processing
- `core/claude_integration.py` - Claude API client with streaming

**Import Pattern**:
```python
from rag_cli.core.embeddings import EmbeddingGenerator
from rag_cli.core.vector_store import VectorStore
from rag_cli.agents.query_decomposer import QueryDecomposer
```

### Plugin Code (rag_cli_plugin)

**Claude Code integration layer** - Bridges core library with Claude Code

**Key Components**:
- `lifecycle/installer.py` - Automated marketplace installation
- `lifecycle/updater.py` - Configuration-preserving updates
- `hooks/user-prompt-submit.py` - Main RAG orchestration (24KB)
- `mcp/unified_server.py` - MCP server with 14 tools
- `services/` - Monitoring, dashboard, service registry

**Import Pattern**:
```python
from rag_cli_plugin.lifecycle.installer import install_dependencies
from rag_cli_plugin.services.service_manager import ServiceManager
```

### Data Flow

```
User Query (Claude Code)
    ↓
UserPromptSubmit Hook
    ↓
Query Classification → Intent Detection
    ↓
┌─────────────────────────────────┐
│  Parallel Processing            │
│  ├─ RAG Retrieval               │
│  │   ├─ HyDE Optimization       │
│  │   ├─ Vector Search (FAISS)   │
│  │   ├─ Keyword Matching        │
│  │   ├─ Hybrid Ranking (0.7+0.3)│
│  │   └─ Cross-Encoder Reranking │
│  │                               │
│  └─ Multi-Agent Framework (MAF) │
│      ├─ Agent Selection          │
│      ├─ Task Decomposition       │
│      └─ Result Synthesis         │
└─────────────────────────────────┘
    ↓
Context Assembly
    ↓
Claude API Call (with context)
    ↓
Response Enhancement
    ↓
Citation Injection (ResponsePost Hook)
    ↓
User Sees Enhanced Response
```

---

## Features

### Core Features

**Local Document Processing**
- Intelligent chunking with 20% overlap for context preservation
- Semantic boundary detection (paragraph, sentence, code block)
- Metadata extraction (title, section, source)
- Support for 5+ document formats

**Vector Search**
- Sub-100ms search latency (IndexFlatL2 for <2K vectors)
- Auto-scaling indexes (HNSW for 2K-1M, IVF for 1M+)
- Hybrid ranking: 70% semantic + 30% keyword
- Two-stage retrieval with cross-encoder reranking

**Claude Integration**
- Streaming responses for better UX
- Context assembly with citation tracking
- Rate limiting (100 requests/min)
- Response caching (100 most recent)

### Advanced Features

**Multi-Agent Framework**
- **Architect Agent**: System design and planning
- **Developer Agent**: Code generation
- **Reviewer Agent**: Code quality analysis
- **Tester Agent**: Test creation
- **Debugger Agent**: Error diagnosis
- **Documenter Agent**: Documentation generation
- **Optimizer Agent**: Performance tuning

**Query Optimization**
- HyDE: Generate hypothetical documents for better retrieval
- Query classification: 10 intent types (factual, code, debug, etc.)
- Semantic caching: 22% hit rate improvement
- Query decomposition for complex questions

**Online Retrieval**
- ArXiv: Academic paper search (free, rate-limited)
- Tavily: AI-optimized web search (free tier with quota tracking)
- Web scraping: Direct URL content extraction
- Automatic deduplication across sources

### Monitoring & Observability

**TCP Server (port 9999)**
- REST API: /status, /logs, /metrics, /health
- JSON responses for easy parsing
- Real-time log streaming
- Event history (last 100 events)

**Web Dashboard**
- Real-time performance metrics
- Query latency histograms
- Cache hit rate tracking
- Error rate monitoring
- Service status indicators

**Logging**
- Rotating file handlers (10MB max, 5 backups)
- Structured logging with context
- Multiple log levels (DEBUG, INFO, WARNING, ERROR)
- Per-component loggers

---

## Installation

### Prerequisites

- Python 3.8 or higher (3.13 recommended)
- 4GB RAM minimum (8GB recommended for large document sets)
- 2GB disk space for dependencies
- Claude Code (latest version for plugin mode)

### Method 1: Claude Code Marketplace (Recommended)

**Automated installation with zero configuration:**

1. Open Claude Code
2. Run marketplace installation:
   ```
   /plugin add rag-cli
   ```
3. Restart Claude Code
4. Verify installation:
   ```
   /rag-project --help
   ```

The marketplace installer automatically:
- Installs all Python dependencies
- Initializes configuration files
- Creates data directories
- Verifies the installation

### Method 2: Manual Installation

**For development or custom setups:**

```bash
# Clone repository
git clone https://github.com/ItMeDiaTech/rag-cli.git
cd rag-cli

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install package in editable mode
pip install -e .

# Run verification
python -c "from rag_cli import __version__; print(f'RAG-CLI v{__version__}')"

# Install as Claude Code plugin (optional)
python -m rag_cli_plugin.lifecycle.installer
```

### Method 3: Development Installation

**For contributors:**

```bash
# Clone and setup development environment
git clone https://github.com/ItMeDiaTech/rag-cli.git
cd rag-cli

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

---

## Quick Start

### 1. Index Your First Project

```bash
# Using slash command in Claude Code
/rag-project /path/to/your/project

# Or using CLI
rag-index ./docs --recursive --pattern "*.md"
```

### 2. Ask Questions

Just ask questions naturally in Claude Code - RAG will automatically enhance responses with relevant context from your indexed documents.

Example:
```
User: How do I configure the API timeout?
Claude: [Enhanced with context from your config documentation]
Based on your project documentation, you can configure the API
timeout in config/settings.json...
```

### 3. Enable/Disable RAG

```bash
# Enable RAG enhancement
/rag-enable

# Disable RAG enhancement
/rag-disable

# Check status
/rag-status
```

### 4. Update RAG-CLI

```bash
# Update to latest version
/update-rag
```

---

## Usage

### Slash Commands

**Project Indexing**
```bash
/rag-project <path>              # Index a project directory
/rag-project ./docs --pattern "*.md" --recursive
```

**RAG Control**
```bash
/rag-enable                      # Enable RAG enhancement
/rag-disable                     # Disable RAG enhancement
/rag-status                      # Show current status
```

**Configuration**
```bash
/rag-maf-config                  # Configure Multi-Agent Framework
/rag-maf-config --enable         # Enable MAF
/rag-maf-config --disable        # Disable MAF
```

**Updates**
```bash
/update-rag                      # Update to latest version
```

### CLI Commands

**Indexing**
```bash
# Index documents
rag-index ./docs --recursive --pattern "*.md"
rag-index ./src --pattern "*.py" --exclude "*test*"

# Index with custom settings
rag-index ./docs --chunk-size 500 --overlap 100
```

**Retrieval**
```bash
# Search documents
rag-retrieve --query "API configuration" --top-k 5

# Interactive mode
rag-retrieve --interactive

# With filtering
rag-retrieve --query "setup" --source "docs/*.md"
```

**Monitoring**
```bash
# Start monitoring server
rag-monitor

# Or with Python module
python -m rag_cli_plugin.services
```

### Python API

**Core Library Usage**
```python
from rag_cli.core.vector_store import VectorStore
from rag_cli.core.embeddings import EmbeddingGenerator
from rag_cli.core.retrieval_pipeline import RetrievalPipeline

# Initialize components
embeddings = EmbeddingGenerator()
vector_store = VectorStore(embeddings)

# Index documents
documents = ["Document 1 content", "Document 2 content"]
vector_store.add_documents(documents)

# Search
pipeline = RetrievalPipeline(vector_store)
results = pipeline.search("your query", top_k=5)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Content: {result.content}\n")
```

**Plugin Integration**
```python
from rag_cli_plugin.services.service_manager import ServiceManager
from rag_cli.core.retrieval_pipeline import RetrievalPipeline

# In a Claude Code hook
def enhance_query(query: str) -> str:
    # Use core library for retrieval
    pipeline = RetrievalPipeline()
    results = pipeline.search(query, top_k=3)

    # Format for Claude
    context = "\n".join([r.content for r in results])
    return f"Context:\n{context}\n\nQuery: {query}"
```

---

## Configuration

### Default Configuration

Configuration files are in `config/`:

**config/defaults/rag_settings.json**
```json
{
  "enabled": false,
  "auto_trigger_threshold": 5,
  "context_limit": 3,
  "relevance_threshold": 0.6,
  "enable_agent_orchestration": true,
  "vector_weight": 0.7,
  "keyword_weight": 0.3,
  "enable_hyde_optimization": true,
  "enable_semantic_caching": true
}
```

**config/defaults/services.json**
```json
{
  "monitoring_server": {
    "enabled": true,
    "host": "localhost",
    "port": 9999,
    "auto_start": false
  },
  "web_dashboard": {
    "enabled": false,
    "host": "localhost",
    "port": 8080
  }
}
```

### Environment Variables

Create `.env` from template:
```bash
cp config/templates/.env.template .env
```

**Key Variables**:
```bash
# Required for standalone mode
ANTHROPIC_API_KEY=your_key_here

# Optional
TAVILY_API_KEY=your_tavily_key
ARXIV_MAX_RESULTS=10
RAG_TOP_K=5
RAG_VECTOR_WEIGHT=0.7
```

### Customization

**Adjust chunk size:**
```python
# In config or environment
CHUNK_SIZE_TOKENS=500
CHUNK_OVERLAP_TOKENS=100
```

**Change embedding model:**
```python
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

**Configure Claude model:**
```python
CLAUDE_MODEL=claude-haiku-4-5-20251001
CLAUDE_MAX_TOKENS=4096
```

---

## Development

### Project Structure

```
rag-cli/
├── src/
│   ├── rag_cli/              # Core library
│   └── rag_cli_plugin/       # Plugin code
├── tests/                    # Test suite
├── scripts/                  # Utilities
├── config/                   # Configuration
├── docs/                     # Documentation
└── data/                     # Runtime data
```

### Development Setup

```bash
# Clone and setup
git clone https://github.com/ItMeDiaTech/rag-cli.git
cd rag-cli
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific tests
pytest tests/test_core.py -v
```

### Testing

**Run all tests:**
```bash
pytest
```

**Run with coverage:**
```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html  # View coverage report
```

**Run specific test categories:**
```bash
pytest tests/test_foundation.py  # Foundation tests
pytest tests/test_core.py         # Core module tests
pytest tests/test_integration.py  # Integration tests
```

### Code Quality

**Format code:**
```bash
black src/
```

**Lint code:**
```bash
pylint src/
```

**Type checking:**
```bash
mypy src/
```

### Contributing Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes and add tests
4. Run test suite: `pytest`
5. Commit changes: `git commit -m "Add your feature"`
6. Push to branch: `git push origin feature/your-feature`
7. Open a Pull Request

**Important**: No emojis in code, documentation, or commit messages.

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Ways to Contribute

- Report bugs via GitHub Issues
- Suggest features or improvements
- Submit pull requests for bug fixes
- Improve documentation
- Write tutorials or blog posts
- Help answer questions in discussions

### Code Standards

- Follow PEP 8 style guide
- Add docstrings to all public functions
- Write tests for new features
- Update documentation for changes
- No emojis in code or documentation
- Professional, clear commit messages

### Testing Requirements

- All tests must pass: `pytest`
- Maintain or improve code coverage
- Add integration tests for new features
- Document test cases in docstrings

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

### Core Technologies

- **FAISS** - Facebook AI Similarity Search for vector operations
- **Sentence Transformers** - Embedding generation (all-MiniLM-L6-v2)
- **Anthropic Claude** - AI response generation
- **LangChain** - Document processing utilities

### Contributors

Created and maintained by DiaTech.

### Related Projects

- [Claude Code](https://claude.com/code) - AI-powered coding assistant
- [Multi-Agent Framework](https://github.com/ItMeDiaTech/multi-agent-framework) - Embedded agent system

---

## Support

- **Documentation**: [GitHub Wiki](https://github.com/ItMeDiaTech/rag-cli/wiki)
- **Issues**: [GitHub Issues](https://github.com/ItMeDiaTech/rag-cli/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ItMeDiaTech/rag-cli/discussions)
- **Email**: support@rag-cli.dev

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and release notes.

**Latest Release**: v2.0.0 - Complete restructuring with dual-package architecture, lifecycle management, and marketplace integration.

---

**Built with passion for better developer experiences.**
