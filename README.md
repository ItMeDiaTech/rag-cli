# RAG-CLI

A powerful local Retrieval-Augmented Generation (RAG) system designed as a Claude Code plugin. Process documents locally, generate embeddings, store vectors in FAISS, and get AI-powered responses using Claude Haiku.

## Features

- **Local-First Architecture**: Everything runs locally except Claude API calls
- **Fast Performance**: <100ms vector search, <5s end-to-end responses
- **Hybrid Search**: Combines semantic vector search with keyword matching for superior accuracy
- **Claude Code Integration**: Seamless plugin for enhanced development workflow
- **Multi-Format Support**: Process MD, PDF, DOCX, HTML, and TXT files
- **Real-Time Monitoring**: TCP server with PowerShell interface for system observability

## Quick Start

### Prerequisites

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- (Optional) Anthropic API key for standalone mode

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-cli.git
cd rag-cli

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Claude Code Deployment

When running as a Claude Code plugin, NO API key is required. The system automatically detects the Claude Code environment and uses its internal interface.

### Standalone Setup (Optional)

For standalone operation or testing outside Claude Code:

```bash
# Set API key for direct Claude API access
export ANTHROPIC_API_KEY="your-api-key-here"
export RAG_CLI_LOG_LEVEL="INFO"
export RAG_CLI_MODE="standalone"
```

### Basic Usage

1. **Index your documents**:
```bash
python scripts/index.py --input data/documents --output data/vectors
```

2. **Search and retrieve**:
```bash
python scripts/retrieve.py "How to configure API authentication?"
```

3. **Start monitoring server**:
```bash
python -m src.monitoring.tcp_server
```

## Operation Modes

RAG-CLI supports three operation modes:

### 1. Claude Code Mode (Default)
- **No API key required**
- Automatically detected when running as Claude Code plugin
- Returns formatted context for Claude's internal processing
- Optimal performance with zero API costs

### 2. Standalone Mode
- Requires Anthropic API key
- Direct API calls to Claude
- Full control over model parameters
- Useful for testing and development

### 3. Hybrid Mode
- Auto-detects environment
- Uses Claude Code when available
- Falls back to API when needed
- Maximum flexibility

Set mode via environment variable:
```bash
export RAG_CLI_MODE="claude_code"  # or "standalone" or "hybrid"
```

## Architecture

### System Components

```
RAG-CLI/
├── src/
│   ├── core/               # Core RAG pipeline
│   │   ├── embeddings.py   # Sentence transformer integration
│   │   ├── vector_store.py # FAISS vector operations
│   │   ├── document_processor.py # Document chunking
│   │   ├── retrieval_pipeline.py # Hybrid search
│   │   └── claude_integration.py # Claude API interface
│   │
│   ├── monitoring/         # Observability
│   │   ├── logger.py      # Structured logging
│   │   └── tcp_server.py  # Monitoring server
│   │
│   └── plugin/            # Claude Code integration
│       ├── skills/        # Agent skills
│       ├── commands/      # Slash commands
│       ├── hooks/         # Event hooks
│       └── mcp/           # MCP server
│
├── scripts/               # CLI utilities
├── tests/                 # Test suites
├── data/                  # Documents and vectors
└── config/                # Configuration files
```

### Data Flow

1. **Document Processing**: Documents → Chunks (400-500 tokens) → Metadata extraction
2. **Embedding Generation**: Chunks → sentence-transformers → 384-dim vectors
3. **Vector Storage**: Embeddings → FAISS index → Persistent storage
4. **Retrieval**: Query → Hybrid search → Reranking → Top-K results
5. **Response Generation**: Context + Query → Claude Haiku → AI response

## Configuration

### Core Settings (`config/default.yaml`)

```yaml
# Operation Mode
mode:
  operation: hybrid     # claude_code, standalone, or hybrid
  claude_code:
    format_context: true
    include_metadata: true
    max_context_length: 10000

# Embeddings
embeddings:
  model_name: sentence-transformers/all-MiniLM-L6-v2
  model_dim: 384
  batch_size: 32
  cache_enabled: true

# Vector Store
vector_store:
  type: faiss
  index_type: flat    # Use 'hnsw' for >100K documents
  save_path: data/vectors

# Retrieval
retrieval:
  top_k: 5
  hybrid_ratio: 0.7   # 70% vector, 30% keyword
  rerank: true
  reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2

# Claude (for standalone mode)
claude:
  model: claude-haiku-4-5-20251001
  max_tokens: 4096
  temperature: 0.7
  api_key_env: ANTHROPIC_API_KEY  # Only needed for standalone
```

## Claude Code Plugin

### Slash Commands

- `/search [query]` - Search indexed documents
- `/rag:enable` - Enable automatic RAG enhancement
- `/rag:disable` - Disable automatic RAG enhancement

### Agent Skills

Access the RAG retrieval skill:
```
/skill rag-retrieval "Your question here"
```

### Automatic Enhancement

When enabled, the UserPromptSubmit hook automatically enhances your queries with relevant context from the document knowledge base.

## API Reference

### Document Indexing

```python
from src.core.document_processor import get_document_processor
from src.core.embeddings import get_embedding_model
from src.core.vector_store import get_vector_store

# Process documents
processor = get_document_processor()
documents = processor.process_directory("data/documents")

# Generate embeddings
model = get_embedding_model()
embeddings = model.encode_batch([doc["content"] for doc in documents])

# Store vectors
store = get_vector_store()
store.add_documents(documents, embeddings)
```

### Document Retrieval

```python
from src.core.retrieval_pipeline import HybridRetriever

# Initialize retriever
retriever = HybridRetriever(vector_store, embedding_model, config)

# Search
results = retriever.search("Your query", top_k=5)
```

### Claude Integration

```python
from src.core.claude_integration import ClaudeAssistant

# Initialize assistant
assistant = ClaudeAssistant(config)

# Generate response
response = assistant.generate_response(query, retrieved_docs)
```

## Monitoring

### TCP Server Interface

The monitoring server runs on port 9999 and accepts these commands:

- `STATUS` - System health and statistics
- `METRICS` - Performance metrics
- `LOGS` - Recent log entries
- `HEALTH` - Health check status

### PowerShell Usage

```powershell
# Check status
./scripts/monitor.ps1 -Command STATUS

# View metrics
./scripts/monitor.ps1 -Command METRICS
```

### Python Client

```python
import socket
import json

def query_monitor(command):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(("localhost", 9999))
        s.send(command.encode())
        response = s.recv(4096).decode()
        return json.loads(response)

status = query_monitor("STATUS")
print(status)
```

## Performance

### Benchmarks

| Operation | Target | Typical |
|-----------|--------|---------|
| Vector Search | <100ms | 45ms |
| End-to-End | <5s | 3.2s |
| Embedding Generation | <500ms | 200ms |
| Document Processing | 0.5s/100 docs | 0.4s/100 docs |

### Optimization Tips

1. **Large Datasets** (>100K docs): Use HNSW index instead of Flat
2. **Memory Constraints**: Enable document streaming
3. **Faster Search**: Reduce top_k and disable reranking
4. **Better Accuracy**: Increase hybrid_ratio for more semantic search

## Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_core.py::TestEmbeddings
```

### Test Coverage

- Unit tests for all core modules
- Integration tests for full pipeline
- Performance benchmarks
- Plugin component validation

## Troubleshooting

### Common Issues

**No results found**:
- Ensure documents are indexed: `ls data/vectors/`
- Lower similarity threshold: `--threshold 0.5`
- Check document processing logs

**Slow performance**:
- Reduce top_k parameter
- Enable caching in configuration
- Use HNSW index for large datasets

**API errors** (Standalone mode only):
- Verify ANTHROPIC_API_KEY is set
- Check rate limits
- Switch to Claude Code mode if running as plugin
- Review logs: `tail -f logs/rag_cli.log`

**Mode detection issues**:
- Check current mode: `python -c "from src.core.claude_code_adapter import get_adapter; print(get_adapter().get_mode_info())"`
- Force mode: `export RAG_CLI_MODE="claude_code"`
- Verify .claude directory exists for Claude Code

### Debug Mode

```bash
export RAG_CLI_LOG_LEVEL=DEBUG
python scripts/retrieve.py "test query" --verbose
```

## Development

### Project Structure

- `src/core/` - Core RAG components
- `src/monitoring/` - Logging and metrics
- `src/plugin/` - Claude Code integration
- `scripts/` - CLI utilities
- `tests/` - Test suites
- `config/` - Configuration files

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add docstrings to all functions
- Run `black` for formatting

## License

MIT License - see LICENSE file for details.

## Support

- GitHub Issues: [Report bugs](https://github.com/yourusername/rag-cli/issues)
- Documentation: [Wiki](https://github.com/yourusername/rag-cli/wiki)
- Discussions: [Community forum](https://github.com/yourusername/rag-cli/discussions)

## Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Anthropic](https://www.anthropic.com/) for Claude API
- [LangChain](https://langchain.com/) for document processing

---

Built with focus on performance, accuracy, and developer experience.