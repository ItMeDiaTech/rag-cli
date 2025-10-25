# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG-CLI is a local Retrieval-Augmented Generation system designed as a Claude Code plugin. It processes documents locally, generates embeddings, stores vectors in FAISS, and uses claude-haiku-4-5-20251001 for response generation.

## Project Structure to Create

```
RAG-CLI/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── document_processor.py    # Document chunking (400-500 tokens)
│   │   ├── embeddings.py            # sentence-transformers/all-MiniLM-L6-v2
│   │   ├── vector_store.py          # FAISS operations
│   │   ├── retrieval_pipeline.py    # Hybrid search + reranking
│   │   └── claude_integration.py    # claude-haiku-4-5-20251001 interface
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── logger.py                # Comprehensive logging
│   │   ├── metrics.py               # Performance tracking
│   │   └── tcp_server.py            # PowerShell interface (port 9999)
│   └── plugin/
│       ├── skills/                  # Agent Skills
│       ├── commands/                 # Slash commands
│       ├── hooks/                    # UserPromptSubmit hooks
│       └── mcp/                      # MCP server configs
├── scripts/
│   ├── index.py                     # Document indexing script
│   ├── retrieve.py                  # Retrieval CLI
│   └── monitor.ps1                  # PowerShell monitoring
├── tests/
│   ├── test_document_processor.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   └── test_integration.py
├── data/
│   ├── documents/                   # Source documents
│   └── vectors/                      # FAISS indexes
├── requirements.txt
├── setup.py
├── pytest.ini
└── .gitignore
```

## Development Commands

### Initial Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies (create requirements.txt first)
pip install sentence-transformers faiss-cpu anthropic flask langchain pytest pytest-asyncio

# Initialize git repository
git init
git add .
git commit -m "feature: initialize RAG-CLI project structure"
```

### Build and Run
```bash
# Index documents
python scripts/index.py --input data/documents --output data/vectors

# Test retrieval
python scripts/retrieve.py --query "How to configure API?" --top-k 5

# Run monitoring server
python -m src.monitoring.tcp_server

# PowerShell monitoring
./scripts/monitor.ps1 -Command STATUS
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

### 1. Document Processing (`src/core/document_processor.py`)
- Chunk size: 400-500 tokens
- Overlap: 10-20% (50-100 tokens)
- Use RecursiveCharacterTextSplitter from langchain
- Add contextual headers (document title, section)
- Support formats: MD, PDF, DOCX, HTML, TXT

### 2. Embeddings (`src/core/embeddings.py`)
- Model: sentence-transformers/all-MiniLM-L6-v2
- Dimensions: 384
- Batch processing for efficiency
- LRU cache for repeated queries

### 3. Vector Store (`src/core/vector_store.py`)
- FAISS IndexFlatL2 for <100K vectors
- FAISS IndexHNSWFlat for 100K-1M vectors
- Save/load functionality
- Metadata storage with pickle

### 4. Retrieval Pipeline (`src/core/retrieval_pipeline.py`)
- Hybrid search: 0.7 vector + 0.3 keyword
- Two-stage: retrieve 10, rerank to 5
- Cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2
- Target latency: <100ms for search

### 5. Claude Integration (`src/core/claude_integration.py`)
- Model: claude-haiku-4-5-20251001
- Stream responses for better UX
- Context assembly from retrieved chunks
- Prompt template with citations

### 6. Monitoring (`src/monitoring/tcp_server.py`)
- TCP server on port 9999
- Endpoints: /status, /logs, /metrics
- JSON responses
- Real-time log streaming

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

1. **Local-first**: Everything runs locally except Claude API
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