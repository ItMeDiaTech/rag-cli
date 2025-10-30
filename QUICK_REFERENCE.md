# RAG-CLI v1.2.0 - Quick Reference

## Installation

```bash
# Clone from GitHub
git clone https://github.com/ItMeDiaTech/rag-cli.git
cd rag-cli

# Install dependencies
pip install -r requirements.txt

# Verify installation
python scripts/verify_installation.py
```

## Claude Code Plugin

```bash
# Install as Claude Code plugin
/plugin install rag-cli

# Check MAF status
/rag-maf-config status

# Enable/disable MAF agents
/rag-maf-config enable
/rag-maf-config disable

# Search documents
/search "your query"
```

## Available Commands

| Command | Purpose |
|---------|---------|
| `/search` | Search indexed documents |
| `/rag-enable` | Enable RAG enhancement |
| `/rag-disable` | Disable RAG |
| `/rag-project` | Analyze current project |
| `/rag-maf-config` | Control Multi-Agent Framework |
| `/update-rag` | Sync plugin files |

## MAF Configuration

```bash
# Check available agents
/rag-maf-config list-agents

# Test MAF health
/rag-maf-config test-connection

# Change execution mode
/rag-maf-config set-mode PARALLEL  # or SEQUENTIAL
```

## Core Functionality

**Document Indexing**:
```bash
python scripts/index.py --input data/documents --output data/vectors
```

**Retrieval**:
```bash
python scripts/retrieve.py "How do I...?"
```

**Monitoring**:
```bash
python -m src.monitoring.tcp_server
```

## Environment Setup

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Then set your values:
```
ANTHROPIC_API_KEY=your_key_here    # Optional (Claude Code mode)
TAVILY_API_KEY=your_tavily_key     # Optional
RAG_CLI_MODE=claude_code           # or standalone, hybrid
```

## 7 Embedded Agents

1. **Debugger** - Error analysis & troubleshooting
2. **Developer** - Code implementation & generation
3. **Reviewer** - Code quality & security checks
4. **Tester** - Test creation & validation
5. **Architect** - System design & planning
6. **Documenter** - Documentation generation
7. **Optimizer** - Performance optimization

## Operating Modes

- **Claude Code Mode** (default): No API key needed, zero cost
- **Standalone Mode**: Requires API key, full control
- **Hybrid Mode**: Auto-detects and uses best available

## Execution Strategies

- **RAG_ONLY**: Simple queries, fast
- **MAF_ONLY**: Pure code analysis
- **PARALLEL_RAG_MAF**: Both simultaneously (default)
- **DECOMPOSED**: Complex multi-part queries

## Default Configuration

- **Parallel Execution**: Enabled
- **RAG + MAF**: Simultaneous
- **Timeout**: 30 seconds
- **Agents**: All 7 available
- **Fallback**: Auto-fallback to RAG-only

## Troubleshooting

```bash
# Verify installation
python scripts/verify_installation.py

# Check MAF health
/rag-maf-config test-connection

# View logs
tail -f logs/application.log

# Test retrieval
python scripts/retrieve.py "test"
```

## Documentation

- **README.md** - Full documentation
- **CHANGELOG.md** - Version history
- **CONTRIBUTING.md** - Development guidelines
- **HOOK_FILES_REFERENCE.md** - Hook documentation
- **RELEASE_RECOMMENDATIONS_v1.2.0.md** - Release process
- **INSTALLATION_SUMMARY_FINAL.md** - Implementation summary

## Support

- **GitHub Issues**: https://github.com/ItMeDiaTech/rag-cli/issues
- **Discussions**: https://github.com/ItMeDiaTech/rag-cli/discussions
- **Docs**: https://github.com/ItMeDiaTech/rag-cli/wiki

## Key Features

✅ Local-first architecture
✅ Embedded Multi-Agent Framework
✅ Parallel execution (RAG + MAF)
✅ Graceful fallback handling
✅ Comprehensive monitoring
✅ Claude Code integration
✅ Zero API cost in Claude Code mode

---

**Version**: 1.2.0
**Status**: Production Ready
**License**: MIT
