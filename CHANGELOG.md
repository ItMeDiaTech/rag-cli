# Changelog - RAG-CLI

All notable changes to RAG-CLI are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.2.0] - 2025-10-30

### Major Features

#### Embedded Multi-Agent Framework (MAF)
- **Complete MAF integration**: All 7 specialized agents embedded in plugin
  - Debugger, Developer, Reviewer, Tester, Architect, Documenter, Optimizer
- **Parallel Execution**: RAG and MAF run simultaneously via asyncio
- **Intelligent Routing**: 4 execution strategies (RAG_ONLY, MAF_ONLY, PARALLEL_RAG_MAF, DECOMPOSED)
- **Graceful Fallback**: Auto-fallback to RAG-only with user notifications

#### New Commands
- `/rag-maf-config`: Control Multi-Agent Framework
  - `status`, `enable`, `disable`, `test-connection`, `list-agents`, `set-mode`

#### Plugin Installation Ready
- Added LICENSE file (MIT)
- Added pyproject.toml (PEP 517/518)
- Added .env.example template
- Fixed all GitHub URLs (yourusername → ItMeDiaTech)
- Synchronized versions to 1.2.0
- Created config template files
- Fixed Python package structure

### Documentation
- MAF_INTEGRATION_v1.2.0.md (architecture)
- IMPLEMENTATION_COMPLETE_v1.2.0.md (implementation)
- PLUGIN_INSTALLATION_FIXES_v1.2.0.md (fixes)
- HOOK_FILES_REFERENCE.md (hook documentation)

### Fixed
- Missing __init__.py files (CRITICAL)
- Broken sync_plugin.py references (CRITICAL)
- GitHub URL placeholders (CRITICAL)
- Version inconsistencies (CRITICAL)
- Missing LICENSE (CRITICAL)
- Broken setup.py entry points (HIGH)
- Missing .env.example (HIGH)

## [1.1.0] - 2025-10-29

### Added
- ArXiv integration for academic papers
- Tavily integration for web search
- Enhanced query classifier (10 intent types)
- Query decomposer for complex queries
- Result synthesizer for RAG+MAF
- Output formatter (550+ lines)
- Web dashboard for monitoring
- Latency tracker

### Improved
- HyDE (70% latency reduction)
- Semantic caching
- Error tracking

## [1.0.0] - 2025-10-25

### Initial Release
- Local RAG pipeline with FAISS
- Claude Code plugin with 5 commands
- 6 active hooks for lifecycle management
- MCP Server with 14 tools
- Monitoring and observability
- Comprehensive documentation

---

For upgrade guide and more details, see the documentation.
