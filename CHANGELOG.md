# Changelog

All notable changes to RAG-CLI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-10-30

### Added
- **Slash Command Blocker Hook** - Prevents Claude from responding to slash commands, showing only execution status
- **Complete Hook Registration** - All 6 hooks now properly registered in `.claude-plugin/hooks.json`
  - UserPromptSubmit (2 hooks: slash-command-blocker + user-prompt-submit)
  - ResponsePost (citation injection)
  - PluginStateChange (settings persistence)
  - FileCreated/FileModified (auto-indexing)
- **Clean Output Formatting System** - Comprehensive `OutputFormatter` class for readable orchestration output
  - Structured headers and sections
  - Progress indicators and metrics tables
  - Document previews with intelligent truncation
  - Collapsible sections for verbose mode
  - Helper functions for RAG, MAF, and hybrid pipeline formatting
- **`/rag-project` Command** - Analyze current project and index relevant documentation
- **Multi-Agent Orchestration Documentation** - Fully documented MAF integration in README and plugin docs
- **Formatted MCP Server Output** - Clean, structured output for search results with metrics
- **Formatted Orchestrator Output** - Readable document previews and structured responses
- **Formatted Hook Output** - User-prompt-submit hook now includes formatted orchestration summaries

### Changed
- **Torch Dependency** - Updated from `>=2.2.0,<2.5.0` to `>=2.6.0` for Python 3.13 compatibility
- **MCP Search Results** - Now displays structured output with headers, timing, and clean document previews
- **Agent Orchestrator** - Response formatting uses new OutputFormatter for consistent presentation
- **Documentation** - Updated README.md and CLAUDE_PLUGIN.md with complete hook listings and priorities

### Fixed
- Python 3.13 compatibility issue with torch version constraint
- Missing hook registrations (only 1 of 6 was registered)
- Missing `/rag-project` command file in plugin distribution
- Inconsistent output formatting across different components

### Technical Details

#### Hook System
- 7 hook implementation files
- 6 registered in hooks.json (update-rag-hook not registered as it's redundant with slash command)
- Priority-based execution order (150, 100, 80, 70, 60, 50)
- All hooks include comprehensive error handling and logging

#### Output Formatting
- 550+ lines of formatting utilities
- Support for markdown, progress bars, metrics tables
- Configurable verbosity levels
- Integration points: MCP server, agent orchestrator, user-prompt-submit hook

#### Commands
- 5 slash commands: `/search`, `/rag-enable`, `/rag-disable`, `/rag-project`, `/update-rag`
- All commands follow consistent format with clear instructions
- Slash command blocker ensures clean execution without AI commentary

#### Multi-Agent Framework
- 6 MAF integration files (maf_connector, orchestrator, 3 agents, monitor)
- 4 routing strategies: RAG_ONLY, MAF_ONLY, PARALLEL_RAG_MAF, DECOMPOSED
- Intelligent query classification and strategy selection
- Formatted output for all orchestration paths

### Installation
Plugin can be installed via GitHub:
```bash
/plugin install rag-cli
```

Or from marketplace:
```bash
/plugin marketplace add ItMeDiaTech/rag-cli
/plugin install rag-cli
```

### Requirements
- Python 3.8+ (recommended 3.13 for latest features)
- 4GB RAM minimum (8GB recommended)
- Dependencies automatically installed from requirements.txt

### Migration Notes
Users upgrading from development versions should:
1. Ensure all hooks are enabled in Claude Code
2. Restart Claude Code to activate new hooks
3. Verify `rag_settings.json` is configured correctly
4. Test `/rag-project` command to index current project

---

## [Unreleased]

### Planned Features
- Real-time progress indicators for long-running operations
- Enhanced MAF agent output formatting
- Custom output templates for different use cases
- Performance metrics dashboard integration
- Additional orchestration strategies

---

## Version History

### Development Milestones
- **Phase 1**: Core RAG pipeline implementation
- **Phase 2**: Multi-agent orchestration integration
- **Phase 3**: Query decomposition and sub-agent routing
- **Phase 4**: Claude Code plugin integration
- **Phase 5**: Hook system and output formatting
- **v1.0.0**: First stable release with complete feature set

### Contributors
- DiaTech - Lead Developer

### License
MIT License - See LICENSE file for details

### Support
- GitHub Issues: https://github.com/ItMeDiaTech/rag-cli/issues
- Documentation: https://github.com/ItMeDiaTech/rag-cli/wiki
