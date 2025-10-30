# RAG-CLI Claude Code Plugin

This document explains how to install and use RAG-CLI as a Claude Code plugin.

## Quick Install (Claude Code Plugin System)

### Method 1: Install from Marketplace (Recommended)

```bash
# Add the RAG-CLI marketplace
/plugin marketplace add ItMeDiaTech/rag-cli

# Browse available plugins
/plugin

# Install RAG-CLI
/plugin install rag-cli
```

After installation, restart Claude Code for the plugin to activate.

What this does:
- Downloads plugin from GitHub repository
- Installs hooks to ~/.claude/hooks/rag-cli/
- Installs commands to ~/.claude/commands/
- Configures MCP server at ~/.claude/mcp/rag-cli.json
- Installs skills to ~/.claude/skills/rag-cli/
- Prompts for API tokens (optional)
- Creates default configuration files

### Method 2: Manual Installation

If you prefer manual installation or the automated method doesn't work:

```bash
# Clone the repository
git clone https://github.com/ItMeDiaTech/rag-cli.git
cd rag-cli

# Install Python dependencies
pip install -r requirements.txt

# Run the sync script to configure Claude Code
python scripts/sync_plugin.py
```

## What Gets Installed

The plugin installs the following components to your `~/.claude/` directory:

### Hooks
Located in `~/.claude/hooks/rag-cli/`:
- `slash-command-blocker.py` (Priority 150) - Prevents Claude from responding to slash commands
- `user-prompt-submit.py` (Priority 100) - Automatically enhances queries with RAG context and multi-agent orchestration
- `response-post.py` (Priority 80) - Adds inline citations to responses
- `error-handler.py` (Priority 70) - Graceful error handling
- `plugin-state-change.py` (Priority 60) - Settings persistence
- `document-indexing.py` (Priority 50) - Auto-indexes new documents

### Commands
Located in `~/.claude/commands/`:
- `/search` - Search indexed documents
- `/rag-enable` - Enable automatic RAG enhancement
- `/rag-disable` - Disable automatic RAG enhancement
- `/rag-project` - Analyze project and index relevant documentation
- `/update-rag` - Sync plugin files

### MCP Server
Located in `~/.claude/mcp/rag-cli.json`:
- Configures the Model Context Protocol server for RAG operations

### Skills
Located in `~/.claude/skills/rag-cli/`:
- `rag-retrieval` - Semantic search skill

## Team Installation

For team projects, add RAG-CLI to your project's `.claude/settings.json`:

```json
{
  "plugins": {
    "rag-cli": {
      "enabled": true,
      "marketplace": "ItMeDiaTech/rag-cli"
    }
  }
}
```

When team members trust the repository folder, plugins will install automatically.

## Initial Setup

After installation, you need to index your documents:

```bash
# Index documents
python scripts/index.py --input /path/to/your/docs

# Enable RAG
/rag-enable

# Test
/search "How do I configure authentication?"
```

## Features

### Automatic Query Enhancement
When enabled, RAG-CLI automatically:
- Classifies your queries to determine relevance
- Searches indexed documents for context
- Injects relevant information into prompts
- Provides inline citations in responses

### Multi-Agent Orchestration
RAG-CLI includes advanced features:
- Adaptive strategy selection based on query type
- Parallel agent execution for complex queries
- Query decomposition for multi-step tasks
- Integration with external Multi-Agent Framework

### Real-Time Monitoring
Monitor plugin performance:
```bash
/watch-rag
```

Opens dashboard at http://localhost:5000 showing:
- Live query processing timeline
- Document retrieval decisions
- Performance metrics
- System logs

## Configuration

Settings are stored in `config/rag_settings.json`:

```json
{
  "enabled": true,
  "auto_trigger_threshold": 5,
  "context_limit": 3,
  "relevance_threshold": 0.6,
  "enable_agent_orchestration": true
}
```

Modify these settings to customize RAG behavior.

## Troubleshooting

### Plugin Not Loading

1. Check installation:
   ```bash
   ls ~/.claude/hooks/rag-cli/
   ls ~/.claude/commands/ | grep rag
   ```

2. Restart Claude Code

3. Verify settings:
   ```bash
   cat config/rag_settings.json
   ```

### No Documents Found

```bash
# Re-index documents
python scripts/index.py --input data/documents

# Verify index
ls data/vectors/
```

### Hooks Not Triggering

1. Ensure RAG is enabled:
   ```bash
   /rag-enable
   ```

2. Check that queries meet minimum threshold (default: 5 words)

3. Verify hooks are executable:
   ```bash
   chmod +x ~/.claude/hooks/rag-cli/*.py
   ```

## Updating the Plugin

### Using Plugin System
```bash
/plugin update rag-cli
```

### Manual Update
```bash
# Update repository
cd rag-cli
git pull

# Re-sync
python scripts/sync_plugin.py
```

## Uninstalling

### Using Plugin System
```bash
/plugin uninstall rag-cli
```

### Manual Uninstall
```bash
# Remove plugin files
rm -rf ~/.claude/hooks/rag-cli
rm ~/.claude/mcp/rag-cli.json
rm ~/.claude/commands/search.md
rm ~/.claude/commands/rag-*.md
rm ~/.claude/commands/update-rag.md
rm ~/.claude/commands/watch-rag.md
rm -rf ~/.claude/skills/rag-cli
```

## Support

- GitHub: https://github.com/ItMeDiaTech/rag-cli
- Issues: https://github.com/ItMeDiaTech/rag-cli/issues
- Discussions: https://github.com/ItMeDiaTech/rag-cli/discussions

## Additional Resources

- Full installation guide: [INSTALL.md](INSTALL.md)
- User documentation: [README.md](README.md)
- Project instructions: [CLAUDE.md](CLAUDE.md)
