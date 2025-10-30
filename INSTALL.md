# RAG-CLI Installation Guide

Complete installation guide for the RAG-CLI Claude Code plugin.

## Prerequisites

- Python 3.8 or higher
- Claude Code CLI installed
- 4GB RAM minimum (8GB recommended)
- Git (for cloning the repository)

## Quick Install

### Option 1: Using pip (Recommended)

```bash
# Install directly from GitHub
pip install git+https://github.com/ItMeDiaTech/rag-cli.git

# Configure Claude Code integration
python -m scripts.sync_plugin
```

### Option 2: Manual Installation

```bash
# Clone the repository
git clone https://github.com/ItMeDiaTech/rag-cli.git
cd rag-cli

# Install dependencies
pip install -r requirements.txt

# Sync plugin files to Claude Code directory
python scripts/sync_plugin.py
```

## What Gets Installed

The installation process configures the following components in your Claude Code directory (`~/.claude/`):

### 1. Hooks (`~/.claude/hooks/rag-cli/`)

- `user-prompt-submit.py` - Automatically enhances queries with document context
- `response-post.py` - Adds inline citations to RAG-enhanced responses
- `error-handler.py` - Graceful error handling with warnings
- `plugin-state-change.py` - Persists settings across restarts
- `document-indexing.py` - Auto-indexes new/modified documents

### 2. Commands (`~/.claude/commands/`)

- `/search` - Search indexed documents and get AI answers
- `/rag-enable` - Enable automatic RAG enhancement
- `/rag-disable` - Disable automatic RAG enhancement
- `/update-rag` - Sync plugin files with Claude Code
- `/watch-rag` - Launch real-time monitoring dashboard

### 3. MCP Server (`~/.claude/mcp/rag-cli.json`)

Configures the Model Context Protocol server for RAG operations with:
- Automatic environment detection
- Project root path configuration
- Claude Code integration mode

### 4. Skills (`~/.claude/skills/rag-cli/`)

- `rag-retrieval` - Semantic search skill for document queries

## Post-Installation

### 1. Index Your Documents

```bash
# Navigate to your project
cd rag-cli

# Index documents from a directory
python scripts/index.py --input /path/to/your/documents

# Or index specific file types
python scripts/index.py --input /path/to/docs --types md,pdf,txt
```

### 2. Enable RAG Enhancement

```bash
# In Claude Code CLI
/rag-enable
```

This enables automatic query enhancement. When you ask questions, RAG-CLI will:
- Classify your query to determine if it's technical
- Search indexed documents for relevant context
- Automatically inject context into your prompt
- Provide citations in responses

### 3. Test the Installation

```bash
# Search your indexed documents
/search "How do I configure the API?"

# Or just ask a question naturally
# (RAG will auto-enhance if the query is relevant)
How do I set up authentication in the API?
```

### 4. Monitor Performance (Optional)

```bash
# Launch the real-time monitoring dashboard
/watch-rag
```

This opens a web dashboard at http://localhost:5000 showing:
- Real-time query processing
- Document retrieval decisions
- Performance metrics
- System logs

## API Token Configuration

### Automatic Token Detection

RAG-CLI can automatically detect and use API tokens from your `.env` file during installation:

1. **Copy the example file**:
   ```bash
   cp .env.example .env
   ```

2. **Add your tokens** (all optional):
   ```bash
   # GitHub token for documentation access
   GITHUB_TOKEN="your_github_token_here"

   # Stack Overflow API key for higher rate limits
   STACKOVERFLOW_KEY="your_stackoverflow_key_here"
   ```

3. **Run installation**:
   ```bash
   python scripts/sync_plugin.py
   ```

The setup script will:
- Check for existing tokens in `.env` file
- Prompt you to enter tokens if not found
- Securely save tokens to `.env` file
- Never commit tokens to version control (`.env` is gitignored)

### Manual Token Configuration

If you skip token configuration during installation, you can add them later:

1. Create or edit `.env` file in project root
2. Add your tokens using the format shown in `.env.example`
3. Restart Claude Code or re-run the sync script

### Token Usage

**GitHub Token**:
- Used for accessing documentation from GitHub repositories
- Provides higher API rate limits
- Optional but recommended for heavy usage

**Stack Overflow Key**:
- Provides higher rate limits for Stack Overflow API
- Completely optional and rarely needed
- Only useful if you frequently query Stack Overflow

Note: When running as a Claude Code plugin, NO Anthropic API key is required.

## Configuration

### RAG Settings

Settings are stored in `config/rag_settings.json`:

```json
{
  "enabled": true,
  "auto_trigger_threshold": 5,
  "context_limit": 3,
  "relevance_threshold": 0.6,
  "cache_queries": true,
  "enable_agent_orchestration": true
}
```

**Key Settings:**

- `enabled` - Enable/disable RAG enhancement
- `auto_trigger_threshold` - Minimum words to trigger RAG (default: 5)
- `context_limit` - Max documents to include (default: 3)
- `relevance_threshold` - Min similarity score (0-1, default: 0.6)
- `enable_agent_orchestration` - Use multi-agent orchestration (default: true)

### Environment Variables

For standalone mode (outside Claude Code):

```bash
# Required for standalone mode only
export ANTHROPIC_API_KEY="your-api-key"

# Optional configuration
export RAG_CLI_MODE="claude_code"     # or "standalone" or "hybrid"
export RAG_CLI_LOG_LEVEL="INFO"       # DEBUG, INFO, WARNING, ERROR
export RAG_CLI_ROOT="/path/to/rag-cli"
```

Note: When running as a Claude Code plugin, NO API key is required.

## Multi-Agent Framework Integration

RAG-CLI includes advanced multi-agent orchestration capabilities:

### Features

- **Adaptive Strategy Selection** - Chooses optimal retrieval strategy based on query type
- **Parallel Execution** - Runs multiple agents concurrently for complex queries
- **Query Decomposition** - Breaks down complex queries into sub-tasks
- **MAF Integration** - Optionally uses external Multi-Agent Framework for advanced reasoning

### Configuration

Enable in `config/rag_settings.json`:

```json
{
  "enable_agent_orchestration": true,
  "orchestration": {
    "enable_maf": true,
    "parallel_threshold_confidence": 0.7,
    "decomposition_complexity_threshold": 0.6,
    "maf_timeout": 30.0
  }
}
```

## Updating the Plugin

### Method 1: Using Update Command

```bash
# In Claude Code CLI
/update-rag
```

### Method 2: Manual Update

```bash
# Pull latest changes
cd rag-cli
git pull

# Sync updated files
python scripts/sync_plugin.py

# Restart Claude Code
```

## Troubleshooting

### Hooks Not Working

1. Verify hooks are installed:
   ```bash
   ls ~/.claude/hooks/rag-cli/
   ```

2. Check hook permissions:
   ```bash
   chmod +x ~/.claude/hooks/rag-cli/*.py
   ```

3. Check RAG settings:
   ```bash
   cat config/rag_settings.json
   ```
   Ensure `"enabled": true`

### No Vector Index Found

```bash
# Create initial index
python scripts/index.py --input data/documents

# Verify index was created
ls data/vectors/
```

### MCP Server Not Starting

1. Check MCP configuration:
   ```bash
   cat ~/.claude/mcp/rag-cli.json
   ```

2. Verify project root path is correct

3. Check Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Performance Issues

1. Check monitoring dashboard for bottlenecks:
   ```bash
   /watch-rag
   ```

2. Reduce context limit in settings:
   ```json
   {
     "context_limit": 2,
     "relevance_threshold": 0.7
   }
   ```

3. Disable agent orchestration for faster responses:
   ```json
   {
     "enable_agent_orchestration": false
   }
   ```

## Uninstallation

To remove RAG-CLI from Claude Code:

```bash
# Remove plugin files
rm -rf ~/.claude/hooks/rag-cli
rm ~/.claude/mcp/rag-cli.json
rm ~/.claude/commands/search.md
rm ~/.claude/commands/rag-*.md
rm ~/.claude/commands/update-rag.md
rm ~/.claude/commands/watch-rag.md

# Optionally remove the installation
pip uninstall rag-cli
```

## Support

- GitHub Issues: https://github.com/ItMeDiaTech/rag-cli/issues
- Documentation: https://github.com/ItMeDiaTech/rag-cli
- Discussions: https://github.com/ItMeDiaTech/rag-cli/discussions

## Next Steps

After installation:

1. Index your project documentation
2. Enable RAG enhancement
3. Test with a few queries
4. Monitor performance with the dashboard
5. Adjust settings as needed
6. Explore advanced features like multi-agent orchestration
