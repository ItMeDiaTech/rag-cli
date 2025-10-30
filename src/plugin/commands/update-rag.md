# Update RAG Plugin

Synchronize RAG-CLI plugin files from the development directory to the Claude Code configuration directory.

## Task

Execute the RAG-CLI plugin sync operation to update all plugin components in the Claude Code configuration:

1. **Prepare for sync**
   - Verify current working directory is RAG-CLI project root
   - Ensure the development files are ready to sync
   - Check if sync should be run in dry-run mode (if user specified `--dry-run`)

2. **Execute sync_plugin.py**
   - Run the sync script: `python sync_plugin.py`
   - Accept command-line arguments if provided:
     - `--dry-run`: Preview changes without applying them
     - `--verbose`: Show detailed sync information
     - `--force`: Force sync even if timestamps match
     - `--no-backup`: Skip backup creation before sync

3. **Report sync results**
   - Display the sync summary showing:
     - New files copied
     - Updated files
     - Deleted obsolete files
     - Any errors encountered
   - Confirm successful sync or report issues

## What Gets Synced

The sync operation updates the following in `~/.claude/`:

### Plugin Components
- **Commands**: `src/plugin/commands/*.md` → `.claude/commands/`
- **Hooks**: `src/plugin/hooks/*.py` → `.claude/plugins/rag-cli/hooks/`
- **Skills**: `src/plugin/skills/` → `.claude/plugins/rag-cli/skills/`
- **MCP Configs**: `src/plugin/mcp/*.json` → `.claude/plugins/rag-cli/mcp/`

### Core Code
- **Core modules**: `src/core/` → `.claude/plugins/rag-cli/src/core/`
- **Monitoring**: `src/monitoring/` → `.claude/plugins/rag-cli/src/monitoring/`

### Data
- **Documents**: `data/documents/` → `.claude/plugins/rag-cli/data/documents/`
- **Vectors**: `data/vectors/` → `.claude/plugins/rag-cli/data/vectors/`

### Configuration
- `requirements.txt`
- `README.md`
- `plugin.json`

## Usage Examples

### Standard sync
```
/update-rag
```

### Preview changes without applying
```
/update-rag --dry-run
```

### Verbose output with detailed logging
```
/update-rag --verbose
```

### Force sync (ignore timestamps)
```
/update-rag --force
```

### Sync without creating backup
```
/update-rag --no-backup
```

## Important Notes

- **Automatic Backup**: By default, creates a timestamped backup of the plugin directory before syncing
- **Smart Merge**: Preserves runtime files like logs, cached state, and vector indexes
- **Cleanup**: Removes obsolete files that no longer exist in the source
- **Validation**: Verifies critical files are present after sync
- **Working Directory**: Must be run from the RAG-CLI project root

## Preserved Files

The sync preserves these runtime files in `.claude/plugins/rag-cli/`:
- `logs/` directory
- `data/vectors/` (existing indexes)
- `*.log` files
- `*state.json` files
- `__pycache__/` directories

## Typical Output

```
============================================================
SYNC SUMMARY
============================================================

New files (3):
  + commands/update-rag.md
  + hooks/user-prompt-submit.py
  + src/core/query_classifier.py

Updated files (12):
  ~ src/core/retrieval_pipeline.py
  ~ src/monitoring/tcp_server.py
  ~ src/monitoring/web_dashboard.py
  ~ hooks/update-rag-hook.py
  ~ src/core/claude_integration.py
  ... and 7 more

Deleted files (1):
  - commands/obsolete-command.md

Total changes: 16
============================================================
```

## Troubleshooting

**Sync fails with "Source directory not found"**
- Ensure you're running from the RAG-CLI project root
- Check that `src/plugin/` directory exists

**Sync fails with "Claude directory not found"**
- Verify `~/.claude/` exists
- Ensure Claude Code is properly installed

**Permission errors**
- On Windows, ensure no files are locked by running processes
- Close monitoring dashboard and TCP server before syncing

**Want to undo a sync**
- Restore from the timestamped backup: `~/.claude/plugins/rag-cli_backup_YYYYMMDD_HHMMSS/`
