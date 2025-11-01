# RAG-CLI Complete Reinstall Guide

This guide will help you completely remove the old installation and reinstall cleanly from the marketplace.

## Step 1: Close Claude Code

**IMPORTANT**: Close Claude Code completely before proceeding. This releases all locked files.

## Step 2: Run Cleanup Script

```cmd
# Navigate to RAG-CLI directory
cd C:\Users\DiaTech\Pictures\DiaTech\Programs\DocHub\development\RAG-CLI

# Run cleanup script
CLEANUP.bat
```

This will remove:
- Old manual installation at `~/.claude/plugins/rag-cli`
- All backup directories `~/.claude/plugins/rag-cli_backup_*`
- Any locked log files

## Step 3: Verify Cleanup

After running the cleanup script, verify these directories are gone:

```bash
# Should NOT exist:
ls ~/.claude/plugins/rag-cli          # Should be gone
ls ~/.claude/mcp/rag-cli.json         # Should be gone
ls ~/.claude/hooks/rag-cli            # Should be gone
ls ~/.claude/commands/*rag*.md        # Should be gone

# Should ONLY exist (marketplace installation):
ls ~/.claude/plugins/marketplaces/rag-cli  # Should exist
```

## Step 4: Restart Claude Code

Open Claude Code in a fresh terminal session.

## Step 5: Remove Old Marketplace Installation

```bash
/plugin uninstall rag-cli
/plugin marketplace remove ItMeDiaTech/rag-cli
```

## Step 6: Reinstall from Updated Repository

```bash
# Add the marketplace (pulls latest from GitHub)
/plugin marketplace add https://github.com/ItMeDiaTech/rag-cli.git

# Install the plugin
/plugin install rag-cli
```

## Step 7: Verify Installation

After installation, check:

1. **MCP Server Status**: Should show "running" not "failed"
   - Command: `python -m src.plugin.mcp.unified_server`
   - Not: `python -m src.monitoring.service_manager` (old/wrong)

2. **Commands Available**:
   - `/search` or `/rag-cli:search`
   - `/rag-cli:rag-enable`
   - `/rag-cli:rag-disable`
   - `/rag-cli:update-rag`

3. **No Duplicates**: Commands should appear only once in the list

## Troubleshooting

### MCP Server Still Shows Old Config

If you still see `python -m src.monitoring.service_manager`:

1. Close Claude Code completely
2. Delete marketplace installation manually:
   ```bash
   rm -rf ~/.claude/plugins/marketplaces/rag-cli
   ```
3. Restart Claude Code
4. Reinstall from Step 6

### Commands Still Duplicated

1. Check for files in `~/.claude/commands/`:
   ```bash
   ls ~/.claude/commands/ | grep rag
   ```
2. Remove any RAG-related files:
   ```bash
   rm ~/.claude/commands/*rag*.md
   rm ~/.claude/commands/search.md
   rm ~/.claude/commands/watch-rag.md
   ```
3. Restart Claude Code

### MCP Server Won't Start

1. Check Python can import the module:
   ```bash
   cd ~/.claude/plugins/marketplaces/rag-cli
   python -m src.plugin.mcp.unified_server
   ```
2. If it fails, check PYTHONPATH is set correctly in plugin.json
3. Verify all source files were synced to the marketplace directory

## Expected Final State

After successful reinstall:

```
~/.claude/
 plugins/
    marketplaces/
        rag-cli/              # Only RAG installation
            .claude-plugin/
               plugin.json
               marketplace.json
               hooks.json
            src/
               core/
               monitoring/
               plugin/
                   commands/  # Command definitions
                   hooks/     # Hook scripts
                   mcp/       # MCP server
            config/
                rag_settings.json
 mcp/
    # NO rag-cli.json file here
```

## Support

If issues persist:
1. Check GitHub issues: https://github.com/ItMeDiaTech/rag-cli/issues
2. Review plugin.json configuration
3. Check Claude Code logs for errors
