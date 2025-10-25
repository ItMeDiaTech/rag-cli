# /update-rag - Update RAG-CLI Plugin

Synchronize RAG-CLI plugin files with your Claude Code installation.

## Usage

```
/update-rag
```

## Description

The `/update-rag` command automatically runs the plugin synchronization process, updating your Claude Code RAG-CLI plugin with the latest implementation files from your development directory.

## What It Does

When you run this command:

1. **Backs up** the current plugin (with timestamp)
2. **Syncs plugin files** (commands, hooks, skills, mcp)
3. **Updates core modules** via symlinks (if possible)
4. **Cleans up** obsolete files
5. **Reports changes** showing what was updated

## Use Cases

- **After code changes**: You've modified RAG-CLI source files and want to update the plugin
- **Before testing**: Ensure your plugin has the latest code
- **After pulling updates**: You've pulled new changes from git
- **Daily workflow**: Keep plugin in sync with development

## Examples

```
# Standard update
/update-rag

# Update with detailed output
/update-rag --verbose

# Preview changes without applying
/update-rag --dry-run

# Force update (ignore file timestamps)
/update-rag --force
```

## Sync Details

The command synchronizes:

- **Commands**: Slash commands in `~/.claude/commands/`
- **Hooks**: Query interceptors in plugin hooks directory
- **Skills**: RAG retrieval skills and tools
- **MCP Server**: Model Context Protocol server
- **Core Code**: Symlinks to `src/core/` and `src/monitoring/`
- **Documentation**: README and plugin configuration

## Options

- `--dry-run` - Preview changes without applying them
- `--verbose` - Show detailed output of all operations
- `--force` - Force sync ignoring file timestamps
- `--no-backup` - Skip backup creation
- `--no-symlink` - Use copy mode instead of symlinks

## Advanced

### Windows Admin Required for Symlinks

If you see "admin required" error:
1. Run Claude Code with administrator privileges, or
2. Use `--no-symlink` flag to copy files instead

### Restore from Backup

If something goes wrong:
1. Find the backup in `~/.claude/plugins/rag-cli_backup_TIMESTAMP/`
2. Restore manually if needed
3. Run `/update-rag` again

## Performance

- **First sync**: 2-5 seconds
- **Typical sync**: 500ms - 1 second
- **Network impact**: None (local only)
- **Background**: Non-blocking operation

## Troubleshooting

### Command Not Found

Make sure your RAG-CLI project is properly installed:
```bash
cd path/to/RAG-CLI
pip install -e .
```

### Sync Fails

1. Check file permissions: `ls -la ~/.claude/plugins/rag-cli`
2. Verify Claude directory exists: `ls ~/.claude`
3. Try with `--verbose` flag for details
4. Check logs: `tail ~/.claude/plugins/rag-cli/logs/rag-cli.log`

### Changes Not Appearing

After sync:
1. Close and reopen Claude Code
2. Reload the plugin in Claude Code
3. Restart Claude Code if using symlinks

## Related Commands

- `/rag:enable` - Enable automatic RAG enhancement
- `/rag:disable` - Disable automatic RAG enhancement
- `/search` - Manually search indexed documents

## See Also

- [PLUGIN_SYNC_README.md](../../PLUGIN_SYNC_README.md) - Complete sync guide
- [CLAUDE.md](../../CLAUDE.md) - Project configuration
- [README.md](../../README.md) - Project overview

## Notes

- Sync preserves your runtime configuration and logs
- Safe to run frequently without data loss
- Automatic backups created with each sync
- Changes are applied immediately after sync completes
