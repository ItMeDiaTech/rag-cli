# RAG-CLI Plugin Sync Guide

This guide explains how to use the plugin synchronization system to keep your Claude Code plugin updated with the latest RAG-CLI implementation.

## Overview

The plugin sync system automatically synchronizes your RAG-CLI plugin files from the development directory to the global Claude Code configuration directory (`~/.claude`). It includes:

- **Smart merge**: Preserves runtime files while syncing code
- **Automatic backup**: Creates timestamped backups before each sync
- **Symlink support**: Links core modules for single source of truth
- **Dry-run mode**: Preview changes before applying them
- **Detailed reporting**: Shows exactly what changed

## File Structure

After sync, your Claude plugin directory will look like:

```
~/.claude/
 plugins/
    rag-cli/
        commands/                 (synced from src/plugin/commands)
           rag-enable.md
           rag-disable.md
           search.md
        hooks/                    (synced from src/plugin/hooks)
           user-prompt-submit.py
        skills/                   (synced from src/plugin/skills)
           rag-retrieval/
               retrieve.py
               SKILL.md
        mcp/                      (synced from src/plugin/mcp)
           server.py
        src/
           core/ -> (symlink to RAG-CLI/src/core)
           monitoring/ -> (symlink to RAG-CLI/src/monitoring)
        plugin.json               (synced)
        README.md                 (synced)
        requirements.txt           (synced)
        logs/                     (preserved - runtime files)
 commands/
     rag-enable.md                (synced)
     rag-disable.md               (synced)
     search.md                    (synced)
```

## Usage

### Quick Start (Windows)

Simply double-click the batch file:

```
update_plugin.bat
```

This will perform a regular sync with automatic backup.

### Command Line

#### Python Script

From the RAG-CLI directory:

```bash
# Normal sync
python sync_plugin.py

# Preview changes (dry-run)
python sync_plugin.py --dry-run

# Sync with detailed output
python sync_plugin.py --verbose

# Force sync (ignore file timestamps)
python sync_plugin.py --force

# Skip backup creation
python sync_plugin.py --no-backup

# Use copy instead of symlinks (if admin required)
python sync_plugin.py --no-symlink
```

#### Batch File (Windows)

```bash
# Normal sync
update_plugin.bat

# Preview changes
update_plugin.bat --dry-run

# Detailed output
update_plugin.bat --verbose

# Help
update_plugin.bat --help
```

#### After Installation

Once installed via `pip install -e .`:

```bash
# From anywhere
rag-sync                  # Normal sync
rag-sync --dry-run       # Preview changes
rag-sync --verbose       # Detailed output
```

## Common Scenarios

### Scenario 1: Preview Changes Before Sync

```bash
python sync_plugin.py --dry-run --verbose
```

Output shows:
- New files to be copied
- Files to be updated
- Obsolete files to be deleted
- Symlinks to be created
- No files are actually modified

### Scenario 2: Sync After Adding New Command

If you added a new slash command file at `src/plugin/commands/my-command.md`:

```bash
python sync_plugin.py
```

Result:
- New file copied to `~/.claude/commands/my-command.md`
- Backup created automatically
- Command available immediately in Claude Code

### Scenario 3: Update Core Module

If you modified `src/core/retrieval_pipeline.py`:

```bash
python sync_plugin.py --verbose
```

Result:
- Symlink to core/ already points to latest version
- No action needed (symlink is live)
- Changes automatically visible in Claude

### Scenario 4: Force Sync Everything

To ignore file timestamps and sync all files:

```bash
python sync_plugin.py --force
```

Useful when:
- File modification times are unreliable
- Forcing a complete refresh
- Troubleshooting sync issues

## Preserved Files

The sync system automatically preserves these files in the plugin directory:

- `logs/` - Runtime log directory
- `*.log` - Log files
- `*state.json` - Plugin state files
- `config.json` - Plugin configuration
- `__pycache__/` - Python cache

These are never deleted or overwritten, ensuring runtime data is safe.

## Symlinks vs Copy

### Symlinks (Default)

```bash
python sync_plugin.py
```

**Advantages:**
- Single source of truth
- Changes to source automatically visible
- No duplication
- Minimal disk usage

**Disadvantages:**
- Requires admin privileges on Windows
- May not work across different drives
- Breaks if source moves

### Copy Mode

```bash
python sync_plugin.py --no-symlink
```

**Advantages:**
- Works without admin privileges
- Portable across drives
- No path dependencies

**Disadvantages:**
- Requires re-sync after source changes
- Duplicates code in two places
- More disk space

## Backup and Recovery

### Automatic Backups

Each sync creates a backup:

```
~/.claude/plugins/rag-cli_backup_20251025_025821/
```

### Manual Recovery

If something goes wrong, restore from backup:

```bash
# Copy backup back to active plugin
cp -r ~/.claude/plugins/rag-cli_backup_20251025_025821/* ~/.claude/plugins/rag-cli/
```

### Skip Backup

For performance (not recommended):

```bash
python sync_plugin.py --no-backup
```

## Troubleshooting

### Admin Required (Windows Symlinks)

Error: "Cannot create symlink (admin required)"

**Solution:**
1. Run Command Prompt as Administrator
2. Run sync script again
3. Or use `--no-symlink` flag

```bash
python sync_plugin.py --no-symlink
```

### Files Not Syncing

Check if files are excluded:

```bash
python sync_plugin.py --verbose --dry-run
```

If files appear in "Files to copy" but don't sync:
- Check file permissions
- Verify Claude directory is writable
- Try `--force` flag

### Symlink Not Working

If symlinks break:

1. Check if source path still exists:
   ```bash
   dir "C:\Users\DiaTech\Pictures\DiaTech\Programs\DocHub\development\RAG-CLI\src\core"
   ```

2. Delete the broken symlink:
   ```bash
   rmdir ~/.claude/plugins/rag-cli/src/core
   ```

3. Re-sync:
   ```bash
   python sync_plugin.py
   ```

### Claude Not Seeing Updates

After syncing, Claude Code might need to:

1. Reload the plugin (restart Claude Code)
2. Refresh the command list
3. Clear the cache

Try:
1. Close Claude Code completely
2. Wait 30 seconds
3. Reopen Claude Code

## Configuration

### Custom Claude Directory

If Claude is installed elsewhere:

```bash
python sync_plugin.py --claude-dir "D:\my-claude-config"
```

### Save Preferred Options

Create `.sync_config.json` in the RAG-CLI directory:

```json
{
  "verbose": true,
  "force": false,
  "backup": true,
  "no_symlink": false,
  "claude_dir": "~/.claude"
}
```

Then sync with saved options:

```bash
python sync_plugin.py
```

## Advanced Usage

### Dry-Run with Verbose Logging

See exactly what would happen:

```bash
python sync_plugin.py --dry-run --verbose > sync_preview.txt
```

Then review `sync_preview.txt` to see all changes.

### Scheduled Sync

Create a Windows Task Scheduler job:

1. Open Task Scheduler
2. Create Basic Task
3. Set trigger (e.g., daily)
4. Set action: `update_plugin.bat`
5. Run with highest privileges (for symlinks)

### Integration with Development Workflow

Add to your git pre-commit hook:

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Sync plugin before committing
python sync_plugin.py --verbose

if [ $? -ne 0 ]; then
    echo "Plugin sync failed!"
    exit 1
fi
```

## Performance

### Sync Times

- First sync: 2-5 seconds
- Subsequent syncs: 500ms - 1s
- With `--force`: 1-2 seconds
- With `--verbose`: 1-2 seconds additional

### Backup Size

- Backup size: Same as plugin directory (typically 5-10 MB)
- Space used: Multiple backups kept, cleanup old ones periodically:

```bash
# Remove old backups (keep last 5)
# Manual: Delete oldest backup folders in ~/.claude/plugins/
```

## FAQ

**Q: Do I need admin privileges?**
A: Only for symlinks. Use `--no-symlink` for copy mode.

**Q: How often should I sync?**
A: After modifying plugin files, or when updating RAG-CLI.

**Q: Will sync overwrite my Claude settings?**
A: No, only plugin code and commands are synced. Config and logs are preserved.

**Q: Can I sync to a different Claude directory?**
A: Yes, use `--claude-dir` flag or config file.

**Q: What if I want two-way sync?**
A: Not supported. Edits in Claude directory won't sync back to source.

**Q: Can I see git history of syncs?**
A: Backups are created with timestamps. To restore:
```bash
cp -r ~/.claude/plugins/rag-cli_backup_TIMESTAMP/* ~/.claude/plugins/rag-cli/
```

## Support

For issues or questions:

1. Check this guide
2. Run with `--verbose` for detailed logs
3. Review `sync_preview.txt` from dry-run
4. Check Claude Code logs in `~/.claude/debug/`

## See Also

- [RAG-CLI README](README.md)
- [CLAUDE.md](CLAUDE.md) - Project configuration
- [RAG-CLI Implementation Guide](RAG-implementation.md)
