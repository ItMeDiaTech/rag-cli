# Update RAG-CLI Plugin Skill

## Overview

The `update-rag` skill provides a Claude Code skill to synchronize your RAG-CLI plugin files with the Claude Code configuration directory.

## Name

`update-rag` or `rag-update`

## Description

Automatically synchronizes RAG-CLI plugin implementation from your development directory to your Claude Code installation. Handles code updates, backup creation, symlinks, and cleanup.

## Capabilities

- **Smart Sync**: Intelligently copies new/updated files while preserving runtime files
- **Automatic Backup**: Creates timestamped backups before syncing
- **Symlinks**: Links core modules for real-time code updates
- **Dry-Run Mode**: Preview changes before applying them
- **Cleanup**: Removes obsolete files automatically
- **Error Handling**: Graceful fallback and detailed error reporting

## Input Parameters

### Basic Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `dry_run` | boolean | No | false | Preview changes without applying them |
| `verbose` | boolean | No | false | Show detailed output |
| `force` | boolean | No | false | Force sync (ignore file timestamps) |

### Advanced Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `no_backup` | boolean | No | false | Skip automatic backup creation |
| `no_symlink` | boolean | No | false | Use copy mode instead of symlinks |
| `preview` | boolean | No | false | Alias for `dry_run` |

## Output

Returns a dictionary with:

```json
{
  "success": true,
  "return_code": 0,
  "output": "... sync output ...",
  "error": null,
  "message": "Plugin sync completed"
}
```

## Usage Examples

### Example 1: Simple Update

```
I need to update the RAG plugin with my latest code changes.
```

The skill will:
1. Backup current plugin
2. Sync all plugin files
3. Create symlinks for core modules
4. Report what changed

### Example 2: Preview Changes

```
Show me what would change if I sync the RAG plugin.
```

The skill will:
1. Run in dry-run mode
2. Show all files that would be copied/updated/deleted
3. Show symlinks that would be created
4. Not make any actual changes

### Example 3: Force Sync

```
Force a complete sync of the RAG plugin, ignoring timestamps.
```

The skill will:
1. Sync all files regardless of modification times
2. Ensure all code is up to date
3. Update any stale files

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| Timeout | Sync took longer than 60 seconds | Check file system, try `--no-backup` |
| Admin Required | Symlinks need admin on Windows | Run as admin or use `--no-symlink` |
| File Not Found | Sync script missing | Reinstall RAG-CLI or check path |
| Permission Denied | Cannot write to `.claude` directory | Check directory permissions |

### Debug Output

For troubleshooting, use:
```
verbose=true, dry_run=true
```

This shows detailed output of all operations without making changes.

## Performance

- **Typical Sync**: 500ms - 2 seconds
- **First Sync**: 2-5 seconds
- **Timeout**: 60 seconds
- **Backup Time**: ~1-2 seconds

## Configuration

The skill can be configured via:

1. **Direct parameters**: Pass options when invoking
2. **Config file**: `.sync_config.json` in RAG-CLI directory
3. **Environment**: `RAG_SYNC_*` environment variables

## Related

- **Command**: `/update-rag` - Slash command version
- **Script**: `sync_plugin.py` - Underlying sync implementation
- **Guide**: `PLUGIN_SYNC_README.md` - Complete documentation

## Skill Location

```
src/plugin/skills/update-rag/update.py
```

## Requirements

- Python 3.8+
- Access to RAG-CLI project directory
- Write access to `~/.claude/plugins/rag-cli/`

## Invocation

### From Claude Code

When this skill is registered with Claude Code, you can invoke it as:

```
@invoke update-rag
@invoke update-rag --verbose
@invoke update-rag --dry-run --verbose
```

### From Python

```python
from src.plugin.skills.update_rag.update import UpdateRagSkill

skill = UpdateRagSkill()
result = skill.execute(SyncOptions(dry_run=True, verbose=True))
print(result['output'])
```

### From Command Line

```bash
python src/plugin/skills/update-rag/update.py --dry-run --verbose
```

## Returns

On success:
```json
{
  "success": true,
  "return_code": 0,
  "output": "... detailed sync report ...",
  "error": null,
  "message": "Plugin sync completed"
}
```

On failure:
```json
{
  "success": false,
  "return_code": 1,
  "output": "",
  "error": "Error message",
  "message": "Plugin sync failed"
}
```

## Notes

- Skill automatically creates backups before syncing
- Preserved files (logs, config, state) are never overwritten
- Symlinks point to development directory for real-time updates
- Safe to run frequently without data loss
- Changes are applied immediately after sync completes

## Troubleshooting

### Skill not found

Ensure RAG-CLI is properly installed:
```bash
cd /path/to/RAG-CLI
pip install -e .
```

### Sync always fails

Try with verbose and dry-run:
```python
result = skill.execute(SyncOptions(dry_run=True, verbose=True))
```

### Admin required on Windows

For symlinks, run Claude Code as administrator, or use `no_symlink=True`.

## See Also

- [PLUGIN_SYNC_README.md](../../../PLUGIN_SYNC_README.md)
- [CLAUDE.md](../../../CLAUDE.md)
- [/update-rag command documentation](../commands/update-rag.md)
