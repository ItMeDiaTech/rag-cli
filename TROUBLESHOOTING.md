# RAG-CLI Troubleshooting Guide

Common issues and solutions when installing or using the RAG-CLI plugin.

## Installation Issues

### Plugin Validation Errors

**Error**: `Failed to load hooks from hooks.json`

**Cause**: The hooks.json file has an invalid structure or your installed version is outdated.

**Solution**:
```bash
# Reinstall to get the latest version
/plugin uninstall rag-cli
/plugin marketplace remove ItMeDiaTech/rag-cli
/plugin marketplace add ItMeDiaTech/rag-cli
/plugin install rag-cli
```

**Verification**: After reinstalling, the hooks.json should have this structure:
```json
{
  "hooks": {
    "UserPromptSubmit": [...],
    "ResponsePost": [...]
  }
}
```

---

### Invalid Plugin Manifest

**Error**: `Plugin rag-cli has an invalid manifest file`

**Causes**:
- Outdated plugin.json schema
- Incorrect author field format
- Invalid hooks reference

**Solution**:
```bash
# Pull latest changes
cd rag-cli
git pull origin master

# Reinstall plugin
/plugin uninstall rag-cli
/plugin marketplace remove ItMeDiaTech/rag-cli
/plugin marketplace add ItMeDiaTech/rag-cli
/plugin install rag-cli
```

---

### MCP Server Failed to Start

**Error**: `Status: failed, Args: -m src.monitoring.service_manager`

**Cause**: Old installation using wrong module path.

**Solution**:
1. Close Claude Code completely
2. Run cleanup script:
   ```cmd
   cd C:\Path\To\RAG-CLI
   CLEANUP.bat
   ```
3. Restart Claude Code
4. Reinstall plugin (see above)

**Expected Result**: MCP server should use `python -m src.plugin.mcp.unified_server`

---

### Duplicate Commands

**Problem**: Commands appear multiple times in Claude Code (/search, /rag-enable, etc.)

**Cause**: Old manual installation conflicting with marketplace installation.

**Solution**:
```bash
# Remove old installations
rm -rf ~/.claude/hooks/rag-cli
rm -rf ~/.claude/plugins/rag-cli
rm ~/.claude/mcp/rag-cli.json
rm ~/.claude/commands/*rag*.md
rm ~/.claude/commands/search.md

# Reinstall from marketplace only
/plugin uninstall rag-cli
/plugin install rag-cli
```

---

## Runtime Issues

### RAG Not Enhancing Queries

**Problem**: Queries not automatically enhanced with document context.

**Checks**:
1. **Is RAG enabled?**
   ```bash
   cat config/rag_settings.json | grep enabled
   ```
   Should show: `"enabled": true`

2. **Are documents indexed?**
   ```bash
   ls data/vectors/
   ```
   Should contain: `faiss_index` or similar

3. **Is query long enough?**
   - Default minimum: 5 words
   - Check: `cat config/rag_settings.json | grep auto_trigger_threshold`

**Solutions**:
```bash
# Enable RAG
/rag-cli:rag-enable

# Index documents
python scripts/index.py --input data/documents

# Lower threshold
# Edit config/rag_settings.json, set "auto_trigger_threshold": 3
```

---

### Hook Not Triggering

**Problem**: user-prompt-submit.py hook not intercepting queries.

**Checks**:
1. **Is hook registered?**
   ```bash
   cat ~/.claude/plugins/marketplaces/rag-cli/.claude-plugin/hooks.json
   ```

2. **Is hook executable?**
   ```bash
   chmod +x ~/.claude/plugins/marketplaces/rag-cli/src/plugin/hooks/*.py
   ```

3. **Check hook logs**:
   ```bash
   tail -f ~/.claude/plugins/rag-cli/logs/rag-cli.log
   ```

---

### Import Errors in MCP Server

**Error**: `ModuleNotFoundError: No module named 'src'`

**Cause**: PYTHONPATH not set correctly.

**Solution**: Verify plugin.json mcpServers configuration includes:
```json
{
  "mcpServers": {
    "rag-server": {
      "env": {
        "PYTHONPATH": "${CLAUDE_PLUGIN_ROOT}",
        "RAG_CLI_ROOT": "${CLAUDE_PLUGIN_ROOT}"
      }
    }
  }
}
```

If incorrect, pull latest changes and reinstall.

---

## Validation

### Pre-Release Validation

Before reporting issues, run the validation script:

```bash
cd rag-cli
python scripts/validate_plugin.py
```

Expected output:
```
[OK] plugin.json is valid
[OK] marketplace.json is valid
[OK] hooks.json is valid
[OK] All required files present
SUCCESS: All validations passed!
```

If validation fails, the repository has an issue. Report at:
https://github.com/ItMeDiaTech/rag-cli/issues

---

## Clean Install Procedure

For a completely clean installation:

1. **Close Claude Code**

2. **Remove all RAG-CLI files**:
   ```bash
   # Run cleanup script
   cd /path/to/rag-cli
   ./CLEANUP.bat  # Windows
   # or
   bash CLEANUP.sh  # Mac/Linux

   # Or manually:
   rm -rf ~/.claude/plugins/rag-cli*
   rm -rf ~/.claude/plugins/marketplaces/rag-cli
   rm ~/.claude/mcp/rag-cli.json
   rm ~/.claude/hooks/rag-cli -r
   rm ~/.claude/commands/*rag*.md
   rm ~/.claude/commands/search.md
   ```

3. **Restart Claude Code**

4. **Fresh Install**:
   ```bash
   /plugin marketplace add ItMeDiaTech/rag-cli
   /plugin install rag-cli
   ```

---

## Getting Help

Still having issues?

1. **Check GitHub Issues**: https://github.com/ItMeDiaTech/rag-cli/issues
2. **Review Documentation**:
   - INSTALL.md - Installation guide
   - REINSTALL_GUIDE.md - Complete reinstall procedure
   - README.md - Feature overview
3. **File a Bug Report**: Include:
   - Error message (full text)
   - Output of `python scripts/validate_plugin.py`
   - Claude Code version
   - Operating system
   - Steps to reproduce
