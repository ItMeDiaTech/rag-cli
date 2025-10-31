# No Emoji Policy for RAG-CLI

## Overview

RAG-CLI maintains a strict **no-emoji policy** for all production code, documentation, and output. This document explains the rationale, implementation, and enforcement of this policy.

## Rationale

### 1. Windows Terminal Compatibility

**Primary Issue**: Emoji characters cause `UnicodeEncodeError` on Windows terminals using CP1252 encoding.

```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2705' in position 43: character maps to <undefined>
```

Windows Command Prompt and PowerShell use CP1252 encoding by default, which cannot display most Unicode emoji characters. This causes runtime errors when trying to print output containing emojis.

### 2. Professional Documentation Standards

- **Clarity**: Text-based indicators like `[OK]`, `[ERROR]`, and `SUCCESS:` are universally readable
- **Consistency**: ASCII characters render identically across all platforms, terminals, and editors
- **Accessibility**: Screen readers and text-to-speech tools handle ASCII better than Unicode emojis

### 3. Cross-Platform Compatibility

- Different terminals have varying levels of emoji support
- Some terminals show emoji as boxes, question marks, or nothing
- ASCII alternatives work everywhere without configuration

## Policy Details

### Prohibited

Emoji characters are **strictly prohibited** in:

- Python source code (`.py` files)
- Markdown documentation (`.md` files)
- Configuration files
- User-facing output from commands
- Log messages
- Error messages
- Status indicators

### Approved Alternatives

Instead of emojis, use these text-based alternatives:

| Emoji | Replacement | Usage |
|-------|-------------|-------|
| ✅ | `[OK]`, `SUCCESS:`, `ENABLED` | Success indicators |
| ❌ | `[ERROR]`, `ERROR:`, `DISABLED` | Error indicators |
| ⚠️ | `[WARNING]`, `WARNING:` | Warning messages |
| 🔍 | `[SEARCH]` | Search operations |
| 📝 | `[NOTE]` | Notes or documentation |
| 🚀 | `[LAUNCH]` | Start/launch operations |
| → | `->` | Right arrow |
| ← | `<-` | Left arrow |

### Exceptions

The following files are **exempt** from the policy (for valid technical reasons):

1. `scripts/remove_emojis.py` - Contains emoji mappings for replacement
2. `scripts/validate_no_emojis.py` - Contains emoji patterns for detection
3. `scripts/verify_installation.py` - Uses emojis for user-facing terminal output (intentional)
4. `tests/**/*.py` - Test files may contain emojis for testing Unicode handling

## Implementation

### Emoji Removal Script

**Location**: `scripts/remove_emojis.py`

**Purpose**: Systematically removes all emoji characters from Python and Markdown files.

**Usage**:
```bash
python scripts/remove_emojis.py
```

**Features**:
- Processes Python files (`.py`) and Markdown files (`.md`)
- Context-sensitive replacements
- Handles status messages, list items, and inline text
- Reports files modified and emoji count

**Results** (v1.2.0):
- Files processed: 154
- Files modified: 45
- Total emojis removed: 984

### Validation Script

**Location**: `scripts/validate_no_emojis.py`

**Purpose**: Validates that no emoji characters exist in the codebase.

**Usage**:
```bash
python scripts/validate_no_emojis.py
```

**Features**:
- Scans Python and Markdown files
- Excludes test files and utility scripts
- Windows terminal-safe error reporting
- Returns exit code 0 (success) or 1 (failure)

**Exit Codes**:
- `0`: No emojis found - validation passed
- `1`: Emojis detected - validation failed

### Pre-Commit Hook

**Location**: `.githooks/pre-commit`

**Purpose**: Automatically validates emoji-free code before allowing commits.

**Setup**:
```bash
# Method 1: Configure git to use .githooks
git config core.hooksPath .githooks

# Method 2: Use setup script
python scripts/setup_git_hooks.py
```

**Behavior**:
- Runs `scripts/validate_no_emojis.py` before each commit
- Blocks commit if emojis are detected
- Provides instructions to fix violations

## Enforcement

### Developer Workflow

1. **Write Code**: Avoid using emojis in all code and documentation
2. **Pre-Commit**: Git hook automatically validates on commit
3. **Fix Violations**: If detected, run `python scripts/remove_emojis.py`
4. **Re-Commit**: Commit passes after emojis are removed

### CI/CD Integration

Add to your CI pipeline:

```yaml
# Example GitHub Actions workflow
- name: Validate No Emojis
  run: python scripts/validate_no_emojis.py
```

### Manual Validation

Developers can manually validate at any time:

```bash
# Check for emojis
python scripts/validate_no_emojis.py

# Fix detected emojis
python scripts/remove_emojis.py

# Verify fix
python scripts/validate_no_emojis.py
```

## Migration Guide

### For Existing Code

If you're migrating existing code with emojis:

1. **Backup**: Create a git commit before changes
2. **Run Removal**: `python scripts/remove_emojis.py`
3. **Review Changes**: Manually review replacements for context
4. **Test**: Ensure commands work without Unicode errors
5. **Commit**: Commit the emoji-free code

### For New Code

When writing new code:

1. **Avoid Emojis**: Don't use emoji characters in the first place
2. **Use Alternatives**: Use text-based indicators from the approved list
3. **Status Messages**: Use `SUCCESS:`, `ERROR:`, `WARNING:` prefixes
4. **List Items**: Use `[OK]` and `[ERROR]` for status lists

## Tools Summary

| Tool | Purpose | Command |
|------|---------|---------|
| `remove_emojis.py` | Remove all emojis from codebase | `python scripts/remove_emojis.py` |
| `validate_no_emojis.py` | Validate emoji-free code | `python scripts/validate_no_emojis.py` |
| `setup_git_hooks.py` | Install pre-commit hooks | `python scripts/setup_git_hooks.py` |
| `.githooks/pre-commit` | Pre-commit validation hook | (automatic on commit) |

## Benefits

### Immediate

- [OK] No more `UnicodeEncodeError` on Windows terminals
- [OK] Consistent output across all platforms
- [OK] Professional, clear documentation

### Long-Term

- [OK] Easier maintenance (no encoding issues)
- [OK] Better accessibility for all users
- [OK] Automated enforcement prevents regression

## References

- Project Policy: `CLAUDE.md` - No Emojis Policy section
- Python PEP 597: https://peps.python.org/pep-0597/ (Encoding standards)
- Windows Code Pages: https://learn.microsoft.com/en-us/windows/win32/intl/code-page-identifiers

## Version History

- **v1.2.0** (2025-10-31): Initial implementation
  - Created emoji removal and validation scripts
  - Implemented pre-commit hooks
  - Removed 984 emojis from 45 files
  - Documented policy and enforcement

---

**Status**: ACTIVE
**Last Updated**: 2025-10-31
**Maintained By**: RAG-CLI Development Team
