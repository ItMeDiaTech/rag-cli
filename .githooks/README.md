# Git Hooks for RAG-CLI

This directory contains git hooks to maintain code quality standards for the RAG-CLI project.

## Available Hooks

### pre-commit

Validates that no emoji characters are present in the codebase before allowing commits.

**Checks:**
- Python files (.py) for emoji characters
- Markdown files (.md) for emoji characters
- Prevents accidental emoji introduction

## Installation

### Method 1: Configure Git to Use This Directory

```bash
git config core.hooksPath .githooks
```

This tells git to use the hooks in `.githooks/` directory instead of `.git/hooks/`.

### Method 2: Copy Hooks Manually

```bash
# On Unix/Linux/Mac
cp .githooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# On Windows (Git Bash)
cp .githooks/pre-commit .git/hooks/pre-commit
```

### Method 3: Use Setup Script

```bash
python scripts/setup_git_hooks.py
```

## Manual Validation

You can manually run the emoji validation at any time:

```bash
python scripts/validate_no_emojis.py
```

If emojis are detected, fix them with:

```bash
python scripts/remove_emojis.py
```

## Why No Emojis?

1. **Windows Terminal Compatibility**: Emoji characters cause UnicodeEncodeError on Windows terminals using CP1252 encoding
2. **Professional Standards**: Text-based indicators are more professional and universally readable
3. **Accessibility**: Screen readers and text-based tools handle ASCII better than Unicode emojis
4. **Consistency**: Plain text ensures consistent rendering across all platforms and editors

## Bypassing Hooks (Not Recommended)

If you absolutely must bypass the pre-commit hook:

```bash
git commit --no-verify
```

However, this is **strongly discouraged** as it violates project standards.
