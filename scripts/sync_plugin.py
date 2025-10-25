#!/usr/bin/env python3
"""Plugin synchronization script for RAG-CLI.

This script synchronizes RAG-CLI plugin files with the Claude Code installation.
It copies commands, hooks, skills, and MCP configuration to the appropriate directories.
"""

import os
import sys
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLAUDE_DIR = Path.home() / '.claude'
PLUGIN_DIR = CLAUDE_DIR / 'plugins' / 'rag-cli'

# Source directories
SRC_COMMANDS = PROJECT_ROOT / 'src' / 'plugin' / 'commands'
SRC_HOOKS = PROJECT_ROOT / 'src' / 'plugin' / 'hooks'
SRC_SKILLS = PROJECT_ROOT / 'src' / 'plugin' / 'skills'
SRC_MCP = PROJECT_ROOT / 'src' / 'plugin' / 'mcp'

# Destination directories
DEST_COMMANDS = CLAUDE_DIR / 'commands'
DEST_HOOKS = CLAUDE_DIR / 'hooks' / 'rag-cli'
DEST_SKILLS = CLAUDE_DIR / 'skills' / 'rag-cli'
DEST_MCP = CLAUDE_DIR / 'mcp'


def create_backup(dest_dir: Path) -> Path:
    """Create a timestamped backup of the destination directory.

    Args:
        dest_dir: Directory to backup

    Returns:
        Path to backup directory
    """
    if not dest_dir.exists():
        return None

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = dest_dir.parent / f"{dest_dir.name}_backup_{timestamp}"

    try:
        shutil.copytree(dest_dir, backup_dir)
        print(f"  Created backup: {backup_dir}")
        return backup_dir
    except Exception as e:
        print(f"  Backup failed: {e}")
        return None


def sync_directory(src: Path, dest: Path, pattern: str = '*') -> Tuple[int, int]:
    """Synchronize files from source to destination.

    Args:
        src: Source directory
        dest: Destination directory
        pattern: File pattern to match

    Returns:
        Tuple of (files_copied, files_skipped)
    """
    if not src.exists():
        print(f"⚠ Source directory not found: {src}")
        return 0, 0

    # Create destination if it doesn't exist
    dest.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    for src_file in src.glob(pattern):
        if src_file.is_file():
            dest_file = dest / src_file.name

            try:
                # Check if file needs updating
                needs_update = (
                    not dest_file.exists() or
                    src_file.stat().st_mtime > dest_file.stat().st_mtime
                )

                if needs_update:
                    shutil.copy2(src_file, dest_file)
                    print(f"  + {src_file.name}")
                    copied += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"  - {src_file.name}: {e}")

    return copied, skipped


def sync_directory_recursive(src: Path, dest: Path) -> Tuple[int, int]:
    """Recursively synchronize a directory tree.

    Args:
        src: Source directory
        dest: Destination directory

    Returns:
        Tuple of (files_copied, files_skipped)
    """
    if not src.exists():
        print(f"  WARNING: Source directory not found: {src}")
        return 0, 0

    dest.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    for src_path in src.rglob('*'):
        if src_path.is_file():
            # Calculate relative path and create corresponding dest path
            rel_path = src_path.relative_to(src)
            dest_path = dest / rel_path

            # Create parent directories if needed
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                needs_update = (
                    not dest_path.exists() or
                    src_path.stat().st_mtime > dest_path.stat().st_mtime
                )

                if needs_update:
                    shutil.copy2(src_path, dest_path)
                    print(f"  + {rel_path}")
                    copied += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"  - {rel_path}: {e}")

    return copied, skipped


def create_mcp_config():
    """Create MCP server configuration if it doesn't exist."""
    mcp_config_path = DEST_MCP / 'rag-cli.json'

    if mcp_config_path.exists():
        print("  + MCP config already exists")
        return

    DEST_MCP.mkdir(parents=True, exist_ok=True)

    config = {
        "command": "python",
        "args": [
            "-m",
            "src.monitoring.service_manager"
        ],
        "cwd": str(PROJECT_ROOT),
        "env": {
            "PYTHONUNBUFFERED": "1",
            "RAG_CLI_MODE": "claude_code"
        }
    }

    with open(mcp_config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"  + Created MCP config: {mcp_config_path}")


def main():
    """Main synchronization function."""
    print("\n" + "="*60)
    print("RAG-CLI Plugin Synchronization")
    print("="*60 + "\n")

    # Check if Claude directory exists
    if not CLAUDE_DIR.exists():
        print(f"ERROR: Claude directory not found: {CLAUDE_DIR}")
        print("  Make sure Claude Code is installed.")
        sys.exit(1)

    print(f"Source: {PROJECT_ROOT}")
    print(f"Destination: {CLAUDE_DIR}\n")

    total_copied = 0
    total_skipped = 0

    # Sync commands
    print("[Commands] Syncing...")
    copied, skipped = sync_directory(SRC_COMMANDS, DEST_COMMANDS, '*.md')
    total_copied += copied
    total_skipped += skipped
    print(f"   {copied} copied, {skipped} skipped\n")

    # Sync hooks
    print("[Hooks] Syncing...")
    copied, skipped = sync_directory(SRC_HOOKS, DEST_HOOKS, '*.py')
    total_copied += copied
    total_skipped += skipped
    print(f"   {copied} copied, {skipped} skipped\n")

    # Sync skills (recursive)
    print("[Skills] Syncing...")
    copied, skipped = sync_directory_recursive(SRC_SKILLS, DEST_SKILLS)
    total_copied += copied
    total_skipped += skipped
    print(f"   {copied} copied, {skipped} skipped\n")

    # Create MCP configuration
    print("[MCP] Setting up MCP server...")
    create_mcp_config()
    print()

    # Summary
    print("="*60)
    print(f"SUCCESS: Sync completed!")
    print(f"  Files updated: {total_copied}")
    print(f"  Files unchanged: {total_skipped}")
    print("="*60 + "\n")

    print("Next steps:")
    print("  1. Restart Claude Code to load the updated plugin")
    print("  2. Run /rag:enable to enable RAG enhancement")
    print("  3. Test with /search command or a query\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nERROR: Sync cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Sync failed: {e}")
        sys.exit(1)
