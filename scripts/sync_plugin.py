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
        print(f"[WARNING] Source directory not found: {src}")
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
    """Create MCP server configuration."""
    mcp_config_path = DEST_MCP / 'rag-cli.json'

    # Always update to ensure correct path
    DEST_MCP.mkdir(parents=True, exist_ok=True)

    # Determine if this is a plugin installation or development mode
    # Plugin installation: Check if we're syncing TO the Claude plugins directory
    is_plugin_install = (PLUGIN_DIR.exists() and
                        PLUGIN_DIR.resolve() != PROJECT_ROOT.resolve())

    if is_plugin_install:
        # For plugin installations, use the installed plugin directory
        rag_root = str(PLUGIN_DIR)
        print(f"  + Using plugin installation path: {rag_root}")
    else:
        # For development mode, use the project root
        rag_root = str(PROJECT_ROOT)
        print(f"  + Using development path: {rag_root}")

    # Note: The MCP config created here is for manual installations only.
    # When installed via plugin system, the mcpServers field in plugin.json is used instead.
    config = {
        "command": "python",
        "args": [
            "-m",
            "src.plugin.mcp.unified_server"
        ],
        "cwd": rag_root,
        "env": {
            "PYTHONUNBUFFERED": "1",
            "RAG_CLI_MODE": "claude_code",
            "RAG_CLI_ROOT": rag_root,
            "PYTHONPATH": rag_root
        }
    }

    with open(mcp_config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"  + Created MCP config: {mcp_config_path}")


def initialize_config_files():
    """Initialize RAG-CLI configuration files if they don't exist."""
    config_dir = PROJECT_ROOT / 'config'
    config_dir.mkdir(parents=True, exist_ok=True)

    # Initialize rag_settings.json
    rag_settings_path = config_dir / 'rag_settings.json'
    if not rag_settings_path.exists():
        default_settings = {
            "enabled": True,
            "enable_agent_orchestration": True,
            "auto_trigger_threshold": 5,
            "context_limit": 3,
            "relevance_threshold": 0.6,
            "cache_queries": True,
            "exclude_patterns": [],
            "orchestration": {
                "enable_maf": True,
                "parallel_threshold_confidence": 0.7,
                "decomposition_complexity_threshold": 0.6,
                "maf_timeout": 30.0
            },
            "metadata": {
                "last_updated": datetime.now().isoformat(),
                "version": "2.0",
                "description": "RAG enhancement settings with multi-agent orchestration support"
            }
        }

        with open(rag_settings_path, 'w') as f:
            json.dump(default_settings, f, indent=2)

        print(f"  + Created RAG settings: {rag_settings_path}")
    else:
        print("  + RAG settings already exist")

    # Initialize default.yaml if it doesn't exist
    default_yaml_path = config_dir / 'default.yaml'
    if not default_yaml_path.exists():
        default_yaml_content = """# RAG-CLI Configuration

# Embedding settings
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  dimensions: 384
  cache_enabled: true

# Vector store settings
vector_store:
  type: "faiss"
  index_type: "IndexFlatL2"
  path: "data/vectors"

# Document processing
document_processing:
  chunk_size: 500
  chunk_overlap: 50
  supported_formats:
    - txt
    - md
    - pdf
    - docx
    - html

# Retrieval settings
retrieval:
  method: "hybrid"
  vector_weight: 0.7
  keyword_weight: 0.3
  reranking: true
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Claude integration
claude:
  model: "claude-haiku-4-5-20251001"
  max_tokens: 4096
  temperature: 0.7
  streaming: true

# Monitoring
monitoring:
  tcp_server: true
  port: 9999
  metrics_enabled: true
  log_rotation: true
"""
        with open(default_yaml_path, 'w') as f:
            f.write(default_yaml_content)

        print(f"  + Created default config: {default_yaml_path}")
    else:
        print("  + Default config already exists")

    # Create data directories
    (PROJECT_ROOT / 'data' / 'documents').mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / 'data' / 'vectors').mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / 'logs').mkdir(parents=True, exist_ok=True)

    print("  + Created data and log directories")


def configure_api_tokens(interactive: bool = True):
    """Configure API tokens from .env or user prompts.

    Args:
        interactive: Whether to prompt user for missing tokens
    """
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.core.token_manager import TokenManager

        print("\n[Tokens] Configuring API tokens...")

        token_manager = TokenManager(PROJECT_ROOT)
        tokens = token_manager.configure_tokens(interactive=interactive)

        # Update config file if tokens were provided
        if tokens['github_token'] or tokens['stackoverflow_key']:
            # Only update if user wants to (for backward compatibility)
            # Tokens in .env are preferred over config file
            pass

        if not tokens['github_token'] and not tokens['stackoverflow_key']:
            print("  [INFO] No API tokens configured (optional)")
            print("  [INFO] You can add them later to .env file")
        else:
            print("  [OK] API tokens configured successfully")

    except ImportError as e:
        print(f"  [WARNING] Token manager not available: {e}")
        print("  [INFO] You can manually add tokens to .env file later")
    except Exception as e:
        print(f"  [WARNING] Token configuration failed: {e}")
        print("  [INFO] You can manually add tokens to .env file later")


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

    # Sync core modules to plugin directory (needed for MCP server)
    print("[Core] Syncing core modules to plugin directory...")
    plugin_src_core = PLUGIN_DIR / 'src' / 'core'
    plugin_src_monitoring = PLUGIN_DIR / 'src' / 'monitoring'
    plugin_src_plugin = PLUGIN_DIR / 'src' / 'plugin'

    core_copied, core_skipped = sync_directory_recursive(PROJECT_ROOT / 'src' / 'core', plugin_src_core)
    mon_copied, mon_skipped = sync_directory_recursive(PROJECT_ROOT / 'src' / 'monitoring', plugin_src_monitoring)
    plug_copied, plug_skipped = sync_directory_recursive(PROJECT_ROOT / 'src' / 'plugin', plugin_src_plugin)

    total_copied += core_copied + mon_copied + plug_copied
    total_skipped += core_skipped + mon_skipped + plug_skipped
    print(f"   {core_copied + mon_copied + plug_copied} copied, {core_skipped + mon_skipped + plug_skipped} skipped\n")

    # Create MCP configuration
    print("[MCP] Setting up MCP server...")
    create_mcp_config()
    print()

    # Initialize config files
    print("[Config] Initializing configuration files...")
    initialize_config_files()
    print()

    # Configure API tokens
    configure_api_tokens(interactive=True)

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
