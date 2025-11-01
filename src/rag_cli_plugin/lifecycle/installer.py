"""
Marketplace installation handler for RAG-CLI plugin.

This module runs automatically after plugin installation via Claude Code marketplace
to install dependencies, initialize configuration, and verify the installation.

IMPORTANT: NO EMOJIS - All output must be professional text only.
"""

import sys
import subprocess
import json
from pathlib import Path
from typing import Optional


def get_plugin_root() -> Path:
    """Find plugin installation directory."""
    import os

    # Try environment variable first (set by Claude Code)
    plugin_root = os.environ.get('CLAUDE_PLUGIN_ROOT')
    if plugin_root:
        return Path(plugin_root)

    # Fallback: resolve from this file's location
    # This file is at: <plugin_root>/src/rag_cli_plugin/lifecycle/installer.py
    return Path(__file__).parent.parent.parent.parent


def install_dependencies() -> bool:
    """Install Python dependencies from requirements.txt"""
    plugin_root = get_plugin_root()
    requirements = plugin_root / "requirements.txt"

    if not requirements.exists():
        print(f"Warning: requirements.txt not found at {requirements}")
        return False

    print("Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "-r", str(requirements),
            "--quiet"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("  Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error installing dependencies: {e}")
        return False


def initialize_config() -> bool:
    """Copy default configs if not present."""
    plugin_root = get_plugin_root()
    config_dir = plugin_root / "config"
    defaults_dir = config_dir / "defaults"

    if not defaults_dir.exists():
        print(f"Warning: defaults directory not found at {defaults_dir}")
        return False

    configs_to_copy = [
        "mcp.json",
        "rag_settings.json",
        "services.json"
    ]

    print("Initializing configuration...")
    success = True

    for config_name in configs_to_copy:
        target = config_dir / config_name
        if not target.exists():
            default = defaults_dir / config_name
            if default.exists():
                import shutil
                try:
                    shutil.copy2(default, target)
                    print(f"  Created config: {config_name}")
                except Exception as e:
                    print(f"  Error copying {config_name}: {e}")
                    success = False
            else:
                print(f"  Warning: Default config not found: {config_name}")
        else:
            print(f"  Config already exists: {config_name}")

    return success


def initialize_data_directories() -> bool:
    """Create data/vectors/cache directories."""
    plugin_root = get_plugin_root()

    dirs = [
        plugin_root / "data" / "vectors",
        plugin_root / "data" / "cache",
        plugin_root / "data" / "documents",
        plugin_root / "logs"
    ]

    print("Creating data directories...")
    success = True

    for dir_path in dirs:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  Created: {dir_path.relative_to(plugin_root)}")
        except Exception as e:
            print(f"  Error creating {dir_path}: {e}")
            success = False

    return success


def verify_installation() -> bool:
    """Run health check to verify installation."""
    print("Verifying installation...")

    try:
        # Try importing core modules
        from rag_cli.core import vector_store, embeddings
        print("  Core library imports OK")

        # Try importing plugin modules
        from rag_cli_plugin.mcp import unified_server
        print("  Plugin imports OK")

        # Check if required files exist
        plugin_root = get_plugin_root()
        required_files = [
            plugin_root / "config" / "rag_settings.json",
            plugin_root / "data",
            plugin_root / "logs"
        ]

        for file_path in required_files:
            if file_path.exists():
                print(f"  Found: {file_path.relative_to(plugin_root)}")
            else:
                print(f"  Missing: {file_path.relative_to(plugin_root)}")
                return False

        print("  Installation verified successfully")
        return True

    except ImportError as e:
        print(f"  Import verification failed: {e}")
        return False
    except Exception as e:
        print(f"  Verification failed: {e}")
        return False


def print_usage_instructions():
    """Print usage instructions after successful installation."""
    print("\n" + "=" * 60)
    print("RAG-CLI installed successfully!")
    print("=" * 60)
    print("\nQuick Start:")
    print("  1. Enable RAG enhancement in Claude Code settings")
    print("  2. Use /rag-project to index your current project")
    print("  3. Ask questions - RAG will enhance responses automatically")
    print("\nCommands:")
    print("  /rag-project <path>  - Index a project for RAG retrieval")
    print("  /update-rag          - Update to latest version")
    print("  /rag-enable          - Enable RAG enhancement")
    print("  /rag-disable         - Disable RAG enhancement")
    print("\nFor more information, see the README.md file.")
    print("=" * 60)


def main():
    """Main installation entrypoint."""
    print("\n" + "=" * 60)
    print("RAG-CLI Marketplace Installation")
    print("=" * 60)

    success = True

    try:
        print("\n[1/4] Installing dependencies...")
        if not install_dependencies():
            print("  Warning: Dependency installation incomplete")
            success = False

        print("\n[2/4] Initializing configuration...")
        if not initialize_config():
            print("  Warning: Configuration initialization incomplete")
            success = False

        print("\n[3/4] Creating data directories...")
        if not initialize_data_directories():
            print("  Warning: Directory creation incomplete")
            success = False

        print("\n[4/4] Verifying installation...")
        if verify_installation():
            print_usage_instructions()
            return 0 if success else 1
        else:
            print("\nInstallation completed with errors")
            print("Please check the output above for details")
            return 1

    except Exception as e:
        print(f"\nInstallation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
