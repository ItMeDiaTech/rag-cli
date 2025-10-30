#!/usr/bin/env python3
"""Installation script for RAG-CLI Claude Code plugin.

This script installs RAG-CLI as a proper Python package and configures
it as a Claude Code plugin.
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{'=' * 80}")
    print(f"  {text}")
    print(f"{'=' * 80}\n")


def print_step(step_num: int, text: str):
    """Print step with number."""
    print(f"[{step_num}] {text}")


def print_success(text: str):
    """Print success message."""
    print(f"[OK] {text}")


def print_error(text: str):
    """Print error message."""
    print(f"[ERROR] {text}", file=sys.stderr)


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print_error(f"Python 3.8 or higher required. Current: {sys.version_info.major}.{sys.version_info.minor}")
        return False
    print_success(f"Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def check_claude_config():
    """Check if Claude Code config directory exists."""
    claude_dir = Path.home() / '.claude'
    if not claude_dir.exists():
        print_error(f"Claude Code config directory not found: {claude_dir}")
        print("Please ensure Claude Code is installed.")
        return False
    print_success(f"Claude Code directory: {claude_dir}")
    return True


def install_package(project_root: Path):
    """Install package using pip in editable mode."""
    print_step(1, "Installing RAG-CLI package...")

    try:
        # Install in editable mode
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-e', str(project_root)],
            capture_output=True,
            text=True,
            check=True
        )

        print_success("Package installed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print_error(f"Package installation failed:")
        print(e.stdout)
        print(e.stderr)
        return False


def create_plugin_directory():
    """Create plugin directory in Claude Code config."""
    print_step(2, "Creating plugin directory...")

    plugin_dir = Path.home() / '.claude' / 'plugins' / 'rag-cli'
    plugin_dir.mkdir(parents=True, exist_ok=True)

    print_success(f"Plugin directory: {plugin_dir}")
    return plugin_dir


def copy_plugin_files(project_root: Path, plugin_dir: Path):
    """Copy plugin metadata and configuration files."""
    print_step(3, "Copying plugin configuration files...")

    files_to_copy = [
        '.claude-plugin/plugin.json',
        '.claude-plugin/hooks.json',
        'config/rag_settings.json',
    ]

    for file_path in files_to_copy:
        src = project_root / file_path
        if src.exists():
            dest = plugin_dir / src.name
            shutil.copy2(src, dest)
            print(f"  Copied: {src.name}")
        else:
            print(f"  Warning: {file_path} not found")

    # Copy command files
    commands_src = project_root / 'src' / 'plugin' / 'commands'
    commands_dest = plugin_dir / 'commands'
    if commands_src.exists():
        shutil.copytree(commands_src, commands_dest, dirs_exist_ok=True)
        print(f"  Copied: commands/")

    print_success("Configuration files copied")


def update_plugin_json(plugin_dir: Path, project_root: Path):
    """Update plugin.json with correct paths."""
    print_step(4, "Updating plugin configuration...")

    plugin_json_path = plugin_dir / 'plugin.json'

    if not plugin_json_path.exists():
        print_error("plugin.json not found")
        return False

    try:
        with open(plugin_json_path, 'r') as f:
            config = json.load(f)

        # Update paths to use installed package
        config['commands'] = ['./commands/']

        # Ensure MCP server uses installed package
        if 'mcpServers' in config:
            for server_name, server_config in config['mcpServers'].items():
                # Set PYTHONPATH to project root for development
                if 'env' not in server_config:
                    server_config['env'] = {}
                server_config['env']['RAG_CLI_ROOT'] = str(plugin_dir)

        with open(plugin_json_path, 'w') as f:
            json.dump(config, f, indent=2)

        print_success("Plugin configuration updated")
        return True

    except Exception as e:
        print_error(f"Failed to update plugin.json: {e}")
        return False


def create_symlinks(project_root: Path, plugin_dir: Path):
    """Create symlinks for data directories."""
    print_step(5, "Creating symlinks for data directories...")

    # Create symlink for data directory
    data_src = project_root / 'data'
    data_dest = plugin_dir / 'data'

    if data_src.exists():
        if data_dest.exists():
            if data_dest.is_symlink():
                data_dest.unlink()
            else:
                shutil.rmtree(data_dest)

        try:
            # Windows requires admin for symlinks, use junction instead
            if sys.platform == 'win32':
                subprocess.run(['mklink', '/J', str(data_dest), str(data_src)], shell=True, check=True)
            else:
                data_dest.symlink_to(data_src, target_is_directory=True)
            print(f"  Linked: data/ -> {data_src}")
        except Exception as e:
            print(f"  Warning: Could not create symlink for data: {e}")
            print(f"  Copying data directory instead...")
            shutil.copytree(data_src, data_dest, dirs_exist_ok=True)

    print_success("Data directories configured")


def verify_installation():
    """Verify that the package is installed correctly."""
    print_step(6, "Verifying installation...")

    try:
        # Test import
        import core
        import monitoring
        import plugin

        print_success("Package imports working")

        # Test entry points
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', 'rag-cli'],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print_success("Package 'rag-cli' is installed")
            return True
        else:
            print_error("Package 'rag-cli' not found in pip")
            return False

    except ImportError as e:
        print_error(f"Import failed: {e}")
        return False


def print_next_steps(plugin_dir: Path):
    """Print next steps for user."""
    print_header("Installation Complete!")

    print("Next steps:")
    print()
    print("1. Restart Claude Code to load the plugin")
    print()
    print("2. Enable the RAG plugin by running:")
    print("   /rag-enable")
    print()
    print("3. Index your documents:")
    print(f"   rag-index {plugin_dir / 'data' / 'documents'}")
    print()
    print("4. Test retrieval:")
    print("   rag-retrieve --query \"Your question here\"")
    print()
    print("5. Configure Multi-Agent Framework:")
    print("   /rag-maf-config")
    print()
    print(f"Plugin installed at: {plugin_dir}")
    print()


def main():
    """Main installation function."""
    print_header("RAG-CLI Plugin Installation")

    project_root = Path(__file__).resolve().parent

    # Step 0: Pre-flight checks
    print_step(0, "Running pre-flight checks...")

    if not check_python_version():
        sys.exit(1)

    if not check_claude_config():
        sys.exit(1)

    print()

    # Step 1: Install package
    if not install_package(project_root):
        sys.exit(1)

    # Step 2-5: Setup plugin
    plugin_dir = create_plugin_directory()
    copy_plugin_files(project_root, plugin_dir)

    if not update_plugin_json(plugin_dir, project_root):
        sys.exit(1)

    create_symlinks(project_root, plugin_dir)

    # Step 6: Verify
    if not verify_installation():
        print_error("Installation verification failed")
        sys.exit(1)

    # Done!
    print_next_steps(plugin_dir)


if __name__ == '__main__':
    main()
