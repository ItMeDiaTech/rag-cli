#!/usr/bin/env python3
"""Setup environment variables and configuration for RAG-CLI.

This script:
1. Detects the current platform (Windows/macOS/Linux)
2. Finds the RAG-CLI root directory
3. Generates proper MCP configuration with environment variables
4. Creates .env file if it doesn't exist
5. Updates global Claude Code MCP configuration
"""

import os
import sys
import json
import platform
from pathlib import Path


def get_project_root() -> Path:
    """Get the RAG-CLI project root directory."""
    # Start from this script's location
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Verify this is the correct root
    if not (project_root / "src" / "core").exists():
        raise RuntimeError(f"Could not find RAG-CLI root. Checked: {project_root}")

    return project_root


def get_maf_root(project_root: Path) -> Path:
    """Get the multi-agent framework root directory."""
    # Assume MAF is in DocHub root
    maf_root = project_root.parent.parent / "multi-agent-framework"

    if not maf_root.exists():
        print(f"Warning: Multi-agent framework not found at {maf_root}")
        return None

    return maf_root


def create_env_file(project_root: Path, maf_root: Path = None):
    """Create .env file from .env.example if it doesn't exist."""
    env_file = project_root / ".env"
    env_example = project_root / ".env.example"

    if env_file.exists():
        print(f".env file already exists: {env_file}")
        return

    # Read example
    if not env_example.exists():
        print(f"Warning: .env.example not found at {env_example}")
        return

    with open(env_example, 'r') as f:
        content = f.read()

    # Replace placeholders
    content = content.replace('/path/to/RAG-CLI', str(project_root))
    if maf_root:
        content = content.replace('/path/to/multi-agent-framework', str(maf_root))

    # Write .env
    with open(env_file, 'w') as f:
        f.write(content)

    print(f"Created .env file: {env_file}")
    print("Please update it with your ANTHROPIC_API_KEY")


def generate_mcp_config(project_root: Path) -> dict:
    """Generate MCP configuration using environment variables."""
    return {
        "command": "python",
        "args": [
            "-m",
            "src.plugin.mcp.unified_server"
        ],
        "cwd": str(project_root),
        "env": {
            "PYTHONUNBUFFERED": "1",
            "RAG_CLI_MODE": "claude_code",
            "RAG_CLI_ROOT": str(project_root)
        }
    }


def update_global_mcp_config(project_root: Path):
    """Update global Claude Code MCP configuration."""
    # Detect Claude Code config directory
    home = Path.home()

    claude_mcp_dir = home / ".claude" / "mcp"
    if not claude_mcp_dir.exists():
        claude_mcp_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created Claude MCP directory: {claude_mcp_dir}")

    mcp_config_file = claude_mcp_dir / "rag-cli.json"

    # Generate configuration
    config = generate_mcp_config(project_root)

    # Write configuration
    with open(mcp_config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Updated MCP configuration: {mcp_config_file}")
    return mcp_config_file


def update_plugin_json(project_root: Path):
    """Update plugin.json to use unified_server."""
    plugin_json = project_root / ".claude-plugin" / "plugin.json"

    if not plugin_json.exists():
        print(f"Warning: plugin.json not found at {plugin_json}")
        return

    with open(plugin_json, 'r') as f:
        config = json.load(f)

    # Update MCP server configuration
    if "components" in config and "mcp_servers" in config["components"]:
        for server in config["components"]["mcp_servers"]:
            if server["name"] == "rag-server":
                server["args"] = ["-m", "src.plugin.mcp.unified_server"]
                print("Updated plugin.json MCP server to use unified_server")

    with open(plugin_json, 'w') as f:
        json.dump(config, f, indent=2)


def set_environment_variables(project_root: Path, maf_root: Path = None):
    """Set environment variables for current session."""
    os.environ["RAG_CLI_ROOT"] = str(project_root)
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["RAG_CLI_MODE"] = "claude_code"

    if maf_root:
        os.environ["MAF_ROOT"] = str(maf_root)

    print("\nEnvironment variables set for current session:")
    print(f"  RAG_CLI_ROOT={os.environ['RAG_CLI_ROOT']}")
    if maf_root:
        print(f"  MAF_ROOT={os.environ['MAF_ROOT']}")


def generate_activation_script(project_root: Path, maf_root: Path = None):
    """Generate platform-specific activation script."""
    system = platform.system()

    if system == "Windows":
        # PowerShell script
        script_path = project_root / "activate.ps1"
        content = f"""# RAG-CLI Environment Activation Script (PowerShell)
# Run this script: .\\activate.ps1

$env:RAG_CLI_ROOT = "{project_root}"
$env:PYTHONUNBUFFERED = "1"
$env:RAG_CLI_MODE = "claude_code"
"""
        if maf_root:
            content += f'$env:MAF_ROOT = "{maf_root}"\n'

        content += """
Write-Host "RAG-CLI environment activated!" -ForegroundColor Green
Write-Host "Environment variables:"
Write-Host "  RAG_CLI_ROOT=$env:RAG_CLI_ROOT"
"""
        if maf_root:
            content += 'Write-Host "  MAF_ROOT=$env:MAF_ROOT"\n'

    else:
        # Bash script
        script_path = project_root / "activate.sh"
        content = f"""#!/bin/bash
# RAG-CLI Environment Activation Script (Bash)
# Run this script: source ./activate.sh

export RAG_CLI_ROOT="{project_root}"
export PYTHONUNBUFFERED="1"
export RAG_CLI_MODE="claude_code"
"""
        if maf_root:
            content += f'export MAF_ROOT="{maf_root}"\n'

        content += """
echo "RAG-CLI environment activated!"
echo "Environment variables:"
echo "  RAG_CLI_ROOT=$RAG_CLI_ROOT"
"""
        if maf_root:
            content += 'echo "  MAF_ROOT=$MAF_ROOT"\n'

    with open(script_path, 'w') as f:
        f.write(content)

    # Make executable on Unix
    if system != "Windows":
        os.chmod(script_path, 0o755)

    print(f"\nActivation script created: {script_path}")
    if system == "Windows":
        print("Run: .\\activate.ps1")
    else:
        print("Run: source ./activate.sh")


def main():
    """Main setup function."""
    print("=" * 60)
    print("RAG-CLI Environment Setup")
    print("=" * 60)
    print()

    # Detect platform
    system = platform.system()
    print(f"Platform: {system}")
    print()

    # Get paths
    project_root = get_project_root()
    print(f"RAG-CLI Root: {project_root}")

    maf_root = get_maf_root(project_root)
    if maf_root:
        print(f"Multi-Agent Framework: {maf_root}")
    print()

    # Create .env file
    create_env_file(project_root, maf_root)
    print()

    # Update MCP configuration
    mcp_config_file = update_global_mcp_config(project_root)
    print()

    # Update plugin.json
    update_plugin_json(project_root)
    print()

    # Set environment variables
    set_environment_variables(project_root, maf_root)
    print()

    # Generate activation script
    generate_activation_script(project_root, maf_root)
    print()

    print("=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Update .env with your ANTHROPIC_API_KEY")
    print("2. Restart Claude Code to load new MCP configuration")
    print("3. Run activation script in terminal sessions:")
    if system == "Windows":
        print("   .\\activate.ps1")
    else:
        print("   source ./activate.sh")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
