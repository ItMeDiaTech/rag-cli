#!/usr/bin/env python3
"""Configure MCP server settings for RAG-CLI development.

This script updates .mcp.json with the correct paths for the current
system while preserving all other MCP server configurations.
"""

import json
import sys
from pathlib import Path


def get_project_root() -> Path:
    """Get the absolute path to the project root directory."""
    # Script is in scripts/, so parent is project root
    return Path(__file__).resolve().parent.parent


def load_existing_config(config_path: Path) -> dict:
    """Load existing .mcp.json or return empty structure."""
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                print(f"Loaded existing configuration from {config_path}")
                return config
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse existing {config_path}: {e}")
            print("Creating new configuration...")
            return {"mcpServers": {}}
    else:
        print(f"No existing configuration found at {config_path}")
        print("Creating new configuration...")
        return {"mcpServers": {}}


def update_rag_cli_config(config: dict, project_root: Path) -> dict:
    """Update only the rag-cli MCP server entry, preserving others."""
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Get source directory for PYTHONPATH
    src_dir = project_root / "src"

    # Update ONLY the rag-cli entry
    config["mcpServers"]["rag-cli"] = {
        "command": "python",
        "args": ["-m", "plugin.mcp.unified_server"],
        "env": {
            "PYTHONUNBUFFERED": "1",
            "RAG_CLI_MODE": "claude_code",
            "PYTHONPATH": str(src_dir),
            "RAG_CLI_ROOT": str(project_root)
        }
    }

    return config


def save_config(config: dict, config_path: Path) -> bool:
    """Save configuration to .mcp.json."""
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print(f"\nConfiguration saved to {config_path}")
        return True
    except Exception as e:
        print(f"Error: Failed to save configuration: {e}", file=sys.stderr)
        return False


def print_summary(config: dict, project_root: Path):
    """Print summary of configuration."""
    print("\n" + "=" * 80)
    print("MCP Configuration Summary")
    print("=" * 80)

    total_servers = len(config.get("mcpServers", {}))
    print(f"\nTotal MCP servers configured: {total_servers}")

    if total_servers > 1:
        other_servers = [name for name in config["mcpServers"].keys() if name != "rag-cli"]
        print(f"Other servers (preserved): {', '.join(other_servers)}")

    print(f"\nRAG-CLI Configuration:")
    print(f"  Project root: {project_root}")
    print(f"  PYTHONPATH:   {project_root / 'src'}")
    print(f"  RAG_CLI_ROOT: {project_root}")

    print("\n" + "=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print("\n1. Restart Claude Code or run '/mcp' to reload MCP servers")
    print("2. Verify connection by running '/mcp' and checking rag-cli status")
    print("3. Start using RAG-CLI with '/rag-enable' and '/search' commands")
    print("\nNote: .mcp.json is gitignored and will not be committed to version control")
    print()


def main():
    """Main configuration function."""
    print("=" * 80)
    print("RAG-CLI MCP Configuration Tool")
    print("=" * 80)
    print()

    # Get paths
    project_root = get_project_root()
    config_path = project_root / ".mcp.json"

    print(f"Project root: {project_root}")
    print(f"Config file:  {config_path}")
    print()

    # Load existing config (or create new)
    config = load_existing_config(config_path)

    # Count existing servers
    existing_server_count = len(config.get("mcpServers", {}))

    # Update rag-cli entry
    config = update_rag_cli_config(config, project_root)

    # Save configuration
    if not save_config(config, config_path):
        sys.exit(1)

    # Print summary
    print_summary(config, project_root)

    # Success message
    if existing_server_count > 0:
        print(f"Successfully updated rag-cli configuration (preserved {existing_server_count} other servers)")
    else:
        print("Successfully created rag-cli configuration")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nConfiguration cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
