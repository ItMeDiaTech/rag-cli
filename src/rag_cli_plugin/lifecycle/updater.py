"""
Update handler for RAG-CLI plugin.

This module handles plugin updates including pre-update backup,
post-update migration, and configuration management.

IMPORTANT: NO EMOJIS - All output must be professional text only.
"""

import sys
import subprocess
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


def get_plugin_root() -> Path:
    """Find plugin installation directory."""
    import os

    # Try environment variable first (set by Claude Code)
    plugin_root = os.environ.get('CLAUDE_PLUGIN_ROOT')
    if plugin_root:
        return Path(plugin_root)

    # Fallback: resolve from this file's location
    return Path(__file__).parent.parent.parent.parent


def get_current_version() -> str:
    """Get currently installed version."""
    try:
        import rag_cli_plugin
        return getattr(rag_cli_plugin, '__version__', 'unknown')
    except ImportError:
        return 'unknown'


def backup_configuration() -> Optional[Path]:
    """Backup current configuration before update."""
    plugin_root = get_plugin_root()
    config_dir = plugin_root / "config"

    if not config_dir.exists():
        print("Warning: Config directory not found, skipping backup")
        return None

    # Create backup directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = plugin_root / "backups" / f"config_backup_{timestamp}"

    try:
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Backup all JSON config files
        config_files = list(config_dir.glob("*.json"))
        for config_file in config_files:
            if config_file.name != "mcp.json":  # Skip MCP config
                shutil.copy2(config_file, backup_dir / config_file.name)

        print(f"Configuration backed up to: {backup_dir.relative_to(plugin_root)}")
        return backup_dir
    except Exception as e:
        print(f"Error backing up configuration: {e}")
        return None


def restore_configuration(backup_dir: Path) -> bool:
    """Restore configuration from backup."""
    plugin_root = get_plugin_root()
    config_dir = plugin_root / "config"

    if not backup_dir.exists():
        print(f"Error: Backup directory not found: {backup_dir}")
        return False

    try:
        for config_file in backup_dir.glob("*.json"):
            target = config_dir / config_file.name
            shutil.copy2(config_file, target)
            print(f"  Restored: {config_file.name}")
        return True
    except Exception as e:
        print(f"Error restoring configuration: {e}")
        return False


def update_dependencies() -> bool:
    """Update Python dependencies from requirements.txt"""
    plugin_root = get_plugin_root()
    requirements = plugin_root / "requirements.txt"

    if not requirements.exists():
        print(f"Warning: requirements.txt not found at {requirements}")
        return False

    print("Updating dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "-r", str(requirements),
            "--upgrade", "--quiet"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("  Dependencies updated successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error updating dependencies: {e}")
        return False


def migrate_configuration(old_version: str, new_version: str) -> bool:
    """Migrate configuration between versions if needed."""
    plugin_root = get_plugin_root()
    config_dir = plugin_root / "config"

    print(f"Checking configuration migration from {old_version} to {new_version}...")

    # Add version-specific migration logic here
    # For now, just verify configs exist

    required_configs = ["rag_settings.json", "services.json"]
    all_exist = True

    for config_name in required_configs:
        config_file = config_dir / config_name
        if not config_file.exists():
            print(f"  Warning: Missing config: {config_name}")
            # Try to copy from defaults
            default_file = config_dir / "defaults" / config_name
            if default_file.exists():
                shutil.copy2(default_file, config_file)
                print(f"  Created {config_name} from defaults")
            else:
                all_exist = False

    if all_exist:
        print("  Configuration migration complete")
    else:
        print("  Configuration migration completed with warnings")

    return all_exist


def run_pre_update() -> int:
    """Run pre-update tasks."""
    print("\n" + "=" * 60)
    print("RAG-CLI Pre-Update")
    print("=" * 60)

    current_version = get_current_version()
    print(f"\nCurrent version: {current_version}")

    print("\n[1/1] Backing up configuration...")
    backup_dir = backup_configuration()

    if backup_dir:
        print("\nPre-update complete")
        return 0
    else:
        print("\nPre-update completed with warnings")
        return 1


def run_post_update() -> int:
    """Run post-update tasks."""
    print("\n" + "=" * 60)
    print("RAG-CLI Post-Update")
    print("=" * 60)

    old_version = get_current_version()  # This might not work after update

    success = True

    try:
        print("\n[1/2] Updating dependencies...")
        if not update_dependencies():
            print("  Warning: Dependency update incomplete")
            success = False

        print("\n[2/2] Migrating configuration...")
        # Get new version after update
        import importlib
        import rag_cli_plugin
        importlib.reload(rag_cli_plugin)
        new_version = getattr(rag_cli_plugin, '__version__', 'unknown')

        if not migrate_configuration(old_version, new_version):
            print("  Warning: Configuration migration incomplete")
            success = False

        print("\n" + "=" * 60)
        print(f"RAG-CLI updated to version {new_version}")
        print("=" * 60)

        return 0 if success else 1

    except Exception as e:
        print(f"\nPost-update failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main updater entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(description="RAG-CLI updater")
    parser.add_argument(
        "--mode",
        choices=["pre", "post"],
        required=True,
        help="Update mode: pre or post"
    )

    args = parser.parse_args()

    if args.mode == "pre":
        return run_pre_update()
    elif args.mode == "post":
        return run_post_update()
    else:
        print(f"Unknown mode: {args.mode}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
