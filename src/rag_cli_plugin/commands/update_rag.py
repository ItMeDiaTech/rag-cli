"""
/update-rag slash command implementation.

Allows users to update RAG-CLI to the latest version from GitHub.

IMPORTANT: NO EMOJIS - All output must be professional text only.
"""

import subprocess
import sys
import requests
from pathlib import Path
from typing import Dict, Any, Optional


def get_plugin_root() -> Path:
    """Resolve plugin root directory."""
    import os

    plugin_root = os.environ.get('CLAUDE_PLUGIN_ROOT')
    if plugin_root:
        return Path(plugin_root)
    return Path(__file__).parent.parent.parent.parent


def check_current_version() -> str:
    """Get currently installed version."""
    try:
        import rag_cli_plugin
        return getattr(rag_cli_plugin, '__version__', 'unknown')
    except ImportError:
        return 'unknown'


def fetch_latest_version() -> Optional[str]:
    """Check GitHub for latest release version."""
    try:
        response = requests.get(
            "https://api.github.com/repos/ItMeDiaTech/rag-cli/releases/latest",
            timeout=5
        )
        if response.ok:
            data = response.json()
            version = data['tag_name'].lstrip('v')
            return version
        return None
    except Exception as e:
        print(f"Warning: Could not fetch latest version: {e}")
        return None


def update_from_git() -> bool:
    """Pull latest from GitHub and reinstall."""
    plugin_root = get_plugin_root()

    print("\nUpdating RAG-CLI from GitHub...")
    print("-" * 60)

    try:
        # Check if this is a git repository
        git_dir = plugin_root / ".git"
        if not git_dir.exists():
            print("Error: Plugin is not a git repository")
            print("Please reinstall from Claude Code marketplace or clone from GitHub")
            return False

        # Pull latest changes
        print("Pulling latest changes...")
        result = subprocess.run(
            ["git", "pull", "origin", "master"],
            cwd=plugin_root,
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)

        # Reinstall package with updated code
        print("\nReinstalling package...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "-e", str(plugin_root),
            "--upgrade", "--quiet"
        ])

        print("\nUpdate complete!")
        print("-" * 60)
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error during update: {e}")
        if e.stderr:
            print(f"Details: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def handle_update_command(args: Dict[str, Any]) -> Dict[str, Any]:
    """Main handler for /update-rag command."""

    print("\n" + "=" * 60)
    print("RAG-CLI Update")
    print("=" * 60)

    current = check_current_version()
    latest = fetch_latest_version()

    print(f"\nCurrent version: {current}")
    if latest:
        print(f"Latest version:  {latest}")
    else:
        print("Latest version:  Unable to determine (continuing anyway)")

    if latest and current == latest:
        print("\nYou are already on the latest version!")
        print("Running update anyway to ensure all files are current...")

    # Perform update
    success = update_from_git()

    if success:
        # Get new version after update
        import importlib
        import rag_cli_plugin
        importlib.reload(rag_cli_plugin)
        new_version = getattr(rag_cli_plugin, '__version__', 'unknown')

        print("\n" + "=" * 60)
        if current != new_version:
            print(f"Successfully updated from {current} to {new_version}")
        else:
            print("Update completed - all files are current")
        print("=" * 60)

        return {
            "success": True,
            "message": f"Updated to version {new_version}",
            "old_version": current,
            "new_version": new_version
        }
    else:
        print("\n" + "=" * 60)
        print("Update failed - see errors above")
        print("=" * 60)

        return {
            "success": False,
            "message": "Update failed",
            "old_version": current,
            "new_version": None
        }


# Entry point for slash command
def main():
    """CLI entry point for testing."""
    result = handle_update_command({})
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
