"""Setup git hooks for RAG-CLI project.

This script configures git to use the hooks in .githooks/ directory
to enforce code quality standards, including the no-emoji policy.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, cwd: Path = None) -> bool:
    """Run a shell command and return success status.

    Args:
        cmd: Command to run as list
        cwd: Working directory

    Returns:
        True if command succeeded, False otherwise
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {' '.join(cmd)}", file=sys.stderr)
        print(f"  Output: {e.stderr}", file=sys.stderr)
        return False


def setup_git_hooks(project_root: Path) -> bool:
    """Configure git to use .githooks directory.

    Args:
        project_root: Path to project root

    Returns:
        True if setup succeeded
    """
    githooks_dir = project_root / ".githooks"

    if not githooks_dir.exists():
        print("[ERROR] .githooks directory not found!", file=sys.stderr)
        return False

    # Configure git to use .githooks
    print("Configuring git to use .githooks directory...")

    if not run_command(
        ["git", "config", "core.hooksPath", ".githooks"],
        cwd=project_root
    ):
        return False

    print("[OK] Git hooks configured successfully!")

    # Make hooks executable on Unix-like systems
    if os.name != 'nt':  # Not Windows
        print("Making hook scripts executable...")
        for hook_file in githooks_dir.glob("*"):
            if hook_file.is_file() and not hook_file.name.endswith('.md'):
                try:
                    hook_file.chmod(0o755)
                    print(f"[OK] Made {hook_file.name} executable")
                except Exception as e:
                    print(f"[WARNING] Failed to chmod {hook_file.name}: {e}")

    return True


def verify_hooks(project_root: Path) -> bool:
    """Verify git hooks are properly configured.

    Args:
        project_root: Path to project root

    Returns:
        True if hooks are configured correctly
    """
    try:
        result = subprocess.run(
            ["git", "config", "--get", "core.hooksPath"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True
        )

        hooks_path = result.stdout.strip()

        if hooks_path == ".githooks":
            print("\n[OK] Git hooks are properly configured!")
            print(f"    Hooks directory: {hooks_path}")
            return True
        else:
            print(f"\n[WARNING] Unexpected hooks path: {hooks_path}")
            return False

    except subprocess.CalledProcessError:
        print("\n[ERROR] Failed to verify git hooks configuration")
        return False


def main():
    """Main execution function."""
    project_root = Path(__file__).parent.parent

    print("=" * 70)
    print("RAG-CLI Git Hooks Setup")
    print("=" * 70)

    # Check if we're in a git repository
    if not (project_root / ".git").exists():
        print("\n[ERROR] Not a git repository!")
        print("Initialize git first: git init")
        return 1

    print("\nSetting up git hooks...")
    print("-" * 70)

    if not setup_git_hooks(project_root):
        print("\n[FAILED] Git hooks setup failed!")
        return 1

    # Verify configuration
    if not verify_hooks(project_root):
        print("\n[WARNING] Verification failed, but hooks may still work")

    print("\n" + "=" * 70)
    print("[SUCCESS] Git hooks setup complete!")
    print("\nAvailable hooks:")
    print("  - pre-commit: Validates no emoji usage")
    print("\nTest the validation:")
    print("  python scripts/validate_no_emojis.py")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
