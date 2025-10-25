#!/usr/bin/env python3
"""Installation helper for RAG-CLI."""

import subprocess
import sys
import os
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.8 or higher."""
    if sys.version_info < (3, 8):
        print(f"Error: Python 3.8+ required (current: {sys.version})")
        return False
    print(f"[OK] Python {sys.version.split()[0]} detected")
    return True


def install_requirements(minimal=False):
    """Install required packages."""
    req_file = "requirements-minimal.txt" if minimal else "requirements.txt"

    print(f"\nInstalling from {req_file}...")

    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])

        print(f"\n[OK] Successfully installed dependencies from {req_file}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n[FAIL] Installation failed: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    dirs = [
        "data/documents",
        "data/vectors",
        "logs",
        "config",
        ".claude"
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    print("[OK] Created required directories")


def check_claude_code_mode():
    """Check if running in Claude Code environment."""
    is_claude_code = os.path.exists('.claude') or os.environ.get('CLAUDE_CODE_ENV')

    if is_claude_code:
        print("\n[OK] Claude Code environment detected")
        print("  No API key required - will use Claude's internal interface")
    else:
        print("\n[WARNING] Not running in Claude Code environment")
        print("  To use standalone mode, set ANTHROPIC_API_KEY environment variable")

    return is_claude_code


def main():
    """Main installation process."""
    print("=" * 60)
    print("RAG-CLI Installation")
    print("=" * 60)

    # Check Python version
    if not check_python_version():
        return 1

    # Ask user about installation mode
    print("\nSelect installation mode:")
    print("1. Claude Code Plugin (minimal, no API key needed)")
    print("2. Full Installation (includes development tools)")

    choice = input("\nEnter choice (1 or 2) [1]: ").strip() or "1"

    minimal = (choice == "1")

    # Install requirements
    if not install_requirements(minimal=minimal):
        return 1

    # Create directories
    create_directories()

    # Check Claude Code mode
    is_claude_code = check_claude_code_mode()

    # Installation complete
    print("\n" + "=" * 60)
    print("Installation Complete!")
    print("=" * 60)

    print("\nNext steps:")
    print("1. Index your documents:")
    print("   python scripts/index.py --input data/documents")
    print("\n2. Test retrieval:")
    print("   python scripts/retrieve.py 'Your test query'")

    if is_claude_code:
        print("\n3. Use Claude Code commands:")
        print("   /search 'Your query'")
        print("   /rag:enable  (to enable automatic enhancement)")
    else:
        print("\n3. For standalone mode, set API key:")
        print("   export ANTHROPIC_API_KEY='your-key-here'")

    print("\nFor more information, see README.md")

    return 0


if __name__ == "__main__":
    sys.exit(main())