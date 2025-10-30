#!/usr/bin/env python3
"""Unified sync script for RAG-CLI plugin components.

This script consolidates sync_global.py and sync_to_plugin.py into a single
comprehensive sync solution that:
1. Syncs to global Claude Code directories (commands, hooks, skills)
2. Syncs to plugin directory for runtime module access
3. Updates MCP configuration with proper environment variables
4. Verifies all components are installed correctly
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Set

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# Paths - use environment variable if set
PROJECT_ROOT = Path(os.getenv('RAG_CLI_ROOT', Path(__file__).resolve().parent.parent))
CLAUDE_HOME = Path.home() / ".claude"

# Global directories
GLOBAL_COMMANDS_DIR = CLAUDE_HOME / "commands"
GLOBAL_HOOKS_DIR = CLAUDE_HOME / "hooks" / "rag-cli"
GLOBAL_SKILLS_DIR = CLAUDE_HOME / "skills" / "rag-cli"
GLOBAL_MCP_DIR = CLAUDE_HOME / "mcp"
PLUGIN_DIR = CLAUDE_HOME / "plugins" / "rag-cli"

# Source directories
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "config"
PROJECT_COMMANDS_DIR = SRC_DIR / "plugin" / "commands"
PROJECT_HOOKS_DIR = SRC_DIR / "plugin" / "hooks"
PROJECT_SKILLS_DIR = SRC_DIR / "plugin" / "skills"
PROJECT_MCP_DIR = SRC_DIR / "plugin" / "mcp"
PROJECT_LOCAL_COMMANDS_DIR = PROJECT_ROOT / ".claude" / "commands"


def print_section(title: str):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKBLUE}{title}{Colors.ENDC}")
    print(f"{Colors.BOLD}{'=' * 70}{Colors.ENDC}")


def print_success(message: str):
    """Print success message."""
    print(f"{Colors.OKGREEN}[OK]{Colors.ENDC} {message}")


def print_warning(message: str):
    """Print warning message."""
    print(f"{Colors.WARNING}[WARN]{Colors.ENDC} {message}")


def print_error(message: str):
    """Print error message."""
    print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {message}")


def sync_file(src: Path, dest: Path, description: str = None) -> bool:
    """Sync a single file from source to destination.

    Args:
        src: Source file path
        dest: Destination file path
        description: Optional description for logging

    Returns:
        True if file was copied, False otherwise
    """
    if not src.exists():
        print_warning(f"Source not found: {src.name}")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)

    # Check if update needed
    if dest.exists():
        src_mtime = src.stat().st_mtime
        dest_mtime = dest.stat().st_mtime
        if src_mtime <= dest_mtime:
            return False  # Destination is newer or same

    try:
        shutil.copy2(src, dest)
        if description:
            print_success(f"{src.name:30} - {description}")
        else:
            print_success(src.name)
        return True
    except Exception as e:
        print_error(f"Failed to copy {src.name}: {e}")
        return False


def sync_directory(src_dir: Path, dest_dir: Path, pattern: str = "*",
                   exclude_patterns: Set[str] = None, description: str = None) -> int:
    """Sync all files matching pattern from source to destination directory.

    Args:
        src_dir: Source directory
        dest_dir: Destination directory
        pattern: Glob pattern for files to sync
        exclude_patterns: Set of patterns to exclude
        description: Optional description for logging

    Returns:
        Number of files copied
    """
    if not src_dir.exists():
        print_warning(f"Source directory not found: {src_dir}")
        return 0

    dest_dir.mkdir(parents=True, exist_ok=True)
    exclude_patterns = exclude_patterns or {"__pycache__", ".pyc", ".pytest_cache", "test_"}

    count = 0
    for item in src_dir.rglob(pattern):
        if item.is_file():
            # Check exclusions
            rel_path = item.relative_to(src_dir)
            if any(excl in str(rel_path) for excl in exclude_patterns):
                continue

            dest_file = dest_dir / rel_path
            if sync_file(item, dest_file):
                count += 1

    if count > 0 and description:
        print(f"  Synced {count} files from {description}")

    return count


def sync_commands() -> Dict[str, int]:
    """Sync all RAG-CLI commands to global commands directory."""
    print_section("SYNCING COMMANDS")

    stats = {"updated": 0, "skipped": 0}

    # Commands from src/plugin/commands/
    commands = [
        ("search.md", "Search indexed documents"),
        ("rag-enable.md", "Enable RAG enhancement"),
        ("rag-disable.md", "Disable RAG enhancement"),
        ("update-rag.md", "Update RAG plugin files"),
    ]

    for cmd_file, description in commands:
        src = PROJECT_COMMANDS_DIR / cmd_file
        dest = GLOBAL_COMMANDS_DIR / cmd_file
        if sync_file(src, dest, description):
            stats["updated"] += 1
        else:
            stats["skipped"] += 1

    # Commands from .claude/commands/ (project-specific)
    if PROJECT_LOCAL_COMMANDS_DIR.exists():
        project_commands = [
            ("watch-rag.md", "Watch RAG system with live dashboard"),
            ("rag-project.md", "Index current project documentation"),
        ]

        for cmd_file, description in project_commands:
            src = PROJECT_LOCAL_COMMANDS_DIR / cmd_file
            dest = GLOBAL_COMMANDS_DIR / cmd_file
            if sync_file(src, dest, description):
                stats["updated"] += 1
            else:
                stats["skipped"] += 1

    print(f"\n{Colors.OKGREEN}Commands: {stats['updated']} updated, {stats['skipped']} unchanged{Colors.ENDC}")
    return stats


def sync_hooks() -> Dict[str, int]:
    """Sync all RAG-CLI hooks to global hooks directory."""
    print_section("SYNCING HOOKS")

    stats = {"updated": 0, "skipped": 0}

    # Hooks from src/plugin/hooks/
    hooks = [
        ("user-prompt-submit.py", "Main RAG enhancement hook"),
        ("update-rag-hook.py", "Update command handler"),
        ("response-post.py", "Inline citations"),
        ("error-handler.py", "Graceful error handling"),
        ("plugin-state-change.py", "Settings persistence"),
        ("document-indexing.py", "Auto-indexing (disabled by default)"),
    ]

    for hook_file, description in hooks:
        src = PROJECT_HOOKS_DIR / hook_file
        dest = GLOBAL_HOOKS_DIR / hook_file
        if sync_file(src, dest, description):
            stats["updated"] += 1
        else:
            stats["skipped"] += 1

    print(f"\n{Colors.OKGREEN}Hooks: {stats['updated']} updated, {stats['skipped']} unchanged{Colors.ENDC}")
    return stats


def sync_skills() -> Dict[str, int]:
    """Sync all RAG-CLI skills to global skills directory."""
    print_section("SYNCING SKILLS")

    stats = {"updated": 0, "skipped": 0}

    # Sync entire skills directory structure
    if PROJECT_SKILLS_DIR.exists():
        for skill_dir in PROJECT_SKILLS_DIR.iterdir():
            if skill_dir.is_dir():
                dest_skill_dir = GLOBAL_SKILLS_DIR / skill_dir.name
                count = sync_directory(skill_dir, dest_skill_dir, "*", description=skill_dir.name)
                if count > 0:
                    stats["updated"] += count
                    print_success(f"Skill '{skill_dir.name}': {count} files")
                else:
                    stats["skipped"] += 1

    print(f"\n{Colors.OKGREEN}Skills: {stats['updated']} files synced{Colors.ENDC}")
    return stats


def sync_plugin_runtime() -> Dict[str, int]:
    """Sync core modules to plugin directory for runtime import access."""
    print_section("SYNCING PLUGIN RUNTIME MODULES")

    stats = {"updated": 0, "skipped": 0}

    exclude_patterns = {"__pycache__", ".pyc", ".pytest_cache", "test_", ".md"}

    # Sync core modules
    dirs_to_sync = [
        ("src/core", "Core RAG modules"),
        ("src/monitoring", "Monitoring modules"),
        ("src/plugin", "Plugin components"),
    ]

    for src_path, description in dirs_to_sync:
        src_dir = PROJECT_ROOT / src_path
        dest_dir = PLUGIN_DIR / src_path
        count = sync_directory(src_dir, dest_dir, "*.py", exclude_patterns, description)
        if count > 0:
            stats["updated"] += count
            print(f"  {description}: {count} files")

    # Sync configuration files
    if CONFIG_DIR.exists():
        config_dest = PLUGIN_DIR / "config"
        count = sync_directory(CONFIG_DIR, config_dest, "*.json", exclude_patterns, "Config files")
        if count > 0:
            stats["updated"] += count

        # Also sync default.yaml
        yaml_src = CONFIG_DIR / "default.yaml"
        yaml_dest = config_dest / "default.yaml"
        if yaml_src.exists():
            if sync_file(yaml_src, yaml_dest, "Default config"):
                stats["updated"] += 1

    # Sync requirements
    for req_file in ["requirements.txt", "requirements-lock.txt"]:
        src = PROJECT_ROOT / req_file
        dest = PLUGIN_DIR / req_file
        if src.exists() and sync_file(src, dest):
            stats["updated"] += 1

    print(f"\n{Colors.OKGREEN}Plugin runtime: {stats['updated']} files synced{Colors.ENDC}")
    return stats


def sync_mcp() -> Dict[str, int]:
    """Update MCP configuration - now handled by setup_env.py."""
    print_section("MCP CONFIGURATION")

    stats = {"updated": 0, "skipped": 0}

    # Check if setup_env has been run
    mcp_config = GLOBAL_MCP_DIR / "rag-cli.json"
    if mcp_config.exists():
        print_success("MCP configuration exists")
        stats["skipped"] += 1
    else:
        print_warning("MCP configuration not found")
        print("  Run: python scripts/setup_env.py")
        stats["updated"] += 1

    return stats


def verify_installation() -> bool:
    """Verify that all components are properly installed."""
    print_section("VERIFYING INSTALLATION")

    checks = {
        "Commands": [
            GLOBAL_COMMANDS_DIR / "search.md",
            GLOBAL_COMMANDS_DIR / "rag-enable.md",
            GLOBAL_COMMANDS_DIR / "update-rag.md",
        ],
        "Hooks": [
            GLOBAL_HOOKS_DIR / "user-prompt-submit.py",
            GLOBAL_HOOKS_DIR / "response-post.py",
            GLOBAL_HOOKS_DIR / "error-handler.py",
        ],
        "MCP": [
            GLOBAL_MCP_DIR / "rag-cli.json",
        ],
        "Plugin Runtime": [
            PLUGIN_DIR / "src" / "core" / "config.py",
            PLUGIN_DIR / "src" / "core" / "vector_store.py",
            PLUGIN_DIR / "src" / "monitoring" / "service_manager.py",
            PLUGIN_DIR / "config" / "hook_config.json",
        ],
    }

    all_present = True

    for category, files in checks.items():
        print(f"\n{Colors.BOLD}{category}:{Colors.ENDC}")
        category_ok = True
        for file_path in files:
            if file_path.exists():
                print_success(file_path.name)
            else:
                print_error(f"Missing: {file_path.name}")
                category_ok = False
                all_present = False

        if category_ok:
            print(f"  {Colors.OKGREEN}All {category.lower()} installed{Colors.ENDC}")

    return all_present


def main():
    """Main sync function."""
    parser = argparse.ArgumentParser(description="Sync RAG-CLI plugin components")
    parser.add_argument("--commands-only", action="store_true", help="Sync only commands")
    parser.add_argument("--hooks-only", action="store_true", help="Sync only hooks")
    parser.add_argument("--runtime-only", action="store_true", help="Sync only runtime modules")
    parser.add_argument("--skip-verify", action="store_true", help="Skip verification step")
    args = parser.parse_args()

    print(f"\n{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}RAG-CLI UNIFIED PLUGIN SYNC{Colors.ENDC}")
    print(f"{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
    print(f"Project Root: {Colors.OKCYAN}{PROJECT_ROOT}{Colors.ENDC}")
    print(f"Claude Home:  {Colors.OKCYAN}{CLAUDE_HOME}{Colors.ENDC}")
    print(f"{Colors.BOLD}{'=' * 70}{Colors.ENDC}")

    # Create base directories
    CLAUDE_HOME.mkdir(parents=True, exist_ok=True)
    GLOBAL_COMMANDS_DIR.mkdir(parents=True, exist_ok=True)
    GLOBAL_HOOKS_DIR.mkdir(parents=True, exist_ok=True)
    GLOBAL_SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    GLOBAL_MCP_DIR.mkdir(parents=True, exist_ok=True)
    PLUGIN_DIR.mkdir(parents=True, exist_ok=True)

    # Sync components based on arguments
    total_stats = {}

    if args.commands_only:
        total_stats["commands"] = sync_commands()
    elif args.hooks_only:
        total_stats["hooks"] = sync_hooks()
    elif args.runtime_only:
        total_stats["runtime"] = sync_plugin_runtime()
    else:
        # Sync all components
        total_stats["commands"] = sync_commands()
        total_stats["hooks"] = sync_hooks()
        total_stats["skills"] = sync_skills()
        total_stats["runtime"] = sync_plugin_runtime()
        total_stats["mcp"] = sync_mcp()

    # Verify installation
    all_present = True
    if not args.skip_verify:
        all_present = verify_installation()

    # Summary
    print_section("SYNC SUMMARY")

    total_updated = sum(stats.get("updated", 0) for stats in total_stats.values())

    print(f"\n{Colors.BOLD}Total files updated: {Colors.OKGREEN}{total_updated}{Colors.ENDC}\n")
    print("Breakdown:")
    for component, stats in total_stats.items():
        updated = stats.get("updated", 0)
        color = Colors.OKGREEN if updated > 0 else Colors.OKCYAN
        print(f"  {component:15} - {color}{updated}{Colors.ENDC} updated")

    print(f"\n{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
    if all_present:
        print(f"{Colors.BOLD}{Colors.OKGREEN}SUCCESS: RAG-CLI is now globally available!{Colors.ENDC}")
        print("\nAvailable commands:")
        print(f"  {Colors.OKCYAN}/search [query]{Colors.ENDC}     - Search indexed documents")
        print(f"  {Colors.OKCYAN}/rag-enable{Colors.ENDC}          - Enable RAG enhancement")
        print(f"  {Colors.OKCYAN}/rag-disable{Colors.ENDC}         - Disable RAG enhancement")
        print(f"  {Colors.OKCYAN}/update-rag{Colors.ENDC}          - Update plugin files")
        print(f"  {Colors.OKCYAN}/watch-rag{Colors.ENDC}           - Open monitoring dashboard")
    else:
        print(f"{Colors.WARNING}WARNING: Some components are missing{Colors.ENDC}")
        print("Please review the verification results above.")
    print(f"{Colors.BOLD}{'=' * 70}{Colors.ENDC}")

    # Next steps
    print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
    print("  1. Restart Claude Code to load updated components")
    print("  2. Verify hooks: claude hooks list")
    print("  3. Test command: /search [query]")
    print("  4. Enable RAG: /rag-enable")
    print()

    return 0 if all_present else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Sync cancelled by user{Colors.ENDC}")
        sys.exit(130)
    except Exception as e:
        print_error(f"Sync failed: {e}")
        sys.exit(1)
