#!/usr/bin/env python3
"""
RAG-CLI Plugin Sync Script

Synchronizes plugin files from RAG-CLI development directory to the global
Claude Code configuration directory (~/.claude/plugins/rag-cli).

Features:
- Smart merge with preserved runtime files
- Symlink support for core code
- Automatic backup before sync
- Dry-run mode for previewing changes
- Detailed logging and reporting
"""

import sys
import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SyncConfig:
    """Configuration for sync operation"""
    source_root: Path
    claude_dir: Path
    plugin_name: str = "rag-cli"
    dry_run: bool = False
    verbose: bool = False
    force: bool = False
    backup: bool = True
    no_symlink: bool = False


@dataclass
class SyncResult:
    """Result of sync operation"""
    files_copied: List[str] = None
    files_updated: List[str] = None
    files_deleted: List[str] = None
    files_preserved: List[str] = None
    symlinks_created: List[str] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.files_copied is None:
            self.files_copied = []
        if self.files_updated is None:
            self.files_updated = []
        if self.files_deleted is None:
            self.files_deleted = []
        if self.files_preserved is None:
            self.files_preserved = []
        if self.symlinks_created is None:
            self.symlinks_created = []
        if self.errors is None:
            self.errors = []


class PluginSyncer:
    """Handles plugin synchronization"""

    # Files/patterns to preserve in .claude directory
    PRESERVED_PATTERNS = {
        'logs/',
        'data/vectors/',
        '*.log',
        '*state.json',
        '.DS_Store',
        '__pycache__',
        '*.pyc',
        '*.pyo'
    }

    # Plugin-specific directories to sync
    PLUGIN_DIRS = {
        'commands': 'commands',
        'hooks': 'hooks',
        'skills': 'skills',
        'mcp': 'mcp',
    }

    # Core directories to copy (no symlinks)
    CORE_DIRS = {
        'core': 'src/core',
        'monitoring': 'src/monitoring',
    }

    # Data directories to sync
    DATA_DIRS = {
        'documents': 'data/documents',
        'vectors': 'data/vectors',
    }

    def __init__(self, config: SyncConfig):
        self.config = config
        self.result = SyncResult()

        # Set logging level
        if config.verbose:
            logger.setLevel(logging.DEBUG)

        # Validate paths
        if not self.config.source_root.exists():
            raise ValueError(f"Source directory not found: {self.config.source_root}")

        if not self.config.claude_dir.exists():
            raise ValueError(f"Claude directory not found: {self.config.claude_dir}")

        self.plugin_dir = self.config.claude_dir / 'plugins' / self.config.plugin_name
        self.commands_dir = self.config.claude_dir / 'commands'

        logger.info(f"Initialized syncer for {self.config.plugin_name}")
        logger.debug(f"Source: {self.config.source_root}")
        logger.debug(f"Destination: {self.plugin_dir}")

    def should_preserve(self, filepath: Path) -> bool:
        """Check if a file should be preserved"""
        filepath_str = str(filepath.relative_to(self.plugin_dir))

        for pattern in self.PRESERVED_PATTERNS:
            if pattern.endswith('/'):
                if filepath_str.startswith(pattern):
                    return True
            elif '*' in pattern:
                if filepath.name.endswith(pattern[1:]):
                    return True
            elif filepath_str == pattern or filepath.name == pattern:
                return True

        return False

    def create_backup(self) -> Optional[Path]:
        """Create backup of plugin directory"""
        if not self.plugin_dir.exists():
            logger.info("No existing plugin directory to backup")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.plugin_dir.parent / f"{self.config.plugin_name}_backup_{timestamp}"

        try:
            logger.info(f"Creating backup: {backup_dir}")
            shutil.copytree(self.plugin_dir, backup_dir, ignore_dangling_symlinks=True)
            logger.info(f"Backup created successfully")
            return backup_dir
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            self.result.errors.append(f"Backup creation failed: {e}")
            return None

    def sync_file(self, src: Path, dst: Path, subdir: str = "") -> bool:
        """Sync a single file"""
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)

            # Check if file needs update
            if dst.exists():
                src_mtime = src.stat().st_mtime
                dst_mtime = dst.stat().st_mtime
                src_size = src.stat().st_size
                dst_size = dst.stat().st_size

                # If same timestamp and size, skip
                if not self.config.force and src_mtime == dst_mtime and src_size == dst_size:
                    logger.debug(f"Skipping unchanged: {src.name}")
                    return False

            if self.config.dry_run:
                if dst.exists():
                    self.result.files_updated.append(f"{subdir}/{src.name}")
                    logger.info(f"[DRY RUN] Would update: {dst.relative_to(self.config.claude_dir)}")
                else:
                    self.result.files_copied.append(f"{subdir}/{src.name}")
                    logger.info(f"[DRY RUN] Would copy: {dst.relative_to(self.config.claude_dir)}")
            else:
                shutil.copy2(src, dst)
                if dst.exists():
                    self.result.files_updated.append(f"{subdir}/{src.name}")
                    logger.info(f"Updated: {src.name}")
                else:
                    self.result.files_copied.append(f"{subdir}/{src.name}")
                    logger.info(f"Copied: {src.name}")

            return True
        except Exception as e:
            error_msg = f"Failed to sync {src.name}: {e}"
            logger.error(error_msg)
            self.result.errors.append(error_msg)
            return False

    def sync_directory(self, src_dir: Path, dst_dir: Path, subdir: str = "") -> int:
        """Sync entire directory"""
        if not src_dir.exists():
            logger.warning(f"Source directory not found: {src_dir}")
            return 0

        count = 0
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Copy/update files
        for src_file in src_dir.rglob('*'):
            if src_file.is_file():
                rel_path = src_file.relative_to(src_dir)
                dst_file = dst_dir / rel_path

                # Skip __pycache__ and other excluded patterns
                if '__pycache__' in src_file.parts:
                    continue

                if self.sync_file(src_file, dst_file, subdir):
                    count += 1

        return count


    def get_tracked_files(self, directory: Path, base_type: str) -> Set[Path]:
        """Get files that are tracked (synced) in a directory"""
        tracked = set()

        if base_type == 'commands':
            # Commands are .md files synced from src/plugin/commands
            if (self.config.source_root / 'src' / 'plugin' / 'commands').exists():
                for f in (self.config.source_root / 'src' / 'plugin' / 'commands').glob('*.md'):
                    tracked.add(directory / f.name)
        elif base_type in self.PLUGIN_DIRS:
            # Plugin directories
            src = self.config.source_root / 'src' / 'plugin' / base_type
            if src.exists():
                for f in src.rglob('*'):
                    if f.is_file():
                        rel = f.relative_to(src)
                        tracked.add(directory / rel)

        return tracked

    def cleanup_obsolete_files(self) -> int:
        """Remove files that no longer exist in source"""
        if not self.plugin_dir.exists():
            return 0

        removed = 0

        # Clean up plugin directories
        for subdir_key, subdir_name in self.PLUGIN_DIRS.items():
            subdir = self.plugin_dir / subdir_name
            if not subdir.exists():
                continue

            # Get tracked files from source
            src_dir = self.config.source_root / 'src' / 'plugin' / subdir_name
            if not src_dir.exists():
                continue

            tracked_files = self.get_tracked_files(subdir, subdir_key)

            # Remove files not in source
            for f in subdir.rglob('*'):
                if f.is_file() and f not in tracked_files:
                    try:
                        if self.config.dry_run:
                            logger.info(f"[DRY RUN] Would delete: {f.relative_to(self.plugin_dir)}")
                            self.result.files_deleted.append(str(f.relative_to(self.plugin_dir)))
                        else:
                            f.unlink()
                            logger.info(f"Deleted: {f.name}")
                            self.result.files_deleted.append(str(f.relative_to(self.plugin_dir)))
                        removed += 1
                    except Exception as e:
                        logger.error(f"Failed to delete {f}: {e}")
                        self.result.errors.append(f"Delete failed: {f.name}: {e}")

        # Clean up commands
        for f in self.commands_dir.glob('rag*.md'):
            src = self.config.source_root / 'src' / 'plugin' / 'commands' / f.name
            if not src.exists():
                try:
                    if self.config.dry_run:
                        logger.info(f"[DRY RUN] Would delete: {f.name}")
                        self.result.files_deleted.append(f.name)
                    else:
                        f.unlink()
                        logger.info(f"Deleted command: {f.name}")
                        self.result.files_deleted.append(f.name)
                    removed += 1
                except Exception as e:
                    logger.error(f"Failed to delete {f}: {e}")

        return removed

    def sync(self) -> SyncResult:
        """Execute full sync"""
        logger.info("Starting plugin sync...")

        # Create backup
        if self.config.backup and not self.config.dry_run:
            self.create_backup()

        # Create plugin directory
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        self.commands_dir.mkdir(parents=True, exist_ok=True)

        # Sync plugin.json
        plugin_json_src = self.config.source_root / '.claude-plugin' / 'plugin.json'
        if plugin_json_src.exists():
            self.sync_file(plugin_json_src, self.plugin_dir / 'plugin.json', 'root')

        # Sync plugin directories
        for subdir_key, subdir_name in self.PLUGIN_DIRS.items():
            src = self.config.source_root / 'src' / 'plugin' / subdir_name
            dst = self.plugin_dir / subdir_name

            if src.exists():
                logger.info(f"Syncing {subdir_name}...")
                self.sync_directory(src, dst, subdir_name)

        # Sync commands
        commands_src = self.config.source_root / 'src' / 'plugin' / 'commands'
        if commands_src.exists():
            logger.info("Syncing commands...")
            for cmd_file in commands_src.glob('*.md'):
                self.sync_file(cmd_file, self.commands_dir / cmd_file.name, 'commands')

        # Sync core code (always copy, no symlinks)
        for core_key, core_src in self.CORE_DIRS.items():
            src = self.config.source_root / core_src
            dst = self.plugin_dir / core_src.split('/')[0] / core_key

            if src.exists():
                logger.info(f"Syncing {core_key}...")
                self.sync_directory(src, dst, core_key)

        # Sync data directories (documents and vectors)
        for data_key, data_src in self.DATA_DIRS.items():
            src = self.config.source_root / data_src
            dst = self.plugin_dir / data_src

            if src.exists():
                logger.info(f"Syncing data/{data_key}...")
                self.sync_directory(src, dst, f"data/{data_key}")

        # Sync README and requirements
        for file_name in ['README.md', 'requirements.txt', 'requirements-minimal.txt']:
            src = self.config.source_root / file_name
            if src.exists():
                self.sync_file(src, self.plugin_dir / file_name, 'root')

        # Cleanup obsolete files
        logger.info("Cleaning up obsolete files...")
        self.cleanup_obsolete_files()

        logger.info("Plugin sync completed")
        return self.result

    def print_summary(self):
        """Print sync summary"""
        print("\n" + "=" * 60)
        print("SYNC SUMMARY")
        print("=" * 60)

        if self.config.dry_run:
            print("[DRY RUN MODE]")
            print()

        if self.result.files_copied:
            print(f"\nNew files ({len(self.result.files_copied)}):")
            for f in self.result.files_copied[:5]:
                print(f"  + {f}")
            if len(self.result.files_copied) > 5:
                print(f"  ... and {len(self.result.files_copied) - 5} more")

        if self.result.files_updated:
            print(f"\nUpdated files ({len(self.result.files_updated)}):")
            for f in self.result.files_updated[:5]:
                print(f"  ~ {f}")
            if len(self.result.files_updated) > 5:
                print(f"  ... and {len(self.result.files_updated) - 5} more")

        if self.result.files_deleted:
            print(f"\nDeleted files ({len(self.result.files_deleted)}):")
            for f in self.result.files_deleted[:5]:
                print(f"  - {f}")
            if len(self.result.files_deleted) > 5:
                print(f"  ... and {len(self.result.files_deleted) - 5} more")

        if self.result.symlinks_created:
            print(f"\nSymlinks created ({len(self.result.symlinks_created)}):")
            for s in self.result.symlinks_created:
                print(f"  -> {s}")

        if self.result.errors:
            print(f"\nErrors ({len(self.result.errors)}):")
            for e in self.result.errors:
                print(f"  ! {e}")

        total = (len(self.result.files_copied) + len(self.result.files_updated) +
                len(self.result.files_deleted) + len(self.result.symlinks_created))

        print(f"\nTotal changes: {total}")
        print("=" * 60 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Synchronize RAG-CLI plugin with Claude Code configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sync_plugin.py                    # Normal sync
  python sync_plugin.py --dry-run          # Preview changes
  python sync_plugin.py --verbose          # Detailed output
  python sync_plugin.py --force            # Force sync (ignore timestamps)
  python sync_plugin.py --no-backup        # Skip backup
        """
    )

    parser.add_argument('--dry-run', action='store_true',
                       help='Preview changes without making them')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Force sync (ignore timestamps)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip backup creation')
    parser.add_argument('--claude-dir', type=Path,
                       default=Path.home() / '.claude',
                       help='Path to Claude configuration directory')

    args = parser.parse_args()

    # Create config
    config = SyncConfig(
        source_root=Path(__file__).parent,
        claude_dir=args.claude_dir,
        dry_run=args.dry_run,
        verbose=args.verbose,
        force=args.force,
        backup=not args.no_backup,
        no_symlink=False  # Always use copy mode (no symlinks)
    )

    try:
        # Run sync
        syncer = PluginSyncer(config)
        result = syncer.sync()
        syncer.print_summary()

        # Return appropriate exit code
        if result.errors and not config.dry_run:
            sys.exit(1)
        sys.exit(0)

    except KeyboardInterrupt:
        logger.info("Sync interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
