"""
Comprehensive import fixer for RAG-CLI v2.0 production release.

This script fixes ALL import issues identified in the production analysis:
1. monitoring.* → rag_cli_plugin.services.*
2. core.* → rag_cli.core.*
3. agents.* → rag_cli.agents.*
4. integrations.* → rag_cli.integrations.*
5. cli.* → rag_cli.cli.*
6. plugin.* → rag_cli_plugin.*

Usage:
    python scripts/utils/fix_all_imports_v2.py
    python scripts/utils/fix_all_imports_v2.py --dry-run  # Preview changes only
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class ImportFixer:
    """Fixes all import patterns for RAG-CLI v2.0 structure."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.stats = {
            'files_scanned': 0,
            'files_modified': 0,
            'imports_fixed': 0,
            'errors': []
        }

        # Import replacement patterns (order matters!)
        self.patterns = [
            # monitoring.* → rag_cli_plugin.services.*
            (r'^from monitoring\.', 'from rag_cli_plugin.services.'),
            (r'^import monitoring\.', 'import rag_cli_plugin.services.'),
            (r'^import monitoring$', 'import rag_cli_plugin.services'),

            # plugin.* → rag_cli_plugin.*
            (r'^from plugin\.', 'from rag_cli_plugin.'),
            (r'^import plugin\.', 'import rag_cli_plugin.'),
            (r'^import plugin$', 'import rag_cli_plugin'),

            # core.* → rag_cli.core.* (only if not already rag_cli.core)
            (r'^from core\.', 'from rag_cli.core.'),
            (r'^import core\.', 'import rag_cli.core.'),
            (r'^import core$', 'import rag_cli.core'),

            # agents.* → rag_cli.agents.*
            (r'^from agents\.', 'from rag_cli.agents.'),
            (r'^import agents\.', 'import rag_cli.agents.'),
            (r'^import agents$', 'import rag_cli.agents'),

            # integrations.* → rag_cli.integrations.*
            (r'^from integrations\.', 'from rag_cli.integrations.'),
            (r'^import integrations\.', 'import rag_cli.integrations.'),
            (r'^import integrations$', 'import rag_cli.integrations'),

            # cli.* → rag_cli.cli.*
            (r'^from cli\.', 'from rag_cli.cli.'),
            (r'^import cli\.', 'import rag_cli.cli.'),
            (r'^import cli$', 'import rag_cli.cli'),
        ]

    def fix_imports_in_file(self, file_path: Path) -> Tuple[int, List[str]]:
        """Fix imports in a single file.

        Returns:
            (number_of_changes, list_of_change_descriptions)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
        except Exception as e:
            return 0, [f"Error reading file: {e}"]

        content = original_content
        changes = []

        # Apply all replacement patterns
        for pattern, replacement in self.patterns:
            matches = list(re.finditer(pattern, content, flags=re.MULTILINE))
            if matches:
                for match in matches:
                    old_line = match.group(0)
                    new_line = re.sub(pattern, replacement, old_line)
                    changes.append(f"  {old_line} -> {new_line}")

                content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

        # Write changes if any were made
        if content != original_content and not self.dry_run:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            except Exception as e:
                return 0, [f"Error writing file: {e}"]

        return len(changes), changes

    def process_directory(self, directory: Path, include_pattern: str = "**/*.py"):
        """Process all Python files in a directory."""

        print(f"\nProcessing directory: {directory}")
        print("=" * 80)

        for py_file in directory.glob(include_pattern):
            # Skip __pycache__ and other excluded directories
            if '__pycache__' in str(py_file) or '.git' in str(py_file):
                continue

            self.stats['files_scanned'] += 1

            num_changes, change_list = self.fix_imports_in_file(py_file)

            if num_changes > 0:
                self.stats['files_modified'] += 1
                self.stats['imports_fixed'] += num_changes

                rel_path = py_file.relative_to(directory.parent.parent)
                print(f"\n{rel_path}")
                for change in change_list:
                    print(change)

    def print_summary(self):
        """Print summary of changes made."""

        print("\n" + "=" * 80)
        print("IMPORT FIX SUMMARY")
        print("=" * 80)
        print(f"Files scanned:    {self.stats['files_scanned']}")
        print(f"Files modified:   {self.stats['files_modified']}")
        print(f"Imports fixed:    {self.stats['imports_fixed']}")

        if self.dry_run:
            print("\nDRY RUN - No files were modified")

        if self.stats['errors']:
            print("\nErrors encountered:")
            for error in self.stats['errors']:
                print(f"  {error}")


def main():
    """Main entry point."""

    # Parse arguments
    dry_run = '--dry-run' in sys.argv

    # Get RAG-CLI root directory
    script_dir = Path(__file__).parent
    rag_cli_root = script_dir.parent.parent

    print("RAG-CLI v2.0 Import Fixer")
    print("=" * 80)
    if dry_run:
        print("DRY RUN MODE - No files will be modified")
    print(f"Root directory: {rag_cli_root}")

    fixer = ImportFixer(dry_run=dry_run)

    # Process src/rag_cli/ directory
    rag_cli_lib = rag_cli_root / 'src' / 'rag_cli'
    if rag_cli_lib.exists():
        fixer.process_directory(rag_cli_lib)
    else:
        print(f"Warning: {rag_cli_lib} does not exist")

    # Process src/rag_cli_plugin/ directory
    rag_cli_plugin = rag_cli_root / 'src' / 'rag_cli_plugin'
    if rag_cli_plugin.exists():
        fixer.process_directory(rag_cli_plugin)
    else:
        print(f"Warning: {rag_cli_plugin} does not exist")

    # Print summary
    fixer.print_summary()

    # Return exit code
    return 0 if not fixer.stats['errors'] else 1


if __name__ == '__main__':
    sys.exit(main())
