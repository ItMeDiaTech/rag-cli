"""
Update imports to use new rag_cli package structure.

This script updates imports from old structure:
  from core.X import Y
  from agents.X import Y

To new structure:
  from rag_cli.core.X import Y
  from rag_cli.agents.X import Y
"""

import os
import re
from pathlib import Path


def update_imports_in_file(file_path: Path, base_dir: Path) -> int:
    """Update imports in a single file. Returns number of changes made."""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    changes = 0

    # Pattern 1: from core. → from rag_cli.core.
    content = re.sub(
        r'^from core\.',
        'from rag_cli.core.',
        content,
        flags=re.MULTILINE
    )

    # Pattern 2: from agents. → from rag_cli.agents.
    content = re.sub(
        r'^from agents\.',
        'from rag_cli.agents.',
        content,
        flags=re.MULTILINE
    )

    # Pattern 3: from integrations. → from rag_cli.integrations.
    content = re.sub(
        r'^from integrations\.',
        'from rag_cli.integrations.',
        content,
        flags=re.MULTILINE
    )

    # Pattern 4: from cli. → from rag_cli.cli.
    content = re.sub(
        r'^from cli\.',
        'from rag_cli.cli.',
        content,
        flags=re.MULTILINE
    )

    # Pattern 5: import core → import rag_cli.core
    content = re.sub(
        r'^import core$',
        'import rag_cli.core',
        content,
        flags=re.MULTILINE
    )

    # Pattern 6: import agents → import rag_cli.agents
    content = re.sub(
        r'^import agents$',
        'import rag_cli.agents',
        content,
        flags=re.MULTILINE
    )

    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        changes = len(re.findall(r'^from rag_cli\.', content, re.MULTILINE))
        print(f"Updated {file_path.relative_to(base_dir)}: {changes} imports fixed")

    return changes


def update_all_imports(base_dir: Path):
    """Update all imports in the rag_cli directory."""

    total_files = 0
    total_changes = 0

    print("Updating imports in rag_cli library...")
    print("=" * 60)

    for py_file in base_dir.glob('**/*.py'):
        if '__pycache__' in str(py_file):
            continue

        changes = update_imports_in_file(py_file, base_dir)
        if changes > 0:
            total_files += 1
            total_changes += changes

    print("=" * 60)
    print(f"Total files updated: {total_files}")
    print(f"Total imports fixed: {total_changes}")


if __name__ == '__main__':
    # Get the RAG-CLI root directory
    script_dir = Path(__file__).parent
    rag_cli_root = script_dir.parent.parent
    rag_cli_lib = rag_cli_root / 'src' / 'rag_cli'

    if not rag_cli_lib.exists():
        print(f"Error: {rag_cli_lib} does not exist")
        exit(1)

    update_all_imports(rag_cli_lib)
    print("\nImport updates complete!")
