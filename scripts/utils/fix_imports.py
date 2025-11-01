#!/usr/bin/env python3
"""Script to fix all imports by removing 'src.' prefix.

This updates all Python files to use imports without the src. prefix,
which is required for proper package installation.
"""

import re
from pathlib import Path
from typing import List, Tuple

def fix_imports_in_file(file_path: Path) -> Tuple[bool, int]:
    """Fix imports in a single file.

    Args:
        file_path: Path to Python file to fix

    Returns:
        Tuple of (was_modified, num_changes)
    """
    content = file_path.read_text(encoding='utf-8')
    original = content
    changes = 0

    # Pattern 1: from module import ...
    # Replace with: from module import ...
    pattern1 = r'from src\.([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import'
    content, count1 = re.subn(pattern1, r'from \1 import', content)
    changes += count1

    # Pattern 2: import module
    # Replace with: import module
    pattern2 = r'import src\.([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)'
    content, count2 = re.subn(pattern2, r'import \1', content)
    changes += count2

    if content != original:
        file_path.write_text(content, encoding='utf-8')
        return True, changes

    return False, 0


def find_python_files(root_dir: Path, exclude_dirs: List[str] = None) -> List[Path]:
    """Find all Python files in directory tree.

    Args:
        root_dir: Root directory to search
        exclude_dirs: List of directory names to exclude

    Returns:
        List of Python file paths
    """
    if exclude_dirs is None:
        exclude_dirs = ['.git', '__pycache__', 'venv', '.venv', 'build', 'dist', '*.egg-info']

    python_files = []

    for py_file in root_dir.rglob('*.py'):
        # Check if file is in excluded directory
        if any(excluded in py_file.parts for excluded in exclude_dirs):
            continue
        python_files.append(py_file)

    return python_files


def main():
    """Main function to fix all imports in the project."""
    project_root = Path(__file__).resolve().parents[1]

    print(f"Fixing imports in: {project_root}")
    print("=" * 80)

    # Find all Python files
    python_files = find_python_files(
        project_root,
        exclude_dirs=['.git', '__pycache__', 'venv', '.venv', 'build', 'dist', 'rag_cli.egg-info']
    )

    print(f"Found {len(python_files)} Python files\n")

    total_modified = 0
    total_changes = 0

    # Process each file
    for py_file in sorted(python_files):
        relative_path = py_file.relative_to(project_root)
        modified, changes = fix_imports_in_file(py_file)

        if modified:
            print(f"[FIXED] {relative_path}: {changes} imports fixed")
            total_modified += 1
            total_changes += changes
        else:
            print(f"[SKIP]  {relative_path}: no changes")

    print("\n" + "=" * 80)
    print(f"Summary:")
    print(f"  Files modified: {total_modified}/{len(python_files)}")
    print(f"  Total imports fixed: {total_changes}")

    if total_modified > 0:
        print(f"\nAll imports have been updated successfully!")
    else:
        print(f"\nNo imports needed updating.")


if __name__ == '__main__':
    main()
