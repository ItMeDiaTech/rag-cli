#!/usr/bin/env python3
"""Script to remove sys.path manipulation code from all Python files.

This script removes:
1. sys.path.insert() calls
2. Path resolution logic for finding project root
3. RAG_CLI_ROOT environment variable usage for path setup
"""

import re
from pathlib import Path
from typing import Tuple

def remove_syspath_manipulation(content: str) -> Tuple[str, bool]:
    """Remove sys.path manipulation from file content.

    Args:
        content: Original file content

    Returns:
        Tuple of (modified_content, was_modified)
    """
    original = content
    modified = False

    # Pattern 1: Remove sys.path.insert() calls
    # Matches: sys.path.insert(0, str(project_root))
    # Matches: sys.path.insert(0, str(Path(__file__).parent.parent))
    pattern1 = r'sys\.path\.insert\([^)]+\)\s*\n'
    if re.search(pattern1, content):
        content = re.sub(pattern1, '', content)
        modified = True

    # Pattern 2: Remove complex project root finding logic
    # This is the multi-strategy path resolution code (40-60 lines in hooks)
    # Look for comment pattern: "# Add project root to path" or "# Strategy 1:"
    lines = content.split('\n')
    new_lines = []
    skip_mode = False
    skip_depth = 0

    for i, line in enumerate(lines):
        # Detect start of path manipulation block
        if any(marker in line for marker in [
            '# Add project root to path',
            '# Strategy 1:',
            '# Add parent directory to path',
            'project_root = None',
            'hook_file = Path(__file__).resolve()'
        ]):
            skip_mode = True
            skip_depth = 0
            continue

        # Count indentation to detect end of block
        if skip_mode:
            # Check if this line starts path resolution
            if line.strip().startswith('if ') and 'RAG_CLI_ROOT' in line:
                skip_depth = len(line) - len(line.lstrip())
                continue

            # Skip lines in the path resolution block
            if line.strip():
                current_indent = len(line) - len(line.lstrip())
                # If we're at the same or deeper indentation, still in block
                if current_indent > 0 or any(marker in line for marker in [
                    'project_root',
                    'potential_paths',
                    '.exists()',
                    'parents[',
                    'RAG_CLI_ROOT',
                    'raise RuntimeError',
                    'sys.path'
                ]):
                    continue
                else:
                    # We've exited the block
                    skip_mode = False
                    skip_depth = 0
            else:
                # Empty line in skip mode
                continue

        new_lines.append(line)

    new_content = '\n'.join(new_lines)

    # Clean up multiple consecutive blank lines
    new_content = re.sub(r'\n{3,}', '\n\n', new_content)

    # If content changed, mark as modified
    if new_content != original:
        return new_content, True

    return content, modified


def process_file(file_path: Path) -> Tuple[bool, str]:
    """Process a single file to remove sys.path manipulation.

    Args:
        file_path: Path to file

    Returns:
        Tuple of (was_modified, status_message)
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        new_content, modified = remove_syspath_manipulation(content)

        if modified:
            file_path.write_text(new_content, encoding='utf-8')
            return True, "sys.path code removed"
        else:
            return False, "no sys.path code found"

    except Exception as e:
        return False, f"error: {str(e)}"


def main():
    """Main function."""
    project_root = Path(__file__).resolve().parents[1]

    print(f"Removing sys.path manipulation from: {project_root}")
    print("=" * 80)

    # Target files: hooks, commands, MCP server, skills
    target_patterns = [
        "src/plugin/hooks/*.py",
        "src/plugin/commands/*.py",
        "src/plugin/mcp/*.py",
        "src/plugin/skills/**/*.py",
    ]

    total_modified = 0
    total_processed = 0

    for pattern in target_patterns:
        files = list(project_root.glob(pattern))

        for file_path in sorted(files):
            if file_path.name == '__init__.py':
                continue  # Skip __init__ files

            relative_path = file_path.relative_to(project_root)
            modified, status = process_file(file_path)

            total_processed += 1
            if modified:
                print(f"[CLEANED] {relative_path}: {status}")
                total_modified += 1
            else:
                print(f"[SKIP]    {relative_path}: {status}")

    print("\n" + "=" * 80)
    print(f"Summary:")
    print(f"  Files processed: {total_processed}")
    print(f"  Files modified: {total_modified}")

    if total_modified > 0:
        print(f"\nSuccessfully cleaned {total_modified} files!")
    else:
        print(f"\nNo files needed cleaning.")


if __name__ == '__main__':
    main()
