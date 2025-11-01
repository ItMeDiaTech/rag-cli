"""Validate that no emoji characters exist in the codebase.

This script can be used as a pre-commit hook or CI check to prevent
emoji characters from being introduced into the codebase.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Emoji detection patterns
EMOJI_PATTERN = re.compile(
    '['
    '\U0001F300-\U0001F9FF'  # Miscellaneous Symbols and Pictographs
    '\U0001F600-\U0001F64F'  # Emoticons
    '\U0001F680-\U0001F6FF'  # Transport and Map
    '\U00002600-\U000027BF'  # Misc symbols
    '\U0001F900-\U0001F9FF'  # Supplemental Symbols
    '\U0000231A-\U0000231B'  # Watches
    '\U000023E9-\U000023FA'  # Media control
    '\U0001F1E6-\U0001F1FF'  # Regional indicators
    '\U0001FA70-\U0001FAFF'  # Extended-A
    ']+',
    flags=re.UNICODE
)

# Common emoji characters to check
SPECIFIC_EMOJIS = [
    '', '', '', '', '', '', '', '', '', '',
    '', '', '', '', '', '', '', '', '',
    '', '', '', '', '', '', '', '', '',
    '', '', '', '', '', ''
]


def check_file_for_emojis(filepath: Path) -> List[Tuple[int, str]]:
    """Check a file for emoji characters.

    Args:
        filepath: Path to file to check

    Returns:
        List of (line_number, line_content) tuples containing emojis
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        violations = []
        for line_num, line in enumerate(content.split('\n'), 1):
            # Check Unicode emoji pattern
            if EMOJI_PATTERN.search(line):
                violations.append((line_num, line.strip()[:100]))
                continue

            # Check specific emojis
            for emoji in SPECIFIC_EMOJIS:
                if emoji in line:
                    violations.append((line_num, line.strip()[:100]))
                    break

        return violations

    except Exception as e:
        print(f"[ERROR] Failed to read {filepath}: {e}", file=sys.stderr)
        return []


def validate_codebase(
    root_dir: Path,
    extensions: List[str] = None,
    exclude_dirs: List[str] = None,
    exclude_files: List[str] = None
) -> Tuple[bool, dict]:
    """Validate entire codebase for emoji usage.

    Args:
        root_dir: Root directory to scan
        extensions: List of file extensions to check (default: ['.py', '.md'])
        exclude_dirs: List of directory names to exclude
        exclude_files: List of file names to exclude (relative paths)

    Returns:
        Tuple of (is_valid, violations_dict)
    """
    if extensions is None:
        extensions = ['.py', '.md']

    if exclude_dirs is None:
        exclude_dirs = [
            '__pycache__', '.git', 'build', 'dist', '.venv', 'venv',
            'node_modules', '.pytest_cache', '.mypy_cache', '.tox',
            'tests'  # Test files may contain emojis for testing purposes
        ]

    if exclude_files is None:
        exclude_files = [
            'scripts/remove_emojis.py',  # Contains emoji mappings for replacement
            'scripts\\remove_emojis.py',  # Windows path
            'scripts/validate_no_emojis.py',  # Contains emoji patterns for detection
            'scripts\\validate_no_emojis.py',  # Windows path
            'scripts/verify_installation.py',  # Uses emojis for user-facing output
            'scripts\\verify_installation.py',  # Windows path
            'test_async_performance.py',  # Test file with emoji output
            'test_hooks.py',  # Test file with emoji output
        ]

    violations = {}

    for root, dirs, files in os.walk(root_dir):
        # Exclude specified directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            # Check if file has target extension
            if not any(file.endswith(ext) for ext in extensions):
                continue

            filepath = Path(root) / file
            rel_path = str(filepath.relative_to(root_dir))

            # Skip excluded files
            if any(rel_path.replace('\\', '/') == excl.replace('\\', '/') for excl in exclude_files):
                continue

            file_violations = check_file_for_emojis(filepath)

            if file_violations:
                violations[rel_path] = file_violations

    is_valid = len(violations) == 0
    return is_valid, violations


def main():
    """Main execution function."""
    project_root = Path(__file__).parent.parent

    print("=" * 70)
    print("RAG-CLI Emoji Validation")
    print("=" * 70)
    print("\nScanning codebase for emoji characters...")
    print("-" * 70)

    is_valid, violations = validate_codebase(project_root)

    if is_valid:
        print("\n[SUCCESS] No emojis found in codebase!")
        print("=" * 70)
        return 0

    # Report violations
    print("\n[FAILED] Emoji characters detected:\n")

    total_violations = 0
    for filepath, file_violations in sorted(violations.items()):
        print(f"\n{filepath}:")
        for line_num, line_content in file_violations[:5]:
            # Safely encode line content for Windows terminal
            try:
                safe_content = line_content[:80].encode('ascii', errors='replace').decode('ascii')
            except Exception:
                safe_content = '<content contains non-ASCII characters>'
            print(f"  Line {line_num}: {safe_content}")
            total_violations += 1

        if len(file_violations) > 5:
            print(f"  ... and {len(file_violations) - 5} more violations")
            total_violations += len(file_violations) - 5

    print("\n" + "-" * 70)
    print(f"\nTotal violations: {total_violations} in {len(violations)} files")
    print("\nTo fix: Run 'python scripts/remove_emojis.py'")
    print("=" * 70)

    return 1


if __name__ == "__main__":
    sys.exit(main())
