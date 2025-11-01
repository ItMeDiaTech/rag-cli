"""Remove all emojis from RAG-CLI codebase.

This script systematically removes emoji characters from Python and Markdown files,
replacing them with appropriate text alternatives to prevent Unicode encoding errors
on Windows terminals and maintain professional documentation standards.
"""

import os
import re
from pathlib import Path
from typing import Dict, Tuple

# Emoji replacement mappings
EMOJI_REPLACEMENTS = {
    # Check marks and crosses
    '': '[OK]',
    '': '[ERROR]',
    '': '[WARNING]',
    '': '[WARNING]',

    # Status indicators
    '': '[SEARCH]',
    '': '[NOTE]',
    '': '[LAUNCH]',
    '': '[TIP]',
    '': '[STAR]',
    '': '[SUCCESS]',
    '': '[NEW]',
    '': '[HOT]',
    '': '[STRONG]',
    '': '[GOOD]',
    '': '[BAD]',
    '': '[TARGET]',
    '': '[STATS]',
    '': '[UP]',
    '': '[DOWN]',
    '': '[CONFIG]',
    '': '[TOOLS]',
    '': '[TOOLS]',
    '': '[SETTINGS]',
    '': '[SETTINGS]',
    '': '[LOCKED]',
    '': '[UNLOCKED]',
    '': '[KEY]',
    '': '[PACKAGE]',
    '': '[DOCS]',
    '': '[BOOK]',
    '': '[FILE]',
    '': '[FOLDER]',
    '': '[INDEX]',
    '': '[INDEX]',

    # Arrows (commonly used in documentation)
    '→': '->',
    '←': '<-',
    '↑': '^',
    '↓': 'v',
    '': '<->',
    '⇒': '=>',
    '⇐': '<=',

    # Bullets and markers
    '•': '*',
    '': '-',
    '': '*',
    '': '-',
    '': '*',
    '': '-',
    '': '*',
    '': '-',

    # Other common symbols
    '°': ' degrees',
    '™': '(TM)',
    '©': '(c)',
    '®': '(R)',
    '§': 'Section',
    '¶': 'Para',
}

# Additional pattern-based replacements for context-sensitive cases
CONTEXT_REPLACEMENTS = {
    # In status messages
    r'\s*([Ee]nabled|[Ss]uccess|[Cc]omplete|[Hh]ealthy)': r'[ENABLED] \1',
    r'\s*([Dd]isabled|[Ee]rror|[Ff]ailed|[Uu]navailable)': r'[DISABLED] \1',

    # In list items
    r'^\s*\s+': '  [OK] ',
    r'^\s*\s+': '  [ERROR] ',
    r'^\s*\s+': '  [WARNING] ',

    # In inline text
    r'\s+\s+': ' [OK] ',
    r'\s+\s+': ' [ERROR] ',
}


def remove_emojis_from_content(content: str, filepath: str) -> Tuple[str, int]:
    """Remove emojis from content and return modified content with count.

    Args:
        content: File content
        filepath: Path to file (for context-aware replacements)

    Returns:
        Tuple of (modified_content, replacement_count)
    """
    original_content = content
    modifications = 0

    # First pass: Context-sensitive replacements
    for pattern, replacement in CONTEXT_REPLACEMENTS.items():
        before = content
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        if content != before:
            modifications += content.count(replacement) - before.count(replacement)

    # Second pass: Direct emoji replacements
    for emoji, replacement in EMOJI_REPLACEMENTS.items():
        if emoji in content:
            count = content.count(emoji)
            content = content.replace(emoji, replacement)
            modifications += count

    # Third pass: Remove any remaining emoji characters using Unicode ranges
    emoji_pattern = re.compile(
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

    remaining_emojis = emoji_pattern.findall(content)
    if remaining_emojis:
        content = emoji_pattern.sub('[*]', content)
        modifications += len(remaining_emojis)

    return content, modifications


def process_file(filepath: Path) -> Tuple[bool, int]:
    """Process a single file to remove emojis.

    Args:
        filepath: Path to file

    Returns:
        Tuple of (success, modification_count)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()

        modified_content, mod_count = remove_emojis_from_content(
            original_content,
            str(filepath)
        )

        if mod_count > 0:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            return True, mod_count

        return True, 0

    except Exception as e:
        print(f"ERROR processing {filepath}: {e}")
        return False, 0


def main():
    """Main execution function."""
    project_root = Path(__file__).parent.parent

    # Files to process
    python_files = list(project_root.glob('src/**/*.py'))
    markdown_files = list(project_root.glob('*.md')) + list(project_root.glob('docs/**/*.md'))
    md_in_commands = list(project_root.glob('src/plugin/commands/*.md'))

    all_files = python_files + markdown_files + md_in_commands

    print("=" * 70)
    print("RAG-CLI Emoji Removal Tool")
    print("=" * 70)
    print(f"\nFound {len(all_files)} files to process:")
    print(f"  - {len(python_files)} Python files")
    print(f"  - {len(markdown_files)} Markdown files")
    print(f"  - {len(md_in_commands)} Command documentation files")
    print("\nProcessing...")
    print("-" * 70)

    total_modifications = 0
    files_modified = 0
    failed_files = 0

    for filepath in all_files:
        success, mod_count = process_file(filepath)

        if not success:
            failed_files += 1
        elif mod_count > 0:
            files_modified += 1
            total_modifications += mod_count
            rel_path = filepath.relative_to(project_root)
            print(f"[OK] {rel_path}: {mod_count} emojis removed")

    print("-" * 70)
    print("\nSummary:")
    print(f"  Files processed: {len(all_files)}")
    print(f"  Files modified: {files_modified}")
    print(f"  Total emojis removed: {total_modifications}")
    print(f"  Failed files: {failed_files}")
    print("\n" + "=" * 70)

    if failed_files > 0:
        print(f"[WARNING] {failed_files} files failed to process")
        return 1

    print("[SUCCESS] All emojis removed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
