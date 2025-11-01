#!/usr/bin/env python3
"""Emoji prevention and removal system for RAG-CLI.

This script:
1. Scans all documentation files for emojis
2. Removes any found emojis
3. Can be used as a pre-commit hook
4. Provides validation for CI/CD pipelines
"""

import re
import sys
import os
from pathlib import Path
from typing import List, Tuple, Set


class EmojiPrevention:
    """System to detect and remove emojis from documentation."""

    # Comprehensive emoji pattern covering all Unicode emoji ranges
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & pictographs
        "\U0001F680-\U0001F6FF"  # Transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed characters
        "\U0001F900-\U0001F9FF"  # Supplemental symbols
        "\U00002600-\U000026FF"  # Miscellaneous symbols
        "\U00002700-\U000027BF"  # Dingbats
        "\U0001FA70-\U0001FAFF"  # Symbols and pictographs extended
        "\U0001F004-\U0001F0CF"  # Additional symbols
        "\u2705\u274C\u2714\u2716"  # Common checkmarks/crosses
        "\u2B50\u2728\u2757\u2764"  # Stars, sparkles, hearts
        "\u2194-\u2199\u21A9\u21AA"  # Arrows
        "\u231A\u231B\u2328\u23CF"  # Tech symbols
        "\u23E9-\u23F3\u23F8-\u23FA"  # Media symbols
        "\u25AA\u25AB\u25B6\u25C0"  # Geometric shapes
        "\u25FB-\u25FE\u2600-\u2604"  # Weather
        "\u260E\u2611\u2614\u2615"  # Objects
        "\u2618\u261D\u2620\u2622"  # Symbols
        "\u2623\u2626\u262A\u262E"  # Religious
        "\u262F\u2638-\u263A\u2640"  # Misc symbols
        "\u2642\u2648-\u2653\u2660"  # Zodiac & cards
        "\u2663\u2665\u2666\u2668"  # Card suits
        "\u267B\u267F\u2692-\u2697"  # Recycling & tools
        "\u2699\u269B\u269C\u26A0"  # Warning & tech
        "\u26A1\u26AA\u26AB\u26B0"  # Electric & circles
        "\u26B1\u26BD\u26BE\u26C4"  # Sports
        "\u26C5\u26C8\u26CE\u26CF"  # Weather
        "\u26D1\u26D3\u26D4\u26E9"  # Symbols
        "\u26EA\u26F0-\u26F5\u26F7"  # Buildings & transport
        "\u26F8\u26F9\u26FA\u26FD"  # Sports & misc
        "]+",
        flags=re.UNICODE
    )

    # File extensions to check
    DOCUMENTATION_EXTENSIONS = {
        '.md', '.txt', '.rst', '.json', '.yaml', '.yml',
        '.toml', '.ini', '.cfg', '.conf', '.py'
    }

    # Directories to exclude
    EXCLUDE_DIRS = {
        '.git', '__pycache__', 'node_modules', 'venv',
        'env', '.venv', 'build', 'dist', '.pytest_cache'
    }

    def __init__(self, root_path: Path = None):
        """Initialize emoji prevention system.

        Args:
            root_path: Root directory to scan (defaults to current directory)
        """
        self.root_path = root_path or Path.cwd()
        self.files_with_emojis: List[Tuple[Path, List[str]]] = []
        self.total_emojis_found = 0

    def scan_file(self, file_path: Path) -> List[str]:
        """Scan a single file for emojis.

        Args:
            file_path: Path to file to scan

        Returns:
            List of emojis found in the file
        """
        emojis_found = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                matches = self.EMOJI_PATTERN.findall(content)
                emojis_found = list(set(matches))  # Unique emojis

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

        return emojis_found

    def remove_emojis_from_file(self, file_path: Path) -> int:
        """Remove all emojis from a file.

        Args:
            file_path: Path to file to clean

        Returns:
            Number of emojis removed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            cleaned_content = self.EMOJI_PATTERN.sub('', content)

            if original_content != cleaned_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)

                # Count removed emojis
                removed = len(self.EMOJI_PATTERN.findall(original_content))
                return removed

        except Exception as e:
            print(f"Error cleaning {file_path}: {e}")

        return 0

    def scan_directory(self, directory: Path = None) -> None:
        """Scan directory recursively for files with emojis.

        Args:
            directory: Directory to scan (defaults to root_path)
        """
        directory = directory or self.root_path

        for file_path in directory.rglob('*'):
            # Skip excluded directories
            if any(excluded in file_path.parts for excluded in self.EXCLUDE_DIRS):
                continue

            # Only check documentation files
            if file_path.is_file() and file_path.suffix.lower() in self.DOCUMENTATION_EXTENSIONS:
                emojis = self.scan_file(file_path)
                if emojis:
                    self.files_with_emojis.append((file_path, emojis))
                    self.total_emojis_found += len(emojis)

    def remove_all_emojis(self) -> int:
        """Remove emojis from all files that contain them.

        Returns:
            Total number of emojis removed
        """
        total_removed = 0

        for file_path, _ in self.files_with_emojis:
            removed = self.remove_emojis_from_file(file_path)
            if removed > 0:
                print(f"Removed {removed} emojis from {file_path}")
                total_removed += removed

        return total_removed

    def generate_report(self) -> str:
        """Generate a report of emoji findings.

        Returns:
            Formatted report string
        """
        if not self.files_with_emojis:
            return "No emojis found in documentation files."

        report = []
        report.append(f"EMOJI DETECTION REPORT")
        report.append(f"=" * 50)
        report.append(f"Total files with emojis: {len(self.files_with_emojis)}")
        report.append(f"Total emojis found: {self.total_emojis_found}")
        report.append("")

        for file_path, emojis in self.files_with_emojis:
            relative_path = file_path.relative_to(self.root_path)
            report.append(f"File: {relative_path}")
            # Don't print actual emojis to avoid encoding issues on Windows
            report.append(f"  Count: {len(emojis)} unique emoji(s) found")
            report.append("")

        return "\n".join(report)

    def validate_clean(self) -> bool:
        """Validate that no emojis exist in documentation.

        Returns:
            True if no emojis found, False otherwise
        """
        self.scan_directory()
        return len(self.files_with_emojis) == 0


def main():
    """Main entry point for emoji prevention."""
    import argparse

    # Fix Windows console encoding
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    parser = argparse.ArgumentParser(description="Emoji prevention and removal for RAG-CLI")
    parser.add_argument('--scan', action='store_true', help='Scan for emojis only')
    parser.add_argument('--remove', action='store_true', help='Remove all found emojis')
    parser.add_argument('--validate', action='store_true', help='Validate no emojis exist (for CI/CD)')
    parser.add_argument('--path', type=str, default='.', help='Path to scan (default: current directory)')

    args = parser.parse_args()

    # Default to scan if no action specified
    if not any([args.scan, args.remove, args.validate]):
        args.scan = True

    # Initialize prevention system
    root_path = Path(args.path).resolve()
    prevention = EmojiPrevention(root_path)

    if args.validate:
        # Validation mode for CI/CD
        print(f"Validating no emojis in: {root_path}")
        if prevention.validate_clean():
            print("PASSED: No emojis found in documentation.")
            sys.exit(0)
        else:
            print("FAILED: Emojis detected in documentation!")
            print(prevention.generate_report())
            sys.exit(1)

    elif args.remove:
        # Scan and remove mode
        print(f"Scanning for emojis in: {root_path}")
        prevention.scan_directory()

        if prevention.files_with_emojis:
            print(prevention.generate_report())
            print("\nRemoving emojis...")
            total_removed = prevention.remove_all_emojis()
            print(f"\nTotal emojis removed: {total_removed}")
            print("Documentation is now emoji-free!")
        else:
            print("No emojis found. Documentation is clean.")

    else:
        # Scan only mode
        print(f"Scanning for emojis in: {root_path}")
        prevention.scan_directory()
        print(prevention.generate_report())

        if prevention.files_with_emojis:
            print("\nTo remove these emojis, run: python emoji_prevention.py --remove")
            sys.exit(1)


if __name__ == "__main__":
    main()