#!/usr/bin/env python3
"""Validate RAG-CLI plugin configuration files before release.

This script ensures all configuration files are valid and complete.
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ASCII characters for cross-platform compatibility
CHECK = '[OK]'
CROSS = '[FAIL]'

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def validate_json_file(file_path: Path, schema_name: str) -> Tuple[bool, List[str]]:
    """Validate a JSON file exists and is valid JSON.

    Args:
        file_path: Path to JSON file
        schema_name: Name for error messages

    Returns:
        Tuple of (is_valid, list of errors)
    """
    errors = []

    if not file_path.exists():
        errors.append(f"{schema_name} not found at {file_path}")
        return False, errors

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return True, []
    except json.JSONDecodeError as e:
        errors.append(f"{schema_name} is not valid JSON: {e}")
        return False, errors


def validate_plugin_json() -> Tuple[bool, List[str]]:
    """Validate plugin.json schema."""
    file_path = PROJECT_ROOT / '.claude-plugin' / 'plugin.json'
    is_valid, errors = validate_json_file(file_path, "plugin.json")

    if not is_valid:
        return False, errors

    with open(file_path, 'r') as f:
        data = json.load(f)

    # Required fields
    required_fields = ['name', 'version', 'description', 'author']
    for field in required_fields:
        if field not in data:
            errors.append(f"plugin.json missing required field: {field}")

    # Author must be an object
    if 'author' in data and isinstance(data['author'], str):
        errors.append("plugin.json author field must be an object with name, email, url")

    # Validate hooks field
    if 'hooks' in data:
        if not isinstance(data['hooks'], str):
            errors.append("plugin.json hooks field must be a string path to hooks.json")
        if not data['hooks'].startswith('./'):
            errors.append("plugin.json hooks path must start with './'")

    # Validate commands field
    if 'commands' in data:
        if not isinstance(data['commands'], list):
            errors.append("plugin.json commands must be an array")

    # Validate mcpServers field
    if 'mcpServers' in data:
        if not isinstance(data['mcpServers'], dict):
            errors.append("plugin.json mcpServers must be an object")
        else:
            for server_name, server_config in data['mcpServers'].items():
                if 'command' not in server_config:
                    errors.append(f"MCP server '{server_name}' missing 'command' field")
                if 'args' not in server_config:
                    errors.append(f"MCP server '{server_name}' missing 'args' field")

    return len(errors) == 0, errors


def validate_marketplace_json() -> Tuple[bool, List[str]]:
    """Validate marketplace.json schema."""
    file_path = PROJECT_ROOT / '.claude-plugin' / 'marketplace.json'
    is_valid, errors = validate_json_file(file_path, "marketplace.json")

    if not is_valid:
        return False, errors

    with open(file_path, 'r') as f:
        data = json.load(f)

    # Required fields
    if 'name' not in data:
        errors.append("marketplace.json missing required field: name")
    if 'plugins' not in data:
        errors.append("marketplace.json missing required field: plugins")

    # Validate plugins array
    if 'plugins' in data:
        if not isinstance(data['plugins'], list):
            errors.append("marketplace.json plugins must be an array")
        else:
            for i, plugin in enumerate(data['plugins']):
                if 'source' not in plugin:
                    errors.append(f"marketplace.json plugins[{i}] missing 'source' field")
                elif not plugin['source'].startswith('./'):
                    errors.append(f"marketplace.json plugins[{i}] source must start with './'")

    return len(errors) == 0, errors


def validate_hooks_json() -> Tuple[bool, List[str]]:
    """Validate hooks.json schema."""
    file_path = PROJECT_ROOT / '.claude-plugin' / 'hooks.json'
    is_valid, errors = validate_json_file(file_path, "hooks.json")

    if not is_valid:
        return False, errors

    with open(file_path, 'r') as f:
        data = json.load(f)

    # Must have top-level 'hooks' field
    if 'hooks' not in data:
        errors.append("hooks.json missing required top-level 'hooks' field")
        return False, errors

    if not isinstance(data['hooks'], dict):
        errors.append("hooks.json 'hooks' field must be an object")
        return False, errors

    # Valid hook event types according to Claude Code
    valid_event_types = {
        'PreToolUse',
        'PostToolUse',
        'Notification',
        'UserPromptSubmit',
        'SessionStart',
        'SessionEnd',
        'Stop',
        'SubagentStop',
        'PreCompact'
    }

    # Validate each hook event type
    for event_type, hooks_array in data['hooks'].items():
        # Check if event type is valid
        if event_type not in valid_event_types:
            errors.append(
                f"hooks.json invalid event type '{event_type}'. "
                f"Valid types: {', '.join(sorted(valid_event_types))}"
            )
            continue

        if not isinstance(hooks_array, list):
            errors.append(f"hooks.json hooks.{event_type} must be an array")
            continue

        for i, hook_entry in enumerate(hooks_array):
            if 'hooks' not in hook_entry:
                errors.append(f"hooks.json hooks.{event_type}[{i}] missing 'hooks' field")

    return len(errors) == 0, errors


def validate_file_structure() -> Tuple[bool, List[str]]:
    """Validate required files and directories exist."""
    errors = []

    required_files = [
        '.claude-plugin/plugin.json',
        '.claude-plugin/marketplace.json',
        '.claude-plugin/hooks.json',
        'src/plugin/commands/search.md',
        'src/plugin/commands/rag-enable.md',
        'src/plugin/commands/rag-disable.md',
        'src/plugin/commands/update-rag.md',
        'src/plugin/hooks/user-prompt-submit.py',
        'src/plugin/hooks/response-post.py',
        'src/plugin/mcp/unified_server.py',
    ]

    for file_path in required_files:
        full_path = PROJECT_ROOT / file_path
        if not full_path.exists():
            errors.append(f"Required file missing: {file_path}")

    return len(errors) == 0, errors


def main():
    """Run all validations."""
    print("=" * 60)
    print("RAG-CLI Plugin Validation")
    print("=" * 60)
    print()

    all_valid = True

    # Validate plugin.json
    print("[1/4] Validating plugin.json...")
    valid, errors = validate_plugin_json()
    if valid:
        print(f"{CHECK} plugin.json is valid")
    else:
        print(f"{CROSS} plugin.json validation failed:")
        for error in errors:
            print(f"  - {error}")
        all_valid = False
    print()

    # Validate marketplace.json
    print("[2/4] Validating marketplace.json...")
    valid, errors = validate_marketplace_json()
    if valid:
        print(f"{CHECK} marketplace.json is valid")
    else:
        print(f"{CROSS} marketplace.json validation failed:")
        for error in errors:
            print(f"  - {error}")
        all_valid = False
    print()

    # Validate hooks.json
    print("[3/4] Validating hooks.json...")
    valid, errors = validate_hooks_json()
    if valid:
        print(f"{CHECK} hooks.json is valid")
    else:
        print(f"{CROSS} hooks.json validation failed:")
        for error in errors:
            print(f"  - {error}")
        all_valid = False
    print()

    # Validate file structure
    print("[4/4] Validating file structure...")
    valid, errors = validate_file_structure()
    if valid:
        print(f"{CHECK} All required files present")
    else:
        print(f"{CROSS} File structure validation failed:")
        for error in errors:
            print(f"  - {error}")
        all_valid = False
    print()

    # Summary
    print("=" * 60)
    if all_valid:
        print(f"SUCCESS: All validations passed!")
        print("The plugin is ready for release.")
        return 0
    else:
        print(f"FAILED: Some validations failed.")
        print("Please fix the errors above before releasing.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
