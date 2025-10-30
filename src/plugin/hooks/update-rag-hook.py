#!/usr/bin/env python3
"""Hook for handling /update-rag command execution.

This hook intercepts the /update-rag slash command and executes the plugin
sync process, capturing and returning the output to Claude Code.
"""

import sys
import os
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path - handle multiple possible locations
# Could be in .claude/plugins/rag-cli (when synced to Claude Code)
# or in development directory
hook_file = Path(__file__).resolve()

# Strategy 1: Check environment variable (most explicit)
project_root = None
if 'RAG_CLI_ROOT' in os.environ:
    env_path = Path(os.environ['RAG_CLI_ROOT'])
    if env_path.exists() and (env_path / 'sync_plugin.py').exists():
        project_root = env_path

# Strategy 2: Try to find project root by walking up from hook location
if project_root is None:
    current = hook_file.parent
    for _ in range(10):  # Search up to 10 levels
        # Check if this is the RAG-CLI root (has sync_plugin.py and src/core)
        if (current / 'sync_plugin.py').exists() and (current / 'src' / 'core').exists():
            project_root = current
            break
        current = current.parent

# Strategy 3: Check common installation locations
if project_root is None:
    potential_paths = [
        # User's home directory plugin location
        Path.home() / '.claude' / 'plugins' / 'rag-cli',
        # Relative to current working directory
        Path.cwd(),
        # Development path (if exists)
        Path.home() / 'Pictures' / 'DiaTech' / 'Programs' / 'DocHub' / 'development' / 'RAG-CLI',
    ]

    for path in potential_paths:
        if path.exists() and (path / 'sync_plugin.py').exists():
            project_root = path
            break

# Strategy 4: Last resort - relative to hook file location
if project_root is None:
    # Assume hook is in src/plugin/hooks/, so project root is 3 levels up
    project_root = hook_file.parents[3]
    # Validate this actually looks like project root
    if not (project_root / 'sync_plugin.py').exists():
        # If validation fails, raise clear error
        raise RuntimeError(
            f"Failed to locate RAG-CLI project root. Searched from: {hook_file}\n"
            f"Please set RAG_CLI_ROOT environment variable to the project directory.\n"
            f"Example: export RAG_CLI_ROOT=/path/to/RAG-CLI"
        )

sys.path.insert(0, str(project_root))

from src.monitoring.logger import get_logger
from src.monitoring.service_manager import ensure_services_running

logger = get_logger(__name__)


def parse_command_args(command: str) -> tuple[str, list[str]]:
    """Parse /update-rag command and extract arguments.

    Args:
        command: User command string (e.g., "/update-rag --dry-run --verbose")

    Returns:
        Tuple of (command_name, list of arguments)
    """
    parts = command.strip().split()
    if not parts or not parts[0].startswith('/update-rag'):
        return None, []

    cmd = parts[0]
    args = parts[1:] if len(parts) > 1 else []

    return cmd, args


def build_sync_command(args: list[str]) -> list[str]:
    """Build the sync_plugin.py command with arguments.

    Args:
        args: Command line arguments from user

    Returns:
        List representing the command to execute
    """
    sync_script = project_root / 'sync_plugin.py'

    cmd = ['python', str(sync_script)]

    # Validate and add arguments
    valid_args = {
        '--dry-run': '--dry-run',
        '--verbose': '--verbose',
        '-v': '--verbose',
        '--force': '--force',
        '-f': '--force',
        '--no-backup': '--no-backup',
        '--no-symlink': '--no-symlink',
    }

    for arg in args:
        if arg in valid_args:
            cmd.append(valid_args[arg])
        elif arg.startswith('--'):
            logger.warning(f"Unknown argument: {arg}")

    return cmd


def execute_sync(cmd: list[str]) -> Dict[str, Any]:
    """Execute the sync command and capture output.

    Args:
        cmd: Command list to execute

    Returns:
        Dictionary with execution results
    """
    try:
        logger.info(f"Executing: {' '.join(cmd)}")

        # Run the command and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )

        return {
            'success': result.returncode == 0,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'error': None
        }

    except subprocess.TimeoutExpired:
        error_msg = "Sync command timed out after 60 seconds"
        logger.error(error_msg)
        return {
            'success': False,
            'return_code': 124,
            'stdout': '',
            'stderr': error_msg,
            'error': 'timeout'
        }
    except FileNotFoundError as e:
        error_msg = f"Failed to find sync script: {e}"
        logger.error(error_msg)
        return {
            'success': False,
            'return_code': 127,
            'stdout': '',
            'stderr': error_msg,
            'error': 'not_found'
        }
    except Exception as e:
        error_msg = f"Sync execution failed: {e}"
        logger.error(error_msg)
        return {
            'success': False,
            'return_code': 1,
            'stdout': '',
            'stderr': error_msg,
            'error': 'execution'
        }


def format_output(result: Dict[str, Any]) -> str:
    """Format sync output for display in Claude Code.

    Args:
        result: Execution result dictionary

    Returns:
        Formatted output string
    """
    output = []

    if result['success']:
        output.append("SUCCESS: Plugin sync completed successfully!\n")
    else:
        output.append("FAILED: Plugin sync failed!\n")

    # Add stdout
    if result['stdout']:
        output.append(result['stdout'])

    # Add stderr if there was an error
    if result['stderr'] and not result['success']:
        output.append(f"\nError output:\n{result['stderr']}")

    return ''.join(output)


def process_hook(event: Dict[str, Any]) -> Dict[str, Any]:
    """Process the /update-rag command.

    Args:
        event: Hook event data containing the user's command

    Returns:
        Modified event with command output
    """
    start_time = time.time()
    logger.info("Hook execution started", hook="UpdateRagCommand")

    try:
        # Ensure monitoring services are running (auto-start if needed)
        try:
            ensure_services_running()
        except Exception as e:
            logger.debug(f"Service startup check failed: {e}")

        prompt = event.get('prompt', '')

        # Check if this is an /update-rag command
        if not prompt.strip().startswith('/update-rag'):
            logger.debug("Not an /update-rag command, skipping")
            return event

        logger.info(f"Processing /update-rag command: {prompt}")

        # Parse command
        cmd_name, args = parse_command_args(prompt)
        if not cmd_name:
            return event

        # Build sync command
        sync_cmd = build_sync_command(args)

        # Execute sync
        result = execute_sync(sync_cmd)

        # Format output
        output = format_output(result)

        # Update event with output
        event['output'] = output
        event['sync_result'] = result
        event['executed'] = True

        logger.info(f"Command execution completed: success={result['success']}")

    except Exception as e:
        logger.error(f"Hook processing failed: {e}")
        event['error'] = str(e)
        event['executed'] = False
    finally:
        execution_time = (time.time() - start_time) * 1000
        logger.info("Hook execution completed",
                   hook="UpdateRagCommand",
                   execution_time_ms=execution_time,
                   success=event.get('sync_result', {}).get('success', False))

    return event


def main():
    """Main function for the hook."""
    try:
        # Read event from stdin
        event_json = sys.stdin.read()
        event = json.loads(event_json)

        # Process the event
        result = process_hook(event)

        # Write result to stdout
        print(json.dumps(result))

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse event JSON: {e}")
        print(json.dumps({'error': f'Invalid JSON: {e}'}))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Hook failed: {e}")
        print(json.dumps({'error': str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
