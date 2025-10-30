#!/usr/bin/env python3
"""SlashCommandSubmit hook to prevent Claude from responding to slash commands.

This hook intercepts user prompts that start with / (slash commands) and prevents
Claude from seeing them, allowing the command to execute without AI commentary.
Returns a brief one-line status message to the user on completion.

Metadata:
  priority: 150
  enabled: true
  triggers: ["UserPromptSubmit"]
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any

# Set environment variable to suppress console logging in hooks
os.environ['CLAUDE_HOOK_CONTEXT'] = '1'
os.environ['RAG_CLI_SUPPRESS_CONSOLE'] = '1'

# Add project root to path - handle multiple possible locations
hook_file = Path(__file__).resolve()

# Strategy 1: Check environment variable (most explicit)
project_root = None
if 'RAG_CLI_ROOT' in os.environ:
    env_path = Path(os.environ['RAG_CLI_ROOT'])
    if env_path.exists() and (env_path / 'src' / 'core').exists():
        project_root = env_path

# Strategy 2: Try to find project root by walking up from hook location
if project_root is None:
    current = hook_file.parent
    for _ in range(10):  # Search up to 10 levels
        if (current / 'src' / 'core').exists() and (current / 'src' / 'monitoring').exists():
            project_root = current
            break
        current = current.parent

# Strategy 3: Check common installation locations
if project_root is None:
    potential_paths = [
        Path.home() / '.claude' / 'plugins' / 'rag-cli',
        Path.cwd(),
        Path.home() / 'Pictures' / 'DiaTech' / 'Programs' / 'DocHub' / 'development' / 'RAG-CLI',
    ]

    for path in potential_paths:
        if path.exists() and (path / 'src' / 'core').exists():
            project_root = path
            break

# Strategy 4: Last resort - relative to hook file location
if project_root is None:
    project_root = hook_file.parents[3]
    if not (project_root / 'src' / 'core').exists():
        raise RuntimeError(
            f"Failed to locate RAG-CLI project root. Searched from: {hook_file}\n"
            f"Please set RAG_CLI_ROOT environment variable to the project directory."
        )

sys.path.insert(0, str(project_root))

from src.monitoring.logger import get_logger

logger = get_logger(__name__)


def is_slash_command(text: str) -> bool:
    """Check if the text is a slash command.

    Args:
        text: User input text

    Returns:
        True if text starts with /, False otherwise
    """
    return text.strip().startswith('/')


def extract_command_name(text: str) -> str:
    """Extract the command name from slash command text.

    Args:
        text: Slash command text

    Returns:
        Command name (without the /)
    """
    stripped = text.strip()
    if not stripped.startswith('/'):
        return ""

    # Remove the / and get the first word
    parts = stripped[1:].split()
    return parts[0] if parts else ""


def process_hook(event: Dict[str, Any]) -> Dict[str, Any]:
    """Process the UserPromptSubmit hook event.

    This hook intercepts slash commands and prevents Claude from responding,
    allowing the command to execute cleanly without AI commentary.

    Args:
        event: Hook event from Claude Code

    Returns:
        Modified event (blocked if slash command) or original event
    """
    try:
        # Extract user prompt from event
        prompt = event.get('prompt', '')

        if not prompt:
            logger.debug("No prompt in event, passing through")
            return event

        # Check if this is a slash command
        if not is_slash_command(prompt):
            logger.debug("Not a slash command, passing through")
            return event

        # Extract command name for status message
        command_name = extract_command_name(prompt)

        logger.info(f"Slash command detected: /{command_name}")

        # Block the prompt from reaching Claude by setting it to empty
        # The slash command will still execute, but Claude won't respond
        blocked_event = event.copy()
        blocked_event['prompt'] = ''

        # Add metadata to indicate this was blocked
        if 'metadata' not in blocked_event:
            blocked_event['metadata'] = {}

        blocked_event['metadata']['slash_command_blocked'] = True
        blocked_event['metadata']['original_command'] = prompt
        blocked_event['metadata']['command_name'] = command_name

        # Create a brief status message for the user
        status_message = f"Executing command: /{command_name}"

        # Add status to response field if it exists
        if 'response' in blocked_event:
            blocked_event['response'] = status_message

        logger.info(f"Blocked slash command /{command_name} from Claude response")

        # Try to send event to monitoring dashboard if available
        try:
            from src.monitoring.service_manager import ServiceManager

            manager = ServiceManager()
            if manager.is_healthy():
                manager.submit_event({
                    'type': 'slash_command_blocked',
                    'command': f"/{command_name}",
                    'timestamp': event.get('timestamp'),
                })
        except Exception as e:
            # Silently fail if monitoring not available
            logger.debug(f"Could not send event to monitoring: {e}")

        return blocked_event

    except Exception as e:
        logger.error(f"Error in slash-command-blocker hook: {e}", exc_info=True)
        # On error, pass through original event to avoid breaking workflow
        return event


def main():
    """Main entry point for hook execution.

    Reads JSON event from stdin, processes it, and writes result to stdout.
    """
    try:
        # Read event from stdin
        event_json = sys.stdin.read()
        event = json.loads(event_json)

        # Process the event
        result = process_hook(event)

        # Write result to stdout
        print(json.dumps(result))

    except Exception as e:
        logger.error(f"Fatal error in slash-command-blocker hook: {e}", exc_info=True)
        # On fatal error, pass through original input
        try:
            print(event_json)
        except:
            print("{}")
        sys.exit(1)


if __name__ == '__main__':
    main()
