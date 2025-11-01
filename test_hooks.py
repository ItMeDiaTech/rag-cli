#!/usr/bin/env python3
"""Test script for RAG-CLI hooks."""

import json
import subprocess
import sys
from pathlib import Path

def test_hook(hook_path: Path, test_event: dict, hook_name: str) -> dict:
    """Test a hook with a sample event.

    Args:
        hook_path: Path to the hook script
        test_event: Event data to send to the hook
        hook_name: Name of the hook for logging

    Returns:
        Result from the hook
    """
    print(f"\n{'='*60}")
    print(f"Testing: {hook_name}")
    print(f"{'='*60}")
    print(f"Input event: {json.dumps(test_event, indent=2)}")

    try:
        # Run the hook
        result = subprocess.run(
            [sys.executable, str(hook_path)],
            input=json.dumps(test_event),
            capture_output=True,
            text=True,
            timeout=10
        )

        print(f"\nReturn code: {result.returncode}")

        if result.stdout:
            print(f"\nStdout:\n{result.stdout}")
            try:
                output_event = json.loads(result.stdout)
                print(f"\nParsed output event: {json.dumps(output_event, indent=2)}")
                return output_event
            except json.JSONDecodeError as e:
                print(f"\nFailed to parse output as JSON: {e}")
                return None

        if result.stderr:
            print(f"\nStderr:\n{result.stderr}")

        return None

    except subprocess.TimeoutExpired:
        print(f"\nERROR: Hook timed out after 10 seconds")
        return None
    except Exception as e:
        print(f"\nERROR: {e}")
        return None


def main():
    """Run hook tests."""
    # Get hook paths from Claude plugin directory
    plugin_dir = Path.home() / '.claude' / 'plugins' / 'rag-cli'
    hooks_dir = plugin_dir / 'hooks'

    user_prompt_hook = hooks_dir / 'user-prompt-submit.py'
    update_rag_hook = hooks_dir / 'update-rag-hook.py'

    # Verify hooks exist
    if not user_prompt_hook.exists():
        print(f"ERROR: Hook not found: {user_prompt_hook}")
        return 1
    if not update_rag_hook.exists():
        print(f"ERROR: Hook not found: {update_rag_hook}")
        return 1

    print("RAG-CLI Hook Testing")
    print("=" * 60)
    print(f"Plugin directory: {plugin_dir}")
    print(f"Hooks directory: {hooks_dir}")

    # Test 1: UserPromptSubmit hook with a technical query
    test_event_1 = {
        "prompt": "How do I configure the embedding model in RAG-CLI?",
        "session_id": "test-session-123",
        "metadata": {}
    }

    result_1 = test_hook(user_prompt_hook, test_event_1, "UserPromptSubmit Hook")

    # Test 2: UserPromptSubmit hook with a command (should skip)
    test_event_2 = {
        "prompt": "/help",
        "session_id": "test-session-123",
        "metadata": {}
    }

    result_2 = test_hook(user_prompt_hook, test_event_2, "UserPromptSubmit Hook (Command)")

    # Test 3: UpdateRagCommand hook
    test_event_3 = {
        "prompt": "/update-rag --dry-run",
        "session_id": "test-session-123",
        "metadata": {}
    }

    result_3 = test_hook(update_rag_hook, test_event_3, "UpdateRagCommand Hook")

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    tests_passed = 0
    tests_total = 3

    if result_1 is not None:
        print(" Test 1 (UserPromptSubmit): PASSED")
        tests_passed += 1
    else:
        print(" Test 1 (UserPromptSubmit): FAILED")

    if result_2 is not None:
        print(" Test 2 (UserPromptSubmit Command): PASSED")
        tests_passed += 1
    else:
        print(" Test 2 (UserPromptSubmit Command): FAILED")

    if result_3 is not None:
        print(" Test 3 (UpdateRagCommand): PASSED")
        tests_passed += 1
    else:
        print(" Test 3 (UpdateRagCommand): FAILED")

    print(f"\nResults: {tests_passed}/{tests_total} tests passed")

    return 0 if tests_passed == tests_total else 1


if __name__ == "__main__":
    sys.exit(main())
