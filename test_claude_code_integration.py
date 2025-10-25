#!/usr/bin/env python3
"""Test script for Claude Code integration."""

import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_mode_detection():
    """Test operation mode detection."""
    print("Testing Mode Detection")
    print("=" * 50)

    from src.core.claude_code_adapter import get_adapter, OperationMode

    adapter = get_adapter()
    mode_info = adapter.get_mode_info()

    print(f"Current Mode: {mode_info['mode']}")
    print(f"Is Claude Code: {mode_info['is_claude_code']}")
    print(f"API Available: {mode_info['api_available']}")
    print(f"Should Use API: {mode_info['should_use_api']}")
    print("\nFeatures:")
    for feature, enabled in mode_info['features'].items():
        status = "[OK]" if enabled else "[DISABLED]"
        print(f"  {feature}: {status}")

    return adapter.mode


def test_context_formatting():
    """Test context formatting for Claude Code."""
    print("\n\nTesting Context Formatting")
    print("=" * 50)

    from src.core.claude_code_adapter import get_adapter

    adapter = get_adapter()

    # Create sample documents
    test_docs = [
        {
            "content": "API authentication can be done using OAuth 2.0 or API keys.",
            "source": "auth_guide.md",
            "score": 0.95,
            "metadata": {"section": "Authentication"}
        },
        {
            "content": "Rate limits apply to all API endpoints: 100 requests per minute.",
            "source": "api_limits.md",
            "score": 0.82,
            "metadata": {"section": "Limits"}
        }
    ]

    query = "How do I authenticate with the API?"

    # Format for skill response
    skill_response = adapter.format_skill_response(test_docs, query)
    print("\nSkill Response Format:")
    print(f"  Status: {skill_response['status']}")
    print(f"  Mode: {skill_response['mode']}")
    print(f"  Sources: {skill_response['sources']}")
    print(f"  Message: {skill_response.get('message', '')[:80]}...")

    # Format for hook enhancement
    hook_enhancement = adapter.format_hook_enhancement(test_docs, query)
    print("\nHook Enhancement Format:")
    print(f"  Length: {len(hook_enhancement)} characters")
    print(f"  Preview: {hook_enhancement[:150]}...")

    return True


def test_claude_integration():
    """Test Claude integration in different modes."""
    print("\n\nTesting Claude Integration")
    print("=" * 50)

    from src.core.claude_integration import ClaudeAssistant
    from src.core.claude_code_adapter import is_claude_code_mode

    # Create assistant
    assistant = ClaudeAssistant()

    print(f"Claude Code Mode: {is_claude_code_mode()}")
    print(f"Client Available: {assistant.client is not None}")

    # Create mock retrieval results
    from dataclasses import dataclass

    @dataclass
    class MockResult:
        text: str
        source: str
        score: float

    mock_results = [
        MockResult(
            text="This is test content about authentication.",
            source="test_auth.md",
            score=0.9
        )
    ]

    # Test response generation
    response = assistant.generate_response("Test query", mock_results)

    print(f"\nResponse Generated:")
    print(f"  Answer Length: {len(response.answer)} characters")
    print(f"  Sources: {response.sources}")
    print(f"  Model: {response.model}")
    print(f"  Cached: {response.cached}")

    return True


def test_skill_execution():
    """Test skill execution in Claude Code mode."""
    print("\n\nTesting Skill Execution")
    print("=" * 50)

    # Set Claude Code mode for testing
    os.environ['RAG_CLI_MODE'] = 'claude_code'

    from src.plugin.skills.rag_retrieval.retrieve import perform_retrieval

    # Test retrieval without LLM (context only)
    result = perform_retrieval(
        query="What is RAG?",
        top_k=2,
        threshold=0.3,
        use_llm=False
    )

    print(f"Retrieval Result:")
    print(f"  Query: {result['query']}")
    print(f"  Sources Found: {len(result.get('sources', []))}")
    print(f"  Mode: {result.get('mode', 'unknown')}")

    # Test with LLM (should use Claude Code mode)
    result_with_llm = perform_retrieval(
        query="What is RAG?",
        top_k=2,
        threshold=0.3,
        use_llm=True
    )

    print(f"\nWith LLM Processing:")
    print(f"  Mode: {result_with_llm.get('mode', 'unknown')}")
    print(f"  Message: {result_with_llm.get('message', '')[:80]}...")

    return True


def main():
    """Run all integration tests."""
    print("RAG-CLI Claude Code Integration Test")
    print("=" * 50)

    tests = [
        ("Mode Detection", test_mode_detection),
        ("Context Formatting", test_context_formatting),
        ("Claude Integration", test_claude_integration),
        ("Skill Execution", test_skill_execution)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, True, result))
            print(f"\n[OK] {test_name} passed")
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n[FAIL] {test_name} failed: {e}")

    # Summary
    print("\n\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for test_name, success, _ in results:
        status = "[OK]" if success else "[FAIL]"
        print(f"{test_name:25} : {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    # Mode recommendations
    mode = test_mode_detection()
    print("\n" + "=" * 50)
    print("Deployment Recommendations")
    print("=" * 50)

    if mode.value == "claude_code":
        print("System is correctly configured for Claude Code deployment.")
        print("No API key required - will use Claude's internal interface.")
    elif mode.value == "hybrid":
        print("System is in hybrid mode - will auto-detect environment.")
        print("API key available for standalone testing if needed.")
    else:
        print("System is in standalone mode.")
        print("Set RAG_CLI_MODE=claude_code for Claude Code deployment.")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())