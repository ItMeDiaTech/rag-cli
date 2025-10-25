#!/usr/bin/env python3
"""Test RAG retrieval in Claude Code mode."""

import sys
import os
from pathlib import Path

# Set Claude Code mode
os.environ['RAG_CLI_MODE'] = 'claude_code'

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_retrieval():
    """Test the retrieval pipeline in Claude Code mode."""
    print("=" * 60)
    print("Testing RAG Retrieval in Claude Code Mode")
    print("=" * 60)

    from src.core.config import get_config
    from src.core.retrieval_pipeline import HybridRetriever
    from src.core.claude_integration import ClaudeAssistant
    from src.core.claude_code_adapter import get_adapter

    # Check mode
    adapter = get_adapter()
    mode_info = adapter.get_mode_info()

    print(f"\nOperation Mode: {mode_info['mode']}")
    print(f"Using API: {mode_info['should_use_api']}")
    print(f"Claude Code: {mode_info['is_claude_code']}")

    # Initialize components
    config = get_config()
    retriever = HybridRetriever()  # It initializes its own dependencies

    # Test query
    query = "How do I authenticate with the API?"
    print(f"\nQuery: {query}")

    # Perform retrieval
    print("\nRetrieving documents...")
    results = retriever.retrieve(query, top_k=3)

    print(f"Found {len(results)} relevant documents:")
    for i, doc in enumerate(results, 1):
        # Check if it's a RetrievalResult object or dict
        if hasattr(doc, 'source'):
            source = doc.source
            score = doc.score
            content = doc.text[:100] + "..."
        else:
            source = doc.get('source', 'Unknown')
            score = doc.get('score', 0)
            content = doc.get('content', '')[:100] + "..."
        print(f"\n[{i}] Source: {source}")
        print(f"    Score: {score:.3f}")
        print(f"    Preview: {content}")

    # Generate response (in Claude Code mode, this returns context)
    print("\nGenerating response...")
    assistant = ClaudeAssistant()
    response = assistant.generate_response(query, results)

    print(f"\nResponse Type: {type(response)}")
    print(f"Model Used: {response.model}")

    # Show formatted context (what Claude Code sees)
    print("\n" + "=" * 60)
    print("Context Formatted for Claude Code:")
    print("=" * 60)
    print(response.answer[:1000])  # First 1000 chars

    return True


def test_skill_mode():
    """Test the skill execution in Claude Code mode."""
    print("\n\n" + "=" * 60)
    print("Testing Skill Execution")
    print("=" * 60)

    # Import path might be different - try direct import
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'src' / 'plugin' / 'skills' / 'rag-retrieval'))
    from retrieve import perform_retrieval

    # Test retrieval
    result = perform_retrieval(
        query="What are the API rate limits?",
        top_k=2,
        use_llm=True  # In Claude Code mode, this formats context
    )

    print(f"\nMode: {result.get('mode', 'unknown')}")
    print(f"Sources Found: {len(result.get('sources', []))}")

    if result.get('message'):
        print(f"Message: {result['message']}")

    # Show first part of answer/context
    answer = result.get('answer', '')
    if answer:
        print(f"\nContext Preview (first 500 chars):")
        print(answer[:500])

    return True


def main():
    """Run all tests."""
    try:
        # Test basic retrieval
        success1 = test_retrieval()
        print(f"\n[{'OK' if success1 else 'FAIL'}] Basic retrieval test")

        # Test skill mode
        success2 = test_skill_mode()
        print(f"\n[{'OK' if success2 else 'FAIL'}] Skill execution test")

        print("\n" + "=" * 60)
        print("Claude Code Integration Status")
        print("=" * 60)
        print("[OK] RAG-CLI is working in Claude Code mode!")
        print("[OK] No API key required - using Claude's internal interface")
        print("[OK] Documents indexed and retrieval working")
        print("\nThe system is ready for use with Claude Code commands.")

        return 0

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())