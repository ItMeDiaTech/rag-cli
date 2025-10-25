#!/usr/bin/env python3
"""Test script to verify foundation components (config and logging) are working."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_config():
    """Test configuration loading."""
    print("Testing configuration system...")

    from src.core.config import load_config, validate_config

    try:
        # Load configuration
        config = load_config()
        print("[OK] Configuration loaded successfully")

        # Check some key values
        print(f"  - Model: {config.claude.model}")
        print(f"  - Chunk size: {config.document_processing.chunk_size}")
        print(f"  - Embeddings model: {config.embeddings.model_name}")
        print(f"  - Log level: {config.monitoring.log_level}")
        print(f"  - TCP port: {config.monitoring.tcp_server['port']}")

        # Validate configuration
        if validate_config():
            print("[OK] Configuration validation passed")
        else:
            print("[FAIL] Configuration validation failed")
            return False

    except Exception as e:
        print(f"[FAIL] Configuration test failed: {e}")
        return False

    return True


def test_logging():
    """Test logging system."""
    print("\nTesting logging system...")

    from src.monitoring.logger import get_logger, get_metrics_logger

    try:
        # Get logger
        logger = get_logger("test_foundation")

        # Test different log levels
        logger.debug("Debug test message", component="test")
        logger.info("Info test message", status="testing")
        logger.warning("Warning test message", threshold=0.8)
        logger.error("Error test message", error_code=500)

        print("[OK] Logger created and tested")

        # Test metrics logger
        metrics = get_metrics_logger()
        metrics.record_latency("test_operation", 100.5)
        metrics.record_success("test_success")
        metrics.record_count("test_count", 42)

        print("[OK] Metrics logger tested")

        # Check if log file was created
        log_file = Path("logs/rag-cli.log")
        if log_file.exists():
            print(f"[OK] Log file created at {log_file}")
            # Show last few lines
            with open(log_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    print(f"  - Log file has {len(lines)} entries")
        else:
            print("[FAIL] Log file not created")
            return False

    except Exception as e:
        print(f"[FAIL] Logging test failed: {e}")
        return False

    return True


def test_decorators():
    """Test logging decorators."""
    print("\nTesting logging decorators...")

    from src.monitoring.logger import log_execution_time, log_api_call
    import time

    try:
        @log_execution_time
        def sample_function(delay=0.1):
            """Sample function to test decorator."""
            time.sleep(delay)
            return "success"

        @log_api_call("test_service")
        def sample_api_call():
            """Sample API call to test decorator."""
            return {"status": "ok"}

        # Test execution time decorator
        result = sample_function()
        print("[OK] Execution time decorator tested")

        # Test API call decorator
        result = sample_api_call()
        print("[OK] API call decorator tested")

    except Exception as e:
        print(f"[FAIL] Decorator test failed: {e}")
        return False

    return True


def main():
    """Run all foundation tests."""
    print("=" * 60)
    print("RAG-CLI Foundation Components Test")
    print("=" * 60)

    all_passed = True

    # Test configuration
    if not test_config():
        all_passed = False

    # Test logging
    if not test_logging():
        all_passed = False

    # Test decorators
    if not test_decorators():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] All foundation tests passed!")
        print("  Configuration and logging systems are ready.")
    else:
        print("[FAILURE] Some tests failed. Please check the errors above.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())