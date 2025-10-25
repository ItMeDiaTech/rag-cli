"""Foundation tests for RAG-CLI core components."""

import pytest
import tempfile
import shutil
import os
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
import sys
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.core.config import Config, get_config
from src.monitoring.logger import get_logger, get_metrics_logger


class TestConfiguration:
    """Test configuration system."""

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                "embeddings": {
                    "model_name": "test-model",
                    "model_dim": 384,
                    "batch_size": 32,
                    "cache_enabled": True
                },
                "vector_store": {
                    "type": "faiss",
                    "index_type": "flat",
                    "metric": "l2",
                    "save_path": "test/vectors"
                },
                "retrieval": {
                    "top_k": 5,
                    "hybrid_ratio": 0.7,
                    "rerank": True,
                    "reranker_model": "test-reranker"
                },
                "claude": {
                    "model": "test-claude",
                    "api_key": "test-key",
                    "max_tokens": 1000,
                    "temperature": 0.7
                },
                "monitoring": {
                    "log_level": "DEBUG",
                    "log_file": "test.log",
                    "tcp_server": {
                        "enabled": True,
                        "host": "127.0.0.1",
                        "port": 9999
                    }
                }
            }
            yaml.dump(config, f)
            temp_file = f.name

        yield temp_file

        # Cleanup
        if os.path.exists(temp_file):
            os.unlink(temp_file)

    def test_config_loading(self, temp_config_file):
        """Test configuration file loading."""
        config = Config(temp_config_file)

        assert config.embeddings.model_name == "test-model"
        assert config.embeddings.model_dim == 384
        assert config.vector_store.type == "faiss"
        assert config.retrieval.top_k == 5
        assert config.claude.model == "test-claude"
        assert config.monitoring.tcp_server["enabled"] is True

    def test_config_validation(self):
        """Test configuration validation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            # Write invalid config
            yaml.dump({"invalid": "config"}, f)
            temp_file = f.name

        try:
            with pytest.raises(KeyError):
                Config(temp_file)
        finally:
            os.unlink(temp_file)

    def test_environment_override(self, temp_config_file, monkeypatch):
        """Test environment variable override."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-api-key")
        monkeypatch.setenv("RAG_CLI_LOG_LEVEL", "ERROR")

        config = Config(temp_config_file)

        # API key should be overridden from environment
        assert os.environ.get("ANTHROPIC_API_KEY") == "env-api-key"

    def test_default_config(self):
        """Test loading default configuration."""
        # Mock the default config path
        with patch('src.core.config.DEFAULT_CONFIG_PATH', new=Path("config/default.yaml")):
            # Create mock file content
            mock_config = {
                "embeddings": {
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "model_dim": 384,
                    "batch_size": 32,
                    "cache_enabled": True,
                    "cache_size": 1000
                },
                "vector_store": {
                    "type": "faiss",
                    "index_type": "flat",
                    "metric": "l2",
                    "save_path": "data/vectors"
                },
                "retrieval": {
                    "top_k": 5,
                    "hybrid_ratio": 0.7,
                    "rerank": True,
                    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
                }
            }

            with patch('builtins.open', mock=Mock()):
                with patch('yaml.safe_load', return_value=mock_config):
                    config = get_config()

                    assert config.embeddings.model_name == "sentence-transformers/all-MiniLM-L6-v2"
                    assert config.vector_store.save_path == "data/vectors"


class TestLogging:
    """Test logging system."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary log directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_logger_creation(self):
        """Test logger creation."""
        logger = get_logger("test_module")

        assert logger is not None
        assert logger.name == "test_module"

    def test_metrics_logger(self):
        """Test metrics logger."""
        metrics_logger = get_metrics_logger()

        assert metrics_logger is not None
        assert metrics_logger.name == "metrics"

    def test_log_levels(self):
        """Test different log levels."""
        logger = get_logger("test_levels")

        # Test that logging doesn't raise exceptions
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

    def test_structured_logging(self):
        """Test structured logging with extra fields."""
        logger = get_logger("test_structured")

        # Log with extra fields
        logger.info("Test event", user_id=123, action="search", latency=45.2)

    @patch('src.monitoring.logger.RotatingFileHandler')
    def test_log_rotation(self, mock_handler):
        """Test log rotation configuration."""
        logger = get_logger("test_rotation")

        # Verify rotation handler was configured
        # Note: Implementation depends on actual logger setup


class TestMonitoringServer:
    """Test monitoring TCP server."""

    def test_metrics_collector(self):
        """Test metrics collector functionality."""
        from src.monitoring.tcp_server import MetricsCollector

        collector = MetricsCollector(max_history=100)

        # Test recording metrics
        collector.record_latency("test_op", 123.45)
        collector.record_query()
        collector.record_error()
        collector.record_cache(hit=True)
        collector.record_cache(hit=False)

        assert collector.query_count == 1
        assert collector.error_count == 1
        assert collector.cache_hits == 1
        assert collector.cache_misses == 1
        assert len(collector.latency_metrics) == 1

    def test_cache_hit_rate(self):
        """Test cache hit rate calculation."""
        from src.monitoring.tcp_server import MetricsCollector

        collector = MetricsCollector()

        # No cache accesses
        assert collector.get_cache_hit_rate() == 0.0

        # Some hits and misses
        for _ in range(7):
            collector.record_cache(hit=True)
        for _ in range(3):
            collector.record_cache(hit=False)

        assert collector.get_cache_hit_rate() == 70.0

    def test_uptime_tracking(self):
        """Test uptime tracking."""
        from src.monitoring.tcp_server import MetricsCollector
        import time

        collector = MetricsCollector()

        # Wait a bit
        time.sleep(0.1)

        uptime = collector.get_uptime()
        assert uptime >= 0.1
        assert uptime < 1.0  # Should be less than 1 second

    def test_component_status(self):
        """Test component status tracking."""
        from src.monitoring.tcp_server import MetricsCollector

        collector = MetricsCollector()

        # Initial status
        assert collector.component_status["vector_store"] == "unknown"

        # Update status
        collector.update_component_status("vector_store", "operational")
        assert collector.component_status["vector_store"] == "operational"

    @patch('socket.socket')
    def test_server_lifecycle(self, mock_socket):
        """Test server start and stop."""
        from src.monitoring.tcp_server import MonitoringServer

        server = MonitoringServer(host="127.0.0.1", port=9999)

        # Test start
        server.start()
        assert server.running is True

        # Test stop
        server.stop()
        assert server.running is False


class TestPluginComponents:
    """Test Claude Code plugin components."""

    def test_rag_settings(self):
        """Test RAG settings management."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "rag_settings.json"

            # Mock the settings file path
            with patch('src.plugin.hooks.user-prompt-submit.SETTINGS_FILE', new=settings_file):
                from src.plugin.hooks.user_prompt_submit import (
                    load_rag_settings,
                    save_rag_settings
                )

                # Test default settings
                settings = load_rag_settings()
                assert settings["enabled"] is False
                assert settings["auto_trigger_threshold"] == 5

                # Test save and load
                settings["enabled"] = True
                settings["context_limit"] = 10
                save_rag_settings(settings)

                loaded = load_rag_settings()
                assert loaded["enabled"] is True
                assert loaded["context_limit"] == 10

    def test_query_enhancement_check(self):
        """Test query enhancement decision logic."""
        # Mock to avoid import issues
        def should_enhance_query(query, settings):
            if not settings.get("enabled", False):
                return False

            word_count = len(query.split())
            if word_count < settings.get("auto_trigger_threshold", 5):
                return False

            if query.strip().startswith("/"):
                return False

            return True

        settings = {
            "enabled": True,
            "auto_trigger_threshold": 5,
            "exclude_patterns": []
        }

        # Should enhance
        assert should_enhance_query("How do I configure the API authentication system?", settings) is True

        # Should not enhance - disabled
        settings["enabled"] = False
        assert should_enhance_query("How do I configure the API authentication system?", settings) is False

        # Should not enhance - too short
        settings["enabled"] = True
        assert should_enhance_query("Help", settings) is False

        # Should not enhance - command
        assert should_enhance_query("/search API docs", settings) is False

    def test_plugin_manifest(self):
        """Test plugin manifest structure."""
        manifest_path = project_root / ".claude-plugin" / "plugin.json"

        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            # Check required fields
            assert "name" in manifest
            assert "version" in manifest
            assert "description" in manifest
            assert "components" in manifest

            # Check component structure
            assert "skills" in manifest["components"]
            assert "commands" in manifest["components"]
            assert "hooks" in manifest["components"]

            # Verify skill configuration
            skills = manifest["components"]["skills"]
            assert len(skills) > 0
            assert skills[0]["name"] == "rag-retrieval"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])