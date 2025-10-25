"""Configuration management for RAG-CLI.

This module handles loading configuration from YAML files, environment variables,
and provides validation and type checking for all configuration values.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
import yaml
from pydantic import BaseModel, Field, validator
import logging

logger = logging.getLogger(__name__)


def resolve_data_path(relative_path: str) -> str:
    """Resolve a data path to work from both project and plugin directories.

    When running from Claude Code plugin, uses plugin directory.
    When running from project, uses project directory.

    Args:
        relative_path: Relative path like "data/vectors/vectors.index"

    Returns:
        Absolute path resolved to correct base directory
    """
    claude_plugin_dir = Path.home() / '.claude' / 'plugins' / 'rag-cli'
    project_root = Path(__file__).resolve().parents[2]

    # Use plugin directory for data if it exists, otherwise use project directory
    if claude_plugin_dir.exists():
        base_dir = claude_plugin_dir
    else:
        base_dir = project_root

    return str(base_dir / relative_path)


class DocumentProcessingConfig(BaseModel):
    """Document processing configuration."""
    chunk_size: int = Field(500, ge=100, le=2000)
    chunk_overlap: int = Field(100, ge=0, le=500)
    separators: List[str] = ["\n\n", "\n", ". ", " ", ""]
    supported_formats: List[str] = [".md", ".txt", ".pdf", ".docx", ".html"]
    add_contextual_headers: bool = True
    metadata_fields: List[str] = ["source", "title", "section", "timestamp", "doc_type"]

    @validator('chunk_overlap')
    def overlap_less_than_chunk_size(cls, v, values):
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError('chunk_overlap must be less than chunk_size')
        return v


class EmbeddingsConfig(BaseModel):
    """Embeddings configuration."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimensions: int = 384
    batch_size: int = Field(32, ge=1, le=256)
    normalize: bool = True
    cache_size: int = Field(1000, ge=0)
    device: str = Field("cpu", pattern="^(cpu|cuda|mps)$")
    max_seq_length: int = Field(256, ge=50, le=512)


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""
    backend: str = "faiss"
    index_type: str = Field("auto", pattern="^(auto|flat|hnsw|ivf)$")
    index_params: Dict[str, Any] = {}
    save_path: str = Field(default_factory=lambda: resolve_data_path("data/vectors/vectors.index"))
    metadata_path: str = Field(default_factory=lambda: resolve_data_path("data/vectors/metadata.pkl"))
    auto_save: bool = True
    backup_enabled: bool = True
    backup_count: int = Field(3, ge=0, le=10)

    @validator('save_path', 'metadata_path', pre=True)
    @classmethod
    def resolve_paths(cls, v):
        """Resolve data paths to work from both project and plugin directories."""
        if v and v.startswith('./data/'):
            # If it's a relative data path, resolve it
            return resolve_data_path(v.lstrip('./'))
        elif v and not Path(v).is_absolute() and 'data/' in v:
            # If it's any relative path with data in it, resolve it
            return resolve_data_path(v)
        return v


class RetrievalConfig(BaseModel):
    """Retrieval pipeline configuration."""
    vector_weight: float = Field(0.7, ge=0.0, le=1.0)
    keyword_weight: float = Field(0.3, ge=0.0, le=1.0)
    initial_candidates: int = Field(10, ge=1, le=100)
    final_results: int = Field(5, ge=1, le=20)
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_batch_size: int = Field(16, ge=1, le=64)
    min_score_threshold: float = Field(0.5, ge=0.0, le=1.0)
    timeout_seconds: int = Field(10, ge=1, le=60)
    cache_enabled: bool = True
    cache_ttl_seconds: int = Field(3600, ge=0)

    @validator('keyword_weight')
    def weights_sum_to_one(cls, v, values):
        if 'vector_weight' in values:
            total = values['vector_weight'] + v
            if abs(total - 1.0) > 0.001:  # Allow small floating point errors
                raise ValueError('vector_weight + keyword_weight must equal 1.0')
        return v

    @validator('final_results')
    def final_less_than_initial(cls, v, values):
        if 'initial_candidates' in values and v > values['initial_candidates']:
            raise ValueError('final_results must be <= initial_candidates')
        return v


class ClaudeConfig(BaseModel):
    """Claude API configuration."""
    model: str = "claude-haiku-4-5-20251001"
    api_key_env: str = "ANTHROPIC_API_KEY"
    max_tokens: int = Field(1024, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    stream: bool = True
    timeout_seconds: int = Field(30, ge=1, le=300)
    max_retries: int = Field(3, ge=0, le=10)
    retry_delay: int = Field(1, ge=1, le=60)
    exponential_backoff: bool = True
    include_citations: bool = True
    citation_format: str = "[Source: {filename}]"
    system_prompt: str = ""
    track_usage: bool = True
    warn_cost_threshold: float = Field(1.0, ge=0.0)


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""
    log_level: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_format: str = Field("json", pattern="^(json|text)$")
    log_file: str = "./logs/rag-cli.log"
    log_rotation: Dict[str, Any] = {"max_bytes": 10485760, "backup_count": 5}
    track_metrics: bool = True
    metrics_port: int = Field(9090, ge=1024, le=65535)
    tcp_server: Dict[str, Any] = {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 9999,
        "endpoints": ["/status", "/logs", "/metrics", "/health"]
    }
    track_latency: bool = True
    latency_buckets: List[float] = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0]


class PluginConfig(BaseModel):
    """Claude Code plugin configuration."""
    enabled: bool = True
    skills: Dict[str, Any] = {}
    hooks: Dict[str, Any] = {}
    commands: Dict[str, Any] = {}
    settings_path: str = ".claude/settings.json"
    auto_enable_on_start: bool = False


class TestingConfig(BaseModel):
    """Testing configuration."""
    golden_dataset: str = "./tests/golden_dataset.json"
    ragas_metrics: List[str] = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    min_scores: Dict[str, float] = {
        "context_precision": 0.8,
        "context_recall": 0.8,
        "faithfulness": 0.7,
        "answer_relevancy": 0.75
    }


class PerformanceConfig(BaseModel):
    """Performance targets configuration."""
    targets: Dict[str, float] = {
        "vector_search_ms": 100,
        "end_to_end_seconds": 5,
        "embedding_speed": 0.5,
        "index_speed": 1000,
        "memory_limit_gb": 2
    }


class DevelopmentConfig(BaseModel):
    """Development settings."""
    hot_reload: bool = False
    verbose_errors: bool = True
    profile_enabled: bool = False
    mock_claude_api: bool = False
    sample_data_path: str = "./tests/sample_data"


class SecurityConfig(BaseModel):
    """Security settings."""
    validate_inputs: bool = True
    max_query_length: int = Field(1000, ge=10, le=10000)
    max_document_size_mb: int = Field(50, ge=1, le=1000)
    allowed_file_extensions: List[str] = [".md", ".txt", ".pdf", ".docx", ".html"]
    sanitize_html: bool = True
    log_queries: bool = False


class Config(BaseModel):
    """Main configuration class."""
    system: Dict[str, Any] = {}
    document_processing: DocumentProcessingConfig = DocumentProcessingConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    claude: ClaudeConfig = ClaudeConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    plugin: PluginConfig = PluginConfig()
    testing: TestingConfig = TestingConfig()
    performance: PerformanceConfig = PerformanceConfig()
    development: DevelopmentConfig = DevelopmentConfig()
    security: SecurityConfig = SecurityConfig()


class ConfigManager:
    """Manages configuration loading and merging."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.

        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self.config_path = config_path or self._get_default_config_path()
        self._config_data: Dict[str, Any] = {}
        self._config: Optional[Config] = None

    @staticmethod
    def _get_default_config_path() -> str:
        """Get default configuration file path."""
        # Look in multiple locations
        possible_paths = [
            Path("config/default.yaml"),
            Path("./config/default.yaml"),
            Path(__file__).parent.parent.parent / "config" / "default.yaml",
        ]

        for path in possible_paths:
            if path.exists():
                return str(path)

        # Return the expected path even if it doesn't exist
        return "config/default.yaml"

    def load(self, override_path: Optional[str] = None) -> Config:
        """Load configuration from file and environment.

        Args:
            override_path: Optional path to override configuration file

        Returns:
            Loaded and validated configuration
        """
        config_path = override_path or self.config_path

        # Load from YAML file
        self._config_data = self._load_yaml(config_path)

        # Override with environment variables
        self._apply_env_overrides()

        # Load from .env file if exists
        self._load_dotenv()

        # Create and validate configuration
        self._config = Config(**self._config_data)

        logger.info(f"Configuration loaded from {config_path}")
        return self._config

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Dictionary of configuration data
        """
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f) or {}
                logger.debug(f"Loaded configuration from {path}")
                return data
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {path}, using defaults")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise

    def _apply_env_overrides(self):
        """Apply environment variable overrides to configuration."""
        # Map of environment variables to config paths
        env_mappings = {
            "RAG_DEBUG": ("system", "debug"),
            "RAG_LOG_LEVEL": ("monitoring", "log_level"),
            "RAG_MODEL": ("claude", "model"),
            "RAG_CHUNK_SIZE": ("document_processing", "chunk_size"),
            "RAG_VECTOR_STORE": ("vector_store", "backend"),
            "RAG_TCP_PORT": ("monitoring", "tcp_server", "port"),
            "ANTHROPIC_API_KEY": None,  # Special handling
        }

        for env_var, config_path in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                if config_path:
                    self._set_nested(self._config_data, config_path, self._parse_value(value))
                    logger.debug(f"Applied environment override: {env_var}")

    def _load_dotenv(self):
        """Load environment variables from .env file."""
        from dotenv import load_dotenv

        # Try multiple .env locations
        env_paths = [
            Path(".env"),
            Path(".env.local"),
            Path(__file__).parent.parent.parent / ".env",
        ]

        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(env_path)
                logger.debug(f"Loaded environment from {env_path}")
                break

    @staticmethod
    def _set_nested(data: Dict[str, Any], path: tuple, value: Any):
        """Set nested dictionary value.

        Args:
            data: Dictionary to modify
            path: Tuple of keys representing path
            value: Value to set
        """
        current = data
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    @staticmethod
    def _parse_value(value: str) -> Any:
        """Parse string value to appropriate type.

        Args:
            value: String value to parse

        Returns:
            Parsed value in appropriate type
        """
        # Try to parse as JSON first (handles lists, dicts, bools)
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

        # Try to parse as number
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # Return as string
        return value

    def get(self) -> Config:
        """Get current configuration.

        Returns:
            Current configuration object
        """
        if self._config is None:
            self.load()
        return self._config

    def save(self, path: Optional[str] = None):
        """Save current configuration to file.

        Args:
            path: Path to save configuration. If None, uses original path.
        """
        save_path = path or self.config_path

        if self._config is None:
            raise ValueError("No configuration loaded")

        # Convert to dictionary
        config_dict = self._config.dict()

        # Write to YAML
        with open(save_path, 'w') as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to {save_path}")

    def validate(self) -> bool:
        """Validate current configuration.

        Returns:
            True if configuration is valid
        """
        if self._config is None:
            self.load()

        # Configuration is already validated by Pydantic
        # Add any additional custom validation here

        # Check that required directories exist or can be created
        required_dirs = [
            Path(self._config.vector_store.save_path).parent,
            Path(self._config.monitoring.log_file).parent,
        ]

        for dir_path in required_dirs:
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    logger.debug(f"Created directory: {dir_path}")
                except Exception as e:
                    logger.error(f"Cannot create required directory {dir_path}: {e}")
                    return False

        # Check API key is set
        api_key = os.environ.get(self._config.claude.api_key_env)
        if not api_key and not self._config.development.mock_claude_api:
            logger.warning(f"API key environment variable '{self._config.claude.api_key_env}' not set")

        return True


# Singleton instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> Config:
    """Get global configuration instance.

    Returns:
        Global configuration object
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.get()


def load_config(path: Optional[str] = None) -> Config:
    """Load configuration from file.

    Args:
        path: Optional path to configuration file

    Returns:
        Loaded configuration
    """
    global _config_manager
    _config_manager = ConfigManager(path)
    return _config_manager.load()


def validate_config() -> bool:
    """Validate current configuration.

    Returns:
        True if configuration is valid
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.validate()


if __name__ == "__main__":
    # Test configuration loading
    config = load_config()
    print(f"Configuration loaded successfully")
    print(f"Model: {config.claude.model}")
    print(f"Chunk size: {config.document_processing.chunk_size}")
    print(f"Log level: {config.monitoring.log_level}")

    if validate_config():
        print("Configuration is valid")
    else:
        print("Configuration validation failed")