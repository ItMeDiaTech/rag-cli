#!/usr/bin/env python3
"""Comprehensive RAG-CLI detection and verification script.

This script checks if RAG-CLI is properly installed and detectable
from both the Python environment and Claude Code plugin system.
"""

import sys
import os
import json
import socket
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional


def print_header(text: str):
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}\n")


def print_success(text: str):
    print(f" {text}")


def print_error(text: str):
    print(f" {text}", file=sys.stderr)


def print_warning(text: str):
    print(f" {text}")


def print_info(text: str):
    print(f"ℹ {text}")


class RAGDetectionVerifier:
    """Verifies RAG-CLI is detectable and functional."""

    def __init__(self):
        self.issues = []
        self.warnings = []
        self.successes = []

    def check_python_imports(self) -> bool:
        """Check if RAG-CLI modules can be imported."""
        print_header("Python Import Check")

        required_modules = [
            "core.config",
            "core.retrieval_pipeline",
            "core.vector_store",
            "core.embeddings",
            "monitoring.tcp_server",
            "plugin.mcp.unified_server"
        ]

        all_imports_ok = True

        for module in required_modules:
            try:
                exec(f"from {module} import *")
                print_success(f"Import successful: {module}")
                self.successes.append(f"Module {module} importable")
            except ImportError as e:
                print_error(f"Import failed: {module} - {e}")
                self.issues.append(f"Cannot import {module}: {e}")
                all_imports_ok = False

        return all_imports_ok

    def check_tcp_server(self) -> bool:
        """Check if RAG TCP server is accessible."""
        print_header("TCP Server Check")

        # Check if port 9999 is listening
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)

        try:
            result = sock.connect_ex(('127.0.0.1', 9999))
            sock.close()

            if result == 0:
                print_success("TCP server is listening on port 9999")
                self.successes.append("TCP server accessible")

                # Try to get status
                try:
                    import requests
                    response = requests.get('http://127.0.0.1:9999/api/status', timeout=2)
                    if response.status_code == 200:
                        print_success("TCP server API is responsive")
                        status = response.json()
                        print_info(f"Server status: {status.get('status', 'unknown')}")
                    else:
                        print_warning(f"TCP server returned status {response.status_code}")
                        self.warnings.append(f"TCP server status code: {response.status_code}")
                except Exception as e:
                    print_warning(f"Could not query TCP server API: {e}")
                    self.warnings.append(f"TCP API query failed: {e}")

                return True
            else:
                print_warning("TCP server not listening on port 9999")
                print_info("You may need to start it with: python -m monitoring.tcp_server")
                self.warnings.append("TCP server not running")
                return False

        except Exception as e:
            print_error(f"TCP connection error: {e}")
            self.issues.append(f"TCP connection failed: {e}")
            return False

    def check_claude_plugin_config(self) -> bool:
        """Check Claude Code plugin configuration."""
        print_header("Claude Code Plugin Configuration")

        claude_dir = Path.home() / '.claude'
        plugin_dir = claude_dir / 'plugins' / 'rag-cli'

        if not claude_dir.exists():
            print_error(f"Claude Code directory not found: {claude_dir}")
            self.issues.append("Claude Code not installed or configured")
            return False

        print_success(f"Claude Code directory found: {claude_dir}")

        if not plugin_dir.exists():
            print_error(f"RAG-CLI plugin not installed: {plugin_dir}")
            self.issues.append("RAG-CLI plugin directory missing")
            print_info("Run install_plugin.py to install the plugin")
            return False

        print_success(f"RAG-CLI plugin directory found: {plugin_dir}")

        # Check plugin.json
        plugin_json = plugin_dir / 'plugin.json'
        if not plugin_json.exists():
            print_error("plugin.json not found")
            self.issues.append("Plugin configuration missing")
            return False

        try:
            with open(plugin_json) as f:
                config = json.load(f)
            print_success("plugin.json is valid JSON")

            # Check for hooks
            if 'hooks' in config:
                hook_count = len(config['hooks'])
                print_info(f"Found {hook_count} hooks configured")

                # Check if user-prompt-submit hook exists (main RAG hook)
                for hook in config['hooks']:
                    if hook.get('event') == 'user-prompt-submit':
                        print_success("user-prompt-submit hook configured")
                        break
                else:
                    print_warning("user-prompt-submit hook not found (main RAG hook)")
                    self.warnings.append("Main RAG hook not configured")

            # Check for MCP servers
            if 'mcpServers' in config:
                print_success(f"MCP servers configured: {list(config['mcpServers'].keys())}")
            else:
                print_warning("No MCP servers configured")
                self.warnings.append("MCP servers not configured")

            return True

        except json.JSONDecodeError as e:
            print_error(f"Invalid JSON in plugin.json: {e}")
            self.issues.append(f"Invalid plugin.json: {e}")
            return False

    def check_environment_variables(self) -> bool:
        """Check environment variables."""
        print_header("Environment Variables")

        important_vars = [
            ('PYTHONPATH', 'Should include RAG-CLI src directory'),
            ('RAG_CLI_MODE', 'Operation mode (claude_code/standalone)'),
            ('RAG_MONITORING_HOST', 'Should be 127.0.0.1 for security'),
            ('RAG_MONITORING_PORT', 'Default is 9999')
        ]

        env_ok = True
        for var, description in important_vars:
            value = os.environ.get(var)
            if value:
                print_success(f"{var} = {value}")
            else:
                print_info(f"{var} not set ({description})")

        # Check for API keys (don't show values)
        api_keys = ['ANTHROPIC_API_KEY', 'TAVILY_API_KEY']
        for key in api_keys:
            if os.environ.get(key):
                print_success(f"{key} is set (hidden)")
            else:
                print_info(f"{key} not set")

        return env_ok

    def check_dependencies(self) -> bool:
        """Check Python dependencies."""
        print_header("Python Dependencies")

        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'list'],
                capture_output=True,
                text=True,
                timeout=10
            )

            required_packages = [
                'numpy',
                'faiss-cpu',
                'sentence-transformers',
                'flask',
                'flask-cors',
                'langchain',
                'rank-bm25',
                'aiofiles'
            ]

            installed = result.stdout
            all_found = True

            for package in required_packages:
                if package.lower() in installed.lower():
                    print_success(f"{package} installed")
                else:
                    print_error(f"{package} NOT installed")
                    self.issues.append(f"Missing dependency: {package}")
                    all_found = False

            return all_found

        except Exception as e:
            print_error(f"Could not check dependencies: {e}")
            self.issues.append(f"Dependency check failed: {e}")
            return False

    def check_data_directories(self) -> bool:
        """Check data directories exist."""
        print_header("Data Directories")

        data_dirs = [
            Path("data/vectors"),
            Path("data/documents"),
            Path("data/cache")
        ]

        all_exist = True
        for dir_path in data_dirs:
            if dir_path.exists():
                print_success(f"Directory exists: {dir_path}")

                # Check if vector store exists
                if dir_path.name == "vectors":
                    index_file = dir_path / "index.faiss"
                    if index_file.exists():
                        print_success(f"Vector index found: {index_file}")
                    else:
                        print_info("No vector index yet (will be created on first index)")

            else:
                print_warning(f"Directory missing: {dir_path}")
                print_info(f"Creating directory: {dir_path}")
                dir_path.mkdir(parents=True, exist_ok=True)

        return all_exist

    def test_rag_functionality(self) -> bool:
        """Test basic RAG functionality."""
        print_header("RAG Functionality Test")

        try:
            # Try to import and initialize core components
            from core.config import get_config
            config = get_config()
            print_success("Configuration loaded successfully")

            from core.embeddings import get_embedding_generator
            embeddings = get_embedding_generator()
            print_success("Embedding generator initialized")

            # Test embedding generation
            test_text = "This is a test query"
            embedding = embeddings.encode([test_text])[0]
            if len(embedding) == 384:  # all-MiniLM-L6-v2 dimension
                print_success(f"Embedding generated successfully (dim={len(embedding)})")
            else:
                print_warning(f"Unexpected embedding dimension: {len(embedding)}")
                self.warnings.append(f"Embedding dimension: {len(embedding)}")

            from core.vector_store import get_vector_store
            vector_store = get_vector_store()
            print_success("Vector store initialized")

            return True

        except Exception as e:
            print_error(f"RAG functionality test failed: {e}")
            self.issues.append(f"RAG test failed: {e}")
            return False

    def generate_report(self) -> None:
        """Generate final report."""
        print_header("Verification Report")

        total_issues = len(self.issues)
        total_warnings = len(self.warnings)
        total_successes = len(self.successes)

        print(f"Successes: {total_successes}")
        print(f"Warnings:  {total_warnings}")
        print(f"Issues:    {total_issues}")

        if total_issues == 0:
            print("\n RAG-CLI appears to be properly installed and configured!")
            if total_warnings > 0:
                print("\n Some warnings were found:")
                for warning in self.warnings:
                    print(f"  - {warning}")
        else:
            print("\n Issues found that may prevent RAG-CLI from working:")
            for issue in self.issues:
                print(f"  - {issue}")

            print("\n Suggested fixes:")
            if "Claude Code not installed" in str(self.issues):
                print("  1. Install Claude Code first")
            if "RAG-CLI plugin directory missing" in str(self.issues):
                print("  1. Run: python install_plugin.py")
            if "Cannot import" in str(self.issues):
                print("  1. Run: pip install -e .")
                print("  2. Ensure PYTHONPATH includes src directory")
            if "Missing dependency" in str(self.issues):
                print("  1. Run: pip install -r requirements.txt")
            if "TCP" in str(self.issues):
                print("  1. Start TCP server: python -m monitoring.tcp_server")

    def run_all_checks(self) -> bool:
        """Run all verification checks."""
        print_header("RAG-CLI Detection and Verification")
        print("This script verifies RAG-CLI is properly installed and detectable.\n")

        # Run checks
        checks = [
            self.check_environment_variables,
            self.check_dependencies,
            self.check_python_imports,
            self.check_claude_plugin_config,
            self.check_data_directories,
            self.check_tcp_server,
            self.test_rag_functionality
        ]

        all_passed = True
        for check in checks:
            try:
                if not check():
                    all_passed = False
            except Exception as e:
                print_error(f"Check failed with error: {e}")
                self.issues.append(f"Check error: {e}")
                all_passed = False

        # Generate report
        self.generate_report()

        return len(self.issues) == 0


def main():
    """Main entry point."""
    verifier = RAGDetectionVerifier()
    success = verifier.run_all_checks()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()