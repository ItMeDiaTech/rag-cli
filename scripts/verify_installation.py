#!/usr/bin/env python3
"""Installation verification script for RAG-CLI.

Checks that all plugin components are properly installed and functional.
"""

import sys
import os
import json
import importlib
from pathlib import Path
from typing import List, Tuple

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


class InstallationVerifier:
    """Verifies RAG-CLI installation completeness."""

    def __init__(self):
        """Initialize verifier."""
        self.plugin_root = Path(__file__).parent.parent
        self.checks_passed = 0
        self.checks_failed = 0
        self.checks_warnings = 0

    def print_header(self, text: str) -> None:
        """Print section header."""
        print(f"\n{BOLD}{BLUE}{'=' * 60}{RESET}")
        print(f"{BOLD}{BLUE}{text}{RESET}")
        print(f"{BOLD}{BLUE}{'=' * 60}{RESET}\n")

    def check_pass(self, message: str) -> None:
        """Print passing check."""
        print(f"{GREEN}✓{RESET} {message}")
        self.checks_passed += 1

    def check_fail(self, message: str) -> None:
        """Print failing check."""
        print(f"{RED}✗{RESET} {message}")
        self.checks_failed += 1

    def check_warn(self, message: str) -> None:
        """Print warning check."""
        print(f"{YELLOW}⚠{RESET} {message}")
        self.checks_warnings += 1

    def check_file_exists(self, path: str, required: bool = True) -> bool:
        """Check if file exists."""
        full_path = self.plugin_root / path
        exists = full_path.exists()

        if exists:
            self.check_pass(f"Found: {path}")
        elif required:
            self.check_fail(f"Missing: {path}")
        else:
            self.check_warn(f"Optional: {path} not found")

        return exists

    def check_python_module(self, module_name: str) -> bool:
        """Check if Python module can be imported."""
        try:
            importlib.import_module(module_name)
            self.check_pass(f"Module importable: {module_name}")
            return True
        except ImportError as e:
            self.check_fail(f"Module import failed: {module_name} ({e})")
            return False

    def check_json_valid(self, path: str) -> bool:
        """Check if JSON file is valid."""
        full_path = self.plugin_root / path
        try:
            with open(full_path) as f:
                json.load(f)
            self.check_pass(f"Valid JSON: {path}")
            return True
        except Exception as e:
            self.check_fail(f"Invalid JSON in {path}: {e}")
            return False

    def verify_directory_structure(self) -> None:
        """Verify plugin directory structure."""
        self.print_header("Directory Structure")

        required_dirs = [
            "src",
            "src/core",
            "src/plugin",
            "src/plugin/commands",
            "src/plugin/hooks",
            "src/plugin/skills",
            "src/plugin/mcp",
            "src/agents",
            "src/agents/maf",
            "src/agents/maf/core",
            "src/agents/maf/agents",
            "src/monitoring",
            "src/integrations",
            "config",
            "data",
            "scripts",
            ".claude-plugin",
        ]

        for dir_name in required_dirs:
            path = self.plugin_root / dir_name
            if path.is_dir():
                self.check_pass(f"Directory exists: {dir_name}/")
            else:
                self.check_fail(f"Directory missing: {dir_name}/")

    def verify_required_files(self) -> None:
        """Verify all required files exist."""
        self.print_header("Required Files")

        required_files = [
            "LICENSE",
            "README.md",
            "setup.py",
            "pyproject.toml",
            ".env.example",
            "requirements.txt",
            ".gitignore",
            ".claude-plugin/plugin.json",
            ".claude-plugin/hooks.json",
            ".claude-plugin/marketplace.json",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            "src/__init__.py",
            "src/core/__init__.py",
            "src/plugin/__init__.py",
            "src/plugin/commands/__init__.py",
            "src/plugin/hooks/__init__.py",
            "src/plugin/mcp/__init__.py",
        ]

        for file_path in required_files:
            self.check_file_exists(file_path)

    def verify_optional_files(self) -> None:
        """Verify optional files."""
        self.print_header("Optional Files")

        optional_files = [
            "HOOK_FILES_REFERENCE.md",
            "MAF_INTEGRATION_v1.2.0.md",
            "IMPLEMENTATION_COMPLETE_v1.2.0.md",
            "PLUGIN_INSTALLATION_FIXES_v1.2.0.md",
        ]

        for file_path in optional_files:
            self.check_file_exists(file_path, required=False)

    def verify_config_files(self) -> None:
        """Verify configuration files."""
        self.print_header("Configuration Files")

        config_files = [
            "config/rag_settings.json",
            "config/hook_config.json",
            "config/citation_config.json",
            "config/error_config.json",
        ]

        for config_file in config_files:
            if self.check_file_exists(config_file, required=False):
                self.check_json_valid(config_file)

        # Check for template files
        self.check_file_exists("config/rag_settings.json.example", required=False)
        self.check_file_exists("config/hook_config.json.example", required=False)

    def verify_plugin_manifest(self) -> None:
        """Verify plugin manifest files."""
        self.print_header("Plugin Manifest")

        # Check plugin.json
        if self.check_file_exists(".claude-plugin/plugin.json"):
            try:
                with open(self.plugin_root / ".claude-plugin/plugin.json") as f:
                    plugin_json = json.load(f)

                # Verify key fields
                if plugin_json.get("version") == "1.2.0":
                    self.check_pass("Plugin version is 1.2.0")
                else:
                    self.check_fail(f"Plugin version is {plugin_json.get('version')}, expected 1.2.0")

                if "name" in plugin_json:
                    self.check_pass(f"Plugin name: {plugin_json['name']}")

                if "commands" in plugin_json:
                    self.check_pass("Commands directory specified")

                if "mcpServers" in plugin_json:
                    self.check_pass("MCP server configured")

            except Exception as e:
                self.check_fail(f"Failed to parse plugin.json: {e}")

    def verify_python_modules(self) -> None:
        """Verify Python modules can be imported."""
        self.print_header("Python Modules")

        modules_to_check = [
            "src.core.document_processor",
            "src.core.embeddings",
            "src.core.vector_store",
            "src.core.retrieval_pipeline",
            "src.core.agent_orchestrator",
            "src.monitoring.logger",
            "src.plugin.mcp.unified_server",
            "src.integrations.maf_connector",
            "src.agents.maf.core.agent",
        ]

        for module in modules_to_check:
            self.check_python_module(module)

    def verify_dependencies(self) -> None:
        """Verify key dependencies are installed."""
        self.print_header("Dependencies")

        required_packages = [
            "sentence_transformers",
            "faiss",
            "anthropic",
            "flask",
            "pydantic",
            "structlog",
        ]

        for package in required_packages:
            try:
                importlib.import_module(package)
                self.check_pass(f"Package installed: {package}")
            except ImportError:
                self.check_fail(f"Package not installed: {package}")

    def verify_commands(self) -> None:
        """Verify slash commands exist."""
        self.print_header("Slash Commands")

        command_files = [
            "src/plugin/commands/search.md",
            "src/plugin/commands/rag-enable.md",
            "src/plugin/commands/rag-disable.md",
            "src/plugin/commands/rag-project.md",
            "src/plugin/commands/rag-maf-config.md",
            "src/plugin/commands/update-rag.md",
        ]

        for cmd_file in command_files:
            self.check_file_exists(cmd_file)

    def verify_hooks(self) -> None:
        """Verify hooks are properly configured."""
        self.print_header("Hooks")

        hook_files = [
            "src/plugin/hooks/slash-command-blocker.py",
            "src/plugin/hooks/user-prompt-submit.py",
            "src/plugin/hooks/response-post.py",
            "src/plugin/hooks/plugin-state-change.py",
            "src/plugin/hooks/document-indexing.py",
        ]

        for hook_file in hook_files:
            self.check_file_exists(hook_file)

    def verify_maf_components(self) -> None:
        """Verify MAF components are embedded."""
        self.print_header("Multi-Agent Framework")

        # Core components
        core_files = [
            "src/agents/maf/core/__init__.py",
            "src/agents/maf/core/agent.py",
            "src/agents/maf/core/orchestrator.py",
            "src/agents/maf/core/agent_communication.py",
            "src/agents/maf/core/memory.py",
            "src/agents/maf/core/task_classifier.py",
            "src/agents/maf/core/claude_cli_unified.py",
        ]

        for file_path in core_files:
            self.check_file_exists(file_path)

        # Agents
        agent_files = [
            "src/agents/maf/agents/__init__.py",
            "src/agents/maf/agents/debugger.py",
            "src/agents/maf/agents/developer.py",
            "src/agents/maf/agents/reviewer.py",
            "src/agents/maf/agents/tester.py",
            "src/agents/maf/agents/architect.py",
            "src/agents/maf/agents/documenter.py",
            "src/agents/maf/agents/optimizer.py",
        ]

        for file_path in agent_files:
            self.check_file_exists(file_path)

    def print_summary(self) -> None:
        """Print verification summary."""
        self.print_header("Verification Summary")

        total = self.checks_passed + self.checks_failed + self.checks_warnings

        print(f"{GREEN}Passed:{RESET}  {self.checks_passed}/{total}")
        if self.checks_warnings > 0:
            print(f"{YELLOW}Warnings:{RESET} {self.checks_warnings}/{total}")
        if self.checks_failed > 0:
            print(f"{RED}Failed:{RESET}  {self.checks_failed}/{total}")

        if self.checks_failed == 0:
            print(f"\n{GREEN}{BOLD}✓ Installation verification PASSED{RESET}")
            print(f"{GREEN}RAG-CLI is ready to use!{RESET}\n")
            return 0
        else:
            print(f"\n{RED}{BOLD}✗ Installation verification FAILED{RESET}")
            print(f"{RED}Please fix the issues above and try again.{RESET}\n")
            return 1

    def run(self) -> int:
        """Run all verification checks."""
        print(f"{BOLD}{BLUE}RAG-CLI Installation Verification{RESET}")
        print(f"Plugin Root: {self.plugin_root}\n")

        self.verify_directory_structure()
        self.verify_required_files()
        self.verify_optional_files()
        self.verify_config_files()
        self.verify_plugin_manifest()
        self.verify_commands()
        self.verify_hooks()
        self.verify_maf_components()
        self.verify_python_modules()
        self.verify_dependencies()

        return self.print_summary()


def main() -> int:
    """Main entry point."""
    verifier = InstallationVerifier()
    return verifier.run()


if __name__ == "__main__":
    sys.exit(main())
