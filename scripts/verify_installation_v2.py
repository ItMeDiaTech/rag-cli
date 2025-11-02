#!/usr/bin/env python3
"""Installation verification script for RAG-CLI v2.0.

Checks that all plugin components are properly installed with the new dual-package structure.
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


class InstallationVerifierV2:
    """Verifies RAG-CLI v2.0 installation completeness."""

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
        print(f"{GREEN}[PASS]{RESET} {message}")
        self.checks_passed += 1

    def check_fail(self, message: str) -> None:
        """Print failing check."""
        print(f"{RED}[FAIL]{RESET} {message}")
        self.checks_failed += 1

    def check_warn(self, message: str) -> None:
        """Print warning check."""
        print(f"{YELLOW}[WARN]{RESET} {message}")
        self.checks_warnings += 1

    def check_dir_exists(self, path: str, required: bool = True) -> bool:
        """Check if directory exists."""
        full_path = self.plugin_root / path
        exists = full_path.exists() and full_path.is_dir()

        if exists:
            self.check_pass(f"Directory exists: {path}")
        elif required:
            self.check_fail(f"Directory missing: {path}")
        else:
            self.check_warn(f"Optional directory: {path} not found")

        return exists

    def check_file_exists(self, path: str, required: bool = True) -> bool:
        """Check if file exists."""
        full_path = self.plugin_root / path
        exists = full_path.exists() and full_path.is_file()

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
            self.check_fail(f"Module import failed: {module_name} ({str(e)[:50]})")
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
            self.check_fail(f"Invalid JSON in {path}: {str(e)[:50]}")
            return False

    def verify_directory_structure(self) -> None:
        """Verify v2.0 dual-package directory structure."""
        self.print_header("Directory Structure (v2.0)")

        # Core directories
        required_dirs = [
            "src",
            "src/rag_cli",
            "src/rag_cli/core",
            "src/rag_cli/agents",
            "src/rag_cli/integrations",
            "src/rag_cli/cli",
            "src/rag_cli/utils",
            "src/rag_cli_plugin",
            "src/rag_cli_plugin/services",
            "src/rag_cli_plugin/hooks",
            "src/rag_cli_plugin/commands",
            "src/rag_cli_plugin/mcp",
            "src/rag_cli_plugin/skills",
            "config",
            "data",
            "scripts",
            ".claude-plugin",
        ]

        for dir_path in required_dirs:
            self.check_dir_exists(dir_path, required=True)

        # Check old directories DON'T exist
        old_dirs = [
            "src/core",
            "src/plugin",
            "src/monitoring",
            "src/agents",
            "src/cli",
            "src/integrations",
        ]

        print(f"\n{BOLD}Checking old v1.x directories removed:{RESET}")
        for dir_path in old_dirs:
            full_path = self.plugin_root / dir_path
            if full_path.exists():
                self.check_fail(f"Old directory still exists (should be deleted): {dir_path}")
            else:
                self.check_pass(f"Old directory removed: {dir_path}")

    def verify_required_files(self) -> None:
        """Verify required files exist."""
        self.print_header("Required Files")

        required_files = [
            "LICENSE",
            "README.md",
            "pyproject.toml",
            "requirements.txt",
            ".gitignore",
            ".claude-plugin/plugin.json",
            ".claude-plugin/hooks.json",
            "MANIFEST.in",
        ]

        for file_path in required_files:
            self.check_file_exists(file_path, required=True)

    def verify_package_init_files(self) -> None:
        """Verify package __init__.py files."""
        self.print_header("Package Structure")

        package_inits = [
            "src/rag_cli/__init__.py",
            "src/rag_cli/core/__init__.py",
            "src/rag_cli/agents/__init__.py",
            "src/rag_cli/integrations/__init__.py",
            "src/rag_cli/cli/__init__.py",
            "src/rag_cli/utils/__init__.py",
            "src/rag_cli_plugin/__init__.py",
            "src/rag_cli_plugin/services/__init__.py",
            "src/rag_cli_plugin/hooks/__init__.py",
            "src/rag_cli_plugin/commands/__init__.py",
            "src/rag_cli_plugin/mcp/__init__.py",
            "src/rag_cli_plugin/skills/__init__.py",
        ]

        for init_file in package_inits:
            self.check_file_exists(init_file, required=True)

    def verify_configuration_files(self) -> None:
        """Verify configuration files."""
        self.print_header("Configuration Files")

        config_files = [
            ("config/rag_settings.json", True),
            ("config/hook_config.json", False),
            ("config/citation_config.json", False),
            ("config/mcp.json", True),
        ]

        for config_file, required in config_files:
            if self.check_file_exists(config_file, required=required):
                self.check_json_valid(config_file)

    def verify_plugin_commands(self) -> None:
        """Verify slash commands exist."""
        self.print_header("Slash Commands")

        commands = [
            "src/rag_cli_plugin/commands/search.md",
            "src/rag_cli_plugin/commands/rag-enable.md",
            "src/rag_cli_plugin/commands/rag-disable.md",
            "src/rag_cli_plugin/commands/rag-project.md",
            "src/rag_cli_plugin/commands/update-rag.md",
        ]

        for command in commands:
            self.check_file_exists(command, required=True)

    def verify_hooks(self) -> None:
        """Verify hooks exist."""
        self.print_header("Hooks")

        hooks = [
            "src/rag_cli_plugin/hooks/user-prompt-submit.py",
            "src/rag_cli_plugin/hooks/document-indexing.py",
            "src/rag_cli_plugin/hooks/session-start.py",
        ]

        for hook in hooks:
            self.check_file_exists(hook, required=True)

    def verify_python_modules(self) -> None:
        """Verify Python modules can be imported."""
        self.print_header("Python Module Imports (v2.0 structure)")

        # Add src to path if not already there
        src_path = str(self.plugin_root / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        core_modules = [
            "rag_cli.core.document_processor",
            "rag_cli.core.embeddings",
            "rag_cli.core.vector_store",
            "rag_cli.core.retrieval_pipeline",
            "rag_cli.core.agent_orchestrator",
            "rag_cli.core.config",
        ]

        plugin_modules = [
            "rag_cli_plugin.services.logger",
            "rag_cli_plugin.services.service_manager",
            "rag_cli_plugin.mcp.unified_server",
        ]

        print(f"\n{BOLD}Core library (rag_cli):{RESET}")
        for module in core_modules:
            self.check_python_module(module)

        print(f"\n{BOLD}Plugin code (rag_cli_plugin):{RESET}")
        for module in plugin_modules:
            self.check_python_module(module)

    def verify_dependencies(self) -> None:
        """Verify required dependencies are installed."""
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
                importlib.import_module(package.replace("-", "_"))
                self.check_pass(f"Package installed: {package}")
            except ImportError:
                self.check_fail(f"Package missing: {package}")

    def print_summary(self) -> None:
        """Print verification summary."""
        self.print_header("Verification Summary")

        total_checks = self.checks_passed + self.checks_failed + self.checks_warnings

        print(f"{GREEN}Passed:{RESET}  {self.checks_passed}/{total_checks}")
        print(f"{YELLOW}Warnings:{RESET} {self.checks_warnings}/{total_checks}")
        print(f"{RED}Failed:{RESET}  {self.checks_failed}/{total_checks}")

        if self.checks_failed == 0:
            print(f"\n{GREEN}{BOLD}[SUCCESS] Installation verification PASSED{RESET}")
            print(f"{GREEN}RAG-CLI v2.0 is ready for use!{RESET}")
            return True
        else:
            print(f"\n{RED}{BOLD}[FAILED] Installation verification FAILED{RESET}")
            print(f"{RED}Please fix the issues above and try again.{RESET}")
            return False

    def run(self) -> bool:
        """Run all verification checks."""
        print(f"{BOLD}{BLUE}RAG-CLI v2.0 Installation Verification{RESET}")
        print(f"Plugin Root: {self.plugin_root}\n")

        self.verify_directory_structure()
        self.verify_required_files()
        self.verify_package_init_files()
        self.verify_configuration_files()
        self.verify_plugin_commands()
        self.verify_hooks()
        self.verify_python_modules()
        self.verify_dependencies()

        return self.print_summary()


def main():
    """Main entry point."""
    verifier = InstallationVerifierV2()
    success = verifier.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
