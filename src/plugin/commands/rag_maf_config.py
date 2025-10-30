"""MAF Configuration Command for RAG-CLI Plugin.

Manages embedded Multi-Agent Framework settings and agent status.
"""

import sys
import json
import asyncio
from pathlib import Path
from typing import Optional

# Add plugin root to path
plugin_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(plugin_root.parent.parent))

from integrations.maf_connector import get_maf_connector
from monitoring.output_formatter import OutputFormatter
from monitoring.logger import get_logger

logger = get_logger(__name__)

class MAFConfigCommand:
    """Manages MAF configuration and status."""

    def __init__(self):
        """Initialize MAF config command."""
        self.maf = get_maf_connector()
        self.formatter = OutputFormatter()
        self.config_file = plugin_root / "config" / "rag_settings.json"

    def execute(self, option: Optional[str] = None) -> str:
        """Execute MAF configuration command.

        Args:
            option: Configuration option (status, enable, disable, test-connection, etc.)

        Returns:
            Formatted output string
        """
        if not option or option == "status":
            return self._show_status()
        elif option == "enable":
            return self._enable_maf()
        elif option == "disable":
            return self._disable_maf()
        elif option == "test-connection":
            return asyncio.run(self._test_connection())
        elif option == "list-agents":
            return self._list_agents()
        elif option.startswith("set-mode"):
            mode = option.split()[-1] if len(option.split()) > 1 else "PARALLEL"
            return self._set_mode(mode)
        else:
            return f"Unknown option: {option}. Use: status, enable, disable, test-connection, list-agents, set-mode"

    def _show_status(self) -> str:
        """Show MAF configuration status.

        Returns:
            Formatted status output
        """
        try:
            config = self._load_config()
            maf_config = config.get("maf", {})

            status = [
                "## MAF Configuration Status",
                "",
                f"**Status**: {'✅ Enabled' if maf_config.get('enabled') else '❌ Disabled'}",
                f"**Mode**: {maf_config.get('mode', 'parallel').upper()}",
                f"**Fallback to RAG**: {'Yes' if maf_config.get('fallback_to_rag') else 'No'}",
                f"**Notifications**: {'Enabled' if maf_config.get('show_notifications') else 'Disabled'}",
                f"**Timeout**: {maf_config.get('timeout_seconds', 30)}s",
                f"**Available Agents**: {len(maf_config.get('agents', []))} agents",
                "",
                f"**Framework Status**: {'Embedded (v1.2.0)' if self.maf.is_available() else 'Not Available'}",
                f"**Available Agents**: {', '.join(self.maf.get_available_agents()) if self.maf.is_available() else 'None'}",
            ]

            return "\n".join(status)

        except Exception as e:
            logger.error(f"Failed to show MAF status: {e}")
            return f"Error retrieving MAF status: {e}"

    def _enable_maf(self) -> str:
        """Enable MAF features.

        Returns:
            Confirmation message
        """
        try:
            config = self._load_config()
            config["ma"]["enabled"] = True
            config["orchestration"]["enable_maf"] = True
            self._save_config(config)

            return "✅ **MAF enabled successfully**\n\nParallel RAG + MAF execution is now active.\n- All 7 agents: debugger, developer, reviewer, tester, architect, documenter, optimizer\n- Execution mode: PARALLEL (simultaneous RAG + MAF)\n- Fallback to RAG-only if MAF unavailable: Yes"

        except Exception as e:
            logger.error(f"Failed to enable MAF: {e}")
            return f"❌ Error enabling MAF: {e}"

    def _disable_maf(self) -> str:
        """Disable MAF features.

        Returns:
            Confirmation message
        """
        try:
            config = self._load_config()
            config["ma"]["enabled"] = False
            config["orchestration"]["enable_maf"] = False
            self._save_config(config)

            return "✅ **MAF disabled**\n\nFalling back to RAG-only retrieval mode.\n- Vector search + keyword search (BM25)\n- Semantic caching enabled\n- Online retrieval fallback (GitHub, StackOverflow, ArXiv, Tavily)"

        except Exception as e:
            logger.error(f"Failed to disable MAF: {e}")
            return f"❌ Error disabling MAF: {e}"

    async def _test_connection(self) -> str:
        """Test MAF connector health.

        Returns:
            Health check result
        """
        try:
            health = await self.maf.health_check()

            result = [
                "## MAF Connector Health Check",
                "",
                f"**Overall Status**: {'✅ Healthy' if health['status'] == 'healthy' else '❌ Unavailable'}",
                f"**MAF Type**: {health.get('maf_type', 'embedded')}",
                f"**Location**: {health.get('maf_location', 'src/agents/maf/')}",
                f"**Version**: {health.get('maf_version', '1.2.0')}",
                f"**Available Agents**: {len(health.get('available_agents', []))}",
            ]

            if health.get('available_agents'):
                result.append("")
                result.append("**Agents Ready**:")
                for agent in health['available_agents']:
                    result.append(f"  - ✅ {agent}")

            return "\n".join(result)

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return f"❌ Health check failed: {e}"

    def _list_agents(self) -> str:
        """List available MAF agents.

        Returns:
            Formatted agent list
        """
        agents = self.maf.get_available_agents()

        if not agents:
            return "❌ No MAF agents available. Embedded MAF framework may not be initialized."

        descriptions = {
            'debugger': 'Error analysis and troubleshooting',
            'developer': 'Code implementation and generation',
            'reviewer': 'Code quality and security analysis',
            'tester': 'Test creation and validation',
            'architect': 'System design and query planning',
            'documenter': 'Documentation generation',
            'optimizer': 'Performance optimization'
        }

        result = ["## Available MAF Agents", "", "All 7 agents embedded in plugin:", ""]

        for agent in agents:
            desc = descriptions.get(agent, 'Agent')
            result.append(f"- **{agent.capitalize()}**: {desc}")

        result.append("")
        result.append("**Execution Strategy**: Parallel (all agents run simultaneously)")
        result.append("**Timeout per Agent**: 30 seconds")
        result.append("**Max Parallel Agents**: 3 (for resource management)")

        return "\n".join(result)

    def _set_mode(self, mode: str) -> str:
        """Set MAF execution mode.

        Args:
            mode: Execution mode (PARALLEL or SEQUENTIAL)

        Returns:
            Confirmation message
        """
        try:
            mode = mode.upper()
            if mode not in ["PARALLEL", "SEQUENTIAL"]:
                return f"❌ Invalid mode '{mode}'. Use: PARALLEL or SEQUENTIAL"

            config = self._load_config()
            config["ma"]["mode"] = mode.lower()
            self._save_config(config)

            if mode == "PARALLEL":
                return "✅ **Execution mode set to PARALLEL**\n\nRAG and MAF agents will execute simultaneously for maximum coverage and comprehensiveness."
            else:
                return "✅ **Execution mode set to SEQUENTIAL**\n\nRAG executes first, then MAF agents run on results. Slower but more resource-efficient."

        except Exception as e:
            logger.error(f"Failed to set mode: {e}")
            return f"❌ Error setting mode: {e}"

    def _load_config(self) -> dict:
        """Load RAG settings configuration.

        Returns:
            Configuration dictionary
        """
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")

        with open(self.config_file, 'r') as f:
            return json.load(f)

    def _save_config(self, config: dict) -> None:
        """Save RAG settings configuration.

        Args:
            config: Configuration dictionary
        """
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info("MAF configuration updated", config_file=str(self.config_file))

def main():
    """Main entry point for /rag-maf-config command."""
    option = sys.argv[1] if len(sys.argv) > 1 else None

    command = MAFConfigCommand()
    result = command.execute(option)

    print(result)

if __name__ == "__main__":
    main()
