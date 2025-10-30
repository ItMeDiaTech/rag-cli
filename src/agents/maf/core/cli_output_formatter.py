#!/usr/bin/env python3
"""
CLI Output Formatter for Multi-Agent Framework

Provides clean, structured, real-time output for Claude Code CLI users.

Design principles:
1. Real-time visibility into agent actions
2. Clear stage progression indicators
3. Structured, scannable format
4. Minimal noise, maximum signal
5. Activity acknowledgment for user confidence
"""

import sys
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

class OutputLevel(Enum):
    """Output verbosity levels"""
    QUIET = 0      # Only errors and final results
    NORMAL = 1     # Standard progress updates
    VERBOSE = 2    # Detailed agent activities
    DEBUG = 3      # Full diagnostic output

class CliOutputFormatter:
    """
    Structured output formatter for Claude Code CLI

    Provides real-time, clean feedback to users about framework activities.
    """

    def __init__(self, level: OutputLevel = OutputLevel.NORMAL):
        self.level = level
        self.start_time = None
        self.current_stage = None
        self.stage_start_time = None
        self.indent_level = 0

        # Output tracking
        self.stages_completed = []
        self.current_activities = []

    def _write(self, message: str, end: str = "\n"):
        """Write to stdout with immediate flush"""
        sys.stdout.write(message + end)
        sys.stdout.flush()

    def _get_elapsed(self) -> str:
        """Get elapsed time since start"""
        if not self.start_time:
            return "0.0s"
        elapsed = time.time() - self.start_time
        if elapsed < 1:
            return f"{elapsed*1000:.0f}ms"
        elif elapsed < 60:
            return f"{elapsed:.1f}s"
        else:
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            return f"{minutes}m {seconds}s"

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        return datetime.now().strftime("%H:%M:%S")

    def _indent(self, text: str) -> str:
        """Apply indentation to text"""
        prefix = "  " * self.indent_level
        return "\n".join(prefix + line for line in text.split("\n"))

    # ==================== Framework Lifecycle ====================

    def task_start(self, task_description: str):
        """Display task start banner"""
        self.start_time = time.time()

        self._write("")
        self._write("=" * 80)
        self._write(" MULTI-AGENT FRAMEWORK - TASK EXECUTION")
        self._write("=" * 80)
        self._write(f" Task: {task_description}")
        self._write(f" Started: {self._get_timestamp()}")
        self._write("=" * 80)
        self._write("")

    def classification_start(self):
        """Display classification phase start"""
        if self.level.value >= OutputLevel.NORMAL.value:
            self._write("[1/4] Analyzing task...")

    def classification_complete(self, task_type: str, workflow: str,
                                agents: List[str], confidence: float):
        """Display classification results"""
        if self.level.value >= OutputLevel.NORMAL.value:
            confidence_label = "HIGH" if confidence >= 0.8 else "MEDIUM" if confidence >= 0.5 else "LOW"

            self._write(f"      Type: {task_type.upper()} ({confidence_label} confidence)")
            self._write(f"      Workflow: {workflow}")
            self._write(f"      Agents: {' -> '.join([a.title() for a in agents])}")
            self._write("")

    def execution_start(self, num_stages: int):
        """Display execution phase start"""
        if self.level.value >= OutputLevel.NORMAL.value:
            self._write(f"[2/4] Executing workflow ({num_stages} stages)...")
            self._write("")

    def stage_start(self, stage_num: int, total_stages: int, agent_name: str,
                   task_summary: str):
        """Display stage start"""
        self.current_stage = stage_num
        self.stage_start_time = time.time()

        if self.level.value >= OutputLevel.NORMAL.value:
            self._write(f"  [{stage_num}/{total_stages}] {agent_name.upper()} Agent")
            self._write(f"      Task: {task_summary}")

            if self.level.value >= OutputLevel.VERBOSE.value:
                self._write(f"      Started: {self._get_timestamp()}")
                self._write("")

    def agent_activity(self, activity: str, detail: Optional[str] = None):
        """Display real-time agent activity"""
        if self.level.value >= OutputLevel.VERBOSE.value:
            timestamp = self._get_timestamp()
            self._write(f"      [{timestamp}] {activity}")

            if detail and self.level.value >= OutputLevel.DEBUG.value:
                self._write(f"                 {detail}")

    def stage_progress(self, progress_pct: int, activity: str):
        """Display progress within a stage"""
        if self.level.value >= OutputLevel.NORMAL.value:
            bar_width = 20
            filled = int(bar_width * progress_pct / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            self._write(f"\r      Progress: [{bar}] {progress_pct}% - {activity}", end="")

            if progress_pct == 100:
                self._write("")  # New line after completion

    def stage_complete(self, agent_name: str, success: bool,
                      summary: Optional[str] = None):
        """Display stage completion"""
        if not self.stage_start_time:
            return

        elapsed = time.time() - self.stage_start_time
        status = "[COMPLETED]" if success else "[FAILED]"

        if self.level.value >= OutputLevel.NORMAL.value:
            self._write(f"      Status: {status} ({elapsed:.2f}s)")

            if summary:
                self._write(f"      Result: {summary}")

            self._write("")

        self.stages_completed.append({
            'agent': agent_name,
            'success': success,
            'elapsed': elapsed
        })

    def results_start(self):
        """Display results phase start"""
        if self.level.value >= OutputLevel.NORMAL.value:
            self._write("[3/4] Compiling results...")
            self._write("")

    def results_summary(self, status: str, total_time: float,
                       stages_completed: int, stages_failed: List[str],
                       key_results: List[str]):
        """Display execution results summary"""
        self._write("=" * 80)
        self._write(" EXECUTION RESULTS")
        self._write("=" * 80)

        # Status
        if status == 'completed':
            self._write(" Status: [SUCCESS]")
        elif status == 'partial':
            self._write(" Status: [PARTIAL SUCCESS]")
        else:
            self._write(" Status: [FAILED]")

        self._write(f" Total Time: {total_time:.2f}s")
        self._write(f" Stages Completed: {stages_completed}")

        if stages_failed:
            self._write(f" Stages Failed: {', '.join(stages_failed)}")

        self._write("")

        # Key results
        if key_results and self.level.value >= OutputLevel.NORMAL.value:
            self._write(" Key Results:")
            for i, result in enumerate(key_results[:5], 1):  # Top 5
                self._write(f"   {i}. {result}")
            self._write("")

    def agent_statistics(self, agent_stats: Dict[str, Dict[str, Any]]):
        """Display agent performance statistics"""
        if self.level.value >= OutputLevel.NORMAL.value and agent_stats:
            self._write(" Agent Performance:")

            for agent_name, stats in agent_stats.items():
                if isinstance(stats, dict):
                    success_rate = stats.get('success_rate', 'N/A')
                    self._write(f"   - {agent_name.title()}: {success_rate}")

            self._write("")

    def cleanup_start(self):
        """Display cleanup phase start"""
        if self.level.value >= OutputLevel.VERBOSE.value:
            self._write("[4/4] Cleaning up resources...")

    def cleanup_complete(self):
        """Display cleanup completion"""
        if self.level.value >= OutputLevel.VERBOSE.value:
            self._write("      Cleanup completed")
            self._write("")

    def task_complete(self):
        """Display task completion banner"""
        elapsed = self._get_elapsed()

        self._write("=" * 80)
        self._write(f" Task completed in {elapsed}")
        self._write("=" * 80)
        self._write("")

    # ==================== Error Handling ====================

    def error(self, error_message: str, context: Optional[str] = None):
        """Display error message"""
        self._write("")
        self._write("!" * 80)
        self._write(" ERROR")
        self._write("!" * 80)
        self._write(f" {error_message}")

        if context:
            self._write(f" Context: {context}")

        self._write("!" * 80)
        self._write("")

    def warning(self, warning_message: str):
        """Display warning message"""
        if self.level.value >= OutputLevel.NORMAL.value:
            self._write(f" [!] Warning: {warning_message}")

    # ==================== Utility Methods ====================

    def separator(self):
        """Display separator line"""
        self._write("-" * 80)

    def info(self, message: str):
        """Display informational message"""
        if self.level.value >= OutputLevel.NORMAL.value:
            self._write(f" [i] {message}")

    def debug(self, message: str):
        """Display debug message"""
        if self.level.value >= OutputLevel.DEBUG.value:
            timestamp = self._get_timestamp()
            self._write(f" [DEBUG {timestamp}] {message}")

    def blank_line(self):
        """Display blank line"""
        self._write("")

# ==================== Convenience Factory ====================

def create_formatter(verbose: bool = False, quiet: bool = False) -> CliOutputFormatter:
    """
    Create output formatter with appropriate verbosity level

    Args:
        verbose: Enable verbose output (agent activities)
        quiet: Enable quiet mode (errors and results only)

    Returns:
        Configured CliOutputFormatter instance
    """
    if quiet:
        level = OutputLevel.QUIET
    elif verbose:
        level = OutputLevel.VERBOSE
    else:
        level = OutputLevel.NORMAL

    return CliOutputFormatter(level)
