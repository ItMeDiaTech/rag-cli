#!/usr/bin/env python3
"""
Update RAG-CLI Plugin Skill

Provides a Claude Code skill to synchronize RAG-CLI plugin files.
This skill can be invoked from Claude Code to run the plugin sync process.
"""

import sys
import json
import subprocess
import click
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

from monitoring.logger import get_logger

logger = get_logger(__name__)

@dataclass
class SyncOptions:
    """Options for plugin sync"""
    dry_run: bool = False
    verbose: bool = False
    force: bool = False
    no_backup: bool = False
    no_symlink: bool = False

class UpdateRagSkill:
    """Skill for updating RAG-CLI plugin"""

    def __init__(self):
        self.sync_script = project_root / 'sync_plugin.py'
        if not self.sync_script.exists():
            raise FileNotFoundError(f"Sync script not found: {self.sync_script}")

    def build_command(self, options: SyncOptions) -> list[str]:
        """Build the sync command with options.

        Args:
            options: Sync options

        Returns:
            Command list to execute
        """
        cmd = ['python', str(self.sync_script)]

        if options.dry_run:
            cmd.append('--dry-run')
        if options.verbose:
            cmd.append('--verbose')
        if options.force:
            cmd.append('--force')
        if options.no_backup:
            cmd.append('--no-backup')
        if options.no_symlink:
            cmd.append('--no-symlink')

        return cmd

    def execute(self, options: SyncOptions) -> Dict[str, Any]:
        """Execute the plugin sync.

        Args:
            options: Sync options

        Returns:
            Dictionary with execution results
        """
        try:
            cmd = self.build_command(options)
            logger.info(f"Executing sync: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(project_root)
            )

            return {
                'success': result.returncode == 0,
                'return_code': result.returncode,
                'output': result.stdout,
                'error': result.stderr if result.returncode != 0 else None,
                'message': 'Plugin sync completed' if result.returncode == 0 else 'Plugin sync failed'
            }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'return_code': 124,
                'output': '',
                'error': 'Sync command timed out after 60 seconds',
                'message': 'Timeout'
            }
        except FileNotFoundError:
            return {
                'success': False,
                'return_code': 127,
                'output': '',
                'error': 'Python not found or sync script missing',
                'message': 'Command not found'
            }
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Sync execution failed: {error_msg}")
            return {
                'success': False,
                'return_code': 1,
                'output': '',
                'error': error_msg,
                'message': 'Execution failed'
            }

    def preview(self) -> Dict[str, Any]:
        """Preview sync without executing (dry-run).

        Returns:
            Dictionary with preview results
        """
        options = SyncOptions(dry_run=True, verbose=True)
        return self.execute(options)

@click.command()
@click.option('--dry-run', is_flag=True, help='Preview changes without applying')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--force', '-', is_flag=True, help='Force sync (ignore timestamps)')
@click.option('--no-backup', is_flag=True, help='Skip backup creation')
@click.option('--no-symlink', is_flag=True, help='Use copy instead of symlinks')
@click.option('--preview', is_flag=True, help='Preview changes (alias for --dry-run)')
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
def update_rag(dry_run, verbose, force, no_backup, no_symlink, preview, output_json):
    """Update RAG-CLI plugin in Claude Code.

    Synchronizes RAG-CLI plugin files with your Claude Code installation,
    including commands, hooks, skills, and core modules.
    """
    try:
        # Create skill
        skill = UpdateRagSkill()

        # Parse options
        options = SyncOptions(
            dry_run=dry_run or preview,
            verbose=verbose,
            force=force,
            no_backup=no_backup,
            no_symlink=no_symlink
        )

        # Execute
        result = skill.execute(options)

        # Output
        if output_json:
            print(json.dumps(result, indent=2))
        else:
            print(result['output'])
            if result['error']:
                print(f"Error: {result['error']}", file=sys.stderr)
            print(f"\n{result['message']}")

        sys.exit(0 if result['success'] else 1)

    except Exception as e:
        if output_json:
            print(json.dumps({
                'success': False,
                'error': str(e),
                'message': 'Failed to execute update'
            }, indent=2))
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """Main entry point for the skill"""
    update_rag()

if __name__ == '__main__':
    main()
