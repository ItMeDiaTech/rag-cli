#!/usr/bin/env python3
"""
Fix MCP server configuration to use unified_server instead of server.py

This script fixes the configuration in ~/.claude/claude_code_config.json
or ~/.claude/plugins/rag-cli/plugin.json to use the correct MCP server path.
"""

import json
import sys
import shutil
import argparse
from pathlib import Path
from typing import Optional


def fix_mcp_config(project_dir: Optional[Path] = None):
    """Fix MCP server configuration files.
    
    Args:
        project_dir: Optional project directory to also create/fix project-specific .mcp.json
    """
    claude_dir = Path.home() / '.claude'
    
    if not claude_dir.exists():
        print(f"Error: Claude directory not found: {claude_dir}")
        return False
    
    fixed = False
    
    # Fix .mcp.json in home directory (primary MCP config location)
    home_mcp_json = Path.home() / '.mcp.json'
    if home_mcp_json.exists():
        try:
            with open(home_mcp_json, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            updated = False
            plugin_dir = claude_dir / 'plugins' / 'rag-cli'
            if 'mcpServers' in config:
                for server_name, server_config in config['mcpServers'].items():
                    if server_name == 'rag-cli':
                        # Ensure env dict exists
                        if 'env' not in server_config:
                            server_config['env'] = {}
                        
                        # Always check and set CLAUDE_PLUGIN_ROOT if missing
                        if 'CLAUDE_PLUGIN_ROOT' not in server_config['env']:
                            print(f"Found missing CLAUDE_PLUGIN_ROOT in {home_mcp_json}")
                            server_config['env']['CLAUDE_PLUGIN_ROOT'] = str(plugin_dir)
                            updated = True
                            fixed = True
                        
                        args = server_config.get('args', [])
                        if args and len(args) > 0:
                            first_arg = str(args[0])
                            if 'server.py' in first_arg or (len(args) == 1 and 'mcp' in first_arg.lower() and first_arg.endswith('.py')):
                                print(f"Found old server.py reference in {home_mcp_json}")
                                print(f"  Old args: {args}")
                                server_config['command'] = 'python'
                                server_config['args'] = ['-m', 'plugin.mcp.unified_server']
                                server_config['env']['PYTHONUNBUFFERED'] = '1'
                                server_config['env']['RAG_CLI_MODE'] = 'claude_code'
                                # Ensure CLAUDE_PLUGIN_ROOT is set
                                if 'CLAUDE_PLUGIN_ROOT' not in server_config['env']:
                                    server_config['env']['CLAUDE_PLUGIN_ROOT'] = str(plugin_dir)
                                server_config['env']['RAG_CLI_ROOT'] = '${CLAUDE_PLUGIN_ROOT}'
                                print(f"  New args: {server_config['args']}")
                                updated = True
                                fixed = True
            
            if updated:
                backup = home_mcp_json.with_suffix('.json.backup')
                print(f"Creating backup: {backup}")
                shutil.copy2(home_mcp_json, backup)
                
                with open(home_mcp_json, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)
                print(f"Updated {home_mcp_json}")
        except Exception as e:
            print(f"Error updating {home_mcp_json}: {e}")
    
    # Fix claude_code_config.json (if it exists)
    claude_code_config = claude_dir / 'claude_code_config.json'
    if claude_code_config.exists():
        try:
            with open(claude_code_config, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            updated = False
            if 'mcpServers' in config:
                for server_name, server_config in config['mcpServers'].items():
                    if server_name == 'rag-cli':
                        # Check if it's using the old server.py path
                        args = server_config.get('args', [])
                        if args and len(args) > 0:
                            # Check for old server.py path (could be absolute or relative)
                            first_arg = str(args[0])
                            if 'server.py' in first_arg or (len(args) == 1 and 'mcp' in first_arg.lower() and first_arg.endswith('.py')):
                                print(f"Found old server.py reference in {claude_code_config}")
                                print(f"  Old args: {args}")
                                # Update to use module path
                                server_config['command'] = 'python'
                                server_config['args'] = ['-m', 'plugin.mcp.unified_server']
                                if 'env' not in server_config:
                                    server_config['env'] = {}
                                server_config['env']['PYTHONUNBUFFERED'] = '1'
                                server_config['env']['RAG_CLI_MODE'] = 'claude_code'
                                server_config['env']['RAG_CLI_ROOT'] = '${CLAUDE_PLUGIN_ROOT}'
                                # Remove cwd if it was pointing to the old path
                                if 'cwd' in server_config:
                                    cwd = server_config['cwd']
                                    if 'server.py' in str(cwd):
                                        del server_config['cwd']
                                print(f"  New args: {server_config['args']}")
                                updated = True
                                fixed = True
                        
                        # Also check command field
                        if server_config.get('command') == 'python' and 'args' in server_config:
                            args = server_config['args']
                            if isinstance(args, list) and args and 'server.py' in str(args[0]):
                                print(f"Found old server.py reference in command args")
                                server_config['args'] = ['-m', 'plugin.mcp.unified_server']
                                updated = True
                                fixed = True
            
            if updated:
                # Backup original
                backup = claude_code_config.with_suffix('.json.backup')
                print(f"Creating backup: {backup}")
                claude_code_config.rename(backup)
                
                # Write updated config
                with open(claude_code_config, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)
                print(f"Updated {claude_code_config}")
        except Exception as e:
            print(f"Error updating {claude_code_config}: {e}")
            return False
    
    # Fix plugin.json in plugins directory (if it exists)
    plugin_dir = claude_dir / 'plugins' / 'rag-cli'
    plugin_json = plugin_dir / 'plugin.json'
    
    if plugin_json.exists():
        try:
            with open(plugin_json, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            updated = False
            if 'mcpServers' in config:
                for server_name, server_config in config['mcpServers'].items():
                    if server_name == 'rag-cli':
                        args = server_config.get('args', [])
                        if args and len(args) > 0:
                            first_arg = str(args[0])
                            if 'server.py' in first_arg or (len(args) == 1 and 'mcp' in first_arg.lower() and first_arg.endswith('.py')):
                                print(f"Found old server.py reference in {plugin_json}")
                                print(f"  Old args: {args}")
                                server_config['command'] = 'python'
                                server_config['args'] = ['-m', 'plugin.mcp.unified_server']
                                if 'env' not in server_config:
                                    server_config['env'] = {}
                                server_config['env']['PYTHONUNBUFFERED'] = '1'
                                server_config['env']['RAG_CLI_MODE'] = 'claude_code'
                                # Set CLAUDE_PLUGIN_ROOT if not already set
                                if 'CLAUDE_PLUGIN_ROOT' not in server_config.get('env', {}):
                                    server_config.setdefault('env', {})['CLAUDE_PLUGIN_ROOT'] = str(plugin_dir)
                                server_config['env']['RAG_CLI_ROOT'] = '${CLAUDE_PLUGIN_ROOT}'
                                print(f"  New args: {server_config['args']}")
                                updated = True
                                fixed = True
            
            if updated:
                # Backup original
                backup = plugin_json.with_suffix('.json.backup')
                print(f"Creating backup: {backup}")
                shutil.copy2(plugin_json, backup)
                
                # Write updated config
                with open(plugin_json, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)
                print(f"Updated {plugin_json}")
        except Exception as e:
            print(f"Error updating {plugin_json}: {e}")
            return False
    
    # Check for any mcp-server.json files that might need updating
    mcp_server_json = plugin_dir / 'mcp-server.json'
    if mcp_server_json.exists():
        try:
            with open(mcp_server_json, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            updated = False
            args = config.get('args', [])
            if args and 'server.py' in str(args):
                print(f"Found old server.py reference in {mcp_server_json}")
                config['command'] = 'python'
                config['args'] = ['-m', 'plugin.mcp.unified_server']
                updated = True
                fixed = True
            
            if updated:
                backup = mcp_server_json.with_suffix('.json.backup')
                print(f"Creating backup: {backup}")
                shutil.copy2(mcp_server_json, backup)
                
                with open(mcp_server_json, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)
                print(f"Updated {mcp_server_json}")
        except Exception as e:
            print(f"Error updating {mcp_server_json}: {e}")
    
    # Optionally fix/create project-specific .mcp.json
    if project_dir:
        project_mcp_json = project_dir / '.mcp.json'
        try:
            # Get the rag-cli config from home .mcp.json
            home_mcp_json = Path.home() / '.mcp.json'
            if home_mcp_json.exists():
                with open(home_mcp_json, 'r', encoding='utf-8') as f:
                    home_config = json.load(f)
                
                if 'mcpServers' in home_config and 'rag-cli' in home_config['mcpServers']:
                    rag_config = home_config['mcpServers']['rag-cli']
                    
                    # Ensure CLAUDE_PLUGIN_ROOT is set in the config we're copying
                    if 'env' not in rag_config:
                        rag_config['env'] = {}
                    if 'CLAUDE_PLUGIN_ROOT' not in rag_config['env']:
                        plugin_dir = claude_dir / 'plugins' / 'rag-cli'
                        rag_config['env']['CLAUDE_PLUGIN_ROOT'] = str(plugin_dir)
                        print(f"Added CLAUDE_PLUGIN_ROOT to rag-cli config")
                    
                    if project_mcp_json.exists():
                        # Update existing project config
                        with open(project_mcp_json, 'r', encoding='utf-8') as f:
                            project_config = json.load(f)
                        
                        if 'mcpServers' not in project_config:
                            project_config['mcpServers'] = {}
                        
                        # Always sync to ensure project config matches home config
                        if 'rag-cli' not in project_config['mcpServers'] or \
                           project_config['mcpServers']['rag-cli'] != rag_config:
                            print(f"Syncing project .mcp.json at {project_mcp_json}")
                            project_config['mcpServers']['rag-cli'] = rag_config.copy()
                            
                            backup = project_mcp_json.with_suffix('.json.backup')
                            if backup.exists():
                                backup.unlink()
                            shutil.copy2(project_mcp_json, backup)
                            
                            with open(project_mcp_json, 'w', encoding='utf-8') as f:
                                json.dump(project_config, f, indent=2)
                            print(f"Updated {project_mcp_json}")
                            fixed = True
                    else:
                        # Create new project config
                        print(f"Creating project .mcp.json at {project_mcp_json}")
                        project_config = {
                            'mcpServers': {
                                'rag-cli': rag_config.copy()
                            }
                        }
                        
                        project_mcp_json.parent.mkdir(parents=True, exist_ok=True)
                        with open(project_mcp_json, 'w', encoding='utf-8') as f:
                            json.dump(project_config, f, indent=2)
                        print(f"Created {project_mcp_json}")
                        fixed = True
                else:
                    print(f"Warning: rag-cli server not found in home .mcp.json")
        except Exception as e:
            print(f"Error handling project .mcp.json: {e}")
            import traceback
            traceback.print_exc()
    
    return fixed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Fix MCP server configuration files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--project-dir',
        type=Path,
        help='Optional project directory to also create/fix project-specific .mcp.json'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Fixing MCP Server Configuration")
    print("=" * 60)
    print()
    
    if fix_mcp_config(project_dir=args.project_dir):
        print()
        print("[SUCCESS] Configuration fixed successfully!")
        print()
        print("Next steps:")
        print("1. Restart Claude Code to apply the changes")
        print("2. The MCP server should now use unified_server.py")
        return 0
    else:
        print()
        print("No configuration files found that need fixing, or all files are already correct.")
        print("If you're still experiencing issues, check:")
        print("  - ~/.mcp.json (primary MCP configuration)")
        print("  - ~/.claude/claude_code_config.json")
        print("  - ~/.claude/plugins/rag-cli/plugin.json")
        print("  - ~/.claude/plugins/rag-cli/mcp-server.json")
        return 0


if __name__ == '__main__':
    sys.exit(main())

