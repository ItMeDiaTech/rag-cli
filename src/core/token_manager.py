#!/usr/bin/env python3
"""Token management for RAG-CLI.

Handles secure token storage and retrieval from .env files and user prompts.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict
import getpass


class TokenManager:
    """Manages API tokens for RAG-CLI."""

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize token manager.

        Args:
            project_root: Root directory of the RAG-CLI project
        """
        self.project_root = project_root or Path(__file__).resolve().parents[2]
        self.env_file = self.project_root / '.env'
        self.config_file = self.project_root / 'config' / 'default.yaml'

    def load_env_file(self) -> Dict[str, str]:
        """Load environment variables from .env file.

        Returns:
            Dictionary of environment variables
        """
        env_vars = {}

        if not self.env_file.exists():
            return env_vars

        try:
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()

                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue

                    # Parse KEY=VALUE format
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()

                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]

                        env_vars[key] = value

        except Exception as e:
            print(f"Warning: Failed to read .env file: {e}")

        return env_vars

    def get_github_token(self) -> Optional[str]:
        """Get GitHub token from .env file or environment.

        Returns:
            GitHub token if found, None otherwise
        """
        # Check .env file first
        env_vars = self.load_env_file()
        token = env_vars.get('GITHUB_TOKEN') or env_vars.get('GH_TOKEN')

        if token:
            return token

        # Fall back to environment variables
        token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')

        return token

    def get_stackoverflow_key(self) -> Optional[str]:
        """Get Stack Overflow API key from .env file or environment.

        Returns:
            Stack Overflow API key if found, None otherwise
        """
        # Check .env file first
        env_vars = self.load_env_file()
        key = env_vars.get('STACKOVERFLOW_KEY') or env_vars.get('SO_API_KEY')

        if key:
            return key

        # Fall back to environment variables
        key = os.environ.get('STACKOVERFLOW_KEY') or os.environ.get('SO_API_KEY')

        return key

    def prompt_for_github_token(self, optional: bool = True) -> Optional[str]:
        """Prompt user for GitHub token.

        Args:
            optional: Whether the token is optional

        Returns:
            GitHub token or None if skipped
        """
        print("\n" + "="*60)
        print("GitHub Token Configuration")
        print("="*60)
        print("\nA GitHub personal access token is used for:")
        print("  - Accessing documentation from GitHub repositories")
        print("  - Higher rate limits for GitHub API")
        print("  - Private repository access (if needed)")
        print("\nTo create a token:")
        print("  1. Go to https://github.com/settings/tokens")
        print("  2. Click 'Generate new token (classic)'")
        print("  3. Give it a name (e.g., 'RAG-CLI')")
        print("  4. Select scopes: 'repo' (for private repos) or 'public_repo' (for public only)")
        print("  5. Click 'Generate token' and copy it")

        if optional:
            print("\nThis is OPTIONAL. Press Enter to skip.")

        try:
            token = getpass.getpass("\nEnter GitHub token (or press Enter to skip): ").strip()

            if not token:
                if optional:
                    print("Skipping GitHub token configuration.")
                    return None
                else:
                    print("GitHub token is required.")
                    return self.prompt_for_github_token(optional=False)

            # Validate token format (basic check)
            if not token.startswith(('ghp_', 'github_pat_', 'gho_', 'ghu_', 'ghs_', 'ghr_')):
                print("\nWarning: Token doesn't match expected GitHub token format.")
                confirm = input("Use this token anyway? (y/N): ").strip().lower()
                if confirm != 'y':
                    return self.prompt_for_github_token(optional=optional)

            return token

        except (KeyboardInterrupt, EOFError):
            print("\n\nToken configuration cancelled.")
            return None

    def prompt_for_stackoverflow_key(self, optional: bool = True) -> Optional[str]:
        """Prompt user for Stack Overflow API key.

        Args:
            optional: Whether the key is optional

        Returns:
            Stack Overflow API key or None if skipped
        """
        print("\n" + "="*60)
        print("Stack Overflow API Key Configuration")
        print("="*60)
        print("\nA Stack Overflow API key provides:")
        print("  - Higher rate limits for Stack Overflow API")
        print("  - Better performance for error resolution")
        print("\nTo get an API key:")
        print("  1. Go to https://stackapps.com/apps/oauth/register")
        print("  2. Register your application")
        print("  3. Copy the API key")

        if optional:
            print("\nThis is OPTIONAL. Press Enter to skip.")

        try:
            key = getpass.getpass("\nEnter Stack Overflow API key (or press Enter to skip): ").strip()

            if not key:
                if optional:
                    print("Skipping Stack Overflow API key configuration.")
                    return None
                else:
                    print("Stack Overflow API key is required.")
                    return self.prompt_for_stackoverflow_key(optional=False)

            return key

        except (KeyboardInterrupt, EOFError):
            print("\n\nAPI key configuration cancelled.")
            return None

    def save_to_env_file(self, github_token: Optional[str] = None,
                         stackoverflow_key: Optional[str] = None,
                         overwrite: bool = False):
        """Save tokens to .env file.

        Args:
            github_token: GitHub token to save
            stackoverflow_key: Stack Overflow API key to save
            overwrite: Whether to overwrite existing .env file
        """
        # Read existing .env if not overwriting
        existing_vars = {}
        if not overwrite and self.env_file.exists():
            existing_vars = self.load_env_file()

        # Update with new tokens
        if github_token:
            existing_vars['GITHUB_TOKEN'] = github_token

        if stackoverflow_key:
            existing_vars['STACKOVERFLOW_KEY'] = stackoverflow_key

        # Write to .env file
        try:
            with open(self.env_file, 'w') as f:
                f.write("# RAG-CLI Environment Configuration\n")
                f.write("# DO NOT COMMIT THIS FILE TO VERSION CONTROL\n\n")

                for key, value in existing_vars.items():
                    f.write(f'{key}="{value}"\n')

            print(f"\n[OK] Tokens saved to {self.env_file}")

            # Set restrictive permissions (Unix-like systems)
            if sys.platform != 'win32':
                os.chmod(self.env_file, 0o600)

        except Exception as e:
            print(f"\nError: Failed to save .env file: {e}")

    def update_config_yaml(self, github_token: Optional[str] = None,
                          stackoverflow_key: Optional[str] = None):
        """Update config/default.yaml with tokens.

        Note: This is NOT recommended for production. Tokens should stay in .env.
        This is only used if user explicitly wants to use YAML config.

        Args:
            github_token: GitHub token to save
            stackoverflow_key: Stack Overflow API key to save
        """
        if not self.config_file.exists():
            print(f"Warning: Config file not found: {self.config_file}")
            return

        try:
            # Read existing config
            with open(self.config_file, 'r') as f:
                lines = f.readlines()

            # Update token lines
            updated_lines = []
            for line in lines:
                if github_token and 'github_token:' in line:
                    indent = len(line) - len(line.lstrip())
                    updated_lines.append(' ' * indent + f'github_token: "{github_token}"  # GitHub personal access token\n')
                elif stackoverflow_key and 'stackoverflow_key:' in line:
                    indent = len(line) - len(line.lstrip())
                    updated_lines.append(' ' * indent + f'stackoverflow_key: "{stackoverflow_key}"  # Stack Overflow API key\n')
                else:
                    updated_lines.append(line)

            # Write updated config
            with open(self.config_file, 'w') as f:
                f.writelines(updated_lines)

            print(f"[OK] Config updated: {self.config_file}")

        except Exception as e:
            print(f"Warning: Failed to update config file: {e}")

    def configure_tokens(self, interactive: bool = True) -> Dict[str, Optional[str]]:
        """Configure tokens interactively or from .env.

        Args:
            interactive: Whether to prompt user for missing tokens

        Returns:
            Dictionary with configured tokens
        """
        tokens = {
            'github_token': None,
            'stackoverflow_key': None
        }

        # Try to load from .env first
        github_token = self.get_github_token()
        stackoverflow_key = self.get_stackoverflow_key()

        if github_token:
            print(f"[OK] GitHub token found in .env file")
            tokens['github_token'] = github_token
        elif interactive:
            print("\n[WARNING] GitHub token not found in .env file")
            github_token = self.prompt_for_github_token(optional=True)
            if github_token:
                tokens['github_token'] = github_token

        if stackoverflow_key:
            print(f"[OK] Stack Overflow API key found in .env file")
            tokens['stackoverflow_key'] = stackoverflow_key
        elif interactive:
            print("\n[WARNING] Stack Overflow API key not found in .env file")
            print("Note: This is completely optional and rarely needed.")
            stackoverflow_key = self.prompt_for_stackoverflow_key(optional=True)
            if stackoverflow_key:
                tokens['stackoverflow_key'] = stackoverflow_key

        # Save to .env if any tokens were provided
        if interactive and (tokens['github_token'] or tokens['stackoverflow_key']):
            save_choice = input("\nSave tokens to .env file? (Y/n): ").strip().lower()
            if save_choice != 'n':
                self.save_to_env_file(
                    github_token=tokens['github_token'],
                    stackoverflow_key=tokens['stackoverflow_key']
                )

        return tokens


def get_token_manager() -> TokenManager:
    """Get the global token manager instance.

    Returns:
        TokenManager instance
    """
    return TokenManager()


if __name__ == '__main__':
    # Test token manager
    manager = get_token_manager()
    tokens = manager.configure_tokens(interactive=True)

    print("\n" + "="*60)
    print("Token Configuration Complete")
    print("="*60)
    print(f"GitHub token configured: {'Yes' if tokens['github_token'] else 'No'}")
    print(f"Stack Overflow key configured: {'Yes' if tokens['stackoverflow_key'] else 'No'}")
