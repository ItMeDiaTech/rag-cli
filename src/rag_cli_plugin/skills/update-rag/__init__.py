"""Update RAG-CLI Plugin Skill

Provides Claude Code skill for synchronizing RAG-CLI plugin files.
"""

from .update import UpdateRagSkill, SyncOptions, update_rag, main

__all__ = ['UpdateRagSkill', 'SyncOptions', 'update_rag', 'main']
