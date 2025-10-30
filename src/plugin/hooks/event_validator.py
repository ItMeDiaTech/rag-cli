#!/usr/bin/env python3
"""Event validation utilities for RAG-CLI hooks.

This module provides JSON schema validation for hook events to prevent
crashes on malformed input.
"""

from typing import Dict, Any, Optional
import logging


class EventValidator:
    """Validates hook event structures."""

    # Event type schemas
    SCHEMAS = {
        'UserPromptSubmit': {
            'required': ['type', 'metadata'],
            'metadata_required': ['rag_enabled'],
        },
        'ResponsePost': {
            'required': ['type', 'metadata'],
            'metadata_required': [],
        },
        'PluginStateChange': {
            'required': ['type', 'metadata'],
            'metadata_required': ['plugin_name', 'state_change'],
        },
        'DocumentIndexing': {
            'required': ['type', 'metadata'],
            'metadata_required': ['file_path'],
        },
        'ErrorOccurred': {
            'required': ['type', 'metadata'],
            'metadata_required': ['error_message'],
        },
    }

    @staticmethod
    def validate_event(event: Dict[str, Any], event_type: Optional[str] = None) -> tuple[bool, str]:
        """Validate an event against its schema.

        Args:
            event: Event dictionary to validate
            event_type: Expected event type. If None, extracted from event['type']

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if event is valid, False otherwise
            - error_message: Detailed error message if invalid, empty string if valid
        """
        if not isinstance(event, dict):
            return False, f"Event must be a dictionary, got {type(event).__name__}"

        # Extract event type
        if event_type is None:
            event_type = event.get('type')

        if not event_type:
            return False, "Event missing 'type' field"

        # Get schema for this event type
        schema = EventValidator.SCHEMAS.get(event_type, {})

        # Check required top-level fields
        required_fields = schema.get('required', [])
        for field in required_fields:
            if field not in event:
                return False, f"Event missing required field: {field}"

        # Check metadata
        metadata = event.get('metadata')
        if metadata is None:
            return False, "Event metadata is None"

        if not isinstance(metadata, dict):
            return False, f"Event metadata must be a dictionary, got {type(metadata).__name__}"

        # Check required metadata fields
        required_metadata = schema.get('metadata_required', [])
        for field in required_metadata:
            if field not in metadata:
                return False, f"Event metadata missing required field: {field}"

        return True, ""

    @staticmethod
    def validate_or_log(event: Dict[str, Any], logger: logging.Logger, event_type: Optional[str] = None) -> bool:
        """Validate an event and log any errors.

        Args:
            event: Event dictionary to validate
            logger: Logger instance to use for error logging
            event_type: Expected event type

        Returns:
            True if valid, False if invalid (error already logged)
        """
        is_valid, error_msg = EventValidator.validate_event(event, event_type)

        if not is_valid:
            logger.error(f"Invalid event structure: {error_msg}. Event: {event}")
            return False

        return True

    @staticmethod
    def safe_get(event: Dict[str, Any], path: str, default: Any = None) -> Any:
        """Safely get nested dictionary values using dot notation.

        Args:
            event: Dictionary to query
            path: Dot-separated path (e.g., 'metadata.rag_enabled')
            default: Default value if path not found

        Returns:
            Value at path, or default if not found
        """
        keys = path.split('.')
        current = event

        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
                if current is None:
                    return default
            else:
                return default

        return current if current is not None else default
