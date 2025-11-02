"""Error tracking system for recurring error detection.

Tracks errors across sessions to identify patterns and trigger online documentation searches.
Platform-agnostic version for use in the core rag_cli library.
"""

import json
import hashlib
import logging
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ErrorOccurrence:
    """Represents a single error occurrence."""
    error_signature: str
    error_type: str
    error_message: str
    timestamp: datetime
    context: str
    resolved: bool = False
    solution: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'ErrorOccurrence':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class ErrorTracker:
    """Tracks recurring errors and manages error history."""

    def __init__(self, log_file: Optional[str] = None, repeated_threshold: int = 3):
        """Initialize error tracker.

        Args:
            log_file: Path to persistent error log file
            repeated_threshold: Number of occurrences to trigger online search
        """
        self.log_file = log_file or './config/error_history.json'
        self.repeated_threshold = repeated_threshold
        self.errors: Dict[str, List[ErrorOccurrence]] = defaultdict(list)
        self.load()

    def compute_error_signature(self, error_text: str) -> str:
        """Compute unique signature for an error.

        Args:
            error_text: Error message or traceback

        Returns:
            Error signature hash
        """
        # Extract key parts of error
        normalized = self._normalize_error(error_text)
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:16]

    def track_error(self, error_text: str, context: str = "") -> ErrorOccurrence:
        """Track an error occurrence.

        Args:
            error_text: Full error message
            context: Optional context about where error occurred

        Returns:
            ErrorOccurrence object
        """
        signature = self.compute_error_signature(error_text)
        error_info = self._extract_error_info(error_text)

        occurrence = ErrorOccurrence(
            error_signature=signature,
            error_type=error_info['type'],
            error_message=error_info['message'],
            timestamp=datetime.now(),
            context=context
        )

        self.errors[signature].append(occurrence)

        # Log the error
        count = len(self.errors[signature])
        logger.info(f"Tracked error {signature}: {error_info['type']} (occurrence #{count})")

        # Save after each addition
        self.save()

        return occurrence

    def is_repeated_error(self, error_text: str) -> bool:
        """Check if error has occurred multiple times.

        Args:
            error_text: Error message to check

        Returns:
            True if error has occurred >= threshold times
        """
        signature = self.compute_error_signature(error_text)
        occurrences = self.errors.get(signature, [])

        # Count unresolved occurrences
        unresolved = [o for o in occurrences if not o.resolved]

        return len(unresolved) >= self.repeated_threshold

    def should_search_online(self, error_text: str) -> bool:
        """Determine if online search should be triggered for this error.

        Args:
            error_text: Error message

        Returns:
            True if should search online
        """
        signature = self.compute_error_signature(error_text)
        occurrences = self.errors.get(signature, [])

        if not occurrences:
            return False

        # Check if this is a repeated error without solution
        unresolved = [o for o in occurrences if not o.resolved]

        if len(unresolved) >= self.repeated_threshold:
            # Check if we haven't searched online recently
            recent_occurrences = unresolved[-self.repeated_threshold:]
            time_span = (recent_occurrences[-1].timestamp - recent_occurrences[0].timestamp).total_seconds()

            # If errors happened within short time span (e.g., 1 hour), trigger search
            if time_span < 3600:
                logger.info(f"Repeated error {signature} detected {len(unresolved)} times, triggering online search")
                return True

        return False

    def mark_resolved(self, error_text: str, solution: str):
        """Mark an error as resolved with a solution.

        Args:
            error_text: Error message
            solution: Solution description or URL
        """
        signature = self.compute_error_signature(error_text)
        occurrences = self.errors.get(signature, [])

        # Mark all unresolved occurrences as resolved
        for occurrence in occurrences:
            if not occurrence.resolved:
                occurrence.resolved = True
                occurrence.solution = solution

        logger.info(f"Marked error {signature} as resolved")
        self.save()

    def get_error_history(self, error_text: Optional[str] = None, limit: int = 10) -> List[ErrorOccurrence]:
        """Get error history.

        Args:
            error_text: Optional specific error to get history for
            limit: Maximum number of occurrences to return

        Returns:
            List of ErrorOccurrence objects
        """
        if error_text:
            signature = self.compute_error_signature(error_text)
            return self.errors.get(signature, [])[:limit]
        else:
            # Return most recent errors across all types
            all_errors = []
            for occurrences in self.errors.values():
                all_errors.extend(occurrences)

            all_errors.sort(key=lambda e: e.timestamp, reverse=True)
            return all_errors[:limit]

    def get_frequent_errors(self, min_count: int = 2) -> Dict[str, List[ErrorOccurrence]]:
        """Get errors that have occurred multiple times.

        Args:
            min_count: Minimum number of occurrences

        Returns:
            Dictionary of error signature to occurrences
        """
        frequent = {}

        for signature, occurrences in self.errors.items():
            unresolved = [o for o in occurrences if not o.resolved]
            if len(unresolved) >= min_count:
                frequent[signature] = unresolved

        return frequent

    def cleanup_old_errors(self, days: int = 30) -> int:
        """Remove old resolved errors from history.

        Args:
            days: Age threshold in days

        Returns:
            Number of error occurrences removed
        """
        cutoff = datetime.now().timestamp() - (days * 24 * 3600)
        removed_count = 0

        for signature in list(self.errors.keys()):
            # Keep unresolved errors, remove old resolved ones
            filtered = [
                e for e in self.errors[signature]
                if not e.resolved or e.timestamp.timestamp() > cutoff
            ]

            removed = len(self.errors[signature]) - len(filtered)
            removed_count += removed

            if filtered:
                self.errors[signature] = filtered
            else:
                del self.errors[signature]

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old error occurrences")
            self.save()

        return removed_count

    def _normalize_error(self, error_text: str) -> str:
        """Normalize error text for signature computation.

        Args:
            error_text: Raw error text

        Returns:
            Normalized error text
        """
        import re

        # Remove file paths
        normalized = re.sub(r'[A-Za-z]:\\[^\s]+', '[PATH]', error_text)
        normalized = re.sub(r'/[^\s]+\.py', '[PATH]', normalized)

        # Remove line numbers
        normalized = re.sub(r'line \d+', 'line [N]', normalized)

        # Remove memory addresses
        normalized = re.sub(r'0x[0-9a-fA-F]+', '0x[ADDR]', normalized)

        # Remove specific values but keep structure
        normalized = re.sub(r'\d+', '[NUM]', normalized)

        # Normalize whitespace
        normalized = ' '.join(normalized.split())

        return normalized.lower()

    def _extract_error_info(self, error_text: str) -> Dict[str, str]:
        """Extract error type and message.

        Args:
            error_text: Error text

        Returns:
            Dictionary with 'type' and 'message' keys
        """
        import re

        lines = error_text.strip().split('\n')

        # Find error type and message (usually last line)
        for line in reversed(lines):
            if ':' in line:
                match = re.match(r'([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)):\s*(.+)', line.strip())
                if match:
                    return {
                        'type': match.group(1),
                        'message': match.group(2)
                    }

        # Fallback: use first line
        if lines:
            parts = lines[0].split(':', 1)
            return {
                'type': parts[0].strip() if len(parts) > 0 else 'Unknown',
                'message': parts[1].strip() if len(parts) > 1 else error_text[:100]
            }

        return {'type': 'Unknown', 'message': error_text[:100]}

    def save(self):
        """Save error history to file."""
        try:
            # Ensure directory exists
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)

            # Convert to serializable format
            data = {
                'errors': {
                    signature: [e.to_dict() for e in occurrences]
                    for signature, occurrences in self.errors.items()
                },
                'saved_at': datetime.now().isoformat(),
                'threshold': self.repeated_threshold
            }

            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved error history to {self.log_file}")

        except Exception as e:
            logger.error(f"Error saving error history: {e}")

    def load(self):
        """Load error history from file."""
        try:
            if not Path(self.log_file).exists():
                logger.debug("No existing error history found")
                return

            with open(self.log_file, 'r') as f:
                data = json.load(f)

            # Load errors
            error_data = data.get('errors', {})
            self.errors = defaultdict(list)

            for signature, occurrences in error_data.items():
                self.errors[signature] = [ErrorOccurrence.from_dict(e) for e in occurrences]

            total_errors = sum(len(occurrences) for occurrences in self.errors.values())
            logger.info(f"Loaded error history: {len(self.errors)} unique errors, {total_errors} total occurrences")

        except Exception as e:
            logger.error(f"Error loading error history: {e}")
            self.errors = defaultdict(list)

    def get_statistics(self) -> Dict[str, any]:
        """Get error tracking statistics.

        Returns:
            Dictionary of statistics
        """
        total_occurrences = sum(len(occurrences) for occurrences in self.errors.values())
        resolved_count = sum(
            1 for occurrences in self.errors.values()
            for e in occurrences if e.resolved
        )
        unresolved_count = total_occurrences - resolved_count

        # Count by error type
        type_counts = defaultdict(int)
        for occurrences in self.errors.values():
            for error in occurrences:
                if not error.resolved:
                    type_counts[error.error_type] += 1

        stats = {
            'unique_errors': len(self.errors),
            'total_occurrences': total_occurrences,
            'resolved': resolved_count,
            'unresolved': unresolved_count,
            'by_type': dict(type_counts),
            'threshold': self.repeated_threshold
        }

        # Find most frequent unresolved error
        frequent = self.get_frequent_errors(min_count=self.repeated_threshold)
        if frequent:
            max_signature = max(frequent.keys(), key=lambda s: len(frequent[s]))
            max_error = frequent[max_signature][0]
            stats['most_frequent'] = {
                'type': max_error.error_type,
                'message': max_error.error_message,
                'count': len(frequent[max_signature])
            }

        return stats


# Singleton instance
_tracker: Optional[ErrorTracker] = None


def get_error_tracker() -> ErrorTracker:
    """Get global error tracker instance.

    Returns:
        ErrorTracker instance
    """
    global _tracker
    if _tracker is None:
        _tracker = ErrorTracker()
    return _tracker
