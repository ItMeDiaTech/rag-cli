"""Duplicate detection for documentation content.

Uses content hashing to identify and filter duplicate documents before indexing.
"""

import hashlib
import logging
import json
import threading
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ContentHash:
    """Represents a content hash with metadata."""
    content_hash: str
    title: str
    source: str
    url: Optional[str]
    indexed_date: datetime
    doc_type: str  # 'local', 'github', 'stackoverflow', 'readthedocs', 'official'

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['indexed_date'] = self.indexed_date.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'ContentHash':
        """Create from dictionary."""
        data['indexed_date'] = datetime.fromisoformat(data['indexed_date'])
        return cls(**data)


class DuplicateDetector:
    """Detects duplicate content using SHA-256 hashing."""

    def __init__(self, hash_file: Optional[str] = None):
        """Initialize duplicate detector.

        Args:
            hash_file: Path to file storing content hashes
        """
        self.hash_file = hash_file or './data/vectors/content_hashes.json'
        self.hashes: Dict[str, ContentHash] = {}
        self._file_lock = threading.Lock()
        self.load()

    def compute_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content.

        Args:
            content: Content to hash

        Returns:
            Hexadecimal hash string
        """
        # Normalize content before hashing
        normalized = self._normalize_content(content)
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    def is_duplicate(self, content: str, fuzzy: bool = False) -> Tuple[bool, Optional[ContentHash]]:
        """Check if content is a duplicate.

        Args:
            content: Content to check
            fuzzy: If True, also check for near-duplicates

        Returns:
            Tuple of (is_duplicate, existing_hash_info)
        """
        content_hash = self.compute_hash(content)

        if content_hash in self.hashes:
            return True, self.hashes[content_hash]

        if fuzzy:
            # Check for fuzzy duplicates
            is_fuzzy_dup, hash_info = self._check_fuzzy_duplicate(content)
            if is_fuzzy_dup:
                return True, hash_info

        return False, None

    def add_hash(self, content: str, title: str, source: str,
                 url: Optional[str] = None, doc_type: str = 'local') -> str:
        """Add content hash to registry.

        Args:
            content: Content to hash
            title: Document title
            source: Source path or identifier
            url: Optional URL
            doc_type: Type of document

        Returns:
            Content hash
        """
        content_hash = self.compute_hash(content)

        if content_hash not in self.hashes:
            self.hashes[content_hash] = ContentHash(
                content_hash=content_hash,
                title=title,
                source=source,
                url=url,
                indexed_date=datetime.now(),
                doc_type=doc_type
            )
            logger.debug(f"Added hash for: {title}")
        else:
            logger.debug(f"Hash already exists for: {title}")

        return content_hash

    def filter_duplicates(self, documents: List[Dict]) -> List[Dict]:
        """Filter out duplicate documents from a list.

        Args:
            documents: List of document dictionaries with 'content', 'title', 'source' keys

        Returns:
            List of non-duplicate documents
        """
        unique_docs = []
        duplicates_found = 0

        for doc in documents:
            content = doc.get('content', '')
            is_dup, _ = self.is_duplicate(content)

            if not is_dup:
                unique_docs.append(doc)

                # Add to hash registry
                self.add_hash(
                    content=content,
                    title=doc.get('title', ''),
                    source=doc.get('source', ''),
                    url=doc.get('url'),
                    doc_type=doc.get('doc_type', 'local')
                )
            else:
                duplicates_found += 1
                logger.debug(f"Filtered duplicate: {doc.get('title', 'Unknown')}")

        if duplicates_found > 0:
            logger.info(f"Filtered {duplicates_found} duplicate documents, kept {len(unique_docs)} unique")

        return unique_docs

    def remove_hash(self, content_hash: str) -> bool:
        """Remove a hash from registry.

        Args:
            content_hash: Hash to remove

        Returns:
            True if removed, False if not found
        """
        if content_hash in self.hashes:
            del self.hashes[content_hash]
            logger.debug(f"Removed hash: {content_hash}")
            return True
        return False

    def get_hash_info(self, content_hash: str) -> Optional[ContentHash]:
        """Get information about a hash.

        Args:
            content_hash: Hash to lookup

        Returns:
            ContentHash object or None if not found
        """
        return self.hashes.get(content_hash)

    def find_by_source(self, source: str) -> List[ContentHash]:
        """Find all hashes from a specific source.

        Args:
            source: Source to search for

        Returns:
            List of ContentHash objects
        """
        return [h for h in self.hashes.values() if h.source == source]

    def find_by_type(self, doc_type: str) -> List[ContentHash]:
        """Find all hashes of a specific type.

        Args:
            doc_type: Document type to search for

        Returns:
            List of ContentHash objects
        """
        return [h for h in self.hashes.values() if h.doc_type == doc_type]

    def cleanup_old_hashes(self, days: int = 90) -> int:
        """Remove hashes older than specified days.

        Args:
            days: Age threshold in days

        Returns:
            Number of hashes removed
        """
        cutoff = datetime.now().timestamp() - (days * 24 * 3600)
        old_hashes = [
            h.content_hash for h in self.hashes.values()
            if h.indexed_date.timestamp() < cutoff
        ]

        for hash_val in old_hashes:
            del self.hashes[hash_val]

        if old_hashes:
            logger.info(f"Cleaned up {len(old_hashes)} old hashes")

        return len(old_hashes)

    def _normalize_content(self, content: str) -> str:
        """Normalize content before hashing.

        Args:
            content: Raw content

        Returns:
            Normalized content
        """
        # Remove extra whitespace
        normalized = ' '.join(content.split())

        # Convert to lowercase for case-insensitive matching
        normalized = normalized.lower()

        # Remove common variations that don't change content meaning
        normalized = normalized.replace('\r\n', '\n')
        normalized = normalized.replace('\t', ' ')

        return normalized

    def _check_fuzzy_duplicate(self, content: str) -> Tuple[bool, Optional[ContentHash]]:
        """Check for fuzzy duplicates using similarity.

        Args:
            content: Content to check

        Returns:
            Tuple of (is_duplicate, hash_info)
        """
        # Simple fuzzy matching: check if content is very similar
        # For production, could use MinHash or SimHash
        normalized = self._normalize_content(content)
        len(normalized)

        for hash_info in self.hashes.values():
            # This is a placeholder - in production would need to store
            # fuzzy hashes or use approximate matching
            # For now, just return False
            pass

        return False, None

    def save(self):
        """Save hashes to file with thread-safe file locking."""
        with self._file_lock:
            try:
                # Ensure directory exists
                Path(self.hash_file).parent.mkdir(parents=True, exist_ok=True)

                # Convert to serializable format
                data = {
                    'hashes': {k: v.to_dict() for k, v in self.hashes.items()},
                    'saved_at': datetime.now().isoformat()
                }

                with open(self.hash_file, 'w') as f:
                    json.dump(data, f, indent=2)

                logger.debug(f"Saved {len(self.hashes)} hashes to {self.hash_file}")

            except (IOError, OSError, PermissionError) as e:
                # Expected errors - file system issues
                logger.error(f"Error saving hashes to {self.hash_file}: {e}")
            except json.JSONEncodeError as e:
                # JSON encoding errors
                logger.error(f"Error encoding hash data: {e}")
            except Exception as e:
                # Unexpected errors - log with traceback
                logger.exception("Unexpected error saving hashes", exc_info=True)

    def load(self):
        """Load hashes from file with thread-safe file locking."""
        with self._file_lock:
            try:
                if not Path(self.hash_file).exists():
                    logger.debug("No existing hash file found, starting fresh")
                    return

                with open(self.hash_file, 'r') as f:
                    data = json.load(f)

                # Load hashes
                hash_data = data.get('hashes', {})
                self.hashes = {k: ContentHash.from_dict(v) for k, v in hash_data.items()}

                logger.info(f"Loaded {len(self.hashes)} hashes from {self.hash_file}")

            except (IOError, OSError, FileNotFoundError) as e:
                # Expected errors - file not found, permission issues
                logger.error(f"Error loading hashes from {self.hash_file}: {e}")
                self.hashes = {}
            except json.JSONDecodeError as e:
                # JSON parsing errors
                logger.error(f"Error parsing hash file {self.hash_file}: {e}")
                self.hashes = {}
            except (KeyError, ValueError, TypeError) as e:
                # Invalid hash data structure
                logger.error(f"Invalid hash data in {self.hash_file}: {e}")
                self.hashes = {}
            except Exception as e:
                # Unexpected errors - log with traceback
                logger.exception("Unexpected error loading hashes", exc_info=True)
                self.hashes = {}

    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about stored hashes.

        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_hashes': len(self.hashes),
            'by_type': {},
            'oldest': None,
            'newest': None
        }

        # Count by type
        for hash_info in self.hashes.values():
            doc_type = hash_info.doc_type
            stats['by_type'][doc_type] = stats['by_type'].get(doc_type, 0) + 1

        # Find oldest and newest
        if self.hashes:
            dates = [h.indexed_date for h in self.hashes.values()]
            stats['oldest'] = min(dates).isoformat()
            stats['newest'] = max(dates).isoformat()

        return stats


# Singleton instance
_detector: Optional[DuplicateDetector] = None


def get_duplicate_detector() -> DuplicateDetector:
    """Get global duplicate detector instance.

    Returns:
        DuplicateDetector instance
    """
    global _detector
    if _detector is None:
        _detector = DuplicateDetector()
    return _detector
