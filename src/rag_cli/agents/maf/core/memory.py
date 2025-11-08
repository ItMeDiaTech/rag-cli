"""
Memory Management System for Multi-Agent Framework
"""

import asyncio
import hashlib
import json
import logging
# import pickle  # SECURITY: Removed - using safe numpy serialization instead
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
# For embeddings - will use sentence-transformers if available
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.warning("sentence-transformers not available - using mock embeddings")


@dataclass
class Memory:
    """Individual memory entry"""
    id: str
    content: str
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]
    timestamp: str
    importance: float
    access_count: int = 0
    last_accessed: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'content': self.content,
            'metadata': self.metadata,
            'timestamp': self.timestamp,
            'importance': self.importance,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed
        }


class EmbeddingManager:
    """Manages text embeddings for semantic search"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embedding_dim = 384  # Default dimension
        self.logger = logging.getLogger('EmbeddingManager')

        if EMBEDDINGS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                self.logger.info(
                    "Loaded embedding model: %s (dim=%s)",
                    model_name,
                    self.embedding_dim
                )
            except (OSError, RuntimeError, ValueError) as e:
                self.logger.error("Failed to load embedding model: %s", e)
                self.model = None

        if not self.model:
            self.embedding_dim = 384  # Mock dimension
            self.logger.warning("Using mock embeddings")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""

        if self.model:
            return self.model.encode(texts)
        else:
            # Mock embeddings
            embeddings = []
            for text in texts:
                # Generate deterministic mock embedding from text hash
                hash_obj = hashlib.blake2b(text.encode(), digest_size=16)
                hash_bytes = hash_obj.digest()

                # Convert to floats
                values = np.frombuffer(hash_bytes, dtype=np.uint8)
                # Repeat to match dimension
                repeat_count = (self.embedding_dim + len(values) - 1) // len(values)
                values = np.tile(values, repeat_count)[:self.embedding_dim]

                # Normalize
                embedding = values.astype(np.float32) / 255.0
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)

            return np.array(embeddings)

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""

        # Ensure proper shape
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)

        # Cosine similarity
        dot_product = np.dot(embedding1, embedding2.T)
        norm1 = np.linalg.norm(embedding1, axis=1)
        norm2 = np.linalg.norm(embedding2, axis=1)

        similarity = dot_product / (norm1[:, np.newaxis] * norm2[np.newaxis, :])
        return float(similarity[0, 0])


class MemoryManager:
    """Manages all memory operations for the framework"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config.get('database_path', 'data/memory.db')
        self.cache_dir = Path(config.get('cache_dir', 'data/cache'))
        self.cache_size = config.get('cache_size', 1000)
        self.max_memories_per_query = config.get('max_memories_per_query', 10)

        # Ensure directories exist
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.embedding_manager = EmbeddingManager(config.get('embedding_model', 'all-MiniLM-L6-v2'))
        self.logger = logging.getLogger('MemoryManager')

        # Memory cache
        self.memory_cache = {}
        self.cache_order = []

        # Statistics
        self.total_memories = 0
        self.total_searches = 0
        self.cache_hits = 0
        self.cache_misses = 0

        # Initialize database
        self._init_database()

        # Start consolidation task
        self.consolidation_interval = config.get('consolidation_interval', 86400)  # 24 hours
        self.consolidation_task = None

        self.logger.info("MemoryManager initialized")

    def _init_database(self):
        """Initialize SQLite database"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create memories table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding BLOB,
                metadata TEXT,
                timestamp TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_access_count ON memories(access_count)')

        # Create consolidation table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS consolidations (
                id TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                memory_ids TEXT,
                timestamp TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

        self.logger.debug("Database initialized")

    async def store(self, memory_data: Dict[str, Any]) -> str:
        """Store a memory"""

        # Generate ID
        memory_id = hashlib.blake2b(
            f"{memory_data.get('content', '')}{time.time()}".encode(),
            digest_size=16
        ).hexdigest()

        # Extract content
        content = str(memory_data.get('content', memory_data))

        # Generate embedding
        embedding = self.embedding_manager.encode([content])[0]

        # Create memory object
        memory = Memory(
            id=memory_id,
            content=content,
            embedding=embedding.tolist(),
            metadata=memory_data.get('metadata', {}),
            timestamp=memory_data.get('timestamp', datetime.now(timezone.utc).isoformat()),
            importance=memory_data.get('importance', 0.5)
        )

        # Store in database
        await self._store_to_database(memory)

        # Add to cache
        self._add_to_cache(memory)

        self.total_memories += 1
        self.logger.debug("Stored memory %s", memory_id)

        return memory_id

    async def _store_to_database(self, memory: Memory):
        """Store memory in database"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Serialize embedding using safe numpy format
            if memory.embedding is not None:
                embedding_blob = np.array(memory.embedding, dtype=np.float32).tobytes()
            else:
                embedding_blob = None

            cursor.execute('''
                INSERT OR REPLACE INTO memories
                (id, content, embedding, metadata, timestamp, importance,
                 access_count, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                memory.id,
                memory.content,
                embedding_blob,
                json.dumps(memory.metadata),
                memory.timestamp,
                memory.importance,
                memory.access_count,
                memory.last_accessed
            ))

            conn.commit()

        finally:
            conn.close()

    async def search(self, query: str, limit: Optional[int] = None) -> List[Memory]:
        """Search memories using semantic similarity"""

        self.total_searches += 1
        limit = limit or self.max_memories_per_query

        self.logger.debug("Searching for: %s...", query[:100])

        # Check cache first
        cache_key = hashlib.blake2b(f"{query}{limit}".encode(), digest_size=16).hexdigest()
        if cache_key in self.memory_cache:
            self.cache_hits += 1
            self.logger.debug("Cache hit for query")
            return self.memory_cache[cache_key]

        self.cache_misses += 1

        # Generate query embedding
        query_embedding = self.embedding_manager.encode([query])[0]

        # Search database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Get all memories (in production, you'd want to limit this)
            cursor.execute('''
                SELECT id, content, embedding, metadata, timestamp, importance,
                       access_count, last_accessed
                FROM memories
                ORDER BY importance DESC, access_count DESC
                LIMIT 1000
            ''')

            results = cursor.fetchall()

            # Calculate similarities
            memories_with_scores = []
            for row in results:
                (memory_id, content, embedding_blob, metadata_str, timestamp,
                 importance, access_count, last_accessed) = row

                if embedding_blob:
                    # Deserialize from safe numpy format
                    embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    similarity = self.embedding_manager.similarity(
                        query_embedding, embedding
                    )

                    # Combined score (similarity + importance)
                    score = similarity * 0.7 + importance * 0.3

                    memory = Memory(
                        id=memory_id,
                        content=content,
                        embedding=embedding,
                        metadata=json.loads(metadata_str) if metadata_str else {},
                        timestamp=timestamp,
                        importance=importance,
                        access_count=access_count,
                        last_accessed=last_accessed
                    )

                    memories_with_scores.append((memory, score))

            # Sort by score and get top results
            memories_with_scores.sort(key=lambda x: x[1], reverse=True)
            top_memories = [m for m, _ in memories_with_scores[:limit]]

            # Update access counts
            for memory in top_memories:
                await self._update_access(memory.id)

            # Cache results
            self._add_to_cache_with_key(cache_key, top_memories)

            self.logger.debug("Found %s relevant memories", len(top_memories))
            return top_memories

        finally:
            conn.close()

    async def get(self, memory_id: str) -> Optional[Memory]:
        """Get a specific memory"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                SELECT id, content, embedding, metadata, timestamp, importance,
                       access_count, last_accessed
                FROM memories
                WHERE id = ?
            ''', (memory_id,))

            row = cursor.fetchone()
            if row:
                (memory_id, content, embedding_blob, metadata_str, timestamp,
                 importance, access_count, last_accessed) = row

                memory = Memory(
                    id=memory_id,
                    content=content,
                    # SECURITY FIX: Use safe numpy deserialization instead of pickle
                    embedding=np.frombuffer(embedding_blob, dtype=np.float32) if embedding_blob else None,
                    metadata=json.loads(metadata_str) if metadata_str else {},
                    timestamp=timestamp,
                    importance=importance,
                    access_count=access_count,
                    last_accessed=last_accessed
                )

                await self._update_access(memory_id)
                return memory

            return None

        finally:
            conn.close()

    async def _update_access(self, memory_id: str):
        """Update memory access statistics"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                UPDATE memories
                SET access_count = access_count + 1,
                    last_accessed = ?
                WHERE id = ?
            ''', (datetime.now(timezone.utc).isoformat(), memory_id))

            conn.commit()

        finally:
            conn.close()

    async def consolidate(self):
        """Consolidate old memories to save space"""

        self.logger.info("Starting memory consolidation")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Find old, rarely accessed memories
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

            cursor.execute('''
                SELECT id, content, metadata, importance
                FROM memories
                WHERE timestamp < ? AND access_count < 5
                ORDER BY importance ASC
                LIMIT 100
            ''', (cutoff_date,))

            old_memories = cursor.fetchall()

            if len(old_memories) > 10:
                # Consolidate memories
                memory_ids = [m[0] for m in old_memories]
                [m[1] for m in old_memories]

                # Create summary (in production, use LLM to summarize)
                summary = f"Consolidated {len(old_memories)} memories from before {cutoff_date}"

                # Store consolidation
                consolidation_id = hashlib.blake2b(f"{summary}{time.time()}".encode(), digest_size=16).hexdigest()

                cursor.execute('''
                    INSERT INTO consolidations (id, summary, memory_ids, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (
                    consolidation_id,
                    summary,
                    json.dumps(memory_ids),
                    datetime.now(timezone.utc).isoformat()
                ))

                # Delete old memories
                placeholders = ','.join('?' * len(memory_ids))
                cursor.execute(f'''
                    DELETE FROM memories
                    WHERE id IN ({placeholders})
                ''', memory_ids)

                conn.commit()

                self.logger.info("Consolidated %s memories", len(old_memories))

        finally:
            conn.close()

    def _add_to_cache(self, memory: Memory):
        """Add memory to cache"""

        # Use LRU cache strategy
        if len(self.cache_order) >= self.cache_size:
            # Remove oldest
            oldest_key = self.cache_order.pop(0)
            if oldest_key in self.memory_cache:
                del self.memory_cache[oldest_key]

        # Add new
        cache_key = memory.id
        self.memory_cache[cache_key] = memory
        self.cache_order.append(cache_key)

    def _add_to_cache_with_key(self, key: str, value: Any):
        """Add to cache with specific key"""

        if len(self.cache_order) >= self.cache_size:
            oldest_key = self.cache_order.pop(0)
            if oldest_key in self.memory_cache:
                del self.memory_cache[oldest_key]

        self.memory_cache[key] = value
        self.cache_order.append(key)

    async def start_consolidation_task(self):
        """Start periodic consolidation task"""

        async def consolidation_loop():
            while True:
                await asyncio.sleep(self.consolidation_interval)
                try:
                    await self.consolidate()
                except Exception as e:
                    self.logger.error("Consolidation failed: %s", e)

        self.consolidation_task = asyncio.create_task(consolidation_loop())
        self.logger.info("Consolidation task started")

    async def stop_consolidation_task(self):
        """Stop consolidation task"""

        if self.consolidation_task:
            self.consolidation_task.cancel()
            try:
                await self.consolidation_task
            except asyncio.CancelledError:
                pass
            self.logger.info("Consolidation task stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('SELECT COUNT(*) FROM memories')
            memory_count = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM consolidations')
            consolidation_count = cursor.fetchone()[0]

            cache_hit_rate = (
                (self.cache_hits / self.total_searches * 100)
                if self.total_searches > 0 else 0
            )

            return {
                'total_memories': memory_count,
                'total_consolidations': consolidation_count,
                'cache_size': len(self.memory_cache),
                'total_searches': self.total_searches,
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cache_hit_rate': f"{cache_hit_rate:.2f}%"
            }

        finally:
            conn.close()

    def export_memories(self, filepath: str):
        """Export memories to file"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                SELECT id, content, metadata, timestamp, importance
                FROM memories
            ''')

            memories = []
            for row in cursor.fetchall():
                memories.append({
                    'id': row[0],
                    'content': row[1],
                    'metadata': json.loads(row[2]) if row[2] else {},
                    'timestamp': row[3],
                    'importance': row[4]
                })

            with open(filepath, 'w') as f:
                json.dump(memories, f, indent=2)

            self.logger.info("Exported %s memories to %s", len(memories), filepath)

        finally:
            conn.close()
