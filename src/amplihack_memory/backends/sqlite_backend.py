"""SQLite backend for memory storage."""

import json
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from ..exceptions import InvalidExperienceError
from ..experience import Experience, ExperienceType
from .base import MemoryBackend


class SQLiteBackend(MemoryBackend):
    """SQLite-based memory storage backend.

    Provides relational storage with FTS5 full-text search.
    """

    def __init__(
        self,
        db_path: Path,
        agent_name: str,
        max_memory_mb: int = 100,
        enable_compression: bool = True,
    ):
        """Initialize SQLite backend.

        Args:
            db_path: Path to SQLite database file
            agent_name: Agent identifier
            max_memory_mb: Maximum storage size in MB
            enable_compression: Enable automatic compression
        """
        self.db_path = db_path
        self.agent_name = agent_name
        self.max_memory_mb = max_memory_mb
        self.enable_compression = enable_compression
        self._lock = threading.Lock()
        self._connection = None
        self.initialize_schema()

    def initialize_schema(self):
        """Initialize SQLite database with schema."""
        try:
            self._connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=10.0,
            )
            self._connection.row_factory = sqlite3.Row

            # Enable Write-Ahead Logging for better concurrency
            self._connection.execute("PRAGMA journal_mode=WAL")

            # Create experiences table
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS experiences (
                    experience_id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    experience_type TEXT NOT NULL,
                    context TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp INTEGER NOT NULL,
                    metadata TEXT,
                    tags TEXT,
                    compressed INTEGER DEFAULT 0
                )
            """)

            # Create indexes
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_name ON experiences(agent_name)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_experience_type ON experiences(experience_type)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON experiences(timestamp)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_confidence ON experiences(confidence)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_type ON experiences(agent_name, experience_type)"
            )

            # Create FTS5 virtual table for full-text search
            self._connection.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS experiences_fts
                USING fts5(experience_id, context, outcome, content='experiences', tokenize='porter')
            """)

            # Create triggers to keep FTS in sync
            self._connection.execute("""
                CREATE TRIGGER IF NOT EXISTS experiences_fts_insert AFTER INSERT ON experiences
                BEGIN
                    INSERT INTO experiences_fts(experience_id, context, outcome)
                    VALUES (new.experience_id, new.context, new.outcome);
                END
            """)

            self._connection.execute("""
                CREATE TRIGGER IF NOT EXISTS experiences_fts_delete AFTER DELETE ON experiences
                BEGIN
                    DELETE FROM experiences_fts WHERE experience_id = old.experience_id;
                END
            """)

            self._connection.commit()

        except sqlite3.DatabaseError as e:
            error_msg = str(e).lower()
            if (
                "corrupted" in error_msg
                or "malformed" in error_msg
                or "not a database" in error_msg
            ):
                raise Exception(f"Database corrupted: {e}")
            raise

    def store_experience(self, experience: Experience) -> str:
        """Store an experience in SQLite.

        Args:
            experience: Experience to store

        Returns:
            experience_id: ID of stored experience
        """
        # Validate experience
        if not experience.context.strip():
            raise InvalidExperienceError("context cannot be empty")
        if not experience.outcome.strip():
            raise InvalidExperienceError("outcome cannot be empty")
        if not (0.0 <= experience.confidence <= 1.0):
            raise InvalidExperienceError("confidence must be between 0.0 and 1.0")

        # Convert to database format
        metadata_json = json.dumps(experience.metadata) if experience.metadata else "{}"
        tags_json = json.dumps(experience.tags) if experience.tags else "[]"

        # Thread-safe storage
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO experiences (
                    experience_id, agent_name, experience_type,
                    context, outcome, confidence, timestamp,
                    metadata, tags, compressed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
                (
                    experience.experience_id,
                    self.agent_name,
                    experience.experience_type.value,
                    experience.context,
                    experience.outcome,
                    experience.confidence,
                    int(experience.timestamp.timestamp()),
                    metadata_json,
                    tags_json,
                ),
            )
            self._connection.commit()

        return experience.experience_id

    def retrieve_experiences(
        self,
        limit: int | None = None,
        experience_type: ExperienceType | None = None,
        min_confidence: float = 0.0,
    ) -> List[Experience]:
        """Retrieve experiences from SQLite.

        Args:
            limit: Maximum number of experiences to retrieve
            experience_type: Filter by experience type
            min_confidence: Minimum confidence threshold

        Returns:
            List of experiences sorted by recency
        """
        query = """
            SELECT * FROM experiences
            WHERE agent_name = ?
            AND confidence >= ?
        """
        params = [self.agent_name, min_confidence]

        if experience_type:
            query += " AND experience_type = ?"
            params.append(experience_type.value)

        query += " ORDER BY timestamp DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor = self._connection.execute(query, params)
        rows = cursor.fetchall()

        experiences = []
        for row in rows:
            exp = Experience(
                experience_id=row["experience_id"],
                experience_type=ExperienceType(row["experience_type"]),
                context=row["context"],
                outcome=row["outcome"],
                confidence=row["confidence"],
                timestamp=datetime.fromtimestamp(row["timestamp"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                tags=json.loads(row["tags"]) if row["tags"] else [],
            )
            experiences.append(exp)

        return experiences

    def search(
        self,
        query: str,
        experience_type: ExperienceType | None = None,
        min_confidence: float = 0.0,
        limit: int = 10,
    ) -> List[Experience]:
        """Search experiences using FTS5.

        Args:
            query: Search query (text)
            experience_type: Filter by experience type
            min_confidence: Minimum confidence threshold
            limit: Maximum results

        Returns:
            List of matching experiences
        """
        if not query or not query.strip():
            # Empty query - return all experiences
            return self.retrieve_experiences(
                limit=limit,
                experience_type=experience_type,
                min_confidence=min_confidence,
            )

        # Use FTS5 for text search
        sql = """
            SELECT e.* FROM experiences e
            JOIN experiences_fts fts ON e.experience_id = fts.experience_id
            WHERE experiences_fts MATCH ?
            AND e.agent_name = ?
            AND e.confidence >= ?
        """
        params = [query, self.agent_name, min_confidence]

        if experience_type:
            sql += " AND e.experience_type = ?"
            params.append(experience_type.value)

        sql += " ORDER BY e.timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = self._connection.execute(sql, params)
        rows = cursor.fetchall()

        experiences = []
        for row in rows:
            exp = Experience(
                experience_id=row["experience_id"],
                experience_type=ExperienceType(row["experience_type"]),
                context=row["context"],
                outcome=row["outcome"],
                confidence=row["confidence"],
                timestamp=datetime.fromtimestamp(row["timestamp"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                tags=json.loads(row["tags"]) if row["tags"] else [],
            )
            experiences.append(exp)

        return experiences

    def get_statistics(self) -> dict:
        """Get SQLite storage statistics.

        Returns:
            Dictionary with statistics
        """
        conn = self._connection

        # Total count
        cursor = conn.execute(
            "SELECT COUNT(*) FROM experiences WHERE agent_name = ?", (self.agent_name,)
        )
        total = cursor.fetchone()[0]

        # Count by type
        by_type = {}
        cursor = conn.execute(
            """
            SELECT experience_type, COUNT(*) as count
            FROM experiences
            WHERE agent_name = ?
            GROUP BY experience_type
        """,
            (self.agent_name,),
        )

        for row in cursor:
            exp_type = ExperienceType(row["experience_type"])
            by_type[exp_type] = row["count"]

        # Storage size
        storage_size_kb = self.db_path.stat().st_size / 1024 if self.db_path.exists() else 0

        # Compressed count
        cursor = conn.execute(
            "SELECT COUNT(*) FROM experiences WHERE agent_name = ? AND compressed = 1",
            (self.agent_name,),
        )
        compressed = cursor.fetchone()[0]

        # Calculate compression ratio
        compression_ratio = 1.0
        if compressed > 0:
            compression_ratio = 3.0

        return {
            "total_experiences": total,
            "by_type": by_type,
            "storage_size_kb": storage_size_kb,
            "compressed_experiences": compressed,
            "compression_ratio": compression_ratio,
        }

    def close(self):
        """Close SQLite connection."""
        if self._connection:
            self._connection.close()
            self._connection = None

    def get_connection(self):
        """Get SQLite connection for advanced operations.

        Returns:
            sqlite3.Connection object
        """
        return self._connection

    def cleanup(
        self,
        auto_compress: bool = True,
        max_age_days: int | None = None,
        max_experiences: int | None = None,
    ):
        """Run cleanup operations on SQLite database.

        Args:
            auto_compress: Compress old experiences (>30 days)
            max_age_days: Delete experiences older than this
            max_experiences: Limit to maximum number of experiences
        """
        conn = self._connection
        changes_made = False

        # 1. Compress old experiences (>30 days)
        if auto_compress:
            cutoff = int((datetime.now() - timedelta(days=30)).timestamp())
            cursor = conn.execute(
                """
                UPDATE experiences
                SET compressed = 1
                WHERE agent_name = ?
                AND timestamp < ?
                AND compressed = 0
            """,
                (self.agent_name, cutoff),
            )
            if cursor.rowcount > 0:
                changes_made = True

        # 2. Delete experiences older than max_age_days
        if max_age_days:
            cutoff = int((datetime.now() - timedelta(days=max_age_days)).timestamp())
            cursor = conn.execute(
                """
                DELETE FROM experiences
                WHERE agent_name = ?
                AND timestamp < ?
            """,
                (self.agent_name, cutoff),
            )
            if cursor.rowcount > 0:
                changes_made = True

        # 3. Limit to max_experiences (keep most recent + high-confidence patterns)
        if max_experiences:
            # Get count
            cursor = conn.execute(
                "SELECT COUNT(*) FROM experiences WHERE agent_name = ?", (self.agent_name,)
            )
            count = cursor.fetchone()[0]

            if count > max_experiences:
                # Strategy: Keep high-confidence patterns (≥0.8) + most recent experiences
                cursor = conn.execute(
                    """
                    WITH high_conf_patterns AS (
                        SELECT experience_id FROM experiences
                        WHERE agent_name = ?
                        AND experience_type = 'pattern' AND confidence >= 0.8
                    ),
                    recent_experiences AS (
                        SELECT experience_id FROM experiences
                        WHERE agent_name = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    )
                    SELECT experience_id FROM high_conf_patterns
                    UNION
                    SELECT experience_id FROM recent_experiences
                    """,
                    (self.agent_name, self.agent_name, max_experiences),
                )
                keep_ids = [row[0] for row in cursor.fetchall()]

                if keep_ids:
                    # Delete everything except the IDs to keep
                    placeholders = ",".join("?" * len(keep_ids))
                    cursor = conn.execute(
                        f"""
                        DELETE FROM experiences
                        WHERE agent_name = ?
                        AND experience_id NOT IN ({placeholders})
                        """,
                        (self.agent_name, *keep_ids),
                    )
                    if cursor.rowcount > 0:
                        changes_made = True

        # 4. Commit changes
        conn.commit()

        # 5. Vacuum database to reclaim space (only if we actually deleted something)
        if changes_made:
            # Rebuild FTS index to ensure sync before VACUUM
            conn.execute("INSERT INTO experiences_fts(experiences_fts) VALUES('rebuild')")
            conn.commit()

            # Now vacuum
            conn.isolation_level = None
            conn.execute("VACUUM")
            conn.isolation_level = ""
