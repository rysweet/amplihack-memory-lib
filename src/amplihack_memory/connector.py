"""Memory connector factory for database lifecycle management."""

from pathlib import Path
from typing import List

from .backends.base import MemoryBackend
from .backends.kuzu_backend import KuzuBackend
from .backends.sqlite_backend import SQLiteBackend
from .experience import Experience, ExperienceType


class MemoryConnector:
    """Database connection management for agent memory.

    Factory class that delegates to specific backend implementations.
    Provides a unified interface regardless of storage backend.

    Attributes:
        agent_name: Name of the agent (used for isolation)
        storage_path: Directory for database files
        max_memory_mb: Maximum storage size in MB
        enable_compression: Whether to enable compression
        backend: Backend type ('kuzu' or 'sqlite')
    """

    def __init__(
        self,
        agent_name: str,
        storage_path: Path | None = None,
        max_memory_mb: int = 100,
        enable_compression: bool = True,
        backend: str = "kuzu",
    ):
        """Initialize memory connector.

        Args:
            agent_name: Agent identifier (must not be empty)
            storage_path: Storage directory (defaults to ~/.amplihack/memory/<agent>)
            max_memory_mb: Maximum storage size in MB
            enable_compression: Enable automatic compression
            backend: Backend type ('kuzu' or 'sqlite', default: 'kuzu')

        Raises:
            ValueError: If agent_name is invalid or backend is unknown
            PermissionError: If storage_path is not writable
        """
        # Validate agent_name
        if not agent_name or not agent_name.strip():
            raise ValueError("agent_name cannot be empty")

        self.agent_name = agent_name.strip()
        self.max_memory_mb = max_memory_mb
        self.enable_compression = enable_compression
        self.backend_type = backend

        # Validate max_memory_mb
        if max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")

        # Set storage path
        if storage_path is None:
            storage_path = Path.home() / ".amplihack" / "memory" / self.agent_name
        self.storage_path = Path(storage_path)

        # Create storage directory
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise PermissionError(f"Cannot create storage directory: {storage_path}") from e

        # Initialize backend
        if backend == "kuzu":
            db_path = self.storage_path / "kuzu_db"
            self._backend: MemoryBackend = KuzuBackend(
                db_path=db_path,
                agent_name=agent_name,
                max_memory_mb=max_memory_mb,
                enable_compression=enable_compression,
            )
        elif backend == "sqlite":
            db_path = self.storage_path / "experiences.db"
            self._backend: MemoryBackend = SQLiteBackend(
                db_path=db_path,
                agent_name=agent_name,
                max_memory_mb=max_memory_mb,
                enable_compression=enable_compression,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'kuzu' or 'sqlite'.")

        # Store db_path for compatibility
        self.db_path = db_path

    def store_experience(self, experience: Experience) -> str:
        """Store an experience in the database.

        Args:
            experience: Experience to store

        Returns:
            experience_id: ID of stored experience

        Raises:
            InvalidExperienceError: If experience is invalid
        """
        return self._backend.store_experience(experience)

    def retrieve_experiences(
        self,
        limit: int | None = None,
        experience_type: ExperienceType | None = None,
        min_confidence: float = 0.0,
    ) -> List[Experience]:
        """Retrieve experiences for this agent.

        Args:
            limit: Maximum number of experiences to retrieve
            experience_type: Filter by experience type
            min_confidence: Minimum confidence threshold

        Returns:
            List of experiences sorted by recency
        """
        return self._backend.retrieve_experiences(
            limit=limit,
            experience_type=experience_type,
            min_confidence=min_confidence,
        )

    def search(
        self,
        query: str,
        experience_type: ExperienceType | None = None,
        min_confidence: float = 0.0,
        limit: int = 10,
    ) -> List[Experience]:
        """Search experiences by text query.

        Args:
            query: Search query (text)
            experience_type: Filter by experience type
            min_confidence: Minimum confidence threshold
            limit: Maximum results

        Returns:
            List of matching experiences
        """
        return self._backend.search(
            query=query,
            experience_type=experience_type,
            min_confidence=min_confidence,
            limit=limit,
        )

    def get_statistics(self) -> dict:
        """Get storage statistics.

        Returns:
            Dictionary with statistics
        """
        return self._backend.get_statistics()

    def close(self) -> None:
        """Close database connection."""
        if self._backend:
            self._backend.close()
            self._backend = None

    @property
    def _connection(self):
        """Get underlying connection for backward compatibility.

        Returns:
            Database connection object (type depends on backend)
        """
        return self._backend.get_connection()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.close()
