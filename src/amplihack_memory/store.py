"""High-level experience storage and retrieval with automatic management."""

import json
from datetime import datetime, timedelta
from pathlib import Path

from .connector import MemoryConnector
from .exceptions import InvalidExperienceError, MemoryQuotaExceededError
from .experience import Experience, ExperienceType


class ExperienceStore:
    """High-level storage and retrieval of experiences with auto-management.

    Features:
    - Automatic compression of old experiences
    - Retention policies (max_age_days, max_experiences)
    - Duplicate detection
    - Full-text search
    - Statistics tracking

    Attributes:
        agent_name: Agent identifier
        auto_compress: Enable automatic compression
        max_age_days: Maximum age for experiences (days)
        max_experiences: Maximum number of experiences
        max_memory_mb: Maximum storage size (MB)
    """

    def __init__(
        self,
        agent_name: str,
        auto_compress: bool = True,
        max_age_days: int | None = None,
        max_experiences: int | None = None,
        max_memory_mb: int = 100,
        storage_path: Path | None = None,
        backend: str = "kuzu",
    ):
        """Initialize experience store.

        Args:
            agent_name: Agent identifier
            auto_compress: Enable automatic compression (default: True)
            max_age_days: Maximum age for experiences (None = no limit)
            max_experiences: Maximum number of experiences (None = no limit)
            max_memory_mb: Maximum storage size in MB
            storage_path: Storage directory (optional)
            backend: Backend type ('kuzu' or 'sqlite', default: 'kuzu')
        """
        self.agent_name = agent_name
        self.auto_compress = auto_compress
        self.max_age_days = max_age_days
        self.max_experiences = max_experiences
        self.max_memory_mb = max_memory_mb

        # Initialize connector
        self.connector = MemoryConnector(
            agent_name=agent_name,
            storage_path=storage_path,
            max_memory_mb=max_memory_mb,
            enable_compression=auto_compress,
            backend=backend,
        )

    def add(self, experience: Experience) -> str:
        """Add experience with automatic management.

        Args:
            experience: Experience to add

        Returns:
            experience_id: ID of stored experience

        Raises:
            InvalidExperienceError: If experience is invalid
            MemoryQuotaExceededError: If storage quota exceeded
        """
        # Validate experience
        self._validate_experience(experience)

        # Check storage quota
        self._check_quota()

        # Store experience
        exp_id = self.connector.store_experience(experience)

        # Run cleanup if needed
        if self.auto_compress or self.max_age_days or self.max_experiences:
            self._cleanup()

        return exp_id

    def search(
        self,
        query: str,
        experience_type: ExperienceType | None = None,
        min_confidence: float = 0.0,
        limit: int = 10,
    ) -> list[Experience]:
        """Search experiences by text query.

        Uses backend-specific full-text search implementation.

        Args:
            query: Search query (text)
            experience_type: Filter by experience type
            min_confidence: Minimum confidence threshold
            limit: Maximum results

        Returns:
            List of matching experiences
        """
        # Delegate to backend's search implementation
        return self.connector.search(
            query=query,
            experience_type=experience_type,
            min_confidence=min_confidence,
            limit=limit,
        )

    def get_statistics(self) -> dict:
        """Get storage statistics.

        Returns:
            Dictionary with statistics:
            - total_experiences: Total count
            - by_type: Count by experience type
            - storage_size_kb: Storage size in KB
            - compressed_experiences: Count of compressed experiences
            - compression_ratio: Compression ratio (if compression enabled)
        """
        # Delegate to backend's statistics implementation
        return self.connector.get_statistics()

    def _cleanup(self):
        """Run cleanup: compression, age limit, count limit."""
        # Delegate to backend-specific cleanup implementation
        self.connector._backend.cleanup(
            auto_compress=self.auto_compress,
            max_age_days=self.max_age_days,
            max_experiences=self.max_experiences,
        )

    def _validate_experience(self, experience: Experience):
        """Validate experience before storage.

        Raises:
            InvalidExperienceError: If validation fails
        """
        if not experience.context.strip():
            raise InvalidExperienceError("context cannot be empty")

        if not experience.outcome.strip():
            raise InvalidExperienceError("outcome cannot be empty")

        if not (0.0 <= experience.confidence <= 1.0):
            raise InvalidExperienceError("confidence must be between 0.0 and 1.0")

    def _check_quota(self):
        """Check if storage quota is exceeded.

        Raises:
            MemoryQuotaExceededError: If quota exceeded
        """
        db_path = self.connector.db_path
        if not db_path.exists():
            return

        size_mb = db_path.stat().st_size / (1024 * 1024)
        if size_mb > self.max_memory_mb:
            raise MemoryQuotaExceededError(
                f"Storage quota exceeded: {size_mb:.1f}MB > {self.max_memory_mb}MB"
            )
