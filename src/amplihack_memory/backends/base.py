"""Abstract base class for memory storage backends."""

from abc import ABC, abstractmethod
from typing import List

from ..experience import Experience, ExperienceType


class MemoryBackend(ABC):
    """Abstract base class for memory storage backends.

    All backend implementations must provide these methods to ensure
    consistent behavior across different storage engines.
    """

    @abstractmethod
    def initialize_schema(self):
        """Initialize the storage schema (tables, indexes, etc.)."""
        pass

    @abstractmethod
    def store_experience(self, experience: Experience) -> str:
        """Store an experience.

        Args:
            experience: Experience to store

        Returns:
            experience_id: ID of stored experience
        """
        pass

    @abstractmethod
    def retrieve_experiences(
        self,
        limit: int | None = None,
        experience_type: ExperienceType | None = None,
        min_confidence: float = 0.0,
    ) -> List[Experience]:
        """Retrieve experiences.

        Args:
            limit: Maximum number of experiences to retrieve
            experience_type: Filter by experience type
            min_confidence: Minimum confidence threshold

        Returns:
            List of experiences sorted by recency
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_statistics(self) -> dict:
        """Get storage statistics.

        Returns:
            Dictionary with statistics
        """
        pass

    @abstractmethod
    def close(self):
        """Close database connection and cleanup resources."""
        pass

    @abstractmethod
    def get_connection(self):
        """Get underlying database connection for advanced operations.

        Returns:
            Database connection object (type depends on backend)
        """
        pass

    @abstractmethod
    def cleanup(
        self,
        auto_compress: bool = True,
        max_age_days: int | None = None,
        max_experiences: int | None = None,
    ):
        """Run cleanup operations.

        Args:
            auto_compress: Compress old experiences (>30 days)
            max_age_days: Delete experiences older than this
            max_experiences: Limit to maximum number of experiences
        """
        pass
