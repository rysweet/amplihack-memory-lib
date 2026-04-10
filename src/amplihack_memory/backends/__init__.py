"""Backend abstraction layer for memory storage."""

from .base import MemoryBackend
from .ladybug_backend import LadybugBackend
from .sqlite_backend import SQLiteBackend

# Backward-compatible alias
KuzuBackend = LadybugBackend

__all__ = ["MemoryBackend", "LadybugBackend", "KuzuBackend", "SQLiteBackend"]
