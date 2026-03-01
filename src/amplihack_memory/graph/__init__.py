"""Common graph abstraction layer for agent graphs and hive mind.

Public API:
    Direction: Edge traversal direction.
    GraphNode: Immutable graph node.
    GraphEdge: Immutable graph edge.
    TraversalResult: Multi-hop traversal result container.
    GraphStore: Protocol all backends implement.
    KuzuGraphStore: Kuzu-backed concrete implementation.
"""

from __future__ import annotations

from .kuzu_store import KuzuGraphStore
from .protocol import GraphStore
from .types import Direction, GraphEdge, GraphNode, TraversalResult

__all__ = [
    "Direction",
    "GraphNode",
    "GraphEdge",
    "TraversalResult",
    "GraphStore",
    "KuzuGraphStore",
]
