"""Common graph abstraction layer for agent graphs and hive mind.

Public API:
    Direction: Edge traversal direction.
    GraphNode: Immutable graph node.
    GraphEdge: Immutable graph edge.
    TraversalResult: Multi-hop traversal result container.
    GraphStore: Protocol all backends implement.
    LadybugGraphStore: Kuzu-backed concrete implementation.
    HiveGraphStore: Kuzu-backed hive mind graph with agent registry.
    AnnotatedResult: Search result with provenance metadata.
    FederatedQueryResult: Container for federated query results.
    FederatedGraphStore: Composite local + hive graph store.
"""

from __future__ import annotations

from .federated_store import AnnotatedResult, FederatedGraphStore, FederatedQueryResult
from .hive_store import HiveGraphStore
from .ladybug_store import LadybugGraphStore
from .protocol import GraphStore
from .types import Direction, GraphEdge, GraphNode, TraversalResult

# Backward-compatible alias
KuzuGraphStore = LadybugGraphStore

__all__ = [
    "Direction",
    "GraphNode",
    "GraphEdge",
    "TraversalResult",
    "GraphStore",
    "LadybugGraphStore",
    "KuzuGraphStore",
    "HiveGraphStore",
    "AnnotatedResult",
    "FederatedQueryResult",
    "FederatedGraphStore",
]
