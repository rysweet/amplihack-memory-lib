"""Common graph abstraction layer for agent graphs and hive mind.

Public API:
    Direction: Edge traversal direction.
    GraphNode: Immutable graph node.
    GraphEdge: Immutable graph edge.
    TraversalResult: Multi-hop traversal result container.
    GraphStore: Protocol all backends implement.
    KuzuGraphStore: Kuzu-backed concrete implementation.
    PostgresGraphStore: PostgreSQL + AGE distributed implementation.
    InMemoryGraphStore: Dict-based fallback for testing.
    create_hive_store: Factory for creating hive graph stores.
    HiveGraphStore: Kuzu-backed hive mind graph with agent registry.
    AnnotatedResult: Search result with provenance metadata.
    FederatedQueryResult: Container for federated query results.
    FederatedGraphStore: Composite local + hive graph store.
"""

from __future__ import annotations

from .federated_store import AnnotatedResult, FederatedGraphStore, FederatedQueryResult
from .hive_store import HiveGraphStore
from .kuzu_store import KuzuGraphStore
from .postgres_store import InMemoryGraphStore, PostgresGraphStore, create_hive_store
from .protocol import GraphStore
from .types import Direction, GraphEdge, GraphNode, TraversalResult

__all__ = [
    "Direction",
    "GraphNode",
    "GraphEdge",
    "TraversalResult",
    "GraphStore",
    "KuzuGraphStore",
    "PostgresGraphStore",
    "InMemoryGraphStore",
    "create_hive_store",
    "HiveGraphStore",
    "AnnotatedResult",
    "FederatedQueryResult",
    "FederatedGraphStore",
]
