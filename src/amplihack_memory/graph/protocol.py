"""GraphStore protocol -- the common interface all graph backends implement.

Public API:
    GraphStore: Runtime-checkable protocol defining the graph store contract.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .types import Direction, GraphEdge, GraphNode, TraversalResult


@runtime_checkable
class GraphStore(Protocol):
    """Common interface for graph storage backends.

    Every concrete implementation (Kuzu, SQLite, in-memory, etc.)
    must satisfy this protocol so higher-level code can swap backends
    without changes.
    """

    # ── identity ──────────────────────────────────────────────

    @property
    def store_id(self) -> str:
        """Unique identifier for this store instance."""
        ...

    # ── node operations ───────────────────────────────────────

    def add_node(
        self,
        node_type: str,
        properties: dict[str, Any],
        node_id: str | None = None,
    ) -> GraphNode:
        """Create a node and return it.

        Args:
            node_type: Table / label for the node (e.g. "Agent").
            properties: Arbitrary properties to store on the node.
            node_id: Optional explicit ID; auto-generated when None.

        Returns:
            The newly created GraphNode.
        """
        ...

    def get_node(self, node_id: str) -> GraphNode | None:
        """Fetch a single node by ID, or None if not found."""
        ...

    def query_nodes(
        self,
        node_type: str,
        filters: dict[str, Any] | None = None,
        limit: int = 50,
    ) -> list[GraphNode]:
        """Return nodes of *node_type* matching optional equality filters."""
        ...

    def search_nodes(
        self,
        node_type: str,
        text_fields: list[str],
        query: str,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[GraphNode]:
        """Full-text keyword search across *text_fields* using CONTAINS."""
        ...

    def update_node(self, node_id: str, properties: dict[str, Any]) -> bool:
        """Update properties on an existing node. Returns True on success."""
        ...

    def delete_node(self, node_id: str) -> bool:
        """Delete a node by ID. Returns True if the node existed."""
        ...

    # ── edge operations ───────────────────────────────────────

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        properties: dict[str, Any] | None = None,
    ) -> GraphEdge:
        """Create a directed edge between two existing nodes.

        Raises:
            KeyError: If either source_id or target_id does not exist.
        """
        ...

    def query_neighbors(
        self,
        node_id: str,
        edge_type: str | None = None,
        direction: Direction = Direction.BOTH,
        limit: int = 50,
    ) -> list[tuple[GraphEdge, GraphNode]]:
        """Return edges and neighbor nodes adjacent to *node_id*."""
        ...

    def delete_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
    ) -> bool:
        """Delete a specific edge. Returns True if the edge existed."""
        ...

    # ── traversal ─────────────────────────────────────────────

    def traverse(
        self,
        start_id: str,
        edge_types: list[str] | None = None,
        max_hops: int = 3,
        direction: Direction = Direction.OUTGOING,
        node_filter: dict[str, Any] | None = None,
    ) -> TraversalResult:
        """BFS traversal from *start_id* up to *max_hops* hops."""
        ...

    # ── lifecycle ─────────────────────────────────────────────

    def close(self) -> None:
        """Release resources held by the store."""
        ...


__all__ = ["GraphStore"]
