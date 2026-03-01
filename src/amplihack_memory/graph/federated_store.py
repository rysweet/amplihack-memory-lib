"""FederatedGraphStore -- composites a local agent graph + hive graph.

Writes go to the local graph. Reads fan out to local + hive and
deduplicate. Traversal crosses graph boundaries transparently.

Public API:
    AnnotatedResult: A search result annotated with source and confidence.
    FederatedQueryResult: Container for federated query results.
    FederatedGraphStore: Composite store implementing the GraphStore protocol.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .protocol import GraphStore
from .types import Direction, GraphEdge, GraphNode, TraversalResult


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class AnnotatedResult:
    """A search result annotated with provenance and confidence metadata.

    Attributes:
        node: The graph node.
        source: Origin of the result -- "local", "hive", or "peer:<agent_id>".
        confirmation_count: Number of cross-agent confirmations.
        confidence: Confidence score derived from trust and confirmations.
    """

    node: GraphNode
    source: str
    confirmation_count: int = 0
    confidence: float = 1.0


@dataclass
class FederatedQueryResult:
    """Container for results from a federated query across stores.

    Attributes:
        results: Annotated results merged from local and hive.
        local_count: Number of results from the local store.
        hive_count: Number of results from the hive store.
        expert_agents: Agent IDs identified as experts for the query.
    """

    results: list[AnnotatedResult] = field(default_factory=list)
    local_count: int = 0
    hive_count: int = 0
    expert_agents: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _content_hash(node: GraphNode) -> str:
    """Deterministic hash of a node's type + properties for deduplication.

    Uses node_type and sorted properties so two nodes with identical
    content from different stores hash the same.
    """
    parts = [node.node_type]
    for k in sorted(node.properties.keys()):
        parts.append(f"{k}={node.properties[k]}")
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _deduplicate_nodes(nodes: list[GraphNode]) -> list[GraphNode]:
    """Remove duplicate nodes by content hash, keeping the first occurrence."""
    seen: set[str] = set()
    unique: list[GraphNode] = []
    for node in nodes:
        h = _content_hash(node)
        if h not in seen:
            seen.add(h)
            unique.append(node)
    return unique


# ---------------------------------------------------------------------------
# FederatedGraphStore
# ---------------------------------------------------------------------------


class FederatedGraphStore:
    """Composes a local agent graph + hive graph behind the GraphStore protocol.

    Writes go to the local graph only. Reads fan out to both local and hive,
    with content-hash deduplication. Traversal crosses graph boundaries
    transparently and sets crossed_boundaries=True when results span both.

    Args:
        local_store: The agent's own KuzuGraphStore.
        hive_store: The shared hive graph store.
    """

    def __init__(self, local_store: GraphStore, hive_store: GraphStore) -> None:
        self._local = local_store
        self._hive = hive_store

    # ── identity ──────────────────────────────────────────────

    @property
    def store_id(self) -> str:
        """Federated store ID combines local and hive IDs."""
        return f"federated:{self._local.store_id}+{self._hive.store_id}"

    # ── write operations (local only) ────────────────────────

    def add_node(
        self,
        node_type: str,
        properties: dict[str, Any],
        node_id: str | None = None,
    ) -> GraphNode:
        """Create a node in the local store only.

        Args:
            node_type: Table/label for the node.
            properties: Node properties.
            node_id: Optional explicit ID.

        Returns:
            The created GraphNode.
        """
        return self._local.add_node(node_type, properties, node_id)

    def update_node(self, node_id: str, properties: dict[str, Any]) -> bool:
        """Update a node in the local store only.

        Args:
            node_id: Node to update.
            properties: Properties to set.

        Returns:
            True if update succeeded.
        """
        return self._local.update_node(node_id, properties)

    def delete_node(self, node_id: str) -> bool:
        """Delete a node from the local store only.

        Args:
            node_id: Node to delete.

        Returns:
            True if the node existed.
        """
        return self._local.delete_node(node_id)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        properties: dict[str, Any] | None = None,
    ) -> GraphEdge:
        """Create an edge in the local store only.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            edge_type: Relationship type.
            properties: Edge properties.

        Returns:
            The created GraphEdge.

        Raises:
            KeyError: If source or target not found in local store.
        """
        return self._local.add_edge(source_id, target_id, edge_type, properties)

    def delete_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
    ) -> bool:
        """Delete an edge from the local store only.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            edge_type: Relationship type.

        Returns:
            True if the edge existed.
        """
        return self._local.delete_edge(source_id, target_id, edge_type)

    # ── read operations (fan out to both stores) ─────────────

    def get_node(self, node_id: str) -> GraphNode | None:
        """Fetch a node by ID, checking local first then hive.

        Args:
            node_id: Node to look up.

        Returns:
            The GraphNode, or None if not found in either store.
        """
        node = self._local.get_node(node_id)
        if node is not None:
            return node
        return self._hive.get_node(node_id)

    def query_nodes(
        self,
        node_type: str,
        filters: dict[str, Any] | None = None,
        limit: int = 50,
    ) -> list[GraphNode]:
        """Query nodes from both stores and deduplicate.

        Args:
            node_type: Node type to query.
            filters: Optional equality filters.
            limit: Maximum results.

        Returns:
            Deduplicated list of GraphNodes from both stores.
        """
        local_results = self._local.query_nodes(node_type, filters, limit)
        hive_results = self._hive.query_nodes(node_type, filters, limit)
        merged = local_results + hive_results
        return _deduplicate_nodes(merged)[:limit]

    def search_nodes(
        self,
        node_type: str,
        text_fields: list[str],
        query: str,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[GraphNode]:
        """Search nodes across both stores with content-hash deduplication.

        Args:
            node_type: Node type to search.
            text_fields: Fields to search in.
            query: Search query.
            filters: Optional equality filters.
            limit: Maximum results.

        Returns:
            Deduplicated search results from both stores.
        """
        local_results = self._local.search_nodes(
            node_type, text_fields, query, filters, limit,
        )
        hive_results = self._hive.search_nodes(
            node_type, text_fields, query, filters, limit,
        )
        merged = local_results + hive_results
        return _deduplicate_nodes(merged)[:limit]

    def query_neighbors(
        self,
        node_id: str,
        edge_type: str | None = None,
        direction: Direction = Direction.BOTH,
        limit: int = 50,
    ) -> list[tuple[GraphEdge, GraphNode]]:
        """Query neighbors from both stores.

        Args:
            node_id: Node to find neighbors of.
            edge_type: Optional edge type filter.
            direction: Traversal direction.
            limit: Maximum results.

        Returns:
            Combined neighbor results from both stores.
        """
        local_neighbors = self._local.query_neighbors(
            node_id, edge_type, direction, limit,
        )
        hive_neighbors = self._hive.query_neighbors(
            node_id, edge_type, direction, limit,
        )

        # Deduplicate by neighbor node content hash
        seen: set[str] = set()
        merged: list[tuple[GraphEdge, GraphNode]] = []
        for edge, node in local_neighbors + hive_neighbors:
            h = _content_hash(node)
            if h not in seen:
                seen.add(h)
                merged.append((edge, node))
        return merged[:limit]

    # ── cross-boundary traversal ─────────────────────────────

    def traverse(
        self,
        start_id: str,
        edge_types: list[str] | None = None,
        max_hops: int = 3,
        direction: Direction = Direction.OUTGOING,
        node_filter: dict[str, Any] | None = None,
    ) -> TraversalResult:
        """BFS traversal that crosses local/hive boundaries.

        At each hop, queries neighbors from BOTH stores. If any node
        comes from a different graph_origin, crossed_boundaries is True.

        Args:
            start_id: Node ID to start from.
            edge_types: Optional edge type filter.
            max_hops: Maximum BFS depth.
            direction: Traversal direction.
            node_filter: Optional property filter on nodes.

        Returns:
            TraversalResult with crossed_boundaries set appropriately.
        """
        # Find the start node in either store
        start_node = self._local.get_node(start_id)
        if start_node is None:
            start_node = self._hive.get_node(start_id)
        if start_node is None:
            return TraversalResult()

        visited_ids: set[str] = {start_id}
        all_nodes: dict[str, GraphNode] = {start_id: start_node}
        all_edges: list[GraphEdge] = []
        paths: list[list] = []
        origins: set[str] = {start_node.graph_origin}

        # BFS queue: (current_node_id, current_path, hops_so_far)
        queue: deque[tuple[str, list, int]] = deque()
        queue.append((start_id, [start_node], 0))

        while queue:
            current_id, current_path, hops = queue.popleft()
            if hops >= max_hops:
                continue

            # Get neighbors from BOTH stores
            neighbors: list[tuple[GraphEdge, GraphNode]] = []
            if edge_types:
                for et in edge_types:
                    neighbors.extend(
                        self._local.query_neighbors(current_id, et, direction)
                    )
                    neighbors.extend(
                        self._hive.query_neighbors(current_id, et, direction)
                    )
            else:
                neighbors.extend(
                    self._local.query_neighbors(current_id, direction=direction)
                )
                neighbors.extend(
                    self._hive.query_neighbors(current_id, direction=direction)
                )

            for edge, neighbor in neighbors:
                # Apply node filter
                if node_filter:
                    if not _matches_filter(neighbor, node_filter):
                        continue

                all_edges.append(edge)
                origins.add(neighbor.graph_origin)

                new_path = current_path + [edge, neighbor]

                if neighbor.node_id not in visited_ids:
                    visited_ids.add(neighbor.node_id)
                    all_nodes[neighbor.node_id] = neighbor
                    queue.append((neighbor.node_id, new_path, hops + 1))

                paths.append(new_path)

        return TraversalResult(
            paths=paths,
            nodes=list(all_nodes.values()),
            edges=all_edges,
            crossed_boundaries=len(origins) > 1,
        )

    # ── federated query ──────────────────────────────────────

    def federated_query(
        self,
        query: str,
        node_type: str = "Fact",
        text_fields: list[str] | None = None,
        limit: int = 20,
    ) -> FederatedQueryResult:
        """High-level federated query across local + hive stores.

        1. Queries the local store.
        2. Queries the hive store.
        3. Finds expert agents via the hive (if HiveAgent nodes exist).
        4. Merges, deduplicates, and annotates results with source info
           and confirmation counts.

        Args:
            query: Search query string.
            node_type: Node type to search (default "Fact").
            text_fields: Fields to search in (default ["content", "text"]).
            limit: Maximum total results.

        Returns:
            FederatedQueryResult with annotated results.
        """
        search_fields = text_fields or ["content"]

        # 1. Query local store
        local_results = self._safe_search(self._local, node_type, search_fields, query, limit)

        # 2. Query hive store
        hive_results = self._safe_search(self._hive, node_type, search_fields, query, limit)

        # 3. Find expert agents via hive (search HiveAgent nodes)
        expert_nodes = self._hive.search_nodes(
            "HiveAgent", ["domain"], query, limit=10,
        )
        expert_agents = [n.node_id for n in expert_nodes]

        # 4. Merge, deduplicate, annotate
        seen_hashes: set[str] = set()
        annotated: list[AnnotatedResult] = []

        # Local results first (higher priority)
        for node in local_results:
            h = _content_hash(node)
            if h not in seen_hashes:
                seen_hashes.add(h)
                annotated.append(AnnotatedResult(
                    node=node,
                    source="local",
                    confirmation_count=0,
                    confidence=1.0,
                ))

        # Then hive results
        hive_count = 0
        for node in hive_results:
            h = _content_hash(node)
            if h not in seen_hashes:
                seen_hashes.add(h)
                # Determine source based on graph_origin
                source = "hive"
                if node.graph_origin and node.graph_origin not in (
                    "__hive__", self._local.store_id,
                ):
                    source = f"peer:{node.graph_origin}"

                annotated.append(AnnotatedResult(
                    node=node,
                    source=source,
                    confirmation_count=0,
                    confidence=0.9,  # Slight discount for non-local
                ))
                hive_count += 1

        return FederatedQueryResult(
            results=annotated[:limit],
            local_count=len(local_results),
            hive_count=hive_count,
            expert_agents=expert_agents,
        )

    # ── private helpers ─────────────────────────────────────

    @staticmethod
    def _safe_search(
        store: GraphStore,
        node_type: str,
        text_fields: list[str],
        query: str,
        limit: int,
    ) -> list[GraphNode]:
        """Search a store, returning empty list if the node type or fields don't exist.

        Kuzu raises RuntimeError when searching columns that don't exist in
        a node table's schema. This wrapper catches that gracefully.
        """
        try:
            return store.search_nodes(node_type, text_fields, query, limit=limit)
        except RuntimeError:
            # Column doesn't exist in this store's schema -- not an error
            return []

    # ── lifecycle ────────────────────────────────────────────

    def close(self) -> None:
        """Close both local and hive stores."""
        self._local.close()
        self._hive.close()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _matches_filter(node: GraphNode, node_filter: dict[str, Any]) -> bool:
    """Check whether a node's properties match all filter criteria."""
    for k, v in node_filter.items():
        if node.properties.get(k) != str(v):
            return False
    return True


__all__ = ["AnnotatedResult", "FederatedQueryResult", "FederatedGraphStore"]
