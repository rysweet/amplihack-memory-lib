"""Graph data structures for the common graph abstraction layer.

Public API:
    Direction: Edge traversal direction enum.
    GraphNode: Immutable node with type and properties.
    GraphEdge: Immutable edge connecting two nodes.
    TraversalResult: Container for multi-hop traversal results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Direction(Enum):
    """Direction for edge traversal queries."""

    OUTGOING = "outgoing"
    INCOMING = "incoming"
    BOTH = "both"


@dataclass(frozen=True)
class GraphNode:
    """An immutable node in the graph.

    Attributes:
        node_id: Unique identifier for the node.
        node_type: The node table/label (e.g. "Agent", "Fact").
        properties: Arbitrary key-value properties stored on the node.
        graph_origin: Identifier for the graph this node belongs to,
            useful when merging results across multiple graph stores.
    """

    node_id: str
    node_type: str
    properties: dict[str, Any] = field(default_factory=dict)
    graph_origin: str = ""


@dataclass(frozen=True)
class GraphEdge:
    """An immutable edge in the graph.

    Attributes:
        edge_id: Unique identifier for the edge (auto-generated if empty).
        source_id: Node ID of the source (tail) node.
        target_id: Node ID of the target (head) node.
        edge_type: The relationship table/label (e.g. "KNOWS", "DEPENDS_ON").
        properties: Arbitrary key-value properties stored on the edge.
        graph_origin: Identifier for the graph this edge belongs to.
    """

    edge_id: str = ""
    source_id: str = ""
    target_id: str = ""
    edge_type: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    graph_origin: str = ""


@dataclass
class TraversalResult:
    """Container for multi-hop graph traversal results.

    Attributes:
        paths: Each path is an alternating list of [node, edge, node, edge, ...].
        nodes: Deduplicated set of all nodes visited.
        edges: Deduplicated set of all edges traversed.
        crossed_boundaries: True if the traversal spanned multiple graph origins.
    """

    paths: list[list] = field(default_factory=list)
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    crossed_boundaries: bool = False


__all__ = ["Direction", "GraphNode", "GraphEdge", "TraversalResult"]
