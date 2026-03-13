//! Graph data structures for the common graph abstraction layer.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Direction for edge traversal queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Direction {
    /// Follow edges leaving the node.
    Outgoing,
    /// Follow edges arriving at the node.
    Incoming,
    /// Follow edges in either direction.
    Both,
}

impl Direction {
    /// Return the lowercase string representation of the direction.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Outgoing => "outgoing",
            Self::Incoming => "incoming",
            Self::Both => "both",
        }
    }
}

/// An immutable node in the graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphNode {
    /// Unique identifier for this node.
    pub node_id: String,
    /// Semantic type label (e.g. `"Fact"`, `"Episode"`).
    pub node_type: String,
    /// Arbitrary key-value properties attached to the node.
    #[serde(default)]
    pub properties: HashMap<String, String>,
    /// Origin graph identifier for federated queries.
    #[serde(default)]
    pub graph_origin: String,
}

impl GraphNode {
    /// Create a new node with the given id and type, empty properties and origin.
    pub fn new(node_id: String, node_type: String) -> Self {
        Self {
            node_id,
            node_type,
            properties: HashMap::new(),
            graph_origin: String::new(),
        }
    }
}

/// An immutable edge in the graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Unique identifier for this edge (may be empty for auto-generated edges).
    #[serde(default)]
    pub edge_id: String,
    /// Node id of the edge source.
    #[serde(default)]
    pub source_id: String,
    /// Node id of the edge target.
    #[serde(default)]
    pub target_id: String,
    /// Relationship type label (e.g. `"SIMILAR_TO"`, `"SUPERSEDES"`).
    #[serde(default)]
    pub edge_type: String,
    /// Arbitrary key-value properties attached to the edge.
    #[serde(default)]
    pub properties: HashMap<String, String>,
    /// Origin graph identifier for federated queries.
    #[serde(default)]
    pub graph_origin: String,
}

impl GraphEdge {
    /// Create a new edge between two nodes with the given relationship type.
    pub fn new(source_id: String, target_id: String, edge_type: String) -> Self {
        Self {
            edge_id: String::new(),
            source_id,
            target_id,
            edge_type,
            properties: HashMap::new(),
            graph_origin: String::new(),
        }
    }
}

/// Container for multi-hop graph traversal results.
#[derive(Debug, Clone, Default)]
pub struct TraversalResult {
    /// Ordered paths discovered during traversal, each a sequence of nodes and edges.
    pub paths: Vec<Vec<TraversalItem>>,
    /// All distinct nodes visited during traversal.
    pub nodes: Vec<GraphNode>,
    /// All distinct edges traversed.
    pub edges: Vec<GraphEdge>,
    /// Whether the traversal crossed federated graph boundaries.
    pub crossed_boundaries: bool,
}

/// Item in a traversal path - either a node or edge.
#[derive(Debug, Clone)]
pub enum TraversalItem {
    Node(GraphNode),
    Edge(GraphEdge),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_node() {
        let node = GraphNode::new("n1".into(), "Fact".into());
        assert_eq!(node.node_id, "n1");
        assert_eq!(node.node_type, "Fact");
    }

    #[test]
    fn test_graph_edge() {
        let edge = GraphEdge::new("n1".into(), "n2".into(), "KNOWS".into());
        assert_eq!(edge.source_id, "n1");
        assert_eq!(edge.target_id, "n2");
    }

    #[test]
    fn test_direction() {
        assert_eq!(Direction::Outgoing.as_str(), "outgoing");
        assert_eq!(Direction::Incoming.as_str(), "incoming");
        assert_eq!(Direction::Both.as_str(), "both");
    }
}
