//! Graph data structures for the common graph abstraction layer.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Direction for edge traversal queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Direction {
    Outgoing,
    Incoming,
    Both,
}

impl Direction {
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
    pub node_id: String,
    pub node_type: String,
    #[serde(default)]
    pub properties: HashMap<String, String>,
    #[serde(default)]
    pub graph_origin: String,
}

impl GraphNode {
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
    #[serde(default)]
    pub edge_id: String,
    #[serde(default)]
    pub source_id: String,
    #[serde(default)]
    pub target_id: String,
    #[serde(default)]
    pub edge_type: String,
    #[serde(default)]
    pub properties: HashMap<String, String>,
    #[serde(default)]
    pub graph_origin: String,
}

impl GraphEdge {
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
    pub paths: Vec<Vec<TraversalItem>>,
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
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
