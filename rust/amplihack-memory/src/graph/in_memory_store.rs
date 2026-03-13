//! InMemoryGraphStore -- Dict-based graph store for testing.
//!
//! Provides a simple in-memory implementation of the GraphStore trait
//! for use in tests and as a fallback when no external database is available.

use std::collections::HashMap;

use super::protocol::GraphStore;
use super::traversal::bfs_traverse;
use super::types::{Direction, GraphEdge, GraphNode, TraversalResult};

/// In-memory graph store backed by HashMaps.
///
/// Edges are indexed by source_id for O(k) outgoing lookups and by
/// target_id (`reverse_edges`) for O(k) incoming lookups.
pub struct InMemoryGraphStore {
    store_id: String,
    nodes: HashMap<String, GraphNode>,
    /// Edges indexed by source_id for efficient outgoing neighbor queries.
    edges: HashMap<String, Vec<GraphEdge>>,
    /// Reverse index: target_id → edges, for O(k) incoming neighbor queries.
    reverse_edges: HashMap<String, Vec<GraphEdge>>,
    graph_origin: String,
}

impl InMemoryGraphStore {
    pub fn new(store_id: Option<&str>) -> Self {
        let id = store_id
            .map(String::from)
            .unwrap_or_else(|| format!("inmem-{}", &uuid::Uuid::new_v4().to_string()[..8]));
        Self {
            store_id: id.clone(),
            nodes: HashMap::new(),
            edges: HashMap::new(),
            reverse_edges: HashMap::new(),
            graph_origin: id,
        }
    }
}

impl GraphStore for InMemoryGraphStore {
    fn store_id(&self) -> &str {
        &self.store_id
    }

    fn add_node(
        &mut self,
        node_type: &str,
        properties: HashMap<String, String>,
        node_id: Option<&str>,
    ) -> crate::Result<GraphNode> {
        let id = node_id
            .map(String::from)
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        let node = GraphNode {
            node_id: id.clone(),
            node_type: node_type.to_string(),
            properties,
            graph_origin: self.graph_origin.clone(),
        };

        self.nodes.insert(id, node.clone());
        Ok(node)
    }

    fn get_node(&self, node_id: &str) -> Option<GraphNode> {
        self.nodes.get(node_id).cloned()
    }

    fn query_nodes(
        &self,
        node_type: &str,
        filters: Option<&HashMap<String, String>>,
        limit: usize,
    ) -> Vec<GraphNode> {
        self.nodes
            .values()
            .filter(|n| n.node_type == node_type)
            .filter(|n| {
                if let Some(f) = filters {
                    f.iter().all(|(k, v)| n.properties.get(k) == Some(v))
                } else {
                    true
                }
            })
            .take(limit)
            .cloned()
            .collect()
    }

    fn search_nodes(
        &self,
        node_type: &str,
        text_fields: &[String],
        query: &str,
        filters: Option<&HashMap<String, String>>,
        limit: usize,
    ) -> Vec<GraphNode> {
        let query_lower = query.to_lowercase();
        self.nodes
            .values()
            .filter(|n| n.node_type == node_type)
            .filter(|n| {
                text_fields.iter().any(|field| {
                    n.properties
                        .get(field)
                        .is_some_and(|v| v.to_lowercase().contains(&query_lower))
                })
            })
            .filter(|n| {
                if let Some(f) = filters {
                    f.iter().all(|(k, v)| n.properties.get(k) == Some(v))
                } else {
                    true
                }
            })
            .take(limit)
            .cloned()
            .collect()
    }

    fn update_node(&mut self, node_id: &str, properties: HashMap<String, String>) -> bool {
        if let Some(node) = self.nodes.get_mut(node_id) {
            for (k, v) in properties {
                node.properties.insert(k, v);
            }
            true
        } else {
            false
        }
    }

    fn delete_node(&mut self, node_id: &str) -> bool {
        // Remove outgoing edges and clean their reverse entries
        if let Some(outgoing) = self.edges.remove(node_id) {
            for edge in &outgoing {
                if let Some(rev) = self.reverse_edges.get_mut(&edge.target_id) {
                    rev.retain(|e| e.edge_id != edge.edge_id);
                }
            }
        }
        // Remove incoming edges (via reverse index) and clean their forward entries
        if let Some(incoming) = self.reverse_edges.remove(node_id) {
            for edge in &incoming {
                if let Some(fwd) = self.edges.get_mut(&edge.source_id) {
                    fwd.retain(|e| e.edge_id != edge.edge_id);
                }
            }
        }
        self.nodes.remove(node_id).is_some()
    }

    fn add_edge(
        &mut self,
        source_id: &str,
        target_id: &str,
        edge_type: &str,
        properties: Option<HashMap<String, String>>,
    ) -> crate::Result<GraphEdge> {
        if !self.nodes.contains_key(source_id) {
            return Err(crate::MemoryError::Internal(format!(
                "source node not found: {source_id}"
            )));
        }
        if !self.nodes.contains_key(target_id) {
            return Err(crate::MemoryError::Internal(format!(
                "target node not found: {target_id}"
            )));
        }

        let edge = GraphEdge {
            edge_id: uuid::Uuid::new_v4().to_string(),
            source_id: source_id.to_string(),
            target_id: target_id.to_string(),
            edge_type: edge_type.to_string(),
            properties: properties.unwrap_or_default(),
            graph_origin: self.graph_origin.clone(),
        };

        self.edges
            .entry(source_id.to_string())
            .or_default()
            .push(edge.clone());
        self.reverse_edges
            .entry(target_id.to_string())
            .or_default()
            .push(edge.clone());
        Ok(edge)
    }

    fn query_neighbors(
        &self,
        node_id: &str,
        edge_type: Option<&str>,
        direction: Direction,
        limit: usize,
    ) -> Vec<(GraphEdge, GraphNode)> {
        let mut results = Vec::new();

        let type_ok = |e: &GraphEdge| edge_type.map_or(true, |et| e.edge_type == et);

        let push_if_valid =
            |edge: &GraphEdge, neighbor_id: &str, results: &mut Vec<(GraphEdge, GraphNode)>| {
                if let Some(node) = self.nodes.get(neighbor_id) {
                    results.push((edge.clone(), node.clone()));
                }
            };

        // Outgoing / Both: lookup edges by source_id — O(k)
        if direction == Direction::Outgoing || direction == Direction::Both {
            if let Some(bucket) = self.edges.get(node_id) {
                for edge in bucket {
                    if type_ok(edge) {
                        push_if_valid(edge, &edge.target_id, &mut results);
                        if results.len() >= limit {
                            return results;
                        }
                    }
                }
            }
        }

        // Incoming / Both: lookup reverse index by target_id — O(k)
        if direction == Direction::Incoming || direction == Direction::Both {
            if let Some(bucket) = self.reverse_edges.get(node_id) {
                for edge in bucket {
                    if type_ok(edge) {
                        push_if_valid(edge, &edge.source_id, &mut results);
                        if results.len() >= limit {
                            return results;
                        }
                    }
                }
            }
        }

        results
    }

    fn delete_edge(&mut self, source_id: &str, target_id: &str, edge_type: &str) -> bool {
        let removed = if let Some(bucket) = self.edges.get_mut(source_id) {
            let len_before = bucket.len();
            bucket.retain(|e| !(e.target_id == target_id && e.edge_type == edge_type));
            bucket.len() < len_before
        } else {
            false
        };
        if removed {
            if let Some(rev) = self.reverse_edges.get_mut(target_id) {
                rev.retain(|e| !(e.source_id == source_id && e.edge_type == edge_type));
            }
        }
        removed
    }

    fn traverse(
        &self,
        start_id: &str,
        edge_types: Option<&[String]>,
        max_hops: usize,
        direction: Direction,
        node_filter: Option<&HashMap<String, String>>,
    ) -> TraversalResult {
        let start_node = match self.get_node(start_id) {
            Some(n) => n,
            None => return TraversalResult::default(),
        };

        bfs_traverse(
            start_node,
            edge_types,
            max_hops,
            node_filter,
            1000,
            |id, et, _depth| self.query_neighbors(id, et, direction, 50),
        )
    }

    fn close(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.reverse_edges.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_get_node() {
        let mut store = InMemoryGraphStore::new(Some("test"));
        let mut props = HashMap::new();
        props.insert("name".into(), "Alice".into());
        let node = store.add_node("Person", props, Some("p1")).unwrap();
        assert_eq!(node.node_id, "p1");
        assert_eq!(node.node_type, "Person");

        let fetched = store.get_node("p1").unwrap();
        assert_eq!(fetched.properties.get("name").unwrap(), "Alice");
    }

    #[test]
    fn test_add_edge_and_neighbors() {
        let mut store = InMemoryGraphStore::new(Some("test"));
        store.add_node("Person", HashMap::new(), Some("a")).unwrap();
        store.add_node("Person", HashMap::new(), Some("b")).unwrap();
        store.add_edge("a", "b", "KNOWS", None).unwrap();

        let neighbors = store.query_neighbors("a", Some("KNOWS"), Direction::Outgoing, 10);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].1.node_id, "b");
    }

    #[test]
    fn test_delete_node() {
        let mut store = InMemoryGraphStore::new(Some("test"));
        store.add_node("Person", HashMap::new(), Some("x")).unwrap();
        assert!(store.delete_node("x"));
        assert!(store.get_node("x").is_none());
    }

    #[test]
    fn test_search_nodes() {
        let mut store = InMemoryGraphStore::new(Some("test"));
        let mut props = HashMap::new();
        props.insert("content".into(), "Rust programming language".into());
        store.add_node("Fact", props, Some("f1")).unwrap();

        let results = store.search_nodes("Fact", &["content".to_string()], "rust", None, 10);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_traverse() {
        let mut store = InMemoryGraphStore::new(Some("test"));
        store.add_node("N", HashMap::new(), Some("a")).unwrap();
        store.add_node("N", HashMap::new(), Some("b")).unwrap();
        store.add_node("N", HashMap::new(), Some("c")).unwrap();
        store.add_edge("a", "b", "LINK", None).unwrap();
        store.add_edge("b", "c", "LINK", None).unwrap();

        let result = store.traverse("a", None, 3, Direction::Outgoing, None);
        assert!(result.nodes.len() >= 2);
    }

    #[test]
    fn test_incoming_neighbors() {
        let mut store = InMemoryGraphStore::new(Some("test"));
        store.add_node("N", HashMap::new(), Some("a")).unwrap();
        store.add_node("N", HashMap::new(), Some("b")).unwrap();
        store.add_node("N", HashMap::new(), Some("c")).unwrap();
        store.add_edge("a", "c", "LINK", None).unwrap();
        store.add_edge("b", "c", "LINK", None).unwrap();

        let incoming = store.query_neighbors("c", None, Direction::Incoming, 10);
        assert_eq!(incoming.len(), 2);
        let ids: Vec<&str> = incoming.iter().map(|(_, n)| n.node_id.as_str()).collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
    }

    #[test]
    fn test_both_direction_neighbors() {
        let mut store = InMemoryGraphStore::new(Some("test"));
        store.add_node("N", HashMap::new(), Some("a")).unwrap();
        store.add_node("N", HashMap::new(), Some("b")).unwrap();
        store.add_node("N", HashMap::new(), Some("c")).unwrap();
        store.add_edge("a", "b", "LINK", None).unwrap();
        store.add_edge("c", "b", "LINK", None).unwrap();

        let both = store.query_neighbors("b", None, Direction::Both, 10);
        assert_eq!(both.len(), 2);

        store.add_node("N", HashMap::new(), Some("d")).unwrap();
        store.add_edge("b", "d", "LINK", None).unwrap();
        let both = store.query_neighbors("b", None, Direction::Both, 10);
        assert_eq!(both.len(), 3);
    }

    #[test]
    fn test_delete_edge_updates_reverse_index() {
        let mut store = InMemoryGraphStore::new(Some("test"));
        store.add_node("N", HashMap::new(), Some("a")).unwrap();
        store.add_node("N", HashMap::new(), Some("b")).unwrap();
        store.add_edge("a", "b", "LINK", None).unwrap();

        assert_eq!(
            store
                .query_neighbors("b", None, Direction::Incoming, 10)
                .len(),
            1
        );
        assert!(store.delete_edge("a", "b", "LINK"));
        assert_eq!(
            store
                .query_neighbors("b", None, Direction::Incoming, 10)
                .len(),
            0
        );
    }

    #[test]
    fn test_delete_node_updates_reverse_index() {
        let mut store = InMemoryGraphStore::new(Some("test"));
        store.add_node("N", HashMap::new(), Some("a")).unwrap();
        store.add_node("N", HashMap::new(), Some("b")).unwrap();
        store.add_node("N", HashMap::new(), Some("c")).unwrap();
        store.add_edge("a", "b", "LINK", None).unwrap();
        store.add_edge("c", "b", "LINK", None).unwrap();

        assert!(store.delete_node("a"));
        let incoming = store.query_neighbors("b", None, Direction::Incoming, 10);
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0].1.node_id, "c");
    }

    #[test]
    fn test_incoming_with_edge_type_filter() {
        let mut store = InMemoryGraphStore::new(Some("test"));
        store.add_node("N", HashMap::new(), Some("a")).unwrap();
        store.add_node("N", HashMap::new(), Some("b")).unwrap();
        store.add_edge("a", "b", "KNOWS", None).unwrap();
        store.add_edge("a", "b", "LIKES", None).unwrap();

        let knows = store.query_neighbors("b", Some("KNOWS"), Direction::Incoming, 10);
        assert_eq!(knows.len(), 1);
        assert_eq!(knows[0].0.edge_type, "KNOWS");

        let all = store.query_neighbors("b", None, Direction::Incoming, 10);
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_self_loop_edge() {
        let mut store = InMemoryGraphStore::new(Some("test"));
        store.add_node("N", HashMap::new(), Some("a")).unwrap();
        store.add_edge("a", "a", "SELF", None).unwrap();

        let outgoing = store.query_neighbors("a", None, Direction::Outgoing, 10);
        assert_eq!(outgoing.len(), 1);

        let incoming = store.query_neighbors("a", None, Direction::Incoming, 10);
        assert_eq!(incoming.len(), 1);

        let both = store.query_neighbors("a", None, Direction::Both, 10);
        assert_eq!(both.len(), 2);
    }
}
