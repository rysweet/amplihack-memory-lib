//! GraphStore trait -- the common interface all graph backends implement.

use std::collections::HashMap;

use super::types::{Direction, GraphEdge, GraphNode, TraversalResult};

/// Common interface for graph storage backends.
pub trait GraphStore {
    /// Unique identifier for this store instance.
    fn store_id(&self) -> &str;

    // -- node operations --

    /// Create a node and return it.
    fn add_node(
        &mut self,
        node_type: &str,
        properties: HashMap<String, String>,
        node_id: Option<&str>,
    ) -> crate::Result<GraphNode>;

    /// Fetch a single node by ID, or None if not found.
    fn get_node(&self, node_id: &str) -> Option<GraphNode>;

    /// Return nodes of `node_type` matching optional equality filters.
    fn query_nodes(
        &self,
        node_type: &str,
        filters: Option<&HashMap<String, String>>,
        limit: usize,
    ) -> Vec<GraphNode>;

    /// Full-text keyword search across `text_fields` using CONTAINS.
    fn search_nodes(
        &self,
        node_type: &str,
        text_fields: &[String],
        query: &str,
        filters: Option<&HashMap<String, String>>,
        limit: usize,
    ) -> Vec<GraphNode>;

    /// Update properties on an existing node. Returns true on success.
    fn update_node(&mut self, node_id: &str, properties: HashMap<String, String>) -> bool;

    /// Delete a node by ID. Returns true if the node existed.
    fn delete_node(&mut self, node_id: &str) -> bool;

    // -- edge operations --

    /// Create a directed edge between two existing nodes.
    fn add_edge(
        &mut self,
        source_id: &str,
        target_id: &str,
        edge_type: &str,
        properties: Option<HashMap<String, String>>,
    ) -> crate::Result<GraphEdge>;

    /// Return edges and neighbor nodes adjacent to `node_id`.
    fn query_neighbors(
        &self,
        node_id: &str,
        edge_type: Option<&str>,
        direction: Direction,
        limit: usize,
    ) -> Vec<(GraphEdge, GraphNode)>;

    /// Delete a specific edge. Returns true if the edge existed.
    fn delete_edge(&mut self, source_id: &str, target_id: &str, edge_type: &str) -> bool;

    // -- traversal --

    /// BFS traversal from `start_id` up to `max_hops` hops.
    fn traverse(
        &self,
        start_id: &str,
        edge_types: Option<&[String]>,
        max_hops: usize,
        direction: Direction,
        node_filter: Option<&HashMap<String, String>>,
    ) -> TraversalResult;

    /// Release resources held by the store.
    fn close(&mut self);
}
