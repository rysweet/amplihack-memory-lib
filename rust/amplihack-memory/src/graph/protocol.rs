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

    /// Fail-closed count of nodes of `node_type` matching optional equality
    /// filters.
    ///
    /// Unlike [`query_nodes`](Self::query_nodes) — which swallows a backend read
    /// error as an empty result — this **propagates** a genuine read failure as
    /// `Err`, while still reporting a genuinely-empty store as `Ok(0)`. That lets
    /// a consumer distinguish a **confirmed-empty** store from a **transient read
    /// failure** (Simard #2561: an auto-restore that trusted a swallowed
    /// all-zeros read could re-insert a snapshot on top of transiently-unreadable
    /// data and duplicate memories once reads recover).
    ///
    /// The default forwards to `query_nodes(...).len()` for backends that cannot
    /// fail (in-memory / test / composite stores); the durable LadybugDB backend
    /// overrides it to fail closed on a real read error.
    fn try_count_nodes(
        &self,
        node_type: &str,
        filters: Option<&HashMap<String, String>>,
    ) -> crate::Result<usize> {
        Ok(self.query_nodes(node_type, filters, usize::MAX).len())
    }

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
    // FUTURE: migrate to Result<bool, MemoryError>
    #[must_use]
    fn update_node(&mut self, node_id: &str, properties: HashMap<String, String>) -> bool;

    /// Delete a node by ID. Returns true if the node existed.
    // FUTURE: migrate to Result<bool, MemoryError>
    #[must_use]
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
    // FUTURE: migrate to Result<bool, MemoryError>
    #[must_use]
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

    /// Flush any write-ahead log into durable storage so a subsequent reopen
    /// needs no replay. A no-op for volatile backends (the default).
    ///
    /// # Errors
    ///
    /// Returns an error if the backend supports checkpointing but the flush
    /// fails.
    fn checkpoint(&self) -> crate::Result<()> {
        Ok(())
    }

    /// Release resources held by the store.
    fn close(&mut self);
}
