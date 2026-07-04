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

    /// Count nodes of `node_type` matching optional equality filters,
    /// **propagating read failures** instead of silently returning zero.
    ///
    /// [`query_nodes`](Self::query_nodes) returns `Vec<GraphNode>` with no error
    /// channel, so a backend whose read transiently fails has to swallow the
    /// failure to an empty result — indistinguishable from a genuinely empty
    /// store. A consumer that self-heals an *empty* store therefore cannot tell
    /// a **confirmed-empty** store (`Ok(0)`) from an **unreadable** one and may
    /// act on a false "empty".
    ///
    /// This method closes that gap: a persistent backend whose reads can fail
    /// (e.g. an lbug store hitting a transient/binder read error) overrides it to
    /// surface the failure as `Err`, so the caller can fail closed. The default
    /// delegates to `query_nodes(...).len()` and is therefore effectively
    /// infallible for volatile backends (in-memory), which never read-fail.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend can encounter a read failure and one
    /// occurs while counting. A genuinely-absent node type is **not** an error —
    /// it is confirmed-empty (`Ok(0)`).
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
