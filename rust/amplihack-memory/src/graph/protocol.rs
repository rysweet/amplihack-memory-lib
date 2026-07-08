//! GraphStore trait -- the common interface all graph backends implement.

use std::collections::HashMap;

use super::types::{Direction, GraphEdge, GraphNode, TraversalResult};

/// Sort `nodes` in place by their `order_by` property, matching the ordering
/// contract of [`GraphStore::query_nodes_ordered`]'s default implementation.
///
/// When `numeric`, each value is parsed as `i64` (a missing or unparsable value
/// sorts as `i64::MIN`, i.e. last for a descending sort); otherwise the raw
/// string value is compared lexicographically (missing → `""`). `descending`
/// reverses the comparison. The sort is stable, so equal keys keep their input
/// order. Shared with the federated backend's global re-sort so every layer
/// orders identically.
pub(crate) fn sort_nodes_by(
    nodes: &mut [GraphNode],
    order_by: &str,
    numeric: bool,
    descending: bool,
) {
    nodes.sort_by(|a, b| {
        let ord = if numeric {
            let pa = a
                .properties
                .get(order_by)
                .and_then(|s| s.parse::<i64>().ok())
                .unwrap_or(i64::MIN);
            let pb = b
                .properties
                .get(order_by)
                .and_then(|s| s.parse::<i64>().ok())
                .unwrap_or(i64::MIN);
            pa.cmp(&pb)
        } else {
            let pa = a.properties.get(order_by).map(String::as_str).unwrap_or("");
            let pb = b.properties.get(order_by).map(String::as_str).unwrap_or("");
            pa.cmp(pb)
        };
        if descending {
            ord.reverse()
        } else {
            ord
        }
    });
}

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

    /// Ordered, filtered read: the backend orders matching rows by `order_by`
    /// and **then** truncates to `limit`, so `LIMIT` always applies *after* the
    /// sort.
    ///
    /// This is the primitive that fixes the prospective-memory truncation bug
    /// (issue #124): [`query_nodes`](Self::query_nodes) emits a bare
    /// `RETURN n LIMIT k` with no `ORDER BY`, so a large store truncates to an
    /// arbitrary window *before* any priority sort, silently dropping
    /// lower-priority rows. `query_nodes_ordered` pushes both the filter and the
    /// ordering into the query so the top-`limit` result is deterministic.
    ///
    /// Unlike `query_nodes` — which is infallible and swallows a backend read
    /// error as an empty result — this **propagates** a read failure (or an
    /// invalid `order_by`) as `Err`, so a populated-but-unreadable store never
    /// masquerades as an empty result (the issue #124 false-empty class).
    ///
    /// * `order_by` — property to order by. Must be a valid identifier
    ///   (`^[A-Za-z_][A-Za-z0-9_]*$`); an invalid value is rejected with
    ///   `Err(MemoryError::InvalidInput)` before any query runs, since it is
    ///   spliced into the `ORDER BY` clause by the database backends.
    /// * `numeric` — when `true`, order by the value parsed/cast as `INT64`
    ///   (required for string-stored numbers like `priority`, so `10` outranks
    ///   `9`); when `false`, order lexicographically.
    /// * `descending` — `true` for `DESC` (highest first), `false` for `ASC`.
    ///
    /// The default implementation serves the in-memory / composite backends that
    /// cannot fail on a read: it fetches all matching rows, sorts them in Rust
    /// (parsing to `i64` when `numeric`, with a missing/unparsable value sorting
    /// as `i64::MIN`), applies `descending`, then truncates to `limit`. The
    /// durable database backends override it to push `ORDER BY … LIMIT` into
    /// their query engine.
    fn query_nodes_ordered(
        &self,
        node_type: &str,
        filters: Option<&HashMap<String, String>>,
        order_by: &str,
        numeric: bool,
        descending: bool,
        limit: usize,
    ) -> crate::Result<Vec<GraphNode>> {
        if !crate::utils::is_valid_identifier(order_by) {
            return Err(crate::MemoryError::InvalidInput(format!(
                "invalid order_by column {order_by:?}: must match [A-Za-z_][A-Za-z0-9_]*"
            )));
        }
        let mut nodes = self.query_nodes(node_type, filters, usize::MAX);
        sort_nodes_by(&mut nodes, order_by, numeric, descending);
        nodes.truncate(limit);
        Ok(nodes)
    }

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

    /// Return every *live* directed edge of `edge_type` as `(source, target)`
    /// node pairs, in one store-locked scan.
    ///
    /// This is the fast path for the ranked-recall graph-proximity term: instead
    /// of issuing one [`query_neighbors`](Self::query_neighbors) call per
    /// candidate node per hop (an N+1 fan-out that made Simard's OODA
    /// prepare-context spin for ~11 min at ~7,590 facts, issue #40), the ranker
    /// loads all edges of the two traversed types once and runs its bounded BFS
    /// against an in-memory adjacency index built from the result.
    ///
    /// Returns `None` if the backend does not implement the fast bulk path; the
    /// caller then falls back to the legacy per-node neighbour queries with
    /// identical results (a performance-only fallback, never a correctness or
    /// isolation downgrade). The provided default is `None`, so every existing
    /// backend compiles and behaves unchanged until it opts in.
    ///
    /// Contract for overrides:
    /// - Include only **live** edges: apply the same tombstone / not-deleted
    ///   filter used elsewhere. Soft-deleted, superseded, or archived edges MUST
    ///   NOT appear.
    /// - Preserve direction: the pair is `(source, target)`, matching the
    ///   directed semantics of the per-node neighbour queries it replaces.
    /// - Do **not** apply tenant scoping here. Per-agent isolation is re-applied
    ///   by the recall layer on every endpoint and every BFS hop.
    /// - An `edge_type` with no matching edges is `Some(vec![])` (the capability
    ///   is supported; there simply are no such edges), never `None`.
    fn bulk_edges_of_type(&self, _edge_type: &str) -> Option<Vec<(GraphNode, GraphNode)>> {
        None
    }

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
