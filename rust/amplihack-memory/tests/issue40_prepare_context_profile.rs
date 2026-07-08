//! Profiling harness for the graph-adjacency-index fix (issue #40).
//!
//! Attributes the ranked-recall graph term's cost to per-node neighbour
//! round-trips and shows the before/after when those are served from a
//! single bulk adjacency scan per edge type instead.
//!
//! `#[ignore]` so it never runs in normal CI (it is a measurement, not an
//! assertion — the durable regression is the round-trip *count* test in
//! `cognitive_memory::tests::ranked_perf_tests`). Run it on demand:
//!
//! ```bash
//! cargo test -p amplihack-memory --test issue40_prepare_context_profile \
//!   -- --ignored --nocapture
//! ```
//!
//! The `ToggleBulkStore` wrapper lets one process measure both paths against
//! the identical fixture: with `support_bulk = false` the recall takes the
//! legacy per-node `query_neighbors` fan-out; with `true` it takes the indexed
//! path. The in-memory backend does not reproduce the persistent backend's
//! per-call Cypher-scan + lock latency (which is what turned this into ~11 min
//! on the daemon), but it does expose the *round-trip count* — the thing that
//! scales with fact count — and the relative wall-clock the fix removes.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use amplihack_memory::{
    CognitiveMemory, Direction, GraphEdge, GraphNode, GraphStore, InMemoryGraphStore, RecallOptions,
};

struct ToggleBulkStore {
    inner: Box<dyn GraphStore + Send>,
    support_bulk: bool,
    query_neighbors: Arc<AtomicUsize>,
    bulk_edges: Arc<AtomicUsize>,
}

impl ToggleBulkStore {
    fn new(
        inner: Box<dyn GraphStore + Send>,
        support_bulk: bool,
    ) -> (Self, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let qn = Arc::new(AtomicUsize::new(0));
        let be = Arc::new(AtomicUsize::new(0));
        (
            Self {
                inner,
                support_bulk,
                query_neighbors: qn.clone(),
                bulk_edges: be.clone(),
            },
            qn,
            be,
        )
    }

    fn in_memory(support_bulk: bool) -> (Self, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        Self::new(
            Box::new(InMemoryGraphStore::new(Some("profile"))),
            support_bulk,
        )
    }
}

impl GraphStore for ToggleBulkStore {
    fn store_id(&self) -> &str {
        self.inner.store_id()
    }
    fn add_node(
        &mut self,
        node_type: &str,
        properties: HashMap<String, String>,
        node_id: Option<&str>,
    ) -> amplihack_memory::Result<GraphNode> {
        self.inner.add_node(node_type, properties, node_id)
    }
    fn get_node(&self, node_id: &str) -> Option<GraphNode> {
        self.inner.get_node(node_id)
    }
    fn query_nodes(
        &self,
        node_type: &str,
        filters: Option<&HashMap<String, String>>,
        limit: usize,
    ) -> Vec<GraphNode> {
        self.inner.query_nodes(node_type, filters, limit)
    }
    fn search_nodes(
        &self,
        node_type: &str,
        text_fields: &[String],
        query: &str,
        filters: Option<&HashMap<String, String>>,
        limit: usize,
    ) -> Vec<GraphNode> {
        self.inner
            .search_nodes(node_type, text_fields, query, filters, limit)
    }
    fn update_node(&mut self, node_id: &str, properties: HashMap<String, String>) -> bool {
        self.inner.update_node(node_id, properties)
    }
    fn delete_node(&mut self, node_id: &str) -> bool {
        self.inner.delete_node(node_id)
    }
    fn add_edge(
        &mut self,
        source_id: &str,
        target_id: &str,
        edge_type: &str,
        properties: Option<HashMap<String, String>>,
    ) -> amplihack_memory::Result<GraphEdge> {
        self.inner
            .add_edge(source_id, target_id, edge_type, properties)
    }
    fn query_neighbors(
        &self,
        node_id: &str,
        edge_type: Option<&str>,
        direction: Direction,
        limit: usize,
    ) -> Vec<(GraphEdge, GraphNode)> {
        self.query_neighbors.fetch_add(1, Ordering::SeqCst);
        self.inner
            .query_neighbors(node_id, edge_type, direction, limit)
    }
    fn bulk_edges_of_type(&self, edge_type: &str) -> Option<Vec<(GraphNode, GraphNode)>> {
        if !self.support_bulk {
            return None;
        }
        self.bulk_edges.fetch_add(1, Ordering::SeqCst);
        self.inner.bulk_edges_of_type(edge_type)
    }
    fn delete_edge(&mut self, source_id: &str, target_id: &str, edge_type: &str) -> bool {
        self.inner.delete_edge(source_id, target_id, edge_type)
    }
    fn traverse(
        &self,
        start_id: &str,
        edge_types: Option<&[String]>,
        max_hops: usize,
        direction: Direction,
        node_filter: Option<&HashMap<String, String>>,
    ) -> amplihack_memory::TraversalResult {
        self.inner
            .traverse(start_id, edge_types, max_hops, direction, node_filter)
    }
    fn close(&mut self) {
        self.inner.close();
    }
}

fn seed(cm: &mut CognitiveMemory, n_facts: usize) {
    let mut ids = Vec::with_capacity(n_facts);
    for i in 0..n_facts {
        let id = cm
            .store_fact(
                &format!("concept-{i}"),
                &format!("rust async runtime fact number {i}"),
                0.5,
                "",
                None,
                None,
            )
            .unwrap();
        ids.push(id);
    }
    for pair in ids.windows(2) {
        cm.link_similar_facts(&pair[0], &pair[1], 0.5).unwrap();
    }
}

fn run(support_bulk: bool, n_facts: usize, hops: usize) -> (usize, usize, u128) {
    let (store, qn, be) = ToggleBulkStore::in_memory(support_bulk);
    run_with_store(store, qn, be, n_facts, hops)
}

fn run_with_store(
    store: ToggleBulkStore,
    qn: Arc<AtomicUsize>,
    be: Arc<AtomicUsize>,
    n_facts: usize,
    hops: usize,
) -> (usize, usize, u128) {
    let mut cm = CognitiveMemory::with_store("profile-agent", Box::new(store)).unwrap();
    seed(&mut cm, n_facts);

    qn.store(0, Ordering::SeqCst);
    be.store(0, Ordering::SeqCst);
    let opts = RecallOptions {
        max_graph_hops: hops,
        record_access: false,
        ..RecallOptions::default()
    };
    let start = Instant::now();
    let _ = cm.recall_facts_ranked("rust async", opts).unwrap();
    let elapsed = start.elapsed().as_micros();
    (
        qn.load(Ordering::SeqCst),
        be.load(Ordering::SeqCst),
        elapsed,
    )
}

#[test]
#[ignore = "profiling harness — run with --ignored --nocapture"]
fn profile_prepare_context_graph_term() {
    // ~7590 = the live cognitive-memory fact count from the issue-#40 report.
    let n_facts = 7_590;
    let hops = 1;

    let (legacy_qn, legacy_be, legacy_us) = run(false, n_facts, hops);
    let (indexed_qn, indexed_be, indexed_us) = run(true, n_facts, hops);

    println!("\n=== issue #40 prepare-context graph-term profile ===");
    println!("facts = {n_facts}, max_graph_hops = {hops}\n");
    println!(
        "{:<10} {:>18} {:>12} {:>14}",
        "path", "query_neighbors", "bulk_edges", "recall_wall"
    );
    println!(
        "{:<10} {:>18} {:>12} {:>11}ms",
        "legacy",
        legacy_qn,
        legacy_be,
        legacy_us / 1000
    );
    println!(
        "{:<10} {:>18} {:>12} {:>11}ms",
        "indexed",
        indexed_qn,
        indexed_be,
        indexed_us / 1000
    );
    let speedup = if indexed_us == 0 {
        f64::INFINITY
    } else {
        legacy_us as f64 / indexed_us as f64
    };
    println!(
        "\nround-trip reduction: {legacy_qn} -> {indexed_qn} per-node neighbour scans \
         (bulk scans: {indexed_be})"
    );
    println!("in-memory wall-clock speedup: {speedup:.1}x\n");

    // Sanity: the indexed path must eliminate the per-node fan-out.
    assert_eq!(indexed_qn, 0, "indexed path must issue zero per-node scans");
    assert!(
        legacy_qn > indexed_qn,
        "legacy path must fan out per-node scans"
    );
}

/// Representative wall-clock on the **real** persistent (LadybugDB) backend,
/// where each `query_neighbors` is a lock-serialized Cypher scan — the cost the
/// in-memory harness above cannot model. This is the profile that reproduces
/// the ~11-min pathology in miniature: legacy fans a Cypher scan out per node,
/// the indexed path issues two typed scans total.
///
/// `n_facts` is kept modest so the *legacy* run stays a few minutes rather than
/// the ~11 min the daemon saw at ~7,590 facts; the wall-clock ratio is the
/// point, not the absolute floor.
#[cfg(feature = "persistent")]
#[test]
#[ignore = "persistent profiling harness — run with --features persistent --ignored --nocapture"]
fn profile_prepare_context_graph_term_persistent() {
    use amplihack_memory::graph::LbugGraphStore;

    let n_facts = 1_500;
    let hops = 1;

    let legacy_dir = tempfile::tempdir().unwrap();
    let legacy_lbug =
        LbugGraphStore::open(&legacy_dir.path().join("legacy.ladybug"), Some("legacy")).unwrap();
    let (legacy_store, legacy_qn_c, legacy_be_c) =
        ToggleBulkStore::new(Box::new(legacy_lbug), false);
    let (legacy_qn, legacy_be, legacy_us) =
        run_with_store(legacy_store, legacy_qn_c, legacy_be_c, n_facts, hops);

    let indexed_dir = tempfile::tempdir().unwrap();
    let indexed_lbug =
        LbugGraphStore::open(&indexed_dir.path().join("indexed.ladybug"), Some("indexed")).unwrap();
    let (indexed_store, indexed_qn_c, indexed_be_c) =
        ToggleBulkStore::new(Box::new(indexed_lbug), true);
    let (indexed_qn, indexed_be, indexed_us) =
        run_with_store(indexed_store, indexed_qn_c, indexed_be_c, n_facts, hops);

    println!("\n=== issue #40 prepare-context graph-term profile (PERSISTENT/lbug) ===");
    println!("facts = {n_facts}, max_graph_hops = {hops}\n");
    println!(
        "{:<10} {:>18} {:>12} {:>14}",
        "path", "query_neighbors", "bulk_edges", "recall_wall"
    );
    println!(
        "{:<10} {:>18} {:>12} {:>11}ms",
        "legacy",
        legacy_qn,
        legacy_be,
        legacy_us / 1000
    );
    println!(
        "{:<10} {:>18} {:>12} {:>11}ms",
        "indexed",
        indexed_qn,
        indexed_be,
        indexed_us / 1000
    );
    let speedup = if indexed_us == 0 {
        f64::INFINITY
    } else {
        legacy_us as f64 / indexed_us as f64
    };
    println!("\npersistent wall-clock speedup: {speedup:.1}x\n");

    assert_eq!(indexed_qn, 0, "indexed path must issue zero per-node scans");
}
