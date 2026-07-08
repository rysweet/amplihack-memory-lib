//! TDD specification tests for the **graph-adjacency-index** performance fix
//! (Simard OODA prepare-context pathology, issue #40).
//!
//! ## What is being fixed
//!
//! `recall_facts_ranked` / `recall_episodes_ranked` compute a graph-proximity
//! term by running a bounded BFS from *every* candidate node. Today that BFS
//! calls [`GraphStore::query_neighbors`] once per node per traversal-edge-type
//! per hop — an **N+1 fan-out** of roughly `3 * N` neighbor round-trips per
//! recall (each a lock-serialized Cypher scan on the persistent backend). At
//! ~7590 facts this makes a single prepare-context call spin for ~11 minutes at
//! ~300% CPU. The output is tiny; the cost is entirely in the per-node scan
//! fan-out.
//!
//! ## The contract these tests pin down
//!
//! The fix adds one additive `GraphStore` capability —
//!
//! ```ignore
//! fn bulk_edges_of_type(&self, edge_type: &str) -> Option<Vec<(GraphNode, GraphNode)>> {
//!     None // default: legacy per-node path preserved
//! }
//! ```
//!
//! — overridden by `InMemoryGraphStore` and `LbugGraphStore`, and refactors the
//! ranked-recall BFS to build a **per-recall in-memory adjacency index** from a
//! small constant number of bulk edge scans (one per traversal edge type:
//! `DERIVES_FROM` + `SIMILAR_TO`) instead of `~3N` per-node `query_neighbors`
//! calls. The in-memory BFS must reproduce the legacy path **byte-for-byte**:
//! same descending-score ordering, same `reasons`, same per-hop `/hop`
//! discount, same per-agent tenant prune on **every** hop, same tombstone /
//! lifecycle exclusion.
//!
//! ## Why this module is RED before the fix
//!
//! It references [`GraphStore::bulk_edges_of_type`], which does not exist yet,
//! so the module fails to compile — the expected TDD "red" state (matching the
//! house style already used by `ranked_tests.rs`). Once the trait method +
//! overrides + BFS refactor land, every test here must go green:
//!
//! * the round-trip **counting** tests prove the neighbor fan-out is gone and
//!   the number of store scans is a small constant **independent of fact
//!   count** (no wall-clock timing — a deterministic call-count regression, per
//!   policy: no wall-clock timeouts on agentic/perf assertions);
//! * the **parity** tests prove the bulk-index BFS still surfaces the same
//!   graph-boosted items across multiple hops; and
//! * the **tenant-isolation** test proves the store-wide bulk scan never leaks
//!   a foreign agent's edges into ranking, even mid-path.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// `CognitiveMemory`, `RecallOptions`, `RecallWeights`, `Scored`, ... .
use super::super::*;

use crate::graph::types::TraversalResult;
use crate::graph::{Direction, GraphEdge, GraphNode, GraphStore, InMemoryGraphStore};
use crate::memory_types::SemanticFact;

// Schema labels, mirrored as local literals so the test does not depend on the
// crate-private constants. Values match `cognitive_memory::types`.
const NT_SEMANTIC: &str = "SemanticMemory";
const ET_SIMILAR_TO: &str = "SIMILAR_TO";
const ET_DERIVES_FROM: &str = "DERIVES_FROM";

// ===========================================================================
// Counting decorator: attributes store round-trips without any wall clock.
// ===========================================================================

/// Snapshot handle over a [`CountingGraphStore`]'s counters.
///
/// Cloned out before the store is moved into a [`CognitiveMemory`] so a test
/// can `reset()` after seeding and then read exactly the round-trips a single
/// recall issued.
#[derive(Clone)]
struct Counters {
    query_nodes: Arc<AtomicUsize>,
    query_neighbors: Arc<AtomicUsize>,
    bulk_edges: Arc<AtomicUsize>,
}

impl Counters {
    fn reset(&self) {
        self.query_nodes.store(0, Ordering::SeqCst);
        self.query_neighbors.store(0, Ordering::SeqCst);
        self.bulk_edges.store(0, Ordering::SeqCst);
    }
    fn query_nodes(&self) -> usize {
        self.query_nodes.load(Ordering::SeqCst)
    }
    fn query_neighbors(&self) -> usize {
        self.query_neighbors.load(Ordering::SeqCst)
    }
    fn bulk_edges(&self) -> usize {
        self.bulk_edges.load(Ordering::SeqCst)
    }
}

/// `GraphStore` decorator that counts the round-trips relevant to the fix:
/// full-store candidate scans (`query_nodes`), per-node neighbor fan-out
/// (`query_neighbors`), and bulk adjacency scans (`bulk_edges_of_type`). Every
/// other operation is a transparent delegate to an inner [`InMemoryGraphStore`].
struct CountingGraphStore {
    inner: InMemoryGraphStore,
    counters: Counters,
}

impl CountingGraphStore {
    fn new() -> (Self, Counters) {
        let counters = Counters {
            query_nodes: Arc::new(AtomicUsize::new(0)),
            query_neighbors: Arc::new(AtomicUsize::new(0)),
            bulk_edges: Arc::new(AtomicUsize::new(0)),
        };
        let store = Self {
            inner: InMemoryGraphStore::new(Some("counting")),
            counters: counters.clone(),
        };
        (store, counters)
    }
}

impl GraphStore for CountingGraphStore {
    fn store_id(&self) -> &str {
        self.inner.store_id()
    }

    fn add_node(
        &mut self,
        node_type: &str,
        properties: HashMap<String, String>,
        node_id: Option<&str>,
    ) -> crate::Result<GraphNode> {
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
        self.counters.query_nodes.fetch_add(1, Ordering::SeqCst);
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
    ) -> crate::Result<GraphEdge> {
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
        self.counters.query_neighbors.fetch_add(1, Ordering::SeqCst);
        self.inner
            .query_neighbors(node_id, edge_type, direction, limit)
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
    ) -> TraversalResult {
        self.inner
            .traverse(start_id, edge_types, max_hops, direction, node_filter)
    }

    /// The additive capability under test. Counts each bulk adjacency scan and
    /// delegates to the inner store's override. Until the trait declares this
    /// method (and `InMemoryGraphStore` overrides it) the whole module fails to
    /// compile — the intended TDD red state.
    fn bulk_edges_of_type(&self, edge_type: &str) -> Option<Vec<(GraphNode, GraphNode)>> {
        self.counters.bulk_edges.fetch_add(1, Ordering::SeqCst);
        self.inner.bulk_edges_of_type(edge_type)
    }

    fn close(&mut self) {
        self.inner.close();
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Read-only recall options with the graph term enabled at one hop.
fn graph_opts(max_graph_hops: usize) -> RecallOptions {
    RecallOptions {
        max_graph_hops,
        record_access: false,
        ..RecallOptions::default()
    }
}

/// Build a `CognitiveMemory` over a counting store, seeded with `n_facts`
/// query-matching facts chained together with `SIMILAR_TO` edges (so the legacy
/// per-node path is forced to fan out neighbor scans across the whole store).
fn seeded_cm(n_facts: usize) -> (CognitiveMemory, Counters) {
    let (store, counters) = CountingGraphStore::new();
    let mut cm = CognitiveMemory::with_store("perf-agent", Box::new(store)).unwrap();

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
    // A SIMILAR_TO chain guarantees every node has neighbors, so the legacy
    // path's `query_neighbors` count genuinely scales with `n_facts`.
    for pair in ids.windows(2) {
        cm.link_similar_facts(&pair[0], &pair[1], 0.5).unwrap();
    }

    (cm, counters)
}

fn find_fact<'a>(
    scored: &'a [Scored<SemanticFact>],
    node_id: &str,
) -> Option<&'a Scored<SemanticFact>> {
    scored.iter().find(|s| s.item.node_id == node_id)
}

fn has_reason(reasons: &[String], needle: &str) -> bool {
    let needle = needle.to_lowercase();
    reasons.iter().any(|r| r.to_lowercase().contains(&needle))
}

/// Node property bag stamped with an `agent_id` so the recall's tenant filter
/// treats it as owned by `agent`.
fn owned_props(concept: &str, content: &str, agent: &str) -> HashMap<String, String> {
    let mut p = HashMap::new();
    p.insert("agent_id".to_string(), agent.to_string());
    p.insert("concept".to_string(), concept.to_string());
    p.insert("content".to_string(), content.to_string());
    p
}

// ===========================================================================
// 1. Round-trip counting regression — the core performance contract
// ===========================================================================

#[test]
fn fact_recall_eliminates_per_node_neighbor_fanout() {
    // The graph term must be served from a bulk adjacency index, not per-node
    // neighbor scans. With the graph term enabled, a single recall issues:
    //   * exactly ONE candidate scan (`query_nodes`),
    //   * ZERO per-node `query_neighbors` calls, and
    //   * a small constant number of bulk adjacency scans (one per traversal
    //     edge type: DERIVES_FROM + SIMILAR_TO = 2).
    let (mut cm, counters) = seeded_cm(200);
    counters.reset();

    let _ = cm.recall_facts_ranked("rust async", graph_opts(1)).unwrap();

    assert_eq!(
        counters.query_nodes(),
        1,
        "recall must fetch candidates with exactly one full-store scan"
    );
    assert_eq!(
        counters.query_neighbors(),
        0,
        "the per-node neighbor fan-out must be gone (served from the index)"
    );
    assert_eq!(
        counters.bulk_edges(),
        2,
        "the adjacency index is built from one bulk scan per traversal edge \
         type (DERIVES_FROM + SIMILAR_TO)"
    );
}

#[test]
fn fact_recall_roundtrips_do_not_scale_with_fact_count() {
    // The load-bearing invariant: store round-trips are CONSTANT in the number
    // of facts. Legacy code's `query_neighbors` count grows ~3N; the fixed code
    // must issue the identical (small, constant) number of scans at 20 facts
    // and at 400 facts.
    let (mut small_cm, small) = seeded_cm(20);
    let (mut large_cm, large) = seeded_cm(400);

    small.reset();
    let _ = small_cm
        .recall_facts_ranked("rust async", graph_opts(2))
        .unwrap();

    large.reset();
    let _ = large_cm
        .recall_facts_ranked("rust async", graph_opts(2))
        .unwrap();

    assert_eq!(
        small.query_nodes(),
        large.query_nodes(),
        "candidate-scan count must not depend on fact count"
    );
    assert_eq!(
        small.query_neighbors(),
        large.query_neighbors(),
        "neighbor round-trips must not depend on fact count"
    );
    assert_eq!(
        small.bulk_edges(),
        large.bulk_edges(),
        "bulk adjacency scans must not depend on fact count (multi-hop BFS \
         still builds the index exactly once)"
    );
    assert_eq!(
        large.query_neighbors(),
        0,
        "no per-node neighbor scans even at 400 facts across 2 hops"
    );
}

#[test]
fn fact_recall_without_graph_term_issues_no_adjacency_scans() {
    // hops=0 disables the graph term entirely: neither per-node neighbor scans
    // nor bulk adjacency scans should happen.
    let (mut cm, counters) = seeded_cm(100);
    counters.reset();

    let _ = cm.recall_facts_ranked("rust async", graph_opts(0)).unwrap();

    assert_eq!(counters.query_neighbors(), 0);
    assert_eq!(
        counters.bulk_edges(),
        0,
        "no adjacency index is needed when the graph term is disabled"
    );
}

#[test]
fn episode_recall_eliminates_per_node_neighbor_fanout() {
    // Episodes take the same index path: DERIVES_FROM (incoming) + SIMILAR_TO
    // (both) => 2 bulk scans, 0 per-node neighbor scans.
    let (store, counters) = CountingGraphStore::new();
    let mut cm = CognitiveMemory::with_store("perf-agent", Box::new(store)).unwrap();

    let mut ids = Vec::new();
    for i in 0..150 {
        let id = cm
            .store_episode(
                &format!("rust async episode {i}"),
                "ops",
                Some(i as i64),
                None,
            )
            .unwrap();
        ids.push(id);
    }
    for pair in ids.windows(2) {
        cm.link_similar_facts(&pair[0], &pair[1], 0.5).unwrap();
    }

    counters.reset();
    let _ = cm
        .recall_episodes_ranked("rust async", graph_opts(1))
        .unwrap();

    assert_eq!(
        counters.query_neighbors(),
        0,
        "episode recall must not per-node query_neighbors either"
    );
    assert_eq!(
        counters.bulk_edges(),
        2,
        "episode recall builds the index from DERIVES_FROM + SIMILAR_TO"
    );
}

// ===========================================================================
// 2. `bulk_edges_of_type` correctness on the in-memory backend
// ===========================================================================

#[test]
fn in_memory_bulk_edges_of_type_returns_live_directed_pairs() {
    let mut store = InMemoryGraphStore::new(Some("bulk"));
    store
        .add_node(NT_SEMANTIC, owned_props("a", "alpha", "me"), Some("a"))
        .unwrap();
    store
        .add_node(NT_SEMANTIC, owned_props("b", "beta", "me"), Some("b"))
        .unwrap();
    store
        .add_node(NT_SEMANTIC, owned_props("c", "gamma", "me"), Some("c"))
        .unwrap();

    store.add_edge("a", "b", ET_SIMILAR_TO, None).unwrap();
    store.add_edge("b", "c", ET_SIMILAR_TO, None).unwrap();
    store.add_edge("a", "c", ET_DERIVES_FROM, None).unwrap();

    // The in-memory backend overrides the capability -> `Some`.
    let sim = store
        .bulk_edges_of_type(ET_SIMILAR_TO)
        .expect("InMemoryGraphStore overrides bulk_edges_of_type");
    assert_eq!(sim.len(), 2, "two SIMILAR_TO edges");
    // Pairs are directed (source, target).
    assert!(
        sim.iter()
            .any(|(s, t)| s.node_id == "a" && t.node_id == "b"),
        "must contain a -> b"
    );
    assert!(
        sim.iter()
            .any(|(s, t)| s.node_id == "b" && t.node_id == "c"),
        "must contain b -> c"
    );

    let der = store
        .bulk_edges_of_type(ET_DERIVES_FROM)
        .expect("Some for DERIVES_FROM");
    assert_eq!(der.len(), 1);
    assert_eq!(der[0].0.node_id, "a", "DERIVES_FROM source");
    assert_eq!(der[0].1.node_id, "c", "DERIVES_FROM target");

    // An edge type with no edges is an empty vector, NOT `None` (the backend
    // supports the capability; there simply are no such edges).
    let empty = store
        .bulk_edges_of_type("NO_SUCH_EDGE")
        .expect("Some even when there are no matching edges");
    assert!(empty.is_empty());
}

#[test]
fn in_memory_bulk_edges_of_type_excludes_deleted_edges() {
    // Tombstone / lifecycle parity: an edge removed from the store must not
    // resurface through the bulk scan (the index must never rank stale links).
    let mut store = InMemoryGraphStore::new(Some("bulk-del"));
    store
        .add_node(NT_SEMANTIC, owned_props("a", "alpha", "me"), Some("a"))
        .unwrap();
    store
        .add_node(NT_SEMANTIC, owned_props("b", "beta", "me"), Some("b"))
        .unwrap();
    store.add_edge("a", "b", ET_SIMILAR_TO, None).unwrap();

    assert_eq!(store.bulk_edges_of_type(ET_SIMILAR_TO).unwrap().len(), 1);
    assert!(store.delete_edge("a", "b", ET_SIMILAR_TO));
    assert!(
        store.bulk_edges_of_type(ET_SIMILAR_TO).unwrap().is_empty(),
        "a deleted edge must not surface through the bulk scan"
    );
}

// ===========================================================================
// 3. Behavioral parity — the index BFS must match the legacy neighbor path
// ===========================================================================

#[test]
fn two_hop_similar_chain_boosts_via_index() {
    // core matches the query; mid is SIMILAR_TO core (1 hop from core); far is
    // SIMILAR_TO mid (2 hops from core). With max_graph_hops>=2, `far` gets a
    // graph boost through the two-hop chain (discounted by /hop) while an
    // isolated node with identical text gets none. This proves the in-memory
    // BFS traverses multiple hops over the bulk index, exactly like the legacy
    // per-node BFS.
    let mut cm = CognitiveMemory::new("parity-agent").unwrap();
    let core = cm
        .store_fact("core", "rust async tokio runtime", 0.0, "", None, None)
        .unwrap();
    let mid = cm
        .store_fact("mid", "totally unrelated middle text", 0.0, "", None, None)
        .unwrap();
    let far = cm
        .store_fact("far", "banana smoothie recipe", 0.0, "", None, None)
        .unwrap();
    let isolated = cm
        .store_fact("iso", "banana smoothie recipe", 0.0, "", None, None)
        .unwrap();
    cm.link_similar_facts(&mid, &core, 0.9).unwrap();
    cm.link_similar_facts(&far, &mid, 0.9).unwrap();

    let res = cm.recall_facts_ranked("rust async", graph_opts(2)).unwrap();
    let far_s = find_fact(&res, &far).unwrap();
    let iso_s = find_fact(&res, &isolated).unwrap();

    assert!(
        far_s.score > iso_s.score,
        "two-hop SIMILAR_TO neighbor must outrank an identical isolated node: \
         {} !> {}",
        far_s.score,
        iso_s.score
    );
    assert!(
        has_reason(&far_s.reasons, "graph"),
        "the two-hop boost must be attributed with a graph reason"
    );
    assert!(
        !has_reason(&iso_s.reasons, "graph"),
        "an isolated node must have no graph reason"
    );
}

#[test]
fn foreign_mid_path_node_does_not_propagate_graph_boost_on_index_path() {
    // Multi-tenant isolation over the STORE-WIDE bulk scan (the critical new
    // risk): anchor (no text match) --SIMILAR_TO--> foreign(intruder)
    // --SIMILAR_TO--> rich(strong match). Even at 2 hops, anchor must NOT be
    // boosted by `rich`, because the only path runs THROUGH a foreign-tenant
    // node, which the per-hop tenant prune must block — the index BFS must not
    // traverse foreign nodes, matching the legacy path.
    let mut cm = CognitiveMemory::new("tenant-agent").unwrap();
    let anchor = cm
        .store_fact("anchor", "banana smoothie recipe", 0.0, "", None, None)
        .unwrap();
    let bridge = cm
        .store_fact("bridge", "neutral bridging node", 0.0, "", None, None)
        .unwrap();
    let rich = cm
        .store_fact("rich", "rust async rust async tokio", 0.0, "", None, None)
        .unwrap();
    cm.link_similar_facts(&anchor, &bridge, 0.9).unwrap();
    cm.link_similar_facts(&bridge, &rich, 0.9).unwrap();

    // Baseline: everything owned -> anchor reaches `rich` at 2 hops and is
    // boosted.
    let owned = cm.recall_facts_ranked("rust async", graph_opts(2)).unwrap();
    let anchor_owned = find_fact(&owned, &anchor).unwrap();
    assert!(
        has_reason(&anchor_owned.reasons, "graph"),
        "owned two-hop path should boost anchor"
    );
    let owned_score = anchor_owned.score;

    // Make the middle node foreign: the path is now severed by the tenant prune.
    let mut foreign = HashMap::new();
    foreign.insert("agent_id".to_string(), "intruder".to_string());
    assert!(cm.set_node_props_for_test(&bridge, foreign));

    let after = cm.recall_facts_ranked("rust async", graph_opts(2)).unwrap();
    let anchor_after = find_fact(&after, &anchor).unwrap();
    assert!(
        !has_reason(&anchor_after.reasons, "graph"),
        "a foreign mid-path node must block boost propagation on the index path"
    );
    assert!(
        anchor_after.score < owned_score,
        "severing the path through the foreign node must lower anchor's score: \
         {} !< {}",
        anchor_after.score,
        owned_score
    );
}

// ===========================================================================
// 4. Persistent (LadybugDB) backend smoke test
// ===========================================================================

#[cfg(feature = "persistent")]
#[test]
fn lbug_bulk_edges_of_type_returns_typed_edges() {
    use crate::graph::LbugGraphStore;

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("bulk.ladybug");
    let mut store = LbugGraphStore::open(&path, Some("bulk-smoke")).unwrap();

    store
        .add_node(NT_SEMANTIC, owned_props("a", "alpha", "me"), Some("a"))
        .unwrap();
    store
        .add_node(NT_SEMANTIC, owned_props("b", "beta", "me"), Some("b"))
        .unwrap();
    store
        .add_node(NT_SEMANTIC, owned_props("c", "gamma", "me"), Some("c"))
        .unwrap();
    store.add_edge("a", "b", ET_SIMILAR_TO, None).unwrap();
    store.add_edge("b", "c", ET_SIMILAR_TO, None).unwrap();

    let sim = store
        .bulk_edges_of_type(ET_SIMILAR_TO)
        .expect("LbugGraphStore overrides bulk_edges_of_type");
    assert_eq!(
        sim.len(),
        2,
        "both SIMILAR_TO edges returned by one typed scan"
    );
    assert!(sim
        .iter()
        .any(|(s, t)| s.node_id == "a" && t.node_id == "b"));
    assert!(sim
        .iter()
        .any(|(s, t)| s.node_id == "b" && t.node_id == "c"));

    // No DERIVES_FROM edges exist -> an empty (but present) result.
    let der = store
        .bulk_edges_of_type(ET_DERIVES_FROM)
        .expect("Some even with no matching edges");
    assert!(der.is_empty());

    store.close();
}

#[cfg(feature = "persistent")]
#[test]
fn persistent_ranked_recall_graph_boost_survives_the_index_refactor() {
    // End-to-end parity on the durable backend: a SIMILAR_TO neighbor still
    // lifts a non-matching fact above an unlinked one after the index refactor.
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("recall.ladybug");
    let mut cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();

    let core = cm
        .store_fact("core", "rust async tokio runtime", 0.0, "", None, None)
        .unwrap();
    let neighbor = cm
        .store_fact("cooking", "banana smoothie recipe", 0.0, "", None, None)
        .unwrap();
    let unlinked = cm
        .store_fact("other", "banana smoothie recipe", 0.0, "", None, None)
        .unwrap();
    cm.link_similar_facts(&neighbor, &core, 0.9).unwrap();

    let res = cm.recall_facts_ranked("rust async", graph_opts(1)).unwrap();
    let neighbor_s = find_fact(&res, &neighbor).unwrap();
    let unlinked_s = find_fact(&res, &unlinked).unwrap();
    assert!(
        neighbor_s.score > unlinked_s.score,
        "SIMILAR_TO-linked fact must outrank the unlinked identical-text fact"
    );
    assert!(has_reason(&neighbor_s.reasons, "graph"));

    cm.close();
}
