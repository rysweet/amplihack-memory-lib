use super::super::types::ET_PROCEDURE_DERIVES_FROM;
use super::super::*;
use crate::graph::Direction;
use crate::MemoryError;

fn make_cm() -> CognitiveMemory {
    CognitiveMemory::new(&format!("test-agent-{}", uuid::Uuid::new_v4())).unwrap()
}

// -- procedural memory --

#[test]
fn test_store_and_search_procedures() {
    let mut cm = make_cm();
    let steps = vec!["cargo build".into(), "cargo test".into()];
    let id = cm.store_procedure("build-rust", &steps, None).unwrap();
    assert!(id.starts_with("proc_"));

    let procs = cm.search_procedures("rust", 10);
    assert_eq!(procs.len(), 1);
    assert_eq!(procs[0].name, "build-rust");
    assert_eq!(procs[0].steps, steps);
}

#[test]
fn test_search_procedures_increments_usage() {
    let mut cm = make_cm();
    let steps = vec!["step1".into()];
    cm.store_procedure("deploy", &steps, None).unwrap();

    let procs = cm.search_procedures_mut("deploy", 10);
    assert_eq!(procs[0].usage_count, 0); // returned before increment

    // After search_procedures_mut, the stored count should be 1
    let procs2 = cm.search_procedures("deploy", 10);
    assert_eq!(procs2[0].usage_count, 1);
}

#[test]
fn test_procedure_with_prerequisites() {
    let mut cm = make_cm();
    let steps = vec!["run tests".into()];
    let prereqs = vec!["install deps".into()];
    cm.store_procedure("test-suite", &steps, Some(&prereqs))
        .unwrap();

    let procs = cm.search_procedures("test", 10);
    assert_eq!(procs[0].prerequisites, prereqs);
}

#[test]
fn test_search_procedures_empty_query_returns_all() {
    let mut cm = make_cm();
    cm.store_procedure("a", &["s".into()], None).unwrap();
    cm.store_procedure("b", &["s".into()], None).unwrap();

    let procs = cm.search_procedures("", 10);
    assert_eq!(procs.len(), 2);
}

// -- Extended coverage: search_procedures_mut --

#[test]
fn test_search_procedures_mut_multiple_increments() {
    let mut cm = make_cm();
    cm.store_procedure("build", &["compile".into()], None)
        .unwrap();

    cm.search_procedures_mut("build", 10);
    cm.search_procedures_mut("build", 10);
    cm.search_procedures_mut("build", 10);

    let procs = cm.search_procedures("build", 10);
    assert_eq!(procs[0].usage_count, 3);
}

// -- Corrected retrieval semantics (the consumer-fork bugs) --

#[test]
fn test_search_procedures_tokenized_multiword_or() {
    let mut cm = make_cm();
    cm.store_procedure(
        "release-pipeline",
        &[
            "compile sources".into(),
            "build container image".into(),
            "deploy".into(),
        ],
        None,
    )
    .unwrap();

    // Out-of-order, mixed-case, matched across name + steps via tokenized
    // CONTAINS (not a whole-string match).
    assert_eq!(
        cm.search_procedures("IMAGE compile", 10).len(),
        1,
        "tokenized procedure recall should match across name and steps"
    );

    // OR semantics: one overlapping token suffices.
    assert_eq!(
        cm.search_procedures("image nonexistent", 10).len(),
        1,
        "one overlapping token should match (OR per token)"
    );

    // No overlap -> no match.
    assert!(cm.search_procedures("kubernetes helm", 10).is_empty());
}

#[test]
fn test_store_procedure_is_idempotent_by_name() {
    let mut cm = make_cm();

    let id1 = cm
        .store_procedure("build", &["step-a".into()], None)
        .unwrap();
    let id2 = cm
        .store_procedure("build", &["step-a".into(), "step-b".into()], None)
        .unwrap();

    // Re-storing the same name reuses the node instead of inserting a duplicate.
    assert_eq!(id1, id2, "re-storing the same name must reuse the node");
    assert_eq!(
        cm.get_memory_stats().get("procedural"),
        Some(&1),
        "idempotent store must not accumulate duplicates"
    );

    // The second store updated the steps in place.
    let procs = cm.search_procedures("build", 10);
    assert_eq!(procs.len(), 1);
    assert_eq!(
        procs[0].steps,
        vec!["step-a".to_string(), "step-b".to_string()]
    );
}

// -- procedure reinforcement on recall (in-memory mirror) --

#[test]
fn test_recall_procedure_increments_and_orders_by_usage() {
    let mut cm = make_cm();
    cm.store_procedure("alpha task", &["shared work".into()], None)
        .unwrap();
    cm.store_procedure("beta task", &["shared work".into()], None)
        .unwrap();

    // Reinforce alpha via a query that matches only alpha.
    for _ in 0..3 {
        let only_alpha = cm.recall_procedure("alpha", 10);
        assert_eq!(only_alpha.len(), 1);
        assert_eq!(only_alpha[0].name, "alpha task");
    }

    // A query matching both returns them ordered by usage_count descending,
    // carrying pre-increment counts.
    let both = cm.recall_procedure("shared", 10);
    assert_eq!(both.len(), 2);
    assert_eq!(both[0].name, "alpha task");
    assert_eq!(both[0].usage_count, 3);
    assert_eq!(both[1].name, "beta task");
    assert_eq!(both[1].usage_count, 0);

    // The reinforced (post-increment) counts now drive ordering.
    let ranked = cm.search_procedures("shared", 10);
    assert_eq!(ranked[0].usage_count, 4);
    assert_eq!(ranked[1].usage_count, 1);
}

// ===========================================================================
// Provenance edges (issue #90):
//   ProceduralMemory --PROCEDURE_DERIVES_FROM--> EpisodicMemory
//
// File: rust/amplihack-memory/src/cognitive_memory/tests/procedural_tests.rs
//
// Failing-first TDD tests for the not-yet-implemented
// `store_procedure_with_provenance`, `store_procedure_with_provenance_strict`,
// the new `ET_PROCEDURE_DERIVES_FROM` constant, and `procedure_provenance`.
// ===========================================================================

/// A procedure stored with one source episode gets exactly one outgoing
/// `PROCEDURE_DERIVES_FROM` edge procedure -> episode. (Invariants I1, I8.)
#[test]
fn test_store_procedure_with_provenance_creates_edge() {
    let mut cm = make_cm();
    let ep = cm
        .store_episode("watched a production deploy", "ops", None, None)
        .unwrap();

    let proc = cm
        .store_procedure_with_provenance(
            "deploy",
            &["build".to_string(), "rollout".to_string()],
            None,
            std::slice::from_ref(&ep),
        )
        .unwrap();
    assert!(proc.starts_with("proc_"));

    let neighbors = cm.graph.query_neighbors(
        &proc,
        Some(ET_PROCEDURE_DERIVES_FROM),
        Direction::Outgoing,
        10,
    );
    assert_eq!(
        neighbors.len(),
        1,
        "expected exactly one PROCEDURE_DERIVES_FROM edge"
    );
    assert_eq!(neighbors[0].0.edge_type, ET_PROCEDURE_DERIVES_FROM);
    assert_eq!(neighbors[0].1.node_id, ep);

    // The procedure is otherwise stored normally.
    let procs = cm.search_procedures("deploy", 10);
    assert_eq!(procs.len(), 1);
    assert_eq!(
        procs[0].steps,
        vec!["build".to_string(), "rollout".to_string()]
    );

    // Public typed read path.
    assert_eq!(cm.procedure_provenance(&proc), vec![ep]);
}

/// Lenient mode skips a missing source episode and still succeeds.
/// (Invariant I2.)
#[test]
fn test_store_procedure_with_provenance_lenient_skips_missing() {
    let mut cm = make_cm();
    let ep = cm.store_episode("real episode", "src", None, None).unwrap();

    let proc = cm
        .store_procedure_with_provenance(
            "p",
            &["s".to_string()],
            None,
            &[ep.clone(), "epi_missing".to_string()],
        )
        .expect("lenient procedure provenance store must succeed");

    assert_eq!(cm.procedure_provenance(&proc), vec![ep]);
}

/// Strict mode rejects a missing source episode with `InvalidInput` and writes
/// zero edges. (Invariant I4.)
#[test]
fn test_store_procedure_with_provenance_strict_errors_on_missing() {
    let mut cm = make_cm();
    let ep = cm.store_episode("real episode", "src", None, None).unwrap();

    let result = cm.store_procedure_with_provenance_strict(
        "p",
        &["s".to_string()],
        None,
        &[ep.clone(), "epi_missing".to_string()],
    );
    assert!(
        matches!(result, Err(MemoryError::InvalidInput(_))),
        "strict mode must reject a missing source episode, got {result:?}"
    );

    let incoming = cm.graph.query_neighbors(
        &ep,
        Some(ET_PROCEDURE_DERIVES_FROM),
        Direction::Incoming,
        10,
    );
    assert!(
        incoming.is_empty(),
        "a strict failure must create zero provenance edges (atomicity)"
    );
}

/// Plain `store_procedure` stays backward compatible: no provenance edges.
/// (Invariant I5.)
#[test]
fn test_store_procedure_backward_compatible_no_edges() {
    let mut cm = make_cm();
    let _ep = cm.store_episode("unrelated", "src", None, None).unwrap();

    let proc = cm.store_procedure("p", &["s".to_string()], None).unwrap();

    let neighbors = cm.graph.query_neighbors(
        &proc,
        Some(ET_PROCEDURE_DERIVES_FROM),
        Direction::Outgoing,
        10,
    );
    assert!(
        neighbors.is_empty(),
        "plain store_procedure must not create provenance edges"
    );
    assert!(cm.procedure_provenance(&proc).is_empty());
}

/// The idempotent upsert-by-name is preserved by the provenance variant:
/// re-storing the same name reuses the node id (provenance never forks the
/// node), updates steps in place, and attaches provenance edges to the single
/// canonical node. (Invariant I6.)
#[test]
fn test_store_procedure_with_provenance_upsert_preserves_node_id() {
    let mut cm = make_cm();
    let ep1 = cm.store_episode("ep1", "src", None, None).unwrap();
    let ep2 = cm.store_episode("ep2", "src", None, None).unwrap();

    let id1 = cm
        .store_procedure_with_provenance(
            "build",
            &["a".to_string()],
            None,
            std::slice::from_ref(&ep1),
        )
        .unwrap();
    let id2 = cm
        .store_procedure_with_provenance(
            "build",
            &["a".to_string(), "b".to_string()],
            None,
            std::slice::from_ref(&ep2),
        )
        .unwrap();

    assert_eq!(id1, id2, "re-storing the same name must reuse the node id");
    assert_eq!(
        cm.get_memory_stats().get("procedural"),
        Some(&1),
        "idempotent upsert must not accumulate duplicate procedure nodes"
    );

    // Steps were updated in place by the second store.
    let procs = cm.search_procedures("build", 10);
    assert_eq!(procs.len(), 1);
    assert_eq!(procs[0].steps, vec!["a".to_string(), "b".to_string()]);

    // Both provenance edges attach to the single canonical node.
    let mut prov = cm.procedure_provenance(&id1);
    prov.sort();
    let mut want = vec![ep1, ep2];
    want.sort();
    assert_eq!(
        prov, want,
        "provenance edges from both upsert calls attach to the same node"
    );
}
