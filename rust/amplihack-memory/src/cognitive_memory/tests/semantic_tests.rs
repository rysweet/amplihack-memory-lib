use super::super::types::ET_DERIVES_FROM;
use super::super::*;
use crate::graph::Direction;
use crate::MemoryError;

fn make_cm() -> CognitiveMemory {
    CognitiveMemory::new(&format!("test-agent-{}", uuid::Uuid::new_v4())).unwrap()
}

// -- semantic memory --

#[test]
fn test_store_and_search_facts() {
    let mut cm = make_cm();
    cm.store_fact("rust", "Rust is a systems language", 0.95, "", None, None)
        .unwrap();
    cm.store_fact("python", "Python is interpreted", 0.9, "", None, None)
        .unwrap();

    let results = cm.search_facts("rust", 10, 0.0);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].concept, "rust");
}

#[test]
fn test_search_facts_confidence_filter() {
    let mut cm = make_cm();
    cm.store_fact("a", "low confidence", 0.3, "", None, None)
        .unwrap();
    cm.store_fact("a", "high confidence", 0.9, "", None, None)
        .unwrap();

    let results = cm.search_facts("confidence", 10, 0.5);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content, "high confidence");
}

#[test]
fn test_search_facts_empty_query_returns_all() {
    let mut cm = make_cm();
    cm.store_fact("a", "content a", 1.0, "", None, None)
        .unwrap();
    cm.store_fact("b", "content b", 0.5, "", None, None)
        .unwrap();

    let results = cm.search_facts("", 10, 0.0);
    assert_eq!(results.len(), 2);
    // Sorted by confidence desc
    assert_eq!(results[0].concept, "a");
}

#[test]
fn test_get_all_facts() {
    let mut cm = make_cm();
    cm.store_fact("x", "xval", 1.0, "", None, None).unwrap();
    cm.store_fact("y", "yval", 0.5, "", None, None).unwrap();

    let all = cm.get_all_facts(50);
    assert_eq!(all.len(), 2);
}

#[test]
fn test_fact_with_tags_and_metadata() {
    let mut cm = make_cm();
    let tags = vec!["lang".to_string(), "systems".to_string()];
    let mut meta = HashMap::new();
    meta.insert(
        "source".to_string(),
        serde_json::Value::String("docs".into()),
    );

    let id = cm
        .store_fact("rust", "fast lang", 0.9, "doc-1", Some(&tags), Some(&meta))
        .unwrap();

    let facts = cm.search_facts("rust", 10, 0.0);
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].tags, tags);
    assert_eq!(
        facts[0].metadata.get("source").unwrap(),
        &serde_json::Value::String("docs".into())
    );
    assert_eq!(facts[0].source_id, "doc-1");
    assert_eq!(facts[0].node_id, id);
}

#[test]
fn test_link_similar_facts() {
    let mut cm = make_cm();
    let a = cm
        .store_fact("rust", "systems lang", 1.0, "", None, None)
        .unwrap();
    let b = cm
        .store_fact("cpp", "systems lang too", 1.0, "", None, None)
        .unwrap();

    assert!(cm.link_similar_facts(&a, &b, 0.85).is_ok());

    // Verify the SIMILAR_TO edge was actually created
    let neighbors = cm.graph.query_neighbors(&a, None, Direction::Both, 10);
    assert!(
        neighbors
            .iter()
            .any(|(e, _)| e.edge_type == "SIMILAR_TO" && e.target_id == b),
        "SIMILAR_TO edge should exist between the two facts"
    );
}

// -- Extended coverage: store_fact --

#[test]
fn test_store_fact_invalid_confidence_nan() {
    let mut cm = make_cm();
    assert!(cm
        .store_fact("c", "content", f64::NAN, "", None, None)
        .is_err());
}

#[test]
fn test_store_fact_invalid_confidence_out_of_range() {
    let mut cm = make_cm();
    assert!(cm.store_fact("c", "content", 1.5, "", None, None).is_err());
    assert!(cm.store_fact("c", "content", -0.1, "", None, None).is_err());
}

#[test]
fn test_store_fact_boundary_confidence() {
    let mut cm = make_cm();
    let id_zero = cm
        .store_fact("c", "zero conf", 0.0, "", None, None)
        .unwrap();
    let id_max = cm.store_fact("c", "max conf", 1.0, "", None, None).unwrap();

    // Verify boundary values were persisted correctly
    let facts = cm.search_facts("conf", 10, 0.0);
    let zero_fact = facts.iter().find(|f| f.node_id == id_zero);
    let max_fact = facts.iter().find(|f| f.node_id == id_max);
    assert!(
        zero_fact.is_some(),
        "zero-confidence fact should be retrievable"
    );
    assert!(
        max_fact.is_some(),
        "max-confidence fact should be retrievable"
    );
    assert!((zero_fact.unwrap().confidence - 0.0).abs() < f64::EPSILON);
    assert!((max_fact.unwrap().confidence - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_store_fact_returns_searchable_id() {
    let mut cm = make_cm();
    let id = cm
        .store_fact("rust-lang", "Rust is safe", 0.95, "src1", None, None)
        .unwrap();
    assert!(id.starts_with("sem_"));

    let facts = cm.search_facts("rust", 10, 0.0);
    assert!(facts.iter().any(|f| f.node_id == id));
}

// -- Corrected retrieval semantics (the consumer-fork bugs) --
//
// These guard the tokenized recall behavior at the *default* (in-memory)
// gate, since the semantics live in the shared cognitive layer and apply to
// every backend. A regression to whole-string CONTAINS or AND-of-tokens
// matching would fail here without needing the `persistent` feature.

#[test]
fn test_search_facts_tokenized_multiword_or() {
    let mut cm = make_cm();
    cm.store_fact(
        "concurrency",
        "The quick brown fox jumps over the lazy dog",
        0.8,
        "src",
        None,
        None,
    )
    .unwrap();

    // Multi-word, out-of-order, mixed-case query. Tokens are OR-matched as
    // CONTAINS substrings, NOT a whole-string CONTAINS — "fox quick" is not a
    // substring of the content, so whole-string matching would return 0.
    assert_eq!(
        cm.search_facts("FOX quick", 10, 0.0).len(),
        1,
        "tokenized, case-insensitive multi-word recall should match"
    );

    // OR semantics: a single overlapping token is sufficient even when another
    // query token is absent. AND-of-tokens matching would return 0 here.
    assert_eq!(
        cm.search_facts("FOX elephant", 10, 0.0).len(),
        1,
        "one overlapping token should match (OR per token)"
    );

    // No token overlaps -> no match.
    assert!(cm.search_facts("elephant zebra", 10, 0.0).is_empty());
}

// ===========================================================================
// Provenance edges (issue #90): SemanticMemory --DERIVES_FROM--> EpisodicMemory
//
// File: rust/amplihack-memory/src/cognitive_memory/tests/semantic_tests.rs
//
// These are failing-first TDD tests. They exercise the not-yet-implemented
// `store_fact_with_provenance`, `store_fact_with_provenance_strict`, and the
// `fact_provenance` read path. Direction is derived -> episode (Outgoing from
// the fact), matching the existing `link_fact_to_episode` assertion pattern.
// ===========================================================================

/// A fact stored with one real source episode gets exactly one outgoing
/// `DERIVES_FROM` edge pointing fact -> episode, and the public read path
/// surfaces that episode id. (Invariants I1, I2, I8.)
#[test]
fn test_store_fact_with_provenance_creates_derives_from_edge() {
    let mut cm = make_cm();
    let ep = cm
        .store_episode("observed that rust compiles", "reading", None, None)
        .unwrap();

    let fact = cm
        .store_fact_with_provenance(
            "rust",
            "Rust is memory safe",
            0.95,
            "",
            None,
            None,
            std::slice::from_ref(&ep),
        )
        .unwrap();

    // The fact node itself is stored and searchable, exactly like store_fact.
    assert!(fact.starts_with("sem_"));
    assert!(cm
        .search_facts("rust", 10, 0.0)
        .iter()
        .any(|f| f.node_id == fact));

    // Exactly one DERIVES_FROM edge, fact -> episode.
    let neighbors = cm
        .graph
        .query_neighbors(&fact, Some(ET_DERIVES_FROM), Direction::Outgoing, 10);
    assert_eq!(
        neighbors.len(),
        1,
        "expected exactly one DERIVES_FROM edge from the fact"
    );
    assert_eq!(neighbors[0].0.edge_type, ET_DERIVES_FROM);
    assert_eq!(neighbors[0].1.node_id, ep);

    // Public typed read path returns the same source episode id.
    assert_eq!(cm.fact_provenance(&fact), vec![ep]);
}

/// Lenient mode: a missing source episode id is skipped with a warning (the
/// call still succeeds) and only the valid episode is linked. (Invariant I2.)
#[test]
fn test_store_fact_with_provenance_lenient_skips_missing_episode() {
    let mut cm = make_cm();
    let ep = cm.store_episode("real episode", "src", None, None).unwrap();

    let fact = cm
        .store_fact_with_provenance(
            "topic",
            "a derived fact",
            0.8,
            "",
            None,
            None,
            &[ep.clone(), "epi_does_not_exist".to_string()],
        )
        .expect("lenient provenance store must succeed despite a missing episode");

    let neighbors = cm
        .graph
        .query_neighbors(&fact, Some(ET_DERIVES_FROM), Direction::Outgoing, 10);
    assert_eq!(
        neighbors.len(),
        1,
        "the missing episode id must be skipped, only the valid one linked"
    );
    assert_eq!(neighbors[0].1.node_id, ep);
    assert_eq!(cm.fact_provenance(&fact), vec![ep]);
}

/// Strict mode: any missing source episode id makes the call return
/// `InvalidInput` and write ZERO provenance edges (validate-then-emit
/// atomicity). (Invariant I4.)
#[test]
fn test_store_fact_with_provenance_strict_errors_on_missing_episode() {
    let mut cm = make_cm();
    let ep = cm.store_episode("real episode", "src", None, None).unwrap();

    let result = cm.store_fact_with_provenance_strict(
        "topic",
        "a derived fact",
        0.8,
        "",
        None,
        None,
        &[ep.clone(), "epi_missing".to_string()],
    );
    assert!(
        matches!(result, Err(MemoryError::InvalidInput(_))),
        "strict mode must reject a missing source episode with InvalidInput, got {result:?}"
    );

    // Atomicity: not even the valid episode may receive an edge from the failed
    // call. No DERIVES_FROM edge points at the episode.
    let incoming = cm
        .graph
        .query_neighbors(&ep, Some(ET_DERIVES_FROM), Direction::Incoming, 10);
    assert!(
        incoming.is_empty(),
        "a strict failure must create zero provenance edges (atomicity)"
    );
}

/// An empty `source_episode_ids` slice creates no edges: the provenance API is
/// observably identical to plain `store_fact` on the no-provenance path.
/// (Invariant I5.)
#[test]
fn test_store_fact_with_provenance_empty_slice_creates_no_edges() {
    let mut cm = make_cm();
    let fact = cm
        .store_fact_with_provenance("c", "content", 0.5, "src", None, None, &[])
        .unwrap();

    let neighbors = cm
        .graph
        .query_neighbors(&fact, Some(ET_DERIVES_FROM), Direction::Outgoing, 10);
    assert!(
        neighbors.is_empty(),
        "empty provenance must create no edges"
    );
    assert!(cm.fact_provenance(&fact).is_empty());

    // Node stored normally.
    let facts = cm.search_facts("content", 10, 0.0);
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].node_id, fact);
}

/// Plain `store_fact` stays backward compatible: it creates no provenance
/// edges and stores the node (including the `source_id` string property)
/// exactly as before. (Invariant I5.)
#[test]
fn test_store_fact_backward_compatible_creates_no_edges() {
    let mut cm = make_cm();
    let _ep = cm
        .store_episode("an unrelated episode", "src", None, None)
        .unwrap();

    let fact = cm
        .store_fact("plain", "a plain fact", 0.9, "src-string", None, None)
        .unwrap();

    let neighbors = cm
        .graph
        .query_neighbors(&fact, Some(ET_DERIVES_FROM), Direction::Outgoing, 10);
    assert!(
        neighbors.is_empty(),
        "plain store_fact must not create provenance edges"
    );
    assert!(cm.fact_provenance(&fact).is_empty());

    // source_id remains a plain string property; it is never auto-converted to
    // an edge.
    let facts = cm.search_facts("plain", 10, 0.0);
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].source_id, "src-string");
}

/// The provenance variant keeps the same confidence validation as
/// `store_fact`.
#[test]
fn test_store_fact_with_provenance_validates_confidence() {
    let mut cm = make_cm();
    assert!(cm
        .store_fact_with_provenance("c", "x", f64::NAN, "", None, None, &[])
        .is_err());
    assert!(cm
        .store_fact_with_provenance("c", "x", 1.5, "", None, None, &[])
        .is_err());
    assert!(cm
        .store_fact_with_provenance("c", "x", -0.1, "", None, None, &[])
        .is_err());
}
