//! File: rust/amplihack-memory/src/cognitive_memory/tests/similarity_tests.rs
//!
//! Failing-first TDD tests for automatic Jaccard `SIMILAR_TO` auto-linking
//! (issue #90). These bind to the not-yet-implemented public contract:
//!   - types  `SimilarityOptions`, `SimilarityReport`, `StoreFactOptions`
//!   - methods `CognitiveMemory::auto_link_similar_facts`,
//!     `CognitiveMemory::rebuild_similarity_links`,
//!     `CognitiveMemory::store_fact_with_options`
//!
//! They are backend-agnostic (default in-memory gate); the same behavior is
//! re-proven for durability under `--features persistent` in
//! `persistent_tests.rs::similar_to_edge_survives_reopen`.
//!
//! Fixture design (so the boundary is deterministic against the composite
//! score `0.5*word + 0.2*tag + 0.3*concept`, see src/similarity.rs):
//!   - `a`,`b`,`c` share concept "memory-safety" + tags ["rust","systems"] +
//!     most content tokens  -> pairwise composite in [0.75, 0.86] (>= 0.60).
//!   - `d`,`e` share nothing with the cluster or each other -> composite 0.0.

use super::super::types::ET_SIMILAR_TO;
use super::super::*;
use crate::graph::Direction;

fn make_cm() -> CognitiveMemory {
    CognitiveMemory::new(&format!("test-agent-{}", uuid::Uuid::new_v4())).unwrap()
}

fn tags(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

/// Store a fact (plain, no auto-linking) and return its id.
fn store(cm: &mut CognitiveMemory, concept: &str, content: &str, conf: f64, t: &[&str]) -> String {
    let tg = tags(t);
    cm.store_fact(concept, content, conf, "", Some(&tg), None)
        .unwrap()
}

/// Three mutually-related facts (`a`,`b`,`c`) + two unrelated (`d`,`e`).
/// Distinct, descending confidences keep `get_all_facts` ordering stable so
/// the `candidate_limit` bound is testable.
struct Corpus {
    a: String,
    b: String,
    c: String,
    d: String,
    e: String,
}

fn build_corpus(cm: &mut CognitiveMemory) -> Corpus {
    Corpus {
        a: store(
            cm,
            "memory-safety",
            "rust borrow checker prevents memory bugs",
            0.95,
            &["rust", "systems"],
        ),
        b: store(
            cm,
            "memory-safety",
            "rust borrow checker prevents memory leaks",
            0.90,
            &["rust", "systems"],
        ),
        c: store(
            cm,
            "memory-safety",
            "rust borrow checker stops memory errors",
            0.85,
            &["rust", "systems"],
        ),
        d: store(
            cm,
            "web-frontend",
            "javascript renders dynamic browser interface",
            0.80,
            &["frontend", "javascript"],
        ),
        e: store(
            cm,
            "database-indexing",
            "postgres btree index speeds query lookups",
            0.75,
            &["database", "sql"],
        ),
    }
}

fn has_similar_edge(cm: &CognitiveMemory, from: &str, to: &str) -> bool {
    cm.graph
        .query_neighbors(from, Some(ET_SIMILAR_TO), Direction::Outgoing, 100)
        .iter()
        .any(|(e, _)| e.target_id == to)
}

fn similar_out_count(cm: &CognitiveMemory, id: &str) -> usize {
    cm.graph
        .query_neighbors(id, Some(ET_SIMILAR_TO), Direction::Outgoing, 100)
        .len()
}

fn similar_in_count(cm: &CognitiveMemory, id: &str) -> usize {
    cm.graph
        .query_neighbors(id, Some(ET_SIMILAR_TO), Direction::Incoming, 100)
        .len()
}

fn similar_degree(cm: &CognitiveMemory, id: &str) -> usize {
    cm.graph
        .query_neighbors(id, Some(ET_SIMILAR_TO), Direction::Both, 100)
        .len()
}

// ---------------------------------------------------------------------------
// auto_link_similar_facts
// ---------------------------------------------------------------------------

/// Core behavior: a `SIMILAR_TO` edge is created from the source fact to every
/// OTHER same-agent fact at/above threshold, and to none below it. (I1, I2.)
#[test]
fn auto_link_creates_edges_only_between_related_facts() {
    let mut cm = make_cm();
    let c = build_corpus(&mut cm);

    let n = cm
        .auto_link_similar_facts(&c.a, &SimilarityOptions::default())
        .unwrap();

    // a links to the two related facts (b, c); never to the unrelated d, e.
    assert_eq!(n, 2, "exactly the two related facts should be linked");
    assert!(has_similar_edge(&cm, &c.a, &c.b));
    assert!(has_similar_edge(&cm, &c.a, &c.c));
    assert!(!has_similar_edge(&cm, &c.a, &c.d));
    assert!(!has_similar_edge(&cm, &c.a, &c.e));

    // Default is bidirectional: reciprocal edges exist back to a.
    assert!(has_similar_edge(&cm, &c.b, &c.a));
    assert!(has_similar_edge(&cm, &c.c, &c.a));

    // The unrelated facts gained no SIMILAR_TO edges at all.
    assert_eq!(similar_degree(&cm, &c.d), 0);
    assert_eq!(similar_degree(&cm, &c.e), 0);
}

/// No self-loops: the source fact is never a candidate for itself. (I3.)
#[test]
fn auto_link_never_creates_self_loop() {
    let mut cm = make_cm();
    let a = store(
        &mut cm,
        "memory-safety",
        "rust borrow checker prevents memory bugs",
        0.95,
        &["rust", "systems"],
    );

    let n = cm
        .auto_link_similar_facts(&a, &SimilarityOptions::default())
        .unwrap();

    assert_eq!(n, 0, "a lone fact links to nothing");
    assert!(!has_similar_edge(&cm, &a, &a), "no self SIMILAR_TO edge");
    assert_eq!(similar_degree(&cm, &a), 0);
}

/// Idempotency: a re-run links nothing new and creates no duplicate edge. (I4.)
#[test]
fn auto_link_is_idempotent() {
    let mut cm = make_cm();
    let a = store(
        &mut cm,
        "memory-safety",
        "rust borrow checker prevents memory bugs",
        0.95,
        &["rust", "systems"],
    );
    let b = store(
        &mut cm,
        "memory-safety",
        "rust borrow checker prevents memory leaks",
        0.90,
        &["rust", "systems"],
    );

    let first = cm
        .auto_link_similar_facts(&a, &SimilarityOptions::default())
        .unwrap();
    assert_eq!(first, 1);
    assert_eq!(similar_out_count(&cm, &a), 1);

    let second = cm
        .auto_link_similar_facts(&a, &SimilarityOptions::default())
        .unwrap();
    assert_eq!(second, 0, "second pass must link nothing new");
    assert_eq!(
        similar_out_count(&cm, &a),
        1,
        "no duplicate SIMILAR_TO edge on re-run"
    );
    let _ = b;
}

/// The threshold gates linking: with a stricter threshold the weaker (but still
/// positive) pair is excluded. composite(a,b)=~0.857, composite(a,c)=0.75. (I2.)
#[test]
fn auto_link_respects_threshold() {
    let mut cm = make_cm();
    let a = store(
        &mut cm,
        "memory-safety",
        "rust borrow checker prevents memory bugs",
        0.95,
        &["rust", "systems"],
    );
    let b = store(
        &mut cm,
        "memory-safety",
        "rust borrow checker prevents memory leaks",
        0.90,
        &["rust", "systems"],
    );
    let c = store(
        &mut cm,
        "memory-safety",
        "rust borrow checker stops memory errors",
        0.85,
        &["rust", "systems"],
    );

    let opts = SimilarityOptions {
        threshold: 0.80,
        ..SimilarityOptions::default()
    };
    let n = cm.auto_link_similar_facts(&a, &opts).unwrap();

    assert_eq!(n, 1, "only the >=0.80 pair (a,b) links at threshold 0.80");
    assert!(has_similar_edge(&cm, &a, &b));
    assert!(
        !has_similar_edge(&cm, &a, &c),
        "the 0.75 pair (a,c) is below the 0.80 threshold"
    );
}

/// `enabled = false` is an inert no-op: returns 0 and writes nothing. (I8.)
#[test]
fn auto_link_disabled_is_noop() {
    let mut cm = make_cm();
    let a = store(
        &mut cm,
        "memory-safety",
        "rust borrow checker prevents memory bugs",
        0.95,
        &["rust", "systems"],
    );
    let _b = store(
        &mut cm,
        "memory-safety",
        "rust borrow checker prevents memory leaks",
        0.90,
        &["rust", "systems"],
    );

    let opts = SimilarityOptions {
        enabled: false,
        ..SimilarityOptions::default()
    };
    let n = cm.auto_link_similar_facts(&a, &opts).unwrap();

    assert_eq!(n, 0);
    assert_eq!(similar_degree(&cm, &a), 0, "disabled must write no edges");
}

/// A missing/unknown fact id is lenient: warn + `Ok(0)`, never an error.
#[test]
fn auto_link_missing_fact_is_lenient() {
    let mut cm = make_cm();
    let _a = store(
        &mut cm,
        "memory-safety",
        "rust borrow checker prevents memory bugs",
        0.95,
        &["rust", "systems"],
    );

    let n = cm
        .auto_link_similar_facts("sem_does_not_exist", &SimilarityOptions::default())
        .expect("missing fact must be lenient, not an error");
    assert_eq!(n, 0);
}

/// `bidirectional = false` creates only the forward edge a -> b. (I5.)
#[test]
fn auto_link_bidirectional_false_only_forward() {
    let mut cm = make_cm();
    let a = store(
        &mut cm,
        "memory-safety",
        "rust borrow checker prevents memory bugs",
        0.95,
        &["rust", "systems"],
    );
    let b = store(
        &mut cm,
        "memory-safety",
        "rust borrow checker prevents memory leaks",
        0.90,
        &["rust", "systems"],
    );

    let opts = SimilarityOptions {
        bidirectional: false,
        ..SimilarityOptions::default()
    };
    let n = cm.auto_link_similar_facts(&a, &opts).unwrap();

    assert_eq!(n, 1);
    assert!(has_similar_edge(&cm, &a, &b), "forward edge a -> b exists");
    assert_eq!(similar_out_count(&cm, &b), 0, "no reciprocal edge b -> a");
    assert_eq!(
        similar_in_count(&cm, &b),
        1,
        "b only receives the inbound a -> b edge"
    );
}

/// `bidirectional = true` (default) also creates the reciprocal b -> a. (I5.)
#[test]
fn auto_link_bidirectional_true_creates_reciprocal() {
    let mut cm = make_cm();
    let a = store(
        &mut cm,
        "memory-safety",
        "rust borrow checker prevents memory bugs",
        0.95,
        &["rust", "systems"],
    );
    let b = store(
        &mut cm,
        "memory-safety",
        "rust borrow checker prevents memory leaks",
        0.90,
        &["rust", "systems"],
    );

    let n = cm
        .auto_link_similar_facts(&a, &SimilarityOptions::default())
        .unwrap();

    assert_eq!(n, 1, "the pair is counted once, not per directed edge");
    assert!(has_similar_edge(&cm, &a, &b));
    assert!(has_similar_edge(&cm, &b, &a), "reciprocal edge present");
}

/// The created edge carries a `similarity_score` property that is a fixed
/// 4-decimal, parseable `f64` string at/above threshold. (Edge data contract.)
#[test]
fn auto_link_writes_parseable_similarity_score_property() {
    let mut cm = make_cm();
    let a = store(
        &mut cm,
        "memory-safety",
        "rust borrow checker prevents memory bugs",
        0.95,
        &["rust", "systems"],
    );
    let _b = store(
        &mut cm,
        "memory-safety",
        "rust borrow checker prevents memory leaks",
        0.90,
        &["rust", "systems"],
    );

    cm.auto_link_similar_facts(&a, &SimilarityOptions::default())
        .unwrap();

    let edges = cm
        .graph
        .query_neighbors(&a, Some(ET_SIMILAR_TO), Direction::Outgoing, 10);
    assert_eq!(edges.len(), 1);

    let raw = edges[0]
        .0
        .properties
        .get("similarity_score")
        .expect("edge must carry a similarity_score property");
    let score: f64 = raw.parse().expect("similarity_score must parse as f64");
    assert!(
        (0.60..=1.0).contains(&score),
        "score {score} must be in [threshold, 1.0]"
    );

    // Fixed `{:.4}` format: exactly four digits after the decimal point.
    let frac = raw
        .split_once('.')
        .map(|(_, f)| f)
        .unwrap_or_else(|| panic!("similarity_score '{raw}' must be a decimal"));
    assert_eq!(
        frac.len(),
        4,
        "similarity_score '{raw}' must use 4-decimal formatting"
    );
}

/// `candidate_limit` bounds how many other facts are scored per source. With a
/// limit of 1, at most one link can be created even with two valid candidates.
/// (I7.)
#[test]
fn auto_link_bounded_by_candidate_limit() {
    let mut cm = make_cm();
    let a = store(
        &mut cm,
        "memory-safety",
        "rust borrow checker prevents memory bugs",
        0.95,
        &["rust", "systems"],
    );
    let _b = store(
        &mut cm,
        "memory-safety",
        "rust borrow checker prevents memory leaks",
        0.90,
        &["rust", "systems"],
    );
    let _c = store(
        &mut cm,
        "memory-safety",
        "rust borrow checker stops memory errors",
        0.85,
        &["rust", "systems"],
    );

    let opts = SimilarityOptions {
        candidate_limit: 1,
        ..SimilarityOptions::default()
    };
    let n = cm.auto_link_similar_facts(&a, &opts).unwrap();

    assert!(
        n <= 1,
        "at most candidate_limit (1) facts may be scored/linked, got {n}"
    );
    assert!(similar_out_count(&cm, &a) <= 1);
}

// ---------------------------------------------------------------------------
// store_fact does NOT auto-link; store_fact_with_options opt-in does
// ---------------------------------------------------------------------------

/// Backward compatibility: plain `store_fact` never creates `SIMILAR_TO`
/// edges, even between strongly related facts.
#[test]
fn store_fact_does_not_auto_link() {
    let mut cm = make_cm();
    let a = store(
        &mut cm,
        "memory-safety",
        "rust borrow checker prevents memory bugs",
        0.95,
        &["rust", "systems"],
    );
    let b = store(
        &mut cm,
        "memory-safety",
        "rust borrow checker prevents memory leaks",
        0.90,
        &["rust", "systems"],
    );

    assert_eq!(similar_degree(&cm, &a), 0);
    assert_eq!(similar_degree(&cm, &b), 0);
}

/// `store_fact_with_options` with `similarity: None` is observably identical to
/// the non-linking store path (no `SIMILAR_TO` edges).
#[test]
fn store_fact_with_options_none_does_not_link() {
    let mut cm = make_cm();
    let a = store(
        &mut cm,
        "memory-safety",
        "rust borrow checker prevents memory bugs",
        0.95,
        &["rust", "systems"],
    );

    let tg = tags(&["rust", "systems"]);
    let opts = StoreFactOptions { similarity: None };
    let b = cm
        .store_fact_with_options(
            "memory-safety",
            "rust borrow checker prevents memory leaks",
            0.90,
            "",
            Some(&tg),
            None,
            &[],
            &opts,
        )
        .unwrap();

    assert!(b.starts_with("sem_"));
    assert!(!has_similar_edge(&cm, &b, &a));
    assert_eq!(similar_degree(&cm, &b), 0);
}

/// `store_fact_with_options` with `similarity: Some(enabled)` auto-links the
/// freshly stored fact to the existing related fact.
#[test]
fn store_fact_with_options_some_auto_links() {
    let mut cm = make_cm();
    let a = store(
        &mut cm,
        "memory-safety",
        "rust borrow checker prevents memory bugs",
        0.95,
        &["rust", "systems"],
    );

    let tg = tags(&["rust", "systems"]);
    let opts = StoreFactOptions {
        similarity: Some(SimilarityOptions::default()),
    };
    let b = cm
        .store_fact_with_options(
            "memory-safety",
            "rust borrow checker prevents memory leaks",
            0.90,
            "",
            Some(&tg),
            None,
            &[],
            &opts,
        )
        .unwrap();

    assert!(
        has_similar_edge(&cm, &b, &a),
        "the new fact must be linked to the related existing fact"
    );
    // Default options are bidirectional.
    assert!(has_similar_edge(&cm, &a, &b));
}

// ---------------------------------------------------------------------------
// rebuild_similarity_links
// ---------------------------------------------------------------------------

/// A full rebuild links every above-threshold pair across all facts exactly
/// once, reports accurate counts, and is idempotent on a second pass. (I4, I6.)
#[test]
fn rebuild_links_all_related_pairs_and_is_idempotent() {
    let mut cm = make_cm();
    let c = build_corpus(&mut cm);

    let report = cm
        .rebuild_similarity_links(&SimilarityOptions::default())
        .unwrap();

    assert_eq!(
        report,
        SimilarityReport {
            facts_processed: 5,
            links_created: 3,
        },
        "three related pairs (a-b, a-c, b-c); d, e isolated"
    );

    assert!(has_similar_edge(&cm, &c.a, &c.b));
    assert!(has_similar_edge(&cm, &c.a, &c.c));
    assert!(has_similar_edge(&cm, &c.b, &c.c));
    assert_eq!(similar_degree(&cm, &c.d), 0);
    assert_eq!(similar_degree(&cm, &c.e), 0);

    // Re-running creates nothing new (non-destructive + idempotent).
    let again = cm
        .rebuild_similarity_links(&SimilarityOptions::default())
        .unwrap();
    assert_eq!(again.links_created, 0, "rebuild must be idempotent");
    assert_eq!(again.facts_processed, 5);
}

/// Disabled rebuild is an inert no-op with an empty report. (I8.)
#[test]
fn rebuild_disabled_is_noop() {
    let mut cm = make_cm();
    let c = build_corpus(&mut cm);

    let opts = SimilarityOptions {
        enabled: false,
        ..SimilarityOptions::default()
    };
    let report = cm.rebuild_similarity_links(&opts).unwrap();

    assert_eq!(
        report,
        SimilarityReport {
            facts_processed: 0,
            links_created: 0,
        }
    );
    assert_eq!(similar_degree(&cm, &c.a), 0, "disabled writes no edges");
}

/// Rebuild is non-destructive: a pre-existing (manual) `SIMILAR_TO` edge is
/// preserved and suppresses a duplicate for that pair. (I4, non-destructive.)
#[test]
fn rebuild_preserves_existing_manual_links() {
    let mut cm = make_cm();
    let a = store(
        &mut cm,
        "memory-safety",
        "rust borrow checker prevents memory bugs",
        0.95,
        &["rust", "systems"],
    );
    let b = store(
        &mut cm,
        "memory-safety",
        "rust borrow checker prevents memory leaks",
        0.90,
        &["rust", "systems"],
    );

    // A manually created edge already connects the only related pair.
    cm.link_similar_facts(&a, &b, 0.5).unwrap();
    assert!(has_similar_edge(&cm, &a, &b));

    let report = cm
        .rebuild_similarity_links(&SimilarityOptions::default())
        .unwrap();

    assert_eq!(
        report.links_created, 0,
        "the manually-linked pair must not be re-created"
    );
    // The manual edge is still present (non-destructive) and not duplicated.
    assert!(has_similar_edge(&cm, &a, &b));
    assert_eq!(similar_out_count(&cm, &a), 1);
}
