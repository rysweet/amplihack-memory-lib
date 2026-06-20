//! File: rust/amplihack-memory/src/cognitive_memory/tests/dedup_tests.rs
//!
//! Failing-first TDD tests for fact DEDUP, SUPERSESSION, and RETENTION
//! (issue #90). These bind to the not-yet-implemented public contract:
//!   - new `SemanticFact` fields (importance, usage_count, last_accessed_at,
//!     expires_at, archived, superseded_by, content_hash, dedup_key)
//!   - types  `DedupMode`, `DedupOptions`, `ProvenanceOptions`, `FactInput`,
//!     `DedupAction`, `StoreFactOutcome`, `DuplicateFactGroup`,
//!     `RetentionPolicy`, `PruneReport`, and `compute_content_hash`
//!   - methods `CognitiveMemory::upsert_fact`,
//!     `CognitiveMemory::supersede_fact`,
//!     `CognitiveMemory::find_duplicate_facts`,
//!     `CognitiveMemory::prune_semantic_memory`
//!
//! They are backend-agnostic (default in-memory gate); durability of the new
//! fields + `SUPERSEDES` edge is re-proven under `--features persistent` in
//! `persistent_tests.rs`.

use super::super::types::ET_SUPERSEDES;
use super::super::*;
use crate::graph::Direction;
use crate::memory_types::SemanticFact;
use crate::MemoryError;
use chrono::{DateTime, Duration, Utc};

fn make_cm() -> CognitiveMemory {
    CognitiveMemory::new(&format!("test-agent-{}", uuid::Uuid::new_v4())).unwrap()
}

fn tags(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

/// Store a fact via the legacy (non-dedup) path and return its id.
fn store(cm: &mut CognitiveMemory, concept: &str, content: &str, conf: f64, t: &[&str]) -> String {
    let tg = tags(t);
    cm.store_fact(concept, content, conf, "", Some(&tg), None)
        .unwrap()
}

/// Fetch a single fact by id (including archived facts).
fn fact(cm: &CognitiveMemory, id: &str) -> SemanticFact {
    cm.get_all_facts(10_000)
        .into_iter()
        .find(|f| f.node_id == id)
        .expect("fact should exist")
}

fn exists(cm: &CognitiveMemory, id: &str) -> bool {
    cm.get_all_facts(10_000).iter().any(|f| f.node_id == id)
}

/// `StoreFactOptions` selecting just a dedup mode (everything else default).
fn dedup_opts(mode: DedupMode) -> StoreFactOptions {
    StoreFactOptions {
        dedup: DedupOptions {
            mode,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn upsert_imp(
    cm: &mut CognitiveMemory,
    concept: &str,
    content: &str,
    conf: f64,
    imp: f64,
) -> String {
    let mut fi = FactInput::new(concept, content, conf);
    fi.importance = Some(imp);
    cm.upsert_fact(fi, &dedup_opts(DedupMode::None))
        .unwrap()
        .node_id
}

#[allow(clippy::too_many_arguments)]
fn upsert_full(
    cm: &mut CognitiveMemory,
    concept: &str,
    content: &str,
    conf: f64,
    imp: Option<f64>,
    expires: Option<DateTime<Utc>>,
    episodes: Vec<String>,
) -> String {
    let mut fi = FactInput::new(concept, content, conf);
    fi.importance = imp;
    fi.expires_at = expires;
    let opts = StoreFactOptions {
        provenance: ProvenanceOptions {
            source_episode_ids: episodes,
            strict: false,
        },
        ..Default::default()
    };
    cm.upsert_fact(fi, &opts).unwrap().node_id
}

fn has_similar_edge(cm: &CognitiveMemory, from: &str, to: &str) -> bool {
    use super::super::types::ET_SIMILAR_TO;
    cm.graph
        .query_neighbors(from, Some(ET_SIMILAR_TO), Direction::Outgoing, 100)
        .iter()
        .any(|(e, _)| e.target_id == to)
}

fn supersedes_targets(cm: &CognitiveMemory, new_id: &str) -> Vec<String> {
    cm.graph
        .query_neighbors(new_id, Some(ET_SUPERSEDES), Direction::Outgoing, 100)
        .into_iter()
        .map(|(_, n)| n.node_id)
        .collect()
}

// ===========================================================================
// New SemanticFact fields: defaults via the legacy store path
// ===========================================================================

/// A plain `store_fact` populates the new lifecycle fields with their defaults:
/// importance defaults to the confidence, counters/flags are zero/false, the
/// content_hash is computed, and the optional fields are `None`. (Backward
/// compatibility of the storage layer.)
#[test]
fn store_fact_populates_new_fields_with_defaults() {
    let mut cm = make_cm();
    let id = store(&mut cm, "rust", "rust is memory safe", 0.8, &["lang"]);

    let f = fact(&cm, &id);
    assert!(
        (f.importance - 0.8).abs() < 1e-9,
        "importance defaults to confidence"
    );
    assert_eq!(f.usage_count, 0, "usage_count defaults to 0");
    assert!(
        f.last_accessed_at.is_none(),
        "last_accessed_at defaults None"
    );
    assert!(f.expires_at.is_none(), "expires_at defaults None");
    assert!(!f.archived, "archived defaults false");
    assert!(f.superseded_by.is_none(), "superseded_by defaults None");
    assert!(f.dedup_key.is_none(), "dedup_key defaults None");
    assert_eq!(
        f.content_hash,
        compute_content_hash("rust", "rust is memory safe"),
        "content_hash is computed from concept + content"
    );
    assert!(!f.content_hash.is_empty());
}

/// `compute_content_hash` is order-sensitive across the concept/content
/// boundary: it must not collide when the split between concept and content
/// shifts (the unit-separator delimiter prevents concatenation collisions).
#[test]
fn content_hash_is_deterministic_and_collision_resistant() {
    assert_eq!(
        compute_content_hash("ab", "c"),
        compute_content_hash("ab", "c"),
        "same inputs -> same hash"
    );
    assert_ne!(
        compute_content_hash("ab", "c"),
        compute_content_hash("a", "bc"),
        "the concept/content boundary must be part of the hash"
    );
    assert_ne!(
        compute_content_hash("rust", "safe"),
        compute_content_hash("rust", "fast"),
        "different content -> different hash"
    );
}

// ===========================================================================
// supersede_fact + ET_SUPERSEDES
// ===========================================================================

/// `supersede_fact(old, new, reason)` archives the old fact, records the
/// pointer to its replacement, and adds exactly one `SUPERSEDES` edge
/// new -> old carrying the reason. (Requirement 2.)
#[test]
fn supersede_fact_archives_old_and_adds_edge() {
    let mut cm = make_cm();
    let a = store(&mut cm, "c", "v1", 0.9, &[]);
    let b = store(&mut cm, "c", "v2", 0.9, &[]);

    cm.supersede_fact(&a, &b, "newer version").unwrap();

    let fa = fact(&cm, &a);
    assert!(fa.archived, "the superseded fact is archived");
    assert_eq!(
        fa.superseded_by.as_deref(),
        Some(b.as_str()),
        "superseded_by points at the replacement"
    );

    // The replacement is untouched.
    let fb = fact(&cm, &b);
    assert!(!fb.archived);
    assert!(fb.superseded_by.is_none());

    // Exactly one SUPERSEDES edge new -> old, carrying the reason.
    assert_eq!(supersedes_targets(&cm, &b), vec![a.clone()]);
    let edges = cm
        .graph
        .query_neighbors(&b, Some(ET_SUPERSEDES), Direction::Outgoing, 10);
    assert_eq!(edges.len(), 1);
    assert_eq!(
        edges[0].0.properties.get("reason").map(String::as_str),
        Some("newer version")
    );
}

/// `supersede_fact` is idempotent on the edge: a repeat does not duplicate the
/// `SUPERSEDES` edge.
#[test]
fn supersede_fact_does_not_duplicate_edge() {
    let mut cm = make_cm();
    let a = store(&mut cm, "c", "v1", 0.9, &[]);
    let b = store(&mut cm, "c", "v2", 0.9, &[]);

    cm.supersede_fact(&a, &b, "r1").unwrap();
    cm.supersede_fact(&a, &b, "r2").unwrap();

    assert_eq!(
        supersedes_targets(&cm, &b),
        vec![a],
        "no duplicate SUPERSEDES edge on repeat"
    );
}

/// `supersede_fact` validates its inputs: unknown ids and self-supersession
/// are rejected with `InvalidInput` and write nothing.
#[test]
fn supersede_fact_rejects_invalid_input() {
    let mut cm = make_cm();
    let a = store(&mut cm, "c", "v1", 0.9, &[]);

    assert!(matches!(
        cm.supersede_fact(&a, "sem_missing", "r"),
        Err(MemoryError::InvalidInput(_))
    ));
    assert!(matches!(
        cm.supersede_fact("sem_missing", &a, "r"),
        Err(MemoryError::InvalidInput(_))
    ));
    assert!(
        matches!(
            cm.supersede_fact(&a, &a, "r"),
            Err(MemoryError::InvalidInput(_))
        ),
        "a fact cannot supersede itself"
    );

    // Nothing was archived by the failed calls.
    assert!(!fact(&cm, &a).archived);
}

// ===========================================================================
// upsert_fact: dedup modes
// ===========================================================================

/// `DedupMode::None` always inserts a fresh node. (Requirement 3, test 9.)
#[test]
fn upsert_none_always_inserts() {
    let mut cm = make_cm();
    let o1 = cm
        .upsert_fact(FactInput::new("c", "v", 0.8), &dedup_opts(DedupMode::None))
        .unwrap();
    let o2 = cm
        .upsert_fact(FactInput::new("c", "v", 0.8), &dedup_opts(DedupMode::None))
        .unwrap();

    assert!(matches!(o1.dedup_action, DedupAction::Inserted));
    assert!(matches!(o2.dedup_action, DedupAction::Inserted));
    assert_ne!(o1.node_id, o2.node_id);
    assert_eq!(cm.get_all_facts(100).len(), 2, "two distinct nodes");
    assert!(!o1.content_hash.is_empty());
}

/// `ExactContentHash`: a second upsert of the same concept+content reuses the
/// existing node (no new node), bumps usage_count, refreshes last_accessed_at,
/// and updates confidence while leaving importance untouched.
/// (Requirement 3, test 1.)
#[test]
fn upsert_exact_content_hash_reuses_existing() {
    let mut cm = make_cm();
    let o1 = cm
        .upsert_fact(
            FactInput::new("rust", "safe", 0.8),
            &dedup_opts(DedupMode::ExactContentHash),
        )
        .unwrap();
    assert!(matches!(o1.dedup_action, DedupAction::Inserted));
    let imp_before = fact(&cm, &o1.node_id).importance;

    let o2 = cm
        .upsert_fact(
            FactInput::new("rust", "safe", 0.95),
            &dedup_opts(DedupMode::ExactContentHash),
        )
        .unwrap();

    match &o2.dedup_action {
        DedupAction::Reused { existing_id } => assert_eq!(*existing_id, o1.node_id),
        other => panic!("expected Reused, got {other:?}"),
    }
    assert_eq!(o2.node_id, o1.node_id, "outcome points at the reused node");
    assert_eq!(
        cm.get_all_facts(100).len(),
        1,
        "reuse must not create a second node"
    );

    let f = fact(&cm, &o1.node_id);
    assert_eq!(f.usage_count, 1, "reuse bumps usage_count");
    assert!(
        (f.confidence - 0.95).abs() < 1e-9,
        "reuse updates confidence"
    );
    assert!(f.last_accessed_at.is_some(), "reuse sets last_accessed_at");
    assert!(
        (f.importance - imp_before).abs() < 1e-9,
        "reuse leaves importance untouched"
    );
}

/// `ExactContentHash` only treats identical concept+content as a duplicate:
/// different content, or the same content under a different concept, inserts a
/// new node.
#[test]
fn upsert_exact_content_hash_distinguishes_concept_and_content() {
    let mut cm = make_cm();
    cm.upsert_fact(
        FactInput::new("rust", "safe", 0.8),
        &dedup_opts(DedupMode::ExactContentHash),
    )
    .unwrap();
    // Different content -> insert.
    let diff_content = cm
        .upsert_fact(
            FactInput::new("rust", "fast", 0.8),
            &dedup_opts(DedupMode::ExactContentHash),
        )
        .unwrap();
    // Same content, different concept -> insert.
    let diff_concept = cm
        .upsert_fact(
            FactInput::new("python", "safe", 0.8),
            &dedup_opts(DedupMode::ExactContentHash),
        )
        .unwrap();

    assert!(matches!(diff_content.dedup_action, DedupAction::Inserted));
    assert!(matches!(diff_concept.dedup_action, DedupAction::Inserted));
    assert_eq!(cm.get_all_facts(100).len(), 3);
}

/// `CallerKey`: a same-key upsert with *changed* content supersedes the prior
/// fact (keeping the latest live + a `SUPERSEDES` edge), while a same-key
/// upsert with *identical* content reuses without churn. (Requirement 3,
/// test 2.)
#[test]
fn upsert_caller_key_supersedes_on_change_and_reuses_on_match() {
    let mut cm = make_cm();
    let key = || DedupMode::CallerKey("k1".to_string());

    let o1 = cm
        .upsert_fact(FactInput::new("c", "v1", 0.8), &dedup_opts(key()))
        .unwrap();
    assert!(matches!(o1.dedup_action, DedupAction::Inserted));
    assert_eq!(
        fact(&cm, &o1.node_id).dedup_key.as_deref(),
        Some("k1"),
        "the caller key is stored on the fact"
    );

    // Changed content under the same key -> supersession.
    let o2 = cm
        .upsert_fact(FactInput::new("c", "v2", 0.9), &dedup_opts(key()))
        .unwrap();
    let (old_id, new_id) = match &o2.dedup_action {
        DedupAction::Superseded { old_id, new_id } => (old_id.clone(), new_id.clone()),
        other => panic!("expected Superseded, got {other:?}"),
    };
    assert_eq!(old_id, o1.node_id);
    assert_eq!(new_id, o2.node_id);
    assert_ne!(new_id, o1.node_id);

    let old = fact(&cm, &o1.node_id);
    assert!(old.archived, "superseded fact archived");
    assert_eq!(old.superseded_by.as_deref(), Some(new_id.as_str()));
    assert_eq!(supersedes_targets(&cm, &new_id), vec![o1.node_id.clone()]);
    assert_eq!(
        fact(&cm, &new_id).dedup_key.as_deref(),
        Some("k1"),
        "the replacement keeps the caller key"
    );

    // Exactly one live (non-archived) fact under the key.
    let live = cm
        .get_all_facts(100)
        .into_iter()
        .filter(|f| !f.archived)
        .count();
    assert_eq!(live, 1);

    // Same key, identical content -> reuse, no new node, no new edge.
    let o3 = cm
        .upsert_fact(FactInput::new("c", "v2", 0.7), &dedup_opts(key()))
        .unwrap();
    match &o3.dedup_action {
        DedupAction::Reused { existing_id } => assert_eq!(*existing_id, new_id),
        other => panic!("expected Reused on identical content, got {other:?}"),
    }
    assert_eq!(
        supersedes_targets(&cm, &new_id),
        vec![o1.node_id],
        "no churn: identical content does not supersede again"
    );
}

/// `SameConceptSimilarity`: an upsert whose content is similar enough (same
/// concept) to an existing fact reuses it; a different concept (with
/// `same_concept_only`) is never a duplicate and inserts. (Requirement 3.)
#[test]
fn upsert_same_concept_similarity_reuses_close_match() {
    let mut cm = make_cm();
    let opts = StoreFactOptions {
        dedup: DedupOptions {
            mode: DedupMode::SameConceptSimilarity,
            similarity_threshold: 0.6,
            same_concept_only: true,
        },
        ..Default::default()
    };

    let o1 = cm
        .upsert_fact(
            FactInput::new(
                "memory-safety",
                "rust borrow checker prevents memory bugs",
                0.9,
            ),
            &opts,
        )
        .unwrap();
    assert!(matches!(o1.dedup_action, DedupAction::Inserted));

    // Near-identical content, same concept -> reuse.
    let o2 = cm
        .upsert_fact(
            FactInput::new(
                "memory-safety",
                "rust borrow checker prevents memory leaks",
                0.85,
            ),
            &opts,
        )
        .unwrap();
    match &o2.dedup_action {
        DedupAction::Reused { existing_id } => assert_eq!(*existing_id, o1.node_id),
        other => panic!("expected Reused, got {other:?}"),
    }
    assert_eq!(cm.get_all_facts(100).len(), 1);

    // Different concept is not a candidate under same_concept_only -> insert.
    let o3 = cm
        .upsert_fact(
            FactInput::new(
                "web-frontend",
                "javascript renders the browser interface",
                0.7,
            ),
            &opts,
        )
        .unwrap();
    assert!(matches!(o3.dedup_action, DedupAction::Inserted));
    assert_eq!(cm.get_all_facts(100).len(), 2);
}

/// Dedup only considers *live* (non-archived) facts: once a fact is superseded
/// (archived), an `ExactContentHash` upsert of the same content inserts a fresh
/// node rather than resurrecting the archived one.
#[test]
fn upsert_dedup_ignores_archived_facts() {
    let mut cm = make_cm();
    let a = store(&mut cm, "rust", "safe", 0.8, &[]);
    let b = store(&mut cm, "rust", "safe-v2", 0.8, &[]);
    cm.supersede_fact(&a, &b, "newer").unwrap(); // a archived

    // Same content as the archived `a` -> must NOT reuse the archived node.
    let o = cm
        .upsert_fact(
            FactInput::new("rust", "safe", 0.9),
            &dedup_opts(DedupMode::ExactContentHash),
        )
        .unwrap();
    assert!(
        matches!(o.dedup_action, DedupAction::Inserted),
        "archived facts are not dedup candidates"
    );
    assert_ne!(o.node_id, a);
}

/// On insert, `upsert_fact` composes the existing provenance + similarity
/// machinery and reports accurate edge counts. (Requirement 3 + backward-compat
/// with #91/#92.)
#[test]
fn upsert_insert_creates_provenance_and_similarity_edges() {
    let mut cm = make_cm();
    let ep = cm
        .store_episode("observed safety", "src", None, None)
        .unwrap();
    let a = store(
        &mut cm,
        "memory-safety",
        "rust borrow checker prevents memory bugs",
        0.95,
        &["rust", "systems"],
    );

    let mut fi = FactInput::new(
        "memory-safety",
        "rust borrow checker prevents memory leaks",
        0.9,
    );
    fi.tags = tags(&["rust", "systems"]);
    let opts = StoreFactOptions {
        similarity: Some(SimilarityOptions::default()),
        provenance: ProvenanceOptions {
            source_episode_ids: vec![ep.clone()],
            strict: false,
        },
        dedup: DedupOptions::default(),
    };

    let outcome = cm.upsert_fact(fi, &opts).unwrap();
    assert!(matches!(outcome.dedup_action, DedupAction::Inserted));
    assert_eq!(outcome.provenance_edges_created, 1);
    assert_eq!(outcome.similarity_links_created, 1);
    assert_eq!(cm.fact_provenance(&outcome.node_id), vec![ep]);
    assert!(has_similar_edge(&cm, &outcome.node_id, &a));
}

/// `upsert_fact` validates confidence exactly like `store_fact`.
#[test]
fn upsert_fact_validates_confidence() {
    let mut cm = make_cm();
    assert!(cm
        .upsert_fact(FactInput::new("c", "x", 1.5), &dedup_opts(DedupMode::None))
        .is_err());
    assert!(cm
        .upsert_fact(
            FactInput::new("c", "x", f64::NAN),
            &dedup_opts(DedupMode::None)
        )
        .is_err());
}

// ===========================================================================
// find_duplicate_facts
// ===========================================================================

/// `ExactContentHash` grouping collects facts sharing a content_hash into one
/// group keyed by that hash, with a representative drawn from the group.
/// (Requirement 4.)
#[test]
fn find_duplicates_groups_by_content_hash() {
    let mut cm = make_cm();
    let a = store(&mut cm, "rust", "safe", 0.9, &[]);
    let b = store(&mut cm, "rust", "safe", 0.8, &[]);
    let _c = store(&mut cm, "python", "fast", 0.7, &[]);

    let groups = cm.find_duplicate_facts(
        &DedupOptions {
            mode: DedupMode::ExactContentHash,
            ..Default::default()
        },
        10,
    );

    assert_eq!(groups.len(), 1, "only the duplicated pair forms a group");
    let g = &groups[0];
    assert_eq!(g.key, compute_content_hash("rust", "safe"));
    assert_eq!(g.fact_ids.len(), 2);
    assert!(g.fact_ids.contains(&a) && g.fact_ids.contains(&b));
    assert!(
        g.fact_ids.contains(&g.representative_id),
        "representative is a member of the group"
    );
}

/// `CallerKey` grouping collects facts sharing a non-empty dedup_key.
/// (Requirement 4.) Facts are seeded with `DedupMode::None` so two live facts
/// can carry the same key.
#[test]
fn find_duplicates_groups_by_caller_key() {
    let mut cm = make_cm();
    let mut k1a = FactInput::new("c", "v1", 0.8);
    k1a.dedup_key = Some("k1".to_string());
    let a = cm
        .upsert_fact(k1a, &dedup_opts(DedupMode::None))
        .unwrap()
        .node_id;

    let mut k1b = FactInput::new("c", "v2", 0.8);
    k1b.dedup_key = Some("k1".to_string());
    let b = cm
        .upsert_fact(k1b, &dedup_opts(DedupMode::None))
        .unwrap()
        .node_id;

    let mut k2 = FactInput::new("c", "v3", 0.8);
    k2.dedup_key = Some("k2".to_string());
    cm.upsert_fact(k2, &dedup_opts(DedupMode::None)).unwrap();

    let groups = cm.find_duplicate_facts(
        &DedupOptions {
            mode: DedupMode::CallerKey("ignored".to_string()),
            ..Default::default()
        },
        10,
    );

    assert_eq!(groups.len(), 1, "only k1 has >= 2 members");
    assert_eq!(groups[0].key, "k1");
    assert_eq!(groups[0].fact_ids.len(), 2);
    assert!(groups[0].fact_ids.contains(&a) && groups[0].fact_ids.contains(&b));
}

/// `DedupMode::None` reports no duplicate groups. (Requirement 4.)
#[test]
fn find_duplicates_none_mode_is_empty() {
    let mut cm = make_cm();
    store(&mut cm, "rust", "safe", 0.9, &[]);
    store(&mut cm, "rust", "safe", 0.8, &[]);

    let groups = cm.find_duplicate_facts(
        &DedupOptions {
            mode: DedupMode::None,
            ..Default::default()
        },
        10,
    );
    assert!(groups.is_empty());
}

// ===========================================================================
// prune_semantic_memory
// ===========================================================================

/// `dry_run` reports the would-be archive/delete counts and mutates nothing.
/// (Requirement 5, test 3.)
#[test]
fn prune_dry_run_reports_without_mutating() {
    let mut cm = make_cm();
    let low = upsert_imp(&mut cm, "c", "low imp", 0.5, 0.1);
    let high = upsert_imp(&mut cm, "c", "high imp", 0.5, 0.9);

    let policy = RetentionPolicy {
        min_importance_to_keep: 0.5,
        dry_run: true,
        ..Default::default()
    };
    let r = cm.prune_semantic_memory(&policy).unwrap();

    assert_eq!(r.would_archive, 1);
    assert_eq!(r.would_delete, 0);
    assert_eq!(r.archived, 0);
    assert_eq!(r.deleted, 0);

    // Nothing changed on disk.
    assert!(!fact(&cm, &low).archived, "dry_run must not archive");
    assert_eq!(cm.get_all_facts(100).len(), 2);
    let _ = high;
}

/// A real prune is a two-tier lifecycle: the first pass archives candidates,
/// the second pass deletes the now-archived candidates (archive-before-delete).
/// (Requirement 5, test 3 + importance retention test 5.)
#[test]
fn prune_archives_then_deletes_low_importance_facts() {
    let mut cm = make_cm();
    let low = upsert_imp(&mut cm, "c", "low imp", 0.5, 0.1);
    let high = upsert_imp(&mut cm, "c", "high imp", 0.5, 0.9);

    let policy = RetentionPolicy {
        min_importance_to_keep: 0.5,
        ..Default::default()
    };

    // Pass 1: archive only.
    let r1 = cm.prune_semantic_memory(&policy).unwrap();
    assert_eq!((r1.archived, r1.deleted), (1, 0));
    assert!(fact(&cm, &low).archived);
    assert!(!fact(&cm, &high).archived);
    assert_eq!(cm.get_all_facts(100).len(), 2, "archive does not delete");

    // Pass 2: delete the archived candidate.
    let r2 = cm.prune_semantic_memory(&policy).unwrap();
    assert_eq!((r2.archived, r2.deleted), (0, 1));
    assert!(!exists(&cm, &low), "archived low-importance fact deleted");
    assert!(exists(&cm, &high), "high-importance fact retained");
}

/// TTL retention: a fact whose concept has an expired TTL is pruned, while a
/// fact of an unconfigured concept is retained. (Requirement 5, test 4.)
#[test]
fn prune_ttl_expires_stale_concept() {
    let mut cm = make_cm();
    let keep = store(&mut cm, "durable", "long lived", 0.9, &[]);
    let expire = store(&mut cm, "ephemeral", "short lived", 0.9, &[]);

    let mut ttl = HashMap::new();
    ttl.insert("ephemeral".to_string(), 0i64); // age >= 0 -> immediately expired
    let policy = RetentionPolicy {
        ttl_seconds_by_concept: ttl,
        ..Default::default()
    };

    let r1 = cm.prune_semantic_memory(&policy).unwrap();
    assert_eq!((r1.archived, r1.deleted), (1, 0));
    assert!(fact(&cm, &expire).archived);
    assert!(!fact(&cm, &keep).archived);

    let r2 = cm.prune_semantic_memory(&policy).unwrap();
    assert_eq!(r2.deleted, 1);
    assert!(!exists(&cm, &expire));
    assert!(
        exists(&cm, &keep),
        "unconfigured concept is never TTL-pruned"
    );
}

/// `expires_at` retention: a fact whose explicit expiry is in the past is a
/// prune candidate. (Requirement 5.)
#[test]
fn prune_expires_at_in_the_past() {
    let mut cm = make_cm();
    let past = Utc::now() - Duration::seconds(3600);
    let expired = upsert_full(&mut cm, "c", "stale", 0.9, Some(0.9), Some(past), vec![]);

    let r1 = cm
        .prune_semantic_memory(&RetentionPolicy::default())
        .unwrap();
    assert_eq!(r1.archived, 1);
    assert!(fact(&cm, &expired).archived);
}

/// `max_facts_per_concept` keeps the highest-importance facts and prunes the
/// lowest-ranked excess. (Requirement 5.)
#[test]
fn prune_enforces_max_facts_per_concept() {
    let mut cm = make_cm();
    let hi = upsert_imp(&mut cm, "topic", "a", 0.5, 0.9);
    let mid = upsert_imp(&mut cm, "topic", "b", 0.5, 0.5);
    let lo = upsert_imp(&mut cm, "topic", "c", 0.5, 0.1);

    let policy = RetentionPolicy {
        max_facts_per_concept: Some(2),
        ..Default::default()
    };
    let r1 = cm.prune_semantic_memory(&policy).unwrap();

    assert_eq!(r1.archived, 1, "only the single excess fact is a candidate");
    assert!(
        fact(&cm, &lo).archived,
        "lowest-importance fact is the excess"
    );
    assert!(!fact(&cm, &hi).archived);
    assert!(!fact(&cm, &mid).archived);
}

/// Over-cap facts are archived first, then deleted on a later pass, so the
/// archived excess does not accumulate forever (archive-before-delete).
#[test]
fn prune_max_facts_archives_then_deletes_excess() {
    let mut cm = make_cm();
    let f0 = upsert_imp(&mut cm, "topic", "a", 0.5, 0.9);
    let f1 = upsert_imp(&mut cm, "topic", "b", 0.5, 0.7);
    let f2 = upsert_imp(&mut cm, "topic", "c", 0.5, 0.3);
    let f3 = upsert_imp(&mut cm, "topic", "d", 0.5, 0.1);

    let policy = RetentionPolicy {
        max_facts_per_concept: Some(2),
        ..Default::default()
    };

    // Pass 1: archive the two lowest-importance excess facts.
    let r1 = cm.prune_semantic_memory(&policy).unwrap();
    assert_eq!((r1.archived, r1.deleted), (2, 0));
    assert!(fact(&cm, &f2).archived && fact(&cm, &f3).archived);
    assert!(!fact(&cm, &f0).archived && !fact(&cm, &f1).archived);

    // Pass 2: the now-archived excess is deleted; the kept facts remain.
    let r2 = cm.prune_semantic_memory(&policy).unwrap();
    assert_eq!((r2.archived, r2.deleted), (0, 2));
    assert!(!exists(&cm, &f2) && !exists(&cm, &f3));
    assert!(exists(&cm, &f0) && exists(&cm, &f1));

    // Pass 3: stable — nothing left to prune.
    let r3 = cm.prune_semantic_memory(&policy).unwrap();
    assert_eq!((r3.archived, r3.deleted), (0, 0));
}

/// Provenance protection: a fact with a `DERIVES_FROM` edge AND importance at
/// or above the keep threshold is never deleted (it may still be archived). A
/// provenance fact below the threshold loses that protection. (Requirement 5,
/// test 6.)
#[test]
fn prune_protects_high_importance_provenance_facts() {
    let mut cm = make_cm();
    let ep = cm.store_episode("obs", "src", None, None).unwrap();
    let past = Utc::now() - Duration::seconds(3600);

    // provenance + high importance -> protected from deletion.
    let prov_hi = upsert_full(
        &mut cm,
        "c",
        "prov high",
        0.9,
        Some(0.9),
        Some(past),
        vec![ep.clone()],
    );
    // no provenance + high importance -> deletable.
    let plain_hi = upsert_full(
        &mut cm,
        "c",
        "plain high",
        0.9,
        Some(0.9),
        Some(past),
        vec![],
    );
    // provenance + LOW importance -> protection lifted, deletable.
    let prov_lo = upsert_full(
        &mut cm,
        "c",
        "prov low",
        0.9,
        Some(0.1),
        Some(past),
        vec![ep.clone()],
    );

    let policy = RetentionPolicy {
        min_importance_to_keep: 0.5,
        ..Default::default()
    };

    // Pass 1: all three are expired -> all archived (protection allows archiving).
    let r1 = cm.prune_semantic_memory(&policy).unwrap();
    assert_eq!(r1.archived, 3);

    // Pass 2: delete archived candidates except the protected one.
    let r2 = cm.prune_semantic_memory(&policy).unwrap();
    assert_eq!((r2.archived, r2.deleted), (0, 2));
    assert!(exists(&cm, &prov_hi), "protected provenance fact survives");
    assert!(
        !exists(&cm, &plain_hi),
        "unprotected high-importance fact deleted"
    );
    assert!(
        !exists(&cm, &prov_lo),
        "below-threshold provenance fact deleted"
    );
}

/// `include_superseded` makes superseded facts prune candidates. Since
/// supersession already archives them, a single pass deletes. (Requirement 5.)
#[test]
fn prune_include_superseded_deletes_superseded_facts() {
    let mut cm = make_cm();
    let a = store(&mut cm, "c", "v1", 0.9, &[]);
    let b = store(&mut cm, "c", "v2", 0.9, &[]);
    cm.supersede_fact(&a, &b, "newer").unwrap();
    assert!(fact(&cm, &a).archived);

    let policy = RetentionPolicy {
        include_superseded: true,
        ..Default::default()
    };
    let r = cm.prune_semantic_memory(&policy).unwrap();

    assert_eq!(r.deleted, 1, "the archived, superseded fact is deleted");
    assert!(!exists(&cm, &a));
    assert!(exists(&cm, &b), "the live replacement is retained");
}

/// Without `include_superseded`, a merely-superseded fact (high importance, no
/// other trigger) is not a prune candidate.
#[test]
fn prune_leaves_superseded_facts_when_not_included() {
    let mut cm = make_cm();
    let a = store(&mut cm, "c", "v1", 0.9, &[]);
    let b = store(&mut cm, "c", "v2", 0.9, &[]);
    cm.supersede_fact(&a, &b, "newer").unwrap();

    let r = cm
        .prune_semantic_memory(&RetentionPolicy::default())
        .unwrap();
    assert_eq!((r.archived, r.deleted), (0, 0));
    assert!(
        exists(&cm, &a),
        "superseded fact retained without include_superseded"
    );
}
