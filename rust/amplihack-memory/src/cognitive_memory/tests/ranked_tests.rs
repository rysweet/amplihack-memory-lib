//! TDD specification tests for **ranked recall** over [`CognitiveMemory`]
//! (issue #90).
//!
//! These tests are written *before* the implementation and define the
//! behavioral contract for:
//!   * [`CognitiveMemory::recall_facts_ranked`]
//!   * [`CognitiveMemory::recall_episodes_ranked`]
//!   * [`CognitiveMemory::record_access`]
//!   * the value types `RecallWeights`, `RecallOptions`, `AccessKind`, `Scored<T>`
//!   * the pure scoring primitives `keyword_jaccard`, `exp_decay`, `usage_boost`
//!
//! They are derived from the consolidated design (Step 5e) and its Gherkin
//! scenarios / formal invariants (I1–I13).
//!
//! ## Required test-only seam
//!
//! Several scenarios (recency decay, usage isolation, lifecycle flags, tenant
//! isolation) need to construct deterministic node state that the public API
//! does not let a test fabricate directly (e.g. an *old* `last_accessed_at`, a
//! foreign `agent_id`). The implementation MUST therefore expose a
//! `#[cfg(test)] pub(crate)` helper on `CognitiveMemory` that writes raw node
//! properties through the backing `GraphStore`:
//!
//! ```ignore
//! #[cfg(test)]
//! pub(crate) fn set_node_props_for_test(
//!     &mut self,
//!     node_id: &str,
//!     props: std::collections::HashMap<String, String>,
//! ) -> bool {
//!     self.graph.update_node(node_id, props)
//! }
//! ```
//!
//! This is the only non-public surface these tests rely on; everything else is
//! the real public ranked-recall API. Until that API exists this module fails
//! to compile — the expected TDD "red" state.

use std::collections::HashMap;

// Brings `CognitiveMemory` plus the ranked-recall re-exports
// (`RecallWeights`, `RecallOptions`, `AccessKind`, `Scored`, `keyword_jaccard`,
// `exp_decay`, `usage_boost`) into scope from `cognitive_memory::mod`.
use super::super::*;

use crate::memory_types::{EpisodicMemory, SemanticFact};
use crate::MemoryError;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Fresh in-memory `CognitiveMemory` with a unique agent name per test.
fn make_cm() -> CognitiveMemory {
    CognitiveMemory::new(&format!("ranked-agent-{}", uuid::Uuid::new_v4())).unwrap()
}

/// Seconds since the Unix epoch, computed the same way the implementation's
/// internal `ts_now()` does, so test-constructed ages line up with scoring.
fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

/// Overwrite a single raw property on a node (see module docs for the required
/// `set_node_props_for_test` seam).
fn set_prop(cm: &mut CognitiveMemory, node_id: &str, key: &str, value: &str) {
    let mut props = HashMap::new();
    props.insert(key.to_string(), value.to_string());
    assert!(
        cm.set_node_props_for_test(node_id, props),
        "set_node_props_for_test failed for node {node_id}"
    );
}

/// Default `RecallOptions` that never mutates the store, so scoring assertions
/// stay isolated from the access-tracking side effect.
fn read_only_opts() -> RecallOptions {
    RecallOptions {
        record_access: false,
        ..RecallOptions::default()
    }
}

fn find_fact<'a>(
    scored: &'a [Scored<SemanticFact>],
    node_id: &str,
) -> Option<&'a Scored<SemanticFact>> {
    scored.iter().find(|s| s.item.node_id == node_id)
}

fn rank_of_fact(scored: &[Scored<SemanticFact>], node_id: &str) -> Option<usize> {
    scored.iter().position(|s| s.item.node_id == node_id)
}

fn find_episode<'a>(
    scored: &'a [Scored<EpisodicMemory>],
    node_id: &str,
) -> Option<&'a Scored<EpisodicMemory>> {
    scored.iter().find(|s| s.item.node_id == node_id)
}

fn get_fact(cm: &CognitiveMemory, node_id: &str) -> SemanticFact {
    cm.get_all_facts(10_000)
        .into_iter()
        .find(|f| f.node_id == node_id)
        .unwrap_or_else(|| panic!("fact {node_id} not found"))
}

/// Case-insensitive substring search across a `Scored` item's reasons.
fn has_reason(reasons: &[String], needle: &str) -> bool {
    let needle = needle.to_lowercase();
    reasons.iter().any(|r| r.to_lowercase().contains(&needle))
}

// ===========================================================================
// 1. Pure scoring primitives (deterministic, no clock, no store)
// ===========================================================================

#[test]
fn keyword_jaccard_partial_overlap() {
    // {rust, async} ∩ {rust, async, programming, tokio} = 2; ∪ = 4 -> 0.5
    let j = keyword_jaccard("rust async", "rust async programming tokio");
    assert!((j - 0.5).abs() < 1e-9, "expected 0.5, got {j}");
}

#[test]
fn keyword_jaccard_identical_is_one() {
    assert!((keyword_jaccard("alpha beta", "beta alpha") - 1.0).abs() < 1e-9);
}

#[test]
fn keyword_jaccard_is_case_insensitive() {
    assert!((keyword_jaccard("Rust ASYNC", "rust async") - 1.0).abs() < 1e-9);
}

#[test]
fn keyword_jaccard_empty_inputs_are_zero() {
    assert_eq!(keyword_jaccard("", "anything"), 0.0);
    assert_eq!(keyword_jaccard("anything", ""), 0.0);
    assert_eq!(keyword_jaccard("", ""), 0.0);
}

#[test]
fn keyword_jaccard_disjoint_is_zero() {
    assert_eq!(keyword_jaccard("rust async", "banana smoothie"), 0.0);
}

#[test]
fn exp_decay_at_zero_age_is_one() {
    assert!((exp_decay(0.0, 100.0) - 1.0).abs() < 1e-12);
}

#[test]
fn exp_decay_at_one_half_life_is_half() {
    // Core recency property used in the Gherkin scenario.
    assert!((exp_decay(100.0, 100.0) - 0.5).abs() < 1e-12);
    assert!((exp_decay(200.0, 100.0) - 0.25).abs() < 1e-12);
}

#[test]
fn exp_decay_nonpositive_half_life_guarded_to_zero() {
    // No div-by-zero / NaN: hl <= 0 yields 0.0.
    assert_eq!(exp_decay(50.0, 0.0), 0.0);
    assert_eq!(exp_decay(50.0, -10.0), 0.0);
}

#[test]
fn exp_decay_is_monotonic_decreasing_in_age() {
    let hl = 7.0 * 86_400.0;
    assert!(exp_decay(0.0, hl) > exp_decay(3600.0, hl));
    assert!(exp_decay(3600.0, hl) > exp_decay(30.0 * 86_400.0, hl));
}

#[test]
fn usage_boost_zero_and_negative_are_zero() {
    assert_eq!(usage_boost(0), 0.0);
    assert_eq!(usage_boost(-5), 0.0);
}

#[test]
fn usage_boost_is_ln_one_plus_n_and_monotonic() {
    assert!((usage_boost(1) - (2.0f64).ln()).abs() < 1e-12);
    assert!(usage_boost(10) > usage_boost(3));
    assert!(usage_boost(3) > usage_boost(1));
    assert!(usage_boost(1) > usage_boost(0));
}

// ===========================================================================
// 2. Defaults (the documented contract for the option/weight structs)
// ===========================================================================

#[test]
fn recall_weights_defaults_match_design() {
    let w = RecallWeights::default();
    assert_eq!(w.text_relevance, 1.0);
    assert_eq!(w.confidence, 0.7);
    assert_eq!(w.importance, 0.5);
    assert_eq!(w.recency, 0.4);
    assert_eq!(w.usage, 0.3);
    assert_eq!(w.graph, 0.6);
}

#[test]
fn recall_options_defaults_match_design() {
    let o = RecallOptions::default();
    assert_eq!(o.limit, 10);
    assert_eq!(o.min_confidence, 0.0);
    assert!(!o.include_archived);
    assert!(!o.include_superseded);
    assert_eq!(o.max_graph_hops, 1);
    assert_eq!(o.recency_half_life_seconds, 604_800.0);
    assert!(o.record_access);
    assert_eq!(o.weights, RecallWeights::default());
}

// ===========================================================================
// 3. Combined-score ranking (I4, I11)
// ===========================================================================

#[test]
fn ranking_orders_by_combined_score() {
    // A recent, high-importance, high-usage fact must outrank an old,
    // low-confidence keyword match even though both mention the query terms.
    let mut cm = make_cm();

    let alpha = cm
        .store_fact("alpha", "rust async notes", 0.2, "", None, None)
        .unwrap();
    let beta = cm
        .store_fact("beta", "rust async deep dive", 0.9, "", None, None)
        .unwrap();

    // Make alpha stale and beta heavily used.
    set_prop(
        &mut cm,
        &alpha,
        "last_accessed_at",
        &(now_secs() - 60 * 86_400).to_string(),
    );
    set_prop(&mut cm, &beta, "usage_count", "5");

    let res = cm
        .recall_facts_ranked("rust async", read_only_opts())
        .unwrap();

    let ra = rank_of_fact(&res, &alpha).expect("alpha present");
    let rb = rank_of_fact(&res, &beta).expect("beta present");
    assert!(rb < ra, "beta (recent/important/used) should outrank alpha");
    assert!(find_fact(&res, &beta).unwrap().score > find_fact(&res, &alpha).unwrap().score);
}

#[test]
fn results_are_sorted_descending_by_score() {
    // I4: i < j => R[i].score >= R[j].score.
    let mut cm = make_cm();
    cm.store_fact("a", "rust async one", 0.9, "", None, None)
        .unwrap();
    cm.store_fact("b", "rust tokio two", 0.5, "", None, None)
        .unwrap();
    cm.store_fact("c", "unrelated three", 0.1, "", None, None)
        .unwrap();

    let res = cm
        .recall_facts_ranked("rust async", read_only_opts())
        .unwrap();
    for w in res.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "scores must be non-increasing: {} then {}",
            w[0].score,
            w[1].score
        );
    }
}

#[test]
fn reasons_are_never_empty() {
    // I6: every returned item carries at least one reason string.
    let mut cm = make_cm();
    cm.store_fact("x", "rust async", 0.9, "", None, None)
        .unwrap();
    let res = cm
        .recall_facts_ranked("rust async", read_only_opts())
        .unwrap();
    assert!(!res.is_empty());
    for s in &res {
        assert!(!s.reasons.is_empty(), "reasons must not be empty");
    }
}

#[test]
fn limit_truncates_results() {
    // I5: |R| <= limit.
    let mut cm = make_cm();
    for i in 0..5 {
        cm.store_fact(&format!("c{i}"), "rust async item", 0.5, "", None, None)
            .unwrap();
    }
    let opts = RecallOptions {
        limit: 2,
        ..read_only_opts()
    };
    let res = cm.recall_facts_ranked("rust async", opts).unwrap();
    assert_eq!(res.len(), 2);
}

#[test]
fn limit_zero_returns_empty() {
    let mut cm = make_cm();
    cm.store_fact("x", "rust async", 0.9, "", None, None)
        .unwrap();
    let opts = RecallOptions {
        limit: 0,
        ..read_only_opts()
    };
    let res = cm.recall_facts_ranked("rust async", opts).unwrap();
    assert!(res.is_empty());
}

// ===========================================================================
// 4. Recency decay (recency term, I7)
// ===========================================================================

#[test]
fn recency_decay_lowers_old_facts() {
    // Two facts equal in everything except last_accessed_at; the recent one
    // must score higher. record_access=false keeps the comparison pure (I7).
    let mut cm = make_cm();
    let recent = cm
        .store_fact("topic", "identical body text", 0.6, "", None, None)
        .unwrap();
    let old = cm
        .store_fact("topic", "identical body text", 0.6, "", None, None)
        .unwrap();

    set_prop(
        &mut cm,
        &old,
        "last_accessed_at",
        &(now_secs() - 30 * 86_400).to_string(),
    );

    let res = cm
        .recall_facts_ranked("identical body", read_only_opts())
        .unwrap();
    let recent_s = find_fact(&res, &recent).unwrap();
    let old_s = find_fact(&res, &old).unwrap();

    assert!(
        recent_s.score > old_s.score,
        "recent {} should beat old {}",
        recent_s.score,
        old_s.score
    );
    assert!(has_reason(&recent_s.reasons, "recency"));
}

// ===========================================================================
// 5. Usage boost (usage term)
// ===========================================================================

#[test]
fn usage_boost_orders_facts() {
    // Two facts equal except usage_count; higher usage scores higher.
    let mut cm = make_cm();
    let low = cm
        .store_fact("topic", "same content here", 0.6, "", None, None)
        .unwrap();
    let high = cm
        .store_fact("topic", "same content here", 0.6, "", None, None)
        .unwrap();

    set_prop(&mut cm, &high, "usage_count", "10");

    let res = cm
        .recall_facts_ranked("same content", read_only_opts())
        .unwrap();
    let high_s = find_fact(&res, &high).unwrap();
    let low_s = find_fact(&res, &low).unwrap();

    assert!(
        high_s.score > low_s.score,
        "higher usage should score higher"
    );
    assert!(has_reason(&high_s.reasons, "usage"));
    // The zero-usage fact contributes no usage term, hence no usage reason.
    assert!(!has_reason(&low_s.reasons, "usage"));
}

// ===========================================================================
// 6. Graph boost via DERIVES_FROM / SIMILAR_TO (graph term, A3)
// ===========================================================================

#[test]
fn graph_boost_similar_neighbor_raises_score() {
    // "core" matches the query strongly; "neighbor" has no textual overlap but
    // is SIMILAR_TO "core", so a 1-hop graph term should lift it.
    let mut cm = make_cm();
    let core = cm
        .store_fact("core", "rust async tokio runtime", 0.0, "", None, None)
        .unwrap();
    let neighbor = cm
        .store_fact("cooking", "banana smoothie recipe", 0.0, "", None, None)
        .unwrap();
    cm.link_similar_facts(&neighbor, &core, 0.9).unwrap();

    let with_graph = RecallOptions {
        max_graph_hops: 1,
        ..read_only_opts()
    };
    let no_graph = RecallOptions {
        max_graph_hops: 0,
        ..read_only_opts()
    };

    let res_g = cm.recall_facts_ranked("rust async", with_graph).unwrap();
    let res_ng = cm.recall_facts_ranked("rust async", no_graph).unwrap();

    let n_g = find_fact(&res_g, &neighbor).unwrap();
    let n_ng = find_fact(&res_ng, &neighbor).unwrap();

    assert!(
        n_g.score > n_ng.score,
        "graph hop should raise neighbor score: {} vs {}",
        n_g.score,
        n_ng.score
    );
    assert!(has_reason(&n_g.reasons, "graph"), "expected a graph reason");
    assert!(
        !has_reason(&n_ng.reasons, "graph"),
        "hops=0 disables graph term"
    );
}

#[test]
fn graph_boost_derives_from_episode() {
    // A fact "core" matches the query; an episode DERIVES_FROM "core" (edge
    // core --DERIVES_FROM--> episode, i.e. Incoming from the episode's view)
    // should be boosted above an unlinked episode with identical text.
    let mut cm = make_cm();
    let core = cm
        .store_fact("core", "rust async tokio", 0.0, "", None, None)
        .unwrap();
    let linked = cm
        .store_episode("did unrelated plumbing work", "ops", Some(1), None)
        .unwrap();
    let unlinked = cm
        .store_episode("did unrelated plumbing work", "ops", Some(2), None)
        .unwrap();
    cm.link_fact_to_episode(&core, &linked).unwrap();

    let opts = RecallOptions {
        max_graph_hops: 1,
        ..read_only_opts()
    };
    let res = cm.recall_episodes_ranked("rust async", opts).unwrap();

    let linked_s = find_episode(&res, &linked).unwrap();
    let unlinked_s = find_episode(&res, &unlinked).unwrap();
    assert!(
        linked_s.score > unlinked_s.score,
        "DERIVES_FROM-linked episode should outrank the unlinked one"
    );
    assert!(has_reason(&linked_s.reasons, "graph"));
}

// ===========================================================================
// 7. Lifecycle exclusion (I2)
// ===========================================================================

#[test]
fn archived_facts_excluded_by_default_and_surfaced_on_opt_in() {
    let mut cm = make_cm();
    let normal = cm
        .store_fact("normal", "rust async normal", 0.8, "", None, None)
        .unwrap();
    let archived = cm
        .store_fact("archived", "rust async archived", 0.8, "", None, None)
        .unwrap();
    set_prop(&mut cm, &archived, "archived", "true");

    let res = cm
        .recall_facts_ranked("rust async", read_only_opts())
        .unwrap();
    assert!(find_fact(&res, &normal).is_some());
    assert!(
        find_fact(&res, &archived).is_none(),
        "archived fact must be excluded by default"
    );

    let opts = RecallOptions {
        include_archived: true,
        ..read_only_opts()
    };
    let res2 = cm.recall_facts_ranked("rust async", opts).unwrap();
    assert!(
        find_fact(&res2, &archived).is_some(),
        "include_archived must surface the archived fact"
    );
}

#[test]
fn superseded_facts_excluded_by_default_and_surfaced_on_opt_in() {
    let mut cm = make_cm();
    let normal = cm
        .store_fact("normal", "rust async normal", 0.8, "", None, None)
        .unwrap();
    let superseded = cm
        .store_fact("old", "rust async superseded", 0.8, "", None, None)
        .unwrap();
    // Mark only superseded_by (independent of the archived flag) to isolate the
    // superseded filter from the archived filter.
    set_prop(&mut cm, &superseded, "superseded_by", "some-newer-id");

    let res = cm
        .recall_facts_ranked("rust async", read_only_opts())
        .unwrap();
    assert!(find_fact(&res, &normal).is_some());
    assert!(
        find_fact(&res, &superseded).is_none(),
        "superseded fact must be excluded by default"
    );

    let opts = RecallOptions {
        include_superseded: true,
        ..read_only_opts()
    };
    let res2 = cm.recall_facts_ranked("rust async", opts).unwrap();
    assert!(
        find_fact(&res2, &superseded).is_some(),
        "include_superseded must surface the superseded fact"
    );
}

#[test]
fn supersede_fact_real_api_excludes_old_by_default() {
    // Realistic path: supersede_fact archives + sets superseded_by on `old`.
    let mut cm = make_cm();
    let old = cm
        .store_fact("topic", "rust async old fact", 0.8, "", None, None)
        .unwrap();
    let new = cm
        .store_fact("topic", "rust async new fact", 0.9, "", None, None)
        .unwrap();
    cm.supersede_fact(&old, &new, "newer info").unwrap();

    let res = cm
        .recall_facts_ranked("rust async", read_only_opts())
        .unwrap();
    assert!(
        find_fact(&res, &old).is_none(),
        "superseded old fact excluded"
    );
    assert!(find_fact(&res, &new).is_some(), "replacement fact retained");
}

#[test]
fn compressed_episodes_excluded() {
    let mut cm = make_cm();
    let normal = cm
        .store_episode("rust async episode normal", "ops", Some(1), None)
        .unwrap();
    let compressed = cm
        .store_episode("rust async episode compressed", "ops", Some(2), None)
        .unwrap();
    set_prop(&mut cm, &compressed, "compressed", "true");

    let res = cm
        .recall_episodes_ranked("rust async", read_only_opts())
        .unwrap();
    assert!(find_episode(&res, &normal).is_some());
    assert!(
        find_episode(&res, &compressed).is_none(),
        "compressed episodes are always excluded"
    );
}

#[test]
fn min_confidence_floor_excludes_low_confidence_facts() {
    // I3: facts below min_confidence are dropped (facts only).
    let mut cm = make_cm();
    let low = cm
        .store_fact("a", "rust async low", 0.2, "", None, None)
        .unwrap();
    let high = cm
        .store_fact("b", "rust async high", 0.9, "", None, None)
        .unwrap();

    let opts = RecallOptions {
        min_confidence: 0.5,
        ..read_only_opts()
    };
    let res = cm.recall_facts_ranked("rust async", opts).unwrap();
    assert!(find_fact(&res, &high).is_some());
    assert!(find_fact(&res, &low).is_none());
}

// ===========================================================================
// 8. Access tracking: record_access + recall side effect (I7, I8, I9)
// ===========================================================================

#[test]
fn record_access_increments_usage_and_sets_last_accessed() {
    // I8: single op bumps usage_count by one and sets last_accessed_at.
    let mut cm = make_cm();
    let id = cm
        .store_fact("topic", "rust async fact", 0.7, "", None, None)
        .unwrap();

    let before = get_fact(&cm, &id);
    assert_eq!(before.usage_count, 0);
    assert!(before.last_accessed_at.is_none());

    cm.record_access(&id, AccessKind::Recall).unwrap();

    let after = get_fact(&cm, &id);
    assert_eq!(after.usage_count, 1);
    assert!(after.last_accessed_at.is_some());
}

#[test]
fn record_access_read_kind_also_increments() {
    let mut cm = make_cm();
    let id = cm
        .store_fact("topic", "rust async fact", 0.7, "", None, None)
        .unwrap();
    cm.record_access(&id, AccessKind::Read).unwrap();
    assert_eq!(get_fact(&cm, &id).usage_count, 1);
}

#[test]
fn record_access_missing_node_is_storage_error() {
    let mut cm = make_cm();
    let err = cm
        .record_access("does-not-exist", AccessKind::Recall)
        .unwrap_err();
    assert!(
        matches!(err, MemoryError::Storage(_)),
        "missing node must yield Storage, got {err:?}"
    );
}

#[test]
fn recall_records_access_for_returned_items_when_enabled() {
    // I9: with record_access=true, each returned item is bumped exactly once,
    // and the *returned* item carries the pre-bump value.
    let mut cm = make_cm();
    let id = cm
        .store_fact("topic", "rust async fact", 0.7, "", None, None)
        .unwrap();

    let opts = RecallOptions {
        record_access: true,
        ..RecallOptions::default()
    };
    let res = cm.recall_facts_ranked("rust async", opts).unwrap();
    let returned = find_fact(&res, &id).expect("fact returned");
    assert_eq!(returned.item.usage_count, 0, "returned item is pre-bump");

    // The stored node, however, has been incremented.
    assert_eq!(get_fact(&cm, &id).usage_count, 1);
}

#[test]
fn recall_does_not_record_access_when_disabled() {
    // I7: record_access=false is a pure read — no mutation.
    let mut cm = make_cm();
    let id = cm
        .store_fact("topic", "rust async fact", 0.7, "", None, None)
        .unwrap();

    let _ = cm
        .recall_facts_ranked("rust async", read_only_opts())
        .unwrap();
    let after = get_fact(&cm, &id);
    assert_eq!(after.usage_count, 0);
    assert!(after.last_accessed_at.is_none());
}

// ===========================================================================
// 9. Tenant isolation (I12, A2/A3/A4)
// ===========================================================================

#[test]
fn cross_agent_recall_excludes_foreign_node() {
    // A node carrying a different agent_id in the same store must never be
    // returned by this agent's ranked recall (A2).
    let mut cm = make_cm();
    let foreign = cm
        .store_fact("foreign", "rust async foreign secret", 0.9, "", None, None)
        .unwrap();
    set_prop(&mut cm, &foreign, "agent_id", "intruder");

    let res = cm
        .recall_facts_ranked("rust async", read_only_opts())
        .unwrap();
    assert!(
        find_fact(&res, &foreign).is_none(),
        "foreign-agent node must be excluded"
    );
}

#[test]
fn cross_agent_record_access_denied_and_leaves_node_unchanged() {
    // A4: record_access on a node owned by another agent returns
    // SecurityViolation and performs no write.
    let mut cm = make_cm();
    let owner = cm.agent_name().to_string();
    let foreign = cm
        .store_fact("foreign", "rust async foreign", 0.9, "", None, None)
        .unwrap();
    set_prop(&mut cm, &foreign, "agent_id", "intruder");

    let err = cm.record_access(&foreign, AccessKind::Recall).unwrap_err();
    assert!(
        matches!(err, MemoryError::SecurityViolation(_)),
        "cross-agent access must be a SecurityViolation, got {err:?}"
    );

    // Reclaim ownership only to read state back and prove nothing was written.
    set_prop(&mut cm, &foreign, "agent_id", &owner);
    let after = get_fact(&cm, &foreign);
    assert_eq!(after.usage_count, 0, "no usage bump on denied access");
    assert!(after.last_accessed_at.is_none());
}

#[test]
fn cross_agent_neighbor_contributes_zero_graph_score() {
    // A3: a SIMILAR_TO neighbor owned by another agent must not contribute to
    // the graph term, even though it matches the query.
    let mut cm = make_cm();
    // anchor has no textual overlap with the query.
    let anchor = cm
        .store_fact("anchor", "banana smoothie recipe", 0.0, "", None, None)
        .unwrap();
    // rich strongly matches the query.
    let rich = cm
        .store_fact("rich", "rust async rust async tokio", 0.0, "", None, None)
        .unwrap();
    cm.link_similar_facts(&anchor, &rich, 0.9).unwrap();

    let opts = RecallOptions {
        max_graph_hops: 1,
        ..read_only_opts()
    };

    // Baseline: rich is owned by this agent -> anchor gets a graph boost.
    let owned = cm.recall_facts_ranked("rust async", opts.clone()).unwrap();
    let anchor_owned = find_fact(&owned, &anchor).unwrap();
    assert!(
        has_reason(&anchor_owned.reasons, "graph"),
        "owned neighbor should produce a graph reason"
    );
    let owned_score = anchor_owned.score;

    // Now make rich foreign: anchor's graph term must drop to zero.
    set_prop(&mut cm, &rich, "agent_id", "intruder");
    let foreign = cm.recall_facts_ranked("rust async", opts).unwrap();
    let anchor_foreign = find_fact(&foreign, &anchor).unwrap();
    assert!(
        !has_reason(&anchor_foreign.reasons, "graph"),
        "foreign neighbor must not contribute a graph reason"
    );
    assert!(
        anchor_foreign.score < owned_score,
        "foreign neighbor must lower anchor score: {} !< {}",
        anchor_foreign.score,
        owned_score
    );
}

// ===========================================================================
// 10. Hostile floats never panic (I13, V3)
// ===========================================================================

#[test]
fn nan_weight_does_not_panic() {
    let mut cm = make_cm();
    cm.store_fact("a", "rust async one", 0.9, "", None, None)
        .unwrap();
    cm.store_fact("b", "rust async two", 0.5, "", None, None)
        .unwrap();

    let opts = RecallOptions {
        weights: RecallWeights {
            text_relevance: f64::NAN,
            ..RecallWeights::default()
        },
        ..read_only_opts()
    };
    let res = cm.recall_facts_ranked("rust async", opts).unwrap();
    assert!(res.len() <= RecallOptions::default().limit);
}

#[test]
fn nan_half_life_does_not_panic() {
    let mut cm = make_cm();
    cm.store_fact("a", "rust async one", 0.9, "", None, None)
        .unwrap();

    let opts = RecallOptions {
        recency_half_life_seconds: f64::NAN,
        ..read_only_opts()
    };
    // Must return Ok without panicking.
    let _ = cm.recall_facts_ranked("rust async", opts).unwrap();
}

// ===========================================================================
// 11. Backward compatibility (I1) — existing keyword search unaffected
// ===========================================================================

#[test]
fn search_facts_still_keyword_only_and_confidence_sorted() {
    // Ranked recall is purely additive: search_facts keeps its old semantics.
    let mut cm = make_cm();
    cm.store_fact("rust", "rust systems language", 0.95, "", None, None)
        .unwrap();
    cm.store_fact("python", "python interpreted", 0.9, "", None, None)
        .unwrap();

    let hits = cm.search_facts("rust", 10, 0.0);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].concept, "rust");
}

// ===========================================================================
// 12. Persistent backend: durability across reopen (I8, I10)
// ===========================================================================

#[cfg(feature = "persistent")]
fn temp_db_path() -> (tempfile::TempDir, std::path::PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("ranked.ladybug");
    (tmp, path)
}

#[cfg(feature = "persistent")]
#[test]
fn record_access_persists_across_reopen() {
    let (_tmp, path) = temp_db_path();

    let id = {
        let mut cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();
        let id = cm
            .store_fact("topic", "rust async durable", 0.8, "", None, None)
            .unwrap();
        cm.record_access(&id, AccessKind::Recall).unwrap();
        cm.checkpoint().unwrap();
        cm.close();
        id
    };

    let cm2 = CognitiveMemory::open_persistent(&path, "agent").unwrap();
    let fact = cm2
        .get_all_facts(100)
        .into_iter()
        .find(|f| f.node_id == id)
        .expect("fact survived reopen");
    assert_eq!(fact.usage_count, 1, "usage_count must persist");
    assert!(
        fact.last_accessed_at.is_some(),
        "last_accessed_at must persist"
    );
}

#[cfg(feature = "persistent")]
#[test]
fn ranked_recall_orders_results_on_persistent_backend() {
    let (_tmp, path) = temp_db_path();
    let mut cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();

    cm.store_fact("match", "rust async tokio", 0.9, "", None, None)
        .unwrap();
    cm.store_fact("other", "banana smoothie", 0.1, "", None, None)
        .unwrap();

    let res = cm
        .recall_facts_ranked("rust async", RecallOptions::default())
        .unwrap();
    assert!(!res.is_empty());
    assert_eq!(
        res[0].item.concept, "match",
        "best keyword match ranks first"
    );
    for s in &res {
        assert!(!s.reasons.is_empty());
    }
}

#[cfg(feature = "persistent")]
#[test]
fn cross_agent_isolation_on_shared_persistent_store() {
    // Two agents share one durable store; agent A must not see agent B's facts,
    // and record_access on B's node is denied.
    let (_tmp, path) = temp_db_path();

    let b_id = {
        let mut cm_b = CognitiveMemory::open_persistent(&path, "agent-b").unwrap();
        let id = cm_b
            .store_fact("secret", "rust async confidential", 0.9, "", None, None)
            .unwrap();
        cm_b.checkpoint().unwrap();
        cm_b.close();
        id
    };

    let mut cm_a = CognitiveMemory::open_persistent(&path, "agent-a").unwrap();
    cm_a.store_fact("mine", "rust async public", 0.9, "", None, None)
        .unwrap();

    let res = cm_a
        .recall_facts_ranked("rust async", read_only_opts())
        .unwrap();
    assert!(
        res.iter().all(|s| s.item.node_id != b_id),
        "agent A must not recall agent B's fact"
    );

    let err = cm_a.record_access(&b_id, AccessKind::Recall).unwrap_err();
    assert!(
        matches!(err, MemoryError::SecurityViolation(_)),
        "record_access on B's node must be denied, got {err:?}"
    );
}
