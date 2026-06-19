//! Persistent-backend tests for [`CognitiveMemory`] over the LadybugDB
//! (`lbug`) `GraphStore`. Gated behind the `persistent` feature.
//!
//! These exercise the four cognitive memory types the de-fork targets
//! (episodes, facts, procedures, prospective triggers), the corrected retrieval
//! semantics (tokenized multi-word recall, keyword-overlap triggers, idempotent
//! procedures), and durability across `close` + reopen.

use std::collections::HashMap;

use crate::cognitive_memory::CognitiveMemory;
use crate::cognitive_memory::WORKING_MEMORY_CAPACITY;

fn temp_db() -> (tempfile::TempDir, std::path::PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("cognitive.ladybug");
    (tmp, path)
}

fn steps(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

#[test]
fn persistent_round_trip_all_types() {
    let (_tmp, path) = temp_db();
    let mut cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();

    // Episode
    let ep = cm
        .store_episode("deployed the service to prod", "ops", Some(1), None)
        .unwrap();
    assert!(!ep.is_empty());
    let episodes = cm.search_episodes(10);
    assert_eq!(episodes.len(), 1);
    assert_eq!(episodes[0].content, "deployed the service to prod");

    // Fact
    cm.store_fact(
        "rust-memory",
        "Rust guarantees memory safety without a garbage collector",
        0.9,
        "book",
        Some(&steps(&["rust", "memory"])),
        None,
    )
    .unwrap();
    let facts = cm.get_all_facts(10);
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].concept, "rust-memory");

    // Procedure
    cm.store_procedure(
        "deploy-service",
        &steps(&["build", "push image", "rollout"]),
        None,
    )
    .unwrap();
    let procs = cm.search_procedures("deploy-service", 10);
    assert_eq!(procs.len(), 1);
    assert_eq!(procs[0].steps, steps(&["build", "push image", "rollout"]));

    // Prospective
    cm.store_prospective("alert on outage", "service outage", "page the on-call", 5)
        .unwrap();

    let stats = cm.get_memory_stats();
    assert_eq!(stats.get("episodic"), Some(&1));
    assert_eq!(stats.get("semantic"), Some(&1));
    assert_eq!(stats.get("procedural"), Some(&1));
    assert_eq!(stats.get("prospective"), Some(&1));
}

#[test]
fn tokenized_multiword_fact_recall() {
    let (_tmp, path) = temp_db();
    let mut cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();

    cm.store_fact(
        "concurrency",
        "The quick brown fox jumps over the lazy dog",
        0.8,
        "src",
        None,
        None,
    )
    .unwrap();

    // Multi-word, out-of-order, mixed-case query: tokens are OR-matched as
    // CONTAINS, so this must still return the fact.
    let hits = cm.search_facts("FOX quick", 10, 0.0);
    assert_eq!(hits.len(), 1, "multi-word tokenized recall should match");

    // OR semantics: a single overlapping token suffices even when another query
    // token is absent (an AND/whole-string match would return 0 here).
    let or_hit = cm.search_facts("FOX elephant", 10, 0.0);
    assert_eq!(or_hit.len(), 1, "one overlapping token should match (OR)");

    // A token that does not appear must not match.
    let misses = cm.search_facts("elephant", 10, 0.0);
    assert!(misses.is_empty());
}

#[test]
fn tokenized_multiword_procedure_recall() {
    let (_tmp, path) = temp_db();
    let mut cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();

    cm.store_procedure(
        "release-pipeline",
        &steps(&["compile sources", "build container image", "deploy"]),
        None,
    )
    .unwrap();

    // Query words match across name + steps, case-insensitively.
    let hits = cm.search_procedures("IMAGE compile", 10);
    assert_eq!(hits.len(), 1, "tokenized procedure recall should match");

    // OR semantics: a single overlapping token suffices.
    let or_hit = cm.search_procedures("image nonexistent", 10);
    assert_eq!(or_hit.len(), 1, "one overlapping token should match (OR)");

    // No overlapping token -> no match.
    assert!(cm.search_procedures("kubernetes helm", 10).is_empty());
}

#[test]
fn procedure_storage_is_idempotent() {
    let (_tmp, path) = temp_db();
    let mut cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();

    let id1 = cm
        .store_procedure("build", &steps(&["step-a"]), None)
        .unwrap();
    let id2 = cm
        .store_procedure("build", &steps(&["step-a", "step-b"]), None)
        .unwrap();

    assert_eq!(id1, id2, "re-storing same name must reuse the node");
    assert_eq!(cm.get_memory_stats().get("procedural"), Some(&1));

    // The second store updated the steps in place.
    let procs = cm.search_procedures("build", 10);
    assert_eq!(procs.len(), 1);
    assert_eq!(procs[0].steps, steps(&["step-a", "step-b"]));
}

#[test]
fn triggers_fire_on_keyword_overlap() {
    let (_tmp, path) = temp_db();
    let mut cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();

    cm.store_prospective(
        "handle timeouts",
        "connection timeout",
        "retry with backoff",
        3,
    )
    .unwrap();

    // Content shares the keyword "timeout" with the trigger condition (not an
    // exact-substring match of the whole condition).
    let fired = cm.check_triggers("the request hit a timeout midway");
    assert_eq!(fired.len(), 1, "keyword overlap should fire the trigger");
    assert_eq!(fired[0].status, "triggered");

    // Non-overlapping content must not fire.
    let none = cm.check_triggers("everything succeeded");
    assert!(none.is_empty());
}

#[test]
fn data_survives_close_and_reopen() {
    let (_tmp, path) = temp_db();

    {
        let mut cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();
        cm.store_episode("first episode", "src", Some(7), None)
            .unwrap();
        cm.store_fact("topic", "durable knowledge", 0.7, "s", None, None)
            .unwrap();
        cm.store_procedure("proc", &steps(&["x"]), None).unwrap();
        cm.store_prospective("p", "alarm bell", "act", 1).unwrap();
        cm.close();
    } // CognitiveMemory dropped -> LadybugDB flushed to disk.

    let mut cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();

    let episodes = cm.search_episodes(10);
    assert_eq!(episodes.len(), 1);
    assert_eq!(episodes[0].content, "first episode");
    assert_eq!(episodes[0].temporal_index, 7);

    let facts = cm.search_facts("durable", 10, 0.0);
    assert_eq!(facts.len(), 1);

    let procs = cm.search_procedures("proc", 10);
    assert_eq!(procs.len(), 1);

    // Trigger stored before the restart still fires after reopen.
    let fired = cm.check_triggers("the alarm went off");
    assert_eq!(fired.len(), 1);

    // Counter recovery: a new auto-indexed episode must come after the
    // persisted temporal_index (7), not collide at 1.
    cm.store_episode("second episode", "src", None, None)
        .unwrap();
    let episodes = cm.search_episodes(10);
    let second = episodes
        .iter()
        .find(|e| e.content == "second episode")
        .unwrap();
    assert!(
        second.temporal_index > 7,
        "counter should recover across reopen"
    );
}

#[test]
fn agents_are_isolated_in_shared_db() {
    let (_tmp, path) = temp_db();

    {
        let mut a = CognitiveMemory::open_persistent(&path, "alice").unwrap();
        a.store_fact("c", "alice fact", 0.5, "s", None, None)
            .unwrap();
        a.close();
    }
    {
        let mut b = CognitiveMemory::open_persistent(&path, "bob").unwrap();
        b.store_fact("c", "bob fact", 0.5, "s", None, None).unwrap();
        // Bob must not see Alice's facts.
        assert_eq!(b.get_all_facts(10).len(), 1);
        assert_eq!(b.get_all_facts(10)[0].content, "bob fact");
        b.close();
    }

    let a = CognitiveMemory::open_persistent(&path, "alice").unwrap();
    let facts = a.get_all_facts(10);
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].content, "alice fact");
}

#[test]
fn metadata_round_trips_through_persistence() {
    let (_tmp, path) = temp_db();
    let mut meta = HashMap::new();
    meta.insert("severity".to_string(), serde_json::json!("high"));

    {
        let mut cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();
        cm.store_episode("an event", "src", Some(1), Some(&meta))
            .unwrap();
        cm.close();
    }

    let cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();
    let episodes = cm.search_episodes(10);
    assert_eq!(episodes.len(), 1);
    assert_eq!(
        episodes[0]
            .metadata
            .get("severity")
            .unwrap()
            .as_str()
            .unwrap(),
        "high"
    );
}

// ---------------------------------------------------------------------------
// Gap 1: distilled-episode flag (mark_episode_distilled / list_undistilled_episodes)
// ---------------------------------------------------------------------------

#[test]
fn distilled_flag_persists_across_reopen() {
    let (_tmp, path) = temp_db();
    let ep;
    {
        let mut cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();
        ep = cm
            .store_episode("episode to distill", "src", Some(1), None)
            .unwrap();

        // A freshly stored episode is undistilled.
        let undistilled = cm.list_undistilled_episodes(10);
        assert_eq!(undistilled.len(), 1);
        assert!(!undistilled[0].distilled);

        // Mark it distilled (one-way latch).
        assert!(cm.mark_episode_distilled(&ep));
        cm.close();
    } // dropped -> flushed to disk

    let cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();
    // The distilled flag survived the reopen.
    let all = cm.search_episodes(10);
    assert_eq!(all.len(), 1);
    assert!(
        all[0].distilled,
        "distilled flag must persist across close + reopen"
    );
    // And the distilled episode is excluded from the undistilled list.
    assert!(
        cm.list_undistilled_episodes(10).is_empty(),
        "distilled episodes must not appear in list_undistilled_episodes"
    );
}

#[test]
fn list_undistilled_episodes_orders_newest_first_and_honors_limit() {
    let (_tmp, path) = temp_db();
    let mut cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();

    let e1 = cm.store_episode("first", "src", None, None).unwrap(); // temporal_index 1
    cm.store_episode("second", "src", None, None).unwrap(); // 2
    cm.store_episode("third", "src", None, None).unwrap(); // 3

    // Distill the oldest episode.
    assert!(cm.mark_episode_distilled(&e1));

    let undistilled = cm.list_undistilled_episodes(10);
    assert_eq!(undistilled.len(), 2, "distilled episode must be excluded");
    // Newest-first ordering by temporal_index.
    assert_eq!(undistilled[0].content, "third");
    assert_eq!(undistilled[1].content, "second");
    assert!(undistilled.iter().all(|e| !e.distilled));

    // The limit argument is honored.
    let limited = cm.list_undistilled_episodes(1);
    assert_eq!(limited.len(), 1);
    assert_eq!(limited[0].content, "third");
}

#[test]
fn mark_episode_distilled_latch_and_ownership() {
    let (_tmp, path) = temp_db();
    let mut alice = CognitiveMemory::open_persistent(&path, "alice").unwrap();
    let ep = alice
        .store_episode("alice event", "src", Some(1), None)
        .unwrap();

    // Unknown id is rejected.
    assert!(!alice.mark_episode_distilled("epi_does_not_exist"));

    // Owned episode is marked, and the latch is idempotent (stays true).
    assert!(alice.mark_episode_distilled(&ep));
    assert!(alice.mark_episode_distilled(&ep));

    // A different agent sharing the DB cannot distill alice's episode.
    let mut bob = CognitiveMemory::open_persistent(&path, "bob").unwrap();
    assert!(
        !bob.mark_episode_distilled(&ep),
        "cross-agent distill must be rejected (ownership check)"
    );

    // Alice's episode is still distilled; bob owns nothing undistilled.
    assert!(alice.list_undistilled_episodes(10).is_empty());
    assert!(bob.list_undistilled_episodes(10).is_empty());
}

// ---------------------------------------------------------------------------
// Gap 2: procedure reinforcement on recall (usage_count increments + ordering)
// ---------------------------------------------------------------------------

#[test]
fn recall_procedure_increments_usage_and_persists() {
    let (_tmp, path) = temp_db();
    {
        let mut cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();
        cm.store_procedure("deploy", &steps(&["build", "ship"]), None)
            .unwrap();

        // First recall reinforces the matched procedure (stored count -> 1).
        let r1 = cm.recall_procedure("deploy", 10);
        assert_eq!(r1.len(), 1);
        assert_eq!(cm.search_procedures("deploy", 10)[0].usage_count, 1);

        // Second recall bumps it again (stored count -> 2).
        cm.recall_procedure("deploy", 10);
        assert_eq!(cm.search_procedures("deploy", 10)[0].usage_count, 2);
        cm.close();
    }

    // The reinforced usage_count persisted across the reopen.
    let cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();
    assert_eq!(
        cm.search_procedures("deploy", 10)[0].usage_count,
        2,
        "reinforced usage_count must persist across reopen"
    );
}

#[test]
fn recall_procedure_orders_by_usage_desc_and_persists() {
    let (_tmp, path) = temp_db();
    {
        let mut cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();
        // Both procedures share the keyword "shared" via their steps.
        cm.store_procedure("alpha task", &steps(&["shared work"]), None)
            .unwrap();
        cm.store_procedure("beta task", &steps(&["shared work"]), None)
            .unwrap();

        // Reinforce alpha three times via a query that matches only alpha.
        for _ in 0..3 {
            let only_alpha = cm.recall_procedure("alpha", 10);
            assert_eq!(only_alpha.len(), 1, "'alpha' matches only the alpha task");
            assert_eq!(only_alpha[0].name, "alpha task");
        }

        // A query matching both returns them ordered by usage_count descending.
        let both = cm.recall_procedure("shared", 10);
        assert_eq!(both.len(), 2);
        assert_eq!(
            both[0].name, "alpha task",
            "more-used procedure ranks first"
        );
        assert_eq!(both[1].name, "beta task");
        assert_eq!(both[0].usage_count, 3);
        assert_eq!(both[1].usage_count, 0);
        cm.close();
    }

    // After reopen the persisted usage drives the same ordering.
    let cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();
    let ranked = cm.search_procedures("shared", 10);
    assert_eq!(ranked.len(), 2);
    assert_eq!(ranked[0].name, "alpha task");
    assert_eq!(ranked[1].name, "beta task");
    // alpha: 3 (alpha-only recalls) + 1 (the shared recall) = 4; beta: 1 (shared recall).
    assert_eq!(ranked[0].usage_count, 4);
    assert_eq!(ranked[1].usage_count, 1);
}

// ---------------------------------------------------------------------------
// Gap 3: episodic content keyword search (case-insensitive substring)
// ---------------------------------------------------------------------------

#[test]
fn search_episodes_by_keyword_is_case_insensitive_substring() {
    let (_tmp, path) = temp_db();
    let mut cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();

    cm.store_episode("Deployed the Payment service to PROD", "ops", None, None)
        .unwrap();
    cm.store_episode("rolled back the cache layer", "ops", None, None)
        .unwrap();
    cm.store_episode("Deployed the Payment service again", "ops", None, None)
        .unwrap();

    // Case-insensitive substring match over content, newest-first.
    let hits = cm.search_episodes_by_keyword("payment", 10);
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].content, "Deployed the Payment service again");
    assert_eq!(hits[1].content, "Deployed the Payment service to PROD");

    // Substring (not whole-token) match, case-insensitive.
    let prod = cm.search_episodes_by_keyword("prod", 10);
    assert_eq!(prod.len(), 1);
    assert_eq!(prod[0].content, "Deployed the Payment service to PROD");

    // A non-matching query returns nothing.
    assert!(cm.search_episodes_by_keyword("kubernetes", 10).is_empty());

    // The limit argument is honored.
    assert_eq!(cm.search_episodes_by_keyword("payment", 1).len(), 1);

    // Compressed episodes are excluded from keyword search.
    cm.consolidate_episodes::<fn(&[String]) -> String>(3, None)
        .unwrap();
    assert!(
        cm.search_episodes_by_keyword("payment", 10).is_empty(),
        "compressed episodes must be excluded from keyword search"
    );
}

// ---------------------------------------------------------------------------
// Sensory + working durability (round-trip across reopen)
// ---------------------------------------------------------------------------

#[test]
fn sensory_round_trip_and_ttl_prune_across_reopen() {
    let (_tmp, path) = temp_db();
    let valid_id;
    {
        let mut cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();
        // Long TTL survives; non-positive TTL is already expired (deterministic, no sleep).
        valid_id = cm.store_sensory("text", "keep me", 3600).unwrap();
        cm.store_sensory("text", "drop me", 0).unwrap();

        // The expired item is never returned by get_sensory, even before prune.
        let live = cm.get_sensory(10);
        assert_eq!(live.len(), 1);
        assert_eq!(live[0].raw_data, "keep me");
        cm.close();
    }

    let mut cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();
    // The valid item survived the reopen.
    let recalled = cm.get_sensory(10);
    assert_eq!(recalled.len(), 1);
    assert_eq!(recalled[0].node_id, valid_id);
    assert_eq!(recalled[0].raw_data, "keep me");

    // Pruning removes the persisted-but-expired item and keeps the valid one.
    let pruned = cm.prune_expired_sensory();
    assert_eq!(pruned, 1, "exactly the expired sensory item is pruned");
    assert_eq!(cm.get_sensory(10).len(), 1);
}

#[test]
fn working_round_trip_and_capacity_eviction_across_reopen() {
    let (_tmp, path) = temp_db();
    {
        let mut cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();
        cm.store_working("goal", "primary goal", "task-1", 0.9)
            .unwrap();
        cm.store_working("context", "some context", "task-1", 0.4)
            .unwrap();
        // A different task is isolated.
        cm.store_working("goal", "other-task goal", "task-2", 0.5)
            .unwrap();
        cm.close();
    }

    let mut cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();
    // Slots persisted and are returned ordered by relevance descending.
    let slots = cm.get_working("task-1");
    assert_eq!(slots.len(), 2);
    assert_eq!(slots[0].content, "primary goal");
    assert_eq!(slots[1].content, "some context");
    assert_eq!(cm.get_working("task-2").len(), 1);

    // Capacity eviction: filling beyond capacity drops the least-relevant slot.
    cm.clear_working("task-1");
    for i in 0..WORKING_MEMORY_CAPACITY {
        // Relevance increases with i, so "slot-0" is the least relevant.
        let rel = (i as f64 + 1.0) / (WORKING_MEMORY_CAPACITY as f64 + 1.0);
        cm.store_working("context", &format!("slot-{i}"), "task-cap", rel)
            .unwrap();
    }
    assert_eq!(cm.get_working("task-cap").len(), WORKING_MEMORY_CAPACITY);

    // One more high-relevance slot evicts the least-relevant existing slot ("slot-0").
    cm.store_working("context", "newcomer", "task-cap", 1.0)
        .unwrap();
    let capped = cm.get_working("task-cap");
    assert_eq!(
        capped.len(),
        WORKING_MEMORY_CAPACITY,
        "working capacity must be enforced"
    );
    assert!(
        !capped.iter().any(|s| s.content == "slot-0"),
        "least-relevant slot must be evicted at capacity"
    );
    assert!(capped.iter().any(|s| s.content == "newcomer"));
}

// ---------------------------------------------------------------------------
// Gap 4: episode temporal ordering is monotonic and persists
// ---------------------------------------------------------------------------

#[test]
fn episode_temporal_index_is_monotonic_and_persists() {
    let (_tmp, path) = temp_db();
    {
        let mut cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();
        cm.store_episode("e1", "src", None, None).unwrap();
        cm.store_episode("e2", "src", None, None).unwrap();
        cm.store_episode("e3", "src", None, None).unwrap();

        let eps = cm.get_episodes(10, false);
        // Newest-first, with strictly increasing temporal indices.
        assert_eq!(eps[0].content, "e3");
        assert_eq!(eps[1].content, "e2");
        assert_eq!(eps[2].content, "e1");
        assert!(eps[0].temporal_index > eps[1].temporal_index);
        assert!(eps[1].temporal_index > eps[2].temporal_index);
        // Auto-index must be a real monotonic counter, never the always-zero default.
        assert!(eps.iter().all(|e| e.temporal_index > 0));
        cm.close();
    }

    // Counter recovers after reopen: a new auto-index exceeds all persisted ones.
    let mut cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();
    let prev_max = cm
        .get_episodes(10, false)
        .iter()
        .map(|e| e.temporal_index)
        .max()
        .unwrap();
    cm.store_episode("e4", "src", None, None).unwrap();
    let e4 = cm
        .get_episodes(10, false)
        .into_iter()
        .find(|e| e.content == "e4")
        .unwrap();
    assert!(
        e4.temporal_index > prev_max,
        "auto temporal_index must keep increasing across reopen"
    );

    // Consolidation is deterministic oldest-first: the two oldest are compressed.
    let cons = cm
        .consolidate_episodes(2, Some(|c: &[String]| c.join(",")))
        .unwrap()
        .unwrap();
    assert!(cons.starts_with("con_"));
    let remaining = cm.get_episodes(10, false);
    assert!(remaining.iter().any(|e| e.content == "e3"));
    assert!(remaining.iter().any(|e| e.content == "e4"));
    assert!(
        !remaining.iter().any(|e| e.content == "e1"),
        "oldest episode (e1) must be consolidated first"
    );
    assert!(!remaining.iter().any(|e| e.content == "e2"));
}

// ---------------------------------------------------------------------------
// Cypher-injection safety: special characters round-trip through persistence
// ---------------------------------------------------------------------------

#[test]
fn special_characters_round_trip_safely() {
    let (_tmp, path) = temp_db();

    // Content and a procedure name laced with Cypher-significant characters:
    // single quotes, double quotes, backslashes, and a newline. If inputs were
    // not escaped these would corrupt the generated Cypher (or inject).
    let nasty_content = "O'Brien said: \"DROP\" \\ all // tables\nline2";
    let nasty_name = "deploy '; MATCH (n) DETACH DELETE n; //";
    let nasty_step = "rm -rf / && echo \"don't\"";
    let ep;

    {
        let mut cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();
        ep = cm
            .store_episode(nasty_content, "src", Some(1), None)
            .unwrap();
        cm.store_procedure(nasty_name, &steps(&[nasty_step]), None)
            .unwrap();
        cm.store_fact("c", nasty_content, 0.9, "s", None, None)
            .unwrap();
        cm.close();
    }

    let mut cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();

    // The episode survived intact (no truncation, no corruption, no deletion).
    let episodes = cm.search_episodes(10);
    assert_eq!(episodes.len(), 1, "no injection deleted the data");
    assert_eq!(episodes[0].node_id, ep);
    assert_eq!(episodes[0].content, nasty_content);

    // Keyword search over the escaped content still matches a literal substring.
    assert_eq!(cm.search_episodes_by_keyword("O'Brien", 10).len(), 1);

    // The procedure name and step round-tripped verbatim.
    let procs = cm.search_procedures("deploy", 10);
    assert_eq!(procs.len(), 1);
    assert_eq!(procs[0].name, nasty_name);
    assert_eq!(procs[0].steps, steps(&[nasty_step]));

    // Distillation still works on the special-character episode.
    assert!(cm.mark_episode_distilled(&ep));
    assert!(cm.list_undistilled_episodes(10).is_empty());

    // The fact persisted with its special-character content intact.
    let facts = cm.get_all_facts(10);
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].content, nasty_content);
}
