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

// ---------------------------------------------------------------------------
// Durability: recovery from an unclean shutdown + checkpoint API (issue #88)
// ---------------------------------------------------------------------------

use std::fs;
use std::path::Path;

use crate::graph::lbug_store::wal_path_for;

/// Write one record of every memory type, tagged so seeds are distinguishable.
fn write_all_types(cm: &mut CognitiveMemory, tag: &str) {
    cm.store_episode(&format!("episode {tag}"), "ops", Some(1), None)
        .unwrap();
    cm.store_fact(
        &format!("concept-{tag}"),
        &format!("durable knowledge {tag}"),
        0.9,
        "book",
        Some(&steps(&["rust", "memory"])),
        None,
    )
    .unwrap();
    cm.store_procedure(&format!("proc-{tag}"), &steps(&["a", "b", "c"]), None)
        .unwrap();
    cm.store_prospective(&format!("trigger {tag}"), "outage", "page on-call", 5)
        .unwrap();
    cm.store_sensory("text", &format!("observation {tag}"), 3600)
        .unwrap();
    cm.store_working("note", &format!("scratch {tag}"), "task-1", 0.5)
        .unwrap();
}

/// Copy every file in `from` into `to` — a snapshot of a killed process' files.
fn copy_dir_files(from: &Path, to: &Path) {
    for entry in fs::read_dir(from).unwrap() {
        let p = entry.unwrap().path();
        if p.is_file() {
            fs::copy(&p, to.join(p.file_name().unwrap())).unwrap();
        }
    }
}

/// Truncate `wal` mid-record so a strict reopen fails to replay it.
fn corrupt_wal_tail(wal: &Path) {
    let len = fs::metadata(wal).unwrap().len();
    assert!(len > 64, "WAL must be non-empty to corrupt (was {len})");
    let f = fs::OpenOptions::new().write(true).open(wal).unwrap();
    f.set_len(len - 41).unwrap();
    f.sync_all().unwrap();
}

/// Return the first sibling of `wal` that looks like a quarantined corrupt WAL.
fn find_quarantine(dir: &Path) -> Option<std::path::PathBuf> {
    fs::read_dir(dir).unwrap().flatten().find_map(|e| {
        let p = e.path();
        if p.file_name()?.to_string_lossy().contains(".wal.corrupt-") {
            Some(p)
        } else {
            None
        }
    })
}

#[test]
fn unclean_shutdown_recovers_without_crash_and_keeps_checkpointed_records() {
    let live = tempfile::tempdir().unwrap();
    let live_db = live.path().join("cognitive.ladybug");

    // 1. Open, write across all memory types, and CHECKPOINT so this batch is
    //    durable in the main DB file (survives even a destroyed WAL).
    let mut cm = CognitiveMemory::open_persistent(&live_db, "agent").unwrap();
    write_all_types(&mut cm, "durable");
    cm.checkpoint().unwrap();
    let durable = cm.get_memory_stats();
    let durable_total = *durable.get("total").unwrap();
    assert!(durable_total >= 6, "expected >= 6 durable records");

    // 2. Write more records that live only in the WAL (uncheckpointed). The
    //    default auto-checkpoint interval is far higher than this, so nothing is
    //    flushed implicitly.
    write_all_types(&mut cm, "wal-only");

    // 3. Simulate an UNCLEAN shutdown: snapshot the on-disk files mid-WAL, then
    //    abandon the handle without a clean close (no final checkpoint).
    let crash = tempfile::tempdir().unwrap();
    copy_dir_files(live.path(), crash.path());
    std::mem::forget(cm);

    let crash_db = crash.path().join("cognitive.ladybug");
    let crash_wal = wal_path_for(&crash_db);
    assert!(crash_wal.exists(), "snapshot must contain the live WAL");
    corrupt_wal_tail(&crash_wal);

    // 4. A plain strict reopen of the corrupt copy fails — this is the incident.
    assert!(
        crate::graph::lbug_store::LbugGraphStore::open(&crash_db, Some("cognitive-agent")).is_err(),
        "strict open of the corrupt WAL should fail"
    );

    // 5. The recovery path opens successfully (no crash) and returns at least the
    //    checkpointed records — no total loss.
    let cm2 = CognitiveMemory::open_persistent_with_recovery(&crash_db, "agent")
        .expect("recovery open must succeed");
    let stats = cm2.get_memory_stats();
    assert!(
        *stats.get("total").unwrap() >= durable_total,
        "recovery must return >= the {durable_total} checkpointed records, got {:?}",
        stats.get("total")
    );
    for ty in [
        "episodic",
        "semantic",
        "procedural",
        "prospective",
        "sensory",
        "working",
    ] {
        assert!(
            stats.get(ty).copied().unwrap_or(0) >= 1,
            "checkpointed {ty} record must survive recovery: {stats:?}"
        );
    }

    // 6. The corrupt WAL was moved aside, never deleted.
    assert!(
        find_quarantine(crash.path()).is_some(),
        "a <wal>.corrupt-<ts> artifact must exist after recovery"
    );

    // 7. The transparent entry point (`open_persistent`) also recovers, proving
    //    existing callers gain crash-resilience with no code change.
    drop(cm2);
    let cm3 = CognitiveMemory::open_persistent(&crash_db, "agent")
        .expect("open_persistent must transparently recover");
    assert!(*cm3.get_memory_stats().get("total").unwrap() >= durable_total);
}

#[test]
fn checkpoint_then_reopen_returns_all_records_and_needs_no_replay() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("cognitive.ladybug");
    let wal = wal_path_for(&path);

    let expected_total;
    {
        let mut cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();
        write_all_types(&mut cm, "a");
        write_all_types(&mut cm, "b");
        cm.checkpoint().unwrap();
        expected_total = *cm.get_memory_stats().get("total").unwrap();

        // After an explicit checkpoint there is no WAL left to replay.
        let wal_len = fs::metadata(&wal).map(|m| m.len()).unwrap_or(0);
        assert!(
            wal_len == 0 || !wal.exists(),
            "WAL should be empty/absent after checkpoint, was {wal_len} bytes"
        );
        std::mem::forget(cm); // no clean close: rely solely on the checkpoint
    }

    // A reopen sees every record even though the handle was never closed.
    let cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();
    assert_eq!(*cm.get_memory_stats().get("total").unwrap(), expected_total);
}

// ===========================================================================
// Provenance edges (issue #90): durability across close + reopen
//
// File: rust/amplihack-memory/src/cognitive_memory/tests/persistent_tests.rs
//
// Failing-first TDD tests proving DERIVES_FROM / PROCEDURE_DERIVES_FROM edges
// survive a drop + reopen on the LadybugDB persistent backend (which also
// proves the new PROCEDURE_DERIVES_FROM rel table is reintrospected on reopen).
// Gated by the `persistent` feature via the parent `tests` module.
// (Invariant I7.)
// ===========================================================================

#[test]
fn fact_provenance_edge_survives_reopen() {
    use crate::cognitive_memory::types::ET_DERIVES_FROM;
    use crate::graph::Direction;

    let (_tmp, path) = temp_db();
    let fact_id;
    let ep_id;
    {
        let mut cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();
        ep_id = cm
            .store_episode("source episode", "src", Some(1), None)
            .unwrap();
        fact_id = cm
            .store_fact_with_provenance(
                "concept",
                "derived knowledge",
                0.9,
                "",
                None,
                None,
                std::slice::from_ref(&ep_id),
            )
            .unwrap();

        // Edge present before close.
        assert_eq!(cm.fact_provenance(&fact_id), vec![ep_id.clone()]);
        cm.close();
    } // dropped -> LadybugDB flushed to disk

    let cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();

    // The DERIVES_FROM edge survived close + reopen, observable via both the
    // public read path and a raw neighbor query.
    assert_eq!(
        cm.fact_provenance(&fact_id),
        vec![ep_id.clone()],
        "fact provenance edge must persist across close + reopen"
    );
    let neighbors =
        cm.graph
            .query_neighbors(&fact_id, Some(ET_DERIVES_FROM), Direction::Outgoing, 10);
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0].1.node_id, ep_id);
}

#[test]
fn procedure_provenance_edge_survives_reopen() {
    use crate::cognitive_memory::types::ET_PROCEDURE_DERIVES_FROM;
    use crate::graph::Direction;

    let (_tmp, path) = temp_db();
    let proc_id;
    let ep_id;
    {
        let mut cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();
        ep_id = cm
            .store_episode("deploy event", "ops", Some(1), None)
            .unwrap();
        proc_id = cm
            .store_procedure_with_provenance(
                "deploy",
                &steps(&["build", "rollout"]),
                None,
                std::slice::from_ref(&ep_id),
            )
            .unwrap();

        assert_eq!(cm.procedure_provenance(&proc_id), vec![ep_id.clone()]);
        cm.close();
    } // dropped -> LadybugDB flushed to disk

    let cm = CognitiveMemory::open_persistent(&path, "agent").unwrap();

    // The PROCEDURE_DERIVES_FROM edge survived reopen; reaching it also proves
    // the new rel table was reintrospected.
    assert_eq!(
        cm.procedure_provenance(&proc_id),
        vec![ep_id.clone()],
        "procedure provenance edge must persist across close + reopen"
    );
    let neighbors = cm.graph.query_neighbors(
        &proc_id,
        Some(ET_PROCEDURE_DERIVES_FROM),
        Direction::Outgoing,
        10,
    );
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0].1.node_id, ep_id);
}
