//! End-to-end conformance suite for the persistent `CognitiveMemory` backend.
//!
//! Every one of the six cognitive memory types is exercised against a
//! LadybugDB-backed `CognitiveMemory::open_persistent` instance, after which the
//! store is DROPPED and RE-OPENED at the same path to prove the system is durable
//! across runs (issue #86 — Simard's sole memory backend).
//!
//! Gated behind the `persistent` feature.

use std::collections::HashMap;

use crate::cognitive_memory::CognitiveMemory;

fn steps(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

#[test]
fn all_six_memory_types_persist_across_drop_and_reopen() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("conformance.ladybug");

    let episode_id;

    // --- Phase 1: populate every memory type, then drop the store. ---
    {
        let mut cm = CognitiveMemory::open_persistent(&path, "conformance-agent").unwrap();

        // sensory: one durable, one already-expired (deterministic TTL, no sleeps).
        cm.store_sensory("text", "durable observation", 3600)
            .unwrap();
        cm.store_sensory("text", "expired observation", 0).unwrap();

        // working
        cm.store_working("goal", "finish the migration", "task-1", 0.95)
            .unwrap();

        // episodic: with metadata and an auto-assigned temporal index.
        let mut meta = HashMap::new();
        meta.insert("severity".to_string(), serde_json::json!("high"));
        episode_id = cm
            .store_episode("migrated the database", "ops", None, Some(&meta))
            .unwrap();

        // semantic
        cm.store_fact(
            "rust",
            "Rust has no garbage collector",
            0.9,
            "book",
            Some(&steps(&["rust", "memory"])),
            None,
        )
        .unwrap();

        // procedural
        cm.store_procedure("deploy", &steps(&["build", "ship"]), None)
            .unwrap();

        // prospective
        cm.store_prospective(
            "watch for outage",
            "service outage detected",
            "page on-call",
            5,
        )
        .unwrap();

        cm.close();
    } // CognitiveMemory dropped -> LadybugDB flushed to disk.

    // --- Phase 2: reopen and assert everything survived with fields intact. ---
    let mut cm = CognitiveMemory::open_persistent(&path, "conformance-agent").unwrap();

    // sensory: the durable item is recalled; the expired item prunes.
    let sensory = cm.get_sensory(10);
    assert_eq!(
        sensory.len(),
        1,
        "only the non-expired sensory item survives recall"
    );
    assert_eq!(sensory[0].raw_data, "durable observation");
    assert_eq!(
        cm.prune_expired_sensory(),
        1,
        "the persisted-but-expired sensory item is pruned after reopen"
    );

    // working: the slot is recalled with its fields intact.
    let working = cm.get_working("task-1");
    assert_eq!(working.len(), 1);
    assert_eq!(working[0].content, "finish the migration");
    assert!((working[0].relevance - 0.95).abs() < 1e-9);

    // episodic: content + metadata + temporal index intact; keyword-searchable; undistilled.
    let episodes = cm.search_episodes(10);
    assert_eq!(episodes.len(), 1);
    assert_eq!(episodes[0].content, "migrated the database");
    assert_eq!(episodes[0].node_id, episode_id);
    assert!(episodes[0].temporal_index > 0);
    assert!(!episodes[0].distilled);
    assert_eq!(
        episodes[0]
            .metadata
            .get("severity")
            .and_then(|v| v.as_str()),
        Some("high")
    );
    assert_eq!(cm.search_episodes_by_keyword("database", 10).len(), 1);
    assert_eq!(cm.list_undistilled_episodes(10).len(), 1);

    // distillation latch works within the reopened store.
    assert!(cm.mark_episode_distilled(&episode_id));
    assert!(cm.list_undistilled_episodes(10).is_empty());

    // semantic: tokenized OR recall matches the persisted fact.
    let facts = cm.search_facts("rust nonexistent", 10, 0.0);
    assert_eq!(facts.len(), 1, "OR-token recall matches the persisted fact");
    assert_eq!(facts[0].concept, "rust");

    // procedural: recall reinforces usage (persisted increment).
    let procs = cm.recall_procedure("deploy", 10);
    assert_eq!(procs.len(), 1);
    assert_eq!(
        cm.search_procedures("deploy", 10)[0].usage_count,
        1,
        "recall must reinforce the procedure's usage_count"
    );

    // prospective: trigger fires on keyword overlap, is one-shot, then resolves.
    let fired = cm.check_triggers("we have a service outage right now");
    assert_eq!(fired.len(), 1);
    assert_eq!(fired[0].status, "triggered");
    assert!(
        cm.check_triggers("another service outage").is_empty(),
        "a triggered prospective is one-shot and must not re-fire"
    );
    cm.resolve_prospective(&fired[0].node_id);

    // Every type is counted in the stats after the round-trip.
    let stats = cm.get_memory_stats();
    assert_eq!(stats.get("sensory"), Some(&1));
    assert_eq!(stats.get("working"), Some(&1));
    assert_eq!(stats.get("episodic"), Some(&1));
    assert_eq!(stats.get("semantic"), Some(&1));
    assert_eq!(stats.get("procedural"), Some(&1));
    assert_eq!(stats.get("prospective"), Some(&1));
}
