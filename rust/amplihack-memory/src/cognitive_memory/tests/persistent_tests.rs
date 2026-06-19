//! Persistent-backend tests for [`CognitiveMemory`] over the LadybugDB
//! (`lbug`) `GraphStore`. Gated behind the `persistent` feature.
//!
//! These exercise the four cognitive memory types the de-fork targets
//! (episodes, facts, procedures, prospective triggers), the corrected retrieval
//! semantics (tokenized multi-word recall, keyword-overlap triggers, idempotent
//! procedures), and durability across `close` + reopen.

use std::collections::HashMap;

use crate::cognitive_memory::CognitiveMemory;

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
