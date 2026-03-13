use super::types::{agent_filter, new_id, ts_now, ET_DERIVES_FROM, NT_PROSPECTIVE};
use super::*;
use crate::graph::types::Direction;

fn make_cm() -> CognitiveMemory {
    CognitiveMemory::new(&format!("test-agent-{}", uuid::Uuid::new_v4())).unwrap()
}

// -- construction -------------------------------------------------------

#[test]
fn test_new_empty_agent_name_rejected() {
    assert!(CognitiveMemory::new("").is_err());
    assert!(CognitiveMemory::new("   ").is_err());
}

#[test]
fn test_new_valid_agent_name() {
    let cm = CognitiveMemory::new("  alice  ").unwrap();
    assert_eq!(cm.agent_name(), "alice");
}

// -- helpers ------------------------------------------------------------

#[test]
fn test_new_id_format() {
    let id = new_id("sen");
    assert!(id.starts_with("sen_"));
    assert_eq!(id.len(), 4 + 12); // prefix_ + 12 hex chars
}

#[test]
fn test_ts_now_positive() {
    assert!(ts_now() > 0);
}

// -- sensory memory -----------------------------------------------------

#[test]
fn test_store_and_get_sensory() {
    let mut cm = make_cm();
    let id = cm.store_sensory("text", "hello world", 300).unwrap();
    assert!(id.starts_with("sen_"));

    let items = cm.get_sensory(10);
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].modality, "text");
    assert_eq!(items[0].raw_data, "hello world");
}

#[test]
fn test_sensory_ordering() {
    let mut cm = make_cm();
    cm.store_sensory("a", "first", 300).unwrap();
    cm.store_sensory("b", "second", 300).unwrap();
    cm.store_sensory("c", "third", 300).unwrap();

    let items = cm.get_sensory(10);
    assert_eq!(items.len(), 3);
    // Most recent first
    assert_eq!(items[0].modality, "c");
    assert_eq!(items[1].modality, "b");
    assert_eq!(items[2].modality, "a");
}

#[test]
fn test_sensory_limit() {
    let mut cm = make_cm();
    for i in 0..5 {
        cm.store_sensory("text", &format!("item-{i}"), 300).unwrap();
    }
    let items = cm.get_sensory(2);
    assert_eq!(items.len(), 2);
}

#[test]
fn test_prune_expired_sensory() {
    let mut cm = make_cm();
    // Use 0 TTL so it expires immediately
    cm.store_sensory("text", "ephemeral", 0).unwrap();
    // Give it a moment
    std::thread::sleep(std::time::Duration::from_millis(1100));

    // This item has a long TTL and should survive
    cm.store_sensory("text", "persistent", 9999).unwrap();

    let pruned = cm.prune_expired_sensory();
    assert_eq!(pruned, 1);

    let remaining = cm.get_sensory(10);
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].raw_data, "persistent");
}

#[test]
fn test_attend_to_sensory() {
    let mut cm = make_cm();
    let sid = cm.store_sensory("error", "segfault", 300).unwrap();
    let ep_id = cm.attend_to_sensory(&sid, "critical error");
    assert!(ep_id.is_some());

    let episodes = cm.get_episodes(10, false);
    assert_eq!(episodes.len(), 1);
    assert!(episodes[0].content.contains("segfault"));
    assert!(episodes[0].content.contains("critical error"));
}

#[test]
fn test_attend_to_missing_sensory() {
    let mut cm = make_cm();
    assert!(cm.attend_to_sensory("nonexistent", "reason").is_none());
}

// -- working memory -----------------------------------------------------

#[test]
fn test_store_and_get_working() {
    let mut cm = make_cm();
    let id = cm
        .store_working("goal", "build feature X", "task-1", 0.9)
        .unwrap();
    assert!(id.starts_with("wrk_"));

    let slots = cm.get_working("task-1");
    assert_eq!(slots.len(), 1);
    assert_eq!(slots[0].content, "build feature X");
    assert_eq!(slots[0].relevance, 0.9);
}

#[test]
fn test_working_memory_capacity_eviction() {
    let mut cm = make_cm();

    // Fill to capacity
    for i in 0..WORKING_MEMORY_CAPACITY {
        cm.store_working("item", &format!("slot-{i}"), "t1", i as f64)
            .unwrap();
    }
    assert_eq!(cm.get_working("t1").len(), WORKING_MEMORY_CAPACITY);

    // Push one more — should evict lowest relevance (0.0)
    cm.store_working("item", "new-slot", "t1", 100.0).unwrap();
    let slots = cm.get_working("t1");
    assert_eq!(slots.len(), WORKING_MEMORY_CAPACITY);

    // The slot with relevance 0.0 should be gone
    assert!(!slots.iter().any(|s| s.content == "slot-0"));
    // The new one should be there
    assert!(slots.iter().any(|s| s.content == "new-slot"));
}

#[test]
fn test_working_task_isolation() {
    let mut cm = make_cm();
    cm.store_working("g", "a", "t1", 1.0).unwrap();
    cm.store_working("g", "b", "t2", 1.0).unwrap();

    assert_eq!(cm.get_working("t1").len(), 1);
    assert_eq!(cm.get_working("t2").len(), 1);
}

#[test]
fn test_clear_working() {
    let mut cm = make_cm();
    cm.store_working("g", "a", "t1", 1.0).unwrap();
    cm.store_working("g", "b", "t1", 1.0).unwrap();

    let cleared = cm.clear_working("t1");
    assert_eq!(cleared, 2);
    assert!(cm.get_working("t1").is_empty());
}

// -- episodic memory ----------------------------------------------------

#[test]
fn test_store_and_get_episodes() {
    let mut cm = make_cm();
    let id = cm
        .store_episode("something happened", "user-session", None, None)
        .unwrap();
    assert!(id.starts_with("epi_"));

    let eps = cm.get_episodes(10, false);
    assert_eq!(eps.len(), 1);
    assert_eq!(eps[0].content, "something happened");
    assert_eq!(eps[0].source_label, "user-session");
    assert!(!eps[0].compressed);
}

#[test]
fn test_episode_temporal_ordering() {
    let mut cm = make_cm();
    cm.store_episode("first", "src", None, None).unwrap();
    cm.store_episode("second", "src", None, None).unwrap();
    cm.store_episode("third", "src", None, None).unwrap();

    let eps = cm.get_episodes(10, false);
    assert_eq!(eps[0].content, "third");
    assert_eq!(eps[1].content, "second");
    assert_eq!(eps[2].content, "first");
}

#[test]
fn test_consolidate_episodes() {
    let mut cm = make_cm();
    for i in 0..5 {
        cm.store_episode(&format!("event-{i}"), "src", None, None)
            .unwrap();
    }

    // Not enough for batch_size=10
    assert!(cm
        .consolidate_episodes::<fn(&[String]) -> String>(10, None)
        .unwrap()
        .is_none());

    // Enough for batch_size=3
    let cons_id = cm
        .consolidate_episodes::<fn(&[String]) -> String>(3, None)
        .unwrap()
        .unwrap();
    assert!(cons_id.starts_with("con_"));

    // The consolidated episodes should be compressed
    let eps = cm.get_episodes(10, false);
    assert_eq!(eps.len(), 2); // only 2 uncompressed remain
}

#[test]
fn test_consolidate_with_custom_summarizer() {
    let mut cm = make_cm();
    for i in 0..3 {
        cm.store_episode(&format!("e{i}"), "src", None, None)
            .unwrap();
    }

    let cons_id = cm
        .consolidate_episodes(
            3,
            Some(|contents: &[String]| format!("SUMMARY({})", contents.len())),
        )
        .unwrap()
        .unwrap();

    // Verify the consolidated node exists
    let node = cm.graph.get_node(&cons_id).unwrap();
    assert_eq!(node.properties.get("summary").unwrap(), "SUMMARY(3)");
}

#[test]
fn test_search_episodes_excludes_compressed() {
    let mut cm = make_cm();
    for i in 0..5 {
        cm.store_episode(&format!("ep-{i}"), "src", None, None)
            .unwrap();
    }
    cm.consolidate_episodes::<fn(&[String]) -> String>(3, None)
        .unwrap();

    let uncompressed = cm.search_episodes(10);
    assert_eq!(uncompressed.len(), 2);

    let all = cm.get_episodes(10, true);
    assert_eq!(all.len(), 5);
}

// -- semantic memory ----------------------------------------------------

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

// -- procedural memory --------------------------------------------------

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

// -- prospective memory -------------------------------------------------

#[test]
fn test_store_and_check_triggers() {
    let mut cm = make_cm();
    cm.store_prospective(
        "notify on deploy",
        "deploy production",
        "send notification",
        5,
    )
    .unwrap();

    // Content that matches
    let triggered = cm.check_triggers("starting deploy to staging");
    assert_eq!(triggered.len(), 1);
    assert_eq!(triggered[0].status, "triggered");
    assert_eq!(triggered[0].action_on_trigger, "send notification");
}

#[test]
fn test_check_triggers_no_match() {
    let mut cm = make_cm();
    cm.store_prospective("remind", "deploy", "do stuff", 1)
        .unwrap();

    let triggered = cm.check_triggers("completely unrelated text");
    assert!(triggered.is_empty());
}

#[test]
fn test_triggered_not_re_triggered() {
    let mut cm = make_cm();
    cm.store_prospective("notify", "deploy", "action", 1)
        .unwrap();

    let t1 = cm.check_triggers("deploy happening");
    assert_eq!(t1.len(), 1);

    // Already triggered — should not trigger again
    let t2 = cm.check_triggers("deploy happening again");
    assert!(t2.is_empty());
}

#[test]
fn test_resolve_prospective() {
    let mut cm = make_cm();
    let id = cm
        .store_prospective("notify", "deploy", "action", 1)
        .unwrap();

    let triggered = cm.check_triggers("deploy now");
    assert_eq!(triggered.len(), 1);

    cm.resolve_prospective(&id);

    // After resolving, it should not appear in pending or triggered checks
    let filter = agent_filter(&cm.agent_name);
    let nodes = cm
        .graph
        .query_nodes(NT_PROSPECTIVE, Some(&filter), usize::MAX);
    let node = nodes.iter().find(|n| n.node_id == id).unwrap();
    assert_eq!(node.properties.get("status").unwrap(), "resolved");
}

#[test]
fn test_prospective_priority_ordering() {
    let mut cm = make_cm();
    cm.store_prospective("low", "deploy", "low-action", 1)
        .unwrap();
    cm.store_prospective("high", "deploy", "high-action", 10)
        .unwrap();

    let triggered = cm.check_triggers("deploy now");
    assert_eq!(triggered.len(), 2);
    // Higher priority first
    assert_eq!(triggered[0].priority, 10);
}

// -- statistics ---------------------------------------------------------

#[test]
fn test_get_memory_stats_empty() {
    let cm = make_cm();
    let stats = cm.get_memory_stats();
    assert_eq!(*stats.get("total").unwrap(), 0);
    assert_eq!(*stats.get("sensory").unwrap(), 0);
}

#[test]
fn test_get_memory_stats_with_data() {
    let mut cm = make_cm();
    cm.store_sensory("text", "a", 300).unwrap();
    cm.store_sensory("text", "b", 300).unwrap();
    cm.store_working("g", "c", "t1", 1.0).unwrap();
    cm.store_episode("ep", "src", None, None).unwrap();
    cm.store_fact("f", "fact", 1.0, "", None, None).unwrap();
    cm.store_procedure("p", &["s".into()], None).unwrap();
    cm.store_prospective("d", "t", "a", 1).unwrap();

    let stats = cm.get_memory_stats();
    assert_eq!(*stats.get("sensory").unwrap(), 2);
    assert_eq!(*stats.get("working").unwrap(), 1);
    assert_eq!(*stats.get("episodic").unwrap(), 1);
    assert_eq!(*stats.get("semantic").unwrap(), 1);
    assert_eq!(*stats.get("procedural").unwrap(), 1);
    assert_eq!(*stats.get("prospective").unwrap(), 1);
    assert_eq!(*stats.get("total").unwrap(), 7);
}

#[test]
fn test_get_statistics_alias() {
    let cm = make_cm();
    let s1 = cm.get_memory_stats();
    let s2 = cm.get_statistics();
    assert_eq!(s1, s2);
}

// -- agent isolation ----------------------------------------------------

#[test]
fn test_agent_isolation() {
    let mut cm_a = CognitiveMemory::new("alice").unwrap();
    let mut cm_b = CognitiveMemory::new("bob").unwrap();

    cm_a.store_fact("x", "alice fact", 1.0, "", None, None)
        .unwrap();
    cm_b.store_fact("x", "bob fact", 1.0, "", None, None)
        .unwrap();

    // Each agent only sees their own facts
    // (separate InMemoryGraphStore instances give natural isolation)
    assert_eq!(cm_a.search_facts("fact", 10, 0.0).len(), 1);
    assert_eq!(cm_b.search_facts("fact", 10, 0.0).len(), 1);
    assert_eq!(cm_a.search_facts("fact", 10, 0.0)[0].content, "alice fact");
    assert_eq!(cm_b.search_facts("fact", 10, 0.0)[0].content, "bob fact");
}

// -- lifecycle ----------------------------------------------------------

#[test]
fn test_close() {
    let mut cm = make_cm();
    cm.store_sensory("text", "data", 300).unwrap();
    cm.close();
    // After close, the store is empty
    assert!(cm.get_sensory(10).is_empty());
}

// -- edge cases ---------------------------------------------------------

#[test]
fn test_episode_explicit_temporal_index() {
    let mut cm = make_cm();
    cm.store_episode("ep1", "src", Some(100), None).unwrap();
    cm.store_episode("ep2", "src", Some(50), None).unwrap();

    let eps = cm.get_episodes(10, false);
    // Sorted by temporal_index desc
    assert_eq!(eps[0].temporal_index, 100);
    assert_eq!(eps[1].temporal_index, 50);
}

#[test]
fn test_episode_auto_increment_after_explicit() {
    let mut cm = make_cm();
    cm.store_episode("ep1", "src", Some(100), None).unwrap();
    cm.store_episode("ep2", "src", None, None).unwrap();

    let eps = cm.get_episodes(10, false);
    // Auto-incremented should be > 100
    assert!(eps.iter().any(|e| e.temporal_index == 101));
}

// ====================================================================
// Extended coverage: store_sensory + get_sensory
// ====================================================================

#[test]
fn test_store_sensory_multiple_modalities() {
    let mut cm = make_cm();
    cm.store_sensory("visual", "red light", 300).unwrap();
    cm.store_sensory("audio", "beep sound", 300).unwrap();
    cm.store_sensory("text", "error message", 300).unwrap();

    let items = cm.get_sensory(10);
    assert_eq!(items.len(), 3);
    let modalities: Vec<&str> = items.iter().map(|i| i.modality.as_str()).collect();
    assert!(modalities.contains(&"visual"));
    assert!(modalities.contains(&"audio"));
    assert!(modalities.contains(&"text"));
}

#[test]
fn test_store_sensory_returns_unique_ids() {
    let mut cm = make_cm();
    let id1 = cm.store_sensory("text", "a", 300).unwrap();
    let id2 = cm.store_sensory("text", "b", 300).unwrap();
    assert_ne!(id1, id2);
}

#[test]
fn test_get_sensory_respects_expiry() {
    let mut cm = make_cm();
    cm.store_sensory("text", "expired", 0).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(10));
    cm.store_sensory("text", "alive", 9999).unwrap();

    let items = cm.get_sensory(10);
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].raw_data, "alive");
}

// ====================================================================
// Extended coverage: expire_sensory
// ====================================================================

#[test]
fn test_expire_sensory_is_alias_for_prune() {
    let mut cm = make_cm();
    cm.store_sensory("text", "gone", 0).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(10));
    cm.store_sensory("text", "stays", 9999).unwrap();

    let pruned = cm.expire_sensory();
    assert_eq!(pruned, 1);
    assert_eq!(cm.get_sensory(10).len(), 1);
}

#[test]
fn test_expire_sensory_nothing_to_prune() {
    let mut cm = make_cm();
    cm.store_sensory("text", "long-lived", 99999).unwrap();
    assert_eq!(cm.expire_sensory(), 0);
}

// ====================================================================
// Extended coverage: store_episode
// ====================================================================

#[test]
fn test_store_episode_with_metadata() {
    let mut cm = make_cm();
    let mut meta = HashMap::new();
    meta.insert(
        "location".to_string(),
        serde_json::Value::String("lab".into()),
    );
    let id = cm
        .store_episode("experiment completed", "lab-session", None, Some(&meta))
        .unwrap();

    let eps = cm.get_episodes(10, false);
    let ep = eps.iter().find(|e| e.node_id == id).unwrap();
    assert_eq!(
        ep.metadata.get("location").and_then(|v| v.as_str()),
        Some("lab")
    );
}

#[test]
fn test_store_episode_returns_unique_ids() {
    let mut cm = make_cm();
    let id1 = cm.store_episode("ep1", "src", None, None).unwrap();
    let id2 = cm.store_episode("ep2", "src", None, None).unwrap();
    assert_ne!(id1, id2);
    assert!(id1.starts_with("epi_"));
}

// ====================================================================
// Extended coverage: store_fact
// ====================================================================

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

// ====================================================================
// Extended coverage: store_working
// ====================================================================

#[test]
fn test_store_working_different_slot_types() {
    let mut cm = make_cm();
    cm.store_working("goal", "build X", "task-1", 0.9).unwrap();
    cm.store_working("context", "we are in beta", "task-1", 0.7)
        .unwrap();
    cm.store_working("constraint", "deadline friday", "task-1", 0.8)
        .unwrap();

    let slots = cm.get_working("task-1");
    assert_eq!(slots.len(), 3);
    // Sorted by relevance desc
    assert_eq!(slots[0].slot_type, "goal");
    assert_eq!(slots[0].relevance, 0.9);
}

#[test]
fn test_store_working_at_exact_capacity() {
    let mut cm = make_cm();
    for i in 0..WORKING_MEMORY_CAPACITY {
        cm.store_working("item", &format!("slot-{i}"), "t1", (i + 1) as f64)
            .unwrap();
    }
    assert_eq!(cm.get_working("t1").len(), WORKING_MEMORY_CAPACITY);

    // Push one more with high relevance — should evict lowest (slot-0 with rel=1.0)
    cm.store_working("item", "overflow-slot", "t1", 999.0)
        .unwrap();
    let slots = cm.get_working("t1");
    assert_eq!(slots.len(), WORKING_MEMORY_CAPACITY);
    assert!(slots.iter().any(|s| s.content == "overflow-slot"));
}

// ====================================================================
// Extended coverage: link_fact_to_episode
// ====================================================================

#[test]
fn test_link_fact_to_episode_creates_edge() {
    let mut cm = make_cm();
    let fact_id = cm
        .store_fact("rust", "Rust is memory safe", 0.95, "", None, None)
        .unwrap();
    let ep_id = cm
        .store_episode("learned about Rust", "reading", None, None)
        .unwrap();

    assert!(cm.link_fact_to_episode(&fact_id, &ep_id).is_ok());

    // Verify edge exists via graph query
    use crate::graph::{Direction as Dir, GraphStore};
    let neighbors = cm
        .graph
        .query_neighbors(&fact_id, Some(ET_DERIVES_FROM), Dir::Outgoing, 10);
    assert!(
        !neighbors.is_empty(),
        "Expected DERIVES_FROM edge from fact to episode"
    );
    assert_eq!(neighbors[0].1.node_id, ep_id);
}

#[test]
fn test_link_fact_to_episode_multiple_links() {
    let mut cm = make_cm();
    let fact_id = cm
        .store_fact("topic", "A general fact", 0.8, "", None, None)
        .unwrap();
    let ep1 = cm.store_episode("episode one", "src", None, None).unwrap();
    let ep2 = cm.store_episode("episode two", "src", None, None).unwrap();

    assert!(cm.link_fact_to_episode(&fact_id, &ep1).is_ok());
    assert!(cm.link_fact_to_episode(&fact_id, &ep2).is_ok());
}

// ====================================================================
// Extended coverage: search_procedures_mut
// ====================================================================

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

// ====================================================================
// Extended coverage: store_prospective
// ====================================================================

#[test]
fn test_store_prospective_returns_valid_id() {
    let mut cm = make_cm();
    let id = cm
        .store_prospective("remind me", "meeting starts", "join call", 5)
        .unwrap();
    assert!(id.starts_with("pro_"));
}

#[test]
fn test_store_prospective_initial_status_pending() {
    let mut cm = make_cm();
    let id = cm
        .store_prospective("task", "trigger", "action", 1)
        .unwrap();

    let filter = agent_filter(&cm.agent_name);
    let nodes = cm
        .graph
        .query_nodes(NT_PROSPECTIVE, Some(&filter), usize::MAX);
    let node = nodes.iter().find(|n| n.node_id == id).unwrap();
    assert_eq!(node.properties.get("status").unwrap(), "pending");
}

// ====================================================================
// Extended coverage: get_memory_stats
// ====================================================================

#[test]
fn test_get_memory_stats_after_deletions() {
    let mut cm = make_cm();
    cm.store_sensory("text", "temp", 0).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(10));
    cm.store_sensory("text", "kept", 9999).unwrap();
    cm.prune_expired_sensory();

    let stats = cm.get_memory_stats();
    assert_eq!(*stats.get("sensory").unwrap(), 1);
    assert_eq!(*stats.get("total").unwrap(), 1);
}

#[test]
fn test_get_memory_stats_all_categories() {
    let cm = make_cm();
    let stats = cm.get_memory_stats();
    assert!(stats.contains_key("working"));
    assert!(stats.contains_key("episodic"));
    assert!(stats.contains_key("semantic"));
    assert!(stats.contains_key("procedural"));
    assert!(stats.contains_key("prospective"));
    assert!(stats.contains_key("total"));
}

// ====================================================================
// Extended coverage: check_triggers
// ====================================================================

#[test]
fn test_check_triggers_partial_word_no_match() {
    let mut cm = make_cm();
    cm.store_prospective("notify", "deployment", "run script", 1)
        .unwrap();

    // "deploy" is not the same word as "deployment" in word-based matching
    let triggered = cm.check_triggers("deploy is happening");
    // This depends on tokenization — "deployment" != "deploy"
    // The trigger uses word-level overlap, so exact word match is needed
    assert!(triggered.is_empty());
}

#[test]
fn test_check_triggers_multiple_triggers() {
    let mut cm = make_cm();
    cm.store_prospective("alert1", "error", "log error", 2)
        .unwrap();
    cm.store_prospective("alert2", "error", "notify admin", 5)
        .unwrap();

    let triggered = cm.check_triggers("an error occurred");
    assert_eq!(triggered.len(), 2);
    // Higher priority first
    assert_eq!(triggered[0].priority, 5);
    assert_eq!(triggered[1].priority, 2);
}

#[test]
fn test_consolidate_episodes_creates_summary() {
    let mut cm = make_cm();
    for i in 0..5 {
        cm.store_episode(&format!("event-{i}"), "src", None, None)
            .unwrap();
    }
    let cons_id = cm
        .consolidate_episodes::<fn(&[String]) -> String>(3, None)
        .unwrap()
        .unwrap();
    assert!(cons_id.starts_with("con_"));
    let uncompressed = cm.get_episodes(10, false);
    assert_eq!(uncompressed.len(), 2);
}

// ====================================================================
// QA audit: Fix 15 — consolidate_episodes creates summary
// ====================================================================
