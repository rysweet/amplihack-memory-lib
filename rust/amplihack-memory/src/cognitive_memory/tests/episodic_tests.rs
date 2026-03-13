use super::super::*;

fn make_cm() -> CognitiveMemory {
    CognitiveMemory::new(&format!("test-agent-{}", uuid::Uuid::new_v4())).unwrap()
}

// -- episodic memory --

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

// -- edge cases --

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

// -- Extended coverage: store_episode --

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
