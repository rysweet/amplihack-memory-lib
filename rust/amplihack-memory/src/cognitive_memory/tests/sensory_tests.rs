use super::super::types::{new_id, ts_now};
use super::super::*;

// -- construction --

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

// -- helpers --

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

// -- sensory memory --

fn make_cm() -> CognitiveMemory {
    CognitiveMemory::new(&format!("test-agent-{}", uuid::Uuid::new_v4())).unwrap()
}

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

// -- Extended coverage: store_sensory + get_sensory --

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

// -- Extended coverage: expire_sensory --

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
