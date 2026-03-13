use super::super::*;

fn make_cm() -> CognitiveMemory {
    CognitiveMemory::new(&format!("test-agent-{}", uuid::Uuid::new_v4())).unwrap()
}

// -- working memory --

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

// -- Extended coverage: store_working --

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
