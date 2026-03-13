use super::super::types::ET_DERIVES_FROM;
use super::super::*;

fn make_cm() -> CognitiveMemory {
    CognitiveMemory::new(&format!("test-agent-{}", uuid::Uuid::new_v4())).unwrap()
}

// -- statistics --

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

// -- agent isolation --

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

// -- lifecycle --

#[test]
fn test_close() {
    let mut cm = make_cm();
    cm.store_sensory("text", "data", 300).unwrap();
    cm.close();
    // After close, the store is empty
    assert!(cm.get_sensory(10).is_empty());
}

// -- link_fact_to_episode --

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

// -- Extended coverage: get_memory_stats --

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
