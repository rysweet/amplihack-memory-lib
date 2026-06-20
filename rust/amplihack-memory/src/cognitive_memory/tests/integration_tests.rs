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
    use crate::graph::Direction as Dir;
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

// ===========================================================================
// Provenance edges (issue #90): plural linking + typed read path
//
// File: rust/amplihack-memory/src/cognitive_memory/tests/integration_tests.rs
//
// Failing-first TDD tests for the not-yet-implemented `link_fact_to_episodes`,
// `fact_provenance`, and `procedure_provenance`.
// ===========================================================================

/// `link_fact_to_episodes` is lenient and returns the count of edges actually
/// created: 2 valid + 1 missing episode id -> returns 2, two edges present.
/// (Invariants I2, I3.)
#[test]
fn test_link_fact_to_episodes_counts_created_edges() {
    let mut cm = make_cm();
    let fact = cm
        .store_fact("topic", "a general fact", 0.8, "", None, None)
        .unwrap();
    let ep1 = cm.store_episode("episode one", "src", None, None).unwrap();
    let ep2 = cm.store_episode("episode two", "src", None, None).unwrap();

    let created = cm
        .link_fact_to_episodes(
            &fact,
            &[ep1.clone(), ep2.clone(), "epi_missing".to_string()],
        )
        .expect("link_fact_to_episodes is lenient and must succeed");
    assert_eq!(
        created, 2,
        "only the two valid episodes should be linked (the missing id is skipped)"
    );

    use crate::graph::Direction as Dir;
    let neighbors = cm
        .graph
        .query_neighbors(&fact, Some(ET_DERIVES_FROM), Dir::Outgoing, 10);
    assert_eq!(neighbors.len(), 2);
    let targets: std::collections::HashSet<String> =
        neighbors.iter().map(|(_, n)| n.node_id.clone()).collect();
    assert!(targets.contains(&ep1));
    assert!(targets.contains(&ep2));

    // Read path agrees.
    let mut prov = cm.fact_provenance(&fact);
    prov.sort();
    let mut want = vec![ep1, ep2];
    want.sort();
    assert_eq!(prov, want);
}

/// The typed read methods return the linked episode ids and `[]` for unknown
/// ids or nodes with no provenance, for both facts and procedures. They keep
/// the `graph` field private from external consumers. (Invariant I8.)
#[test]
fn test_fact_and_procedure_provenance_read_path() {
    let mut cm = make_cm();

    // Unknown ids return empty (no panic).
    assert!(cm.fact_provenance("does-not-exist").is_empty());
    assert!(cm.procedure_provenance("does-not-exist").is_empty());

    let ep1 = cm.store_episode("ep one", "src", None, None).unwrap();
    let ep2 = cm.store_episode("ep two", "src", None, None).unwrap();

    // A node with no provenance edges returns [].
    let plain = cm
        .store_fact("c", "no provenance", 0.5, "", None, None)
        .unwrap();
    assert!(cm.fact_provenance(&plain).is_empty());

    // A fact stored with provenance surfaces all its source episodes.
    let fact = cm
        .store_fact_with_provenance(
            "c",
            "derived",
            0.9,
            "",
            None,
            None,
            &[ep1.clone(), ep2.clone()],
        )
        .unwrap();
    let mut got = cm.fact_provenance(&fact);
    got.sort();
    let mut want = vec![ep1.clone(), ep2.clone()];
    want.sort();
    assert_eq!(got, want, "fact_provenance must return all linked episodes");

    // Procedure read path mirrors the fact read path.
    let proc = cm
        .store_procedure_with_provenance("p", &["s".to_string()], None, std::slice::from_ref(&ep1))
        .unwrap();
    assert_eq!(cm.procedure_provenance(&proc), vec![ep1]);
}
