use super::super::*;

fn make_mem() -> HierarchicalMemory {
    HierarchicalMemory::new("test_agent").unwrap()
}

// -- Temporal chains --

#[test]
fn test_temporal_chain() {
    let mut mem = make_mem();

    let mut tm1 = HashMap::new();
    tm1.insert("temporal_index".into(), serde_json::json!(1));
    mem.store_knowledge(
        "Team has 5 members",
        "team size",
        0.8,
        None,
        "",
        &[],
        Some(&tm1),
    )
    .unwrap();

    let mut tm2 = HashMap::new();
    tm2.insert("temporal_index".into(), serde_json::json!(2));
    mem.store_knowledge(
        "Team has 8 members",
        "team size",
        0.9,
        None,
        "",
        &[],
        Some(&tm2),
    )
    .unwrap();

    let chain = mem.get_temporal_chain("team");
    assert!(
        chain.len() >= 2,
        "Expected at least 2 nodes in temporal chain"
    );

    // First node in chain should be the newest (head)
    assert!(chain[0].content.contains("8"));
}

// -- Knowledge evolution --

#[test]
fn test_knowledge_evolution() {
    let mut mem = make_mem();

    let mut tm1 = HashMap::new();
    tm1.insert("temporal_index".into(), serde_json::json!(1));
    mem.store_knowledge(
        "Population is 5000 people",
        "city population",
        0.8,
        None,
        "",
        &[],
        Some(&tm1),
    )
    .unwrap();

    let mut tm2 = HashMap::new();
    tm2.insert("temporal_index".into(), serde_json::json!(2));
    mem.store_knowledge(
        "Population is 7000 people",
        "city population",
        0.9,
        None,
        "",
        &[],
        Some(&tm2),
    )
    .unwrap();

    let evolution = mem.get_knowledge_evolution("population");
    assert!(!evolution.is_empty());

    // At least one entry should have a reason
    let has_reason = evolution.iter().any(|(_, reason)| reason.is_some());
    assert!(has_reason, "Expected at least one evolution reason");
}

// -- Extended coverage: get_temporal_chain --

#[test]
fn test_temporal_chain_empty_concept() {
    let mem = make_mem();
    let chain = mem.get_temporal_chain("");
    assert!(chain.is_empty());
}

#[test]
fn test_temporal_chain_no_matching_concept() {
    let mut mem = make_mem();
    mem.store_knowledge("Unrelated fact", "other", 0.8, None, "", &[], None)
        .unwrap();
    let chain = mem.get_temporal_chain("nonexistent_concept_xyz");
    assert!(chain.is_empty());
}

#[test]
fn test_temporal_chain_single_node() {
    let mut mem = make_mem();
    let mut tm = HashMap::new();
    tm.insert("temporal_index".into(), serde_json::json!(1));
    mem.store_knowledge(
        "Solo fact here",
        "solo-concept",
        0.9,
        None,
        "",
        &[],
        Some(&tm),
    )
    .unwrap();

    let chain = mem.get_temporal_chain("solo-concept");
    assert_eq!(chain.len(), 1);
    assert!(chain[0].content.contains("Solo fact"));
}

#[test]
fn test_temporal_chain_three_versions() {
    let mut mem = make_mem();
    for i in 1..=3 {
        let mut tm = HashMap::new();
        tm.insert("temporal_index".into(), serde_json::json!(i));
        mem.store_knowledge(
            &format!("Server count is {}", i * 10),
            "server count",
            0.8,
            None,
            "",
            &[],
            Some(&tm),
        )
        .unwrap();
    }

    let chain = mem.get_temporal_chain("server");
    assert!(chain.len() >= 2, "Expected multiple versions in chain");
}

// -- Extended coverage: get_knowledge_evolution --

#[test]
fn test_knowledge_evolution_empty_concept() {
    let mem = make_mem();
    let evolution = mem.get_knowledge_evolution("");
    assert!(evolution.is_empty());
}

#[test]
fn test_knowledge_evolution_single_node_no_reason() {
    let mut mem = make_mem();
    mem.store_knowledge("Singular fact", "single-topic", 0.9, None, "", &[], None)
        .unwrap();

    let evolution = mem.get_knowledge_evolution("single-topic");
    if !evolution.is_empty() {
        assert!(evolution[0].1.is_none());
    }
}
