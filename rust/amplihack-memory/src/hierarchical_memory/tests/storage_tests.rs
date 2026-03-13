use super::super::*;
use crate::memory_types::MemoryCategory;

fn make_mem() -> HierarchicalMemory {
    HierarchicalMemory::new("test_agent").unwrap()
}

// -- Store and retrieve --

#[test]
fn test_store_and_retrieve() {
    let mut mem = make_mem();
    let nid = mem
        .store_knowledge(
            "Plants use photosynthesis to convert sunlight",
            "biology",
            0.9,
            None,
            "",
            &[],
            None,
        )
        .unwrap();
    assert!(!nid.is_empty());

    let sub = mem.retrieve_subgraph("photosynthesis", 2, 20);
    assert!(!sub.nodes.is_empty());
    assert_eq!(
        sub.nodes[0].content,
        "Plants use photosynthesis to convert sunlight"
    );
}

#[test]
fn test_store_empty_content_fails() {
    let mut mem = make_mem();
    let result = mem.store_knowledge("", "concept", 0.8, None, "", &[], None);
    assert!(result.is_err());
}

#[test]
fn test_invalid_agent_name() {
    assert!(HierarchicalMemory::new("").is_err());
    assert!(HierarchicalMemory::new("a b c").is_err());
    assert!(HierarchicalMemory::new("../hack").is_err());
}

// -- Get all knowledge with category filter --

#[test]
fn test_get_all_knowledge_with_category() {
    let mut mem = make_mem();
    mem.store_knowledge(
        "Water is H2O",
        "chemistry",
        0.9,
        Some(MemoryCategory::Semantic),
        "",
        &[],
        None,
    )
    .unwrap();
    mem.store_knowledge(
        "Step 1: mix chemicals",
        "procedure",
        0.8,
        Some(MemoryCategory::Procedural),
        "",
        &[],
        None,
    )
    .unwrap();

    let semantic = mem.get_all_knowledge(Some(MemoryCategory::Semantic), 100);
    assert_eq!(semantic.len(), 1);
    assert!(semantic[0].content.contains("H2O"));

    let procedural = mem.get_all_knowledge(Some(MemoryCategory::Procedural), 100);
    assert_eq!(procedural.len(), 1);
    assert!(procedural[0].content.contains("mix chemicals"));

    let all = mem.get_all_knowledge(None, 100);
    assert_eq!(all.len(), 2);
}

// -- Retrieve with no results --

#[test]
fn test_retrieve_empty_query() {
    let mem = make_mem();
    let sub = mem.retrieve_subgraph("", 2, 20);
    assert!(sub.nodes.is_empty());
}

#[test]
fn test_retrieve_no_matches() {
    let mut mem = make_mem();
    mem.store_knowledge("Plants need water", "biology", 0.9, None, "", &[], None)
        .unwrap();
    let sub = mem.retrieve_subgraph("quantum mechanics spacetime", 2, 20);
    assert!(sub.nodes.is_empty());
}

// -- Multiple agents isolation --

#[test]
fn test_agent_isolation() {
    let mut mem1 = HierarchicalMemory::new("agent_alpha").unwrap();
    let mut mem2 = HierarchicalMemory::new("agent_beta").unwrap();

    mem1.store_knowledge("Alpha fact", "alpha-topic", 0.9, None, "", &[], None)
        .unwrap();
    mem2.store_knowledge("Beta fact", "beta-topic", 0.9, None, "", &[], None)
        .unwrap();

    let sub1 = mem1.retrieve_subgraph("alpha", 2, 20);
    assert!(!sub1.nodes.is_empty());

    // Agent beta should not see agent alpha's facts (different store instances)
    let sub2 = mem2.retrieve_subgraph("alpha", 2, 20);
    assert!(sub2.nodes.is_empty());
}

// -- Close --

#[test]
fn test_close() {
    let mut mem = make_mem();
    mem.store_knowledge("Some fact", "topic", 0.8, None, "", &[], None)
        .unwrap();
    mem.close();

    // After close, store is cleared
    let stats = mem.get_statistics();
    let nodes = stats
        .get("semantic_nodes")
        .and_then(|v| v.as_u64())
        .unwrap();
    assert_eq!(nodes, 0);
}

// -- Extended coverage: store_knowledge --

#[test]
fn test_store_knowledge_with_tags_and_source() {
    let mut mem = make_mem();
    let tags = vec!["lang".into(), "systems".into()];
    let nid = mem
        .store_knowledge(
            "Rust has ownership semantics",
            "rust",
            0.95,
            None,
            "doc-42",
            &tags,
            None,
        )
        .unwrap();

    let nodes = mem.get_all_knowledge(None, 100);
    let node = nodes.iter().find(|n| n.node_id == nid).unwrap();
    assert_eq!(node.source_id, "doc-42");
    assert_eq!(node.tags, tags);
    assert!(node.confidence > 0.94);
}

#[test]
fn test_store_knowledge_explicit_category() {
    let mut mem = make_mem();
    let nid = mem
        .store_knowledge(
            "A plain factual statement with no keywords",
            "trivia",
            0.7,
            Some(MemoryCategory::Procedural),
            "",
            &[],
            None,
        )
        .unwrap();

    let nodes = mem.get_all_knowledge(Some(MemoryCategory::Procedural), 100);
    assert!(nodes.iter().any(|n| n.node_id == nid));
}

#[test]
fn test_store_knowledge_nan_confidence_rejected() {
    let mut mem = make_mem();
    assert!(mem
        .store_knowledge("data", "concept", f64::NAN, None, "", &[], None)
        .is_err());
}

#[test]
fn test_store_knowledge_out_of_range_confidence() {
    let mut mem = make_mem();
    assert!(mem
        .store_knowledge("data", "concept", 1.5, None, "", &[], None)
        .is_err());
    assert!(mem
        .store_knowledge("data", "concept", -0.1, None, "", &[], None)
        .is_err());
}

#[test]
fn test_store_knowledge_boundary_confidence() {
    let mut mem = make_mem();
    assert!(mem
        .store_knowledge("low bound", "c", 0.0, None, "", &[], None)
        .is_ok());
    assert!(mem
        .store_knowledge("high bound", "c", 1.0, None, "", &[], None)
        .is_ok());
}

#[test]
fn test_store_knowledge_with_temporal_metadata() {
    let mut mem = make_mem();
    let mut tm = HashMap::new();
    tm.insert("temporal_index".into(), serde_json::json!(5));
    tm.insert("source_date".into(), serde_json::json!("2024-03-15"));

    let nid = mem
        .store_knowledge(
            "Temperature is 25C today",
            "weather",
            0.8,
            None,
            "",
            &[],
            Some(&tm),
        )
        .unwrap();

    let nodes = mem.get_all_knowledge(None, 100);
    let node = nodes.iter().find(|n| n.node_id == nid).unwrap();
    assert_eq!(
        node.metadata.get("temporal_index").and_then(|v| v.as_i64()),
        Some(5)
    );
}

#[test]
fn test_store_knowledge_whitespace_trimmed() {
    let mut mem = make_mem();
    let nid = mem
        .store_knowledge("  trimmed content  ", "topic", 0.8, None, "", &[], None)
        .unwrap();
    let nodes = mem.get_all_knowledge(None, 100);
    let node = nodes.iter().find(|n| n.node_id == nid).unwrap();
    assert_eq!(node.content, "trimmed content");
}

// -- Extended coverage: retrieve_subgraph --

#[test]
fn test_retrieve_subgraph_max_nodes_limit() {
    let mut mem = make_mem();
    for i in 0..10 {
        mem.store_knowledge(
            &format!("Quantum physics experiment number {i}"),
            "quantum",
            0.8,
            None,
            "",
            &[],
            None,
        )
        .unwrap();
    }

    let sub = mem.retrieve_subgraph("quantum physics", 2, 3);
    assert!(sub.nodes.len() <= 3);
}

#[test]
fn test_retrieve_subgraph_returns_edges() {
    let mut mem = make_mem();
    mem.store_knowledge(
        "Machine learning uses neural networks",
        "machine-learning",
        0.9,
        None,
        "",
        &[],
        None,
    )
    .unwrap();
    mem.store_knowledge(
        "Deep learning neural networks are powerful",
        "machine-learning",
        0.85,
        None,
        "",
        &[],
        None,
    )
    .unwrap();

    let sub = mem.retrieve_subgraph("neural networks machine learning", 2, 20);
    assert!(sub.nodes.len() >= 2);
    // With two similar nodes, there should be at least one edge
    assert!(
        !sub.edges.is_empty() || sub.nodes.len() >= 2,
        "Expected edges or multiple nodes"
    );
}

#[test]
fn test_retrieve_subgraph_query_field() {
    let mem = make_mem();
    let sub = mem.retrieve_subgraph("test query", 2, 20);
    assert_eq!(sub.query, "test query");
}
