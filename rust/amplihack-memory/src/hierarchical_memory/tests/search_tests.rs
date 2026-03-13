use super::super::*;
use crate::graph::Direction;
use crate::memory_types::MemoryCategory;

fn make_mem() -> HierarchicalMemory {
    HierarchicalMemory::new("test_agent").unwrap()
}

// -- Similarity edge creation --

#[test]
fn test_similarity_edges() {
    let mut mem = make_mem();
    let _id1 = mem
        .store_knowledge(
            "Rust programming language is fast",
            "rust",
            0.9,
            None,
            "",
            &[],
            None,
        )
        .unwrap();
    let id2 = mem
        .store_knowledge(
            "Rust programming language has memory safety",
            "rust",
            0.85,
            None,
            "",
            &[],
            None,
        )
        .unwrap();

    // The second node should have a SIMILAR_TO edge to the first
    let stats = mem.get_statistics();
    let sim_edges = stats
        .get("similar_to_edges")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    assert!(sim_edges > 0, "Expected similarity edges to be created");

    // Retrieve subgraph should find both via similarity
    let sub = mem.retrieve_subgraph("rust programming", 2, 20);
    assert!(sub.nodes.len() >= 2);
    assert!(!sub.edges.is_empty() || sub.nodes.len() >= 2);

    // Verify through direct neighbor query
    let neighbors = mem
        .store
        .query_neighbors(&id2, Some("SIMILAR_TO"), Direction::Outgoing, 10);
    assert!(!neighbors.is_empty());
}

// -- Contradiction detection and SUPERSEDES edges --

#[test]
fn test_contradiction_and_supersedes() {
    let mut mem = make_mem();

    let mut tm1 = HashMap::new();
    tm1.insert("temporal_index".into(), serde_json::json!(1));

    let id1 = mem
        .store_knowledge(
            "Klaebo has 9 gold medals",
            "klaebo medals",
            0.8,
            None,
            "",
            &[],
            Some(&tm1),
        )
        .unwrap();

    let mut tm2 = HashMap::new();
    tm2.insert("temporal_index".into(), serde_json::json!(2));

    let id2 = mem
        .store_knowledge(
            "Klaebo has 10 gold medals",
            "klaebo medals",
            0.9,
            None,
            "",
            &[],
            Some(&tm2),
        )
        .unwrap();

    // Should have SUPERSEDES edge from id2 -> id1
    let neighbors = mem
        .store
        .query_neighbors(&id2, Some("SUPERSEDES"), Direction::Outgoing, 10);
    assert!(
        !neighbors.is_empty(),
        "Expected SUPERSEDES edge from new to old fact"
    );
    assert_eq!(neighbors[0].1.node_id, id1);

    let reason = neighbors[0].0.properties.get("reason").unwrap();
    assert!(reason.contains("Updated values"));
}

// -- Concept search --

#[test]
fn test_search_by_concept() {
    let mut mem = make_mem();
    mem.store_knowledge(
        "Water boils at 100 degrees celsius",
        "physics",
        0.95,
        None,
        "",
        &[],
        None,
    )
    .unwrap();
    mem.store_knowledge(
        "Iron melts at 1538 degrees",
        "chemistry",
        0.9,
        None,
        "",
        &[],
        None,
    )
    .unwrap();

    let results = mem.search_by_concept(&["physics".to_string()], 10);
    assert_eq!(results.len(), 1);
    assert!(results[0].content.contains("Water"));

    let results = mem.search_by_concept(&["degrees".to_string()], 10);
    assert!(results.len() >= 2);
}

// -- Entity search --

#[test]
fn test_search_by_entity() {
    let mut mem = make_mem();
    mem.store_knowledge(
        "Sarah Chen works at the research lab",
        "Sarah Chen",
        0.9,
        None,
        "",
        &[],
        None,
    )
    .unwrap();
    mem.store_knowledge(
        "John Smith is a teacher",
        "John Smith",
        0.85,
        None,
        "",
        &[],
        None,
    )
    .unwrap();

    let results = mem.search_by_entity("Sarah Chen", 10);
    assert!(!results.is_empty());
    assert!(results[0].content.contains("Sarah Chen"));

    let results = mem.search_by_entity("John", 10);
    assert!(!results.is_empty());
}

// -- Edge case: concept search with empty/short keywords --

#[test]
fn test_search_by_concept_short_keyword() {
    let mut mem = make_mem();
    mem.store_knowledge("Fact about AI", "ai-topic", 0.9, None, "", &[], None)
        .unwrap();

    // Keywords ≤ 2 chars are skipped
    let results = mem.search_by_concept(&["ai".to_string()], 10);
    assert!(results.is_empty());
}

#[test]
fn test_search_by_concept_with_str_slices() {
    // Fix 10: verify search_by_concept accepts &[&str] via AsRef<str>
    let mut mem = make_mem();
    mem.store_knowledge("Gravity pulls objects", "physics", 0.9, None, "", &[], None)
        .unwrap();
    let results = mem.search_by_concept(&["physics"], 10);
    assert!(!results.is_empty());
}

// -- Subgraph to_llm_context --

#[test]
fn test_to_llm_context_empty() {
    let sub = KnowledgeSubgraph::default();
    assert_eq!(sub.to_llm_context(false), "No relevant knowledge found.");
}

#[test]
fn test_to_llm_context_basic() {
    let sub = KnowledgeSubgraph {
        query: "test query".to_string(),
        nodes: vec![KnowledgeNode {
            node_id: "abc12345-1234-1234-1234-123456789abc".into(),
            category: MemoryCategory::Semantic,
            content: "The sky is blue".into(),
            concept: "nature".into(),
            confidence: 0.9,
            ..Default::default()
        }],
        edges: vec![],
    };

    let ctx = sub.to_llm_context(false);
    assert!(ctx.contains("test query"));
    assert!(ctx.contains("The sky is blue"));
    assert!(ctx.contains("[nature]"));
    assert!(ctx.contains("0.9"));
}

#[test]
fn test_to_llm_context_chronological() {
    let mut meta1 = HashMap::new();
    meta1.insert("temporal_index".into(), serde_json::json!(2));
    meta1.insert("source_date".into(), serde_json::json!("2024-01-02"));

    let mut meta2 = HashMap::new();
    meta2.insert("temporal_index".into(), serde_json::json!(1));
    meta2.insert("temporal_order".into(), serde_json::json!("Day 1"));

    let sub = KnowledgeSubgraph {
        query: "timeline".to_string(),
        nodes: vec![
            KnowledgeNode {
                node_id: "node2".into(),
                concept: "event".into(),
                content: "Second event".into(),
                confidence: 0.8,
                metadata: meta1,
                ..Default::default()
            },
            KnowledgeNode {
                node_id: "node1".into(),
                concept: "event".into(),
                content: "First event".into(),
                confidence: 0.9,
                metadata: meta2,
                ..Default::default()
            },
        ],
        edges: vec![],
    };

    let ctx = sub.to_llm_context(true);
    assert!(ctx.contains("chronological"));
    // First event (temporal_index=1) should appear before second (temporal_index=2)
    let pos_first = ctx.find("First event").unwrap();
    let pos_second = ctx.find("Second event").unwrap();
    assert!(pos_first < pos_second);
}

#[test]
fn test_to_llm_context_with_contradictions() {
    // Test with Value::String("true") — the format produced by create_similarity_edges
    let mut edge_meta_str = HashMap::new();
    edge_meta_str.insert(
        "contradiction".into(),
        serde_json::Value::String("true".to_string()),
    );
    edge_meta_str.insert("conflicting_values".into(), serde_json::json!("5 vs 8"));

    let sub = KnowledgeSubgraph {
        query: "team".to_string(),
        nodes: vec![KnowledgeNode {
            node_id: "n1".into(),
            concept: "team".into(),
            content: "Team has 5 members".into(),
            confidence: 0.8,
            ..Default::default()
        }],
        edges: vec![KnowledgeEdge {
            source_id: "n1-source-12345678".into(),
            target_id: "n2-target-12345678".into(),
            relationship: "SIMILAR_TO".into(),
            weight: 0.7,
            metadata: edge_meta_str,
        }],
    };

    let ctx = sub.to_llm_context(false);
    assert!(ctx.contains("Contradictions detected"));
    assert!(ctx.contains("5 vs 8"));
    assert!(ctx.contains("Relationships:"));

    // Also verify with Value::Bool(true) for backward compat
    let mut edge_meta_bool = HashMap::new();
    edge_meta_bool.insert("contradiction".into(), serde_json::json!(true));
    edge_meta_bool.insert("conflicting_values".into(), serde_json::json!("3 vs 7"));

    let sub2 = KnowledgeSubgraph {
        query: "team".to_string(),
        nodes: vec![KnowledgeNode {
            node_id: "n1".into(),
            concept: "team".into(),
            content: "Team has 3 members".into(),
            confidence: 0.8,
            ..Default::default()
        }],
        edges: vec![KnowledgeEdge {
            source_id: "n1-source-12345678".into(),
            target_id: "n2-target-12345678".into(),
            relationship: "SIMILAR_TO".into(),
            weight: 0.7,
            metadata: edge_meta_bool,
        }],
    };

    let ctx2 = sub2.to_llm_context(false);
    assert!(ctx2.contains("Contradictions detected"));
    assert!(ctx2.contains("3 vs 7"));
}
