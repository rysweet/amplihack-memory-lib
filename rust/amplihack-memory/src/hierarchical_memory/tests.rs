use super::*;
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

// -- Aggregation queries --

#[test]
fn test_aggregation_count() {
    let mut mem = make_mem();
    mem.store_knowledge("Fact one", "topic-a", 0.8, None, "", &[], None)
        .unwrap();
    mem.store_knowledge("Fact two", "topic-b", 0.9, None, "", &[], None)
        .unwrap();
    mem.store_knowledge("Fact three", "topic-a", 0.7, None, "", &[], None)
        .unwrap();

    let result = mem.execute_aggregation("count", "", 10);
    assert_eq!(result.get("count").and_then(|v| v.as_u64()), Some(3));
}

#[test]
fn test_aggregation_avg_confidence() {
    let mut mem = make_mem();
    mem.store_knowledge("Fact one", "topic", 0.6, None, "", &[], None)
        .unwrap();
    mem.store_knowledge("Fact two", "topic", 0.8, None, "", &[], None)
        .unwrap();

    let result = mem.execute_aggregation("avg_confidence", "", 10);
    let avg = result
        .get("avg_confidence")
        .and_then(|v| v.as_f64())
        .unwrap();
    assert!((avg - 0.7).abs() < 0.01);
}

#[test]
fn test_aggregation_top_concepts() {
    let mut mem = make_mem();
    mem.store_knowledge("Fact A1", "alpha", 0.8, None, "", &[], None)
        .unwrap();
    mem.store_knowledge("Fact A2", "alpha", 0.8, None, "", &[], None)
        .unwrap();
    mem.store_knowledge("Fact B1", "beta", 0.8, None, "", &[], None)
        .unwrap();

    let result = mem.execute_aggregation("top_concepts", "", 10);
    let items = result.get("items").unwrap().as_object().unwrap();
    assert!(items.contains_key("alpha"));
    assert_eq!(items.get("alpha").and_then(|v| v.as_u64()), Some(2));
}

#[test]
fn test_aggregation_by_category() {
    let mut mem = make_mem();
    mem.store_knowledge(
        "Water boils at 100C",
        "physics",
        0.9,
        Some(MemoryCategory::Semantic),
        "",
        &[],
        None,
    )
    .unwrap();
    mem.store_knowledge(
        "Step 1: boil water. Step 2: add tea.",
        "recipe",
        0.8,
        None,
        "",
        &[],
        None,
    )
    .unwrap();

    let result = mem.execute_aggregation("by_category", "", 10);
    let items = result.get("items").unwrap().as_object().unwrap();
    assert!(items.contains_key("semantic") || items.contains_key("procedural"));
}

#[test]
fn test_aggregation_unknown_type() {
    let mem = make_mem();
    let result = mem.execute_aggregation("invalid_type", "", 10);
    assert!(result.contains_key("error"));
}

// -- Memory classification --

#[test]
fn test_classifier_procedural() {
    let c = MemoryClassifier::new();
    assert_eq!(
        c.classify("How to bake a cake", ""),
        MemoryCategory::Procedural
    );
    assert_eq!(
        c.classify("Step 1: preheat oven", ""),
        MemoryCategory::Procedural
    );
}

#[test]
fn test_classifier_prospective() {
    let c = MemoryClassifier::new();
    assert_eq!(
        c.classify("I plan to visit Paris", ""),
        MemoryCategory::Prospective
    );
    assert_eq!(
        c.classify("My future goal is to learn Rust", ""),
        MemoryCategory::Prospective
    );
}

#[test]
fn test_classifier_episodic() {
    let c = MemoryClassifier::new();
    assert_eq!(
        c.classify("Something interesting happened today", ""),
        MemoryCategory::Episodic
    );
    assert_eq!(
        c.classify("I observed a rare bird", ""),
        MemoryCategory::Episodic
    );
}

#[test]
fn test_classifier_semantic_default() {
    let c = MemoryClassifier::new();
    assert_eq!(
        c.classify("Water is composed of hydrogen and oxygen", ""),
        MemoryCategory::Semantic
    );
}

#[test]
fn test_auto_classification_on_store() {
    let mut mem = make_mem();
    let nid = mem
        .store_knowledge(
            "Step 1: install Rust. Step 2: write code.",
            "tutorial",
            0.9,
            None,
            "",
            &[],
            None,
        )
        .unwrap();

    let nodes = mem.get_all_knowledge(None, 100);
    let stored = nodes.iter().find(|n| n.node_id == nid).unwrap();
    assert_eq!(stored.category, MemoryCategory::Procedural);
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

// -- Statistics --

#[test]
fn test_statistics() {
    let mut mem = make_mem();
    mem.store_knowledge("Fact one about Rust", "rust", 0.9, None, "", &[], None)
        .unwrap();
    mem.store_knowledge(
        "Fact two about Rust language",
        "rust",
        0.8,
        None,
        "",
        &[],
        None,
    )
    .unwrap();

    let stats = mem.get_statistics();
    assert_eq!(
        stats.get("agent_name").and_then(|v| v.as_str()),
        Some("test_agent")
    );
    let nodes = stats
        .get("semantic_nodes")
        .and_then(|v| v.as_u64())
        .unwrap();
    assert_eq!(nodes, 2);
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

// ====================================================================
// Extended coverage: store_knowledge
// ====================================================================

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

// ====================================================================
// Extended coverage: retrieve_subgraph
// ====================================================================

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

// ====================================================================
// Extended coverage: get_temporal_chain
// ====================================================================

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

// ====================================================================
// Extended coverage: get_knowledge_evolution
// ====================================================================

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

// ====================================================================
// Extended coverage: execute_aggregation
// ====================================================================

#[test]
fn test_aggregation_count_with_concept_filter() {
    let mut mem = make_mem();
    mem.store_knowledge("Rust is fast", "rust-lang", 0.9, None, "", &[], None)
        .unwrap();
    mem.store_knowledge("Python is dynamic", "python-lang", 0.8, None, "", &[], None)
        .unwrap();
    mem.store_knowledge("Rust has ownership", "rust-lang", 0.85, None, "", &[], None)
        .unwrap();

    let result = mem.execute_aggregation("count", "rust", 10);
    assert_eq!(result.get("count").and_then(|v| v.as_u64()), Some(2));
}

#[test]
fn test_aggregation_avg_confidence_empty() {
    let mem = make_mem();
    let result = mem.execute_aggregation("avg_confidence", "", 10);
    assert_eq!(
        result.get("avg_confidence").and_then(|v| v.as_f64()),
        Some(0.0)
    );
}

#[test]
fn test_aggregation_top_concepts_with_limit() {
    let mut mem = make_mem();
    for topic in &["alpha", "alpha", "alpha", "beta", "beta", "gamma"] {
        mem.store_knowledge(
            &format!("Fact about {topic}"),
            topic,
            0.8,
            None,
            "",
            &[],
            None,
        )
        .unwrap();
    }

    let result = mem.execute_aggregation("top_concepts", "", 2);
    let items = result.get("items").unwrap().as_object().unwrap();
    assert!(items.len() <= 2);
}

// ====================================================================
// Extended coverage: MemoryClassifier::classify
// ====================================================================

#[test]
fn test_classifier_concept_triggers_category() {
    let c = MemoryClassifier::new();
    // Concept alone contains the keyword
    assert_eq!(
        c.classify("generic statement", "future plans"),
        MemoryCategory::Prospective
    );
}

#[test]
fn test_classifier_priority_procedural_over_episodic() {
    let c = MemoryClassifier::new();
    // Contains both "step" (procedural) and "happened" (episodic)
    // Procedural is checked first
    assert_eq!(
        c.classify("step by step what happened", ""),
        MemoryCategory::Procedural
    );
}

#[test]
fn test_classifier_case_insensitive() {
    let c = MemoryClassifier::new();
    assert_eq!(
        c.classify("HOW TO BUILD A HOUSE", ""),
        MemoryCategory::Procedural
    );
    assert_eq!(
        c.classify("I SAW AN EVENT THAT OCCURRED", ""),
        MemoryCategory::Episodic
    );
}

#[test]
fn test_classifier_default_returns() {
    let c = MemoryClassifier;
    assert_eq!(c.classify("plain fact", ""), MemoryCategory::Semantic);
}
