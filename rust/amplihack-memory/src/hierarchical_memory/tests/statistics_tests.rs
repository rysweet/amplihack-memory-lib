use super::super::*;
use crate::memory_types::MemoryCategory;

fn make_mem() -> HierarchicalMemory {
    HierarchicalMemory::new("test_agent").unwrap()
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
    let items = result.get("items").unwrap().as_array().unwrap();
    let item_map: std::collections::HashMap<String, u64> = items
        .iter()
        .filter_map(|v| {
            let pair = v.as_array()?;
            Some((pair[0].as_str()?.to_string(), pair[1].as_u64()?))
        })
        .collect();
    assert!(item_map.contains_key("alpha"));
    assert_eq!(item_map.get("alpha"), Some(&2));
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
    let items = result.get("items").unwrap().as_array().unwrap();
    assert!(items.len() <= 2);
}

#[test]
fn test_qa_top_concepts_is_array() {
    let mut mem = make_mem();
    for _ in 0..5 {
        mem.store_knowledge("Plants grow", "plants", 0.9, None, "", &[], None)
            .unwrap();
    }
    for _ in 0..3 {
        mem.store_knowledge("Animals eat", "animals", 0.8, None, "", &[], None)
            .unwrap();
    }
    let result = mem.execute_aggregation("top_concepts", "", 10);
    let items = result.get("items").unwrap();
    assert!(items.is_array());
}
