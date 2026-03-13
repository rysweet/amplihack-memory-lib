use super::super::*;
use crate::graph::Direction;

fn make_cm() -> CognitiveMemory {
    CognitiveMemory::new(&format!("test-agent-{}", uuid::Uuid::new_v4())).unwrap()
}

// -- semantic memory --

#[test]
fn test_store_and_search_facts() {
    let mut cm = make_cm();
    cm.store_fact("rust", "Rust is a systems language", 0.95, "", None, None)
        .unwrap();
    cm.store_fact("python", "Python is interpreted", 0.9, "", None, None)
        .unwrap();

    let results = cm.search_facts("rust", 10, 0.0);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].concept, "rust");
}

#[test]
fn test_search_facts_confidence_filter() {
    let mut cm = make_cm();
    cm.store_fact("a", "low confidence", 0.3, "", None, None)
        .unwrap();
    cm.store_fact("a", "high confidence", 0.9, "", None, None)
        .unwrap();

    let results = cm.search_facts("confidence", 10, 0.5);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content, "high confidence");
}

#[test]
fn test_search_facts_empty_query_returns_all() {
    let mut cm = make_cm();
    cm.store_fact("a", "content a", 1.0, "", None, None)
        .unwrap();
    cm.store_fact("b", "content b", 0.5, "", None, None)
        .unwrap();

    let results = cm.search_facts("", 10, 0.0);
    assert_eq!(results.len(), 2);
    // Sorted by confidence desc
    assert_eq!(results[0].concept, "a");
}

#[test]
fn test_get_all_facts() {
    let mut cm = make_cm();
    cm.store_fact("x", "xval", 1.0, "", None, None).unwrap();
    cm.store_fact("y", "yval", 0.5, "", None, None).unwrap();

    let all = cm.get_all_facts(50);
    assert_eq!(all.len(), 2);
}

#[test]
fn test_fact_with_tags_and_metadata() {
    let mut cm = make_cm();
    let tags = vec!["lang".to_string(), "systems".to_string()];
    let mut meta = HashMap::new();
    meta.insert(
        "source".to_string(),
        serde_json::Value::String("docs".into()),
    );

    let id = cm
        .store_fact("rust", "fast lang", 0.9, "doc-1", Some(&tags), Some(&meta))
        .unwrap();

    let facts = cm.search_facts("rust", 10, 0.0);
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].tags, tags);
    assert_eq!(
        facts[0].metadata.get("source").unwrap(),
        &serde_json::Value::String("docs".into())
    );
    assert_eq!(facts[0].source_id, "doc-1");
    assert_eq!(facts[0].node_id, id);
}

#[test]
fn test_link_similar_facts() {
    let mut cm = make_cm();
    let a = cm
        .store_fact("rust", "systems lang", 1.0, "", None, None)
        .unwrap();
    let b = cm
        .store_fact("cpp", "systems lang too", 1.0, "", None, None)
        .unwrap();

    assert!(cm.link_similar_facts(&a, &b, 0.85).is_ok());

    // Verify the SIMILAR_TO edge was actually created
    let neighbors = cm.graph.query_neighbors(&a, None, Direction::Both, 10);
    assert!(
        neighbors
            .iter()
            .any(|(e, _)| e.edge_type == "SIMILAR_TO" && e.target_id == b),
        "SIMILAR_TO edge should exist between the two facts"
    );
}

// -- Extended coverage: store_fact --

#[test]
fn test_store_fact_invalid_confidence_nan() {
    let mut cm = make_cm();
    assert!(cm
        .store_fact("c", "content", f64::NAN, "", None, None)
        .is_err());
}

#[test]
fn test_store_fact_invalid_confidence_out_of_range() {
    let mut cm = make_cm();
    assert!(cm.store_fact("c", "content", 1.5, "", None, None).is_err());
    assert!(cm.store_fact("c", "content", -0.1, "", None, None).is_err());
}

#[test]
fn test_store_fact_boundary_confidence() {
    let mut cm = make_cm();
    let id_zero = cm
        .store_fact("c", "zero conf", 0.0, "", None, None)
        .unwrap();
    let id_max = cm.store_fact("c", "max conf", 1.0, "", None, None).unwrap();

    // Verify boundary values were persisted correctly
    let facts = cm.search_facts("conf", 10, 0.0);
    let zero_fact = facts.iter().find(|f| f.node_id == id_zero);
    let max_fact = facts.iter().find(|f| f.node_id == id_max);
    assert!(
        zero_fact.is_some(),
        "zero-confidence fact should be retrievable"
    );
    assert!(
        max_fact.is_some(),
        "max-confidence fact should be retrievable"
    );
    assert!((zero_fact.unwrap().confidence - 0.0).abs() < f64::EPSILON);
    assert!((max_fact.unwrap().confidence - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_store_fact_returns_searchable_id() {
    let mut cm = make_cm();
    let id = cm
        .store_fact("rust-lang", "Rust is safe", 0.95, "src1", None, None)
        .unwrap();
    assert!(id.starts_with("sem_"));

    let facts = cm.search_facts("rust", 10, 0.0);
    assert!(facts.iter().any(|f| f.node_id == id));
}
