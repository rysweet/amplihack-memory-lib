use std::collections::HashMap;

use crate::graph::kuzu_store::KuzuGraphStore;
use crate::graph::protocol::GraphStore;
use crate::graph::types::Direction;

fn make_store() -> (KuzuGraphStore, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().unwrap();
    let db_path = tmp.path().join("test_kuzu_db");
    let store = KuzuGraphStore::new(&db_path, Some("test"), None).unwrap();
    (store, tmp)
}

#[test]
fn test_add_and_get_node() {
    let (mut store, _tmp) = make_store();
    let mut props = HashMap::new();
    props.insert("name".into(), "Alice".into());
    let node = store.add_node("Person", props, Some("p1")).unwrap();
    assert_eq!(node.node_id, "p1");
    assert_eq!(node.node_type, "Person");

    let fetched = store.get_node("p1").unwrap();
    assert_eq!(fetched.properties.get("name").unwrap(), "Alice");
    assert_eq!(fetched.node_type, "Person");
}

#[test]
fn test_add_node_auto_id() {
    let (mut store, _tmp) = make_store();
    let node = store.add_node("Agent", HashMap::new(), None).unwrap();
    assert!(!node.node_id.is_empty());
}

#[test]
fn test_query_nodes() {
    let (mut store, _tmp) = make_store();
    let mut props1 = HashMap::new();
    props1.insert("role".into(), "developer".into());
    store.add_node("Person", props1, Some("q1")).unwrap();

    let mut props2 = HashMap::new();
    props2.insert("role".into(), "manager".into());
    store.add_node("Person", props2, Some("q2")).unwrap();

    let all = store.query_nodes("Person", None, 50);
    assert_eq!(all.len(), 2);

    let mut filter = HashMap::new();
    filter.insert("role".into(), "developer".into());
    let filtered = store.query_nodes("Person", Some(&filter), 50);
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].node_id, "q1");
}

#[test]
fn test_search_nodes() {
    let (mut store, _tmp) = make_store();
    let mut props = HashMap::new();
    props.insert("content".into(), "Rust programming language".into());
    store.add_node("Fact", props, Some("f1")).unwrap();

    let mut props2 = HashMap::new();
    props2.insert("content".into(), "Python scripting".into());
    store.add_node("Fact", props2, Some("f2")).unwrap();

    let results = store.search_nodes("Fact", &["content".to_string()], "rust", None, 10);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].node_id, "f1");

    let results2 = store.search_nodes("Fact", &["content".to_string()], "PYTHON", None, 10);
    assert_eq!(results2.len(), 1);
    assert_eq!(results2[0].node_id, "f2");
}

#[test]
fn test_update_node() {
    let (mut store, _tmp) = make_store();
    let mut props = HashMap::new();
    props.insert("name".into(), "Bob".into());
    store.add_node("Person", props, Some("u1")).unwrap();

    let mut updates = HashMap::new();
    updates.insert("name".into(), "Robert".into());
    assert!(store.update_node("u1", updates));

    let fetched = store.get_node("u1").unwrap();
    assert_eq!(fetched.properties.get("name").unwrap(), "Robert");
}

#[test]
fn test_delete_node() {
    let (mut store, _tmp) = make_store();
    store
        .add_node("Person", HashMap::new(), Some("d1"))
        .unwrap();
    assert!(store.delete_node("d1"));
    assert!(store.get_node("d1").is_none());
    assert!(!store.delete_node("nonexistent"));
}

#[test]
fn test_add_edge_and_neighbors() {
    let (mut store, _tmp) = make_store();
    store.add_node("Person", HashMap::new(), Some("a")).unwrap();
    store.add_node("Person", HashMap::new(), Some("b")).unwrap();
    let edge = store.add_edge("a", "b", "KNOWS", None).unwrap();
    assert_eq!(edge.source_id, "a");
    assert_eq!(edge.target_id, "b");
    assert_eq!(edge.edge_type, "KNOWS");

    let neighbors = store.query_neighbors("a", Some("KNOWS"), Direction::Outgoing, 10);
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0].1.node_id, "b");

    let incoming = store.query_neighbors("b", Some("KNOWS"), Direction::Incoming, 10);
    assert_eq!(incoming.len(), 1);
    assert_eq!(incoming[0].1.node_id, "a");
}

#[test]
fn test_delete_edge() {
    let (mut store, _tmp) = make_store();
    store
        .add_node("Person", HashMap::new(), Some("e1"))
        .unwrap();
    store
        .add_node("Person", HashMap::new(), Some("e2"))
        .unwrap();
    store.add_edge("e1", "e2", "FOLLOWS", None).unwrap();

    assert!(store.delete_edge("e1", "e2", "FOLLOWS"));

    let neighbors = store.query_neighbors("e1", Some("FOLLOWS"), Direction::Outgoing, 10);
    assert!(neighbors.is_empty());
}

#[test]
fn test_traverse() {
    let (mut store, _tmp) = make_store();
    store.add_node("N", HashMap::new(), Some("a")).unwrap();
    store.add_node("N", HashMap::new(), Some("b")).unwrap();
    store.add_node("N", HashMap::new(), Some("c")).unwrap();
    store.add_edge("a", "b", "LINK", None).unwrap();
    store.add_edge("b", "c", "LINK", None).unwrap();

    let result = store.traverse("a", None, 3, Direction::Outgoing, None);
    assert!(result.nodes.len() >= 2);
    assert!(!result.edges.is_empty());
}

#[test]
fn test_edge_with_properties() {
    let (mut store, _tmp) = make_store();
    store
        .add_node("Person", HashMap::new(), Some("ep1"))
        .unwrap();
    store
        .add_node("Person", HashMap::new(), Some("ep2"))
        .unwrap();

    let mut edge_props = HashMap::new();
    edge_props.insert("weight".into(), "0.9".into());
    let edge = store
        .add_edge("ep1", "ep2", "RELATED", Some(edge_props))
        .unwrap();
    assert_eq!(edge.properties.get("weight").unwrap(), "0.9");
}

#[test]
fn test_close() {
    let (mut store, _tmp) = make_store();
    store
        .add_node("Person", HashMap::new(), Some("cl1"))
        .unwrap();
    store.close();
}

#[test]
fn test_invalid_identifier() {
    let (mut store, _tmp) = make_store();
    let result = store.add_node("invalid-type", HashMap::new(), Some("x"));
    assert!(result.is_err());
}

#[test]
fn test_query_nonexistent_type() {
    let (store, _tmp) = make_store();
    let results = store.query_nodes("NonExistent", None, 50);
    assert!(results.is_empty());
}
