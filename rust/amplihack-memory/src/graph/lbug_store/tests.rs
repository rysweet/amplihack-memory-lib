//! Direct `GraphStore` trait tests for [`LbugGraphStore`].

use std::collections::HashMap;

use super::LbugGraphStore;
use crate::graph::protocol::GraphStore;
use crate::graph::types::Direction;

fn props(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn open_temp() -> (tempfile::TempDir, LbugGraphStore) {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("graph.ladybug");
    let store = LbugGraphStore::open(&path, Some("test-store")).unwrap();
    (tmp, store)
}

#[test]
fn add_and_query_round_trip() {
    let (_tmp, mut store) = open_temp();
    store
        .add_node(
            "Thing",
            props(&[("node_id", "n1"), ("agent_id", "a"), ("name", "hello")]),
            Some("n1"),
        )
        .unwrap();

    let filter = props(&[("agent_id", "a")]);
    let nodes = store.query_nodes("Thing", Some(&filter), 10);
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].node_id, "n1");
    assert_eq!(nodes[0].properties.get("name").unwrap(), "hello");
    // node_id must be present in properties for converter compatibility.
    assert_eq!(nodes[0].properties.get("node_id").unwrap(), "n1");
}

#[test]
fn query_missing_table_is_empty() {
    let (_tmp, store) = open_temp();
    assert!(store.query_nodes("NeverCreated", None, 10).is_empty());
}

#[test]
fn get_update_delete_node() {
    let (_tmp, mut store) = open_temp();
    store
        .add_node(
            "Thing",
            props(&[("node_id", "n1"), ("agent_id", "a"), ("status", "pending")]),
            Some("n1"),
        )
        .unwrap();

    let got = store.get_node("n1").expect("node should exist");
    assert_eq!(got.properties.get("status").unwrap(), "pending");

    assert!(store.update_node("n1", props(&[("status", "done")])));
    let got = store.get_node("n1").unwrap();
    assert_eq!(got.properties.get("status").unwrap(), "done");

    assert!(store.delete_node("n1"));
    assert!(store.get_node("n1").is_none());
}

#[test]
fn escapes_quotes_in_values() {
    let (_tmp, mut store) = open_temp();
    let tricky = "O'Brien \\ \"x\" line\nbreak";
    store
        .add_node(
            "Thing",
            props(&[("node_id", "n1"), ("agent_id", "a"), ("name", tricky)]),
            Some("n1"),
        )
        .unwrap();
    let got = store.get_node("n1").unwrap();
    assert_eq!(got.properties.get("name").unwrap(), tricky);
}

#[test]
fn unbounded_limit_returns_all() {
    let (_tmp, mut store) = open_temp();
    for i in 0..5 {
        store
            .add_node(
                "Thing",
                props(&[("node_id", &format!("n{i}")), ("agent_id", "a")]),
                Some(&format!("n{i}")),
            )
            .unwrap();
    }
    let nodes = store.query_nodes("Thing", None, usize::MAX);
    assert_eq!(nodes.len(), 5);
}

#[test]
fn add_edge_and_query_neighbors() {
    let (_tmp, mut store) = open_temp();
    store
        .add_node(
            "Thing",
            props(&[("node_id", "a1"), ("agent_id", "a")]),
            Some("a1"),
        )
        .unwrap();
    store
        .add_node(
            "Thing",
            props(&[("node_id", "b1"), ("agent_id", "a")]),
            Some("b1"),
        )
        .unwrap();

    store
        .add_edge("a1", "b1", "LINKS", Some(props(&[("weight", "5")])))
        .unwrap();

    let neighbors = store.query_neighbors("a1", Some("LINKS"), Direction::Outgoing, 10);
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0].1.node_id, "b1");
    assert_eq!(neighbors[0].0.properties.get("weight").unwrap(), "5");

    assert!(store.delete_edge("a1", "b1", "LINKS"));
    assert!(store
        .query_neighbors("a1", Some("LINKS"), Direction::Outgoing, 10)
        .is_empty());
}

#[test]
fn data_survives_close_and_reopen() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("graph.ladybug");

    {
        let mut store = LbugGraphStore::open(&path, Some("s")).unwrap();
        store
            .add_node(
                "Thing",
                props(&[("node_id", "n1"), ("agent_id", "a"), ("name", "persisted")]),
                Some("n1"),
            )
            .unwrap();
        store.close();
    } // Database dropped here -> WAL flushed to disk.

    let store = LbugGraphStore::open(&path, Some("s")).unwrap();
    let nodes = store.query_nodes("Thing", None, 10);
    assert_eq!(nodes.len(), 1, "reopened store must see persisted node");
    assert_eq!(nodes[0].properties.get("name").unwrap(), "persisted");

    // get_node after reopen must resolve the table via catalog introspection.
    let got = store.get_node("n1").expect("get_node after reopen");
    assert_eq!(got.properties.get("name").unwrap(), "persisted");
}

#[test]
fn invalid_identifiers_are_rejected() {
    let (_tmp, mut store) = open_temp();
    assert!(store
        .add_node("Bad-Type", props(&[("node_id", "x")]), Some("x"))
        .is_err());
    assert!(store
        .add_node("Thing", props(&[("bad-key", "v")]), Some("x"))
        .is_err());
}

#[test]
fn query_neighbors_all_edge_types_and_directions() {
    let (_tmp, mut store) = open_temp();
    for id in ["a1", "b1", "c1"] {
        store
            .add_node(
                "Thing",
                props(&[("node_id", id), ("agent_id", "a")]),
                Some(id),
            )
            .unwrap();
    }
    // a1 -LINKS-> b1, a1 -REFS-> c1, c1 -LINKS-> a1
    store.add_edge("a1", "b1", "LINKS", None).unwrap();
    store.add_edge("a1", "c1", "REFS", None).unwrap();
    store.add_edge("c1", "a1", "LINKS", None).unwrap();

    // edge_type=None must return both outgoing edge types in one pass.
    let out = store.query_neighbors("a1", None, Direction::Outgoing, 10);
    assert_eq!(out.len(), 2);
    let mut types: Vec<&str> = out.iter().map(|(e, _)| e.edge_type.as_str()).collect();
    types.sort_unstable();
    assert_eq!(types, ["LINKS", "REFS"]);

    // Incoming should see the c1 -LINKS-> a1 edge.
    let inc = store.query_neighbors("a1", None, Direction::Incoming, 10);
    assert_eq!(inc.len(), 1);
    assert_eq!(inc[0].1.node_id, "c1");

    // Both = outgoing + incoming.
    let both = store.query_neighbors("a1", None, Direction::Both, 10);
    assert_eq!(both.len(), 3);

    // Filtering by edge_type narrows results.
    let refs = store.query_neighbors("a1", Some("REFS"), Direction::Outgoing, 10);
    assert_eq!(refs.len(), 1);
    assert_eq!(refs[0].1.node_id, "c1");

    // Unknown edge type yields nothing (no binder error).
    assert!(store
        .query_neighbors("a1", Some("NOPE"), Direction::Both, 10)
        .is_empty());
}

#[test]
fn get_node_resolves_across_types_after_reopen() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("graph.ladybug");
    {
        let mut store = LbugGraphStore::open(&path, Some("s")).unwrap();
        store
            .add_node(
                "Alpha",
                props(&[("node_id", "x1"), ("name", "ax")]),
                Some("x1"),
            )
            .unwrap();
        store
            .add_node(
                "Beta",
                props(&[("node_id", "y1"), ("name", "by")]),
                Some("y1"),
            )
            .unwrap();
        store.close();
    }

    // Fresh handle: id_table_cache is cold, so get_node exercises the
    // single-query label-less resolution across multiple node tables.
    let store = LbugGraphStore::open(&path, Some("s")).unwrap();
    let beta = store.get_node("y1").expect("beta resolves after reopen");
    assert_eq!(beta.node_type, "Beta");
    assert_eq!(beta.properties.get("name").unwrap(), "by");

    let alpha = store.get_node("x1").expect("alpha resolves after reopen");
    assert_eq!(alpha.node_type, "Alpha");

    // A missing id resolves to None without error.
    assert!(store.get_node("missing").is_none());
}

#[test]
fn execute_error_does_not_leak_query_values() {
    // A failing statement must not surface interpolated string values (stored
    // memory content, agent_id) in the propagating error. The engine may echo
    // schema identifiers (validated to [A-Za-z_][A-Za-z0-9_]*), but never the
    // single-quoted values; the full query text is logged at debug level only.
    let (_tmp, store) = open_temp();
    let secret = "s3cr3t-memory-value";

    let cypher = format!("MATCH (n:Missing) WHERE n.node_id = '{secret}' RETURN bogus_fn(n)");
    let err = store
        .execute(&cypher)
        .expect_err("invalid Cypher should error");
    assert!(
        !err.to_string().contains(secret),
        "error must not leak interpolated values: {err}"
    );

    let qerr = store
        .query_rows(&cypher)
        .expect_err("invalid query should error");
    assert!(
        !qerr.to_string().contains(secret),
        "error must not leak interpolated values: {qerr}"
    );
    assert!(qerr.to_string().contains("Cypher query failed"));
}
