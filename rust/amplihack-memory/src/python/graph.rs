//! PyO3 bindings for graph operations — nodes, edges, traversal, queries.

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::graph::in_memory_store::InMemoryGraphStore;
use crate::graph::protocol::GraphStore;
use crate::graph::types::{Direction, GraphEdge, GraphNode};

use super::helpers::mem_err;

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

fn node_to_pydict(py: Python<'_>, node: &GraphNode) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    d.set_item("node_id", &node.node_id)?;
    d.set_item("node_type", &node.node_type)?;
    let props = PyDict::new_bound(py);
    for (k, v) in &node.properties {
        props.set_item(k, v)?;
    }
    d.set_item("properties", props)?;
    d.set_item("graph_origin", &node.graph_origin)?;
    Ok(d.to_object(py))
}

fn edge_to_pydict(py: Python<'_>, edge: &GraphEdge) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    d.set_item("edge_id", &edge.edge_id)?;
    d.set_item("source", &edge.source_id)?;
    d.set_item("target", &edge.target_id)?;
    d.set_item("edge_type", &edge.edge_type)?;
    let props = PyDict::new_bound(py);
    for (k, v) in &edge.properties {
        props.set_item(k, v)?;
    }
    d.set_item("properties", props)?;
    d.set_item("graph_origin", &edge.graph_origin)?;
    Ok(d.to_object(py))
}

fn parse_direction(s: &str) -> PyResult<Direction> {
    match s.to_lowercase().as_str() {
        "outgoing" | "out" => Ok(Direction::Outgoing),
        "incoming" | "in" => Ok(Direction::Incoming),
        "both" => Ok(Direction::Both),
        other => Err(PyValueError::new_err(format!(
            "Invalid direction '{other}'. Use 'outgoing', 'incoming', or 'both'."
        ))),
    }
}

fn pydict_to_string_map(dict: Option<&Bound<'_, PyDict>>) -> PyResult<HashMap<String, String>> {
    match dict {
        None => Ok(HashMap::new()),
        Some(d) => {
            let mut map = HashMap::new();
            for (k, v) in d {
                let key: String = k.extract()?;
                let val: String = v.extract()?;
                map.insert(key, val);
            }
            Ok(map)
        }
    }
}

// ---------------------------------------------------------------------------
// PyGraphStore
// ---------------------------------------------------------------------------

/// In-memory graph store for node/edge CRUD and traversal.
///
/// Example
/// -------
/// ```python
/// from amplihack_memory_rs import GraphStore
/// g = GraphStore()
/// g.add_node("n1", "Person", {"name": "Alice"})
/// g.add_node("n2", "Person", {"name": "Bob"})
/// g.add_edge("n1", "n2", "KNOWS")
/// print(g.query_neighbors("n1", 1))
/// ```
#[pyclass(name = "GraphStore")]
pub struct PyGraphStore {
    inner: InMemoryGraphStore,
}

#[pymethods]
impl PyGraphStore {
    /// Create a new in-memory graph store.
    ///
    /// Parameters
    /// ----------
    /// store_id : str, optional
    ///     Identifier for this store. Auto-generated if omitted.
    #[new]
    #[pyo3(signature = (store_id=None))]
    fn new(store_id: Option<&str>) -> Self {
        Self {
            inner: InMemoryGraphStore::new(store_id),
        }
    }

    /// Add a node to the graph.
    ///
    /// Returns the created node as a dict, or raises on failure.
    #[pyo3(signature = (id, node_type, properties=None))]
    fn add_node(
        &mut self,
        py: Python<'_>,
        id: &str,
        node_type: &str,
        properties: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        let props = pydict_to_string_map(properties)?;
        let node = self
            .inner
            .add_node(node_type, props, Some(id))
            .map_err(mem_err)?;
        node_to_pydict(py, &node)
    }

    /// Get a node by ID, returning a dict or None.
    fn get_node(&self, py: Python<'_>, id: &str) -> PyResult<Option<PyObject>> {
        match self.inner.get_node(id) {
            Some(node) => Ok(Some(node_to_pydict(py, &node)?)),
            None => Ok(None),
        }
    }

    /// Update properties on an existing node.
    ///
    /// Returns True if the node was found and updated.
    #[pyo3(signature = (id, properties))]
    fn update_node(&mut self, id: &str, properties: &Bound<'_, PyDict>) -> PyResult<bool> {
        let props = pydict_to_string_map(Some(properties))?;
        Ok(self.inner.update_node(id, props))
    }

    /// Delete a node by ID.
    ///
    /// Returns True if the node existed and was removed.
    fn delete_node(&mut self, id: &str) -> bool {
        self.inner.delete_node(id)
    }

    /// Add a directed edge between two existing nodes.
    ///
    /// Returns the created edge as a dict, or raises on failure.
    #[pyo3(signature = (source, target, edge_type, properties=None))]
    fn add_edge(
        &mut self,
        py: Python<'_>,
        source: &str,
        target: &str,
        edge_type: &str,
        properties: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        let props = match properties {
            Some(d) => Some(pydict_to_string_map(Some(d))?),
            None => None,
        };
        let edge = self
            .inner
            .add_edge(source, target, edge_type, props)
            .map_err(mem_err)?;
        edge_to_pydict(py, &edge)
    }

    /// Get edges adjacent to a node.
    ///
    /// Parameters
    /// ----------
    /// node_id : str
    ///     The node whose edges to retrieve.
    /// direction : str, optional
    ///     "outgoing", "incoming", or "both" (default: "both").
    /// edge_type : str, optional
    ///     Filter by edge type.
    /// limit : int, optional
    ///     Max number of results (default: 100).
    ///
    /// Returns a list of dicts, each containing edge and neighbor info.
    #[pyo3(signature = (node_id, direction="both", edge_type=None, limit=100))]
    fn get_edges(
        &self,
        py: Python<'_>,
        node_id: &str,
        direction: &str,
        edge_type: Option<&str>,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        let dir = parse_direction(direction)?;
        let results = self.inner.query_neighbors(node_id, edge_type, dir, limit);
        results
            .iter()
            .map(|(edge, neighbor)| {
                let d = PyDict::new_bound(py);
                d.set_item("edge", edge_to_pydict(py, edge)?)?;
                d.set_item("neighbor", node_to_pydict(py, neighbor)?)?;
                Ok(d.to_object(py))
            })
            .collect()
    }

    /// Delete a specific edge.
    ///
    /// Returns True if the edge existed and was removed.
    fn delete_edge(&mut self, source: &str, target: &str, edge_type: &str) -> bool {
        self.inner.delete_edge(source, target, edge_type)
    }

    /// Search nodes by keyword across all string properties.
    ///
    /// Parameters
    /// ----------
    /// query : str
    ///     Search text to match against node properties.
    /// limit : int, optional
    ///     Max results (default: 10).
    /// node_type : str, optional
    ///     Filter to a specific node type (default: searches all types).
    #[pyo3(signature = (query, limit=10, node_type=None))]
    fn search_nodes(
        &self,
        py: Python<'_>,
        query: &str,
        limit: usize,
        node_type: Option<&str>,
    ) -> PyResult<Vec<PyObject>> {
        let nodes = self.inner.search_all_properties(query, node_type, limit);
        nodes.iter().map(|n| node_to_pydict(py, n)).collect()
    }

    /// Find neighbors within N hops of a node.
    ///
    /// Parameters
    /// ----------
    /// node_id : str
    ///     Start node for traversal.
    /// depth : int, optional
    ///     Maximum traversal hops (default: 1).
    /// edge_type : str, optional
    ///     Filter traversal to a specific edge type.
    /// direction : str, optional
    ///     "outgoing", "incoming", or "both" (default: "both").
    ///
    /// Returns a list of node dicts discovered during traversal.
    #[pyo3(signature = (node_id, depth=1, edge_type=None, direction="both"))]
    fn query_neighbors(
        &self,
        py: Python<'_>,
        node_id: &str,
        depth: usize,
        edge_type: Option<&str>,
        direction: &str,
    ) -> PyResult<Vec<PyObject>> {
        let dir = parse_direction(direction)?;
        let edge_types: Option<Vec<String>> = edge_type.map(|et| vec![et.to_string()]);
        let result = self.inner.traverse(
            node_id,
            edge_types.as_deref(),
            depth,
            dir,
            None,
        );
        result
            .nodes
            .iter()
            .map(|node| node_to_pydict(py, node))
            .collect()
    }

    /// Return the number of nodes in the graph.
    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Return the number of edges in the graph.
    fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    fn __repr__(&self) -> String {
        format!(
            "GraphStore(store_id='{}', nodes={}, edges={})",
            self.inner.store_id(),
            self.inner.node_count(),
            self.inner.edge_count(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_direction() {
        assert_eq!(parse_direction("outgoing").unwrap(), Direction::Outgoing);
        assert_eq!(parse_direction("out").unwrap(), Direction::Outgoing);
        assert_eq!(parse_direction("incoming").unwrap(), Direction::Incoming);
        assert_eq!(parse_direction("in").unwrap(), Direction::Incoming);
        assert_eq!(parse_direction("both").unwrap(), Direction::Both);
        assert_eq!(parse_direction("BOTH").unwrap(), Direction::Both);
        assert!(parse_direction("invalid").is_err());
    }

    #[test]
    fn test_pydict_to_string_map_none() {
        let map = pydict_to_string_map(None).unwrap();
        assert!(map.is_empty());
    }
}
