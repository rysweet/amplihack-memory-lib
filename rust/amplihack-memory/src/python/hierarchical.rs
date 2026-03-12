//! PyO3 wrapper for the `HierarchicalMemory` knowledge graph.
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::hierarchical_memory::HierarchicalMemory;

use super::converters::{knowledge_edge_to_dict, knowledge_node_to_dict};
use super::helpers::{json_to_py, mem_err, parse_category, pydict_to_hashmap};

/// Hierarchical knowledge graph memory with concept-based retrieval.
#[pyclass(name = "HierarchicalMemory")]
pub struct PyHierarchicalMemory {
    inner: HierarchicalMemory,
}

#[pymethods]
impl PyHierarchicalMemory {
    #[new]
    fn new(agent_name: &str) -> PyResult<Self> {
        let hm = HierarchicalMemory::new(agent_name).map_err(mem_err)?;
        Ok(Self { inner: hm })
    }

    /// Store a knowledge node.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (content, concept, confidence, source_id, tags=None, metadata=None, category=None))]
    fn store_knowledge(
        &mut self,
        content: &str,
        concept: &str,
        confidence: f64,
        source_id: &str,
        tags: Option<Vec<String>>,
        metadata: Option<&Bound<'_, PyDict>>,
        category: Option<&str>,
    ) -> PyResult<String> {
        let tag_vec = tags.unwrap_or_default();
        let cat = category.map(parse_category).transpose()?;
        let meta = match metadata {
            Some(d) => Some(pydict_to_hashmap(Some(d))?),
            None => None,
        };
        self.inner
            .store_knowledge(
                content,
                concept,
                confidence,
                cat,
                source_id,
                &tag_vec,
                meta.as_ref(),
            )
            .map_err(mem_err)
    }

    /// Retrieve a subgraph around a query.
    #[pyo3(signature = (query, max_depth=2, max_nodes=20))]
    fn retrieve_subgraph(
        &self,
        py: Python<'_>,
        query: &str,
        max_depth: usize,
        max_nodes: usize,
    ) -> PyResult<PyObject> {
        let sg = self.inner.retrieve_subgraph(query, max_depth, max_nodes);
        let d = PyDict::new_bound(py);
        let nodes: PyResult<Vec<PyObject>> = sg
            .nodes
            .iter()
            .map(|n| knowledge_node_to_dict(py, n))
            .collect();
        let edges: PyResult<Vec<PyObject>> = sg
            .edges
            .iter()
            .map(|e| knowledge_edge_to_dict(py, e))
            .collect();
        d.set_item("nodes", PyList::new_bound(py, nodes?))?;
        d.set_item("edges", PyList::new_bound(py, edges?))?;
        d.set_item("query", &sg.query)?;
        Ok(d.to_object(py))
    }

    /// Search by concept keywords.
    #[pyo3(signature = (keywords, limit=10))]
    fn search_by_concept(
        &self,
        py: Python<'_>,
        keywords: Vec<String>,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        self.inner
            .search_by_concept(&keywords, limit)
            .iter()
            .map(|n| knowledge_node_to_dict(py, n))
            .collect()
    }

    /// Search by entity name.
    #[pyo3(signature = (entity_name, limit=10))]
    fn search_by_entity(
        &self,
        py: Python<'_>,
        entity_name: &str,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        self.inner
            .search_by_entity(entity_name, limit)
            .iter()
            .map(|n| knowledge_node_to_dict(py, n))
            .collect()
    }

    /// Get all stored knowledge, optionally filtered by category.
    #[pyo3(signature = (category=None, limit=100))]
    fn get_all_knowledge(
        &self,
        py: Python<'_>,
        category: Option<&str>,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        let cat = category.map(parse_category).transpose()?;
        self.inner
            .get_all_knowledge(cat, limit)
            .iter()
            .map(|n| knowledge_node_to_dict(py, n))
            .collect()
    }

    /// Get statistics as a dict.
    fn get_statistics(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats = self.inner.get_statistics();
        let d = PyDict::new_bound(py);
        for (k, v) in &stats {
            d.set_item(k, json_to_py(py, v)?)?;
        }
        Ok(d.to_object(py))
    }

    fn close(&mut self) {
        self.inner.close();
    }
}
