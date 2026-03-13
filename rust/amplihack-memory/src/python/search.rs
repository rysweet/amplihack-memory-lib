//! PyO3 wrapper for the `SemanticSearchEngine` struct.
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::semantic_search::{calculate_relevance, SemanticSearchEngine};

use super::experience::PyExperience;
use super::helpers::hashmap_to_pydict;

/// Semantic search engine with TF-IDF relevance scoring.
#[pyclass(name = "SemanticSearchEngine")]
pub struct PySemanticSearchEngine {
    inner: SemanticSearchEngine,
}

#[pymethods]
impl PySemanticSearchEngine {
    /// Create a new empty SemanticSearchEngine.
    #[new]
    fn new() -> Self {
        Self {
            inner: SemanticSearchEngine::new(vec![]),
        }
    }

    /// Add an experience to the search index.
    fn add_experience(&mut self, experience: &PyExperience) {
        self.inner.add_experience(experience.inner.clone());
    }

    /// Search for experiences matching a query.
    ///
    /// Args:
    ///     query: Search query string
    ///     limit: Maximum number of results to return
    ///
    /// Returns a list of dicts, each with the experience fields plus a "score".
    #[pyo3(signature = (query, limit=10))]
    fn search(&self, py: Python<'_>, query: &str, limit: usize) -> PyResult<PyObject> {
        let results = self.inner.search(query, limit);
        let list: Vec<PyObject> = results
            .iter()
            .map(|exp| {
                let score = calculate_relevance(exp, query);
                let d = PyDict::new_bound(py);
                d.set_item("id", &exp.experience_id)?;
                d.set_item("experience_type", exp.experience_type.as_str())?;
                d.set_item("context", &exp.context)?;
                d.set_item("outcome", &exp.outcome)?;
                d.set_item("confidence", exp.confidence)?;
                d.set_item("tags", exp.tags.to_object(py))?;
                d.set_item("timestamp", exp.timestamp.to_rfc3339())?;
                d.set_item("metadata", hashmap_to_pydict(py, &exp.metadata)?)?;
                d.set_item("score", score)?;
                Ok(d.to_object(py))
            })
            .collect::<PyResult<_>>()?;
        Ok(PyList::new_bound(py, list).to_object(py))
    }

    /// Return the number of experiences in the corpus.
    fn corpus_size(&self) -> usize {
        self.inner.corpus_size()
    }

    /// Check if the search index is ready.
    fn is_indexed(&self) -> bool {
        self.inner.is_indexed()
    }

    /// Remove an experience by ID.
    fn remove_experience(&mut self, experience_id: &str) {
        self.inner.remove_experience(experience_id);
    }

    fn __repr__(&self) -> String {
        format!(
            "SemanticSearchEngine(corpus_size={}, indexed={})",
            self.inner.corpus_size(),
            self.inner.is_indexed()
        )
    }
}
