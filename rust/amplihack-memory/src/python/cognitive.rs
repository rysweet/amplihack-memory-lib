//! PyO3 wrapper for the `CognitiveMemory` six-type memory system.
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::cognitive_memory::CognitiveMemory;

use super::converters::{
    episode_to_dict, fact_to_dict, procedure_to_dict, prospective_to_dict, sensory_to_dict,
    working_to_dict,
};
use super::helpers::{mem_err, pydict_to_hashmap};

/// Six-type cognitive memory system (sensory, working, episodic, semantic,
/// procedural, prospective) backed by an in-memory graph.
#[pyclass(name = "CognitiveMemory")]
pub struct PyCognitiveMemory {
    inner: CognitiveMemory,
}

#[pymethods]
impl PyCognitiveMemory {
    #[new]
    fn new(agent_name: &str) -> PyResult<Self> {
        let cm = CognitiveMemory::new(agent_name).map_err(mem_err)?;
        Ok(Self { inner: cm })
    }

    // -- Sensory --

    /// Record a sensory observation.
    #[pyo3(signature = (modality, raw_data, ttl_seconds=30))]
    fn record_sensory(
        &mut self,
        modality: &str,
        raw_data: &str,
        ttl_seconds: i64,
    ) -> PyResult<String> {
        self.inner
            .store_sensory(modality, raw_data, ttl_seconds)
            .map_err(mem_err)
    }

    /// Get recent sensory items as list of dicts.
    #[pyo3(signature = (limit=10))]
    fn get_recent_sensory(&self, py: Python<'_>, limit: usize) -> PyResult<Vec<PyObject>> {
        self.inner
            .get_sensory(limit)
            .iter()
            .map(|s| sensory_to_dict(py, s))
            .collect()
    }

    /// Prune expired sensory items. Returns count removed.
    fn prune_expired_sensory(&mut self) -> usize {
        self.inner.prune_expired_sensory()
    }

    // -- Working --

    /// Push an item to working memory.
    #[pyo3(signature = (slot_type, content, task_id, relevance=1.0))]
    fn push_working(
        &mut self,
        slot_type: &str,
        content: &str,
        task_id: &str,
        relevance: f64,
    ) -> PyResult<String> {
        self.inner
            .store_working(slot_type, content, task_id, relevance)
            .map_err(mem_err)
    }

    /// Recall working memory items for a task.
    fn recall_working(&self, py: Python<'_>, task_id: &str) -> PyResult<Vec<PyObject>> {
        self.inner
            .get_working(task_id)
            .iter()
            .map(|w| working_to_dict(py, w))
            .collect()
    }

    /// Clear working memory for a task. Returns count cleared.
    fn clear_working(&mut self, task_id: &str) -> usize {
        self.inner.clear_working(task_id)
    }

    // -- Episodic --

    /// Store an episodic memory.
    #[pyo3(signature = (content, source_label, temporal_index=None, metadata=None))]
    fn store_episode(
        &mut self,
        content: &str,
        source_label: &str,
        temporal_index: Option<i64>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<String> {
        let meta = match metadata {
            Some(d) => Some(pydict_to_hashmap(Some(d))?),
            None => None,
        };
        self.inner
            .store_episode(content, source_label, temporal_index, meta.as_ref())
            .map_err(mem_err)
    }

    /// Recall episodes.
    #[pyo3(signature = (limit=10))]
    fn recall_episodes(&self, py: Python<'_>, limit: usize) -> PyResult<Vec<PyObject>> {
        self.inner
            .search_episodes(limit)
            .iter()
            .map(|e| episode_to_dict(py, e))
            .collect()
    }

    // -- Semantic --

    /// Store a semantic fact.
    #[pyo3(signature = (concept, content, confidence, source_id, tags=None, metadata=None))]
    fn store_fact(
        &mut self,
        concept: &str,
        content: &str,
        confidence: f64,
        source_id: &str,
        tags: Option<Vec<String>>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<String> {
        let tag_vec: Vec<String> = tags.unwrap_or_default();
        let meta = match metadata {
            Some(d) => Some(pydict_to_hashmap(Some(d))?),
            None => None,
        };
        self.inner
            .store_fact(
                concept,
                content,
                confidence,
                source_id,
                Some(&tag_vec),
                meta.as_ref(),
            )
            .map_err(mem_err)
    }

    /// Search semantic facts.
    #[pyo3(signature = (query, limit=10, min_confidence=0.0))]
    fn search_facts(
        &self,
        py: Python<'_>,
        query: &str,
        limit: usize,
        min_confidence: f64,
    ) -> PyResult<Vec<PyObject>> {
        self.inner
            .search_facts(query, limit, min_confidence)
            .iter()
            .map(|f| fact_to_dict(py, f))
            .collect()
    }

    // -- Procedural --

    /// Store a procedure.
    #[pyo3(signature = (name, steps, prerequisites=None))]
    fn store_procedure(
        &mut self,
        name: &str,
        steps: Vec<String>,
        prerequisites: Option<Vec<String>>,
    ) -> PyResult<String> {
        let prereqs = prerequisites.unwrap_or_default();
        self.inner
            .store_procedure(name, &steps, Some(&prereqs))
            .map_err(mem_err)
    }

    /// Recall procedures matching a query.
    #[pyo3(signature = (query, limit=5))]
    fn recall_procedures(
        &mut self,
        py: Python<'_>,
        query: &str,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        self.inner
            .search_procedures_mut(query, limit)
            .iter()
            .map(|p| procedure_to_dict(py, p))
            .collect()
    }

    // -- Prospective --

    /// Store a prospective (future-oriented) memory.
    #[pyo3(signature = (description, trigger_condition, action_on_trigger, priority=0))]
    fn store_prospective(
        &mut self,
        description: &str,
        trigger_condition: &str,
        action_on_trigger: &str,
        priority: i32,
    ) -> PyResult<String> {
        self.inner
            .store_prospective(description, trigger_condition, action_on_trigger, priority)
            .map_err(mem_err)
    }

    /// Check which prospective memories are triggered by content.
    fn check_triggers(&mut self, py: Python<'_>, content: &str) -> PyResult<Vec<PyObject>> {
        self.inner
            .check_triggers(content)
            .iter()
            .map(|p| prospective_to_dict(py, p))
            .collect()
    }

    // -- Stats --

    /// Get memory statistics as a dict.
    fn get_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats = self.inner.get_statistics();
        let d = PyDict::new_bound(py);
        for (k, v) in &stats {
            d.set_item(k, *v)?;
        }
        Ok(d.to_object(py))
    }

    fn close(&mut self) {
        self.inner.close();
    }
}
