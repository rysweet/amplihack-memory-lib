//! PyO3 bindings exposing amplihack-memory as a Python extension module.
//!
//! Compiled only with `--features python`. Produces a `amplihack_memory_rs`
//! native module importable from Python.

use std::collections::HashMap;
use std::path::Path;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::backends::base::StorageStatistics;
use crate::cognitive_memory::CognitiveMemory;
use crate::connector::MemoryConnector;
use crate::contradiction::detect_contradiction;
use crate::entity_extraction::extract_entity_name;
use crate::experience::{Experience, ExperienceType};
use crate::hierarchical_memory::{HierarchicalMemory, KnowledgeEdge, KnowledgeNode};
use crate::memory_types::{
    EpisodicMemory, MemoryCategory, ProceduralMemory, ProspectiveMemory, SemanticFact, SensoryItem,
    WorkingMemorySlot,
};
use crate::security::CredentialScrubber;
use crate::similarity::compute_word_similarity;
use crate::store::ExperienceStore;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn mem_err(e: crate::MemoryError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn parse_experience_type(s: &str) -> PyResult<ExperienceType> {
    s.parse::<ExperienceType>()
        .map_err(|e| PyRuntimeError::new_err(e))
}

fn json_to_py(py: Python<'_>, val: &serde_json::Value) -> PyObject {
    match val {
        serde_json::Value::Null => py.None(),
        serde_json::Value::Bool(b) => b.to_object(py),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_object(py)
            } else {
                n.as_f64().unwrap_or(0.0).to_object(py)
            }
        }
        serde_json::Value::String(s) => s.to_object(py),
        serde_json::Value::Array(arr) => {
            let items: Vec<PyObject> = arr.iter().map(|v| json_to_py(py, v)).collect();
            PyList::new_bound(py, items).to_object(py)
        }
        serde_json::Value::Object(map) => {
            let d = PyDict::new_bound(py);
            for (k, v) in map {
                d.set_item(k, json_to_py(py, v)).unwrap();
            }
            d.to_object(py)
        }
    }
}

fn py_to_json(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<serde_json::Value> {
    if obj.is_none() {
        Ok(serde_json::Value::Null)
    } else if let Ok(b) = obj.extract::<bool>() {
        Ok(serde_json::Value::Bool(b))
    } else if let Ok(i) = obj.extract::<i64>() {
        Ok(serde_json::json!(i))
    } else if let Ok(f) = obj.extract::<f64>() {
        Ok(serde_json::json!(f))
    } else if let Ok(s) = obj.extract::<String>() {
        Ok(serde_json::Value::String(s))
    } else if let Ok(list) = obj.downcast::<PyList>() {
        let items: PyResult<Vec<serde_json::Value>> =
            list.iter().map(|item| py_to_json(&item)).collect();
        Ok(serde_json::Value::Array(items?))
    } else if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (k, v) in dict {
            let key: String = k.extract()?;
            map.insert(key, py_to_json(&v)?);
        }
        Ok(serde_json::Value::Object(map))
    } else {
        Ok(serde_json::Value::String(obj.str()?.to_string()))
    }
}

fn pydict_to_hashmap(
    dict: Option<&Bound<'_, PyDict>>,
) -> PyResult<HashMap<String, serde_json::Value>> {
    match dict {
        None => Ok(HashMap::new()),
        Some(d) => {
            let mut map = HashMap::new();
            for (k, v) in d {
                let key: String = k.extract()?;
                map.insert(key, py_to_json(&v)?);
            }
            Ok(map)
        }
    }
}

fn hashmap_to_pydict(
    py: Python<'_>,
    map: &HashMap<String, serde_json::Value>,
) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    for (k, v) in map {
        d.set_item(k, json_to_py(py, v))?;
    }
    Ok(d.to_object(py))
}

fn storage_stats_to_dict(py: Python<'_>, stats: &StorageStatistics) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    d.set_item("total_experiences", stats.total_experiences)?;
    d.set_item("storage_size_kb", stats.storage_size_kb)?;
    d.set_item("compressed_experiences", stats.compressed_experiences)?;
    d.set_item("compression_ratio", stats.compression_ratio)?;
    let by_type = PyDict::new_bound(py);
    for (et, count) in &stats.by_type {
        by_type.set_item(et.as_str(), count)?;
    }
    d.set_item("by_type", by_type)?;
    Ok(d.to_object(py))
}

fn parse_category(s: &str) -> MemoryCategory {
    match s.to_lowercase().as_str() {
        "sensory" => MemoryCategory::Sensory,
        "working" => MemoryCategory::Working,
        "episodic" => MemoryCategory::Episodic,
        "procedural" => MemoryCategory::Procedural,
        "prospective" => MemoryCategory::Prospective,
        _ => MemoryCategory::Semantic,
    }
}

// ---------------------------------------------------------------------------
// PyExperience
// ---------------------------------------------------------------------------

/// A single experience record wrapping the Rust `Experience` struct.
#[pyclass(name = "Experience")]
#[derive(Clone)]
pub struct PyExperience {
    inner: Experience,
}

#[pymethods]
impl PyExperience {
    /// Create a new Experience.
    ///
    /// Args:
    ///     experience_type: One of "success", "failure", "pattern", "insight"
    ///     context: Context string
    ///     outcome: Outcome string
    ///     confidence: Float 0.0–1.0
    ///     tags: Optional list of tag strings
    ///     metadata: Optional dict of metadata
    #[new]
    #[pyo3(signature = (experience_type, context, outcome, confidence, tags=None, metadata=None))]
    fn new(
        experience_type: &str,
        context: String,
        outcome: String,
        confidence: f64,
        tags: Option<Vec<String>>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let et = parse_experience_type(experience_type)?;
        let mut exp = Experience::new(et, context, outcome, confidence).map_err(mem_err)?;
        if let Some(t) = tags {
            exp.tags = t;
        }
        if let Some(m) = metadata {
            exp.metadata = pydict_to_hashmap(Some(m))?;
        }
        Ok(Self { inner: exp })
    }

    #[getter]
    fn id(&self) -> &str {
        &self.inner.experience_id
    }

    #[getter]
    fn experience_type(&self) -> &str {
        self.inner.experience_type.as_str()
    }

    #[getter]
    fn context(&self) -> &str {
        &self.inner.context
    }

    #[getter]
    fn outcome(&self) -> &str {
        &self.inner.outcome
    }

    #[getter]
    fn confidence(&self) -> f64 {
        self.inner.confidence
    }

    #[getter]
    fn tags(&self) -> Vec<String> {
        self.inner.tags.clone()
    }

    #[getter]
    fn timestamp(&self) -> String {
        self.inner.timestamp.to_rfc3339()
    }

    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        hashmap_to_pydict(py, &self.inner.metadata)
    }

    /// Serialize to a Python dict.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let d = PyDict::new_bound(py);
        d.set_item("id", &self.inner.experience_id)?;
        d.set_item("experience_type", self.inner.experience_type.as_str())?;
        d.set_item("context", &self.inner.context)?;
        d.set_item("outcome", &self.inner.outcome)?;
        d.set_item("confidence", self.inner.confidence)?;
        d.set_item("tags", self.inner.tags.to_object(py))?;
        d.set_item("timestamp", self.inner.timestamp.to_rfc3339())?;
        d.set_item("metadata", hashmap_to_pydict(py, &self.inner.metadata)?)?;
        Ok(d.to_object(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "Experience(id={}, type={}, confidence={})",
            self.inner.experience_id,
            self.inner.experience_type.as_str(),
            self.inner.confidence
        )
    }
}

impl PyExperience {
    fn from_rust(exp: Experience) -> Self {
        Self { inner: exp }
    }
}

// ---------------------------------------------------------------------------
// PyMemoryConnector
// ---------------------------------------------------------------------------

/// SQLite-backed memory connector for storing and retrieving experiences.
#[pyclass(name = "MemoryConnector")]
pub struct PyMemoryConnector {
    inner: MemoryConnector,
}

#[pymethods]
impl PyMemoryConnector {
    /// Create a new MemoryConnector.
    ///
    /// Args:
    ///     agent_name: Name of the agent
    ///     db_path: Optional path to the SQLite database directory
    #[new]
    #[pyo3(signature = (agent_name, db_path=None))]
    fn new(agent_name: &str, db_path: Option<&str>) -> PyResult<Self> {
        let path = db_path.map(Path::new);
        let conn = MemoryConnector::new(agent_name, path, 100, false).map_err(mem_err)?;
        Ok(Self { inner: conn })
    }

    /// Store an experience and return its id.
    fn store_experience(&mut self, experience: &PyExperience) -> PyResult<String> {
        self.inner
            .store_experience(&experience.inner)
            .map_err(mem_err)
    }

    /// Retrieve experiences, optionally filtered.
    #[pyo3(signature = (limit=None, experience_type=None, min_confidence=0.0))]
    fn retrieve_experiences(
        &self,
        limit: Option<usize>,
        experience_type: Option<&str>,
        min_confidence: f64,
    ) -> PyResult<Vec<PyExperience>> {
        let et = experience_type.map(parse_experience_type).transpose()?;
        self.inner
            .retrieve_experiences(limit, et, min_confidence)
            .map(|v| v.into_iter().map(PyExperience::from_rust).collect())
            .map_err(mem_err)
    }

    /// Full-text search for experiences.
    #[pyo3(signature = (query, experience_type=None, min_confidence=0.0, limit=10))]
    fn search(
        &self,
        query: &str,
        experience_type: Option<&str>,
        min_confidence: f64,
        limit: usize,
    ) -> PyResult<Vec<PyExperience>> {
        let et = experience_type.map(parse_experience_type).transpose()?;
        self.inner
            .search(query, et, min_confidence, limit)
            .map(|v| v.into_iter().map(PyExperience::from_rust).collect())
            .map_err(mem_err)
    }

    /// Get storage statistics as a dict.
    fn get_statistics(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats = self.inner.get_statistics().map_err(mem_err)?;
        storage_stats_to_dict(py, &stats)
    }

    /// Close the connector.
    fn close(&mut self) {
        self.inner.close();
    }
}

// ---------------------------------------------------------------------------
// PyExperienceStore
// ---------------------------------------------------------------------------

/// Higher-level experience store with auto-cleanup policies.
#[pyclass(name = "ExperienceStore")]
pub struct PyExperienceStore {
    inner: ExperienceStore,
}

#[pymethods]
impl PyExperienceStore {
    /// Create a new ExperienceStore.
    ///
    /// Args:
    ///     agent_name: Name of the agent
    ///     db_path: Optional storage directory path
    #[new]
    #[pyo3(signature = (agent_name, db_path=None))]
    fn new(agent_name: &str, db_path: Option<&str>) -> PyResult<Self> {
        let path = db_path.map(Path::new);
        let store =
            ExperienceStore::new(agent_name, false, None, None, 100, path).map_err(mem_err)?;
        Ok(Self { inner: store })
    }

    /// Add an experience and return its id.
    fn add(&mut self, experience: &PyExperience) -> PyResult<String> {
        self.inner.add(&experience.inner).map_err(mem_err)
    }

    /// Search experiences by query.
    #[pyo3(signature = (query, experience_type=None, min_confidence=0.0, limit=10))]
    fn search(
        &self,
        query: &str,
        experience_type: Option<&str>,
        min_confidence: f64,
        limit: usize,
    ) -> PyResult<Vec<PyExperience>> {
        let et = experience_type.map(parse_experience_type).transpose()?;
        self.inner
            .search(query, et, min_confidence, limit)
            .map(|v| v.into_iter().map(PyExperience::from_rust).collect())
            .map_err(mem_err)
    }

    /// Get storage statistics as a dict.
    fn get_statistics(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats = self.inner.get_statistics().map_err(mem_err)?;
        storage_stats_to_dict(py, &stats)
    }
}

// ---------------------------------------------------------------------------
// PyCognitiveMemory
// ---------------------------------------------------------------------------

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
            .record_sensory(modality, raw_data, ttl_seconds)
            .map_err(mem_err)
    }

    /// Get recent sensory items as list of dicts.
    #[pyo3(signature = (limit=10))]
    fn get_recent_sensory(&self, py: Python<'_>, limit: usize) -> Vec<PyObject> {
        self.inner
            .get_recent_sensory(limit)
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
            .push_working(slot_type, content, task_id, relevance)
            .map_err(mem_err)
    }

    /// Recall working memory items for a task.
    fn recall_working(&self, py: Python<'_>, task_id: &str) -> Vec<PyObject> {
        self.inner
            .recall_working(task_id)
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
    fn recall_episodes(&self, py: Python<'_>, limit: usize) -> Vec<PyObject> {
        self.inner
            .recall_episodes(limit)
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
    ) -> Vec<PyObject> {
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
    fn recall_procedures(&mut self, py: Python<'_>, query: &str, limit: usize) -> Vec<PyObject> {
        self.inner
            .recall_procedures_mut(query, limit)
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
    fn check_triggers(&mut self, py: Python<'_>, content: &str) -> Vec<PyObject> {
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

// ---------------------------------------------------------------------------
// PyHierarchicalMemory
// ---------------------------------------------------------------------------

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
        let cat = category.map(|c| parse_category(c));
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
        let cat = category.map(|c| parse_category(c));
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
            d.set_item(k, json_to_py(py, v))?;
        }
        Ok(d.to_object(py))
    }

    fn close(&mut self) {
        self.inner.close();
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Extract an entity name from text given a concept hint.
#[pyfunction]
#[pyo3(name = "extract_entities")]
fn py_extract_entities(text: &str, concept: &str) -> String {
    extract_entity_name(text, concept)
}

/// Compute Jaccard word-level similarity between two strings (0.0–1.0).
#[pyfunction]
#[pyo3(name = "jaccard_similarity")]
fn py_jaccard_similarity(a: &str, b: &str) -> f64 {
    compute_word_similarity(a, b)
}

/// Detect whether two facts contradict each other.
/// Returns a dict with `contradiction` (bool) and `conflicting_values` (str),
/// or None if no contradiction is detected.
#[pyfunction]
#[pyo3(name = "detect_contradiction")]
fn py_detect_contradiction(
    py: Python<'_>,
    fact1: &str,
    fact2: &str,
    concept1: &str,
    concept2: &str,
) -> PyResult<Option<PyObject>> {
    match detect_contradiction(fact1, fact2, concept1, concept2) {
        Some(cr) => {
            let d = PyDict::new_bound(py);
            d.set_item("contradiction", cr.contradiction)?;
            d.set_item("conflicting_values", &cr.conflicting_values)?;
            Ok(Some(d.to_object(py)))
        }
        None => Ok(None),
    }
}

/// Scrub credentials from text. Returns (scrubbed_text, had_credentials).
#[pyfunction]
#[pyo3(name = "scrub_credentials")]
fn py_scrub_credentials(text: &str) -> (String, bool) {
    let scrubber = CredentialScrubber::new();
    scrubber.scrub_text(text)
}

// ---------------------------------------------------------------------------
// Type-to-dict helpers
// ---------------------------------------------------------------------------

fn sensory_to_dict(py: Python<'_>, s: &SensoryItem) -> PyObject {
    let d = PyDict::new_bound(py);
    d.set_item("node_id", &s.node_id).unwrap();
    d.set_item("modality", &s.modality).unwrap();
    d.set_item("raw_data", &s.raw_data).unwrap();
    d.set_item("observation_order", s.observation_order)
        .unwrap();
    d.set_item("expires_at", s.expires_at).unwrap();
    d.to_object(py)
}

fn working_to_dict(py: Python<'_>, w: &WorkingMemorySlot) -> PyObject {
    let d = PyDict::new_bound(py);
    d.set_item("node_id", &w.node_id).unwrap();
    d.set_item("slot_type", &w.slot_type).unwrap();
    d.set_item("content", &w.content).unwrap();
    d.set_item("relevance", w.relevance).unwrap();
    d.set_item("task_id", &w.task_id).unwrap();
    d.to_object(py)
}

fn episode_to_dict(py: Python<'_>, e: &EpisodicMemory) -> PyObject {
    let d = PyDict::new_bound(py);
    d.set_item("node_id", &e.node_id).unwrap();
    d.set_item("content", &e.content).unwrap();
    d.set_item("source_label", &e.source_label).unwrap();
    d.set_item("temporal_index", e.temporal_index).unwrap();
    d.set_item("compressed", e.compressed).unwrap();
    d.to_object(py)
}

fn fact_to_dict(py: Python<'_>, f: &SemanticFact) -> PyObject {
    let d = PyDict::new_bound(py);
    d.set_item("node_id", &f.node_id).unwrap();
    d.set_item("concept", &f.concept).unwrap();
    d.set_item("content", &f.content).unwrap();
    d.set_item("confidence", f.confidence).unwrap();
    d.set_item("source_id", &f.source_id).unwrap();
    d.set_item("tags", f.tags.to_object(py)).unwrap();
    d.to_object(py)
}

fn procedure_to_dict(py: Python<'_>, p: &ProceduralMemory) -> PyObject {
    let d = PyDict::new_bound(py);
    d.set_item("node_id", &p.node_id).unwrap();
    d.set_item("name", &p.name).unwrap();
    d.set_item("steps", p.steps.to_object(py)).unwrap();
    d.set_item("prerequisites", p.prerequisites.to_object(py))
        .unwrap();
    d.set_item("usage_count", p.usage_count).unwrap();
    d.to_object(py)
}

fn prospective_to_dict(py: Python<'_>, p: &ProspectiveMemory) -> PyObject {
    let d = PyDict::new_bound(py);
    d.set_item("node_id", &p.node_id).unwrap();
    d.set_item("description", &p.description).unwrap();
    d.set_item("trigger_condition", &p.trigger_condition)
        .unwrap();
    d.set_item("action_on_trigger", &p.action_on_trigger)
        .unwrap();
    d.set_item("status", &p.status).unwrap();
    d.set_item("priority", p.priority).unwrap();
    d.to_object(py)
}

fn knowledge_node_to_dict(py: Python<'_>, n: &KnowledgeNode) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    d.set_item("node_id", &n.node_id)?;
    d.set_item("category", format!("{:?}", n.category))?;
    d.set_item("content", &n.content)?;
    d.set_item("concept", &n.concept)?;
    d.set_item("confidence", n.confidence)?;
    d.set_item("source_id", &n.source_id)?;
    d.set_item("created_at", &n.created_at)?;
    d.set_item("tags", n.tags.to_object(py))?;
    d.set_item("metadata", hashmap_to_pydict(py, &n.metadata)?)?;
    Ok(d.to_object(py))
}

fn knowledge_edge_to_dict(py: Python<'_>, e: &KnowledgeEdge) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    d.set_item("source_id", &e.source_id)?;
    d.set_item("target_id", &e.target_id)?;
    d.set_item("relationship", &e.relationship)?;
    d.set_item("weight", e.weight)?;
    d.set_item("metadata", hashmap_to_pydict(py, &e.metadata)?)?;
    Ok(d.to_object(py))
}

// ---------------------------------------------------------------------------
// Module entry point
// ---------------------------------------------------------------------------

/// Native Rust implementation of amplihack-memory, exposed as a Python module.
#[pymodule]
#[pyo3(name = "amplihack_memory_rs")]
pub fn amplihack_memory_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<PyExperience>()?;
    m.add_class::<PyMemoryConnector>()?;
    m.add_class::<PyExperienceStore>()?;
    m.add_class::<PyCognitiveMemory>()?;
    m.add_class::<PyHierarchicalMemory>()?;

    // Functions
    m.add_function(wrap_pyfunction!(py_extract_entities, m)?)?;
    m.add_function(wrap_pyfunction!(py_jaccard_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(py_detect_contradiction, m)?)?;
    m.add_function(wrap_pyfunction!(py_scrub_credentials, m)?)?;

    Ok(())
}
