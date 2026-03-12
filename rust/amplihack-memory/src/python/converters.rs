//! Type-to-dict converter functions for Python bindings.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::hierarchical_memory::{KnowledgeEdge, KnowledgeNode};
use crate::memory_types::{
    EpisodicMemory, ProceduralMemory, ProspectiveMemory, SemanticFact, SensoryItem,
    WorkingMemorySlot,
};

use super::helpers::hashmap_to_pydict;

pub(crate) fn sensory_to_dict(py: Python<'_>, s: &SensoryItem) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    d.set_item("node_id", &s.node_id)?;
    d.set_item("modality", &s.modality)?;
    d.set_item("raw_data", &s.raw_data)?;
    d.set_item("observation_order", s.observation_order)?;
    d.set_item("expires_at", s.expires_at)?;
    d.set_item("created_at", s.created_at.to_rfc3339())?;
    Ok(d.to_object(py))
}

pub(crate) fn working_to_dict(py: Python<'_>, w: &WorkingMemorySlot) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    d.set_item("node_id", &w.node_id)?;
    d.set_item("slot_type", &w.slot_type)?;
    d.set_item("content", &w.content)?;
    d.set_item("relevance", w.relevance)?;
    d.set_item("task_id", &w.task_id)?;
    d.set_item("created_at", w.created_at.to_rfc3339())?;
    Ok(d.to_object(py))
}

pub(crate) fn episode_to_dict(py: Python<'_>, e: &EpisodicMemory) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    d.set_item("node_id", &e.node_id)?;
    d.set_item("content", &e.content)?;
    d.set_item("source_label", &e.source_label)?;
    d.set_item("temporal_index", e.temporal_index)?;
    d.set_item("compressed", e.compressed)?;
    d.set_item("created_at", e.created_at.to_rfc3339())?;
    d.set_item("metadata", hashmap_to_pydict(py, &e.metadata)?)?;
    Ok(d.to_object(py))
}

pub(crate) fn fact_to_dict(py: Python<'_>, f: &SemanticFact) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    d.set_item("node_id", &f.node_id)?;
    d.set_item("concept", &f.concept)?;
    d.set_item("content", &f.content)?;
    d.set_item("confidence", f.confidence)?;
    d.set_item("source_id", &f.source_id)?;
    d.set_item("tags", f.tags.to_object(py))?;
    d.set_item("created_at", f.created_at.to_rfc3339())?;
    d.set_item("metadata", hashmap_to_pydict(py, &f.metadata)?)?;
    Ok(d.to_object(py))
}

pub(crate) fn procedure_to_dict(py: Python<'_>, p: &ProceduralMemory) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    d.set_item("node_id", &p.node_id)?;
    d.set_item("name", &p.name)?;
    d.set_item("steps", p.steps.to_object(py))?;
    d.set_item("prerequisites", p.prerequisites.to_object(py))?;
    d.set_item("usage_count", p.usage_count)?;
    d.set_item("created_at", p.created_at.to_rfc3339())?;
    Ok(d.to_object(py))
}

pub(crate) fn prospective_to_dict(py: Python<'_>, p: &ProspectiveMemory) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    d.set_item("node_id", &p.node_id)?;
    d.set_item("description", &p.description)?;
    d.set_item("trigger_condition", &p.trigger_condition)?;
    d.set_item("action_on_trigger", &p.action_on_trigger)?;
    d.set_item("status", &p.status)?;
    d.set_item("priority", p.priority)?;
    d.set_item("created_at", p.created_at.to_rfc3339())?;
    Ok(d.to_object(py))
}

pub(crate) fn knowledge_node_to_dict(py: Python<'_>, n: &KnowledgeNode) -> PyResult<PyObject> {
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

pub(crate) fn knowledge_edge_to_dict(py: Python<'_>, e: &KnowledgeEdge) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    d.set_item("source_id", &e.source_id)?;
    d.set_item("target_id", &e.target_id)?;
    d.set_item("relationship", &e.relationship)?;
    d.set_item("weight", e.weight)?;
    d.set_item("metadata", hashmap_to_pydict(py, &e.metadata)?)?;
    Ok(d.to_object(py))
}
