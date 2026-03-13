//! Node-to-struct converter functions for cognitive memory.

use std::collections::HashMap;

use chrono::{DateTime, Utc};

use crate::memory_types::{
    EpisodicMemory, ProceduralMemory, ProspectiveMemory, SemanticFact, SensoryItem,
    WorkingMemorySlot,
};

use super::types::ts_to_datetime;

// ---------------------------------------------------------------------------
// Property extraction helpers
// ---------------------------------------------------------------------------

pub(crate) fn prop_str(props: &HashMap<String, String>, key: &str) -> String {
    props.get(key).cloned().unwrap_or_default()
}

pub(crate) fn prop_i64(props: &HashMap<String, String>, key: &str) -> i64 {
    props
        .get(key)
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(0)
}

pub(crate) fn prop_f64(props: &HashMap<String, String>, key: &str) -> f64 {
    props
        .get(key)
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0)
}

pub(crate) fn prop_datetime(props: &HashMap<String, String>, key: &str) -> DateTime<Utc> {
    ts_to_datetime(prop_i64(props, key))
}

// ---------------------------------------------------------------------------
// Node-to-struct converters
// ---------------------------------------------------------------------------

pub(crate) fn node_to_sensory(props: &HashMap<String, String>) -> SensoryItem {
    SensoryItem {
        node_id: prop_str(props, "node_id"),
        modality: prop_str(props, "modality"),
        raw_data: prop_str(props, "raw_data"),
        observation_order: prop_i64(props, "observation_order"),
        expires_at: prop_f64(props, "expires_at"),
        created_at: prop_datetime(props, "created_at"),
    }
}

pub(crate) fn node_to_working(props: &HashMap<String, String>) -> WorkingMemorySlot {
    WorkingMemorySlot {
        node_id: prop_str(props, "node_id"),
        slot_type: prop_str(props, "slot_type"),
        content: prop_str(props, "content"),
        relevance: prop_f64(props, "relevance"),
        task_id: prop_str(props, "task_id"),
        created_at: prop_datetime(props, "created_at"),
    }
}

pub(crate) fn node_to_episodic(props: &HashMap<String, String>) -> EpisodicMemory {
    let meta_str = prop_str(props, "metadata");
    let metadata: HashMap<String, serde_json::Value> =
        serde_json::from_str(&meta_str).unwrap_or_default();

    EpisodicMemory {
        node_id: prop_str(props, "node_id"),
        content: prop_str(props, "content"),
        source_label: prop_str(props, "source_label"),
        temporal_index: prop_i64(props, "temporal_index"),
        compressed: prop_str(props, "compressed") == "true",
        created_at: prop_datetime(props, "created_at"),
        metadata,
    }
}

pub(crate) fn node_to_fact(props: &HashMap<String, String>) -> SemanticFact {
    let tags_str = prop_str(props, "tags");
    let tags: Vec<String> = serde_json::from_str(&tags_str).unwrap_or_default();

    let meta_str = prop_str(props, "metadata");
    let metadata: HashMap<String, serde_json::Value> =
        serde_json::from_str(&meta_str).unwrap_or_default();

    SemanticFact {
        node_id: prop_str(props, "node_id"),
        concept: prop_str(props, "concept"),
        content: prop_str(props, "content"),
        confidence: prop_f64(props, "confidence"),
        source_id: prop_str(props, "source_id"),
        tags,
        metadata,
        created_at: prop_datetime(props, "created_at"),
    }
}

pub(crate) fn node_to_procedural(props: &HashMap<String, String>) -> ProceduralMemory {
    let steps_str = prop_str(props, "steps");
    let steps: Vec<String> = serde_json::from_str(&steps_str).unwrap_or_default();

    let prereqs_str = prop_str(props, "prerequisites");
    let prerequisites: Vec<String> = serde_json::from_str(&prereqs_str).unwrap_or_default();

    ProceduralMemory {
        node_id: prop_str(props, "node_id"),
        name: prop_str(props, "name"),
        steps,
        prerequisites,
        usage_count: prop_i64(props, "usage_count"),
        created_at: prop_datetime(props, "created_at"),
    }
}

pub(crate) fn node_to_prospective(props: &HashMap<String, String>) -> ProspectiveMemory {
    ProspectiveMemory {
        node_id: prop_str(props, "node_id"),
        description: prop_str(props, "desc_text"),
        trigger_condition: prop_str(props, "trigger_condition"),
        action_on_trigger: prop_str(props, "action_on_trigger"),
        status: prop_str(props, "status"),
        priority: prop_i64(props, "priority").clamp(i32::MIN as i64, i32::MAX as i64) as i32,
        created_at: prop_datetime(props, "created_at"),
    }
}
