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

/// Read an optional string prop: absent or empty maps to `None`.
pub(crate) fn prop_opt_str(props: &HashMap<String, String>, key: &str) -> Option<String> {
    props.get(key).filter(|v| !v.is_empty()).cloned()
}

/// Read an optional unix-timestamp prop as `DateTime<Utc>`: absent, empty, or
/// unparseable maps to `None`.
pub(crate) fn prop_opt_datetime(
    props: &HashMap<String, String>,
    key: &str,
) -> Option<DateTime<Utc>> {
    props
        .get(key)
        .filter(|v| !v.is_empty())
        .and_then(|v| v.parse::<i64>().ok())
        .map(ts_to_datetime)
}

/// Read a boolean prop encoded as the string `"true"` (anything else is false).
pub(crate) fn prop_bool(props: &HashMap<String, String>, key: &str) -> bool {
    props.get(key).map(|v| v == "true").unwrap_or(false)
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
        distilled: prop_str(props, "distilled") == "true",
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

    let concept = prop_str(props, "concept");
    let content = prop_str(props, "content");
    let confidence = prop_f64(props, "confidence");

    // `importance` falls back to `confidence` when the prop is absent/empty so
    // facts stored before the field existed keep a sensible value (rather than
    // `prop_f64`'s `0.0`).
    let importance = props
        .get("importance")
        .filter(|v| !v.is_empty())
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(confidence);

    // `content_hash` is recomputed from concept + content when absent/empty so
    // legacy facts participate in exact-content dedup without a migration.
    let content_hash = match prop_opt_str(props, "content_hash") {
        Some(h) => h,
        None => super::dedup::compute_content_hash(&concept, &content),
    };

    SemanticFact {
        node_id: prop_str(props, "node_id"),
        concept,
        content,
        confidence,
        source_id: prop_str(props, "source_id"),
        tags,
        metadata,
        created_at: prop_datetime(props, "created_at"),
        importance,
        usage_count: prop_i64(props, "usage_count"),
        last_accessed_at: prop_opt_datetime(props, "last_accessed_at"),
        expires_at: prop_opt_datetime(props, "expires_at"),
        archived: prop_bool(props, "archived"),
        superseded_by: prop_opt_str(props, "superseded_by"),
        content_hash,
        dedup_key: prop_opt_str(props, "dedup_key"),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn full_props() -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("node_id".into(), "n1".into());
        m.insert("modality".into(), "text".into());
        m.insert("raw_data".into(), "hello".into());
        m.insert("observation_order".into(), "5".into());
        m.insert("expires_at".into(), "99.9".into());
        m.insert("created_at".into(), "1700000000".into());
        m.insert("slot_type".into(), "goal".into());
        m.insert("content".into(), "test content".into());
        m.insert("relevance".into(), "0.75".into());
        m.insert("task_id".into(), "t1".into());
        m.insert("source_label".into(), "src".into());
        m.insert("temporal_index".into(), "42".into());
        m.insert("compressed".into(), "true".into());
        m.insert("metadata".into(), r#"{"k":"v"}"#.into());
        m.insert("concept".into(), "rust".into());
        m.insert("confidence".into(), "0.95".into());
        m.insert("source_id".into(), "s1".into());
        m.insert("tags".into(), r#"["t1","t2"]"#.into());
        m.insert("name".into(), "proc1".into());
        m.insert("steps".into(), r#"["a","b"]"#.into());
        m.insert("prerequisites".into(), r#"["p1"]"#.into());
        m.insert("usage_count".into(), "10".into());
        m.insert("desc_text".into(), "description".into());
        m.insert("trigger_condition".into(), "when x".into());
        m.insert("action_on_trigger".into(), "do y".into());
        m.insert("status".into(), "pending".into());
        m.insert("priority".into(), "3".into());
        m
    }

    #[test]
    fn test_node_to_sensory_all_fields() {
        let p = full_props();
        let s = node_to_sensory(&p);
        assert_eq!(s.node_id, "n1");
        assert_eq!(s.modality, "text");
        assert_eq!(s.raw_data, "hello");
        assert_eq!(s.observation_order, 5);
        assert!((s.expires_at - 99.9).abs() < 0.01);
    }

    #[test]
    fn test_node_to_sensory_empty_map() {
        let s = node_to_sensory(&HashMap::new());
        assert!(s.node_id.is_empty());
        assert_eq!(s.observation_order, 0);
    }

    #[test]
    fn test_node_to_working_all_fields() {
        let p = full_props();
        let w = node_to_working(&p);
        assert_eq!(w.node_id, "n1");
        assert_eq!(w.slot_type, "goal");
        assert_eq!(w.content, "test content");
        assert!((w.relevance - 0.75).abs() < 0.01);
        assert_eq!(w.task_id, "t1");
    }

    #[test]
    fn test_node_to_working_empty_map() {
        let w = node_to_working(&HashMap::new());
        assert!(w.node_id.is_empty());
        assert!((w.relevance - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_node_to_episodic_all_fields() {
        let p = full_props();
        let e = node_to_episodic(&p);
        assert_eq!(e.node_id, "n1");
        assert_eq!(e.content, "test content");
        assert_eq!(e.source_label, "src");
        assert_eq!(e.temporal_index, 42);
        assert!(e.compressed);
        assert_eq!(e.metadata.get("k").unwrap().as_str().unwrap(), "v");
    }

    #[test]
    fn test_node_to_episodic_empty_map() {
        let e = node_to_episodic(&HashMap::new());
        assert!(e.node_id.is_empty());
        assert!(!e.compressed);
        assert!(e.metadata.is_empty());
    }

    #[test]
    fn test_node_to_episodic_invalid_metadata_json() {
        let mut p = HashMap::new();
        p.insert("metadata".into(), "not json".into());
        let e = node_to_episodic(&p);
        assert!(e.metadata.is_empty());
    }

    #[test]
    fn test_node_to_fact_all_fields() {
        let p = full_props();
        let f = node_to_fact(&p);
        assert_eq!(f.node_id, "n1");
        assert_eq!(f.concept, "rust");
        assert!((f.confidence - 0.95).abs() < 0.01);
        assert_eq!(f.tags, vec!["t1", "t2"]);
        assert_eq!(f.metadata.get("k").unwrap().as_str().unwrap(), "v");
    }

    #[test]
    fn test_node_to_fact_empty_map() {
        let f = node_to_fact(&HashMap::new());
        assert!(f.tags.is_empty());
        assert!(f.metadata.is_empty());
    }

    #[test]
    fn test_node_to_fact_invalid_tags_json() {
        let mut p = HashMap::new();
        p.insert("tags".into(), "not json".into());
        let f = node_to_fact(&p);
        assert!(f.tags.is_empty());
    }

    #[test]
    fn test_node_to_fact_new_lifecycle_fields() {
        let mut p = HashMap::new();
        p.insert("node_id".into(), "n1".into());
        p.insert("concept".into(), "rust".into());
        p.insert("content".into(), "safe".into());
        p.insert("confidence".into(), "0.9".into());
        p.insert("importance".into(), "0.42".into());
        p.insert("usage_count".into(), "7".into());
        p.insert("last_accessed_at".into(), "1700000500".into());
        p.insert("expires_at".into(), "1800000000".into());
        p.insert("archived".into(), "true".into());
        p.insert("superseded_by".into(), "sem_newer".into());
        p.insert("content_hash".into(), "deadbeef".into());
        p.insert("dedup_key".into(), "caller-key".into());

        let f = node_to_fact(&p);
        assert!((f.importance - 0.42).abs() < 1e-9);
        assert_eq!(f.usage_count, 7);
        assert!(f.last_accessed_at.is_some());
        assert!(f.expires_at.is_some());
        assert!(f.archived);
        assert_eq!(f.superseded_by.as_deref(), Some("sem_newer"));
        assert_eq!(f.content_hash, "deadbeef");
        assert_eq!(f.dedup_key.as_deref(), Some("caller-key"));
    }

    #[test]
    fn test_node_to_fact_legacy_defaults() {
        // A node persisted before the lifecycle fields existed: only the
        // original props are present. The converter must default-populate the
        // new fields without panicking and never break on missing data.
        let mut p = HashMap::new();
        p.insert("node_id".into(), "n1".into());
        p.insert("concept".into(), "rust".into());
        p.insert("content".into(), "memory safe".into());
        p.insert("confidence".into(), "0.95".into());

        let f = node_to_fact(&p);
        assert!(
            (f.importance - 0.95).abs() < 1e-9,
            "importance falls back to confidence when absent"
        );
        assert_eq!(f.usage_count, 0);
        assert!(f.last_accessed_at.is_none());
        assert!(f.expires_at.is_none());
        assert!(!f.archived);
        assert!(f.superseded_by.is_none());
        assert!(f.dedup_key.is_none());
        assert!(
            !f.content_hash.is_empty(),
            "content_hash is recomputed when the stored prop is absent"
        );
    }

    #[test]
    fn test_node_to_fact_empty_options_are_none() {
        // Empty-string props (lbug's DEFAULT '' for ALTER-added columns) map to
        // None / defaults rather than Some("").
        let mut p = HashMap::new();
        p.insert("concept".into(), "c".into());
        p.insert("content".into(), "x".into());
        p.insert("confidence".into(), "0.5".into());
        p.insert("superseded_by".into(), "".into());
        p.insert("dedup_key".into(), "".into());
        p.insert("last_accessed_at".into(), "".into());
        p.insert("expires_at".into(), "".into());
        p.insert("content_hash".into(), "".into());
        p.insert("archived".into(), "".into());

        let f = node_to_fact(&p);
        assert!(f.superseded_by.is_none());
        assert!(f.dedup_key.is_none());
        assert!(f.last_accessed_at.is_none());
        assert!(f.expires_at.is_none());
        assert!(!f.archived);
        assert!(
            !f.content_hash.is_empty(),
            "empty content_hash prop triggers a recompute"
        );
    }

    #[test]
    fn test_node_to_procedural_all_fields() {
        let p = full_props();
        let pr = node_to_procedural(&p);
        assert_eq!(pr.node_id, "n1");
        assert_eq!(pr.name, "proc1");
        assert_eq!(pr.steps, vec!["a", "b"]);
        assert_eq!(pr.prerequisites, vec!["p1"]);
        assert_eq!(pr.usage_count, 10);
    }

    #[test]
    fn test_node_to_procedural_empty_map() {
        let pr = node_to_procedural(&HashMap::new());
        assert!(pr.steps.is_empty());
        assert!(pr.prerequisites.is_empty());
    }

    #[test]
    fn test_node_to_prospective_all_fields() {
        let p = full_props();
        let pr = node_to_prospective(&p);
        assert_eq!(pr.node_id, "n1");
        assert_eq!(pr.description, "description");
        assert_eq!(pr.trigger_condition, "when x");
        assert_eq!(pr.action_on_trigger, "do y");
        assert_eq!(pr.status, "pending");
        assert_eq!(pr.priority, 3);
    }

    #[test]
    fn test_node_to_prospective_empty_map() {
        let pr = node_to_prospective(&HashMap::new());
        assert!(pr.node_id.is_empty());
        assert_eq!(pr.priority, 0);
    }

    #[test]
    fn test_prop_i64_non_numeric() {
        let mut m = HashMap::new();
        m.insert("key".into(), "not_a_number".into());
        assert_eq!(prop_i64(&m, "key"), 0);
    }

    #[test]
    fn test_prop_f64_non_numeric() {
        let mut m = HashMap::new();
        m.insert("key".into(), "abc".into());
        assert!((prop_f64(&m, "key") - 0.0).abs() < 0.01);
    }
}
