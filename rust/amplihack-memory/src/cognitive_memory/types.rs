//! Helper structs, constants, and utility functions for cognitive memory.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{DateTime, TimeZone, Utc};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of working-memory slots per task.
pub const WORKING_MEMORY_CAPACITY: usize = 20;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a short unique id with a human-readable prefix.
pub(crate) fn new_id(prefix: &str) -> String {
    let hex = uuid::Uuid::new_v4().to_string().replace('-', "");
    format!("{}_{}", prefix, &hex[..12])
}

/// Current Unix timestamp as i64.
pub(crate) fn ts_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// Convert a unix timestamp (i64) to `DateTime<Utc>`.
pub(crate) fn ts_to_datetime(ts: i64) -> DateTime<Utc> {
    Utc.timestamp_opt(ts, 0).single().unwrap_or_default()
}

/// Build an agent-id filter map for `query_nodes` / `search_nodes`.
pub(crate) fn agent_filter(agent_id: &str) -> HashMap<String, String> {
    let mut m = HashMap::new();
    m.insert("agent_id".to_string(), agent_id.to_string());
    m
}

// ---------------------------------------------------------------------------
// Node-type labels (matching the Python schema table names)
// ---------------------------------------------------------------------------

/// Node-type label for sensory memory nodes.
pub(crate) const NT_SENSORY: &str = "SensoryMemory";
/// Node-type label for working memory nodes.
pub(crate) const NT_WORKING: &str = "WorkingMemory";
/// Node-type label for episodic memory nodes.
pub(crate) const NT_EPISODIC: &str = "EpisodicMemory";
/// Node-type label for semantic memory nodes.
pub(crate) const NT_SEMANTIC: &str = "SemanticMemory";
/// Node-type label for procedural memory nodes.
pub(crate) const NT_PROCEDURAL: &str = "ProceduralMemory";
/// Node-type label for prospective memory nodes.
pub(crate) const NT_PROSPECTIVE: &str = "ProspectiveMemory";
/// Node-type label for consolidated episode nodes.
pub(crate) const NT_CONSOLIDATED: &str = "ConsolidatedEpisode";

// Edge-type labels
/// Edge-type label for sensory-to-working attention transitions.
pub(crate) const ET_ATTENDED_TO: &str = "ATTENDED_TO";
/// Edge-type label for episode consolidation relationships.
pub(crate) const ET_CONSOLIDATES: &str = "CONSOLIDATES";
/// Edge-type label for similarity links between nodes.
pub(crate) const ET_SIMILAR_TO: &str = "SIMILAR_TO";
/// Edge-type label indicating one node derives from another.
pub(crate) const ET_DERIVES_FROM: &str = "DERIVES_FROM";
