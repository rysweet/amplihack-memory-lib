//! Cognitive memory type definitions.
//!
//! Six memory categories modeled after human cognition:
//!   SENSORY    - Raw, short-lived observations (auto-expire)
//!   WORKING    - Active task context with bounded capacity
//!   EPISODIC   - Autobiographical events (consolidatable)
//!   SEMANTIC   - Distilled facts and knowledge
//!   PROCEDURAL - Reusable step-by-step procedures
//!   PROSPECTIVE - Future-oriented trigger-action pairs

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Six cognitive memory types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryCategory {
    Sensory,
    Working,
    Episodic,
    Semantic,
    Procedural,
    Prospective,
}

impl MemoryCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Sensory => "sensory",
            Self::Working => "working",
            Self::Episodic => "episodic",
            Self::Semantic => "semantic",
            Self::Procedural => "procedural",
            Self::Prospective => "prospective",
        }
    }
}

impl std::str::FromStr for MemoryCategory {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "sensory" => Ok(Self::Sensory),
            "working" => Ok(Self::Working),
            "episodic" => Ok(Self::Episodic),
            "semantic" => Ok(Self::Semantic),
            "procedural" => Ok(Self::Procedural),
            "prospective" => Ok(Self::Prospective),
            _ => Err(format!("unknown memory category: {s}")),
        }
    }
}

impl std::fmt::Display for MemoryCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Short-lived raw observation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensoryItem {
    pub node_id: String,
    pub modality: String,
    pub raw_data: String,
    pub observation_order: i64,
    pub expires_at: f64,
    pub created_at: DateTime<Utc>,
}

/// Active task-context slot (bounded capacity).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemorySlot {
    pub node_id: String,
    pub slot_type: String,
    pub content: String,
    pub relevance: f64,
    pub task_id: String,
    pub created_at: DateTime<Utc>,
}

/// Autobiographical event record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicMemory {
    pub node_id: String,
    pub content: String,
    pub source_label: String,
    pub temporal_index: i64,
    pub compressed: bool,
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Distilled knowledge fact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFact {
    pub node_id: String,
    pub concept: String,
    pub content: String,
    pub confidence: f64,
    #[serde(default)]
    pub source_id: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
}

/// Reusable step-by-step procedure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProceduralMemory {
    pub node_id: String,
    pub name: String,
    #[serde(default)]
    pub steps: Vec<String>,
    #[serde(default)]
    pub prerequisites: Vec<String>,
    #[serde(default)]
    pub usage_count: i64,
    pub created_at: DateTime<Utc>,
}

/// Future-oriented trigger-action pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProspectiveMemory {
    pub node_id: String,
    pub description: String,
    pub trigger_condition: String,
    pub action_on_trigger: String,
    #[serde(default = "default_status")]
    pub status: String,
    #[serde(default = "default_priority")]
    pub priority: i32,
    pub created_at: DateTime<Utc>,
}

fn default_status() -> String {
    "pending".into()
}

fn default_priority() -> i32 {
    1
}

/// Summary produced by consolidating multiple episodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidatedEpisode {
    pub node_id: String,
    pub summary: String,
    pub original_count: i64,
    pub created_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_category_roundtrip() {
        for cat in [
            MemoryCategory::Sensory,
            MemoryCategory::Working,
            MemoryCategory::Episodic,
            MemoryCategory::Semantic,
            MemoryCategory::Procedural,
            MemoryCategory::Prospective,
        ] {
            assert_eq!(cat.as_str().parse::<MemoryCategory>(), Ok(cat));
        }
    }

    #[test]
    fn test_sensory_item() {
        let item = SensoryItem {
            node_id: "s1".into(),
            modality: "text".into(),
            raw_data: "hello".into(),
            observation_order: 1,
            expires_at: 1000.0,
            created_at: Utc::now(),
        };
        assert_eq!(item.modality, "text");
    }
}
