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

/// Six cognitive memory categories modeled after human memory architecture.
///
/// Each category corresponds to a distinct memory system with its own
/// storage characteristics, retention policy, and access patterns.
///
/// # Examples
///
/// ```
/// use amplihack_memory::MemoryCategory;
///
/// let cat = MemoryCategory::Semantic;
/// assert_eq!(cat.as_str(), "semantic");
/// assert_eq!("semantic".parse::<MemoryCategory>(), Ok(cat));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryCategory {
    /// Raw, short-lived observations that auto-expire (e.g. tool output, user message).
    Sensory,
    /// Active task-context slots with bounded capacity for the current task.
    Working,
    /// Autobiographical event records that can be consolidated over time.
    Episodic,
    /// Distilled, long-lived knowledge facts with confidence scores.
    Semantic,
    /// Reusable step-by-step procedures (how-to knowledge).
    Procedural,
    /// Future-oriented trigger-action pairs that fire when conditions are met.
    Prospective,
}

impl MemoryCategory {
    /// Return the lowercase string representation of the category.
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

/// Short-lived raw observation captured from the environment.
///
/// Sensory items represent unprocessed input (tool output, user messages, etc.)
/// and automatically expire after a configured TTL. They are the entry point
/// for all information entering the cognitive memory system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensoryItem {
    /// Unique identifier for this sensory item.
    pub node_id: String,
    /// Input channel (e.g. `"text"`, `"tool_output"`, `"observation"`).
    pub modality: String,
    /// The raw, unprocessed observation data.
    pub raw_data: String,
    /// Monotonically increasing order within the current session.
    pub observation_order: i64,
    /// Unix timestamp (fractional seconds) after which this item expires.
    pub expires_at: f64,
    /// When this observation was recorded.
    pub created_at: DateTime<Utc>,
}

/// Active task-context slot with bounded capacity.
///
/// Working memory holds the information actively being used for the current
/// task. The total number of slots is capped at a configured maximum;
/// least-relevant slots are evicted when the limit is reached.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemorySlot {
    /// Unique identifier for this slot.
    pub node_id: String,
    /// Semantic type of the slot (e.g. `"goal"`, `"context"`, `"constraint"`).
    pub slot_type: String,
    /// The content held in this working memory slot.
    pub content: String,
    /// Relevance score in `[0.0, 1.0]` used for eviction ordering.
    pub relevance: f64,
    /// Identifier of the task this slot is associated with.
    pub task_id: String,
    /// When this slot was created.
    pub created_at: DateTime<Utc>,
}

/// Autobiographical event record representing a specific agent experience.
///
/// Episodes capture what happened, when, and from which source. Over time,
/// multiple related episodes can be consolidated into a
/// [`ConsolidatedEpisode`] to save storage while preserving key information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicMemory {
    /// Unique identifier for this episode.
    pub node_id: String,
    /// Textual description of what happened.
    pub content: String,
    /// Label identifying the source (e.g. agent name or session id).
    pub source_label: String,
    /// Monotonically increasing temporal ordering index.
    pub temporal_index: i64,
    /// Whether this episode has been compressed during consolidation.
    pub compressed: bool,
    /// When this episode was recorded.
    pub created_at: DateTime<Utc>,
    /// Arbitrary key-value metadata attached to the episode.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Distilled knowledge fact with an associated confidence score.
///
/// Semantic facts represent long-lived knowledge extracted or declared by the
/// agent. They are indexed by `concept` for retrieval and support similarity
/// linking via tags. Confidence decays or grows based on contradiction
/// detection and validation feedback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFact {
    /// Unique identifier for this fact.
    pub node_id: String,
    /// The concept or topic this fact relates to (used as a lookup key).
    pub concept: String,
    /// The factual content itself.
    pub content: String,
    /// Confidence score in `[0.0, 1.0]` indicating reliability.
    pub confidence: f64,
    /// Identifier of the source that produced this fact (e.g. episode id).
    #[serde(default)]
    pub source_id: String,
    /// Categorical tags for similarity linking and filtering.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Arbitrary key-value metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    /// When this fact was recorded.
    pub created_at: DateTime<Utc>,
}

/// Reusable step-by-step procedure (how-to knowledge).
///
/// Procedural memories encode learned multi-step processes that the agent can
/// recall and execute. Each procedure tracks prerequisites and a usage count
/// so frequently used procedures can be surfaced more readily.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProceduralMemory {
    /// Unique identifier for this procedure.
    pub node_id: String,
    /// Human-readable name of the procedure.
    pub name: String,
    /// Ordered list of steps to execute.
    #[serde(default)]
    pub steps: Vec<String>,
    /// Conditions that must be satisfied before execution.
    #[serde(default)]
    pub prerequisites: Vec<String>,
    /// Number of times this procedure has been invoked.
    #[serde(default)]
    pub usage_count: i64,
    /// When this procedure was first recorded.
    pub created_at: DateTime<Utc>,
}

/// Future-oriented trigger-action pair (intention memory).
///
/// Prospective memories represent deferred intentions: when a specified
/// trigger condition is detected, the associated action should be executed.
/// They remain in a `"pending"` state until resolved or explicitly dismissed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProspectiveMemory {
    /// Unique identifier for this prospective memory.
    pub node_id: String,
    /// Human-readable description of the intention.
    pub description: String,
    /// Condition that must be met before the action fires.
    pub trigger_condition: String,
    /// Action to perform when the trigger condition is satisfied.
    pub action_on_trigger: String,
    /// Current lifecycle status (`"pending"`, `"resolved"`, `"dismissed"`).
    #[serde(default = "default_status")]
    pub status: String,
    /// Priority level (higher = more urgent). Defaults to `1`.
    #[serde(default = "default_priority")]
    pub priority: i32,
    /// When this intention was recorded.
    pub created_at: DateTime<Utc>,
}

fn default_status() -> String {
    "pending".into()
}

fn default_priority() -> i32 {
    1
}

/// Summary produced by consolidating multiple episodic memories.
///
/// Consolidation merges several related [`EpisodicMemory`] records into a
/// single summary, reducing storage while preserving the essential narrative.
/// The original episodes are typically marked as compressed after
/// consolidation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidatedEpisode {
    /// Unique identifier for this consolidated record.
    pub node_id: String,
    /// Merged summary text covering all original episodes.
    pub summary: String,
    /// Number of original episodes that were consolidated.
    pub original_count: i64,
    /// When this consolidation was performed.
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
