//! Cognitive memory system with six memory types.
//!
//! Provides a single [`CognitiveMemory`] struct that manages all six cognitive
//! memory types (sensory, working, episodic, semantic, procedural, prospective)
//! backed by an [`InMemoryGraphStore`].
//!
//! Each agent gets full isolation via an `agent_id` property stored on every
//! graph node.

mod converters;
mod types;

pub use types::WORKING_MEMORY_CAPACITY;

use std::collections::HashMap;

use crate::graph::in_memory_store::InMemoryGraphStore;
use crate::graph::protocol::GraphStore;
use crate::memory_types::{
    EpisodicMemory, MemoryCategory, ProceduralMemory, ProspectiveMemory, SemanticFact, SensoryItem,
    WorkingMemorySlot,
};
use crate::{MemoryError, Result};

use converters::{
    node_to_episodic, node_to_fact, node_to_procedural, node_to_prospective, node_to_sensory,
    node_to_working,
};
use types::{
    agent_filter, new_id, ts_now, ET_ATTENDED_TO, ET_CONSOLIDATES, ET_DERIVES_FROM, ET_SIMILAR_TO,
    NT_CONSOLIDATED, NT_EPISODIC, NT_PROCEDURAL, NT_PROSPECTIVE, NT_SEMANTIC, NT_SENSORY,
    NT_WORKING,
};

// ---------------------------------------------------------------------------
// CognitiveMemory
// ---------------------------------------------------------------------------

/// Six-type cognitive memory backed by an in-memory graph store.
///
/// The struct owns an [`InMemoryGraphStore`] and provides methods corresponding
/// to every public method in the Python `CognitiveMemory` class.
pub struct CognitiveMemory {
    agent_name: String,
    graph: InMemoryGraphStore,
    sensory_order: i64,
    temporal_index: i64,
}

impl CognitiveMemory {
    /// Create a new `CognitiveMemory` for the given agent.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::InvalidInput` if `agent_name` is empty.
    pub fn new(agent_name: &str) -> Result<Self> {
        let trimmed = agent_name.trim();
        if trimmed.is_empty() {
            return Err(MemoryError::InvalidInput(
                "agent_name cannot be empty".into(),
            ));
        }
        let store_id = format!("cognitive-{}", trimmed);
        Ok(Self {
            agent_name: trimmed.to_string(),
            graph: InMemoryGraphStore::new(Some(&store_id)),
            sensory_order: 0,
            temporal_index: 0,
        })
    }

    /// The agent name this memory is scoped to.
    pub fn agent_name(&self) -> &str {
        &self.agent_name
    }

    // ======================================================================
    // SENSORY MEMORY
    // ======================================================================

    /// Record a short-lived sensory observation.
    ///
    /// Returns the `node_id` of the newly created sensory item.
    pub fn record_sensory(
        &mut self,
        modality: &str,
        raw_data: &str,
        ttl_seconds: i64,
    ) -> Result<String> {
        let node_id = new_id("sen");
        let now = ts_now();
        self.sensory_order += 1;

        let mut props = HashMap::new();
        props.insert("node_id".to_string(), node_id.clone());
        props.insert("agent_id".to_string(), self.agent_name.clone());
        props.insert("modality".to_string(), modality.to_string());
        props.insert("raw_data".to_string(), raw_data.to_string());
        props.insert(
            "observation_order".to_string(),
            self.sensory_order.to_string(),
        );
        props.insert("expires_at".to_string(), (now + ttl_seconds).to_string());
        props.insert("created_at".to_string(), now.to_string());

        self.graph
            .add_node(NT_SENSORY, props, Some(&node_id))
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        Ok(node_id)
    }

    /// Return the most recent sensory items that have not expired.
    pub fn get_recent_sensory(&self, limit: usize) -> Vec<SensoryItem> {
        let now = ts_now();
        let filter = agent_filter(&self.agent_name);
        let nodes = self
            .graph
            .query_nodes(NT_SENSORY, Some(&filter), usize::MAX);

        let mut items: Vec<SensoryItem> = nodes
            .into_iter()
            .filter(|n| {
                n.properties
                    .get("expires_at")
                    .and_then(|v| v.parse::<i64>().ok())
                    .is_some_and(|exp| exp > now)
            })
            .map(|n| node_to_sensory(&n.properties))
            .collect();

        // Sort by observation_order descending
        items.sort_by(|a, b| b.observation_order.cmp(&a.observation_order));
        items.truncate(limit);
        items
    }

    /// Promote a sensory item to an episodic memory and link them.
    ///
    /// Returns the `node_id` of the new episodic memory, or `None` if the
    /// sensory item was not found or has expired.
    pub fn attend_to_sensory(&mut self, sensory_id: &str, reason: &str) -> Option<String> {
        let now = ts_now();
        let node = self.graph.get_node(sensory_id)?;

        if node.node_type != NT_SENSORY {
            return None;
        }
        if node.properties.get("agent_id")? != &self.agent_name {
            return None;
        }
        let expires_at: i64 = node.properties.get("expires_at")?.parse().ok()?;
        if expires_at <= now {
            return None;
        }

        let modality = node.properties.get("modality").cloned().unwrap_or_default();
        let raw_data = node.properties.get("raw_data").cloned().unwrap_or_default();
        let content = format!("[{modality}] {raw_data} -- attended: {reason}");

        let ep_id = self
            .store_episode(&content, "sensory-attention", None, None)
            .ok()?;

        // Best-effort edge creation
        let _ = self.graph.add_edge(
            sensory_id,
            &ep_id,
            ET_ATTENDED_TO,
            Some({
                let mut m = HashMap::new();
                m.insert("attended_at".to_string(), now.to_string());
                m
            }),
        );

        Some(ep_id)
    }

    /// Delete sensory items past their expiry time. Returns count pruned.
    pub fn prune_expired_sensory(&mut self) -> usize {
        let now = ts_now();
        let filter = agent_filter(&self.agent_name);
        let nodes = self
            .graph
            .query_nodes(NT_SENSORY, Some(&filter), usize::MAX);

        let expired_ids: Vec<String> = nodes
            .into_iter()
            .filter(|n| {
                n.properties
                    .get("expires_at")
                    .and_then(|v| v.parse::<i64>().ok())
                    .is_some_and(|exp| exp <= now)
            })
            .map(|n| n.node_id)
            .collect();

        let count = expired_ids.len();
        for id in expired_ids {
            self.graph.delete_node(&id);
        }
        count
    }

    /// Alias matching the Python method name `expire_sensory`.
    pub fn expire_sensory(&mut self) -> usize {
        self.prune_expired_sensory()
    }

    // ======================================================================
    // WORKING MEMORY
    // ======================================================================

    /// Push a slot into working memory for a given task.
    ///
    /// If the task already has [`WORKING_MEMORY_CAPACITY`] slots, the
    /// least-relevant slot is evicted.
    pub fn push_working(
        &mut self,
        slot_type: &str,
        content: &str,
        task_id: &str,
        relevance: f64,
    ) -> Result<String> {
        // Evict if at capacity
        let existing = self.recall_working(task_id);
        if existing.len() >= WORKING_MEMORY_CAPACITY {
            if let Some(lowest) = existing.iter().min_by(|a, b| {
                a.relevance
                    .partial_cmp(&b.relevance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                self.graph.delete_node(&lowest.node_id);
            }
        }

        let node_id = new_id("wrk");
        let now = ts_now();

        let mut props = HashMap::new();
        props.insert("node_id".to_string(), node_id.clone());
        props.insert("agent_id".to_string(), self.agent_name.clone());
        props.insert("slot_type".to_string(), slot_type.to_string());
        props.insert("content".to_string(), content.to_string());
        props.insert("relevance".to_string(), relevance.to_string());
        props.insert("task_id".to_string(), task_id.to_string());
        props.insert("created_at".to_string(), now.to_string());

        self.graph
            .add_node(NT_WORKING, props, Some(&node_id))
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        Ok(node_id)
    }

    /// Retrieve working memory slots for a task, ordered by relevance descending.
    pub fn recall_working(&self, task_id: &str) -> Vec<WorkingMemorySlot> {
        let mut filter = agent_filter(&self.agent_name);
        filter.insert("task_id".to_string(), task_id.to_string());
        let nodes = self
            .graph
            .query_nodes(NT_WORKING, Some(&filter), usize::MAX);

        let mut slots: Vec<WorkingMemorySlot> = nodes
            .iter()
            .map(|n| node_to_working(&n.properties))
            .collect();

        slots.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        slots
    }

    /// Clear all working memory slots for a task. Returns count cleared.
    pub fn clear_working(&mut self, task_id: &str) -> usize {
        let mut filter = agent_filter(&self.agent_name);
        filter.insert("task_id".to_string(), task_id.to_string());
        let nodes = self
            .graph
            .query_nodes(NT_WORKING, Some(&filter), usize::MAX);

        let count = nodes.len();
        for n in nodes {
            self.graph.delete_node(&n.node_id);
        }
        count
    }

    // ======================================================================
    // EPISODIC MEMORY
    // ======================================================================

    /// Store an episodic memory.
    ///
    /// If `temporal_index` is `None`, an auto-incrementing index is used.
    pub fn store_episode(
        &mut self,
        content: &str,
        source_label: &str,
        temporal_index: Option<i64>,
        metadata: Option<&HashMap<String, serde_json::Value>>,
    ) -> Result<String> {
        let node_id = new_id("epi");
        let now = ts_now();

        let tidx = match temporal_index {
            Some(idx) => {
                if idx > self.temporal_index {
                    self.temporal_index = idx;
                }
                idx
            }
            None => {
                self.temporal_index += 1;
                self.temporal_index
            }
        };

        let meta_json = metadata
            .map(|m| serde_json::to_string(m).unwrap_or_else(|_| "{}".into()))
            .unwrap_or_else(|| "{}".into());

        let mut props = HashMap::new();
        props.insert("node_id".to_string(), node_id.clone());
        props.insert("agent_id".to_string(), self.agent_name.clone());
        props.insert("content".to_string(), content.to_string());
        props.insert("source_label".to_string(), source_label.to_string());
        props.insert("temporal_index".to_string(), tidx.to_string());
        props.insert("compressed".to_string(), "false".to_string());
        props.insert("metadata".to_string(), meta_json);
        props.insert("created_at".to_string(), now.to_string());

        self.graph
            .add_node(NT_EPISODIC, props, Some(&node_id))
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        Ok(node_id)
    }

    /// Retrieve episodic memories, sorted by temporal_index descending.
    ///
    /// If `include_compressed` is false (default), compressed episodes are excluded.
    pub fn get_episodes(&self, limit: usize, include_compressed: bool) -> Vec<EpisodicMemory> {
        let filter = agent_filter(&self.agent_name);
        let nodes = self
            .graph
            .query_nodes(NT_EPISODIC, Some(&filter), usize::MAX);

        let mut episodes: Vec<EpisodicMemory> = nodes
            .into_iter()
            .filter(|n| {
                if include_compressed {
                    true
                } else {
                    n.properties.get("compressed").map_or(true, |v| v != "true")
                }
            })
            .map(|n| node_to_episodic(&n.properties))
            .collect();

        // Sort by temporal_index descending
        episodes.sort_by(|a, b| b.temporal_index.cmp(&a.temporal_index));
        episodes.truncate(limit);
        episodes
    }

    /// Alias matching the Python method name `recall_episodes`.
    pub fn recall_episodes(&self, limit: usize) -> Vec<EpisodicMemory> {
        self.get_episodes(limit, false)
    }

    /// Consolidate the oldest un-compressed episodes into a summary.
    ///
    /// Returns the `node_id` of the `ConsolidatedEpisode`, or `None` if
    /// fewer than `batch_size` unconsolidated episodes exist.
    ///
    /// If `summarizer` is `None`, a simple `" | "` concatenation is used.
    pub fn consolidate_episodes<F>(
        &mut self,
        batch_size: usize,
        summarizer: Option<F>,
    ) -> Option<String>
    where
        F: FnOnce(&[String]) -> String,
    {
        let filter = agent_filter(&self.agent_name);
        let nodes = self
            .graph
            .query_nodes(NT_EPISODIC, Some(&filter), usize::MAX);

        // Collect un-compressed episodes sorted by temporal_index ascending
        let mut candidates: Vec<(String, String, i64)> = nodes
            .into_iter()
            .filter(|n| n.properties.get("compressed").map_or(true, |v| v != "true"))
            .map(|n| {
                let nid = n.node_id.clone();
                let content = n.properties.get("content").cloned().unwrap_or_default();
                let tidx: i64 = n
                    .properties
                    .get("temporal_index")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
                (nid, content, tidx)
            })
            .collect();

        candidates.sort_by_key(|(_, _, tidx)| *tidx);

        if candidates.len() < batch_size {
            return None;
        }

        let batch: Vec<(String, String, i64)> = candidates.into_iter().take(batch_size).collect();

        let contents: Vec<String> = batch.iter().map(|(_, c, _)| c.clone()).collect();
        let summary = match summarizer {
            Some(f) => f(&contents),
            None => contents.join(" | "),
        };

        // Create consolidated node
        let cons_id = new_id("con");
        let now = ts_now();

        let mut props = HashMap::new();
        props.insert("node_id".to_string(), cons_id.clone());
        props.insert("agent_id".to_string(), self.agent_name.clone());
        props.insert("summary".to_string(), summary);
        props.insert("original_count".to_string(), batch.len().to_string());
        props.insert("created_at".to_string(), now.to_string());

        if self
            .graph
            .add_node(NT_CONSOLIDATED, props, Some(&cons_id))
            .is_err()
        {
            return None;
        }

        // Mark originals as compressed and create edges
        for (ep_id, _, _) in &batch {
            let mut update = HashMap::new();
            update.insert("compressed".to_string(), "true".to_string());
            self.graph.update_node(ep_id, update);

            // Best-effort edge
            let _ = self.graph.add_edge(
                &cons_id,
                ep_id,
                ET_CONSOLIDATES,
                Some({
                    let mut m = HashMap::new();
                    m.insert("consolidated_at".to_string(), now.to_string());
                    m
                }),
            );
        }

        Some(cons_id)
    }

    // ======================================================================
    // SEMANTIC MEMORY
    // ======================================================================

    /// Store a semantic fact.
    ///
    /// Vector embeddings are not available in the Rust port; keyword search
    /// is used for retrieval.
    pub fn store_fact(
        &mut self,
        concept: &str,
        content: &str,
        confidence: f64,
        source_id: &str,
        tags: Option<&[String]>,
        metadata: Option<&HashMap<String, serde_json::Value>>,
    ) -> Result<String> {
        if confidence.is_nan() || !(0.0..=1.0).contains(&confidence) {
            return Err(MemoryError::InvalidInput(
                "confidence must be between 0.0 and 1.0".into(),
            ));
        }

        let node_id = new_id("sem");
        let now = ts_now();
        let tags_json = tags
            .map(|t| serde_json::to_string(t).unwrap_or_else(|_| "[]".into()))
            .unwrap_or_else(|| "[]".into());
        let meta_json = metadata
            .map(|m| serde_json::to_string(m).unwrap_or_else(|_| "{}".into()))
            .unwrap_or_else(|| "{}".into());

        let mut props = HashMap::new();
        props.insert("node_id".to_string(), node_id.clone());
        props.insert("agent_id".to_string(), self.agent_name.clone());
        props.insert("concept".to_string(), concept.to_string());
        props.insert("content".to_string(), content.to_string());
        props.insert("confidence".to_string(), confidence.to_string());
        props.insert("source_id".to_string(), source_id.to_string());
        props.insert("tags".to_string(), tags_json);
        props.insert("metadata".to_string(), meta_json);
        props.insert("created_at".to_string(), now.to_string());

        self.graph
            .add_node(NT_SEMANTIC, props, Some(&node_id))
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        Ok(node_id)
    }

    /// Search semantic facts using keyword matching.
    ///
    /// Words from `query` are matched against both `concept` and `content`
    /// fields (case-insensitive). Results are filtered by `min_confidence`
    /// and sorted by confidence descending.
    pub fn search_facts(
        &self,
        query: &str,
        limit: usize,
        min_confidence: f64,
    ) -> Vec<SemanticFact> {
        let keywords: Vec<String> = query
            .split_whitespace()
            .filter(|w| !w.is_empty())
            .map(|w| w.to_lowercase())
            .collect();

        if keywords.is_empty() {
            return self.get_all_facts(limit);
        }

        let filter = agent_filter(&self.agent_name);
        let nodes = self
            .graph
            .query_nodes(NT_SEMANTIC, Some(&filter), usize::MAX);

        let mut facts: Vec<SemanticFact> = nodes
            .into_iter()
            .filter(|n| {
                let conf: f64 = n
                    .properties
                    .get("confidence")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0.0);
                if conf < min_confidence {
                    return false;
                }
                let concept = n
                    .properties
                    .get("concept")
                    .map(|s| s.to_lowercase())
                    .unwrap_or_default();
                let content = n
                    .properties
                    .get("content")
                    .map(|s| s.to_lowercase())
                    .unwrap_or_default();

                keywords
                    .iter()
                    .any(|kw| concept.contains(kw) || content.contains(kw))
            })
            .map(|n| node_to_fact(&n.properties))
            .collect();

        facts.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        facts.truncate(limit);
        facts
    }

    /// Return all semantic facts for this agent, sorted by confidence descending.
    pub fn get_all_facts(&self, limit: usize) -> Vec<SemanticFact> {
        let filter = agent_filter(&self.agent_name);
        let nodes = self
            .graph
            .query_nodes(NT_SEMANTIC, Some(&filter), usize::MAX);

        let mut facts: Vec<SemanticFact> = nodes
            .into_iter()
            .map(|n| node_to_fact(&n.properties))
            .collect();

        facts.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        facts.truncate(limit);
        facts
    }

    /// Link two semantic facts with a SIMILAR_TO edge.
    pub fn link_similar_facts(
        &mut self,
        fact_id_a: &str,
        fact_id_b: &str,
        similarity_score: f64,
    ) -> Result<()> {
        let mut props = HashMap::new();
        props.insert("similarity_score".to_string(), similarity_score.to_string());
        self.graph
            .add_edge(fact_id_a, fact_id_b, ET_SIMILAR_TO, Some(props))
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Link a semantic fact to its source episode.
    pub fn link_fact_to_episode(&mut self, fact_id: &str, episode_id: &str) -> Result<()> {
        let now = ts_now();
        let mut props = HashMap::new();
        props.insert("derived_at".to_string(), now.to_string());
        self.graph
            .add_edge(fact_id, episode_id, ET_DERIVES_FROM, Some(props))
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        Ok(())
    }

    // ======================================================================
    // PROCEDURAL MEMORY
    // ======================================================================

    /// Store a reusable procedure.
    pub fn store_procedure(
        &mut self,
        name: &str,
        steps: &[String],
        prerequisites: Option<&[String]>,
    ) -> Result<String> {
        let node_id = new_id("proc");
        let now = ts_now();
        let steps_json = serde_json::to_string(steps).unwrap_or_else(|_| "[]".into());
        let prereqs_json = prerequisites
            .map(|p| serde_json::to_string(p).unwrap_or_else(|_| "[]".into()))
            .unwrap_or_else(|| "[]".into());

        let mut props = HashMap::new();
        props.insert("node_id".to_string(), node_id.clone());
        props.insert("agent_id".to_string(), self.agent_name.clone());
        props.insert("name".to_string(), name.to_string());
        props.insert("steps".to_string(), steps_json);
        props.insert("prerequisites".to_string(), prereqs_json);
        props.insert("usage_count".to_string(), "0".to_string());
        props.insert("created_at".to_string(), now.to_string());

        self.graph
            .add_node(NT_PROCEDURAL, props, Some(&node_id))
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        Ok(node_id)
    }

    /// Recall procedures matching a query, incrementing their usage count.
    ///
    /// Words from `query` are matched against `name` and `steps` fields.
    /// Results are sorted by `usage_count` descending.
    pub fn recall_procedures(&self, query: &str, limit: usize) -> Vec<ProceduralMemory> {
        let filter = agent_filter(&self.agent_name);
        let nodes = self
            .graph
            .query_nodes(NT_PROCEDURAL, Some(&filter), usize::MAX);

        let keywords: Vec<String> = query
            .split_whitespace()
            .filter(|w| !w.is_empty())
            .map(|w| w.to_lowercase())
            .collect();

        let mut procs: Vec<ProceduralMemory> = nodes
            .into_iter()
            .filter(|n| {
                if keywords.is_empty() {
                    return true;
                }
                let name = n
                    .properties
                    .get("name")
                    .map(|s| s.to_lowercase())
                    .unwrap_or_default();
                let steps = n
                    .properties
                    .get("steps")
                    .map(|s| s.to_lowercase())
                    .unwrap_or_default();
                keywords
                    .iter()
                    .any(|kw| name.contains(kw) || steps.contains(kw))
            })
            .map(|n| node_to_procedural(&n.properties))
            .collect();

        procs.sort_by(|a, b| b.usage_count.cmp(&a.usage_count));
        procs.truncate(limit);
        procs
    }

    /// Recall procedures and increment their usage counts (mutable version).
    pub fn recall_procedures_mut(&mut self, query: &str, limit: usize) -> Vec<ProceduralMemory> {
        let procs = self.recall_procedures(query, limit);

        // Increment usage_count for recalled procedures
        for proc in &procs {
            let new_count = (proc.usage_count + 1).to_string();
            let mut update = HashMap::new();
            update.insert("usage_count".to_string(), new_count);
            self.graph.update_node(&proc.node_id, update);
        }

        procs
    }

    // ======================================================================
    // PROSPECTIVE MEMORY
    // ======================================================================

    /// Store a trigger-action pair for future evaluation.
    pub fn store_prospective(
        &mut self,
        description: &str,
        trigger_condition: &str,
        action_on_trigger: &str,
        priority: i32,
    ) -> Result<String> {
        let node_id = new_id("pro");
        let now = ts_now();

        let mut props = HashMap::new();
        props.insert("node_id".to_string(), node_id.clone());
        props.insert("agent_id".to_string(), self.agent_name.clone());
        props.insert("desc_text".to_string(), description.to_string());
        props.insert(
            "trigger_condition".to_string(),
            trigger_condition.to_string(),
        );
        props.insert(
            "action_on_trigger".to_string(),
            action_on_trigger.to_string(),
        );
        props.insert("status".to_string(), "pending".to_string());
        props.insert("priority".to_string(), priority.to_string());
        props.insert("created_at".to_string(), now.to_string());

        self.graph
            .add_node(NT_PROSPECTIVE, props, Some(&node_id))
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        Ok(node_id)
    }

    /// Check pending prospective memories against provided content.
    ///
    /// Uses a keyword-overlap heuristic: if any word from the
    /// `trigger_condition` appears in `content`, the prospective memory
    /// is considered triggered (status updated to `"triggered"`).
    pub fn check_triggers(&mut self, content: &str) -> Vec<ProspectiveMemory> {
        let mut filter = agent_filter(&self.agent_name);
        filter.insert("status".to_string(), "pending".to_string());
        let nodes = self
            .graph
            .query_nodes(NT_PROSPECTIVE, Some(&filter), usize::MAX);

        let mut candidates: Vec<ProspectiveMemory> = nodes
            .iter()
            .map(|n| node_to_prospective(&n.properties))
            .collect();

        // Sort by priority descending
        candidates.sort_by(|a, b| b.priority.cmp(&a.priority));

        let content_lower = content.to_lowercase();
        let mut triggered = Vec::new();

        for mut pm in candidates {
            let trigger_words: Vec<String> = pm
                .trigger_condition
                .split_whitespace()
                .filter(|w| !w.is_empty())
                .map(|w| w.to_lowercase())
                .collect();

            let content_words: std::collections::HashSet<&str> =
                content_lower.split_whitespace().collect();
            if trigger_words
                .iter()
                .any(|w| content_words.contains(w.as_str()))
            {
                // Mark as triggered
                let mut update = HashMap::new();
                update.insert("status".to_string(), "triggered".to_string());
                self.graph.update_node(&pm.node_id, update);

                pm.status = "triggered".to_string();
                triggered.push(pm);
            }
        }

        triggered
    }

    /// Mark a prospective memory as resolved.
    pub fn resolve_prospective(&mut self, node_id: &str) {
        let mut update = HashMap::new();
        update.insert("status".to_string(), "resolved".to_string());
        self.graph.update_node(node_id, update);
    }

    // ======================================================================
    // STATISTICS
    // ======================================================================

    /// Return counts per memory type.
    ///
    /// Returns a map with keys for each `MemoryCategory` (as string) plus
    /// a `"total"` key.
    pub fn get_memory_stats(&self) -> HashMap<String, usize> {
        let tables = [
            (MemoryCategory::Sensory, NT_SENSORY),
            (MemoryCategory::Working, NT_WORKING),
            (MemoryCategory::Episodic, NT_EPISODIC),
            (MemoryCategory::Semantic, NT_SEMANTIC),
            (MemoryCategory::Procedural, NT_PROCEDURAL),
            (MemoryCategory::Prospective, NT_PROSPECTIVE),
        ];

        let mut stats = HashMap::new();
        let mut total = 0usize;
        let filter = agent_filter(&self.agent_name);

        for (category, node_type) in &tables {
            let count = self
                .graph
                .query_nodes(node_type, Some(&filter), usize::MAX)
                .len();
            stats.insert(category.as_str().to_string(), count);
            total += count;
        }

        stats.insert("total".to_string(), total);
        stats
    }

    /// Alias matching the Python method name `get_statistics`.
    pub fn get_statistics(&self) -> HashMap<String, usize> {
        self.get_memory_stats()
    }

    // ======================================================================
    // LIFECYCLE
    // ======================================================================

    /// Release graph resources.
    pub fn close(&mut self) {
        self.graph.close();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::types::Direction;

    fn make_cm() -> CognitiveMemory {
        CognitiveMemory::new("test-agent").unwrap()
    }

    // -- construction -------------------------------------------------------

    #[test]
    fn test_new_empty_agent_name_rejected() {
        assert!(CognitiveMemory::new("").is_err());
        assert!(CognitiveMemory::new("   ").is_err());
    }

    #[test]
    fn test_new_valid_agent_name() {
        let cm = CognitiveMemory::new("  alice  ").unwrap();
        assert_eq!(cm.agent_name(), "alice");
    }

    // -- helpers ------------------------------------------------------------

    #[test]
    fn test_new_id_format() {
        let id = new_id("sen");
        assert!(id.starts_with("sen_"));
        assert_eq!(id.len(), 4 + 12); // prefix_ + 12 hex chars
    }

    #[test]
    fn test_ts_now_positive() {
        assert!(ts_now() > 0);
    }

    // -- sensory memory -----------------------------------------------------

    #[test]
    fn test_record_and_get_sensory() {
        let mut cm = make_cm();
        let id = cm.record_sensory("text", "hello world", 300).unwrap();
        assert!(id.starts_with("sen_"));

        let items = cm.get_recent_sensory(10);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].modality, "text");
        assert_eq!(items[0].raw_data, "hello world");
    }

    #[test]
    fn test_sensory_ordering() {
        let mut cm = make_cm();
        cm.record_sensory("a", "first", 300).unwrap();
        cm.record_sensory("b", "second", 300).unwrap();
        cm.record_sensory("c", "third", 300).unwrap();

        let items = cm.get_recent_sensory(10);
        assert_eq!(items.len(), 3);
        // Most recent first
        assert_eq!(items[0].modality, "c");
        assert_eq!(items[1].modality, "b");
        assert_eq!(items[2].modality, "a");
    }

    #[test]
    fn test_sensory_limit() {
        let mut cm = make_cm();
        for i in 0..5 {
            cm.record_sensory("text", &format!("item-{i}"), 300)
                .unwrap();
        }
        let items = cm.get_recent_sensory(2);
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_prune_expired_sensory() {
        let mut cm = make_cm();
        // Use 0 TTL so it expires immediately
        cm.record_sensory("text", "ephemeral", 0).unwrap();
        // Give it a moment
        std::thread::sleep(std::time::Duration::from_millis(10));

        // This item has a long TTL and should survive
        cm.record_sensory("text", "persistent", 9999).unwrap();

        let pruned = cm.prune_expired_sensory();
        assert_eq!(pruned, 1);

        let remaining = cm.get_recent_sensory(10);
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].raw_data, "persistent");
    }

    #[test]
    fn test_attend_to_sensory() {
        let mut cm = make_cm();
        let sid = cm.record_sensory("error", "segfault", 300).unwrap();
        let ep_id = cm.attend_to_sensory(&sid, "critical error");
        assert!(ep_id.is_some());

        let episodes = cm.get_episodes(10, false);
        assert_eq!(episodes.len(), 1);
        assert!(episodes[0].content.contains("segfault"));
        assert!(episodes[0].content.contains("critical error"));
    }

    #[test]
    fn test_attend_to_missing_sensory() {
        let mut cm = make_cm();
        assert!(cm.attend_to_sensory("nonexistent", "reason").is_none());
    }

    // -- working memory -----------------------------------------------------

    #[test]
    fn test_push_and_recall_working() {
        let mut cm = make_cm();
        let id = cm
            .push_working("goal", "build feature X", "task-1", 0.9)
            .unwrap();
        assert!(id.starts_with("wrk_"));

        let slots = cm.recall_working("task-1");
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0].content, "build feature X");
        assert_eq!(slots[0].relevance, 0.9);
    }

    #[test]
    fn test_working_memory_capacity_eviction() {
        let mut cm = make_cm();

        // Fill to capacity
        for i in 0..WORKING_MEMORY_CAPACITY {
            cm.push_working("item", &format!("slot-{i}"), "t1", i as f64)
                .unwrap();
        }
        assert_eq!(cm.recall_working("t1").len(), WORKING_MEMORY_CAPACITY);

        // Push one more — should evict lowest relevance (0.0)
        cm.push_working("item", "new-slot", "t1", 100.0).unwrap();
        let slots = cm.recall_working("t1");
        assert_eq!(slots.len(), WORKING_MEMORY_CAPACITY);

        // The slot with relevance 0.0 should be gone
        assert!(!slots.iter().any(|s| s.content == "slot-0"));
        // The new one should be there
        assert!(slots.iter().any(|s| s.content == "new-slot"));
    }

    #[test]
    fn test_working_task_isolation() {
        let mut cm = make_cm();
        cm.push_working("g", "a", "t1", 1.0).unwrap();
        cm.push_working("g", "b", "t2", 1.0).unwrap();

        assert_eq!(cm.recall_working("t1").len(), 1);
        assert_eq!(cm.recall_working("t2").len(), 1);
    }

    #[test]
    fn test_clear_working() {
        let mut cm = make_cm();
        cm.push_working("g", "a", "t1", 1.0).unwrap();
        cm.push_working("g", "b", "t1", 1.0).unwrap();

        let cleared = cm.clear_working("t1");
        assert_eq!(cleared, 2);
        assert!(cm.recall_working("t1").is_empty());
    }

    // -- episodic memory ----------------------------------------------------

    #[test]
    fn test_store_and_get_episodes() {
        let mut cm = make_cm();
        let id = cm
            .store_episode("something happened", "user-session", None, None)
            .unwrap();
        assert!(id.starts_with("epi_"));

        let eps = cm.get_episodes(10, false);
        assert_eq!(eps.len(), 1);
        assert_eq!(eps[0].content, "something happened");
        assert_eq!(eps[0].source_label, "user-session");
        assert!(!eps[0].compressed);
    }

    #[test]
    fn test_episode_temporal_ordering() {
        let mut cm = make_cm();
        cm.store_episode("first", "src", None, None).unwrap();
        cm.store_episode("second", "src", None, None).unwrap();
        cm.store_episode("third", "src", None, None).unwrap();

        let eps = cm.get_episodes(10, false);
        assert_eq!(eps[0].content, "third");
        assert_eq!(eps[1].content, "second");
        assert_eq!(eps[2].content, "first");
    }

    #[test]
    fn test_consolidate_episodes() {
        let mut cm = make_cm();
        for i in 0..5 {
            cm.store_episode(&format!("event-{i}"), "src", None, None)
                .unwrap();
        }

        // Not enough for batch_size=10
        assert!(cm
            .consolidate_episodes::<fn(&[String]) -> String>(10, None)
            .is_none());

        // Enough for batch_size=3
        let cons_id = cm
            .consolidate_episodes::<fn(&[String]) -> String>(3, None)
            .unwrap();
        assert!(cons_id.starts_with("con_"));

        // The consolidated episodes should be compressed
        let eps = cm.get_episodes(10, false);
        assert_eq!(eps.len(), 2); // only 2 uncompressed remain
    }

    #[test]
    fn test_consolidate_with_custom_summarizer() {
        let mut cm = make_cm();
        for i in 0..3 {
            cm.store_episode(&format!("e{i}"), "src", None, None)
                .unwrap();
        }

        let cons_id = cm
            .consolidate_episodes(
                3,
                Some(|contents: &[String]| format!("SUMMARY({})", contents.len())),
            )
            .unwrap();

        // Verify the consolidated node exists
        let node = cm.graph.get_node(&cons_id).unwrap();
        assert_eq!(node.properties.get("summary").unwrap(), "SUMMARY(3)");
    }

    #[test]
    fn test_recall_episodes_excludes_compressed() {
        let mut cm = make_cm();
        for i in 0..5 {
            cm.store_episode(&format!("ep-{i}"), "src", None, None)
                .unwrap();
        }
        cm.consolidate_episodes::<fn(&[String]) -> String>(3, None);

        let uncompressed = cm.recall_episodes(10);
        assert_eq!(uncompressed.len(), 2);

        let all = cm.get_episodes(10, true);
        assert_eq!(all.len(), 5);
    }

    // -- semantic memory ----------------------------------------------------

    #[test]
    fn test_store_and_search_facts() {
        let mut cm = make_cm();
        cm.store_fact("rust", "Rust is a systems language", 0.95, "", None, None)
            .unwrap();
        cm.store_fact("python", "Python is interpreted", 0.9, "", None, None)
            .unwrap();

        let results = cm.search_facts("rust", 10, 0.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].concept, "rust");
    }

    #[test]
    fn test_search_facts_confidence_filter() {
        let mut cm = make_cm();
        cm.store_fact("a", "low confidence", 0.3, "", None, None)
            .unwrap();
        cm.store_fact("a", "high confidence", 0.9, "", None, None)
            .unwrap();

        let results = cm.search_facts("confidence", 10, 0.5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "high confidence");
    }

    #[test]
    fn test_search_facts_empty_query_returns_all() {
        let mut cm = make_cm();
        cm.store_fact("a", "content a", 1.0, "", None, None)
            .unwrap();
        cm.store_fact("b", "content b", 0.5, "", None, None)
            .unwrap();

        let results = cm.search_facts("", 10, 0.0);
        assert_eq!(results.len(), 2);
        // Sorted by confidence desc
        assert_eq!(results[0].concept, "a");
    }

    #[test]
    fn test_get_all_facts() {
        let mut cm = make_cm();
        cm.store_fact("x", "xval", 1.0, "", None, None).unwrap();
        cm.store_fact("y", "yval", 0.5, "", None, None).unwrap();

        let all = cm.get_all_facts(50);
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_fact_with_tags_and_metadata() {
        let mut cm = make_cm();
        let tags = vec!["lang".to_string(), "systems".to_string()];
        let mut meta = HashMap::new();
        meta.insert(
            "source".to_string(),
            serde_json::Value::String("docs".into()),
        );

        let id = cm
            .store_fact("rust", "fast lang", 0.9, "doc-1", Some(&tags), Some(&meta))
            .unwrap();

        let facts = cm.search_facts("rust", 10, 0.0);
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].tags, tags);
        assert_eq!(
            facts[0].metadata.get("source").unwrap(),
            &serde_json::Value::String("docs".into())
        );
        assert_eq!(facts[0].source_id, "doc-1");
        assert_eq!(facts[0].node_id, id);
    }

    #[test]
    fn test_link_similar_facts() {
        let mut cm = make_cm();
        let a = cm
            .store_fact("rust", "systems lang", 1.0, "", None, None)
            .unwrap();
        let b = cm
            .store_fact("cpp", "systems lang too", 1.0, "", None, None)
            .unwrap();

        assert!(cm.link_similar_facts(&a, &b, 0.85).is_ok());

        // Verify the SIMILAR_TO edge was actually created
        let neighbors = cm.graph.query_neighbors(&a, None, Direction::Both, 10);
        assert!(
            neighbors
                .iter()
                .any(|(e, _)| e.edge_type == "SIMILAR_TO" && e.target_id == b),
            "SIMILAR_TO edge should exist between the two facts"
        );
    }

    // -- procedural memory --------------------------------------------------

    #[test]
    fn test_store_and_recall_procedures() {
        let mut cm = make_cm();
        let steps = vec!["cargo build".into(), "cargo test".into()];
        let id = cm.store_procedure("build-rust", &steps, None).unwrap();
        assert!(id.starts_with("proc_"));

        let procs = cm.recall_procedures("rust", 10);
        assert_eq!(procs.len(), 1);
        assert_eq!(procs[0].name, "build-rust");
        assert_eq!(procs[0].steps, steps);
    }

    #[test]
    fn test_recall_procedures_increments_usage() {
        let mut cm = make_cm();
        let steps = vec!["step1".into()];
        cm.store_procedure("deploy", &steps, None).unwrap();

        let procs = cm.recall_procedures_mut("deploy", 10);
        assert_eq!(procs[0].usage_count, 0); // returned before increment

        // After recall_procedures_mut, the stored count should be 1
        let procs2 = cm.recall_procedures("deploy", 10);
        assert_eq!(procs2[0].usage_count, 1);
    }

    #[test]
    fn test_procedure_with_prerequisites() {
        let mut cm = make_cm();
        let steps = vec!["run tests".into()];
        let prereqs = vec!["install deps".into()];
        cm.store_procedure("test-suite", &steps, Some(&prereqs))
            .unwrap();

        let procs = cm.recall_procedures("test", 10);
        assert_eq!(procs[0].prerequisites, prereqs);
    }

    #[test]
    fn test_recall_procedures_empty_query_returns_all() {
        let mut cm = make_cm();
        cm.store_procedure("a", &["s".into()], None).unwrap();
        cm.store_procedure("b", &["s".into()], None).unwrap();

        let procs = cm.recall_procedures("", 10);
        assert_eq!(procs.len(), 2);
    }

    // -- prospective memory -------------------------------------------------

    #[test]
    fn test_store_and_check_triggers() {
        let mut cm = make_cm();
        cm.store_prospective(
            "notify on deploy",
            "deploy production",
            "send notification",
            5,
        )
        .unwrap();

        // Content that matches
        let triggered = cm.check_triggers("starting deploy to staging");
        assert_eq!(triggered.len(), 1);
        assert_eq!(triggered[0].status, "triggered");
        assert_eq!(triggered[0].action_on_trigger, "send notification");
    }

    #[test]
    fn test_check_triggers_no_match() {
        let mut cm = make_cm();
        cm.store_prospective("remind", "deploy", "do stuff", 1)
            .unwrap();

        let triggered = cm.check_triggers("completely unrelated text");
        assert!(triggered.is_empty());
    }

    #[test]
    fn test_triggered_not_re_triggered() {
        let mut cm = make_cm();
        cm.store_prospective("notify", "deploy", "action", 1)
            .unwrap();

        let t1 = cm.check_triggers("deploy happening");
        assert_eq!(t1.len(), 1);

        // Already triggered — should not trigger again
        let t2 = cm.check_triggers("deploy happening again");
        assert!(t2.is_empty());
    }

    #[test]
    fn test_resolve_prospective() {
        let mut cm = make_cm();
        let id = cm
            .store_prospective("notify", "deploy", "action", 1)
            .unwrap();

        let triggered = cm.check_triggers("deploy now");
        assert_eq!(triggered.len(), 1);

        cm.resolve_prospective(&id);

        // After resolving, it should not appear in pending or triggered checks
        let filter = agent_filter(&cm.agent_name);
        let nodes = cm
            .graph
            .query_nodes(NT_PROSPECTIVE, Some(&filter), usize::MAX);
        let node = nodes.iter().find(|n| n.node_id == id).unwrap();
        assert_eq!(node.properties.get("status").unwrap(), "resolved");
    }

    #[test]
    fn test_prospective_priority_ordering() {
        let mut cm = make_cm();
        cm.store_prospective("low", "deploy", "low-action", 1)
            .unwrap();
        cm.store_prospective("high", "deploy", "high-action", 10)
            .unwrap();

        let triggered = cm.check_triggers("deploy now");
        assert_eq!(triggered.len(), 2);
        // Higher priority first
        assert_eq!(triggered[0].priority, 10);
    }

    // -- statistics ---------------------------------------------------------

    #[test]
    fn test_get_memory_stats_empty() {
        let cm = make_cm();
        let stats = cm.get_memory_stats();
        assert_eq!(*stats.get("total").unwrap(), 0);
        assert_eq!(*stats.get("sensory").unwrap(), 0);
    }

    #[test]
    fn test_get_memory_stats_with_data() {
        let mut cm = make_cm();
        cm.record_sensory("text", "a", 300).unwrap();
        cm.record_sensory("text", "b", 300).unwrap();
        cm.push_working("g", "c", "t1", 1.0).unwrap();
        cm.store_episode("ep", "src", None, None).unwrap();
        cm.store_fact("f", "fact", 1.0, "", None, None).unwrap();
        cm.store_procedure("p", &["s".into()], None).unwrap();
        cm.store_prospective("d", "t", "a", 1).unwrap();

        let stats = cm.get_memory_stats();
        assert_eq!(*stats.get("sensory").unwrap(), 2);
        assert_eq!(*stats.get("working").unwrap(), 1);
        assert_eq!(*stats.get("episodic").unwrap(), 1);
        assert_eq!(*stats.get("semantic").unwrap(), 1);
        assert_eq!(*stats.get("procedural").unwrap(), 1);
        assert_eq!(*stats.get("prospective").unwrap(), 1);
        assert_eq!(*stats.get("total").unwrap(), 7);
    }

    #[test]
    fn test_get_statistics_alias() {
        let cm = make_cm();
        let s1 = cm.get_memory_stats();
        let s2 = cm.get_statistics();
        assert_eq!(s1, s2);
    }

    // -- agent isolation ----------------------------------------------------

    #[test]
    fn test_agent_isolation() {
        let mut cm_a = CognitiveMemory::new("alice").unwrap();
        let mut cm_b = CognitiveMemory::new("bob").unwrap();

        cm_a.store_fact("x", "alice fact", 1.0, "", None, None)
            .unwrap();
        cm_b.store_fact("x", "bob fact", 1.0, "", None, None)
            .unwrap();

        // Each agent only sees their own facts
        // (separate InMemoryGraphStore instances give natural isolation)
        assert_eq!(cm_a.search_facts("fact", 10, 0.0).len(), 1);
        assert_eq!(cm_b.search_facts("fact", 10, 0.0).len(), 1);
        assert_eq!(cm_a.search_facts("fact", 10, 0.0)[0].content, "alice fact");
        assert_eq!(cm_b.search_facts("fact", 10, 0.0)[0].content, "bob fact");
    }

    // -- lifecycle ----------------------------------------------------------

    #[test]
    fn test_close() {
        let mut cm = make_cm();
        cm.record_sensory("text", "data", 300).unwrap();
        cm.close();
        // After close, the store is empty
        assert!(cm.get_recent_sensory(10).is_empty());
    }

    // -- edge cases ---------------------------------------------------------

    #[test]
    fn test_episode_explicit_temporal_index() {
        let mut cm = make_cm();
        cm.store_episode("ep1", "src", Some(100), None).unwrap();
        cm.store_episode("ep2", "src", Some(50), None).unwrap();

        let eps = cm.get_episodes(10, false);
        // Sorted by temporal_index desc
        assert_eq!(eps[0].temporal_index, 100);
        assert_eq!(eps[1].temporal_index, 50);
    }

    #[test]
    fn test_episode_auto_increment_after_explicit() {
        let mut cm = make_cm();
        cm.store_episode("ep1", "src", Some(100), None).unwrap();
        cm.store_episode("ep2", "src", None, None).unwrap();

        let eps = cm.get_episodes(10, false);
        // Auto-incremented should be > 100
        assert!(eps.iter().any(|e| e.temporal_index == 101));
    }

    // ====================================================================
    // Extended coverage: record_sensory + get_recent_sensory
    // ====================================================================

    #[test]
    fn test_record_sensory_multiple_modalities() {
        let mut cm = make_cm();
        cm.record_sensory("visual", "red light", 300).unwrap();
        cm.record_sensory("audio", "beep sound", 300).unwrap();
        cm.record_sensory("text", "error message", 300).unwrap();

        let items = cm.get_recent_sensory(10);
        assert_eq!(items.len(), 3);
        let modalities: Vec<&str> = items.iter().map(|i| i.modality.as_str()).collect();
        assert!(modalities.contains(&"visual"));
        assert!(modalities.contains(&"audio"));
        assert!(modalities.contains(&"text"));
    }

    #[test]
    fn test_record_sensory_returns_unique_ids() {
        let mut cm = make_cm();
        let id1 = cm.record_sensory("text", "a", 300).unwrap();
        let id2 = cm.record_sensory("text", "b", 300).unwrap();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_get_recent_sensory_respects_expiry() {
        let mut cm = make_cm();
        cm.record_sensory("text", "expired", 0).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        cm.record_sensory("text", "alive", 9999).unwrap();

        let items = cm.get_recent_sensory(10);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].raw_data, "alive");
    }

    // ====================================================================
    // Extended coverage: expire_sensory
    // ====================================================================

    #[test]
    fn test_expire_sensory_is_alias_for_prune() {
        let mut cm = make_cm();
        cm.record_sensory("text", "gone", 0).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        cm.record_sensory("text", "stays", 9999).unwrap();

        let pruned = cm.expire_sensory();
        assert_eq!(pruned, 1);
        assert_eq!(cm.get_recent_sensory(10).len(), 1);
    }

    #[test]
    fn test_expire_sensory_nothing_to_prune() {
        let mut cm = make_cm();
        cm.record_sensory("text", "long-lived", 99999).unwrap();
        assert_eq!(cm.expire_sensory(), 0);
    }

    // ====================================================================
    // Extended coverage: store_episode
    // ====================================================================

    #[test]
    fn test_store_episode_with_metadata() {
        let mut cm = make_cm();
        let mut meta = HashMap::new();
        meta.insert(
            "location".to_string(),
            serde_json::Value::String("lab".into()),
        );
        let id = cm
            .store_episode("experiment completed", "lab-session", None, Some(&meta))
            .unwrap();

        let eps = cm.get_episodes(10, false);
        let ep = eps.iter().find(|e| e.node_id == id).unwrap();
        assert_eq!(
            ep.metadata.get("location").and_then(|v| v.as_str()),
            Some("lab")
        );
    }

    #[test]
    fn test_store_episode_returns_unique_ids() {
        let mut cm = make_cm();
        let id1 = cm.store_episode("ep1", "src", None, None).unwrap();
        let id2 = cm.store_episode("ep2", "src", None, None).unwrap();
        assert_ne!(id1, id2);
        assert!(id1.starts_with("epi_"));
    }

    // ====================================================================
    // Extended coverage: store_fact
    // ====================================================================

    #[test]
    fn test_store_fact_invalid_confidence_nan() {
        let mut cm = make_cm();
        assert!(cm
            .store_fact("c", "content", f64::NAN, "", None, None)
            .is_err());
    }

    #[test]
    fn test_store_fact_invalid_confidence_out_of_range() {
        let mut cm = make_cm();
        assert!(cm.store_fact("c", "content", 1.5, "", None, None).is_err());
        assert!(cm.store_fact("c", "content", -0.1, "", None, None).is_err());
    }

    #[test]
    fn test_store_fact_boundary_confidence() {
        let mut cm = make_cm();
        let id_zero = cm
            .store_fact("c", "zero conf", 0.0, "", None, None)
            .unwrap();
        let id_max = cm.store_fact("c", "max conf", 1.0, "", None, None).unwrap();

        // Verify boundary values were persisted correctly
        let facts = cm.search_facts("conf", 10, 0.0);
        let zero_fact = facts.iter().find(|f| f.node_id == id_zero);
        let max_fact = facts.iter().find(|f| f.node_id == id_max);
        assert!(
            zero_fact.is_some(),
            "zero-confidence fact should be retrievable"
        );
        assert!(
            max_fact.is_some(),
            "max-confidence fact should be retrievable"
        );
        assert!((zero_fact.unwrap().confidence - 0.0).abs() < f64::EPSILON);
        assert!((max_fact.unwrap().confidence - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_store_fact_returns_searchable_id() {
        let mut cm = make_cm();
        let id = cm
            .store_fact("rust-lang", "Rust is safe", 0.95, "src1", None, None)
            .unwrap();
        assert!(id.starts_with("sem_"));

        let facts = cm.search_facts("rust", 10, 0.0);
        assert!(facts.iter().any(|f| f.node_id == id));
    }

    // ====================================================================
    // Extended coverage: push_working
    // ====================================================================

    #[test]
    fn test_push_working_different_slot_types() {
        let mut cm = make_cm();
        cm.push_working("goal", "build X", "task-1", 0.9).unwrap();
        cm.push_working("context", "we are in beta", "task-1", 0.7)
            .unwrap();
        cm.push_working("constraint", "deadline friday", "task-1", 0.8)
            .unwrap();

        let slots = cm.recall_working("task-1");
        assert_eq!(slots.len(), 3);
        // Sorted by relevance desc
        assert_eq!(slots[0].slot_type, "goal");
        assert_eq!(slots[0].relevance, 0.9);
    }

    #[test]
    fn test_push_working_at_exact_capacity() {
        let mut cm = make_cm();
        for i in 0..WORKING_MEMORY_CAPACITY {
            cm.push_working("item", &format!("slot-{i}"), "t1", (i + 1) as f64)
                .unwrap();
        }
        assert_eq!(cm.recall_working("t1").len(), WORKING_MEMORY_CAPACITY);

        // Push one more with high relevance — should evict lowest (slot-0 with rel=1.0)
        cm.push_working("item", "overflow-slot", "t1", 999.0)
            .unwrap();
        let slots = cm.recall_working("t1");
        assert_eq!(slots.len(), WORKING_MEMORY_CAPACITY);
        assert!(slots.iter().any(|s| s.content == "overflow-slot"));
    }

    // ====================================================================
    // Extended coverage: link_fact_to_episode
    // ====================================================================

    #[test]
    fn test_link_fact_to_episode_creates_edge() {
        let mut cm = make_cm();
        let fact_id = cm
            .store_fact("rust", "Rust is memory safe", 0.95, "", None, None)
            .unwrap();
        let ep_id = cm
            .store_episode("learned about Rust", "reading", None, None)
            .unwrap();

        assert!(cm.link_fact_to_episode(&fact_id, &ep_id).is_ok());

        // Verify edge exists via graph query
        use crate::graph::{Direction, GraphStore};
        let neighbors =
            cm.graph
                .query_neighbors(&fact_id, Some(ET_DERIVES_FROM), Direction::Outgoing, 10);
        assert!(
            !neighbors.is_empty(),
            "Expected DERIVES_FROM edge from fact to episode"
        );
        assert_eq!(neighbors[0].1.node_id, ep_id);
    }

    #[test]
    fn test_link_fact_to_episode_multiple_links() {
        let mut cm = make_cm();
        let fact_id = cm
            .store_fact("topic", "A general fact", 0.8, "", None, None)
            .unwrap();
        let ep1 = cm.store_episode("episode one", "src", None, None).unwrap();
        let ep2 = cm.store_episode("episode two", "src", None, None).unwrap();

        assert!(cm.link_fact_to_episode(&fact_id, &ep1).is_ok());
        assert!(cm.link_fact_to_episode(&fact_id, &ep2).is_ok());
    }

    // ====================================================================
    // Extended coverage: recall_procedures_mut
    // ====================================================================

    #[test]
    fn test_recall_procedures_mut_multiple_increments() {
        let mut cm = make_cm();
        cm.store_procedure("build", &["compile".into()], None)
            .unwrap();

        cm.recall_procedures_mut("build", 10);
        cm.recall_procedures_mut("build", 10);
        cm.recall_procedures_mut("build", 10);

        let procs = cm.recall_procedures("build", 10);
        assert_eq!(procs[0].usage_count, 3);
    }

    // ====================================================================
    // Extended coverage: store_prospective
    // ====================================================================

    #[test]
    fn test_store_prospective_returns_valid_id() {
        let mut cm = make_cm();
        let id = cm
            .store_prospective("remind me", "meeting starts", "join call", 5)
            .unwrap();
        assert!(id.starts_with("pro_"));
    }

    #[test]
    fn test_store_prospective_initial_status_pending() {
        let mut cm = make_cm();
        let id = cm
            .store_prospective("task", "trigger", "action", 1)
            .unwrap();

        let filter = agent_filter(&cm.agent_name);
        let nodes = cm
            .graph
            .query_nodes(NT_PROSPECTIVE, Some(&filter), usize::MAX);
        let node = nodes.iter().find(|n| n.node_id == id).unwrap();
        assert_eq!(node.properties.get("status").unwrap(), "pending");
    }

    // ====================================================================
    // Extended coverage: get_memory_stats
    // ====================================================================

    #[test]
    fn test_get_memory_stats_after_deletions() {
        let mut cm = make_cm();
        cm.record_sensory("text", "temp", 0).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        cm.record_sensory("text", "kept", 9999).unwrap();
        cm.prune_expired_sensory();

        let stats = cm.get_memory_stats();
        assert_eq!(*stats.get("sensory").unwrap(), 1);
        assert_eq!(*stats.get("total").unwrap(), 1);
    }

    #[test]
    fn test_get_memory_stats_all_categories() {
        let cm = make_cm();
        let stats = cm.get_memory_stats();
        assert!(stats.contains_key("working"));
        assert!(stats.contains_key("episodic"));
        assert!(stats.contains_key("semantic"));
        assert!(stats.contains_key("procedural"));
        assert!(stats.contains_key("prospective"));
        assert!(stats.contains_key("total"));
    }

    // ====================================================================
    // Extended coverage: check_triggers
    // ====================================================================

    #[test]
    fn test_check_triggers_partial_word_no_match() {
        let mut cm = make_cm();
        cm.store_prospective("notify", "deployment", "run script", 1)
            .unwrap();

        // "deploy" is not the same word as "deployment" in word-based matching
        let triggered = cm.check_triggers("deploy is happening");
        // This depends on tokenization — "deployment" != "deploy"
        // The trigger uses word-level overlap, so exact word match is needed
        assert!(triggered.is_empty());
    }

    #[test]
    fn test_check_triggers_multiple_triggers() {
        let mut cm = make_cm();
        cm.store_prospective("alert1", "error", "log error", 2)
            .unwrap();
        cm.store_prospective("alert2", "error", "notify admin", 5)
            .unwrap();

        let triggered = cm.check_triggers("an error occurred");
        assert_eq!(triggered.len(), 2);
        // Higher priority first
        assert_eq!(triggered[0].priority, 5);
        assert_eq!(triggered[1].priority, 2);
    }
}
