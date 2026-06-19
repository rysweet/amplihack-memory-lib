//! Episodic memory methods for [`CognitiveMemory`].

use std::collections::HashMap;

use crate::memory_types::EpisodicMemory;
use crate::{MemoryError, Result};

use tracing::warn;

use super::converters::node_to_episodic;
use super::types::{agent_filter, new_id, ts_now, ET_CONSOLIDATES, NT_CONSOLIDATED, NT_EPISODIC};
use super::CognitiveMemory;

impl CognitiveMemory {
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
            .map(|m| {
                serde_json::to_string(m).unwrap_or_else(|e| {
                    warn!("store_episode: failed to serialize metadata: {e}");
                    "{}".into()
                })
            })
            .unwrap_or_else(|| "{}".into());

        let mut props = HashMap::new();
        props.insert("node_id".to_string(), node_id.clone());
        props.insert("agent_id".to_string(), self.agent_name.clone());
        props.insert("content".to_string(), content.to_string());
        props.insert("source_label".to_string(), source_label.to_string());
        props.insert("temporal_index".to_string(), tidx.to_string());
        props.insert("compressed".to_string(), "false".to_string());
        props.insert("distilled".to_string(), "false".to_string());
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
                    n.properties.get("compressed").is_none_or(|v| v != "true")
                }
            })
            .map(|n| node_to_episodic(&n.properties))
            .collect();

        // Sort by temporal_index descending
        episodes.sort_by_key(|e| std::cmp::Reverse(e.temporal_index));
        episodes.truncate(limit);
        episodes
    }

    /// Search episodes, excluding compressed ones.
    pub fn search_episodes(&self, limit: usize) -> Vec<EpisodicMemory> {
        self.get_episodes(limit, false)
    }

    /// Search episodes by case-insensitive substring over content.
    ///
    /// Returns un-compressed episodes whose lowercased `content` contains the
    /// lowercased `query`, newest-first (temporal_index descending), capped at
    /// `limit`. This is a whole-query substring match — distinct from semantic
    /// fact search, which OR-matches individual tokens.
    pub fn search_episodes_by_keyword(&self, query: &str, limit: usize) -> Vec<EpisodicMemory> {
        let query_lc = query.to_lowercase();
        let filter = agent_filter(&self.agent_name);
        let nodes = self
            .graph
            .query_nodes(NT_EPISODIC, Some(&filter), usize::MAX);

        let mut episodes: Vec<EpisodicMemory> = nodes
            .into_iter()
            .filter(|n| n.properties.get("compressed").is_none_or(|v| v != "true"))
            .filter(|n| {
                n.properties
                    .get("content")
                    .map(|c| c.to_lowercase().contains(&query_lc))
                    .unwrap_or(false)
            })
            .map(|n| node_to_episodic(&n.properties))
            .collect();

        episodes.sort_by_key(|e| std::cmp::Reverse(e.temporal_index));
        episodes.truncate(limit);
        episodes
    }

    /// Mark an episode as distilled — a one-way latch that persists across reopen.
    ///
    /// Returns `true` when the episode exists, is an episodic node owned by this
    /// agent, and the flag was persisted. Returns `false` for unknown ids,
    /// non-episode nodes, or episodes owned by a different agent. Idempotent:
    /// re-marking an already-distilled episode still returns `true`.
    pub fn mark_episode_distilled(&mut self, episode_id: &str) -> bool {
        let Some(node) = self.graph.get_node(episode_id) else {
            return false;
        };
        if node.node_type != NT_EPISODIC {
            return false;
        }
        if node.properties.get("agent_id").map(String::as_str) != Some(self.agent_name.as_str()) {
            return false;
        }
        let mut update = HashMap::new();
        update.insert("distilled".to_string(), "true".to_string());
        self.graph.update_node(episode_id, update)
    }

    /// List this agent's episodes that have not yet been distilled.
    ///
    /// Excludes episodes whose `distilled` flag is set, ordered newest-first
    /// (temporal_index descending) and capped at `limit`. Compressed episodes
    /// are intentionally NOT excluded: distillation may consume any
    /// not-yet-distilled episode regardless of its consolidation state.
    pub fn list_undistilled_episodes(&self, limit: usize) -> Vec<EpisodicMemory> {
        let filter = agent_filter(&self.agent_name);
        let nodes = self
            .graph
            .query_nodes(NT_EPISODIC, Some(&filter), usize::MAX);

        let mut episodes: Vec<EpisodicMemory> = nodes
            .into_iter()
            .filter(|n| n.properties.get("distilled").is_none_or(|v| v != "true"))
            .map(|n| node_to_episodic(&n.properties))
            .collect();

        episodes.sort_by_key(|e| std::cmp::Reverse(e.temporal_index));
        episodes.truncate(limit);
        episodes
    }

    /// Deprecated: renamed to [`search_episodes`](Self::search_episodes).
    #[deprecated(since = "0.2.0", note = "renamed to search_episodes")]
    pub fn recall_episodes(&self, limit: usize) -> Vec<EpisodicMemory> {
        self.search_episodes(limit)
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
    ) -> Result<Option<String>>
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
            .filter(|n| n.properties.get("compressed").is_none_or(|v| v != "true"))
            .map(|mut n| {
                let tidx: i64 = n
                    .properties
                    .get("temporal_index")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
                let content = n.properties.remove("content").unwrap_or_default();
                (n.node_id, content, tidx)
            })
            .collect();

        candidates.sort_by_key(|(_, _, tidx)| *tidx);

        if candidates.len() < batch_size {
            return Ok(None);
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

        self.graph
            .add_node(NT_CONSOLIDATED, props, Some(&cons_id))
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        // Mark originals as compressed and create edges
        for (idx, (ep_id, _, _)) in batch.iter().enumerate() {
            let mut update = HashMap::new();
            update.insert("compressed".to_string(), "true".to_string());
            if !self.graph.update_node(ep_id, update) {
                warn!("consolidate_episodes: failed to mark episode {ep_id} as compressed");
            }

            // Best-effort edge
            if let Err(e) = self.graph.add_edge(
                &cons_id,
                ep_id,
                ET_CONSOLIDATES,
                Some({
                    let mut m = HashMap::new();
                    m.insert("consolidated_at".to_string(), now.to_string());
                    m
                }),
            ) {
                warn!("consolidate_episodes: failed to add CONSOLIDATES edge: {e}");
                // Rollback: revert compressed flags on all episodes in this batch
                for (rollback_id, _, _) in &batch {
                    if !self.graph.update_node(rollback_id, {
                        let mut rollback = HashMap::new();
                        rollback.insert("compressed".to_string(), "false".to_string());
                        rollback
                    }) {
                        warn!("consolidate_episodes: failed to rollback compressed flag on {rollback_id}");
                    }
                }
                // Delete previously-created edges for this consolidated node
                for (prev_ep_id, _, _) in &batch[..idx] {
                    let _ = self
                        .graph
                        .delete_edge(&cons_id, prev_ep_id, ET_CONSOLIDATES);
                }
                // Delete the consolidated node itself
                let _ = self.graph.delete_node(&cons_id);
                return Err(MemoryError::Storage(format!(
                    "failed to add CONSOLIDATES edge: {e}"
                )));
            }
        }

        Ok(Some(cons_id))
    }
}
