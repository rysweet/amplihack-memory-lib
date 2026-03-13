//! Procedural memory methods for [`CognitiveMemory`].

use std::collections::HashMap;

use crate::memory_types::ProceduralMemory;
use crate::{MemoryError, Result};

use crate::graph::protocol::GraphStore;
use tracing::warn;

use super::converters::node_to_procedural;
use super::types::{agent_filter, new_id, ts_now, NT_PROCEDURAL};
use super::CognitiveMemory;

impl CognitiveMemory {
    /// Store a reusable procedure.
    pub fn store_procedure(
        &mut self,
        name: &str,
        steps: &[String],
        prerequisites: Option<&[String]>,
    ) -> Result<String> {
        let node_id = new_id("proc");
        let now = ts_now();
        let steps_json = serde_json::to_string(steps).unwrap_or_else(|e| {
            warn!("store_procedure: failed to serialize steps: {e}");
            "[]".into()
        });
        let prereqs_json = prerequisites
            .map(|p| {
                serde_json::to_string(p).unwrap_or_else(|e| {
                    warn!("store_procedure: failed to serialize prerequisites: {e}");
                    "[]".into()
                })
            })
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

    /// Search procedures matching a query.
    ///
    /// Words from `query` are matched against `name` and `steps` fields.
    /// Results are sorted by `usage_count` descending.
    pub fn search_procedures(&self, query: &str, limit: usize) -> Vec<ProceduralMemory> {
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

    /// Search procedures and increment their usage counts (mutable version).
    pub fn search_procedures_mut(&mut self, query: &str, limit: usize) -> Vec<ProceduralMemory> {
        let procs = self.search_procedures(query, limit);

        // Increment usage_count for recalled procedures
        for proc in &procs {
            let new_count = (proc.usage_count + 1).to_string();
            let mut update = HashMap::new();
            update.insert("usage_count".to_string(), new_count);
            let _ = self.graph.update_node(&proc.node_id, update);
        }

        procs
    }

    /// Deprecated: renamed to [`search_procedures`](Self::search_procedures).
    #[deprecated(since = "0.2.0", note = "renamed to search_procedures")]
    pub fn recall_procedures(&self, query: &str, limit: usize) -> Vec<ProceduralMemory> {
        self.search_procedures(query, limit)
    }

    /// Deprecated: renamed to [`search_procedures_mut`](Self::search_procedures_mut).
    #[deprecated(since = "0.2.0", note = "renamed to search_procedures_mut")]
    pub fn recall_procedures_mut(&mut self, query: &str, limit: usize) -> Vec<ProceduralMemory> {
        self.search_procedures_mut(query, limit)
    }
}
