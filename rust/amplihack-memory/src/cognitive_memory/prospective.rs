//! Prospective memory methods for [`CognitiveMemory`].

use std::collections::HashMap;

use crate::memory_types::ProspectiveMemory;
use crate::{MemoryError, Result};

use crate::graph::protocol::GraphStore;

use super::converters::node_to_prospective;
use super::types::{agent_filter, new_id, ts_now, NT_PROSPECTIVE};
use super::CognitiveMemory;

impl CognitiveMemory {
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
}
