//! Working memory methods for [`CognitiveMemory`].

use std::collections::HashMap;

use crate::memory_types::WorkingMemorySlot;
use crate::{MemoryError, Result};

use crate::graph::protocol::GraphStore;

use super::converters::node_to_working;
use super::types::{agent_filter, new_id, ts_now, NT_WORKING, WORKING_MEMORY_CAPACITY};
use super::CognitiveMemory;

impl CognitiveMemory {
    /// Store a slot into working memory for a given task.
    ///
    /// If the task already has [`WORKING_MEMORY_CAPACITY`] slots, the
    /// least-relevant slot is evicted.
    pub fn store_working(
        &mut self,
        slot_type: &str,
        content: &str,
        task_id: &str,
        relevance: f64,
    ) -> Result<String> {
        // Evict if at capacity
        let existing = self.get_working(task_id);
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
    pub fn get_working(&self, task_id: &str) -> Vec<WorkingMemorySlot> {
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

    /// Deprecated: renamed to [`store_working`](Self::store_working).
    #[deprecated(since = "0.2.0", note = "renamed to store_working")]
    pub fn push_working(
        &mut self,
        slot_type: &str,
        content: &str,
        task_id: &str,
        relevance: f64,
    ) -> Result<String> {
        self.store_working(slot_type, content, task_id, relevance)
    }

    /// Deprecated: renamed to [`get_working`](Self::get_working).
    #[deprecated(since = "0.2.0", note = "renamed to get_working")]
    pub fn recall_working(&self, task_id: &str) -> Vec<WorkingMemorySlot> {
        self.get_working(task_id)
    }
}
