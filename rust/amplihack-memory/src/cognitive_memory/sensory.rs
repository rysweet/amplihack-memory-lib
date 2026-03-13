//! Sensory memory methods for [`CognitiveMemory`].

use std::collections::HashMap;

use crate::memory_types::SensoryItem;
use crate::Result;

use crate::graph::protocol::GraphStore;
use tracing::warn;

use super::converters::node_to_sensory;
use super::types::{agent_filter, new_id, ts_now, ET_ATTENDED_TO, NT_SENSORY};
use super::CognitiveMemory;

impl CognitiveMemory {
    /// Store a short-lived sensory observation.
    ///
    /// Returns the `node_id` of the newly created sensory item.
    pub fn store_sensory(
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
            .map_err(|e| crate::MemoryError::Storage(e.to_string()))?;

        Ok(node_id)
    }

    /// Return the most recent sensory items that have not expired.
    pub fn get_sensory(&self, limit: usize) -> Vec<SensoryItem> {
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

        let modality = node
            .properties
            .get("modality")
            .map(|s| s.as_str())
            .unwrap_or("");
        let raw_data = node
            .properties
            .get("raw_data")
            .map(|s| s.as_str())
            .unwrap_or("");
        let content = format!("[{modality}] {raw_data} -- attended: {reason}");

        let ep_id = self
            .store_episode(&content, "sensory-attention", None, None)
            .ok()?;

        // Best-effort edge creation
        if let Err(e) = self.graph.add_edge(
            sensory_id,
            &ep_id,
            ET_ATTENDED_TO,
            Some({
                let mut m = HashMap::new();
                m.insert("attended_at".to_string(), now.to_string());
                m
            }),
        ) {
            warn!("attend_sensory: failed to add ATTENDED_TO edge: {e}");
        }

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

    /// Deprecated: renamed to [`store_sensory`](Self::store_sensory).
    #[deprecated(since = "0.2.0", note = "renamed to store_sensory")]
    pub fn record_sensory(
        &mut self,
        modality: &str,
        raw_data: &str,
        ttl_seconds: i64,
    ) -> Result<String> {
        self.store_sensory(modality, raw_data, ttl_seconds)
    }

    /// Deprecated: renamed to [`get_sensory`](Self::get_sensory).
    #[deprecated(since = "0.2.0", note = "renamed to get_sensory")]
    pub fn get_recent_sensory(&self, limit: usize) -> Vec<SensoryItem> {
        self.get_sensory(limit)
    }
}
