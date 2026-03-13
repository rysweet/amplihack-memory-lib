//! HiveGraphStore -- In-memory hive mind graph with agent registry.
//!
//! Extends InMemoryGraphStore with hive-specific operations.

use std::collections::HashMap;

use super::in_memory_store::InMemoryGraphStore;
use super::protocol::GraphStore;
use super::types::{Direction, GraphEdge, GraphNode, TraversalResult};

/// The hive mind's own graph -- metadata about the agent collective.
pub struct HiveGraphStore {
    inner: InMemoryGraphStore,
}

impl HiveGraphStore {
    pub fn new(store_id: Option<&str>) -> Self {
        let id = store_id.unwrap_or("__hive__");
        Self {
            inner: InMemoryGraphStore::new(Some(id)),
        }
    }

    /// Register an agent in the hive graph.
    pub fn register_agent(
        &mut self,
        agent_id: &str,
        domain: &str,
        trust: f64,
    ) -> crate::Result<GraphNode> {
        if !(0.0..=1.0).contains(&trust) {
            return Err(crate::MemoryError::InvalidInput(
                "trust must be between 0.0 and 1.0".into(),
            ));
        }
        let mut props = HashMap::new();
        props.insert("domain".into(), domain.to_string());
        props.insert("trust".into(), trust.to_string());
        props.insert("fact_count".into(), "0".into());
        props.insert(
            "joined_at".into(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64()
                .to_string(),
        );
        props.insert("status".into(), "active".into());
        self.inner.add_node("HiveAgent", props, Some(agent_id))
    }

    /// Get the trust score for a registered agent.
    pub fn get_agent_trust(&self, agent_id: &str) -> f64 {
        self.inner
            .get_node(agent_id)
            .and_then(|n| n.properties.get("trust")?.parse::<f64>().ok())
            .unwrap_or(0.0)
    }

    /// Update the trust score for a registered agent.
    pub fn update_trust(&mut self, agent_id: &str, new_trust: f64) -> crate::Result<()> {
        if !(0.0..=1.0).contains(&new_trust) {
            return Err(crate::MemoryError::InvalidInput(
                "trust must be between 0.0 and 1.0".into(),
            ));
        }
        if self.inner.get_node(agent_id).is_none() {
            return Err(crate::MemoryError::Internal(format!(
                "Agent not registered in hive: {agent_id}"
            )));
        }
        let mut props = HashMap::new();
        props.insert("trust".into(), new_trust.to_string());
        let _ = self.inner.update_node(agent_id, props);
        Ok(())
    }

    /// Increment the fact count for an agent.
    pub fn increment_fact_count(&mut self, agent_id: &str) {
        if let Some(node) = self.inner.get_node(agent_id) {
            let current: i64 = node
                .properties
                .get("fact_count")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            let mut props = HashMap::new();
            props.insert("fact_count".into(), (current + 1).to_string());
            let _ = self.inner.update_node(agent_id, props);
        }
    }

    /// Find agents registered as experts in a given domain.
    pub fn get_expert_agents(&self, domain: &str) -> Vec<GraphNode> {
        let mut results =
            self.inner
                .search_nodes("HiveAgent", &["domain".to_string()], domain, None, 50);
        results.sort_by(|a, b| {
            let ta: f64 = a
                .properties
                .get("trust")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0);
            let tb: f64 = b
                .properties
                .get("trust")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0);
            tb.partial_cmp(&ta).unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Record that one agent confirms another agent's fact.
    pub fn add_confirmation(
        &mut self,
        confirming_agent_id: &str,
        confirmed_agent_id: &str,
        confidence: f64,
    ) -> crate::Result<()> {
        let mut props = HashMap::new();
        props.insert(
            "confirmed_at".into(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64()
                .to_string(),
        );
        props.insert("confidence".into(), confidence.to_string());
        self.inner.add_edge(
            confirming_agent_id,
            confirmed_agent_id,
            "CONFIRMED_BY",
            Some(props),
        )?;
        Ok(())
    }

    /// Record a contradiction between two agents.
    pub fn add_contradiction(
        &mut self,
        agent_a_id: &str,
        agent_b_id: &str,
        description: &str,
    ) -> crate::Result<()> {
        let mut props = HashMap::new();
        props.insert(
            "detected_at".into(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64()
                .to_string(),
        );
        props.insert("resolved".into(), "false".into());
        props.insert("description".into(), description.to_string());
        self.inner
            .add_edge(agent_a_id, agent_b_id, "CONTRADICTS", Some(props))?;
        Ok(())
    }

    /// Record a semantic bridge between two agents.
    pub fn add_semantic_bridge(
        &mut self,
        agent_a_id: &str,
        agent_b_id: &str,
        similarity: f64,
    ) -> crate::Result<()> {
        let mut props = HashMap::new();
        props.insert("similarity".into(), similarity.to_string());
        props.insert(
            "detected_at".into(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64()
                .to_string(),
        );
        self.inner
            .add_edge(agent_a_id, agent_b_id, "SAME_CONCEPT", Some(props))?;
        Ok(())
    }

    /// Count how many times this agent's facts have been confirmed.
    pub fn get_confirmation_count(&self, agent_id: &str) -> usize {
        self.inner
            .query_neighbors(agent_id, Some("CONFIRMED_BY"), Direction::Incoming, 1000)
            .len()
    }
}

impl GraphStore for HiveGraphStore {
    fn store_id(&self) -> &str {
        self.inner.store_id()
    }

    fn add_node(
        &mut self,
        node_type: &str,
        properties: HashMap<String, String>,
        node_id: Option<&str>,
    ) -> crate::Result<GraphNode> {
        self.inner.add_node(node_type, properties, node_id)
    }

    fn get_node(&self, node_id: &str) -> Option<GraphNode> {
        self.inner.get_node(node_id)
    }

    fn query_nodes(
        &self,
        node_type: &str,
        filters: Option<&HashMap<String, String>>,
        limit: usize,
    ) -> Vec<GraphNode> {
        self.inner.query_nodes(node_type, filters, limit)
    }

    fn search_nodes(
        &self,
        node_type: &str,
        text_fields: &[String],
        query: &str,
        filters: Option<&HashMap<String, String>>,
        limit: usize,
    ) -> Vec<GraphNode> {
        self.inner
            .search_nodes(node_type, text_fields, query, filters, limit)
    }

    fn update_node(&mut self, node_id: &str, properties: HashMap<String, String>) -> bool {
        self.inner.update_node(node_id, properties)
    }

    fn delete_node(&mut self, node_id: &str) -> bool {
        self.inner.delete_node(node_id)
    }

    fn add_edge(
        &mut self,
        source_id: &str,
        target_id: &str,
        edge_type: &str,
        properties: Option<HashMap<String, String>>,
    ) -> crate::Result<GraphEdge> {
        self.inner
            .add_edge(source_id, target_id, edge_type, properties)
    }

    fn query_neighbors(
        &self,
        node_id: &str,
        edge_type: Option<&str>,
        direction: Direction,
        limit: usize,
    ) -> Vec<(GraphEdge, GraphNode)> {
        self.inner
            .query_neighbors(node_id, edge_type, direction, limit)
    }

    fn delete_edge(&mut self, source_id: &str, target_id: &str, edge_type: &str) -> bool {
        self.inner.delete_edge(source_id, target_id, edge_type)
    }

    fn traverse(
        &self,
        start_id: &str,
        edge_types: Option<&[String]>,
        max_hops: usize,
        direction: Direction,
        node_filter: Option<&HashMap<String, String>>,
    ) -> TraversalResult {
        self.inner
            .traverse(start_id, edge_types, max_hops, direction, node_filter)
    }

    fn close(&mut self) {
        self.inner.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_trust() {
        let mut hive = HiveGraphStore::new(None);
        hive.register_agent("agent-1", "biology", 0.9).unwrap();
        assert!((hive.get_agent_trust("agent-1") - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_update_trust() {
        let mut hive = HiveGraphStore::new(None);
        hive.register_agent("agent-1", "math", 0.5).unwrap();
        hive.update_trust("agent-1", 0.8).unwrap();
        assert!((hive.get_agent_trust("agent-1") - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_expert_agents() {
        let mut hive = HiveGraphStore::new(None);
        hive.register_agent("a1", "biology", 0.9).unwrap();
        hive.register_agent("a2", "biology", 0.7).unwrap();
        hive.register_agent("a3", "math", 0.8).unwrap();

        let experts = hive.get_expert_agents("biology");
        assert_eq!(experts.len(), 2);
        // Should be sorted by trust descending
        let t0: f64 = experts[0].properties.get("trust").unwrap().parse().unwrap();
        let t1: f64 = experts[1].properties.get("trust").unwrap().parse().unwrap();
        assert!(t0 >= t1);
    }

    #[test]
    fn test_confirmation_count() {
        let mut hive = HiveGraphStore::new(None);
        hive.register_agent("a1", "bio", 1.0).unwrap();
        hive.register_agent("a2", "bio", 1.0).unwrap();
        hive.add_confirmation("a2", "a1", 0.9).unwrap();
        assert_eq!(hive.get_confirmation_count("a1"), 1);
    }

    #[test]
    fn test_increment_fact_count() {
        let mut hive = HiveGraphStore::new(None);
        hive.register_agent("agent-x", "math", 0.8).unwrap();

        let node_before = hive.inner.get_node("agent-x").unwrap();
        let count_before: i64 = node_before
            .properties
            .get("fact_count")
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(count_before, 0);

        hive.increment_fact_count("agent-x");
        hive.increment_fact_count("agent-x");

        let node_after = hive.inner.get_node("agent-x").unwrap();
        let count_after: i64 = node_after
            .properties
            .get("fact_count")
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(count_after, 2);
    }

    #[test]
    fn test_add_contradiction() {
        let mut hive = HiveGraphStore::new(None);
        hive.register_agent("a1", "biology", 0.9).unwrap();
        hive.register_agent("a2", "biology", 0.8).unwrap();

        hive.add_contradiction("a1", "a2", "conflicting findings")
            .unwrap();

        let neighbors =
            hive.inner
                .query_neighbors("a1", Some("CONTRADICTS"), Direction::Outgoing, 10);
        assert_eq!(neighbors.len(), 1);
        let (edge, target) = &neighbors[0];
        assert_eq!(edge.edge_type, "CONTRADICTS");
        assert_eq!(target.node_id, "a2");
        assert_eq!(
            edge.properties.get("description").unwrap(),
            "conflicting findings"
        );
        assert_eq!(edge.properties.get("resolved").unwrap(), "false");
    }

    #[test]
    fn test_add_semantic_bridge() {
        let mut hive = HiveGraphStore::new(None);
        hive.register_agent("a1", "nlp", 0.9).unwrap();
        hive.register_agent("a2", "linguistics", 0.8).unwrap();

        hive.add_semantic_bridge("a1", "a2", 0.75).unwrap();

        let neighbors =
            hive.inner
                .query_neighbors("a1", Some("SAME_CONCEPT"), Direction::Outgoing, 10);
        assert_eq!(neighbors.len(), 1);
        let (edge, target) = &neighbors[0];
        assert_eq!(edge.edge_type, "SAME_CONCEPT");
        assert_eq!(target.node_id, "a2");
        assert!(
            (edge
                .properties
                .get("similarity")
                .unwrap()
                .parse::<f64>()
                .unwrap()
                - 0.75)
                .abs()
                < 0.01
        );
    }
}
