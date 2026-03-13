//! Pattern evolution and temporal tracking operations.

use std::collections::HashSet;

use crate::graph::{Direction, GraphStore};

use super::helpers::graph_node_to_knowledge_node;
use super::types::KnowledgeNode;
use super::HierarchicalMemory;

impl HierarchicalMemory {
    /// Follow SUPERSEDES edges to build a temporal chain for a concept.
    ///
    /// Returns nodes ordered from newest to oldest showing how knowledge
    /// about the concept evolved over time.
    pub fn get_temporal_chain(&self, concept: &str) -> Vec<KnowledgeNode> {
        if concept.is_empty() {
            return Vec::new();
        }

        // Find all nodes matching the concept
        let agent_filter = self.agent_filter();
        let matching = self.store.search_nodes(
            "SemanticMemory",
            &["concept".to_string(), "content".to_string()],
            &concept.to_lowercase(),
            Some(&agent_filter),
            100,
        );

        if matching.is_empty() {
            return Vec::new();
        }

        // Build chain by following SUPERSEDES edges
        let mut chain: Vec<KnowledgeNode> = Vec::new();
        let mut visited: HashSet<String> = HashSet::new();

        // Find the "newest" node (one that isn't superseded by anything)
        let mut head_ids: Vec<String> = Vec::new();
        for node in &matching {
            let incoming = self.store.query_neighbors(
                &node.node_id,
                Some("SUPERSEDES"),
                Direction::Incoming,
                1,
            );
            if incoming.is_empty() {
                head_ids.push(node.node_id.clone());
            }
        }

        // If no head found, use all matching nodes
        if head_ids.is_empty() {
            head_ids = matching.into_iter().map(|n| n.node_id).collect();
        }

        // Walk from each head through SUPERSEDES chain
        for head_id in &head_ids {
            let mut current_id = head_id.clone();
            while !visited.contains(&current_id) {
                visited.insert(current_id.clone());
                if let Some(gn) = self.store.get_node(&current_id) {
                    chain.push(graph_node_to_knowledge_node(&gn));
                }
                // Follow outgoing SUPERSEDES to older node
                let next = self.store.query_neighbors(
                    &current_id,
                    Some("SUPERSEDES"),
                    Direction::Outgoing,
                    1,
                );
                if let Some((_, neighbor)) = next.first() {
                    current_id = neighbor.node_id.clone();
                } else {
                    break;
                }
            }
        }

        chain
    }

    /// Track how knowledge about a concept evolved over time.
    ///
    /// Similar to `get_temporal_chain` but includes SUPERSEDES edge metadata
    /// showing what changed between versions.
    pub fn get_knowledge_evolution(&self, concept: &str) -> Vec<(KnowledgeNode, Option<String>)> {
        let chain = self.get_temporal_chain(concept);
        if chain.len() <= 1 {
            return chain.into_iter().map(|n| (n, None)).collect();
        }

        let mut evolution: Vec<(KnowledgeNode, Option<String>)> = Vec::new();

        for (i, node) in chain.iter().enumerate() {
            if i + 1 < chain.len() {
                // Check for SUPERSEDES edge to next (older) node
                let neighbors = self.store.query_neighbors(
                    &node.node_id,
                    Some("SUPERSEDES"),
                    Direction::Outgoing,
                    10,
                );
                let reason = neighbors.iter().find_map(|(edge, neighbor)| {
                    if neighbor.node_id == chain[i + 1].node_id {
                        edge.properties.get("reason").cloned()
                    } else {
                        None
                    }
                });
                evolution.push((node.clone(), reason));
            } else {
                evolution.push((node.clone(), None));
            }
        }

        evolution
    }
}
