//! Hierarchical memory system using in-memory graph store.
//!
//! Port of the Python `HierarchicalMemory` class. Uses `InMemoryGraphStore`
//! for the graph backend instead of Kuzu.
//!
//! Features:
//! - Five memory categories matching cognitive science model
//! - Auto-classification of incoming knowledge
//! - Similarity edges computed on store for Graph RAG traversal
//! - Contradiction detection and SUPERSEDES edge creation
//! - Entity-centric and concept-based retrieval
//!
//! Public API:
//!     KnowledgeNode, KnowledgeEdge, KnowledgeSubgraph, MemoryClassifier, HierarchicalMemory

mod consolidation;
mod experience_ops;
mod helpers;
mod pattern_ops;
mod types;

#[cfg(test)]
mod tests;

pub use types::{KnowledgeEdge, KnowledgeNode, KnowledgeSubgraph, MemoryClassifier};

use std::collections::{HashMap, HashSet};

use crate::entity_extraction::MULTI_WORD_NAME_RE;
use crate::graph::{Direction, GraphStore, InMemoryGraphStore};

use helpers::{graph_node_to_knowledge_node, parse_edge_metadata, rank_and_truncate, STORE_ID_RE};

const DEFAULT_SIMILARITY_THRESHOLD: f64 = 0.3;
/// Similarity score above which contradiction detection is triggered.
const CONTRADICTION_SIMILARITY_THRESHOLD: f64 = 0.5;
/// Decay factor applied to confidence when a node is superseded.
const SUPERSEDE_CONFIDENCE_DECAY: f64 = 0.5;
/// Minimum confidence after decay.
const MIN_CONFIDENCE: f64 = 0.1;

// ---------------------------------------------------------------------------
// HierarchicalMemory
// ---------------------------------------------------------------------------

/// Hierarchical memory system backed by an in-memory graph store.
///
/// Creates and manages a knowledge graph with:
/// - SemanticMemory nodes for factual knowledge
/// - SIMILAR_TO edges computed via text similarity
/// - SUPERSEDES edges for temporal contradiction tracking
pub struct HierarchicalMemory {
    pub agent_name: String,
    store: InMemoryGraphStore,
    classifier: MemoryClassifier,
    /// Running count of SIMILAR_TO edges created (avoids O(n×E) in get_statistics).
    similarity_edge_count: usize,
    /// Running count of SUPERSEDES edges created (avoids O(n×E) in get_statistics).
    supersedes_edge_count: usize,
}

impl HierarchicalMemory {
    /// Create a new hierarchical memory for the given agent.
    ///
    /// # Errors
    /// Returns error if `agent_name` is empty or invalid.
    pub fn new(agent_name: &str) -> crate::Result<Self> {
        let cleaned = agent_name.trim();
        if cleaned.is_empty() {
            return Err(crate::MemoryError::InvalidInput(
                "agent_name cannot be empty".into(),
            ));
        }

        let valid = &*STORE_ID_RE;
        if !valid.is_match(cleaned) {
            return Err(crate::MemoryError::InvalidInput(format!(
                "agent_name must be alphanumeric with hyphens/underscores, 1-64 chars, got: {cleaned:?}"
            )));
        }

        Ok(Self {
            agent_name: cleaned.to_string(),
            store: InMemoryGraphStore::new(Some(cleaned)),
            classifier: MemoryClassifier::new(),
            similarity_edge_count: 0,
            supersedes_edge_count: 0,
        })
    }

    /// Retrieve a knowledge subgraph relevant to a query via Graph RAG.
    ///
    /// 1. Keyword search for seed nodes
    /// 2. Expand via SIMILAR_TO edges (up to `max_depth` hops)
    /// 3. Rank by confidence × keyword relevance
    pub fn retrieve_subgraph(
        &self,
        query: &str,
        max_depth: usize,
        max_nodes: usize,
    ) -> KnowledgeSubgraph {
        let query = query.trim();
        if query.is_empty() {
            return KnowledgeSubgraph {
                query: query.to_string(),
                ..Default::default()
            };
        }

        let keywords: Vec<String> = query
            .to_lowercase()
            .split_whitespace()
            .map(String::from)
            .collect();

        let seed_ids = self.find_seed_nodes(&keywords, query, max_nodes);
        let (expanded_nodes, edges) = self.expand_neighbors(&seed_ids, max_depth, max_nodes);

        let mut all_nodes: Vec<KnowledgeNode> = seed_ids;
        all_nodes.extend(expanded_nodes);
        let mut all_nodes = rank_and_truncate(all_nodes, &keywords, max_nodes);

        self.mark_superseded(&mut all_nodes);

        KnowledgeSubgraph {
            query: query.to_string(),
            nodes: all_nodes,
            edges,
        }
    }

    /// Find seed nodes by entity search and keyword search.
    fn find_seed_nodes(
        &self,
        keywords: &[String],
        entity: &str,
        max_nodes: usize,
    ) -> Vec<KnowledgeNode> {
        let keyword_limit = std::cmp::max(max_nodes * 3, 60);
        let mut seen_ids: HashSet<String> = HashSet::new();
        let mut seed_nodes: Vec<KnowledgeNode> = Vec::new();

        for node in self.entity_seed_search(entity, keyword_limit) {
            if seen_ids.insert(node.node_id.clone()) {
                seed_nodes.push(node);
            }
        }

        let agent_filter = self.agent_filter();
        for keyword in keywords {
            if keyword.len() <= 2 {
                continue;
            }
            let results = self.store.search_nodes(
                "SemanticMemory",
                &["content".to_string(), "concept".to_string()],
                keyword,
                Some(&agent_filter),
                keyword_limit,
            );
            for gn in results {
                if seen_ids.insert(gn.node_id.clone()) {
                    seed_nodes.push(graph_node_to_knowledge_node(&gn));
                }
            }
        }

        seed_nodes
    }

    /// Expand seed nodes via SIMILAR_TO edges (1-hop and optional 2-hop).
    fn expand_neighbors(
        &self,
        seed_nodes: &[KnowledgeNode],
        max_depth: usize,
        max_nodes: usize,
    ) -> (Vec<KnowledgeNode>, Vec<KnowledgeEdge>) {
        let mut seen_ids: HashSet<String> = seed_nodes.iter().map(|n| n.node_id.clone()).collect();
        let mut expanded_nodes: Vec<KnowledgeNode> = Vec::new();
        let mut edges: Vec<KnowledgeEdge> = Vec::new();

        for seed in seed_nodes {
            if seen_ids.len() >= max_nodes {
                break;
            }

            let neighbors = self.store.query_neighbors(
                &seed.node_id,
                Some("SIMILAR_TO"),
                Direction::Outgoing,
                50,
            );

            for (ge, gn) in &neighbors {
                let weight: f64 = ge
                    .properties
                    .get("weight")
                    .and_then(|w| w.parse().ok())
                    .unwrap_or(1.0);

                if seen_ids.len() < max_nodes
                    && seen_ids.insert(gn.node_id.clone())
                    && gn.properties.get("agent_id") == Some(&self.agent_name)
                {
                    expanded_nodes.push(graph_node_to_knowledge_node(gn));
                }

                edges.push(KnowledgeEdge {
                    source_id: seed.node_id.clone(),
                    target_id: gn.node_id.clone(),
                    relationship: "SIMILAR_TO".to_string(),
                    weight,
                    metadata: parse_edge_metadata(ge),
                });
            }

            if max_depth >= 2 {
                for (_, hop1_node) in &neighbors {
                    if seen_ids.len() >= max_nodes {
                        break;
                    }
                    let hop2 = self.store.query_neighbors(
                        &hop1_node.node_id,
                        Some("SIMILAR_TO"),
                        Direction::Outgoing,
                        50,
                    );
                    for (ge2, gn2) in &hop2 {
                        if gn2.node_id == seed.node_id {
                            continue;
                        }
                        let weight: f64 = ge2
                            .properties
                            .get("weight")
                            .and_then(|w| w.parse().ok())
                            .unwrap_or(1.0);

                        if seen_ids.len() < max_nodes
                            && seen_ids.insert(gn2.node_id.clone())
                            && gn2.properties.get("agent_id") == Some(&self.agent_name)
                        {
                            expanded_nodes.push(graph_node_to_knowledge_node(gn2));
                        }

                        edges.push(KnowledgeEdge {
                            source_id: hop1_node.node_id.clone(),
                            target_id: gn2.node_id.clone(),
                            relationship: "SIMILAR_TO".to_string(),
                            weight,
                            metadata: parse_edge_metadata(ge2),
                        });
                    }
                }
            }
        }

        (expanded_nodes, edges)
    }

    /// Search for nodes matching a concept keyword.
    pub fn search_by_concept(&self, keywords: &[String], limit: usize) -> Vec<KnowledgeNode> {
        if keywords.is_empty() {
            return Vec::new();
        }

        let mut nodes = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        let agent_filter = self.agent_filter();

        for kw in keywords {
            let kw_lower = kw.trim().to_lowercase();
            if kw_lower.len() <= 2 {
                continue;
            }
            let results = self.store.search_nodes(
                "SemanticMemory",
                &["concept".to_string(), "content".to_string()],
                &kw_lower,
                Some(&agent_filter),
                limit,
            );
            for gn in results {
                if seen.insert(gn.node_id.clone()) {
                    nodes.push(graph_node_to_knowledge_node(&gn));
                }
            }
        }
        nodes
    }

    /// Entity-centric retrieval. Searches by `entity_name` property,
    /// falling back to content/concept text search.
    pub fn search_by_entity(&self, entity_name: &str, limit: usize) -> Vec<KnowledgeNode> {
        let entity_name = entity_name.trim();
        if entity_name.is_empty() {
            return Vec::new();
        }

        let entity_lower = entity_name.to_lowercase();
        let agent_filter = self.agent_filter();

        // Primary: search entity_name field
        let results = self.store.search_nodes(
            "SemanticMemory",
            &["entity_name".to_string()],
            &entity_lower,
            Some(&agent_filter),
            limit,
        );

        if !results.is_empty() {
            return results.iter().map(graph_node_to_knowledge_node).collect();
        }

        // Fallback: search content/concept
        let results = self.store.search_nodes(
            "SemanticMemory",
            &["content".to_string(), "concept".to_string()],
            &entity_lower,
            Some(&agent_filter),
            limit,
        );
        results.iter().map(graph_node_to_knowledge_node).collect()
    }

    /// Release resources held by the memory store.
    pub fn close(&mut self) {
        self.store.close();
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn agent_filter(&self) -> HashMap<String, String> {
        let mut f = HashMap::new();
        f.insert("agent_id".into(), self.agent_name.clone());
        f
    }

    fn entity_seed_search(&self, query: &str, limit: usize) -> Vec<KnowledgeNode> {
        let mut nodes = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        let agent_filter = self.agent_filter();

        // Extract entity candidates from query
        let mut candidates: Vec<String> = MULTI_WORD_NAME_RE
            .find_iter(query)
            .map(|m| m.as_str().to_string())
            .collect();

        if candidates.is_empty() {
            // Fallback: use words > 3 chars
            candidates = query
                .to_lowercase()
                .split_whitespace()
                .filter(|w| w.len() > 3)
                .map(|w| w.trim_end_matches("'s").to_string())
                .collect();
        }

        for candidate in &candidates {
            let candidate_lower = candidate.to_lowercase();
            if candidate_lower.len() <= 2 {
                continue;
            }
            let results = self.store.search_nodes(
                "SemanticMemory",
                &["entity_name".to_string()],
                &candidate_lower,
                Some(&agent_filter),
                limit,
            );
            for gn in results {
                if seen.insert(gn.node_id.clone()) {
                    nodes.push(graph_node_to_knowledge_node(&gn));
                }
            }
        }

        nodes.truncate(limit);
        nodes
    }
}
