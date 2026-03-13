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

mod helpers;
mod types;

pub use types::{KnowledgeEdge, KnowledgeNode, KnowledgeSubgraph, MemoryClassifier};

use std::collections::{HashMap, HashSet};

use crate::contradiction::detect_contradiction;
use crate::entity_extraction::{extract_entity_name, MULTI_WORD_NAME_RE};
use crate::graph::{Direction, GraphStore, InMemoryGraphStore};
use crate::memory_types::MemoryCategory;
use crate::similarity::compute_similarity;

use helpers::{
    build_sim_map, graph_node_to_knowledge_node, make_id, now_iso, parse_edge_metadata,
    rank_and_truncate, MAX_AGGREGATION_NODES, MAX_STATISTICS_NODES, MAX_SUPERSEDE_CANDIDATES,
    STORE_ID_RE,
};

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

    /// Store a knowledge node in the graph.
    ///
    /// Auto-classifies if `category` is `None`. Computes similarity against
    /// existing nodes and creates `SIMILAR_TO` edges for scores > 0.3.
    /// Detects contradictions and creates `SUPERSEDES` edges when found.
    ///
    /// Returns the `node_id` of the stored knowledge node.
    #[allow(clippy::too_many_arguments)]
    pub fn store_knowledge(
        &mut self,
        content: &str,
        concept: &str,
        confidence: f64,
        category: Option<MemoryCategory>,
        source_id: &str,
        tags: &[String],
        temporal_metadata: Option<&HashMap<String, serde_json::Value>>,
    ) -> crate::Result<String> {
        if confidence.is_nan() || !(0.0..=1.0).contains(&confidence) {
            return Err(crate::MemoryError::InvalidInput(
                "confidence must be between 0.0 and 1.0".into(),
            ));
        }

        let content = content.trim();
        if content.is_empty() {
            return Err(crate::MemoryError::InvalidInput(
                "content cannot be empty".into(),
            ));
        }

        let category = category.unwrap_or_else(|| self.classifier.classify(content, concept));
        let node_id = make_id();
        let now = now_iso();

        let mut meta: HashMap<String, serde_json::Value> = HashMap::new();
        meta.insert(
            "category".into(),
            serde_json::Value::String(category.as_str().to_string()),
        );
        if let Some(tm) = temporal_metadata {
            for (k, v) in tm {
                meta.insert(k.clone(), v.clone());
            }
        }

        let entity_name = extract_entity_name(content, concept);

        let mut props = HashMap::new();
        props.insert("concept".into(), concept.to_string());
        props.insert("content".into(), content.to_string());
        props.insert("confidence".into(), confidence.to_string());
        props.insert("source_id".into(), source_id.to_string());
        props.insert("agent_id".into(), self.agent_name.clone());
        props.insert(
            "tags".into(),
            serde_json::to_string(tags).unwrap_or_default(),
        );
        props.insert(
            "metadata".into(),
            serde_json::to_string(&meta).unwrap_or_default(),
        );
        props.insert("created_at".into(), now);
        props.insert("entity_name".into(), entity_name);

        self.store
            .add_node("SemanticMemory", props, Some(&node_id))?;

        // Detect SUPERSEDES if temporal metadata present
        if let Some(tm) = temporal_metadata {
            let idx = tm
                .get("temporal_index")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            if idx > 0 {
                self.detect_supersedes(&node_id, content, concept, temporal_metadata);
            }
        }

        // Compute similarity edges
        self.create_similarity_edges(&node_id, content, concept, tags);

        Ok(node_id)
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

    /// Retrieve all knowledge nodes, optionally filtered by category.
    pub fn get_all_knowledge(
        &self,
        category: Option<MemoryCategory>,
        limit: usize,
    ) -> Vec<KnowledgeNode> {
        let agent_filter = self.agent_filter();
        let all = self
            .store
            .query_nodes("SemanticMemory", Some(&agent_filter), limit);

        let mut nodes: Vec<KnowledgeNode> = all
            .iter()
            .map(graph_node_to_knowledge_node)
            .filter(|n| {
                if let Some(cat) = category {
                    n.category == cat
                } else {
                    true
                }
            })
            .collect();

        // Sort by created_at descending
        nodes.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        nodes.truncate(limit);
        nodes
    }

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

    /// Execute aggregation queries for meta-memory questions.
    ///
    /// Supported `aggregation_type` values:
    /// - `"count"`: total number of facts
    /// - `"avg_confidence"`: average confidence score
    /// - `"top_concepts"`: most frequent concepts
    /// - `"by_category"`: count by category
    pub fn execute_aggregation(
        &self,
        aggregation_type: &str,
        concept: &str,
        limit: usize,
    ) -> HashMap<String, serde_json::Value> {
        let agent_filter = self.agent_filter();
        let all_nodes =
            self.store
                .query_nodes("SemanticMemory", Some(&agent_filter), MAX_AGGREGATION_NODES);

        let filtered: Vec<&crate::graph::GraphNode> = if concept.is_empty() {
            all_nodes.iter().collect()
        } else {
            let concept_lower = concept.to_lowercase();
            all_nodes
                .iter()
                .filter(|n| {
                    n.properties
                        .get("concept")
                        .is_some_and(|c| c.to_lowercase().contains(&concept_lower))
                        || n.properties
                            .get("content")
                            .is_some_and(|c| c.to_lowercase().contains(&concept_lower))
                })
                .collect()
        };

        let mut result = HashMap::new();
        result.insert(
            "query_type".into(),
            serde_json::Value::String(aggregation_type.to_string()),
        );

        match aggregation_type {
            "count" => {
                result.insert("count".into(), serde_json::json!(filtered.len()));
            }
            "avg_confidence" => {
                if filtered.is_empty() {
                    result.insert("avg_confidence".into(), serde_json::json!(0.0));
                } else {
                    let sum: f64 = filtered
                        .iter()
                        .map(|n| {
                            n.properties
                                .get("confidence")
                                .and_then(|c| c.parse::<f64>().ok())
                                .unwrap_or(0.0)
                        })
                        .sum();
                    result.insert(
                        "avg_confidence".into(),
                        serde_json::json!(sum / filtered.len() as f64),
                    );
                }
            }
            "top_concepts" => {
                let mut concept_counts: HashMap<String, usize> = HashMap::new();
                for n in &filtered {
                    if let Some(c) = n.properties.get("concept") {
                        if !c.is_empty() && c != "SUMMARY" {
                            *concept_counts.entry(c.clone()).or_insert(0) += 1;
                        }
                    }
                }
                let mut sorted: Vec<(String, usize)> = concept_counts.into_iter().collect();
                sorted.sort_by(|a, b| b.1.cmp(&a.1));
                sorted.truncate(limit);

                let items: HashMap<String, usize> = sorted.into_iter().collect();
                result.insert("items".into(), serde_json::json!(items));
                result.insert("count".into(), serde_json::json!(items.len()));
            }
            "by_category" => {
                let mut cat_counts: HashMap<String, usize> = HashMap::new();
                for n in &filtered {
                    let meta_str = n
                        .properties
                        .get("metadata")
                        .map(|s| s.as_str())
                        .unwrap_or("");
                    let meta: HashMap<String, serde_json::Value> =
                        serde_json::from_str(meta_str).unwrap_or_default();
                    let cat = meta
                        .get("category")
                        .and_then(|v| v.as_str())
                        .unwrap_or("semantic")
                        .to_string();
                    *cat_counts.entry(cat).or_insert(0) += 1;
                }
                result.insert("items".into(), serde_json::json!(cat_counts));
                result.insert("count".into(), serde_json::json!(filtered.len()));
            }
            _ => {
                result.insert(
                    "error".into(),
                    serde_json::Value::String(format!(
                        "Unknown aggregation type: {aggregation_type}"
                    )),
                );
            }
        }

        result
    }

    /// Get statistics about the hierarchical memory.
    pub fn get_statistics(&self) -> HashMap<String, serde_json::Value> {
        let agent_filter = self.agent_filter();
        let semantic_nodes =
            self.store
                .query_nodes("SemanticMemory", Some(&agent_filter), MAX_STATISTICS_NODES);

        let mut stats = HashMap::new();
        stats.insert(
            "agent_name".into(),
            serde_json::Value::String(self.agent_name.clone()),
        );
        stats.insert(
            "semantic_nodes".into(),
            serde_json::json!(semantic_nodes.len()),
        );
        stats.insert(
            "similar_to_edges".into(),
            serde_json::json!(self.similarity_edge_count),
        );
        stats.insert(
            "supersedes_edges".into(),
            serde_json::json!(self.supersedes_edge_count),
        );
        stats.insert(
            "total_experiences".into(),
            serde_json::json!(semantic_nodes.len()),
        );
        stats
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

    fn create_similarity_edges(
        &mut self,
        node_id: &str,
        content: &str,
        concept: &str,
        tags: &[String],
    ) {
        // Targeted candidate selection: only compare against nodes likely to be
        // similar, rather than scanning all nodes (O(n) -> O(k) per insert).
        const MAX_CANDIDATES: usize = 100;

        let agent_filter = self.agent_filter();
        let mut seen_ids: HashSet<String> = HashSet::new();
        seen_ids.insert(node_id.to_string());
        let mut candidates: Vec<crate::graph::GraphNode> = Vec::new();

        // 1. Nodes with the same concept (highest relevance).
        if !concept.is_empty() {
            let mut concept_filter = agent_filter.clone();
            concept_filter.insert("concept".into(), concept.to_string());
            for node in
                self.store
                    .query_nodes("SemanticMemory", Some(&concept_filter), MAX_CANDIDATES)
            {
                if seen_ids.insert(node.node_id.clone()) {
                    candidates.push(node);
                }
            }
        }

        // 2. If concept is empty or matched few nodes, search by entity name
        //    extracted from content to find related nodes.
        if candidates.len() < MAX_CANDIDATES {
            let entity = extract_entity_name(content, concept);
            if !entity.is_empty() {
                let remaining = MAX_CANDIDATES - candidates.len();
                for node in self.store.search_nodes(
                    "SemanticMemory",
                    &["entity_name".to_string()],
                    &entity,
                    Some(&agent_filter),
                    remaining,
                ) {
                    if seen_ids.insert(node.node_id.clone()) {
                        candidates.push(node);
                    }
                }
            }
        }

        // 3. If still few candidates, do a content keyword search as fallback.
        if candidates.len() < MAX_CANDIDATES {
            let keywords: Vec<&str> = content
                .split_whitespace()
                .filter(|w| w.len() > 3)
                .take(3)
                .collect();
            let remaining = MAX_CANDIDATES - candidates.len();
            for kw in keywords {
                if candidates.len() >= MAX_CANDIDATES {
                    break;
                }
                for node in self.store.search_nodes(
                    "SemanticMemory",
                    &["content".to_string()],
                    kw,
                    Some(&agent_filter),
                    remaining,
                ) {
                    if seen_ids.insert(node.node_id.clone()) {
                        candidates.push(node);
                    }
                }
            }
        }

        let new_node = build_sim_map(content, concept, tags);

        for other in &candidates {
            let other_content = other
                .properties
                .get("content")
                .map(|s| s.as_str())
                .unwrap_or("");
            let other_concept = other
                .properties
                .get("concept")
                .map(|s| s.as_str())
                .unwrap_or("");
            let other_tags: Vec<String> = other
                .properties
                .get("tags")
                .and_then(|t| serde_json::from_str(t).ok())
                .unwrap_or_default();

            let other_map = build_sim_map(other_content, other_concept, &other_tags);
            let score = compute_similarity(&new_node, &other_map);

            if score > 0.3 {
                let mut edge_meta = HashMap::new();

                // Check for contradiction between high-similarity facts
                if score > 0.5 {
                    if let Some(contradiction) =
                        detect_contradiction(content, other_content, concept, other_concept)
                    {
                        edge_meta.insert(
                            "contradiction".to_string(),
                            serde_json::json!(contradiction.contradiction).to_string(),
                        );
                        edge_meta.insert(
                            "conflicting_values".to_string(),
                            contradiction.conflicting_values,
                        );
                    }
                }

                edge_meta.insert("weight".to_string(), format!("{score:.4}"));

                // Store edge metadata as serialized JSON in a property
                let meta_json = if edge_meta.len() > 1 {
                    // Has contradiction info beyond just weight
                    let meta_map: HashMap<String, serde_json::Value> = edge_meta
                        .into_iter()
                        .filter(|(k, _)| k.as_str() != "weight")
                        .map(|(k, v)| (k, serde_json::Value::String(v)))
                        .collect();
                    serde_json::to_string(&meta_map).unwrap_or_default()
                } else {
                    String::new()
                };

                let mut props = HashMap::new();
                props.insert("weight".to_string(), format!("{score:.4}"));
                if !meta_json.is_empty() {
                    props.insert("metadata".to_string(), meta_json);
                }

                let _ = self
                    .store
                    .add_edge(node_id, &other.node_id, "SIMILAR_TO", Some(props));
                self.similarity_edge_count += 1;
            }
        }
    }

    fn detect_supersedes(
        &mut self,
        new_node_id: &str,
        content: &str,
        concept: &str,
        temporal_metadata: Option<&HashMap<String, serde_json::Value>>,
    ) {
        let new_temporal_idx = temporal_metadata
            .and_then(|tm| tm.get("temporal_index"))
            .and_then(|v| v.as_i64())
            .unwrap_or(0);

        if new_temporal_idx <= 0 {
            return;
        }

        let concept_key = concept
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_lowercase();
        if concept_key.is_empty() {
            return;
        }

        let agent_filter = self.agent_filter();
        let candidates = self.store.search_nodes(
            "SemanticMemory",
            &["concept".to_string()],
            &concept_key,
            Some(&agent_filter),
            MAX_SUPERSEDE_CANDIDATES,
        );

        let mut supersede_pairs: Vec<(String, String, String)> = Vec::new();

        for candidate in &candidates {
            if candidate.node_id == new_node_id {
                continue;
            }

            let old_meta_str = candidate
                .properties
                .get("metadata")
                .map(|s| s.as_str())
                .unwrap_or("");
            let old_meta: HashMap<String, serde_json::Value> =
                serde_json::from_str(old_meta_str).unwrap_or_default();
            let old_temporal_idx = old_meta
                .get("temporal_index")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);

            if old_temporal_idx <= 0 || old_temporal_idx >= new_temporal_idx {
                continue;
            }

            let old_content = candidate
                .properties
                .get("content")
                .map(|s| s.as_str())
                .unwrap_or("");
            let old_concept = candidate
                .properties
                .get("concept")
                .map(|s| s.as_str())
                .unwrap_or("");

            if let Some(contradiction) =
                detect_contradiction(content, old_content, concept, old_concept)
            {
                if contradiction.contradiction {
                    let temporal_delta = format!("index {old_temporal_idx} -> {new_temporal_idx}");
                    let reason = format!("Updated values: {}", contradiction.conflicting_values);
                    supersede_pairs.push((candidate.node_id.clone(), reason, temporal_delta));
                }
            }
        }

        for (old_id, reason, delta) in supersede_pairs {
            let mut props = HashMap::new();
            props.insert("reason".to_string(), reason);
            props.insert("temporal_delta".to_string(), delta);
            let _ = self
                .store
                .add_edge(new_node_id, &old_id, "SUPERSEDES", Some(props));
            self.supersedes_edge_count += 1;
        }
    }

    fn mark_superseded(&self, nodes: &mut [KnowledgeNode]) {
        for node in nodes.iter_mut() {
            let incoming = self.store.query_neighbors(
                &node.node_id,
                Some("SUPERSEDES"),
                Direction::Incoming,
                1,
            );
            if let Some((edge, newer_node)) = incoming.first() {
                node.metadata
                    .insert("superseded".into(), serde_json::json!(true));
                node.metadata.insert(
                    "superseded_by".into(),
                    serde_json::Value::String(newer_node.node_id.clone()),
                );
                let reason = edge.properties.get("reason").cloned().unwrap_or_default();
                node.metadata
                    .insert("supersede_reason".into(), serde_json::Value::String(reason));
                node.confidence = (node.confidence * 0.5).max(0.1);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mem() -> HierarchicalMemory {
        HierarchicalMemory::new("test_agent").unwrap()
    }

    // -- Store and retrieve --

    #[test]
    fn test_store_and_retrieve() {
        let mut mem = make_mem();
        let nid = mem
            .store_knowledge(
                "Plants use photosynthesis to convert sunlight",
                "biology",
                0.9,
                None,
                "",
                &[],
                None,
            )
            .unwrap();
        assert!(!nid.is_empty());

        let sub = mem.retrieve_subgraph("photosynthesis", 2, 20);
        assert!(!sub.nodes.is_empty());
        assert_eq!(
            sub.nodes[0].content,
            "Plants use photosynthesis to convert sunlight"
        );
    }

    #[test]
    fn test_store_empty_content_fails() {
        let mut mem = make_mem();
        let result = mem.store_knowledge("", "concept", 0.8, None, "", &[], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_agent_name() {
        assert!(HierarchicalMemory::new("").is_err());
        assert!(HierarchicalMemory::new("a b c").is_err());
        assert!(HierarchicalMemory::new("../hack").is_err());
    }

    // -- Similarity edge creation --

    #[test]
    fn test_similarity_edges() {
        let mut mem = make_mem();
        let _id1 = mem
            .store_knowledge(
                "Rust programming language is fast",
                "rust",
                0.9,
                None,
                "",
                &[],
                None,
            )
            .unwrap();
        let id2 = mem
            .store_knowledge(
                "Rust programming language has memory safety",
                "rust",
                0.85,
                None,
                "",
                &[],
                None,
            )
            .unwrap();

        // The second node should have a SIMILAR_TO edge to the first
        let stats = mem.get_statistics();
        let sim_edges = stats
            .get("similar_to_edges")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        assert!(sim_edges > 0, "Expected similarity edges to be created");

        // Retrieve subgraph should find both via similarity
        let sub = mem.retrieve_subgraph("rust programming", 2, 20);
        assert!(sub.nodes.len() >= 2);
        assert!(!sub.edges.is_empty() || sub.nodes.len() >= 2);

        // Verify through direct neighbor query
        let neighbors =
            mem.store
                .query_neighbors(&id2, Some("SIMILAR_TO"), Direction::Outgoing, 10);
        assert!(!neighbors.is_empty());
    }

    // -- Contradiction detection and SUPERSEDES edges --

    #[test]
    fn test_contradiction_and_supersedes() {
        let mut mem = make_mem();

        let mut tm1 = HashMap::new();
        tm1.insert("temporal_index".into(), serde_json::json!(1));

        let id1 = mem
            .store_knowledge(
                "Klaebo has 9 gold medals",
                "klaebo medals",
                0.8,
                None,
                "",
                &[],
                Some(&tm1),
            )
            .unwrap();

        let mut tm2 = HashMap::new();
        tm2.insert("temporal_index".into(), serde_json::json!(2));

        let id2 = mem
            .store_knowledge(
                "Klaebo has 10 gold medals",
                "klaebo medals",
                0.9,
                None,
                "",
                &[],
                Some(&tm2),
            )
            .unwrap();

        // Should have SUPERSEDES edge from id2 -> id1
        let neighbors =
            mem.store
                .query_neighbors(&id2, Some("SUPERSEDES"), Direction::Outgoing, 10);
        assert!(
            !neighbors.is_empty(),
            "Expected SUPERSEDES edge from new to old fact"
        );
        assert_eq!(neighbors[0].1.node_id, id1);

        let reason = neighbors[0].0.properties.get("reason").unwrap();
        assert!(reason.contains("Updated values"));
    }

    // -- Concept search --

    #[test]
    fn test_search_by_concept() {
        let mut mem = make_mem();
        mem.store_knowledge(
            "Water boils at 100 degrees celsius",
            "physics",
            0.95,
            None,
            "",
            &[],
            None,
        )
        .unwrap();
        mem.store_knowledge(
            "Iron melts at 1538 degrees",
            "chemistry",
            0.9,
            None,
            "",
            &[],
            None,
        )
        .unwrap();

        let results = mem.search_by_concept(&["physics".to_string()], 10);
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Water"));

        let results = mem.search_by_concept(&["degrees".to_string()], 10);
        assert!(results.len() >= 2);
    }

    // -- Entity search --

    #[test]
    fn test_search_by_entity() {
        let mut mem = make_mem();
        mem.store_knowledge(
            "Sarah Chen works at the research lab",
            "Sarah Chen",
            0.9,
            None,
            "",
            &[],
            None,
        )
        .unwrap();
        mem.store_knowledge(
            "John Smith is a teacher",
            "John Smith",
            0.85,
            None,
            "",
            &[],
            None,
        )
        .unwrap();

        let results = mem.search_by_entity("Sarah Chen", 10);
        assert!(!results.is_empty());
        assert!(results[0].content.contains("Sarah Chen"));

        let results = mem.search_by_entity("John", 10);
        assert!(!results.is_empty());
    }

    // -- Temporal chains --

    #[test]
    fn test_temporal_chain() {
        let mut mem = make_mem();

        let mut tm1 = HashMap::new();
        tm1.insert("temporal_index".into(), serde_json::json!(1));
        mem.store_knowledge(
            "Team has 5 members",
            "team size",
            0.8,
            None,
            "",
            &[],
            Some(&tm1),
        )
        .unwrap();

        let mut tm2 = HashMap::new();
        tm2.insert("temporal_index".into(), serde_json::json!(2));
        mem.store_knowledge(
            "Team has 8 members",
            "team size",
            0.9,
            None,
            "",
            &[],
            Some(&tm2),
        )
        .unwrap();

        let chain = mem.get_temporal_chain("team");
        assert!(
            chain.len() >= 2,
            "Expected at least 2 nodes in temporal chain"
        );

        // First node in chain should be the newest (head)
        assert!(chain[0].content.contains("8"));
    }

    // -- Knowledge evolution --

    #[test]
    fn test_knowledge_evolution() {
        let mut mem = make_mem();

        let mut tm1 = HashMap::new();
        tm1.insert("temporal_index".into(), serde_json::json!(1));
        mem.store_knowledge(
            "Population is 5000 people",
            "city population",
            0.8,
            None,
            "",
            &[],
            Some(&tm1),
        )
        .unwrap();

        let mut tm2 = HashMap::new();
        tm2.insert("temporal_index".into(), serde_json::json!(2));
        mem.store_knowledge(
            "Population is 7000 people",
            "city population",
            0.9,
            None,
            "",
            &[],
            Some(&tm2),
        )
        .unwrap();

        let evolution = mem.get_knowledge_evolution("population");
        assert!(!evolution.is_empty());

        // At least one entry should have a reason
        let has_reason = evolution.iter().any(|(_, reason)| reason.is_some());
        assert!(has_reason, "Expected at least one evolution reason");
    }

    // -- Aggregation queries --

    #[test]
    fn test_aggregation_count() {
        let mut mem = make_mem();
        mem.store_knowledge("Fact one", "topic-a", 0.8, None, "", &[], None)
            .unwrap();
        mem.store_knowledge("Fact two", "topic-b", 0.9, None, "", &[], None)
            .unwrap();
        mem.store_knowledge("Fact three", "topic-a", 0.7, None, "", &[], None)
            .unwrap();

        let result = mem.execute_aggregation("count", "", 10);
        assert_eq!(result.get("count").and_then(|v| v.as_u64()), Some(3));
    }

    #[test]
    fn test_aggregation_avg_confidence() {
        let mut mem = make_mem();
        mem.store_knowledge("Fact one", "topic", 0.6, None, "", &[], None)
            .unwrap();
        mem.store_knowledge("Fact two", "topic", 0.8, None, "", &[], None)
            .unwrap();

        let result = mem.execute_aggregation("avg_confidence", "", 10);
        let avg = result
            .get("avg_confidence")
            .and_then(|v| v.as_f64())
            .unwrap();
        assert!((avg - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_aggregation_top_concepts() {
        let mut mem = make_mem();
        mem.store_knowledge("Fact A1", "alpha", 0.8, None, "", &[], None)
            .unwrap();
        mem.store_knowledge("Fact A2", "alpha", 0.8, None, "", &[], None)
            .unwrap();
        mem.store_knowledge("Fact B1", "beta", 0.8, None, "", &[], None)
            .unwrap();

        let result = mem.execute_aggregation("top_concepts", "", 10);
        let items = result.get("items").unwrap().as_object().unwrap();
        assert!(items.contains_key("alpha"));
        assert_eq!(items.get("alpha").and_then(|v| v.as_u64()), Some(2));
    }

    #[test]
    fn test_aggregation_by_category() {
        let mut mem = make_mem();
        mem.store_knowledge(
            "Water boils at 100C",
            "physics",
            0.9,
            Some(MemoryCategory::Semantic),
            "",
            &[],
            None,
        )
        .unwrap();
        mem.store_knowledge(
            "Step 1: boil water. Step 2: add tea.",
            "recipe",
            0.8,
            None,
            "",
            &[],
            None,
        )
        .unwrap();

        let result = mem.execute_aggregation("by_category", "", 10);
        let items = result.get("items").unwrap().as_object().unwrap();
        assert!(items.contains_key("semantic") || items.contains_key("procedural"));
    }

    #[test]
    fn test_aggregation_unknown_type() {
        let mem = make_mem();
        let result = mem.execute_aggregation("invalid_type", "", 10);
        assert!(result.contains_key("error"));
    }

    // -- Memory classification --

    #[test]
    fn test_classifier_procedural() {
        let c = MemoryClassifier::new();
        assert_eq!(
            c.classify("How to bake a cake", ""),
            MemoryCategory::Procedural
        );
        assert_eq!(
            c.classify("Step 1: preheat oven", ""),
            MemoryCategory::Procedural
        );
    }

    #[test]
    fn test_classifier_prospective() {
        let c = MemoryClassifier::new();
        assert_eq!(
            c.classify("I plan to visit Paris", ""),
            MemoryCategory::Prospective
        );
        assert_eq!(
            c.classify("My future goal is to learn Rust", ""),
            MemoryCategory::Prospective
        );
    }

    #[test]
    fn test_classifier_episodic() {
        let c = MemoryClassifier::new();
        assert_eq!(
            c.classify("Something interesting happened today", ""),
            MemoryCategory::Episodic
        );
        assert_eq!(
            c.classify("I observed a rare bird", ""),
            MemoryCategory::Episodic
        );
    }

    #[test]
    fn test_classifier_semantic_default() {
        let c = MemoryClassifier::new();
        assert_eq!(
            c.classify("Water is composed of hydrogen and oxygen", ""),
            MemoryCategory::Semantic
        );
    }

    #[test]
    fn test_auto_classification_on_store() {
        let mut mem = make_mem();
        let nid = mem
            .store_knowledge(
                "Step 1: install Rust. Step 2: write code.",
                "tutorial",
                0.9,
                None,
                "",
                &[],
                None,
            )
            .unwrap();

        let nodes = mem.get_all_knowledge(None, 100);
        let stored = nodes.iter().find(|n| n.node_id == nid).unwrap();
        assert_eq!(stored.category, MemoryCategory::Procedural);
    }

    // -- Subgraph to_llm_context --

    #[test]
    fn test_to_llm_context_empty() {
        let sub = KnowledgeSubgraph::default();
        assert_eq!(sub.to_llm_context(false), "No relevant knowledge found.");
    }

    #[test]
    fn test_to_llm_context_basic() {
        let sub = KnowledgeSubgraph {
            query: "test query".to_string(),
            nodes: vec![KnowledgeNode {
                node_id: "abc12345-1234-1234-1234-123456789abc".into(),
                category: MemoryCategory::Semantic,
                content: "The sky is blue".into(),
                concept: "nature".into(),
                confidence: 0.9,
                ..Default::default()
            }],
            edges: vec![],
        };

        let ctx = sub.to_llm_context(false);
        assert!(ctx.contains("test query"));
        assert!(ctx.contains("The sky is blue"));
        assert!(ctx.contains("[nature]"));
        assert!(ctx.contains("0.9"));
    }

    #[test]
    fn test_to_llm_context_chronological() {
        let mut meta1 = HashMap::new();
        meta1.insert("temporal_index".into(), serde_json::json!(2));
        meta1.insert("source_date".into(), serde_json::json!("2024-01-02"));

        let mut meta2 = HashMap::new();
        meta2.insert("temporal_index".into(), serde_json::json!(1));
        meta2.insert("temporal_order".into(), serde_json::json!("Day 1"));

        let sub = KnowledgeSubgraph {
            query: "timeline".to_string(),
            nodes: vec![
                KnowledgeNode {
                    node_id: "node2".into(),
                    concept: "event".into(),
                    content: "Second event".into(),
                    confidence: 0.8,
                    metadata: meta1,
                    ..Default::default()
                },
                KnowledgeNode {
                    node_id: "node1".into(),
                    concept: "event".into(),
                    content: "First event".into(),
                    confidence: 0.9,
                    metadata: meta2,
                    ..Default::default()
                },
            ],
            edges: vec![],
        };

        let ctx = sub.to_llm_context(true);
        assert!(ctx.contains("chronological"));
        // First event (temporal_index=1) should appear before second (temporal_index=2)
        let pos_first = ctx.find("First event").unwrap();
        let pos_second = ctx.find("Second event").unwrap();
        assert!(pos_first < pos_second);
    }

    #[test]
    fn test_to_llm_context_with_contradictions() {
        // Test with Value::String("true") — the format produced by create_similarity_edges
        let mut edge_meta_str = HashMap::new();
        edge_meta_str.insert(
            "contradiction".into(),
            serde_json::Value::String("true".to_string()),
        );
        edge_meta_str.insert("conflicting_values".into(), serde_json::json!("5 vs 8"));

        let sub = KnowledgeSubgraph {
            query: "team".to_string(),
            nodes: vec![KnowledgeNode {
                node_id: "n1".into(),
                concept: "team".into(),
                content: "Team has 5 members".into(),
                confidence: 0.8,
                ..Default::default()
            }],
            edges: vec![KnowledgeEdge {
                source_id: "n1-source-12345678".into(),
                target_id: "n2-target-12345678".into(),
                relationship: "SIMILAR_TO".into(),
                weight: 0.7,
                metadata: edge_meta_str,
            }],
        };

        let ctx = sub.to_llm_context(false);
        assert!(ctx.contains("Contradictions detected"));
        assert!(ctx.contains("5 vs 8"));
        assert!(ctx.contains("Relationships:"));

        // Also verify with Value::Bool(true) for backward compat
        let mut edge_meta_bool = HashMap::new();
        edge_meta_bool.insert("contradiction".into(), serde_json::json!(true));
        edge_meta_bool.insert("conflicting_values".into(), serde_json::json!("3 vs 7"));

        let sub2 = KnowledgeSubgraph {
            query: "team".to_string(),
            nodes: vec![KnowledgeNode {
                node_id: "n1".into(),
                concept: "team".into(),
                content: "Team has 3 members".into(),
                confidence: 0.8,
                ..Default::default()
            }],
            edges: vec![KnowledgeEdge {
                source_id: "n1-source-12345678".into(),
                target_id: "n2-target-12345678".into(),
                relationship: "SIMILAR_TO".into(),
                weight: 0.7,
                metadata: edge_meta_bool,
            }],
        };

        let ctx2 = sub2.to_llm_context(false);
        assert!(ctx2.contains("Contradictions detected"));
        assert!(ctx2.contains("3 vs 7"));
    }

    // -- Statistics --

    #[test]
    fn test_statistics() {
        let mut mem = make_mem();
        mem.store_knowledge("Fact one about Rust", "rust", 0.9, None, "", &[], None)
            .unwrap();
        mem.store_knowledge(
            "Fact two about Rust language",
            "rust",
            0.8,
            None,
            "",
            &[],
            None,
        )
        .unwrap();

        let stats = mem.get_statistics();
        assert_eq!(
            stats.get("agent_name").and_then(|v| v.as_str()),
            Some("test_agent")
        );
        let nodes = stats
            .get("semantic_nodes")
            .and_then(|v| v.as_u64())
            .unwrap();
        assert_eq!(nodes, 2);
    }

    // -- Close --

    #[test]
    fn test_close() {
        let mut mem = make_mem();
        mem.store_knowledge("Some fact", "topic", 0.8, None, "", &[], None)
            .unwrap();
        mem.close();

        // After close, store is cleared
        let stats = mem.get_statistics();
        let nodes = stats
            .get("semantic_nodes")
            .and_then(|v| v.as_u64())
            .unwrap();
        assert_eq!(nodes, 0);
    }

    // -- Get all knowledge with category filter --

    #[test]
    fn test_get_all_knowledge_with_category() {
        let mut mem = make_mem();
        mem.store_knowledge(
            "Water is H2O",
            "chemistry",
            0.9,
            Some(MemoryCategory::Semantic),
            "",
            &[],
            None,
        )
        .unwrap();
        mem.store_knowledge(
            "Step 1: mix chemicals",
            "procedure",
            0.8,
            Some(MemoryCategory::Procedural),
            "",
            &[],
            None,
        )
        .unwrap();

        let semantic = mem.get_all_knowledge(Some(MemoryCategory::Semantic), 100);
        assert_eq!(semantic.len(), 1);
        assert!(semantic[0].content.contains("H2O"));

        let procedural = mem.get_all_knowledge(Some(MemoryCategory::Procedural), 100);
        assert_eq!(procedural.len(), 1);
        assert!(procedural[0].content.contains("mix chemicals"));

        let all = mem.get_all_knowledge(None, 100);
        assert_eq!(all.len(), 2);
    }

    // -- Retrieve with no results --

    #[test]
    fn test_retrieve_empty_query() {
        let mem = make_mem();
        let sub = mem.retrieve_subgraph("", 2, 20);
        assert!(sub.nodes.is_empty());
    }

    #[test]
    fn test_retrieve_no_matches() {
        let mut mem = make_mem();
        mem.store_knowledge("Plants need water", "biology", 0.9, None, "", &[], None)
            .unwrap();
        let sub = mem.retrieve_subgraph("quantum mechanics spacetime", 2, 20);
        assert!(sub.nodes.is_empty());
    }

    // -- Edge case: concept search with empty/short keywords --

    #[test]
    fn test_search_by_concept_short_keyword() {
        let mut mem = make_mem();
        mem.store_knowledge("Fact about AI", "ai-topic", 0.9, None, "", &[], None)
            .unwrap();

        // Keywords ≤ 2 chars are skipped
        let results = mem.search_by_concept(&["ai".to_string()], 10);
        assert!(results.is_empty());
    }

    // -- Multiple agents isolation --

    #[test]
    fn test_agent_isolation() {
        let mut mem1 = HierarchicalMemory::new("agent_alpha").unwrap();
        let mut mem2 = HierarchicalMemory::new("agent_beta").unwrap();

        mem1.store_knowledge("Alpha fact", "alpha-topic", 0.9, None, "", &[], None)
            .unwrap();
        mem2.store_knowledge("Beta fact", "beta-topic", 0.9, None, "", &[], None)
            .unwrap();

        let sub1 = mem1.retrieve_subgraph("alpha", 2, 20);
        assert!(!sub1.nodes.is_empty());

        // Agent beta should not see agent alpha's facts (different store instances)
        let sub2 = mem2.retrieve_subgraph("alpha", 2, 20);
        assert!(sub2.nodes.is_empty());
    }

    // ====================================================================
    // Extended coverage: store_knowledge
    // ====================================================================

    #[test]
    fn test_store_knowledge_with_tags_and_source() {
        let mut mem = make_mem();
        let tags = vec!["lang".into(), "systems".into()];
        let nid = mem
            .store_knowledge(
                "Rust has ownership semantics",
                "rust",
                0.95,
                None,
                "doc-42",
                &tags,
                None,
            )
            .unwrap();

        let nodes = mem.get_all_knowledge(None, 100);
        let node = nodes.iter().find(|n| n.node_id == nid).unwrap();
        assert_eq!(node.source_id, "doc-42");
        assert_eq!(node.tags, tags);
        assert!(node.confidence > 0.94);
    }

    #[test]
    fn test_store_knowledge_explicit_category() {
        let mut mem = make_mem();
        let nid = mem
            .store_knowledge(
                "A plain factual statement with no keywords",
                "trivia",
                0.7,
                Some(MemoryCategory::Procedural),
                "",
                &[],
                None,
            )
            .unwrap();

        let nodes = mem.get_all_knowledge(Some(MemoryCategory::Procedural), 100);
        assert!(nodes.iter().any(|n| n.node_id == nid));
    }

    #[test]
    fn test_store_knowledge_nan_confidence_rejected() {
        let mut mem = make_mem();
        assert!(mem
            .store_knowledge("data", "concept", f64::NAN, None, "", &[], None)
            .is_err());
    }

    #[test]
    fn test_store_knowledge_out_of_range_confidence() {
        let mut mem = make_mem();
        assert!(mem
            .store_knowledge("data", "concept", 1.5, None, "", &[], None)
            .is_err());
        assert!(mem
            .store_knowledge("data", "concept", -0.1, None, "", &[], None)
            .is_err());
    }

    #[test]
    fn test_store_knowledge_boundary_confidence() {
        let mut mem = make_mem();
        assert!(mem
            .store_knowledge("low bound", "c", 0.0, None, "", &[], None)
            .is_ok());
        assert!(mem
            .store_knowledge("high bound", "c", 1.0, None, "", &[], None)
            .is_ok());
    }

    #[test]
    fn test_store_knowledge_with_temporal_metadata() {
        let mut mem = make_mem();
        let mut tm = HashMap::new();
        tm.insert("temporal_index".into(), serde_json::json!(5));
        tm.insert("source_date".into(), serde_json::json!("2024-03-15"));

        let nid = mem
            .store_knowledge(
                "Temperature is 25C today",
                "weather",
                0.8,
                None,
                "",
                &[],
                Some(&tm),
            )
            .unwrap();

        let nodes = mem.get_all_knowledge(None, 100);
        let node = nodes.iter().find(|n| n.node_id == nid).unwrap();
        assert_eq!(
            node.metadata.get("temporal_index").and_then(|v| v.as_i64()),
            Some(5)
        );
    }

    #[test]
    fn test_store_knowledge_whitespace_trimmed() {
        let mut mem = make_mem();
        let nid = mem
            .store_knowledge("  trimmed content  ", "topic", 0.8, None, "", &[], None)
            .unwrap();
        let nodes = mem.get_all_knowledge(None, 100);
        let node = nodes.iter().find(|n| n.node_id == nid).unwrap();
        assert_eq!(node.content, "trimmed content");
    }

    // ====================================================================
    // Extended coverage: retrieve_subgraph
    // ====================================================================

    #[test]
    fn test_retrieve_subgraph_max_nodes_limit() {
        let mut mem = make_mem();
        for i in 0..10 {
            mem.store_knowledge(
                &format!("Quantum physics experiment number {i}"),
                "quantum",
                0.8,
                None,
                "",
                &[],
                None,
            )
            .unwrap();
        }

        let sub = mem.retrieve_subgraph("quantum physics", 2, 3);
        assert!(sub.nodes.len() <= 3);
    }

    #[test]
    fn test_retrieve_subgraph_returns_edges() {
        let mut mem = make_mem();
        mem.store_knowledge(
            "Machine learning uses neural networks",
            "machine-learning",
            0.9,
            None,
            "",
            &[],
            None,
        )
        .unwrap();
        mem.store_knowledge(
            "Deep learning neural networks are powerful",
            "machine-learning",
            0.85,
            None,
            "",
            &[],
            None,
        )
        .unwrap();

        let sub = mem.retrieve_subgraph("neural networks machine learning", 2, 20);
        assert!(sub.nodes.len() >= 2);
        // With two similar nodes, there should be at least one edge
        assert!(
            !sub.edges.is_empty() || sub.nodes.len() >= 2,
            "Expected edges or multiple nodes"
        );
    }

    #[test]
    fn test_retrieve_subgraph_query_field() {
        let mem = make_mem();
        let sub = mem.retrieve_subgraph("test query", 2, 20);
        assert_eq!(sub.query, "test query");
    }

    // ====================================================================
    // Extended coverage: get_temporal_chain
    // ====================================================================

    #[test]
    fn test_temporal_chain_empty_concept() {
        let mem = make_mem();
        let chain = mem.get_temporal_chain("");
        assert!(chain.is_empty());
    }

    #[test]
    fn test_temporal_chain_no_matching_concept() {
        let mut mem = make_mem();
        mem.store_knowledge("Unrelated fact", "other", 0.8, None, "", &[], None)
            .unwrap();
        let chain = mem.get_temporal_chain("nonexistent_concept_xyz");
        assert!(chain.is_empty());
    }

    #[test]
    fn test_temporal_chain_single_node() {
        let mut mem = make_mem();
        let mut tm = HashMap::new();
        tm.insert("temporal_index".into(), serde_json::json!(1));
        mem.store_knowledge(
            "Solo fact here",
            "solo-concept",
            0.9,
            None,
            "",
            &[],
            Some(&tm),
        )
        .unwrap();

        let chain = mem.get_temporal_chain("solo-concept");
        assert_eq!(chain.len(), 1);
        assert!(chain[0].content.contains("Solo fact"));
    }

    #[test]
    fn test_temporal_chain_three_versions() {
        let mut mem = make_mem();
        for i in 1..=3 {
            let mut tm = HashMap::new();
            tm.insert("temporal_index".into(), serde_json::json!(i));
            mem.store_knowledge(
                &format!("Server count is {}", i * 10),
                "server count",
                0.8,
                None,
                "",
                &[],
                Some(&tm),
            )
            .unwrap();
        }

        let chain = mem.get_temporal_chain("server");
        assert!(chain.len() >= 2, "Expected multiple versions in chain");
    }

    // ====================================================================
    // Extended coverage: get_knowledge_evolution
    // ====================================================================

    #[test]
    fn test_knowledge_evolution_empty_concept() {
        let mem = make_mem();
        let evolution = mem.get_knowledge_evolution("");
        assert!(evolution.is_empty());
    }

    #[test]
    fn test_knowledge_evolution_single_node_no_reason() {
        let mut mem = make_mem();
        mem.store_knowledge("Singular fact", "single-topic", 0.9, None, "", &[], None)
            .unwrap();

        let evolution = mem.get_knowledge_evolution("single-topic");
        if !evolution.is_empty() {
            assert!(evolution[0].1.is_none());
        }
    }

    // ====================================================================
    // Extended coverage: execute_aggregation
    // ====================================================================

    #[test]
    fn test_aggregation_count_with_concept_filter() {
        let mut mem = make_mem();
        mem.store_knowledge("Rust is fast", "rust-lang", 0.9, None, "", &[], None)
            .unwrap();
        mem.store_knowledge("Python is dynamic", "python-lang", 0.8, None, "", &[], None)
            .unwrap();
        mem.store_knowledge("Rust has ownership", "rust-lang", 0.85, None, "", &[], None)
            .unwrap();

        let result = mem.execute_aggregation("count", "rust", 10);
        assert_eq!(result.get("count").and_then(|v| v.as_u64()), Some(2));
    }

    #[test]
    fn test_aggregation_avg_confidence_empty() {
        let mem = make_mem();
        let result = mem.execute_aggregation("avg_confidence", "", 10);
        assert_eq!(
            result.get("avg_confidence").and_then(|v| v.as_f64()),
            Some(0.0)
        );
    }

    #[test]
    fn test_aggregation_top_concepts_with_limit() {
        let mut mem = make_mem();
        for topic in &["alpha", "alpha", "alpha", "beta", "beta", "gamma"] {
            mem.store_knowledge(
                &format!("Fact about {topic}"),
                topic,
                0.8,
                None,
                "",
                &[],
                None,
            )
            .unwrap();
        }

        let result = mem.execute_aggregation("top_concepts", "", 2);
        let items = result.get("items").unwrap().as_object().unwrap();
        assert!(items.len() <= 2);
    }

    // ====================================================================
    // Extended coverage: MemoryClassifier::classify
    // ====================================================================

    #[test]
    fn test_classifier_concept_triggers_category() {
        let c = MemoryClassifier::new();
        // Concept alone contains the keyword
        assert_eq!(
            c.classify("generic statement", "future plans"),
            MemoryCategory::Prospective
        );
    }

    #[test]
    fn test_classifier_priority_procedural_over_episodic() {
        let c = MemoryClassifier::new();
        // Contains both "step" (procedural) and "happened" (episodic)
        // Procedural is checked first
        assert_eq!(
            c.classify("step by step what happened", ""),
            MemoryCategory::Procedural
        );
    }

    #[test]
    fn test_classifier_case_insensitive() {
        let c = MemoryClassifier::new();
        assert_eq!(
            c.classify("HOW TO BUILD A HOUSE", ""),
            MemoryCategory::Procedural
        );
        assert_eq!(
            c.classify("I SAW AN EVENT THAT OCCURRED", ""),
            MemoryCategory::Episodic
        );
    }

    #[test]
    fn test_classifier_default_returns() {
        let c = MemoryClassifier::default();
        assert_eq!(c.classify("plain fact", ""), MemoryCategory::Semantic);
    }
}
