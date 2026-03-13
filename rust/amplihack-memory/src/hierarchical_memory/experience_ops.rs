//! Experience storage operations for hierarchical memory.

use std::collections::{HashMap, HashSet};

use crate::contradiction::detect_contradiction;
use crate::entity_extraction::extract_entity_name;
use crate::graph::{Direction, GraphStore};
use crate::memory_types::MemoryCategory;
use crate::similarity::compute_similarity;
use tracing::warn;

use super::helpers::{build_sim_map, make_id, now_iso, MAX_SUPERSEDE_CANDIDATES};
use super::types::KnowledgeNode;
use super::{
    HierarchicalMemory, CONTRADICTION_SIMILARITY_THRESHOLD, DEFAULT_SIMILARITY_THRESHOLD,
    MIN_CONFIDENCE, SUPERSEDE_CONFIDENCE_DECAY,
};

impl HierarchicalMemory {
    /// Store a knowledge node in the graph.
    ///
    /// Auto-classifies if `category` is `None`. Computes similarity against
    /// existing nodes and creates `SIMILAR_TO` edges for scores > [`DEFAULT_SIMILARITY_THRESHOLD`].
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

            if score > DEFAULT_SIMILARITY_THRESHOLD {
                let mut edge_meta = HashMap::new();

                // Check for contradiction between high-similarity facts
                if score > CONTRADICTION_SIMILARITY_THRESHOLD {
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

                match self
                    .store
                    .add_edge(node_id, &other.node_id, "SIMILAR_TO", Some(props))
                {
                    Ok(_) => self.similarity_edge_count += 1,
                    Err(e) => warn!("detect_similarity: failed to add SIMILAR_TO edge: {e}"),
                }
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
            match self
                .store
                .add_edge(new_node_id, &old_id, "SUPERSEDES", Some(props))
            {
                Ok(_) => self.supersedes_edge_count += 1,
                Err(e) => warn!("detect_supersedes: failed to add SUPERSEDES edge: {e}"),
            }
        }
    }

    pub(super) fn mark_superseded(&self, nodes: &mut [KnowledgeNode]) {
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
                node.confidence =
                    (node.confidence * SUPERSEDE_CONFIDENCE_DECAY).max(MIN_CONFIDENCE);
            }
        }
    }
}
