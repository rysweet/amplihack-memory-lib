//! Semantic memory methods for [`CognitiveMemory`].

use std::collections::HashMap;

use crate::memory_types::SemanticFact;
use crate::{MemoryError, Result};

use crate::graph::protocol::GraphStore;
use tracing::warn;

use super::converters::node_to_fact;
use super::types::{agent_filter, new_id, ts_now, ET_DERIVES_FROM, ET_SIMILAR_TO, NT_SEMANTIC};
use super::CognitiveMemory;

impl CognitiveMemory {
    /// Store a semantic fact.
    ///
    /// Vector embeddings are not available in the Rust port; keyword search
    /// is used for retrieval.
    pub fn store_fact(
        &mut self,
        concept: &str,
        content: &str,
        confidence: f64,
        source_id: &str,
        tags: Option<&[String]>,
        metadata: Option<&HashMap<String, serde_json::Value>>,
    ) -> Result<String> {
        if confidence.is_nan() || !(0.0..=1.0).contains(&confidence) {
            return Err(MemoryError::InvalidInput(
                "confidence must be between 0.0 and 1.0".into(),
            ));
        }

        let node_id = new_id("sem");
        let now = ts_now();
        let tags_json = tags
            .map(|t| {
                serde_json::to_string(t).unwrap_or_else(|e| {
                    warn!("store_fact: failed to serialize tags: {e}");
                    "[]".into()
                })
            })
            .unwrap_or_else(|| "[]".into());
        let meta_json = metadata
            .map(|m| {
                serde_json::to_string(m).unwrap_or_else(|e| {
                    warn!("store_fact: failed to serialize metadata: {e}");
                    "{}".into()
                })
            })
            .unwrap_or_else(|| "{}".into());

        let mut props = HashMap::new();
        props.insert("node_id".to_string(), node_id.clone());
        props.insert("agent_id".to_string(), self.agent_name.clone());
        props.insert("concept".to_string(), concept.to_string());
        props.insert("content".to_string(), content.to_string());
        props.insert("confidence".to_string(), confidence.to_string());
        props.insert("source_id".to_string(), source_id.to_string());
        props.insert("tags".to_string(), tags_json);
        props.insert("metadata".to_string(), meta_json);
        props.insert("created_at".to_string(), now.to_string());

        self.graph
            .add_node(NT_SEMANTIC, props, Some(&node_id))
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        Ok(node_id)
    }

    /// Search semantic facts using keyword matching.
    ///
    /// Words from `query` are matched against both `concept` and `content`
    /// fields (case-insensitive). Results are filtered by `min_confidence`
    /// and sorted by confidence descending.
    pub fn search_facts(
        &self,
        query: &str,
        limit: usize,
        min_confidence: f64,
    ) -> Vec<SemanticFact> {
        let keywords: Vec<String> = query
            .split_whitespace()
            .filter(|w| !w.is_empty())
            .map(|w| w.to_lowercase())
            .collect();

        if keywords.is_empty() {
            return self.get_all_facts(limit);
        }

        let filter = agent_filter(&self.agent_name);
        let nodes = self
            .graph
            .query_nodes(NT_SEMANTIC, Some(&filter), usize::MAX);

        let mut facts: Vec<SemanticFact> = nodes
            .into_iter()
            .filter(|n| {
                let conf: f64 = n
                    .properties
                    .get("confidence")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0.0);
                if conf < min_confidence {
                    return false;
                }
                let concept = n
                    .properties
                    .get("concept")
                    .map(|s| s.to_lowercase())
                    .unwrap_or_default();
                let content = n
                    .properties
                    .get("content")
                    .map(|s| s.to_lowercase())
                    .unwrap_or_default();

                keywords
                    .iter()
                    .any(|kw| concept.contains(kw) || content.contains(kw))
            })
            .map(|n| node_to_fact(&n.properties))
            .collect();

        facts.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        facts.truncate(limit);
        facts
    }

    /// Return all semantic facts for this agent, sorted by confidence descending.
    pub fn get_all_facts(&self, limit: usize) -> Vec<SemanticFact> {
        let filter = agent_filter(&self.agent_name);
        let nodes = self
            .graph
            .query_nodes(NT_SEMANTIC, Some(&filter), usize::MAX);

        let mut facts: Vec<SemanticFact> = nodes
            .into_iter()
            .map(|n| node_to_fact(&n.properties))
            .collect();

        facts.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        facts.truncate(limit);
        facts
    }

    /// Link two semantic facts with a SIMILAR_TO edge.
    pub fn link_similar_facts(
        &mut self,
        fact_id_a: &str,
        fact_id_b: &str,
        similarity_score: f64,
    ) -> Result<()> {
        let mut props = HashMap::new();
        props.insert("similarity_score".to_string(), similarity_score.to_string());
        self.graph
            .add_edge(fact_id_a, fact_id_b, ET_SIMILAR_TO, Some(props))
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Link a semantic fact to its source episode.
    pub fn link_fact_to_episode(&mut self, fact_id: &str, episode_id: &str) -> Result<()> {
        let now = ts_now();
        let mut props = HashMap::new();
        props.insert("derived_at".to_string(), now.to_string());
        self.graph
            .add_edge(fact_id, episode_id, ET_DERIVES_FROM, Some(props))
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        Ok(())
    }
}
