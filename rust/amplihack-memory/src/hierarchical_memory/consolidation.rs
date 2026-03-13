//! Memory consolidation, aggregation and statistics.

use std::collections::HashMap;

use crate::graph::GraphStore;
use crate::memory_types::MemoryCategory;

use super::helpers::{graph_node_to_knowledge_node, MAX_AGGREGATION_NODES, MAX_STATISTICS_NODES};
use super::types::KnowledgeNode;
use super::HierarchicalMemory;

impl HierarchicalMemory {
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
}
