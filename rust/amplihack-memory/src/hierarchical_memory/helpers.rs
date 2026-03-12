//! Free-standing helper functions and constants for the hierarchical memory.

use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;

use crate::graph::GraphNode;
use crate::memory_types::MemoryCategory;

use super::types::KnowledgeNode;

// ---------------------------------------------------------------------------
// Module-level constants
// ---------------------------------------------------------------------------

/// Maximum number of nodes to scan when computing statistics.
pub(crate) const MAX_STATISTICS_NODES: usize = 100_000;

/// Maximum number of nodes to scan in aggregation queries.
pub(crate) const MAX_AGGREGATION_NODES: usize = 10_000;

/// Maximum number of candidate nodes to evaluate for SUPERSEDES edges.
pub(crate) const MAX_SUPERSEDE_CANDIDATES: usize = 20;

// ---------------------------------------------------------------------------
// Module-level compiled regexes (c2-012)
// ---------------------------------------------------------------------------

/// Regex to validate store / agent identifiers.
pub(crate) static STORE_ID_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,63}$").unwrap());

// ---------------------------------------------------------------------------
// ID and timestamp helpers
// ---------------------------------------------------------------------------

pub(crate) fn make_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

pub(crate) fn now_iso() -> String {
    chrono::Utc::now()
        .format("%Y-%m-%dT%H:%M:%S%.6f")
        .to_string()
}

// ---------------------------------------------------------------------------
// Graph conversion helpers
// ---------------------------------------------------------------------------

pub(crate) fn build_sim_map(
    content: &str,
    concept: &str,
    tags: &[String],
) -> HashMap<String, serde_json::Value> {
    let mut m = HashMap::new();
    m.insert(
        "content".into(),
        serde_json::Value::String(content.to_string()),
    );
    m.insert(
        "concept".into(),
        serde_json::Value::String(concept.to_string()),
    );
    m.insert("tags".into(), serde_json::json!(tags));
    m
}

pub(crate) fn graph_node_to_knowledge_node(gn: &GraphNode) -> KnowledgeNode {
    let tags: Vec<String> = gn
        .properties
        .get("tags")
        .and_then(|t| serde_json::from_str(t).ok())
        .unwrap_or_default();

    let metadata: HashMap<String, serde_json::Value> = gn
        .properties
        .get("metadata")
        .and_then(|m| serde_json::from_str(m).ok())
        .unwrap_or_default();

    let category_str = metadata
        .get("category")
        .and_then(|v| v.as_str())
        .unwrap_or("semantic");
    let category = category_str
        .parse::<MemoryCategory>()
        .unwrap_or(MemoryCategory::Semantic);

    let confidence: f64 = gn
        .properties
        .get("confidence")
        .and_then(|c| c.parse().ok())
        .unwrap_or(0.8);

    KnowledgeNode {
        node_id: gn.node_id.clone(),
        category,
        content: gn.properties.get("content").cloned().unwrap_or_default(),
        concept: gn.properties.get("concept").cloned().unwrap_or_default(),
        confidence,
        source_id: gn.properties.get("source_id").cloned().unwrap_or_default(),
        created_at: gn.properties.get("created_at").cloned().unwrap_or_default(),
        tags,
        metadata,
    }
}

pub(crate) fn parse_edge_metadata(
    ge: &crate::graph::GraphEdge,
) -> HashMap<String, serde_json::Value> {
    ge.properties
        .get("metadata")
        .and_then(|m| serde_json::from_str(m).ok())
        .unwrap_or_default()
}

fn rank_score(node: &KnowledgeNode, keywords: &[String]) -> f64 {
    let content_lower = node.content.to_lowercase();
    let keyword_hits = keywords
        .iter()
        .filter(|kw| kw.len() > 2 && content_lower.contains(kw.as_str()))
        .count();
    let keyword_relevance = keyword_hits as f64 / keywords.len().max(1) as f64;
    node.confidence * (0.5 + 0.5 * keyword_relevance)
}

/// Rank nodes by confidence × keyword relevance, then truncate to `max_nodes`.
pub(crate) fn rank_and_truncate(
    mut nodes: Vec<KnowledgeNode>,
    keywords: &[String],
    max_nodes: usize,
) -> Vec<KnowledgeNode> {
    nodes.sort_by(|a, b| {
        let score_a = rank_score(a, keywords);
        let score_b = rank_score(b, keywords);
        score_b
            .partial_cmp(&score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    nodes.truncate(max_nodes);
    nodes
}
