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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{GraphEdge, GraphNode};

    fn make_knode(id: &str, content: &str, confidence: f64) -> KnowledgeNode {
        KnowledgeNode {
            node_id: id.to_string(),
            content: content.to_string(),
            confidence,
            ..KnowledgeNode::default()
        }
    }

    #[test]
    fn test_build_sim_map() {
        let m = build_sim_map("content", "concept", &["t1".into()]);
        assert_eq!(m.get("content").unwrap().as_str().unwrap(), "content");
        assert_eq!(m.get("concept").unwrap().as_str().unwrap(), "concept");
    }

    #[test]
    fn test_build_sim_map_empty() {
        let m = build_sim_map("", "", &[]);
        assert_eq!(m.get("content").unwrap().as_str().unwrap(), "");
    }

    #[test]
    fn test_graph_node_to_knowledge_node_defaults() {
        let gn = GraphNode::new("n1".into(), "Fact".into());
        let kn = graph_node_to_knowledge_node(&gn);
        assert_eq!(kn.node_id, "n1");
        assert!((kn.confidence - 0.8).abs() < 0.01);
        assert_eq!(kn.category, MemoryCategory::Semantic);
    }

    #[test]
    fn test_graph_node_to_knowledge_node_full() {
        let mut props = HashMap::new();
        props.insert("content".into(), "Rust is fast".into());
        props.insert("confidence".into(), "0.95".into());
        props.insert("tags".into(), r#"["lang"]"#.into());
        props.insert("metadata".into(), r#"{"category":"procedural"}"#.into());
        let gn = GraphNode {
            node_id: "n2".into(),
            node_type: "F".into(),
            properties: props,
            graph_origin: String::new(),
        };
        let kn = graph_node_to_knowledge_node(&gn);
        assert!((kn.confidence - 0.95).abs() < 0.01);
        assert_eq!(kn.category, MemoryCategory::Procedural);
    }

    #[test]
    fn test_parse_edge_metadata_valid() {
        let mut props = HashMap::new();
        props.insert("metadata".into(), r#"{"w":0.5}"#.into());
        let edge = GraphEdge {
            edge_id: String::new(),
            source_id: "a".into(),
            target_id: "b".into(),
            edge_type: "L".into(),
            properties: props,
            graph_origin: String::new(),
        };
        let meta = parse_edge_metadata(&edge);
        assert!((meta.get("w").unwrap().as_f64().unwrap() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_parse_edge_metadata_missing() {
        let edge = GraphEdge::new("a".into(), "b".into(), "L".into());
        assert!(parse_edge_metadata(&edge).is_empty());
    }

    #[test]
    fn test_parse_edge_metadata_invalid() {
        let mut props = HashMap::new();
        props.insert("metadata".into(), "not json".into());
        let edge = GraphEdge {
            edge_id: String::new(),
            source_id: "a".into(),
            target_id: "b".into(),
            edge_type: "L".into(),
            properties: props,
            graph_origin: String::new(),
        };
        assert!(parse_edge_metadata(&edge).is_empty());
    }

    #[test]
    fn test_rank_and_truncate_basic() {
        let nodes = vec![
            make_knode("a", "rust programming", 0.5),
            make_knode("b", "rust compiler", 0.9),
        ];
        let result = rank_and_truncate(nodes, &["rust".into()], 1);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_rank_and_truncate_empty() {
        let result = rank_and_truncate(vec![], &["kw".into()], 10);
        assert!(result.is_empty());
    }

    #[test]
    fn test_store_id_re() {
        assert!(STORE_ID_RE.is_match("my-agent"));
        assert!(!STORE_ID_RE.is_match(""));
        assert!(!STORE_ID_RE.is_match("-bad"));
    }

    #[test]
    fn test_make_id_and_now_iso() {
        let id = make_id();
        assert_eq!(id.len(), 36);
        let ts = now_iso();
        assert!(ts.starts_with("20"));
    }
}
