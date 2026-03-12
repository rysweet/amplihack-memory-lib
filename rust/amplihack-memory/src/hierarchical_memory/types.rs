//! Data types for the hierarchical memory system.

use std::collections::HashMap;

use crate::memory_types::MemoryCategory;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A node in the knowledge graph.
#[derive(Debug, Clone)]
pub struct KnowledgeNode {
    pub node_id: String,
    pub category: MemoryCategory,
    pub content: String,
    pub concept: String,
    pub confidence: f64,
    pub source_id: String,
    pub created_at: String,
    pub tags: Vec<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Default for KnowledgeNode {
    fn default() -> Self {
        Self {
            node_id: String::new(),
            category: MemoryCategory::Semantic,
            content: String::new(),
            concept: String::new(),
            confidence: 0.8,
            source_id: String::new(),
            created_at: String::new(),
            tags: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

/// An edge in the knowledge graph.
#[derive(Debug, Clone)]
pub struct KnowledgeEdge {
    pub source_id: String,
    pub target_id: String,
    pub relationship: String,
    pub weight: f64,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Default for KnowledgeEdge {
    fn default() -> Self {
        Self {
            source_id: String::new(),
            target_id: String::new(),
            relationship: String::new(),
            weight: 1.0,
            metadata: HashMap::new(),
        }
    }
}

/// A subgraph of knowledge nodes and edges returned by `retrieve_subgraph()`.
#[derive(Debug, Clone, Default)]
pub struct KnowledgeSubgraph {
    pub nodes: Vec<KnowledgeNode>,
    pub edges: Vec<KnowledgeEdge>,
    pub query: String,
}

impl KnowledgeSubgraph {
    /// Format subgraph as LLM-readable context string.
    ///
    /// If `chronological` is true, sorts facts by `temporal_index` metadata
    /// instead of confidence. Useful for temporal reasoning questions.
    pub fn to_llm_context(&self, chronological: bool) -> String {
        if self.nodes.is_empty() {
            return "No relevant knowledge found.".to_string();
        }

        let mut lines = vec![format!("Knowledge graph context for: {}\n", self.query)];

        if chronological {
            lines.push(self.format_nodes_chronological(&self.nodes));
        } else {
            lines.push(self.format_nodes_by_confidence(&self.nodes));
        }

        let contradictions = self.format_contradictions();
        if !contradictions.is_empty() {
            lines.push(String::new());
            lines.push(contradictions);
        }

        let relationships = self.format_relationships();
        if !relationships.is_empty() {
            lines.push(String::new());
            lines.push(relationships);
        }

        lines.join("\n")
    }

    /// Format nodes sorted by temporal_index for chronological display.
    fn format_nodes_chronological(&self, nodes: &[KnowledgeNode]) -> String {
        let mut sorted_nodes = nodes.to_vec();
        sorted_nodes.sort_by(|a, b| {
            let t_a = a
                .metadata
                .get("temporal_index")
                .and_then(|v| v.as_i64())
                .unwrap_or(999999);
            let t_b = b
                .metadata
                .get("temporal_index")
                .and_then(|v| v.as_i64())
                .unwrap_or(999999);
            t_a.cmp(&t_b).then_with(|| a.created_at.cmp(&b.created_at))
        });

        let mut lines = vec!["Facts (in chronological order):".to_string()];
        for (i, node) in sorted_nodes.iter().enumerate() {
            let mut time_marker = String::new();
            let mut source_marker = String::new();
            if let Some(src_date) = node.metadata.get("source_date").and_then(|v| v.as_str()) {
                if !src_date.is_empty() {
                    time_marker = format!(" [Date: {src_date}]");
                }
            } else if let Some(t_order) =
                node.metadata.get("temporal_order").and_then(|v| v.as_str())
            {
                if !t_order.is_empty() {
                    time_marker = format!(" [Time: {t_order}]");
                }
            }
            if let Some(src_label) = node.metadata.get("source_label").and_then(|v| v.as_str()) {
                if !src_label.is_empty() {
                    source_marker = format!(" [Source: {src_label}]");
                }
            }
            lines.push(format!(
                "  {}. [{}]{}{} {} (confidence: {:.1})",
                i + 1,
                node.concept,
                time_marker,
                source_marker,
                node.content,
                node.confidence
            ));
        }
        lines.join("\n")
    }

    /// Format nodes sorted by confidence (highest first).
    fn format_nodes_by_confidence(&self, nodes: &[KnowledgeNode]) -> String {
        let mut sorted_nodes = nodes.to_vec();
        sorted_nodes.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut lines = vec!["Facts:".to_string()];
        for (i, node) in sorted_nodes.iter().enumerate() {
            let mut source_marker = String::new();
            if let Some(src_label) = node.metadata.get("source_label").and_then(|v| v.as_str()) {
                if !src_label.is_empty() {
                    source_marker = format!(" [Source: {src_label}]");
                }
            }
            lines.push(format!(
                "  {}. [{}]{} {} (confidence: {:.1})",
                i + 1,
                node.concept,
                source_marker,
                node.content,
                node.confidence
            ));
        }
        lines.join("\n")
    }

    /// Format contradiction warning lines from edges with contradiction metadata.
    fn format_contradictions(&self) -> String {
        let contradiction_edges: Vec<&KnowledgeEdge> = self
            .edges
            .iter()
            .filter(|e| {
                e.metadata
                    .get("contradiction")
                    .map(|v| v.as_str() == Some("true") || v.as_bool().unwrap_or(false))
                    .unwrap_or(false)
            })
            .collect();

        if contradiction_edges.is_empty() {
            return String::new();
        }

        let mut lines = vec!["Contradictions detected:".to_string()];
        for edge in &contradiction_edges {
            let conflict = edge
                .metadata
                .get("conflicting_values")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            lines.push(format!(
                "  - WARNING: Conflicting information found: {conflict}"
            ));
        }
        lines.join("\n")
    }

    /// Format relationship lines from edges.
    fn format_relationships(&self) -> String {
        if self.edges.is_empty() {
            return String::new();
        }

        let mut lines = vec!["Relationships:".to_string()];
        for edge in &self.edges {
            let src_short = if edge.source_id.len() >= 8 {
                &edge.source_id[..8]
            } else {
                &edge.source_id
            };
            let tgt_short = if edge.target_id.len() >= 8 {
                &edge.target_id[..8]
            } else {
                &edge.target_id
            };
            lines.push(format!(
                "  - {}.. {} {}.. (weight: {:.2})",
                src_short, edge.relationship, tgt_short, edge.weight
            ));
        }
        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// MemoryClassifier
// ---------------------------------------------------------------------------

/// Rule-based classifier for memory categories.
///
/// Uses keyword patterns to classify content into memory categories.
pub struct MemoryClassifier;

impl MemoryClassifier {
    const PROCEDURAL_KEYWORDS: &[&str] = &[
        "step",
        "steps",
        "how to",
        "procedure",
        "process",
        "method",
        "recipe",
        "instructions",
    ];
    const PROSPECTIVE_KEYWORDS: &[&str] = &[
        "plan", "goal", "future", "will", "should", "todo", "intend", "schedule",
    ];
    const EPISODIC_KEYWORDS: &[&str] = &[
        "happened",
        "event",
        "occurred",
        "experience",
        "observed",
        "saw",
        "noticed",
    ];

    /// Create a new `MemoryClassifier` with default keyword patterns.
    pub fn new() -> Self {
        Self
    }

    /// Classify content into a memory category.
    pub fn classify(&self, content: &str, concept: &str) -> MemoryCategory {
        let text = format!("{content} {concept}").to_lowercase();

        if Self::PROCEDURAL_KEYWORDS.iter().any(|kw| text.contains(kw)) {
            return MemoryCategory::Procedural;
        }

        if Self::PROSPECTIVE_KEYWORDS
            .iter()
            .any(|kw| text.contains(kw))
        {
            return MemoryCategory::Prospective;
        }

        if Self::EPISODIC_KEYWORDS.iter().any(|kw| text.contains(kw)) {
            return MemoryCategory::Episodic;
        }

        MemoryCategory::Semantic
    }
}

impl Default for MemoryClassifier {
    fn default() -> Self {
        Self::new()
    }
}
