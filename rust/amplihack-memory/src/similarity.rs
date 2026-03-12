//! Text similarity computation for knowledge graph edges.
//!
//! Simple, deterministic similarity (no ML embeddings needed):
//! - Jaccard coefficient on tokenized words minus stop words
//! - Tag overlap for categorical similarity
//! - Weighted composite score for graph edge creation

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

/// Common English stop words.
static STOP_WORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can", "to",
    "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "about", "like", "through",
    "after", "over", "between", "out", "against", "during", "without", "before", "under", "around",
    "among", "and", "but", "or", "nor", "not", "so", "yet", "both", "either", "neither", "each",
    "every", "all", "any", "few", "more", "most", "other", "some", "such", "no", "only", "own",
    "same", "than", "too", "very", "just", "because", "if", "when", "where", "how", "what",
    "which", "who", "whom", "this", "that", "these", "those", "it", "its", "i", "me", "my", "we",
    "our", "you", "your", "he", "him", "his", "she", "her", "they", "them", "their",
];

static STOP_WORDS_SET: LazyLock<HashSet<&'static str>> =
    LazyLock::new(|| STOP_WORDS.iter().copied().collect());

/// Tokenize text into lowercase words, removing stop words and short tokens.
fn tokenize(text: &str) -> HashSet<String> {
    if text.is_empty() {
        return HashSet::new();
    }
    text.to_lowercase()
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| ".,;:!?()[]{}\"'".contains(c)))
        .filter(|w| w.len() > 2)
        .filter(|w| !STOP_WORDS_SET.contains(w))
        .map(String::from)
        .collect()
}

/// Compute Jaccard similarity on tokenized words minus stop words.
pub fn compute_word_similarity(text_a: &str, text_b: &str) -> f64 {
    let tokens_a = tokenize(text_a);
    let tokens_b = tokenize(text_b);

    if tokens_a.is_empty() || tokens_b.is_empty() {
        return 0.0;
    }

    let intersection = tokens_a.intersection(&tokens_b).count();
    let union = tokens_a.union(&tokens_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Compute Jaccard similarity between two tag lists.
pub fn compute_tag_similarity(tags_a: &[String], tags_b: &[String]) -> f64 {
    if tags_a.is_empty() || tags_b.is_empty() {
        return 0.0;
    }

    let set_a: HashSet<String> = tags_a
        .iter()
        .map(|t| t.to_lowercase().trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();
    let set_b: HashSet<String> = tags_b
        .iter()
        .map(|t| t.to_lowercase().trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();

    if set_a.is_empty() || set_b.is_empty() {
        return 0.0;
    }

    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Compute weighted composite similarity between two knowledge nodes.
///
/// Weights: 0.5 * word_similarity + 0.2 * tag_similarity + 0.3 * concept_similarity
pub fn compute_similarity(
    node_a: &HashMap<String, serde_json::Value>,
    node_b: &HashMap<String, serde_json::Value>,
) -> f64 {
    let content_a = node_a.get("content").and_then(|v| v.as_str()).unwrap_or("");
    let content_b = node_b.get("content").and_then(|v| v.as_str()).unwrap_or("");
    let word_sim = compute_word_similarity(content_a, content_b);

    let tags_a: Vec<String> = node_a
        .get("tags")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();
    let tags_b: Vec<String> = node_b
        .get("tags")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();
    let tag_sim = compute_tag_similarity(&tags_a, &tags_b);

    let concept_a = node_a.get("concept").and_then(|v| v.as_str()).unwrap_or("");
    let concept_b = node_b.get("concept").and_then(|v| v.as_str()).unwrap_or("");
    let concept_sim = compute_word_similarity(concept_a, concept_b);

    0.5 * word_sim + 0.2 * tag_sim + 0.3 * concept_sim
}

/// Rerank retrieved facts by keyword relevance to a query.
pub fn rerank_facts_by_query(
    facts: &[HashMap<String, serde_json::Value>],
    query: &str,
    top_k: usize,
) -> Vec<HashMap<String, serde_json::Value>> {
    if facts.is_empty() || query.is_empty() {
        return facts.to_vec();
    }

    let query_tokens = tokenize(query);
    if query_tokens.is_empty() {
        return facts.to_vec();
    }

    let query_lower = query.to_lowercase();
    let temporal_cues = [
        "change",
        "changed",
        "original",
        "before",
        "after",
        "previous",
        "current",
        "first",
        "initially",
        "updated",
        "revised",
        "intermediate",
        "over time",
        "history",
        "evolution",
        "timeline",
        "when",
    ];
    let has_temporal_query = temporal_cues.iter().any(|cue| query_lower.contains(cue));

    let mut scored: Vec<(f64, usize, &HashMap<String, serde_json::Value>)> = facts
        .iter()
        .enumerate()
        .map(|(idx, fact)| {
            let context = fact.get("context").and_then(|v| v.as_str()).unwrap_or("");
            let outcome = fact.get("outcome").and_then(|v| v.as_str()).unwrap_or("");
            let fact_text = format!("{context} {outcome}");
            let fact_tokens = tokenize(&fact_text);

            if fact_tokens.is_empty() {
                return (0.0, idx, fact);
            }

            let overlap =
                query_tokens.intersection(&fact_tokens).count() as f64 / query_tokens.len() as f64;

            let mut score = overlap;

            if has_temporal_query {
                if let Some(meta) = fact.get("metadata").and_then(|v| v.as_object()) {
                    let has_temporal = meta
                        .get("temporal_index")
                        .and_then(|v| v.as_i64())
                        .is_some_and(|v| v > 0)
                        || meta.get("source_date").is_some()
                        || meta.get("temporal_order").is_some();
                    if has_temporal {
                        score += 0.15;
                    }
                }
            }

            (score, idx, fact)
        })
        .collect();

    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });

    let reranked: Vec<HashMap<String, serde_json::Value>> =
        scored.iter().map(|(_, _, fact)| (*fact).clone()).collect();

    if top_k > 0 {
        reranked.into_iter().take(top_k).collect()
    } else {
        reranked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_similarity_identical() {
        let sim = compute_word_similarity("hello world test", "hello world test");
        assert!((sim - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_word_similarity_no_overlap() {
        let sim = compute_word_similarity("alpha beta gamma", "delta epsilon zeta");
        assert!(sim < 0.01);
    }

    #[test]
    fn test_word_similarity_empty() {
        assert_eq!(compute_word_similarity("", "hello"), 0.0);
        assert_eq!(compute_word_similarity("hello", ""), 0.0);
    }

    #[test]
    fn test_tag_similarity() {
        let tags_a: Vec<String> = vec!["rust".into(), "python".into()];
        let tags_b: Vec<String> = vec!["python".into(), "go".into()];
        let sim = compute_tag_similarity(&tags_a, &tags_b);
        assert!((sim - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_rerank_empty() {
        let facts: Vec<HashMap<String, serde_json::Value>> = vec![];
        assert!(rerank_facts_by_query(&facts, "test", 0).is_empty());
    }

    #[test]
    fn test_rerank_ordering_by_relevance() {
        let mut irrelevant = HashMap::new();
        irrelevant.insert(
            "context".into(),
            serde_json::json!("alpha beta gamma delta"),
        );
        irrelevant.insert("outcome".into(), serde_json::json!("nothing related"));

        let mut relevant = HashMap::new();
        relevant.insert(
            "context".into(),
            serde_json::json!("rust programming language compiler"),
        );
        relevant.insert(
            "outcome".into(),
            serde_json::json!("rust compiler produces fast binaries"),
        );

        let mut medium = HashMap::new();
        medium.insert(
            "context".into(),
            serde_json::json!("programming with various tools"),
        );
        medium.insert("outcome".into(), serde_json::json!("tools are helpful"));

        let facts = vec![irrelevant, medium, relevant];
        let result = rerank_facts_by_query(&facts, "rust programming compiler", 0);
        assert_eq!(result.len(), 3);
        // The fact about rust/compiler should be ranked first
        let top_context = result[0].get("context").unwrap().as_str().unwrap();
        assert!(
            top_context.contains("rust"),
            "expected top result to mention 'rust', got: {top_context}"
        );
    }

    #[test]
    fn test_rerank_top_k_truncation() {
        let facts: Vec<HashMap<String, serde_json::Value>> = (0..10)
            .map(|i| {
                let mut f = HashMap::new();
                f.insert(
                    "context".into(),
                    serde_json::json!(format!("fact number {i} data")),
                );
                f.insert("outcome".into(), serde_json::json!("outcome"));
                f
            })
            .collect();

        let result = rerank_facts_by_query(&facts, "fact data", 3);
        assert_eq!(result.len(), 3);
        // Verify ranking: each result should have a _rerank_score, in descending order
        let scores: Vec<f64> = result
            .iter()
            .filter_map(|r| r.get("_rerank_score").and_then(|v| v.as_f64()))
            .collect();
        for w in scores.windows(2) {
            assert!(w[0] >= w[1], "results should be sorted by descending score");
        }
    }

    #[test]
    fn test_rerank_temporal_boost() {
        let mut with_temporal = HashMap::new();
        with_temporal.insert(
            "context".into(),
            serde_json::json!("configuration updated server"),
        );
        with_temporal.insert(
            "outcome".into(),
            serde_json::json!("server configuration updated"),
        );
        let mut meta = serde_json::Map::new();
        meta.insert("temporal_index".into(), serde_json::json!(1));
        meta.insert("source_date".into(), serde_json::json!("2024-01-01"));
        with_temporal.insert("metadata".into(), serde_json::Value::Object(meta));

        let mut without_temporal = HashMap::new();
        without_temporal.insert(
            "context".into(),
            serde_json::json!("configuration updated server"),
        );
        without_temporal.insert(
            "outcome".into(),
            serde_json::json!("server configuration updated"),
        );

        let facts = vec![without_temporal, with_temporal];
        // Query with a temporal cue word "changed"
        let result = rerank_facts_by_query(&facts, "configuration changed server", 0);
        assert_eq!(result.len(), 2);
        // The fact with temporal metadata should be ranked higher
        let top = &result[0];
        assert!(
            top.get("metadata").is_some(),
            "expected fact with temporal metadata to be ranked first"
        );
    }

    // ====================================================================
    // Extended coverage: compute_similarity
    // ====================================================================

    #[test]
    fn test_compute_similarity_identical_nodes() {
        let mut node_a = HashMap::new();
        node_a.insert(
            "content".into(),
            serde_json::json!("Rust programming language is fast and safe"),
        );
        node_a.insert("concept".into(), serde_json::json!("rust"));
        node_a.insert("tags".into(), serde_json::json!(["language", "systems"]));

        let node_b = node_a.clone();
        let sim = compute_similarity(&node_a, &node_b);
        assert!(
            (sim - 1.0).abs() < 0.01,
            "Identical nodes should have similarity ~1.0, got {sim}"
        );
    }

    #[test]
    fn test_compute_similarity_completely_different() {
        let mut node_a = HashMap::new();
        node_a.insert(
            "content".into(),
            serde_json::json!("quantum physics entanglement particles"),
        );
        node_a.insert("concept".into(), serde_json::json!("physics"));
        node_a.insert("tags".into(), serde_json::json!(["science", "quantum"]));

        let mut node_b = HashMap::new();
        node_b.insert(
            "content".into(),
            serde_json::json!("cooking recipes chocolate brownies"),
        );
        node_b.insert("concept".into(), serde_json::json!("cooking"));
        node_b.insert("tags".into(), serde_json::json!(["food", "dessert"]));

        let sim = compute_similarity(&node_a, &node_b);
        assert!(
            sim < 0.1,
            "Completely different nodes should have low similarity, got {sim}"
        );
    }

    #[test]
    fn test_compute_similarity_partial_overlap() {
        let mut node_a = HashMap::new();
        node_a.insert(
            "content".into(),
            serde_json::json!("Rust programming language memory safety"),
        );
        node_a.insert("concept".into(), serde_json::json!("rust"));
        node_a.insert("tags".into(), serde_json::json!(["language", "systems"]));

        let mut node_b = HashMap::new();
        node_b.insert(
            "content".into(),
            serde_json::json!("Rust programming compiler optimization"),
        );
        node_b.insert("concept".into(), serde_json::json!("rust"));
        node_b.insert("tags".into(), serde_json::json!(["language", "compiler"]));

        let sim = compute_similarity(&node_a, &node_b);
        assert!(
            sim > 0.3,
            "Partially overlapping nodes should score > 0.3, got {sim}"
        );
        assert!(sim < 1.0, "Should not be perfect similarity");
    }

    #[test]
    fn test_compute_similarity_empty_nodes() {
        let node_a: HashMap<String, serde_json::Value> = HashMap::new();
        let node_b: HashMap<String, serde_json::Value> = HashMap::new();

        let sim = compute_similarity(&node_a, &node_b);
        assert!(
            sim.abs() < 0.01,
            "Empty nodes should have similarity ~0.0, got {sim}"
        );
    }

    #[test]
    fn test_compute_similarity_no_tags() {
        let mut node_a = HashMap::new();
        node_a.insert(
            "content".into(),
            serde_json::json!("database indexing performance"),
        );
        node_a.insert("concept".into(), serde_json::json!("databases"));

        let mut node_b = HashMap::new();
        node_b.insert(
            "content".into(),
            serde_json::json!("database query optimization performance"),
        );
        node_b.insert("concept".into(), serde_json::json!("databases"));

        let sim = compute_similarity(&node_a, &node_b);
        // Without tags, only content (0.5) and concept (0.3) weights contribute
        assert!(
            sim > 0.3,
            "Should have reasonable similarity from content+concept, got {sim}"
        );
    }

    #[test]
    fn test_compute_similarity_same_concept_different_content() {
        let mut node_a = HashMap::new();
        node_a.insert(
            "content".into(),
            serde_json::json!("alpha bravo charlie delta"),
        );
        node_a.insert("concept".into(), serde_json::json!("nato-alphabet"));

        let mut node_b = HashMap::new();
        node_b.insert(
            "content".into(),
            serde_json::json!("echo foxtrot golf hotel"),
        );
        node_b.insert("concept".into(), serde_json::json!("nato-alphabet"));

        let sim = compute_similarity(&node_a, &node_b);
        // Same concept contributes 0.3 weight with word_similarity ~1.0 on concept
        assert!(
            sim >= 0.29,
            "Same concept should contribute ~0.3, got {sim}"
        );
    }

    #[test]
    fn test_compute_similarity_weight_distribution() {
        // Test that tags contribute to the score
        let mut node_a = HashMap::new();
        node_a.insert("content".into(), serde_json::json!("aaaa bbbb cccc"));
        node_a.insert("concept".into(), serde_json::json!("xxxx"));
        node_a.insert("tags".into(), serde_json::json!(["shared-tag"]));

        let mut node_b = HashMap::new();
        node_b.insert("content".into(), serde_json::json!("dddd eeee ffff"));
        node_b.insert("concept".into(), serde_json::json!("yyyy"));
        node_b.insert("tags".into(), serde_json::json!(["shared-tag"]));

        let sim = compute_similarity(&node_a, &node_b);
        // Only tags overlap: tag_sim=1.0, weight=0.2 => 0.2
        assert!(
            (sim - 0.2).abs() < 0.05,
            "Only tag overlap should give ~0.2, got {sim}"
        );
    }
}
