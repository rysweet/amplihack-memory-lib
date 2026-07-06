//! Deterministic recall-quality measurement primitives.
//!
//! This module hosts the memory-measurement scoring math that both a fixed
//! benchmark and a live-telemetry rail need to share. Keeping it here — rather
//! than forked into a downstream consumer — is what lets a benchmark score and a
//! production self-measurement be *the same quantity*, so a claimed cognition
//! improvement can be validated on the benchmark AND observed live without the
//! two rails silently drifting apart.
//!
//! The primitive is intentionally decoupled from any consumer type: it scores
//! plain `(concept, content)` pairs, so it carries no dependency on a caller's
//! fact/record struct.

/// Tokenize a recall query: whitespace-split, lowercased, dropping
/// punctuation-only tokens (e.g. the wildcard `*`) so a wildcard/empty query
/// yields no tokens and is treated as "no measurable relevance target".
fn query_tokens(query: &str) -> Vec<String> {
    query
        .split_whitespace()
        .map(str::to_lowercase)
        .filter(|t| t.chars().any(char::is_alphanumeric))
        .collect()
}

/// Whether a `(concept, content)` item is query-relevant under the keyword
/// proxy: its lowercased `concept` or `content` contains at least one query
/// token as a substring (mirroring a keyword recall gate).
fn item_is_relevant(concept: &str, content: &str, tokens: &[String]) -> bool {
    let concept = concept.to_lowercase();
    let content = content.to_lowercase();
    tokens
        .iter()
        .any(|t| concept.contains(t.as_str()) || content.contains(t.as_str()))
}

/// Precision@k for a ranked recall result over decoupled `(concept, content)`
/// pairs: of the top-`k` returned `items`, the fraction that are
/// **query-relevant**.
///
/// Relevance is a coarse keyword proxy (see [`item_is_relevant`]): an item counts
/// when its `concept`/`content` contains a query token as a substring. It is
/// deliberately broader than an exact-token ranker score (e.g. `cat` matches
/// `concatenate`), which is acceptable for a self-metric baseline: it needs no
/// external ground-truth labels — the query itself is the relevance oracle — and
/// it moves in the same direction as ranking quality.
///
/// Returns `None` (undefined, **not** `0.0`) when the query has no usable tokens
/// (empty / wildcard `*`) or the result set is empty, so callers skip emitting a
/// meaningless sample rather than dragging the mean toward zero. `k` is clamped
/// to the number of returned items.
pub fn precision_at_k(query: &str, items: &[(&str, &str)], k: usize) -> Option<f64> {
    let tokens = query_tokens(query);
    if tokens.is_empty() {
        return None;
    }
    let window = k.min(items.len());
    if window == 0 {
        return None;
    }
    let relevant = items[..window]
        .iter()
        .filter(|(concept, content)| item_is_relevant(concept, content, &tokens))
        .count();
    Some(relevant as f64 / window as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The parity gate: these nine pure-math cases are the authoritative
    // semantics of precision@k. A downstream adapter that delegates here is
    // expected to reproduce every one of them.

    #[test]
    fn all_relevant_is_one() {
        let items = [
            ("kafka streaming", "backpressure"),
            ("kafka broker", "partition rebalance"),
        ];
        assert_eq!(precision_at_k("kafka", &items, 2), Some(1.0));
    }

    #[test]
    fn half_relevant_over_window() {
        let items = [
            ("kafka streaming", "backpressure"),
            ("kafka broker", "rebalance"),
            ("postgres index", "btree bloat"),
            ("redis cache", "eviction"),
        ];
        assert_eq!(precision_at_k("kafka", &items, 2), Some(1.0));
        assert_eq!(precision_at_k("kafka", &items, 4), Some(0.5));
    }

    #[test]
    fn matches_on_content_not_only_concept() {
        let items = [("infra note", "the kafka consumer lagged")];
        assert_eq!(precision_at_k("kafka", &items, 1), Some(1.0));
    }

    #[test]
    fn zero_when_top_k_all_irrelevant() {
        let items = [("postgres", "vacuum"), ("redis", "ttl")];
        assert_eq!(precision_at_k("kafka", &items, 2), Some(0.0));
    }

    #[test]
    fn clamps_k_to_result_len() {
        let items = [("kafka", "lag")];
        assert_eq!(precision_at_k("kafka", &items, 10), Some(1.0));
    }

    #[test]
    fn none_for_empty_results() {
        let items: [(&str, &str); 0] = [];
        assert_eq!(precision_at_k("kafka", &items, 5), None);
    }

    #[test]
    fn none_for_wildcard_or_empty_query() {
        let items = [("kafka", "lag")];
        assert_eq!(precision_at_k("*", &items, 1), None);
        assert_eq!(precision_at_k("   ", &items, 1), None);
        assert_eq!(precision_at_k("", &items, 1), None);
    }

    #[test]
    fn multi_token_query_is_case_insensitive() {
        let items = [("Kafka Streaming", "Backpressure"), ("unrelated", "topic")];
        assert_eq!(precision_at_k("KAFKA streaming", &items, 2), Some(0.5));
    }

    #[test]
    fn window_clamps_to_shorter_result_set() {
        let items = [("kafka one", "x"), ("kafka two", "y")];
        // k beyond the two results clamps to 2; both relevant → 1.0.
        assert_eq!(precision_at_k("kafka", &items, 9), Some(1.0));
    }
}
