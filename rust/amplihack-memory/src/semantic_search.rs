//! Semantic search and relevance scoring for experiences.

use chrono::Utc;
use std::collections::HashSet;

use crate::experience::{Experience, ExperienceType};

/// Simple TF-IDF similarity calculator (word overlap approximation).
pub struct TfIdfSimilarity;

impl TfIdfSimilarity {
    /// Calculate similarity between two texts using Jaccard similarity.
    pub fn calculate(text1: &str, text2: &str) -> f64 {
        if text1.is_empty() || text2.is_empty() {
            return 0.0;
        }

        let words1: HashSet<String> = text1
            .to_lowercase()
            .split_whitespace()
            .map(String::from)
            .collect();
        let words2: HashSet<String> = text2
            .to_lowercase()
            .split_whitespace()
            .map(String::from)
            .collect();

        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }
}

/// Calculate relevance score for experience given query.
pub fn calculate_relevance(experience: &Experience, query: &str) -> f64 {
    let mut similarity = TfIdfSimilarity::calculate(&experience.context, query);

    // Type weighting
    match experience.experience_type {
        ExperienceType::Pattern => similarity *= 1.5,
        ExperienceType::Insight => similarity *= 1.3,
        _ => {}
    }

    // Confidence boost
    similarity *= 0.5 + experience.confidence * 0.5;

    // Recency boost (decay over 90 days)
    let age_days = (Utc::now() - experience.timestamp).num_days().max(0);
    let recency_factor = (1.0 - (age_days as f64 / 90.0) * 0.3).max(0.7);
    similarity *= recency_factor;

    similarity.min(1.0)
}

/// Retrieve most relevant experiences for current context.
pub fn retrieve_relevant_experiences(
    experiences: &[Experience],
    current_context: &str,
    top_k: usize,
    min_similarity: f64,
) -> Vec<Experience> {
    let mut scored: Vec<(f64, &Experience)> = experiences
        .iter()
        .map(|exp| (calculate_relevance(exp, current_context), exp))
        .filter(|(relevance, _)| *relevance >= min_similarity)
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    scored
        .into_iter()
        .take(top_k)
        .map(|(_, exp)| exp.clone())
        .collect()
}

/// Search engine with index for fast similarity search.
pub struct SemanticSearchEngine {
    experiences: Vec<Experience>,
}

impl SemanticSearchEngine {
    /// Create a new search engine pre-loaded with the given experiences.
    ///
    /// The engine is immediately ready for queries after construction.
    pub fn new(experiences: Vec<Experience>) -> Self {
        Self { experiences }
    }

    /// Return the number of experiences currently in the corpus.
    pub fn corpus_size(&self) -> usize {
        self.experiences.len()
    }

    /// Check whether the engine is indexed and ready for queries.
    ///
    /// This implementation uses a linear scan, so it is always considered
    /// indexed. Returns `true` unconditionally.
    pub fn is_indexed(&self) -> bool {
        true
    }

    /// Search the corpus for experiences most relevant to `query`.
    ///
    /// Returns up to `top_k` experiences sorted by descending relevance score.
    /// Relevance is computed via [`calculate_relevance`], which combines
    /// Jaccard word similarity, experience-type weighting, confidence, and
    /// recency decay.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<Experience> {
        retrieve_relevant_experiences(&self.experiences, query, top_k, 0.0)
    }

    /// Append an experience to the corpus so it becomes searchable.
    pub fn add_experience(&mut self, experience: Experience) {
        self.experiences.push(experience);
    }

    /// Remove an experience from the corpus by its id.
    ///
    /// If no experience with the given id exists, this is a no-op.
    pub fn remove_experience(&mut self, experience_id: &str) {
        self.experiences
            .retain(|e| e.experience_id != experience_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_exp(id: &str, etype: ExperienceType, context: &str, confidence: f64) -> Experience {
        Experience::from_parts(
            id.into(),
            etype,
            context.into(),
            "outcome".into(),
            confidence,
            Utc::now(),
            Default::default(),
            vec![],
        )
        .unwrap()
    }

    #[test]
    fn test_tfidf_identical() {
        let sim = TfIdfSimilarity::calculate("hello world", "hello world");
        assert!((sim - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_tfidf_empty() {
        assert_eq!(TfIdfSimilarity::calculate("", "hello"), 0.0);
    }

    #[test]
    fn test_relevance_pattern_boost() {
        let pattern_exp = Experience::from_parts(
            "e1".into(),
            ExperienceType::Pattern,
            "test context".into(),
            "test outcome".into(),
            0.9,
            Utc::now(),
            Default::default(),
            vec![],
        )
        .unwrap();
        let success_exp = Experience::from_parts(
            "e2".into(),
            ExperienceType::Success,
            "test context".into(),
            "test outcome".into(),
            0.9,
            Utc::now(),
            Default::default(),
            vec![],
        )
        .unwrap();

        let pattern_rel = calculate_relevance(&pattern_exp, "test context");
        let success_rel = calculate_relevance(&success_exp, "test context");
        assert!(pattern_rel > success_rel);
    }

    #[test]
    fn test_search_engine() {
        let exp = Experience::from_parts(
            "e1".into(),
            ExperienceType::Success,
            "rust programming".into(),
            "compiled successfully".into(),
            0.9,
            Utc::now(),
            Default::default(),
            vec![],
        )
        .unwrap();
        let engine = SemanticSearchEngine::new(vec![exp]);
        assert_eq!(engine.corpus_size(), 1);
        assert!(engine.is_indexed());

        let results = engine.search("rust", 10);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_add_and_remove_experience() {
        let mut engine = SemanticSearchEngine::new(vec![]);
        assert_eq!(engine.corpus_size(), 0);

        let exp = Experience::from_parts(
            "exp-add-rm".into(),
            ExperienceType::Insight,
            "neural network training".into(),
            "model converged".into(),
            0.85,
            Utc::now(),
            Default::default(),
            vec![],
        )
        .unwrap();

        engine.add_experience(exp);
        assert_eq!(engine.corpus_size(), 1);

        let results = engine.search("neural network", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].experience_id, "exp-add-rm");

        engine.remove_experience("exp-add-rm");
        assert_eq!(engine.corpus_size(), 0);

        let results_after = engine.search("neural network", 10);
        assert!(
            results_after.is_empty(),
            "search should return nothing after removal"
        );
    }

    // --- New parity tests ---

    #[test]
    fn test_tfidf_completely_different() {
        let sim = TfIdfSimilarity::calculate("alpha beta gamma", "delta epsilon zeta");
        assert!(sim < 0.01, "disjoint word sets should have ~0 similarity");
    }

    #[test]
    fn test_tfidf_partial_overlap() {
        let sim = TfIdfSimilarity::calculate("hello world foo", "hello world bar");
        // Jaccard: intersection=2 (hello,world), union=4 (hello,world,foo,bar) => 0.5
        assert!((sim - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_tfidf_both_empty() {
        assert_eq!(TfIdfSimilarity::calculate("", ""), 0.0);
    }

    #[test]
    fn test_tfidf_case_insensitivity() {
        let sim = TfIdfSimilarity::calculate("Hello WORLD", "hello world");
        assert!(
            (sim - 1.0).abs() < 0.01,
            "case should not affect similarity"
        );
    }

    #[test]
    fn test_tfidf_single_word_match() {
        let sim = TfIdfSimilarity::calculate("hello", "hello");
        assert!((sim - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_tfidf_single_word_mismatch() {
        let sim = TfIdfSimilarity::calculate("hello", "goodbye");
        assert!(sim < 0.01);
    }

    #[test]
    fn test_search_engine_creation_empty() {
        let engine = SemanticSearchEngine::new(vec![]);
        assert_eq!(engine.corpus_size(), 0);
        assert!(engine.is_indexed());
    }

    #[test]
    fn test_search_engine_empty_index_search() {
        let engine = SemanticSearchEngine::new(vec![]);
        let results = engine.search("anything", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_engine_index_multiple_documents() {
        let experiences = vec![
            make_exp("e1", ExperienceType::Success, "rust memory safety", 0.9),
            make_exp("e2", ExperienceType::Insight, "python dynamic typing", 0.8),
            make_exp(
                "e3",
                ExperienceType::Failure,
                "java null pointer exception",
                0.7,
            ),
        ];
        let engine = SemanticSearchEngine::new(experiences);
        assert_eq!(engine.corpus_size(), 3);
    }

    #[test]
    fn test_search_result_ordering_by_relevance() {
        let experiences = vec![
            make_exp(
                "e1",
                ExperienceType::Success,
                "rust systems programming language",
                0.9,
            ),
            make_exp("e2", ExperienceType::Success, "rust", 0.9),
            make_exp("e3", ExperienceType::Success, "python is great", 0.9),
        ];
        let engine = SemanticSearchEngine::new(experiences);

        let results = engine.search("rust", 10);
        // "rust" alone should be a perfect match to query "rust"
        // The result with "rust" context should rank highest (Jaccard = 1.0)
        assert!(!results.is_empty());
        assert_eq!(results[0].experience_id, "e2");
    }

    #[test]
    fn test_search_with_similarity_threshold() {
        let experiences = vec![
            make_exp(
                "e1",
                ExperienceType::Success,
                "rust programming language",
                0.9,
            ),
            make_exp(
                "e2",
                ExperienceType::Success,
                "completely unrelated topic",
                0.9,
            ),
        ];

        // Use retrieve_relevant_experiences directly for min_similarity control
        let results = retrieve_relevant_experiences(&experiences, "rust programming", 10, 0.3);
        for r in &results {
            let rel = calculate_relevance(r, "rust programming");
            assert!(rel >= 0.3);
        }
    }

    #[test]
    fn test_search_top_k_limiting() {
        let mut experiences = Vec::new();
        for i in 0..20 {
            experiences.push(make_exp(
                &format!("e{i}"),
                ExperienceType::Success,
                &format!("common shared word {i}"),
                0.9,
            ));
        }
        let engine = SemanticSearchEngine::new(experiences);

        let results = engine.search("common shared word", 5);
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_relevance_insight_boost() {
        let insight = make_exp("i1", ExperienceType::Insight, "test query words", 0.9);
        let success = make_exp("s1", ExperienceType::Success, "test query words", 0.9);

        let insight_rel = calculate_relevance(&insight, "test query words");
        let success_rel = calculate_relevance(&success, "test query words");
        assert!(
            insight_rel > success_rel,
            "insight should be boosted above success"
        );
    }

    #[test]
    fn test_relevance_confidence_boost() {
        let high_conf = make_exp("h1", ExperienceType::Success, "same context here", 1.0);
        let low_conf = make_exp("l1", ExperienceType::Success, "same context here", 0.1);

        let high_rel = calculate_relevance(&high_conf, "same context here");
        let low_rel = calculate_relevance(&low_conf, "same context here");
        assert!(
            high_rel > low_rel,
            "higher confidence should yield higher relevance"
        );
    }

    #[test]
    fn test_search_engine_duplicate_documents() {
        let mut engine = SemanticSearchEngine::new(vec![]);
        let exp1 = make_exp("dup1", ExperienceType::Success, "duplicate content", 0.9);
        let exp2 = make_exp("dup2", ExperienceType::Success, "duplicate content", 0.9);

        engine.add_experience(exp1);
        engine.add_experience(exp2);
        assert_eq!(engine.corpus_size(), 2);

        let results = engine.search("duplicate content", 10);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_engine_remove_nonexistent() {
        let mut engine = SemanticSearchEngine::new(vec![]);
        engine.add_experience(make_exp("e1", ExperienceType::Success, "data", 0.9));
        engine.remove_experience("nonexistent-id");
        assert_eq!(
            engine.corpus_size(),
            1,
            "removing nonexistent id should be a no-op"
        );
    }

    #[test]
    fn test_retrieve_relevant_min_similarity_filters() {
        let experiences = vec![
            make_exp("e1", ExperienceType::Success, "exact match query", 0.9),
            make_exp(
                "e2",
                ExperienceType::Success,
                "completely different words",
                0.9,
            ),
        ];

        // Very high threshold should filter out low-relevance matches
        let results = retrieve_relevant_experiences(&experiences, "exact match query", 10, 0.9);
        for r in &results {
            assert!(r.context.contains("exact"));
        }
    }

    #[test]
    fn test_search_engine_large_corpus() {
        let mut experiences = Vec::new();
        for i in 0..100 {
            experiences.push(make_exp(
                &format!("exp{i}"),
                ExperienceType::Success,
                &format!("document number {i} with some content"),
                0.5 + (i as f64) * 0.005,
            ));
        }
        let engine = SemanticSearchEngine::new(experiences);
        assert_eq!(engine.corpus_size(), 100);

        let results = engine.search("document number content", 5);
        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_search_engine_add_then_search() {
        let mut engine = SemanticSearchEngine::new(vec![]);
        assert!(engine.search("anything", 10).is_empty());

        engine.add_experience(make_exp(
            "a1",
            ExperienceType::Success,
            "newly added item",
            0.8,
        ));
        let results = engine.search("newly added", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].experience_id, "a1");
    }

    #[test]
    fn test_relevance_zero_similarity_query() {
        let exp = make_exp("e1", ExperienceType::Success, "alpha beta gamma", 0.9);
        let rel = calculate_relevance(&exp, "xyznonexistent");
        assert!(rel < 0.01, "no word overlap should yield ~0 relevance");
    }
}
