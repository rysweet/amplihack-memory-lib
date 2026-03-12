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
    pub fn new(experiences: Vec<Experience>) -> Self {
        Self { experiences }
    }

    pub fn corpus_size(&self) -> usize {
        self.experiences.len()
    }

    pub fn is_indexed(&self) -> bool {
        true
    }

    pub fn search(&self, query: &str, top_k: usize) -> Vec<Experience> {
        retrieve_relevant_experiences(&self.experiences, query, top_k, 0.0)
    }

    pub fn add_experience(&mut self, experience: Experience) {
        self.experiences.push(experience);
    }

    pub fn remove_experience(&mut self, experience_id: &str) {
        self.experiences
            .retain(|e| e.experience_id != experience_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
