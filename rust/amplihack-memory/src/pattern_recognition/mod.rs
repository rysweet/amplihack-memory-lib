//! Pattern recognition from discoveries.

mod detection;
mod scoring;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_extended;

use chrono::{DateTime, Utc};
use std::collections::HashMap;

use crate::experience::{Experience, ExperienceType};

pub use detection::recognize_patterns;
pub use scoring::{calculate_pattern_confidence, extract_pattern_key};

/// Detect recurring patterns from discoveries.
pub struct PatternDetector {
    threshold: usize,
    min_confidence: f64,
    patterns: HashMap<String, PatternData>,
    validations: HashMap<String, ValidationData>,
}

#[derive(Debug, Clone)]
pub(crate) struct PatternData {
    pub(crate) count: usize,
    pub(crate) examples: Vec<HashMap<String, serde_json::Value>>,
    pub(crate) first_seen: DateTime<Utc>,
    pub(crate) confidence: f64,
    pub(crate) base_confidence: f64,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ValidationData {
    pub(crate) successes: usize,
    pub(crate) failures: usize,
}

impl PatternDetector {
    /// Create a detector that requires `threshold` observations and
    /// `min_confidence` (0.0–1.0) before promoting a pattern.
    pub fn new(threshold: usize, min_confidence: f64) -> Self {
        Self {
            threshold,
            min_confidence,
            patterns: HashMap::new(),
            validations: HashMap::new(),
        }
    }

    /// Add a discovery for pattern tracking.
    pub fn add_discovery(&mut self, discovery: &HashMap<String, serde_json::Value>) {
        let key = extract_pattern_key(discovery);

        let pattern = self.patterns.entry(key).or_insert_with(|| PatternData {
            count: 0,
            examples: Vec::new(),
            first_seen: Utc::now(),
            confidence: 0.0,
            base_confidence: 0.0,
        });

        pattern.count += 1;

        if pattern.examples.len() < 5 {
            pattern.examples.push(discovery.clone());
        }

        if pattern.count >= self.threshold {
            let base_conf = calculate_pattern_confidence(pattern.count, self.threshold);
            pattern.base_confidence = base_conf;
            pattern.confidence = base_conf;
        }
    }

    /// Get occurrence count for a pattern type.
    pub fn get_occurrence_count(&self, pattern_type: &str) -> usize {
        self.patterns.get(pattern_type).map_or(0, |d| d.count)
    }

    /// Check if pattern has been recognized.
    pub fn is_pattern_recognized(&self, pattern_type: &str) -> bool {
        self.patterns
            .get(pattern_type)
            .is_some_and(|d| d.count >= self.threshold)
    }

    /// Get all recognized patterns as experiences.
    pub fn get_recognized_patterns(&self, min_confidence: Option<f64>) -> Vec<Experience> {
        let min_conf = min_confidence.unwrap_or(self.min_confidence);
        let mut patterns = Vec::new();

        for (key, data) in &self.patterns {
            if data.count < self.threshold || data.confidence < min_conf {
                continue;
            }

            let context = self.create_pattern_context(key, data);
            let outcome = self.create_pattern_outcome(key, data);

            let mut metadata = HashMap::new();
            metadata.insert("occurrences".into(), serde_json::json!(data.count));
            metadata.insert(
                "first_seen".into(),
                serde_json::json!(data.first_seen.to_rfc3339()),
            );

            let exp = Experience::from_parts(
                format!("pattern_{}", &uuid::Uuid::new_v4().to_string()[..8]),
                ExperienceType::Pattern,
                context,
                outcome,
                data.confidence,
                data.first_seen,
                metadata,
                vec![],
            );
            if let Ok(e) = exp {
                patterns.push(e);
            }
        }

        patterns
    }

    /// Update pattern confidence based on validation.
    pub fn validate_pattern(&mut self, pattern_type: &str, success: bool) {
        let validation = self
            .validations
            .entry(pattern_type.to_string())
            .or_default();

        if success {
            validation.successes += 1;
        } else {
            validation.failures += 1;
        }

        let total = validation.successes + validation.failures;
        let success_rate = validation.successes as f64 / total as f64;
        let failures = validation.failures;

        if let Some(data) = self.patterns.get_mut(pattern_type) {
            let base_conf = if data.base_confidence > 0.0 {
                data.base_confidence
            } else {
                data.confidence
            };
            let mut multiplier = 0.7 + (0.4 * success_rate);
            if failures > 5 {
                let penalty = ((failures - 5) as f64 * 0.02).min(0.2);
                multiplier -= penalty;
            }
            data.confidence = (base_conf * multiplier).clamp(0.1, 0.95);
        }
    }

    fn create_pattern_context(&self, _key: &str, data: &PatternData) -> String {
        if data.examples.is_empty() {
            return "Pattern detected".to_string();
        }

        let example = &data.examples[0];
        let pattern_type = example
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        format!(
            "Pattern '{}' occurs frequently ({} times)",
            pattern_type, data.count
        )
    }

    fn create_pattern_outcome(&self, _key: &str, data: &PatternData) -> String {
        format!(
            "Check for this pattern in similar contexts - it occurs in {} places",
            data.count
        )
    }
}
