//! Pattern recognition from discoveries.

use chrono::{DateTime, Utc};
use std::collections::HashMap;

use crate::experience::{Experience, ExperienceType};

/// Detect recurring patterns from discoveries.
pub struct PatternDetector {
    threshold: usize,
    min_confidence: f64,
    patterns: HashMap<String, PatternData>,
    validations: HashMap<String, ValidationData>,
}

#[derive(Debug, Clone)]
struct PatternData {
    count: usize,
    examples: Vec<HashMap<String, serde_json::Value>>,
    first_seen: DateTime<Utc>,
    confidence: f64,
    base_confidence: f64,
}

#[derive(Debug, Clone, Default)]
struct ValidationData {
    successes: usize,
    failures: usize,
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
        for (key, data) in &self.patterns {
            if key.contains(pattern_type) {
                return data.count;
            }
        }
        0
    }

    /// Check if pattern has been recognized.
    pub fn is_pattern_recognized(&self, pattern_type: &str) -> bool {
        for (key, data) in &self.patterns {
            if key.contains(pattern_type) {
                return data.count >= self.threshold;
            }
        }
        false
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

        for (key, data) in self.patterns.iter_mut() {
            if key.contains(pattern_type) {
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

/// Extract normalized pattern key from discovery.
pub fn extract_pattern_key(discovery: &HashMap<String, serde_json::Value>) -> String {
    let disc_type = discovery
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    if let Some(link_type) = discovery.get("link_type").and_then(|v| v.as_str()) {
        return format!("{disc_type}_{link_type}");
    }

    if let Some(file) = discovery.get("file").and_then(|v| v.as_str()) {
        if file.ends_with(".md") {
            let prefix = if file.contains('_') {
                file.split('_').next().unwrap_or(file)
            } else {
                file.rsplit('.').next_back().unwrap_or(file)
            };
            return format!("{prefix}_{disc_type}");
        }
        return disc_type.to_string();
    }

    disc_type.to_string()
}

/// Calculate pattern confidence based on occurrences.
pub fn calculate_pattern_confidence(occurrences: usize, _threshold: usize) -> f64 {
    (0.5 + (occurrences as f64 * 0.1)).min(0.95)
}

/// Recognize new patterns from discoveries.
pub fn recognize_patterns(
    current_discoveries: &[HashMap<String, serde_json::Value>],
    known_patterns: Option<&[Experience]>,
    threshold: usize,
) -> Vec<Experience> {
    let mut detector = PatternDetector::new(threshold, 0.5);

    for discovery in current_discoveries {
        detector.add_discovery(discovery);
    }

    let mut new_patterns = detector.get_recognized_patterns(None);

    if let Some(known) = known_patterns {
        let known_keys: std::collections::HashSet<String> = known
            .iter()
            .filter_map(|p| {
                if let Some(start) = p.context.find("Pattern '") {
                    let start = start + "Pattern '".len();
                    if let Some(end) = p.context[start..].find('\'') {
                        return Some(p.context[start..start + end].to_string());
                    }
                }
                if p.context.to_lowercase().contains("known_pattern") {
                    return Some("known_pattern".to_string());
                }
                None
            })
            .collect();

        new_patterns.retain(|p| {
            if let Some(start) = p.context.find("Pattern '") {
                let start = start + "Pattern '".len();
                if let Some(end) = p.context[start..].find('\'') {
                    let key = &p.context[start..start + end];
                    return !known_keys.contains(key);
                }
            }
            true
        });
    }

    new_patterns
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_discovery(disc_type: &str) -> HashMap<String, serde_json::Value> {
        let mut d = HashMap::new();
        d.insert("type".into(), serde_json::json!(disc_type));
        d
    }

    #[test]
    fn test_pattern_detection() {
        let mut detector = PatternDetector::new(3, 0.5);
        for _ in 0..3 {
            detector.add_discovery(&make_discovery("test_type"));
        }
        assert!(detector.is_pattern_recognized("test_type"));
        assert_eq!(detector.get_occurrence_count("test_type"), 3);
    }

    #[test]
    fn test_below_threshold() {
        let mut detector = PatternDetector::new(3, 0.5);
        for _ in 0..2 {
            detector.add_discovery(&make_discovery("test_type"));
        }
        assert!(!detector.is_pattern_recognized("test_type"));
    }

    #[test]
    fn test_confidence_formula() {
        assert!((calculate_pattern_confidence(3, 3) - 0.8).abs() < 0.01);
        assert!((calculate_pattern_confidence(10, 3) - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_extract_pattern_key() {
        let mut d = HashMap::new();
        d.insert("type".into(), serde_json::json!("broken_link"));
        d.insert("link_type".into(), serde_json::json!("external"));
        assert_eq!(extract_pattern_key(&d), "broken_link_external");
    }

    #[test]
    fn test_validate_pattern() {
        let mut detector = PatternDetector::new(3, 0.5);
        for _ in 0..5 {
            detector.add_discovery(&make_discovery("validate_me"));
        }

        detector.validate_pattern("validate_me", true);
        detector.validate_pattern("validate_me", true);

        let patterns = detector.get_recognized_patterns(None);
        assert!(!patterns.is_empty());
    }

    #[test]
    fn test_recognize_patterns() {
        let discoveries: Vec<HashMap<String, serde_json::Value>> =
            (0..5).map(|_| make_discovery("frequent")).collect();

        let patterns = recognize_patterns(&discoveries, None, 3);
        assert!(!patterns.is_empty());
    }

    #[test]
    fn test_recognize_patterns_excludes_known() {
        // Create discoveries of two pattern types, each above threshold
        let mut discoveries = Vec::new();
        for _ in 0..5 {
            discoveries.push(make_discovery("known_type"));
        }
        for _ in 0..5 {
            discoveries.push(make_discovery("new_type"));
        }

        // First pass: discover both patterns
        let all_patterns = recognize_patterns(&discoveries, None, 3);
        assert!(
            all_patterns.len() >= 2,
            "expected at least 2 patterns, got {}",
            all_patterns.len()
        );

        // Build known_patterns from the "known_type" pattern
        let known: Vec<Experience> = all_patterns
            .iter()
            .filter(|p| p.context.contains("known_type"))
            .cloned()
            .collect();
        assert!(
            !known.is_empty(),
            "should have found the 'known_type' pattern"
        );

        // Second pass with known_patterns: should exclude known_type
        let new_patterns = recognize_patterns(&discoveries, Some(&known), 3);
        for p in &new_patterns {
            assert!(
                !p.context.contains("known_type"),
                "known pattern should have been excluded, but found: {}",
                p.context
            );
        }
    }
}
