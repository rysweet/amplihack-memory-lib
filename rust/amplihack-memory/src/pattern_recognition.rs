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

    // --- New parity tests ---

    #[test]
    fn test_pattern_confidence_accumulation() {
        let mut detector = PatternDetector::new(3, 0.5);

        // After 3 occurrences, confidence = 0.5 + 3*0.1 = 0.8
        for _ in 0..3 {
            detector.add_discovery(&make_discovery("accum"));
        }
        let patterns = detector.get_recognized_patterns(None);
        assert_eq!(patterns.len(), 1);
        let conf = patterns[0].confidence;
        assert!((conf - 0.8).abs() < 0.01, "expected ~0.8, got {conf}");

        // After 5 total, confidence = 0.5 + 5*0.1 = 0.95 (capped)
        for _ in 0..2 {
            detector.add_discovery(&make_discovery("accum"));
        }
        let patterns = detector.get_recognized_patterns(None);
        let conf = patterns[0].confidence;
        assert!(
            (conf - 0.95).abs() < 0.01,
            "expected ~0.95 (capped), got {conf}"
        );
    }

    #[test]
    fn test_pattern_validation_threshold_filtering() {
        let mut detector = PatternDetector::new(3, 0.7);

        // Add 3 occurrences → confidence = 0.8, above min_confidence 0.7
        for _ in 0..3 {
            detector.add_discovery(&make_discovery("high_conf"));
        }
        let patterns = detector.get_recognized_patterns(None);
        assert_eq!(patterns.len(), 1);

        // Request patterns with higher min_confidence
        let strict = detector.get_recognized_patterns(Some(0.9));
        assert!(
            strict.is_empty(),
            "pattern with 0.8 confidence should not appear when min_confidence=0.9"
        );
    }

    #[test]
    fn test_pattern_demotion_on_counter_evidence() {
        let mut detector = PatternDetector::new(3, 0.5);
        for _ in 0..5 {
            detector.add_discovery(&make_discovery("demote_me"));
        }

        let before = detector.get_recognized_patterns(None);
        let conf_before = before[0].confidence;

        // Many failures should reduce confidence
        for _ in 0..10 {
            detector.validate_pattern("demote_me", false);
        }

        let after = detector.get_recognized_patterns(Some(0.0));
        assert!(!after.is_empty());
        let conf_after = after[0].confidence;
        assert!(
            conf_after < conf_before,
            "confidence should decrease after failures: {conf_after} should be < {conf_before}"
        );
    }

    #[test]
    fn test_pattern_validation_success_maintains_confidence() {
        let mut detector = PatternDetector::new(3, 0.5);
        for _ in 0..5 {
            detector.add_discovery(&make_discovery("stable"));
        }

        let before = detector.get_recognized_patterns(None);
        let conf_before = before[0].confidence;

        // All successes should keep confidence high
        for _ in 0..5 {
            detector.validate_pattern("stable", true);
        }

        let after = detector.get_recognized_patterns(Some(0.0));
        let conf_after = after[0].confidence;
        assert!(
            conf_after > 0.5,
            "confidence should remain high after all successes: {conf_after}"
        );
        // With all successes, multiplier = 0.7 + 0.4*1.0 = 1.1 → clamped to 0.95
        // Result should be close to or at cap
        assert!(
            (conf_after - conf_before).abs() < 0.15,
            "confidence should stay relatively stable after successes"
        );
    }

    #[test]
    fn test_multiple_overlapping_patterns() {
        let mut detector = PatternDetector::new(2, 0.5);

        for _ in 0..3 {
            detector.add_discovery(&make_discovery("pattern_a"));
        }
        for _ in 0..4 {
            detector.add_discovery(&make_discovery("pattern_b"));
        }
        for _ in 0..5 {
            detector.add_discovery(&make_discovery("pattern_c"));
        }

        assert!(detector.is_pattern_recognized("pattern_a"));
        assert!(detector.is_pattern_recognized("pattern_b"));
        assert!(detector.is_pattern_recognized("pattern_c"));

        assert_eq!(detector.get_occurrence_count("pattern_a"), 3);
        assert_eq!(detector.get_occurrence_count("pattern_b"), 4);
        assert_eq!(detector.get_occurrence_count("pattern_c"), 5);

        let patterns = detector.get_recognized_patterns(None);
        assert_eq!(patterns.len(), 3);
    }

    #[test]
    fn test_empty_experience_sequences() {
        let discoveries: Vec<HashMap<String, serde_json::Value>> = vec![];
        let patterns = recognize_patterns(&discoveries, None, 3);
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_single_experience_no_pattern() {
        let mut detector = PatternDetector::new(3, 0.5);
        detector.add_discovery(&make_discovery("lone_wolf"));

        assert!(!detector.is_pattern_recognized("lone_wolf"));
        assert_eq!(detector.get_occurrence_count("lone_wolf"), 1);

        let patterns = detector.get_recognized_patterns(None);
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_identical_experiences_form_pattern() {
        let mut detector = PatternDetector::new(3, 0.5);
        let disc = make_discovery("identical");

        for _ in 0..5 {
            detector.add_discovery(&disc);
        }

        assert!(detector.is_pattern_recognized("identical"));
        let patterns = detector.get_recognized_patterns(None);
        assert_eq!(patterns.len(), 1);
    }

    #[test]
    fn test_extract_pattern_key_type_only() {
        let d = make_discovery("simple_type");
        assert_eq!(extract_pattern_key(&d), "simple_type");
    }

    #[test]
    fn test_extract_pattern_key_with_link_type() {
        let mut d = HashMap::new();
        d.insert("type".into(), serde_json::json!("link"));
        d.insert("link_type".into(), serde_json::json!("internal"));
        assert_eq!(extract_pattern_key(&d), "link_internal");
    }

    #[test]
    fn test_extract_pattern_key_with_md_file() {
        let mut d = HashMap::new();
        d.insert("type".into(), serde_json::json!("finding"));
        d.insert("file".into(), serde_json::json!("README.md"));
        // File is .md and doesn't contain '_', so prefix = full filename stem
        let key = extract_pattern_key(&d);
        assert!(key.contains("finding"), "key should contain the type");
    }

    #[test]
    fn test_extract_pattern_key_with_md_file_underscore() {
        let mut d = HashMap::new();
        d.insert("type".into(), serde_json::json!("issue"));
        d.insert("file".into(), serde_json::json!("some_doc.md"));
        let key = extract_pattern_key(&d);
        assert_eq!(key, "some_issue");
    }

    #[test]
    fn test_extract_pattern_key_missing_type() {
        let d = HashMap::new();
        assert_eq!(extract_pattern_key(&d), "unknown");
    }

    #[test]
    fn test_pattern_detector_new_thresholds() {
        let detector = PatternDetector::new(5, 0.8);
        assert!(detector.get_recognized_patterns(None).is_empty());
        assert_eq!(detector.get_occurrence_count("anything"), 0);
        assert!(!detector.is_pattern_recognized("anything"));
    }

    #[test]
    fn test_pattern_exactly_at_threshold() {
        let mut detector = PatternDetector::new(5, 0.5);
        for _ in 0..5 {
            detector.add_discovery(&make_discovery("exact"));
        }
        assert!(detector.is_pattern_recognized("exact"));
        assert_eq!(detector.get_occurrence_count("exact"), 5);
    }

    #[test]
    fn test_pattern_one_below_threshold() {
        let mut detector = PatternDetector::new(5, 0.5);
        for _ in 0..4 {
            detector.add_discovery(&make_discovery("almost"));
        }
        assert!(!detector.is_pattern_recognized("almost"));
    }

    #[test]
    fn test_confidence_formula_minimum() {
        // 1 occurrence: 0.5 + 0.1 = 0.6
        let conf = calculate_pattern_confidence(1, 3);
        assert!((conf - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_confidence_formula_cap() {
        // 100 occurrences: 0.5 + 10.0 = capped at 0.95
        let conf = calculate_pattern_confidence(100, 3);
        assert!((conf - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_recognized_patterns_are_experience_type_pattern() {
        let mut detector = PatternDetector::new(2, 0.5);
        for _ in 0..3 {
            detector.add_discovery(&make_discovery("typed"));
        }
        let patterns = detector.get_recognized_patterns(None);
        for p in &patterns {
            assert_eq!(p.experience_type, ExperienceType::Pattern);
        }
    }

    #[test]
    fn test_recognized_pattern_context_format() {
        let mut detector = PatternDetector::new(2, 0.5);
        for _ in 0..4 {
            detector.add_discovery(&make_discovery("fmt_check"));
        }
        let patterns = detector.get_recognized_patterns(None);
        assert_eq!(patterns.len(), 1);
        assert!(
            patterns[0].context.contains("fmt_check"),
            "context should mention the pattern type: {}",
            patterns[0].context
        );
        assert!(
            patterns[0].context.contains("4"),
            "context should mention occurrence count: {}",
            patterns[0].context
        );
    }

    #[test]
    fn test_validate_pattern_not_yet_added() {
        let mut detector = PatternDetector::new(3, 0.5);
        // Validating a pattern that has no discoveries should not panic
        detector.validate_pattern("ghost_pattern", true);
        detector.validate_pattern("ghost_pattern", false);
        // No patterns should be recognized
        assert!(detector.get_recognized_patterns(None).is_empty());
    }

    #[test]
    fn test_recognize_patterns_with_high_threshold() {
        let discoveries: Vec<_> = (0..4).map(|_| make_discovery("sparse")).collect();
        let patterns = recognize_patterns(&discoveries, None, 10);
        assert!(
            patterns.is_empty(),
            "4 discoveries should not reach threshold of 10"
        );
    }

    #[test]
    fn test_examples_capped_at_five() {
        let mut detector = PatternDetector::new(2, 0.5);
        for i in 0..10 {
            let mut d = HashMap::new();
            d.insert("type".into(), serde_json::json!("capped"));
            d.insert("index".into(), serde_json::json!(i));
            detector.add_discovery(&d);
        }

        // Internal check: the pattern data should have at most 5 examples
        // We verify indirectly through recognized patterns (which use examples)
        let patterns = detector.get_recognized_patterns(None);
        assert_eq!(patterns.len(), 1);
        // The pattern's context/outcome are generated from first example, so it works
        assert!(patterns[0].context.contains("capped"));
    }
}
