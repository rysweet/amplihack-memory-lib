use std::collections::HashMap;

use crate::experience::Experience;
use crate::pattern_recognition::{
    calculate_pattern_confidence, extract_pattern_key, recognize_patterns, PatternDetector,
};

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
    assert!((calculate_pattern_confidence(3) - 0.8).abs() < 0.01);
    assert!((calculate_pattern_confidence(10) - 0.95).abs() < 0.01);
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
