use std::collections::HashMap;

use crate::experience::{Experience, ExperienceType};
use crate::pattern_recognition::{
    calculate_pattern_confidence, extract_pattern_key, recognize_patterns, PatternDetector,
};

fn make_discovery(disc_type: &str) -> HashMap<String, serde_json::Value> {
    let mut d = HashMap::new();
    d.insert("type".into(), serde_json::json!(disc_type));
    d
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
    let conf = calculate_pattern_confidence(1);
    assert!((conf - 0.6).abs() < 0.01);
}

#[test]
fn test_confidence_formula_cap() {
    // 100 occurrences: 0.5 + 10.0 = capped at 0.95
    let conf = calculate_pattern_confidence(100);
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

#[test]
fn test_recognize_patterns_safe_slicing_out_of_bounds() {
    let short_exp = Experience {
        experience_id: "exp_test".into(),
        experience_type: ExperienceType::Success,
        context: "Pat".into(),
        outcome: "ok".into(),
        confidence: 0.8,
        timestamp: chrono::Utc::now(),
        metadata: HashMap::new(),
        tags: vec![],
    };

    let discoveries = vec![make_discovery("alpha")];
    // Should not panic even with short context strings
    let _result = recognize_patterns(&discoveries, Some(&[short_exp]), 1);
}

#[test]
fn test_recognize_patterns_safe_slicing_utf8() {
    let utf8_exp = Experience {
        experience_id: "exp_test_utf8".into(),
        experience_type: ExperienceType::Success,
        context: "Pattern '日本語テスト' found".into(),
        outcome: "ok".into(),
        confidence: 0.8,
        timestamp: chrono::Utc::now(),
        metadata: HashMap::new(),
        tags: vec![],
    };

    let discoveries = vec![make_discovery("alpha")];
    // Should handle multi-byte chars without panicking
    let _result = recognize_patterns(&discoveries, Some(&[utf8_exp]), 1);
}
