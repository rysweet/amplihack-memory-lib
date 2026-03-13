//! Tests for the Ladybug backend.

use super::*;

fn make_backend() -> (LadybugBackend, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().unwrap();
    let db_path = tmp.path().join("test_ladybug_backend");
    let backend = LadybugBackend::new(&db_path, "test-agent", true).unwrap();
    (backend, tmp)
}

fn make_experience(
    exp_type: ExperienceType,
    context: &str,
    outcome: &str,
    confidence: f64,
) -> Experience {
    Experience::new(exp_type, context.into(), outcome.into(), confidence).unwrap()
}

#[test]
fn test_create_backend() {
    let (_backend, _tmp) = make_backend();
}

#[test]
fn test_initialize_schema() {
    let (mut backend, _tmp) = make_backend();
    // Schema already initialized by new(); calling again should be idempotent.
    backend.initialize_schema().unwrap();
}

#[test]
fn test_store_and_retrieve() {
    let (mut backend, _tmp) = make_backend();
    let exp = make_experience(
        ExperienceType::Success,
        "test context data",
        "test outcome data",
        0.9,
    );

    let id = backend.store_experience(&exp).unwrap();
    assert!(!id.is_empty());

    let results = backend.retrieve_experiences(Some(10), None, 0.0).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].context, "test context data");
    assert_eq!(results[0].outcome, "test outcome data");
    assert_eq!(results[0].experience_type, ExperienceType::Success);
}

#[test]
fn test_store_validates_empty_context() {
    let (mut backend, _tmp) = make_backend();
    let mut exp = make_experience(ExperienceType::Success, "placeholder", "outcome", 0.5);
    exp.context = "".to_string();
    let result = backend.store_experience(&exp);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("context"),
        "error should mention context: {err}"
    );
}

#[test]
fn test_store_validates_confidence_range() {
    let (mut backend, _tmp) = make_backend();
    let mut exp = make_experience(ExperienceType::Success, "ctx", "outcome", 0.5);
    exp.confidence = 1.5;
    let result = backend.store_experience(&exp);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("confidence"),
        "error should mention confidence: {err}"
    );
}

#[test]
fn test_search_by_text() {
    let (mut backend, _tmp) = make_backend();
    backend
        .store_experience(&make_experience(
            ExperienceType::Success,
            "rust programming rocks",
            "compiled successfully",
            0.9,
        ))
        .unwrap();
    backend
        .store_experience(&make_experience(
            ExperienceType::Failure,
            "python scripting error",
            "import failed",
            0.5,
        ))
        .unwrap();

    let results = MemoryBackend::search(&backend, "rust", None, 0.0, 10).unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].context.contains("rust"));
}

#[test]
fn test_search_by_type() {
    let (mut backend, _tmp) = make_backend();
    backend
        .store_experience(&make_experience(
            ExperienceType::Success,
            "success ctx",
            "success out",
            0.8,
        ))
        .unwrap();
    backend
        .store_experience(&make_experience(
            ExperienceType::Failure,
            "failure ctx",
            "failure out",
            0.4,
        ))
        .unwrap();

    let results = backend
        .retrieve_experiences(Some(10), Some(ExperienceType::Success), 0.0)
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].experience_type, ExperienceType::Success);
}

#[test]
fn test_search_min_confidence() {
    let (mut backend, _tmp) = make_backend();
    backend
        .store_experience(&make_experience(
            ExperienceType::Insight,
            "low conf context",
            "low conf outcome",
            0.2,
        ))
        .unwrap();
    backend
        .store_experience(&make_experience(
            ExperienceType::Insight,
            "high conf context",
            "high conf outcome",
            0.9,
        ))
        .unwrap();

    let results = backend.retrieve_experiences(Some(10), None, 0.5).unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].confidence >= 0.5);
}

#[test]
fn test_retrieve_with_limit() {
    let (mut backend, _tmp) = make_backend();
    for i in 0..5 {
        backend
            .store_experience(&make_experience(
                ExperienceType::Success,
                &format!("context {i}"),
                &format!("outcome {i}"),
                0.8,
            ))
            .unwrap();
    }

    let results = backend.retrieve_experiences(Some(3), None, 0.0).unwrap();
    assert_eq!(results.len(), 3);
}

#[test]
fn test_get_statistics() {
    let (mut backend, _tmp) = make_backend();
    backend
        .store_experience(&make_experience(
            ExperienceType::Pattern,
            "pattern context",
            "pattern outcome",
            0.95,
        ))
        .unwrap();

    let stats = MemoryBackend::get_statistics(&backend).unwrap();
    assert_eq!(stats.total_experiences, 1);
    assert_eq!(*stats.by_type.get(&ExperienceType::Pattern).unwrap(), 1);
}

#[test]
fn test_cleanup_max_experiences() {
    let (mut backend, _tmp) = make_backend();
    for i in 0..5 {
        backend
            .store_experience(&make_experience(
                ExperienceType::Success,
                &format!("ctx {i}"),
                &format!("out {i}"),
                0.5,
            ))
            .unwrap();
    }

    backend.cleanup(false, None, Some(2)).unwrap();

    let results = backend.retrieve_experiences(None, None, 0.0).unwrap();
    assert!(results.len() <= 2, "expected <= 2, got {}", results.len());
}

#[test]
fn test_experience_backend_trait() {
    let (mut backend, _tmp) = make_backend();
    let exp = make_experience(
        ExperienceType::Insight,
        "trait context",
        "trait outcome",
        0.7,
    );

    let id = ExperienceBackend::add(&mut backend, &exp).unwrap();
    assert!(!id.is_empty());

    let results = ExperienceBackend::search(&backend, "trait", None, 0.0, 10).unwrap();
    assert_eq!(results.len(), 1);

    let stats = ExperienceBackend::get_statistics(&backend).unwrap();
    assert_eq!(stats.total_experiences, 1);
}

#[test]
fn test_roundtrip_metadata() {
    let (mut backend, _tmp) = make_backend();
    let mut exp = make_experience(ExperienceType::Success, "meta ctx", "meta out", 0.8);
    exp.metadata.insert(
        "tool".to_string(),
        serde_json::Value::String("cargo".to_string()),
    );
    exp.metadata
        .insert("count".to_string(), serde_json::json!(42));

    backend.store_experience(&exp).unwrap();

    let results = backend.retrieve_experiences(Some(1), None, 0.0).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].metadata.get("tool").and_then(|v| v.as_str()),
        Some("cargo")
    );
    assert_eq!(
        results[0].metadata.get("count").and_then(|v| v.as_i64()),
        Some(42)
    );
}

#[test]
fn test_roundtrip_tags() {
    let (mut backend, _tmp) = make_backend();
    let mut exp = make_experience(ExperienceType::Success, "tags ctx", "tags out", 0.8);
    exp.tags = vec!["rust".into(), "graph".into(), "memory".into()];

    backend.store_experience(&exp).unwrap();

    let results = backend.retrieve_experiences(Some(1), None, 0.0).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].tags, vec!["rust", "graph", "memory"]);
}

#[test]
fn test_multiple_stores() {
    let (mut backend, _tmp) = make_backend();
    for i in 0..10 {
        backend
            .store_experience(&make_experience(
                ExperienceType::Success,
                &format!("multi ctx {i}"),
                &format!("multi out {i}"),
                0.5 + (i as f64) * 0.05,
            ))
            .unwrap();
    }

    let results = backend.retrieve_experiences(None, None, 0.0).unwrap();
    assert_eq!(results.len(), 10);
}

#[test]
fn test_close_is_safe() {
    let (mut backend, _tmp) = make_backend();
    backend.close(); // should not panic
    backend.close(); // calling twice should also be fine
}

#[test]
fn test_empty_search() {
    let (backend, _tmp) = make_backend();
    let results = MemoryBackend::search(&backend, "nonexistent_xyz", None, 0.0, 10).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_store_with_special_characters() {
    let (mut backend, _tmp) = make_backend();
    let exp = make_experience(
        ExperienceType::Success,
        "context with 'quotes' and \\backslash",
        "outcome with \"double quotes\" and $dollar",
        0.7,
    );

    let id = backend.store_experience(&exp).unwrap();
    assert!(!id.is_empty());

    let results = backend.retrieve_experiences(Some(1), None, 0.0).unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].context.contains("'quotes'"));
    assert!(results[0].outcome.contains("$dollar"));
}
