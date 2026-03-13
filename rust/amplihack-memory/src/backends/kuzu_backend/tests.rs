//! Tests for the Kuzu backend.

use super::*;

fn make_backend() -> KuzuBackend {
    let tmp = tempfile::TempDir::new().unwrap();
    let db_path = tmp.path().join("test_kuzu_backend");
    let backend = KuzuBackend::new(&db_path, "test-agent", 100, true).unwrap();
    std::mem::forget(tmp);
    backend
}

#[test]
fn test_store_and_retrieve() {
    let mut backend = make_backend();
    let exp = Experience::new(
        ExperienceType::Success,
        "test context data".into(),
        "test outcome data".into(),
        0.9,
    )
    .unwrap();

    let id = backend.store_experience(&exp).unwrap();
    assert!(!id.is_empty());

    let results = backend.retrieve_experiences(Some(10), None, 0.0).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].context, "test context data");
    assert_eq!(results[0].outcome, "test outcome data");
}

#[test]
fn test_search() {
    let mut backend = make_backend();
    let exp1 = Experience::new(
        ExperienceType::Success,
        "rust programming rocks".into(),
        "compiled successfully".into(),
        0.9,
    )
    .unwrap();
    let exp2 = Experience::new(
        ExperienceType::Failure,
        "python scripting error".into(),
        "import failed".into(),
        0.5,
    )
    .unwrap();

    backend.store_experience(&exp1).unwrap();
    backend.store_experience(&exp2).unwrap();

    let results = MemoryBackend::search(&backend, "rust", None, 0.0, 10).unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].context.contains("rust"));
}

#[test]
fn test_filter_by_type() {
    let mut backend = make_backend();
    let exp1 = Experience::new(
        ExperienceType::Success,
        "success context".into(),
        "success outcome".into(),
        0.8,
    )
    .unwrap();
    let exp2 = Experience::new(
        ExperienceType::Failure,
        "failure context".into(),
        "failure outcome".into(),
        0.4,
    )
    .unwrap();

    backend.store_experience(&exp1).unwrap();
    backend.store_experience(&exp2).unwrap();

    let results = backend
        .retrieve_experiences(Some(10), Some(ExperienceType::Success), 0.0)
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].experience_type, ExperienceType::Success);
}

#[test]
fn test_statistics() {
    let mut backend = make_backend();
    let exp = Experience::new(
        ExperienceType::Pattern,
        "pattern context".into(),
        "pattern outcome".into(),
        0.95,
    )
    .unwrap();

    backend.store_experience(&exp).unwrap();

    let stats = MemoryBackend::get_statistics(&backend).unwrap();
    assert_eq!(stats.total_experiences, 1);
    assert_eq!(*stats.by_type.get(&ExperienceType::Pattern).unwrap(), 1);
}

#[test]
fn test_min_confidence_filter() {
    let mut backend = make_backend();
    let exp_low = Experience::new(
        ExperienceType::Insight,
        "low conf context".into(),
        "low conf outcome".into(),
        0.2,
    )
    .unwrap();
    let exp_high = Experience::new(
        ExperienceType::Insight,
        "high conf context".into(),
        "high conf outcome".into(),
        0.9,
    )
    .unwrap();

    backend.store_experience(&exp_low).unwrap();
    backend.store_experience(&exp_high).unwrap();

    let results = backend.retrieve_experiences(Some(10), None, 0.5).unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].confidence >= 0.5);
}

#[test]
fn test_cleanup() {
    let mut backend = make_backend();
    let exp = Experience::new(
        ExperienceType::Success,
        "cleanup test".into(),
        "cleanup outcome".into(),
        0.7,
    )
    .unwrap();
    backend.store_experience(&exp).unwrap();

    // Should not error
    backend.cleanup(false, None, None).unwrap();
}
