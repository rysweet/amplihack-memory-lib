use super::*;
use crate::experience::{Experience, ExperienceType};
use tempfile::TempDir;

#[test]
fn test_connector_creation() {
    let tmp = TempDir::new().unwrap();
    let connector = MemoryConnector::new("test-agent", Some(tmp.path()), 100, true).unwrap();
    assert_eq!(connector.agent_name, "test-agent");
}

#[test]
fn test_empty_agent_name() {
    let tmp = TempDir::new().unwrap();
    let result = MemoryConnector::new("", Some(tmp.path()), 100, true);
    assert!(result.is_err());
}

#[test]
fn test_store_and_retrieve() {
    let tmp = TempDir::new().unwrap();
    let mut connector = MemoryConnector::new("test-agent", Some(tmp.path()), 100, true).unwrap();

    let exp = Experience::new(
        ExperienceType::Success,
        "test context".into(),
        "test outcome".into(),
        0.9,
    )
    .unwrap();

    connector.store_experience(&exp).unwrap();

    let results = connector.retrieve_experiences(Some(10), None, 0.0).unwrap();
    assert_eq!(results.len(), 1);
}

// --- New parity tests ---

#[test]
fn test_connector_creation_with_path() {
    let tmp = TempDir::new().unwrap();
    let connector = MemoryConnector::new("path-agent", Some(tmp.path()), 50, false).unwrap();
    assert_eq!(connector.agent_name, "path-agent");
    assert_eq!(connector.storage_path, tmp.path());
    assert!(connector.db_path.exists());
}

#[test]
fn test_connector_with_backend_sqlite() {
    let tmp = TempDir::new().unwrap();
    let connector = MemoryConnector::with_backend(
        "backend-agent",
        Some(tmp.path()),
        100,
        true,
        BackendType::Sqlite,
    )
    .unwrap();
    assert_eq!(connector.agent_name, "backend-agent");
}

#[test]
fn test_connector_whitespace_agent_name() {
    let tmp = TempDir::new().unwrap();
    let result = MemoryConnector::new("   ", Some(tmp.path()), 100, true);
    assert!(result.is_err(), "whitespace-only agent name should fail");
}

#[test]
fn test_connector_invalid_max_memory() {
    let tmp = TempDir::new().unwrap();
    let result = MemoryConnector::new("agent", Some(tmp.path()), 0, true);
    assert!(result.is_err(), "zero max_memory_mb should fail");

    let result2 = MemoryConnector::new("agent", Some(tmp.path()), -1, true);
    assert!(result2.is_err(), "negative max_memory_mb should fail");
}

#[test]
fn test_connector_store_retrieve_roundtrip() {
    let tmp = TempDir::new().unwrap();
    let mut connector =
        MemoryConnector::new("roundtrip-agent", Some(tmp.path()), 100, true).unwrap();

    let exp = Experience::new(
        ExperienceType::Insight,
        "insight about architecture".into(),
        "modular design works best".into(),
        0.85,
    )
    .unwrap();

    let stored_id = connector.store_experience(&exp).unwrap();
    assert!(!stored_id.is_empty());

    let results = connector.retrieve_experiences(Some(10), None, 0.0).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].context, "insight about architecture");
    assert_eq!(results[0].outcome, "modular design works best");
    assert_eq!(results[0].experience_type, ExperienceType::Insight);
    assert!((results[0].confidence - 0.85).abs() < 0.01);
}

#[test]
fn test_connector_store_multiple_types() {
    let tmp = TempDir::new().unwrap();
    let mut connector =
        MemoryConnector::new("multi-type-agent", Some(tmp.path()), 100, true).unwrap();

    let types = [
        ExperienceType::Success,
        ExperienceType::Failure,
        ExperienceType::Pattern,
        ExperienceType::Insight,
    ];

    for et in &types {
        let exp = Experience::new(
            *et,
            format!("{} context", et.as_str()),
            format!("{} outcome", et.as_str()),
            0.7,
        )
        .unwrap();
        connector.store_experience(&exp).unwrap();
    }

    let all = connector
        .retrieve_experiences(Some(100), None, 0.0)
        .unwrap();
    assert_eq!(all.len(), 4);

    // Filter by type
    let successes = connector
        .retrieve_experiences(Some(100), Some(ExperienceType::Success), 0.0)
        .unwrap();
    assert_eq!(successes.len(), 1);
    assert_eq!(successes[0].experience_type, ExperienceType::Success);
}

#[test]
fn test_connector_search_functionality() {
    let tmp = TempDir::new().unwrap();
    let mut connector = MemoryConnector::new("search-agent", Some(tmp.path()), 100, true).unwrap();

    let exp1 = Experience::new(
        ExperienceType::Success,
        "database migration completed".into(),
        "schema updated".into(),
        0.9,
    )
    .unwrap();
    let exp2 = Experience::new(
        ExperienceType::Failure,
        "network timeout error".into(),
        "retry logic needed".into(),
        0.6,
    )
    .unwrap();

    connector.store_experience(&exp1).unwrap();
    connector.store_experience(&exp2).unwrap();

    let db_results = connector.search("database", None, 0.0, 10).unwrap();
    assert_eq!(db_results.len(), 1);
    assert!(db_results[0].context.contains("database"));

    let net_results = connector.search("network", None, 0.0, 10).unwrap();
    assert_eq!(net_results.len(), 1);
    assert!(net_results[0].context.contains("network"));
}

#[test]
fn test_connector_search_with_type_filter() {
    let tmp = TempDir::new().unwrap();
    let mut connector = MemoryConnector::new("filter-agent", Some(tmp.path()), 100, true).unwrap();

    let exp1 = Experience::new(
        ExperienceType::Success,
        "context alpha".into(),
        "outcome alpha".into(),
        0.9,
    )
    .unwrap();
    let exp2 = Experience::new(
        ExperienceType::Failure,
        "context alpha".into(),
        "outcome beta".into(),
        0.8,
    )
    .unwrap();

    connector.store_experience(&exp1).unwrap();
    connector.store_experience(&exp2).unwrap();

    let filtered = connector
        .search("alpha", Some(ExperienceType::Failure), 0.0, 10)
        .unwrap();
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].experience_type, ExperienceType::Failure);
}

#[test]
fn test_connector_statistics() {
    let tmp = TempDir::new().unwrap();
    let mut connector = MemoryConnector::new("stats-agent", Some(tmp.path()), 100, false).unwrap();

    let stats_empty = connector.get_statistics().unwrap();
    assert_eq!(stats_empty.total_experiences, 0);

    for i in 0..5 {
        let exp = Experience::new(
            ExperienceType::Success,
            format!("ctx{i}"),
            format!("out{i}"),
            0.8,
        )
        .unwrap();
        connector.store_experience(&exp).unwrap();
    }

    let stats = connector.get_statistics().unwrap();
    assert_eq!(stats.total_experiences, 5);
    assert!(stats.storage_size_kb > 0.0);
}

#[test]
fn test_connector_close_and_reopen() {
    let tmp = TempDir::new().unwrap();

    // First session: store an experience
    {
        let mut c = MemoryConnector::new("reopen-agent", Some(tmp.path()), 100, true).unwrap();
        let exp = Experience::new(
            ExperienceType::Success,
            "persistent data".into(),
            "should survive close".into(),
            0.9,
        )
        .unwrap();
        c.store_experience(&exp).unwrap();
        c.close();
    }

    // Second session: reopen and verify data persists
    {
        let c = MemoryConnector::new("reopen-agent", Some(tmp.path()), 100, true).unwrap();
        let results = c.retrieve_experiences(Some(10), None, 0.0).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].context, "persistent data");
    }
}

#[test]
fn test_connector_cleanup() {
    let tmp = TempDir::new().unwrap();
    let mut connector = MemoryConnector::new("cleanup-agent", Some(tmp.path()), 100, true).unwrap();

    for i in 0..10 {
        let exp = Experience::new(
            ExperienceType::Success,
            format!("cleanup ctx {i}"),
            format!("cleanup out {i}"),
            0.5,
        )
        .unwrap();
        connector.store_experience(&exp).unwrap();
    }

    // Cleanup with max_experiences limit
    connector.cleanup(false, None, Some(3)).unwrap();

    let stats = connector.get_statistics().unwrap();
    assert!(
        stats.total_experiences <= 3,
        "expected at most 3 after cleanup, got {}",
        stats.total_experiences
    );
}
