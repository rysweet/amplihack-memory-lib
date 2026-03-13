use super::*;
use crate::experience::{Experience, ExperienceType};
use tempfile::TempDir;

#[test]
fn test_connector_retrieve_with_limit() {
    let tmp = TempDir::new().unwrap();
    let mut connector = MemoryConnector::new("limit-agent", Some(tmp.path()), 100, true).unwrap();

    for i in 0..10 {
        let exp = Experience::new(
            ExperienceType::Success,
            format!("limit ctx {i}"),
            format!("limit out {i}"),
            0.8,
        )
        .unwrap();
        connector.store_experience(&exp).unwrap();
    }

    let limited = connector.retrieve_experiences(Some(3), None, 0.0).unwrap();
    assert!(limited.len() <= 3);

    let all = connector.retrieve_experiences(None, None, 0.0).unwrap();
    assert_eq!(all.len(), 10);
}

#[test]
fn test_connector_retrieve_with_confidence_filter() {
    let tmp = TempDir::new().unwrap();
    let mut connector = MemoryConnector::new("conf-agent", Some(tmp.path()), 100, true).unwrap();

    let low = Experience::new(
        ExperienceType::Success,
        "low confidence".into(),
        "outcome low".into(),
        0.2,
    )
    .unwrap();
    let high = Experience::new(
        ExperienceType::Success,
        "high confidence".into(),
        "outcome high".into(),
        0.9,
    )
    .unwrap();

    connector.store_experience(&low).unwrap();
    connector.store_experience(&high).unwrap();

    let filtered = connector.retrieve_experiences(Some(10), None, 0.5).unwrap();
    assert_eq!(filtered.len(), 1);
    assert!(filtered[0].confidence >= 0.5);
}

#[test]
fn test_connector_concurrent_access_threads() {
    use std::sync::{Arc, Mutex};
    use std::thread;

    let tmp = TempDir::new().unwrap();
    let connector = Arc::new(Mutex::new(
        MemoryConnector::new("thread-agent", Some(tmp.path()), 100, true).unwrap(),
    ));

    let mut handles = vec![];
    for i in 0..5 {
        let c = Arc::clone(&connector);
        handles.push(thread::spawn(move || {
            let exp = Experience::new(
                ExperienceType::Success,
                format!("thread {i} context"),
                format!("thread {i} outcome"),
                0.8,
            )
            .unwrap();
            let mut conn = c.lock().unwrap();
            conn.store_experience(&exp).unwrap();
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let conn = connector.lock().unwrap();
    let results = conn.retrieve_experiences(Some(100), None, 0.0).unwrap();
    assert_eq!(results.len(), 5);
}

#[test]
fn test_connector_multiple_connectors_same_db() {
    let tmp = TempDir::new().unwrap();

    // First connector writes
    {
        let mut c1 = MemoryConnector::new("shared-agent", Some(tmp.path()), 100, true).unwrap();
        let exp = Experience::new(
            ExperienceType::Success,
            "from connector one".into(),
            "written by c1".into(),
            0.9,
        )
        .unwrap();
        c1.store_experience(&exp).unwrap();
    }

    // Second connector reads
    {
        let c2 = MemoryConnector::new("shared-agent", Some(tmp.path()), 100, true).unwrap();
        let results = c2.retrieve_experiences(Some(10), None, 0.0).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].context, "from connector one");
    }
}

#[test]
fn test_connector_empty_search() {
    let tmp = TempDir::new().unwrap();
    let connector =
        MemoryConnector::new("empty-search-agent", Some(tmp.path()), 100, true).unwrap();

    let results = connector.search("anything", None, 0.0, 10).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_connector_search_limit() {
    let tmp = TempDir::new().unwrap();
    let mut connector =
        MemoryConnector::new("search-limit-agent", Some(tmp.path()), 100, true).unwrap();

    for i in 0..10 {
        let exp = Experience::new(
            ExperienceType::Success,
            format!("searchable item {i}"),
            format!("outcome {i}"),
            0.8,
        )
        .unwrap();
        connector.store_experience(&exp).unwrap();
    }

    let limited = connector.search("searchable", None, 0.0, 3).unwrap();
    assert!(limited.len() <= 3);
}

#[test]
fn test_connector_agent_name_trimmed() {
    let tmp = TempDir::new().unwrap();
    let connector = MemoryConnector::new("  trimmed-agent  ", Some(tmp.path()), 100, true).unwrap();
    assert_eq!(connector.agent_name(), "trimmed-agent");
}

#[test]
fn test_connector_db_path_is_set() {
    let tmp = TempDir::new().unwrap();
    let connector = MemoryConnector::new("dbpath-agent", Some(tmp.path()), 100, true).unwrap();
    assert!(
        connector
            .db_path()
            .to_string_lossy()
            .contains("experiences.db"),
        "db_path should point to experiences.db: {:?}",
        connector.db_path()
    );
}

// ====================================================================
// #21: Agent name validation — path traversal prevention
// ====================================================================

#[test]
fn test_agent_name_rejects_path_traversal() {
    let tmp = TempDir::new().unwrap();
    assert!(MemoryConnector::new("../etc/passwd", Some(tmp.path()), 100, true).is_err());
}

#[test]
fn test_agent_name_rejects_slash() {
    let tmp = TempDir::new().unwrap();
    assert!(MemoryConnector::new("agent/evil", Some(tmp.path()), 100, true).is_err());
}

#[test]
fn test_agent_name_rejects_backslash() {
    let tmp = TempDir::new().unwrap();
    assert!(MemoryConnector::new("agent\\evil", Some(tmp.path()), 100, true).is_err());
}

#[test]
fn test_agent_name_rejects_dots() {
    let tmp = TempDir::new().unwrap();
    assert!(MemoryConnector::new("agent..name", Some(tmp.path()), 100, true).is_err());
}

#[test]
fn test_agent_name_rejects_special_chars() {
    let tmp = TempDir::new().unwrap();
    assert!(MemoryConnector::new("agent@name", Some(tmp.path()), 100, true).is_err());
    assert!(MemoryConnector::new("agent name", Some(tmp.path()), 100, true).is_err());
    assert!(MemoryConnector::new("agent!name", Some(tmp.path()), 100, true).is_err());
}

#[test]
fn test_agent_name_allows_valid_names() {
    let tmp = TempDir::new().unwrap();
    assert!(MemoryConnector::new("valid-agent", Some(tmp.path()), 100, true).is_ok());
    assert!(MemoryConnector::new("agent_123", Some(tmp.path()), 100, true).is_ok());
    assert!(MemoryConnector::new("MyAgent", Some(tmp.path()), 100, true).is_ok());
}
