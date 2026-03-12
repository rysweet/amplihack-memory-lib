//! Memory connector factory for database lifecycle management.

use std::path::{Path, PathBuf};

use crate::backends::base::{MemoryBackend, StorageStatistics};
use crate::backends::sqlite_backend::SqliteBackend;
use crate::errors::MemoryError;
use crate::experience::{Experience, ExperienceType};

/// Selects which database backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// SQLite with FTS5 (default).
    Sqlite,
    /// Kuzu graph database (requires `kuzu` feature).
    #[cfg(feature = "kuzu")]
    Kuzu,
}

/// Wrapper enum so `MemoryConnector` can hold either backend type.
enum BackendInner {
    Sqlite(SqliteBackend),
    #[cfg(feature = "kuzu")]
    Kuzu(crate::backends::kuzu_backend::KuzuBackend),
}

/// Database connection management for agent memory.
///
/// Factory class that delegates to the selected backend implementation.
pub struct MemoryConnector {
    pub agent_name: String,
    pub storage_path: PathBuf,
    pub db_path: PathBuf,
    backend: BackendInner,
}

impl MemoryConnector {
    /// Create a new connector with the default (SQLite) backend.
    pub fn new(
        agent_name: &str,
        storage_path: Option<&Path>,
        max_memory_mb: i32,
        enable_compression: bool,
    ) -> crate::Result<Self> {
        Self::with_backend(
            agent_name,
            storage_path,
            max_memory_mb,
            enable_compression,
            BackendType::Sqlite,
        )
    }

    /// Create a new connector with an explicit backend type.
    pub fn with_backend(
        agent_name: &str,
        storage_path: Option<&Path>,
        max_memory_mb: i32,
        enable_compression: bool,
        backend_type: BackendType,
    ) -> crate::Result<Self> {
        if agent_name.trim().is_empty() {
            return Err(MemoryError::InvalidInput(
                "agent_name cannot be empty".into(),
            ));
        }

        if max_memory_mb <= 0 {
            return Err(MemoryError::InvalidInput(
                "max_memory_mb must be positive".into(),
            ));
        }

        let agent = agent_name.trim().to_string();

        let storage = match storage_path {
            Some(p) => PathBuf::from(p),
            None => dirs::home_dir()
                .ok_or_else(|| MemoryError::Internal("could not determine home directory".into()))?
                .join(".amplihack")
                .join("memory")
                .join(&agent),
        };

        std::fs::create_dir_all(&storage)
            .map_err(|e| MemoryError::Storage(format!("Cannot create storage directory: {e}")))?;

        let (db_path, backend) = match backend_type {
            BackendType::Sqlite => {
                let db_path = storage.join("experiences.db");
                let b = SqliteBackend::new(&db_path, &agent, max_memory_mb, enable_compression)?;
                (db_path, BackendInner::Sqlite(b))
            }
            #[cfg(feature = "kuzu")]
            BackendType::Kuzu => {
                let db_path = storage.join("kuzu_db");
                let b = crate::backends::kuzu_backend::KuzuBackend::new(
                    &db_path,
                    &agent,
                    max_memory_mb,
                    enable_compression,
                )?;
                (db_path, BackendInner::Kuzu(b))
            }
        };

        Ok(Self {
            agent_name: agent,
            storage_path: storage,
            db_path,
            backend,
        })
    }

    /// Store an experience in the database.
    pub fn store_experience(&mut self, experience: &Experience) -> crate::Result<String> {
        match &mut self.backend {
            BackendInner::Sqlite(b) => b.store_experience(experience),
            #[cfg(feature = "kuzu")]
            BackendInner::Kuzu(b) => b.store_experience(experience),
        }
    }

    /// Retrieve experiences for this agent.
    pub fn retrieve_experiences(
        &self,
        limit: Option<usize>,
        experience_type: Option<ExperienceType>,
        min_confidence: f64,
    ) -> crate::Result<Vec<Experience>> {
        match &self.backend {
            BackendInner::Sqlite(b) => {
                b.retrieve_experiences(limit, experience_type, min_confidence)
            }
            #[cfg(feature = "kuzu")]
            BackendInner::Kuzu(b) => b.retrieve_experiences(limit, experience_type, min_confidence),
        }
    }

    /// Search experiences by text query.
    pub fn search(
        &self,
        query: &str,
        experience_type: Option<ExperienceType>,
        min_confidence: f64,
        limit: usize,
    ) -> crate::Result<Vec<Experience>> {
        match &self.backend {
            BackendInner::Sqlite(b) => b.search(query, experience_type, min_confidence, limit),
            #[cfg(feature = "kuzu")]
            BackendInner::Kuzu(b) => {
                MemoryBackend::search(b, query, experience_type, min_confidence, limit)
            }
        }
    }

    /// Get storage statistics.
    pub fn get_statistics(&self) -> crate::Result<StorageStatistics> {
        match &self.backend {
            BackendInner::Sqlite(b) => b.get_statistics(),
            #[cfg(feature = "kuzu")]
            BackendInner::Kuzu(b) => MemoryBackend::get_statistics(b),
        }
    }

    /// Run cleanup operations.
    pub fn cleanup(
        &mut self,
        auto_compress: bool,
        max_age_days: Option<i64>,
        max_experiences: Option<usize>,
    ) -> crate::Result<()> {
        match &mut self.backend {
            BackendInner::Sqlite(b) => b.cleanup(auto_compress, max_age_days, max_experiences),
            #[cfg(feature = "kuzu")]
            BackendInner::Kuzu(b) => b.cleanup(auto_compress, max_age_days, max_experiences),
        }
    }

    /// Close database connection.
    pub fn close(&mut self) {
        match &mut self.backend {
            BackendInner::Sqlite(b) => b.close(),
            #[cfg(feature = "kuzu")]
            BackendInner::Kuzu(b) => b.close(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let mut connector =
            MemoryConnector::new("test-agent", Some(tmp.path()), 100, true).unwrap();

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
        let mut connector =
            MemoryConnector::new("search-agent", Some(tmp.path()), 100, true).unwrap();

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
        let mut connector =
            MemoryConnector::new("filter-agent", Some(tmp.path()), 100, true).unwrap();

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
        let mut connector =
            MemoryConnector::new("stats-agent", Some(tmp.path()), 100, false).unwrap();

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
        let mut connector =
            MemoryConnector::new("cleanup-agent", Some(tmp.path()), 100, true).unwrap();

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

    #[test]
    fn test_connector_retrieve_with_limit() {
        let tmp = TempDir::new().unwrap();
        let mut connector =
            MemoryConnector::new("limit-agent", Some(tmp.path()), 100, true).unwrap();

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
        let mut connector =
            MemoryConnector::new("conf-agent", Some(tmp.path()), 100, true).unwrap();

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
        let connector =
            MemoryConnector::new("  trimmed-agent  ", Some(tmp.path()), 100, true).unwrap();
        assert_eq!(connector.agent_name, "trimmed-agent");
    }

    #[test]
    fn test_connector_db_path_is_set() {
        let tmp = TempDir::new().unwrap();
        let connector = MemoryConnector::new("dbpath-agent", Some(tmp.path()), 100, true).unwrap();
        assert!(
            connector
                .db_path
                .to_string_lossy()
                .contains("experiences.db"),
            "db_path should point to experiences.db: {:?}",
            connector.db_path
        );
    }
}
