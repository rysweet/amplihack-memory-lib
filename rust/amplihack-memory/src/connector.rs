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
}
