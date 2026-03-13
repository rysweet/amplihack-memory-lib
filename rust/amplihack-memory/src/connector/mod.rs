//! Memory connector factory for database lifecycle management.

mod operations;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_extended;

use std::path::{Path, PathBuf};

use crate::backends::sqlite_backend::SqliteBackend;
use crate::errors::MemoryError;

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
pub(crate) enum BackendInner {
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
    pub(crate) backend: BackendInner,
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

        // Reject agent names that could cause path traversal or contain
        // characters outside the safe set [a-zA-Z0-9_-].
        if !agent
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return Err(MemoryError::InvalidInput(
                "agent_name must only contain ASCII alphanumeric characters, hyphens, and underscores".into(),
            ));
        }

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
}
