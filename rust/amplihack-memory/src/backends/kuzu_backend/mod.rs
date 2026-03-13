//! Kuzu graph database backend for memory storage.
//!
//! Provides graph-based experience storage using Kuzu via PyO3.
//! Implements both `MemoryBackend` and `ExperienceBackend` traits.

mod search;
mod storage;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use chrono::{TimeZone, Utc};
use pyo3::prelude::*;

use super::base::{ExperienceBackend, MemoryBackend, StorageStatistics};
use crate::errors::MemoryError;
use crate::experience::{Experience, ExperienceType};

/// Kuzu graph database backend for memory storage.
///
/// Stores experiences as nodes in a Kuzu graph database, with relationship
/// tables for similarity and causality links.
pub struct KuzuBackend {
    pub(crate) db_path: PathBuf,
    pub(crate) agent_name: String,
    pub(crate) _max_memory_mb: i32,
    pub(crate) _enable_compression: bool,
    /// Python kuzu.Database object
    pub(crate) py_db: PyObject,
    /// Python kuzu.Connection object
    pub(crate) py_conn: PyObject,
}

impl KuzuBackend {
    /// Create a new Kuzu-backed experience store.
    pub fn new(
        db_path: &Path,
        agent_name: &str,
        max_memory_mb: i32,
        enable_compression: bool,
    ) -> crate::Result<Self> {
        let (py_db, py_conn) = Python::with_gil(|py| -> crate::Result<(PyObject, PyObject)> {
            let kuzu = py.import_bound("kuzu").map_err(|e| {
                MemoryError::Storage(format!("Failed to import kuzu Python module: {e}"))
            })?;

            let db_cls = kuzu
                .getattr("Database")
                .map_err(|e| MemoryError::Storage(format!("Failed to get Database class: {e}")))?;
            let kwargs = pyo3::types::PyDict::new_bound(py);
            kwargs
                .set_item("max_db_size", 1_073_741_824_u64)
                .map_err(|e| MemoryError::Storage(format!("param error: {e}")))?;
            let db = db_cls
                .call((db_path.to_string_lossy().to_string(),), Some(&kwargs))
                .map_err(|e| {
                    MemoryError::Storage(format!("Failed to create Kuzu database: {e}"))
                })?;

            let conn = kuzu.call_method1("Connection", (&db,)).map_err(|e| {
                MemoryError::Storage(format!("Failed to create Kuzu connection: {e}"))
            })?;

            Ok((db.unbind(), conn.unbind()))
        })?;

        let mut backend = Self {
            db_path: db_path.to_path_buf(),
            agent_name: agent_name.to_string(),
            _max_memory_mb: max_memory_mb,
            _enable_compression: enable_compression,
            py_db,
            py_conn,
        };

        backend.initialize_schema()?;
        Ok(backend)
    }

    /// Execute a Cypher query with parameters.
    pub(crate) fn execute(
        &self,
        py: Python<'_>,
        cypher: &str,
        params: &pyo3::Bound<'_, pyo3::types::PyDict>,
    ) -> crate::Result<PyObject> {
        let conn = self.py_conn.bind(py);
        conn.call_method("execute", (cypher, params), None)
            .map(|r| r.unbind())
            .map_err(|e| {
                tracing::debug!("Cypher execution failed — query: {cypher}");
                MemoryError::Storage(format!("Cypher execution failed: {e}"))
            })
    }

    /// Execute a Cypher query without parameters.
    pub(crate) fn execute_no_params(
        &self,
        py: Python<'_>,
        cypher: &str,
    ) -> crate::Result<PyObject> {
        let conn = self.py_conn.bind(py);
        conn.call_method1("execute", (cypher,))
            .map(|r| r.unbind())
            .map_err(|e| {
                tracing::debug!("Cypher execution failed — query: {cypher}");
                MemoryError::Storage(format!("Cypher execution failed: {e}"))
            })
    }

    /// Parse an Experience from a Kuzu result row.
    pub(crate) fn row_to_experience(
        _py: Python<'_>,
        row: &Bound<'_, PyAny>,
    ) -> crate::Result<Experience> {
        let exp_id: String = row
            .get_item(0)
            .map_err(|e| MemoryError::Storage(format!("Failed to get experience_id: {e}")))?
            .extract()
            .map_err(|e| MemoryError::Storage(format!("Failed to extract experience_id: {e}")))?;

        let exp_type_str: String = row
            .get_item(1)
            .map_err(|e| MemoryError::Storage(format!("Failed to get experience_type: {e}")))?
            .extract()
            .map_err(|e| MemoryError::Storage(format!("Failed to extract experience_type: {e}")))?;

        let context: String = row
            .get_item(2)
            .map_err(|e| MemoryError::Storage(format!("Failed to get context: {e}")))?
            .extract()
            .map_err(|e| MemoryError::Storage(format!("Failed to extract context: {e}")))?;

        let outcome: String = row
            .get_item(3)
            .map_err(|e| MemoryError::Storage(format!("Failed to get outcome: {e}")))?
            .extract()
            .map_err(|e| MemoryError::Storage(format!("Failed to extract outcome: {e}")))?;

        let confidence: f64 = row
            .get_item(4)
            .map_err(|e| MemoryError::Storage(format!("Failed to get confidence: {e}")))?
            .extract()
            .map_err(|e| MemoryError::Storage(format!("Failed to extract confidence: {e}")))?;

        let timestamp_i64: i64 = row
            .get_item(5)
            .map_err(|e| MemoryError::Storage(format!("Failed to get timestamp: {e}")))?
            .extract()
            .map_err(|e| MemoryError::Storage(format!("Failed to extract timestamp: {e}")))?;

        let metadata_str: String = row
            .get_item(6)
            .map_err(|e| MemoryError::Storage(format!("Failed to get metadata: {e}")))?
            .extract()
            .unwrap_or_else(|_| "{}".to_string());

        let tags_str: String = row
            .get_item(7)
            .map_err(|e| MemoryError::Storage(format!("Failed to get tags: {e}")))?
            .extract()
            .unwrap_or_else(|_| "[]".to_string());

        let experience_type: ExperienceType = exp_type_str
            .parse()
            .map_err(|e: String| MemoryError::Storage(e))?;

        let timestamp = Utc
            .timestamp_opt(timestamp_i64, 0)
            .single()
            .unwrap_or_else(Utc::now);

        let metadata: HashMap<String, serde_json::Value> =
            serde_json::from_str(&metadata_str).unwrap_or_default();
        let tags: Vec<String> = serde_json::from_str(&tags_str).unwrap_or_default();

        Ok(Experience {
            experience_id: exp_id,
            experience_type,
            context,
            outcome,
            confidence,
            timestamp,
            metadata,
            tags,
        })
    }
}

impl ExperienceBackend for KuzuBackend {
    fn add(&mut self, experience: &Experience) -> crate::Result<String> {
        self.store_experience(experience)
    }

    fn search(
        &self,
        query: &str,
        experience_type: Option<ExperienceType>,
        min_confidence: f64,
        limit: usize,
    ) -> crate::Result<Vec<Experience>> {
        MemoryBackend::search(self, query, experience_type, min_confidence, limit)
    }

    fn get_statistics(&self) -> crate::Result<StorageStatistics> {
        MemoryBackend::get_statistics(self)
    }
}

/// Calculate total file size in a directory (recursive).
pub(crate) fn walkdir_size(path: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let ft = entry.file_type();
            if let Ok(ft) = ft {
                if ft.is_file() {
                    if let Ok(meta) = entry.metadata() {
                        total += meta.len();
                    }
                } else if ft.is_dir() {
                    total += walkdir_size(&entry.path());
                }
            }
        }
    }
    total
}
