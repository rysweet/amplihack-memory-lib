//! PyO3 wrapper for `SecureMemoryBackend` with `SqliteBackend`.
#![allow(clippy::useless_conversion)]

use std::path::PathBuf;

use pyo3::prelude::*;

use crate::backends::sqlite_backend::SqliteBackend;
use crate::backends::ExperienceBackend;
use crate::experience::ExperienceType;
use crate::security::{AgentCapabilities, ScopeLevel, SecureMemoryBackend};

use super::experience::PyExperience;
use super::helpers::{mem_err, parse_experience_type, storage_stats_to_dict};

/// Parse a scope level string into a ScopeLevel enum.
fn parse_scope(s: &str) -> PyResult<ScopeLevel> {
    match s {
        "session_only" => Ok(ScopeLevel::SessionOnly),
        "cross_session_read" => Ok(ScopeLevel::CrossSessionRead),
        "cross_session_write" => Ok(ScopeLevel::CrossSessionWrite),
        "global_read" => Ok(ScopeLevel::GlobalRead),
        "global_write" => Ok(ScopeLevel::GlobalWrite),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown scope level: '{s}'. Expected one of: session_only, \
             cross_session_read, cross_session_write, global_read, global_write"
        ))),
    }
}

/// Experience store wrapped with capability-based access control
/// and automatic credential scrubbing.
#[pyclass(name = "SecureMemoryBackend")]
pub struct PySecureMemoryBackend {
    inner: SecureMemoryBackend<SqliteBackend>,
}

#[pymethods]
impl PySecureMemoryBackend {
    /// Create a new SecureMemoryBackend.
    ///
    /// Args:
    ///     agent_name: Name of the agent
    ///     db_path: Path to the SQLite database file
    ///     scope: Access scope level string (e.g. "session_only", "global_read")
    ///     allowed_types: List of allowed experience type strings (empty = all)
    ///     max_query_cost: Maximum allowed query cost (positive int)
    ///     can_access_patterns: Whether agent can access pattern experiences
    ///     memory_quota_mb: Memory quota in MB (positive int)
    #[new]
    #[pyo3(signature = (
        agent_name,
        db_path,
        scope="session_only",
        allowed_types=None,
        max_query_cost=50,
        can_access_patterns=false,
        memory_quota_mb=10,
    ))]
    fn new(
        agent_name: &str,
        db_path: &str,
        scope: &str,
        allowed_types: Option<Vec<String>>,
        max_query_cost: i32,
        can_access_patterns: bool,
        memory_quota_mb: i32,
    ) -> PyResult<Self> {
        let scope_level = parse_scope(scope)?;
        let types: Vec<ExperienceType> = allowed_types
            .unwrap_or_default()
            .iter()
            .map(|s| parse_experience_type(s))
            .collect::<PyResult<_>>()?;

        let caps = AgentCapabilities::new(
            scope_level,
            types,
            max_query_cost,
            can_access_patterns,
            memory_quota_mb,
        )
        .map_err(mem_err)?;

        let path = PathBuf::from(db_path);
        let backend =
            SqliteBackend::new(&path, agent_name, memory_quota_mb, false).map_err(mem_err)?;

        Ok(Self {
            inner: SecureMemoryBackend::new(backend, caps),
        })
    }

    /// Add an experience with security checks and credential scrubbing.
    fn add(&mut self, experience: &PyExperience) -> PyResult<String> {
        self.inner
            .add_experience(&experience.inner)
            .map_err(mem_err)
    }

    /// Search experiences with security checks.
    #[pyo3(signature = (query, experience_type=None, min_confidence=0.0, limit=10))]
    fn search(
        &self,
        query: &str,
        experience_type: Option<&str>,
        min_confidence: f64,
        limit: usize,
    ) -> PyResult<Vec<PyExperience>> {
        let et = experience_type.map(parse_experience_type).transpose()?;
        self.inner
            .search(query, et, min_confidence, limit)
            .map(|v| v.into_iter().map(PyExperience::from_rust).collect())
            .map_err(mem_err)
    }

    /// Validate a custom SQL query against cost limits.
    fn validate_query(&self, sql: &str) -> PyResult<()> {
        self.inner.validate_custom_query(sql).map_err(mem_err)
    }

    /// Get storage statistics as a dict.
    fn get_statistics(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats = self.inner.store.get_statistics().map_err(mem_err)?;
        storage_stats_to_dict(py, &stats)
    }

    fn __repr__(&self) -> String {
        "SecureMemoryBackend(...)".to_string()
    }
}
