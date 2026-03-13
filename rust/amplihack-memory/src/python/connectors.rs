//! PyO3 wrappers for `MemoryConnector` and `ExperienceStore`.
#![allow(clippy::useless_conversion)]

use std::path::Path;

use pyo3::prelude::*;

use crate::connector::MemoryConnector;
use crate::store::ExperienceStore;

use super::experience::PyExperience;
use super::helpers::{mem_err, parse_experience_type, storage_stats_to_dict};

fn validate_db_path(db_path: &str) -> PyResult<()> {
    if db_path.contains("..") {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "db_path must not contain '..' path traversal",
        ));
    }
    if std::path::Path::new(db_path).is_absolute() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "db_path must be a relative path, not absolute",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// PyMemoryConnector
// ---------------------------------------------------------------------------

/// SQLite-backed memory connector for storing and retrieving experiences.
#[pyclass(name = "MemoryConnector")]
pub struct PyMemoryConnector {
    inner: MemoryConnector,
}

#[pymethods]
impl PyMemoryConnector {
    /// Create a new MemoryConnector.
    ///
    /// Args:
    ///     agent_name: Name of the agent
    ///     db_path: Optional path to the SQLite database directory
    #[new]
    #[pyo3(signature = (agent_name, db_path=None))]
    fn new(agent_name: &str, db_path: Option<&str>) -> PyResult<Self> {
        if let Some(p) = db_path {
            validate_db_path(p)?;
        }
        let path = db_path.map(Path::new);
        let conn = MemoryConnector::new(agent_name, path, 100, false).map_err(mem_err)?;
        Ok(Self { inner: conn })
    }

    /// Store an experience and return its id.
    fn store_experience(&mut self, experience: &PyExperience) -> PyResult<String> {
        self.inner
            .store_experience(&experience.inner)
            .map_err(mem_err)
    }

    /// Retrieve experiences, optionally filtered.
    #[pyo3(signature = (limit=None, experience_type=None, min_confidence=0.0))]
    fn retrieve_experiences(
        &self,
        limit: Option<usize>,
        experience_type: Option<&str>,
        min_confidence: f64,
    ) -> PyResult<Vec<PyExperience>> {
        let et = experience_type.map(parse_experience_type).transpose()?;
        self.inner
            .retrieve_experiences(limit, et, min_confidence)
            .map(|v| v.into_iter().map(PyExperience::from_rust).collect())
            .map_err(mem_err)
    }

    /// Full-text search for experiences.
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

    /// Get storage statistics as a dict.
    fn get_statistics(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats = self.inner.get_statistics().map_err(mem_err)?;
        storage_stats_to_dict(py, &stats)
    }

    /// Close the connector.
    fn close(&mut self) {
        self.inner.close();
    }
}

// ---------------------------------------------------------------------------
// PyExperienceStore
// ---------------------------------------------------------------------------

/// Higher-level experience store with auto-cleanup policies.
#[pyclass(name = "ExperienceStore")]
pub struct PyExperienceStore {
    inner: ExperienceStore,
}

#[pymethods]
impl PyExperienceStore {
    /// Create a new ExperienceStore.
    ///
    /// Args:
    ///     agent_name: Name of the agent
    ///     db_path: Optional storage directory path
    #[new]
    #[pyo3(signature = (agent_name, db_path=None))]
    fn new(agent_name: &str, db_path: Option<&str>) -> PyResult<Self> {
        if let Some(p) = db_path {
            validate_db_path(p)?;
        }
        let path = db_path.map(Path::new);
        let store =
            ExperienceStore::new(agent_name, false, None, None, 100, path).map_err(mem_err)?;
        Ok(Self { inner: store })
    }

    /// Add an experience and return its id.
    fn add(&mut self, experience: &PyExperience) -> PyResult<String> {
        self.inner.add(&experience.inner).map_err(mem_err)
    }

    /// Search experiences by query.
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

    /// Get storage statistics as a dict.
    fn get_statistics(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats = self.inner.get_statistics().map_err(mem_err)?;
        storage_stats_to_dict(py, &stats)
    }
}
