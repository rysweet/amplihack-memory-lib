//! Kuzu graph database backend for memory storage.
//!
//! Provides graph-based experience storage using Kuzu via PyO3.
//! Implements both `MemoryBackend` and `ExperienceBackend` traits.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use chrono::{TimeZone, Utc};
use pyo3::prelude::*;

use super::base::{ExperienceBackend, MemoryBackend, StorageStatistics};
use crate::errors::MemoryError;
use crate::experience::{Experience, ExperienceType};
use crate::security::QueryValidator;

use tracing::{debug, error, warn};

/// Kuzu graph database backend for memory storage.
///
/// Stores experiences as nodes in a Kuzu graph database, with relationship
/// tables for similarity and causality links.
pub struct KuzuBackend {
    db_path: PathBuf,
    agent_name: String,
    _max_memory_mb: i32,
    _enable_compression: bool,
    /// Python kuzu.Database object
    py_db: PyObject,
    /// Python kuzu.Connection object
    py_conn: PyObject,
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
    fn execute(
        &self,
        py: Python<'_>,
        cypher: &str,
        params: &pyo3::Bound<'_, pyo3::types::PyDict>,
    ) -> crate::Result<PyObject> {
        let conn = self.py_conn.bind(py);
        conn.call_method("execute", (cypher, params), None)
            .map(|r| r.unbind())
            .map_err(|e| {
                debug!("Cypher execution failed — query: {cypher}");
                MemoryError::Storage(format!("Cypher execution failed: {e}"))
            })
    }

    /// Execute a Cypher query without parameters.
    fn execute_no_params(&self, py: Python<'_>, cypher: &str) -> crate::Result<PyObject> {
        let conn = self.py_conn.bind(py);
        conn.call_method1("execute", (cypher,))
            .map(|r| r.unbind())
            .map_err(|e| {
                debug!("Cypher execution failed — query: {cypher}");
                MemoryError::Storage(format!("Cypher execution failed: {e}"))
            })
    }

    /// Parse an Experience from a Kuzu result row.
    fn row_to_experience(_py: Python<'_>, row: &Bound<'_, PyAny>) -> crate::Result<Experience> {
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

impl MemoryBackend for KuzuBackend {
    fn initialize_schema(&mut self) -> crate::Result<()> {
        Python::with_gil(|py| {
            // Create Experience node table
            if let Err(e) = self.execute_no_params(
                py,
                "CREATE NODE TABLE IF NOT EXISTS Experience(\
                    experience_id STRING, \
                    agent_name STRING, \
                    experience_type STRING, \
                    context STRING, \
                    outcome STRING, \
                    confidence DOUBLE, \
                    timestamp INT64, \
                    metadata STRING, \
                    tags STRING, \
                    compressed BOOLEAN, \
                    PRIMARY KEY(experience_id)\
                )",
            ) {
                error!("initialize_schema: failed to create Experience node table: {e}");
            }

            // Create SIMILAR_TO relationship table
            if let Err(e) = self.execute_no_params(
                py,
                "CREATE REL TABLE IF NOT EXISTS SIMILAR_TO(\
                    FROM Experience TO Experience, \
                    similarity_score DOUBLE\
                )",
            ) {
                error!("initialize_schema: failed to create SIMILAR_TO rel table: {e}");
            }

            // Create LEADS_TO relationship table
            if let Err(e) = self.execute_no_params(
                py,
                "CREATE REL TABLE IF NOT EXISTS LEADS_TO(\
                    FROM Experience TO Experience, \
                    causal_strength DOUBLE\
                )",
            ) {
                error!("initialize_schema: failed to create LEADS_TO rel table: {e}");
            }

            Ok(())
        })
    }

    fn store_experience(&mut self, experience: &Experience) -> crate::Result<String> {
        if experience.context.trim().is_empty() {
            return Err(MemoryError::InvalidExperience(
                "context cannot be empty".into(),
            ));
        }
        if experience.outcome.trim().is_empty() {
            return Err(MemoryError::InvalidExperience(
                "outcome cannot be empty".into(),
            ));
        }
        if !(0.0..=1.0).contains(&experience.confidence) {
            return Err(MemoryError::InvalidExperience(
                "confidence must be between 0.0 and 1.0".into(),
            ));
        }

        let metadata_json =
            serde_json::to_string(&experience.metadata).unwrap_or_else(|_| "{}".to_string());
        let tags_json =
            serde_json::to_string(&experience.tags).unwrap_or_else(|_| "[]".to_string());

        Python::with_gil(|py| {
            let params = pyo3::types::PyDict::new_bound(py);
            params
                .set_item("id", &experience.experience_id)
                .map_err(|e| MemoryError::Storage(format!("param error: {e}")))?;
            params
                .set_item("agent", &self.agent_name)
                .map_err(|e| MemoryError::Storage(format!("param error: {e}")))?;
            params
                .set_item("type", experience.experience_type.as_str())
                .map_err(|e| MemoryError::Storage(format!("param error: {e}")))?;
            params
                .set_item("context", &experience.context)
                .map_err(|e| MemoryError::Storage(format!("param error: {e}")))?;
            params
                .set_item("outcome", &experience.outcome)
                .map_err(|e| MemoryError::Storage(format!("param error: {e}")))?;
            params
                .set_item("conf", experience.confidence)
                .map_err(|e| MemoryError::Storage(format!("param error: {e}")))?;
            params
                .set_item("ts", experience.timestamp.timestamp())
                .map_err(|e| MemoryError::Storage(format!("param error: {e}")))?;
            params
                .set_item("metadata", &metadata_json)
                .map_err(|e| MemoryError::Storage(format!("param error: {e}")))?;
            params
                .set_item("tags", &tags_json)
                .map_err(|e| MemoryError::Storage(format!("param error: {e}")))?;

            self.execute(
                py,
                "CREATE (:Experience {\
                    experience_id: $id, \
                    agent_name: $agent, \
                    experience_type: $type, \
                    context: $context, \
                    outcome: $outcome, \
                    confidence: $conf, \
                    timestamp: $ts, \
                    metadata: $metadata, \
                    tags: $tags, \
                    compressed: false\
                })",
                &params,
            )?;

            Ok(experience.experience_id.clone())
        })
    }

    fn retrieve_experiences(
        &self,
        limit: Option<usize>,
        experience_type: Option<ExperienceType>,
        min_confidence: f64,
    ) -> crate::Result<Vec<Experience>> {
        Python::with_gil(|py| {
            let mut where_clauses = vec![
                "e.agent_name = $agent".to_string(),
                "e.confidence >= $min_conf".to_string(),
            ];
            let params = pyo3::types::PyDict::new_bound(py);
            params
                .set_item("agent", &self.agent_name)
                .map_err(|e| MemoryError::Storage(format!("param error: {e}")))?;
            params
                .set_item("min_conf", min_confidence)
                .map_err(|e| MemoryError::Storage(format!("param error: {e}")))?;

            if let Some(et) = experience_type {
                where_clauses.push("e.experience_type = $exp_type".to_string());
                params
                    .set_item("exp_type", et.as_str())
                    .map_err(|e| MemoryError::Storage(format!("param error: {e}")))?;
            }

            let where_clause = where_clauses.join(" AND ");
            let mut query = format!(
                "MATCH (e:Experience) WHERE {where_clause} \
                 RETURN e.experience_id, e.experience_type, e.context, e.outcome, \
                        e.confidence, e.timestamp, e.metadata, e.tags \
                 ORDER BY e.timestamp DESC"
            );

            if let Some(lim) = limit {
                // Safety: `lim` is a `usize`, guaranteed to be a non-negative integer.
                query.push_str(&format!(" LIMIT {lim}"));
            }

            if !QueryValidator::is_safe_cypher(&query) {
                return Err(MemoryError::SecurityViolation(
                    "constructed Cypher query failed safety check".into(),
                ));
            }

            let result = self.execute(py, &query, &params)?;
            let result_ref = result.bind(py);

            let mut experiences = Vec::new();
            loop {
                let has_next: bool = result_ref
                    .call_method0("has_next")
                    .map_err(|e| MemoryError::Storage(format!("has_next failed: {e}")))?
                    .extract()
                    .map_err(|e| MemoryError::Storage(format!("has_next extract failed: {e}")))?;

                if !has_next {
                    break;
                }

                let row = result_ref
                    .call_method0("get_next")
                    .map_err(|e| MemoryError::Storage(format!("get_next failed: {e}")))?;

                experiences.push(Self::row_to_experience(py, &row)?);
            }

            Ok(experiences)
        })
    }

    fn search(
        &self,
        query: &str,
        experience_type: Option<ExperienceType>,
        min_confidence: f64,
        limit: usize,
    ) -> crate::Result<Vec<Experience>> {
        if query.trim().is_empty() {
            return self.retrieve_experiences(Some(limit), experience_type, min_confidence);
        }

        Python::with_gil(|py| {
            let mut where_clauses = vec![
                "e.agent_name = $agent".to_string(),
                "e.confidence >= $min_conf".to_string(),
                "(lower(e.context) CONTAINS lower($query) OR lower(e.outcome) CONTAINS lower($query))".to_string(),
            ];
            let params = pyo3::types::PyDict::new_bound(py);
            params
                .set_item("agent", &self.agent_name)
                .map_err(|e| MemoryError::Storage(format!("param error: {e}")))?;
            params
                .set_item("min_conf", min_confidence)
                .map_err(|e| MemoryError::Storage(format!("param error: {e}")))?;
            params
                .set_item("query", query)
                .map_err(|e| MemoryError::Storage(format!("param error: {e}")))?;

            if let Some(et) = experience_type {
                where_clauses.push("e.experience_type = $exp_type".to_string());
                params
                    .set_item("exp_type", et.as_str())
                    .map_err(|e| MemoryError::Storage(format!("param error: {e}")))?;
            }

            let where_clause = where_clauses.join(" AND ");
            // Safety: `limit` is a `usize`, guaranteed to be a non-negative integer.
            let kuzu_query = format!(
                "MATCH (e:Experience) WHERE {where_clause} \
                 RETURN e.experience_id, e.experience_type, e.context, e.outcome, \
                        e.confidence, e.timestamp, e.metadata, e.tags \
                 ORDER BY e.timestamp DESC \
                 LIMIT {limit}"
            );

            if !QueryValidator::is_safe_cypher(&kuzu_query) {
                return Err(MemoryError::SecurityViolation(
                    "constructed Cypher query failed safety check".into(),
                ));
            }

            let result = self.execute(py, &kuzu_query, &params)?;
            let result_ref = result.bind(py);

            let mut experiences = Vec::new();
            loop {
                let has_next: bool = result_ref
                    .call_method0("has_next")
                    .map_err(|e| MemoryError::Storage(format!("has_next failed: {e}")))?
                    .extract()
                    .map_err(|e| MemoryError::Storage(format!("has_next extract failed: {e}")))?;

                if !has_next {
                    break;
                }

                let row = result_ref
                    .call_method0("get_next")
                    .map_err(|e| MemoryError::Storage(format!("get_next failed: {e}")))?;

                experiences.push(Self::row_to_experience(py, &row)?);
            }

            Ok(experiences)
        })
    }

    fn get_statistics(&self) -> crate::Result<StorageStatistics> {
        Python::with_gil(|py| {
            let params = pyo3::types::PyDict::new_bound(py);
            params
                .set_item("agent", &self.agent_name)
                .map_err(|e| MemoryError::Storage(format!("param error: {e}")))?;

            // Total count
            let result = self.execute(
                py,
                "MATCH (e:Experience) WHERE e.agent_name = $agent RETURN COUNT(e) as total",
                &params,
            )?;
            let result_ref = result.bind(py);
            let total: usize = if result_ref
                .call_method0("has_next")
                .map_err(|e| MemoryError::Storage(format!("has_next failed: {e}")))?
                .extract::<bool>()
                .unwrap_or(false)
            {
                let row = result_ref
                    .call_method0("get_next")
                    .map_err(|e| MemoryError::Storage(format!("get_next failed: {e}")))?;
                row.get_item(0)
                    .map_err(|e| MemoryError::Storage(format!("get_item failed: {e}")))?
                    .extract::<usize>()
                    .unwrap_or(0)
            } else {
                0
            };

            // Count by type
            let result = self.execute(
                py,
                "MATCH (e:Experience) WHERE e.agent_name = $agent \
                 RETURN e.experience_type, COUNT(e) as count",
                &params,
            )?;
            let result_ref = result.bind(py);
            let mut by_type = HashMap::new();
            loop {
                let has_next: bool = result_ref
                    .call_method0("has_next")
                    .map_err(|e| MemoryError::Storage(format!("has_next failed: {e}")))?
                    .extract()
                    .unwrap_or(false);
                if !has_next {
                    break;
                }
                let row = result_ref
                    .call_method0("get_next")
                    .map_err(|e| MemoryError::Storage(format!("get_next failed: {e}")))?;
                if let (Ok(type_str), Ok(count)) = (
                    row.get_item(0).and_then(|v| v.extract::<String>()),
                    row.get_item(1).and_then(|v| v.extract::<usize>()),
                ) {
                    if let Ok(et) = type_str.parse::<ExperienceType>() {
                        by_type.insert(et, count);
                    }
                }
            }

            // Storage size (approximate from database directory)
            let storage_size_kb = if self.db_path.exists() {
                walkdir_size(&self.db_path) as f64 / 1024.0
            } else {
                0.0
            };

            // Compressed count
            let result = self.execute(
                py,
                "MATCH (e:Experience) WHERE e.agent_name = $agent AND e.compressed = true \
                 RETURN COUNT(e) as compressed",
                &params,
            )?;
            let result_ref = result.bind(py);
            let compressed: usize = if result_ref
                .call_method0("has_next")
                .map_err(|e| MemoryError::Storage(format!("has_next failed: {e}")))?
                .extract::<bool>()
                .unwrap_or(false)
            {
                let row = result_ref
                    .call_method0("get_next")
                    .map_err(|e| MemoryError::Storage(format!("get_next failed: {e}")))?;
                row.get_item(0)
                    .map_err(|e| MemoryError::Storage(format!("get_item failed: {e}")))?
                    .extract::<usize>()
                    .unwrap_or(0)
            } else {
                0
            };

            let compression_ratio = if compressed > 0 { 3.0 } else { 1.0 };

            Ok(StorageStatistics {
                total_experiences: total,
                by_type,
                storage_size_kb,
                compressed_experiences: compressed,
                compression_ratio,
            })
        })
    }

    fn close(&mut self) {
        Python::with_gil(|py| {
            // Explicitly close connection before dropping
            if let Ok(conn) = self.py_conn.bind(py).getattr("close") {
                if let Err(e) = conn.call0() {
                    warn!("close: failed to close kuzu connection: {e}");
                }
            }
            if let Ok(db) = self.py_db.bind(py).getattr("close") {
                if let Err(e) = db.call0() {
                    warn!("close: failed to close kuzu database: {e}");
                }
            }
        });
    }

    fn cleanup(
        &mut self,
        auto_compress: bool,
        max_age_days: Option<i64>,
        max_experiences: Option<usize>,
    ) -> crate::Result<()> {
        Python::with_gil(|py| {
            let params = pyo3::types::PyDict::new_bound(py);
            params
                .set_item("agent", &self.agent_name)
                .map_err(|e| MemoryError::Storage(format!("param error: {e}")))?;

            // 1. Mark old experiences as compressed (>30 days)
            if auto_compress {
                let cutoff = (chrono::Utc::now() - chrono::Duration::days(30)).timestamp();
                params
                    .set_item("cutoff", cutoff)
                    .map_err(|e| MemoryError::Storage(format!("param error: {e}")))?;
                if let Err(e) = self.execute(
                    py,
                    "MATCH (e:Experience) \
                     WHERE e.agent_name = $agent AND e.timestamp < $cutoff AND e.compressed = false \
                     SET e.compressed = true",
                    &params,
                ) {
                    warn!("cleanup: failed to auto-compress old experiences: {e}");
                }
            }

            // 2. Delete experiences older than max_age_days
            if let Some(days) = max_age_days {
                let cutoff = (chrono::Utc::now() - chrono::Duration::days(days)).timestamp();
                let age_params = pyo3::types::PyDict::new_bound(py);
                age_params
                    .set_item("agent", &self.agent_name)
                    .map_err(|e| MemoryError::Storage(format!("param error: {e}")))?;
                age_params
                    .set_item("cutoff", cutoff)
                    .map_err(|e| MemoryError::Storage(format!("param error: {e}")))?;
                if let Err(e) = self.execute(
                    py,
                    "MATCH (e:Experience) \
                     WHERE e.agent_name = $agent AND e.timestamp < $cutoff \
                     DETACH DELETE e",
                    &age_params,
                ) {
                    warn!("cleanup: failed to delete experiences older than {days} days: {e}");
                }
            }

            // 3. Limit to max_experiences
            if let Some(max_exp) = max_experiences {
                let count_params = pyo3::types::PyDict::new_bound(py);
                count_params
                    .set_item("agent", &self.agent_name)
                    .map_err(|e| MemoryError::Storage(format!("param error: {e}")))?;

                let result = self.execute(
                    py,
                    "MATCH (e:Experience) WHERE e.agent_name = $agent RETURN COUNT(e) as count",
                    &count_params,
                )?;
                let result_ref = result.bind(py);

                let count: usize = if result_ref
                    .call_method0("has_next")
                    .map_err(|e| MemoryError::Storage(format!("has_next failed: {e}")))?
                    .extract::<bool>()
                    .unwrap_or(false)
                {
                    let row = result_ref
                        .call_method0("get_next")
                        .map_err(|e| MemoryError::Storage(format!("get_next failed: {e}")))?;
                    row.get_item(0)
                        .and_then(|v| v.extract::<usize>())
                        .unwrap_or(0)
                } else {
                    0
                };

                if count > max_exp {
                    // Get high-confidence pattern IDs to keep
                    let keep_params = pyo3::types::PyDict::new_bound(py);
                    keep_params
                        .set_item("agent", &self.agent_name)
                        .map_err(|e| MemoryError::Storage(format!("param error: {e}")))?;

                    let result = self.execute(
                        py,
                        "MATCH (e:Experience) \
                         WHERE e.agent_name = $agent AND e.experience_type = 'pattern' AND e.confidence >= 0.8 \
                         RETURN e.experience_id",
                        &keep_params,
                    )?;
                    let result_ref = result.bind(py);

                    let mut keep_ids = std::collections::HashSet::new();
                    loop {
                        let has_next: bool = result_ref
                            .call_method0("has_next")
                            .map_err(|e| MemoryError::Storage(format!("has_next failed: {e}")))?
                            .extract()
                            .unwrap_or(false);
                        if !has_next {
                            break;
                        }
                        let row = result_ref
                            .call_method0("get_next")
                            .map_err(|e| MemoryError::Storage(format!("get_next failed: {e}")))?;
                        if let Ok(id) = row.get_item(0).and_then(|v| v.extract::<String>()) {
                            keep_ids.insert(id);
                        }
                    }

                    // Get most recent IDs
                    // Safety: `max_exp` is a `usize`, guaranteed to be a non-negative integer.
                    let recent_query = format!(
                        "MATCH (e:Experience) WHERE e.agent_name = $agent \
                         RETURN e.experience_id ORDER BY e.timestamp DESC LIMIT {max_exp}"
                    );
                    let result = self.execute(py, &recent_query, &keep_params)?;
                    let result_ref = result.bind(py);
                    loop {
                        let has_next: bool = result_ref
                            .call_method0("has_next")
                            .map_err(|e| MemoryError::Storage(format!("has_next failed: {e}")))?
                            .extract()
                            .unwrap_or(false);
                        if !has_next {
                            break;
                        }
                        let row = result_ref
                            .call_method0("get_next")
                            .map_err(|e| MemoryError::Storage(format!("get_next failed: {e}")))?;
                        if let Ok(id) = row.get_item(0).and_then(|v| v.extract::<String>()) {
                            keep_ids.insert(id);
                        }
                    }

                    // Get all IDs and delete those not in keep_ids
                    let result = self.execute(
                        py,
                        "MATCH (e:Experience) WHERE e.agent_name = $agent RETURN e.experience_id",
                        &keep_params,
                    )?;
                    let result_ref = result.bind(py);

                    let mut all_ids = Vec::new();
                    loop {
                        let has_next: bool = result_ref
                            .call_method0("has_next")
                            .map_err(|e| MemoryError::Storage(format!("has_next failed: {e}")))?
                            .extract()
                            .unwrap_or(false);
                        if !has_next {
                            break;
                        }
                        let row = result_ref
                            .call_method0("get_next")
                            .map_err(|e| MemoryError::Storage(format!("get_next failed: {e}")))?;
                        if let Ok(id) = row.get_item(0).and_then(|v| v.extract::<String>()) {
                            all_ids.push(id);
                        }
                    }

                    for exp_id in all_ids {
                        if !keep_ids.contains(&exp_id) {
                            let del_params = pyo3::types::PyDict::new_bound(py);
                            del_params
                                .set_item("id", &exp_id)
                                .map_err(|e| MemoryError::Storage(format!("param error: {e}")))?;
                            if let Err(e) = self.execute(
                                py,
                                "MATCH (e:Experience {experience_id: $id}) DETACH DELETE e",
                                &del_params,
                            ) {
                                warn!("cleanup: failed to delete excess experience {exp_id}: {e}");
                            }
                        }
                    }
                }
            }

            Ok(())
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
fn walkdir_size(path: &Path) -> u64 {
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

#[cfg(test)]
mod tests {
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
}
