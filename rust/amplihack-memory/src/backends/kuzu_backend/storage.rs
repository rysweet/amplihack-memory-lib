//! MemoryBackend trait implementation for the Kuzu backend.

use super::KuzuBackend;
use crate::backends::base::{walkdir_size, MemoryBackend, StorageStatistics};
use crate::errors::MemoryError;
use crate::experience::{Experience, ExperienceType};
use crate::security::QueryValidator;
use pyo3::prelude::*;
use std::collections::HashMap;
use tracing::warn;

/// Set a key-value pair on a Python dict, mapping errors to `MemoryError`.
fn set_param<V: IntoPyObject<'_>>(
    params: &pyo3::Bound<'_, pyo3::types::PyDict>,
    key: &str,
    value: V,
) -> crate::Result<()> {
    params
        .set_item(key, value)
        .map_err(|e| MemoryError::Storage(format!("param error: {e}")))
}

/// Read a single scalar from the first row of a Kuzu result, or return a default.
fn scalar_from_result<T: for<'py> pyo3::FromPyObject<'py>>(
    result: &Bound<'_, PyAny>,
    default: T,
) -> crate::Result<T> {
    let has_next: bool = result
        .call_method0("has_next")
        .map_err(|e| MemoryError::Storage(format!("has_next failed: {e}")))?
        .extract()
        .unwrap_or(false);
    if !has_next {
        return Ok(default);
    }
    let row = result
        .call_method0("get_next")
        .map_err(|e| MemoryError::Storage(format!("get_next failed: {e}")))?;
    Ok(row
        .get_item(0)
        .map_err(|e| MemoryError::Storage(format!("get_item failed: {e}")))?
        .extract::<T>()
        .unwrap_or(default))
}

/// Collect all string IDs from a Kuzu query result.
fn collect_ids(result_ref: &Bound<'_, PyAny>) -> crate::Result<Vec<String>> {
    let mut ids = Vec::new();
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
            ids.push(id);
        }
    }
    Ok(ids)
}

impl MemoryBackend for KuzuBackend {
    fn initialize_schema(&mut self) -> crate::Result<()> {
        Python::with_gil(|py| {
            self.execute_no_params(
                py,
                "CREATE NODE TABLE IF NOT EXISTS Experience(\
                    experience_id STRING, agent_name STRING, experience_type STRING, \
                    context STRING, outcome STRING, confidence DOUBLE, timestamp INT64, \
                    metadata STRING, tags STRING, compressed BOOLEAN, \
                    PRIMARY KEY(experience_id))",
            )?;
            self.execute_no_params(
                py,
                "CREATE REL TABLE IF NOT EXISTS SIMILAR_TO(\
                    FROM Experience TO Experience, similarity_score DOUBLE)",
            )?;
            self.execute_no_params(
                py,
                "CREATE REL TABLE IF NOT EXISTS LEADS_TO(\
                    FROM Experience TO Experience, causal_strength DOUBLE)",
            )?;
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
            set_param(&params, "id", &experience.experience_id)?;
            set_param(&params, "agent", &self.agent_name)?;
            set_param(&params, "type", experience.experience_type.as_str())?;
            set_param(&params, "context", &experience.context)?;
            set_param(&params, "outcome", &experience.outcome)?;
            set_param(&params, "conf", experience.confidence)?;
            set_param(&params, "ts", experience.timestamp.timestamp())?;
            set_param(&params, "metadata", &metadata_json)?;
            set_param(&params, "tags", &tags_json)?;

            self.execute(
                py,
                "CREATE (:Experience {\
                    experience_id: $id, agent_name: $agent, experience_type: $type, \
                    context: $context, outcome: $outcome, confidence: $conf, \
                    timestamp: $ts, metadata: $metadata, tags: $tags, compressed: false})",
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
            set_param(&params, "agent", &self.agent_name)?;
            set_param(&params, "min_conf", min_confidence)?;

            if let Some(et) = experience_type {
                where_clauses.push("e.experience_type = $exp_type".to_string());
                set_param(&params, "exp_type", et.as_str())?;
            }

            let where_clause = where_clauses.join(" AND ");
            let mut query = format!(
                "MATCH (e:Experience) WHERE {where_clause} \
                 RETURN e.experience_id, e.experience_type, e.context, e.outcome, \
                        e.confidence, e.timestamp, e.metadata, e.tags \
                 ORDER BY e.timestamp DESC"
            );
            if let Some(lim) = limit {
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
                experiences.push(KuzuBackend::row_to_experience(py, &row)?);
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
        super::search::search_experiences(self, query, experience_type, min_confidence, limit)
    }

    fn get_statistics(&self) -> crate::Result<StorageStatistics> {
        Python::with_gil(|py| {
            let params = pyo3::types::PyDict::new_bound(py);
            set_param(&params, "agent", &self.agent_name)?;

            let result = self.execute(
                py,
                "MATCH (e:Experience) WHERE e.agent_name = $agent RETURN COUNT(e) as total",
                &params,
            )?;
            let total: usize = scalar_from_result(result.bind(py), 0)?;

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

            let storage_size_kb = if self.db_path.exists() {
                walkdir_size(&self.db_path) as f64 / 1024.0
            } else {
                0.0
            };

            let result = self.execute(
                py,
                "MATCH (e:Experience) WHERE e.agent_name = $agent AND e.compressed = true \
                 RETURN COUNT(e) as compressed",
                &params,
            )?;
            let compressed: usize = scalar_from_result(result.bind(py), 0)?;
            let compression_ratio = 1.0;

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
            set_param(&params, "agent", &self.agent_name)?;

            if auto_compress {
                let cutoff = (chrono::Utc::now() - chrono::Duration::days(30)).timestamp();
                set_param(&params, "cutoff", cutoff)?;
                self.execute(py,
                    "MATCH (e:Experience) \
                     WHERE e.agent_name = $agent AND e.timestamp < $cutoff AND e.compressed = false \
                     SET e.compressed = true", &params)?;
            }

            if let Some(days) = max_age_days {
                let cutoff = (chrono::Utc::now() - chrono::Duration::days(days)).timestamp();
                let age_params = pyo3::types::PyDict::new_bound(py);
                set_param(&age_params, "agent", &self.agent_name)?;
                set_param(&age_params, "cutoff", cutoff)?;
                self.execute(
                    py,
                    "MATCH (e:Experience) \
                     WHERE e.agent_name = $agent AND e.timestamp < $cutoff \
                     DETACH DELETE e",
                    &age_params,
                )?;
            }

            if let Some(max_exp) = max_experiences {
                let count_params = pyo3::types::PyDict::new_bound(py);
                set_param(&count_params, "agent", &self.agent_name)?;
                let result = self.execute(
                    py,
                    "MATCH (e:Experience) WHERE e.agent_name = $agent RETURN COUNT(e) as count",
                    &count_params,
                )?;
                let count: usize = scalar_from_result(result.bind(py), 0)?;
                if count > max_exp {
                    self.cleanup_excess(py, max_exp, &count_params)?;
                }
            }
            Ok(())
        })
    }
}

impl KuzuBackend {
    /// Delete experiences exceeding `max_exp`, keeping high-confidence and recent entries.
    fn cleanup_excess(
        &self,
        py: Python<'_>,
        max_exp: usize,
        keep_params: &pyo3::Bound<'_, pyo3::types::PyDict>,
    ) -> crate::Result<()> {
        let result = self.execute(
            py,
            "MATCH (e:Experience) \
             WHERE e.agent_name = $agent AND e.experience_type = 'pattern' AND e.confidence >= 0.8 \
             RETURN e.experience_id",
            keep_params,
        )?;
        let mut keep_ids: std::collections::HashSet<String> =
            collect_ids(result.bind(py))?.into_iter().collect();

        let recent_query = format!(
            "MATCH (e:Experience) WHERE e.agent_name = $agent \
             RETURN e.experience_id ORDER BY e.timestamp DESC LIMIT {max_exp}"
        );
        let result = self.execute(py, &recent_query, keep_params)?;
        keep_ids.extend(collect_ids(result.bind(py))?);

        let result = self.execute(
            py,
            "MATCH (e:Experience) WHERE e.agent_name = $agent RETURN e.experience_id",
            keep_params,
        )?;
        let all_ids = collect_ids(result.bind(py))?;

        for exp_id in all_ids {
            if !keep_ids.contains(&exp_id) {
                let del_params = pyo3::types::PyDict::new_bound(py);
                set_param(&del_params, "id", &exp_id)?;
                if let Err(e) = self.execute(
                    py,
                    "MATCH (e:Experience {experience_id: $id}) DETACH DELETE e",
                    &del_params,
                ) {
                    warn!("cleanup: failed to delete excess experience {exp_id}: {e}");
                }
            }
        }
        Ok(())
    }
}
