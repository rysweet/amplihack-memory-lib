//! Search implementation for the Kuzu backend.

use pyo3::prelude::*;

use super::KuzuBackend;
use crate::errors::MemoryError;
use crate::experience::{Experience, ExperienceType};
use crate::security::QueryValidator;

/// Execute a text search against experiences in the Kuzu graph.
///
/// Matches experiences whose context or outcome contain `query` (case-insensitive).
pub(crate) fn search_experiences(
    backend: &KuzuBackend,
    query: &str,
    experience_type: Option<ExperienceType>,
    min_confidence: f64,
    limit: usize,
) -> crate::Result<Vec<Experience>> {
    Python::with_gil(|py| {
        let mut where_clauses = vec![
            "e.agent_name = $agent".to_string(),
            "e.confidence >= $min_conf".to_string(),
            "(lower(e.context) CONTAINS lower($query) OR lower(e.outcome) CONTAINS lower($query))"
                .to_string(),
        ];
        let params = pyo3::types::PyDict::new_bound(py);
        params
            .set_item("agent", &backend.agent_name)
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

        let result = backend.execute(py, &kuzu_query, &params)?;
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
