//! FTS5 full-text search helpers for the SQLite backend.

use rusqlite::Connection;

use super::SqliteBackend;
use crate::errors::MemoryError;
use crate::experience::{Experience, ExperienceType};

use super::escape_fts5_query;

/// Execute an FTS5 search against the experiences table.
///
/// This is the implementation behind `MemoryBackend::search` for [`SqliteBackend`].
pub(crate) fn fts5_search(
    conn: &Connection,
    agent_name: &str,
    query: &str,
    experience_type: Option<ExperienceType>,
    min_confidence: f64,
    limit: usize,
) -> crate::Result<Vec<Experience>> {
    let escaped_query = escape_fts5_query(query);

    let mut sql = String::from(
        "SELECT e.* FROM experiences e \
         JOIN experiences_fts fts ON e.experience_id = fts.experience_id \
         WHERE experiences_fts MATCH ?1 \
         AND e.agent_name = ?2 \
         AND e.confidence >= ?3",
    );
    let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = vec![
        Box::new(escaped_query),
        Box::new(agent_name.to_string()),
        Box::new(min_confidence),
    ];

    if let Some(et) = experience_type {
        sql.push_str(" AND e.experience_type = ?4");
        param_values.push(Box::new(et.as_str().to_string()));
    }

    // Safety: 'limit' is a 'usize'.
    sql.push_str(&format!(" ORDER BY e.timestamp DESC LIMIT {limit}"));

    let params_refs: Vec<&dyn rusqlite::types::ToSql> =
        param_values.iter().map(|p| p.as_ref()).collect();

    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| MemoryError::Storage(format!("Failed to prepare search: {e}")))?;

    let rows = stmt
        .query_map(params_refs.as_slice(), SqliteBackend::row_to_experience)
        .map_err(|e| MemoryError::Storage(format!("Failed to execute search: {e}")))?;

    Ok(rows.filter_map(|r| r.ok()).collect())
}
