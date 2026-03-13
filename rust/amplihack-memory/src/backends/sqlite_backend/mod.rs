//! SQLite backend for memory storage with FTS5 full-text search.

mod search;
mod storage;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use chrono::{TimeZone, Utc};
use rusqlite::Connection;

use super::base::{ExperienceBackend, MemoryBackend, StorageStatistics};
use crate::errors::MemoryError;
use crate::experience::{Experience, ExperienceType};

/// SQLite-based memory storage backend.
pub struct SqliteBackend {
    pub(crate) db_path: PathBuf,
    pub(crate) agent_name: String,
    pub(crate) conn: Mutex<Connection>,
}

impl SqliteBackend {
    /// Open (or create) a SQLite-backed experience store.
    ///
    /// `db_path` is the database file, `agent_name` scopes stored data,
    /// `max_memory_mb` caps storage, and `enable_compression` toggles payload compression.
    pub fn new(
        db_path: &Path,
        agent_name: &str,
        _max_memory_mb: i32,
        _enable_compression: bool,
    ) -> crate::Result<Self> {
        let conn = Connection::open(db_path)
            .map_err(|e| MemoryError::Storage(format!("Failed to open database: {e}")))?;

        conn.execute_batch("PRAGMA journal_mode=WAL;")
            .map_err(|e| MemoryError::Storage(format!("Failed to set WAL: {e}")))?;

        let backend = Self {
            db_path: db_path.to_path_buf(),
            agent_name: agent_name.to_string(),
            conn: Mutex::new(conn),
        };

        // We need a mutable ref for init, but we just created it
        // so there are no other references
        {
            let conn = backend
                .conn
                .lock()
                .map_err(|e| MemoryError::Internal(format!("mutex poisoned: {e}")))?;
            Self::init_schema(&conn)?;
        }

        Ok(backend)
    }

    fn init_schema(conn: &Connection) -> crate::Result<()> {
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS experiences (
                experience_id TEXT PRIMARY KEY,
                agent_name TEXT NOT NULL,
                experience_type TEXT NOT NULL,
                context TEXT NOT NULL,
                outcome TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp INTEGER NOT NULL,
                metadata TEXT,
                tags TEXT,
                compressed INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_agent_name ON experiences(agent_name);
            CREATE INDEX IF NOT EXISTS idx_experience_type ON experiences(experience_type);
            CREATE INDEX IF NOT EXISTS idx_timestamp ON experiences(timestamp);
            CREATE INDEX IF NOT EXISTS idx_confidence ON experiences(confidence);
            CREATE INDEX IF NOT EXISTS idx_agent_type ON experiences(agent_name, experience_type);

            CREATE VIRTUAL TABLE IF NOT EXISTS experiences_fts
            USING fts5(experience_id, context, outcome, content='experiences', tokenize='porter');

            CREATE TRIGGER IF NOT EXISTS experiences_fts_insert AFTER INSERT ON experiences
            BEGIN
                INSERT INTO experiences_fts(experience_id, context, outcome)
                VALUES (new.experience_id, new.context, new.outcome);
            END;

            CREATE TRIGGER IF NOT EXISTS experiences_fts_delete AFTER DELETE ON experiences
            BEGIN
                INSERT INTO experiences_fts(experiences_fts, rowid, experience_id, context, outcome)
                VALUES('delete', old.rowid, old.experience_id, old.context, old.outcome);
            END;
            ",
        )
        .map_err(|e| MemoryError::Storage(format!("Schema initialization failed: {e}")))?;

        Ok(())
    }

    pub(crate) fn row_to_experience(
        row: &rusqlite::Row,
    ) -> std::result::Result<Experience, rusqlite::Error> {
        let experience_id: String = row.get("experience_id")?;
        let exp_type_str: String = row.get("experience_type")?;
        let context: String = row.get("context")?;
        let outcome: String = row.get("outcome")?;
        let confidence: f64 = row.get("confidence")?;
        let timestamp_i64: i64 = row.get("timestamp")?;
        let metadata_str: Option<String> = row.get("metadata")?;
        let tags_str: Option<String> = row.get("tags")?;

        // Skip rows with unrecognized experience types — a corrupted row
        // should not prevent reading other valid rows.
        let experience_type = exp_type_str.parse::<ExperienceType>().map_err(|_| {
            rusqlite::Error::InvalidColumnType(
                1,
                "experience_type".to_string(),
                rusqlite::types::Type::Text,
            )
        })?;

        let timestamp = Utc
            .timestamp_opt(timestamp_i64, 0)
            .single()
            .unwrap_or_else(Utc::now);

        let metadata: HashMap<String, serde_json::Value> = metadata_str
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();

        let tags: Vec<String> = tags_str
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();

        Experience::from_parts(
            experience_id,
            experience_type,
            context,
            outcome,
            confidence,
            timestamp,
            metadata,
            tags,
        )
        .map_err(|e| rusqlite::Error::InvalidParameterName(e.to_string()))
    }
}

/// Escape user-supplied text for FTS5 MATCH queries.
///
/// Wraps each whitespace-delimited token in double quotes so that FTS5
/// special operators (`AND`, `OR`, `NOT`, `NEAR`, `*`) are treated as
/// literal search terms rather than query syntax.
pub(crate) fn escape_fts5_query(query: &str) -> String {
    query
        .split_whitespace()
        .map(|term| {
            let escaped = term.replace('"', "\"\"");
            format!("\"{escaped}\"")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

impl ExperienceBackend for SqliteBackend {
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
