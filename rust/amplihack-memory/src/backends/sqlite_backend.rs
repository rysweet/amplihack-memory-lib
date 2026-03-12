//! SQLite backend for memory storage with FTS5 full-text search.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use chrono::{TimeZone, Utc};
use rusqlite::{params, Connection};

use super::base::{ExperienceBackend, MemoryBackend, StorageStatistics};
use crate::errors::MemoryError;
use crate::experience::{Experience, ExperienceType};

/// SQLite-based memory storage backend.
pub struct SqliteBackend {
    db_path: PathBuf,
    agent_name: String,
    _max_memory_mb: i32,
    _enable_compression: bool,
    conn: Mutex<Connection>,
}

impl SqliteBackend {
    /// Open (or create) a SQLite-backed experience store.
    ///
    /// `db_path` is the database file, `agent_name` scopes stored data,
    /// `max_memory_mb` caps storage, and `enable_compression` toggles payload compression.
    pub fn new(
        db_path: &Path,
        agent_name: &str,
        max_memory_mb: i32,
        enable_compression: bool,
    ) -> crate::Result<Self> {
        let conn = Connection::open(db_path)
            .map_err(|e| MemoryError::Storage(format!("Failed to open database: {e}")))?;

        conn.execute_batch("PRAGMA journal_mode=WAL;")
            .map_err(|e| MemoryError::Storage(format!("Failed to set WAL: {e}")))?;

        let backend = Self {
            db_path: db_path.to_path_buf(),
            agent_name: agent_name.to_string(),
            _max_memory_mb: max_memory_mb,
            _enable_compression: enable_compression,
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

    fn row_to_experience(row: &rusqlite::Row) -> std::result::Result<Experience, rusqlite::Error> {
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

impl MemoryBackend for SqliteBackend {
    fn initialize_schema(&mut self) -> crate::Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::Internal(format!("mutex poisoned: {e}")))?;
        Self::init_schema(&conn)
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
            serde_json::to_string(&experience.metadata).unwrap_or_else(|_| "{}".into());
        let tags_json = serde_json::to_string(&experience.tags).unwrap_or_else(|_| "[]".into());

        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::Internal(format!("mutex poisoned: {e}")))?;
        conn.execute(
            "INSERT INTO experiences (experience_id, agent_name, experience_type, \
             context, outcome, confidence, timestamp, metadata, tags, compressed) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 0)",
            params![
                experience.experience_id,
                self.agent_name,
                experience.experience_type.as_str(),
                experience.context,
                experience.outcome,
                experience.confidence,
                experience.timestamp.timestamp(),
                metadata_json,
                tags_json,
            ],
        )
        .map_err(|e| MemoryError::Storage(format!("Failed to store experience: {e}")))?;

        Ok(experience.experience_id.clone())
    }

    fn retrieve_experiences(
        &self,
        limit: Option<usize>,
        experience_type: Option<ExperienceType>,
        min_confidence: f64,
    ) -> crate::Result<Vec<Experience>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::Internal(format!("mutex poisoned: {e}")))?;

        let mut sql =
            String::from("SELECT * FROM experiences WHERE agent_name = ?1 AND confidence >= ?2");
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> =
            vec![Box::new(self.agent_name.clone()), Box::new(min_confidence)];

        if let Some(et) = experience_type {
            sql.push_str(" AND experience_type = ?3");
            param_values.push(Box::new(et.as_str().to_string()));
        }

        sql.push_str(" ORDER BY timestamp DESC");

        if let Some(lim) = limit {
            sql.push_str(&format!(" LIMIT {lim}"));
        }

        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| MemoryError::Storage(format!("Failed to prepare query: {e}")))?;

        let rows = stmt
            .query_map(params_refs.as_slice(), Self::row_to_experience)
            .map_err(|e| MemoryError::Storage(format!("Failed to execute query: {e}")))?;

        Ok(rows.filter_map(|r| r.ok()).collect())
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

        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::Internal(format!("mutex poisoned: {e}")))?;

        let mut sql = String::from(
            "SELECT e.* FROM experiences e \
             JOIN experiences_fts fts ON e.experience_id = fts.experience_id \
             WHERE experiences_fts MATCH ?1 \
             AND e.agent_name = ?2 \
             AND e.confidence >= ?3",
        );
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = vec![
            Box::new(query.to_string()),
            Box::new(self.agent_name.clone()),
            Box::new(min_confidence),
        ];

        if let Some(et) = experience_type {
            sql.push_str(" AND e.experience_type = ?4");
            param_values.push(Box::new(et.as_str().to_string()));
        }

        sql.push_str(&format!(" ORDER BY e.timestamp DESC LIMIT {limit}"));

        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| MemoryError::Storage(format!("Failed to prepare search: {e}")))?;

        let rows = stmt
            .query_map(params_refs.as_slice(), Self::row_to_experience)
            .map_err(|e| MemoryError::Storage(format!("Failed to execute search: {e}")))?;

        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    fn get_statistics(&self) -> crate::Result<StorageStatistics> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::Internal(format!("mutex poisoned: {e}")))?;

        let total: usize = conn
            .query_row(
                "SELECT COUNT(*) FROM experiences WHERE agent_name = ?1",
                params![self.agent_name],
                |row| row.get(0),
            )
            .unwrap_or(0);

        let mut by_type = HashMap::new();
        let mut stmt = conn
            .prepare(
                "SELECT experience_type, COUNT(*) as count FROM experiences \
                 WHERE agent_name = ?1 GROUP BY experience_type",
            )
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map(params![self.agent_name], |row| {
                let type_str: String = row.get(0)?;
                let count: usize = row.get(1)?;
                Ok((type_str, count))
            })
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        for row in rows.flatten() {
            if let Ok(et) = row.0.parse::<ExperienceType>() {
                by_type.insert(et, row.1);
            }
        }

        let storage_size_kb = if self.db_path.exists() {
            std::fs::metadata(&self.db_path)
                .map(|m| m.len() as f64 / 1024.0)
                .unwrap_or(0.0)
        } else {
            0.0
        };

        let compressed: usize = conn
            .query_row(
                "SELECT COUNT(*) FROM experiences WHERE agent_name = ?1 AND compressed = 1",
                params![self.agent_name],
                |row| row.get(0),
            )
            .unwrap_or(0);

        // No actual compression implemented; ratio is always 1:1
        let compression_ratio = 1.0;

        Ok(StorageStatistics {
            total_experiences: total,
            by_type,
            storage_size_kb,
            compressed_experiences: compressed,
            compression_ratio,
        })
    }

    fn close(&mut self) {
        // Connection will be closed when dropped
    }

    fn cleanup(
        &mut self,
        auto_compress: bool,
        max_age_days: Option<i64>,
        max_experiences: Option<usize>,
    ) -> crate::Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::Internal(format!("mutex poisoned: {e}")))?;
        let mut changes_made = false;

        if auto_compress {
            let cutoff = (Utc::now() - chrono::Duration::days(30)).timestamp();
            let changed = conn
                .execute(
                    "UPDATE experiences SET compressed = 1 \
                     WHERE agent_name = ?1 AND timestamp < ?2 AND compressed = 0",
                    params![self.agent_name, cutoff],
                )
                .map_err(|e| {
                    MemoryError::Storage(format!("Failed to compress experiences: {e}"))
                })?;
            if changed > 0 {
                changes_made = true;
            }
        }

        if let Some(days) = max_age_days {
            let cutoff = (Utc::now() - chrono::Duration::days(days)).timestamp();
            let changed = conn
                .execute(
                    "DELETE FROM experiences WHERE agent_name = ?1 AND timestamp < ?2",
                    params![self.agent_name, cutoff],
                )
                .map_err(|e| {
                    MemoryError::Storage(format!("Failed to delete old experiences: {e}"))
                })?;
            if changed > 0 {
                changes_made = true;
            }
        }

        if let Some(max_exp) = max_experiences {
            let count: usize = conn
                .query_row(
                    "SELECT COUNT(*) FROM experiences WHERE agent_name = ?1",
                    params![self.agent_name],
                    |row| row.get(0),
                )
                .unwrap_or(0);

            if count > max_exp {
                conn.execute(
                    "DELETE FROM experiences WHERE agent_name = ?1 \
                     AND experience_id NOT IN ( \
                         SELECT experience_id FROM experiences \
                         WHERE agent_name = ?1 AND experience_type = 'pattern' AND confidence >= 0.8 \
                         UNION \
                         SELECT experience_id FROM ( \
                             SELECT experience_id FROM experiences \
                             WHERE agent_name = ?1 ORDER BY timestamp DESC LIMIT ?2 \
                         ) \
                     )",
                    params![self.agent_name, max_exp as i64],
                )
                .map_err(|e| MemoryError::Storage(format!("Failed to trim experiences: {e}")))?;
                changes_made = true;
            }
        }

        if changes_made {
            conn.execute_batch("INSERT INTO experiences_fts(experiences_fts) VALUES('rebuild');")
                .map_err(|e| MemoryError::Storage(format!("Failed to rebuild FTS index: {e}")))?;
        }

        Ok(())
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn test_backend() -> (SqliteBackend, NamedTempFile) {
        let tmp = NamedTempFile::new().unwrap();
        let backend = SqliteBackend::new(tmp.path(), "test-agent", 100, true).unwrap();
        (backend, tmp)
    }

    fn test_experience() -> Experience {
        Experience::new(
            ExperienceType::Success,
            "test context".into(),
            "test outcome".into(),
            0.9,
        )
        .unwrap()
    }

    #[test]
    fn test_store_and_retrieve() {
        let (mut backend, _tmp) = test_backend();
        let exp = test_experience();
        let id = backend.store_experience(&exp).unwrap();
        assert!(!id.is_empty());

        let results = backend.retrieve_experiences(Some(10), None, 0.0).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].context, "test context");
    }

    #[test]
    fn test_search_fts() {
        let (mut backend, _tmp) = test_backend();
        let exp = test_experience();
        backend.store_experience(&exp).unwrap();

        let results = MemoryBackend::search(&backend, "test", None, 0.0, 10).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_statistics() {
        let (mut backend, _tmp) = test_backend();
        let exp = test_experience();
        backend.store_experience(&exp).unwrap();

        let stats = MemoryBackend::get_statistics(&backend).unwrap();
        assert_eq!(stats.total_experiences, 1);
    }

    #[test]
    fn test_filter_by_type() {
        let (mut backend, _tmp) = test_backend();
        let success =
            Experience::new(ExperienceType::Success, "ctx".into(), "out".into(), 0.9).unwrap();
        let failure =
            Experience::new(ExperienceType::Failure, "ctx2".into(), "out2".into(), 0.8).unwrap();
        backend.store_experience(&success).unwrap();
        backend.store_experience(&failure).unwrap();

        let results = backend
            .retrieve_experiences(Some(10), Some(ExperienceType::Success), 0.0)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].experience_type, ExperienceType::Success);
    }

    #[test]
    fn test_min_confidence_filter() {
        let (mut backend, _tmp) = test_backend();
        let high = Experience::new(
            ExperienceType::Success,
            "high".into(),
            "high out".into(),
            0.9,
        )
        .unwrap();
        let low = Experience::new(
            ExperienceType::Success,
            "low conf".into(),
            "low out".into(),
            0.3,
        )
        .unwrap();
        backend.store_experience(&high).unwrap();
        backend.store_experience(&low).unwrap();

        let results = backend.retrieve_experiences(Some(10), None, 0.5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].context, "high");
    }

    #[test]
    fn test_cleanup_auto_compress() {
        let (mut backend, _tmp) = test_backend();
        // Store an experience with a timestamp older than 30 days
        let old_ts = Utc::now() - chrono::Duration::days(60);
        let exp = Experience::with_timestamp(
            ExperienceType::Success,
            "old context".into(),
            "old outcome".into(),
            0.9,
            old_ts,
        )
        .unwrap();
        backend.store_experience(&exp).unwrap();

        // Also store a recent experience
        let recent = test_experience();
        backend.store_experience(&recent).unwrap();

        backend.cleanup(true, None, None).unwrap();

        let stats = MemoryBackend::get_statistics(&backend).unwrap();
        assert_eq!(stats.total_experiences, 2);
        // Only the old one should be compressed
        assert_eq!(stats.compressed_experiences, 1);
    }

    #[test]
    fn test_cleanup_max_age_days() {
        let (mut backend, _tmp) = test_backend();
        let old_ts = Utc::now() - chrono::Duration::days(100);
        let old_exp = Experience::with_timestamp(
            ExperienceType::Success,
            "ancient context".into(),
            "ancient outcome".into(),
            0.9,
            old_ts,
        )
        .unwrap();
        backend.store_experience(&old_exp).unwrap();

        let recent = test_experience();
        backend.store_experience(&recent).unwrap();

        // Delete experiences older than 50 days
        backend.cleanup(false, Some(50), None).unwrap();

        let results = backend.retrieve_experiences(Some(10), None, 0.0).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].context, "test context");
    }

    #[test]
    fn test_cleanup_max_experiences() {
        let (mut backend, _tmp) = test_backend();
        // Store 5 experiences with staggered timestamps
        for i in 0..5 {
            let ts = Utc::now() - chrono::Duration::seconds(50 - i);
            let exp = Experience::with_timestamp(
                ExperienceType::Success,
                format!("ctx {i}"),
                format!("out {i}"),
                0.5,
                ts,
            )
            .unwrap();
            backend.store_experience(&exp).unwrap();
        }

        let before = backend.retrieve_experiences(Some(100), None, 0.0).unwrap();
        assert_eq!(before.len(), 5);

        // Trim to max 3 experiences
        backend.cleanup(false, None, Some(3)).unwrap();

        let after = backend.retrieve_experiences(Some(100), None, 0.0).unwrap();
        assert!(after.len() <= 3, "expected at most 3, got {}", after.len());
    }
}
