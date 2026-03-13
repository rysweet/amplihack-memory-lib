//! `MemoryBackend` trait implementation for [`SqliteBackend`].

use std::collections::HashMap;

use chrono::Utc;
use rusqlite::params;

use super::SqliteBackend;
use crate::backends::base::{MemoryBackend, StorageStatistics};
use crate::errors::MemoryError;
use crate::experience::{Experience, ExperienceType};

use super::search::fts5_search;

impl MemoryBackend for SqliteBackend {
    fn initialize_schema(&mut self) -> crate::Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| MemoryError::Internal(format!("mutex poisoned: {e}")))?;
        Self::init_schema(&conn)
    }

    fn store_experience(&mut self, experience: &Experience) -> crate::Result<String> {
        experience.validate()?;

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
            // Safety: 'lim' is a 'usize', guaranteed to be a non-negative integer.
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

        fts5_search(
            &conn,
            &self.agent_name,
            query,
            experience_type,
            min_confidence,
            limit,
        )
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
