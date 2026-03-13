//! LadybugDB graph database backend for memory storage.
//!
//! Native Rust implementation using `ladybug-graph-rs`.
//! Implements both `MemoryBackend` and `ExperienceBackend` traits.

mod search;
mod storage;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use chrono::{TimeZone, Utc};
use ladybug_graph_rs::{Graph, GraphConfig, Property};

use super::base::{ExperienceBackend, MemoryBackend, StorageStatistics};
use crate::errors::MemoryError;
use crate::experience::{Experience, ExperienceType};

use tracing::debug;

/// LadybugDB graph database backend for memory storage.
///
/// Stores experiences as nodes in a LadybugDB graph database, with relationship
/// tables for similarity and causality links.
pub struct LadybugBackend {
    pub(crate) graph: Graph,
    pub(crate) db_path: PathBuf,
    pub(crate) agent_name: String,
    pub(crate) enable_compression: bool,
}

impl LadybugBackend {
    /// Create a new LadybugDB-backed experience store.
    pub fn new(db_path: &Path, agent_name: &str, enable_compression: bool) -> crate::Result<Self> {
        let config = GraphConfig::new()
            .buffer_pool_size(256 * 1024 * 1024)
            .max_db_size(1024 * 1024 * 1024);
        let graph = Graph::open_with_config(db_path, config)
            .map_err(|e| MemoryError::Storage(format!("Failed to open LadybugDB: {e}")))?;

        let mut backend = Self {
            graph,
            db_path: db_path.to_path_buf(),
            agent_name: agent_name.to_string(),
            enable_compression,
        };

        backend.initialize_schema()?;
        Ok(backend)
    }

    /// Execute a parameterised Cypher query and return rows.
    pub(crate) fn exec(
        &self,
        cypher: &str,
        params: &[(&str, Property)],
    ) -> crate::Result<Vec<Vec<Property>>> {
        self.graph.execute(cypher, params).map_err(|e| {
            debug!("Cypher execution failed — query: {cypher}");
            MemoryError::Storage(format!("Cypher execution failed: {e}"))
        })
    }

    /// Execute a Cypher statement with no expected results.
    pub(crate) fn exec_mut(&self, cypher: &str) -> crate::Result<()> {
        self.graph.execute_cypher(cypher).map_err(|e| {
            debug!("Cypher execution failed — query: {cypher}");
            MemoryError::Storage(format!("Cypher execution failed: {e}"))
        })
    }

    /// Parse a row (Vec<Property>) into an Experience.
    ///
    /// Expected column order:
    ///   0: experience_id, 1: experience_type, 2: context, 3: outcome,
    ///   4: confidence, 5: timestamp, 6: metadata, 7: tags
    pub(crate) fn row_to_experience(row: &[Property]) -> crate::Result<Experience> {
        let exp_id = row
            .first()
            .and_then(|p| p.as_str())
            .ok_or_else(|| MemoryError::Storage("missing experience_id".into()))?
            .to_string();

        let exp_type_str = row
            .get(1)
            .and_then(|p| p.as_str())
            .ok_or_else(|| MemoryError::Storage("missing experience_type".into()))?;

        let context = row
            .get(2)
            .and_then(|p| p.as_str())
            .ok_or_else(|| MemoryError::Storage("missing context".into()))?
            .to_string();

        let outcome = row
            .get(3)
            .and_then(|p| p.as_str())
            .ok_or_else(|| MemoryError::Storage("missing outcome".into()))?
            .to_string();

        let confidence = row
            .get(4)
            .and_then(|p| p.as_f64())
            .ok_or_else(|| MemoryError::Storage("missing confidence".into()))?;

        let timestamp_i64 = row
            .get(5)
            .and_then(|p| p.as_i64())
            .ok_or_else(|| MemoryError::Storage("missing timestamp".into()))?;

        let metadata_str = row.get(6).and_then(|p| p.as_str()).unwrap_or("{}");

        let tags_str = row.get(7).and_then(|p| p.as_str()).unwrap_or("[]");

        let experience_type: ExperienceType = exp_type_str
            .parse()
            .map_err(|e: String| MemoryError::Storage(e))?;

        let timestamp = Utc
            .timestamp_opt(timestamp_i64, 0)
            .single()
            .unwrap_or_else(Utc::now);

        let metadata: HashMap<String, serde_json::Value> =
            serde_json::from_str(metadata_str).unwrap_or_default();
        let tags: Vec<String> = serde_json::from_str(tags_str).unwrap_or_default();

        Experience::from_parts(
            exp_id,
            experience_type,
            context,
            outcome,
            confidence,
            timestamp,
            metadata,
            tags,
        )
    }

    /// Build the standard RETURN clause used by retrieve and search queries.
    pub(crate) const RETURN_COLS: &'static str = "e.experience_id, e.experience_type, e.context, \
         e.outcome, e.confidence, e.timestamp, \
         e.metadata, e.tags";
}

impl ExperienceBackend for LadybugBackend {
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
            if let Ok(ft) = entry.file_type() {
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
