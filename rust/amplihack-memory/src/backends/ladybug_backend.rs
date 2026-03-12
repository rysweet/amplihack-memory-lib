//! LadybugDB graph database backend for memory storage.
//!
//! Native Rust implementation using `ladybug-graph-rs`.
//! Implements both `MemoryBackend` and `ExperienceBackend` traits.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use chrono::{TimeZone, Utc};
use ladybug_graph_rs::{Graph, GraphConfig, Property};

use super::base::{ExperienceBackend, MemoryBackend, StorageStatistics};
use crate::errors::MemoryError;
use crate::experience::{Experience, ExperienceType};

/// LadybugDB graph database backend for memory storage.
///
/// Stores experiences as nodes in a LadybugDB graph database, with relationship
/// tables for similarity and causality links.
pub struct LadybugBackend {
    graph: Graph,
    db_path: PathBuf,
    agent_name: String,
    enable_compression: bool,
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
    fn exec(&self, cypher: &str, params: &[(&str, Property)]) -> crate::Result<Vec<Vec<Property>>> {
        self.graph.execute(cypher, params).map_err(|e| {
            MemoryError::Storage(format!("Cypher execution failed: {e}\nQuery: {cypher}"))
        })
    }

    /// Execute a Cypher statement with no expected results.
    fn exec_mut(&self, cypher: &str) -> crate::Result<()> {
        self.graph.execute_cypher(cypher).map_err(|e| {
            MemoryError::Storage(format!("Cypher execution failed: {e}\nQuery: {cypher}"))
        })
    }

    /// Parse a row (Vec<Property>) into an Experience.
    ///
    /// Expected column order:
    ///   0: experience_id, 1: experience_type, 2: context, 3: outcome,
    ///   4: confidence, 5: timestamp, 6: metadata, 7: tags
    fn row_to_experience(row: &[Property]) -> crate::Result<Experience> {
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
    const RETURN_COLS: &'static str = "e.experience_id, e.experience_type, e.context, \
                                       e.outcome, e.confidence, e.timestamp, \
                                       e.metadata, e.tags";
}

impl MemoryBackend for LadybugBackend {
    fn initialize_schema(&mut self) -> crate::Result<()> {
        self.exec_mut(
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
        )?;

        self.exec_mut(
            "CREATE REL TABLE IF NOT EXISTS SIMILAR_TO(\
                FROM Experience TO Experience, \
                similarity_score DOUBLE\
            )",
        )?;

        self.exec_mut(
            "CREATE REL TABLE IF NOT EXISTS LEADS_TO(\
                FROM Experience TO Experience, \
                causal_strength DOUBLE\
            )",
        )?;

        Ok(())
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

        self.exec(
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
            &[
                ("id", Property::from(experience.experience_id.as_str())),
                ("agent", Property::from(self.agent_name.as_str())),
                ("type", Property::from(experience.experience_type.as_str())),
                ("context", Property::from(experience.context.as_str())),
                ("outcome", Property::from(experience.outcome.as_str())),
                ("conf", Property::Double(experience.confidence)),
                ("ts", Property::Int64(experience.timestamp.timestamp())),
                ("metadata", Property::from(metadata_json.as_str())),
                ("tags", Property::from(tags_json.as_str())),
            ],
        )?;

        Ok(experience.experience_id.clone())
    }

    fn retrieve_experiences(
        &self,
        limit: Option<usize>,
        experience_type: Option<ExperienceType>,
        min_confidence: f64,
    ) -> crate::Result<Vec<Experience>> {
        let mut where_clauses = vec![
            "e.agent_name = $agent".to_string(),
            "e.confidence >= $min_conf".to_string(),
        ];
        let mut params: Vec<(&str, Property)> = vec![
            ("agent", Property::from(self.agent_name.as_str())),
            ("min_conf", Property::Double(min_confidence)),
        ];

        let type_str;
        if let Some(et) = experience_type {
            where_clauses.push("e.experience_type = $exp_type".to_string());
            type_str = et.as_str().to_string();
            params.push(("exp_type", Property::from(type_str.as_str())));
        }

        let where_clause = where_clauses.join(" AND ");
        let mut query = format!(
            "MATCH (e:Experience) WHERE {where_clause} \
             RETURN {} \
             ORDER BY e.timestamp DESC",
            Self::RETURN_COLS
        );

        if let Some(lim) = limit {
            query.push_str(&format!(" LIMIT {lim}"));
        }

        let rows = self.exec(&query, &params)?;
        rows.iter().map(|r| Self::row_to_experience(r)).collect()
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

        let mut where_clauses = vec![
            "e.agent_name = $agent".to_string(),
            "e.confidence >= $min_conf".to_string(),
            "(lower(e.context) CONTAINS lower($query) OR \
              lower(e.outcome) CONTAINS lower($query))"
                .to_string(),
        ];
        let mut params: Vec<(&str, Property)> = vec![
            ("agent", Property::from(self.agent_name.as_str())),
            ("min_conf", Property::Double(min_confidence)),
            ("query", Property::from(query)),
        ];

        let type_str;
        if let Some(et) = experience_type {
            where_clauses.push("e.experience_type = $exp_type".to_string());
            type_str = et.as_str().to_string();
            params.push(("exp_type", Property::from(type_str.as_str())));
        }

        let where_clause = where_clauses.join(" AND ");
        let cypher = format!(
            "MATCH (e:Experience) WHERE {where_clause} \
             RETURN {} \
             ORDER BY e.timestamp DESC \
             LIMIT {limit}",
            Self::RETURN_COLS
        );

        let rows = self.exec(&cypher, &params)?;
        rows.iter().map(|r| Self::row_to_experience(r)).collect()
    }

    fn get_statistics(&self) -> crate::Result<StorageStatistics> {
        let params: Vec<(&str, Property)> =
            vec![("agent", Property::from(self.agent_name.as_str()))];

        // Total count
        let rows = self.exec(
            "MATCH (e:Experience) WHERE e.agent_name = $agent RETURN COUNT(e)",
            &params,
        )?;
        let total = rows
            .first()
            .and_then(|r| r.first())
            .and_then(|p| p.as_i64())
            .unwrap_or(0) as usize;

        // Count by type
        let rows = self.exec(
            "MATCH (e:Experience) WHERE e.agent_name = $agent \
             RETURN e.experience_type, COUNT(e)",
            &params,
        )?;
        let mut by_type = HashMap::new();
        for row in &rows {
            if let (Some(type_str), Some(count)) = (
                row.first().and_then(|p| p.as_str()),
                row.get(1).and_then(|p| p.as_i64()),
            ) {
                if let Ok(et) = type_str.parse::<ExperienceType>() {
                    by_type.insert(et, count as usize);
                }
            }
        }

        // Storage size
        let storage_size_kb = if self.db_path.exists() {
            walkdir_size(&self.db_path) as f64 / 1024.0
        } else {
            0.0
        };

        // Compressed count
        let rows = self.exec(
            "MATCH (e:Experience) WHERE e.agent_name = $agent AND e.compressed = true \
             RETURN COUNT(e)",
            &params,
        )?;
        let compressed = rows
            .first()
            .and_then(|r| r.first())
            .and_then(|p| p.as_i64())
            .unwrap_or(0) as usize;

        let compression_ratio = if compressed > 0 { 3.0 } else { 1.0 };

        Ok(StorageStatistics {
            total_experiences: total,
            by_type,
            storage_size_kb,
            compressed_experiences: compressed,
            compression_ratio,
        })
    }

    fn close(&mut self) {
        // Graph handles cleanup on drop — nothing to do.
    }

    fn cleanup(
        &mut self,
        auto_compress: bool,
        max_age_days: Option<i64>,
        max_experiences: Option<usize>,
    ) -> crate::Result<()> {
        let agent = self.agent_name.clone();

        // 1. Mark old experiences as compressed (>30 days)
        if auto_compress && self.enable_compression {
            let cutoff = (Utc::now() - chrono::Duration::days(30)).timestamp();
            let _ = self.exec(
                "MATCH (e:Experience) \
                 WHERE e.agent_name = $agent AND e.timestamp < $cutoff AND e.compressed = false \
                 SET e.compressed = true",
                &[
                    ("agent", Property::from(agent.as_str())),
                    ("cutoff", Property::Int64(cutoff)),
                ],
            );
        }

        // 2. Delete experiences older than max_age_days
        if let Some(days) = max_age_days {
            let cutoff = (Utc::now() - chrono::Duration::days(days)).timestamp();
            let _ = self.exec(
                "MATCH (e:Experience) \
                 WHERE e.agent_name = $agent AND e.timestamp < $cutoff \
                 DETACH DELETE e",
                &[
                    ("agent", Property::from(agent.as_str())),
                    ("cutoff", Property::Int64(cutoff)),
                ],
            );
        }

        // 3. Limit to max_experiences
        if let Some(max_exp) = max_experiences {
            let count_rows = self.exec(
                "MATCH (e:Experience) WHERE e.agent_name = $agent RETURN COUNT(e)",
                &[("agent", Property::from(agent.as_str()))],
            )?;
            let count = count_rows
                .first()
                .and_then(|r| r.first())
                .and_then(|p| p.as_i64())
                .unwrap_or(0) as usize;

            if count > max_exp {
                // Collect IDs to keep: high-confidence patterns + most recent
                let mut keep_ids = std::collections::HashSet::new();

                let pattern_rows = self.exec(
                    "MATCH (e:Experience) \
                     WHERE e.agent_name = $agent AND e.experience_type = 'pattern' AND e.confidence >= 0.8 \
                     RETURN e.experience_id",
                    &[("agent", Property::from(agent.as_str()))],
                )?;
                for row in &pattern_rows {
                    if let Some(id) = row.first().and_then(|p| p.as_str()) {
                        keep_ids.insert(id.to_string());
                    }
                }

                let recent_query = format!(
                    "MATCH (e:Experience) WHERE e.agent_name = $agent \
                     RETURN e.experience_id ORDER BY e.timestamp DESC LIMIT {max_exp}"
                );
                let recent_rows =
                    self.exec(&recent_query, &[("agent", Property::from(agent.as_str()))])?;
                for row in &recent_rows {
                    if let Some(id) = row.first().and_then(|p| p.as_str()) {
                        keep_ids.insert(id.to_string());
                    }
                }

                // Get all IDs and delete those not kept
                let all_rows = self.exec(
                    "MATCH (e:Experience) WHERE e.agent_name = $agent RETURN e.experience_id",
                    &[("agent", Property::from(agent.as_str()))],
                )?;
                for row in &all_rows {
                    if let Some(id) = row.first().and_then(|p| p.as_str()) {
                        if !keep_ids.contains(id) {
                            let _ = self.exec(
                                "MATCH (e:Experience {experience_id: $id}) DETACH DELETE e",
                                &[("id", Property::from(id))],
                            );
                        }
                    }
                }
            }
        }

        Ok(())
    }
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
fn walkdir_size(path: &Path) -> u64 {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_backend() -> (LadybugBackend, tempfile::TempDir) {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("test_ladybug_backend");
        let backend = LadybugBackend::new(&db_path, "test-agent", true).unwrap();
        (backend, tmp)
    }

    fn make_experience(
        exp_type: ExperienceType,
        context: &str,
        outcome: &str,
        confidence: f64,
    ) -> Experience {
        Experience::new(exp_type, context.into(), outcome.into(), confidence).unwrap()
    }

    #[test]
    fn test_create_backend() {
        let (_backend, _tmp) = make_backend();
    }

    #[test]
    fn test_initialize_schema() {
        let (mut backend, _tmp) = make_backend();
        // Schema already initialized by new(); calling again should be idempotent.
        backend.initialize_schema().unwrap();
    }

    #[test]
    fn test_store_and_retrieve() {
        let (mut backend, _tmp) = make_backend();
        let exp = make_experience(
            ExperienceType::Success,
            "test context data",
            "test outcome data",
            0.9,
        );

        let id = backend.store_experience(&exp).unwrap();
        assert!(!id.is_empty());

        let results = backend.retrieve_experiences(Some(10), None, 0.0).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].context, "test context data");
        assert_eq!(results[0].outcome, "test outcome data");
        assert_eq!(results[0].experience_type, ExperienceType::Success);
    }

    #[test]
    fn test_store_validates_empty_context() {
        let (mut backend, _tmp) = make_backend();
        let mut exp = make_experience(ExperienceType::Success, "placeholder", "outcome", 0.5);
        exp.context = "".to_string();
        let result = backend.store_experience(&exp);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("context"),
            "error should mention context: {err}"
        );
    }

    #[test]
    fn test_store_validates_confidence_range() {
        let (mut backend, _tmp) = make_backend();
        let mut exp = make_experience(ExperienceType::Success, "ctx", "outcome", 0.5);
        exp.confidence = 1.5;
        let result = backend.store_experience(&exp);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("confidence"),
            "error should mention confidence: {err}"
        );
    }

    #[test]
    fn test_search_by_text() {
        let (mut backend, _tmp) = make_backend();
        backend
            .store_experience(&make_experience(
                ExperienceType::Success,
                "rust programming rocks",
                "compiled successfully",
                0.9,
            ))
            .unwrap();
        backend
            .store_experience(&make_experience(
                ExperienceType::Failure,
                "python scripting error",
                "import failed",
                0.5,
            ))
            .unwrap();

        let results = MemoryBackend::search(&backend, "rust", None, 0.0, 10).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].context.contains("rust"));
    }

    #[test]
    fn test_search_by_type() {
        let (mut backend, _tmp) = make_backend();
        backend
            .store_experience(&make_experience(
                ExperienceType::Success,
                "success ctx",
                "success out",
                0.8,
            ))
            .unwrap();
        backend
            .store_experience(&make_experience(
                ExperienceType::Failure,
                "failure ctx",
                "failure out",
                0.4,
            ))
            .unwrap();

        let results = backend
            .retrieve_experiences(Some(10), Some(ExperienceType::Success), 0.0)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].experience_type, ExperienceType::Success);
    }

    #[test]
    fn test_search_min_confidence() {
        let (mut backend, _tmp) = make_backend();
        backend
            .store_experience(&make_experience(
                ExperienceType::Insight,
                "low conf context",
                "low conf outcome",
                0.2,
            ))
            .unwrap();
        backend
            .store_experience(&make_experience(
                ExperienceType::Insight,
                "high conf context",
                "high conf outcome",
                0.9,
            ))
            .unwrap();

        let results = backend.retrieve_experiences(Some(10), None, 0.5).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].confidence >= 0.5);
    }

    #[test]
    fn test_retrieve_with_limit() {
        let (mut backend, _tmp) = make_backend();
        for i in 0..5 {
            backend
                .store_experience(&make_experience(
                    ExperienceType::Success,
                    &format!("context {i}"),
                    &format!("outcome {i}"),
                    0.8,
                ))
                .unwrap();
        }

        let results = backend.retrieve_experiences(Some(3), None, 0.0).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_get_statistics() {
        let (mut backend, _tmp) = make_backend();
        backend
            .store_experience(&make_experience(
                ExperienceType::Pattern,
                "pattern context",
                "pattern outcome",
                0.95,
            ))
            .unwrap();

        let stats = MemoryBackend::get_statistics(&backend).unwrap();
        assert_eq!(stats.total_experiences, 1);
        assert_eq!(*stats.by_type.get(&ExperienceType::Pattern).unwrap(), 1);
    }

    #[test]
    fn test_cleanup_max_experiences() {
        let (mut backend, _tmp) = make_backend();
        for i in 0..5 {
            backend
                .store_experience(&make_experience(
                    ExperienceType::Success,
                    &format!("ctx {i}"),
                    &format!("out {i}"),
                    0.5,
                ))
                .unwrap();
        }

        backend.cleanup(false, None, Some(2)).unwrap();

        let results = backend.retrieve_experiences(None, None, 0.0).unwrap();
        assert!(results.len() <= 2, "expected <= 2, got {}", results.len());
    }

    #[test]
    fn test_experience_backend_trait() {
        let (mut backend, _tmp) = make_backend();
        let exp = make_experience(
            ExperienceType::Insight,
            "trait context",
            "trait outcome",
            0.7,
        );

        let id = ExperienceBackend::add(&mut backend, &exp).unwrap();
        assert!(!id.is_empty());

        let results = ExperienceBackend::search(&backend, "trait", None, 0.0, 10).unwrap();
        assert_eq!(results.len(), 1);

        let stats = ExperienceBackend::get_statistics(&backend).unwrap();
        assert_eq!(stats.total_experiences, 1);
    }

    #[test]
    fn test_roundtrip_metadata() {
        let (mut backend, _tmp) = make_backend();
        let mut exp = make_experience(ExperienceType::Success, "meta ctx", "meta out", 0.8);
        exp.metadata.insert(
            "tool".to_string(),
            serde_json::Value::String("cargo".to_string()),
        );
        exp.metadata
            .insert("count".to_string(), serde_json::json!(42));

        backend.store_experience(&exp).unwrap();

        let results = backend.retrieve_experiences(Some(1), None, 0.0).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].metadata.get("tool").and_then(|v| v.as_str()),
            Some("cargo")
        );
        assert_eq!(
            results[0].metadata.get("count").and_then(|v| v.as_i64()),
            Some(42)
        );
    }

    #[test]
    fn test_roundtrip_tags() {
        let (mut backend, _tmp) = make_backend();
        let mut exp = make_experience(ExperienceType::Success, "tags ctx", "tags out", 0.8);
        exp.tags = vec!["rust".into(), "graph".into(), "memory".into()];

        backend.store_experience(&exp).unwrap();

        let results = backend.retrieve_experiences(Some(1), None, 0.0).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].tags, vec!["rust", "graph", "memory"]);
    }

    #[test]
    fn test_multiple_stores() {
        let (mut backend, _tmp) = make_backend();
        for i in 0..10 {
            backend
                .store_experience(&make_experience(
                    ExperienceType::Success,
                    &format!("multi ctx {i}"),
                    &format!("multi out {i}"),
                    0.5 + (i as f64) * 0.05,
                ))
                .unwrap();
        }

        let results = backend.retrieve_experiences(None, None, 0.0).unwrap();
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_close_is_safe() {
        let (mut backend, _tmp) = make_backend();
        backend.close(); // should not panic
        backend.close(); // calling twice should also be fine
    }

    #[test]
    fn test_empty_search() {
        let (backend, _tmp) = make_backend();
        let results = MemoryBackend::search(&backend, "nonexistent_xyz", None, 0.0, 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_store_with_special_characters() {
        let (mut backend, _tmp) = make_backend();
        let exp = make_experience(
            ExperienceType::Success,
            "context with 'quotes' and \\backslash",
            "outcome with \"double quotes\" and $dollar",
            0.7,
        );

        let id = backend.store_experience(&exp).unwrap();
        assert!(!id.is_empty());

        let results = backend.retrieve_experiences(Some(1), None, 0.0).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].context.contains("'quotes'"));
        assert!(results[0].outcome.contains("$dollar"));
    }
}
