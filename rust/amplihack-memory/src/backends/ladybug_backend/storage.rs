//! MemoryBackend trait implementation for the Ladybug backend.

use std::collections::HashMap;

use chrono::Utc;
use ladybug_graph_rs::Property;

use super::{walkdir_size, LadybugBackend};
use crate::backends::base::{MemoryBackend, StorageStatistics};
use crate::errors::MemoryError;
use crate::experience::{Experience, ExperienceType};

use tracing::warn;

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
            // Safety: `lim` is a `usize`, guaranteed to be a non-negative integer.
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

        super::search::search_experiences(self, query, experience_type, min_confidence, limit)
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
            if let Err(e) = self.exec(
                "MATCH (e:Experience) \
                 WHERE e.agent_name = $agent AND e.timestamp < $cutoff AND e.compressed = false \
                 SET e.compressed = true",
                &[
                    ("agent", Property::from(agent.as_str())),
                    ("cutoff", Property::Int64(cutoff)),
                ],
            ) {
                warn!("cleanup: failed to auto-compress old experiences: {e}");
            }
        }

        // 2. Delete experiences older than max_age_days
        if let Some(days) = max_age_days {
            let cutoff = (Utc::now() - chrono::Duration::days(days)).timestamp();
            if let Err(e) = self.exec(
                "MATCH (e:Experience) \
                 WHERE e.agent_name = $agent AND e.timestamp < $cutoff \
                 DETACH DELETE e",
                &[
                    ("agent", Property::from(agent.as_str())),
                    ("cutoff", Property::Int64(cutoff)),
                ],
            ) {
                warn!("cleanup: failed to delete experiences older than {days} days: {e}");
            }
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

                // Safety: `max_exp` is a `usize`, guaranteed to be a non-negative integer.
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
                            if let Err(e) = self.exec(
                                "MATCH (e:Experience {experience_id: $id}) DETACH DELETE e",
                                &[("id", Property::from(id))],
                            ) {
                                warn!("cleanup: failed to delete excess experience {id}: {e}");
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
