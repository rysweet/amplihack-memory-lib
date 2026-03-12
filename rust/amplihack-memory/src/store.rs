//! High-level experience storage and retrieval with automatic management.

use std::path::Path;

use crate::backends::base::StorageStatistics;
use crate::connector::MemoryConnector;
use crate::errors::MemoryError;
use crate::experience::{Experience, ExperienceType};

/// High-level storage and retrieval of experiences with auto-management.
///
/// Features:
/// - Automatic compression of old experiences
/// - Retention policies (max_age_days, max_experiences)
/// - Duplicate detection
/// - Full-text search
/// - Statistics tracking
pub struct ExperienceStore {
    pub agent_name: String,
    pub auto_compress: bool,
    pub max_age_days: Option<i64>,
    pub max_experiences: Option<usize>,
    pub max_memory_mb: i32,
    connector: MemoryConnector,
}

impl ExperienceStore {
    pub fn new(
        agent_name: &str,
        auto_compress: bool,
        max_age_days: Option<i64>,
        max_experiences: Option<usize>,
        max_memory_mb: i32,
        storage_path: Option<&Path>,
    ) -> crate::Result<Self> {
        let connector =
            MemoryConnector::new(agent_name, storage_path, max_memory_mb, auto_compress)?;

        Ok(Self {
            agent_name: agent_name.to_string(),
            auto_compress,
            max_age_days,
            max_experiences,
            max_memory_mb,
            connector,
        })
    }

    /// Add experience with automatic management.
    pub fn add(&mut self, experience: &Experience) -> crate::Result<String> {
        self.validate_experience(experience)?;
        self.check_quota()?;

        let exp_id = self.connector.store_experience(experience)?;

        if self.auto_compress || self.max_age_days.is_some() || self.max_experiences.is_some() {
            self.connector
                .cleanup(self.auto_compress, self.max_age_days, self.max_experiences)?;
        }

        Ok(exp_id)
    }

    /// Search experiences by text query.
    pub fn search(
        &self,
        query: &str,
        experience_type: Option<ExperienceType>,
        min_confidence: f64,
        limit: usize,
    ) -> crate::Result<Vec<Experience>> {
        self.connector
            .search(query, experience_type, min_confidence, limit)
    }

    /// Get storage statistics.
    pub fn get_statistics(&self) -> crate::Result<StorageStatistics> {
        self.connector.get_statistics()
    }

    fn validate_experience(&self, experience: &Experience) -> crate::Result<()> {
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
        Ok(())
    }

    fn check_quota(&self) -> crate::Result<()> {
        if self.connector.db_path.exists() {
            if let Ok(meta) = std::fs::metadata(&self.connector.db_path) {
                let size_mb = meta.len() as f64 / (1024.0 * 1024.0);
                if size_mb > self.max_memory_mb as f64 {
                    return Err(MemoryError::MemoryQuotaExceeded(format!(
                        "Storage quota exceeded: {size_mb:.1}MB > {}MB",
                        self.max_memory_mb
                    )));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_store() -> (ExperienceStore, TempDir) {
        let tmp = TempDir::new().unwrap();
        let store =
            ExperienceStore::new("test-agent", true, None, None, 100, Some(tmp.path())).unwrap();
        (store, tmp)
    }

    #[test]
    fn test_add_and_search() {
        let (mut store, _tmp) = test_store();
        let exp = Experience::new(
            ExperienceType::Success,
            "test context".into(),
            "test outcome".into(),
            0.9,
        )
        .unwrap();

        store.add(&exp).unwrap();

        let results = store.search("test", None, 0.0, 10).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_statistics() {
        let (mut store, _tmp) = test_store();
        let exp =
            Experience::new(ExperienceType::Success, "ctx".into(), "out".into(), 0.8).unwrap();
        store.add(&exp).unwrap();

        let stats = store.get_statistics().unwrap();
        assert_eq!(stats.total_experiences, 1);
    }
}
