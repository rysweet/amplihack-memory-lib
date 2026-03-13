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
    /// Create a new experience store.
    ///
    /// Opens (or creates) a SQLite database at `storage_path` for the given
    /// `agent_name`. When `storage_path` is `None`, a default location under
    /// the user's data directory is used.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Storage`] if the database cannot be opened.
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
            let meta = std::fs::metadata(&self.connector.db_path)
                .map_err(|e| MemoryError::Storage(format!("Cannot read db metadata: {e}")))?;
            let size_mb = meta.len() as f64 / (1024.0 * 1024.0);
            if size_mb > self.max_memory_mb as f64 {
                return Err(MemoryError::MemoryQuotaExceeded(format!(
                    "Storage quota exceeded: {size_mb:.1}MB > {}MB",
                    self.max_memory_mb
                )));
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

    // --- New parity tests ---

    #[test]
    fn test_store_creation_with_defaults() {
        let tmp = TempDir::new().unwrap();
        let store =
            ExperienceStore::new("default-agent", true, None, None, 100, Some(tmp.path())).unwrap();
        assert_eq!(store.agent_name, "default-agent");
        assert!(store.auto_compress);
        assert!(store.max_age_days.is_none());
        assert!(store.max_experiences.is_none());
        assert_eq!(store.max_memory_mb, 100);
    }

    #[test]
    fn test_store_creation_with_custom_config() {
        let tmp = TempDir::new().unwrap();
        let store = ExperienceStore::new(
            "custom-agent",
            false,
            Some(30),
            Some(500),
            50,
            Some(tmp.path()),
        )
        .unwrap();
        assert_eq!(store.agent_name, "custom-agent");
        assert!(!store.auto_compress);
        assert_eq!(store.max_age_days, Some(30));
        assert_eq!(store.max_experiences, Some(500));
        assert_eq!(store.max_memory_mb, 50);
    }

    #[test]
    fn test_store_add_and_retrieve_experience() {
        let (mut store, _tmp) = test_store();
        let exp = Experience::new(
            ExperienceType::Insight,
            "learned something".into(),
            "the insight outcome".into(),
            0.75,
        )
        .unwrap();

        let id = store.add(&exp).unwrap();
        assert!(!id.is_empty());

        let results = store.search("learned", None, 0.0, 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].context, "learned something");
        assert_eq!(results[0].outcome, "the insight outcome");
    }

    #[test]
    fn test_store_search_with_query_matching() {
        let (mut store, _tmp) = test_store();
        let exp1 = Experience::new(
            ExperienceType::Success,
            "rust compilation succeeded".into(),
            "binary produced".into(),
            0.9,
        )
        .unwrap();
        let exp2 = Experience::new(
            ExperienceType::Failure,
            "python import failed".into(),
            "module not found".into(),
            0.8,
        )
        .unwrap();

        store.add(&exp1).unwrap();
        store.add(&exp2).unwrap();

        let rust_results = store.search("rust", None, 0.0, 10).unwrap();
        assert_eq!(rust_results.len(), 1);
        assert!(rust_results[0].context.contains("rust"));

        let python_results = store.search("python", None, 0.0, 10).unwrap();
        assert_eq!(python_results.len(), 1);
        assert!(python_results[0].context.contains("python"));
    }

    #[test]
    fn test_store_search_with_confidence_filter() {
        let (mut store, _tmp) = test_store();

        let high_conf = Experience::new(
            ExperienceType::Success,
            "high confidence context".into(),
            "high confidence outcome".into(),
            0.95,
        )
        .unwrap();
        let low_conf = Experience::new(
            ExperienceType::Success,
            "low confidence context".into(),
            "low confidence outcome".into(),
            0.3,
        )
        .unwrap();

        store.add(&high_conf).unwrap();
        store.add(&low_conf).unwrap();

        // Both returned with min_confidence 0.0
        let all = store.search("confidence", None, 0.0, 10).unwrap();
        assert_eq!(all.len(), 2);

        // Only high returned with min_confidence 0.5
        let filtered = store.search("confidence", None, 0.5, 10).unwrap();
        assert_eq!(filtered.len(), 1);
        assert!(filtered[0].confidence >= 0.5);
    }

    #[test]
    fn test_store_statistics_accuracy() {
        let (mut store, _tmp) = test_store();

        let s = store.get_statistics().unwrap();
        assert_eq!(s.total_experiences, 0);

        let exp1 =
            Experience::new(ExperienceType::Success, "ctx1".into(), "out1".into(), 0.9).unwrap();
        let exp2 =
            Experience::new(ExperienceType::Failure, "ctx2".into(), "out2".into(), 0.7).unwrap();
        let exp3 =
            Experience::new(ExperienceType::Success, "ctx3".into(), "out3".into(), 0.8).unwrap();

        store.add(&exp1).unwrap();
        store.add(&exp2).unwrap();
        store.add(&exp3).unwrap();

        let stats = store.get_statistics().unwrap();
        assert_eq!(stats.total_experiences, 3);
        assert_eq!(
            *stats.by_type.get(&ExperienceType::Success).unwrap_or(&0),
            2
        );
        assert_eq!(
            *stats.by_type.get(&ExperienceType::Failure).unwrap_or(&0),
            1
        );
    }

    #[test]
    fn test_store_empty_behavior() {
        let (store, _tmp) = test_store();

        let results = store.search("anything", None, 0.0, 10).unwrap();
        assert!(results.is_empty());

        let stats = store.get_statistics().unwrap();
        assert_eq!(stats.total_experiences, 0);
    }

    #[test]
    fn test_store_multiple_experience_types() {
        let (mut store, _tmp) = test_store();

        let types = [
            ExperienceType::Success,
            ExperienceType::Failure,
            ExperienceType::Pattern,
            ExperienceType::Insight,
        ];

        for et in &types {
            let exp = Experience::new(
                *et,
                format!("{} context data", et.as_str()),
                format!("{} outcome data", et.as_str()),
                0.8,
            )
            .unwrap();
            store.add(&exp).unwrap();
        }

        let stats = store.get_statistics().unwrap();
        assert_eq!(stats.total_experiences, 4);

        // Filter by type
        let successes = store
            .search("context", Some(ExperienceType::Success), 0.0, 10)
            .unwrap();
        assert_eq!(successes.len(), 1);

        let failures = store
            .search("context", Some(ExperienceType::Failure), 0.0, 10)
            .unwrap();
        assert_eq!(failures.len(), 1);
    }

    #[test]
    fn test_store_validation_empty_context() {
        let (mut store, _tmp) = test_store();
        let exp =
            Experience::new(ExperienceType::Success, "valid".into(), "valid".into(), 0.5).unwrap();

        // Manually construct an experience with empty context
        let bad = Experience {
            context: "   ".into(),
            ..exp
        };
        let result = store.add(&bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_store_validation_empty_outcome() {
        let (mut store, _tmp) = test_store();
        let exp =
            Experience::new(ExperienceType::Success, "valid".into(), "valid".into(), 0.5).unwrap();

        let bad = Experience {
            outcome: "  ".into(),
            ..exp
        };
        let result = store.add(&bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_store_validation_confidence_out_of_range() {
        let (mut store, _tmp) = test_store();
        let exp =
            Experience::new(ExperienceType::Success, "valid".into(), "valid".into(), 0.5).unwrap();

        let bad_high = Experience {
            confidence: 1.5,
            ..exp.clone()
        };
        assert!(store.add(&bad_high).is_err());

        let bad_low = Experience {
            confidence: -0.1,
            ..exp
        };
        assert!(store.add(&bad_low).is_err());
    }

    #[test]
    fn test_store_retention_max_experiences() {
        let tmp = TempDir::new().unwrap();
        let mut store = ExperienceStore::new(
            "retention-agent",
            true,
            None,
            Some(3),
            100,
            Some(tmp.path()),
        )
        .unwrap();

        for i in 0..5 {
            let exp = Experience::new(
                ExperienceType::Success,
                format!("context {i}"),
                format!("outcome {i}"),
                0.8,
            )
            .unwrap();
            store.add(&exp).unwrap();
        }

        let stats = store.get_statistics().unwrap();
        assert!(
            stats.total_experiences <= 3,
            "expected at most 3 experiences after cleanup, got {}",
            stats.total_experiences
        );
    }

    #[test]
    fn test_store_search_limit() {
        let (mut store, _tmp) = test_store();

        for i in 0..10 {
            let exp = Experience::new(
                ExperienceType::Success,
                format!("searchable context {i}"),
                format!("outcome {i}"),
                0.8,
            )
            .unwrap();
            store.add(&exp).unwrap();
        }

        let limited = store.search("searchable", None, 0.0, 3).unwrap();
        assert!(limited.len() <= 3);
    }

    #[test]
    fn test_store_search_no_match() {
        let (mut store, _tmp) = test_store();
        let exp = Experience::new(
            ExperienceType::Success,
            "alpha beta".into(),
            "gamma delta".into(),
            0.9,
        )
        .unwrap();
        store.add(&exp).unwrap();

        let results = store.search("zzzznonexistent", None, 0.0, 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_store_many_adds_and_stats() {
        let (mut store, _tmp) = test_store();

        for i in 0..20 {
            let et = if i % 2 == 0 {
                ExperienceType::Success
            } else {
                ExperienceType::Failure
            };
            let exp = Experience::new(
                et,
                format!("ctx{i}"),
                format!("out{i}"),
                0.5 + (i as f64) * 0.02,
            )
            .unwrap();
            store.add(&exp).unwrap();
        }

        let stats = store.get_statistics().unwrap();
        assert_eq!(stats.total_experiences, 20);
        assert_eq!(
            *stats.by_type.get(&ExperienceType::Success).unwrap_or(&0),
            10
        );
        assert_eq!(
            *stats.by_type.get(&ExperienceType::Failure).unwrap_or(&0),
            10
        );
    }

    #[test]
    fn test_store_boundary_confidence_values() {
        let (mut store, _tmp) = test_store();

        let exp_zero = Experience::new(
            ExperienceType::Success,
            "zero conf".into(),
            "outcome zero".into(),
            0.0,
        )
        .unwrap();
        store.add(&exp_zero).unwrap();

        let exp_one = Experience::new(
            ExperienceType::Success,
            "one conf".into(),
            "outcome one".into(),
            1.0,
        )
        .unwrap();
        store.add(&exp_one).unwrap();

        let stats = store.get_statistics().unwrap();
        assert_eq!(stats.total_experiences, 2);
    }
}
