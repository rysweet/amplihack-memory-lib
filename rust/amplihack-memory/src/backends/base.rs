//! Abstract base traits for memory storage backends.

use crate::experience::{Experience, ExperienceType};

/// Trait for memory storage backends.
pub trait MemoryBackend {
    /// Initialize the storage schema.
    fn initialize_schema(&mut self) -> crate::Result<()>;

    /// Store an experience.
    fn store_experience(&mut self, experience: &Experience) -> crate::Result<String>;

    /// Retrieve experiences.
    fn retrieve_experiences(
        &self,
        limit: Option<usize>,
        experience_type: Option<ExperienceType>,
        min_confidence: f64,
    ) -> crate::Result<Vec<Experience>>;

    /// Search experiences by text query.
    fn search(
        &self,
        query: &str,
        experience_type: Option<ExperienceType>,
        min_confidence: f64,
        limit: usize,
    ) -> crate::Result<Vec<Experience>>;

    /// Get storage statistics.
    fn get_statistics(&self) -> crate::Result<StorageStatistics>;

    /// Close database connection.
    fn close(&mut self);

    /// Run cleanup operations.
    fn cleanup(
        &mut self,
        auto_compress: bool,
        max_age_days: Option<i64>,
        max_experiences: Option<usize>,
    ) -> crate::Result<()>;
}

/// Trait alias for experience storage operations used by [`crate::security::SecureMemoryBackend`].
pub trait ExperienceBackend {
    /// Store an experience and return its assigned id.
    fn add(&mut self, experience: &Experience) -> crate::Result<String>;
    /// Search experiences by text query with optional type and confidence filters.
    fn search(
        &self,
        query: &str,
        experience_type: Option<ExperienceType>,
        min_confidence: f64,
        limit: usize,
    ) -> crate::Result<Vec<Experience>>;
    /// Return aggregate storage statistics.
    fn get_statistics(&self) -> crate::Result<StorageStatistics>;
}

/// Aggregate statistics about the storage backend's contents.
#[derive(Debug, Clone)]
pub struct StorageStatistics {
    /// Total number of stored experiences across all types.
    pub total_experiences: usize,
    /// Breakdown of experience counts by type.
    pub by_type: std::collections::HashMap<ExperienceType, usize>,
    /// Approximate on-disk size in kilobytes.
    pub storage_size_kb: f64,
    /// Number of experiences that have been compressed.
    pub compressed_experiences: usize,
    /// Ratio of compressed to total experiences (1.0 = none compressed).
    pub compression_ratio: f64,
}

impl Default for StorageStatistics {
    fn default() -> Self {
        Self {
            total_experiences: 0,
            by_type: std::collections::HashMap::new(),
            storage_size_kb: 0.0,
            compressed_experiences: 0,
            compression_ratio: 1.0,
        }
    }
}
