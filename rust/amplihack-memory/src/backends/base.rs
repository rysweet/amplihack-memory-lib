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

/// Trait alias for experience storage operations used by SecureMemoryBackend.
pub trait ExperienceBackend {
    fn add(&mut self, experience: &Experience) -> crate::Result<String>;
    fn search(
        &self,
        query: &str,
        experience_type: Option<ExperienceType>,
        min_confidence: f64,
        limit: usize,
    ) -> crate::Result<Vec<Experience>>;
    fn get_statistics(&self) -> crate::Result<StorageStatistics>;
}

/// Storage statistics.
#[derive(Debug, Clone)]
pub struct StorageStatistics {
    pub total_experiences: usize,
    pub by_type: std::collections::HashMap<ExperienceType, usize>,
    pub storage_size_kb: f64,
    pub compressed_experiences: usize,
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
