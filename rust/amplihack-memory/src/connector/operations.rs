//! Operations on [`MemoryConnector`]: store, retrieve, search, statistics, cleanup, close.

use crate::backends::base::{MemoryBackend, StorageStatistics};
use crate::experience::{Experience, ExperienceType};

use super::{BackendInner, MemoryConnector};

impl MemoryConnector {
    /// Store an experience in the database.
    pub fn store_experience(&mut self, experience: &Experience) -> crate::Result<String> {
        match &mut self.backend {
            BackendInner::Sqlite(b) => b.store_experience(experience),
            #[cfg(feature = "kuzu")]
            BackendInner::Kuzu(b) => b.store_experience(experience),
        }
    }

    /// Retrieve experiences for this agent.
    pub fn retrieve_experiences(
        &self,
        limit: Option<usize>,
        experience_type: Option<ExperienceType>,
        min_confidence: f64,
    ) -> crate::Result<Vec<Experience>> {
        match &self.backend {
            BackendInner::Sqlite(b) => {
                b.retrieve_experiences(limit, experience_type, min_confidence)
            }
            #[cfg(feature = "kuzu")]
            BackendInner::Kuzu(b) => b.retrieve_experiences(limit, experience_type, min_confidence),
        }
    }

    /// Search experiences by text query.
    pub fn search(
        &self,
        query: &str,
        experience_type: Option<ExperienceType>,
        min_confidence: f64,
        limit: usize,
    ) -> crate::Result<Vec<Experience>> {
        match &self.backend {
            BackendInner::Sqlite(b) => b.search(query, experience_type, min_confidence, limit),
            #[cfg(feature = "kuzu")]
            BackendInner::Kuzu(b) => {
                MemoryBackend::search(b, query, experience_type, min_confidence, limit)
            }
        }
    }

    /// Get storage statistics.
    pub fn get_statistics(&self) -> crate::Result<StorageStatistics> {
        match &self.backend {
            BackendInner::Sqlite(b) => b.get_statistics(),
            #[cfg(feature = "kuzu")]
            BackendInner::Kuzu(b) => MemoryBackend::get_statistics(b),
        }
    }

    /// Run cleanup operations.
    pub fn cleanup(
        &mut self,
        auto_compress: bool,
        max_age_days: Option<i64>,
        max_experiences: Option<usize>,
    ) -> crate::Result<()> {
        match &mut self.backend {
            BackendInner::Sqlite(b) => b.cleanup(auto_compress, max_age_days, max_experiences),
            #[cfg(feature = "kuzu")]
            BackendInner::Kuzu(b) => b.cleanup(auto_compress, max_age_days, max_experiences),
        }
    }

    /// Close database connection.
    pub fn close(&mut self) {
        match &mut self.backend {
            BackendInner::Sqlite(b) => b.close(),
            #[cfg(feature = "kuzu")]
            BackendInner::Kuzu(b) => b.close(),
        }
    }
}
