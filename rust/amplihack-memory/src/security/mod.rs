//! Security layer for memory-enabled agents.
//!
//! Provides capability-based access control, credential scrubbing,
//! and query validation to ensure safe memory operations.

mod credential_scrubber;
mod query_validator;
mod scope_enforcer;

#[cfg(test)]
mod tests;

pub use credential_scrubber::CredentialScrubber;
pub use query_validator::QueryValidator;
pub use scope_enforcer::{AgentCapabilities, ScopeLevel};

use crate::errors::MemoryError;
use crate::experience::{Experience, ExperienceType};

/// Wrapper around an experience backend that enforces capability-based access control.
///
/// All write operations are credential-scrubbed before reaching the inner store.
pub struct SecureMemoryBackend<S> {
    /// The underlying experience storage backend.
    pub store: S,
    /// Capabilities controlling what operations are permitted.
    pub capabilities: AgentCapabilities,
    scrubber: CredentialScrubber,
}

impl<S: crate::backends::ExperienceBackend> SecureMemoryBackend<S> {
    /// Wrap an experience backend with security enforcement.
    ///
    /// Applies `capabilities` for access control and scrubs credentials on write.
    pub fn new(store: S, capabilities: AgentCapabilities) -> Self {
        Self {
            store,
            capabilities,
            scrubber: CredentialScrubber::new(),
        }
    }

    /// Add experience with security checks.
    pub fn add_experience(&mut self, experience: &Experience) -> crate::Result<String> {
        if !self.capabilities.can_store_experience(experience) {
            return Err(MemoryError::SecurityViolation(format!(
                "Agent not allowed to store {} experiences",
                experience.experience_type
            )));
        }

        let (scrubbed, _) = self.scrubber.scrub_experience(experience)?;
        self.store.add(&scrubbed)
    }

    /// Search experiences with security checks.
    pub fn search(
        &self,
        query: &str,
        experience_type: Option<ExperienceType>,
        min_confidence: f64,
        limit: usize,
    ) -> crate::Result<Vec<Experience>> {
        if let Some(et) = experience_type {
            if !self.capabilities.can_retrieve_experience_type(et) {
                return Err(MemoryError::SecurityViolation(format!(
                    "Agent not allowed to retrieve {} experiences",
                    et
                )));
            }
        }

        self.store
            .search(query, experience_type, min_confidence, limit)
    }

    /// Validate custom SQL query.
    pub fn validate_custom_query(&self, sql: &str) -> crate::Result<()> {
        if !QueryValidator::is_safe_query(sql) {
            return Err(MemoryError::SecurityViolation(
                "Only SELECT queries are allowed".into(),
            ));
        }
        QueryValidator::validate_query(sql, self.capabilities.max_query_cost)
    }
}
