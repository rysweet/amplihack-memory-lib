//! Scope-based access control for memory-enabled agents.

use crate::errors::MemoryError;
use crate::experience::{Experience, ExperienceType};

pub enum ScopeLevel {
    /// Access limited to the current session only.
    SessionOnly,
    /// Read access across sessions of the same agent.
    CrossSessionRead,
    /// Read and write access across sessions of the same agent.
    CrossSessionWrite,
    /// Read access to any agent's memory in the system.
    GlobalRead,
    /// Full read/write access to any agent's memory in the system.
    GlobalWrite,
}

impl ScopeLevel {
    // Scope enforcement is not yet integrated at memory access points.
    // ScopeLevel variants are defined for future capability-based access control.
}

/// Security capabilities governing what a memory-enabled agent can do.
///
/// Capabilities are checked before every store/retrieve/query operation
/// by `SecureMemoryBackend`.
pub struct AgentCapabilities {
    /// The maximum scope this agent is allowed to access.
    pub scope: ScopeLevel,
    /// Which experience types the agent may store and retrieve.
    /// An empty list means all types are allowed.
    pub allowed_experience_types: Vec<ExperienceType>,
    /// Maximum estimated query cost before a query is rejected.
    pub max_query_cost: i32,
    /// Whether the agent may access pattern-type experiences.
    pub can_access_patterns: bool,
    /// Maximum storage quota in megabytes.
    pub memory_quota_mb: i32,
}

impl AgentCapabilities {
    /// Create agent capabilities with the given scope, allowed experience types,
    /// query cost limit, pattern access flag, and memory quota.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::InvalidInput`] if `max_query_cost` or
    /// `memory_quota_mb` is not positive.
    pub fn new(
        scope: ScopeLevel,
        allowed_experience_types: Vec<ExperienceType>,
        max_query_cost: i32,
        can_access_patterns: bool,
        memory_quota_mb: i32,
    ) -> crate::Result<Self> {
        if max_query_cost <= 0 {
            return Err(MemoryError::InvalidInput(
                "max_query_cost must be positive integer".into(),
            ));
        }
        if memory_quota_mb <= 0 {
            return Err(MemoryError::InvalidInput(
                "memory_quota_mb must be positive integer".into(),
            ));
        }
        Ok(Self {
            scope,
            allowed_experience_types,
            max_query_cost,
            can_access_patterns,
            memory_quota_mb,
        })
    }

    /// Check if agent can store this experience type.
    pub fn can_store_experience(&self, experience: &Experience) -> bool {
        if self.allowed_experience_types.is_empty() {
            return true;
        }
        self.allowed_experience_types
            .contains(&experience.experience_type)
    }

    /// Check if agent can retrieve this experience type.
    pub fn can_retrieve_experience_type(&self, experience_type: ExperienceType) -> bool {
        if self.allowed_experience_types.is_empty() {
            return true;
        }
        if experience_type == ExperienceType::Pattern && !self.can_access_patterns {
            return false;
        }
        self.allowed_experience_types.contains(&experience_type)
    }
}
