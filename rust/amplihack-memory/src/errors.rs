//! Custom error types for amplihack-memory.

use thiserror::Error;

/// Base error type for memory operations.
#[derive(Debug, Error)]
pub enum MemoryError {
    /// An experience could not be found.
    #[error("experience not found: {0}")]
    ExperienceNotFound(String),

    /// An experience failed validation.
    #[error("invalid experience: {0}")]
    InvalidExperience(String),

    /// Memory quota has been exceeded.
    #[error("memory quota exceeded: {0}")]
    MemoryQuotaExceeded(String),

    /// Security policy violation.
    #[error("security violation: {0}")]
    SecurityViolation(String),

    /// Query cost exceeded the allowed limit.
    #[error("query cost exceeded: {0}")]
    QueryCostExceeded(String),

    /// Database or storage error.
    #[error("storage error: {0}")]
    Storage(String),

    /// Invalid input provided by the caller.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Generic internal error.
    #[error("{0}")]
    Internal(String),
}

/// Convenience type alias for `std::result::Result<T, MemoryError>`.
pub type Result<T> = std::result::Result<T, MemoryError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_experience_not_found() {
        let e = MemoryError::ExperienceNotFound("abc123".into());
        assert_eq!(e.to_string(), "experience not found: abc123");
    }

    #[test]
    fn test_display_invalid_experience() {
        let e = MemoryError::InvalidExperience("bad data".into());
        assert_eq!(e.to_string(), "invalid experience: bad data");
    }

    #[test]
    fn test_display_memory_quota_exceeded() {
        let e = MemoryError::MemoryQuotaExceeded("over limit".into());
        assert_eq!(e.to_string(), "memory quota exceeded: over limit");
    }

    #[test]
    fn test_display_security_violation() {
        let e = MemoryError::SecurityViolation("unauthorized".into());
        assert_eq!(e.to_string(), "security violation: unauthorized");
    }

    #[test]
    fn test_display_query_cost_exceeded() {
        let e = MemoryError::QueryCostExceeded("too expensive".into());
        assert_eq!(e.to_string(), "query cost exceeded: too expensive");
    }

    #[test]
    fn test_display_storage() {
        let e = MemoryError::Storage("disk full".into());
        assert_eq!(e.to_string(), "storage error: disk full");
    }

    #[test]
    fn test_display_invalid_input() {
        let e = MemoryError::InvalidInput("missing field".into());
        assert_eq!(e.to_string(), "invalid input: missing field");
    }

    #[test]
    fn test_display_internal() {
        let e = MemoryError::Internal("something broke".into());
        assert_eq!(e.to_string(), "something broke");
    }
}
