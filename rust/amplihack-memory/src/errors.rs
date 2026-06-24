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

    /// A store with a material on-disk footprint read back empty (or unreadable)
    /// — the #107 silent empty-read signature. The store-open path fails closed
    /// with this rather than returning a usable handle over an apparently-empty
    /// graph (which a consumer could then checkpoint and upgrade to v41, making
    /// the loss permanent). `footprint_bytes` is the committed on-disk size that
    /// proves the store *should* be populated; `read_count` is what the read path
    /// actually returned (typically `0`).
    #[error(
        "suspected data loss: {read_count} nodes read from a populated \
         {footprint_bytes}-byte store (store left intact, not checkpointed)"
    )]
    SuspectedDataLoss {
        /// Committed on-disk footprint (bytes) of the store that read back empty.
        footprint_bytes: u64,
        /// Node count the read path returned (the empty/suspect value).
        read_count: usize,
    },

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

    #[test]
    fn test_display_suspected_data_loss() {
        // #107: the fail-closed empty-read error must name both the on-disk
        // footprint (proof the store should be populated) and the empty count it
        // read back, so operators can see the silent-empty signature at a glance.
        let e = MemoryError::SuspectedDataLoss {
            footprint_bytes: 30_961_664,
            read_count: 0,
        };
        let s = e.to_string();
        assert!(s.contains("suspected data loss"), "got: {s}");
        assert!(
            s.contains("30961664"),
            "footprint must be surfaced, got: {s}"
        );
        assert!(s.contains('0'), "read count must be surfaced, got: {s}");
    }
}
