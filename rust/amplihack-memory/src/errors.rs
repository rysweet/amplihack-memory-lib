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

pub type Result<T> = std::result::Result<T, MemoryError>;
