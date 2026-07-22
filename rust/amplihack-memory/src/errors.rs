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

    /// The Design C applier's stamped lease epoch no longer matches the live
    /// on-disk lease epoch at apply time — another process acquired the store
    /// lease. The applier **fails closed** rather than mutating the store under a
    /// stale epoch. This is the concrete `NoSplitBrain` guard from
    /// `specs/FencedApplier.tla`. Fields are structural only (no user data): the
    /// epoch the applier believed it held vs. the epoch actually live.
    #[error("epoch fenced: stamped epoch {expected} but live lease epoch is {found}")]
    EpochFenced {
        /// The epoch this applier stamped when it acquired the lease.
        expected: u64,
        /// The epoch currently live in the on-disk lease.
        found: u64,
    },

    /// A CRC mismatch on a **committed interior** record of the durable intent
    /// log (i.e. not a torn tail). Such a record was acked, so it is **never**
    /// skipped — skipping it would violate `PrefixConsistency`. Fields are
    /// structural only: the segment index and byte offset of the bad frame.
    #[error("intent-log corruption: bad crc at segment {segment} offset {offset}")]
    LogCorruption {
        /// Zero-based segment index containing the corrupt frame.
        segment: u64,
        /// Byte offset of the corrupt frame within its segment.
        offset: u64,
    },

    /// A durable-log record of an unknown `WriteIntent` kind or version was
    /// read. It is **never silently dropped** (dropping an acked intent would
    /// break `NoLostAckedWrite`); the applier fails closed so an operator
    /// deploys a newer daemon rather than losing the write.
    #[error("unsupported intent kind or version in durable log")]
    UnsupportedIntentVersion,

    /// Another process (or another live handle in this process) already holds
    /// the single-writer ownership lock on the persistent store. Opening the
    /// store for writing takes an exclusive, non-blocking `flock` and **fails
    /// closed** with this error rather than opening a second concurrent writer
    /// over the same `lbug` store (which would be split-brain at the store
    /// layer). This is the store-ownership half of `NoSplitBrain`
    /// (`specs/FencedApplier.tla`); the lock releases when the holding handle is
    /// dropped. `path` is the store path whose lock is held (structural only).
    #[error("store already locked by another single-writer owner: {path}")]
    AlreadyLocked {
        /// The persistent store path whose writer lock is already held.
        path: String,
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
