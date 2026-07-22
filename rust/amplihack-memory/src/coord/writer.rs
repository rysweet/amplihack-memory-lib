//! Consumer-side writer client for the durable shared intent log.
//!
//! [`WriterClient`] is the *only* write path an ephemeral engineer has: it links
//! the coordination layer only and **cannot** open the `lbug` store. `append`
//! returns a [`LogOffset`](crate::coord::LogOffset) only after the record is
//! `fsync`'d — the durability ack. After it returns the process may crash, be
//! `SIGKILL`'d, or have its worktree destroyed; the write is already durable and
//! will be applied by the daemon's single fenced applier (`NoLostAckedWrite`).

use uuid::Uuid;

use crate::coord::intent_log::SegmentedLog;
use crate::coord::{CoordConfig, LogOffset, WriteIntent};
use crate::errors::Result;

/// A connected writer over an already-provisioned coordination directory.
pub struct WriterClient {
    log: SegmentedLog,
}

impl WriterClient {
    /// Connect to the coordination directory.
    ///
    /// # Errors
    /// **Fails closed** if the coordination directory is missing or unreadable —
    /// there is deliberately no per-agent fallback store (that would be the
    /// rejected design B, which loses writes).
    pub fn connect(config: &CoordConfig) -> Result<Self> {
        Ok(Self {
            log: SegmentedLog::open(config)?,
        })
    }

    /// Append a pre-built intent (already carrying its `intent_id`) and return
    /// its durable offset. Returns **only after** the record is `fsync`'d.
    ///
    /// Multi-process concurrent-safe and at-least-once at the transport level;
    /// exactly-once is enforced at apply time by the applier (keyed on
    /// `intent_id`), so re-appending the same `intent_id` is safe.
    ///
    /// # Errors
    /// [`crate::MemoryError::Storage`] on any framing/I-O failure.
    pub fn append(&self, intent: &WriteIntent) -> Result<LogOffset> {
        self.log.append(intent)
    }

    /// Assign a fresh v4 `intent_id`, append, and return both the durable offset
    /// and the id (so the caller can correlate the eventual apply).
    ///
    /// # Errors
    /// [`crate::MemoryError::Storage`] on any framing/I-O failure.
    pub fn append_new(&self, mut intent: WriteIntent) -> Result<(LogOffset, Uuid)> {
        let id = Uuid::new_v4();
        intent.set_intent_id(id);
        let off = self.log.append(&intent)?;
        Ok((off, id))
    }
}
