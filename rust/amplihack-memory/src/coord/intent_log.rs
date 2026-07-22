//! Durable, append-only, multi-process-safe shared intent log
//! (`PrefixConsistency` + `NoLostAckedWrite`, `specs/DurableLog.tla`).
//!
//! Writers append [`WriteIntent`]s; each record is framed `len | payload |
//! crc32`, written under a brief append `flock` with `O_APPEND`, and `fsync`'d
//! before the offset is returned — the `fsync` is the durability ack. The log is
//! a single append-only total order split into fixed-size segments; a single
//! [`Applier`](crate::coord::Applier) consumes it strictly in order.
//!
//! On reopen the reader distinguishes a **torn tail** (a partial final record —
//! quarantined and truncated, because it was never acked) from
//! **committed-interior corruption** (a CRC mismatch on a non-tail record — a
//! hard [`MemoryError::LogCorruption`], never skipped).

use std::collections::HashMap;
use std::io::Write;
#[cfg(feature = "persistent")]
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::coord::{coord_dirs_exist, crc32, io_err, CoordConfig, FlockGuard};
use crate::errors::{MemoryError, Result};
use crate::{AccessKind, FactInput, StoreFactOptions};

/// An opaque, totally-ordered position in the durable log (segment index + byte
/// offset). Returned by append, persisted by the applier as the applied-index,
/// and monotonically increasing across every append.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct LogOffset {
    /// Zero-based segment index.
    pub(crate) segment: u64,
    /// Byte offset of the record within its segment.
    pub(crate) offset: u64,
}

impl LogOffset {
    /// The origin of an empty log (segment 0, offset 0).
    #[cfg(feature = "persistent")]
    pub(crate) const fn origin() -> Self {
        Self {
            segment: 0,
            offset: 0,
        }
    }
}

/// A versioned, `serde`-tagged mutation intent. One variant per store mutation a
/// consumer performs; each carries an `intent_id` used for idempotent replay.
///
/// The wire form is internally tagged on `kind` (snake_case) and uses
/// `deny_unknown_fields`: an unknown kind or field **fails closed** (dropping an
/// acked intent would break `NoLostAckedWrite`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
pub enum WriteIntent {
    /// Store a semantic fact (`CognitiveMemory::store_fact`).
    StoreFact {
        /// Idempotency key for this intent.
        intent_id: Uuid,
        /// Authoring agent.
        agent_name: String,
        /// Concept / topic key.
        concept: String,
        /// Factual content.
        content: String,
        /// Confidence in `[0.0, 1.0]`.
        confidence: f64,
        /// Opaque source identifier.
        source_id: String,
        /// Optional categorical tags.
        tags: Option<Vec<String>>,
        /// Optional arbitrary metadata.
        metadata: Option<HashMap<String, serde_json::Value>>,
    },
    /// Store a procedure (`CognitiveMemory::store_procedure`).
    StoreProcedure {
        /// Idempotency key for this intent.
        intent_id: Uuid,
        /// Authoring agent.
        agent_name: String,
        /// Procedure name (idempotent upsert-by-name).
        name: String,
        /// Ordered steps.
        steps: Vec<String>,
        /// Optional prerequisites.
        prerequisites: Option<Vec<String>>,
    },
    /// Store an episode (`CognitiveMemory::store_episode`).
    StoreEpisode {
        /// Idempotency key for this intent.
        intent_id: Uuid,
        /// Authoring agent.
        agent_name: String,
        /// Episode content.
        content: String,
        /// Source label.
        source_label: String,
        /// Optional explicit temporal index.
        temporal_index: Option<i64>,
        /// Optional arbitrary metadata.
        metadata: Option<HashMap<String, serde_json::Value>>,
    },
    /// Store a prospective trigger-action memory (`store_prospective`).
    StoreProspective {
        /// Idempotency key for this intent.
        intent_id: Uuid,
        /// Authoring agent.
        agent_name: String,
        /// Human-readable description.
        description: String,
        /// Trigger condition.
        trigger_condition: String,
        /// Action to take on trigger.
        action_on_trigger: String,
        /// Priority.
        priority: i32,
    },
    /// Link a fact to its source episodes (`link_fact_to_episodes`).
    LinkFactToEpisodes {
        /// Idempotency key for this intent.
        intent_id: Uuid,
        /// Authoring agent.
        agent_name: String,
        /// Fact node id.
        fact_id: String,
        /// Source episode ids.
        episode_ids: Vec<String>,
    },
    /// Link two similar facts (`link_similar_facts`).
    LinkSimilarFacts {
        /// Idempotency key for this intent.
        intent_id: Uuid,
        /// Authoring agent.
        agent_name: String,
        /// First fact id.
        fact_id_a: String,
        /// Second fact id.
        fact_id_b: String,
        /// Similarity score.
        similarity_score: f64,
    },
    /// Upsert a fact with dedup/provenance options (`upsert_fact`).
    UpsertFact {
        /// Idempotency key for this intent.
        intent_id: Uuid,
        /// Authoring agent.
        agent_name: String,
        /// Fact input payload.
        input: FactInput,
        /// Store options (dedup, provenance, similarity).
        options: StoreFactOptions,
    },
    /// Record an access (usage bump) on a node (`record_access`).
    RecordAccess {
        /// Idempotency key for this intent.
        intent_id: Uuid,
        /// Authoring agent.
        agent_name: String,
        /// Target node id.
        node_id: String,
        /// Kind of access. Renamed on the wire to avoid colliding with the
        /// enum's `kind` discriminant tag.
        #[serde(rename = "access_kind")]
        kind: AccessKind,
    },
}

impl WriteIntent {
    /// This intent's idempotency key.
    pub fn intent_id(&self) -> Uuid {
        match self {
            WriteIntent::StoreFact { intent_id, .. }
            | WriteIntent::StoreProcedure { intent_id, .. }
            | WriteIntent::StoreEpisode { intent_id, .. }
            | WriteIntent::StoreProspective { intent_id, .. }
            | WriteIntent::LinkFactToEpisodes { intent_id, .. }
            | WriteIntent::LinkSimilarFacts { intent_id, .. }
            | WriteIntent::UpsertFact { intent_id, .. }
            | WriteIntent::RecordAccess { intent_id, .. } => *intent_id,
        }
    }

    /// Overwrite this intent's `intent_id` (used by `append_new`).
    pub(crate) fn set_intent_id(&mut self, id: Uuid) {
        match self {
            WriteIntent::StoreFact { intent_id, .. }
            | WriteIntent::StoreProcedure { intent_id, .. }
            | WriteIntent::StoreEpisode { intent_id, .. }
            | WriteIntent::StoreProspective { intent_id, .. }
            | WriteIntent::LinkFactToEpisodes { intent_id, .. }
            | WriteIntent::LinkSimilarFacts { intent_id, .. }
            | WriteIntent::UpsertFact { intent_id, .. }
            | WriteIntent::RecordAccess { intent_id, .. } => *intent_id = id,
        }
    }
}

/// Length-prefix (4) + crc (4) framing overhead per record.
const FRAME_OVERHEAD: u64 = 8;

/// A decoded raw record read from the log plus the position immediately after
/// it (the applier's next resume point).
#[cfg(feature = "persistent")]
pub(crate) struct RawRecord {
    /// The JSON-encoded `WriteIntent` payload.
    pub(crate) payload: Vec<u8>,
    /// Position immediately after this record.
    pub(crate) next: LogOffset,
}

/// One on-disk segment file.
struct Segment {
    index: u64,
    // Read only by the persistent-only reader (`read_frame_at`); the writer path
    // uses `index`/`len` alone.
    #[cfg_attr(not(feature = "persistent"), allow(dead_code))]
    path: PathBuf,
    len: u64,
}

/// A handle onto the segmented, append-only intent log under a coord dir.
pub(crate) struct SegmentedLog {
    config: CoordConfig,
}

impl SegmentedLog {
    /// Open the log over an already-provisioned coordination directory. Fails
    /// closed if the coord dir does not exist (no per-agent fallback store).
    pub(crate) fn open(config: &CoordConfig) -> Result<Self> {
        if !coord_dirs_exist(config) {
            return Err(MemoryError::Storage(
                "coordination directory does not exist (fail closed; no fallback store)".into(),
            ));
        }
        Ok(Self {
            config: config.clone(),
        })
    }

    /// Open and exclusively `flock` the shared append lock file. The lock is
    /// **not** ownership — it only serializes the append write syscall + offset
    /// assignment (and torn-tail truncation) across processes. The returned
    /// `File` must be kept alive alongside the guard for the lock to hold.
    fn acquire_append_lock(&self) -> Result<(std::fs::File, FlockGuard)> {
        let lock_path = self.config.intent_log_dir().join(".append.lock");
        let lock_file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .mode(0o600)
            .open(&lock_path)
            .map_err(|e| io_err("open-append-lock", e))?;
        let guard = FlockGuard::acquire(lock_file.as_raw_fd())?;
        Ok((lock_file, guard))
    }

    /// Append an intent, returning its durable [`LogOffset`] only after the
    /// record is `fsync`'d (the ack). Multi-process concurrent-safe: the whole
    /// append runs under a brief exclusive `flock` on a lock file, so records
    /// never interleave and every offset is distinct.
    pub(crate) fn append(&self, intent: &WriteIntent) -> Result<LogOffset> {
        let payload = serde_json::to_vec(intent)
            .map_err(|e| MemoryError::Storage(format!("coord serialize-intent: {e}")))?;
        let payload_len = u32::try_from(payload.len())
            .map_err(|_| MemoryError::Storage("intent payload too large".into()))?;
        if payload_len > self.config.max_frame_bytes {
            return Err(MemoryError::Storage(format!(
                "intent payload {payload_len} exceeds max_frame_bytes {}",
                self.config.max_frame_bytes
            )));
        }
        let frame_len = FRAME_OVERHEAD + u64::from(payload_len);

        // Serialize all appends (across processes) with a brief flock. The lock
        // is NOT ownership — it only serializes the write syscall + offset
        // assignment so records never interleave.
        let (_lock_file, _guard) = self.acquire_append_lock()?;

        // Choose the target segment: roll over once the current one reaches the
        // size target (rollover only at record boundaries, so segments contain
        // whole frames).
        let segs = self.segments()?;
        let target_index = match segs.last() {
            None => 0,
            Some(last) if last.len >= self.config.segment_bytes => last.index + 1,
            Some(last) => last.index,
        };
        let seg_path = self.segment_path(target_index);
        let existing_len = segs
            .iter()
            .find(|s| s.index == target_index)
            .map_or(0, |s| s.len);

        let mut frame = Vec::with_capacity(frame_len as usize);
        frame.extend_from_slice(&payload_len.to_be_bytes());
        frame.extend_from_slice(&payload);
        frame.extend_from_slice(&crc32(&payload).to_be_bytes());

        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .mode(0o600)
            .open(&seg_path)
            .map_err(|e| io_err("open-segment", e))?;
        file.write_all(&frame)
            .map_err(|e| io_err("append-frame", e))?;
        if self.config.fsync_on_append {
            file.sync_all().map_err(|e| io_err("fsync-segment", e))?;
        }

        Ok(LogOffset {
            segment: target_index,
            offset: existing_len,
        })
    }

    /// Sorted list of segment files under the intent-log dir.
    fn segments(&self) -> Result<Vec<Segment>> {
        let dir = self.config.intent_log_dir();
        let mut out = Vec::new();
        let rd = match std::fs::read_dir(&dir) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
            Err(e) => return Err(io_err("read-log-dir", e)),
        };
        for entry in rd {
            let entry = entry.map_err(|e| io_err("read-log-entry", e))?;
            let path = entry.path();
            let name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n,
                None => continue,
            };
            let index = match name
                .strip_suffix(".seg")
                .and_then(|s| s.parse::<u64>().ok())
            {
                Some(i) => i,
                None => continue,
            };
            let len = entry
                .metadata()
                .map_err(|e| io_err("stat-segment", e))?
                .len();
            out.push(Segment { index, path, len });
        }
        out.sort_by_key(|s| s.index);
        Ok(out)
    }

    fn segment_path(&self, index: u64) -> PathBuf {
        self.config
            .intent_log_dir()
            .join(format!("{index:012}.seg"))
    }
}

// Reader half — only the applier (persistent) consumes the log.
#[cfg(feature = "persistent")]
enum FrameParse {
    /// A complete, CRC-valid frame.
    Complete(RawRecord),
    /// A short / implausible frame at EOF (a torn tail, or a live writer mid-append).
    Torn,
    /// A fully-present frame whose CRC does not match — committed-interior corruption.
    Corrupt,
}

#[cfg(feature = "persistent")]
impl SegmentedLog {
    /// Read the next record at or after `pos`, advancing across segment
    /// boundaries. Returns `Ok(None)` when caught up (or on a quarantined torn
    /// tail). A committed-interior CRC/length fault is a hard
    /// [`MemoryError::LogCorruption`].
    pub(crate) fn read_next(&self, pos: LogOffset) -> Result<Option<RawRecord>> {
        let segs = self.segments()?;
        if segs.is_empty() {
            return Ok(None);
        }
        let max_index = segs.last().map_or(0, |s| s.index);

        let mut cur = pos;
        loop {
            let seg = match segs.iter().find(|s| s.index == cur.segment) {
                Some(s) => s,
                None => {
                    // Requested segment is missing. If it is below the current
                    // range it was compacted after apply (fine — nothing to
                    // read there); if above the range there is nothing yet.
                    return Ok(None);
                }
            };
            if cur.offset >= seg.len {
                if cur.segment < max_index {
                    cur = LogOffset {
                        segment: cur.segment + 1,
                        offset: 0,
                    };
                    continue;
                }
                return Ok(None);
            }
            let is_last = cur.segment == max_index;
            return match self.parse_frame_at(&seg.path, cur, seg.len)? {
                FrameParse::Complete(rec) => Ok(Some(rec)),
                FrameParse::Corrupt => Err(MemoryError::LogCorruption {
                    segment: cur.segment,
                    offset: cur.offset,
                }),
                FrameParse::Torn => self.quarantine_torn_tail(&seg.path, cur, is_last),
            };
        }
    }

    /// Parse the single frame at `pos` within a segment of length `seg_len`,
    /// classifying it as complete, torn (short at EOF), or interior-corrupt.
    fn parse_frame_at(
        &self,
        seg_path: &std::path::Path,
        pos: LogOffset,
        seg_len: u64,
    ) -> Result<FrameParse> {
        let remaining = seg_len - pos.offset;
        if remaining < FRAME_OVERHEAD {
            return Ok(FrameParse::Torn);
        }
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .open(seg_path)
            .map_err(|e| io_err("open-segment-read", e))?;
        file.seek(SeekFrom::Start(pos.offset))
            .map_err(|e| io_err("seek-segment", e))?;

        let mut len_buf = [0u8; 4];
        file.read_exact(&mut len_buf)
            .map_err(|e| io_err("read-frame-len", e))?;
        let payload_len = u32::from_be_bytes(len_buf);
        if u64::from(payload_len) > u64::from(self.config.max_frame_bytes) {
            return Ok(FrameParse::Torn);
        }
        let need = 4 + u64::from(payload_len) + 4;
        if remaining < need {
            return Ok(FrameParse::Torn);
        }
        let mut payload = vec![0u8; payload_len as usize];
        file.read_exact(&mut payload)
            .map_err(|e| io_err("read-frame-payload", e))?;
        let mut crc_buf = [0u8; 4];
        file.read_exact(&mut crc_buf)
            .map_err(|e| io_err("read-frame-crc", e))?;
        let stored_crc = u32::from_be_bytes(crc_buf);
        if crc32(&payload) != stored_crc {
            return Ok(FrameParse::Corrupt);
        }
        Ok(FrameParse::Complete(RawRecord {
            payload,
            next: LogOffset {
                segment: pos.segment,
                offset: pos.offset + need,
            },
        }))
    }

    /// A short/implausible frame at EOF. This can be a genuine torn tail (a
    /// writer that died mid-append before its `fsync`/ack) **or** a writer that
    /// is *currently* mid-`write_all`. We must never truncate the latter, so we
    /// take the same `.append.lock` the writers hold, re-stat the segment, and
    /// re-parse: if a writer completed the frame in the meantime it is a normal
    /// record; if it is still short with no writer holding the lock it is a
    /// genuine torn tail and is truncated (it was never acked). A short frame in
    /// a non-last segment is interior corruption and is never truncated.
    fn quarantine_torn_tail(
        &self,
        seg_path: &std::path::Path,
        pos: LogOffset,
        is_last: bool,
    ) -> Result<Option<RawRecord>> {
        if !is_last {
            return Err(MemoryError::LogCorruption {
                segment: pos.segment,
                offset: pos.offset,
            });
        }
        // Serialize against live appends before any destructive action.
        let (_lock_file, _guard) = self.acquire_append_lock()?;
        let seg_len = std::fs::metadata(seg_path)
            .map(|m| m.len())
            .map_err(|e| io_err("stat-tail", e))?;
        match self.parse_frame_at(seg_path, pos, seg_len)? {
            // A writer completed the frame between our unlocked read and the
            // lock — it is a normal record, not a torn tail.
            FrameParse::Complete(rec) => Ok(Some(rec)),
            FrameParse::Corrupt => Err(MemoryError::LogCorruption {
                segment: pos.segment,
                offset: pos.offset,
            }),
            // Still short with the append lock held: no writer is mid-write, so
            // this is a genuine never-acked torn tail. Truncate it away.
            FrameParse::Torn => {
                let file = std::fs::OpenOptions::new()
                    .write(true)
                    .open(seg_path)
                    .map_err(|e| io_err("open-truncate-tail", e))?;
                file.set_len(pos.offset)
                    .map_err(|e| io_err("truncate-tail", e))?;
                file.sync_all().map_err(|e| io_err("fsync-truncate", e))?;
                Ok(None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intent_serializes_tagged_snake_case() {
        let intent = WriteIntent::RecordAccess {
            intent_id: Uuid::new_v4(),
            agent_name: "e".into(),
            node_id: "sem_1".into(),
            kind: AccessKind::Recall,
        };
        let v = serde_json::to_value(&intent).expect("serialize");
        assert_eq!(v["kind"], "record_access");
    }

    #[test]
    fn unknown_kind_fails_closed() {
        let bogus = serde_json::json!({ "kind": "boom", "intent_id": Uuid::new_v4() });
        assert!(serde_json::from_value::<WriteIntent>(bogus).is_err());
    }

    #[test]
    fn log_offset_orders_by_segment_then_offset() {
        let a = LogOffset {
            segment: 0,
            offset: 100,
        };
        let b = LogOffset {
            segment: 1,
            offset: 0,
        };
        assert!(b > a);
    }
}
