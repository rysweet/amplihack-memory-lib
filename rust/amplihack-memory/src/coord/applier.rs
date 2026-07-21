//! The single fenced applier (`PrefixConsistency` + `NoSplitBrain`).
//!
//! Exactly one [`Applier`], run by the long-lived daemon, drains the durable
//! shared log strictly **in order** and applies each [`WriteIntent`] to the
//! `lbug` store via the single-writer path
//! ([`CognitiveMemory::open_persistent`]). It holds the store lease and
//! **re-reads the live epoch and fences before every apply**: if another process
//! has acquired the lease, the apply is rejected with
//! [`MemoryError::EpochFenced`] and the applier fails closed — it never "wins" by
//! being alive. Progress is recorded exactly-once via a durable applied-index
//! (checkpoint → `fsync` → atomic-rename) plus idempotent replay keyed on
//! `intent_id`.

use std::collections::HashSet;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use uuid::Uuid;

use crate::coord::intent_log::SegmentedLog;
use crate::coord::lease::Epoch;
use crate::coord::{io_err, CoordConfig, Lease, LogOffset, WriteIntent};
use crate::errors::{MemoryError, Result};
use crate::CognitiveMemory;

/// The daemon-only fenced consumer of the durable log.
pub struct Applier {
    config: CoordConfig,
    memory: CognitiveMemory,
    /// Held for the applier's lifetime; dropping it releases the lease.
    _lease: Lease,
    epoch: Epoch,
    log: SegmentedLog,
    applied_index: LogOffset,
    seen: HashSet<Uuid>,
}

impl Applier {
    /// Acquire the store lease (bumping the epoch), open the single `lbug`
    /// writer, and resume from the durable applied-index.
    ///
    /// # Errors
    /// [`MemoryError::Storage`] if the lease/store/index cannot be opened.
    pub fn open(store_path: &Path, agent_name: &str, config: &CoordConfig) -> Result<Self> {
        let lease = Lease::acquire(config, agent_name)?;
        let epoch = lease.epoch();
        let memory = CognitiveMemory::open_persistent(store_path, agent_name)?;
        let log = SegmentedLog::open(config)?;
        let applied_index = load_applied_index(config)?;
        Ok(Self {
            config: config.clone(),
            memory,
            _lease: lease,
            epoch,
            log,
            applied_index,
            seen: HashSet::new(),
        })
    }

    /// Apply all currently-durable records at or after the applied-index, in
    /// order, exactly-once, fencing before each. Returns the number of records
    /// **effectively** applied (duplicates by `intent_id` are collapsed).
    ///
    /// # Errors
    /// [`MemoryError::EpochFenced`] if the lease epoch has moved on (fail
    /// closed); [`MemoryError::LogCorruption`] on a committed-interior bad
    /// frame; [`MemoryError::UnsupportedIntentVersion`] on an unknown record.
    pub fn drain(&mut self) -> Result<usize> {
        let mut applied = 0usize;
        let mut pos = self.applied_index;

        while let Some(rec) = self.log.read_next(pos)? {
            // Fence BEFORE applying: a stale epoch must never mutate the store.
            let live = Lease::current_epoch(&self.config)?;
            if live != self.epoch {
                return Err(MemoryError::EpochFenced {
                    expected: self.epoch,
                    found: live,
                });
            }
            let intent: WriteIntent = serde_json::from_slice(&rec.payload)
                .map_err(|_| MemoryError::UnsupportedIntentVersion)?;
            let id = intent.intent_id();
            // Exactly-once across restart: the in-memory `seen` set is a fast
            // path, but the DURABLE boundary is the in-store applied-intent
            // ledger — it survives a crash that rewound the applied-index, so a
            // replay never re-applies an effect. Effect then ledger-mark ride the
            // same checkpoint below.
            if !self.seen.contains(&id) && !self.memory.intent_applied(id) {
                apply_intent(&mut self.memory, &intent)?;
                self.memory.mark_intent_applied(id)?;
                applied += 1;
            }
            self.seen.insert(id);
            pos = rec.next;
        }

        if pos != self.applied_index {
            // Durably advance: checkpoint the store, then atomically move the
            // applied-index past everything applied. Order matters — the store
            // is persisted before its cursor advances, so a crash never leaves
            // the index ahead of the store.
            self.memory.checkpoint()?;
            persist_applied_index(&self.config, pos)?;
            self.applied_index = pos;
        }
        Ok(applied)
    }

    /// Drain as records arrive until `shutdown` returns `true`.
    ///
    /// # Errors
    /// Propagates any [`drain`](Self::drain) error (e.g. fail-closed
    /// [`MemoryError::EpochFenced`]).
    pub fn run(&mut self, shutdown: impl Fn() -> bool) -> Result<()> {
        while !shutdown() {
            self.drain()?;
            std::thread::sleep(Duration::from_millis(50));
        }
        Ok(())
    }

    /// The applied-index (durable cursor) as last committed.
    pub fn applied_index(&self) -> LogOffset {
        self.applied_index
    }
}

/// Apply one intent to the store via the single-writer mutator surface. Each
/// variant maps 1:1 to an existing [`CognitiveMemory`] method.
fn apply_intent(memory: &mut CognitiveMemory, intent: &WriteIntent) -> Result<()> {
    match intent {
        WriteIntent::StoreFact {
            concept,
            content,
            confidence,
            source_id,
            tags,
            metadata,
            ..
        } => {
            memory.store_fact(
                concept,
                content,
                *confidence,
                source_id,
                tags.as_deref(),
                metadata.as_ref(),
            )?;
        }
        WriteIntent::StoreProcedure {
            name,
            steps,
            prerequisites,
            ..
        } => {
            memory.store_procedure(name, steps, prerequisites.as_deref())?;
        }
        WriteIntent::StoreEpisode {
            content,
            source_label,
            temporal_index,
            metadata,
            ..
        } => {
            memory.store_episode(content, source_label, *temporal_index, metadata.as_ref())?;
        }
        WriteIntent::StoreProspective {
            description,
            trigger_condition,
            action_on_trigger,
            priority,
            ..
        } => {
            memory.store_prospective(
                description,
                trigger_condition,
                action_on_trigger,
                *priority,
            )?;
        }
        WriteIntent::LinkFactToEpisodes {
            fact_id,
            episode_ids,
            ..
        } => {
            memory.link_fact_to_episodes(fact_id, episode_ids)?;
        }
        WriteIntent::LinkSimilarFacts {
            fact_id_a,
            fact_id_b,
            similarity_score,
            ..
        } => {
            memory.link_similar_facts(fact_id_a, fact_id_b, *similarity_score)?;
        }
        WriteIntent::UpsertFact { input, options, .. } => {
            memory.upsert_fact(input.clone(), options)?;
        }
        WriteIntent::RecordAccess { node_id, kind, .. } => {
            memory.record_access(node_id, *kind)?;
        }
    }
    Ok(())
}

/// Load the durable applied-index, or the log origin if none has been written.
fn load_applied_index(config: &CoordConfig) -> Result<LogOffset> {
    match std::fs::read(config.applied_index_path()) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map_err(|e| MemoryError::Storage(format!("coord parse-applied-index: {e}"))),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(LogOffset::origin()),
        Err(e) => Err(io_err("read-applied-index", e)),
    }
}

/// Atomically advance the durable applied-index: write a temp file, `fsync` it,
/// rename it over the cursor, then `fsync` the directory.
fn persist_applied_index(config: &CoordConfig, pos: LogOffset) -> Result<()> {
    use std::os::unix::fs::OpenOptionsExt;
    let final_path = config.applied_index_path();
    let tmp_path = config.base_dir.join(".applied-index.tmp");
    let bytes = serde_json::to_vec(&pos)
        .map_err(|e| MemoryError::Storage(format!("coord serialize-applied-index: {e}")))?;
    {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .mode(0o600)
            .open(&tmp_path)
            .map_err(|e| io_err("open-index-tmp", e))?;
        f.write_all(&bytes)
            .map_err(|e| io_err("write-index-tmp", e))?;
        f.sync_all().map_err(|e| io_err("fsync-index-tmp", e))?;
    }
    std::fs::rename(&tmp_path, &final_path).map_err(|e| io_err("rename-index", e))?;
    if let Ok(dir) = std::fs::File::open(&config.base_dir) {
        let _ = dir.sync_all();
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Coordinator — the daemon's front door (applier + read server sharing one store)
// ---------------------------------------------------------------------------

/// The daemon coordinator: it owns the single `lbug` writer and the store lease,
/// and exposes the fenced applier loop plus (with `ipc`) the read server. Both
/// go through the one `open_persistent` handle, preserving single-writer
/// discipline. Wrap in an [`Arc`] to drive `run_applier`/`serve_reads` on
/// separate threads.
pub struct Coordinator {
    config: CoordConfig,
    memory: Arc<Mutex<CognitiveMemory>>,
    _lease: Lease,
    epoch: Epoch,
    log: SegmentedLog,
    applied_index: Mutex<LogOffset>,
    seen: Mutex<HashSet<Uuid>>,
}

impl Coordinator {
    /// Open the coordinator: acquire the lease, open the single `lbug` writer,
    /// and resume from the durable applied-index.
    ///
    /// # Errors
    /// [`MemoryError::Storage`] if the lease/store/index cannot be opened.
    pub fn open(store_path: &Path, agent_name: &str, config: CoordConfig) -> Result<Self> {
        let lease = Lease::acquire(&config, agent_name)?;
        let epoch = lease.epoch();
        let memory = CognitiveMemory::open_persistent(store_path, agent_name)?;
        let log = SegmentedLog::open(&config)?;
        let applied_index = load_applied_index(&config)?;
        Ok(Self {
            config,
            memory: Arc::new(Mutex::new(memory)),
            _lease: lease,
            epoch,
            log,
            applied_index: Mutex::new(applied_index),
            seen: Mutex::new(HashSet::new()),
        })
    }

    /// Run the fenced applier loop until `shutdown` returns `true`.
    ///
    /// # Errors
    /// Propagates fail-closed [`MemoryError::EpochFenced`] and any apply error.
    pub fn run_applier(&self, shutdown: impl Fn() -> bool) -> Result<()> {
        while !shutdown() {
            self.drain_once()?;
            std::thread::sleep(Duration::from_millis(50));
        }
        Ok(())
    }

    /// One fenced drain pass over the shared store.
    fn drain_once(&self) -> Result<()> {
        let mut pos = *self.applied_index.lock().expect("applied-index mutex");
        let start = pos;
        while let Some(rec) = self.log.read_next(pos)? {
            let live = Lease::current_epoch(&self.config)?;
            if live != self.epoch {
                return Err(MemoryError::EpochFenced {
                    expected: self.epoch,
                    found: live,
                });
            }
            let intent: WriteIntent = serde_json::from_slice(&rec.payload)
                .map_err(|_| MemoryError::UnsupportedIntentVersion)?;
            let id = intent.intent_id();
            let mut mem = self.memory.lock().expect("store mutex");
            let seen_new = self.seen.lock().expect("seen mutex").insert(id);
            // Durable exactly-once: consult the in-store ledger (survives a
            // restart with an empty `seen`) before applying, then mark it.
            if seen_new && !mem.intent_applied(id) {
                apply_intent(&mut mem, &intent)?;
                mem.mark_intent_applied(id)?;
            }
            pos = rec.next;
        }
        if pos != start {
            self.memory.lock().expect("store mutex").checkpoint()?;
            persist_applied_index(&self.config, pos)?;
            *self.applied_index.lock().expect("applied-index mutex") = pos;
        }
        Ok(())
    }

    /// Serve ranked-recall/ping reads over the coord socket until `shutdown`
    /// returns `true`. The read plane is read-only by contract.
    ///
    /// # Errors
    /// [`MemoryError::Storage`] on bind/serve failure.
    #[cfg(feature = "ipc")]
    pub fn serve_reads(&self, shutdown: impl Fn() -> bool) -> Result<()> {
        let server = crate::coord::ipc::IpcServer::bind(&self.config)?;
        server.serve_shared(&self.memory, shutdown)
    }
}
