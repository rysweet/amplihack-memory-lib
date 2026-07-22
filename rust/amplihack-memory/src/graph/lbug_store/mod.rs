//! `LbugGraphStore` -- LadybugDB-backed persistent implementation of [`GraphStore`].
//!
//! This is the persistent counterpart to [`InMemoryGraphStore`](crate::graph::InMemoryGraphStore).
//! It executes Cypher against an embedded LadybugDB instance via the published
//! [`lbug`] crate (the same engine Simard's private cognitive-memory module uses),
//! and faithfully implements the library's [`GraphStore`](crate::graph::protocol::GraphStore)
//! trait so it can back [`CognitiveMemory`](crate::cognitive_memory::CognitiveMemory)
//! without any change to the cognitive-memory logic.
//!
//! ## Design
//!
//! Modeled on the in-tree `KuzuGraphStore` (LadybugDB
//! is a fork of Kùzu and speaks the same Cypher dialect), but executes through
//! lbug's native Rust API instead of PyO3:
//!
//! * **Dynamic schema** — node/rel tables are created on first use; every
//!   user-supplied property maps to a `STRING` column. This mirrors the generic
//!   key/value node model the `GraphStore` trait promises and matches the
//!   in-memory backend's behavior so converters round-trip identically.
//! * **Reopen-safe** — on first access the existing catalog is introspected
//!   (`CALL show_tables` / `CALL table_info`) so data written in a previous
//!   process is visible after `close` + reopen.
//! * **Durability** — writes are serialized through a [`Mutex`](std::sync::Mutex), every mutating
//!   operation issues a per-write `fsync` barrier (data file + parent
//!   directory), and `close` issues a `CHECKPOINT` so a subsequent reopen sees
//!   all committed writes. Without the barrier a crash between two writes could
//!   lose an acknowledged write, since LadybugDB only flushes its WAL on
//!   `Database::drop`.
//! * **Bounded crash loss** — the store auto-checkpoints after every
//!   [`AUTO_CHECKPOINT_WRITES`](crate::graph::lbug_store::AUTO_CHECKPOINT_WRITES) mutating operations (and always on `close` /
//!   `Drop`), flushing the write-ahead log into the main database file. An
//!   unclean shutdown therefore loses at most the handful of writes accumulated
//!   since the last checkpoint, rather than every uncheckpointed record.
//! * **Corrupt-WAL recovery** — [`LbugGraphStore::open_with_recovery`] opens a
//!   store whose WAL was left partially written by an unclean shutdown. A strict
//!   open is attempted first; if it fails because the WAL replay cannot complete
//!   (the failure mode that previously made the store permanently unopenable),
//!   the corrupt WAL is quarantined to `<wal>.corrupt-<ts>`, a resilient open
//!   replays the good prefix, and a `CHECKPOINT` folds the recovered records into
//!   the main database file so a subsequent clean reopen needs no replay. If that
//!   checkpoint-after-recovery itself **fails** (#2550 — the recovered records
//!   would otherwise live only in the quarantined WAL and be reset to empty on a
//!   later open), the recovered graph is **salvaged into a fresh database**: it is
//!   dumped while the resilient handle is still open, the un-checkpointable
//!   original is quarantined, and the records are reloaded and checkpointed into a
//!   clean database so the pre-corruption prefix survives durably.
//! * **Catalog / main-DB corruption recovery** — if the engine cannot open the
//!   main database even with the WAL fully quarantined (a corrupt catalog / main
//!   file, e.g. left by a failed CHECKPOINT — the cause of the #95 crash loop),
//!   [`open_with_recovery`](LbugGraphStore::open_with_recovery) first attempts a
//!   **read-only salvage** of any still-readable records (#2550 — never reset a
//!   store that still holds recoverable records); it then quarantines the entire
//!   database to `<db_path>.corrupt-<ts>` (moved aside, never deleted) and opens a
//!   fresh database, reloading any salvaged records so the store self-heals
//!   instead of failing forever. The strict [`open`](LbugGraphStore::open) stays
//!   strict and errors.
//! * **Configurable limits** — the LadybugDB buffer-pool cap and maximum
//!   database size are read from `AMPLIHACK_MEMORY_BUFFER_POOL_BYTES` /
//!   `AMPLIHACK_MEMORY_MAX_DB_BYTES` (with larger, safer defaults than the old
//!   hardcoded 128 MiB / 1 GiB) so a busy host can raise the pool and avoid the
//!   checkpoint-time buffer-pool exhaustion that corrupted the catalog (#95).
//! * **Injection-safe** — table/column/edge identifiers are validated against
//!   `[A-Za-z_][A-Za-z0-9_]*`, all string values are escaped via `escape_cypher`,
//!   and `LIMIT` is a type-safe `usize`.

mod store_impl;

#[cfg(test)]
mod tests;

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use lbug::{Connection, Database, SystemConfig, Value};
use tracing::{debug, info, warn};

use crate::graph::protocol::GraphStore;
use crate::graph::types::{Direction, GraphEdge, GraphNode};
use crate::MemoryError;

/// Number of mutating operations after which the store auto-checkpoints,
/// folding the write-ahead log into the main database file. This bounds how
/// much acknowledged-but-uncheckpointed data an unclean shutdown can strand in
/// the WAL (and therefore put at risk if that WAL is later found corrupt).
pub const AUTO_CHECKPOINT_WRITES: u64 = 128;

/// Environment variable overriding the LadybugDB buffer-pool size cap, in bytes.
///
/// The pool is allocated *lazily* by lbug, so this is a ceiling — not an upfront
/// allocation. Raising it is therefore cheap and prevents the checkpoint-time
/// "buffer pool is full" exhaustion that previously corrupted the catalog (#95).
pub const ENV_BUFFER_POOL_BYTES: &str = "AMPLIHACK_MEMORY_BUFFER_POOL_BYTES";

/// Environment variable overriding the LadybugDB maximum database size, in bytes.
///
/// `max_db_size` is only an mmap address-space *reservation*, not eager
/// allocation, so a large value costs nothing until data actually grows.
pub const ENV_MAX_DB_BYTES: &str = "AMPLIHACK_MEMORY_MAX_DB_BYTES";

/// Default buffer-pool cap (1 GiB), raised from the previous 128 MiB that caused
/// auto-CHECKPOINT to exhaust the pool on a busy host. lbug allocates the pool
/// lazily, so this larger cap is safe.
pub(crate) const DEFAULT_BUFFER_POOL_BYTES: u64 = 1 << 30;

/// Floor for the buffer-pool cap (64 MiB): an absurdly small override is clamped
/// up to this so the engine can't be starved.
pub(crate) const MIN_BUFFER_POOL_BYTES: u64 = 64 * 1024 * 1024;

/// Default maximum database size (16 GiB), raised from the previous 1 GiB. This
/// is only an mmap reservation, so a large default does not allocate memory.
pub(crate) const DEFAULT_MAX_DB_BYTES: u64 = 16 << 30;

/// Floor for the maximum database size (1 GiB).
pub(crate) const MIN_MAX_DB_BYTES: u64 = 1 << 30;

/// How [`LbugGraphStore::open_with_recovery`] opened the store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalRecoveryOutcome {
    /// The write-ahead log replayed cleanly; no recovery was performed and no
    /// artifact was written.
    Clean,
    /// A corrupt WAL tail was quarantined; the good prefix was replayed and made
    /// durable in the main database file. Normally the prefix is folded in by a
    /// checkpoint; if that checkpoint-after-recovery fails (#2550) the recovered
    /// graph is salvaged into a fresh database instead, and this outcome is also
    /// reported when a corrupt catalog's still-readable records are salvaged via a
    /// read-only open. In every case the reported records survive a strict reopen.
    RecoveredPrefix,
    /// The WAL was unusable even in resilient mode; it was quarantined and the
    /// store was opened from the last good checkpoint only.
    CheckpointOnly,
    /// The main database / catalog itself was corrupt and unopenable even with
    /// the WAL fully out of the way (the failure mode that previously
    /// crash-looped consumers, #95), **and** no records could be salvaged from it
    /// via a read-only open (#2550). The corrupt database was quarantined to
    /// `<db_path>.corrupt-<ts>` (moved aside, never deleted) and a fresh, empty
    /// database was opened at `db_path` so the store self-heals instead of
    /// failing forever. `recovered_records` is `0`.
    RebuiltAfterCorruption,
    /// The store opened without engine error but a materially non-empty store
    /// read back **empty** — the #107 silent empty-read signature. The gate
    /// **writes nothing**: it does not checkpoint, does not upgrade the on-disk
    /// format, does not rebuild, and writes no `*.corrupt-*` artifact, so the
    /// store bytes are left exactly as opened (and remain readable by the prior
    /// engine line for rollback). The returned store is **sealed**:
    /// [`checkpoint`](LbugGraphStore::checkpoint) is a hard error and
    /// auto-checkpoint is disabled, so a caller cannot accidentally upgrade an
    /// apparently-empty store. Distinct from both [`Clean`](Self::Clean) and
    /// [`RebuiltAfterCorruption`](Self::RebuiltAfterCorruption).
    SuspectedDataLoss,
}

/// Structured report describing a recovery open. Surfaced via a `warn!` log and
/// returned from [`LbugGraphStore::open_with_recovery`] so callers (and tests)
/// can see how many records survived and where the corrupt WAL was moved.
#[derive(Debug, Clone)]
pub struct WalRecovery {
    /// How the store was opened.
    pub outcome: WalRecoveryOutcome,
    /// Number of records (graph nodes) present after recovery. Only computed on
    /// a recovery path; `0` for [`WalRecoveryOutcome::Clean`].
    pub recovered_records: usize,
    /// Where the corrupt artifact was quarantined, if recovery ran. For a WAL
    /// recovery this is the moved-aside WAL (`<wal>.corrupt-<ts>`); for
    /// [`WalRecoveryOutcome::RebuiltAfterCorruption`] it is the moved-aside
    /// database (`<db_path>.corrupt-<ts>`). The artifact is always moved aside —
    /// never deleted.
    pub quarantined_wal: Option<PathBuf>,
}

impl WalRecovery {
    fn clean() -> Self {
        Self {
            outcome: WalRecoveryOutcome::Clean,
            recovered_records: 0,
            quarantined_wal: None,
        }
    }

    /// `true` if a corrupt WAL was detected and recovery (of any kind) ran.
    pub fn recovered(&self) -> bool {
        self.outcome != WalRecoveryOutcome::Clean
    }
}

/// LadybugDB-backed persistent [`GraphStore`].
pub struct LbugGraphStore {
    pub(crate) store_id: String,
    pub(crate) db_path: PathBuf,
    pub(crate) db: Database,
    /// Serializes writes and schema-cache mutations across threads.
    pub(crate) lock: Mutex<()>,
    /// Set once the on-disk catalog has been introspected into the caches below.
    pub(crate) schema_loaded: Cell<bool>,
    pub(crate) known_node_tables: RefCell<HashSet<String>>,
    pub(crate) node_table_columns: RefCell<HashMap<String, HashSet<String>>>,
    /// (rel_name, from_table, to_table) tuples known to exist.
    pub(crate) known_rel_tables: RefCell<HashSet<(String, String, String)>>,
    /// node_id -> node table, to resolve a node's label without scanning.
    pub(crate) id_table_cache: RefCell<HashMap<String, String>>,
    /// Mutating operations applied since the last checkpoint.
    pub(crate) writes_since_checkpoint: Cell<u64>,
    /// Auto-checkpoint after this many writes (`0` disables count-based
    /// checkpointing; used by tests that need data to remain in the WAL).
    pub(crate) checkpoint_interval: Cell<u64>,
    /// The most recent checkpoint error (e.g. buffer-pool exhaustion), or `None`
    /// after a successful checkpoint. Surfaced via [`last_checkpoint_error`] so
    /// consumers can report store health.
    ///
    /// [`last_checkpoint_error`]: LbugGraphStore::last_checkpoint_error
    pub(crate) last_checkpoint_error: RefCell<Option<String>>,
    /// Effective buffer-pool cap (bytes) this store was opened with.
    pub(crate) buffer_pool_bytes: u64,
    /// Effective max database size (bytes) this store was opened with.
    pub(crate) max_db_bytes: u64,
    /// #107 empty-read safety gate: set when a store with a material on-disk
    /// footprint read back empty. A **sealed** store refuses every checkpoint
    /// (`checkpoint`, the trait method, `do_checkpoint`, auto-checkpoint, and the
    /// drop checkpoint) so an apparently-empty store can never be upgraded to v41
    /// — the on-disk bytes are left intact for a prior-engine rollback.
    pub(crate) sealed: Cell<bool>,
}

// SAFETY: lbug::Database is internally synchronized (it declares Send + Sync).
// All access to the interior-mutable caches is serialized through `lock`, and
// the store is only ever used behind `&mut`/`&` from a single `CognitiveMemory`.
unsafe impl Send for LbugGraphStore {}

impl LbugGraphStore {
    /// Open (or create) a LadybugDB database at `db_path`.
    ///
    /// This is the strict open: if the on-disk write-ahead log cannot be fully
    /// replayed (e.g. it was left partially written by an unclean shutdown) this
    /// returns an error. Use [`open_with_recovery`](Self::open_with_recovery)
    /// (or [`CognitiveMemory::open_persistent`](crate::cognitive_memory::CognitiveMemory::open_persistent),
    /// which wraps it) to tolerate a corrupt WAL.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Storage`] if the parent directory cannot be created
    /// or the database cannot be opened.
    pub fn open(db_path: &Path, store_id: Option<&str>) -> crate::Result<Self> {
        Self::prepare_parent(db_path)?;
        let db = open_database(db_path, true).map_err(|e| {
            MemoryError::Storage(format!(
                "failed to open LadybugDB at {}: {e}",
                db_path.display()
            ))
        })?;
        Ok(Self::from_parts(db, db_path, store_id))
    }

    /// Open (or create) a LadybugDB database at `db_path`, recovering from a
    /// corrupt / partially-written WAL instead of failing.
    ///
    /// A strict [`open`](Self::open) is attempted first. If it succeeds the store
    /// is returned with [`WalRecoveryOutcome::Clean`] and no artifact is written.
    /// If the strict open fails *and* a WAL file is present (the signature of an
    /// unclean shutdown), the WAL — which can no longer be replayed — is
    /// quarantined to `<wal>.corrupt-<unix_ts>` (moved aside, never deleted), a
    /// resilient open replays the recoverable prefix, and a `CHECKPOINT` folds
    /// the recovered records into the main database file so the next clean reopen
    /// needs no replay. The returned [`WalRecovery`] reports the outcome and how
    /// many records survived; a `warn!` is also emitted.
    ///
    /// If the main database itself is corrupt — it will not open even with the
    /// WAL fully quarantined, or no WAL is present at all (a corrupt catalog /
    /// main file, e.g. left by a failed CHECKPOINT, the cause of the #95 crash
    /// loop) — the entire database is quarantined to `<db_path>.corrupt-<unix_ts>`
    /// (moved aside, never deleted) and a fresh, empty database is opened so the
    /// store self-heals instead of failing forever
    /// ([`WalRecoveryOutcome::RebuiltAfterCorruption`], `recovered_records = 0`).
    /// This rebuild-on-corruption behaviour is intentional for this resilient
    /// entry point; the strict [`open`](Self::open) never rebuilds and still
    /// errors.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Storage`] only if even a fresh database cannot be
    /// opened after quarantining the corrupt one (e.g. the parent directory is
    /// unwritable).
    pub fn open_with_recovery(
        db_path: &Path,
        store_id: Option<&str>,
    ) -> crate::Result<(Self, WalRecovery)> {
        Self::prepare_parent(db_path)?;

        // #107 empty-read safety gate — a READ-ONLY peek run *before* any
        // read-write open. A read-only open cannot checkpoint, upgrade v40→v41,
        // or rewrite the file (a read-write open mutates the file on drop even
        // with no writes), so if the peek finds a populated store reading back
        // empty we seal and return it without ever touching the bytes — the v40
        // store stays readable by a 0.15.x binary for rollback.
        if let Some(sealed) = Self::peek_empty_read_gate(db_path, store_id) {
            return Ok(sealed);
        }

        // 1. Fast path: strict open. A corrupt WAL surfaces as Err (lbug/cxx
        //    converts the C++ replay assertion into a Rust error rather than
        //    aborting); catch_unwind additionally contains any panic.
        match try_open_database(db_path, true) {
            Ok(db) => {
                let store = Self::from_parts(db, db_path, store_id);
                Ok((store, WalRecovery::clean()))
            }
            Err(strict_err) => {
                let wal = wal_path_for(db_path);
                if !wal.exists() {
                    // No WAL: the strict open did not fail on WAL replay. On the
                    // resilient entry point a persistent main-DB open failure is
                    // treated as catalog / main-file corruption and self-healed
                    // by quarantining the corrupt database and rebuilding a fresh
                    // one, rather than erroring (which previously crash-looped the
                    // consumer forever, #95).
                    warn!(
                        db_path = %db_path.display(),
                        error = %strict_err,
                        "lbug_store: main database failed to open and no WAL is present; \
                         treating as catalog corruption and rebuilding from empty"
                    );
                    return Self::rebuild_after_corruption(db_path, store_id, None);
                }
                warn!(
                    db_path = %db_path.display(),
                    error = %strict_err,
                    "lbug_store: WAL replay failed on open; attempting recovery"
                );
                Self::recover(db_path, store_id, &wal)
            }
        }
    }

    /// Recovery slow path: quarantine the corrupt WAL, then open resiliently and
    /// checkpoint, falling back to a checkpoint-only open if even that fails.
    fn recover(
        db_path: &Path,
        store_id: Option<&str>,
        wal: &Path,
    ) -> crate::Result<(Self, WalRecovery)> {
        // Preserve the incident artifact *before* a resilient open (which will
        // consume/truncate the WAL) gets a chance to mutate it.
        let quarantine = quarantine_path(wal);
        let copied = copy_wal_aside(db_path, wal, &quarantine);

        // 2. Resilient open: replay the good prefix, ignore the unreplayable tail.
        match try_open_database(db_path, false) {
            Ok(db) => {
                let store = Self::from_parts(db, db_path, store_id);
                let recovered = store.count_all_nodes();
                // Fold the recovered prefix into the main DB so a later clean
                // reopen needs no replay. A *failed* checkpoint here is the #2550
                // data-loss trap: the recovered prefix would then live only in the
                // quarantined/consumed WAL, and a later open would read the
                // pre-recovery (near-empty) main file and could re-quarantine and
                // reset the store to empty — permanently dropping the recovered
                // memories. So on checkpoint failure we do NOT return a store whose
                // records are not durably persisted: we salvage the recovered graph
                // into a fresh, clean database where a checkpoint succeeds.
                match store.recovery_checkpoint() {
                    Ok(()) => {
                        let report = WalRecovery {
                            outcome: WalRecoveryOutcome::RecoveredPrefix,
                            recovered_records: recovered,
                            quarantined_wal: copied.clone(),
                        };
                        warn!(
                            db_path = %db_path.display(),
                            recovered_records = recovered,
                            quarantined_wal = ?copied,
                            "lbug_store: recovered from corrupt WAL (good prefix replayed + checkpointed)"
                        );
                        Ok((store, report))
                    }
                    Err(checkpoint_err) => {
                        warn!(
                            db_path = %db_path.display(),
                            error = %checkpoint_err,
                            recovered_records = recovered,
                            "lbug_store: checkpoint after WAL recovery failed; salvaging the \
                             recovered graph into a fresh database so the pre-corruption prefix \
                             survives durably instead of being reset to empty on a later open"
                        );
                        // Capture the recovered graph while this resilient handle is
                        // still open — it is the only live copy until we persist it.
                        // Dump directly (rather than trusting the possibly-lossy
                        // `count_all_nodes()` gate) so a transient count failure can
                        // never misclassify a populated store as "nothing to salvage".
                        // An empty dump salvages to a fresh empty DB (records = 0),
                        // which also durably clears the corrupt WAL.
                        let dump = store.dump_graph();
                        drop(store);
                        Self::salvage_rebuild(db_path, store_id, dump, copied)
                    }
                }
            }
            Err(resilient_err) => {
                // 3. Hard fallback: even resilient replay failed. Move the WAL
                //    (and any sidecars) fully aside so the engine opens from the
                //    last good checkpoint with no WAL to replay.
                warn!(
                    db_path = %db_path.display(),
                    error = %resilient_err,
                    "lbug_store: resilient WAL replay failed; opening from last checkpoint only"
                );
                let moved = move_wal_aside(db_path, wal, &quarantine, copied.is_some());
                match try_open_database(db_path, true) {
                    Ok(db) => {
                        let store = Self::from_parts(db, db_path, store_id);
                        let recovered = store.count_all_nodes();
                        let report = WalRecovery {
                            outcome: WalRecoveryOutcome::CheckpointOnly,
                            recovered_records: recovered,
                            quarantined_wal: moved.or(copied),
                        };
                        warn!(
                            db_path = %db_path.display(),
                            recovered_records = recovered,
                            "lbug_store: opened from last checkpoint after unrecoverable WAL"
                        );
                        Ok((store, report))
                    }
                    Err(checkpoint_err) => {
                        // 4. The main database won't open even with the WAL gone:
                        //    the catalog / main file itself is corrupt. Quarantine
                        //    the whole database and rebuild fresh so the store
                        //    self-heals instead of crash-looping (#95).
                        warn!(
                            db_path = %db_path.display(),
                            error = %checkpoint_err,
                            "lbug_store: main database still unopenable after quarantining the \
                             WAL; treating as catalog corruption and rebuilding from empty"
                        );
                        Self::rebuild_after_corruption(db_path, store_id, moved.or(copied))
                    }
                }
            }
        }
    }

    /// Catalog / main-database corruption recovery.
    ///
    /// Reached when the engine cannot open the main database even with the WAL
    /// fully out of the way — the on-disk catalog / main file is corrupt beyond
    /// WAL replay (e.g. a failed CHECKPOINT left it in the state that surfaced as
    /// "table 0 doesn't exist in catalog", #95). The entire corrupt database is
    /// quarantined to `<db_path>.corrupt-<unix_ts>` (moved aside, never deleted;
    /// preserved via copy if a rename is risky) and a fresh, empty database is
    /// opened at `db_path` so the consumer keeps running instead of
    /// crash-looping. `prior_quarantine` is any WAL quarantine recorded earlier
    /// in the same recovery, used only as a fallback if the database itself was
    /// absent.
    fn rebuild_after_corruption(
        db_path: &Path,
        store_id: Option<&str>,
        prior_quarantine: Option<PathBuf>,
    ) -> crate::Result<(Self, WalRecovery)> {
        // #2550: never reset a store that still holds recoverable records. Before
        // quarantining and rebuilding empty, attempt a READ-ONLY salvage of the
        // corrupt database (a read-only open cannot mutate or further corrupt it).
        // If anything is still readable, route it through the verified
        // salvage-rebuild path (quarantine the corrupt DB → reload → checkpoint →
        // strict-reopen-verify) so a store that still holds records is never reset
        // to empty and its durability is proven, not assumed.
        if let Some(dump) = Self::read_only_dump(db_path) {
            return Self::salvage_rebuild(db_path, store_id, dump, prior_quarantine);
        }

        // Nothing salvageable: quarantine the corrupt database and open a fresh,
        // empty one so the store self-heals instead of crash-looping (#95).
        let quarantine = quarantine_path(db_path);
        let moved = quarantine_db_artifacts(db_path, &quarantine)?;
        let db = open_database(db_path, true).map_err(|e| {
            MemoryError::Storage(format!(
                "failed to open a fresh LadybugDB at {} after quarantining a corrupt catalog: {e}",
                db_path.display()
            ))
        })?;
        let store = Self::from_parts(db, db_path, store_id);
        let quarantined = moved.or(prior_quarantine);
        warn!(
            db_path = %db_path.display(),
            quarantined_db = ?quarantined,
            "lbug_store: main database/catalog was corrupt and unopenable; quarantined it \
             and rebuilt a fresh empty database so the store self-heals instead of \
             crash-looping"
        );
        let report = WalRecovery {
            outcome: WalRecoveryOutcome::RebuiltAfterCorruption,
            recovered_records: 0,
            quarantined_wal: quarantined,
        };
        Ok((store, report))
    }

    /// Checkpoint issued immediately after a resilient WAL replay, folding the
    /// recovered prefix into the main database file.
    ///
    /// Extracted from [`recover`](Self::recover) so a test can deterministically
    /// force *this* checkpoint to fail (reproducing the #2550
    /// checkpoint-after-recovery failure) without perturbing any other checkpoint
    /// — including the fresh-database checkpoint the salvage path relies on.
    fn recovery_checkpoint(&self) -> crate::Result<()> {
        #[cfg(test)]
        if test_hooks::take_forced_recovery_checkpoint_failure() {
            return Err(MemoryError::Storage(
                "injected checkpoint-after-recovery failure (test): cannot open <db>.wal.checkpoint"
                    .to_string(),
            ));
        }
        self.do_checkpoint()
    }

    /// Read every live node and edge into memory so a recovered graph can be
    /// salvaged into a fresh database (see [`salvage_rebuild`](Self::salvage_rebuild)).
    ///
    /// Tombstoned (soft-deleted) rows are excluded — the dump contains exactly the
    /// live records a normal read would return. Edges are enumerated by walking
    /// each node's *outgoing* relationships, which reuses the tombstone-aware
    /// neighbour reader and visits every edge exactly once (each edge has a single
    /// source node).
    pub(crate) fn dump_graph(&self) -> GraphDump {
        self.ensure_schema_loaded();
        let tables: Vec<String> = self.known_node_tables.borrow().iter().cloned().collect();
        let mut nodes: Vec<GraphNode> = Vec::new();
        for table in &tables {
            nodes.extend(self.query_nodes(table, None, usize::MAX));
        }
        let mut edges: Vec<GraphEdge> = Vec::new();
        for node in &nodes {
            for (edge, _neighbor) in
                self.query_neighbors(&node.node_id, None, Direction::Outgoing, usize::MAX)
            {
                edges.push(edge);
            }
        }
        GraphDump { nodes, edges }
    }

    /// Reload a [`GraphDump`] into this (fresh) store, recreating every node and
    /// then every edge. Returns the number of nodes reloaded. Reserved columns are
    /// stripped ([`salvage_props`]) so `add_node`/`add_edge` re-derive them (the id
    /// is passed explicitly and `graph_origin` becomes this store's id). A row that
    /// fails to reload is logged rather than aborting the whole salvage.
    /// Reload a [`GraphDump`] into this (fresh) store, recreating every node and
    /// then every edge. Returns `(nodes_reloaded, edges_reloaded)`. Reserved
    /// columns are stripped ([`salvage_props`]) so `add_node`/`add_edge` re-derive
    /// them (the id is passed explicitly and `graph_origin` becomes this store's
    /// id). A row that fails to reload is logged rather than aborting the whole
    /// salvage — the caller compares the returned counts against the dump to
    /// surface any partial reload.
    pub(crate) fn reload_graph(&mut self, dump: &GraphDump) -> (usize, usize) {
        let mut nodes_reloaded = 0usize;
        for node in &dump.nodes {
            match self.add_node(
                &node.node_type,
                salvage_props(&node.properties),
                Some(&node.node_id),
            ) {
                Ok(_) => nodes_reloaded += 1,
                Err(e) => warn!(
                    node_id = %node.node_id,
                    node_type = %node.node_type,
                    error = %e,
                    "lbug_store: failed to reload a salvaged node"
                ),
            }
        }
        let mut edges_reloaded = 0usize;
        for edge in &dump.edges {
            match self.add_edge(
                &edge.source_id,
                &edge.target_id,
                &edge.edge_type,
                Some(salvage_props(&edge.properties)),
            ) {
                Ok(_) => edges_reloaded += 1,
                Err(e) => warn!(
                    source = %edge.source_id,
                    target = %edge.target_id,
                    edge_type = %edge.edge_type,
                    error = %e,
                    "lbug_store: failed to reload a salvaged edge"
                ),
            }
        }
        (nodes_reloaded, edges_reloaded)
    }

    /// Salvage a recovered graph into a fresh, clean database.
    ///
    /// Reached when a resilient WAL replay recovered records but the in-place
    /// `CHECKPOINT` that would fold them into the main database file failed
    /// (#2550: a failed checkpoint-after-recovery left the recovered prefix only
    /// in the WAL, and a later open then reset the store to empty). The recovered
    /// graph — captured into `dump` while the resilient store was still open — is
    /// reloaded into a brand-new database at `db_path`, whose checkpoint succeeds
    /// because it starts clean. The un-checkpointable original is quarantined to
    /// `<db_path>.corrupt-<ts>` (moved aside, never deleted).
    ///
    /// Durability is **verified, not assumed**: the freshly-loaded database is
    /// dropped (flushing its clean WAL) and then **strict-reopened**, and the
    /// returned store is that strict-reopened handle. So the returned
    /// `recovered_records` is a count read back through a clean strict open — the
    /// same open a later process performs — which is exactly the durability
    /// guarantee. A short reload (fewer records than dumped) is surfaced loudly
    /// rather than silently reported as a clean recovery.
    fn salvage_rebuild(
        db_path: &Path,
        store_id: Option<&str>,
        dump: GraphDump,
        prior_quarantine: Option<PathBuf>,
    ) -> crate::Result<(Self, WalRecovery)> {
        let dumped_nodes = dump.nodes.len();
        let dumped_edges = dump.edges.len();
        let quarantine = quarantine_path(db_path);
        let moved = quarantine_db_artifacts(db_path, &quarantine)?;

        // Load the salvaged graph into a fresh database, then drop it so its clean
        // WAL is flushed before we re-open to verify durability.
        let (nodes_reloaded, edges_reloaded) = {
            let db = open_database(db_path, true).map_err(|e| {
                MemoryError::Storage(format!(
                    "failed to open a fresh LadybugDB at {} while salvaging a recovered graph: {e}",
                    db_path.display()
                ))
            })?;
            let mut store = Self::from_parts(db, db_path, store_id);
            let counts = store.reload_graph(&dump);
            // A fresh database has no corrupt WAL, so this checkpoint folds the
            // reload into the main file. Even if it fails, the reload is in the
            // fresh, clean WAL (flushed on the drop below) and replays cleanly on
            // the strict reopen — which is what actually verifies durability.
            if let Err(e) = store.checkpoint() {
                warn!(
                    db_path = %db_path.display(),
                    error = %e,
                    "lbug_store: checkpoint of the salvaged database failed; the reload is in the \
                     fresh clean WAL and will be verified by the strict reopen below"
                );
            }
            counts
        };

        // Verify durability the same way a later process would see it: a strict
        // reopen (replays the fresh clean WAL, or none if the checkpoint succeeded).
        let db = open_database(db_path, true).map_err(|e| {
            MemoryError::Storage(format!(
                "salvaged database at {} could not be strict-reopened to verify durability: {e}",
                db_path.display()
            ))
        })?;
        let store = Self::from_parts(db, db_path, store_id);
        let survived = store.count_all_nodes();
        let quarantined = moved.or(prior_quarantine);
        // `recovered_records` is the honest, strict-reopen-verified node count.
        // A partial reload (nodes or edges) is preserved for availability but
        // surfaced at error level with the expected vs surviving counts, never
        // silently reported as a complete recovery.
        if survived < dumped_nodes || edges_reloaded < dumped_edges {
            tracing::error!(
                db_path = %db_path.display(),
                dumped_nodes,
                dumped_edges,
                nodes_reloaded,
                edges_reloaded,
                survived,
                quarantined = ?quarantined,
                "lbug_store: salvage recovered fewer records than were dumped (partial recovery); \
                 the durable survivors are kept and the corrupt original is preserved at the \
                 quarantine path for offline recovery of the remainder"
            );
        } else {
            warn!(
                db_path = %db_path.display(),
                recovered_records = survived,
                dumped_nodes,
                dumped_edges,
                edges_reloaded,
                quarantined = ?quarantined,
                "lbug_store: salvaged the recovered graph into a fresh database after a failed \
                 checkpoint-after-recovery; the pre-corruption prefix is durable (verified by \
                 strict reopen)"
            );
        }
        let report = WalRecovery {
            outcome: WalRecoveryOutcome::RecoveredPrefix,
            recovered_records: survived,
            quarantined_wal: quarantined,
        };
        Ok((store, report))
    }

    /// Best-effort read-only salvage of a possibly-corrupt database: open it
    /// **read-only** (which can never mutate or further corrupt it) and dump every
    /// live node/edge. Returns `None` if even a read-only open fails or the store
    /// is empty, in which case the caller rebuilds from empty.
    fn read_only_dump(db_path: &Path) -> Option<GraphDump> {
        let db = try_open_database_read_only(db_path).ok()?;
        let store = Self::from_parts(db, db_path, None);
        // Belt-and-suspenders: seal so the drop below never attempts a `CHECKPOINT`
        // against the store we are only inspecting (the engine is opened read-only,
        // but sealing matches the #107 read-only-peek pattern and guarantees no
        // write is even attempted).
        store.seal();
        let dump = store.dump_graph();
        drop(store);
        if dump.nodes.is_empty() {
            None
        } else {
            Some(dump)
        }
    }

    /// Create the parent directory of `db_path` if needed.
    fn prepare_parent(db_path: &Path) -> crate::Result<()> {
        if let Some(parent) = db_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    MemoryError::Storage(format!(
                        "failed to create parent directory {}: {e}",
                        parent.display()
                    ))
                })?;
            }
        }
        Ok(())
    }

    /// Assemble a store around an already-opened [`Database`].
    fn from_parts(db: Database, db_path: &Path, store_id: Option<&str>) -> Self {
        let id = store_id
            .map(String::from)
            .unwrap_or_else(|| format!("lbug-{}", &uuid::Uuid::new_v4().to_string()[..8]));
        let buffer_env = std::env::var(ENV_BUFFER_POOL_BYTES).ok();
        let max_db_env = std::env::var(ENV_MAX_DB_BYTES).ok();
        let (buffer_pool_bytes, max_db_bytes) =
            effective_limits(buffer_env.as_deref(), max_db_env.as_deref());
        Self {
            store_id: id,
            db_path: db_path.to_path_buf(),
            db,
            lock: Mutex::new(()),
            schema_loaded: Cell::new(false),
            known_node_tables: RefCell::new(HashSet::new()),
            node_table_columns: RefCell::new(HashMap::new()),
            known_rel_tables: RefCell::new(HashSet::new()),
            id_table_cache: RefCell::new(HashMap::new()),
            writes_since_checkpoint: Cell::new(0),
            checkpoint_interval: Cell::new(AUTO_CHECKPOINT_WRITES),
            last_checkpoint_error: RefCell::new(None),
            buffer_pool_bytes,
            max_db_bytes,
            sealed: Cell::new(false),
        }
    }

    /// Effective buffer-pool cap (bytes) this store was opened with. Reflects
    /// any [`ENV_BUFFER_POOL_BYTES`] override and the clamped default.
    pub fn buffer_pool_bytes(&self) -> u64 {
        self.buffer_pool_bytes
    }

    /// Effective maximum database size (bytes) this store was opened with.
    /// Reflects any [`ENV_MAX_DB_BYTES`] override and the clamped default.
    pub fn max_db_bytes(&self) -> u64 {
        self.max_db_bytes
    }

    /// The most recent checkpoint error, or `None` if the last checkpoint
    /// succeeded. A failed auto-checkpoint (commonly buffer-pool exhaustion) is
    /// recorded here and cleared on the next successful checkpoint, so consumers
    /// can surface store health without parsing logs.
    pub fn last_checkpoint_error(&self) -> Option<String> {
        self.last_checkpoint_error.borrow().clone()
    }

    /// The on-disk database path.
    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    /// Override the auto-checkpoint write interval (`0` disables count-based
    /// checkpointing). Intended for in-crate tests that need writes to remain in
    /// the WAL so an unclean shutdown can be simulated.
    #[cfg(test)]
    pub(crate) fn set_checkpoint_interval(&self, writes: u64) {
        self.checkpoint_interval.set(writes);
    }

    /// Mutating operations applied since the last checkpoint (test introspection).
    #[cfg(test)]
    pub(crate) fn pending_writes(&self) -> u64 {
        self.writes_since_checkpoint.get()
    }

    /// Force a checkpoint: flush the write-ahead log into the main database file
    /// so a subsequent clean reopen needs no WAL replay.
    ///
    /// Acquires the write lock; safe to call concurrently with reads/writes.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Storage`] if the `CHECKPOINT` statement or the
    /// durability barrier fails.
    pub fn checkpoint(&self) -> crate::Result<()> {
        let _guard = self.acquire_lock();
        self.do_checkpoint()
    }

    /// Issue `CHECKPOINT` + durability barrier and reset the write counter.
    /// Caller is responsible for holding [`acquire_lock`](Self::acquire_lock)
    /// when used outside single-threaded recovery.
    ///
    /// On a failed `CHECKPOINT` the error is recorded in
    /// [`last_checkpoint_error`](Self::last_checkpoint_error) (store-health
    /// signal); a subsequent successful checkpoint clears it.
    pub(crate) fn do_checkpoint(&self) -> crate::Result<()> {
        // #107: a sealed store (the gate tripped on a suspected empty read) must
        // never checkpoint — that is exactly the v40→v41 upgrade that would make
        // an apparently-empty store permanently unreadable by a 0.15.x binary.
        if self.sealed.get() {
            return Err(MemoryError::Storage(
                "store sealed by the #107 empty-read gate; refusing to checkpoint \
                 (a checkpoint would upgrade an apparently-empty store to v41 and \
                 make the suspected data loss permanent)"
                    .to_string(),
            ));
        }
        if let Err(e) = self.execute("CHECKPOINT") {
            *self.last_checkpoint_error.borrow_mut() = Some(e.to_string());
            return Err(e);
        }
        self.writes_since_checkpoint.set(0);
        self.post_write_barrier()?;
        *self.last_checkpoint_error.borrow_mut() = None;
        Ok(())
    }

    /// Record a mutating operation and checkpoint once the configured interval is
    /// reached. Best-effort: a failed checkpoint is logged, not propagated (the
    /// write itself already succeeded and was fsync'd). Caller must hold the
    /// write lock.
    pub(crate) fn note_write_and_maybe_checkpoint(&self) {
        let interval = self.checkpoint_interval.get();
        if interval == 0 {
            return;
        }
        let next = self.writes_since_checkpoint.get().saturating_add(1);
        if next >= interval {
            if let Err(e) = self.do_checkpoint() {
                warn!(
                    error = %e,
                    "lbug_store: auto-checkpoint failed — this can indicate buffer-pool \
                     exhaustion (raise the cap via {}); recorded as last_checkpoint_error, \
                     will retry on the next write",
                    ENV_BUFFER_POOL_BYTES
                );
                // do_checkpoint resets the counter only on success; reset here so
                // we retry on the next write rather than every write.
                self.writes_since_checkpoint.set(0);
            }
        } else {
            self.writes_since_checkpoint.set(next);
        }
    }

    /// Total number of graph nodes across every node table. Used to report how
    /// many records survived a recovery open.
    pub(crate) fn count_all_nodes(&self) -> usize {
        let mut total = 0usize;
        let rows = match self.query_rows("CALL show_tables() RETURN *") {
            Ok(r) => r,
            Err(_) => return 0,
        };
        for row in rows {
            let (name, ttype) = match table_row_name_and_type(&row) {
                Some(v) => v,
                None => continue,
            };
            if !ttype.to_ascii_uppercase().contains("NODE") || !is_valid_identifier(&name) {
                continue;
            }
            if let Ok(cnt) = self.query_rows(&format!("MATCH (n:{name}) RETURN count(n)")) {
                if let Some(v) = cnt.first().and_then(|r| r.first()) {
                    total += value_as_usize(v);
                }
            }
        }
        total
    }

    /// #107 empty-read safety gate — a **read-only peek** run before any
    /// read-write open (see [`open_with_recovery`](Self::open_with_recovery)).
    ///
    /// Decides, *without trusting the suspect read as its own baseline*, whether
    /// a store is the silent empty-read: it compares the freshly-read node count
    /// against an **independent on-disk signal**, the committed-footprint probe
    /// ([`committed_footprint`]). The decision:
    ///
    /// | Committed footprint | Node count | Result |
    /// | --- | --- | --- |
    /// | `Empty` | anything | `None` — fresh/empty store; proceed to a normal open. |
    /// | `NonEmpty` | `> 0` | `None` — populated store that read populated; proceed. |
    /// | `NonEmpty` | `0` (or unreadable) | `Some(sealed, SuspectedDataLoss)` — trip. |
    ///
    /// The peek opens **read-only**, so it can never checkpoint, upgrade v40→v41,
    /// rebuild, or rewrite the file. On a trip it returns a **sealed** store
    /// (checkpoint refused, auto-checkpoint off, drop checkpoint suppressed) and
    /// `WalRecoveryOutcome::SuspectedDataLoss`; **nothing is written** and no
    /// `*.corrupt-*` artifact is created, so the on-disk (v40) bytes stay exactly
    /// as they are for a prior-engine rollback. A read-only open that fails (e.g.
    /// a pending WAL needs replay) returns `None` so the normal recovery open
    /// handles it.
    fn peek_empty_read_gate(db_path: &Path, store_id: Option<&str>) -> Option<(Self, WalRecovery)> {
        let bytes = match committed_footprint(db_path) {
            CommittedFootprint::Empty => return None,
            CommittedFootprint::NonEmpty { bytes } => bytes,
        };

        // Open read-only so this peek cannot mutate a single byte of the store.
        let db = try_open_database_read_only(db_path).ok()?;
        let store = Self::from_parts(db, db_path, store_id);
        // The peek handle must never write, whatever happens to it.
        store.seal();

        // `count_all_nodes` uses unfiltered `count(n)` (it never names `_deleted`)
        // so it stays trustworthy even when the labeled read path is broken; a
        // `0` here over a material footprint is the catastrophic catalog-empty
        // signature. The test seam forces this observation deterministically.
        let read_count = store.count_all_nodes();
        let observed_empty = read_count == 0 || force_empty_read_for_test();
        if !observed_empty {
            // Healthy: drop the read-only peek (sealed, so its drop writes
            // nothing) and let the caller do a normal read-write open.
            return None;
        }

        warn!(
            db_path = %db_path.display(),
            footprint_bytes = bytes,
            read_count,
            "lbug_store: #107 empty-read gate TRIPPED — a {bytes}-byte store read back \
             empty; returning a SEALED, read-only handle (no checkpoint, no v41 upgrade, \
             no rebuild) so the on-disk bytes stay intact for rollback"
        );
        Some((
            store,
            WalRecovery {
                outcome: WalRecoveryOutcome::SuspectedDataLoss,
                recovered_records: 0,
                quarantined_wal: None,
            },
        ))
    }

    /// Seal the store after the #107 gate trips: refuse every checkpoint and
    /// disable count-based auto-checkpoint, so an apparently-empty store can
    /// never be upgraded to v41 (which would make the loss permanent).
    pub(crate) fn seal(&self) {
        self.sealed.set(true);
        self.checkpoint_interval.set(0);
    }

    // -- connection / execution helpers --------------------------------------

    fn conn(&self) -> crate::Result<Connection<'_>> {
        Connection::new(&self.db)
            .map_err(|e| MemoryError::Storage(format!("failed to open LadybugDB connection: {e}")))
    }

    /// Run a Cypher statement, discarding the result. Returns an error on failure.
    ///
    /// The returned error deliberately omits the Cypher text: because all values
    /// are interpolated into the query string, embedding it in a propagating
    /// error would leak stored memory content (and the agent's `agent_id`) into
    /// logs or any caller that surfaces the error. The full query is emitted at
    /// `debug` level for operators instead (mirrors the other backends).
    pub(crate) fn execute(&self, cypher: &str) -> crate::Result<()> {
        self.conn()?.query(cypher).map_err(|e| {
            debug!("lbug_store: Cypher execution failed — query: {cypher}");
            MemoryError::Storage(format!("Cypher execution failed: {e}"))
        })?;
        Ok(())
    }

    /// Run a Cypher query and materialize all result rows.
    ///
    /// See [`execute`](Self::execute) for why the interpolated query is not
    /// included in the returned error.
    pub(crate) fn query_rows(&self, cypher: &str) -> crate::Result<Vec<Vec<Value>>> {
        let conn = self.conn()?;
        let result = conn.query(cypher).map_err(|e| {
            debug!("lbug_store: Cypher query failed — query: {cypher}");
            MemoryError::Storage(format!("Cypher query failed: {e}"))
        })?;
        Ok(result.collect())
    }

    // -- durability ----------------------------------------------------------

    /// fsync the database file and its parent directory after a successful write.
    ///
    /// Without this barrier a `SIGKILL` between two writes could lose an
    /// acknowledged write because LadybugDB only flushes its WAL on
    /// `Database::drop`. Best-effort: a missing data file (LadybugDB may operate
    /// purely from the WAL before its first checkpoint) is tolerated.
    pub(crate) fn post_write_barrier(&self) -> crate::Result<()> {
        if self.db_path.exists() {
            if let Err(e) = open_and_fsync(&self.db_path) {
                if !is_not_found(&e) {
                    return Err(e);
                }
            }
        }
        let parent = self
            .db_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        if parent.exists() {
            open_and_fsync(parent)?;
        }
        Ok(())
    }

    // -- schema introspection / management -----------------------------------

    pub(crate) fn acquire_lock(&self) -> std::sync::MutexGuard<'_, ()> {
        self.lock.lock().unwrap_or_else(|e| {
            warn!("lbug_store mutex poisoned, recovering: {e}");
            e.into_inner()
        })
    }

    /// Populate the schema caches from the on-disk catalog exactly once.
    ///
    /// Makes the store reopen-safe: tables and columns created by a previous
    /// process become visible to read/update/delete paths without requiring a
    /// fresh `add_node` first.
    pub(crate) fn ensure_schema_loaded(&self) {
        if let Err(e) = self.try_ensure_schema_loaded() {
            // Best-effort for the infallible read/update/delete paths: log and
            // move on. We deliberately do NOT latch `schema_loaded = true` here.
            //
            // Latching on failure would (a) poison the fail-closed count path
            // (#2561) — `try_ensure_schema_loaded` would then short-circuit `Ok`
            // over an empty cache, and `try_count_nodes` would read that empty
            // cache as a *confirmed*-empty store — and (b) strand this handle as
            // "loaded, no tables" for its whole lifetime after a merely-transient
            // catalog failure. Leaving the flag unset lets the next read retry
            // and self-heal once the catalog is readable again. (In the healthy
            // case the first successful load latches the flag, so a warm store
            // still introspects exactly once.)
            warn!("lbug_store: show_tables introspection failed: {e}");
        }
    }

    /// Fallible variant of [`ensure_schema_loaded`](Self::ensure_schema_loaded).
    ///
    /// Populates the schema caches from the on-disk catalog, **propagating** a
    /// catalog read failure instead of swallowing it. The schema is marked
    /// loaded only on success, so a transient `show_tables` failure can be
    /// retried on the next call (and stays visible to the fail-closed count
    /// path, Simard #2561) rather than being permanently latched as
    /// "loaded, no tables".
    pub(crate) fn try_ensure_schema_loaded(&self) -> crate::Result<()> {
        if self.schema_loaded.get() {
            return Ok(());
        }

        // Test seam (#2561): simulate an unreadable catalog so the fail-closed
        // count path can be exercised deterministically. No-op outside tests.
        if force_read_error_for_test() {
            return Err(MemoryError::Storage(
                "forced catalog read error (test seam)".to_string(),
            ));
        }

        let rows = self.query_rows("CALL show_tables() RETURN *")?;

        for row in rows {
            let (name, ttype) = match table_row_name_and_type(&row) {
                Some(v) => v,
                None => continue,
            };
            let tupper = ttype.to_ascii_uppercase();
            if tupper.contains("REL") {
                if let Some((from, to)) = self.introspect_rel_endpoints(&name) {
                    // Back-fill the tombstone column on stores created before
                    // the #100 soft-delete fix so reads/deletes can rely on it.
                    self.ensure_soft_delete_column(&name);
                    self.known_rel_tables.borrow_mut().insert((name, from, to));
                }
            } else if tupper.contains("NODE") {
                self.ensure_soft_delete_column(&name);
                let cols = self.introspect_table_columns(&name);
                self.known_node_tables.borrow_mut().insert(name.clone());
                self.node_table_columns.borrow_mut().insert(name, cols);
            }
        }

        self.schema_loaded.set(true);
        Ok(())
    }

    /// Return the user-defined column names of `table` from the catalog.
    fn introspect_table_columns(&self, table: &str) -> HashSet<String> {
        let mut cols = HashSet::new();
        let cypher = format!("CALL table_info('{}') RETURN *", escape_cypher(table));
        if let Ok(rows) = self.query_rows(&cypher) {
            for row in rows {
                // table_info rows are [property_id, name, type, ...]; the column
                // name is the first String value in the row.
                if let Some(col) = row.iter().find_map(value_as_str) {
                    if col != "node_id" && col != "graph_origin" {
                        cols.insert(col.to_string());
                    }
                }
            }
        }
        cols
    }

    /// `true` if `table`'s on-disk schema carries the soft-delete tombstone
    /// column ([`DELETED_COL`]).
    ///
    /// Consults the node-table column cache populated by
    /// [`ensure_schema_loaded`](Self::ensure_schema_loaded); for tables absent
    /// from that cache (e.g. rel tables) it introspects the catalog directly.
    /// Callers should have run `ensure_schema_loaded` first so the cache — and
    /// any `_deleted` back-fill — is up to date.
    pub(crate) fn table_has_deleted_column(&self, table: &str) -> bool {
        if let Some(cols) = self.node_table_columns.borrow().get(table) {
            return cols.contains(DELETED_COL);
        }
        self.introspect_table_columns(table).contains(DELETED_COL)
    }

    /// The live-row tombstone predicate ([`not_deleted`]) for a **labeled** read
    /// of `table`, or `None` when `table`'s schema lacks [`DELETED_COL`].
    ///
    /// #107 (silent empty-read): lbug 0.17.1 strictly binds a labeled
    /// `MATCH (n:Label)` against the table schema, so emitting
    /// `(n._deleted IS NULL OR n._deleted = '')` against a legacy table that
    /// predates the soft-delete column is a hard binder error
    /// (`Cannot find property _deleted`) — which the read helpers swallow to an
    /// empty result, silently dropping every row over a fully-populated store.
    /// Gating the predicate on the actual on-disk schema keeps a labeled read
    /// correct whether or not the back-fill
    /// ([`ensure_soft_delete_column`](Self::ensure_soft_delete_column)) has added
    /// the column. (lbug 0.15.4 tolerated the missing property as `NULL`, which
    /// is why the same store read correctly there.)
    pub(crate) fn tombstone_filter(&self, table: &str, alias: &str) -> Option<String> {
        self.table_has_deleted_column(table)
            .then(|| not_deleted(alias))
    }

    /// Ensure the reserved soft-delete tombstone column ([`DELETED_COL`]) exists
    /// on `table`.
    ///
    /// Soft deletes (`SET <alias>._deleted = '1'`) replace physical `DELETE`
    /// against committed CSR rel groups, which corrupts lbug's CSR node-group
    /// index (`getGroup(UINT32_MAX)` SEGV, #100). The column must therefore be
    /// present on every node and rel table before a delete or a filtered read.
    /// Idempotent and reopen-safe: introspects first so a redundant
    /// `ALTER ... ADD` (which errors when the column already exists) is skipped.
    /// Best-effort — a failure is logged and the column simply stays absent
    /// (reads then show, rather than hide, rows on that table).
    fn ensure_soft_delete_column(&self, table: &str) {
        if !is_valid_identifier(table) {
            return;
        }
        if self.introspect_table_columns(table).contains(DELETED_COL) {
            return;
        }
        let ddl = format!("ALTER TABLE {table} ADD {DELETED_COL} STRING DEFAULT ''");
        if let Err(e) = self.execute(&ddl) {
            warn!("ensure_soft_delete_column: failed to add {DELETED_COL} to {table}: {e}");
        }
    }

    /// Return (from_table, to_table) for a rel table, if discoverable.
    fn introspect_rel_endpoints(&self, rel: &str) -> Option<(String, String)> {
        let cypher = format!("CALL show_connection('{}') RETURN *", escape_cypher(rel));
        let rows = self.query_rows(&cypher).ok()?;
        let row = rows.into_iter().next()?;
        let strings: Vec<String> = row
            .iter()
            .filter_map(|v| value_as_str(v).map(String::from))
            .collect();
        match strings.as_slice() {
            [from, to, ..] => Some((from.clone(), to.clone())),
            _ => None,
        }
    }

    /// Ensure a node table exists with (at least) the reserved columns plus the
    /// given extra `STRING` columns. Idempotent and reopen-safe.
    pub(crate) fn ensure_node_table(
        &self,
        table: &str,
        extra_cols: &HashSet<String>,
    ) -> crate::Result<()> {
        validate_identifier(table)?;
        for c in extra_cols {
            validate_identifier(c)?;
        }
        self.ensure_schema_loaded();

        let known = self.known_node_tables.borrow().contains(table);
        if known {
            // Add any genuinely new columns via ALTER (best-effort).
            let missing: Vec<String> = {
                let cache = self.node_table_columns.borrow();
                let existing = cache.get(table);
                extra_cols
                    .iter()
                    .filter(|c| existing.is_none_or(|s| !s.contains(*c)))
                    .cloned()
                    .collect()
            };
            for col in missing {
                let ddl = format!("ALTER TABLE {table} ADD {col} STRING DEFAULT ''");
                if let Err(e) = self.execute(&ddl) {
                    warn!("ensure_node_table: failed to add column {col} to {table}: {e}");
                } else {
                    self.node_table_columns
                        .borrow_mut()
                        .entry(table.to_string())
                        .or_default()
                        .insert(col);
                }
            }
            return Ok(());
        }

        let mut defs = vec![
            "node_id STRING".to_string(),
            "graph_origin STRING".to_string(),
            format!("{DELETED_COL} STRING DEFAULT ''"),
        ];
        for col in extra_cols {
            defs.push(format!("{col} STRING DEFAULT ''"));
        }
        let ddl = format!(
            "CREATE NODE TABLE IF NOT EXISTS {table}({}, PRIMARY KEY(node_id))",
            defs.join(", ")
        );
        self.execute(&ddl)?;

        self.known_node_tables
            .borrow_mut()
            .insert(table.to_string());
        // The table is created WITH the soft-delete tombstone column (above), so
        // record it in the column cache alongside the user columns. Without this
        // a table created in-session reads back as "no `_deleted` column", which
        // disables tombstone filtering for its labeled reads until the handle is
        // reopened — so soft-deleted rows would leak back into query results
        // (e.g. a pruned applied-intent marker still counted).
        let mut cached_cols = extra_cols.clone();
        cached_cols.insert(DELETED_COL.to_string());
        self.node_table_columns
            .borrow_mut()
            .insert(table.to_string(), cached_cols);
        Ok(())
    }

    /// Ensure a rel table exists between `from`/`to` node tables.
    pub(crate) fn ensure_rel_table(
        &self,
        rel: &str,
        from: &str,
        to: &str,
        extra_cols: &HashSet<String>,
    ) -> crate::Result<()> {
        validate_identifier(rel)?;
        validate_identifier(from)?;
        validate_identifier(to)?;
        for c in extra_cols {
            validate_identifier(c)?;
        }
        self.ensure_schema_loaded();

        let key = (rel.to_string(), from.to_string(), to.to_string());
        if self.known_rel_tables.borrow().contains(&key) {
            return Ok(());
        }
        self.ensure_node_table(from, &HashSet::new())?;
        self.ensure_node_table(to, &HashSet::new())?;

        let mut defs = vec![
            format!("FROM {from} TO {to}"),
            "edge_id STRING".to_string(),
            "graph_origin STRING".to_string(),
            format!("{DELETED_COL} STRING DEFAULT ''"),
        ];
        for col in extra_cols {
            defs.push(format!("{col} STRING DEFAULT ''"));
        }
        let ddl = format!("CREATE REL TABLE IF NOT EXISTS {rel}({})", defs.join(", "));
        self.execute(&ddl)?;
        self.known_rel_tables.borrow_mut().insert(key);
        Ok(())
    }
}

impl Drop for LbugGraphStore {
    /// Always-on durability: checkpoint the WAL into the main database file when
    /// the store is dropped, so even a forgotten `close` leaves the data durable
    /// without an unbounded WAL to replay. Best-effort and silent on error —
    /// LadybugDB also force-checkpoints on its own drop as a backstop.
    fn drop(&mut self) {
        // #107: a sealed store must never checkpoint on drop either — that would
        // upgrade an apparently-empty store to v41 and destroy the rollback path.
        if self.sealed.get() {
            debug!(
                "lbug_store: skipping drop checkpoint — store sealed by the #107 empty-read gate"
            );
            return;
        }
        if let Err(e) = self.execute("CHECKPOINT") {
            debug!("lbug_store: checkpoint on drop failed (engine will retry): {e}");
        }
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// An in-memory snapshot of every live node and edge in a store.
///
/// Captured by [`LbugGraphStore::dump_graph`] and replayed by
/// [`LbugGraphStore::reload_graph`] to salvage a recovered graph into a fresh
/// database when an in-place checkpoint fails (#2550), so the pre-corruption
/// prefix is persisted durably instead of being reset to empty.
pub(crate) struct GraphDump {
    /// Every live node, with its type and (non-reserved) properties.
    pub(crate) nodes: Vec<GraphNode>,
    /// Every live edge, with its endpoints, type and (non-reserved) properties.
    pub(crate) edges: Vec<GraphEdge>,
}

/// Strip reserved/internal columns from a salvaged node/edge property map so a
/// reload via `add_node`/`add_edge` re-derives them: the id is passed explicitly,
/// `graph_origin` is set to the destination store, and the tombstone column
/// defaults to "live". Underscore-prefixed columns are engine/internal by
/// convention and are never user data.
///
/// Note: salvage preserves node ids, node/edge types, endpoints, and all user
/// properties (the durable-recall payload). It does **not** preserve edge ids
/// (`add_edge` mints a fresh `edge_id`) or the original per-record `graph_origin`
/// (reset to the destination store) — those are internal/federation metadata, not
/// recall content.
fn salvage_props(properties: &HashMap<String, String>) -> HashMap<String, String> {
    properties
        .iter()
        .filter(|(k, _)| {
            k.as_str() != "node_id"
                && k.as_str() != "edge_id"
                && k.as_str() != "graph_origin"
                && !k.starts_with('_')
        })
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

/// Test-only deterministic fault injection for the recovery path.
#[cfg(test)]
pub(crate) mod test_hooks {
    use std::cell::Cell;

    thread_local! {
        /// Number of upcoming [`LbugGraphStore::recovery_checkpoint`] calls on this
        /// thread that must be forced to fail. Thread-local so parallel tests do
        /// not interfere; recovery runs synchronously on the caller's thread.
        static FORCED_RECOVERY_CHECKPOINT_FAILURES: Cell<u32> = const { Cell::new(0) };
    }

    /// Force the next `n` checkpoint-after-recovery attempts on this thread to
    /// fail, reproducing the #2550 "checkpoint after recovery failed" incident
    /// deterministically. Only the recovery in-place checkpoint consults this — the
    /// salvage path's fresh-database checkpoint is unaffected.
    pub(crate) fn force_recovery_checkpoint_failures(n: u32) {
        FORCED_RECOVERY_CHECKPOINT_FAILURES.with(|c| c.set(n));
    }

    /// Returns `true` (and decrements the pending count) if a forced recovery
    /// checkpoint failure is queued.
    pub(crate) fn take_forced_recovery_checkpoint_failure() -> bool {
        FORCED_RECOVERY_CHECKPOINT_FAILURES.with(|c| {
            let n = c.get();
            if n > 0 {
                c.set(n - 1);
                true
            } else {
                false
            }
        })
    }
}

/// Reserved tombstone column used for soft deletes (#100).
///
/// `delete_node` / `delete_edge` mark a row deleted by `SET`ting this column to
/// [`DELETED_MARK`] instead of issuing a physical `DELETE` against a committed
/// CSR rel group — the operation that drives lbug's CSR node-group index to the
/// `UINT32_MAX` sentinel and SEGVs the next scan (`getGroup(UINT32_MAX)`). Reads
/// filter tombstoned rows out via [`not_deleted`]. The name is `_`-prefixed so
/// `node_val_to_graph_node` / `rel_val_to_edge` already strip it from returned
/// properties (the existing reserved-underscore convention).
pub(crate) const DELETED_COL: &str = "_deleted";

/// Value written to [`DELETED_COL`] to mark a row deleted. Live rows hold the
/// column default (`''`) or `NULL` (back-filled rows on older stores).
pub(crate) const DELETED_MARK: &str = "1";

/// A Cypher predicate that is true only for *live* (non-tombstoned) rows bound
/// to `alias`. Tolerant of both the `''` default and a `NULL` back-fill so it
/// stays correct on stores migrated by `ALTER TABLE ... ADD` as well as fresh
/// ones, and never hides a live row.
pub(crate) fn not_deleted(alias: &str) -> String {
    format!("({alias}.{DELETED_COL} IS NULL OR {alias}.{DELETED_COL} = '')")
}

/// On-disk footprint threshold (bytes) below which a store is treated as
/// fresh/empty by the [`committed_footprint`] probe.
///
/// Empirically (lbug 0.17.1, this crate's `SystemConfig`) a checkpointed store
/// measures ~16 KiB empty (no tables), ~48 KiB with one node, and ~132 KiB with
/// ten nodes; the live production store is ~31 MB. 32 KiB sits above the
/// empty/cognitive-empty baseline and below a single-node store, so a genuinely
/// new store is `Empty` while any store that has ever committed a row is
/// `NonEmpty`.
pub(crate) const EMPTY_STORE_FOOTPRINT_THRESHOLD: u64 = 32 * 1024;

/// Independent on-disk evidence of whether a store *should* be populated, used
/// by the #107 empty-read gate ([`LbugGraphStore::empty_read_gate`]) so the
/// decision does not rely on the (suspect) catalog read as its own baseline.
///
/// It reads the durable database file size — not the catalog rows the bug zeroes
/// out — so it stays trustworthy even when the read path is broken.
pub(crate) enum CommittedFootprint {
    /// No durable evidence of committed data (size at/near the fresh-store
    /// baseline). A legitimately new or empty store.
    Empty,
    /// A material committed footprint (the store was written with real data by a
    /// previous process).
    NonEmpty {
        /// The committed on-disk size in bytes.
        bytes: u64,
    },
}

/// Probe the committed on-disk footprint of the single-file store at `db_path`.
pub(crate) fn committed_footprint(db_path: &Path) -> CommittedFootprint {
    let bytes = std::fs::metadata(db_path).map(|m| m.len()).unwrap_or(0);
    if bytes > EMPTY_STORE_FOOTPRINT_THRESHOLD {
        CommittedFootprint::NonEmpty { bytes }
    } else {
        CommittedFootprint::Empty
    }
}

/// Whether tests have forced the #107 gate to observe the post-open read as
/// empty (see [`test_seam`]). Always `false` outside tests.
#[cfg(test)]
fn force_empty_read_for_test() -> bool {
    test_seam::force_empty_read()
}
#[cfg(not(test))]
fn force_empty_read_for_test() -> bool {
    false
}

/// Whether tests have forced the fail-closed count path to observe a genuine
/// backend read error (see [`test_seam`]). Always `false` outside tests.
///
/// Lets a test exercise the #2561 fail-closed propagation deterministically —
/// a transient read failure on an *existing, populated* table must surface as
/// `Err`, not a swallowed empty — without depending on a specific broken engine
/// build.
#[cfg(test)]
pub(crate) fn force_read_error_for_test() -> bool {
    test_seam::force_read_error()
}
#[cfg(not(test))]
pub(crate) fn force_read_error_for_test() -> bool {
    false
}

/// The system configuration used for every open.
///
/// Buffer-pool and max-database-size limits are resolved from the environment
/// (`AMPLIHACK_MEMORY_BUFFER_POOL_BYTES` / `AMPLIHACK_MEMORY_MAX_DB_BYTES`) via
/// [`effective_limits`], falling back to larger, safer defaults than the old
/// hardcoded 128 MiB pool / 1 GiB cap that let an auto-CHECKPOINT exhaust the
/// buffer pool and corrupt the catalog (#95). lbug allocates the pool lazily and
/// `max_db_size` is only an mmap reservation, so the larger caps cost nothing
/// until data actually needs them.
///
/// `auto_checkpoint(true)` lets LadybugDB bound the WAL on its own as a safety
/// net; the store additionally checkpoints every [`AUTO_CHECKPOINT_WRITES`]
/// writes (see [`LbugGraphStore::note_write_and_maybe_checkpoint`]).
/// `throw_on_wal_replay_failure` selects strict vs. resilient WAL replay: strict
/// (`true`) errors on a corrupt WAL, resilient (`false`) replays the good prefix
/// and ignores the unreplayable tail.
fn system_config(throw_on_wal_replay_failure: bool) -> SystemConfig {
    let buffer_env = std::env::var(ENV_BUFFER_POOL_BYTES).ok();
    let max_db_env = std::env::var(ENV_MAX_DB_BYTES).ok();
    let (buffer_pool, max_db) = effective_limits(buffer_env.as_deref(), max_db_env.as_deref());
    log_effective_limits_once(buffer_pool, max_db);
    SystemConfig::default()
        .max_db_size(max_db)
        .buffer_pool_size(buffer_pool)
        .auto_checkpoint(true)
        .throw_on_wal_replay_failure(throw_on_wal_replay_failure)
}

/// Parse a byte count from an override string: a strictly-positive integer, or
/// `None` for anything missing / unparseable / zero.
fn parse_positive_bytes(s: &str) -> Option<u64> {
    match s.trim().parse::<u64>() {
        Ok(n) if n > 0 => Some(n),
        _ => None,
    }
}

/// Resolve the effective buffer-pool cap (bytes) from the optional
/// [`ENV_BUFFER_POOL_BYTES`] override. A valid positive value is clamped up to
/// [`MIN_BUFFER_POOL_BYTES`]; anything missing or invalid falls back to
/// [`DEFAULT_BUFFER_POOL_BYTES`]. Pure (takes the value, not the env) so it is
/// unit-testable without mutating process state.
pub(crate) fn resolve_buffer_pool_bytes(env: Option<&str>) -> u64 {
    match env.and_then(parse_positive_bytes) {
        Some(n) => n.max(MIN_BUFFER_POOL_BYTES),
        None => DEFAULT_BUFFER_POOL_BYTES,
    }
}

/// Resolve the effective maximum database size (bytes) from the optional
/// [`ENV_MAX_DB_BYTES`] override. A valid positive value is clamped up to
/// [`MIN_MAX_DB_BYTES`]; anything missing or invalid falls back to
/// [`DEFAULT_MAX_DB_BYTES`]. Pure and unit-testable.
pub(crate) fn resolve_max_db_bytes(env: Option<&str>) -> u64 {
    match env.and_then(parse_positive_bytes) {
        Some(n) => n.max(MIN_MAX_DB_BYTES),
        None => DEFAULT_MAX_DB_BYTES,
    }
}

/// Resolve both limits and enforce the invariant `buffer_pool <= max_db_size`
/// (a buffer pool larger than the whole database is meaningless). Returns
/// `(buffer_pool_bytes, max_db_bytes)`.
pub(crate) fn effective_limits(buffer_env: Option<&str>, max_db_env: Option<&str>) -> (u64, u64) {
    let max_db = resolve_max_db_bytes(max_db_env);
    let buffer_pool = resolve_buffer_pool_bytes(buffer_env).min(max_db);
    (buffer_pool, max_db)
}

/// Emit the effective limits exactly once per process (recovery re-opens call
/// [`system_config`] repeatedly; we don't want to spam the log).
fn log_effective_limits_once(buffer_pool: u64, max_db: u64) {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        info!(
            buffer_pool_bytes = buffer_pool,
            max_db_bytes = max_db,
            "lbug_store: effective LadybugDB limits (override via {} / {})",
            ENV_BUFFER_POOL_BYTES,
            ENV_MAX_DB_BYTES
        );
    });
}

/// Open a [`Database`] with the given replay strictness, mapping the engine
/// error to a `String`.
fn open_database(db_path: &Path, strict: bool) -> std::result::Result<Database, String> {
    Database::new(db_path, system_config(strict)).map_err(|e| e.to_string())
}

/// System configuration for the #107 read-only empty-read peek. Read-only so the
/// peek cannot mutate the store; auto-checkpoint disabled belt-and-suspenders.
fn read_only_system_config() -> SystemConfig {
    let buffer_env = std::env::var(ENV_BUFFER_POOL_BYTES).ok();
    let max_db_env = std::env::var(ENV_MAX_DB_BYTES).ok();
    let (buffer_pool, max_db) = effective_limits(buffer_env.as_deref(), max_db_env.as_deref());
    SystemConfig::default()
        .max_db_size(max_db)
        .buffer_pool_size(buffer_pool)
        .auto_checkpoint(false)
        .read_only(true)
        .throw_on_wal_replay_failure(true)
}

/// Open a [`Database`] **read-only**, containing any panic like
/// [`try_open_database`]. Used by the #107 empty-read peek so the gate never
/// mutates the store it is inspecting.
fn try_open_database_read_only(db_path: &Path) -> std::result::Result<Database, String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Database::new(db_path, read_only_system_config()).map_err(|e| e.to_string())
    })) {
        Ok(res) => res,
        Err(panic) => Err(format!(
            "panic during read-only open: {}",
            panic_message(panic)
        )),
    }
}

/// Open a [`Database`], additionally containing any panic (a corrupt WAL surfaces
/// as `Err` in practice, but `catch_unwind` guarantees we never crash the
/// process during a recovery attempt).
fn try_open_database(db_path: &Path, strict: bool) -> std::result::Result<Database, String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        open_database(db_path, strict)
    })) {
        Ok(res) => res,
        Err(panic) => Err(format!("panic during open: {}", panic_message(panic))),
    }
}

fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = panic.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = panic.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// The write-ahead-log path for `db_path`.
///
/// LadybugDB appends `.wal` to the *full* database filename (extension included),
/// so `cognitive.ladybug` -> `cognitive.ladybug.wal`. (This is intentionally not
/// `Path::with_extension`, which would wrongly produce `cognitive.wal`.)
pub(crate) fn wal_path_for(db_path: &Path) -> PathBuf {
    let mut name: OsString = db_path.as_os_str().to_os_string();
    name.push(".wal");
    PathBuf::from(name)
}

/// Quarantine target for a corrupt WAL: `<wal>.corrupt-<unix_ts>`.
fn quarantine_path(wal: &Path) -> PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut name: OsString = wal.as_os_str().to_os_string();
    name.push(format!(".corrupt-{ts}"));
    PathBuf::from(name)
}

/// Sidecar files LadybugDB may write alongside the WAL (shadow pages, temporary
/// WAL segments). Returns existing siblings whose name starts with the WAL or db
/// filename and carries a `.shadow` / `.wal.` infix.
fn wal_sidecars(db_path: &Path, wal: &Path) -> Vec<PathBuf> {
    let parent = wal.parent().unwrap_or_else(|| Path::new("."));
    let db_name = db_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let wal_name = wal
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(parent) {
        for entry in entries.flatten() {
            let fname = entry.file_name().to_string_lossy().to_string();
            if fname == wal_name {
                continue; // the WAL itself is handled separately
            }
            let is_sidecar = fname.starts_with(&format!("{wal_name}."))
                || (fname.starts_with(&db_name) && fname.contains(".shadow"));
            if is_sidecar {
                out.push(entry.path());
            }
        }
    }
    out
}

/// Copy the corrupt WAL (and any sidecars) to `quarantine`, preserving the
/// incident artifact before a resilient open consumes the live WAL. Returns the
/// quarantine path on success.
fn copy_wal_aside(db_path: &Path, wal: &Path, quarantine: &Path) -> Option<PathBuf> {
    match std::fs::copy(wal, quarantine) {
        Ok(_) => {
            for sidecar in wal_sidecars(db_path, wal) {
                if let Some(name) = sidecar.file_name() {
                    let mut dst: OsString = quarantine.as_os_str().to_os_string();
                    dst.push(".");
                    dst.push(name);
                    let _ = std::fs::copy(&sidecar, PathBuf::from(dst));
                }
            }
            Some(quarantine.to_path_buf())
        }
        Err(e) => {
            warn!(
                "lbug_store: failed to quarantine corrupt WAL {}: {e}",
                wal.display()
            );
            None
        }
    }
}

/// Move the WAL (and sidecars) fully aside so the engine opens with no WAL to
/// replay. If a copy was already quarantined, the live WAL is just removed (the
/// artifact is already preserved); otherwise it is renamed to `quarantine`.
fn move_wal_aside(
    db_path: &Path,
    wal: &Path,
    quarantine: &Path,
    already_copied: bool,
) -> Option<PathBuf> {
    let result = if already_copied {
        let _ = std::fs::remove_file(wal);
        Some(quarantine.to_path_buf())
    } else {
        match std::fs::rename(wal, quarantine) {
            Ok(_) => Some(quarantine.to_path_buf()),
            Err(_) => {
                let _ = std::fs::remove_file(wal);
                None
            }
        }
    };
    for sidecar in wal_sidecars(db_path, wal) {
        let _ = std::fs::remove_file(&sidecar);
    }
    result
}

/// Quarantine a corrupt main database (and any leftover WAL / sidecars) to
/// `quarantine`, preserving everything before clearing `db_path` so a fresh
/// database can be created there.
///
/// The main file is moved with [`move_path_aside`] (atomic rename, falling back
/// to copy-then-remove so the artifact is never lost); on a hard failure to
/// preserve it the original is left intact and an error is returned (we never
/// destroy the corrupt database we can't first copy). Leftover WAL / sidecars
/// are best-effort moved next to the quarantined main file. Returns the
/// quarantine path of the main file if it existed.
fn quarantine_db_artifacts(db_path: &Path, quarantine: &Path) -> crate::Result<Option<PathBuf>> {
    let moved = if db_path.exists() {
        let dst = move_path_aside(db_path, quarantine).map_err(|e| {
            MemoryError::Storage(format!(
                "failed to quarantine corrupt database {}: {e}",
                db_path.display()
            ))
        })?;
        Some(dst)
    } else {
        None
    };

    // Best-effort: move any leftover WAL + sidecars aside too so the fresh open
    // starts from a clean directory. They are preserved next to the main file's
    // quarantine path, never silently deleted.
    let wal = wal_path_for(db_path);
    if wal.exists() {
        let mut wal_q: OsString = quarantine.as_os_str().to_os_string();
        wal_q.push(".wal");
        let _ = move_path_aside(&wal, &PathBuf::from(wal_q));
    }
    for sidecar in wal_sidecars(db_path, &wal) {
        if let Some(name) = sidecar.file_name() {
            let mut dst: OsString = quarantine.as_os_str().to_os_string();
            dst.push(".");
            dst.push(name);
            let _ = move_path_aside(&sidecar, &PathBuf::from(dst));
        }
    }

    Ok(moved)
}

/// Move `src` to `dst`, preserving the data: prefer an atomic rename, and on
/// failure (e.g. a cross-device `EXDEV`) fall back to copy-then-remove so the
/// artifact is preserved at `dst` before the original is cleared. Errors only if
/// the artifact could not be preserved at all, in which case `src` is left
/// intact.
fn move_path_aside(src: &Path, dst: &Path) -> std::io::Result<PathBuf> {
    if std::fs::rename(src, dst).is_ok() {
        return Ok(dst.to_path_buf());
    }
    // Rename failed: preserve a copy first, then clear the original.
    std::fs::copy(src, dst)?;
    let _ = std::fs::remove_file(src);
    Ok(dst.to_path_buf())
}

/// Interpret a scalar [`Value`] as a `usize` count (from `count(n)` results).
fn value_as_usize(v: &Value) -> usize {
    match v {
        Value::Int64(i) => (*i).max(0) as usize,
        Value::Int32(i) => (*i).max(0) as usize,
        Value::UInt64(i) => *i as usize,
        Value::UInt32(i) => *i as usize,
        other => value_to_string(other).parse::<usize>().unwrap_or(0),
    }
}

/// `true` if `name` is a safe Cypher identifier (`[A-Za-z_][A-Za-z0-9_]*`).
pub(crate) fn is_valid_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

pub(crate) fn validate_identifier(name: &str) -> crate::Result<()> {
    if is_valid_identifier(name) {
        Ok(())
    } else {
        Err(MemoryError::InvalidInput(format!(
            "Invalid identifier: {name:?}. Must match [A-Za-z_][A-Za-z0-9_]*"
        )))
    }
}

/// Escape a string for safe interpolation inside a single-quoted Cypher literal.
pub(crate) fn escape_cypher(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\0' => out.push_str("\\0"),
            _ => out.push(c),
        }
    }
    out
}

/// Borrow the `&str` from a [`Value::String`], else `None`.
pub(crate) fn value_as_str(v: &Value) -> Option<&str> {
    match v {
        Value::String(s) => Some(s.as_str()),
        _ => None,
    }
}

/// Render any scalar [`Value`] as the string we store/return. All cognitive
/// properties are stored in `STRING` columns, so this is dominated by the
/// `Value::String` case; numeric/bool fallbacks keep the helper total.
pub(crate) fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Int64(i) => i.to_string(),
        Value::Int32(i) => i.to_string(),
        Value::Int16(i) => i.to_string(),
        Value::Int8(i) => i.to_string(),
        Value::UInt64(i) => i.to_string(),
        Value::UInt32(i) => i.to_string(),
        Value::UInt16(i) => i.to_string(),
        Value::UInt8(i) => i.to_string(),
        Value::Int128(i) => i.to_string(),
        Value::Double(d) => d.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Null(_) => String::new(),
        other => format!("{other}"),
    }
}

/// Extract (table name, table type) from a `show_tables` result row.
///
/// LadybugDB returns `[id, name, type, ...]`; we read the name (first String
/// value) and the type (the String value equal to NODE/REL, falling back to the
/// value immediately following the name).
fn table_row_name_and_type(row: &[Value]) -> Option<(String, String)> {
    let name = row.iter().find_map(value_as_str)?.to_string();
    let ttype = row
        .iter()
        .filter_map(value_as_str)
        .find(|s| {
            let u = s.to_ascii_uppercase();
            u == "NODE" || u == "REL"
        })
        .map(String::from)
        .or_else(|| {
            // Fall back to the String value after the name.
            let mut seen_name = false;
            for v in row {
                if let Some(s) = value_as_str(v) {
                    if seen_name {
                        return Some(s.to_string());
                    }
                    if s == name {
                        seen_name = true;
                    }
                }
            }
            None
        })?;
    Some((name, ttype))
}

fn open_and_fsync(path: &Path) -> crate::Result<()> {
    let f = std::fs::OpenOptions::new()
        .read(true)
        .open(path)
        .map_err(|e| MemoryError::Storage(format!("fsync open {}: {e}", path.display())))?;
    f.sync_all()
        .map_err(|e| MemoryError::Storage(format!("fsync {}: {e}", path.display())))?;
    Ok(())
}

fn is_not_found(err: &MemoryError) -> bool {
    matches!(err, MemoryError::Storage(reason) if reason.contains("No such file or directory"))
}

/// Test-only seam for the #107 empty-read safety gate.
///
/// Lets a test force the store-open path to *observe* a populated store as if it
/// had read back empty, so the gate's fail-closed behaviour
/// ([`WalRecoveryOutcome::SuspectedDataLoss`]) can be exercised deterministically
/// without depending on a specific broken engine build. The gate (in
/// `open_with_recovery`) will consult [`test_seam::force_empty_read`] when
/// deciding whether the post-open node count should be treated as empty.
///
/// Defined at the end of the module so it does not trip
/// `clippy::items_after_test_module` (the lint flags real items declared after an
/// inline `#[cfg(test)]` module).
#[cfg(test)]
pub(crate) mod test_seam {
    use std::cell::Cell;

    thread_local! {
        static FORCE_EMPTY_READ: Cell<bool> = const { Cell::new(false) };
        static FORCE_READ_ERROR: Cell<bool> = const { Cell::new(false) };
    }

    /// Force (or stop forcing) the open path to treat the store as read-empty.
    ///
    /// Thread-local so a test that flips it does **not** pollute the #107 gate of
    /// other tests running in parallel (the gate runs on the same thread as the
    /// `open_with_recovery` call that observes it).
    pub(crate) fn set_force_empty_read(on: bool) {
        FORCE_EMPTY_READ.with(|c| c.set(on));
    }

    /// Whether the open path should treat the post-open read as empty.
    pub(crate) fn force_empty_read() -> bool {
        FORCE_EMPTY_READ.with(|c| c.get())
    }

    /// Force (or stop forcing) the fail-closed count read primitive
    /// (`try_match_return_nodes`) to fail with a synthetic backend read error.
    ///
    /// Thread-local for the same parallel-isolation reason as
    /// [`set_force_empty_read`]. Used by the #2561 fail-closed count tests to
    /// simulate a transient read failure on an existing, populated table.
    pub(crate) fn set_force_read_error(on: bool) {
        FORCE_READ_ERROR.with(|c| c.set(on));
    }

    /// Whether the fail-closed count read primitive should fail.
    pub(crate) fn force_read_error() -> bool {
        FORCE_READ_ERROR.with(|c| c.get())
    }
}
