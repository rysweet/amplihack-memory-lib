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
//! * **Durability** — writes are serialized through a [`Mutex`], every mutating
//!   operation issues a per-write `fsync` barrier (data file + parent
//!   directory), and `close` issues a `CHECKPOINT` so a subsequent reopen sees
//!   all committed writes. Without the barrier a crash between two writes could
//!   lose an acknowledged write, since LadybugDB only flushes its WAL on
//!   `Database::drop`.
//! * **Bounded crash loss** — the store auto-checkpoints after every
//!   [`AUTO_CHECKPOINT_WRITES`] mutating operations (and always on `close` /
//!   `Drop`), flushing the write-ahead log into the main database file. An
//!   unclean shutdown therefore loses at most the handful of writes accumulated
//!   since the last checkpoint, rather than every uncheckpointed record.
//! * **Corrupt-WAL recovery** — [`LbugGraphStore::open_with_recovery`] opens a
//!   store whose WAL was left partially written by an unclean shutdown. A strict
//!   open is attempted first; if it fails because the WAL replay cannot complete
//!   (the failure mode that previously made the store permanently unopenable),
//!   the corrupt WAL is quarantined to `<wal>.corrupt-<ts>`, a resilient open
//!   replays the good prefix, and a `CHECKPOINT` folds the recovered records into
//!   the main database file so a subsequent clean reopen needs no replay.
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
use tracing::{debug, warn};

use crate::MemoryError;

/// Number of mutating operations after which the store auto-checkpoints,
/// folding the write-ahead log into the main database file. This bounds how
/// much acknowledged-but-uncheckpointed data an unclean shutdown can strand in
/// the WAL (and therefore put at risk if that WAL is later found corrupt).
pub const AUTO_CHECKPOINT_WRITES: u64 = 128;

/// How [`LbugGraphStore::open_with_recovery`] opened the store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalRecoveryOutcome {
    /// The write-ahead log replayed cleanly; no recovery was performed and no
    /// artifact was written.
    Clean,
    /// A corrupt WAL tail was quarantined; the good prefix was replayed and
    /// checkpointed into the main database file.
    RecoveredPrefix,
    /// The WAL was unusable even in resilient mode; it was quarantined and the
    /// store was opened from the last good checkpoint only.
    CheckpointOnly,
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
    /// Where the corrupt WAL (and any sidecars) were quarantined, if recovery
    /// ran. The bad WAL is moved aside — never deleted.
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

/// LadybugDB-backed persistent [`GraphStore`](crate::graph::protocol::GraphStore).
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
    /// # Errors
    ///
    /// Returns [`MemoryError::Storage`] if the database cannot be opened even
    /// after quarantining the WAL (e.g. the parent directory is unwritable or the
    /// main database file itself is unreadable).
    pub fn open_with_recovery(
        db_path: &Path,
        store_id: Option<&str>,
    ) -> crate::Result<(Self, WalRecovery)> {
        Self::prepare_parent(db_path)?;

        // 1. Fast path: strict open. A corrupt WAL surfaces as Err (lbug/cxx
        //    converts the C++ replay assertion into a Rust error rather than
        //    aborting); catch_unwind additionally contains any panic.
        match try_open_database(db_path, true) {
            Ok(db) => Ok((
                Self::from_parts(db, db_path, store_id),
                WalRecovery::clean(),
            )),
            Err(strict_err) => {
                let wal = wal_path_for(db_path);
                if !wal.exists() {
                    // No WAL: the failure is not WAL-replay related (bad path,
                    // quota, permissions, ...). Surface it unchanged.
                    return Err(MemoryError::Storage(format!(
                        "failed to open LadybugDB at {}: {strict_err}",
                        db_path.display()
                    )));
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
                // Fold the recovered prefix into the main DB so a later clean
                // reopen needs no replay, then count what survived.
                if let Err(e) = store.do_checkpoint() {
                    warn!("lbug_store: checkpoint after recovery failed: {e}");
                }
                let recovered = store.count_all_nodes();
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
                let db = open_database(db_path, true).map_err(|e| {
                    MemoryError::Storage(format!(
                        "failed to open LadybugDB at {} even after quarantining the WAL: {e}",
                        db_path.display()
                    ))
                })?;
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
        }
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
    pub(crate) fn do_checkpoint(&self) -> crate::Result<()> {
        self.execute("CHECKPOINT")?;
        self.writes_since_checkpoint.set(0);
        self.post_write_barrier()?;
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
                warn!("lbug_store: auto-checkpoint failed: {e}");
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
        if self.schema_loaded.get() {
            return;
        }
        self.schema_loaded.set(true);

        let rows = match self.query_rows("CALL show_tables() RETURN *") {
            Ok(r) => r,
            Err(e) => {
                warn!("lbug_store: show_tables introspection failed: {e}");
                return;
            }
        };

        for row in rows {
            let (name, ttype) = match table_row_name_and_type(&row) {
                Some(v) => v,
                None => continue,
            };
            let tupper = ttype.to_ascii_uppercase();
            if tupper.contains("REL") {
                if let Some((from, to)) = self.introspect_rel_endpoints(&name) {
                    self.known_rel_tables.borrow_mut().insert((name, from, to));
                }
            } else if tupper.contains("NODE") {
                let cols = self.introspect_table_columns(&name);
                self.known_node_tables.borrow_mut().insert(name.clone());
                self.node_table_columns.borrow_mut().insert(name, cols);
            }
        }
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
        self.node_table_columns
            .borrow_mut()
            .insert(table.to_string(), extra_cols.clone());
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
        if let Err(e) = self.execute("CHECKPOINT") {
            debug!("lbug_store: checkpoint on drop failed (engine will retry): {e}");
        }
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// The system configuration used for every open.
///
/// `auto_checkpoint(true)` lets LadybugDB bound the WAL on its own as a safety
/// net; the store additionally checkpoints every [`AUTO_CHECKPOINT_WRITES`]
/// writes (see [`LbugGraphStore::note_write_and_maybe_checkpoint`]).
/// `throw_on_wal_replay_failure` selects strict vs. resilient WAL replay: strict
/// (`true`) errors on a corrupt WAL, resilient (`false`) replays the good prefix
/// and ignores the unreplayable tail.
fn system_config(throw_on_wal_replay_failure: bool) -> SystemConfig {
    SystemConfig::default()
        // Cap the mmap reservation so we don't request a huge address space in
        // constrained environments (mirrors the Kùzu backend).
        .max_db_size(1 << 30)
        .buffer_pool_size(128 * 1024 * 1024)
        .auto_checkpoint(true)
        .throw_on_wal_replay_failure(throw_on_wal_replay_failure)
}

/// Open a [`Database`] with the given replay strictness, mapping the engine
/// error to a `String`.
fn open_database(db_path: &Path, strict: bool) -> std::result::Result<Database, String> {
    Database::new(db_path, system_config(strict)).map_err(|e| e.to_string())
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
