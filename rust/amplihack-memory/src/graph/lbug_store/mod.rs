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
//! * **Injection-safe** — table/column/edge identifiers are validated against
//!   `[A-Za-z_][A-Za-z0-9_]*`, all string values are escaped via `escape_cypher`,
//!   and `LIMIT` is a type-safe `usize`.

mod store_impl;

#[cfg(test)]
mod tests;

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use lbug::{Connection, Database, SystemConfig, Value};
use tracing::{debug, warn};

use crate::MemoryError;

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
}

// SAFETY: lbug::Database is internally synchronized (it declares Send + Sync).
// All access to the interior-mutable caches is serialized through `lock`, and
// the store is only ever used behind `&mut`/`&` from a single `CognitiveMemory`.
unsafe impl Send for LbugGraphStore {}

impl LbugGraphStore {
    /// Open (or create) a LadybugDB database at `db_path`.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Storage`] if the parent directory cannot be created
    /// or the database cannot be opened.
    pub fn open(db_path: &Path, store_id: Option<&str>) -> crate::Result<Self> {
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

        let id = store_id
            .map(String::from)
            .unwrap_or_else(|| format!("lbug-{}", &uuid::Uuid::new_v4().to_string()[..8]));

        let config = SystemConfig::default()
            // Cap the mmap reservation so we don't request a huge address space
            // in constrained environments (mirrors the Kùzu backend).
            .max_db_size(1 << 30)
            .buffer_pool_size(128 * 1024 * 1024);

        let db = Database::new(db_path, config).map_err(|e| {
            MemoryError::Storage(format!(
                "failed to open LadybugDB at {}: {e}",
                db_path.display()
            ))
        })?;

        Ok(Self {
            store_id: id,
            db_path: db_path.to_path_buf(),
            db,
            lock: Mutex::new(()),
            schema_loaded: Cell::new(false),
            known_node_tables: RefCell::new(HashSet::new()),
            node_table_columns: RefCell::new(HashMap::new()),
            known_rel_tables: RefCell::new(HashSet::new()),
            id_table_cache: RefCell::new(HashMap::new()),
        })
    }

    /// The on-disk database path.
    pub fn db_path(&self) -> &Path {
        &self.db_path
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

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

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
