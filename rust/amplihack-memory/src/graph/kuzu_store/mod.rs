//! KuzuGraphStore -- Kuzu graph database implementation of the GraphStore trait.
//!
//! Uses PyO3 to bridge to Kuzu's Python API since no native C headers are available.
//! Manages dynamic schema (node and rel tables created on demand), parameterized
//! Cypher queries, and BFS traversal.

mod connection;
mod helpers;
mod queries;
mod schema;

#[cfg(test)]
mod tests;

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Kuzu graph database implementation of the GraphStore trait.
///
/// Bridges to the Python Kuzu API via PyO3. Supports dynamic schema:
/// node/rel tables are created on first use. All Cypher queries use
/// parameterized bindings.
pub struct KuzuGraphStore {
    pub(crate) store_id: String,
    pub(crate) db_path: PathBuf,
    #[allow(dead_code)]
    pub(crate) py_db: PyObject,
    pub(crate) py_conn: PyObject,
    pub(crate) lock: Mutex<()>,
    pub(crate) known_node_tables: HashSet<String>,
    pub(crate) node_table_columns: HashMap<String, HashSet<String>>,
    pub(crate) known_rel_tables: HashSet<(String, String, String)>,
    pub(crate) id_table_cache: RefCell<HashMap<String, String>>,
}

impl KuzuGraphStore {
    /// Create a new KuzuGraphStore.
    pub fn new(
        db_path: &Path,
        store_id: Option<&str>,
        buffer_pool_size: Option<usize>,
    ) -> crate::Result<Self> {
        let id = store_id
            .map(String::from)
            .unwrap_or_else(|| format!("kuzu-{}", &uuid::Uuid::new_v4().to_string()[..8]));

        let pool_size = buffer_pool_size.unwrap_or(256 * 1024 * 1024);

        let (py_db, py_conn) = Python::with_gil(|py| -> crate::Result<(PyObject, PyObject)> {
            let kuzu = py.import_bound("kuzu").map_err(|e| {
                crate::MemoryError::Storage(format!("Failed to import kuzu Python module: {e}"))
            })?;

            let db_cls = kuzu.getattr("Database").map_err(|e| {
                crate::MemoryError::Storage(format!("Failed to get Database class: {e}"))
            })?;
            let kwargs = PyDict::new_bound(py);
            kwargs
                .set_item("buffer_pool_size", pool_size)
                .map_err(|e| crate::MemoryError::Storage(format!("param error: {e}")))?;
            // Limit max_db_size to avoid 8TB mmap in constrained environments
            kwargs
                .set_item("max_db_size", 1_073_741_824_u64) // 1GB
                .map_err(|e| crate::MemoryError::Storage(format!("param error: {e}")))?;
            let db = db_cls
                .call((db_path.to_string_lossy().to_string(),), Some(&kwargs))
                .map_err(|e| {
                    crate::MemoryError::Storage(format!("Failed to create Kuzu database: {e}"))
                })?;

            let conn = kuzu.call_method1("Connection", (&db,)).map_err(|e| {
                crate::MemoryError::Storage(format!("Failed to create Kuzu connection: {e}"))
            })?;

            Ok((db.unbind(), conn.unbind()))
        })?;

        Ok(Self {
            store_id: id,
            db_path: db_path.to_path_buf(),
            py_db,
            py_conn,
            lock: Mutex::new(()),
            known_node_tables: HashSet::new(),
            node_table_columns: HashMap::new(),
            known_rel_tables: HashSet::new(),
            id_table_cache: RefCell::new(HashMap::new()),
        })
    }

    /// Returns the database path.
    pub fn db_path(&self) -> &Path {
        &self.db_path
    }
}
