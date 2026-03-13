//! KuzuGraphStore -- Kuzu graph database implementation of the GraphStore trait.
//!
//! Uses PyO3 to bridge to Kuzu's Python API since no native C headers are available.
//! Manages dynamic schema (node and rel tables created on demand), parameterized
//! Cypher queries, and BFS traversal.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::protocol::GraphStore;
use super::traversal::bfs_traverse;
use super::types::{Direction, GraphEdge, GraphNode, TraversalResult};

use tracing::warn;

fn is_valid_identifier(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    let mut chars = name.chars();
    let first = chars.next().unwrap();
    if !first.is_ascii_alphabetic() && first != '_' {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn validate_identifier(name: &str) -> crate::Result<()> {
    if !is_valid_identifier(name) {
        return Err(crate::MemoryError::InvalidInput(format!(
            "Invalid identifier: {name:?}. Must match [A-Za-z_][A-Za-z0-9_]*"
        )));
    }
    Ok(())
}

/// Kuzu graph database implementation of the GraphStore trait.
///
/// Bridges to the Python Kuzu API via PyO3. Supports dynamic schema:
/// node/rel tables are created on first use. All Cypher queries use
/// parameterized bindings.
pub struct KuzuGraphStore {
    store_id: String,
    db_path: PathBuf,
    #[allow(dead_code)]
    py_db: PyObject,
    py_conn: PyObject,
    lock: Mutex<()>,
    known_node_tables: HashSet<String>,
    node_table_columns: HashMap<String, HashSet<String>>,
    known_rel_tables: HashSet<(String, String, String)>,
    id_table_cache: RefCell<HashMap<String, String>>,
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

    /// Execute a Cypher query with a PyDict of parameters.
    fn execute_cypher(
        &self,
        py: Python<'_>,
        cypher: &str,
        params: &Bound<'_, PyDict>,
    ) -> crate::Result<PyObject> {
        let conn = self.py_conn.bind(py);
        conn.call_method("execute", (cypher, params), None)
            .map(|r| r.unbind())
            .map_err(|e| {
                crate::MemoryError::Storage(format!(
                    "Cypher execution failed: {e}\nQuery: {cypher}"
                ))
            })
    }

    /// Execute a Cypher query without parameters.
    fn execute_no_params(&self, py: Python<'_>, cypher: &str) -> crate::Result<PyObject> {
        let conn = self.py_conn.bind(py);
        conn.call_method1("execute", (cypher,))
            .map(|r| r.unbind())
            .map_err(|e| {
                crate::MemoryError::Storage(format!(
                    "Cypher execution failed: {e}\nQuery: {cypher}"
                ))
            })
    }

    fn ensure_node_table(
        &mut self,
        table_name: &str,
        columns: Option<&HashMap<String, String>>,
    ) -> crate::Result<()> {
        validate_identifier(table_name)?;

        let extra_cols: HashMap<&String, &String> = columns
            .map(|c| {
                c.iter()
                    .filter(|(k, _)| k.as_str() != "node_id" && k.as_str() != "graph_origin")
                    .collect()
            })
            .unwrap_or_default();

        for col_name in extra_cols.keys() {
            validate_identifier(col_name)?;
        }

        if self.known_node_tables.contains(table_name) {
            let known = self
                .node_table_columns
                .entry(table_name.to_string())
                .or_default();
            let new_cols: Vec<(String, String)> = extra_cols
                .iter()
                .filter(|(k, _)| !known.contains(k.as_str()))
                .map(|(k, v)| ((*k).clone(), (*v).clone()))
                .collect();

            if !new_cols.is_empty() {
                Python::with_gil(|py| -> crate::Result<()> {
                    for (col_name, col_type) in &new_cols {
                        let ddl = format!(
                            "ALTER TABLE {table_name} ADD {col_name} {col_type} DEFAULT ''"
                        );
                        if let Err(e) = self.execute_no_params(py, &ddl) {
                            warn!("ensure_node_table: failed to add column {col_name} to {table_name}: {e}");
                        }
                    }
                    Ok(())
                })?;

                let known = self
                    .node_table_columns
                    .entry(table_name.to_string())
                    .or_default();
                for (col_name, _) in new_cols {
                    known.insert(col_name);
                }
            }
            return Ok(());
        }

        let mut col_defs = vec![
            "node_id STRING".to_string(),
            "graph_origin STRING".to_string(),
        ];
        for (col_name, col_type) in &extra_cols {
            col_defs.push(format!("{col_name} {col_type}"));
        }

        let col_defs_str = col_defs.join(", ");
        let ddl = format!(
            "CREATE NODE TABLE IF NOT EXISTS {table_name}({col_defs_str}, PRIMARY KEY(node_id))"
        );

        Python::with_gil(|py| self.execute_no_params(py, &ddl))?;

        self.known_node_tables.insert(table_name.to_string());
        let col_set: HashSet<String> = extra_cols.keys().map(|k| (*k).clone()).collect();
        self.node_table_columns
            .insert(table_name.to_string(), col_set);
        Ok(())
    }

    fn ensure_rel_table(
        &mut self,
        table_name: &str,
        from_table: &str,
        to_table: &str,
        columns: Option<&HashMap<String, String>>,
    ) -> crate::Result<()> {
        validate_identifier(table_name)?;
        validate_identifier(from_table)?;
        validate_identifier(to_table)?;

        let key = (
            table_name.to_string(),
            from_table.to_string(),
            to_table.to_string(),
        );
        if self.known_rel_tables.contains(&key) {
            return Ok(());
        }

        if !self.known_node_tables.contains(from_table) {
            self.ensure_node_table(from_table, None)?;
        }
        if !self.known_node_tables.contains(to_table) {
            self.ensure_node_table(to_table, None)?;
        }

        let mut col_defs = vec![
            "edge_id STRING".to_string(),
            "graph_origin STRING".to_string(),
        ];
        if let Some(cols) = columns {
            for (col_name, col_type) in cols {
                if col_name != "edge_id" && col_name != "graph_origin" {
                    validate_identifier(col_name)?;
                    col_defs.push(format!("{col_name} {col_type}"));
                }
            }
        }

        let col_defs_str = col_defs.join(", ");
        let ddl = format!(
            "CREATE REL TABLE IF NOT EXISTS {table_name}\
             (FROM {from_table} TO {to_table}, {col_defs_str})"
        );

        Python::with_gil(|py| self.execute_no_params(py, &ddl))?;
        self.known_rel_tables.insert(key);
        Ok(())
    }

    /// Convert a Python dict (Kuzu result row) to a GraphNode.
    fn row_to_node(row_data: &Bound<'_, PyAny>, table: &str) -> crate::Result<GraphNode> {
        let node_id: String = row_data
            .get_item("node_id")
            .ok()
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_default();

        let graph_origin: String = row_data
            .get_item("graph_origin")
            .ok()
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_default();

        let skip: HashSet<&str> = ["node_id", "graph_origin", "_id", "_label"]
            .iter()
            .copied()
            .collect();

        let mut properties = HashMap::new();
        if let Ok(items) = row_data.call_method0("items") {
            if let Ok(iter) = items.iter() {
                for item in iter.flatten() {
                    if let (Ok(key), Ok(val)) = (
                        item.get_item(0).and_then(|k| k.extract::<String>()),
                        item.get_item(1),
                    ) {
                        if !skip.contains(key.as_str()) {
                            let val_str =
                                val.extract::<String>().unwrap_or_else(|_| format!("{val}"));
                            properties.insert(key, val_str);
                        }
                    }
                }
            }
        }

        Ok(GraphNode {
            node_id,
            node_type: table.to_string(),
            properties,
            graph_origin,
        })
    }

    /// Convert a Python relationship dict to a GraphEdge.
    fn rel_to_edge(
        rel_data: &Bound<'_, PyAny>,
        rel_name: &str,
        anchor_id: &str,
        direction: &str,
        neighbor_id: &str,
    ) -> crate::Result<GraphEdge> {
        let edge_id: String = rel_data
            .get_item("edge_id")
            .ok()
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_default();

        let graph_origin: String = rel_data
            .get_item("graph_origin")
            .ok()
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_default();

        let skip: HashSet<&str> = ["edge_id", "graph_origin", "_id", "_label", "_src", "_dst"]
            .iter()
            .copied()
            .collect();

        let mut properties = HashMap::new();
        if let Ok(items) = rel_data.call_method0("items") {
            if let Ok(iter) = items.iter() {
                for item in iter.flatten() {
                    if let (Ok(key), Ok(val)) = (
                        item.get_item(0).and_then(|k| k.extract::<String>()),
                        item.get_item(1),
                    ) {
                        if !skip.contains(key.as_str()) {
                            let val_str =
                                val.extract::<String>().unwrap_or_else(|_| format!("{val}"));
                            properties.insert(key, val_str);
                        }
                    }
                }
            }
        }

        let (source_id, target_id) = if direction == "outgoing" {
            (anchor_id.to_string(), neighbor_id.to_string())
        } else {
            (neighbor_id.to_string(), anchor_id.to_string())
        };

        Ok(GraphEdge {
            edge_id,
            source_id,
            target_id,
            edge_type: rel_name.to_string(),
            properties,
            graph_origin,
        })
    }

    fn get_rel_tables_for(&self, edge_type: Option<&str>) -> Vec<(String, String, String)> {
        self.known_rel_tables
            .iter()
            .filter(|(rel_name, _, _)| edge_type.is_none() || edge_type == Some(rel_name.as_str()))
            .cloned()
            .collect()
    }

    fn get_node_from_table(&self, node_id: &str, table: &str) -> Option<GraphNode> {
        Python::with_gil(|py| {
            let cypher = format!("MATCH (n:{table}) WHERE n.node_id = $nid RETURN n");
            let params = PyDict::new_bound(py);
            params.set_item("nid", node_id).ok()?;

            let result = self.execute_cypher(py, &cypher, &params).ok()?;
            let result_ref = result.bind(py);

            let has_next: bool = result_ref.call_method0("has_next").ok()?.extract().ok()?;
            if !has_next {
                return None;
            }

            let row = result_ref.call_method0("get_next").ok()?;
            let row_data = row.get_item(0).ok()?;
            Self::row_to_node(&row_data, table).ok()
        })
    }

    fn query_directed_neighbors(
        &self,
        node_id: &str,
        node_table: &str,
        neighbor_table: &str,
        rel_name: &str,
        direction: &str,
        limit: usize,
    ) -> Vec<(GraphEdge, GraphNode)> {
        Python::with_gil(|py| {
            let params = PyDict::new_bound(py);
            if params.set_item("nid", node_id).is_err() {
                return Vec::new();
            }

            // SAFETY: `limit` is `usize`, so it cannot contain Cypher injection.
            // Kuzu does not support parameterized LIMIT, so interpolation is required.
            let cypher = if direction == "outgoing" {
                format!(
                    "MATCH (a:{node_table})-[r:{rel_name}]->(b:{neighbor_table}) \
                     WHERE a.node_id = $nid RETURN r, b LIMIT {limit}"
                )
            } else {
                format!(
                    "MATCH (a:{neighbor_table})-[r:{rel_name}]->(b:{node_table}) \
                     WHERE b.node_id = $nid RETURN r, a LIMIT {limit}"
                )
            };

            let result = match self.execute_cypher(py, &cypher, &params) {
                Ok(r) => r,
                Err(_) => return Vec::new(),
            };
            let result_ref = result.bind(py);

            let mut pairs = Vec::new();
            loop {
                let has_next: bool = match result_ref.call_method0("has_next") {
                    Ok(v) => match v.extract() {
                        Ok(b) => b,
                        Err(_) => break,
                    },
                    Err(_) => break,
                };
                if !has_next {
                    break;
                }

                let row = match result_ref.call_method0("get_next") {
                    Ok(r) => r,
                    Err(_) => break,
                };

                let rel_data = match row.get_item(0) {
                    Ok(d) => d,
                    Err(_) => continue,
                };
                let neighbor_data = match row.get_item(1) {
                    Ok(d) => d,
                    Err(_) => continue,
                };

                let neighbor = match Self::row_to_node(&neighbor_data, neighbor_table) {
                    Ok(n) => n,
                    Err(_) => continue,
                };
                let edge = match Self::rel_to_edge(
                    &rel_data,
                    rel_name,
                    node_id,
                    direction,
                    &neighbor.node_id,
                ) {
                    Ok(e) => e,
                    Err(_) => continue,
                };

                pairs.push((edge, neighbor));
            }
            pairs
        })
    }
}

impl GraphStore for KuzuGraphStore {
    fn store_id(&self) -> &str {
        &self.store_id
    }

    fn add_node(
        &mut self,
        node_type: &str,
        properties: HashMap<String, String>,
        node_id: Option<&str>,
    ) -> crate::Result<GraphNode> {
        validate_identifier(node_type)?;
        for k in properties.keys() {
            validate_identifier(k)?;
        }

        let nid = node_id
            .map(String::from)
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        let col_types: HashMap<String, String> = properties
            .keys()
            .map(|k| (k.clone(), "STRING".to_string()))
            .collect();
        self.ensure_node_table(node_type, Some(&col_types))?;

        let store_id = self.store_id.clone();

        Python::with_gil(|py| {
            let mut set_parts = vec![
                "node_id: $node_id".to_string(),
                "graph_origin: $graph_origin".to_string(),
            ];
            let params = PyDict::new_bound(py);
            params
                .set_item("node_id", &nid)
                .map_err(|e| crate::MemoryError::Storage(format!("param error: {e}")))?;
            params
                .set_item("graph_origin", &store_id)
                .map_err(|e| crate::MemoryError::Storage(format!("param error: {e}")))?;

            for (idx, (k, v)) in properties.iter().enumerate() {
                let pname = format!("p{idx}");
                set_parts.push(format!("{k}: ${pname}"));
                params
                    .set_item(&pname, v)
                    .map_err(|e| crate::MemoryError::Storage(format!("param error: {e}")))?;
            }

            let set_clause = set_parts.join(", ");
            let cypher = format!("CREATE (:{node_type} {{{set_clause}}})");
            self.execute_cypher(py, &cypher, &params)?;

            Ok(())
        })?;

        self.id_table_cache
            .borrow_mut()
            .insert(nid.clone(), node_type.to_string());

        Ok(GraphNode {
            node_id: nid,
            node_type: node_type.to_string(),
            properties,
            graph_origin: store_id,
        })
    }

    fn get_node(&self, node_id: &str) -> Option<GraphNode> {
        let _guard = self.lock.lock().unwrap_or_else(|e| {
            warn!("mutex poisoned, recovering: {e}");
            e.into_inner()
        });

        if let Some(cached_table) = self.id_table_cache.borrow().get(node_id).cloned() {
            if let Some(node) = self.get_node_from_table(node_id, &cached_table) {
                return Some(node);
            }
        }

        let cached = self.id_table_cache.borrow().get(node_id).cloned();
        for table in &self.known_node_tables {
            if cached.as_ref() == Some(table) {
                continue;
            }
            if let Some(node) = self.get_node_from_table(node_id, table) {
                self.id_table_cache
                    .borrow_mut()
                    .insert(node_id.to_string(), table.clone());
                return Some(node);
            }
        }

        None
    }

    fn query_nodes(
        &self,
        node_type: &str,
        filters: Option<&HashMap<String, String>>,
        limit: usize,
    ) -> Vec<GraphNode> {
        if !is_valid_identifier(node_type) || !self.known_node_tables.contains(node_type) {
            return Vec::new();
        }

        let _guard = self.lock.lock().unwrap_or_else(|e| {
            warn!("mutex poisoned, recovering: {e}");
            e.into_inner()
        });

        Python::with_gil(|py| {
            let mut where_parts: Vec<String> = Vec::new();
            let params = PyDict::new_bound(py);

            if let Some(f) = filters {
                for (idx, (k, v)) in f.iter().enumerate() {
                    if !is_valid_identifier(k) {
                        return Vec::new();
                    }
                    let pname = format!("f{idx}");
                    where_parts.push(format!("n.{k} = ${pname}"));
                    if params.set_item(&pname, v).is_err() {
                        return Vec::new();
                    }
                }
            }

            let where_clause = if where_parts.is_empty() {
                String::new()
            } else {
                format!(" WHERE {}", where_parts.join(" AND "))
            };
            // `limit: usize` is type-safe; no injection risk from interpolation.
            let cypher = format!("MATCH (n:{node_type}){where_clause} RETURN n LIMIT {limit}");

            let result = match self.execute_cypher(py, &cypher, &params) {
                Ok(r) => r,
                Err(_) => return Vec::new(),
            };
            Self::collect_nodes(py, &result, node_type)
        })
    }

    fn search_nodes(
        &self,
        node_type: &str,
        text_fields: &[String],
        query: &str,
        filters: Option<&HashMap<String, String>>,
        limit: usize,
    ) -> Vec<GraphNode> {
        if !is_valid_identifier(node_type) || !self.known_node_tables.contains(node_type) {
            return Vec::new();
        }
        for field in text_fields {
            if !is_valid_identifier(field) {
                return Vec::new();
            }
        }

        let _guard = self.lock.lock().unwrap_or_else(|e| {
            warn!("mutex poisoned, recovering: {e}");
            e.into_inner()
        });

        Python::with_gil(|py| {
            let mut where_parts: Vec<String> = Vec::new();
            let params = PyDict::new_bound(py);

            let text_clauses: Vec<String> = text_fields
                .iter()
                .map(|field| format!("lower(n.{field}) CONTAINS $query"))
                .collect();
            if !text_clauses.is_empty() {
                where_parts.push(format!("({})", text_clauses.join(" OR ")));
            }
            if params.set_item("query", query.to_lowercase()).is_err() {
                return Vec::new();
            }

            if let Some(f) = filters {
                for (idx, (k, v)) in f.iter().enumerate() {
                    if !is_valid_identifier(k) {
                        return Vec::new();
                    }
                    let pname = format!("f{idx}");
                    where_parts.push(format!("n.{k} = ${pname}"));
                    if params.set_item(&pname, v).is_err() {
                        return Vec::new();
                    }
                }
            }

            let where_clause = if where_parts.is_empty() {
                String::new()
            } else {
                format!(" WHERE {}", where_parts.join(" AND "))
            };
            // `limit: usize` is type-safe; no injection risk from interpolation.
            let cypher = format!("MATCH (n:{node_type}){where_clause} RETURN n LIMIT {limit}");

            let result = match self.execute_cypher(py, &cypher, &params) {
                Ok(r) => r,
                Err(_) => return Vec::new(),
            };
            Self::collect_nodes(py, &result, node_type)
        })
    }

    fn update_node(&mut self, node_id: &str, properties: HashMap<String, String>) -> bool {
        for k in properties.keys() {
            if !is_valid_identifier(k) {
                return false;
            }
        }

        let table = {
            let _guard = self.lock.lock().unwrap_or_else(|e| {
                warn!("mutex poisoned, recovering: {e}");
                e.into_inner()
            });
            self.id_table_cache.borrow().get(node_id).cloned()
        };

        let table = match table {
            Some(t) => t,
            None => match self.get_node(node_id) {
                Some(n) => n.node_type,
                None => return false,
            },
        };

        // Ensure any new property columns exist
        let col_types: HashMap<String, String> = properties
            .keys()
            .map(|k| (k.clone(), "STRING".to_string()))
            .collect();
        if self.ensure_node_table(&table, Some(&col_types)).is_err() {
            return false;
        }

        if properties.is_empty() {
            return true;
        }

        let _guard = self.lock.lock().unwrap_or_else(|e| {
            warn!("mutex poisoned, recovering: {e}");
            e.into_inner()
        });

        Python::with_gil(|py| {
            let mut set_parts: Vec<String> = Vec::new();
            let params = PyDict::new_bound(py);
            if params.set_item("nid", node_id).is_err() {
                return false;
            }

            for (idx, (k, v)) in properties.iter().enumerate() {
                let pname = format!("u{idx}");
                set_parts.push(format!("n.{k} = ${pname}"));
                if params.set_item(&pname, v).is_err() {
                    return false;
                }
            }

            let set_clause = set_parts.join(", ");
            let cypher = format!("MATCH (n:{table}) WHERE n.node_id = $nid SET {set_clause}");
            self.execute_cypher(py, &cypher, &params).is_ok()
        })
    }

    fn delete_node(&mut self, node_id: &str) -> bool {
        let table = {
            let _guard = self.lock.lock().unwrap_or_else(|e| {
                warn!("mutex poisoned, recovering: {e}");
                e.into_inner()
            });
            self.id_table_cache.borrow().get(node_id).cloned()
        };

        let table = match table {
            Some(t) => t,
            None => match self.get_node(node_id) {
                Some(n) => n.node_type,
                None => return false,
            },
        };

        let _guard = self.lock.lock().unwrap_or_else(|e| {
            warn!("mutex poisoned, recovering: {e}");
            e.into_inner()
        });

        let result = Python::with_gil(|py| {
            let cypher = format!("MATCH (n:{table}) WHERE n.node_id = $nid DETACH DELETE n");
            let params = PyDict::new_bound(py);
            if params.set_item("nid", node_id).is_err() {
                return false;
            }
            self.execute_cypher(py, &cypher, &params).is_ok()
        });

        if result {
            self.id_table_cache.borrow_mut().remove(node_id);
        }
        result
    }

    fn add_edge(
        &mut self,
        source_id: &str,
        target_id: &str,
        edge_type: &str,
        properties: Option<HashMap<String, String>>,
    ) -> crate::Result<GraphEdge> {
        validate_identifier(edge_type)?;
        let props = properties.unwrap_or_default();
        for k in props.keys() {
            validate_identifier(k)?;
        }

        let src_node = self.get_node(source_id).ok_or_else(|| {
            crate::MemoryError::Internal(format!("source node not found: {source_id}"))
        })?;
        let tgt_node = self.get_node(target_id).ok_or_else(|| {
            crate::MemoryError::Internal(format!("target node not found: {target_id}"))
        })?;

        let eid = uuid::Uuid::new_v4().to_string();

        let col_types: HashMap<String, String> = props
            .keys()
            .map(|k| (k.clone(), "STRING".to_string()))
            .collect();
        self.ensure_rel_table(
            edge_type,
            &src_node.node_type,
            &tgt_node.node_type,
            Some(&col_types),
        )?;

        let _guard = self.lock.lock().unwrap_or_else(|e| {
            warn!("mutex poisoned, recovering: {e}");
            e.into_inner()
        });

        Python::with_gil(|py| {
            let mut set_parts = vec![
                "edge_id: $eid".to_string(),
                "graph_origin: $graph_origin".to_string(),
            ];
            let params = PyDict::new_bound(py);
            params
                .set_item("sid", source_id)
                .map_err(|e| crate::MemoryError::Storage(format!("param error: {e}")))?;
            params
                .set_item("tid", target_id)
                .map_err(|e| crate::MemoryError::Storage(format!("param error: {e}")))?;
            params
                .set_item("eid", &eid)
                .map_err(|e| crate::MemoryError::Storage(format!("param error: {e}")))?;
            params
                .set_item("graph_origin", &self.store_id)
                .map_err(|e| crate::MemoryError::Storage(format!("param error: {e}")))?;

            for (idx, (k, v)) in props.iter().enumerate() {
                let pname = format!("ep{idx}");
                set_parts.push(format!("{k}: ${pname}"));
                params
                    .set_item(&pname, v)
                    .map_err(|e| crate::MemoryError::Storage(format!("param error: {e}")))?;
            }

            let set_clause = set_parts.join(", ");
            let src_type = &src_node.node_type;
            let tgt_type = &tgt_node.node_type;
            let cypher = format!(
                "MATCH (a:{src_type}), (b:{tgt_type}) \
                 WHERE a.node_id = $sid AND b.node_id = $tid \
                 CREATE (a)-[:{edge_type} {{{set_clause}}}]->(b)"
            );
            self.execute_cypher(py, &cypher, &params)?;

            Ok(GraphEdge {
                edge_id: eid,
                source_id: source_id.to_string(),
                target_id: target_id.to_string(),
                edge_type: edge_type.to_string(),
                properties: props,
                graph_origin: self.store_id.clone(),
            })
        })
    }

    fn query_neighbors(
        &self,
        node_id: &str,
        edge_type: Option<&str>,
        direction: Direction,
        limit: usize,
    ) -> Vec<(GraphEdge, GraphNode)> {
        if let Some(et) = edge_type {
            if !is_valid_identifier(et) {
                return Vec::new();
            }
        }

        let node = match self.get_node(node_id) {
            Some(n) => n,
            None => return Vec::new(),
        };

        let _guard = self.lock.lock().unwrap_or_else(|e| {
            warn!("mutex poisoned, recovering: {e}");
            e.into_inner()
        });

        let rel_tables = self.get_rel_tables_for(edge_type);
        let mut results = Vec::new();

        for (rel_name, from_table, to_table) in &rel_tables {
            if (direction == Direction::Outgoing || direction == Direction::Both)
                && from_table == &node.node_type
            {
                results.extend(self.query_directed_neighbors(
                    node_id,
                    &node.node_type,
                    to_table,
                    rel_name,
                    "outgoing",
                    limit,
                ));
            }

            if (direction == Direction::Incoming || direction == Direction::Both)
                && to_table == &node.node_type
            {
                results.extend(self.query_directed_neighbors(
                    node_id,
                    &node.node_type,
                    from_table,
                    rel_name,
                    "incoming",
                    limit,
                ));
            }
        }

        results.truncate(limit);
        results
    }

    fn delete_edge(&mut self, source_id: &str, target_id: &str, edge_type: &str) -> bool {
        if !is_valid_identifier(edge_type) {
            return false;
        }

        let src_node = match self.get_node(source_id) {
            Some(n) => n,
            None => return false,
        };
        let tgt_node = match self.get_node(target_id) {
            Some(n) => n,
            None => return false,
        };

        let _guard = self.lock.lock().unwrap_or_else(|e| {
            warn!("mutex poisoned, recovering: {e}");
            e.into_inner()
        });

        let src_type = &src_node.node_type;
        let tgt_type = &tgt_node.node_type;

        Python::with_gil(|py| {
            let cypher = format!(
                "MATCH (a:{src_type})-[r:{edge_type}]->(b:{tgt_type}) \
                 WHERE a.node_id = $sid AND b.node_id = $tid DELETE r"
            );
            let params = PyDict::new_bound(py);
            if params.set_item("sid", source_id).is_err()
                || params.set_item("tid", target_id).is_err()
            {
                return false;
            }
            self.execute_cypher(py, &cypher, &params).is_ok()
        })
    }

    fn traverse(
        &self,
        start_id: &str,
        edge_types: Option<&[String]>,
        max_hops: usize,
        direction: Direction,
        node_filter: Option<&HashMap<String, String>>,
    ) -> TraversalResult {
        let start_node = match self.get_node(start_id) {
            Some(n) => n,
            None => return TraversalResult::default(),
        };

        bfs_traverse(
            start_node,
            edge_types,
            max_hops,
            node_filter,
            1000,
            |id, et, _depth| self.query_neighbors(id, et, direction, 50),
        )
    }

    fn close(&mut self) {
        let _guard = self.lock.lock().unwrap_or_else(|e| {
            warn!("mutex poisoned, recovering: {e}");
            e.into_inner()
        });
        self.id_table_cache.borrow_mut().clear();
        self.known_node_tables.clear();
        self.known_rel_tables.clear();
        self.node_table_columns.clear();
    }
}

// Helper to collect nodes from a query result (avoids duplication).
impl KuzuGraphStore {
    fn collect_nodes(py: Python<'_>, result: &PyObject, node_type: &str) -> Vec<GraphNode> {
        let result_ref = result.bind(py);
        let mut nodes = Vec::new();
        loop {
            let has_next: bool = match result_ref.call_method0("has_next") {
                Ok(v) => match v.extract() {
                    Ok(b) => b,
                    Err(_) => break,
                },
                Err(_) => break,
            };
            if !has_next {
                break;
            }
            let row = match result_ref.call_method0("get_next") {
                Ok(r) => r,
                Err(_) => break,
            };
            if let Ok(row_data) = row.get_item(0) {
                if let Ok(node) = Self::row_to_node(&row_data, node_type) {
                    nodes.push(node);
                }
            }
        }
        nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store() -> (KuzuGraphStore, tempfile::TempDir) {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("test_kuzu_db");
        let store = KuzuGraphStore::new(&db_path, Some("test"), None).unwrap();
        (store, tmp)
    }

    #[test]
    fn test_add_and_get_node() {
        let (mut store, _tmp) = make_store();
        let mut props = HashMap::new();
        props.insert("name".into(), "Alice".into());
        let node = store.add_node("Person", props, Some("p1")).unwrap();
        assert_eq!(node.node_id, "p1");
        assert_eq!(node.node_type, "Person");

        let fetched = store.get_node("p1").unwrap();
        assert_eq!(fetched.properties.get("name").unwrap(), "Alice");
        assert_eq!(fetched.node_type, "Person");
    }

    #[test]
    fn test_add_node_auto_id() {
        let (mut store, _tmp) = make_store();
        let node = store.add_node("Agent", HashMap::new(), None).unwrap();
        assert!(!node.node_id.is_empty());
    }

    #[test]
    fn test_query_nodes() {
        let (mut store, _tmp) = make_store();
        let mut props1 = HashMap::new();
        props1.insert("role".into(), "developer".into());
        store.add_node("Person", props1, Some("q1")).unwrap();

        let mut props2 = HashMap::new();
        props2.insert("role".into(), "manager".into());
        store.add_node("Person", props2, Some("q2")).unwrap();

        let all = store.query_nodes("Person", None, 50);
        assert_eq!(all.len(), 2);

        let mut filter = HashMap::new();
        filter.insert("role".into(), "developer".into());
        let filtered = store.query_nodes("Person", Some(&filter), 50);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].node_id, "q1");
    }

    #[test]
    fn test_search_nodes() {
        let (mut store, _tmp) = make_store();
        let mut props = HashMap::new();
        props.insert("content".into(), "Rust programming language".into());
        store.add_node("Fact", props, Some("f1")).unwrap();

        let mut props2 = HashMap::new();
        props2.insert("content".into(), "Python scripting".into());
        store.add_node("Fact", props2, Some("f2")).unwrap();

        let results = store.search_nodes("Fact", &["content".to_string()], "rust", None, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, "f1");

        let results2 = store.search_nodes("Fact", &["content".to_string()], "PYTHON", None, 10);
        assert_eq!(results2.len(), 1);
        assert_eq!(results2[0].node_id, "f2");
    }

    #[test]
    fn test_update_node() {
        let (mut store, _tmp) = make_store();
        let mut props = HashMap::new();
        props.insert("name".into(), "Bob".into());
        store.add_node("Person", props, Some("u1")).unwrap();

        let mut updates = HashMap::new();
        updates.insert("name".into(), "Robert".into());
        assert!(store.update_node("u1", updates));

        let fetched = store.get_node("u1").unwrap();
        assert_eq!(fetched.properties.get("name").unwrap(), "Robert");
    }

    #[test]
    fn test_delete_node() {
        let (mut store, _tmp) = make_store();
        store
            .add_node("Person", HashMap::new(), Some("d1"))
            .unwrap();
        assert!(store.delete_node("d1"));
        assert!(store.get_node("d1").is_none());
        assert!(!store.delete_node("nonexistent"));
    }

    #[test]
    fn test_add_edge_and_neighbors() {
        let (mut store, _tmp) = make_store();
        store.add_node("Person", HashMap::new(), Some("a")).unwrap();
        store.add_node("Person", HashMap::new(), Some("b")).unwrap();
        let edge = store.add_edge("a", "b", "KNOWS", None).unwrap();
        assert_eq!(edge.source_id, "a");
        assert_eq!(edge.target_id, "b");
        assert_eq!(edge.edge_type, "KNOWS");

        let neighbors = store.query_neighbors("a", Some("KNOWS"), Direction::Outgoing, 10);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].1.node_id, "b");

        let incoming = store.query_neighbors("b", Some("KNOWS"), Direction::Incoming, 10);
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0].1.node_id, "a");
    }

    #[test]
    fn test_delete_edge() {
        let (mut store, _tmp) = make_store();
        store
            .add_node("Person", HashMap::new(), Some("e1"))
            .unwrap();
        store
            .add_node("Person", HashMap::new(), Some("e2"))
            .unwrap();
        store.add_edge("e1", "e2", "FOLLOWS", None).unwrap();

        assert!(store.delete_edge("e1", "e2", "FOLLOWS"));

        let neighbors = store.query_neighbors("e1", Some("FOLLOWS"), Direction::Outgoing, 10);
        assert!(neighbors.is_empty());
    }

    #[test]
    fn test_traverse() {
        let (mut store, _tmp) = make_store();
        store.add_node("N", HashMap::new(), Some("a")).unwrap();
        store.add_node("N", HashMap::new(), Some("b")).unwrap();
        store.add_node("N", HashMap::new(), Some("c")).unwrap();
        store.add_edge("a", "b", "LINK", None).unwrap();
        store.add_edge("b", "c", "LINK", None).unwrap();

        let result = store.traverse("a", None, 3, Direction::Outgoing, None);
        assert!(result.nodes.len() >= 2);
        assert!(!result.edges.is_empty());
    }

    #[test]
    fn test_edge_with_properties() {
        let (mut store, _tmp) = make_store();
        store
            .add_node("Person", HashMap::new(), Some("ep1"))
            .unwrap();
        store
            .add_node("Person", HashMap::new(), Some("ep2"))
            .unwrap();

        let mut edge_props = HashMap::new();
        edge_props.insert("weight".into(), "0.9".into());
        let edge = store
            .add_edge("ep1", "ep2", "RELATED", Some(edge_props))
            .unwrap();
        assert_eq!(edge.properties.get("weight").unwrap(), "0.9");
    }

    #[test]
    fn test_close() {
        let (mut store, _tmp) = make_store();
        store
            .add_node("Person", HashMap::new(), Some("cl1"))
            .unwrap();
        store.close();
    }

    #[test]
    fn test_invalid_identifier() {
        let (mut store, _tmp) = make_store();
        let result = store.add_node("invalid-type", HashMap::new(), Some("x"));
        assert!(result.is_err());
    }

    #[test]
    fn test_query_nonexistent_type() {
        let (store, _tmp) = make_store();
        let results = store.query_nodes("NonExistent", None, 50);
        assert!(results.is_empty());
    }
}
