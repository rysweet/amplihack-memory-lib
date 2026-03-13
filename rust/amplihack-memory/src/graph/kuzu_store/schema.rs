use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;
use tracing::warn;

use super::KuzuGraphStore;

pub(crate) fn is_valid_identifier(name: &str) -> bool {
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

pub(crate) fn validate_identifier(name: &str) -> crate::Result<()> {
    if !is_valid_identifier(name) {
        return Err(crate::MemoryError::InvalidInput(format!(
            "Invalid identifier: {name:?}. Must match [A-Za-z_][A-Za-z0-9_]*"
        )));
    }
    Ok(())
}

impl KuzuGraphStore {
    pub(crate) fn ensure_node_table(
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

    pub(crate) fn ensure_rel_table(
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
}
