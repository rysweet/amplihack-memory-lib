use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;
use pyo3::types::PyDict;
use tracing::warn;

use super::schema::{is_valid_identifier, validate_identifier};
use super::KuzuGraphStore;
use crate::graph::protocol::GraphStore;
use crate::graph::types::{GraphEdge, GraphNode};

/// Extract string properties from a Kuzu Python row/rel dict, skipping reserved keys.
fn extract_properties(data: &Bound<'_, PyAny>, skip: &HashSet<&str>) -> HashMap<String, String> {
    let mut properties = HashMap::new();
    if let Ok(items) = data.call_method0("items") {
        if let Ok(iter) = items.iter() {
            for item in iter.flatten() {
                if let (Ok(key), Ok(val)) = (
                    item.get_item(0).and_then(|k| k.extract::<String>()),
                    item.get_item(1),
                ) {
                    if !skip.contains(key.as_str()) {
                        let val_str = val.extract::<String>().unwrap_or_else(|_| format!("{val}"));
                        properties.insert(key, val_str);
                    }
                }
            }
        }
    }
    properties
}

impl KuzuGraphStore {
    /// Acquire the internal mutex, recovering from poisoning.
    pub(crate) fn acquire_lock(&self) -> std::sync::MutexGuard<'_, ()> {
        self.lock.lock().unwrap_or_else(|e| {
            warn!("mutex poisoned, recovering: {e}");
            e.into_inner()
        })
    }

    /// Convert a Python dict (Kuzu result row) to a GraphNode.
    pub(crate) fn row_to_node(
        row_data: &Bound<'_, PyAny>,
        table: &str,
    ) -> crate::Result<GraphNode> {
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

        let properties = extract_properties(row_data, &skip);

        Ok(GraphNode {
            node_id,
            node_type: table.to_string(),
            properties,
            graph_origin,
        })
    }

    /// Convert a Python relationship dict to a GraphEdge.
    pub(crate) fn rel_to_edge(
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

        let properties = extract_properties(rel_data, &skip);

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

    pub(crate) fn get_rel_tables_for(
        &self,
        edge_type: Option<&str>,
    ) -> Vec<(String, String, String)> {
        self.known_rel_tables
            .iter()
            .filter(|(rel_name, _, _)| edge_type.is_none() || edge_type == Some(rel_name.as_str()))
            .cloned()
            .collect()
    }

    pub(crate) fn get_node_from_table(&self, node_id: &str, table: &str) -> Option<GraphNode> {
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

    pub(crate) fn query_directed_neighbors(
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
                Err(e) => {
                    tracing::warn!("query_directed_neighbors failed: {e}");
                    return Vec::new();
                }
            };
            let result_ref = result.bind(py);

            let mut pairs = Vec::new();
            while result_ref
                .call_method0("has_next")
                .ok()
                .and_then(|v| v.extract::<bool>().ok())
                == Some(true)
            {
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

    pub(crate) fn collect_nodes(
        py: Python<'_>,
        result: &PyObject,
        node_type: &str,
    ) -> Vec<GraphNode> {
        let result_ref = result.bind(py);
        let mut nodes = Vec::new();
        while result_ref
            .call_method0("has_next")
            .ok()
            .and_then(|v| v.extract::<bool>().ok())
            == Some(true)
        {
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

    pub(crate) fn search_nodes_impl(
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

        let _guard = self.acquire_lock();

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

    pub(crate) fn add_edge_impl(
        &mut self,
        source_id: &str,
        target_id: &str,
        edge_type: &str,
        properties: Option<HashMap<String, String>>,
    ) -> crate::Result<GraphEdge> {
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

        let _guard = self.acquire_lock();

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
}
