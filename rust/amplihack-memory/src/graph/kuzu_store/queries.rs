use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::schema::{is_valid_identifier, validate_identifier};
use super::KuzuGraphStore;
use crate::graph::protocol::GraphStore;
use crate::graph::traversal::bfs_traverse;
use crate::graph::types::{Direction, GraphEdge, GraphNode, TraversalResult};

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
        let _guard = self.acquire_lock();

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

        let _guard = self.acquire_lock();

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
        self.search_nodes_impl(node_type, text_fields, query, filters, limit)
    }

    fn update_node(&mut self, node_id: &str, properties: HashMap<String, String>) -> bool {
        for k in properties.keys() {
            if !is_valid_identifier(k) {
                return false;
            }
        }

        let table = {
            let _guard = self.acquire_lock();
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

        let _guard = self.acquire_lock();

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
            let _guard = self.acquire_lock();
            self.id_table_cache.borrow().get(node_id).cloned()
        };

        let table = match table {
            Some(t) => t,
            None => match self.get_node(node_id) {
                Some(n) => n.node_type,
                None => return false,
            },
        };

        let _guard = self.acquire_lock();

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
        self.add_edge_impl(source_id, target_id, edge_type, properties)
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

        let _guard = self.acquire_lock();

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

        let _guard = self.acquire_lock();

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
        let _guard = self.acquire_lock();
        self.id_table_cache.borrow_mut().clear();
        self.known_node_tables.clear();
        self.known_rel_tables.clear();
        self.node_table_columns.clear();
    }
}
