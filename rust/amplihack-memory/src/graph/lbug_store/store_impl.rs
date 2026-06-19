//! `GraphStore` trait implementation for [`LbugGraphStore`].

use std::collections::{HashMap, HashSet};

use lbug::Value;
use tracing::warn;

use super::{
    escape_cypher, is_valid_identifier, validate_identifier, value_to_string, LbugGraphStore,
};
use crate::graph::protocol::GraphStore;
use crate::graph::traversal::bfs_traverse;
use crate::graph::types::{Direction, GraphEdge, GraphNode, TraversalResult};
use crate::MemoryError;

/// `LIMIT k` clause, or empty when `limit` is effectively unbounded
/// (`usize::MAX` is what `CognitiveMemory` passes to mean "all rows", and it
/// overflows LadybugDB's signed `LIMIT`).
fn limit_clause(limit: usize) -> String {
    if limit >= i64::MAX as usize {
        String::new()
    } else {
        format!(" LIMIT {limit}")
    }
}

/// Append `n.key = 'value'` equality predicates to `where_parts` for each
/// supplied filter. Returns `false` if any key is not a safe identifier, in
/// which case the caller should yield no rows rather than risk injection.
fn append_equality_filters(
    where_parts: &mut Vec<String>,
    filters: Option<&HashMap<String, String>>,
) -> bool {
    let Some(filters) = filters else {
        return true;
    };
    for (k, v) in filters {
        if !is_valid_identifier(k) {
            return false;
        }
        where_parts.push(format!("n.{k} = '{}'", escape_cypher(v)));
    }
    true
}

impl LbugGraphStore {
    /// Build a [`GraphNode`] from an lbug node value.
    ///
    /// `node_id` is retained in `properties` (matching the in-memory backend, so
    /// the cognitive-memory converters keep working) while `graph_origin` is
    /// lifted into the dedicated field and Kùzu-internal keys (`_id`, `_label`)
    /// are dropped.
    fn node_val_to_graph_node(&self, node: &lbug::NodeVal, fallback_type: &str) -> GraphNode {
        let label = node.get_label_name();
        let node_type = if label.is_empty() {
            fallback_type.to_string()
        } else {
            label.clone()
        };

        let mut properties = HashMap::new();
        let mut node_id = String::new();
        let mut graph_origin = String::new();

        for (k, v) in node.get_properties() {
            if k.starts_with('_') {
                continue;
            }
            let val = value_to_string(v);
            if k == "graph_origin" {
                graph_origin = val;
                continue;
            }
            if k == "node_id" {
                node_id = val.clone();
            }
            properties.insert(k.clone(), val);
        }

        GraphNode {
            node_id,
            node_type,
            properties,
            graph_origin,
        }
    }

    fn rel_val_to_edge(
        &self,
        rel: &lbug::RelVal,
        rel_name: &str,
        anchor_id: &str,
        direction: &str,
        neighbor_id: &str,
    ) -> GraphEdge {
        let mut properties = HashMap::new();
        let mut edge_id = String::new();
        let mut graph_origin = String::new();

        for (k, v) in rel.get_properties() {
            if k.starts_with('_') {
                continue;
            }
            let val = value_to_string(v);
            match k.as_str() {
                "graph_origin" => graph_origin = val,
                "edge_id" => edge_id = val,
                _ => {
                    properties.insert(k.clone(), val);
                }
            }
        }

        let (source_id, target_id) = if direction == "outgoing" {
            (anchor_id.to_string(), neighbor_id.to_string())
        } else {
            (neighbor_id.to_string(), anchor_id.to_string())
        };

        GraphEdge {
            edge_id,
            source_id,
            target_id,
            edge_type: rel_name.to_string(),
            properties,
            graph_origin,
        }
    }

    fn collect_nodes(&self, rows: Vec<Vec<Value>>, fallback_type: &str) -> Vec<GraphNode> {
        rows.into_iter()
            .filter_map(|row| match row.into_iter().next() {
                Some(Value::Node(nv)) => Some(self.node_val_to_graph_node(&nv, fallback_type)),
                _ => None,
            })
            .collect()
    }

    /// Run `MATCH (n:node_type) [WHERE ...] RETURN n [LIMIT k]` and collect the
    /// nodes. A missing table (fresh/never-created type) simply yields no rows.
    fn match_return_nodes(
        &self,
        node_type: &str,
        where_parts: &[String],
        limit: usize,
    ) -> Vec<GraphNode> {
        let where_clause = if where_parts.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", where_parts.join(" AND "))
        };
        let cypher = format!(
            "MATCH (n:{node_type}){where_clause} RETURN n{}",
            limit_clause(limit)
        );
        match self.query_rows(&cypher) {
            Ok(rows) => self.collect_nodes(rows, node_type),
            Err(_) => Vec::new(),
        }
    }

    fn get_node_from_table(&self, node_id: &str, table: &str) -> Option<GraphNode> {
        if !is_valid_identifier(table) {
            return None;
        }
        let cypher = format!(
            "MATCH (n:{table}) WHERE n.node_id = '{}' RETURN n LIMIT 1",
            escape_cypher(node_id)
        );
        let rows = self.query_rows(&cypher).ok()?;
        self.collect_nodes(rows, table).into_iter().next()
    }

    /// Resolve the node table for `node_id` via the id cache, falling back to a
    /// full [`get_node`](GraphStore::get_node) scan.
    fn resolve_table(&self, node_id: &str) -> Option<String> {
        if let Some(t) = self.id_table_cache.borrow().get(node_id).cloned() {
            return Some(t);
        }
        self.get_node(node_id).map(|n| n.node_type)
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
        let lc = limit_clause(limit);
        let escaped = escape_cypher(node_id);
        let cypher = if direction == "outgoing" {
            format!(
                "MATCH (a:{node_table})-[r:{rel_name}]->(b:{neighbor_table}) \
                 WHERE a.node_id = '{escaped}' RETURN r, b{lc}"
            )
        } else {
            format!(
                "MATCH (a:{neighbor_table})-[r:{rel_name}]->(b:{node_table}) \
                 WHERE b.node_id = '{escaped}' RETURN r, a{lc}"
            )
        };

        let rows = match self.query_rows(&cypher) {
            Ok(r) => r,
            Err(e) => {
                warn!("query_directed_neighbors failed: {e}");
                return Vec::new();
            }
        };

        let mut pairs = Vec::new();
        for row in rows {
            let mut it = row.into_iter();
            let rel_v = it.next();
            let node_v = it.next();
            let neighbor = match node_v {
                Some(Value::Node(nv)) => self.node_val_to_graph_node(&nv, neighbor_table),
                _ => continue,
            };
            let edge = match rel_v {
                Some(Value::Rel(rv)) => {
                    self.rel_val_to_edge(&rv, rel_name, node_id, direction, &neighbor.node_id)
                }
                _ => continue,
            };
            pairs.push((edge, neighbor));
        }
        pairs
    }
}

impl GraphStore for LbugGraphStore {
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

        let extra: HashSet<String> = properties
            .keys()
            .filter(|k| k.as_str() != "node_id" && k.as_str() != "graph_origin")
            .cloned()
            .collect();

        let _guard = self.acquire_lock();
        self.ensure_node_table(node_type, &extra)?;

        let mut parts = vec![
            format!("node_id: '{}'", escape_cypher(&nid)),
            format!("graph_origin: '{}'", escape_cypher(&self.store_id)),
        ];
        for (k, v) in &properties {
            if k == "node_id" || k == "graph_origin" {
                continue;
            }
            parts.push(format!("{k}: '{}'", escape_cypher(v)));
        }
        let cypher = format!("CREATE (:{node_type} {{{}}})", parts.join(", "));
        self.execute(&cypher)?;
        self.post_write_barrier()?;

        self.id_table_cache
            .borrow_mut()
            .insert(nid.clone(), node_type.to_string());

        let mut props = properties;
        props
            .entry("node_id".to_string())
            .or_insert_with(|| nid.clone());

        Ok(GraphNode {
            node_id: nid,
            node_type: node_type.to_string(),
            properties: props,
            graph_origin: self.store_id.clone(),
        })
    }

    fn get_node(&self, node_id: &str) -> Option<GraphNode> {
        self.ensure_schema_loaded();
        let _guard = self.acquire_lock();

        if let Some(t) = self.id_table_cache.borrow().get(node_id).cloned() {
            if let Some(n) = self.get_node_from_table(node_id, &t) {
                return Some(n);
            }
        }

        let tables: Vec<String> = self.known_node_tables.borrow().iter().cloned().collect();
        for t in tables {
            if let Some(n) = self.get_node_from_table(node_id, &t) {
                self.id_table_cache
                    .borrow_mut()
                    .insert(node_id.to_string(), t);
                return Some(n);
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
        if !is_valid_identifier(node_type) {
            return Vec::new();
        }
        let _guard = self.acquire_lock();

        let mut where_parts: Vec<String> = Vec::new();
        if !append_equality_filters(&mut where_parts, filters) {
            return Vec::new();
        }
        self.match_return_nodes(node_type, &where_parts, limit)
    }

    fn search_nodes(
        &self,
        node_type: &str,
        text_fields: &[String],
        query: &str,
        filters: Option<&HashMap<String, String>>,
        limit: usize,
    ) -> Vec<GraphNode> {
        if !is_valid_identifier(node_type) {
            return Vec::new();
        }
        for field in text_fields {
            if !is_valid_identifier(field) {
                return Vec::new();
            }
        }
        let _guard = self.acquire_lock();

        let mut where_parts: Vec<String> = Vec::new();
        if !text_fields.is_empty() {
            let qlower = escape_cypher(&query.to_lowercase());
            let clauses: Vec<String> = text_fields
                .iter()
                .map(|fld| format!("lower(n.{fld}) CONTAINS '{qlower}'"))
                .collect();
            where_parts.push(format!("({})", clauses.join(" OR ")));
        }
        if !append_equality_filters(&mut where_parts, filters) {
            return Vec::new();
        }
        self.match_return_nodes(node_type, &where_parts, limit)
    }

    fn update_node(&mut self, node_id: &str, properties: HashMap<String, String>) -> bool {
        for k in properties.keys() {
            if !is_valid_identifier(k) {
                return false;
            }
        }

        let table = match self.resolve_table(node_id) {
            Some(t) => t,
            None => return false,
        };

        let extra: HashSet<String> = properties
            .keys()
            .filter(|k| k.as_str() != "node_id" && k.as_str() != "graph_origin")
            .cloned()
            .collect();

        let _guard = self.acquire_lock();
        if self.ensure_node_table(&table, &extra).is_err() {
            return false;
        }

        let set_parts: Vec<String> = properties
            .iter()
            .filter(|(k, _)| k.as_str() != "node_id")
            .map(|(k, v)| format!("n.{k} = '{}'", escape_cypher(v)))
            .collect();
        if set_parts.is_empty() {
            return true;
        }

        let cypher = format!(
            "MATCH (n:{table}) WHERE n.node_id = '{}' SET {}",
            escape_cypher(node_id),
            set_parts.join(", ")
        );
        if self.execute(&cypher).is_ok() {
            let _ = self.post_write_barrier();
            true
        } else {
            false
        }
    }

    fn delete_node(&mut self, node_id: &str) -> bool {
        let table = match self.resolve_table(node_id) {
            Some(t) => t,
            None => return false,
        };

        let _guard = self.acquire_lock();
        let cypher = format!(
            "MATCH (n:{table}) WHERE n.node_id = '{}' DETACH DELETE n",
            escape_cypher(node_id)
        );
        if self.execute(&cypher).is_ok() {
            self.id_table_cache.borrow_mut().remove(node_id);
            let _ = self.post_write_barrier();
            true
        } else {
            false
        }
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

        let src = self
            .get_node(source_id)
            .ok_or_else(|| MemoryError::Internal(format!("source node not found: {source_id}")))?;
        let tgt = self
            .get_node(target_id)
            .ok_or_else(|| MemoryError::Internal(format!("target node not found: {target_id}")))?;

        let eid = uuid::Uuid::new_v4().to_string();
        let extra: HashSet<String> = props
            .keys()
            .filter(|k| k.as_str() != "edge_id" && k.as_str() != "graph_origin")
            .cloned()
            .collect();

        let _guard = self.acquire_lock();
        self.ensure_rel_table(edge_type, &src.node_type, &tgt.node_type, &extra)?;

        let mut parts = vec![
            format!("edge_id: '{}'", escape_cypher(&eid)),
            format!("graph_origin: '{}'", escape_cypher(&self.store_id)),
        ];
        for (k, v) in &props {
            if k == "edge_id" || k == "graph_origin" {
                continue;
            }
            parts.push(format!("{k}: '{}'", escape_cypher(v)));
        }

        let cypher = format!(
            "MATCH (a:{}), (b:{}) WHERE a.node_id = '{}' AND b.node_id = '{}' \
             CREATE (a)-[:{edge_type} {{{}}}]->(b)",
            src.node_type,
            tgt.node_type,
            escape_cypher(source_id),
            escape_cypher(target_id),
            parts.join(", ")
        );
        self.execute(&cypher)?;
        self.post_write_barrier()?;

        Ok(GraphEdge {
            edge_id: eid,
            source_id: source_id.to_string(),
            target_id: target_id.to_string(),
            edge_type: edge_type.to_string(),
            properties: props,
            graph_origin: self.store_id.clone(),
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

        let _guard = self.acquire_lock();
        let rel_tables: Vec<(String, String, String)> = self
            .known_rel_tables
            .borrow()
            .iter()
            .filter(|(rel, _, _)| edge_type.is_none() || edge_type == Some(rel.as_str()))
            .cloned()
            .collect();

        let mut results = Vec::new();
        for (rel, from, to) in &rel_tables {
            if (direction == Direction::Outgoing || direction == Direction::Both)
                && from == &node.node_type
            {
                results.extend(self.query_directed_neighbors(
                    node_id,
                    &node.node_type,
                    to,
                    rel,
                    "outgoing",
                    limit,
                ));
            }
            if (direction == Direction::Incoming || direction == Direction::Both)
                && to == &node.node_type
            {
                results.extend(self.query_directed_neighbors(
                    node_id,
                    &node.node_type,
                    from,
                    rel,
                    "incoming",
                    limit,
                ));
            }
        }

        if limit < results.len() {
            results.truncate(limit);
        }
        results
    }

    fn delete_edge(&mut self, source_id: &str, target_id: &str, edge_type: &str) -> bool {
        if !is_valid_identifier(edge_type) {
            return false;
        }

        let src = match self.get_node(source_id) {
            Some(n) => n,
            None => return false,
        };
        let tgt = match self.get_node(target_id) {
            Some(n) => n,
            None => return false,
        };

        let _guard = self.acquire_lock();
        let cypher = format!(
            "MATCH (a:{})-[r:{edge_type}]->(b:{}) \
             WHERE a.node_id = '{}' AND b.node_id = '{}' DELETE r",
            src.node_type,
            tgt.node_type,
            escape_cypher(source_id),
            escape_cypher(target_id)
        );
        if self.execute(&cypher).is_ok() {
            let _ = self.post_write_barrier();
            true
        } else {
            false
        }
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
        // Best-effort: flush the WAL into the main DB file so a subsequent open
        // (after this handle is dropped) sees all committed writes.
        let _ = self.execute("CHECKPOINT");
        self.id_table_cache.borrow_mut().clear();
        self.schema_loaded.set(false);
        self.known_node_tables.borrow_mut().clear();
        self.node_table_columns.borrow_mut().clear();
        self.known_rel_tables.borrow_mut().clear();
    }
}
