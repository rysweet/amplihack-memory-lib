//! `GraphStore` trait implementation for [`LbugGraphStore`].

use std::collections::{HashMap, HashSet};

use lbug::Value;
use tracing::warn;

use super::{
    escape_cypher, is_valid_identifier, not_deleted, validate_identifier, value_to_string,
    LbugGraphStore, DELETED_COL, DELETED_MARK,
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

    /// Run `MATCH (n:node_type) WHERE ... RETURN n [LIMIT k]` and collect the
    /// nodes. A missing table (fresh/never-created type) simply yields no rows.
    /// Soft-deleted (tombstoned, #100) nodes are always filtered out.
    ///
    /// Infallible: a backend read error is swallowed to an empty result (the
    /// historical read-path behaviour every consumer of the [`GraphStore`] read
    /// methods relies on). The fail-closed count path uses
    /// [`try_match_return_nodes`](Self::try_match_return_nodes) instead so it can
    /// distinguish a genuine read error from a genuinely-empty table
    /// (Simard #2561).
    fn match_return_nodes(
        &self,
        node_type: &str,
        where_parts: &[String],
        limit: usize,
    ) -> Vec<GraphNode> {
        self.try_match_return_nodes(node_type, where_parts, limit)
            .unwrap_or_default()
    }

    /// Fallible variant of [`match_return_nodes`](Self::match_return_nodes):
    /// **propagate** a backend query failure as `Err` instead of swallowing it
    /// to an empty result.
    ///
    /// This is the read primitive behind the fail-closed count
    /// ([`try_count_nodes`](GraphStore::try_count_nodes)) so a transient read
    /// error at startup is never mistaken for a genuinely-empty store (Simard
    /// #2561). Only the query itself is fallible here — an empty (but readable)
    /// table still yields `Ok(vec![])`.
    fn try_match_return_nodes(
        &self,
        node_type: &str,
        where_parts: &[String],
        limit: usize,
    ) -> crate::Result<Vec<GraphNode>> {
        // Test seam (#2561): deterministically simulate a transient backend read
        // failure so the fail-closed count path can be exercised without a
        // specific broken engine build. No-op outside tests.
        if super::force_read_error_for_test() {
            return Err(MemoryError::Storage(
                "forced read error (test seam)".to_string(),
            ));
        }
        let mut parts = where_parts.to_vec();
        // #107: only filter tombstones when the table actually carries the
        // `_deleted` column. A legacy table that predates soft-delete would make
        // lbug 0.17.1 binder-error on the missing property, and the swallowed
        // error would silently read the populated store back as empty.
        if let Some(filter) = self.tombstone_filter(node_type, "n") {
            parts.push(filter);
        }
        let where_clause = if parts.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", parts.join(" AND "))
        };
        let cypher = format!(
            "MATCH (n:{node_type}){where_clause} RETURN n{}",
            limit_clause(limit)
        );
        let rows = self.query_rows(&cypher)?;
        Ok(self.collect_nodes(rows, node_type))
    }

    fn get_node_from_table(&self, node_id: &str, table: &str) -> Option<GraphNode> {
        if !is_valid_identifier(table) {
            return None;
        }
        let mut where_parts = vec![format!("n.node_id = '{}'", escape_cypher(node_id))];
        if let Some(filter) = self.tombstone_filter(table, "n") {
            where_parts.push(filter);
        }
        let cypher = format!(
            "MATCH (n:{table}) WHERE {} RETURN n LIMIT 1",
            where_parts.join(" AND ")
        );
        let rows = self.query_rows(&cypher).ok()?;
        self.collect_nodes(rows, table).into_iter().next()
    }

    /// Does a row with `node_id` physically exist in `table`, *including*
    /// soft-deleted (tombstoned, #100) rows? Used by [`add_node`] to revive a
    /// tombstoned id rather than `CREATE` a duplicate primary key.
    fn physical_node_exists(&self, table: &str, node_id: &str) -> bool {
        if !is_valid_identifier(table) {
            return false;
        }
        let cypher = format!(
            "MATCH (n:{table}) WHERE n.node_id = '{}' RETURN n LIMIT 1",
            escape_cypher(node_id)
        );
        matches!(self.query_rows(&cypher), Ok(rows) if !rows.is_empty())
    }

    /// Resolve the node table for `node_id` via the id cache, falling back to a
    /// full [`get_node`](GraphStore::get_node) scan.
    fn resolve_table(&self, node_id: &str) -> Option<String> {
        if let Some(t) = self.id_table_cache.borrow().get(node_id).cloned() {
            return Some(t);
        }
        self.get_node(node_id).map(|n| n.node_type)
    }

    /// Distinct relationship-type names currently known to the catalog, sorted
    /// for a deterministic scan order.
    ///
    /// A rel name can appear in [`known_rel_tables`](LbugGraphStore::known_rel_tables)
    /// with several `(from, to)` endpoint pairs, but each edge lives in exactly
    /// one rel table, so deduplicating by name lets a typed fan-out
    /// (`MATCH (a)-[r:NAME]->(b)`) visit every edge exactly once without the
    /// double-counting a per-tuple iteration would cause.
    fn distinct_rel_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .known_rel_tables
            .borrow()
            .iter()
            .map(|(rel, _, _)| rel.clone())
            .collect();
        names.sort();
        names.dedup();
        names
    }

    /// Fetch neighbors of one rel type in a single direction with a *typed*
    /// `MATCH (a)-[r:rel_type]->(b)` scan.
    ///
    /// `rel_type` must already be a valid identifier (the caller checks);
    /// it is interpolated bare as a rel label. A typed scan touches a single
    /// rel table's CSR storage and never enters lbug's multi-rel-table scanner
    /// — see [`query_neighbors_directed`](Self::query_neighbors_directed).
    fn query_neighbors_one_type(
        &self,
        node_id: &str,
        rel_type: &str,
        direction: &str,
        limit: usize,
    ) -> Vec<(GraphEdge, GraphNode)> {
        let lc = limit_clause(limit);
        let escaped = escape_cypher(node_id);
        // #107: gate the labeled rel tombstone filter on the rel table actually
        // carrying `_deleted` (a legacy rel table without it would binder-error
        // under lbug 0.17.1 and silently read zero edges). The `a`/`b` endpoints
        // are unlabeled, which lbug binds leniently, so their filter stays.
        let r_filter = self.tombstone_filter(rel_type, "r");
        let cypher = if direction == "outgoing" {
            let mut wp = vec![format!("a.node_id = '{escaped}'")];
            if let Some(f) = &r_filter {
                wp.push(f.clone());
            }
            wp.push(not_deleted("b"));
            format!(
                "MATCH (a)-[r:{rel_type}]->(b) WHERE {} RETURN r, b{lc}",
                wp.join(" AND ")
            )
        } else {
            let mut wp = vec![format!("b.node_id = '{escaped}'")];
            if let Some(f) = &r_filter {
                wp.push(f.clone());
            }
            wp.push(not_deleted("a"));
            format!(
                "MATCH (a)-[r:{rel_type}]->(b) WHERE {} RETURN r, a{lc}",
                wp.join(" AND ")
            )
        };

        let rows = match self.query_rows(&cypher) {
            Ok(r) => r,
            Err(e) => {
                warn!("query_neighbors_one_type({rel_type}) failed: {e}");
                return Vec::new();
            }
        };

        let mut pairs = Vec::new();
        for row in rows {
            let mut it = row.into_iter();
            let rel_v = it.next();
            let node_v = it.next();
            let neighbor = match node_v {
                Some(Value::Node(nv)) => self.node_val_to_graph_node(&nv, ""),
                _ => continue,
            };
            let edge = match rel_v {
                // The rel type is fixed by the typed query, so take it from
                // `rel_type` rather than `RelVal::get_label_name` (which only
                // mattered for the old label-less multi-table scan).
                Some(Value::Rel(rv)) => {
                    self.rel_val_to_edge(&rv, rel_type, node_id, direction, &neighbor.node_id)
                }
                _ => continue,
            };
            pairs.push((edge, neighbor));
        }
        pairs
    }

    /// Fetch neighbors in a single direction.
    ///
    /// When `edge_type` is `Some(t)` a single typed scan is issued. When it is
    /// `None` we **fan out one typed scan per known rel type** and union the
    /// results, instead of issuing a single label-less `MATCH (a)-[r]->(b)`
    /// across every rel table at once.
    ///
    /// A label-less multi-rel-table scan is exactly what drives lbug's
    /// `RelTableCollectionScanner::scan` / `ScanMultiRelTable` /
    /// `CSRNodeGroup::scanCommittedInMemRandom` into a `getGroup(UINT32_MAX)`
    /// null-pointer dereference and SIGSEGVs the process (#100). Each typed
    /// scan touches a single rel table's CSR and avoids that scanner entirely.
    /// Because every edge lives in exactly one rel table, the per-type union
    /// returns each edge exactly once (matching the old single-query result
    /// set; ordering across rel types was never part of the contract).
    ///
    /// Defense-in-depth note (Step 3, #100): skipping node groups whose CSR
    /// index is the `UINT32_MAX` sentinel before dereferencing is internal to
    /// lbug's C++ engine and is not exposed to the Rust bindings, so it cannot
    /// be done here. Avoiding the multi-rel-table scanner via typed scans (this
    /// function) and keeping `known_rel_tables` the authoritative rel-type list
    /// is the feasible, durability-neutral mitigation.
    fn query_neighbors_directed(
        &self,
        node_id: &str,
        edge_type: Option<&str>,
        direction: &str,
        limit: usize,
    ) -> Vec<(GraphEdge, GraphNode)> {
        match edge_type {
            Some(et) => self.query_neighbors_one_type(node_id, et, direction, limit),
            None => {
                let mut pairs = Vec::new();
                for rel in self.distinct_rel_names() {
                    // Catalog names are always valid identifiers; skip anything
                    // else rather than interpolate an unsafe rel label.
                    if !is_valid_identifier(&rel) {
                        continue;
                    }
                    pairs.extend(self.query_neighbors_one_type(node_id, &rel, direction, limit));
                    if pairs.len() >= limit {
                        break;
                    }
                }
                if limit < pairs.len() {
                    pairs.truncate(limit);
                }
                pairs
            }
        }
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

        // If a row with this id physically exists in the table — including a
        // soft-deleted (tombstoned, #100) one left behind by `delete_node` —
        // a `CREATE` would violate `PRIMARY KEY(node_id)`. Revive/overwrite it
        // instead: clear the tombstone and refresh `graph_origin` + properties.
        // This is what makes the consolidation "delete then re-add the same id"
        // churn work once deletes no longer physically remove the row.
        let cypher = if self.physical_node_exists(node_type, &nid) {
            let mut sets = vec![
                format!("n.graph_origin = '{}'", escape_cypher(&self.store_id)),
                format!("n.{DELETED_COL} = ''"),
            ];
            for (k, v) in &properties {
                if k == "node_id" || k == "graph_origin" {
                    continue;
                }
                sets.push(format!("n.{k} = '{}'", escape_cypher(v)));
            }
            format!(
                "MATCH (n:{node_type}) WHERE n.node_id = '{}' SET {}",
                escape_cypher(&nid),
                sets.join(", ")
            )
        } else {
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
            format!("CREATE (:{node_type} {{{}}})", parts.join(", "))
        };
        self.execute(&cypher)?;
        self.post_write_barrier()?;
        self.note_write_and_maybe_checkpoint();

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

        // Fast path: we already know which table holds this id.
        if let Some(t) = self.id_table_cache.borrow().get(node_id).cloned() {
            if let Some(n) = self.get_node_from_table(node_id, &t) {
                return Some(n);
            }
        }

        // Cold path: one label-less lookup resolves the node (and its real
        // label) across every node table in a single query, instead of issuing
        // one query per table. An unlabeled MATCH is a binder error on an empty
        // catalog, so it is only attempted once a table exists; any other binder
        // error (e.g. a catalog table lacking `node_id`) falls through to the
        // per-table scan below.
        if !self.known_node_tables.borrow().is_empty() {
            let cypher = format!(
                "MATCH (n) WHERE n.node_id = '{}' AND {} RETURN n LIMIT 1",
                escape_cypher(node_id),
                not_deleted("n")
            );
            if let Ok(rows) = self.query_rows(&cypher) {
                let found = self.collect_nodes(rows, "").into_iter().next();
                if let Some(node) = found {
                    self.id_table_cache
                        .borrow_mut()
                        .insert(node_id.to_string(), node.node_type.clone());
                    return Some(node);
                }
                // The query succeeded and matched nothing: the id is absent from
                // every table, so the per-table scan would also find nothing.
                return None;
            }
        }

        // Fallback per-table scan (pathological catalogs the single query above
        // could not bind).
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
        // #107: populate the column cache from the on-disk catalog (and back-fill
        // a missing `_deleted` column) before the labeled, tombstone-filtered
        // read so a legacy store does not read back empty.
        self.ensure_schema_loaded();
        let _guard = self.acquire_lock();

        let mut where_parts: Vec<String> = Vec::new();
        if !append_equality_filters(&mut where_parts, filters) {
            return Vec::new();
        }
        self.match_return_nodes(node_type, &where_parts, limit)
    }

    fn try_count_nodes(
        &self,
        node_type: &str,
        filters: Option<&HashMap<String, String>>,
    ) -> crate::Result<usize> {
        // An invalid identifier can never name a real table, so it is a
        // confirmed zero for that type rather than a read error.
        if !is_valid_identifier(node_type) {
            return Ok(0);
        }

        // Fail closed on a suspected-empty store. A **sealed** store tripped the
        // #107 empty-read gate — a material on-disk footprint that read back
        // empty — so its "emptiness" is *suspected data loss*, not a confirmed
        // empty store. Reporting it as a trustworthy `0` would let Simard's
        // auto-restore (#2561) re-insert a snapshot over still-present but
        // transiently-unreadable data. Surface it as an error instead.
        if self.sealed.get() {
            return Err(MemoryError::Storage(
                "store sealed by the #107 empty-read safety gate; node count is not \
                 trustworthy (suspected data loss, not a confirmed-empty store)"
                    .to_string(),
            ));
        }

        // #107: populate the catalog cache before the labeled, tombstone-filtered
        // read. Use the *fallible* schema load so a catalog read failure is a
        // propagated error rather than a silent "loaded, no tables" (which would
        // masquerade as a confirmed-empty store below).
        self.try_ensure_schema_loaded()?;
        let _guard = self.acquire_lock();

        // A genuinely-absent node-type table is a confirmed zero for that type
        // (fresh/empty store), not a read error. `try_ensure_schema_loaded`
        // proved the catalog is readable, so this distinction is trustworthy.
        if !self.known_node_tables.borrow().contains(node_type) {
            return Ok(0);
        }

        let mut where_parts: Vec<String> = Vec::new();
        if !append_equality_filters(&mut where_parts, filters) {
            return Ok(0);
        }

        // Existing table + readable catalog: any failure now is a genuine read
        // error on a table that *should* be readable, so propagate it.
        self.try_match_return_nodes(node_type, &where_parts, usize::MAX)
            .map(|nodes| nodes.len())
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
        // #107: same as query_nodes — load the schema (and back-fill `_deleted`)
        // so the labeled, tombstone-filtered read does not silently read empty.
        self.ensure_schema_loaded();
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
            if let Err(e) = self.post_write_barrier() {
                warn!("update_node: durability barrier failed for {node_id}: {e}");
            }
            self.note_write_and_maybe_checkpoint();
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
        // The table label is interpolated bare (no quoting) into the Phase B
        // query, so it must be a safe identifier. Real tables always are, but
        // refuse rather than risk injection if a corrupt cache entry slips in.
        if !is_valid_identifier(&table) {
            return false;
        }

        // Populate `known_rel_tables` from the on-disk catalog so the guard
        // below is correct on a freshly reopened store. Does not take the lock.
        self.ensure_schema_loaded();

        let _guard = self.acquire_lock();
        let escaped = escape_cypher(node_id);

        // Phase A — soft-delete (tombstone) every incident relationship.
        //
        // The relationships are *not* physically removed. Issuing `DELETE r`
        // against a relationship that lives in a committed CSR rel group drives
        // lbug's CSR node-group index to the `UINT32_MAX` sentinel; the next
        // scan to touch that table then dereferences a null group and SEGVs the
        // process (`getGroup(UINT32_MAX)`, #100 — version-independent across
        // lbug 0.15.3/0.15.4/0.17.1). Instead each incident edge is marked
        // deleted with `SET r._deleted = '1'` (a property write that never
        // mutates the CSR adjacency structure), and every read filters
        // tombstoned rows out (`not_deleted`). One typed `SET` pass per known
        // rel type, both directions, covers outbound, inbound, self-loop, and
        // parallel edges; typed (not label-less) scans also avoid lbug's
        // `ScanMultiRelTable` path. With no rel tables the list is empty and we
        // fall through to tombstoning the node below.
        for rel in self.distinct_rel_names() {
            // The rel label is interpolated bare; catalog names are always
            // valid identifiers. Fail-closed (leave the node and its edges
            // fully intact) rather than risk injecting an unsafe label.
            if !is_valid_identifier(&rel) {
                warn!("delete_node: refusing non-identifier rel type {rel:?} for {node_id}");
                return false;
            }
            let outgoing = format!(
                "MATCH (a)-[r:{rel}]->(b) WHERE a.node_id = '{escaped}' SET r.{DELETED_COL} = '{DELETED_MARK}'"
            );
            let incoming = format!(
                "MATCH (a)-[r:{rel}]->(b) WHERE b.node_id = '{escaped}' SET r.{DELETED_COL} = '{DELETED_MARK}'"
            );
            for cypher in [&outgoing, &incoming] {
                if let Err(e) = self.execute(cypher) {
                    // Fail-closed: leave the node fully intact (and its routing
                    // cache entry) rather than half-delete its edges.
                    warn!("delete_node: incident-edge cleanup failed for {node_id} on {rel}: {e}");
                    return false;
                }
            }
        }

        // Phase B — soft-delete the node itself. A plain `DELETE n` is safe for
        // an edgeless node, but the node still physically owns its (now
        // tombstoned) relationships, so a physical delete would either fail or
        // need `DETACH` (which re-enters the crashing CSR path). Tombstoning the
        // node keeps it out of every read while leaving the CSR groups
        // untouched. `add_node` revives the row if the id is recreated.
        let cypher = format!(
            "MATCH (n:{table}) WHERE n.node_id = '{escaped}' SET n.{DELETED_COL} = '{DELETED_MARK}'"
        );
        if self.execute(&cypher).is_ok() {
            self.id_table_cache.borrow_mut().remove(node_id);
            if let Err(e) = self.post_write_barrier() {
                warn!("delete_node: durability barrier failed for {node_id}: {e}");
            }
            self.note_write_and_maybe_checkpoint();
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
        self.note_write_and_maybe_checkpoint();

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

        self.ensure_schema_loaded();
        let _guard = self.acquire_lock();

        // Early-out when there are no matching rel tables: an unknown
        // `edge_type` or an empty catalog yields no neighbors, and skipping the
        // scan avoids issuing queries against rel tables that do not exist.
        // (The neighbor fetch itself now uses typed per-rel-type scans, so the
        // crashing label-less multi-rel-table path is never taken — see
        // `query_neighbors_directed`.)
        let has_rel = {
            let rels = self.known_rel_tables.borrow();
            match edge_type {
                Some(et) => rels.iter().any(|(rel, _, _)| rel == et),
                None => !rels.is_empty(),
            }
        };
        if !has_rel {
            return Vec::new();
        }

        let mut results = Vec::new();
        if direction == Direction::Outgoing || direction == Direction::Both {
            results.extend(self.query_neighbors_directed(node_id, edge_type, "outgoing", limit));
        }
        if direction == Direction::Incoming || direction == Direction::Both {
            results.extend(self.query_neighbors_directed(node_id, edge_type, "incoming", limit));
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
        // Soft-delete (tombstone) the matching edge(s) rather than `DELETE r`:
        // a physical relationship delete against a committed CSR group corrupts
        // lbug's CSR node-group index and SEGVs the next scan
        // (`getGroup(UINT32_MAX)`, #100). Setting `r._deleted` is a property
        // write that leaves the CSR adjacency intact; reads filter it out.
        let cypher = format!(
            "MATCH (a:{})-[r:{edge_type}]->(b:{}) \
             WHERE a.node_id = '{}' AND b.node_id = '{}' AND {} \
             SET r.{DELETED_COL} = '{DELETED_MARK}'",
            src.node_type,
            tgt.node_type,
            escape_cypher(source_id),
            escape_cypher(target_id),
            not_deleted("r")
        );
        if self.execute(&cypher).is_ok() {
            if let Err(e) = self.post_write_barrier() {
                warn!("delete_edge: durability barrier failed for {source_id}->{target_id}: {e}");
            }
            self.note_write_and_maybe_checkpoint();
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

    fn checkpoint(&self) -> crate::Result<()> {
        let _guard = self.acquire_lock();
        self.do_checkpoint()
    }

    fn close(&mut self) {
        let _guard = self.acquire_lock();
        // Best-effort: flush the WAL into the main DB file so a subsequent open
        // (after this handle is dropped) sees all committed writes.
        if let Err(e) = self.do_checkpoint() {
            warn!("close: CHECKPOINT failed; relying on WAL flush at drop: {e}");
        }
        self.id_table_cache.borrow_mut().clear();
        self.schema_loaded.set(false);
        self.known_node_tables.borrow_mut().clear();
        self.node_table_columns.borrow_mut().clear();
        self.known_rel_tables.borrow_mut().clear();
    }
}
