//! FederatedGraphStore -- composites a local agent graph + hive graph.
//!
//! Writes go to the local graph. Reads fan out to local + hive and deduplicate.

use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

use super::protocol::GraphStore;
use super::traversal::bfs_traverse;
use super::types::{Direction, GraphEdge, GraphNode, TraversalResult};

/// A search result annotated with provenance and confidence metadata.
#[derive(Debug, Clone)]
pub struct AnnotatedResult {
    pub node: GraphNode,
    pub source: String,
    pub confirmation_count: i32,
    pub confidence: f64,
}

/// Container for results from a federated query across stores.
#[derive(Debug, Clone, Default)]
pub struct FederatedQueryResult {
    pub results: Vec<AnnotatedResult>,
    pub local_count: usize,
    pub hive_count: usize,
    pub expert_agents: Vec<String>,
}

fn content_hash(node: &GraphNode) -> String {
    let mut parts = vec![node.node_type.clone()];
    let mut keys: Vec<&String> = node.properties.keys().collect();
    keys.sort();
    for k in keys {
        parts.push(format!("{}={}", k, node.properties[k]));
    }
    let raw = parts.join("|");
    let mut hasher = Sha256::new();
    hasher.update(raw.as_bytes());
    let hash = hasher.finalize();
    hex::encode(&hash[..8])
}

fn deduplicate_nodes(nodes: Vec<GraphNode>) -> Vec<GraphNode> {
    let mut seen_ids: HashSet<String> = HashSet::new();
    let mut seen_hashes: HashSet<String> = HashSet::new();
    let mut unique = Vec::new();

    for node in nodes {
        if !node.node_id.is_empty() && seen_ids.contains(&node.node_id) {
            continue;
        }
        let h = content_hash(&node);
        if seen_hashes.contains(&h) {
            continue;
        }
        if !node.node_id.is_empty() {
            seen_ids.insert(node.node_id.clone());
        }
        seen_hashes.insert(h);
        unique.push(node);
    }
    unique
}

mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }
}

/// Composes a local agent graph + hive graph behind the GraphStore trait.
pub struct FederatedGraphStore {
    local: Box<dyn GraphStore>,
    hive: Box<dyn GraphStore>,
}

impl FederatedGraphStore {
    /// Create a federated store composing a `local` agent graph and a shared `hive` graph.
    pub fn new(local: Box<dyn GraphStore>, hive: Box<dyn GraphStore>) -> Self {
        Self { local, hive }
    }

    /// High-level federated query across local + hive stores.
    pub fn federated_query(
        &self,
        query: &str,
        node_type: &str,
        text_fields: Option<&[String]>,
        limit: usize,
    ) -> FederatedQueryResult {
        let search_fields: Vec<String> = text_fields
            .map(|tf| tf.to_vec())
            .unwrap_or_else(|| vec!["content".to_string()]);

        let local_results =
            safe_search(self.local.as_ref(), node_type, &search_fields, query, limit);
        let hive_results = safe_search(self.hive.as_ref(), node_type, &search_fields, query, limit);

        let expert_nodes =
            self.hive
                .search_nodes("HiveAgent", &["domain".to_string()], query, None, 10);
        let expert_agents: Vec<String> = expert_nodes.iter().map(|n| n.node_id.clone()).collect();

        let mut seen_hashes: HashSet<String> = HashSet::new();
        let mut annotated: Vec<AnnotatedResult> = Vec::new();

        for node in &local_results {
            let h = content_hash(node);
            if seen_hashes.insert(h) {
                annotated.push(AnnotatedResult {
                    node: node.clone(),
                    source: "local".into(),
                    confirmation_count: 0,
                    confidence: 1.0,
                });
            }
        }

        let mut hive_count = 0;
        for node in &hive_results {
            let h = content_hash(node);
            if seen_hashes.insert(h) {
                let source = if !node.graph_origin.is_empty()
                    && node.graph_origin != "__hive__"
                    && node.graph_origin != self.local.store_id()
                {
                    format!("peer:{}", node.graph_origin)
                } else {
                    "hive".into()
                };

                annotated.push(AnnotatedResult {
                    node: node.clone(),
                    source,
                    confirmation_count: 0,
                    confidence: 0.9,
                });
                hive_count += 1;
            }
        }

        FederatedQueryResult {
            results: annotated.into_iter().take(limit).collect(),
            local_count: local_results.len(),
            hive_count,
            expert_agents,
        }
    }

    /// Return a combined store identifier in the form `federated:<local>+<hive>`.
    pub fn store_id_combined(&self) -> String {
        format!(
            "federated:{}+{}",
            self.local.store_id(),
            self.hive.store_id()
        )
    }
}

impl GraphStore for FederatedGraphStore {
    fn store_id(&self) -> &str {
        // This is a simplification; the Python version builds a dynamic string
        // For trait compliance we return local store_id
        self.local.store_id()
    }

    fn add_node(
        &mut self,
        node_type: &str,
        properties: HashMap<String, String>,
        node_id: Option<&str>,
    ) -> crate::Result<GraphNode> {
        self.local.add_node(node_type, properties, node_id)
    }

    fn get_node(&self, node_id: &str) -> Option<GraphNode> {
        self.local
            .get_node(node_id)
            .or_else(|| self.hive.get_node(node_id))
    }

    fn query_nodes(
        &self,
        node_type: &str,
        filters: Option<&HashMap<String, String>>,
        limit: usize,
    ) -> Vec<GraphNode> {
        let local = self.local.query_nodes(node_type, filters, limit);
        let hive = self.hive.query_nodes(node_type, filters, limit);
        let merged: Vec<GraphNode> = local.into_iter().chain(hive).collect();
        deduplicate_nodes(merged).into_iter().take(limit).collect()
    }

    fn search_nodes(
        &self,
        node_type: &str,
        text_fields: &[String],
        query: &str,
        filters: Option<&HashMap<String, String>>,
        limit: usize,
    ) -> Vec<GraphNode> {
        let local = self
            .local
            .search_nodes(node_type, text_fields, query, filters, limit);
        let hive = self
            .hive
            .search_nodes(node_type, text_fields, query, filters, limit);
        let merged: Vec<GraphNode> = local.into_iter().chain(hive).collect();
        deduplicate_nodes(merged).into_iter().take(limit).collect()
    }

    fn update_node(&mut self, node_id: &str, properties: HashMap<String, String>) -> bool {
        self.local.update_node(node_id, properties)
    }

    fn delete_node(&mut self, node_id: &str) -> bool {
        self.local.delete_node(node_id)
    }

    fn add_edge(
        &mut self,
        source_id: &str,
        target_id: &str,
        edge_type: &str,
        properties: Option<HashMap<String, String>>,
    ) -> crate::Result<GraphEdge> {
        self.local
            .add_edge(source_id, target_id, edge_type, properties)
    }

    fn query_neighbors(
        &self,
        node_id: &str,
        edge_type: Option<&str>,
        direction: Direction,
        limit: usize,
    ) -> Vec<(GraphEdge, GraphNode)> {
        let local = self
            .local
            .query_neighbors(node_id, edge_type, direction, limit);
        let hive = self
            .hive
            .query_neighbors(node_id, edge_type, direction, limit);

        let mut seen: HashSet<String> = HashSet::new();
        let mut merged = Vec::new();
        for (edge, node) in local.into_iter().chain(hive) {
            let h = format!("{}:{}", edge.edge_type, content_hash(&node));
            if seen.insert(h) {
                merged.push((edge, node));
            }
        }
        merged.into_iter().take(limit).collect()
    }

    fn delete_edge(&mut self, source_id: &str, target_id: &str, edge_type: &str) -> bool {
        self.local.delete_edge(source_id, target_id, edge_type)
    }

    fn traverse(
        &self,
        start_id: &str,
        edge_types: Option<&[String]>,
        max_hops: usize,
        direction: Direction,
        node_filter: Option<&HashMap<String, String>>,
    ) -> TraversalResult {
        let start_node = self
            .local
            .get_node(start_id)
            .or_else(|| self.hive.get_node(start_id));

        let start_node = match start_node {
            Some(n) => n,
            None => return TraversalResult::default(),
        };

        bfs_traverse(
            start_node,
            edge_types,
            max_hops,
            node_filter,
            1000,
            |id, et, _depth| {
                let local = self.local.query_neighbors(id, et, direction, 50);
                let hive = self.hive.query_neighbors(id, et, direction, 50);

                // Deduplicate edges by (source_id, target_id, edge_type)
                let mut seen: HashSet<(String, String, String)> = HashSet::new();
                let mut deduped = Vec::new();
                for pair in local.into_iter().chain(hive) {
                    let key = (
                        pair.0.source_id.clone(),
                        pair.0.target_id.clone(),
                        pair.0.edge_type.clone(),
                    );
                    if seen.insert(key) {
                        deduped.push(pair);
                    }
                }
                deduped
            },
        )
    }

    fn close(&mut self) {
        self.local.close();
        self.hive.close();
    }
}

fn safe_search(
    store: &dyn GraphStore,
    node_type: &str,
    text_fields: &[String],
    query: &str,
    limit: usize,
) -> Vec<GraphNode> {
    store.search_nodes(node_type, text_fields, query, None, limit)
}

#[allow(dead_code)]
fn matches_filter(node: &GraphNode, node_filter: &HashMap<String, String>) -> bool {
    node_filter
        .iter()
        .all(|(k, v)| node.properties.get(k) == Some(v))
}

#[cfg(test)]
mod tests {
    use super::super::in_memory_store::InMemoryGraphStore;
    use super::*;

    #[test]
    fn test_federated_writes_to_local() {
        let local = Box::new(InMemoryGraphStore::new(Some("local")));
        let hive = Box::new(InMemoryGraphStore::new(Some("hive")));
        let mut fed = FederatedGraphStore::new(local, hive);

        fed.add_node("Fact", HashMap::new(), Some("f1")).unwrap();
        assert!(fed.get_node("f1").is_some());
    }

    #[test]
    fn test_federated_reads_from_both() {
        let mut local = InMemoryGraphStore::new(Some("local"));
        let mut hive = InMemoryGraphStore::new(Some("hive"));

        let mut props = HashMap::new();
        props.insert("content".into(), "local fact".into());
        local.add_node("Fact", props, Some("l1")).unwrap();

        let mut props = HashMap::new();
        props.insert("content".into(), "hive fact".into());
        hive.add_node("Fact", props, Some("h1")).unwrap();

        let fed = FederatedGraphStore::new(Box::new(local), Box::new(hive));
        let results = fed.query_nodes("Fact", None, 50);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_federated_query_deduplication() {
        let mut local = InMemoryGraphStore::new(Some("local"));
        let mut hive = InMemoryGraphStore::new(Some("hive"));

        // Same content in both stores — should deduplicate to one result.
        let mut props = HashMap::new();
        props.insert("content".into(), "shared fact".into());
        local.add_node("Fact", props.clone(), Some("dup1")).unwrap();
        hive.add_node("Fact", props, Some("dup2")).unwrap();

        let fed = FederatedGraphStore::new(Box::new(local), Box::new(hive));
        let result = fed.federated_query("shared", "Fact", None, 10);
        assert_eq!(
            result.results.len(),
            1,
            "duplicate content across stores should be deduplicated"
        );
    }

    #[test]
    fn test_federated_query_source_annotation() {
        let mut local = InMemoryGraphStore::new(Some("local"));
        let mut hive = InMemoryGraphStore::new(Some("__hive__"));

        let mut props = HashMap::new();
        props.insert("content".into(), "local only".into());
        local.add_node("Fact", props, Some("l1")).unwrap();

        let mut props = HashMap::new();
        props.insert("content".into(), "hive only".into());
        hive.add_node("Fact", props, Some("h1")).unwrap();

        let fed = FederatedGraphStore::new(Box::new(local), Box::new(hive));
        let result = fed.federated_query("only", "Fact", None, 10);

        assert_eq!(result.results.len(), 2);
        let sources: Vec<&str> = result.results.iter().map(|r| r.source.as_str()).collect();
        assert!(sources.contains(&"local"), "should have a local source");
        assert!(sources.contains(&"hive"), "should have a hive source");
    }

    #[test]
    fn test_federated_query_limit_enforcement() {
        let mut local = InMemoryGraphStore::new(Some("local"));
        let hive = InMemoryGraphStore::new(Some("hive"));

        for i in 0..5 {
            let mut props = HashMap::new();
            props.insert("content".into(), format!("fact number {i}"));
            local
                .add_node("Fact", props, Some(&format!("f{i}")))
                .unwrap();
        }

        let fed = FederatedGraphStore::new(Box::new(local), Box::new(hive));
        let result = fed.federated_query("fact", "Fact", None, 2);
        assert!(
            result.results.len() <= 2,
            "limit=2 but got {} results",
            result.results.len()
        );
    }

    #[test]
    fn test_federated_query_empty_query() {
        let local = InMemoryGraphStore::new(Some("local"));
        let hive = InMemoryGraphStore::new(Some("hive"));

        let fed = FederatedGraphStore::new(Box::new(local), Box::new(hive));
        let result = fed.federated_query("anything", "Fact", None, 10);
        assert!(result.results.is_empty());
        assert_eq!(result.local_count, 0);
        assert_eq!(result.hive_count, 0);
    }

    #[test]
    fn test_federated_query_matching_vs_non_matching() {
        let mut local = InMemoryGraphStore::new(Some("local"));
        let hive = InMemoryGraphStore::new(Some("hive"));

        let mut props = HashMap::new();
        props.insert("content".into(), "rust programming".into());
        local.add_node("Fact", props, Some("f1")).unwrap();

        let mut props = HashMap::new();
        props.insert("content".into(), "python scripting".into());
        local.add_node("Fact", props, Some("f2")).unwrap();

        let fed = FederatedGraphStore::new(Box::new(local), Box::new(hive));

        let matching = fed.federated_query("rust", "Fact", None, 10);
        assert_eq!(matching.results.len(), 1);
        assert_eq!(
            matching.results[0].node.properties.get("content").unwrap(),
            "rust programming"
        );

        let non_matching = fed.federated_query("golang", "Fact", None, 10);
        assert!(non_matching.results.is_empty());
    }
}
