//! Shared BFS traversal logic used by all graph store implementations.

use std::collections::{HashMap, HashSet, VecDeque};

use super::types::{GraphEdge, GraphNode, TraversalItem, TraversalResult};

/// Generic BFS traversal that delegates neighbor fetching to a caller-provided closure.
///
/// `get_neighbors` receives `(current_node_id, optional_edge_type, depth)` and returns
/// the `(edge, neighbor_node)` pairs reachable from the current node.
///
/// `max_results` caps the total number of paths collected to bound memory usage.
/// Pass `usize::MAX` for effectively unlimited results.
pub fn bfs_traverse<F>(
    start_node: GraphNode,
    edge_types: Option<&[String]>,
    max_hops: usize,
    node_filter: Option<&HashMap<String, String>>,
    max_results: usize,
    get_neighbors: F,
) -> TraversalResult
where
    F: Fn(&str, Option<&str>, usize) -> Vec<(GraphEdge, GraphNode)>,
{
    let start_id = start_node.node_id.clone();

    let mut visited: HashSet<String> = HashSet::new();
    visited.insert(start_id.clone());
    let mut all_nodes: HashMap<String, GraphNode> = HashMap::new();
    all_nodes.insert(start_id.clone(), start_node.clone());
    let mut all_edges: Vec<GraphEdge> = Vec::new();
    let mut paths: Vec<Vec<TraversalItem>> = Vec::new();
    let mut origins: HashSet<String> = HashSet::new();
    origins.insert(start_node.graph_origin.clone());

    let mut queue: VecDeque<(String, Vec<TraversalItem>, usize)> = VecDeque::new();
    queue.push_back((start_id, vec![TraversalItem::Node(start_node)], 0));

    while let Some((current_id, current_path, hops)) = queue.pop_front() {
        if hops >= max_hops {
            continue;
        }

        let neighbors = if let Some(etypes) = edge_types {
            let mut all = Vec::new();
            for et in etypes {
                all.extend(get_neighbors(&current_id, Some(et), hops));
            }
            all
        } else {
            get_neighbors(&current_id, None, hops)
        };

        for (edge, neighbor) in neighbors {
            if let Some(filter) = node_filter {
                if !filter
                    .iter()
                    .all(|(k, v)| neighbor.properties.get(k) == Some(v))
                {
                    continue;
                }
            }

            all_edges.push(edge.clone());
            origins.insert(neighbor.graph_origin.clone());

            let mut new_path = current_path.clone();
            new_path.push(TraversalItem::Edge(edge));
            new_path.push(TraversalItem::Node(neighbor.clone()));

            if !visited.contains(&neighbor.node_id) {
                visited.insert(neighbor.node_id.clone());
                all_nodes.insert(neighbor.node_id.clone(), neighbor.clone());
                queue.push_back((neighbor.node_id.clone(), new_path.clone(), hops + 1));
            }

            paths.push(new_path);

            if paths.len() >= max_results {
                break;
            }
        }

        if paths.len() >= max_results {
            break;
        }
    }

    TraversalResult {
        paths,
        nodes: all_nodes.into_values().collect(),
        edges: all_edges,
        crossed_boundaries: origins.len() > 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(id: &str, origin: &str) -> GraphNode {
        GraphNode {
            node_id: id.into(),
            node_type: "Test".into(),
            properties: HashMap::new(),
            graph_origin: origin.into(),
        }
    }

    fn make_node_with_props(id: &str, origin: &str, props: HashMap<String, String>) -> GraphNode {
        GraphNode {
            node_id: id.into(),
            node_type: "Test".into(),
            properties: props,
            graph_origin: origin.into(),
        }
    }

    fn make_edge(src: &str, tgt: &str, etype: &str) -> GraphEdge {
        GraphEdge {
            edge_id: format!("{src}->{tgt}"),
            source_id: src.into(),
            target_id: tgt.into(),
            edge_type: etype.into(),
            properties: HashMap::new(),
            graph_origin: String::new(),
        }
    }

    #[test]
    fn test_bfs_empty_graph_no_neighbors() {
        let start = make_node("a", "local");
        let result = bfs_traverse(start, None, 3, None, 100, |_id, _et, _depth| vec![]);
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.edges.len(), 0);
    }

    #[test]
    fn test_bfs_single_node_zero_hops() {
        let start = make_node("a", "local");
        let result = bfs_traverse(start, None, 0, None, 100, |_id, _et, _depth| {
            vec![(make_edge("a", "b", "LINK"), make_node("b", "local"))]
        });
        assert_eq!(result.nodes.len(), 1);
    }

    #[test]
    fn test_bfs_linear_chain() {
        let start = make_node("a", "local");
        let result = bfs_traverse(start, None, 3, None, 100, |id, _et, _depth| match id {
            "a" => vec![(make_edge("a", "b", "LINK"), make_node("b", "local"))],
            "b" => vec![(make_edge("b", "c", "LINK"), make_node("c", "local"))],
            _ => vec![],
        });
        assert_eq!(result.nodes.len(), 3);
        assert_eq!(result.edges.len(), 2);
    }

    #[test]
    fn test_bfs_cyclic_graph() {
        let start = make_node("a", "local");
        let result = bfs_traverse(start, None, 10, None, 100, |id, _et, _depth| match id {
            "a" => vec![(make_edge("a", "b", "LINK"), make_node("b", "local"))],
            "b" => vec![(make_edge("b", "c", "LINK"), make_node("c", "local"))],
            "c" => vec![(make_edge("c", "a", "LINK"), make_node("a", "local"))],
            _ => vec![],
        });
        assert_eq!(result.nodes.len(), 3);
    }

    #[test]
    fn test_bfs_node_filter() {
        let start = make_node("a", "local");
        let mut filter = HashMap::new();
        filter.insert("role".into(), "keep".into());

        let result = bfs_traverse(
            start,
            None,
            3,
            Some(&filter),
            100,
            |id, _et, _depth| match id {
                "a" => {
                    let mut p1 = HashMap::new();
                    p1.insert("role".into(), "keep".into());
                    let mut p2 = HashMap::new();
                    p2.insert("role".into(), "skip".into());
                    vec![
                        (
                            make_edge("a", "b", "LINK"),
                            make_node_with_props("b", "local", p1),
                        ),
                        (
                            make_edge("a", "c", "LINK"),
                            make_node_with_props("c", "local", p2),
                        ),
                    ]
                }
                _ => vec![],
            },
        );
        assert!(result.nodes.iter().any(|n| n.node_id == "b"));
        assert!(!result.nodes.iter().any(|n| n.node_id == "c"));
    }

    #[test]
    fn test_bfs_max_depth() {
        let start = make_node("a", "local");
        let result = bfs_traverse(start, None, 1, None, 100, |id, _et, _depth| match id {
            "a" => vec![(make_edge("a", "b", "LINK"), make_node("b", "local"))],
            "b" => vec![(make_edge("b", "c", "LINK"), make_node("c", "local"))],
            _ => vec![],
        });
        let ids: Vec<&str> = result.nodes.iter().map(|n| n.node_id.as_str()).collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
        assert!(!ids.contains(&"c"));
    }

    #[test]
    fn test_bfs_max_results_caps_paths() {
        let start = make_node("a", "local");
        let result = bfs_traverse(start, None, 3, None, 1, |id, _et, _depth| match id {
            "a" => vec![
                (make_edge("a", "b", "LINK"), make_node("b", "local")),
                (make_edge("a", "c", "LINK"), make_node("c", "local")),
            ],
            _ => vec![],
        });
        assert_eq!(result.paths.len(), 1);
    }

    #[test]
    fn test_bfs_crossed_boundaries() {
        let start = make_node("a", "local");
        let result = bfs_traverse(start, None, 2, None, 100, |id, _et, _depth| match id {
            "a" => vec![(make_edge("a", "b", "LINK"), make_node("b", "remote"))],
            _ => vec![],
        });
        assert!(result.crossed_boundaries);
    }

    #[test]
    fn test_bfs_edge_type_filter() {
        let start = make_node("a", "local");
        let etypes = vec!["KNOWS".to_string()];
        let result = bfs_traverse(start, Some(&etypes), 2, None, 100, |_id, et, _depth| {
            assert_eq!(et, Some("KNOWS"));
            vec![(make_edge("a", "b", "KNOWS"), make_node("b", "local"))]
        });
        assert!(result.nodes.len() >= 2);
    }
}
