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
