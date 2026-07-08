//! Ranked (scored) recall over [`CognitiveMemory`].
//!
//! Adds a scored recall path alongside the existing keyword-only
//! `search_facts` / `search_episodes_by_keyword`. Ranking combines six signals
//! — text relevance, confidence, importance, recency, usage, and graph
//! proximity — into a single un-normalized score, with documented defaults,
//! lifecycle filtering (archived / superseded / compressed excluded by
//! default), and optional access tracking (`usage_count` + `last_accessed_at`)
//! that persists across reopen.
//!
//! This module is purely additive and backward compatible: no existing public
//! signature, sort order, or serde shape changes. All logic lives behind the
//! existing [`GraphStore`](crate::graph::GraphStore) seam, so behavior is
//! identical on the in-memory and `--features persistent` (LadybugDB) backends.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::graph::types::{Direction, GraphNode};
use crate::memory_types::{EpisodicMemory, SemanticFact};
use crate::{MemoryError, Result};

use super::converters::{node_to_episodic, node_to_fact, prop_bool, prop_i64, prop_opt_datetime};
use super::types::{
    agent_filter, ts_now, ET_DERIVES_FROM, ET_SIMILAR_TO, NT_EPISODIC, NT_SEMANTIC,
};
use super::CognitiveMemory;

use tracing::debug;

/// Upper bound on graph-traversal depth, regardless of `max_graph_hops`.
///
/// Caps neighbor fan-out (security R6/D5): a caller-supplied
/// `RecallOptions::max_graph_hops` is clamped to this value before traversal.
const MAX_GRAPH_HOPS: usize = 3;

// ---------------------------------------------------------------------------
// Public value types
// ---------------------------------------------------------------------------

/// Per-signal weights for [`recall_facts_ranked`](CognitiveMemory::recall_facts_ranked)
/// and [`recall_episodes_ranked`](CognitiveMemory::recall_episodes_ranked).
///
/// Each field scales one scoring term. Weights are un-normalized — only their
/// relative magnitudes matter. The [`Default`] reflects the tuned baseline:
/// text relevance dominates, with confidence and graph proximity as the next
/// strongest signals.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RecallWeights {
    /// Weight on keyword Jaccard overlap between the query and the item text.
    pub text_relevance: f64,
    /// Weight on the fact's confidence (no-op for episodes; treated as 0).
    pub confidence: f64,
    /// Weight on the fact's importance (no-op for episodes; treated as 0).
    pub importance: f64,
    /// Weight on exponential recency decay of the last access / creation time.
    pub recency: f64,
    /// Weight on the sub-linear usage boost (`ln(1 + usage_count)`).
    pub usage: f64,
    /// Weight on the best graph-neighbor relevance within `max_graph_hops`.
    pub graph: f64,
}

impl Default for RecallWeights {
    fn default() -> Self {
        Self {
            text_relevance: 1.0,
            confidence: 0.7,
            importance: 0.5,
            recency: 0.4,
            usage: 0.3,
            graph: 0.6,
        }
    }
}

/// How a node was accessed. Recorded by
/// [`record_access`](CognitiveMemory::record_access); both variants currently
/// increment `usage_count` identically (reserved for future differential
/// weighting). The serde wire form is the variant name (`"Read"` / `"Recall"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessKind {
    /// A direct read of a single node.
    Read,
    /// An access made while recalling/ranking results.
    Recall,
}

/// Tunable options for ranked recall.
///
/// Construct via [`Default`] and struct-update so new fields stay
/// source-compatible:
///
/// ```
/// use amplihack_memory::RecallOptions;
/// let opts = RecallOptions { limit: 25, ..Default::default() };
/// assert_eq!(opts.limit, 25);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecallOptions {
    /// Maximum number of results to return. `0` yields an empty result.
    pub limit: usize,
    /// Minimum confidence floor (facts only; no-op for episodes).
    pub min_confidence: f64,
    /// Include archived facts. Defaults to `false`.
    pub include_archived: bool,
    /// Include superseded facts. Defaults to `false`.
    pub include_superseded: bool,
    /// Maximum graph-traversal hops for the graph term. `0` disables the graph
    /// term; values above [`MAX_GRAPH_HOPS`] are clamped.
    pub max_graph_hops: usize,
    /// Recency half-life in seconds. `<= 0` makes the recency term `0`.
    pub recency_half_life_seconds: f64,
    /// Record access (bump `usage_count` + `last_accessed_at`) for every
    /// returned item. Defaults to `true`; set `false` for a pure read.
    pub record_access: bool,
    /// Per-signal weights.
    pub weights: RecallWeights,
}

impl Default for RecallOptions {
    fn default() -> Self {
        Self {
            limit: 10,
            min_confidence: 0.0,
            include_archived: false,
            include_superseded: false,
            max_graph_hops: 1,
            recency_half_life_seconds: 604_800.0,
            record_access: true,
            weights: RecallWeights::default(),
        }
    }
}

/// A scored recall result: the recalled item, its combined score, and a list of
/// human-readable, numeric/label-only `reasons` explaining the score.
///
/// `reasons` never embeds raw content, neighbor body text, or the query — each
/// entry is one positive scoring term (or a single baseline marker when every
/// term is zero), so the vector is always non-empty.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scored<T> {
    /// The recalled item (a [`SemanticFact`] or [`EpisodicMemory`]).
    pub item: T,
    /// The combined, un-normalized relevance score (higher is better).
    pub score: f64,
    /// One short, payload-free explanation per positive scoring term.
    pub reasons: Vec<String>,
}

// ---------------------------------------------------------------------------
// Pure scoring primitives (no `self`, no clock — deterministic & unit-testable)
// ---------------------------------------------------------------------------

/// Jaccard similarity of the lowercased whitespace-token *sets* of `query` and
/// `text`: `|A ∩ B| / |A ∪ B|`. Returns `0.0` when either side is empty (so an
/// empty union never divides by zero).
pub fn keyword_jaccard(query: &str, text: &str) -> f64 {
    let a: HashSet<String> = query.split_whitespace().map(str::to_lowercase).collect();
    let b: HashSet<String> = text.split_whitespace().map(str::to_lowercase).collect();
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(&b).count() as f64;
    let union = a.union(&b).count() as f64;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Exponential recency decay: `2^(-age / half_life)` in `(0, 1]`.
///
/// `age = 0` yields `1.0`; `age = half_life` yields `0.5`. Guarded so a
/// non-positive `half_life_seconds` returns `0.0` (no division by zero / NaN).
pub fn exp_decay(age_seconds: f64, half_life_seconds: f64) -> f64 {
    if half_life_seconds <= 0.0 {
        return 0.0;
    }
    2f64.powf(-age_seconds / half_life_seconds)
}

/// Sub-linear usage boost: `ln(1 + max(0, usage_count))`.
///
/// `0` at `usage_count <= 0`, monotonically increasing, and sub-linear so a
/// large usage count cannot dominate the combined score.
pub fn usage_boost(usage_count: i64) -> f64 {
    (1.0 + usage_count.max(0) as f64).ln()
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build the scoring text for a neighbor node: `concept + " " + content` when a
/// concept is present (facts), else `content` (episodes).
fn node_text(node: &GraphNode) -> String {
    let concept = node
        .properties
        .get("concept")
        .map(String::as_str)
        .unwrap_or("");
    let content = node
        .properties
        .get("content")
        .map(String::as_str)
        .unwrap_or("");
    match (concept.is_empty(), content.is_empty()) {
        (true, _) => content.to_string(),
        (false, true) => concept.to_string(),
        (false, false) => format!("{concept} {content}"),
    }
}

// ---------------------------------------------------------------------------
// Per-recall graph-adjacency index (issue #40)
// ---------------------------------------------------------------------------

/// Direction-aware in-memory adjacency for a single edge type.
///
/// `outgoing[src]` lists the target nodes of edges `src -> *`; `incoming[tgt]`
/// lists the source nodes of edges `* -> tgt`. These are exactly the neighbour
/// sets [`GraphStore::query_neighbors`](crate::graph::GraphStore::query_neighbors)
/// returns for [`Direction::Outgoing`] / [`Direction::Incoming`], so the graph
/// BFS reads identical results — without a per-node database round-trip.
#[derive(Default)]
struct EdgeAdjacency {
    outgoing: HashMap<String, Vec<GraphNode>>,
    incoming: HashMap<String, Vec<GraphNode>>,
}

/// Per-recall adjacency index over the traversed edge types, built once from a
/// small constant number of bulk edge scans (one per distinct edge type).
///
/// Replaces the ranked-recall graph term's ~`3N` per-node `query_neighbors`
/// fan-out (the Simard OODA prepare-context pathology, issue #40) with two
/// store scans per recall while keeping the bounded BFS — and therefore the
/// score, ordering, and reasons — byte-identical to the legacy per-node path.
struct GraphAdjacencyIndex {
    by_type: HashMap<String, EdgeAdjacency>,
}

impl GraphAdjacencyIndex {
    /// Neighbours of `node_id` along `edge_type` in `direction`, in the same
    /// order [`GraphStore::query_neighbors`](crate::graph::GraphStore::query_neighbors)
    /// returns them (outgoing before incoming for [`Direction::Both`]).
    fn neighbors(&self, node_id: &str, edge_type: &str, direction: Direction) -> Vec<GraphNode> {
        let mut out = Vec::new();
        let Some(adj) = self.by_type.get(edge_type) else {
            return out;
        };
        if matches!(direction, Direction::Outgoing | Direction::Both) {
            if let Some(v) = adj.outgoing.get(node_id) {
                out.extend(v.iter().cloned());
            }
        }
        if matches!(direction, Direction::Incoming | Direction::Both) {
            if let Some(v) = adj.incoming.get(node_id) {
                out.extend(v.iter().cloned());
            }
        }
        out
    }
}

impl CognitiveMemory {
    // ======================================================================
    // RANKED RECALL
    // ======================================================================

    /// Ranked recall over this agent's semantic facts.
    ///
    /// Scores every non-excluded fact by the weighted combination of text
    /// relevance, confidence, importance, recency, usage, and graph proximity
    /// (see module docs), returns them sorted by descending score (NaN-safe,
    /// deterministic tie-break), and — when `options.record_access` is `true`
    /// (the default) — records a [`Recall`](AccessKind::Recall) access for each
    /// returned item (affecting subsequent calls, not the returned values).
    ///
    /// Archived and superseded facts are excluded unless opted in via
    /// `options`; facts below `options.min_confidence` are dropped. The existing
    /// [`search_facts`](Self::search_facts) / [`get_all_facts`](Self::get_all_facts)
    /// remain unchanged.
    ///
    /// # Errors
    ///
    /// Only when `options.record_access` is `true` and a post-rank access write
    /// fails: the first failure returns that `MemoryError`.
    pub fn recall_facts_ranked(
        &mut self,
        query: &str,
        options: RecallOptions,
    ) -> Result<Vec<Scored<SemanticFact>>> {
        let filter = agent_filter(&self.agent_name);
        let nodes = self
            .graph
            .query_nodes(NT_SEMANTIC, Some(&filter), usize::MAX);
        let now = ts_now();
        let traversals = [
            (ET_DERIVES_FROM, Direction::Outgoing),
            (ET_SIMILAR_TO, Direction::Both),
        ];
        let index = self.build_graph_index(&traversals, &options);

        let mut scored: Vec<Scored<SemanticFact>> = Vec::new();
        for node in &nodes {
            let fact = node_to_fact(&node.properties);

            if fact.archived && !options.include_archived {
                continue;
            }
            if fact.superseded_by.is_some() && !options.include_superseded {
                continue;
            }
            if fact.confidence < options.min_confidence {
                continue;
            }

            let text = format!("{} {}", fact.concept, fact.content);
            let reference_time = fact
                .last_accessed_at
                .map(|d| d.timestamp())
                .unwrap_or_else(|| fact.created_at.timestamp());

            let (score, reasons) = self.score_signals(
                query,
                &text,
                fact.confidence,
                fact.importance,
                fact.usage_count,
                reference_time,
                now,
                &fact.node_id,
                &traversals,
                &options,
                index.as_ref(),
            );
            scored.push(Scored {
                item: fact,
                score,
                reasons,
            });
        }

        scored.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then(b.item.confidence.total_cmp(&a.item.confidence))
                .then(a.item.node_id.cmp(&b.item.node_id))
        });
        scored.truncate(options.limit);

        if options.record_access {
            let ids: Vec<String> = scored.iter().map(|s| s.item.node_id.clone()).collect();
            for id in ids {
                self.record_access(&id, AccessKind::Recall)?;
            }
        }

        Ok(scored)
    }

    /// Ranked recall over this agent's episodic memories.
    ///
    /// Mirrors [`recall_facts_ranked`](Self::recall_facts_ranked) for episodes.
    /// Confidence, importance, archived, superseded, and `min_confidence` are
    /// no-ops for episodes; compressed episodes are always excluded (parity with
    /// [`search_episodes_by_keyword`](Self::search_episodes_by_keyword)). The
    /// graph term follows `DERIVES_FROM` incoming (toward the source fact) and
    /// `SIMILAR_TO` in both directions.
    ///
    /// # Errors
    ///
    /// Only when `options.record_access` is `true` and a post-rank access write
    /// fails: the first failure returns that `MemoryError`.
    pub fn recall_episodes_ranked(
        &mut self,
        query: &str,
        options: RecallOptions,
    ) -> Result<Vec<Scored<EpisodicMemory>>> {
        let filter = agent_filter(&self.agent_name);
        let nodes = self
            .graph
            .query_nodes(NT_EPISODIC, Some(&filter), usize::MAX);
        let now = ts_now();
        let traversals = [
            (ET_DERIVES_FROM, Direction::Incoming),
            (ET_SIMILAR_TO, Direction::Both),
        ];
        let index = self.build_graph_index(&traversals, &options);

        let mut scored: Vec<Scored<EpisodicMemory>> = Vec::new();
        for node in &nodes {
            if prop_bool(&node.properties, "compressed") {
                continue;
            }

            let usage_count = prop_i64(&node.properties, "usage_count");
            let reference_time = prop_opt_datetime(&node.properties, "last_accessed_at")
                .map(|d| d.timestamp())
                .unwrap_or_else(|| prop_i64(&node.properties, "created_at"));

            let episode = node_to_episodic(&node.properties);
            let text = episode.content.clone();

            let (score, reasons) = self.score_signals(
                query,
                &text,
                0.0,
                0.0,
                usage_count,
                reference_time,
                now,
                &episode.node_id,
                &traversals,
                &options,
                index.as_ref(),
            );
            scored.push(Scored {
                item: episode,
                score,
                reasons,
            });
        }

        scored.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then(b.item.temporal_index.cmp(&a.item.temporal_index))
                .then(a.item.node_id.cmp(&b.item.node_id))
        });
        scored.truncate(options.limit);

        if options.record_access {
            let ids: Vec<String> = scored.iter().map(|s| s.item.node_id.clone()).collect();
            for id in ids {
                self.record_access(&id, AccessKind::Recall)?;
            }
        }

        Ok(scored)
    }

    /// Record an access to a node: increment its `usage_count` (saturating) and
    /// set `last_accessed_at` to now. Persisted, so the bump survives reopen.
    ///
    /// Works for any node type owned by this agent. Writes only the two static
    /// counter properties — never the query or any content.
    ///
    /// # Errors
    ///
    /// - [`MemoryError::Storage`] if `node_id` does not exist.
    /// - [`MemoryError::SecurityViolation`] if the node is owned by a different
    ///   agent (no write is performed).
    /// - [`MemoryError::Storage`] if the backend update fails.
    pub fn record_access(&mut self, node_id: &str, kind: AccessKind) -> Result<()> {
        let _ = kind; // Read and Recall increment identically (reserved).

        let node = self.graph.get_node(node_id).ok_or_else(|| {
            MemoryError::Storage(format!("record_access: node {node_id} not found"))
        })?;

        if node.properties.get("agent_id").map(String::as_str) != Some(self.agent_name.as_str()) {
            return Err(MemoryError::SecurityViolation(
                "record_access: node not owned by current agent".into(),
            ));
        }

        let prev = prop_i64(&node.properties, "usage_count");
        let mut props = HashMap::new();
        props.insert(
            "usage_count".to_string(),
            prev.saturating_add(1).to_string(),
        );
        props.insert("last_accessed_at".to_string(), ts_now().to_string());

        if !self.graph.update_node(node_id, props) {
            return Err(MemoryError::Storage(format!(
                "record_access: update failed for {node_id}"
            )));
        }
        Ok(())
    }

    // ======================================================================
    // SCORING (internal)
    // ======================================================================

    /// Compute the combined score and `reasons` for one candidate.
    ///
    /// `confidence` / `importance` are passed as `0.0` for episodes so their
    /// terms vanish. Every positive weighted term contributes one numeric /
    /// label-only `reasons` entry; if all terms are zero a single baseline
    /// marker is emitted so the vector is never empty.
    #[allow(clippy::too_many_arguments)]
    fn score_signals(
        &self,
        query: &str,
        text: &str,
        confidence: f64,
        importance: f64,
        usage_count: i64,
        reference_time: i64,
        now: i64,
        node_id: &str,
        traversals: &[(&str, Direction)],
        options: &RecallOptions,
        index: Option<&GraphAdjacencyIndex>,
    ) -> (f64, Vec<String>) {
        let w = &options.weights;
        let mut score = 0.0;
        let mut reasons: Vec<String> = Vec::new();

        // Text relevance.
        let jaccard = keyword_jaccard(query, text);
        let text_term = w.text_relevance * jaccard;
        if text_term > 0.0 {
            reasons.push(format!("text {text_term:.4} (jaccard={jaccard:.4})"));
        }
        score += text_term;

        // Confidence (facts only; 0 for episodes).
        let conf_term = w.confidence * confidence;
        if conf_term > 0.0 {
            reasons.push(format!("confidence {conf_term:.4}"));
        }
        score += conf_term;

        // Importance (facts only; 0 for episodes).
        let imp_term = w.importance * importance;
        if imp_term > 0.0 {
            reasons.push(format!("importance {imp_term:.4}"));
        }
        score += imp_term;

        // Recency.
        let age = (now - reference_time).max(0) as f64;
        let recency_term = w.recency * exp_decay(age, options.recency_half_life_seconds);
        if recency_term > 0.0 {
            reasons.push(format!("recency {recency_term:.4} (age={}s)", age as i64));
        }
        score += recency_term;

        // Usage.
        let usage_term = w.usage * usage_boost(usage_count);
        if usage_term > 0.0 {
            reasons.push(format!("usage {usage_term:.4} (n={usage_count})"));
        }
        score += usage_term;

        // Graph proximity.
        if options.max_graph_hops > 0 {
            let (best, meta) =
                self.best_edge_score(node_id, query, traversals, options.max_graph_hops, index);
            let graph_term = w.graph * best;
            if graph_term > 0.0 {
                if let Some((label, hop)) = meta {
                    reasons.push(format!("graph {graph_term:.4} ({label} hop{hop})"));
                }
            }
            score += graph_term;
        }

        if reasons.is_empty() {
            reasons.push("baseline (no positive signals)".to_string());
        }

        (score, reasons)
    }

    /// Best graph-neighbor relevance reachable from `start_id` within
    /// `min(max_hops, MAX_GRAPH_HOPS)`.
    ///
    /// Runs a bounded BFS over the supplied `(edge_type, direction)` pairs,
    /// scoring each agent-owned neighbor by `keyword_jaccard(query, text) / hop`
    /// and returning the maximum (with the edge label + hop of the winner for
    /// the reason string). Foreign-tenant neighbors are pruned entirely — they
    /// neither contribute a score nor are traversed through (security A3).
    ///
    /// When `index` is `Some`, each hop's neighbours are read from the per-recall
    /// in-memory adjacency index (issue #40) instead of a per-node
    /// `query_neighbors` round-trip; the walk, per-hop `/hop` discount, tenant
    /// prune, and `best` selection are otherwise identical, so the returned score
    /// is byte-identical to the legacy path.
    fn best_edge_score(
        &self,
        start_id: &str,
        query: &str,
        traversals: &[(&str, Direction)],
        max_hops: usize,
        index: Option<&GraphAdjacencyIndex>,
    ) -> (f64, Option<(String, usize)>) {
        let hops = max_hops.min(MAX_GRAPH_HOPS);
        if hops == 0 {
            return (0.0, None);
        }

        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(start_id.to_string());
        let mut frontier: Vec<String> = vec![start_id.to_string()];
        let mut best = 0.0_f64;
        let mut best_meta: Option<(String, usize)> = None;

        for hop in 1..=hops {
            let mut next: Vec<String> = Vec::new();
            for nid in &frontier {
                for (edge_type, direction) in traversals {
                    let neighbors = match index {
                        // Fast path: read the neighbour set from the pre-built
                        // adjacency index (no database round-trip).
                        Some(idx) => idx.neighbors(nid, edge_type, *direction),
                        // Legacy path: one per-node neighbour scan, preserved for
                        // backends without the bulk capability.
                        None => self
                            .graph
                            .query_neighbors(nid, Some(edge_type), *direction, usize::MAX)
                            .into_iter()
                            .map(|(_, neighbor)| neighbor)
                            .collect(),
                    };
                    for neighbor in neighbors {
                        if !visited.insert(neighbor.node_id.clone()) {
                            continue;
                        }
                        // A3: drop foreign-tenant neighbors before they score or
                        // expand the frontier.
                        if neighbor.properties.get("agent_id").map(String::as_str)
                            != Some(self.agent_name.as_str())
                        {
                            continue;
                        }
                        let s = keyword_jaccard(query, &node_text(&neighbor)) / hop as f64;
                        if s > best {
                            best = s;
                            best_meta = Some(((*edge_type).to_string(), hop));
                        }
                        next.push(neighbor.node_id);
                    }
                }
            }
            frontier = next;
        }

        (best, best_meta)
    }

    /// Build the per-recall graph-adjacency index for `traversals`, or `None`
    /// when the graph term is disabled (`max_graph_hops == 0`) or the backend
    /// does not support the bulk-scan capability (all-or-nothing fold to the
    /// legacy per-node path — the index and legacy path are never mixed within a
    /// single recall).
    ///
    /// Issues exactly one [`bulk_edges_of_type`](crate::graph::GraphStore::bulk_edges_of_type)
    /// scan per **distinct** edge type in `traversals` (two: `DERIVES_FROM` and
    /// `SIMILAR_TO`), independent of the fact count — this is what removes the
    /// ~`3N` per-node fan-out behind issue #40. Emits a single `graph_path`
    /// tracing line (`indexed` | `legacy`, counts/labels only — never memory
    /// content) so the active path is verifiable from a live log.
    fn build_graph_index(
        &self,
        traversals: &[(&str, Direction)],
        options: &RecallOptions,
    ) -> Option<GraphAdjacencyIndex> {
        if options.max_graph_hops == 0 {
            return None;
        }

        let mut by_type: HashMap<String, EdgeAdjacency> = HashMap::new();
        for (edge_type, _direction) in traversals {
            if by_type.contains_key(*edge_type) {
                continue; // one bulk scan per distinct edge type
            }
            // All-or-nothing: a `None` from any required edge type discards the
            // partial index so the whole recall takes the legacy path.
            let edges = match self.graph.bulk_edges_of_type(edge_type) {
                Some(edges) => edges,
                None => {
                    debug!(
                        target: "amplihack_memory::cognitive_memory",
                        graph_path = "legacy",
                        "ranked recall graph term: backend lacks bulk_edges_of_type"
                    );
                    return None;
                }
            };
            let mut adj = EdgeAdjacency::default();
            for (src, tgt) in edges {
                adj.outgoing
                    .entry(src.node_id.clone())
                    .or_default()
                    .push(tgt.clone());
                adj.incoming
                    .entry(tgt.node_id.clone())
                    .or_default()
                    .push(src);
            }
            by_type.insert((*edge_type).to_string(), adj);
        }

        debug!(
            target: "amplihack_memory::cognitive_memory",
            graph_path = "indexed",
            edge_types = by_type.len(),
            "ranked recall graph term: bulk adjacency index"
        );
        Some(GraphAdjacencyIndex { by_type })
    }

    /// Test-only seam: overwrite raw node properties through the backing
    /// `GraphStore`, used to fabricate deterministic state (old timestamps,
    /// foreign `agent_id`, lifecycle flags) the public API can't construct
    /// directly. Returns `true` on success.
    #[cfg(test)]
    pub(crate) fn set_node_props_for_test(
        &mut self,
        node_id: &str,
        props: HashMap<String, String>,
    ) -> bool {
        self.graph.update_node(node_id, props)
    }
}
