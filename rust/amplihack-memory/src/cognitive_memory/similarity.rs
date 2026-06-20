//! Automatic `SIMILAR_TO` linking for semantic facts.
//!
//! Connects related semantic facts with `SIMILAR_TO` edges using the
//! deterministic Jaccard/tag/concept composite score from
//! [`crate::similarity::compute_similarity`] (no ML embeddings required).
//!
//! Two entry points are provided on [`CognitiveMemory`]:
//!
//! * [`auto_link_similar_facts`](CognitiveMemory::auto_link_similar_facts) —
//!   link one fact to its above-threshold neighbours. The primary, explicit
//!   call.
//! * [`rebuild_similarity_links`](CognitiveMemory::rebuild_similarity_links) —
//!   backfill/maintenance pass that ensures every above-threshold pair across
//!   all facts is linked.
//!
//! Both are additive and idempotent: an existing `SIMILAR_TO` edge (whether
//! auto- or manually created via
//! [`link_similar_facts`](CognitiveMemory::link_similar_facts)) suppresses a
//! duplicate for that unordered pair, and re-running creates nothing new.

use std::collections::HashMap;

use tracing::warn;

use crate::graph::types::Direction;
use crate::memory_types::SemanticFact;
use crate::similarity::compute_similarity;
use crate::{MemoryError, Result};

use super::types::ET_SIMILAR_TO;
use super::CognitiveMemory;

/// Tuning knobs for automatic `SIMILAR_TO` linking.
///
/// [`Default`] enables linking with a `0.60` composite threshold, scoring up to
/// `50` candidate facts per source and storing reciprocal (bidirectional)
/// edges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimilarityOptions {
    /// When `false`, linking methods are inert no-ops (return `0` / an empty
    /// report and write no edges). Lets a single options value also gate the
    /// store-time hook.
    pub enabled: bool,
    /// Minimum composite similarity (`0.5*word + 0.2*tag + 0.3*concept`) for a
    /// pair to be linked. A pair is linked when `score >= threshold`.
    pub threshold: f64,
    /// Upper bound on how many other facts are scored per source fact.
    pub candidate_limit: usize,
    /// When `true`, store both `A -> B` and `B -> A`; when `false`, only the
    /// forward `source -> candidate` edge.
    pub bidirectional: bool,
}

impl Default for SimilarityOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: 0.60,
            candidate_limit: 50,
            bidirectional: true,
        }
    }
}

/// Summary of a [`rebuild_similarity_links`](CognitiveMemory::rebuild_similarity_links)
/// pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimilarityReport {
    /// Number of semantic facts considered.
    pub facts_processed: usize,
    /// Number of new unordered pairs linked this pass (pre-existing edges are
    /// preserved and not counted).
    pub links_created: usize,
}

/// Options for [`store_fact_with_options`](CognitiveMemory::store_fact_with_options).
///
/// [`Default`] (`similarity: None`) makes the call behave exactly like
/// [`store_fact`](CognitiveMemory::store_fact): no `SIMILAR_TO` edges are
/// created, preserving backward compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct StoreFactOptions {
    /// When `Some(opts)` (and `opts.enabled`), the freshly stored fact is
    /// auto-linked to its above-threshold neighbours after storage.
    pub similarity: Option<SimilarityOptions>,
}

/// Build the property map [`compute_similarity`] expects from a fact.
fn fact_to_sim_map(fact: &SemanticFact) -> HashMap<String, serde_json::Value> {
    let mut map = HashMap::with_capacity(3);
    map.insert(
        "content".to_string(),
        serde_json::Value::String(fact.content.clone()),
    );
    map.insert(
        "concept".to_string(),
        serde_json::Value::String(fact.concept.clone()),
    );
    map.insert("tags".to_string(), serde_json::json!(fact.tags));
    map
}

impl CognitiveMemory {
    /// Return `true` if a `SIMILAR_TO` edge already connects `a` and `b` in
    /// either direction (the unordered-pair idempotency key).
    fn similar_pair_exists(&self, a: &str, b: &str) -> bool {
        self.graph
            .query_neighbors(a, Some(ET_SIMILAR_TO), Direction::Both, usize::MAX)
            .iter()
            .any(|(_, neighbor)| neighbor.node_id == b)
    }

    /// Create a directed `SIMILAR_TO` edge `from -> to` carrying a fixed
    /// 4-decimal `similarity_score` property. Both endpoints must already exist.
    fn add_similar_edge(&mut self, from: &str, to: &str, score: f64) -> Result<()> {
        let mut props = HashMap::with_capacity(1);
        props.insert("similarity_score".to_string(), format!("{score:.4}"));
        self.graph
            .add_edge(from, to, ET_SIMILAR_TO, Some(props))
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Link a single new (unordered) pair, honouring `bidirectional`. Assumes
    /// the caller has already confirmed the pair is not yet connected.
    fn link_similar_pair(
        &mut self,
        source_id: &str,
        candidate_id: &str,
        score: f64,
        options: &SimilarityOptions,
    ) -> Result<()> {
        self.add_similar_edge(source_id, candidate_id, score)?;
        if options.bidirectional {
            self.add_similar_edge(candidate_id, source_id, score)?;
        }
        Ok(())
    }

    /// Auto-link `fact_id` to every other same-agent semantic fact whose
    /// composite similarity is at or above `options.threshold`.
    ///
    /// For each newly linked pair a `SIMILAR_TO` edge (carrying a
    /// `similarity_score` property) is created from `fact_id` to the candidate,
    /// plus a reciprocal edge when `options.bidirectional`. At most
    /// `options.candidate_limit` other facts are scored. Returns the number of
    /// candidate facts newly linked this call (unordered pairs created); a
    /// reciprocal edge is part of a pair, not counted separately.
    ///
    /// Idempotent: a pair already connected by a `SIMILAR_TO` edge (in either
    /// direction, auto- or manually created) is skipped, so re-running returns
    /// `0` and creates no duplicates. Never creates a self-loop.
    ///
    /// Lenient: when `options.enabled` is `false` this is an inert no-op
    /// (`Ok(0)`), and an unknown `fact_id` logs a warning and returns `Ok(0)`
    /// rather than erroring.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::Storage` if the backend rejects an edge write.
    pub fn auto_link_similar_facts(
        &mut self,
        fact_id: &str,
        options: &SimilarityOptions,
    ) -> Result<usize> {
        if !options.enabled {
            return Ok(0);
        }

        let facts = self.get_all_facts(usize::MAX);
        let Some(source) = facts.iter().find(|f| f.node_id == fact_id) else {
            warn!("auto_link_similar_facts: fact {fact_id} not found; skipping");
            return Ok(0);
        };
        let source_map = fact_to_sim_map(source);

        let candidates: Vec<SemanticFact> = facts
            .iter()
            .filter(|f| f.node_id != fact_id)
            .take(options.candidate_limit)
            .cloned()
            .collect();

        let mut created = 0usize;
        for candidate in &candidates {
            if self.similar_pair_exists(fact_id, &candidate.node_id) {
                continue;
            }
            let score = compute_similarity(&source_map, &fact_to_sim_map(candidate));
            if score >= options.threshold {
                self.link_similar_pair(fact_id, &candidate.node_id, score, options)?;
                created += 1;
            }
        }

        Ok(created)
    }

    /// Recompute `SIMILAR_TO` links across all of this agent's semantic facts
    /// for backfill / maintenance.
    ///
    /// Additive and non-destructive: existing edges (including manual ones from
    /// [`link_similar_facts`](Self::link_similar_facts)) are preserved and
    /// suppress duplicates for their pair. Every above-threshold unordered pair
    /// ends up linked exactly once. Each fact scores at most
    /// `options.candidate_limit` others. Running twice creates `0` new links.
    ///
    /// When `options.enabled` is `false` this is an inert no-op returning an
    /// empty [`SimilarityReport`].
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::Storage` if the backend rejects an edge write.
    pub fn rebuild_similarity_links(
        &mut self,
        options: &SimilarityOptions,
    ) -> Result<SimilarityReport> {
        if !options.enabled {
            return Ok(SimilarityReport {
                facts_processed: 0,
                links_created: 0,
            });
        }

        let facts = self.get_all_facts(usize::MAX);
        let mut report = SimilarityReport {
            facts_processed: facts.len(),
            links_created: 0,
        };

        for i in 0..facts.len() {
            let source = &facts[i];
            let source_map = fact_to_sim_map(source);

            let candidates: Vec<&SemanticFact> = facts
                .iter()
                .filter(|f| f.node_id != source.node_id)
                .take(options.candidate_limit)
                .collect();

            for candidate in candidates {
                if self.similar_pair_exists(&source.node_id, &candidate.node_id) {
                    continue;
                }
                let score = compute_similarity(&source_map, &fact_to_sim_map(candidate));
                if score >= options.threshold {
                    self.link_similar_pair(&source.node_id, &candidate.node_id, score, options)?;
                    report.links_created += 1;
                }
            }
        }

        Ok(report)
    }

    /// Store a semantic fact and optionally auto-link it on store.
    ///
    /// Identical to
    /// [`store_fact_with_provenance`](Self::store_fact_with_provenance) for the
    /// storage and provenance behaviour, then — when
    /// `options.similarity = Some(opts)` and `opts.enabled` — runs
    /// [`auto_link_similar_facts`](Self::auto_link_similar_facts) on the new
    /// fact. With the default `options.similarity = None` no `SIMILAR_TO` edges
    /// are created, so this is a backward-compatible superset of
    /// [`store_fact`](Self::store_fact).
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::InvalidInput` if `confidence` is out of range, or
    /// `MemoryError::Storage` if the node, a provenance edge, or a similarity
    /// edge cannot be persisted.
    #[allow(clippy::too_many_arguments)]
    pub fn store_fact_with_options(
        &mut self,
        concept: &str,
        content: &str,
        confidence: f64,
        source_id: &str,
        tags: Option<&[String]>,
        metadata: Option<&HashMap<String, serde_json::Value>>,
        source_episode_ids: &[String],
        options: &StoreFactOptions,
    ) -> Result<String> {
        let node_id = self.store_fact_with_provenance(
            concept,
            content,
            confidence,
            source_id,
            tags,
            metadata,
            source_episode_ids,
        )?;

        if let Some(sim_opts) = &options.similarity {
            if sim_opts.enabled {
                self.auto_link_similar_facts(&node_id, sim_opts)?;
            }
        }

        Ok(node_id)
    }
}
