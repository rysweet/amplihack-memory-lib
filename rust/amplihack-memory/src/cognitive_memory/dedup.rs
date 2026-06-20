//! Fact deduplication, supersession, and retention for [`CognitiveMemory`].
//!
//! This module adds three capabilities on top of the semantic-fact store so
//! consumers can keep the fact graph from accumulating unbounded duplicate or
//! stale knowledge:
//!
//! * **Dedup** — [`upsert_fact`](CognitiveMemory::upsert_fact) stores a fact
//!   but first checks, per a [`DedupMode`], whether an equivalent live fact
//!   already exists. A hit is either *reused* (confidence/usage refreshed) or
//!   *superseded* (a new version replaces the old) instead of inserting a
//!   duplicate. [`find_duplicate_facts`](CognitiveMemory::find_duplicate_facts)
//!   reports existing duplicates without changing anything.
//! * **Supersession** — [`supersede_fact`](CognitiveMemory::supersede_fact)
//!   archives an old fact, records a `superseded_by` back-pointer, and links the
//!   replacement to it with a `SUPERSEDES` edge.
//! * **Retention** — [`prune_semantic_memory`](CognitiveMemory::prune_semantic_memory)
//!   applies a [`RetentionPolicy`] using an archive-before-delete lifecycle:
//!   the first pass archives candidates, a later pass deletes the now-archived
//!   ones. High-importance, provenance-bearing facts are protected from
//!   deletion.
//!
//! All entry points are additive; the legacy `store_fact*` / `get_all_facts`
//! APIs keep their existing behaviour and the new fact fields default cleanly
//! for facts written before this module existed.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Utc};
use sha2::{Digest, Sha256};

use crate::graph::types::Direction;
use crate::memory_types::SemanticFact;
use crate::similarity::compute_similarity;
use crate::{MemoryError, Result};

use super::similarity::SimilarityOptions;
use super::types::{ts_now, ET_SUPERSEDES, NT_SEMANTIC};
use super::CognitiveMemory;

// ---------------------------------------------------------------------------
// content hashing
// ---------------------------------------------------------------------------

/// Compute the deterministic content hash for a fact.
///
/// The hash is the lowercase hex SHA-256 of `concept`, a `0x1F` unit-separator
/// byte, then `content`. The separator makes the concept/content boundary part
/// of the hash so e.g. `("ab", "c")` and `("a", "bc")` do not collide. Content
/// is hashed verbatim (no trimming or case-folding) so dedup is exact.
pub fn compute_content_hash(concept: &str, content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(concept.as_bytes());
    hasher.update([0x1f]);
    hasher.update(content.as_bytes());
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for b in digest {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

// ---------------------------------------------------------------------------
// option / input / outcome types
// ---------------------------------------------------------------------------

/// Strategy for detecting that a fact being stored duplicates an existing one.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum DedupMode {
    /// No deduplication: every upsert inserts a fresh fact.
    #[default]
    None,
    /// Treat facts with an identical `concept + content` hash as duplicates.
    ExactContentHash,
    /// Treat facts sharing the given caller-supplied key as the same logical
    /// fact (a new version with changed content supersedes the old one).
    CallerKey(String),
    /// Treat a sufficiently similar fact (optionally restricted to the same
    /// concept) as a duplicate, using the deterministic composite similarity.
    SameConceptSimilarity,
}

/// Tuning knobs controlling dedup detection.
///
/// [`Default`] disables deduplication (`mode = None`) with a `0.60` similarity
/// threshold and same-concept restriction, so it is a safe no-op default.
#[derive(Debug, Clone, PartialEq)]
pub struct DedupOptions {
    /// The duplicate-detection strategy.
    pub mode: DedupMode,
    /// Minimum composite similarity for `SameConceptSimilarity` to treat two
    /// facts as duplicates (a pair matches when `score >= threshold`).
    pub similarity_threshold: f64,
    /// When `true`, only same-concept facts are similarity-dedup candidates.
    pub same_concept_only: bool,
}

impl Default for DedupOptions {
    fn default() -> Self {
        Self {
            mode: DedupMode::None,
            similarity_threshold: 0.60,
            same_concept_only: true,
        }
    }
}

/// Provenance inputs for an [`upsert_fact`](CognitiveMemory::upsert_fact) insert.
///
/// [`Default`] links no episodes and is lenient (`strict = false`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ProvenanceOptions {
    /// Episode ids the new fact derives from (each gets a `DERIVES_FROM` edge).
    pub source_episode_ids: Vec<String>,
    /// When `true`, a missing source episode fails the whole insert atomically
    /// (mirrors `store_fact_with_provenance_strict`).
    pub strict: bool,
}

/// Options bundle for [`upsert_fact`](CognitiveMemory::upsert_fact) and
/// [`store_fact_with_options`](CognitiveMemory::store_fact_with_options).
///
/// [`Default`] performs a plain insert: no similarity linking, no provenance
/// edges, and no deduplication — observably identical to
/// [`store_fact`](CognitiveMemory::store_fact).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StoreFactOptions {
    /// When `Some(opts)` (and `opts.enabled`), a freshly inserted fact is
    /// auto-linked to its above-threshold neighbours.
    pub similarity: Option<SimilarityOptions>,
    /// Provenance edges to create for an inserted fact.
    pub provenance: ProvenanceOptions,
    /// Deduplication behaviour applied before insertion.
    pub dedup: DedupOptions,
}

/// A fact to store via [`upsert_fact`](CognitiveMemory::upsert_fact).
///
/// Construct with [`FactInput::new`] and set the optional fields as needed.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FactInput {
    /// The concept / topic key.
    pub concept: String,
    /// The factual content.
    pub content: String,
    /// Confidence in `[0.0, 1.0]`.
    pub confidence: f64,
    /// Opaque source identifier (stored as a string property).
    pub source_id: String,
    /// Categorical tags for similarity linking and filtering.
    pub tags: Vec<String>,
    /// Arbitrary metadata.
    pub metadata: HashMap<String, serde_json::Value>,
    /// Optional importance override; defaults to `confidence` when `None`.
    pub importance: Option<f64>,
    /// Optional expiry instant used by retention.
    pub expires_at: Option<DateTime<Utc>>,
    /// Optional caller dedup key (used/overwritten by `DedupMode::CallerKey`).
    pub dedup_key: Option<String>,
}

impl FactInput {
    /// Create a `FactInput` with the required fields; the rest default.
    pub fn new(concept: impl Into<String>, content: impl Into<String>, confidence: f64) -> Self {
        Self {
            concept: concept.into(),
            content: content.into(),
            confidence,
            ..Default::default()
        }
    }
}

/// What an [`upsert_fact`](CognitiveMemory::upsert_fact) call did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DedupAction {
    /// A new fact node was created.
    Inserted,
    /// An existing live fact was reused (confidence/usage refreshed).
    Reused {
        /// The id of the reused fact.
        existing_id: String,
    },
    /// A new fact superseded an existing one.
    Superseded {
        /// The archived, superseded fact.
        old_id: String,
        /// The new replacement fact.
        new_id: String,
    },
}

/// Result of an [`upsert_fact`](CognitiveMemory::upsert_fact) call.
#[derive(Debug, Clone, PartialEq)]
pub struct StoreFactOutcome {
    /// The id of the live fact after the call (inserted/reused/new replacement).
    pub node_id: String,
    /// What happened.
    pub dedup_action: DedupAction,
    /// The content hash of the input fact.
    pub content_hash: String,
    /// How many `SIMILAR_TO` links were created (0 unless inserted with
    /// similarity enabled).
    pub similarity_links_created: usize,
    /// How many `DERIVES_FROM` provenance edges were created.
    pub provenance_edges_created: usize,
}

/// A group of facts detected as duplicates of one another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DuplicateFactGroup {
    /// The grouping key (content hash, caller key, or representative id).
    pub key: String,
    /// The member fact ids, oldest first.
    pub fact_ids: Vec<String>,
    /// The representative (oldest) fact id.
    pub representative_id: String,
}

/// Policy controlling [`prune_semantic_memory`](CognitiveMemory::prune_semantic_memory).
///
/// [`Default`] is a no-op-ish policy: no per-concept cap, no TTLs, keep
/// everything with importance `>= 0.0`, ignore superseded facts, and actually
/// mutate (`dry_run = false`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RetentionPolicy {
    /// Keep at most this many live facts per concept (highest importance wins).
    pub max_facts_per_concept: Option<usize>,
    /// Per-concept max age in seconds; a fact older than its concept's TTL is a
    /// prune candidate.
    pub ttl_seconds_by_concept: HashMap<String, i64>,
    /// Facts with importance below this threshold are prune candidates.
    pub min_importance_to_keep: f64,
    /// When `true`, superseded (archived) facts are also prune candidates.
    pub include_superseded: bool,
    /// When `true`, only report counts; make no changes.
    pub dry_run: bool,
}

/// Summary of a [`prune_semantic_memory`](CognitiveMemory::prune_semantic_memory)
/// pass. For a real run the `would_*` counts are zero; for a `dry_run` the
/// `archived` / `deleted` counts are zero.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PruneReport {
    /// Facts archived this pass.
    pub archived: usize,
    /// Facts deleted this pass.
    pub deleted: usize,
    /// Facts that would be archived (dry run).
    pub would_archive: usize,
    /// Facts that would be deleted (dry run).
    pub would_delete: usize,
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Build the property map [`compute_similarity`] expects from raw fields.
fn sim_map(concept: &str, content: &str, tags: &[String]) -> HashMap<String, serde_json::Value> {
    let mut m = HashMap::with_capacity(3);
    m.insert(
        "content".to_string(),
        serde_json::Value::String(content.to_string()),
    );
    m.insert(
        "concept".to_string(),
        serde_json::Value::String(concept.to_string()),
    );
    m.insert("tags".to_string(), serde_json::json!(tags));
    m
}

/// Age basis for retention: the last access time, falling back to creation.
fn age_basis_ts(f: &SemanticFact) -> i64 {
    f.last_accessed_at.unwrap_or(f.created_at).timestamp()
}

/// Build a [`DuplicateFactGroup`] from members, ordered oldest-first.
fn make_group(key: String, mut members: Vec<SemanticFact>) -> DuplicateFactGroup {
    members.sort_by(|a, b| {
        a.created_at
            .cmp(&b.created_at)
            .then_with(|| a.node_id.cmp(&b.node_id))
    });
    let representative_id = members
        .first()
        .map(|f| f.node_id.clone())
        .unwrap_or_default();
    let fact_ids = members.into_iter().map(|f| f.node_id).collect();
    DuplicateFactGroup {
        key,
        fact_ids,
        representative_id,
    }
}

/// Group facts by a derived key, dropping facts whose key is `None`.
fn group_by_key(
    facts: &[SemanticFact],
    key_of: impl Fn(&SemanticFact) -> Option<String>,
) -> Vec<DuplicateFactGroup> {
    let mut map: HashMap<String, Vec<SemanticFact>> = HashMap::new();
    for f in facts {
        if let Some(k) = key_of(f) {
            map.entry(k).or_default().push(f.clone());
        }
    }
    map.into_iter()
        .map(|(key, members)| make_group(key, members))
        .collect()
}

// ---------------------------------------------------------------------------
// CognitiveMemory methods
// ---------------------------------------------------------------------------

impl CognitiveMemory {
    /// Store a fact, deduplicating against existing live facts per `options.dedup`.
    ///
    /// Depending on the [`DedupMode`] and whether a matching live fact exists,
    /// this either **inserts** a new fact (creating any provenance / similarity
    /// edges from `options`), **reuses** an existing fact (bumping its usage
    /// count, refreshing `last_accessed_at`, and updating its confidence while
    /// leaving importance untouched), or **supersedes** an existing fact with a
    /// new version. Only non-archived facts are dedup candidates.
    ///
    /// With the default `options.dedup.mode = None` every call inserts, making
    /// this a backward-compatible superset of
    /// [`store_fact`](Self::store_fact).
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::InvalidInput` if `confidence` is out of range (or,
    /// in strict provenance mode, a source episode is missing), or
    /// `MemoryError::Storage` if a node or edge cannot be persisted.
    pub fn upsert_fact(
        &mut self,
        input: FactInput,
        options: &StoreFactOptions,
    ) -> Result<StoreFactOutcome> {
        if input.confidence.is_nan() || !(0.0..=1.0).contains(&input.confidence) {
            return Err(MemoryError::InvalidInput(
                "confidence must be between 0.0 and 1.0".into(),
            ));
        }

        let content_hash = compute_content_hash(&input.concept, &input.content);
        let hit = self.find_dedup_hit(&input, &content_hash, &options.dedup);

        match (&options.dedup.mode, hit) {
            (DedupMode::CallerKey(k), Some(existing)) => {
                if existing.content_hash == content_hash {
                    self.reuse_fact(existing, &input, content_hash)
                } else {
                    let key = k.clone();
                    self.supersede_with_new(existing, input, options, content_hash, key)
                }
            }
            (DedupMode::CallerKey(k), None) => {
                let mut input = input;
                input.dedup_key = Some(k.clone());
                self.insert_fact(input, options, content_hash)
            }
            (DedupMode::ExactContentHash, Some(existing))
            | (DedupMode::SameConceptSimilarity, Some(existing)) => {
                self.reuse_fact(existing, &input, content_hash)
            }
            _ => self.insert_fact(input, options, content_hash),
        }
    }

    /// Archive `old_id`, point it at its replacement `new_id`, and link the two
    /// with a `SUPERSEDES` edge (`new -> old`) carrying `reason`.
    ///
    /// The old fact's `superseded_by` is set and `archived` becomes `true`. The
    /// `SUPERSEDES` edge is created once: a repeat call is idempotent on the
    /// edge. Both ids must resolve to existing semantic-fact nodes and must
    /// differ.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::InvalidInput` if either id is missing or they are
    /// equal, or `MemoryError::Storage` if the archive update or edge write
    /// fails.
    pub fn supersede_fact(&mut self, old_id: &str, new_id: &str, reason: &str) -> Result<()> {
        if old_id == new_id {
            return Err(MemoryError::InvalidInput(
                "a fact cannot supersede itself".into(),
            ));
        }
        if !self.is_semantic_fact(old_id) {
            return Err(MemoryError::InvalidInput(format!(
                "fact {old_id} not found"
            )));
        }
        if !self.is_semantic_fact(new_id) {
            return Err(MemoryError::InvalidInput(format!(
                "fact {new_id} not found"
            )));
        }

        let mut props = HashMap::new();
        props.insert("superseded_by".to_string(), new_id.to_string());
        props.insert("archived".to_string(), "true".to_string());
        if !self.graph.update_node(old_id, props) {
            return Err(MemoryError::Storage(format!(
                "failed to archive superseded fact {old_id}"
            )));
        }

        let already = self
            .graph
            .query_neighbors(new_id, Some(ET_SUPERSEDES), Direction::Outgoing, usize::MAX)
            .iter()
            .any(|(_, n)| n.node_id == old_id);
        if !already {
            let mut eprops = HashMap::new();
            eprops.insert("reason".to_string(), reason.to_string());
            eprops.insert("superseded_at".to_string(), ts_now().to_string());
            self.graph
                .add_edge(new_id, old_id, ET_SUPERSEDES, Some(eprops))
                .map_err(|e| MemoryError::Storage(e.to_string()))?;
        }

        Ok(())
    }

    /// Report groups of live facts that duplicate one another under `options`.
    ///
    /// Grouping key depends on the [`DedupMode`]: `ExactContentHash` groups by
    /// content hash, `CallerKey` groups by non-empty dedup key, and
    /// `SameConceptSimilarity` clusters above-threshold pairs (optionally
    /// same-concept). `None` returns no groups. Only groups with at least two
    /// members are returned, sorted by size descending; `limit` (when non-zero)
    /// caps the number of groups. This is read-only: nothing is mutated.
    pub fn find_duplicate_facts(
        &self,
        options: &DedupOptions,
        limit: usize,
    ) -> Vec<DuplicateFactGroup> {
        let live: Vec<SemanticFact> = self
            .get_all_facts(usize::MAX)
            .into_iter()
            .filter(|f| !f.archived)
            .collect();

        let mut groups = match &options.mode {
            DedupMode::None => return Vec::new(),
            DedupMode::ExactContentHash => group_by_key(&live, |f| Some(f.content_hash.clone())),
            DedupMode::CallerKey(_) => {
                group_by_key(&live, |f| f.dedup_key.clone().filter(|k| !k.is_empty()))
            }
            DedupMode::SameConceptSimilarity => self.group_by_similarity(&live, options),
        };

        groups.retain(|g| g.fact_ids.len() >= 2);
        groups.sort_by(|a, b| {
            b.fact_ids
                .len()
                .cmp(&a.fact_ids.len())
                .then_with(|| a.key.cmp(&b.key))
        });
        if limit > 0 {
            groups.truncate(limit);
        }
        groups
    }

    /// Apply a [`RetentionPolicy`] to this agent's facts, archive-before-delete.
    ///
    /// A fact is a prune *candidate* if any policy trigger fires: importance
    /// below `min_importance_to_keep`, a past `expires_at`, an exceeded
    /// per-concept TTL, being over the per-concept cap (lowest-importance
    /// excess), or — with `include_superseded` — being superseded.
    ///
    /// Each candidate is handled in two tiers: a not-yet-archived candidate is
    /// archived; an already-archived candidate is deleted. Thus one pass
    /// archives fresh candidates and a later pass deletes them. A fact with at
    /// least one `DERIVES_FROM` edge **and** importance at or above the keep
    /// threshold is protected from deletion (it may still be archived). With
    /// `dry_run`, only the `would_*` counts are populated and nothing changes.
    ///
    /// # Errors
    ///
    /// Currently infallible, but returns `Result` for forward compatibility
    /// with backends that report storage errors.
    pub fn prune_semantic_memory(&mut self, policy: &RetentionPolicy) -> Result<PruneReport> {
        let facts = self.get_all_facts(usize::MAX);
        let now = ts_now();
        let excess = self.compute_excess(&facts, policy);
        let mut report = PruneReport::default();

        for f in &facts {
            if !self.prune_candidate(f, policy, now, &excess) {
                continue;
            }
            if !f.archived {
                if policy.dry_run {
                    report.would_archive += 1;
                } else {
                    let mut props = HashMap::new();
                    props.insert("archived".to_string(), "true".to_string());
                    if self.graph.update_node(&f.node_id, props) {
                        report.archived += 1;
                    }
                }
            } else if !self.protected_from_delete(f, policy) {
                if policy.dry_run {
                    report.would_delete += 1;
                } else if self.graph.delete_node(&f.node_id) {
                    report.deleted += 1;
                }
            }
        }

        Ok(report)
    }

    // -- internals --

    fn is_semantic_fact(&self, id: &str) -> bool {
        self.graph
            .get_node(id)
            .is_some_and(|n| n.node_type == NT_SEMANTIC)
    }

    /// Find a live fact this input duplicates, per the dedup mode.
    fn find_dedup_hit(
        &self,
        input: &FactInput,
        content_hash: &str,
        dedup: &DedupOptions,
    ) -> Option<SemanticFact> {
        let live = self
            .get_all_facts(usize::MAX)
            .into_iter()
            .filter(|f| !f.archived);

        match &dedup.mode {
            DedupMode::None => None,
            DedupMode::ExactContentHash => {
                live.into_iter().find(|f| f.content_hash == content_hash)
            }
            DedupMode::CallerKey(k) => live
                .into_iter()
                .find(|f| f.dedup_key.as_deref() == Some(k.as_str())),
            DedupMode::SameConceptSimilarity => {
                let src = sim_map(&input.concept, &input.content, &input.tags);
                live.into_iter().find(|f| {
                    if dedup.same_concept_only && f.concept != input.concept {
                        return false;
                    }
                    compute_similarity(&src, &sim_map(&f.concept, &f.content, &f.tags))
                        >= dedup.similarity_threshold
                })
            }
        }
    }

    fn insert_fact(
        &mut self,
        input: FactInput,
        options: &StoreFactOptions,
        content_hash: String,
    ) -> Result<StoreFactOutcome> {
        let node_id = self.store_fact_with_provenance_inner(
            &input.concept,
            &input.content,
            input.confidence,
            &input.source_id,
            Some(input.tags.as_slice()),
            Some(&input.metadata),
            &options.provenance.source_episode_ids,
            options.provenance.strict,
            input.importance,
            input.expires_at,
            input.dedup_key.clone(),
        )?;

        let provenance_edges_created = self.fact_provenance(&node_id).len();
        let similarity_links_created = self.auto_link_on_insert(&node_id, options)?;

        Ok(StoreFactOutcome {
            node_id,
            dedup_action: DedupAction::Inserted,
            content_hash,
            similarity_links_created,
            provenance_edges_created,
        })
    }

    fn reuse_fact(
        &mut self,
        existing: SemanticFact,
        input: &FactInput,
        content_hash: String,
    ) -> Result<StoreFactOutcome> {
        let mut props = HashMap::new();
        props.insert(
            "usage_count".to_string(),
            (existing.usage_count + 1).to_string(),
        );
        props.insert("last_accessed_at".to_string(), ts_now().to_string());
        props.insert("confidence".to_string(), input.confidence.to_string());
        if !self.graph.update_node(&existing.node_id, props) {
            return Err(MemoryError::Storage(format!(
                "failed to update reused fact {}",
                existing.node_id
            )));
        }

        Ok(StoreFactOutcome {
            node_id: existing.node_id.clone(),
            dedup_action: DedupAction::Reused {
                existing_id: existing.node_id,
            },
            content_hash,
            similarity_links_created: 0,
            provenance_edges_created: 0,
        })
    }

    fn supersede_with_new(
        &mut self,
        old: SemanticFact,
        mut input: FactInput,
        options: &StoreFactOptions,
        content_hash: String,
        key: String,
    ) -> Result<StoreFactOutcome> {
        input.dedup_key = Some(key);
        let node_id = self.store_fact_with_provenance_inner(
            &input.concept,
            &input.content,
            input.confidence,
            &input.source_id,
            Some(input.tags.as_slice()),
            Some(&input.metadata),
            &options.provenance.source_episode_ids,
            options.provenance.strict,
            input.importance,
            input.expires_at,
            input.dedup_key.clone(),
        )?;

        let provenance_edges_created = self.fact_provenance(&node_id).len();
        self.supersede_fact(&old.node_id, &node_id, "callerkey")?;
        let similarity_links_created = self.auto_link_on_insert(&node_id, options)?;

        Ok(StoreFactOutcome {
            node_id: node_id.clone(),
            dedup_action: DedupAction::Superseded {
                old_id: old.node_id,
                new_id: node_id,
            },
            content_hash,
            similarity_links_created,
            provenance_edges_created,
        })
    }

    fn auto_link_on_insert(&mut self, node_id: &str, options: &StoreFactOptions) -> Result<usize> {
        if let Some(sim) = &options.similarity {
            if sim.enabled {
                return self.auto_link_similar_facts(node_id, sim);
            }
        }
        Ok(0)
    }

    fn group_by_similarity(
        &self,
        facts: &[SemanticFact],
        options: &DedupOptions,
    ) -> Vec<DuplicateFactGroup> {
        let mut used = vec![false; facts.len()];
        let mut groups = Vec::new();

        for i in 0..facts.len() {
            if used[i] {
                continue;
            }
            used[i] = true;
            let mut members = vec![facts[i].clone()];
            let src = sim_map(&facts[i].concept, &facts[i].content, &facts[i].tags);

            for (j, used_j) in used.iter_mut().enumerate().skip(i + 1) {
                if *used_j {
                    continue;
                }
                if options.same_concept_only && facts[j].concept != facts[i].concept {
                    continue;
                }
                let score = compute_similarity(
                    &src,
                    &sim_map(&facts[j].concept, &facts[j].content, &facts[j].tags),
                );
                if score >= options.similarity_threshold {
                    *used_j = true;
                    members.push(facts[j].clone());
                }
            }

            if members.len() >= 2 {
                groups.push(make_group(facts[i].node_id.clone(), members));
            }
        }

        groups
    }

    /// Compute the set of fact ids that exceed `max_facts_per_concept`.
    ///
    /// Facts are ranked per concept with live facts before archived ones, then
    /// by importance descending (newer wins on ties); everything past the cap is
    /// excess. Sorting archived facts last means the cap first archives the
    /// lowest-importance live facts, then — on a subsequent pass — the
    /// now-archived excess becomes deletable (archive-before-delete) instead of
    /// accumulating forever.
    fn compute_excess(&self, facts: &[SemanticFact], policy: &RetentionPolicy) -> HashSet<String> {
        let mut excess = HashSet::new();
        let Some(max) = policy.max_facts_per_concept else {
            return excess;
        };

        let mut by_concept: HashMap<&str, Vec<&SemanticFact>> = HashMap::new();
        for f in facts {
            by_concept.entry(f.concept.as_str()).or_default().push(f);
        }

        for (_concept, mut group) in by_concept {
            if group.len() <= max {
                continue;
            }
            group.sort_by(|a, b| {
                a.archived
                    .cmp(&b.archived)
                    .then_with(|| {
                        b.importance
                            .partial_cmp(&a.importance)
                            .unwrap_or(Ordering::Equal)
                    })
                    .then_with(|| age_basis_ts(b).cmp(&age_basis_ts(a)))
            });
            for f in group.iter().skip(max) {
                excess.insert(f.node_id.clone());
            }
        }

        excess
    }

    fn prune_candidate(
        &self,
        f: &SemanticFact,
        policy: &RetentionPolicy,
        now: i64,
        excess: &HashSet<String>,
    ) -> bool {
        if f.importance < policy.min_importance_to_keep {
            return true;
        }
        if let Some(exp) = f.expires_at {
            if now >= exp.timestamp() {
                return true;
            }
        }
        if let Some(ttl) = policy.ttl_seconds_by_concept.get(&f.concept) {
            if now - age_basis_ts(f) >= *ttl {
                return true;
            }
        }
        if excess.contains(&f.node_id) {
            return true;
        }
        if policy.include_superseded && f.superseded_by.is_some() {
            return true;
        }
        false
    }

    /// A fact is protected from deletion when it carries provenance and its
    /// importance is at or above the keep threshold.
    fn protected_from_delete(&self, f: &SemanticFact, policy: &RetentionPolicy) -> bool {
        f.importance >= policy.min_importance_to_keep
            && !self.fact_provenance(&f.node_id).is_empty()
    }
}
