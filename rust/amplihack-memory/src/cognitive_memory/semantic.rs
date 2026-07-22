//! Semantic memory methods for [`CognitiveMemory`].

use std::collections::HashMap;

use chrono::{DateTime, Utc};

use crate::memory_types::SemanticFact;
use crate::{MemoryError, Result};

use tracing::warn;

use super::converters::node_to_fact;
use super::dedup::compute_content_hash;
use super::types::{agent_filter, new_id, ts_now, ET_DERIVES_FROM, ET_SIMILAR_TO, NT_SEMANTIC};
use super::CognitiveMemory;

impl CognitiveMemory {
    /// Store a semantic fact.
    ///
    /// Vector embeddings are not available in the Rust port; keyword search
    /// is used for retrieval.
    ///
    /// Backward-compatible: this is exactly
    /// [`store_fact_with_provenance`](Self::store_fact_with_provenance) with no
    /// source episodes, so it never creates provenance edges.
    pub fn store_fact(
        &mut self,
        concept: &str,
        content: &str,
        confidence: f64,
        source_id: &str,
        tags: Option<&[String]>,
        metadata: Option<&HashMap<String, serde_json::Value>>,
    ) -> Result<String> {
        self.store_fact_with_provenance(
            concept,
            content,
            confidence,
            source_id,
            tags,
            metadata,
            &[],
        )
    }

    /// Store a semantic fact and link it to the episodes it was derived from.
    ///
    /// Identical to [`store_fact`](Self::store_fact) but additionally creates a
    /// `DERIVES_FROM` edge from the new fact node to each id in
    /// `source_episode_ids`, turning the flat fact store into a connected
    /// provenance graph. `source_id` remains an opaque string property and is
    /// never auto-converted into an edge.
    ///
    /// Lenient: a source-episode id that does not resolve to an existing
    /// [`EpisodicMemory`](crate::memory_types::EpisodicMemory) node is skipped
    /// with a warning rather than failing the call. Use
    /// [`store_fact_with_provenance_strict`](Self::store_fact_with_provenance_strict)
    /// to reject missing episodes instead.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::InvalidInput` if `confidence` is outside
    /// `0.0..=1.0`, or `MemoryError::Storage` if the node or an edge cannot be
    /// persisted.
    #[allow(clippy::too_many_arguments)]
    pub fn store_fact_with_provenance(
        &mut self,
        concept: &str,
        content: &str,
        confidence: f64,
        source_id: &str,
        tags: Option<&[String]>,
        metadata: Option<&HashMap<String, serde_json::Value>>,
        source_episode_ids: &[String],
    ) -> Result<String> {
        self.store_fact_with_provenance_inner(
            concept,
            content,
            confidence,
            source_id,
            tags,
            metadata,
            source_episode_ids,
            false,
            None,
            None,
            None,
            None,
        )
    }

    /// Strict variant of
    /// [`store_fact_with_provenance`](Self::store_fact_with_provenance): any id
    /// in `source_episode_ids` that is not an existing episode node makes the
    /// whole call fail with `MemoryError::InvalidInput` and write zero edges
    /// (validate-then-emit atomicity).
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::InvalidInput` if `confidence` is out of range or
    /// any source episode is missing, or `MemoryError::Storage` on a backend
    /// failure.
    #[allow(clippy::too_many_arguments)]
    pub fn store_fact_with_provenance_strict(
        &mut self,
        concept: &str,
        content: &str,
        confidence: f64,
        source_id: &str,
        tags: Option<&[String]>,
        metadata: Option<&HashMap<String, serde_json::Value>>,
        source_episode_ids: &[String],
    ) -> Result<String> {
        self.store_fact_with_provenance_inner(
            concept,
            content,
            confidence,
            source_id,
            tags,
            metadata,
            source_episode_ids,
            true,
            None,
            None,
            None,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn store_fact_with_provenance_inner(
        &mut self,
        concept: &str,
        content: &str,
        confidence: f64,
        source_id: &str,
        tags: Option<&[String]>,
        metadata: Option<&HashMap<String, serde_json::Value>>,
        source_episode_ids: &[String],
        strict: bool,
        importance: Option<f64>,
        expires_at: Option<DateTime<Utc>>,
        dedup_key: Option<String>,
        forced_id: Option<&str>,
    ) -> Result<String> {
        if confidence.is_nan() || !(0.0..=1.0).contains(&confidence) {
            return Err(MemoryError::InvalidInput(
                "confidence must be between 0.0 and 1.0".into(),
            ));
        }

        // Validate-then-emit: in strict mode reject before any write so a
        // failure leaves neither a node nor any provenance edge behind.
        if strict {
            for ep in source_episode_ids {
                if !self.is_source_episode(ep) {
                    return Err(MemoryError::InvalidInput(format!(
                        "source episode {ep} not found"
                    )));
                }
            }
        }

        // `forced_id` makes the fenced applier's effect replay-safe (F2): a
        // deterministic `sem_{intent_id}` collapses a crash-window replay onto
        // the SAME node via the store's same-PK upsert instead of minting a
        // duplicate. Normal callers pass `None` and get a fresh id.
        let node_id = forced_id.map(String::from).unwrap_or_else(|| new_id("sem"));
        let now = ts_now();
        let tags_json = tags
            .map(|t| {
                serde_json::to_string(t).unwrap_or_else(|e| {
                    warn!("store_fact: failed to serialize tags: {e}");
                    "[]".into()
                })
            })
            .unwrap_or_else(|| "[]".into());
        let meta_json = metadata
            .map(|m| {
                serde_json::to_string(m).unwrap_or_else(|e| {
                    warn!("store_fact: failed to serialize metadata: {e}");
                    "{}".into()
                })
            })
            .unwrap_or_else(|| "{}".into());

        let mut props = HashMap::new();
        props.insert("node_id".to_string(), node_id.clone());
        props.insert("agent_id".to_string(), self.agent_name.clone());
        props.insert("concept".to_string(), concept.to_string());
        props.insert("content".to_string(), content.to_string());
        props.insert("confidence".to_string(), confidence.to_string());
        props.insert("source_id".to_string(), source_id.to_string());
        props.insert("tags".to_string(), tags_json);
        props.insert("metadata".to_string(), meta_json);
        props.insert("created_at".to_string(), now.to_string());

        // Lifecycle / dedup props. Written on every store so the columns exist
        // from the first write (the persistent backend creates them lazily) and
        // legacy readers see explicit defaults.
        props.insert(
            "importance".to_string(),
            importance.unwrap_or(confidence).to_string(),
        );
        props.insert("usage_count".to_string(), "0".to_string());
        props.insert("archived".to_string(), "false".to_string());
        props.insert(
            "content_hash".to_string(),
            compute_content_hash(concept, content),
        );
        props.insert("last_accessed_at".to_string(), String::new());
        props.insert(
            "expires_at".to_string(),
            expires_at
                .map(|d| d.timestamp().to_string())
                .unwrap_or_default(),
        );
        props.insert("superseded_by".to_string(), String::new());
        props.insert("dedup_key".to_string(), dedup_key.unwrap_or_default());

        self.graph
            .add_node(NT_SEMANTIC, props, Some(&node_id))
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        for ep in source_episode_ids {
            if self.is_source_episode(ep) {
                self.add_provenance_edge(&node_id, ep, ET_DERIVES_FROM)?;
            } else {
                warn!(
                    "store_fact_with_provenance: source episode {ep} not found; \
                     skipping DERIVES_FROM edge"
                );
            }
        }

        Ok(node_id)
    }

    /// Search semantic facts using keyword matching.
    ///
    /// Words from `query` are matched against both `concept` and `content`
    /// fields (case-insensitive). Results are filtered by `min_confidence`
    /// and sorted by confidence descending.
    pub fn search_facts(
        &self,
        query: &str,
        limit: usize,
        min_confidence: f64,
    ) -> Vec<SemanticFact> {
        let keywords: Vec<String> = query
            .split_whitespace()
            .filter(|w| !w.is_empty())
            .map(|w| w.to_lowercase())
            .collect();

        if keywords.is_empty() {
            return self.get_all_facts(limit);
        }

        let filter = agent_filter(&self.agent_name);
        let nodes = self
            .graph
            .query_nodes(NT_SEMANTIC, Some(&filter), usize::MAX);

        let mut facts: Vec<SemanticFact> = nodes
            .into_iter()
            .filter(|n| {
                let conf: f64 = n
                    .properties
                    .get("confidence")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0.0);
                if conf < min_confidence {
                    return false;
                }
                let concept = n
                    .properties
                    .get("concept")
                    .map(|s| s.to_lowercase())
                    .unwrap_or_default();
                let content = n
                    .properties
                    .get("content")
                    .map(|s| s.to_lowercase())
                    .unwrap_or_default();

                keywords
                    .iter()
                    .any(|kw| concept.contains(kw) || content.contains(kw))
            })
            .map(|n| node_to_fact(&n.properties))
            .collect();

        facts.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        facts.truncate(limit);
        facts
    }

    /// Return all semantic facts for this agent, sorted by confidence descending.
    pub fn get_all_facts(&self, limit: usize) -> Vec<SemanticFact> {
        let filter = agent_filter(&self.agent_name);
        let nodes = self
            .graph
            .query_nodes(NT_SEMANTIC, Some(&filter), usize::MAX);

        let mut facts: Vec<SemanticFact> = nodes
            .into_iter()
            .map(|n| node_to_fact(&n.properties))
            .collect();

        facts.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        facts.truncate(limit);
        facts
    }

    /// Link two semantic facts with a SIMILAR_TO edge.
    ///
    /// Idempotent: a repeat call (including a fenced-applier replay of a
    /// `LinkSimilarFacts` intent across the F2 crash window) does not add a
    /// duplicate edge when one already links the pair in this direction.
    pub fn link_similar_facts(
        &mut self,
        fact_id_a: &str,
        fact_id_b: &str,
        similarity_score: f64,
    ) -> Result<()> {
        if self.edge_of_type_exists(fact_id_a, fact_id_b, ET_SIMILAR_TO) {
            return Ok(());
        }
        let mut props = HashMap::new();
        props.insert("similarity_score".to_string(), similarity_score.to_string());
        self.graph
            .add_edge(fact_id_a, fact_id_b, ET_SIMILAR_TO, Some(props))
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Fenced-applier entry point for a `StoreFact` intent: store the fact under
    /// a DETERMINISTIC node id (`sem_{intent_id}`) so a crash-window replay
    /// upserts the same node instead of minting a duplicate (F2 exactly-once).
    ///
    /// # Errors
    /// As [`store_fact`](Self::store_fact).
    #[cfg(feature = "persistent")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_store_fact(
        &mut self,
        intent_id: uuid::Uuid,
        concept: &str,
        content: &str,
        confidence: f64,
        source_id: &str,
        tags: Option<&[String]>,
        metadata: Option<&HashMap<String, serde_json::Value>>,
    ) -> Result<String> {
        let forced = format!("sem_{intent_id}");
        self.store_fact_with_provenance_inner(
            concept,
            content,
            confidence,
            source_id,
            tags,
            metadata,
            &[],
            false,
            None,
            None,
            None,
            Some(&forced),
        )
    }

    /// Link a semantic fact to its source episode.
    pub fn link_fact_to_episode(&mut self, fact_id: &str, episode_id: &str) -> Result<()> {
        let now = ts_now();
        let mut props = HashMap::new();
        props.insert("derived_at".to_string(), now.to_string());
        self.graph
            .add_edge(fact_id, episode_id, ET_DERIVES_FROM, Some(props))
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Link a fact to several source episodes at once, returning how many edges
    /// were actually created.
    ///
    /// Lenient, mirroring
    /// [`store_fact_with_provenance`](Self::store_fact_with_provenance): each id
    /// in `episode_ids` that resolves to an existing episode node gets a
    /// `DERIVES_FROM` edge from `fact_id`; ids that do not are skipped with a
    /// warning. The returned count reflects only the edges created.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::Storage` if the backend rejects an edge whose
    /// endpoints both exist (e.g. `fact_id` itself is missing).
    pub fn link_fact_to_episodes(
        &mut self,
        fact_id: &str,
        episode_ids: &[String],
    ) -> Result<usize> {
        let mut created = 0;
        for ep in episode_ids {
            if self.is_source_episode(ep) {
                self.add_provenance_edge(fact_id, ep, ET_DERIVES_FROM)?;
                created += 1;
            } else {
                warn!("link_fact_to_episodes: source episode {ep} not found; skipping");
            }
        }
        Ok(created)
    }

    /// Return the ids of the episodes a fact was derived from.
    ///
    /// Reads the `DERIVES_FROM` provenance edges outgoing from `fact_id`.
    /// Returns an empty vector for an unknown id or a fact with no provenance.
    pub fn fact_provenance(&self, fact_id: &str) -> Vec<String> {
        self.provenance_targets(fact_id, ET_DERIVES_FROM)
    }
}
