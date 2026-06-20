//! Procedural memory methods for [`CognitiveMemory`].

use std::collections::HashMap;

use crate::memory_types::ProceduralMemory;
use crate::{MemoryError, Result};

use tracing::warn;

use super::converters::node_to_procedural;
use super::types::{agent_filter, new_id, ts_now, ET_PROCEDURE_DERIVES_FROM, NT_PROCEDURAL};
use super::CognitiveMemory;

impl CognitiveMemory {
    /// Store a reusable procedure.
    ///
    /// Idempotent by `name` within an agent: re-storing a procedure with a name
    /// that already exists updates the existing node (steps/prerequisites) in
    /// place and returns its `node_id` rather than creating a duplicate. This
    /// matches the semantics consumers expect when re-registering procedures
    /// (e.g. on agent restart) and avoids unbounded duplicate accumulation.
    ///
    /// Backward-compatible: this is exactly
    /// [`store_procedure_with_provenance`](Self::store_procedure_with_provenance)
    /// with no source episodes, so it never creates provenance edges.
    pub fn store_procedure(
        &mut self,
        name: &str,
        steps: &[String],
        prerequisites: Option<&[String]>,
    ) -> Result<String> {
        self.store_procedure_with_provenance(name, steps, prerequisites, &[])
    }

    /// Store a procedure and link it to the episodes it was derived from.
    ///
    /// Identical to [`store_procedure`](Self::store_procedure) — including the
    /// idempotent upsert-by-name — but additionally creates a
    /// `PROCEDURE_DERIVES_FROM` edge from the (possibly pre-existing) procedure
    /// node to each id in `source_episode_ids`. Re-storing the same name reuses
    /// the canonical node, so provenance edges from multiple calls all attach to
    /// a single node rather than forking it.
    ///
    /// Lenient: a source-episode id that does not resolve to an existing
    /// [`EpisodicMemory`](crate::memory_types::EpisodicMemory) node is skipped
    /// with a warning. Use
    /// [`store_procedure_with_provenance_strict`](Self::store_procedure_with_provenance_strict)
    /// to reject missing episodes instead.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::Storage` if the node or an edge cannot be
    /// persisted.
    pub fn store_procedure_with_provenance(
        &mut self,
        name: &str,
        steps: &[String],
        prerequisites: Option<&[String]>,
        source_episode_ids: &[String],
    ) -> Result<String> {
        self.store_procedure_with_provenance_inner(
            name,
            steps,
            prerequisites,
            source_episode_ids,
            false,
        )
    }

    /// Strict variant of
    /// [`store_procedure_with_provenance`](Self::store_procedure_with_provenance):
    /// any id in `source_episode_ids` that is not an existing episode node makes
    /// the whole call fail with `MemoryError::InvalidInput` and write zero edges
    /// (and perform no upsert), giving validate-then-emit atomicity.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::InvalidInput` if any source episode is missing, or
    /// `MemoryError::Storage` on a backend failure.
    pub fn store_procedure_with_provenance_strict(
        &mut self,
        name: &str,
        steps: &[String],
        prerequisites: Option<&[String]>,
        source_episode_ids: &[String],
    ) -> Result<String> {
        self.store_procedure_with_provenance_inner(
            name,
            steps,
            prerequisites,
            source_episode_ids,
            true,
        )
    }

    fn store_procedure_with_provenance_inner(
        &mut self,
        name: &str,
        steps: &[String],
        prerequisites: Option<&[String]>,
        source_episode_ids: &[String],
        strict: bool,
    ) -> Result<String> {
        // Validate-then-emit: in strict mode reject before any write (including
        // the upsert) so a failure changes nothing.
        if strict {
            for ep in source_episode_ids {
                if !self.is_source_episode(ep) {
                    return Err(MemoryError::InvalidInput(format!(
                        "source episode {ep} not found"
                    )));
                }
            }
        }

        let node_id = self.upsert_procedure(name, steps, prerequisites)?;

        for ep in source_episode_ids {
            if self.is_source_episode(ep) {
                self.add_provenance_edge(&node_id, ep, ET_PROCEDURE_DERIVES_FROM)?;
            } else {
                warn!(
                    "store_procedure_with_provenance: source episode {ep} not found; \
                     skipping PROCEDURE_DERIVES_FROM edge"
                );
            }
        }

        Ok(node_id)
    }

    /// Idempotent upsert-by-name: update an existing procedure's
    /// steps/prerequisites in place (returning its id) or insert a new node.
    fn upsert_procedure(
        &mut self,
        name: &str,
        steps: &[String],
        prerequisites: Option<&[String]>,
    ) -> Result<String> {
        let now = ts_now();
        let steps_json = serde_json::to_string(steps).unwrap_or_else(|e| {
            warn!("store_procedure: failed to serialize steps: {e}");
            "[]".into()
        });
        let prereqs_json = prerequisites
            .map(|p| {
                serde_json::to_string(p).unwrap_or_else(|e| {
                    warn!("store_procedure: failed to serialize prerequisites: {e}");
                    "[]".into()
                })
            })
            .unwrap_or_else(|| "[]".into());

        // Idempotency: dedup by (agent_id, name). If a procedure with this name
        // already exists, update it in place instead of inserting a duplicate.
        if let Some(existing_id) = self.find_procedure_id_by_name(name) {
            let mut update = HashMap::new();
            update.insert("steps".to_string(), steps_json);
            update.insert("prerequisites".to_string(), prereqs_json);
            if !self.graph.update_node(&existing_id, update) {
                warn!("store_procedure: failed to update existing procedure {existing_id}");
            }
            return Ok(existing_id);
        }

        let node_id = new_id("proc");
        let mut props = HashMap::new();
        props.insert("node_id".to_string(), node_id.clone());
        props.insert("agent_id".to_string(), self.agent_name.clone());
        props.insert("name".to_string(), name.to_string());
        props.insert("steps".to_string(), steps_json);
        props.insert("prerequisites".to_string(), prereqs_json);
        props.insert("usage_count".to_string(), "0".to_string());
        props.insert("created_at".to_string(), now.to_string());

        self.graph
            .add_node(NT_PROCEDURAL, props, Some(&node_id))
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        Ok(node_id)
    }

    /// Return the ids of the episodes a procedure was derived from.
    ///
    /// Reads the `PROCEDURE_DERIVES_FROM` provenance edges outgoing from
    /// `procedure_id`. Returns an empty vector for an unknown id or a procedure
    /// with no provenance.
    pub fn procedure_provenance(&self, procedure_id: &str) -> Vec<String> {
        self.provenance_targets(procedure_id, ET_PROCEDURE_DERIVES_FROM)
    }

    /// Return the `node_id` of an existing procedure with `name` for this agent,
    /// or `None`. Used to make [`store_procedure`](Self::store_procedure) idempotent.
    fn find_procedure_id_by_name(&self, name: &str) -> Option<String> {
        let mut filter = agent_filter(&self.agent_name);
        filter.insert("name".to_string(), name.to_string());
        self.graph
            .query_nodes(NT_PROCEDURAL, Some(&filter), 1)
            .into_iter()
            .next()
            .map(|n| n.node_id)
    }

    /// Search procedures matching a query.
    ///
    /// Words from `query` are matched against `name` and `steps` fields.
    /// Results are sorted by `usage_count` descending.
    pub fn search_procedures(&self, query: &str, limit: usize) -> Vec<ProceduralMemory> {
        let filter = agent_filter(&self.agent_name);
        let nodes = self
            .graph
            .query_nodes(NT_PROCEDURAL, Some(&filter), usize::MAX);

        let keywords: Vec<String> = query
            .split_whitespace()
            .filter(|w| !w.is_empty())
            .map(|w| w.to_lowercase())
            .collect();

        let mut procs: Vec<ProceduralMemory> = nodes
            .into_iter()
            .filter(|n| {
                if keywords.is_empty() {
                    return true;
                }
                let name = n
                    .properties
                    .get("name")
                    .map(|s| s.to_lowercase())
                    .unwrap_or_default();
                let steps = n
                    .properties
                    .get("steps")
                    .map(|s| s.to_lowercase())
                    .unwrap_or_default();
                keywords
                    .iter()
                    .any(|kw| name.contains(kw) || steps.contains(kw))
            })
            .map(|n| node_to_procedural(&n.properties))
            .collect();

        procs.sort_by_key(|p| std::cmp::Reverse(p.usage_count));
        procs.truncate(limit);
        procs
    }

    /// Search procedures and increment their usage counts (mutable version).
    pub fn search_procedures_mut(&mut self, query: &str, limit: usize) -> Vec<ProceduralMemory> {
        let procs = self.search_procedures(query, limit);

        // Increment usage_count for recalled procedures
        for proc in &procs {
            let new_count = (proc.usage_count + 1).to_string();
            let mut update = HashMap::new();
            update.insert("usage_count".to_string(), new_count);
            let _ = self.graph.update_node(&proc.node_id, update);
        }

        procs
    }

    /// Recall procedures matching `query`, reinforcing each match.
    ///
    /// On every recall the matched procedures' persisted `usage_count` is
    /// incremented (reinforcement), and results are returned ordered by
    /// `usage_count` descending so frequently-used procedures rank first. The
    /// returned structs carry the pre-increment counts; the reinforced values
    /// are what persist and drive subsequent orderings.
    pub fn recall_procedure(&mut self, query: &str, limit: usize) -> Vec<ProceduralMemory> {
        self.search_procedures_mut(query, limit)
    }

    /// Deprecated: renamed to [`search_procedures`](Self::search_procedures).
    #[deprecated(since = "0.2.0", note = "renamed to search_procedures")]
    pub fn recall_procedures(&self, query: &str, limit: usize) -> Vec<ProceduralMemory> {
        self.search_procedures(query, limit)
    }

    /// Deprecated: renamed to [`search_procedures_mut`](Self::search_procedures_mut).
    #[deprecated(since = "0.2.0", note = "renamed to search_procedures_mut")]
    pub fn recall_procedures_mut(&mut self, query: &str, limit: usize) -> Vec<ProceduralMemory> {
        self.search_procedures_mut(query, limit)
    }
}
