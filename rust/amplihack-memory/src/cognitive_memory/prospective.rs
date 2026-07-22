//! Prospective memory methods for [`CognitiveMemory`].

use std::collections::HashMap;

use crate::memory_types::ProspectiveMemory;
use crate::{MemoryError, Result};

use tracing::{error, warn};

use super::converters::node_to_prospective;
use super::types::{agent_filter, new_id, ts_now, NT_PROSPECTIVE};
use super::CognitiveMemory;

/// Default lifecycle status a freshly stored prospective memory carries.
const DEFAULT_PROSPECTIVE_STATUS: &str = "pending";

impl CognitiveMemory {
    /// Store a trigger-action pair for future evaluation.
    ///
    /// The memory starts in the [`DEFAULT_PROSPECTIVE_STATUS`] (`"pending"`)
    /// status, so it is eligible to fire from
    /// [`check_triggers`](Self::check_triggers). Use
    /// [`store_prospective_with_status`](Self::store_prospective_with_status)
    /// when a caller must persist an explicit status (e.g. a restore path that
    /// round-trips a previously `resolved`/`triggered` memory).
    pub fn store_prospective(
        &mut self,
        description: &str,
        trigger_condition: &str,
        action_on_trigger: &str,
        priority: i32,
    ) -> Result<String> {
        self.store_prospective_with_status(
            description,
            trigger_condition,
            action_on_trigger,
            priority,
            DEFAULT_PROSPECTIVE_STATUS,
        )
    }

    /// Store a trigger-action pair with an explicit lifecycle `status`
    /// (`"pending"` / `"triggered"` / `"resolved"`).
    ///
    /// [`store_prospective`](Self::store_prospective) always starts a memory
    /// `"pending"`. This variant lets a restore path preserve the original
    /// status captured in a snapshot, so a `resolved`/`triggered` prospective
    /// does **not** come back `"pending"` and re-fire from
    /// [`check_triggers`](Self::check_triggers) after an import (Simard issue
    /// #2562). An empty `status` is normalised to
    /// [`DEFAULT_PROSPECTIVE_STATUS`] so a malformed snapshot cannot persist a
    /// blank status that would silently never match either the pending or the
    /// resolved paths.
    pub fn store_prospective_with_status(
        &mut self,
        description: &str,
        trigger_condition: &str,
        action_on_trigger: &str,
        priority: i32,
        status: &str,
    ) -> Result<String> {
        self.store_prospective_with_status_inner(
            description,
            trigger_condition,
            action_on_trigger,
            priority,
            status,
            None,
        )
    }

    /// Fenced-applier entry point for a `StoreProspective` intent: store the
    /// prospective memory under a DETERMINISTIC node id (`pro_{intent_id}`) so a
    /// crash-window replay upserts the same node instead of duplicating it
    /// (F2 exactly-once).
    ///
    /// # Errors
    /// As [`store_prospective`](Self::store_prospective).
    #[cfg(feature = "persistent")]
    pub(crate) fn apply_store_prospective(
        &mut self,
        intent_id: uuid::Uuid,
        description: &str,
        trigger_condition: &str,
        action_on_trigger: &str,
        priority: i32,
    ) -> Result<String> {
        let forced = format!("pro_{intent_id}");
        self.store_prospective_with_status_inner(
            description,
            trigger_condition,
            action_on_trigger,
            priority,
            DEFAULT_PROSPECTIVE_STATUS,
            Some(&forced),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn store_prospective_with_status_inner(
        &mut self,
        description: &str,
        trigger_condition: &str,
        action_on_trigger: &str,
        priority: i32,
        status: &str,
        forced_id: Option<&str>,
    ) -> Result<String> {
        let node_id = forced_id.map(String::from).unwrap_or_else(|| new_id("pro"));
        let now = ts_now();

        let status = if status.is_empty() {
            DEFAULT_PROSPECTIVE_STATUS
        } else {
            status
        };

        let mut props = HashMap::new();
        props.insert("node_id".to_string(), node_id.clone());
        props.insert("agent_id".to_string(), self.agent_name.clone());
        props.insert("desc_text".to_string(), description.to_string());
        props.insert(
            "trigger_condition".to_string(),
            trigger_condition.to_string(),
        );
        props.insert(
            "action_on_trigger".to_string(),
            action_on_trigger.to_string(),
        );
        props.insert("status".to_string(), status.to_string());
        props.insert("priority".to_string(), priority.to_string());
        props.insert("created_at".to_string(), now.to_string());

        self.graph
            .add_node(NT_PROSPECTIVE, props, Some(&node_id))
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        Ok(node_id)
    }

    /// Return up to `limit` prospective memories for this agent, in every
    /// status (`pending` / `triggered` / `resolved`), ordered by priority
    /// (highest first).
    ///
    /// A **pure read** (`&self`): unlike [`check_triggers`](Self::check_triggers)
    /// it neither filters by content nor mutates any node's status, so it is
    /// safe to call from a backup / export path. Added for the Simard verified
    /// snapshot (Simard issue #2550): a complete cognitive-memory backup must
    /// enumerate every prospective memory so a restore round-trips them. The
    /// prior snapshot captured only facts + procedures, which is exactly why a
    /// corruption-reset dropped every prospective trigger with no way back.
    ///
    /// When the store holds more than `limit` prospectives this returns the true
    /// **top-`limit` by priority**: the ordering is pushed into the query so the
    /// database sorts *before* it truncates (issue #124). The prior
    /// implementation truncated to an arbitrary `limit` window first and sorted
    /// in Rust afterwards, silently dropping lower-priority nodes.
    ///
    /// The `-> Vec` signature is frozen for backward compatibility, so this
    /// cannot return `Err`. It is nonetheless **fail-closed**: on a genuine
    /// backend read error it logs via `tracing::error!` (the error only — never
    /// node payloads) and returns an empty `Vec`, rather than silently masking
    /// the failure. A caller that must distinguish a confirmed-empty store from
    /// an unreadable one should use [`get_prospective_by_trigger`](Self::get_prospective_by_trigger),
    /// which propagates the error.
    pub fn get_all_prospective(&self, limit: usize) -> Vec<ProspectiveMemory> {
        let filter = agent_filter(&self.agent_name);
        match self.graph.query_nodes_ordered(
            NT_PROSPECTIVE,
            Some(&filter),
            "priority",
            true,
            true,
            limit,
        ) {
            Ok(nodes) => sorted_prospectives(&nodes),
            Err(e) => {
                error!("get_all_prospective: prospective read failed: {e}");
                Vec::new()
            }
        }
    }

    /// Return up to `limit` prospective memories for **this agent** whose
    /// `trigger_condition` **equals** `trigger`, ordered by priority descending
    /// (highest first), across every status (`pending` / `triggered` /
    /// `resolved`).
    ///
    /// A **pure read** (`&self`): it neither mutates node status nor runs the
    /// keyword-overlap heuristic used by [`check_triggers`](Self::check_triggers).
    /// The match is an **exact-equality** filter on `trigger_condition` pushed
    /// into the query, so the `LIMIT` bounds only *matching* nodes: a
    /// sentinel-tagged prospective (e.g. a `"creative-idea-thread"` dashboard
    /// node) is returned even when the store holds far more than `limit`
    /// higher-priority nodes carrying other triggers.
    ///
    /// This is the fix for the issue #124 user-facing bug: a dashboard that
    /// called `get_all_prospective(512)` and filtered by a sentinel in Rust
    /// showed **zero** ideas in a large store, because the low-priority idea
    /// nodes fell outside the arbitrary 512-row window. Filtering in the query
    /// removes that window entirely.
    ///
    /// **Fail-closed:** a genuine backend read error is **propagated** as `Err`,
    /// never masked as an empty `Ok(vec![])`. An empty `Ok(vec![])` therefore
    /// means a *confirmed* absence of matching prospectives.
    ///
    /// # Errors
    ///
    /// Returns `Err(MemoryError)` if the backend read fails.
    pub fn get_prospective_by_trigger(
        &self,
        trigger: &str,
        limit: usize,
    ) -> Result<Vec<ProspectiveMemory>> {
        let mut filter = agent_filter(&self.agent_name);
        filter.insert("trigger_condition".to_string(), trigger.to_string());
        let nodes = self.graph.query_nodes_ordered(
            NT_PROSPECTIVE,
            Some(&filter),
            "priority",
            true,
            true,
            limit,
        )?;
        Ok(sorted_prospectives(&nodes))
    }

    /// Check pending prospective memories against provided content.
    ///
    /// Uses a keyword-overlap heuristic: if any word from the
    /// `trigger_condition` appears in `content`, the prospective memory
    /// is considered triggered (status updated to `"triggered"`).
    pub fn check_triggers(&mut self, content: &str) -> Vec<ProspectiveMemory> {
        let mut filter = agent_filter(&self.agent_name);
        filter.insert("status".to_string(), "pending".to_string());
        let nodes = self
            .graph
            .query_nodes(NT_PROSPECTIVE, Some(&filter), usize::MAX);

        let mut candidates: Vec<ProspectiveMemory> = nodes
            .iter()
            .map(|n| node_to_prospective(&n.properties))
            .collect();

        // Sort by priority descending
        candidates.sort_by_key(|pm| std::cmp::Reverse(pm.priority));

        let content_lower = content.to_lowercase();
        let mut triggered = Vec::new();

        for mut pm in candidates {
            let trigger_words: Vec<String> = pm
                .trigger_condition
                .split_whitespace()
                .filter(|w| !w.is_empty())
                .map(|w| w.to_lowercase())
                .collect();

            let content_words: std::collections::HashSet<&str> =
                content_lower.split_whitespace().collect();
            if trigger_words
                .iter()
                .any(|w| content_words.contains(w.as_str()))
            {
                // Mark as triggered
                let mut update = HashMap::new();
                update.insert("status".to_string(), "triggered".to_string());
                if !self.graph.update_node(&pm.node_id, update) {
                    warn!(
                        "check_triggers: failed to update prospective node {}",
                        pm.node_id
                    );
                }

                pm.status = "triggered".to_string();
                triggered.push(pm);
            }
        }

        triggered
    }

    /// Mark a prospective memory as resolved.
    pub fn resolve_prospective(&mut self, node_id: &str) {
        let mut update = HashMap::new();
        update.insert("status".to_string(), "resolved".to_string());
        if !self.graph.update_node(node_id, update) {
            warn!("resolve_prospective: failed to update node {node_id}");
        }
    }
}

/// Map graph nodes to [`ProspectiveMemory`] and re-sort by priority descending.
///
/// The backend already ordered and truncated to the requested window; this
/// final Rust sort makes the observable order identical across every backend
/// (including the in-memory default), so callers see a single deterministic
/// contract regardless of which store is underneath.
fn sorted_prospectives(nodes: &[crate::graph::types::GraphNode]) -> Vec<ProspectiveMemory> {
    let mut out: Vec<ProspectiveMemory> = nodes
        .iter()
        .map(|n| node_to_prospective(&n.properties))
        .collect();
    out.sort_by_key(|pm| std::cmp::Reverse(pm.priority));
    out
}
