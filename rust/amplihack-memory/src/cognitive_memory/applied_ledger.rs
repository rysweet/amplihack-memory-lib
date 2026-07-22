//! Durable exactly-once ledger for applied intents (F2, `PrefixConsistency`).
//!
//! The fenced applier records each successfully-applied `WriteIntent` by its
//! `intent_id` as a marker node **in the store itself**, so the marker is
//! checkpointed atomically alongside the intent's effect. On a restart — even
//! one that rewound the durable applied-index (a crash in the
//! checkpoint → index-persist window) — the applier consults this ledger and
//! skips any intent whose marker is already present, making replay genuinely
//! idempotent across process restarts rather than relying on a process-local
//! `seen` set that is empty after a crash.
//!
//! # Bounded ledger (N2)
//!
//! Each marker records the intent's log offset (`applied_offset`, an opaque,
//! lexicographically-ordered key). Once the durable applied-index has advanced
//! strictly past a marker's offset, that marker can never be needed again — a
//! restart resumes at the durable index and never re-reads the covered record —
//! so [`prune_applied_below`](CognitiveMemory::prune_applied_below) deletes it.
//! The applier prunes **only after** the applied-index is durably persisted, so
//! pruning is delete-after-persist and crash-safe: a crash mid-prune leaves
//! extra markers (harmless, idempotent), never fewer than safe. This keeps the
//! ledger bounded rather than growing one permanent node per intent ever applied.

use std::collections::HashMap;

use uuid::Uuid;

use crate::{MemoryError, Result};

use super::types::{ts_now, NT_APPLIED_INTENT};
use super::CognitiveMemory;

/// The ledger node id derived from an intent's idempotency key. A distinct
/// prefix keeps it from ever colliding with a memory node id.
fn applied_intent_node_id(intent_id: Uuid) -> String {
    format!("aintent_{intent_id}")
}

impl CognitiveMemory {
    /// `true` if `intent_id` has a durable applied-marker in the store.
    pub(crate) fn intent_applied(&self, intent_id: Uuid) -> bool {
        self.graph
            .get_node(&applied_intent_node_id(intent_id))
            .is_some()
    }

    /// Record `intent_id` as applied by writing its ledger marker node, stamped
    /// with the intent's durable log offset (`applied_offset`) so the ledger can
    /// later be pruned below the durable applied-index (N2). The marker rides the
    /// store's WAL/checkpoint with the intent's effect, so a later replay is
    /// idempotent. Writing the same marker twice is harmless: the id is a primary
    /// key, so it collapses to one node.
    ///
    /// # Errors
    /// [`MemoryError::Storage`] if the marker node cannot be written.
    pub(crate) fn mark_intent_applied(
        &mut self,
        intent_id: Uuid,
        applied_offset: &str,
    ) -> Result<()> {
        let id = applied_intent_node_id(intent_id);
        let mut props = HashMap::new();
        props.insert("node_id".to_string(), id.clone());
        props.insert("intent_id".to_string(), intent_id.to_string());
        props.insert("applied_at".to_string(), ts_now().to_string());
        props.insert("applied_offset".to_string(), applied_offset.to_string());
        self.graph
            .add_node(NT_APPLIED_INTENT, props, Some(&id))
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Count of live applied-intent markers currently in the store. Bounded once
    /// [`prune_applied_below`](Self::prune_applied_below) runs; used by tests to
    /// assert the ledger does not grow without bound (N2).
    pub(crate) fn applied_intent_count(&self) -> usize {
        self.graph
            .query_nodes(NT_APPLIED_INTENT, None, usize::MAX)
            .len()
    }

    /// Delete every applied-intent marker whose `applied_offset` is strictly
    /// below `watermark` (the durable applied-index order key). Safe because the
    /// caller invokes this only AFTER the applied-index is durably persisted past
    /// those offsets, so a restart resumes beyond them and never re-reads their
    /// records — the markers are provably unreachable. A marker missing its
    /// `applied_offset` (legacy round-1 marker) is left in place: without an
    /// offset we cannot prove it is below the watermark, so we never risk
    /// dropping a still-needed marker.
    ///
    /// # Errors
    /// [`MemoryError::Storage`] if enumerating the markers fails. Individual
    /// deletes are best-effort (a failed delete just leaves an extra marker,
    /// which is harmless and retried next prune).
    pub(crate) fn prune_applied_below(&mut self, watermark: &str) -> Result<()> {
        let markers = self.graph.query_nodes(NT_APPLIED_INTENT, None, usize::MAX);
        for marker in markers {
            let Some(offset) = marker.properties.get("applied_offset") else {
                continue;
            };
            if offset.as_str() < watermark {
                let _ = self.graph.delete_node(&marker.node_id);
            }
        }
        Ok(())
    }
}
