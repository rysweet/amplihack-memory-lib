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

    /// Record `intent_id` as applied by writing its ledger marker node. The
    /// marker rides the store's WAL/checkpoint with the intent's effect, so a
    /// later replay is idempotent. Writing the same marker twice is harmless:
    /// the id is a primary key, so it collapses to one node.
    ///
    /// # Errors
    /// [`MemoryError::Storage`] if the marker node cannot be written.
    pub(crate) fn mark_intent_applied(&mut self, intent_id: Uuid) -> Result<()> {
        let id = applied_intent_node_id(intent_id);
        let mut props = HashMap::new();
        props.insert("node_id".to_string(), id.clone());
        props.insert("intent_id".to_string(), intent_id.to_string());
        props.insert("applied_at".to_string(), ts_now().to_string());
        self.graph
            .add_node(NT_APPLIED_INTENT, props, Some(&id))
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        Ok(())
    }
}
