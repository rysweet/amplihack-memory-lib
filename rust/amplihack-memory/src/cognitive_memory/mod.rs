//! Cognitive memory system with six memory types.
//!
//! Provides a single [`CognitiveMemory`] struct that manages all six cognitive
//! memory types (sensory, working, episodic, semantic, procedural, prospective)
//! over any [`GraphStore`] backend.
//!
//! The default constructor ([`CognitiveMemory::new`]) keeps the historical
//! in-memory behavior. Callers that need durability can supply their own backend
//! via [`CognitiveMemory::with_store`], or — with the `persistent` feature —
//! open a LadybugDB-backed instance via `CognitiveMemory::open_persistent`.
//!
//! Each agent gets full isolation via an `agent_id` property stored on every
//! graph node.

mod converters;
mod dedup;
mod episodic;
mod procedural;
mod prospective;
mod ranked;
mod semantic;
mod sensory;
mod similarity;
mod types;
mod working;

pub use dedup::{
    compute_content_hash, DedupAction, DedupMode, DedupOptions, DuplicateFactGroup, FactInput,
    ProvenanceOptions, PruneReport, RetentionPolicy, StoreFactOptions, StoreFactOutcome,
};
pub use ranked::{
    exp_decay, keyword_jaccard, usage_boost, AccessKind, RecallOptions, RecallWeights, Scored,
};
pub use similarity::{SimilarityOptions, SimilarityReport};
pub use types::WORKING_MEMORY_CAPACITY;

use std::collections::HashMap;

use crate::graph::in_memory_store::InMemoryGraphStore;
use crate::graph::protocol::GraphStore;
use crate::graph::types::Direction;
use crate::memory_types::MemoryCategory;
use crate::{MemoryError, Result};

use types::{
    agent_filter, ts_now, NT_EPISODIC, NT_PROCEDURAL, NT_PROSPECTIVE, NT_SEMANTIC, NT_SENSORY,
    NT_WORKING,
};

// ---------------------------------------------------------------------------
// CognitiveMemory
// ---------------------------------------------------------------------------

/// Six-type cognitive memory over a pluggable [`GraphStore`] backend.
///
/// The struct owns a `Box<dyn GraphStore + Send>` and provides methods corresponding
/// to every public method in the Python `CognitiveMemory` class. The default
/// [`CognitiveMemory::new`] uses an [`InMemoryGraphStore`]; [`CognitiveMemory::with_store`]
/// accepts any backend, and `CognitiveMemory::open_persistent` (feature
/// `persistent`) opens a LadybugDB-backed durable instance.
pub struct CognitiveMemory {
    agent_name: String,
    graph: Box<dyn GraphStore + Send>,
    sensory_order: i64,
    temporal_index: i64,
}

impl CognitiveMemory {
    /// Create a new in-memory `CognitiveMemory` for the given agent.
    ///
    /// Backward-compatible with prior releases: storage is an
    /// [`InMemoryGraphStore`] and nothing is persisted across process restarts.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::InvalidInput` if `agent_name` is empty.
    pub fn new(agent_name: &str) -> Result<Self> {
        let trimmed = Self::validate_agent_name(agent_name)?;
        let store_id = format!("cognitive-{trimmed}");
        Self::with_store(
            agent_name,
            Box::new(InMemoryGraphStore::new(Some(&store_id))),
        )
    }

    /// Create a `CognitiveMemory` backed by a caller-supplied [`GraphStore`].
    ///
    /// This is the seam that makes the cognitive memory backend-pluggable:
    /// supply an in-memory, LadybugDB, or any other `GraphStore` implementation.
    /// The backend must be [`Send`] so the resulting `CognitiveMemory` can cross
    /// thread boundaries (required by the PyO3 `#[pyclass]` wrapper and by any
    /// consumer that moves it between threads).
    /// Counters (`temporal_index`, `sensory_order`) are recovered from any
    /// pre-existing data for this agent so auto-incrementing indices continue
    /// monotonically across reopens.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::InvalidInput` if `agent_name` is empty.
    pub fn with_store(agent_name: &str, store: Box<dyn GraphStore + Send>) -> Result<Self> {
        let trimmed = Self::validate_agent_name(agent_name)?;
        let mut cm = Self {
            agent_name: trimmed,
            graph: store,
            sensory_order: 0,
            temporal_index: 0,
        };
        cm.recover_counters();
        Ok(cm)
    }

    /// Open (or create) a LadybugDB-backed persistent `CognitiveMemory`.
    ///
    /// Data is durably stored at `path` and survives `close` + reopen. The
    /// backend implements the same [`GraphStore`] trait as the in-memory store,
    /// so all cognitive-memory methods behave identically — only durability and
    /// crash-safety differ.
    ///
    /// This is resilient to an unclean shutdown: if the on-disk write-ahead log
    /// was left partially written (the failure that previously made the store
    /// permanently unopenable), it transparently delegates to
    /// [`open_persistent_with_recovery`](Self::open_persistent_with_recovery),
    /// which quarantines the corrupt WAL and recovers the last good checkpoint
    /// plus any replayable prefix rather than returning an error.
    ///
    /// Requires the `persistent` cargo feature.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::InvalidInput` if `agent_name` is empty, or
    /// `MemoryError::Storage` if the database cannot be opened even after WAL
    /// recovery.
    #[cfg(feature = "persistent")]
    pub fn open_persistent(path: impl AsRef<std::path::Path>, agent_name: &str) -> Result<Self> {
        Self::open_persistent_with_recovery(path, agent_name)
    }

    /// Open a LadybugDB-backed persistent `CognitiveMemory`, recovering from a
    /// corrupt / partially-written WAL instead of failing.
    ///
    /// A strict open is attempted first. If it fails because the WAL cannot be
    /// replayed (e.g. after a process was killed mid-write), the unreplayable WAL
    /// is moved aside to `<wal>.corrupt-<timestamp>`, the recoverable prefix is
    /// replayed, and a checkpoint folds it into the main database file. A
    /// structured `warn!` reports how many records survived. The clean case is
    /// identical to a strict open: no artifact is written and no warning emitted.
    ///
    /// [`open_persistent`](Self::open_persistent) delegates to this method, so
    /// existing callers gain recovery automatically.
    ///
    /// Requires the `persistent` cargo feature.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::InvalidInput` if `agent_name` is empty, or
    /// `MemoryError::Storage` if the database cannot be opened even after
    /// quarantining the WAL.
    #[cfg(feature = "persistent")]
    pub fn open_persistent_with_recovery(
        path: impl AsRef<std::path::Path>,
        agent_name: &str,
    ) -> Result<Self> {
        let trimmed = Self::validate_agent_name(agent_name)?;
        let store_id = format!("cognitive-{trimmed}");
        let (store, recovery) = crate::graph::lbug_store::LbugGraphStore::open_with_recovery(
            path.as_ref(),
            Some(&store_id),
        )?;
        // #107: fail closed if the empty-read gate tripped. A populated store
        // that read back empty must NOT be handed back as a usable (empty) graph
        // — a consumer would run "clean" cycles over it and checkpoint, upgrading
        // an empty store to v41 and destroying the data permanently. The store is
        // sealed and its v40 bytes are intact; surface a hard error so the caller
        // rolls back (or deploys a build with the read fix) instead.
        if recovery.outcome == crate::graph::WalRecoveryOutcome::SuspectedDataLoss {
            let footprint_bytes = std::fs::metadata(store.db_path())
                .map(|m| m.len())
                .unwrap_or(0);
            // The gate set `recovered_records` to the (empty) count it reacted to;
            // use it rather than re-reading `count_all_nodes()`, whose unfiltered
            // `count(n)` can disagree with the suspect read the gate rejected.
            let read_count = recovery.recovered_records;
            tracing::error!(
                agent = %trimmed,
                footprint_bytes,
                read_count,
                "CognitiveMemory::open_persistent: #107 empty-read gate tripped — refusing \
                 to open a populated store that read back empty (store left intact, not \
                 checkpointed)"
            );
            return Err(MemoryError::SuspectedDataLoss {
                footprint_bytes,
                read_count,
            });
        }
        if recovery.recovered() {
            tracing::warn!(
                agent = %trimmed,
                outcome = ?recovery.outcome,
                recovered_records = recovery.recovered_records,
                quarantined_wal = ?recovery.quarantined_wal,
                "CognitiveMemory::open_persistent recovered from a corrupt WAL"
            );
        }
        Self::with_store(agent_name, Box::new(store))
    }

    /// Flush durable state: force the backend to checkpoint its write-ahead log
    /// into the main store so a subsequent reopen needs no replay.
    ///
    /// For the in-memory backend this is a no-op. For the LadybugDB persistent
    /// backend it issues a `CHECKPOINT`, bounding how much data an unclean
    /// shutdown could strand in the WAL. The persistent backend also
    /// auto-checkpoints periodically and on close/drop; call this explicitly at
    /// known-safe points (e.g. after a batch of writes) for a stronger guarantee.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::Storage` if the backend supports checkpointing but
    /// the flush fails.
    pub fn checkpoint(&self) -> Result<()> {
        self.graph.checkpoint()
    }

    fn validate_agent_name(agent_name: &str) -> Result<String> {
        let trimmed = agent_name.trim();
        if trimmed.is_empty() {
            return Err(MemoryError::InvalidInput(
                "agent_name cannot be empty".into(),
            ));
        }
        Ok(trimmed.to_string())
    }

    /// Recover the auto-incrementing `temporal_index` and `sensory_order`
    /// counters from any persisted nodes for this agent.
    ///
    /// A no-op for an empty (e.g. freshly created in-memory) store.
    fn recover_counters(&mut self) {
        let filter = agent_filter(&self.agent_name);

        let max_temporal = self
            .graph
            .query_nodes(NT_EPISODIC, Some(&filter), usize::MAX)
            .iter()
            .filter_map(|n| {
                n.properties
                    .get("temporal_index")
                    .and_then(|v| v.parse::<i64>().ok())
            })
            .max()
            .unwrap_or(0);
        if max_temporal > self.temporal_index {
            self.temporal_index = max_temporal;
        }

        let max_sensory = self
            .graph
            .query_nodes(NT_SENSORY, Some(&filter), usize::MAX)
            .iter()
            .filter_map(|n| {
                n.properties
                    .get("observation_order")
                    .and_then(|v| v.parse::<i64>().ok())
            })
            .max()
            .unwrap_or(0);
        if max_sensory > self.sensory_order {
            self.sensory_order = max_sensory;
        }
    }

    /// The agent name this memory is scoped to.
    pub fn agent_name(&self) -> &str {
        &self.agent_name
    }

    // ======================================================================
    // STATISTICS
    // ======================================================================

    /// Return counts per memory type.
    ///
    /// Returns a map with keys for each `MemoryCategory` (as string) plus
    /// a `"total"` key.
    pub fn get_memory_stats(&self) -> HashMap<String, usize> {
        let tables = [
            (MemoryCategory::Sensory, NT_SENSORY),
            (MemoryCategory::Working, NT_WORKING),
            (MemoryCategory::Episodic, NT_EPISODIC),
            (MemoryCategory::Semantic, NT_SEMANTIC),
            (MemoryCategory::Procedural, NT_PROCEDURAL),
            (MemoryCategory::Prospective, NT_PROSPECTIVE),
        ];

        let mut stats = HashMap::new();
        let mut total = 0usize;
        let filter = agent_filter(&self.agent_name);

        for (category, node_type) in &tables {
            let count = self
                .graph
                .query_nodes(node_type, Some(&filter), usize::MAX)
                .len();
            stats.insert(category.as_str().to_string(), count);
            total += count;
        }

        stats.insert("total".to_string(), total);
        stats
    }

    /// Alias matching the Python method name `get_statistics`.
    pub fn get_statistics(&self) -> HashMap<String, usize> {
        self.get_memory_stats()
    }

    // ======================================================================
    // PROVENANCE (shared helpers)
    // ======================================================================

    /// Return `true` if `id` refers to an existing episodic-memory node.
    ///
    /// Used to gate provenance-edge creation: a `DERIVES_FROM` /
    /// `PROCEDURE_DERIVES_FROM` edge may only target a real
    /// [`EpisodicMemory`](crate::memory_types::EpisodicMemory) node, and both
    /// graph backends reject [`add_edge`](crate::graph::GraphStore::add_edge)
    /// for a missing endpoint, so callers must pre-check.
    fn is_source_episode(&self, id: &str) -> bool {
        self.graph
            .get_node(id)
            .is_some_and(|n| n.node_type == NT_EPISODIC)
    }

    /// Create a provenance edge `source_id --edge_type--> episode_id` carrying a
    /// `derived_at` timestamp. Both endpoints must already exist.
    fn add_provenance_edge(
        &mut self,
        source_id: &str,
        episode_id: &str,
        edge_type: &str,
    ) -> Result<()> {
        let mut props = HashMap::new();
        props.insert("derived_at".to_string(), ts_now().to_string());
        self.graph
            .add_edge(source_id, episode_id, edge_type, Some(props))
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Return the ids of every node reachable from `node_id` along an outgoing
    /// `edge_type` edge. Empty for unknown ids or nodes with no such edges.
    fn provenance_targets(&self, node_id: &str, edge_type: &str) -> Vec<String> {
        self.graph
            .query_neighbors(node_id, Some(edge_type), Direction::Outgoing, usize::MAX)
            .into_iter()
            .map(|(_, node)| node.node_id)
            .collect()
    }

    // ======================================================================
    // LIFECYCLE
    // ======================================================================

    /// Release graph resources.
    pub fn close(&mut self) {
        self.graph.close();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
