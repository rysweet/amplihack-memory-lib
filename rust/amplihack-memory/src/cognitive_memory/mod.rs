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
mod episodic;
mod procedural;
mod prospective;
mod semantic;
mod sensory;
mod types;
mod working;

pub use types::WORKING_MEMORY_CAPACITY;

use std::collections::HashMap;

use crate::graph::in_memory_store::InMemoryGraphStore;
use crate::graph::protocol::GraphStore;
use crate::memory_types::MemoryCategory;
use crate::{MemoryError, Result};

use types::{
    agent_filter, NT_EPISODIC, NT_PROCEDURAL, NT_PROSPECTIVE, NT_SEMANTIC, NT_SENSORY, NT_WORKING,
};

// ---------------------------------------------------------------------------
// CognitiveMemory
// ---------------------------------------------------------------------------

/// Six-type cognitive memory over a pluggable [`GraphStore`] backend.
///
/// The struct owns a `Box<dyn GraphStore>` and provides methods corresponding
/// to every public method in the Python `CognitiveMemory` class. The default
/// [`CognitiveMemory::new`] uses an [`InMemoryGraphStore`]; [`CognitiveMemory::with_store`]
/// accepts any backend, and `CognitiveMemory::open_persistent` (feature
/// `persistent`) opens a LadybugDB-backed durable instance.
pub struct CognitiveMemory {
    agent_name: String,
    graph: Box<dyn GraphStore>,
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
    /// Counters (`temporal_index`, `sensory_order`) are recovered from any
    /// pre-existing data for this agent so auto-incrementing indices continue
    /// monotonically across reopens.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::InvalidInput` if `agent_name` is empty.
    pub fn with_store(agent_name: &str, store: Box<dyn GraphStore>) -> Result<Self> {
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
    /// Requires the `persistent` cargo feature.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::InvalidInput` if `agent_name` is empty, or
    /// `MemoryError::Storage` if the database cannot be opened.
    #[cfg(feature = "persistent")]
    pub fn open_persistent(path: impl AsRef<std::path::Path>, agent_name: &str) -> Result<Self> {
        let trimmed = Self::validate_agent_name(agent_name)?;
        let store_id = format!("cognitive-{trimmed}");
        let store = crate::graph::lbug_store::LbugGraphStore::open(path.as_ref(), Some(&store_id))?;
        Self::with_store(agent_name, Box::new(store))
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
