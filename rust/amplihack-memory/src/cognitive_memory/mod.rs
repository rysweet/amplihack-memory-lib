//! Cognitive memory system with six memory types.
//!
//! Provides a single [`CognitiveMemory`] struct that manages all six cognitive
//! memory types (sensory, working, episodic, semantic, procedural, prospective)
//! backed by an [`InMemoryGraphStore`].
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

/// Six-type cognitive memory backed by an in-memory graph store.
///
/// The struct owns an [`InMemoryGraphStore`] and provides methods corresponding
/// to every public method in the Python `CognitiveMemory` class.
pub struct CognitiveMemory {
    agent_name: String,
    graph: InMemoryGraphStore,
    sensory_order: i64,
    temporal_index: i64,
}

impl CognitiveMemory {
    /// Create a new `CognitiveMemory` for the given agent.
    ///
    /// # Errors
    ///
    /// Returns `MemoryError::InvalidInput` if `agent_name` is empty.
    pub fn new(agent_name: &str) -> Result<Self> {
        let trimmed = agent_name.trim();
        if trimmed.is_empty() {
            return Err(MemoryError::InvalidInput(
                "agent_name cannot be empty".into(),
            ));
        }
        let store_id = format!("cognitive-{}", trimmed);
        Ok(Self {
            agent_name: trimmed.to_string(),
            graph: InMemoryGraphStore::new(Some(&store_id)),
            sensory_order: 0,
            temporal_index: 0,
        })
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
