//! amplihack-memory: Standalone memory system for goal-seeking agents.
//!
//! This crate provides graph-based persistent memory with:
//! - Six cognitive memory types (sensory, working, episodic, semantic, procedural, prospective)
//! - Deterministic text similarity (Jaccard, no ML required)
//! - Entity extraction and contradiction detection
//! - Security layer with capability-based access control
//! - SQLite backend with FTS5 full-text search
//! - Graph abstraction with in-memory and federated store implementations
//! - Pattern recognition from discoveries

#[cfg(all(feature = "python", feature = "kuzu"))]
compile_error!(
    "Features `python` and `kuzu` are mutually exclusive. \
     `python` uses pyo3/extension-module (Rust as Python extension), \
     `kuzu` uses pyo3/auto-initialize (Rust calls Python)."
);

pub mod backends;
pub mod cognitive_memory;
pub mod connector;
pub mod contradiction;
pub mod entity_extraction;
pub mod errors;
pub mod experience;
pub mod graph;
pub mod hierarchical_memory;
pub mod memory_types;
pub mod pattern_recognition;
#[cfg(feature = "python")]
pub mod python_module;
pub mod security;
pub mod semantic_search;
pub mod similarity;
pub mod store;

// Re-exports for convenience
pub use errors::{MemoryError, Result};

pub use experience::{Experience, ExperienceType};

pub use memory_types::{
    ConsolidatedEpisode, EpisodicMemory, MemoryCategory, ProceduralMemory, ProspectiveMemory,
    SemanticFact, SensoryItem, WorkingMemorySlot,
};

pub use similarity::{
    compute_similarity, compute_tag_similarity, compute_word_similarity, rerank_facts_by_query,
};

pub use contradiction::{detect_contradiction, ContradictionResult};
pub use entity_extraction::extract_entity_name;

#[cfg(feature = "kuzu")]
pub use graph::KuzuGraphStore;
pub use graph::{
    AnnotatedResult, Direction, FederatedGraphStore, FederatedQueryResult, GraphEdge, GraphNode,
    GraphStore, HiveGraphStore, InMemoryGraphStore, TraversalItem, TraversalResult,
};

pub use security::{
    AgentCapabilities, CredentialScrubber, QueryValidator, ScopeLevel, SecureMemoryBackend,
};

pub use semantic_search::{
    calculate_relevance, retrieve_relevant_experiences, SemanticSearchEngine, TfIdfSimilarity,
};

pub use pattern_recognition::{
    calculate_pattern_confidence, extract_pattern_key, recognize_patterns, PatternDetector,
};

#[cfg(feature = "kuzu")]
pub use backends::KuzuBackend;
pub use backends::{ExperienceBackend, MemoryBackend, SqliteBackend};
pub use cognitive_memory::CognitiveMemory;
pub use connector::{BackendType, MemoryConnector};
pub use hierarchical_memory::{
    HierarchicalMemory, KnowledgeEdge, KnowledgeNode, KnowledgeSubgraph, MemoryClassifier,
};
pub use store::ExperienceStore;
