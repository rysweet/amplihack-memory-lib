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
//!
//! # Quick start
//!
//! ```no_run
//! use amplihack_memory::{Experience, ExperienceType, ExperienceStore};
//! use std::path::Path;
//!
//! // Create a store backed by SQLite
//! let mut store = ExperienceStore::new(
//!     "my-agent",
//!     true,         // auto-compress old experiences
//!     Some(30),     // retain at most 30 days
//!     Some(1000),   // retain at most 1000 experiences
//!     100,          // 100 MB quota
//!     Some(Path::new("/tmp/agent-memory")),
//! ).expect("failed to open store");
//!
//! // Record a new experience
//! let exp = Experience::new(
//!     ExperienceType::Success,
//!     "compiled the project".into(),
//!     "zero warnings produced".into(),
//!     0.95,
//! ).unwrap();
//! store.add(&exp).unwrap();
//!
//! // Search later
//! let results = store.search("compile", None, 0.0, 10).unwrap();
//! ```

#[cfg(all(feature = "python", feature = "kuzu"))]
compile_error!(
    "Features `python` and `kuzu` are mutually exclusive. \
     `python` uses pyo3/extension-module (Rust as Python extension), \
     `kuzu` uses pyo3/auto-initialize (Rust calls Python)."
);

#[cfg(all(feature = "kuzu", feature = "ladybug"))]
compile_error!(
    "Features `kuzu` and `ladybug` are mutually exclusive. \
     Use one graph backend at a time."
);

/// Storage backend abstraction layer (SQLite, Kùzu, Ladybug).
pub mod backends;
/// Six-category cognitive memory system modeled after human cognition.
pub mod cognitive_memory;
/// High-level memory connector that selects and initializes a backend.
pub mod connector;
/// Contradiction detection between memory facts.
pub mod contradiction;
/// Named-entity extraction from free-text content.
pub mod entity_extraction;
/// Error types and the crate-wide `Result` alias.
pub mod errors;
/// Core experience data model and serialization.
pub mod experience;
/// Graph abstraction layer with in-memory, Hive, and federated stores.
pub mod graph;
/// Hierarchical knowledge graph with classification and similarity linking.
pub mod hierarchical_memory;
/// Cognitive memory type definitions (sensory, working, episodic, etc.).
pub mod memory_types;
/// Pattern recognition engine for extracting recurring patterns from experiences.
pub mod pattern_recognition;
/// Python bindings via PyO3.
#[cfg(feature = "python")]
pub mod python;
/// Security layer with capability-based access control and credential scrubbing.
pub mod security;
/// TF-IDF-based semantic search and relevance scoring.
pub mod semantic_search;
/// Deterministic text similarity functions (Jaccard, word-level, tag-level).
pub mod similarity;
/// Managed experience store with auto-compression and retention policies.
pub mod store;
/// Internal utility functions shared across modules.
pub(crate) mod utils;

// Re-exports for convenience

/// Crate-wide error type and result alias.
pub use errors::{MemoryError, Result};

/// Core experience data types.
pub use experience::{Experience, ExperienceType};

/// Cognitive memory structs for each of the six memory categories.
pub use memory_types::{
    ConsolidatedEpisode, EpisodicMemory, MemoryCategory, ProceduralMemory, ProspectiveMemory,
    SemanticFact, SensoryItem, WorkingMemorySlot,
};

/// Deterministic text-similarity scoring functions.
pub use similarity::{
    compute_similarity, compute_tag_similarity, compute_word_similarity, rerank_facts_by_query,
};

/// Contradiction detection result types.
pub use contradiction::{detect_contradiction, ContradictionResult};
/// Entity-name extraction from free-text.
pub use entity_extraction::extract_entity_name;

/// Kùzu-backed graph store (requires the `kuzu` feature).
#[cfg(feature = "kuzu")]
pub use graph::KuzuGraphStore;
/// Graph abstraction types, stores, and traversal primitives.
pub use graph::{
    AnnotatedResult, Direction, FederatedGraphStore, FederatedQueryResult, GraphEdge, GraphNode,
    GraphStore, HiveGraphStore, InMemoryGraphStore, TraversalItem, TraversalResult,
};

/// Security primitives: capabilities, credential scrubbing, query validation.
pub use security::{
    AgentCapabilities, CredentialScrubber, QueryValidator, ScopeLevel, SecureMemoryBackend,
};

/// Semantic search engine and relevance scorer.
#[allow(deprecated)]
pub use semantic_search::{
    calculate_relevance, retrieve_relevant_experiences, tfidf_similarity, JaccardSimilarity,
    SemanticSearchEngine, TfIdfSimilarity,
};

/// Pattern recognition utilities for discovering recurring experience patterns.
pub use pattern_recognition::{
    calculate_pattern_confidence, extract_pattern_key, recognize_patterns, PatternDetector,
};

/// Kùzu storage backend (requires the `kuzu` feature).
#[cfg(feature = "kuzu")]
pub use backends::KuzuBackend;
/// Ladybug graph storage backend (requires the `ladybug` feature).
#[cfg(feature = "ladybug")]
pub use backends::LadybugBackend;
/// Core backend traits and SQLite implementation.
pub use backends::{ExperienceBackend, MemoryBackend, SqliteBackend};
/// Unified cognitive memory interface over the graph layer.
pub use cognitive_memory::CognitiveMemory;
/// Backend selector and memory connector entry point.
pub use connector::{BackendType, MemoryConnector};
/// Hierarchical knowledge graph with classification, subgraph queries, and edge types.
pub use hierarchical_memory::{
    HierarchicalMemory, KnowledgeEdge, KnowledgeNode, KnowledgeSubgraph, MemoryClassifier,
};
/// Managed experience store with auto-compression and retention policies.
pub use store::ExperienceStore;
