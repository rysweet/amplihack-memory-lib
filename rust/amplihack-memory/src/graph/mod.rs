//! Common graph abstraction layer for agent graphs and hive mind.

/// Federated multi-graph query aggregation.
pub mod federated_store;
/// Hive-mind shared graph store.
pub mod hive_store;
/// In-memory graph store for testing and lightweight use.
pub mod in_memory_store;
/// Kùzu-backed persistent graph store (requires the `kuzu` feature).
#[cfg(feature = "kuzu")]
pub mod kuzu_store;
/// Common graph store trait defining the storage protocol.
pub mod protocol;
/// Multi-hop graph traversal algorithms.
pub mod traversal;
/// Core graph data types (nodes, edges, directions, traversal results).
pub mod types;

/// Annotated query result and federated graph store.
pub use federated_store::{AnnotatedResult, FederatedGraphStore, FederatedQueryResult};
/// Hive-mind shared graph store implementation.
pub use hive_store::HiveGraphStore;
/// In-memory graph store implementation.
pub use in_memory_store::InMemoryGraphStore;
/// Kùzu-backed graph store implementation.
#[cfg(feature = "kuzu")]
pub use kuzu_store::KuzuGraphStore;
/// The common `GraphStore` trait all backends implement.
pub use protocol::GraphStore;
/// Core graph primitives re-exported for convenience.
pub use types::{Direction, GraphEdge, GraphNode, TraversalItem, TraversalResult};
