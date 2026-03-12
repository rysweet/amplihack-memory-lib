//! Common graph abstraction layer for agent graphs and hive mind.

pub mod federated_store;
pub mod hive_store;
pub mod in_memory_store;
#[cfg(feature = "kuzu")]
pub mod kuzu_store;
pub mod protocol;
pub mod traversal;
pub mod types;

pub use federated_store::{AnnotatedResult, FederatedGraphStore, FederatedQueryResult};
pub use hive_store::HiveGraphStore;
pub use in_memory_store::InMemoryGraphStore;
#[cfg(feature = "kuzu")]
pub use kuzu_store::KuzuGraphStore;
pub use protocol::GraphStore;
pub use types::{Direction, GraphEdge, GraphNode, TraversalItem, TraversalResult};
