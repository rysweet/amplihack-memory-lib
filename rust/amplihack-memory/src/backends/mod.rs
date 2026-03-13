//! Backend abstraction layer for memory storage.

/// Abstract base traits for memory storage backends.
pub mod base;
/// Kùzu graph-database backend (requires the `kuzu` feature).
#[cfg(feature = "kuzu")]
pub mod kuzu_backend;
/// Ladybug graph backend (requires the `ladybug` feature).
#[cfg(feature = "ladybug")]
pub mod ladybug_backend;
/// SQLite-based persistent storage backend with FTS5 full-text search.
pub mod sqlite_backend;

/// Core storage traits re-exported for convenience.
pub use base::{ExperienceBackend, MemoryBackend};
/// Kùzu backend implementation.
#[cfg(feature = "kuzu")]
pub use kuzu_backend::KuzuBackend;
/// Ladybug backend implementation.
#[cfg(feature = "ladybug")]
pub use ladybug_backend::LadybugBackend;
/// SQLite backend implementation.
pub use sqlite_backend::SqliteBackend;
