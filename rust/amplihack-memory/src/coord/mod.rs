//! Design C — a transactionally-safe multi-writer coordination layer over the
//! single-writer `lbug` cognitive store.
//!
//! This module is the Rust implementation of the design that was **formally
//! verified with TLA+** in [`specs/`](../../../specs/README.md); those specs are
//! the source of truth. See [`docs/coordination_layer.md`](../../docs/coordination_layer.md)
//! for the full contract. Every public guarantee here is pinned to a
//! model-checked invariant:
//!
//! * **`NoSplitBrain`** (`FencedApplier.tla`) — store ownership is a monotonic
//!   [`Epoch`](lease::Epoch) in an on-disk [`Lease`](lease::Lease); the
//!   applier fences on every apply. `kill(pid, 0)` liveness is never used.
//! * **`PrefixConsistency` + `NoLostAckedWrite`** (`DurableLog.tla`) — writers
//!   append [`WriteIntent`]s to a durable, `fsync`-on-append shared log (the
//!   ack); a single applier drains it strictly in order, exactly once.
//!
//! # Feature gating
//!
//! * `coord` — the consumer surface ([`CoordConfig`], [`Lease`](lease::Lease),
//!   [`WriteIntent`], [`WriterClient`], [`LogOffset`]). Builds **without** the
//!   `lbug` engine, so an engineer binary that enables only `coord` cannot name
//!   `CognitiveMemory::open_persistent`.
//! * `ipc` — the read client [`RankedRecallClient`] transport.
//! * `persistent` — the daemon-only pieces: `Applier`, `Coordinator`, and (with
//!   `ipc`) `IpcServer`, all of which own the single `lbug` writer.

use std::io;
use std::path::{Path, PathBuf};

use crate::errors::{MemoryError, Result};

pub mod intent_log;
pub mod lease;
pub mod writer;

#[cfg(feature = "persistent")]
pub mod applier;

#[cfg(feature = "ipc")]
pub mod ipc;

pub use intent_log::{LogOffset, WriteIntent};
pub use lease::{Epoch, Lease};
pub use writer::WriterClient;

#[cfg(feature = "persistent")]
pub use applier::{Applier, Coordinator};

#[cfg(feature = "ipc")]
pub use ipc::RankedRecallClient;

#[cfg(all(feature = "ipc", feature = "persistent"))]
pub use ipc::IpcServer;

/// Default maximum size of a single intent-log segment before rollover (64 MiB).
pub const DEFAULT_SEGMENT_BYTES: u64 = 64 * 1024 * 1024;

/// Default hard cap on a single log record / IPC frame (16 MiB). Bounds a
/// hostile or corrupt length prefix *before* any body allocation.
pub const DEFAULT_MAX_FRAME_BYTES: u32 = 16 * 1024 * 1024;

/// Fixed filename of the read-transport Unix domain socket under the coord dir.
pub const DEFAULT_SOCKET_NAME: &str = "read.sock";

/// Configuration + on-disk layout of a coordination directory.
///
/// All coordination state (lease, intent log, applied-index, read socket) lives
/// under a single [`base_dir`](Self::base_dir). Construct with
/// [`for_store`](Self::for_store) to place it beside the store at
/// `<store>.coord`, discoverable next to the store it coordinates and captured
/// by an ordinary store-directory backup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoordConfig {
    /// Root of the coordination directory (`0o700`).
    pub base_dir: PathBuf,
    /// Max size of an intent-log segment before rollover.
    pub segment_bytes: u64,
    /// Fixed UDS filename under `base_dir`.
    pub socket_name: String,
    /// Hard cap on a single log record / IPC frame (hostile-input defence).
    pub max_frame_bytes: u32,
    /// Retained for wire/config compatibility. **Non-authoritative:** the append
    /// ack always `fsync`s the record + directory regardless of this flag (F3) —
    /// the `fsync` *is* the durability ack, so it can never be disabled. A
    /// non-`fsync`'d append is never treated as an ack.
    pub fsync_on_append: bool,
}

impl Default for CoordConfig {
    fn default() -> Self {
        Self {
            base_dir: PathBuf::from("coord"),
            segment_bytes: DEFAULT_SEGMENT_BYTES,
            socket_name: DEFAULT_SOCKET_NAME.to_string(),
            max_frame_bytes: DEFAULT_MAX_FRAME_BYTES,
            fsync_on_append: true,
        }
    }
}

impl CoordConfig {
    /// Coordination directory placed **beside** the store at `<store>.coord`.
    ///
    /// The `lbug` persistent store is a single file at `store`; the coordination
    /// directory must therefore be a sibling, not a child (a child path would be
    /// destroyed the moment `open_persistent` creates the store file). `.coord`
    /// sorts next to the store and its `.wal`/`.corrupt-*` sidecars, so an
    /// ordinary directory backup captures it, and it never collides with them.
    pub fn for_store(store: impl AsRef<Path>) -> Self {
        let mut base = store.as_ref().as_os_str().to_owned();
        base.push(".coord");
        Self {
            base_dir: PathBuf::from(base),
            ..Self::default()
        }
    }

    /// Path of the on-disk lease file.
    pub(crate) fn lease_path(&self) -> PathBuf {
        self.base_dir.join("lease")
    }

    /// Path of the dedicated lease RMW lock file. Kept separate from the lease
    /// record itself so the `flock` serialization survives the atomic-`rename`
    /// lease update (F5), which replaces the lease inode.
    pub(crate) fn lease_lock_path(&self) -> PathBuf {
        self.base_dir.join("lease.lock")
    }

    /// Directory holding the segmented, append-only intent log.
    pub(crate) fn intent_log_dir(&self) -> PathBuf {
        self.base_dir.join("intent-log")
    }

    /// Path of the durable applied-index cursor.
    #[cfg(feature = "persistent")]
    pub(crate) fn applied_index_path(&self) -> PathBuf {
        self.base_dir.join("applied-index")
    }

    /// Path of the read-transport socket.
    pub fn socket_path(&self) -> PathBuf {
        self.base_dir.join(&self.socket_name)
    }
}

/// Map a raw I/O error to [`MemoryError::Storage`] with a structural (payload-
/// free) context label.
pub(crate) fn io_err(context: &str, e: io::Error) -> MemoryError {
    MemoryError::Storage(format!("coord {context}: {e}"))
}

/// Create the coordination directory tree (`base_dir` and `intent-log/`) with
/// least-privilege `0o700` permissions, tolerating a concurrent creator. Called
/// on the daemon-side provisioning path (lease acquire); consumer `connect`
/// paths deliberately do **not** create it (no per-agent fallback store).
pub(crate) fn ensure_coord_dirs(config: &CoordConfig) -> Result<()> {
    create_dir_0700(&config.base_dir)?;
    create_dir_0700(&config.intent_log_dir())?;
    Ok(())
}

/// Create a single directory `0o700` if absent, tolerating races and missing
/// parents (recursive).
fn create_dir_0700(dir: &Path) -> Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    match std::fs::DirBuilder::new()
        .mode(0o700)
        .recursive(true)
        .create(dir)
    {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => Ok(()),
        Err(e) => Err(io_err("create-coord-dir", e)),
    }
}

/// Return `true` iff the coordination directory has been provisioned (both the
/// base dir and the intent-log dir exist). Consumers fail closed when this is
/// `false` rather than inventing a fallback store.
pub(crate) fn coord_dirs_exist(config: &CoordConfig) -> bool {
    config.base_dir.is_dir() && config.intent_log_dir().is_dir()
}

// ---------------------------------------------------------------------------
// CRC32 (IEEE) — crash-safe framing checksum, no external dependency.
// ---------------------------------------------------------------------------

/// IEEE CRC-32 of `bytes`. Used as the per-record framing checksum so a torn or
/// corrupt frame is detected on read (the same crash-safe idiom as `lbug`'s
/// WAL). A small self-contained implementation avoids a new dependency.
pub(crate) fn crc32(bytes: &[u8]) -> u32 {
    // Standard reflected IEEE polynomial 0xEDB88320, computed without a static
    // table (frames are small and appends are already fsync-bound, so the table
    // build cost is irrelevant and this keeps the code dependency-free).
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

// ---------------------------------------------------------------------------
// flock — brief advisory lock for RMW/append serialization.
// ---------------------------------------------------------------------------

// SAFETY: `flock(2)` is declared here to avoid pulling in the `libc` crate for a
// single, well-specified syscall. The layer targets Linux, where these ABI
// constants and the signature are stable.
extern "C" {
    fn flock(fd: std::os::raw::c_int, operation: std::os::raw::c_int) -> std::os::raw::c_int;
}

const LOCK_EX: std::os::raw::c_int = 2;
const LOCK_UN: std::os::raw::c_int = 8;

/// An exclusive advisory lock held on an open file descriptor for the duration
/// of a brief read-modify-write / append. It is a **serialization primitive
/// only** — it is emphatically *not* store ownership (ownership is the epoch).
pub(crate) struct FlockGuard {
    fd: std::os::raw::c_int,
}

impl FlockGuard {
    /// Take an exclusive `flock` on `fd`, blocking until acquired.
    pub(crate) fn acquire(fd: std::os::raw::c_int) -> Result<Self> {
        // SAFETY: `fd` is a valid open descriptor owned by the caller for the
        // lifetime of the guard.
        let rc = unsafe { flock(fd, LOCK_EX) };
        if rc != 0 {
            return Err(io_err("flock", io::Error::last_os_error()));
        }
        Ok(Self { fd })
    }
}

impl Drop for FlockGuard {
    fn drop(&mut self) {
        // SAFETY: releasing the lock we hold on our own descriptor.
        unsafe {
            flock(self.fd, LOCK_UN);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_known_vectors() {
        // "123456789" -> 0xCBF43926 is the canonical IEEE CRC-32 check value.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn for_store_places_coord_beside_store() {
        let cfg = CoordConfig::for_store("/tmp/x");
        assert_eq!(cfg.base_dir, PathBuf::from("/tmp/x.coord"));
        assert_eq!(cfg.socket_name, "read.sock");
        assert_eq!(cfg.segment_bytes, DEFAULT_SEGMENT_BYTES);
        assert!(cfg.fsync_on_append);
    }
}
