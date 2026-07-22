//! Single-writer OS lock for the persistent `lbug` store (`NoSplitBrain`).
//!
//! The coordination layer's epoch fence bounds a lease-steal race to a single
//! in-flight apply, but it does not stop a *superseded* owner from keeping the
//! store open for write while a new owner also opens it — two processes mutating
//! one single-writer WAL is corruption. This module takes an **exclusive,
//! non-blocking** `flock` on a store-adjacent lock file at open time, held for
//! the store handle's lifetime and released on `Drop`. A second concurrent
//! open therefore **fails closed**, while a legitimate sequential reopen (after
//! the prior handle drops) succeeds — the belt to the fence's suspenders.

use std::os::unix::io::AsRawFd;
use std::os::unix::io::RawFd;
use std::path::{Path, PathBuf};

use crate::errors::{MemoryError, Result};

// SAFETY: `flock(2)` is declared directly to avoid depending on `libc` for a
// single stable Linux syscall (the same idiom the `coord` layer uses).
extern "C" {
    fn flock(fd: std::os::raw::c_int, operation: std::os::raw::c_int) -> std::os::raw::c_int;
}

const LOCK_EX: std::os::raw::c_int = 2;
const LOCK_NB: std::os::raw::c_int = 4;
const LOCK_UN: std::os::raw::c_int = 8;

/// The single-writer lock file placed beside the store at `<store>.writer.lock`.
fn lock_path_for(store_path: &Path) -> PathBuf {
    let mut p = store_path.as_os_str().to_owned();
    p.push(".writer.lock");
    PathBuf::from(p)
}

/// An exclusive OS lock proving this process is the sole live writer of a store.
/// Dropping it (or the process exiting) releases the lock so the next owner can
/// take over.
#[derive(Debug)]
pub(crate) struct StoreWriterLock {
    fd: RawFd,
    // The lock is tied to this open descriptor; keep it alive for the guard's
    // lifetime (closing the file also drops the lock).
    _file: std::fs::File,
}

impl StoreWriterLock {
    /// Acquire the exclusive single-writer lock for `store_path`, failing closed
    /// if another live handle already holds it.
    ///
    /// # Errors
    /// [`MemoryError::Storage`] if the lock file cannot be created, or if the
    /// lock is already held by another writer (`EWOULDBLOCK`).
    pub(crate) fn acquire(store_path: &Path) -> Result<Self> {
        use std::os::unix::fs::OpenOptionsExt;

        // The lock file is a sibling of the store file; ensure its directory
        // exists (the store's own open also creates it, but the lock is taken
        // first, before the store is opened).
        if let Some(parent) = store_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    MemoryError::Storage(format!("store-lock create parent dir: {e}"))
                })?;
            }
        }

        let path = lock_path_for(store_path);
        let file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .mode(0o600)
            .open(&path)
            .map_err(|e| {
                MemoryError::Storage(format!("store-lock open {}: {e}", path.display()))
            })?;

        let fd = file.as_raw_fd();
        // SAFETY: `fd` is a valid descriptor owned by `file` for the guard's life.
        let rc = unsafe { flock(fd, LOCK_EX | LOCK_NB) };
        if rc != 0 {
            let err = std::io::Error::last_os_error();
            return Err(MemoryError::Storage(format!(
                "store {} is already open by another writer (single-writer lock held): {err}",
                store_path.display()
            )));
        }
        Ok(Self { fd, _file: file })
    }
}

impl Drop for StoreWriterLock {
    fn drop(&mut self) {
        // SAFETY: releasing the exclusive lock we hold on our own descriptor.
        unsafe {
            flock(self.fd, LOCK_UN);
        }
    }
}
