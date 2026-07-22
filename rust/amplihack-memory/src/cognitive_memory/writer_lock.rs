//! Single-writer ownership lock for the persistent `lbug` store (F4).
//!
//! `open_persistent` takes an exclusive, **non-blocking** advisory `flock` on a
//! store-adjacent lock file before opening the store. A second opener — in any
//! process, or a second live handle in this process — is refused immediately
//! with [`MemoryError::AlreadyLocked`] rather than opening a concurrent writer
//! over the same store (which would be split-brain at the store layer). The
//! lock is bound to the returned handle's lifetime and released on `Drop`, so a
//! legitimate reopen after the first handle is dropped still succeeds.
//!
//! Ownership is fail-closed and never consults process liveness (`kill(pid, 0)`)
//! — that is TOCTOU/pid-reuse unsafe. This is the store-ownership half of
//! `NoSplitBrain` (`specs/FencedApplier.tla`).

use std::os::unix::io::AsRawFd;
use std::path::Path;

use crate::errors::{MemoryError, Result};

// SAFETY: `flock(2)` is declared here to avoid pulling in the `libc` crate for a
// single, well-specified syscall. The persistent backend targets Linux, where
// these ABI constants and the signature are stable.
extern "C" {
    fn flock(fd: std::os::raw::c_int, operation: std::os::raw::c_int) -> std::os::raw::c_int;
}

const LOCK_EX: std::os::raw::c_int = 2;
const LOCK_NB: std::os::raw::c_int = 4;
const LOCK_UN: std::os::raw::c_int = 8;

/// An exclusive, non-blocking advisory lock on a store's writer-lock file, held
/// for the lifetime of the owning [`CognitiveMemory`](super::CognitiveMemory)
/// handle and released on `Drop`.
#[derive(Debug)]
pub(crate) struct WriterLock {
    // The open file keeps the descriptor (and thus the `flock`) alive; it is
    // never read/written. Dropping it after `LOCK_UN` closes the descriptor.
    file: std::fs::File,
}

impl WriterLock {
    /// Take the exclusive writer lock for `store_path`, failing closed if it is
    /// already held.
    ///
    /// # Errors
    /// [`MemoryError::AlreadyLocked`] if another owner holds the lock;
    /// [`MemoryError::Storage`] if the lock file cannot be created/opened.
    pub(crate) fn acquire(store_path: &Path) -> Result<Self> {
        let lock_path = Self::lock_path(store_path);
        if let Some(parent) = lock_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| MemoryError::Storage(format!("writer-lock create-parent: {e}")))?;
            }
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .map_err(|e| MemoryError::Storage(format!("writer-lock open: {e}")))?;

        // SAFETY: `file` owns a valid descriptor for the call.
        let rc = unsafe { flock(file.as_raw_fd(), LOCK_EX | LOCK_NB) };
        if rc != 0 {
            let err = std::io::Error::last_os_error();
            return match err.raw_os_error() {
                // EWOULDBLOCK / EAGAIN: the lock is held by another owner.
                Some(libc_ewouldblock) if is_would_block(libc_ewouldblock) => {
                    Err(MemoryError::AlreadyLocked {
                        path: store_path.display().to_string(),
                    })
                }
                _ => Err(MemoryError::Storage(format!("writer-lock flock: {err}"))),
            };
        }
        Ok(Self { file })
    }

    /// The path of the writer-lock file placed beside the store.
    fn lock_path(store_path: &Path) -> std::path::PathBuf {
        let mut p = store_path.as_os_str().to_owned();
        p.push(".writer.lock");
        std::path::PathBuf::from(p)
    }
}

/// `EWOULDBLOCK` and `EAGAIN` share the value on Linux (11), but check both of
/// the canonical values defensively.
fn is_would_block(errno: i32) -> bool {
    const EAGAIN: i32 = 11;
    const EWOULDBLOCK: i32 = 11;
    errno == EAGAIN || errno == EWOULDBLOCK
}

impl Drop for WriterLock {
    fn drop(&mut self) {
        // SAFETY: releasing the lock we hold on our own descriptor. The file is
        // closed immediately afterwards when `file` drops.
        unsafe {
            flock(self.file.as_raw_fd(), LOCK_UN);
        }
    }
}
