//! On-disk, epoch-fenced store lease (`NoSplitBrain`, `specs/FencedApplier.tla`).
//!
//! The [`Epoch`] is a **monotonic** fencing token and the **sole** ownership
//! signal for the single-writer `lbug` store. Acquiring or renewing the lease
//! bumps the epoch under a brief `flock`-guarded read-modify-write; the applier
//! stamps its epoch and re-checks it before every apply, so a stale-epoch write
//! is rejected. Process **liveness (`kill(pid, 0)`) is never consulted** — there
//! is no PID/liveness API here at all, by design.

use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;

use crate::coord::{crc32, ensure_coord_dirs, io_err, CoordConfig, FlockGuard};
use crate::errors::{MemoryError, Result};

/// A monotonic lease epoch: the fencing token and sole store-ownership signal.
pub type Epoch = u64;

const LEASE_MAGIC: &[u8; 4] = b"LEA1";
const LEASE_VERSION: u16 = 1;
/// Fixed header size: magic(4) + version(2) + epoch(8) + holder_len(2).
const HEADER_LEN: usize = 4 + 2 + 8 + 2;
/// Defensive upper bound on the holder label length read back from disk.
const MAX_HOLDER_LEN: usize = 4096;

/// A held store lease carrying its [`Epoch`].
///
/// Constructed by [`acquire`](Self::acquire); the epoch it holds only ever
/// becomes stale relative to a later acquisition — it never silently changes
/// under the holder, which is exactly what makes the applier's fence reject a
/// superseded owner.
#[derive(Debug)]
pub struct Lease {
    config: CoordConfig,
    epoch: Epoch,
    holder: String,
}

impl Lease {
    /// Acquire (or steal) the store lease, minting a strictly greater
    /// [`Epoch`]. The RMW runs under a brief exclusive `flock` on the lease
    /// file; the `flock` **serializes** the update — it is *not* ownership.
    ///
    /// Acquisition deliberately permits an unsafe steal (it may fire while a
    /// previous holder is still running), mirroring `FencedApplier.tla`'s
    /// `Acquire`. Safety comes from the fence at *apply* time, not from refusing
    /// to acquire. Provisions the coordination directory on first use.
    ///
    /// # Errors
    /// [`MemoryError::Storage`] if the coord dir or lease file cannot be created
    /// / read / written, or if an existing lease record is corrupt.
    pub fn acquire(config: &CoordConfig, holder: &str) -> Result<Self> {
        ensure_coord_dirs(config)?;

        // Serialize the RMW on a DEDICATED lock file whose inode is stable. The
        // lease record itself is updated by atomic rename (F5), which replaces
        // the lease inode — so an `flock` taken on the lease file would be
        // defeated by the swap. A separate `lease.lock` keeps the `flock`
        // serialization intact across the rename.
        let lock_file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .mode(0o600)
            .open(config.lease_lock_path())
            .map_err(|e| io_err("open-lease-lock", e))?;

        let _guard = FlockGuard::acquire(lock_file.as_raw_fd())?;

        let current = read_epoch_at(config)?.unwrap_or(0);
        let next = current
            .checked_add(1)
            .ok_or_else(|| MemoryError::Storage("lease epoch overflow".into()))?;

        write_record_atomic(config, next, holder)?;

        Ok(Self {
            config: config.clone(),
            epoch: next,
            holder: holder.to_string(),
        })
    }

    /// Renew the held lease, bumping the epoch again to extend fencing.
    ///
    /// # Errors
    /// [`MemoryError::Storage`] on any I/O or corrupt-record failure.
    pub fn renew(&mut self) -> Result<Epoch> {
        let lock_file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .mode(0o600)
            .open(self.config.lease_lock_path())
            .map_err(|e| io_err("open-lease-lock-renew", e))?;

        let _guard = FlockGuard::acquire(lock_file.as_raw_fd())?;

        let current = read_epoch_at(&self.config)?.unwrap_or(0);
        let next = current
            .checked_add(1)
            .ok_or_else(|| MemoryError::Storage("lease epoch overflow".into()))?;
        write_record_atomic(&self.config, next, &self.holder)?;
        self.epoch = next;
        Ok(next)
    }

    /// Release the lease. The on-disk epoch is left in place so the next
    /// acquirer strictly increases it; only this in-memory handle is consumed.
    #[allow(clippy::unnecessary_wraps)]
    pub fn release(self) -> Result<()> {
        Ok(())
    }

    /// This lease's epoch.
    pub fn epoch(&self) -> Epoch {
        self.epoch
    }

    /// Read the current on-disk lease epoch without acquiring.
    ///
    /// Lock-free: because a lease update is committed by atomic `rename` (F5),
    /// the lease path always resolves to a complete record — a reader can never
    /// observe a torn/empty lease — so no `flock` is required to read it.
    ///
    /// # Errors
    /// **Fails closed** with [`MemoryError::Storage`] if no lease has ever been
    /// written (a missing/empty lease is not epoch `0`, which a stale writer
    /// could then match) or if the record is corrupt.
    pub fn current_epoch(config: &CoordConfig) -> Result<Epoch> {
        let bytes = match std::fs::read(config.lease_path()) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(MemoryError::Storage(
                    "no store lease exists (fail closed)".into(),
                ));
            }
            Err(e) => return Err(io_err("read-lease-current", e)),
        };
        if bytes.is_empty() {
            return Err(MemoryError::Storage(
                "store lease is empty (fail closed)".into(),
            ));
        }
        decode_record(&bytes)
    }
}

/// Read and validate the on-disk lease record, returning its epoch, or `None` if
/// the lease has never been written (missing or empty file).
fn read_epoch_at(config: &CoordConfig) -> Result<Option<Epoch>> {
    match std::fs::read(config.lease_path()) {
        Ok(bytes) if bytes.is_empty() => Ok(None),
        Ok(bytes) => decode_record(&bytes).map(Some),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(io_err("read-lease", e)),
    }
}

/// F5: commit a lease record atomically — write it to a same-dir temp file,
/// `fsync` the temp, `rename` it over the lease path, then `fsync` the parent
/// directory (the proven `persist_applied_index` idiom). A crash mid-update
/// therefore leaves EITHER the old complete record OR the new complete record —
/// the live lease is never observed empty/torn, so a never-committed update can
/// never destroy the previously-committed epoch. The parent-dir `fsync` also
/// covers F3 for first-time lease creation.
fn write_record_atomic(config: &CoordConfig, epoch: Epoch, holder: &str) -> Result<()> {
    let record = encode_record(epoch, holder)?;
    let final_path = config.lease_path();
    let tmp_path = config.base_dir.join(".lease.tmp");
    {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .mode(0o600)
            .open(&tmp_path)
            .map_err(|e| io_err("open-lease-tmp", e))?;
        f.write_all(&record)
            .map_err(|e| io_err("write-lease-tmp", e))?;
        f.sync_all().map_err(|e| io_err("fsync-lease-tmp", e))?;
    }
    std::fs::rename(&tmp_path, &final_path).map_err(|e| io_err("rename-lease", e))?;
    let dir = std::fs::File::open(&config.base_dir).map_err(|e| io_err("open-lease-dir", e))?;
    dir.sync_all().map_err(|e| io_err("fsync-lease-dir", e))?;
    Ok(())
}

/// `magic | version | epoch | holder_len | holder | crc32`.
fn encode_record(epoch: Epoch, holder: &str) -> Result<Vec<u8>> {
    let hb = holder.as_bytes();
    let holder_len = u16::try_from(hb.len())
        .map_err(|_| MemoryError::Storage("lease holder label too long".into()))?;
    let mut buf = Vec::with_capacity(HEADER_LEN + hb.len() + 4);
    buf.extend_from_slice(LEASE_MAGIC);
    buf.extend_from_slice(&LEASE_VERSION.to_be_bytes());
    buf.extend_from_slice(&epoch.to_be_bytes());
    buf.extend_from_slice(&holder_len.to_be_bytes());
    buf.extend_from_slice(hb);
    let crc = crc32(&buf);
    buf.extend_from_slice(&crc.to_be_bytes());
    Ok(buf)
}

fn decode_record(bytes: &[u8]) -> Result<Epoch> {
    if bytes.len() < HEADER_LEN + 4 {
        return Err(MemoryError::Storage("lease record truncated".into()));
    }
    if &bytes[0..4] != LEASE_MAGIC {
        return Err(MemoryError::Storage("lease record bad magic".into()));
    }
    let version = u16::from_be_bytes([bytes[4], bytes[5]]);
    if version != LEASE_VERSION {
        return Err(MemoryError::Storage(format!(
            "unsupported lease version {version}"
        )));
    }
    let epoch = Epoch::from_be_bytes([
        bytes[6], bytes[7], bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13],
    ]);
    let holder_len = u16::from_be_bytes([bytes[14], bytes[15]]) as usize;
    if holder_len > MAX_HOLDER_LEN {
        return Err(MemoryError::Storage(
            "lease holder length implausible".into(),
        ));
    }
    let end = HEADER_LEN + holder_len;
    if bytes.len() != end + 4 {
        return Err(MemoryError::Storage("lease record length mismatch".into()));
    }
    let stored_crc =
        u32::from_be_bytes([bytes[end], bytes[end + 1], bytes[end + 2], bytes[end + 3]]);
    if crc32(&bytes[..end]) != stored_crc {
        return Err(MemoryError::Storage("lease record crc mismatch".into()));
    }
    Ok(epoch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_round_trips() {
        let enc = encode_record(42, "holder-x").expect("encode");
        assert_eq!(decode_record(&enc).expect("decode"), 42);
    }

    #[test]
    fn corrupt_record_fails_closed() {
        let mut enc = encode_record(7, "h").expect("encode");
        let last = enc.len() - 1;
        enc[last] ^= 0xFF; // flip a crc byte
        assert!(decode_record(&enc).is_err());
    }
}
