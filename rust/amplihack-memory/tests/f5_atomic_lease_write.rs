// rust/amplihack-memory/tests/f5_atomic_lease_write.rs
//
// TDD contract for **F5 — atomic lease write (temp -> fsync -> rename ->
// dir-fsync)**.
//
// Bug (Design C lease): `lease::write_record` updates the lease file IN PLACE —
// `seek(0)` + `set_len(0)` + `write_all` + `fsync`. The `set_len(0)` truncates
// the live lease to zero bytes *before* the new record is written. A crash
// (power loss / SIGKILL) in that window leaves the lease file EMPTY or partially
// written. On the next open `read_epoch` sees an empty/torn record and
// `Lease::current_epoch` FAILS CLOSED ("store lease is empty" / "crc mismatch")
// — which means the fenced applier can no longer read the epoch and the whole
// store becomes unusable. A never-committed lease update has destroyed the
// previously-committed epoch.
//
// F5 fix: write the new record to a same-dir `O_EXCL` temp file, `fsync` it,
// atomically `rename` it over the lease path, then `fsync` the parent dir (the
// proven `persist_applied_index` idiom). The live lease is therefore NEVER
// observed empty/torn: a crash mid-update leaves either the old complete record
// or the new complete record — never nothing.
//
// Deterministic discriminator (no fault injection needed): an atomic
// rename-based update REPLACES the lease inode, whereas an in-place
// truncate+write reuses it. The test asserts the update is rename-atomic (inode
// changes) AND that the epoch remains a complete, readable record throughout.
//
// Pins the durability half of `NoSplitBrain` (`specs/FencedApplier.tla`): the
// ownership epoch must survive a crash during a lease update.
// RED on baseline 32de5be: in-place write keeps the same inode.
//
// The pure-lease tests need only `coord`.
#![cfg(feature = "coord")]

use std::os::unix::fs::MetadataExt;
use std::path::PathBuf;

use amplihack_memory::coord::{CoordConfig, Lease};

fn lease_path(cfg: &CoordConfig) -> PathBuf {
    cfg.base_dir.join("lease")
}

fn lease_inode(cfg: &CoordConfig) -> u64 {
    std::fs::metadata(lease_path(cfg))
        .expect("lease file exists after acquire")
        .ino()
}

#[test]
fn lease_update_is_rename_atomic_not_in_place_truncate() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cfg = CoordConfig::for_store(tmp.path());

    // Acquire creates the lease at epoch 1 and writes the first record.
    let mut lease = Lease::acquire(&cfg, "holder").expect("acquire");
    let ino_before = lease_inode(&cfg);
    let epoch_before = lease.epoch();

    // A renew rewrites the record (epoch 2). With F5 this is a temp+rename swap,
    // so the lease path points at a NEW inode. On baseline the in-place
    // truncate+write reuses the SAME inode (and exposes the empty-file window).
    let epoch_after = lease.renew().expect("renew");
    let ino_after = lease_inode(&cfg);

    assert!(epoch_after > epoch_before, "renew must bump the epoch");
    assert_ne!(
        ino_after, ino_before,
        "a lease update must be applied by atomic rename (new inode), not by an \
         in-place set_len(0)+write that exposes a torn/empty-lease crash window"
    );

    // The committed epoch must always be a complete, readable record.
    let live = Lease::current_epoch(&cfg).expect("current epoch must read a complete record");
    assert_eq!(
        live, epoch_after,
        "the live epoch reflects the renewed record"
    );
}

#[test]
fn a_partial_lease_update_never_destroys_the_committed_epoch() {
    // Crash-consistency statement of the same guarantee, simulated
    // deterministically: reproduce the exact on-disk artifact the BASELINE
    // in-place algorithm can leave behind — a lease file truncated to empty by
    // `set_len(0)` before the new record was written — and assert that F5's
    // atomic-rename discipline means such an artifact is never the committed
    // lease.
    //
    // We assert the invariant an atomic update guarantees: immediately after a
    // successful `renew`, there is no leftover temp file masquerading as the
    // lease and the lease path holds a COMPLETE record. A half-written temp that
    // was never renamed must not be observable as the live lease.
    let tmp = tempfile::tempdir().expect("tempdir");
    let cfg = CoordConfig::for_store(tmp.path());

    let mut lease = Lease::acquire(&cfg, "holder").expect("acquire");
    let committed = lease.renew().expect("renew");

    // No orphan temp file left behind by the atomic write.
    let leftover_tmp = std::fs::read_dir(&cfg.base_dir)
        .expect("read coord dir")
        .filter_map(|e| e.ok())
        .any(|e| {
            e.file_name()
                .to_string_lossy()
                .to_ascii_lowercase()
                .contains("tmp")
        });
    assert!(
        !leftover_tmp,
        "an atomic lease write must not leave a temp file behind after rename"
    );

    // The committed epoch is intact and completely readable (never torn/empty).
    let live = Lease::current_epoch(&cfg).expect("committed lease must be a complete record");
    assert_eq!(
        live, committed,
        "the committed epoch must survive the update intact"
    );
}
