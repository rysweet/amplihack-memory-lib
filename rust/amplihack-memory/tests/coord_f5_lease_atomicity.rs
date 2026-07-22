// rust/amplihack-memory/tests/coord_f5_lease_atomicity.rs
//
// TDD crash-atomicity contract for **[F5] non-atomic in-place lease write ->
// crash can reset the epoch** (`src/coord/lease.rs::write_record`).
//
// The bug: `write_record` does `seek(0) + set_len(0) + write_all + sync_all` in
// place. A crash after the truncate is durable but before the record is written
// leaves the lease file EMPTY; `read_epoch` then returns `None` and the next
// `acquire` mints epoch 1 — a MONOTONICITY BREAK. A previously-fenced stale
// holder can then match epoch 1 again -> split brain (`NoSplitBrain`,
// `specs/FencedApplier.tla`).
//
// The fix: write the record to a temp file, `fsync`, atomically `rename` over the
// lease, then `fsync` the dir — the exact temp+fsync+rename+dir-fsync pattern
// `persist_applied_index` already uses. The on-disk `lease` file is then ALWAYS a
// complete record: it is never observed empty or torn, so the epoch can never
// reset.
//
// This test proves the atomicity externally: under heavy concurrent `acquire`
// contention, a watcher reads the raw `lease` file WITHOUT taking the flock. With
// the in-place write it catches the `set_len(0) -> write` window as an empty/torn
// file (the exact state that resets the epoch) and fails. With the atomic-rename
// fix every observation is a complete record and it passes. A companion test
// pins plain monotonicity across acquire/renew/reopen.
//
// Pure-lease layer: only `coord` is required.
#![cfg(feature = "coord")]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use amplihack_memory::coord::{CoordConfig, Lease};

/// The first four bytes of a valid lease record (`LEASE_MAGIC`), and the minimum
/// record length (header 16 + crc 4). Duplicated here on purpose: an integration
/// test only sees the public surface, so it validates the on-disk shape directly.
const LEASE_MAGIC: &[u8; 4] = b"LEA1";
const MIN_RECORD_LEN: usize = 16 + 4;

#[test]
fn epoch_is_monotonic_across_acquire_renew_and_reopen() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cfg = CoordConfig::for_store(tmp.path());

    let mut l = Lease::acquire(&cfg, "holder").expect("acquire");
    let e1 = l.epoch();
    let e2 = l.renew().expect("renew");
    assert!(e2 > e1, "renew must strictly increase the epoch");
    drop(l);

    // Reopen from disk: the next acquire must continue strictly upward, never
    // reset to 1.
    let l3 = Lease::acquire(&cfg, "holder-2").expect("re-acquire after reopen");
    assert!(
        l3.epoch() > e2,
        "acquire after reopen must exceed the last persisted epoch {e2}, got {}",
        l3.epoch()
    );
}

#[test]
fn concurrent_acquire_never_exposes_an_empty_or_torn_lease() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cfg = CoordConfig::for_store(tmp.path());
    let lease_path = cfg.base_dir.join("lease");

    // Seed a lease so the file exists and is non-empty from the start; any later
    // empty/torn observation is therefore caused by a non-atomic WRITE, not by
    // first creation.
    Lease::acquire(&cfg, "seed")
        .expect("seed acquire")
        .release()
        .expect("release seed");

    let stop = Arc::new(AtomicBool::new(false));

    // Writer threads hammer `acquire`; each call runs `write_record`, which under
    // the buggy in-place path truncates the file to zero before writing.
    let mut writers = Vec::new();
    for _ in 0..4 {
        let cfg = cfg.clone();
        let stop = Arc::clone(&stop);
        writers.push(std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                // A failed acquire (e.g. transient) would itself be a bug; unwrap
                // so the test surfaces it.
                let l = Lease::acquire(&cfg, "w").expect("concurrent acquire");
                let _ = l.release();
            }
        }));
    }

    // Watcher: read the raw lease file WITHOUT the flock, as fast as possible.
    // A valid record is >= MIN_RECORD_LEN bytes and starts with LEASE_MAGIC.
    // Anything else (empty, short, bad magic) is the torn window the atomic
    // write must never expose.
    let saw_torn = Arc::new(AtomicBool::new(false));
    let reads = Arc::new(AtomicU64::new(0));
    let watcher = {
        let saw_torn = Arc::clone(&saw_torn);
        let reads = Arc::clone(&reads);
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                if let Ok(bytes) = std::fs::read(&lease_path) {
                    reads.fetch_add(1, Ordering::Relaxed);
                    let valid = bytes.len() >= MIN_RECORD_LEN && &bytes[0..4] == LEASE_MAGIC;
                    if !valid {
                        saw_torn.store(true, Ordering::Relaxed);
                    }
                }
                // An ENOENT is only possible during an atomic rename swap and is
                // itself acceptable (old->new is atomic); a real torn state is an
                // existing-but-empty/short file, caught above.
            }
        })
    };

    std::thread::sleep(Duration::from_millis(1500));
    stop.store(true, Ordering::Relaxed);
    for w in writers {
        w.join().expect("writer join");
    }
    watcher.join().expect("watcher join");

    assert!(
        reads.load(Ordering::Relaxed) > 0,
        "watcher must have observed the lease file at least once"
    );
    assert!(
        !saw_torn.load(Ordering::Relaxed),
        "the lease write must be crash-atomic: a concurrent reader must NEVER see \
         an empty or torn lease file (that state resets the epoch to 1 on the next \
         acquire and breaks monotonicity). Use temp+fsync+rename+dir-fsync."
    );
}

#[test]
fn epoch_never_regresses_under_a_final_reopen_after_contention() {
    // Belt-and-suspenders: after heavy contention, the persisted epoch a fresh
    // reopen sees must be far above 1 — it must never have reset mid-run.
    let tmp = tempfile::tempdir().expect("tempdir");
    let cfg = CoordConfig::for_store(tmp.path());

    let mut last = 0u64;
    let deadline = Instant::now() + Duration::from_millis(300);
    while Instant::now() < deadline {
        let l = Lease::acquire(&cfg, "seq").expect("acquire");
        let e = l.epoch();
        assert!(
            e > last,
            "epoch must never regress across acquisitions: {e} !> {last}"
        );
        last = e;
        let _ = l.release();
    }
    assert!(
        last > 1,
        "expected the epoch to advance well beyond 1, got {last}"
    );
}
