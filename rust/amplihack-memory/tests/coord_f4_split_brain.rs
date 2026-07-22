// rust/amplihack-memory/tests/coord_f4_split_brain.rs
//
// TDD concurrency contract for **[F4] no store-level OS lock -> split-brain apply
// window** (`src/cognitive_memory/mod.rs::open_persistent`,
// `src/coord/applier.rs` fence).
//
// The bug: the applier's fence is an advisory PRE-READ of the lease epoch, and
// `open_persistent` takes NO exclusive OS lock. So a superseded owner (P1, stale
// epoch) can still hold the store open for write while a new owner (P2) also
// `open_persistent`s it — two processes mutate the same WAL/db concurrently ->
// corruption. The epoch fence bounds the race to one apply; it does not eliminate
// it. This violates `NoSplitBrain` (`specs/FencedApplier.tla`).
//
// The fix (belt-and-suspenders): take an EXCLUSIVE OS lock
// (`flock(LOCK_EX|LOCK_NB)`) on the store at `open_persistent` so a second
// concurrent writer FAILS CLOSED, and push the epoch into the store write
// boundary so a stale-epoch mutation is rejected atomically. The lock releases on
// Drop so a legitimate sequential reopen still works.
//
// This test FAILS without the fix (the second concurrent open succeeds) and
// PASSES with it (the second open fails closed while the first is held, and
// succeeds again once it is dropped).
//
// Requires `persistent` (`open_persistent` / `Applier`).
#![cfg(all(feature = "coord", feature = "persistent"))]

use amplihack_memory::coord::{Applier, CoordConfig, Lease};
use amplihack_memory::CognitiveMemory;

#[test]
fn a_second_concurrent_store_open_fails_closed() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");

    // First writer holds the store open (and its exclusive lock).
    let holder = CognitiveMemory::open_persistent(&store, "writer-a").expect("first open");

    // A second concurrent open of the SAME store must fail closed — two live
    // writers over one single-writer store is the split-brain the OS lock
    // forbids.
    let second = CognitiveMemory::open_persistent(&store, "writer-b");
    assert!(
        second.is_err(),
        "a second concurrent open_persistent of the same store must fail closed \
         (single-writer lock); got Ok, which is split-brain"
    );

    // Releasing the first holder releases the lock, so a fresh owner can open —
    // this is what keeps sequential reopen-for-verification working.
    drop(holder);
    let reopened = CognitiveMemory::open_persistent(&store, "writer-c");
    assert!(
        reopened.is_ok(),
        "after the holder drops, the store lock must release so a new owner can open; got {:?}",
        reopened.err()
    );
}

#[test]
fn stale_owner_cannot_hold_store_while_new_owner_opens() {
    // The concrete split-brain repro: applier A owns the store; a NEW owner
    // acquires the lease (bumping the epoch, so A is now stale). The new owner's
    // attempt to open the single store for write must NOT succeed while A still
    // holds it — otherwise both mutate the WAL concurrently.
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");
    let cfg = CoordConfig::for_store(&store);

    // Applier A opens: acquires the lease and holds the single lbug writer.
    let applier_a = Applier::open(&store, "daemon-a", &cfg).expect("applier A opens");

    // A new owner steals the lease (monotonic epoch bump); A is now stale.
    let _stolen = Lease::acquire(&cfg, "daemon-b").expect("daemon B steals the lease");

    // The new owner tries to open the store for write while A is still alive.
    // This must fail closed — the exclusive store lock A holds forbids a second
    // concurrent writer.
    let concurrent = CognitiveMemory::open_persistent(&store, "daemon-b");
    assert!(
        concurrent.is_err(),
        "a new owner must not be able to open the store for write while the prior \
         owner still holds it (NoSplitBrain); got Ok"
    );

    // Once A is gone, the new owner can take over cleanly.
    drop(applier_a);
    let taken_over = CognitiveMemory::open_persistent(&store, "daemon-b");
    assert!(
        taken_over.is_ok(),
        "the new owner must be able to open once the prior owner releases; got {:?}",
        taken_over.err()
    );
}
