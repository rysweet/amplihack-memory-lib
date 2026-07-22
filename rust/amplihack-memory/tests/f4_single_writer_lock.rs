// rust/amplihack-memory/tests/f4_single_writer_lock.rs
//
// TDD contract for **F4 — single-writer ownership lock on the lbug store**.
//
// Bug (empirically confirmed on baseline 32de5be): two concurrent
// `CognitiveMemory::open_persistent` calls on the SAME store path both SUCCEED.
// Nothing at the store-handle layer enforces single-writer discipline — the only
// ownership signal is the coordination-layer epoch fence, which protects the
// *applier's* writes but does nothing to stop a second process from opening the
// same lbug store directly and writing concurrently (split-brain at the store).
//
// F4 fix: `open_persistent_with_recovery` takes an advisory writer lock on the
// store with `flock(LOCK_EX | LOCK_NB)` — fail-CLOSED and non-blocking. A second
// opener fails immediately with a typed ownership error (the design names it
// `MemoryError::AlreadyLocked { path }`); it must NEVER fall back to an unlocked
// open, and must NEVER consult pid-liveness (`kill(pid,0)`) to reclaim (TOCTOU /
// pid-reuse). The lock is bound to the handle's lifetime and released on `Drop`,
// so a legitimate reopen after the first handle is dropped still succeeds.
//
// Pins `NoSplitBrain` at the store-ownership layer (`specs/FencedApplier.tla`).
//
// This test is intentionally decoupled from the exact error variant name (which
// does not exist yet) so the file COMPILES on baseline and fails at RUNTIME:
//   * baseline: the second open returns Ok -> assertion fails (RED), and
//   * post-fix: the second open returns a fail-closed lock error -> GREEN.
//
// Requires `persistent`.
#![cfg(all(feature = "coord", feature = "persistent"))]

use amplihack_memory::coord::{Applier, CoordConfig};
use amplihack_memory::CognitiveMemory;

fn err_mentions_lock_ownership(msg: &str) -> bool {
    let m = msg.to_ascii_lowercase();
    m.contains("lock")
        || m.contains("single-writer")
        || m.contains("single writer")
        || m.contains("already")
        || m.contains("owned")
        || m.contains("in use")
}

#[test]
fn second_concurrent_open_fails_closed() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");

    // First writer opens and HOLDS the store.
    let first = CognitiveMemory::open_persistent(&store, "writer-1").expect("first open ok");

    // A second writer must be REFUSED while the first holds the lock. Baseline
    // returns Ok here (the confirmed split-brain gap).
    let second = CognitiveMemory::open_persistent(&store, "writer-2");
    assert!(
        second.is_err(),
        "a second concurrent open of the same store must fail closed (single-writer), \
         but it succeeded — split-brain at the store handle"
    );
    let msg = format!("{}", second.err().unwrap());
    assert!(
        err_mentions_lock_ownership(&msg),
        "the refusal must be a typed ownership/lock error, got: {msg:?}"
    );

    // The lock is bound to the handle lifetime: after the holder is dropped, a
    // legitimate reopen must succeed (Drop-released, not a permanent lockout).
    drop(first);
    let reopened = CognitiveMemory::open_persistent(&store, "writer-3");
    assert!(
        reopened.is_ok(),
        "after the holder drops, a fresh single-writer open must succeed: {:?}",
        reopened.err()
    );
}

#[test]
fn second_applier_open_is_refused_at_the_store_layer() {
    // The single-writer guarantee must hold through the coordination front door
    // too: while one Applier owns the store, a second Applier::open on the same
    // store must be refused at the store-ownership layer — epoch fencing alone
    // (which only fences at APPLY time) does not prevent a second process from
    // opening the lbug writer.
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");
    let cfg = CoordConfig::for_store(&store);

    let first = Applier::open(&store, "daemon-a", &cfg).expect("first applier opens");

    let second = Applier::open(&store, "daemon-b", &cfg);
    assert!(
        second.is_err(),
        "a second Applier must not be able to open the same lbug store while the first holds it"
    );

    // And the lock releases with the handle.
    drop(first);
    let third = Applier::open(&store, "daemon-c", &cfg);
    assert!(
        third.is_ok(),
        "after the first applier drops, a new applier must be able to acquire the store: {:?}",
        third.err()
    );
}
