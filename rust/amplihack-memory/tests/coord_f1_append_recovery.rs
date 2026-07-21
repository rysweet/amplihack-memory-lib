// rust/amplihack-memory/tests/coord_f1_append_recovery.rs
//
// TDD crash-injection contract for **[F1] torn-tail-then-append** in the durable
// shared intent log (`src/coord/intent_log.rs`).
//
// The bug: `append` computes its write position purely from the on-disk segment
// length and `O_APPEND`s WITHOUT first validating that the tail is a complete,
// CRC-valid frame. If a writer is SIGKILL'd mid-`write_all` (before its
// fsync/ack) it leaves a partial frame; `flock` releases on death; the NEXT
// writer then appends its complete, fsync'd, ACKED frame R' AFTER the garbage.
// On read this is either quarantined as a torn tail (TRUNCATING away the acked
// R' -> lost acked write) or the partial's length prefix plausibly spans into R'
// -> CRC mismatch -> hard `LogCorruption` -> the applier is wedged and R' never
// applies. Both violate `NoLostAckedWrite` / `PrefixConsistency`
// (`specs/DurableLog.tla`).
//
// The fix: on the append path, under the append flock, recover the tail BEFORE
// writing — validate the segment ends on a complete CRC-valid frame boundary and
// `set_len` away any trailing torn partial before computing the write position.
// Recovery only ever discards a torn PARTIAL, never a complete fsynced frame.
//
// This test FAILS without the fix (drain errors with LogCorruption, or the acked
// survivor is truncated away) and PASSES with it.
//
// Verification goes through the applier + lbug store, so `persistent` is required.
#![cfg(all(feature = "coord", feature = "persistent"))]

use std::fs::OpenOptions;
use std::io::Write;

use amplihack_memory::coord::{Applier, CoordConfig, Lease, WriteIntent, WriterClient};
use amplihack_memory::{CognitiveMemory, RecallOptions};
use uuid::Uuid;

fn provision(cfg: &CoordConfig) {
    let lease = Lease::acquire(cfg, "provision").expect("provision coord dir");
    lease.release().expect("release provision lease");
}

fn fact(content: &str) -> WriteIntent {
    WriteIntent::StoreFact {
        intent_id: Uuid::new_v4(),
        agent_name: "engineer".into(),
        concept: "torn-tail".into(),
        content: content.into(),
        confidence: 0.9,
        source_id: "engineer".into(),
        tags: None,
        metadata: None,
    }
}

/// The single segment file of a fresh log (`{index:012}.seg`).
fn first_segment(cfg: &CoordConfig) -> std::path::PathBuf {
    cfg.base_dir.join("intent-log").join("000000000000.seg")
}

fn recall_count(store: &std::path::Path, needle: &str) -> usize {
    let mut mem = CognitiveMemory::open_persistent(store, "daemon").expect("reopen store");
    mem.recall_facts_ranked(
        "torn tail frame survivor",
        RecallOptions {
            limit: 1000,
            record_access: false,
            ..Default::default()
        },
    )
    .expect("recall")
    .iter()
    .filter(|h| h.item.content == needle)
    .count()
}

/// Write a torn PARTIAL frame directly onto the segment: a 4-byte length prefix
/// claiming `declared_payload` bytes, followed by only `wrote` (< declared)
/// payload bytes and NO crc — exactly what a writer SIGKILL'd mid-`write_all`
/// leaves behind. `flock` released on death is modelled by simply dropping the
/// file handle here.
fn inject_torn_partial(seg: &std::path::Path, declared_payload: u32, wrote: usize) {
    let mut f = OpenOptions::new()
        .append(true)
        .open(seg)
        .expect("open segment to inject torn tail");
    f.write_all(&declared_payload.to_be_bytes())
        .expect("write torn length prefix");
    f.write_all(&vec![0xABu8; wrote])
        .expect("write torn partial payload");
    f.sync_all().expect("fsync torn partial");
}

#[test]
fn append_after_torn_tail_preserves_the_acked_write() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");
    let cfg = CoordConfig::for_store(&store);
    provision(&cfg);

    let w = WriterClient::connect(&cfg).expect("connect");

    // Two complete, fsync'd, ACKED frames.
    w.append(&fact("pre-0")).expect("append pre-0 acked");
    w.append(&fact("pre-1")).expect("append pre-1 acked");

    // A writer died mid-append: a torn partial now sits at the tail. Declared
    // length 32 but only 3 payload bytes on disk and no crc. When a real frame
    // follows, the declared frame (need = 4 + 32 + 4) fits within the remaining
    // bytes and its "crc" is read out of the survivor -> interior corruption.
    inject_torn_partial(&first_segment(&cfg), 32, 3);

    // The next writer appends a complete, fsync'd, ACKED frame. With the F1 fix
    // this recovers (truncates) the torn tail first; without it, R' lands after
    // the garbage.
    w.append(&fact("survivor")).expect("append survivor acked");

    // Drain the durable log. Without the fix this returns LogCorruption (wedged)
    // or silently drops the survivor; with the fix every acked frame applies.
    let mut applier = Applier::open(&store, "daemon", &cfg).expect("applier opens");
    let drained = applier.drain();
    assert!(
        drained.is_ok(),
        "a torn tail followed by an acked append must never corrupt or wedge \
         the log; drain returned {drained:?}"
    );
    drop(applier);

    // The acked survivor and both earlier acked frames must be present exactly
    // once; nothing is lost, nothing is duplicated, the torn partial is gone.
    assert_eq!(
        recall_count(&store, "survivor"),
        1,
        "the acked write appended after the torn tail must survive (NoLostAckedWrite)"
    );
    assert_eq!(
        recall_count(&store, "pre-0"),
        1,
        "earlier acked frame intact"
    );
    assert_eq!(
        recall_count(&store, "pre-1"),
        1,
        "earlier acked frame intact"
    );
}

#[test]
fn recovery_never_truncates_past_a_crc_valid_frame() {
    // Integrity guard tied to the F1 security control: recovery must discard ONLY
    // a trailing torn partial, never a complete fsynced frame that precedes it.
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");
    let cfg = CoordConfig::for_store(&store);
    provision(&cfg);

    let w = WriterClient::connect(&cfg).expect("connect");
    w.append(&fact("keep-0")).expect("append keep-0 acked");
    w.append(&fact("keep-1")).expect("append keep-1 acked");

    // Length of the two complete frames before we corrupt the tail.
    let clean_len = std::fs::metadata(first_segment(&cfg))
        .expect("stat segment")
        .len();

    // A torn partial at the very end, with no following frame.
    inject_torn_partial(&first_segment(&cfg), 64, 5);

    // Draining recovers the tail: the two complete frames stay, only the partial
    // is dropped.
    let mut applier = Applier::open(&store, "daemon", &cfg).expect("applier opens");
    applier
        .drain()
        .expect("drain recovers torn tail without loss");
    drop(applier);

    assert_eq!(
        recall_count(&store, "keep-0"),
        1,
        "complete frame preserved"
    );
    assert_eq!(
        recall_count(&store, "keep-1"),
        1,
        "complete frame preserved"
    );

    // The segment was truncated back to exactly the last CRC-valid boundary —
    // never further (which would eat an acked frame).
    let recovered_len = std::fs::metadata(first_segment(&cfg))
        .expect("stat segment after recovery")
        .len();
    assert_eq!(
        recovered_len, clean_len,
        "recovery must truncate to the last CRC-valid frame boundary, not past it"
    );
}
