// rust/amplihack-memory/tests/f1_torn_tail_quarantine.rs
//
// TDD contract for **F1 — torn-tail quarantine BEFORE a new append**.
//
// Bug (Design C coordination layer): `SegmentedLog::append` chooses the target
// segment and computes `existing_len` from raw file metadata WITHOUT first
// quarantining a never-acked torn tail left by a previously-crashed writer. A
// torn tail is a partial final frame (a writer died mid-`write_all`, before its
// `fsync`/ack). When the next writer appends a *complete* frame after it, the
// torn bytes stop being an EOF tail and become **committed-interior** bytes:
//   * a plausible-but-wrong length prefix makes the reader parse across the torn
//     region into the following frame, fail CRC, and raise the hard, unskippable
//     `MemoryError::LogCorruption` — a single never-acked partial write has
//     poisoned the entire durable log; OR
//   * an oversized length prefix makes the reader classify the region as a torn
//     tail at *read* time and truncate it away — taking the following ACKED
//     frame with it (a lost acked write → violates `NoLostAckedWrite`).
//
// F1 fix: under the append `flock`, CRC-validate and quarantine/truncate a
// torn tail *before* computing `existing_len` and appending, so:
//   1. a torn tail is never converted into interior corruption, and
//   2. a later acked frame is never eaten by a lazy read-time truncation.
//
// Pins `PrefixConsistency` + `NoLostAckedWrite` (`specs/DurableLog.tla`).
//
// Both tests are RED on baseline 32de5be:
//   * `moderate` -> `drain()` errors with `LogCorruption` (poisoned log).
//   * `oversized` -> the following acked frame is truncated away (lost write).
//
// Requires `persistent` (verification drains through the applier + lbug store).
#![cfg(all(feature = "coord", feature = "persistent"))]

use std::io::Write;
use std::path::PathBuf;

use amplihack_memory::coord::{Applier, CoordConfig, WriteIntent, WriterClient};
use amplihack_memory::{CognitiveMemory, MemoryError, RecallOptions};
use uuid::Uuid;

/// Provision the coord dir (daemon-side step) so a `WriterClient` can connect.
fn provision(cfg: &CoordConfig) {
    let lease = amplihack_memory::coord::Lease::acquire(cfg, "provision").expect("provision");
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

/// Path of segment 0 under the coord dir (matches `SegmentedLog::segment_path`:
/// 12-digit zero-padded index + `.seg` under `intent-log/`).
fn segment0_path(cfg: &CoordConfig) -> PathBuf {
    cfg.base_dir
        .join("intent-log")
        .join(format!("{:012}.seg", 0))
}

/// Append raw bytes directly to segment 0 with `O_APPEND`, simulating a writer
/// that died mid-`write_all` (a never-acked torn tail: a length prefix followed
/// by a *partial* payload and NO crc).
fn append_torn_tail(cfg: &CoordConfig, declared_payload_len: u32, partial_payload: &[u8]) {
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(segment0_path(cfg))
        .expect("open segment 0 to inject torn tail");
    f.write_all(&declared_payload_len.to_be_bytes())
        .expect("write torn length prefix");
    f.write_all(partial_payload).expect("write partial payload");
    // Deliberately no crc, and fewer payload bytes than declared: a torn tail.
    f.sync_all().ok();
}

fn recall_count(store: &std::path::Path, needle: &str) -> usize {
    let mut mem = CognitiveMemory::open_persistent(store, "engineer").expect("reopen store");
    mem.recall_facts_ranked(
        "torn-tail",
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

#[test]
fn torn_tail_before_new_append_does_not_poison_the_log() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");
    let cfg = CoordConfig::for_store(&store);
    provision(&cfg);

    let w = WriterClient::connect(&cfg).expect("connect");

    // (A) A first, fully-acked write.
    w.append(&fact("record-A")).expect("append A acked");

    // A previous writer crashed mid-append here: a plausible (in-range) length
    // prefix with a truncated payload and no crc. 40 < max_frame_bytes so the
    // reader treats it as a real frame length, not an implausible tail.
    append_torn_tail(&cfg, 40, b"partial-bytes");

    // (B) The next writer appends a fully-acked write. With F1 the torn tail is
    // quarantined first, so B lands cleanly right after A.
    w.append(&fact("record-B")).expect("append B acked");

    // The daemon drains. F1: both A and B apply, no corruption. Baseline: the
    // torn region is now interior, the reader parses across it into B, fails CRC,
    // and raises the hard, unskippable LogCorruption — the log is poisoned.
    let mut applier = Applier::open(&store, "engineer", &cfg).expect("applier opens");
    let drained = applier.drain();
    drop(applier);

    assert!(
        drained.is_ok(),
        "a never-acked torn tail must not poison the durable log; \
         drain returned {drained:?}"
    );
    assert_eq!(drained.unwrap(), 2, "both acked writes A and B must apply");

    assert_eq!(
        recall_count(&store, "record-A"),
        1,
        "A present exactly once"
    );
    assert_eq!(
        recall_count(&store, "record-B"),
        1,
        "B present exactly once"
    );
}

#[test]
fn oversized_length_prefix_torn_tail_is_quarantined_without_losing_the_following_acked_write() {
    // Security + durability: a hostile/corrupt OVERSIZED length prefix at a torn
    // tail must be bounded (never drives a huge allocation) AND must never cause
    // a following ACKED frame to be truncated away.
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");
    let cfg = CoordConfig::for_store(&store);
    provision(&cfg);

    let w = WriterClient::connect(&cfg).expect("connect");
    w.append(&fact("record-A")).expect("append A acked");

    // Torn tail whose length prefix far exceeds max_frame_bytes (hostile input).
    append_torn_tail(&cfg, u32::MAX, b"x");

    // A fully-acked write follows the torn tail.
    w.append(&fact("record-B")).expect("append B acked");

    let mut applier = Applier::open(&store, "engineer", &cfg).expect("applier opens");
    let drained = applier
        .drain()
        .expect("oversized torn tail must be quarantined, not fatal");
    drop(applier);

    // With F1 the oversized torn tail is quarantined at append time, so BOTH
    // acked writes survive. On baseline the read-time quarantine truncates the
    // segment at the torn offset and eats the following acked frame B.
    assert!(
        drained >= 2,
        "both acked writes must apply, applied {drained}"
    );
    assert_eq!(recall_count(&store, "record-A"), 1, "A must survive");
    assert_eq!(
        recall_count(&store, "record-B"),
        1,
        "the ACKED write after an oversized torn tail must NOT be lost"
    );
}

#[test]
fn interior_corruption_is_still_a_hard_error_after_f1() {
    // Guardrail: F1 must NOT weaken the interior-corruption contract. A CRC fault
    // on a genuinely committed (non-tail) record is still an unskippable
    // LogCorruption — F1 only rescues never-acked torn tails.
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");
    let cfg = CoordConfig::for_store(&store);
    provision(&cfg);

    let w = WriterClient::connect(&cfg).expect("connect");
    w.append(&fact("record-A")).expect("append A acked");
    w.append(&fact("record-B")).expect("append B acked");

    // Corrupt a byte inside the FIRST (committed, interior) frame's payload.
    // Offset 4 skips the 4-byte length prefix and lands in A's payload.
    let seg = segment0_path(&cfg);
    let mut bytes = std::fs::read(&seg).expect("read segment");
    bytes[8] ^= 0xFF;
    std::fs::write(&seg, &bytes).expect("write corrupted segment");

    let mut applier = Applier::open(&store, "engineer", &cfg).expect("applier opens");
    let err = applier
        .drain()
        .expect_err("interior corruption must remain a hard error");
    assert!(
        matches!(err, MemoryError::LogCorruption { .. }),
        "committed-interior corruption must stay LogCorruption, got {err:?}"
    );
}
