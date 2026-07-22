// rust/amplihack-memory/tests/coord_n1_scan_bound.rs
//
// TDD contract for **[N1, PERF] append() does an O(n^2) full-segment rescan**
// in the durable shared intent log (`src/coord/intent_log.rs`).
//
// THE BUG. On every `append`, under the append `flock`, the writer calls
// `recover_tail(last_segment)`, which re-reads and CRC-validates the ENTIRE last
// segment (up to `segment_bytes`, default 64 MiB) — even in the common clean-tail
// case where the segment already ends on a complete, fsynced frame boundary. So
// filling one segment with `n` frames costs O(n) CRC work on EACH append =>
// O(n^2) total, and it holds the append flock for the whole scan, serializing
// every writer. This is load-bearing tech debt (fix-now per no-defer rule).
//
// THE FIX (design R3/N1). Track/persist a verified clean-tail offset (atomic
// `.clean-tail` sidecar: temp+fsync+rename+dir-fsync). When the observed tail
// equals the known clean boundary, `append` SKIPS the full re-CRC and scans only
// the (empty) unverified suffix; a full `recover_tail` runs ONLY when the tail is
// NOT a known-clean frame boundary. F1's guarantees are preserved unchanged: a
// torn PARTIAL tail is still truncated to the last clean frame, and a bad-CRC
// frame FOLLOWED BY bytes (committed-interior corruption) still hard-fails as
// `LogCorruption` — the clean-tail is only ever a monotone lower bound, never an
// excuse to skip CRC on a consumed frame.
//
// OBSERVABILITY. The implementer exposes a process-global counter in the public
// `intent_log` module:
//
//     amplihack_memory::coord::intent_log::frames_scanned() -> u64
//     amplihack_memory::coord::intent_log::reset_frames_scanned()
//
// incremented once per frame the append-path recovery CRC-validates. On a
// known-clean tail the fix scans ~0 frames per append; without it, ~n frames per
// append.
//
// Requires `persistent` (drain/F1 check goes through `open_persistent`).
#![cfg(all(feature = "coord", feature = "persistent"))]

use std::fs::OpenOptions;
use std::io::Write;

use amplihack_memory::coord::intent_log::{frames_scanned, reset_frames_scanned};
use amplihack_memory::coord::{Applier, CoordConfig, Lease, WriteIntent, WriterClient};
use amplihack_memory::{CognitiveMemory, RecallOptions};
use uuid::Uuid;

/// Number of frames to pre-fill into the single segment. Large enough that an
/// O(n)-per-append full rescan is unmistakably above the bounded-scan budget.
const PREFILL: usize = 400;

/// Per-append scan budget on a KNOWN-CLEAN tail. The clean-tail fast path should
/// scan effectively nothing; a small constant absorbs at most the single new
/// frame / boundary check. It is INDEPENDENT of `PREFILL` — that independence is
/// the whole point (O(1) vs O(n)).
const CLEAN_TAIL_SCAN_BUDGET: u64 = 16;

fn provision(cfg: &CoordConfig) {
    let lease = Lease::acquire(cfg, "provision").expect("provision coord dir");
    lease.release().expect("release provision lease");
}

fn fact(content: &str) -> WriteIntent {
    WriteIntent::StoreFact {
        intent_id: Uuid::new_v4(),
        agent_name: "engineer".into(),
        concept: "scan-bound".into(),
        content: content.into(),
        confidence: 0.9,
        source_id: "engineer".into(),
        tags: None,
        metadata: None,
    }
}

fn first_segment(cfg: &CoordConfig) -> std::path::PathBuf {
    cfg.base_dir.join("intent-log").join("000000000000.seg")
}

fn recall_count(store: &std::path::Path, needle: &str) -> usize {
    let mut mem = CognitiveMemory::open_persistent(store, "daemon").expect("reopen store");
    mem.recall_facts_ranked(
        "scan bound frame survivor",
        RecallOptions {
            limit: 2000,
            record_access: false,
            ..Default::default()
        },
    )
    .expect("recall")
    .iter()
    .filter(|h| h.item.content == needle)
    .count()
}

/// Inject a torn PARTIAL frame (declared length, short payload, no crc) — what a
/// writer SIGKILL'd mid-`write_all` leaves behind (same as the F1 test).
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
fn append_on_known_clean_tail_scans_bounded_frames() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");
    let cfg = CoordConfig::for_store(&store);
    provision(&cfg);

    let w = WriterClient::connect(&cfg).expect("connect");

    // Fill one segment with many complete, fsynced, acked frames. (All land in
    // segment 0: the default segment is 64 MiB and these payloads are tiny.)
    for i in 0..PREFILL {
        w.append(&fact(&format!("fill-{i}"))).expect("append acked");
    }

    // The tail is now a known-clean frame boundary. Measure the scan cost of ONE
    // further append in isolation.
    reset_frames_scanned();
    w.append(&fact("measured")).expect("append acked");
    let scanned = frames_scanned();

    assert!(
        scanned <= CLEAN_TAIL_SCAN_BUDGET,
        "append on a known-clean tail must NOT re-CRC the whole segment: it \
         scanned {scanned} frames (budget {CLEAN_TAIL_SCAN_BUDGET}), which grows \
         with the {PREFILL}-frame fill => O(n^2) append (N1). With the clean-tail \
         fast path this is O(1)."
    );
}

#[test]
fn append_scan_cost_does_not_grow_with_segment_fill() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");
    let cfg = CoordConfig::for_store(&store);
    provision(&cfg);

    let w = WriterClient::connect(&cfg).expect("connect");

    // Scan cost of an append early in the segment...
    for i in 0..20 {
        w.append(&fact(&format!("early-{i}")))
            .expect("append acked");
    }
    reset_frames_scanned();
    w.append(&fact("probe-early")).expect("append acked");
    let scanned_early = frames_scanned();

    // ...must be no worse than an append deep into a well-filled segment. With
    // the O(n^2) bug `scanned_late` balloons with the fill; with the clean-tail
    // fast path both are ~0 and independent of position.
    for i in 0..PREFILL {
        w.append(&fact(&format!("bulk-{i}"))).expect("append acked");
    }
    reset_frames_scanned();
    w.append(&fact("probe-late")).expect("append acked");
    let scanned_late = frames_scanned();

    assert!(
        scanned_late <= scanned_early + CLEAN_TAIL_SCAN_BUDGET,
        "append scan cost must be independent of segment fill: early={scanned_early} \
         frames, late (after {PREFILL} more) ={scanned_late} frames. A late cost \
         that grows with fill is the O(n^2) rescan (N1)."
    );
}

#[test]
fn torn_tail_recovery_still_preserves_f1_semantics() {
    // The clean-tail fast path is a hint only: a torn tail must still be
    // recovered (truncated to the last clean frame) and the following acked
    // append must survive — exactly F1's `NoLostAckedWrite`.
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");
    let cfg = CoordConfig::for_store(&store);
    provision(&cfg);

    let w = WriterClient::connect(&cfg).expect("connect");
    w.append(&fact("pre-0")).expect("append pre-0 acked");
    w.append(&fact("pre-1")).expect("append pre-1 acked");

    // A writer died mid-append: torn partial at the tail (declared 32, only 3
    // bytes, no crc) — its declared frame would otherwise span into the next
    // real frame and read a bogus crc => interior corruption.
    inject_torn_partial(&first_segment(&cfg), 32, 3);

    // The next append must recover (truncate) the torn tail first, NOT trust a
    // stale clean-tail hint and bury the garbage. The clean-tail lower bound
    // predates the injected bytes, so recovery must still run over the suffix.
    w.append(&fact("survivor")).expect("append survivor acked");

    // Drain must not wedge (LogCorruption) or drop the survivor.
    let mut applier = Applier::open(&store, "daemon", &cfg).expect("applier opens");
    let drained = applier.drain();
    assert!(
        drained.is_ok(),
        "a torn tail followed by an acked append must never corrupt or wedge the \
         log even with the clean-tail fast path; drain returned {drained:?}"
    );
    drop(applier);

    for c in ["survivor", "pre-0", "pre-1"] {
        assert_eq!(
            recall_count(&store, c),
            1,
            "acked frame {c:?} must be present exactly once (F1 preserved under N1)"
        );
    }
}
