// rust/amplihack-memory/tests/coord_n2_ledger_prune.rs
//
// TDD contract for **[N2, BLOCKING] the exactly-once ledger grows UNBOUNDED**
// in the fenced applier (`src/coord/applier.rs` + `cognitive_memory/applied_ledger.rs`).
//
// THE BUG. Round-1 F2 records one permanent `NT_APPLIED_INTENT` marker node per
// applied intent (`aintent_{intent_id}`) and NEVER prunes it. On a durable,
// long-lived daemon the cognitive store therefore grows without bound — one
// marker for every write ever applied — and `intent_applied` lookups degrade
// over time. A durable store cannot ship an unbounded, monotonically-growing
// side table.
//
// THE FIX (design R2/N2). A durable low-watermark already exists: the applied
// index. Every marker for an intent at/below the durably-advanced index is
// provably safe to drop — a replay from an index that already covers it will
// never re-read it. So after `persist_applied_index` succeeds (STRICTLY after —
// never before the cursor is durable), the drain prunes `NT_APPLIED_INTENT`
// markers strictly below the watermark, in BOTH `Applier::drain` and
// `Coordinator::drain_once`. Pruning is delete-after-persist, hence crash-safe:
// a crash mid-prune just leaves extra markers, never fewer than safe, and never
// drops a marker whose effect is not yet durably past the index.
//
// OBSERVABILITY. The implementer exposes `Applier::applied_intent_count() -> usize`
// (count of live `NT_APPLIED_INTENT` markers). Without the fix this equals the
// total number of intents ever applied; with the fix it is bounded (below-
// watermark markers are gone) regardless of how many batches have been drained.
//
// This test drains many batches, asserts the ledger is BOUNDED (not total), and
// asserts exactly-once still holds across a restart AFTER pruning — a
// pruned-but-durable intent is not re-applied because the applied-index already
// covers it. Without the prune the count grows with the batch count (FAILS the
// bound); with it, it stays small.
//
// Requires `persistent` (apply goes through `open_persistent`).
#![cfg(all(feature = "coord", feature = "persistent"))]

use amplihack_memory::coord::{Applier, CoordConfig, Lease, WriteIntent, WriterClient};
use amplihack_memory::{CognitiveMemory, RecallOptions};
use uuid::Uuid;

/// Batches drained, and intents per batch. Total applied = BATCHES * PER_BATCH.
/// BATCHES is large relative to PER_BATCH so an unbounded ledger (== total) is
/// unmistakably distinct from a bounded one (<= one batch).
const BATCHES: usize = 8;
const PER_BATCH: usize = 5;

fn provision(cfg: &CoordConfig) {
    let lease = Lease::acquire(cfg, "provision").expect("provision coord dir");
    lease.release().expect("release provision lease");
}

fn fact(content: &str) -> WriteIntent {
    WriteIntent::StoreFact {
        intent_id: Uuid::new_v4(),
        agent_name: "daemon".into(),
        concept: "ledger-prune".into(),
        content: content.into(),
        confidence: 0.9,
        source_id: "daemon".into(),
        tags: None,
        metadata: None,
    }
}

fn fact_count(store: &std::path::Path, needle: &str) -> usize {
    let mut mem = CognitiveMemory::open_persistent(store, "daemon").expect("reopen store");
    mem.recall_facts_ranked(
        "ledger prune fact record",
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

#[test]
fn ledger_is_bounded_below_the_durable_applied_index() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");
    let cfg = CoordConfig::for_store(&store);
    provision(&cfg);

    let w = WriterClient::connect(&cfg).expect("connect");

    let total = BATCHES * PER_BATCH;

    // Drain many batches through ONE long-lived applier (models a daemon). Each
    // drain advances the durable applied-index and then prunes below it.
    let mut applier = Applier::open(&store, "daemon", &cfg).expect("daemon opens");
    for b in 0..BATCHES {
        for i in 0..PER_BATCH {
            w.append(&fact(&format!("f-b{b}-i{i}")))
                .expect("append acked");
        }
        applier.drain().expect("daemon drains batch");
    }

    // THE BOUND. All markers whose intents sit below the (now end-of-log)
    // durable applied-index must have been pruned. Without the fix the ledger
    // holds one marker per applied intent == `total`; with it, it is bounded by
    // at most a single in-flight batch, independent of `BATCHES`. Read it while
    // the daemon (the exclusive writer) is still alive.
    let ledger = applier.applied_intent_count();
    assert!(
        ledger <= PER_BATCH,
        "the applied-intent ledger must be bounded below the durable applied \
         index (<= {PER_BATCH}), but holds {ledger} of {total} markers — it is \
         growing unbounded (N2)"
    );

    // Release the single-writer lock before reopening the store to verify the
    // effects landed exactly once.
    drop(applier);
    assert_eq!(
        fact_count(&store, "f-b0-i0"),
        1,
        "each fact applies exactly once"
    );
}

#[test]
fn exactly_once_holds_across_restart_after_prune() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");
    let cfg = CoordConfig::for_store(&store);
    provision(&cfg);

    let w = WriterClient::connect(&cfg).expect("connect");
    let total = BATCHES * PER_BATCH;
    for b in 0..BATCHES {
        for i in 0..PER_BATCH {
            w.append(&fact(&format!("r-b{b}-i{i}")))
                .expect("append acked");
        }
    }

    // Daemon 1 drains everything: applied-index durable at end-of-log, markers
    // below it pruned.
    {
        let mut a = Applier::open(&store, "daemon", &cfg).expect("daemon 1 opens");
        // Drain in a couple of passes so pruning runs more than once.
        a.drain().expect("daemon 1 drain");
        a.drain().expect("daemon 1 drain (idempotent tail)");
        let ledger = a.applied_intent_count();
        assert!(
            ledger <= PER_BATCH,
            "ledger must be pruned to a bounded size (<= {PER_BATCH}); got {ledger}"
        );
    }

    // Daemon 2 resumes from the durable applied-index (which already covers every
    // pruned intent). Because the cursor covers them, nothing is re-read and the
    // pruned-but-durable intents are NOT re-applied.
    {
        let mut a = Applier::open(&store, "daemon", &cfg).expect("daemon 2 opens");
        let reapplied = a.drain().expect("daemon 2 drain");
        assert_eq!(
            reapplied, 0,
            "after a clean drain + prune, a restarted applier resumes at the \
             durable index and re-applies nothing"
        );
    }

    // Every one of the `total` facts is present EXACTLY once — pruning the ledger
    // did not reintroduce the F2 duplicate window.
    for b in 0..BATCHES {
        for i in 0..PER_BATCH {
            assert_eq!(
                fact_count(&store, &format!("r-b{b}-i{i}")),
                1,
                "fact r-b{b}-i{i} must be applied exactly once across the \
                 restart-after-prune ({total} total)"
            );
        }
    }
}
