// rust/amplihack-memory/tests/f2_exactly_once_ledger.rs
//
// TDD contract for **F2 — exactly-once apply via a store-resident idempotency
// ledger**.
//
// Bug (Design C applier): duplicate suppression is an IN-MEMORY `HashSet<Uuid>`
// (`Applier::seen`). Progress is made durable in two steps — `checkpoint()` (the
// store effect) THEN `persist_applied_index()` (the cursor). There is a crash
// window BETWEEN them: if the daemon dies after the effect is checkpointed but
// before the applied-index advances, the durable cursor still points *before*
// those records. On restart the applier has an EMPTY in-memory `seen` set and
// re-reads the same records from the log — and RE-APPLIES them. For a
// non-idempotent effect (e.g. `store_fact`, which inserts a new node) this
// double-applies, corrupting the store's cognitive memory (duplicate nodes).
//
// F2 fix: co-commit a store-resident applied-intent ledger (keyed on
// `intent_id`) under the SAME checkpoint barrier as the effect, and consult it
// before applying. Recovery then derives idempotency from the store itself, so a
// crash in the checkpoint↔cursor window can never double-apply. The in-memory
// set is demoted to a fast-path filter only.
//
// Deterministic simulation of the crash window (no real crash needed): drain
// fully (effect + ledger become durable), then ROLL THE APPLIED-INDEX BACK to
// the log origin by deleting the durable cursor file — exactly the on-disk state
// a crash-after-checkpoint-before-cursor-advance leaves. A fresh applier (empty
// `seen`) then re-drains. F2 dedups via the store ledger; baseline re-applies.
//
// Pins `PrefixConsistency` exactly-once (`specs/DurableLog.tla`).
// RED on baseline 32de5be: the record double-applies (recall_count == 2).
//
// Requires `persistent` (the applier applies via `open_persistent`).
#![cfg(all(feature = "coord", feature = "persistent"))]

use amplihack_memory::coord::{Applier, CoordConfig, WriteIntent, WriterClient};
use amplihack_memory::{CognitiveMemory, RecallOptions};
use uuid::Uuid;

fn provision(cfg: &CoordConfig) {
    let lease = amplihack_memory::coord::Lease::acquire(cfg, "daemon").expect("provision");
    lease.release().expect("release provision lease");
}

fn fact(id: Uuid, content: &str) -> WriteIntent {
    WriteIntent::StoreFact {
        intent_id: id,
        agent_name: "daemon".into(),
        concept: "exactly-once-ledger".into(),
        content: content.into(),
        confidence: 0.9,
        source_id: "daemon".into(),
        tags: None,
        metadata: None,
    }
}

fn recall_count(store: &std::path::Path, needle: &str) -> usize {
    let mut mem = CognitiveMemory::open_persistent(store, "daemon").expect("reopen store");
    mem.recall_facts_ranked(
        "exactly-once-ledger record",
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

/// Simulate a crash that landed AFTER `checkpoint()` but BEFORE the durable
/// applied-index advanced: the store effects are persisted, but the cursor is
/// (as far as the next daemon can tell) still at the log origin.
fn rewind_applied_index_to_origin(cfg: &CoordConfig) {
    let idx = cfg.base_dir.join("applied-index");
    // Missing cursor => `load_applied_index` resumes from the origin, replaying
    // every record — the precise crash-window state we must be idempotent under.
    let _ = std::fs::remove_file(idx);
}

#[test]
fn crash_between_checkpoint_and_cursor_advance_does_not_double_apply() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");
    let cfg = CoordConfig::for_store(&store);
    provision(&cfg);

    // Append 5 distinct facts.
    let w = WriterClient::connect(&cfg).expect("connect");
    for n in 0..5 {
        w.append(&fact(Uuid::new_v4(), &format!("record-{n}")))
            .expect("append acked");
    }

    // Daemon 1 drains: effects (and, with F2, the applied-intent ledger) become
    // durable via the checkpoint barrier.
    {
        let mut a = Applier::open(&store, "daemon", &cfg).expect("daemon 1 opens");
        assert_eq!(a.drain().expect("daemon 1 drain"), 5, "all 5 applied once");
    }

    // The cursor advance was lost in the crash window; the store effects were not.
    rewind_applied_index_to_origin(&cfg);

    // Daemon 2 resumes with an EMPTY in-memory seen set and a rewound cursor, so
    // it re-reads all 5 records. F2: the store-resident ledger recognises them as
    // already-applied and applies NOTHING new. Baseline: it re-applies all 5.
    let reapplied = {
        let mut a = Applier::open(&store, "daemon", &cfg).expect("daemon 2 opens");
        a.drain().expect("daemon 2 drain")
    };
    assert_eq!(
        reapplied, 0,
        "after a checkpoint↔cursor crash, replay must be a no-op (ledger dedup), \
         but {reapplied} records were re-applied"
    );

    // The decisive corruption check: every fact is present EXACTLY once.
    for n in 0..5 {
        assert_eq!(
            recall_count(&store, &format!("record-{n}")),
            1,
            "record-{n} must be applied exactly once across the crash window, not duplicated"
        );
    }
}

#[test]
fn replayed_intent_id_is_a_no_op_for_a_fresh_applier() {
    // A narrower statement of the same guarantee: idempotency is keyed on
    // `intent_id` and is DURABLE — a brand-new applier (no in-memory history)
    // must still refuse to re-apply an intent whose effect is already in the
    // store.
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");
    let cfg = CoordConfig::for_store(&store);
    provision(&cfg);

    let id = Uuid::new_v4();
    let w = WriterClient::connect(&cfg).expect("connect");
    w.append(&fact(id, "only-record")).expect("append acked");

    // First applier applies it and makes the ledger durable.
    {
        let mut a = Applier::open(&store, "daemon", &cfg).expect("applier 1");
        assert_eq!(a.drain().expect("drain 1"), 1);
    }

    // Same log record replayed to a fresh applier after a lost cursor advance.
    rewind_applied_index_to_origin(&cfg);
    {
        let mut a = Applier::open(&store, "daemon", &cfg).expect("applier 2");
        assert_eq!(
            a.drain().expect("drain 2"),
            0,
            "a durable ledger must make the replayed intent_id a no-op"
        );
    }

    assert_eq!(
        recall_count(&store, "only-record"),
        1,
        "the replayed intent must not create a duplicate node"
    );
}
