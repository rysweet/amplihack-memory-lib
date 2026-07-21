// rust/amplihack-memory/tests/applier_exactly_once.rs
//
// TDD contract for the **single fenced applier**: it drains the durable log
// strictly IN ORDER and applies each intent to the lbug store EXACTLY ONCE — no
// gaps, no reorder, no duplicates — and that property holds across a daemon
// restart.
//
// Pins `PrefixConsistency` (`specs/DurableLog.tla`): the store is always an
// in-order prefix of the log. Exactly-once is enforced by a durable applied-index
// (checkpoint -> fsync -> atomic-rename) plus idempotent replay keyed by
// `intent_id`.
//
// Requires `persistent` (the applier applies via `open_persistent`).
#![cfg(all(feature = "coord", feature = "persistent"))]

use amplihack_memory::coord::{Applier, CoordConfig, WriteIntent, WriterClient};
use amplihack_memory::{CognitiveMemory, RecallOptions};
use uuid::Uuid;

fn provision(cfg: &CoordConfig) {
    let lease = amplihack_memory::coord::Lease::acquire(cfg, "provision").expect("provision");
    lease.release().expect("release provision lease");
}

fn fact(id: Uuid, n: usize) -> WriteIntent {
    WriteIntent::StoreFact {
        intent_id: id,
        agent_name: "daemon".into(),
        concept: "exactly-once".into(),
        content: format!("record-{n}"),
        confidence: 0.9,
        source_id: "daemon".into(),
        tags: None,
        metadata: None,
    }
}

fn recall_count(store: &std::path::Path, needle: &str) -> usize {
    let mut mem = CognitiveMemory::open_persistent(store, "daemon").expect("reopen store");
    mem.recall_facts_ranked(
        "exactly-once record",
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
fn applier_applies_each_intent_exactly_once_in_order() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");
    let cfg = CoordConfig::for_store(&store);
    provision(&cfg);

    // Append 10 distinct intents.
    let w = WriterClient::connect(&cfg).expect("connect");
    for n in 0..10 {
        w.append(&fact(Uuid::new_v4(), n)).expect("append acked");
    }

    // Drain once.
    let first = {
        let mut a = Applier::open(&store, "daemon", &cfg).expect("applier opens");
        a.drain().expect("first drain")
    };
    assert_eq!(first, 10, "first drain applies all 10 records");

    // A second drain with NO new records must apply nothing (idempotent; the
    // durable applied-index already covers them).
    let second = {
        let mut a = Applier::open(&store, "daemon", &cfg).expect("applier reopens");
        a.drain().expect("second drain")
    };
    assert_eq!(second, 0, "re-draining an unchanged log applies nothing");

    // Every record present exactly once, in order (0..10 all present, none twice).
    for n in 0..10 {
        assert_eq!(
            recall_count(&store, &format!("record-{n}")),
            1,
            "record-{n} once"
        );
    }
}

#[test]
fn re_appended_same_intent_id_is_applied_at_most_once() {
    // At-least-once transport + idempotent apply => a writer that re-sends the
    // SAME intent_id (e.g. crashed uncertain whether the append landed) results in
    // the intent being applied EXACTLY ONCE.
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");
    let cfg = CoordConfig::for_store(&store);
    provision(&cfg);

    let w = WriterClient::connect(&cfg).expect("connect");
    let dup = Uuid::new_v4();
    w.append(&fact(dup, 0)).expect("first append");
    w.append(&fact(dup, 0))
        .expect("duplicate-id append accepted at transport");
    w.append(&fact(Uuid::new_v4(), 1))
        .expect("a distinct intent");

    let applied = {
        let mut a = Applier::open(&store, "daemon", &cfg).expect("applier opens");
        a.drain().expect("drain")
    };
    // Transport saw 3 records; apply deduped the duplicate id => 2 effective.
    assert!(applied >= 2, "at least the 2 distinct intents are applied");
    assert_eq!(
        recall_count(&store, "record-0"),
        1,
        "duplicate intent_id applied once"
    );
    assert_eq!(
        recall_count(&store, "record-1"),
        1,
        "distinct intent applied once"
    );
}

#[test]
fn exactly_once_survives_a_daemon_restart_mid_stream() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");
    let cfg = CoordConfig::for_store(&store);
    provision(&cfg);

    let w = WriterClient::connect(&cfg).expect("connect");
    for n in 0..6 {
        w.append(&fact(Uuid::new_v4(), n)).expect("append batch 1");
    }

    // "Daemon 1" drains the first batch, then goes away (dropped => releases
    // lease + closes store), simulating a restart.
    let applied_1 = {
        let mut a = Applier::open(&store, "daemon", &cfg).expect("daemon 1 opens");
        a.drain().expect("daemon 1 drain")
    };
    assert_eq!(applied_1, 6, "daemon 1 applies the first 6");

    // More writes arrive while the daemon is down.
    for n in 6..12 {
        w.append(&fact(Uuid::new_v4(), n)).expect("append batch 2");
    }

    // "Daemon 2" resumes from the DURABLE applied-index: it must apply ONLY the 6
    // new records, never re-applying the first 6.
    let applied_2 = {
        let mut a = Applier::open(&store, "daemon", &cfg).expect("daemon 2 opens");
        a.drain().expect("daemon 2 drain")
    };
    assert_eq!(
        applied_2, 6,
        "after restart the applier resumes from applied-index and applies only new records"
    );

    // All 12 present exactly once — no gaps, no duplicates across the restart.
    for n in 0..12 {
        assert_eq!(
            recall_count(&store, &format!("record-{n}")),
            1,
            "record-{n} applied exactly once across the restart"
        );
    }
}
