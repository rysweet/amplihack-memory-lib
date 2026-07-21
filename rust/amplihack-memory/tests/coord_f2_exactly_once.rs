// rust/amplihack-memory/tests/coord_f2_exactly_once.rs
//
// TDD crash-injection contract for **[F2] cross-restart duplicate apply** in the
// fenced applier (`src/coord/applier.rs`).
//
// The bug: exactly-once is claimed via a durable applied-index plus idempotent
// replay keyed on `intent_id`, but (a) the `seen` set is process-local and empty
// after a restart, (b) `apply_intent` drops the `intent_id` so the store mints a
// FRESH node every call, and (c) `drain` applies a whole batch, `checkpoint`s the
// store, and ONLY THEN persists the applied-index. A crash in that window — the
// store effect is durable but the index never advanced — re-applies every record
// on restart against an empty `seen`, producing DUPLICATE memories and
// double-bumping `temporal_index`. This breaks exactly-once
// (`PrefixConsistency`, `specs/DurableLog.tla`).
//
// The fix: make the store effect and the applied-index advance one durable unit,
// and/or persist an idempotency ledger of applied `intent_id`s INSIDE the store
// so replay is genuinely idempotent across a restart.
//
// This test reproduces the crash window faithfully: after a clean drain (store
// effects durable), it rolls the applied-index cursor back — modelling "the store
// effect landed but the index persist never happened" — and re-drains with a
// fresh applier (empty in-memory `seen`). Without the fix this DUPLICATES every
// record; with the fix the in-store ledger collapses the replay to exactly once.
//
// Requires `persistent` (apply goes through `open_persistent`).
#![cfg(all(feature = "coord", feature = "persistent"))]

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
        agent_name: "daemon".into(),
        concept: "dedup".into(),
        content: content.into(),
        confidence: 0.9,
        source_id: "daemon".into(),
        tags: None,
        metadata: None,
    }
}

fn episode(content: &str) -> WriteIntent {
    WriteIntent::StoreEpisode {
        intent_id: Uuid::new_v4(),
        agent_name: "daemon".into(),
        content: content.into(),
        source_label: "session".into(),
        temporal_index: None,
        metadata: None,
    }
}

fn fact_count(store: &std::path::Path, needle: &str) -> usize {
    let mut mem = CognitiveMemory::open_persistent(store, "daemon").expect("reopen store");
    mem.recall_facts_ranked(
        "dedup fact record",
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

fn episode_count(store: &std::path::Path, needle: &str) -> usize {
    let mem = CognitiveMemory::open_persistent(store, "daemon").expect("reopen store");
    mem.search_episodes(1000)
        .iter()
        .filter(|e| e.content == needle)
        .count()
}

/// Roll the durable applied-index cursor all the way back to the log origin by
/// removing it. This models a crash in the checkpoint -> index-persist window:
/// the store effects are already durable, but the index never advanced.
fn crash_before_index_persist(cfg: &CoordConfig) {
    let idx = cfg.base_dir.join("applied-index");
    // The index MUST exist here (a clean drain wrote it); removing it rewinds the
    // cursor without touching the already-durable store.
    std::fs::remove_file(&idx).expect("rewind applied-index (simulate crash window)");
}

#[test]
fn replay_after_index_rewind_creates_no_duplicate_facts() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");
    let cfg = CoordConfig::for_store(&store);
    provision(&cfg);

    let w = WriterClient::connect(&cfg).expect("connect");
    for i in 0..4 {
        w.append(&fact(&format!("fact-{i}"))).expect("append acked");
    }

    // Daemon 1 drains cleanly: store effects AND index are now durable.
    let applied_1 = {
        let mut a = Applier::open(&store, "daemon", &cfg).expect("daemon 1 opens");
        a.drain().expect("daemon 1 drain")
    };
    assert_eq!(applied_1, 4, "daemon 1 applies all 4 facts");

    // CRASH in the checkpoint -> index-persist window: the effects are durable
    // but the index persist was lost.
    crash_before_index_persist(&cfg);

    // Daemon 2 resumes from the (rewound) index with an empty in-memory `seen`
    // and re-drains the very same records.
    {
        let mut a = Applier::open(&store, "daemon", &cfg).expect("daemon 2 opens");
        a.drain().expect("daemon 2 drain");
    }

    // Each fact must exist EXACTLY ONCE. Without the fix the replay mints fresh
    // duplicate nodes (count == 2).
    for i in 0..4 {
        assert_eq!(
            fact_count(&store, &format!("fact-{i}")),
            1,
            "fact-{i} must be applied exactly once across the crash-window replay"
        );
    }
}

#[test]
fn replay_after_index_rewind_does_not_double_bump_temporal_index() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");
    let cfg = CoordConfig::for_store(&store);
    provision(&cfg);

    let w = WriterClient::connect(&cfg).expect("connect");
    w.append(&episode("ep-0")).expect("append episode acked");

    // Clean drain: episode + its temporal_index are durable.
    {
        let mut a = Applier::open(&store, "daemon", &cfg).expect("daemon 1 opens");
        assert_eq!(a.drain().expect("drain"), 1, "one episode applied");
    }

    crash_before_index_persist(&cfg);

    // Resume and re-drain from the rewound cursor.
    {
        let mut a = Applier::open(&store, "daemon", &cfg).expect("daemon 2 opens");
        a.drain().expect("re-drain");
    }

    // Exactly one episode with our content — a duplicate would also have
    // re-bumped `temporal_index`, corrupting episodic ordering.
    assert_eq!(
        episode_count(&store, "ep-0"),
        1,
        "the episode must not be duplicated (and its temporal_index not double-bumped) on replay"
    );
}
