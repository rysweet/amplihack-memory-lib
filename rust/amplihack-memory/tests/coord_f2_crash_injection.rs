// rust/amplihack-memory/tests/coord_f2_crash_injection.rs
//
// TDD crash-injection contract for **[F2-residual, CRITICAL] duplicate-on-crash
// across the effect<->marker window** in the fenced applier
// (`src/coord/applier.rs`).
//
// THE RESIDUAL BUG (beyond round-1 F2). Round-1 made exactly-once rest on a
// durable in-store idempotency ledger (`aintent_{intent_id}` marker) consulted
// before each apply. But the effect and its marker are still TWO separate,
// non-atomic store writes, and the effect itself is non-idempotent
// (`store_fact`/`store_episode` mint a FRESH `new_id(...)` per call, dropping
// `intent_id`). The applier's store keeps the default auto-checkpoint every 128
// writes, which fires MID-DRAIN. So a durability boundary (auto-checkpoint, or a
// SIGKILL) can land BETWEEN an effect's last write and its marker write:
//
//     apply_intent(&mut mem, &intent)?;   // effect  -> durable at checkpoint@128
//     <<< crash / auto-checkpoint boundary HERE — marker NOT yet written >>>
//     mem.mark_intent_applied(id)?;       // marker  -> lost
//     ...            persist_applied_index -> never ran
//
// On restart the marker is absent AND the applied-index never advanced, so the
// ledger cannot collapse the replay: the record is re-read and the
// non-idempotent effect mints a DUPLICATE node (and re-bumps `temporal_index`).
// The round-1 `coord_f2_exactly_once` test only rewinds the index AFTER a fully
// clean drain (marker present), so it never exercises this window. This breaks
// exactly-once / `PrefixConsistency` (`specs/DurableLog.tla`).
//
// THE FIX (design D1/R1): make the effect REPLAY-SAFE by deriving the effect
// node id DETERMINISTICALLY from `intent_id` and relying on the store's existing
// same-PK upsert, so replay UPSERTs the same node instead of minting a
// duplicate, and the episode `temporal_index` is bumped create-only
// (get-before-put). The effect<->marker window then becomes harmless. Applied to
// BOTH `Applier::drain` and `Coordinator::drain_once`.
//
// HOW THIS TEST REPRODUCES THE WINDOW. The applier exposes an env-gated
// crash-injection seam the implementer MUST add:
//
//     AMPLIHACK_COORD_CRASH_AFTER_EFFECTS=<N>
//
// When set, the drain path — after applying the N-th effect of the pass AND
// `checkpoint()`-ing the store (effect now durable) — returns
// `Err(MemoryError::Storage("injected crash after effect (test seam)"))` WITHOUT
// writing that intent's ledger marker and WITHOUT persisting the applied-index.
// This is a faithful in-process model of the auto-checkpoint@128 / SIGKILL
// landing in the effect->marker gap.
//
// The test asserts (a) the first drain actually returned that injected error
// (guards that the seam is wired at all), (b) the effect is durable exactly once
// after the "crash", then (c) re-drains with a FRESH applier (empty in-memory
// `seen`, marker absent, index un-advanced) and asserts the memory is applied
// EXACTLY ONCE — no duplicate node, no double `temporal_index` bump. Without the
// deterministic-id fix the replay duplicates (count == 2); with it, exactly once.
//
// Requires `persistent` (apply goes through `open_persistent`).
#![cfg(all(feature = "coord", feature = "persistent"))]

use amplihack_memory::coord::{Applier, CoordConfig, Lease, WriteIntent, WriterClient};
use amplihack_memory::{CognitiveMemory, RecallOptions};
use uuid::Uuid;

/// Env seam name the applier must honor in BOTH drain paths (see file header).
const CRASH_AFTER_EFFECTS: &str = "AMPLIHACK_COORD_CRASH_AFTER_EFFECTS";

fn provision(cfg: &CoordConfig) {
    let lease = Lease::acquire(cfg, "provision").expect("provision coord dir");
    lease.release().expect("release provision lease");
}

fn fact(content: &str) -> WriteIntent {
    WriteIntent::StoreFact {
        intent_id: Uuid::new_v4(),
        agent_name: "daemon".into(),
        concept: "crash-window".into(),
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
        "crash window fact record",
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

fn episodes_with(store: &std::path::Path, needle: &str) -> Vec<i64> {
    let mem = CognitiveMemory::open_persistent(store, "daemon").expect("reopen store");
    mem.search_episodes(1000)
        .into_iter()
        .filter(|e| e.content == needle)
        .map(|e| e.temporal_index)
        .collect()
}

/// Drive one drain with the crash seam armed to fire after the first effect.
/// Must return the injected error (proving the seam is present and fired in the
/// effect->marker window). The `Applier` is dropped on return, releasing the
/// lease so a fresh applier can resume.
fn drain_crashing_after_first_effect(store: &std::path::Path, cfg: &CoordConfig) {
    std::env::set_var(CRASH_AFTER_EFFECTS, "1");
    let result = {
        let mut a = Applier::open(store, "daemon", cfg).expect("crashing daemon opens");
        a.drain()
    };
    std::env::remove_var(CRASH_AFTER_EFFECTS);
    assert!(
        result.is_err(),
        "the crash seam ({CRASH_AFTER_EFFECTS}=1) must fire in the effect->marker \
         window and surface as an error (got {result:?}); a clean Ok means the \
         seam is not wired and the residual F2 window is untested"
    );
}

#[test]
fn crash_between_effect_and_marker_applies_fact_exactly_once() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");
    let cfg = CoordConfig::for_store(&store);
    provision(&cfg);

    // One acked fact on the durable log.
    let w = WriterClient::connect(&cfg).expect("connect");
    w.append(&fact("crash-fact")).expect("append acked");

    // Daemon 1: applies the effect, the store checkpoints (effect durable), then
    // "crashes" BEFORE the marker and BEFORE the applied-index advances.
    drain_crashing_after_first_effect(&store, &cfg);

    // The effect is durable exactly once at this point (the checkpoint persisted
    // it). If this is already 2 something is very wrong; if 0 the seam failed to
    // checkpoint the effect before injecting.
    assert_eq!(
        fact_count(&store, "crash-fact"),
        1,
        "the effect must be durable exactly once after the injected crash"
    );

    // Daemon 2: fresh process — empty in-memory `seen`, the marker was never
    // written, and the applied-index never advanced, so it RE-READS the same
    // record and re-applies its effect. This is the residual window.
    {
        let mut a = Applier::open(&store, "daemon", &cfg).expect("resuming daemon opens");
        a.drain().expect("resuming daemon drains cleanly");
    }

    // Exactly-once across the crash-window replay. Without the deterministic-id
    // fix the replay mints a fresh duplicate node (count == 2).
    assert_eq!(
        fact_count(&store, "crash-fact"),
        1,
        "the fact must be applied EXACTLY ONCE across the effect<->marker crash \
         window (deterministic effect id collapses the replay)"
    );
}

#[test]
fn crash_between_effect_and_marker_does_not_duplicate_or_double_bump_episode() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");
    let cfg = CoordConfig::for_store(&store);
    provision(&cfg);

    let w = WriterClient::connect(&cfg).expect("connect");
    w.append(&episode("crash-ep"))
        .expect("append episode acked");

    // Daemon 1 crashes in the effect->marker window (episode durable, temporal
    // index bumped ONCE, marker + index not persisted).
    drain_crashing_after_first_effect(&store, &cfg);

    let after_crash = episodes_with(&store, "crash-ep");
    assert_eq!(
        after_crash.len(),
        1,
        "the episode must be durable exactly once after the injected crash"
    );
    let temporal_after_crash = after_crash[0];

    // Daemon 2 resumes and re-reads the same record.
    {
        let mut a = Applier::open(&store, "daemon", &cfg).expect("resuming daemon opens");
        a.drain().expect("resuming daemon drains cleanly");
    }

    // Exactly one episode with our content, and its temporal_index is UNCHANGED
    // — a duplicate would add a second node and a create-time re-bump would move
    // the index, corrupting episodic ordering.
    let after_replay = episodes_with(&store, "crash-ep");
    assert_eq!(
        after_replay.len(),
        1,
        "the episode must not be duplicated on the crash-window replay"
    );
    assert_eq!(
        after_replay[0], temporal_after_crash,
        "the episode's temporal_index must not be double-bumped on replay \
         (create-only bump / get-before-put)"
    );
}
