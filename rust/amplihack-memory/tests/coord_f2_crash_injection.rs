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
use amplihack_memory::{CognitiveMemory, DedupMode, FactInput, RecallOptions, StoreFactOptions};
use uuid::Uuid;

/// Env seam name the applier must honor in BOTH drain paths (see file header).
const CRASH_AFTER_EFFECTS: &str = "AMPLIHACK_COORD_CRASH_AFTER_EFFECTS";

/// The crash seam is driven by a PROCESS-GLOBAL env var, so these tests must not
/// run concurrently (one test's armed seam would fire in another's clean drain).
/// Serialize them regardless of the harness thread count. Poison-tolerant: a
/// panicking test still releases a usable guard to the next.
static SEAM_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn seam_guard() -> std::sync::MutexGuard<'static, ()> {
    SEAM_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

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

/// An `UpsertFact` intent with exact-content-hash dedup, so a second one with the
/// same `concept`+`content` HITS the first and drives `reuse_fact` (the
/// non-idempotent `usage_count += 1` read-modify-write).
fn upsert_exact(concept: &str, content: &str) -> WriteIntent {
    let input = FactInput::new(concept, content, 0.9);
    let options = StoreFactOptions {
        dedup: amplihack_memory::DedupOptions {
            mode: DedupMode::ExactContentHash,
            ..Default::default()
        },
        ..Default::default()
    };
    WriteIntent::UpsertFact {
        intent_id: Uuid::new_v4(),
        agent_name: "daemon".into(),
        input,
        options,
    }
}

/// `usage_count` of the (single) live fact whose content matches `needle`.
fn usage_count(store: &std::path::Path, needle: &str) -> i64 {
    let mut mem = CognitiveMemory::open_persistent(store, "daemon").expect("reopen store");
    let hits = mem
        .recall_facts_ranked(
            "reuse fact usage count",
            RecallOptions {
                limit: 1000,
                record_access: false,
                ..Default::default()
            },
        )
        .expect("recall");
    let matching: Vec<_> = hits.iter().filter(|h| h.item.content == needle).collect();
    assert_eq!(
        matching.len(),
        1,
        "expected exactly one live fact with content {needle:?}, found {}",
        matching.len()
    );
    matching[0].item.usage_count
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
    let _seam = seam_guard();
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
    let _seam = seam_guard();
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

// ---------------------------------------------------------------------------
// [F2-residual, MEDIUM] non-idempotent UpsertFact reuse (`dedup::reuse_fact`)
// ---------------------------------------------------------------------------
//
// Round-2 made the CREATE-family effects replay-safe with deterministic ids, and
// `RecordAccess` replay-safe with a `last_access_intent` stamp. But `reuse_fact`
// (reached when an `UpsertFact` dedup-HITS an existing fact) was left as a
// non-idempotent read-modify-write: `usage_count = existing.usage_count + 1`.
// It gets no deterministic id and no intent stamp, so a crash landing in the
// effect->marker window makes the durable-but-un-marked bump replay and
// double-count (5 -> 6 on the first apply, 6 -> 7 on the replay).
//
// NOTE ON THE FIX. `set_checkpoint_interval(0)` on the applier store (applied in
// `src/coord/applier.rs`) does NOT by itself close this: un-checkpointed WAL
// writes are replayed on reopen (the store's own `open_with_recovery_survives_*`
// tests rely on exactly that), so the effect is durable at its own write, not at
// a checkpoint — the effect<->marker window is a per-write boundary. The
// load-bearing fix is per-effect replay-safety: `reuse_fact` now stamps the node
// with `last_reuse_intent = sem_{intent_id}` (the applier's per-intent id) and
// skips the bump when the stamp already matches, mirroring `apply_record_access`.
//
// This test proves that stamp guard is present and NON-VACUOUS: removing the
// `last_reuse_intent` stamp/skip from `reuse_fact` reintroduces the
// double-increment (final `usage_count == base + 2` instead of `base + 1`).
#[test]
fn crash_between_effect_and_marker_bumps_reused_fact_usage_exactly_once() {
    let _seam = seam_guard();
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");
    let cfg = CoordConfig::for_store(&store);
    provision(&cfg);

    let w = WriterClient::connect(&cfg).expect("connect");

    // Seed the fact (first UpsertFact INSERTs — no dedup hit yet). Drain cleanly
    // so it is applied, marked, and the applied-index advances past it.
    w.append(&upsert_exact("reuse", "reuse-me"))
        .expect("append seed");
    {
        let mut a = Applier::open(&store, "daemon", &cfg).expect("seed daemon opens");
        a.drain().expect("seed daemon drains cleanly");
    }
    let base = usage_count(&store, "reuse-me");

    // A second UpsertFact with identical concept+content HITS the seed via
    // ExactContentHash -> `reuse_fact` runs and bumps usage_count.
    w.append(&upsert_exact("reuse", "reuse-me"))
        .expect("append reuse");

    // Daemon crashes in the effect->marker window: the reuse effect is applied +
    // checkpointed (usage_count durably base+1), then the crash fires BEFORE the
    // marker and BEFORE the applied-index advances.
    drain_crashing_after_first_effect(&store, &cfg);

    assert_eq!(
        usage_count(&store, "reuse-me"),
        base + 1,
        "the reuse must bump usage_count exactly once and be durable after the \
         injected crash (effect checkpointed, marker not yet written)"
    );

    // Fresh daemon: marker absent + index un-advanced, so it RE-READS the same
    // UpsertFact and re-applies `reuse_fact`. The `last_reuse_intent` stamp
    // (durable with the prior bump) must collapse the replay to a no-op bump.
    {
        let mut a = Applier::open(&store, "daemon", &cfg).expect("resuming daemon opens");
        a.drain().expect("resuming daemon drains cleanly");
    }

    assert_eq!(
        usage_count(&store, "reuse-me"),
        base + 1,
        "usage_count must advance by EXACTLY 1 across the effect<->marker crash \
         window (the reuse intent-stamp guard collapses the replay). Removing the \
         `last_reuse_intent` stamp reintroduces the double-increment (base + 2)."
    );
}
