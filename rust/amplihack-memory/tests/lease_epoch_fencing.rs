// rust/amplihack-memory/tests/lease_epoch_fencing.rs
//
// TDD contract for the Design C **lease + epoch fencing** primitive
// (`amplihack_memory::coord::{Lease, Epoch}`).
//
// Pins the `NoSplitBrain` invariant from `specs/FencedApplier.tla`:
//   * The store lease carries a MONOTONIC epoch. Acquire/renew bump it.
//   * The epoch is the SOLE ownership signal. Liveness (`kill(pid, 0)`) is never
//     used to decide ownership — there is no PID/liveness API here at all.
//   * A holder whose epoch != the live lease epoch is STALE and must be fenced:
//     the applier rejects a stale-epoch apply with `EpochFenced` (the split-brain
//     repro that TLC proved impossible under `Fencing = TRUE`).
//
// The pure-lease tests are `feature = "coord"`. The end-to-end
// "a stale epoch cannot mutate the store" repro additionally needs the applier,
// so it is gated on `feature = "persistent"` as well.
#![cfg(feature = "coord")]

use amplihack_memory::coord::{CoordConfig, Epoch, Lease};

fn fresh_cfg() -> (tempfile::TempDir, CoordConfig) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cfg = CoordConfig::for_store(tmp.path());
    (tmp, cfg)
}

#[test]
fn acquire_bumps_the_epoch_monotonically() {
    let (_tmp, cfg) = fresh_cfg();

    let l1 = Lease::acquire(&cfg, "holder-a").expect("first acquire");
    let e1 = l1.epoch();
    // Simulate the previous holder going away (or being stolen from): a second
    // acquire MUST mint a strictly greater epoch. Acquire deliberately permits an
    // unsafe steal (mirrors FencedApplier.tla's Acquire); safety comes from the
    // fence at apply time, not from refusing to acquire.
    l1.release().expect("release first");

    let l2 = Lease::acquire(&cfg, "holder-b").expect("second acquire");
    let e2 = l2.epoch();
    assert!(
        e2 > e1,
        "epoch must strictly increase across acquisitions: {e2} > {e1}"
    );
}

#[test]
fn renew_bumps_the_epoch_again() {
    let (_tmp, cfg) = fresh_cfg();
    let mut l = Lease::acquire(&cfg, "holder").expect("acquire");
    let before = l.epoch();
    let renewed: Epoch = l.renew().expect("renew");
    assert!(
        renewed > before,
        "renew must extend fencing by bumping the epoch"
    );
    assert_eq!(l.epoch(), renewed, "the lease reflects its renewed epoch");
}

#[test]
fn current_epoch_reads_the_live_value_without_acquiring() {
    let (_tmp, cfg) = fresh_cfg();
    let l = Lease::acquire(&cfg, "holder").expect("acquire");
    let live = Lease::current_epoch(&cfg).expect("read current epoch");
    assert_eq!(
        live,
        l.epoch(),
        "current_epoch must equal the live holder's epoch"
    );

    // After another acquisition the previous holder's epoch is now STALE.
    let stale = l.epoch();
    l.release().expect("release");
    let l2 = Lease::acquire(&cfg, "holder-2").expect("re-acquire");
    let now_live = Lease::current_epoch(&cfg).expect("read current epoch again");
    assert_eq!(now_live, l2.epoch());
    assert!(
        stale < now_live,
        "the earlier epoch {stale} must be fenced by the newer live epoch {now_live}"
    );
}

#[test]
fn current_epoch_fails_closed_when_no_lease_exists() {
    // No lease has ever been acquired: reading the epoch must fail closed rather
    // than inventing a default that a stale writer could then match.
    let tmp = tempfile::tempdir().expect("tempdir");
    let cfg = CoordConfig::for_store(tmp.path());
    assert!(
        Lease::current_epoch(&cfg).is_err(),
        "reading a non-existent lease must fail closed, not return epoch 0"
    );
}

#[test]
fn a_stale_lease_epoch_never_equals_the_live_epoch() {
    // Core fencing property expressed purely at the lease layer: once the epoch
    // has moved on, an older holder can NEVER observe its own epoch as live —
    // which is exactly what makes the applier's fence reject it.
    let (_tmp, cfg) = fresh_cfg();

    let old = Lease::acquire(&cfg, "p1").expect("p1 acquire");
    let old_epoch = old.epoch();

    // p2 steals/acquires while p1 still "believes" it holds old_epoch.
    let _new = Lease::acquire(&cfg, "p2").expect("p2 acquire");

    let live = Lease::current_epoch(&cfg).expect("live epoch");
    assert_ne!(
        old_epoch, live,
        "the stale holder's epoch must not match the live epoch"
    );
    assert!(old_epoch < live, "and it must be strictly older");
}

// ---------------------------------------------------------------------------
// End-to-end split-brain repro: a stale-epoch applier CANNOT mutate the store.
// This is the concrete `NoSplitBrain` guarantee and needs the applier + lbug,
// so it is additionally gated on `persistent`.
// ---------------------------------------------------------------------------
#[cfg(feature = "persistent")]
mod fenced_apply {
    use super::*;
    use amplihack_memory::coord::{Applier, WriteIntent, WriterClient};
    use amplihack_memory::MemoryError;
    use uuid::Uuid;

    fn fact(agent: &str, tag: &str) -> WriteIntent {
        WriteIntent::StoreFact {
            intent_id: Uuid::new_v4(),
            agent_name: agent.into(),
            concept: "epoch".into(),
            content: format!("write from {tag}"),
            confidence: 0.9,
            source_id: agent.into(),
            tags: None,
            metadata: None,
        }
    }

    #[test]
    fn stale_epoch_apply_is_fenced_split_brain_impossible() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let store = tmp.path().join("store");
        let cfg = CoordConfig::for_store(&store);

        // Applier A opens: acquires the lease at some epoch and opens the single
        // lbug writer.
        let mut applier_a = Applier::open(&store, "daemon-a", &cfg).expect("applier A opens");

        // Meanwhile a *new* owner acquires the lease, bumping the epoch. From this
        // moment applier A is holding a STALE epoch (acquire is monotonic — proven
        // by the pure-lease tests above).
        let _stolen = Lease::acquire(&cfg, "daemon-b").expect("daemon B steals the lease");

        // A writer appends an intent to the durable log.
        let w = WriterClient::connect(&cfg).expect("writer connects");
        w.append(&fact("engineer-1", "post-steal"))
            .expect("append acked");

        // Applier A tries to drain. It MUST fence on the stale epoch and refuse to
        // apply — never corrupting the store by writing under a dead epoch.
        let err = applier_a
            .drain()
            .expect_err("a stale-epoch applier must fail closed, not apply");
        assert!(
            matches!(err, MemoryError::EpochFenced { .. }),
            "stale apply must be rejected with EpochFenced, got {err:?}"
        );
    }
}
