// rust/amplihack-memory/tests/writer_client.rs
//
// TDD contract for the Design C **writer client** (`amplihack_memory::coord::WriterClient`).
//
// Pins these guarantees from `docs/coordination_layer.md` / the TLA+ specs:
//   * `append` / `append_new` return a `LogOffset` **only after** the record is
//     fsync'd — the durability ack (`NoLostAckedWrite`, DurableLog.tla).
//   * `append_new` assigns a fresh v4 `intent_id` and returns it for correlation.
//   * Offsets are totally ordered and strictly monotonic within a process.
//   * `connect` **fails closed** when the coord dir is missing (no per-agent
//     fallback store — that would be the rejected design B).
//   * Re-appending the *same* `intent_id` is safe (at-least-once transport;
//     exactly-once at apply is the applier's job — see applier_exactly_once.rs).
//
// Feature: `coord` (the consumer surface — NO lbug engine linked here).
//
// This file is written BEFORE the implementation exists; under
// `cargo test --features coord` it MUST fail to compile / fail its assertions
// until `coord` lands. That is the intended red state of TDD.
#![cfg(feature = "coord")]

use amplihack_memory::coord::{CoordConfig, LogOffset, WriteIntent, WriterClient};
use std::collections::HashSet;
use std::path::PathBuf;
use uuid::Uuid;

/// Create a fresh, isolated coord directory and return (tempdir, config).
///
/// Bootstraps the on-disk coordination layout through the public `coord` API
/// (acquiring then releasing a lease is the daemon-side provisioning step that
/// creates the `0o700` coord dir race-free). Writer-only tests reuse the dir the
/// daemon would have created in production.
fn bootstrap() -> (tempfile::TempDir, CoordConfig) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cfg = CoordConfig::for_store(tmp.path());
    // Provision the coord dir via the lease API (creates <store>/coord/...).
    let lease = amplihack_memory::coord::Lease::acquire(&cfg, "bootstrap")
        .expect("bootstrap lease acquire creates the coord dir");
    lease.release().expect("release bootstrap lease");
    (tmp, cfg)
}

fn sample_fact(agent: &str, n: usize) -> WriteIntent {
    WriteIntent::StoreFact {
        intent_id: Uuid::new_v4(),
        agent_name: agent.into(),
        concept: format!("concept-{n}"),
        content: format!("content number {n}"),
        confidence: 0.9,
        source_id: agent.into(),
        tags: Some(vec!["build".into()]),
        metadata: None,
    }
}

#[test]
fn append_returns_a_durable_offset() {
    let (_tmp, cfg) = bootstrap();
    let w = WriterClient::connect(&cfg).expect("connect to existing coord dir");

    // A successful append is the durability ack: it only returns after fsync.
    let off: LogOffset = w
        .append(&sample_fact("engineer-1", 0))
        .expect("append acked");

    // A second append must land at a strictly greater offset (total order).
    let off2 = w
        .append(&sample_fact("engineer-1", 1))
        .expect("second append acked");
    assert!(
        off2 > off,
        "offsets must be strictly monotonic: {off2:?} > {off:?}"
    );
}

#[test]
fn append_new_assigns_and_returns_a_fresh_intent_id() {
    let (_tmp, cfg) = bootstrap();
    let w = WriterClient::connect(&cfg).expect("connect");

    let intent = WriteIntent::StoreProcedure {
        intent_id: Uuid::nil(), // append_new must override this with a fresh v4
        agent_name: "engineer-7".into(),
        name: "release-build".into(),
        steps: vec!["cargo fmt".into(), "cargo clippy".into()],
        prerequisites: None,
    };

    let (off, id) = w.append_new(intent).expect("append_new acked");
    assert_ne!(
        id,
        Uuid::nil(),
        "append_new must assign a fresh, non-nil intent_id"
    );
    assert_eq!(id.get_version_num(), 4, "intent_id must be a v4 UUID");

    // The returned id must be usable to correlate; a later append is ordered after.
    let (off2, id2) = w
        .append_new(sample_fact("engineer-7", 99))
        .expect("second append_new acked");
    assert!(off2 > off, "monotonic offsets across append_new");
    assert_ne!(id, id2, "each append_new mints a distinct intent_id");
}

#[test]
fn many_appends_yield_strictly_increasing_distinct_offsets() {
    let (_tmp, cfg) = bootstrap();
    let w = WriterClient::connect(&cfg).expect("connect");

    let mut prev: Option<LogOffset> = None;
    let mut seen = HashSet::new();
    for n in 0..64 {
        let off = w
            .append(&sample_fact("engineer-1", n))
            .expect("append acked");
        if let Some(p) = prev {
            assert!(off > p, "offset {off:?} must exceed previous {p:?}");
        }
        // Serialize to compare distinctness without requiring Hash on LogOffset.
        let key = serde_json::to_string(&off).expect("LogOffset is Serialize");
        assert!(seen.insert(key), "offset {off:?} must be unique");
        prev = Some(off);
    }
    assert_eq!(seen.len(), 64);
}

#[test]
fn connect_fails_closed_when_coord_dir_is_missing() {
    // Point at a coord dir that was never provisioned. There must be NO silent
    // per-agent fallback store; connect must return an error.
    let tmp = tempfile::tempdir().expect("tempdir");
    let missing = tmp.path().join("does-not-exist");
    let cfg = CoordConfig {
        base_dir: PathBuf::from(&missing),
        ..CoordConfig::default()
    };

    let res = WriterClient::connect(&cfg);
    assert!(
        res.is_err(),
        "connect must fail closed on a missing coord dir, got Ok(_) (silent fallback = design B)"
    );
    assert!(
        !missing.exists(),
        "a failed connect must not silently create a fallback store directory"
    );
}

#[test]
fn re_appending_the_same_intent_id_is_accepted_at_transport() {
    // The transport is at-least-once; a writer that crashed uncertain whether its
    // append landed may re-send the SAME intent_id. That must be accepted here
    // (dedup to exactly-once happens at APPLY time, keyed by intent_id — proven
    // in applier_exactly_once.rs). This test only asserts the append side does
    // not reject a duplicate id.
    let (_tmp, cfg) = bootstrap();
    let w = WriterClient::connect(&cfg).expect("connect");

    let id = Uuid::new_v4();
    let make = || WriteIntent::StoreFact {
        intent_id: id,
        agent_name: "engineer-dup".into(),
        concept: "idempotent".into(),
        content: "same intent twice".into(),
        confidence: 0.8,
        source_id: "engineer-dup".into(),
        tags: None,
        metadata: None,
    };

    let off1 = w.append(&make()).expect("first append acked");
    let off2 = w
        .append(&make())
        .expect("duplicate-id append also acked at transport");
    assert!(off2 > off1, "the duplicate lands at its own (later) offset");
}

#[test]
fn write_intent_serialization_is_tagged_and_rejects_unknown_fields() {
    // The wire form must be `#[serde(tag = "kind")]`, snake_case, and
    // deny_unknown_fields — an unknown kind/field must fail closed (never a
    // silent drop, which would break NoLostAckedWrite).
    let intent = WriteIntent::RecordAccess {
        intent_id: Uuid::new_v4(),
        agent_name: "engineer-1".into(),
        node_id: "sem_abc".into(),
        kind: amplihack_memory::AccessKind::Recall,
    };
    let json = serde_json::to_value(&intent).expect("serialize");
    assert_eq!(
        json["kind"], "record_access",
        "must be serde-tagged snake_case"
    );

    // Round-trips.
    let back: WriteIntent = serde_json::from_value(json).expect("round-trip");
    match back {
        WriteIntent::RecordAccess { node_id, .. } => assert_eq!(node_id, "sem_abc"),
        other => panic!("wrong variant round-tripped: {other:?}"),
    }

    // An unknown kind must be rejected, not silently ignored.
    let bogus = serde_json::json!({ "kind": "obliterate_everything", "intent_id": Uuid::new_v4() });
    assert!(
        serde_json::from_value::<WriteIntent>(bogus).is_err(),
        "unknown intent kind must fail closed"
    );

    // An unknown extra field on a known variant must also be rejected.
    let extra = serde_json::json!({
        "kind": "record_access",
        "intent_id": Uuid::new_v4(),
        "agent_name": "e",
        "node_id": "sem_abc",
        "kind_of_access": "Recall",
        "surprise": true
    });
    assert!(
        serde_json::from_value::<WriteIntent>(extra).is_err(),
        "deny_unknown_fields must reject unexpected fields"
    );
}
