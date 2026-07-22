// rust/amplihack-memory/tests/no_direct_open.rs
//
// TDD contract for the compile-time guarantee that an **engineer / consumer
// build cannot open the store directly**.
//
// Design C's core safety promise (docs/coordination_layer.md, "Cargo features")
// is enforced *by construction*: in a `--features coord` (no `persistent`) build,
// `CognitiveMemory::open_persistent`, the `lbug` backend, and the `Applier` are
// NOT compiled in and cannot be named. An ephemeral engineer therefore cannot
// open `lbug` even by mistake — the symbol does not exist in its build.
//
// This test is the runtime half of that guarantee. It compiles and runs under
// the coord-only feature matrix and exercises the FULL consumer surface
// (WriterClient + WriteIntent + RankedRecallClient types) using ONLY the `coord`
// feature — never referencing `open_persistent`/`lbug`. If someone accidentally
// made the writer/read client depend on the lbug engine, the coord-only build
// would fail to compile and CI would catch it here.
//
// The negative (must-NOT-compile) half — literally naming `open_persistent` in a
// coord-only build — is asserted by the CI feature-matrix build itself
// (`--features coord` with `-D warnings`); a `trybuild` compile-fail case would
// add a dev-dependency and is intentionally avoided.
#![cfg(feature = "coord")]

use amplihack_memory::coord::{CoordConfig, WriteIntent, WriterClient};
use uuid::Uuid;

/// The consumer write path must be fully usable with NO lbug engine linked.
#[test]
fn consumer_can_append_without_the_lbug_engine() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cfg = CoordConfig::for_store(tmp.path());

    // Provision the coord dir through the coord-only lease API.
    let lease = amplihack_memory::coord::Lease::acquire(&cfg, "provision").expect("provision");
    lease.release().expect("release");

    // Append through the writer client — this is the ONLY write path a consumer
    // has, and it links the coordination layer only.
    let w = WriterClient::connect(&cfg).expect("connect");
    let off = w
        .append(&WriteIntent::StoreFact {
            intent_id: Uuid::new_v4(),
            agent_name: "engineer".into(),
            concept: "no-direct-open".into(),
            content: "written via the log, never via open_persistent".into(),
            confidence: 0.9,
            source_id: "engineer".into(),
            tags: None,
            metadata: None,
        })
        .expect("append acked");
    // The offset is Serialize (opaque, ordered) — usable without any lbug type.
    let _ = serde_json::to_string(&off).expect("LogOffset serializes");
}

/// Under a coord-only build (no `persistent`), the store-owning symbols must be
/// absent. We assert the build's own feature flags to document the matrix this
/// test is meant to run in; the true enforcement is that this file compiles at
/// all while referencing only `coord` types.
#[test]
#[cfg(not(feature = "persistent"))]
#[allow(clippy::assertions_on_constants)] // intentional: documents the required feature matrix
fn coord_only_build_has_no_persistent_feature() {
    assert!(
        !cfg!(feature = "persistent"),
        "this test must run in the coord-only matrix (--features coord, NO persistent), \
         proving the consumer surface never pulls in the lbug engine"
    );
    // The `Applier` and `open_persistent` are `#[cfg(feature = \"persistent\")]`
    // and are therefore un-nameable here. Merely compiling this file — whose only
    // amplihack_memory imports are `coord::{CoordConfig, WriteIntent,
    // WriterClient}` and `AccessKind`-free consumer types — is the assertion.
}

/// Sanity: the consumer types we rely on are re-exported and nameable from the
/// crate's `coord` module without touching the persistent surface.
#[test]
fn consumer_surface_is_reachable() {
    #[allow(dead_code)]
    fn _assert_types(_c: &CoordConfig, _w: &WriterClient, _i: &WriteIntent) {}
    // Read-client type must also be reachable in a consumer build (feature `ipc`
    // is what wires its transport; the type name lives under `coord`).
    #[cfg(feature = "ipc")]
    #[allow(dead_code)]
    fn _assert_read_client(_r: &amplihack_memory::coord::RankedRecallClient) {}
}
