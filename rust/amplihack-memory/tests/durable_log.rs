// rust/amplihack-memory/tests/durable_log.rs
//
// TDD contract for the Design C **durable shared intent log** — specifically the
// headline guarantee: a write **survives the writer's death after the ack**.
//
// Pins `NoLostAckedWrite` (`specs/DurableLog.tla`): once `WriterClient::append`
// returns (the record is fsync'd), the submitting process may crash / be
// SIGKILL'd / have its worktree destroyed, and the write is STILL applied by the
// daemon's single fenced applier. Durability is the fsync, not clean shutdown.
//
// Mechanics: a real child process appends N intents and then `abort()`s (SIGABRT
// — no destructors, no buffer flush) immediately after the ack. The parent then
// runs the `Applier`, drains the log, and verifies every acked write landed in
// the store. This is the hermetic `append -> kill -> still applied` repro.
//
// Requires `persistent` (verification goes through the applier + lbug store).
#![cfg(all(feature = "coord", feature = "persistent"))]

use amplihack_memory::coord::{Applier, CoordConfig, WriteIntent, WriterClient};
use amplihack_memory::{CognitiveMemory, RecallOptions};
use std::process::Command;
use uuid::Uuid;

const CHILD_ENV: &str = "AMH_DURABLE_CHILD";
const N_WRITES: usize = 5;

/// The child-process entry point. Under a normal `cargo test` run the env var is
/// unset and this returns immediately (a trivial pass). When the parent re-execs
/// this binary with the env var set, it appends N intents and then ABORTS right
/// after the last ack — proving durability cannot depend on a clean exit.
#[test]
fn durable_writer_child_then_abort() {
    let coord_dir = match std::env::var(CHILD_ENV) {
        Ok(v) => v,
        Err(_) => return, // parent/normal run: no-op
    };
    let cfg = CoordConfig {
        base_dir: coord_dir.into(),
        ..CoordConfig::default()
    };
    let w = WriterClient::connect(&cfg).expect("child connects to coord dir");
    for n in 0..N_WRITES {
        let id = Uuid::new_v4();
        w.append(&WriteIntent::StoreFact {
            intent_id: id,
            agent_name: "ephemeral-engineer".into(),
            concept: "durable".into(),
            content: format!("acked-write-{n}"),
            confidence: 0.9,
            source_id: "ephemeral-engineer".into(),
            tags: None,
            metadata: None,
        })
        .expect("child append acked (fsync'd)");
        // Marker so the parent can confirm the child really appended.
        println!("CHILD_ACKED:{n}");
    }
    // Flush the marker, then die ABRUPTLY. No clean shutdown, no Drop, no flush
    // beyond the fsync that `append` already performed.
    use std::io::Write;
    std::io::stdout().flush().ok();
    std::process::abort();
}

#[test]
fn acked_writes_survive_writer_death_after_ack() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");
    let cfg = CoordConfig::for_store(&store);

    // Provision the coord dir (daemon-side step) so the child can connect.
    {
        let lease = amplihack_memory::coord::Lease::acquire(&cfg, "provision")
            .expect("provision coord dir");
        lease.release().expect("release provision lease");
    }
    let coord_dir = cfg.base_dir.clone();

    // Spawn the child that appends then aborts.
    let exe = std::env::current_exe().expect("current test exe");
    let output = Command::new(&exe)
        .arg("durable_writer_child_then_abort")
        .arg("--exact")
        .arg("--nocapture")
        .env(CHILD_ENV, &coord_dir)
        .output()
        .expect("spawn child writer");

    // The child must have died via signal (abort => not a clean exit), AFTER
    // acking all writes. We tolerate any non-clean termination; what matters is
    // that the acks were emitted before death.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let acked = stdout
        .lines()
        .filter(|l| l.contains("CHILD_ACKED:"))
        .count();
    assert_eq!(
        acked, N_WRITES,
        "child must ack all {N_WRITES} writes before dying; stdout was:\n{stdout}"
    );
    assert!(
        !output.status.success(),
        "child was supposed to abort() after the ack, not exit cleanly"
    );

    // Now the daemon side: open the applier and drain the durable log. Every
    // acked write MUST be applied even though its author is long dead.
    let applied = {
        let mut applier = Applier::open(&store, "daemon", &cfg).expect("applier opens");
        applier.drain().expect("drain the durable log")
    };
    assert!(
        applied >= N_WRITES,
        "applier must apply at least the {N_WRITES} acked writes, applied {applied}"
    );

    // Verify the facts are actually in the store (applier dropped => single
    // writer, safe to reopen for verification).
    let mut mem = CognitiveMemory::open_persistent(&store, "daemon").expect("reopen store");
    let hits = mem
        .recall_facts_ranked(
            "acked-write durable",
            RecallOptions {
                limit: 100,
                record_access: false,
                ..Default::default()
            },
        )
        .expect("recall");
    for n in 0..N_WRITES {
        let needle = format!("acked-write-{n}");
        assert!(
            hits.iter().any(|h| h.item.content.contains(&needle)),
            "acked write '{needle}' must be present in the store after the writer's death"
        );
    }
}
