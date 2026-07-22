// rust/amplihack-memory/tests/f3_durable_append.rs
//
// TDD contract for **F3 — durable append: fsync-on-ack is UNCONDITIONAL**.
//
// Bug (Design C intent log): `SegmentedLog::append` only `fsync`s the segment
// when `CoordConfig::fsync_on_append` is true:
//
//     if self.config.fsync_on_append { file.sync_all()?; }
//
// The `fsync` IS the durability ack (`WriterClient::append` returns a `LogOffset`
// == "this write is durable"). Gating it on a config knob means a caller can set
// `fsync_on_append = false` and silently downgrade the ack to a NON-durable
// promise: a power loss right after the ack loses an "acked" write, violating
// `NoLostAckedWrite` (`specs/DurableLog.tla`). F3 makes the ack `fsync`
// unconditional (and adds a parent-directory `fsync` on segment/lease creation
// so a newly-created segment's directory entry is durable too).
//
// fsync durability is not observable from safe in-process Rust (page-cache reads
// are coherent whether or not fsync ran, and only true power loss loses
// un-fsynced pages). This file therefore verifies F3 two ways:
//
//   1. `append_fsyncs_on_ack_even_when_fsync_flag_disabled` — a SYSCALL-level
//      discriminator: it runs a child that appends ONE record with
//      `fsync_on_append = false` under `strace` and asserts at least one
//      `fsync`/`fdatasync` syscall occurred. This is deterministic and
//      discriminating on Linux (baseline: zero fsyncs in the append path;
//      post-F3: >= 1). It SKIPS (does not fail) where ptrace is unavailable.
//
//   2. `acked_write_survives_writer_abort_with_fsync_disabled` — an end-to-end
//      behavioural contract: a child appends with the flag OFF then `abort()`s;
//      the write must still be applied. On hosts without filesystem
//      fault-injection this exercises the process-death case (which the page
//      cache already survives); under the CI durability harness
//      (dm-flakey / eatmydata) it becomes the true power-loss regression.
//
// The strace test needs only `coord`; the end-to-end test additionally needs
// `persistent`.
#![cfg(feature = "coord")]

use amplihack_memory::coord::{CoordConfig, WriteIntent, WriterClient};
use uuid::Uuid;

const COORD_DIR_ENV: &str = "AMH_F3_COORD_DIR";
const FSYNC_CHILD_ENV: &str = "AMH_F3_FSYNC_CHILD";

fn flag_off_cfg(coord_dir: &str) -> CoordConfig {
    CoordConfig {
        base_dir: coord_dir.into(),
        // The knob is explicitly OFF — F3's contract is that the ack fsync
        // happens ANYWAY.
        fsync_on_append: false,
        ..CoordConfig::default()
    }
}

fn a_fact(tag: &str) -> WriteIntent {
    WriteIntent::StoreFact {
        intent_id: Uuid::new_v4(),
        agent_name: "engineer".into(),
        concept: "durable-append".into(),
        content: format!("f3-{tag}"),
        confidence: 0.9,
        source_id: "engineer".into(),
        tags: None,
        metadata: None,
    }
}

/// Child entry point for the strace test: connect with `fsync_on_append = false`
/// and append exactly one record, then exit cleanly. No-op under a normal run.
#[test]
fn f3_fsync_child_appends_with_flag_off() {
    let coord_dir = match std::env::var(COORD_DIR_ENV) {
        Ok(v) if std::env::var(FSYNC_CHILD_ENV).is_ok() => v,
        _ => return, // parent/normal run: no-op
    };
    let w = WriterClient::connect(&flag_off_cfg(&coord_dir)).expect("child connect");
    w.append(&a_fact("acked")).expect("child append acked");
    println!("F3_CHILD_ACKED");
}

#[test]
fn append_fsyncs_on_ack_even_when_fsync_flag_disabled() {
    use std::process::Command;

    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");
    let cfg = CoordConfig::for_store(&store);

    // Provision the coord dir in the PARENT (not under strace) so the child only
    // performs connect + append — any fsync the trace captures is the append's.
    {
        let lease =
            amplihack_memory::coord::Lease::acquire(&cfg, "provision").expect("provision coord");
        lease.release().expect("release provision lease");
    }
    let coord_dir = cfg.base_dir.to_string_lossy().to_string();
    let trace_file = tmp.path().join("append.strace");
    let exe = std::env::current_exe().expect("current test exe");

    let run = Command::new("strace")
        .arg("-f")
        .arg("-e")
        .arg("trace=fsync,fdatasync")
        .arg("-o")
        .arg(&trace_file)
        .arg(&exe)
        .arg("f3_fsync_child_appends_with_flag_off")
        .arg("--exact")
        .arg("--nocapture")
        .env(FSYNC_CHILD_ENV, "1")
        .env(COORD_DIR_ENV, &coord_dir)
        .output();

    let output = match run {
        Ok(o) => o,
        Err(e) => {
            eprintln!("SKIP: strace could not be executed ({e}); F3 syscall check skipped");
            return;
        }
    };
    let strace_err = String::from_utf8_lossy(&output.stderr);
    if strace_err.contains("ptrace")
        || strace_err.contains("PTRACE")
        || strace_err
            .to_ascii_lowercase()
            .contains("operation not permitted")
    {
        eprintln!("SKIP: ptrace not permitted in this environment; F3 syscall check skipped");
        return;
    }

    let child_stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        child_stdout.contains("F3_CHILD_ACKED"),
        "child must have appended (acked) before we assert on fsyncs; stdout:\n{child_stdout}\n\
         strace stderr:\n{strace_err}"
    );

    let trace = std::fs::read_to_string(&trace_file).unwrap_or_default();
    let fsyncs = trace
        .lines()
        .filter(|l| l.contains("fsync(") || l.contains("fdatasync("))
        .count();

    assert!(
        fsyncs >= 1,
        "F3: the append ack must fsync even when fsync_on_append == false, but the \
         traced child issued {fsyncs} fsync/fdatasync syscalls.\nTrace:\n{trace}"
    );
}

// ---------------------------------------------------------------------------
// End-to-end durability contract (needs the applier + lbug store).
// ---------------------------------------------------------------------------
#[cfg(feature = "persistent")]
mod end_to_end {
    use super::*;
    use amplihack_memory::coord::Applier;
    use amplihack_memory::{CognitiveMemory, RecallOptions};
    use std::process::Command;

    const ABORT_CHILD_ENV: &str = "AMH_F3_ABORT_CHILD";

    /// Child: append with `fsync_on_append = false`, then die abruptly via
    /// `abort()` (no clean shutdown / Drop). Durability must rest on the ack
    /// fsync alone, not on an orderly exit.
    #[test]
    fn f3_abort_child_appends_flag_off_then_aborts() {
        let coord_dir = match std::env::var(COORD_DIR_ENV) {
            Ok(v) if std::env::var(ABORT_CHILD_ENV).is_ok() => v,
            _ => return,
        };
        let w = WriterClient::connect(&flag_off_cfg(&coord_dir)).expect("child connect");
        w.append(&a_fact("survives-abort"))
            .expect("child append acked");
        use std::io::Write;
        println!("F3_ABORT_CHILD_ACKED");
        std::io::stdout().flush().ok();
        std::process::abort();
    }

    #[test]
    fn acked_write_survives_writer_abort_with_fsync_disabled() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let store = tmp.path().join("store");
        let cfg = CoordConfig::for_store(&store);
        {
            let lease = amplihack_memory::coord::Lease::acquire(&cfg, "provision")
                .expect("provision coord");
            lease.release().expect("release provision lease");
        }
        let coord_dir = cfg.base_dir.to_string_lossy().to_string();
        let exe = std::env::current_exe().expect("current test exe");

        let output = Command::new(&exe)
            .arg("end_to_end::f3_abort_child_appends_flag_off_then_aborts")
            .arg("--exact")
            .arg("--nocapture")
            .env(ABORT_CHILD_ENV, "1")
            .env(COORD_DIR_ENV, &coord_dir)
            .output()
            .expect("spawn abort child");

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("F3_ABORT_CHILD_ACKED"),
            "child must ack before aborting; stdout:\n{stdout}"
        );
        assert!(
            !output.status.success(),
            "child was supposed to abort() after the ack, not exit cleanly"
        );

        // The acked write must be applied despite the writer's abrupt death and
        // the fsync flag being disabled. (On a fault-injection host this is the
        // true post-power-loss regression for F3.)
        let applied = {
            let mut applier = Applier::open(&store, "engineer", &cfg).expect("applier opens");
            applier.drain().expect("drain durable log")
        };
        assert!(
            applied >= 1,
            "the acked write must be applied, applied {applied}"
        );

        let mut mem = CognitiveMemory::open_persistent(&store, "engineer").expect("reopen");
        let hits = mem
            .recall_facts_ranked(
                "durable-append",
                RecallOptions {
                    limit: 100,
                    record_access: false,
                    ..Default::default()
                },
            )
            .expect("recall");
        assert!(
            hits.iter().any(|h| h.item.content == "f3-survives-abort"),
            "the acked (fsync-disabled) write must be durable after the writer's death"
        );
    }
}
