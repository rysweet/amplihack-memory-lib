// rust/amplihack-memory/tests/coord_f3_dir_fsync.rs
//
// TDD durability contract for **[F3] missing directory fsync on new-segment
// creation** in the durable shared intent log (`src/coord/intent_log.rs`).
//
// The bug: `append` `fsync`s the segment FILE but never `fsync`s the containing
// `intent-log/` DIRECTORY. `sync_all` persists the file's data + inode, not the
// parent directory entry that links name -> inode. A power loss after the ack but
// before that dir entry is durable can vanish the whole freshly-created segment
// file — losing an acked write (`NoLostAckedWrite`, `specs/DurableLog.tla`). This
// hits the very first append (creating `000000000000.seg`) and every rollover.
//
// The fix: after creating a NEW segment file, `open(dir)+sync_all()` the parent
// directory before returning the ack (mirroring `persist_applied_index`, which
// already does this correctly).
//
// Because there is no in-tree fault-injection filesystem, this asserts the fix
// STRUCTURALLY at the syscall boundary: a child process performs one append that
// creates the first segment, run under `strace`; the trace MUST show an
// `fsync`/`fdatasync` whose fd resolves to the `intent-log` directory itself.
// Without the fix only the `.seg` file is fsync'd and this assertion fails.
//
// Only `coord` is needed (the child uses `WriterClient`; no store).
#![cfg(feature = "coord")]

use amplihack_memory::coord::{CoordConfig, Lease, WriteIntent, WriterClient};
use std::process::Command;
use uuid::Uuid;

const CHILD_ENV: &str = "AMH_F3_COORD_DIR";

/// Child entry point: under a normal `cargo test` run the env var is unset and
/// this returns immediately. When the parent re-execs this binary under `strace`
/// with the env var set, it connects and performs ONE append — which creates the
/// first segment file and (with the fix) fsyncs the parent directory.
#[test]
fn f3_child_appends_one_record() {
    let coord_dir = match std::env::var(CHILD_ENV) {
        Ok(v) => v,
        Err(_) => return, // parent/normal run: no-op
    };
    let cfg = CoordConfig {
        base_dir: coord_dir.into(),
        ..CoordConfig::default()
    };
    let w = WriterClient::connect(&cfg).expect("child connects to coord dir");
    w.append(&WriteIntent::StoreFact {
        intent_id: Uuid::new_v4(),
        agent_name: "engineer".into(),
        concept: "dir-fsync".into(),
        content: "creates-first-segment".into(),
        confidence: 0.9,
        source_id: "engineer".into(),
        tags: None,
        metadata: None,
    })
    .expect("child append acked");
}

/// `true` if `strace` is usable in this environment (present + can trace a direct
/// child). ptrace_scope=1 permits tracing direct children, which is exactly what
/// we do here.
fn strace_available() -> bool {
    Command::new("strace")
        .arg("-V")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[test]
fn new_segment_creation_fsyncs_the_parent_directory() {
    if !strace_available() {
        eprintln!("skipping F3 strace assertion: strace not available in this environment");
        return;
    }

    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");
    let cfg = CoordConfig::for_store(&store);

    // Provision the coord dir (base + intent-log/) OUTSIDE strace so only the
    // child's segment-creating append shows up in the trace.
    {
        let lease = Lease::acquire(&cfg, "provision").expect("provision coord dir");
        lease.release().expect("release provision lease");
    }
    let coord_dir = cfg.base_dir.clone();
    let intent_log_dir = coord_dir.join("intent-log");
    assert!(
        intent_log_dir.is_dir(),
        "intent-log dir must be provisioned before the child appends"
    );
    // The child's append must CREATE the first segment (not append to an existing
    // one), so the new-segment dir-fsync path is exercised.
    assert!(
        !intent_log_dir.join("000000000000.seg").exists(),
        "no segment must exist yet; the child's append creates it"
    );

    let exe = std::env::current_exe().expect("current test exe");
    // `-f` follow, `-y` resolve fds to their paths so an fsync on the directory
    // shows `fsync(N</.../intent-log>)`, distinct from the segment file
    // `fsync(N</.../intent-log/000000000000.seg>)`.
    let output = Command::new("strace")
        .args(["-f", "-y", "-e", "trace=fsync,fdatasync"])
        .arg(&exe)
        .arg("f3_child_appends_one_record")
        .arg("--exact")
        .env(CHILD_ENV, &coord_dir)
        .output()
        .expect("run child under strace");

    // strace writes the trace to stderr.
    let trace = String::from_utf8_lossy(&output.stderr);

    // Sanity: the child really created + fsync'd the segment file.
    assert!(
        trace.contains("000000000000.seg>"),
        "expected the child to fsync the new segment file; strace trace was:\n{trace}"
    );

    // The F3 assertion: an fsync/fdatasync whose fd resolves to the intent-log
    // DIRECTORY itself (path ends at `intent-log`, not a file within it).
    let dir_fsynced = trace.lines().any(|line| {
        (line.contains("fsync(") || line.contains("fdatasync(")) && line.contains("/intent-log>")
    });
    assert!(
        dir_fsynced,
        "new-segment creation MUST fsync the parent intent-log directory \
         (F3: otherwise a power loss can vanish the acked segment). \
         No dir-fsync found; strace trace was:\n{trace}"
    );
}
