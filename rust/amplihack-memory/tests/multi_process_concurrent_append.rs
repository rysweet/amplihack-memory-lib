// rust/amplihack-memory/tests/multi_process_concurrent_append.rs
//
// TDD contract for **multi-process concurrent append safety** of the durable
// shared intent log.
//
// Pins the write-side of `PrefixConsistency`/`NoLostAckedWrite`
// (`specs/DurableLog.tla`): any number of `WriterClient`s in DIFFERENT processes
// may append at once and records NEVER interleave, tear, or get lost. The log is
// a single append-only total order (length-framed + `O_APPEND` + brief `flock` +
// per-record fsync).
//
// Mechanics: the parent spawns W real child processes; each appends M unique
// intents and prints its acked offsets. The parent asserts:
//   * every child acked all M writes (no lost acks under contention), and
//   * all W*M offsets are DISTINCT (no two records were assigned the same slot —
//     i.e. no interleaving/overwrite).
//
// When `persistent` is also enabled, the parent additionally drains the log and
// verifies all W*M distinct contents materialize with no loss, dup, or torn
// frame — an end-to-end corruption check.
#![cfg(feature = "coord")]

use amplihack_memory::coord::{CoordConfig, WriteIntent, WriterClient};
use std::collections::HashSet;
use std::process::Command;
use uuid::Uuid;

const CHILD_ENV: &str = "AMH_CONCURRENT_CHILD";
const WRITER_ID_ENV: &str = "AMH_WRITER_ID";
const COORD_DIR_ENV: &str = "AMH_COORD_DIR";
const N_WRITERS: usize = 4;
const N_PER_WRITER: usize = 25;

/// Child entry point: appends N_PER_WRITER uniquely-tagged facts and prints each
/// acked offset as `OFFSET:<json>`. No-op under a normal test run.
#[test]
fn concurrent_append_child() {
    let writer_id = match std::env::var(WRITER_ID_ENV) {
        Ok(v) => v,
        Err(_) => return,
    };
    if std::env::var(CHILD_ENV).is_err() {
        return;
    }
    let coord_dir = std::env::var(COORD_DIR_ENV).expect("coord dir env");
    let cfg = CoordConfig {
        base_dir: coord_dir.into(),
        ..CoordConfig::default()
    };
    let w = WriterClient::connect(&cfg).expect("child connect");
    for n in 0..N_PER_WRITER {
        let off = w
            .append(&WriteIntent::StoreFact {
                intent_id: Uuid::new_v4(),
                agent_name: format!("writer-{writer_id}"),
                concept: "concurrent".into(),
                // Globally-unique content so the parent can verify no loss/dup.
                content: format!("w{writer_id}-item{n}"),
                confidence: 0.9,
                source_id: format!("writer-{writer_id}"),
                tags: None,
                metadata: None,
            })
            .expect("child append acked under contention");
        let j = serde_json::to_string(&off).expect("offset serializes");
        println!("OFFSET:{j}");
    }
    use std::io::Write;
    std::io::stdout().flush().ok();
}

fn provision(cfg: &CoordConfig) {
    let lease = amplihack_memory::coord::Lease::acquire(cfg, "provision").expect("provision");
    lease.release().expect("release provision lease");
}

#[test]
fn concurrent_multi_process_append_never_interleaves_or_loses() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");
    let cfg = CoordConfig::for_store(&store);
    provision(&cfg);
    let coord_dir = cfg.base_dir.to_string_lossy().to_string();

    let exe = std::env::current_exe().expect("current exe");

    // Launch all writers concurrently (spawn, not wait) so their appends actually
    // contend on the log.
    let mut children = Vec::new();
    for id in 0..N_WRITERS {
        let child = Command::new(&exe)
            .arg("concurrent_append_child")
            .arg("--exact")
            .arg("--nocapture")
            .env(CHILD_ENV, "1")
            .env(WRITER_ID_ENV, id.to_string())
            .env(COORD_DIR_ENV, &coord_dir)
            .stdout(std::process::Stdio::piped())
            .spawn()
            .expect("spawn writer child");
        children.push(child);
    }

    // Collect every acked offset across every process.
    let mut all_offsets: Vec<String> = Vec::new();
    for child in children {
        let out = child.wait_with_output().expect("child completes");
        assert!(
            out.status.success(),
            "each writer child must exit cleanly after acking all writes"
        );
        let stdout = String::from_utf8_lossy(&out.stdout);
        let mut acked = 0;
        for line in stdout.lines() {
            if let Some(j) = line.strip_prefix("OFFSET:") {
                all_offsets.push(j.to_string());
                acked += 1;
            }
        }
        assert_eq!(
            acked, N_PER_WRITER,
            "every writer must ack all {N_PER_WRITER} writes under contention"
        );
    }

    // No two acked records may share an offset — that would be an interleave /
    // overwrite (log-slot collision).
    let total = N_WRITERS * N_PER_WRITER;
    assert_eq!(all_offsets.len(), total, "expected {total} acked offsets");
    let distinct: HashSet<&String> = all_offsets.iter().collect();
    assert_eq!(
        distinct.len(),
        total,
        "all {total} concurrent-append offsets must be DISTINCT (no interleave/overwrite)"
    );

    verify_applied(&store, &cfg);
}

// Under `persistent`, drain the log and prove every unique write materialized
// exactly once — no loss, no duplication, no torn/corrupt frame.
#[cfg(feature = "persistent")]
fn verify_applied(store: &std::path::Path, cfg: &CoordConfig) {
    use amplihack_memory::coord::Applier;
    use amplihack_memory::{CognitiveMemory, RecallOptions};

    let applied = {
        let mut applier = Applier::open(store, "daemon", cfg).expect("applier opens");
        applier.drain().expect("drain concurrent log")
    };
    let total = N_WRITERS * N_PER_WRITER;
    assert!(
        applied >= total,
        "applier must apply all {total} concurrently-appended records, applied {applied}"
    );

    let mut mem = CognitiveMemory::open_persistent(store, "daemon").expect("reopen store");
    let hits = mem
        .recall_facts_ranked(
            "concurrent",
            RecallOptions {
                limit: total * 4,
                record_access: false,
                ..Default::default()
            },
        )
        .expect("recall");
    for wid in 0..N_WRITERS {
        for n in 0..N_PER_WRITER {
            let needle = format!("w{wid}-item{n}");
            let count = hits.iter().filter(|h| h.item.content == needle).count();
            assert_eq!(
                count, 1,
                "content '{needle}' must materialize exactly once (no loss/dup/corruption)"
            );
        }
    }
}

#[cfg(not(feature = "persistent"))]
fn verify_applied(_store: &std::path::Path, _cfg: &CoordConfig) {
    // Without the applier we can only assert the write-side distinctness above;
    // the end-to-end materialization check runs in the `persistent` matrix.
}
