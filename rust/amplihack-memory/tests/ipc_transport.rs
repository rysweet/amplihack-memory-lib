// rust/amplihack-memory/tests/ipc_transport.rs
//
// TDD contract for the Design C **read plane**: a framed-JSON Unix-domain-socket
// transport that serves ranked recall + ping from the single daemon-owned store.
//
// Pins (docs/coordination_layer.md, "Security model" / "Read plane"):
//   * `RankedRecallClient::ping` round-trips a health probe.
//   * `recall_facts_ranked` returns the identical `Scored<SemanticFact>` shape as
//     the in-process API, served out-of-process — reads never open lbug directly.
//   * The read plane is READ-ONLY BY CONTRACT: the server forces
//     `RecallOptions.record_access = false`, so a recall over IPC can NEVER bump
//     usage counters / mutate the store.
//   * Frames are size-capped (`max_frame_bytes`) and `options.limit` is clamped —
//     hostile-input / OOM defence.
//   * Peer authentication via `SO_PEERCRED`: a same-UID peer is accepted (the
//     cross-UID rejection path maps to `SecurityViolation`, exercised where the
//     harness can drop privileges; here we assert same-UID acceptance).
//
// Server needs `ipc` + `persistent`; the client needs `ipc`. This file therefore
// requires the full daemon feature set.
#![cfg(all(feature = "coord", feature = "ipc", feature = "persistent"))]

use amplihack_memory::coord::{CoordConfig, IpcServer, RankedRecallClient};
use amplihack_memory::{CognitiveMemory, RecallOptions};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Seed a store with a couple of facts and return its path + coord config.
fn seed_store() -> (tempfile::TempDir, std::path::PathBuf, CoordConfig) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");
    let cfg = CoordConfig::for_store(&store);
    {
        let mut mem = CognitiveMemory::open_persistent(&store, "daemon").expect("open store");
        mem.store_fact(
            "release",
            "cargo build --release is warning-clean on 1.85",
            0.95,
            "seed",
            Some(&["build".to_string(), "toolchain".to_string()]),
            None,
        )
        .expect("seed fact 1");
        mem.store_fact(
            "recall",
            "ranked recall scores by keyword, recency, and usage",
            0.9,
            "seed",
            Some(&["recall".to_string()]),
            None,
        )
        .expect("seed fact 2");
    }
    (tmp, store, cfg)
}

/// Spin up the IPC read server on a background thread and return a stop-handle
/// plus the join handle. The server owns the store for the duration.
fn spawn_server(
    store: std::path::PathBuf,
    cfg: CoordConfig,
) -> (Arc<AtomicBool>, thread::JoinHandle<()>) {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_thread = stop.clone();
    let handle = thread::spawn(move || {
        let mut mem =
            CognitiveMemory::open_persistent(&store, "daemon").expect("server opens store");
        let server = IpcServer::bind(&cfg).expect("bind read.sock");
        server
            .serve(&mut mem, move || stop_thread.load(Ordering::Relaxed))
            .expect("serve reads");
    });
    (stop, handle)
}

/// Wait until the socket is connectable (server bound) or time out.
fn wait_for_server(cfg: &CoordConfig) -> RankedRecallClient {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match RankedRecallClient::connect(cfg) {
            Ok(c) => return c,
            Err(_) if Instant::now() < deadline => thread::sleep(Duration::from_millis(25)),
            Err(e) => panic!("server never came up: {e:?}"),
        }
    }
}

#[test]
fn ping_round_trips() {
    let (_tmp, store, cfg) = seed_store();
    let (stop, handle) = spawn_server(store, cfg.clone());

    let client = wait_for_server(&cfg);
    client
        .ping()
        .expect("ping must round-trip against a live server");

    stop.store(true, Ordering::Relaxed);
    // Nudge the accept loop with one more connection so it observes shutdown.
    let _ = RankedRecallClient::connect(&cfg).and_then(|c| c.ping());
    handle.join().expect("server thread joins");
}

#[test]
fn ranked_recall_over_ipc_returns_scored_facts() {
    let (_tmp, store, cfg) = seed_store();
    let (stop, handle) = spawn_server(store, cfg.clone());

    let client = wait_for_server(&cfg);
    let hits = client
        .recall_facts_ranked(
            "release build toolchain",
            RecallOptions {
                limit: 10,
                ..Default::default()
            },
        )
        .expect("recall over IPC");

    assert!(
        !hits.is_empty(),
        "recall must return the seeded matching fact"
    );
    assert!(
        hits.iter()
            .any(|h| h.item.content.contains("warning-clean")),
        "the release-build fact must rank into the results"
    );
    // Same shape as the in-process API: scored + non-empty reasons.
    for h in &hits {
        assert!(h.score >= 0.0);
        assert!(
            !h.reasons.is_empty(),
            "each Scored result carries >=1 reason"
        );
    }

    stop.store(true, Ordering::Relaxed);
    let _ = RankedRecallClient::connect(&cfg).and_then(|c| c.ping());
    handle.join().expect("server joins");
}

#[test]
fn read_plane_never_mutates_the_store_record_access_forced_off() {
    // Even if the client asks for record_access = true, the server must force it
    // OFF: a read must NEVER bump usage_count / last_accessed_at. We prove this by
    // recalling many times over IPC, then reopening the store and confirming the
    // ranking has not been perturbed by usage boosts (a pure read leaves usage at
    // its seeded baseline).
    let (_tmp, store, cfg) = seed_store();

    // Baseline ranking captured with an explicit non-mutating in-process read.
    let baseline_top = {
        let mut mem = CognitiveMemory::open_persistent(&store, "daemon").expect("open");
        let hits = mem
            .recall_facts_ranked(
                "recall",
                RecallOptions {
                    limit: 5,
                    record_access: false,
                    ..Default::default()
                },
            )
            .expect("baseline recall");
        hits.into_iter().map(|h| h.item.content).collect::<Vec<_>>()
    };

    let (stop, handle) = spawn_server(store.clone(), cfg.clone());
    let client = wait_for_server(&cfg);

    // Hammer the read path asking (in vain) to record access.
    for _ in 0..20 {
        let _ = client
            .recall_facts_ranked(
                "recall",
                RecallOptions {
                    limit: 5,
                    record_access: true,
                    ..Default::default()
                },
            )
            .expect("recall over IPC");
    }

    stop.store(true, Ordering::Relaxed);
    let _ = RankedRecallClient::connect(&cfg).and_then(|c| c.ping());
    handle.join().expect("server joins");

    // Reopen and re-rank with a pure read: order must be unchanged, proving the
    // 20 IPC reads did NOT mutate usage counters.
    let after_top = {
        let mut mem = CognitiveMemory::open_persistent(&store, "daemon").expect("reopen");
        let hits = mem
            .recall_facts_ranked(
                "recall",
                RecallOptions {
                    limit: 5,
                    record_access: false,
                    ..Default::default()
                },
            )
            .expect("after recall");
        hits.into_iter().map(|h| h.item.content).collect::<Vec<_>>()
    };
    assert_eq!(
        baseline_top, after_top,
        "IPC reads must not perturb ranking — record_access must be forced off server-side"
    );
}

#[test]
fn oversized_limit_is_clamped_not_honored_verbatim() {
    // A client asking for an absurd limit must not OOM the server; the server
    // clamps `options.limit`. We only have 2 seeded facts, so the practical
    // assertion is that the call SUCCEEDS and returns a bounded result rather than
    // erroring or hanging on an unbounded allocation.
    let (_tmp, store, cfg) = seed_store();
    let (stop, handle) = spawn_server(store, cfg.clone());
    let client = wait_for_server(&cfg);

    let hits = client
        .recall_facts_ranked(
            "recall build",
            RecallOptions {
                limit: usize::MAX,
                ..Default::default()
            },
        )
        .expect("an oversized limit must be clamped and still succeed");
    assert!(
        hits.len() <= 2,
        "cannot return more than the 2 seeded facts"
    );

    stop.store(true, Ordering::Relaxed);
    let _ = RankedRecallClient::connect(&cfg).and_then(|c| c.ping());
    handle.join().expect("server joins");
}

#[test]
fn socket_is_created_private_0600() {
    use std::os::unix::fs::PermissionsExt;
    let (_tmp, store, cfg) = seed_store();
    let (stop, handle) = spawn_server(store, cfg.clone());
    let _client = wait_for_server(&cfg);

    let sock = cfg.base_dir.join("read.sock");
    let mode = std::fs::metadata(&sock)
        .expect("socket exists")
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(
        mode, 0o600,
        "read.sock must be created 0o600 (got {mode:o})"
    );

    stop.store(true, Ordering::Relaxed);
    let _ = RankedRecallClient::connect(&cfg).and_then(|c| c.ping());
    handle.join().expect("server joins");
}
