//! Direct `GraphStore` trait tests for [`LbugGraphStore`].

use std::collections::HashMap;

use super::LbugGraphStore;
use crate::graph::protocol::GraphStore;
use crate::graph::types::Direction;

fn props(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn open_temp() -> (tempfile::TempDir, LbugGraphStore) {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("graph.ladybug");
    let store = LbugGraphStore::open(&path, Some("test-store")).unwrap();
    (tmp, store)
}

#[test]
fn add_and_query_round_trip() {
    let (_tmp, mut store) = open_temp();
    store
        .add_node(
            "Thing",
            props(&[("node_id", "n1"), ("agent_id", "a"), ("name", "hello")]),
            Some("n1"),
        )
        .unwrap();

    let filter = props(&[("agent_id", "a")]);
    let nodes = store.query_nodes("Thing", Some(&filter), 10);
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].node_id, "n1");
    assert_eq!(nodes[0].properties.get("name").unwrap(), "hello");
    // node_id must be present in properties for converter compatibility.
    assert_eq!(nodes[0].properties.get("node_id").unwrap(), "n1");
}

#[test]
fn query_missing_table_is_empty() {
    let (_tmp, store) = open_temp();
    assert!(store.query_nodes("NeverCreated", None, 10).is_empty());
}

#[test]
fn get_update_delete_node() {
    let (_tmp, mut store) = open_temp();
    store
        .add_node(
            "Thing",
            props(&[("node_id", "n1"), ("agent_id", "a"), ("status", "pending")]),
            Some("n1"),
        )
        .unwrap();

    let got = store.get_node("n1").expect("node should exist");
    assert_eq!(got.properties.get("status").unwrap(), "pending");

    assert!(store.update_node("n1", props(&[("status", "done")])));
    let got = store.get_node("n1").unwrap();
    assert_eq!(got.properties.get("status").unwrap(), "done");

    assert!(store.delete_node("n1"));
    assert!(store.get_node("n1").is_none());
}

#[test]
fn escapes_quotes_in_values() {
    let (_tmp, mut store) = open_temp();
    let tricky = "O'Brien \\ \"x\" line\nbreak";
    store
        .add_node(
            "Thing",
            props(&[("node_id", "n1"), ("agent_id", "a"), ("name", tricky)]),
            Some("n1"),
        )
        .unwrap();
    let got = store.get_node("n1").unwrap();
    assert_eq!(got.properties.get("name").unwrap(), tricky);
}

#[test]
fn unbounded_limit_returns_all() {
    let (_tmp, mut store) = open_temp();
    for i in 0..5 {
        store
            .add_node(
                "Thing",
                props(&[("node_id", &format!("n{i}")), ("agent_id", "a")]),
                Some(&format!("n{i}")),
            )
            .unwrap();
    }
    let nodes = store.query_nodes("Thing", None, usize::MAX);
    assert_eq!(nodes.len(), 5);
}

#[test]
fn add_edge_and_query_neighbors() {
    let (_tmp, mut store) = open_temp();
    store
        .add_node(
            "Thing",
            props(&[("node_id", "a1"), ("agent_id", "a")]),
            Some("a1"),
        )
        .unwrap();
    store
        .add_node(
            "Thing",
            props(&[("node_id", "b1"), ("agent_id", "a")]),
            Some("b1"),
        )
        .unwrap();

    store
        .add_edge("a1", "b1", "LINKS", Some(props(&[("weight", "5")])))
        .unwrap();

    let neighbors = store.query_neighbors("a1", Some("LINKS"), Direction::Outgoing, 10);
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0].1.node_id, "b1");
    assert_eq!(neighbors[0].0.properties.get("weight").unwrap(), "5");

    assert!(store.delete_edge("a1", "b1", "LINKS"));
    assert!(store
        .query_neighbors("a1", Some("LINKS"), Direction::Outgoing, 10)
        .is_empty());
}

#[test]
fn data_survives_close_and_reopen() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("graph.ladybug");

    {
        let mut store = LbugGraphStore::open(&path, Some("s")).unwrap();
        store
            .add_node(
                "Thing",
                props(&[("node_id", "n1"), ("agent_id", "a"), ("name", "persisted")]),
                Some("n1"),
            )
            .unwrap();
        store.close();
    } // Database dropped here -> WAL flushed to disk.

    let store = LbugGraphStore::open(&path, Some("s")).unwrap();
    let nodes = store.query_nodes("Thing", None, 10);
    assert_eq!(nodes.len(), 1, "reopened store must see persisted node");
    assert_eq!(nodes[0].properties.get("name").unwrap(), "persisted");

    // get_node after reopen must resolve the table via catalog introspection.
    let got = store.get_node("n1").expect("get_node after reopen");
    assert_eq!(got.properties.get("name").unwrap(), "persisted");
}

#[test]
fn invalid_identifiers_are_rejected() {
    let (_tmp, mut store) = open_temp();
    assert!(store
        .add_node("Bad-Type", props(&[("node_id", "x")]), Some("x"))
        .is_err());
    assert!(store
        .add_node("Thing", props(&[("bad-key", "v")]), Some("x"))
        .is_err());
}

#[test]
fn query_neighbors_all_edge_types_and_directions() {
    let (_tmp, mut store) = open_temp();
    for id in ["a1", "b1", "c1"] {
        store
            .add_node(
                "Thing",
                props(&[("node_id", id), ("agent_id", "a")]),
                Some(id),
            )
            .unwrap();
    }
    // a1 -LINKS-> b1, a1 -REFS-> c1, c1 -LINKS-> a1
    store.add_edge("a1", "b1", "LINKS", None).unwrap();
    store.add_edge("a1", "c1", "REFS", None).unwrap();
    store.add_edge("c1", "a1", "LINKS", None).unwrap();

    // edge_type=None must return both outgoing edge types in one pass.
    let out = store.query_neighbors("a1", None, Direction::Outgoing, 10);
    assert_eq!(out.len(), 2);
    let mut types: Vec<&str> = out.iter().map(|(e, _)| e.edge_type.as_str()).collect();
    types.sort_unstable();
    assert_eq!(types, ["LINKS", "REFS"]);

    // Incoming should see the c1 -LINKS-> a1 edge.
    let inc = store.query_neighbors("a1", None, Direction::Incoming, 10);
    assert_eq!(inc.len(), 1);
    assert_eq!(inc[0].1.node_id, "c1");

    // Both = outgoing + incoming.
    let both = store.query_neighbors("a1", None, Direction::Both, 10);
    assert_eq!(both.len(), 3);

    // Filtering by edge_type narrows results.
    let refs = store.query_neighbors("a1", Some("REFS"), Direction::Outgoing, 10);
    assert_eq!(refs.len(), 1);
    assert_eq!(refs[0].1.node_id, "c1");

    // Unknown edge type yields nothing (no binder error).
    assert!(store
        .query_neighbors("a1", Some("NOPE"), Direction::Both, 10)
        .is_empty());
}

#[test]
fn get_node_resolves_across_types_after_reopen() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("graph.ladybug");
    {
        let mut store = LbugGraphStore::open(&path, Some("s")).unwrap();
        store
            .add_node(
                "Alpha",
                props(&[("node_id", "x1"), ("name", "ax")]),
                Some("x1"),
            )
            .unwrap();
        store
            .add_node(
                "Beta",
                props(&[("node_id", "y1"), ("name", "by")]),
                Some("y1"),
            )
            .unwrap();
        store.close();
    }

    // Fresh handle: id_table_cache is cold, so get_node exercises the
    // single-query label-less resolution across multiple node tables.
    let store = LbugGraphStore::open(&path, Some("s")).unwrap();
    let beta = store.get_node("y1").expect("beta resolves after reopen");
    assert_eq!(beta.node_type, "Beta");
    assert_eq!(beta.properties.get("name").unwrap(), "by");

    let alpha = store.get_node("x1").expect("alpha resolves after reopen");
    assert_eq!(alpha.node_type, "Alpha");

    // A missing id resolves to None without error.
    assert!(store.get_node("missing").is_none());
}

#[test]
fn execute_error_does_not_leak_query_values() {
    // A failing statement must not surface interpolated string values (stored
    // memory content, agent_id) in the propagating error. The engine may echo
    // schema identifiers (validated to [A-Za-z_][A-Za-z0-9_]*), but never the
    // single-quoted values; the full query text is logged at debug level only.
    let (_tmp, store) = open_temp();
    let secret = "s3cr3t-memory-value";

    let cypher = format!("MATCH (n:Missing) WHERE n.node_id = '{secret}' RETURN bogus_fn(n)");
    let err = store
        .execute(&cypher)
        .expect_err("invalid Cypher should error");
    assert!(
        !err.to_string().contains(secret),
        "error must not leak interpolated values: {err}"
    );

    let qerr = store
        .query_rows(&cypher)
        .expect_err("invalid query should error");
    assert!(
        !qerr.to_string().contains(secret),
        "error must not leak interpolated values: {qerr}"
    );
    assert!(qerr.to_string().contains("Cypher query failed"));
}

// ---------------------------------------------------------------------------
// Durability: corrupt-WAL recovery + checkpoint API + auto-checkpoint
// ---------------------------------------------------------------------------

use std::fs;
use std::path::{Path, PathBuf};

use super::{
    effective_limits, resolve_buffer_pool_bytes, resolve_max_db_bytes, wal_path_for,
    WalRecoveryOutcome, DEFAULT_BUFFER_POOL_BYTES, DEFAULT_MAX_DB_BYTES, MIN_BUFFER_POOL_BYTES,
    MIN_MAX_DB_BYTES,
};

/// Insert `n` simple nodes into a fresh "T" table.
fn add_things(store: &mut LbugGraphStore, start: usize, n: usize) {
    for i in start..start + n {
        let id = format!("n{i}");
        store
            .add_node(
                "T",
                props(&[("node_id", &id), ("agent_id", "a")]),
                Some(&id),
            )
            .unwrap();
    }
}

/// Copy every file in `from` into a fresh dir, returning the dir + the copied
/// db path. Mimics snapshotting the on-disk state of a killed process.
fn crash_snapshot(from: &Path, db_name: &str) -> (tempfile::TempDir, PathBuf) {
    let crash = tempfile::tempdir().unwrap();
    for entry in fs::read_dir(from).unwrap() {
        let p = entry.unwrap().path();
        if p.is_file() {
            fs::copy(&p, crash.path().join(p.file_name().unwrap())).unwrap();
        }
    }
    let db = crash.path().join(db_name);
    (crash, db)
}

/// Truncate `wal` mid-record so a strict open fails to replay it.
fn corrupt_wal_tail(wal: &Path) {
    let len = fs::metadata(wal).unwrap().len();
    assert!(len > 64, "WAL should be non-empty to corrupt: {}", len);
    let f = fs::OpenOptions::new().write(true).open(wal).unwrap();
    f.set_len(len - 41).unwrap();
    f.sync_all().unwrap();
}

#[test]
fn checkpoint_makes_clean_reopen_need_no_wal_replay() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("graph.ladybug");
    let wal = wal_path_for(&path);

    {
        let mut store = LbugGraphStore::open(&path, Some("s")).unwrap();
        store.set_checkpoint_interval(0); // keep everything in the WAL until we ask
        add_things(&mut store, 0, 40);
        // The WAL holds the writes; checkpoint folds them into the main DB file.
        store.checkpoint().unwrap();
        // After a checkpoint there is nothing left to replay.
        let wal_len = fs::metadata(&wal).map(|m| m.len()).unwrap_or(0);
        assert!(
            wal_len == 0 || !wal.exists(),
            "WAL should be empty/absent after checkpoint, was {wal_len} bytes"
        );
        assert_eq!(store.pending_writes(), 0, "checkpoint resets write counter");
        // Forget so Drop/close doesn't run another checkpoint — prove the
        // explicit checkpoint alone made the data durable.
        std::mem::forget(store);
    }

    // A strict reopen (the one that crashes on a bad WAL) succeeds and sees all
    // records, because the checkpoint left no WAL to replay.
    let store = LbugGraphStore::open(&path, Some("s")).unwrap();
    assert_eq!(store.count_all_nodes(), 40);
}

#[test]
fn open_with_recovery_survives_corrupt_wal_and_returns_checkpointed_records() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("graph.ladybug");

    {
        let mut store = LbugGraphStore::open(&path, Some("s")).unwrap();
        store.set_checkpoint_interval(0);
        // Durable batch: checkpointed into the main DB file.
        add_things(&mut store, 0, 30);
        store.checkpoint().unwrap();
        // Uncheckpointed batch: lives only in the WAL.
        add_things(&mut store, 30, 30);
        std::mem::forget(store); // unclean: no close, no final checkpoint
    }

    // Snapshot the on-disk files and corrupt the WAL tail of the copy.
    let (crash, crash_db) = crash_snapshot(path.parent().unwrap(), "graph.ladybug");
    let crash_wal = wal_path_for(&crash_db);
    assert!(crash_wal.exists(), "snapshot must include the WAL");
    corrupt_wal_tail(&crash_wal);

    // A strict open of the corrupt copy must fail (this is the incident).
    assert!(
        LbugGraphStore::open(&crash_db, Some("s")).is_err(),
        "strict open of a corrupt WAL should error"
    );

    // Recovery opens successfully (no crash) and reports the outcome.
    let (store, report) =
        LbugGraphStore::open_with_recovery(&crash_db, Some("s")).expect("recovery must open");
    assert_eq!(report.outcome, WalRecoveryOutcome::RecoveredPrefix);
    assert!(report.recovered(), "report should flag that recovery ran");
    assert!(
        report.recovered_records >= 30,
        "at least the 30 checkpointed records must survive, got {}",
        report.recovered_records
    );
    assert_eq!(store.count_all_nodes(), report.recovered_records);

    // The corrupt WAL was moved aside (never deleted).
    let quarantine = report.quarantined_wal.expect("a quarantine path");
    assert!(
        quarantine.exists(),
        "corrupt WAL must be preserved at {quarantine:?}"
    );
    assert!(quarantine
        .file_name()
        .unwrap()
        .to_string_lossy()
        .contains(".corrupt-"));

    drop(store);

    // Recovery checkpointed the survivors, so a subsequent STRICT reopen of the
    // same path now needs no replay and no longer crashes.
    let reopened = LbugGraphStore::open(&crash_db, Some("s")).expect("clean reopen after recovery");
    assert!(reopened.count_all_nodes() >= 30);

    drop(crash);
}

#[test]
fn open_with_recovery_is_a_noop_on_a_clean_store() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("graph.ladybug");
    {
        let mut store = LbugGraphStore::open(&path, Some("s")).unwrap();
        add_things(&mut store, 0, 10);
        store.close();
    }
    let (store, report) = LbugGraphStore::open_with_recovery(&path, Some("s")).unwrap();
    assert_eq!(report.outcome, WalRecoveryOutcome::Clean);
    assert!(!report.recovered());
    assert!(report.quarantined_wal.is_none());
    assert_eq!(store.count_all_nodes(), 10);
}

#[test]
fn auto_checkpoint_bounds_uncheckpointed_writes() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("graph.ladybug");

    let mut store = LbugGraphStore::open(&path, Some("s")).unwrap();
    store.set_checkpoint_interval(4); // checkpoint every 4 writes
    add_things(&mut store, 0, 10); // checkpoints fire after writes 4 and 8
    assert!(
        store.pending_writes() < 4,
        "at most interval-1 writes may be uncheckpointed, was {}",
        store.pending_writes()
    );

    // Snapshot, drop the WAL entirely (simulate total WAL loss), strict-open:
    // only the auto-checkpointed records survive — and there must be some,
    // proving auto-checkpoint persisted data with no explicit checkpoint/close.
    let (crash, crash_db) = crash_snapshot(path.parent().unwrap(), "graph.ladybug");
    std::mem::forget(store);
    let crash_wal = wal_path_for(&crash_db);
    let _ = fs::remove_file(&crash_wal);

    let recovered = LbugGraphStore::open(&crash_db, Some("s")).unwrap();
    let n = recovered.count_all_nodes();
    assert!(
        n >= 8,
        "auto-checkpoint should have persisted >= 8 of 10 records, got {n}"
    );
    assert!(
        n < 10,
        "the last writes should still have been WAL-only, got {n}"
    );
    drop(crash);
}

// ---------------------------------------------------------------------------
// Configurable buffer-pool / max-db-size resolution (pure functions)
// ---------------------------------------------------------------------------

#[test]
fn resolves_larger_defaults_when_unset() {
    assert_eq!(resolve_buffer_pool_bytes(None), DEFAULT_BUFFER_POOL_BYTES);
    assert_eq!(resolve_max_db_bytes(None), DEFAULT_MAX_DB_BYTES);
    // The new defaults must be larger than the old hardcoded 128 MiB / 1 GiB
    // that allowed the buffer pool to exhaust and corrupt the catalog (#95).
    assert_eq!(
        DEFAULT_BUFFER_POOL_BYTES,
        1 << 30,
        "buffer pool default = 1 GiB"
    );
    assert_eq!(DEFAULT_MAX_DB_BYTES, 16u64 << 30, "max db default = 16 GiB");
}

#[test]
fn resolves_valid_env_overrides() {
    // 2 GiB pool / 32 GiB max — both above their minimums, used verbatim.
    assert_eq!(resolve_buffer_pool_bytes(Some("2147483648")), 2_147_483_648);
    assert_eq!(resolve_max_db_bytes(Some("34359738368")), 34_359_738_368);
}

#[test]
fn clamps_overrides_below_minimum() {
    assert_eq!(
        resolve_buffer_pool_bytes(Some("1024")),
        MIN_BUFFER_POOL_BYTES
    );
    assert_eq!(resolve_max_db_bytes(Some("1024")), MIN_MAX_DB_BYTES);
}

#[test]
fn invalid_overrides_fall_back_to_default() {
    for bad in [
        "",
        "   ",
        "abc",
        "-5",
        "0",
        "12.5",
        "1_000",
        "9999999999999999999999",
    ] {
        assert_eq!(
            resolve_buffer_pool_bytes(Some(bad)),
            DEFAULT_BUFFER_POOL_BYTES,
            "buffer pool should default for invalid override {bad:?}"
        );
        assert_eq!(
            resolve_max_db_bytes(Some(bad)),
            DEFAULT_MAX_DB_BYTES,
            "max db should default for invalid override {bad:?}"
        );
    }
}

#[test]
fn effective_limits_keeps_buffer_pool_at_or_below_max_db() {
    // A buffer pool larger than the whole database is clamped down to max_db.
    let (buf, max) = effective_limits(Some("9999999999999"), Some("1073741824"));
    assert_eq!(max, 1 << 30, "1 GiB max honored");
    assert_eq!(buf, 1 << 30, "buffer pool clamped to max_db");
    assert!(buf <= max);

    // Defaults already satisfy the invariant (1 GiB pool <= 16 GiB max).
    let (dbuf, dmax) = effective_limits(None, None);
    assert!(dbuf <= dmax);
    assert_eq!(
        (dbuf, dmax),
        (DEFAULT_BUFFER_POOL_BYTES, DEFAULT_MAX_DB_BYTES)
    );
}

// ---------------------------------------------------------------------------
// Catalog / main-DB corruption recovery (no more crash loop, #95)
// ---------------------------------------------------------------------------

/// Overwrite the main database file with garbage so the catalog/header is
/// unreadable — the on-disk shape of the #95 incident ("table 0 doesn't exist
/// in catalog") where a failed CHECKPOINT corrupted the main file.
fn corrupt_db_file(db: &Path) {
    let len = fs::metadata(db).unwrap().len();
    assert!(len > 0, "db file should be non-empty to corrupt: {len}");
    fs::write(db, vec![0xABu8; len as usize]).unwrap();
    let f = fs::OpenOptions::new().write(true).open(db).unwrap();
    f.sync_all().unwrap();
}

/// Assert a recovery report describes a fresh rebuild: empty, quarantined to a
/// `*.corrupt-*` sibling that still exists on disk, and the returned store is
/// empty and writable.
fn assert_rebuilt(store: &mut LbugGraphStore, report: &super::WalRecovery) {
    assert_eq!(
        report.outcome,
        WalRecoveryOutcome::RebuiltAfterCorruption,
        "a corrupt catalog must trigger a rebuild"
    );
    assert!(report.recovered(), "rebuild counts as recovery");
    assert_eq!(
        report.recovered_records, 0,
        "a fresh database has no records"
    );

    let quarantine = report
        .quarantined_wal
        .as_ref()
        .expect("the corrupt database must be quarantined, never deleted");
    assert!(
        quarantine.exists(),
        "corrupt database must be preserved at {quarantine:?}"
    );
    assert!(
        quarantine
            .file_name()
            .unwrap()
            .to_string_lossy()
            .contains(".corrupt-"),
        "quarantine path must be a *.corrupt-* sibling, was {quarantine:?}"
    );

    assert_eq!(store.count_all_nodes(), 0, "rebuilt store starts empty");

    // The rebuilt store must be writable.
    store
        .add_node(
            "T",
            props(&[("node_id", "fresh1"), ("agent_id", "a")]),
            Some("fresh1"),
        )
        .unwrap();
    assert_eq!(
        store.count_all_nodes(),
        1,
        "rebuilt store must accept writes"
    );
    assert_eq!(store.get_node("fresh1").unwrap().node_id, "fresh1");
}

#[test]
fn open_with_recovery_rebuilds_after_corrupt_catalog_no_wal() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("graph.ladybug");

    {
        let mut store = LbugGraphStore::open(&path, Some("s")).unwrap();
        add_things(&mut store, 0, 20);
        store.checkpoint().unwrap(); // fold everything into the main DB file
        store.close();
    } // drop -> final checkpoint; the WAL is now empty/absent.

    // Snapshot the on-disk files, drop any WAL (this is the no-WAL case), and
    // corrupt the main database file of the copy.
    let (crash, crash_db) = crash_snapshot(path.parent().unwrap(), "graph.ladybug");
    let crash_wal = wal_path_for(&crash_db);
    let _ = fs::remove_file(&crash_wal);
    assert!(!crash_wal.exists(), "no-WAL case: WAL must be absent");
    corrupt_db_file(&crash_db);

    // A strict open of a corrupt catalog must fail (this is the incident).
    assert!(
        LbugGraphStore::open(&crash_db, Some("s")).is_err(),
        "strict open of a corrupt catalog should error"
    );

    // Resilient open self-heals: quarantine + fresh empty DB, no crash loop.
    let (mut store, report) =
        LbugGraphStore::open_with_recovery(&crash_db, Some("s")).expect("recovery must open");
    assert_rebuilt(&mut store, &report);

    drop(store);
    // A subsequent strict reopen now succeeds (fresh DB, nothing to replay).
    let reopened = LbugGraphStore::open(&crash_db, Some("s")).expect("clean reopen after rebuild");
    assert_eq!(reopened.count_all_nodes(), 1);

    drop(reopened);
    drop(crash);
}

#[test]
fn open_with_recovery_rebuilds_after_corrupt_catalog_with_wal_present() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("graph.ladybug");

    {
        let mut store = LbugGraphStore::open(&path, Some("s")).unwrap();
        store.set_checkpoint_interval(0); // keep writes in the WAL until we ask
        add_things(&mut store, 0, 20);
        store.checkpoint().unwrap(); // some records folded into the main DB file
        add_things(&mut store, 20, 20); // these stay in the WAL
        std::mem::forget(store); // unclean: no close, no final checkpoint
    }

    // Snapshot: both a main DB file AND a non-empty WAL are present.
    let (crash, crash_db) = crash_snapshot(path.parent().unwrap(), "graph.ladybug");
    let crash_wal = wal_path_for(&crash_db);
    assert!(
        crash_wal.exists(),
        "WAL-present case: snapshot must include a WAL"
    );
    // Corrupt the main DB file; the WAL stays in place so recovery goes through
    // the WAL path first, then discovers the main file itself is unopenable.
    corrupt_db_file(&crash_db);

    assert!(
        LbugGraphStore::open(&crash_db, Some("s")).is_err(),
        "strict open of a corrupt catalog should error even with a WAL present"
    );

    let (mut store, report) =
        LbugGraphStore::open_with_recovery(&crash_db, Some("s")).expect("recovery must open");
    assert_rebuilt(&mut store, &report);

    drop(store);
    drop(crash);
}

// ---------------------------------------------------------------------------
// Checkpoint-failure signal + effective-config readback (store health, #95)
// ---------------------------------------------------------------------------

#[test]
fn effective_config_getters_report_resolved_limits() {
    let (_tmp, store) = open_temp();

    // The getters must report exactly the limits `from_parts` resolved from the
    // environment (the larger defaults when unset) — i.e. the store's *actual*
    // open-time config — so a consumer can read it back for health/telemetry
    // without re-deriving it or parsing logs.
    let buffer_env = std::env::var(super::ENV_BUFFER_POOL_BYTES).ok();
    let max_db_env = std::env::var(super::ENV_MAX_DB_BYTES).ok();
    let (expected_buf, expected_max) =
        effective_limits(buffer_env.as_deref(), max_db_env.as_deref());

    assert_eq!(
        store.buffer_pool_bytes(),
        expected_buf,
        "buffer_pool_bytes() must report the resolved effective buffer-pool cap"
    );
    assert_eq!(
        store.max_db_bytes(),
        expected_max,
        "max_db_bytes() must report the resolved effective max database size"
    );

    // Invariants that must hold regardless of any environment override.
    assert!(
        store.buffer_pool_bytes() >= MIN_BUFFER_POOL_BYTES,
        "effective buffer pool must never fall below the floor"
    );
    assert!(
        store.max_db_bytes() >= MIN_MAX_DB_BYTES,
        "effective max db size must never fall below the floor"
    );
    assert!(
        store.buffer_pool_bytes() <= store.max_db_bytes(),
        "effective buffer pool must never exceed effective max db size"
    );
}

#[test]
fn last_checkpoint_error_is_none_on_a_healthy_store() {
    let (_tmp, mut store) = open_temp();

    // A brand-new store has never failed a checkpoint.
    assert!(
        store.last_checkpoint_error().is_none(),
        "fresh store must report no checkpoint error"
    );

    // A successful explicit checkpoint must leave/keep store health clear.
    add_things(&mut store, 0, 5);
    store.checkpoint().expect("checkpoint should succeed");
    assert!(
        store.last_checkpoint_error().is_none(),
        "a successful checkpoint must report healthy (no recorded error)"
    );

    // Successful AUTO-checkpoints (the path that records buffer-pool exhaustion
    // on failure) must NOT record a spurious error on a healthy store.
    store.set_checkpoint_interval(4);
    add_things(&mut store, 5, 12); // crosses the interval -> auto-checkpoints fire
    assert!(
        store.last_checkpoint_error().is_none(),
        "healthy auto-checkpoints must not record a checkpoint error"
    );
}

// ---------------------------------------------------------------------------
// Safe node deletion: never hit lbug 0.15.3's buggy DETACH-DELETE-with-CSR-rels
// path (#98). In a long-running consumer daemon the pinned engine SIGSEGVs in
//   getGroup(groupIdx = 4294967295 / UINT32_MAX) -> null unique_ptr deref
//   <- CSRNodeGroup::scanCommittedInMemRandom <- CSRNodeGroup::scan
//   <- RelTableScanState::scanNext <- RelTable::detachDeleteForCSRRels
//   <- RelTable::detachDelete <- DeleteNode (Cypher DETACH DELETE)
// when a `DETACH DELETE n` removes a node that owns relationships in a
// *committed* (checkpointed) CSR rel group. `delete_node` must therefore strip
// every incident edge first, then plain-`DELETE` the now-isolated node — never
// emit `DETACH`.
//
// The native crash is timing/state-dependent: it depends on internal CSR
// node-group allocation state that accumulates over hours of consolidation, so
// it is not deterministically reproducible from a fresh, small store. These
// tests therefore (a) pin down the *behavioral contract* of the rewritten
// delete — node + every incident edge (outgoing, incoming, self-loop) removed,
// unrelated data untouched, `true` returned — and (b) act as a **crash
// tripwire**: each forces a checkpoint to materialize a committed CSR group and
// then deletes an edge-bearing node, so if the engine ever does take the
// detachDelete CSR path and SIGSEGVs, the test binary dies and the suite fails.
// A re-introduction of `DETACH DELETE` would reopen the crash on the consumer;
// these tests lock in the delete-edges-first behavior that prevents it.
// ---------------------------------------------------------------------------

/// REGRESSION (#98): deleting a node that owns *committed* CSR relationships
/// (outgoing, incoming, and a self-loop) must succeed, remove the node and all
/// of its incident edges, leave unrelated nodes intact, and — above all — not
/// SIGSEGV in lbug's `detachDeleteForCSRRels`. The edge-bearing committed-CSR
/// delete is exactly the path that crashes the consumer's daemon under the old
/// `DETACH DELETE`; the delete-edges-first rewrite must satisfy this contract
/// without ever emitting `DETACH`.
#[test]
fn delete_node_with_committed_csr_edges_does_not_crash() {
    let (_tmp, mut store) = open_temp();

    for id in ["a", "b", "c"] {
        store
            .add_node(
                "Thing",
                props(&[("node_id", id), ("agent_id", "x")]),
                Some(id),
            )
            .unwrap();
    }

    // a owns three relationships of distinct types so the deleted node has a
    // populated CSR adjacency in every shape that detachDelete walks:
    //   - outgoing  a -[LINKS]-> b
    //   - incoming  b -[REL2]-> a
    //   - self/loop a -[SELF]-> a
    store.add_edge("a", "b", "LINKS", None).unwrap();
    store.add_edge("b", "a", "REL2", None).unwrap();
    store.add_edge("a", "a", "SELF", None).unwrap();

    // Fold the writes into the main DB file so the rels live in a *committed*
    // CSR node group — the exact precondition for the getGroup(UINT32_MAX)
    // null-deref. Without this the rels stay in the in-memory delta and the
    // crashing code path is never reached.
    store
        .checkpoint()
        .expect("checkpoint must commit the CSR group");

    // The call that previously SIGSEGV'd. Surviving past it == the fix works.
    assert!(
        store.delete_node("a"),
        "deleting an edge-bearing node must report success"
    );

    // The node itself is gone.
    assert!(
        store.get_node("a").is_none(),
        "the deleted node must no longer resolve"
    );

    // Every edge incident to `a` (both directions + the self-loop) is gone:
    // b's only relationships were a->b and b->a, so b is now fully isolated.
    assert!(
        store
            .query_neighbors("a", None, Direction::Both, 10)
            .is_empty(),
        "the deleted node must have no remaining incident edges"
    );
    assert!(
        store
            .query_neighbors("b", None, Direction::Both, 10)
            .is_empty(),
        "edges pointing at/from the deleted node must be removed from its neighbors"
    );

    // An unrelated node is untouched by the targeted delete.
    assert!(
        store.get_node("b").is_some(),
        "a surviving neighbor must remain after the delete"
    );
    assert!(
        store.get_node("c").is_some(),
        "an unrelated node must be untouched by the delete"
    );
}

/// REGRESSION (#98) + injection hardening (SR-1): the edge-bearing delete path
/// must keep escaping `node_id` everywhere it is interpolated into Cypher. A
/// node whose id contains quotes/backslashes is given committed CSR edges, then
/// deleted; it must succeed, vanish, and take only its own edges with it.
#[test]
fn delete_node_with_edges_escapes_injection_id() {
    let (_tmp, mut store) = open_temp();

    // A node_id packed with Cypher-significant characters. If any interpolation
    // skips escape_cypher the edge-delete or node-delete statement breaks (or,
    // worse, mutates unrelated rows) and these assertions fail.
    let weird = "a'b\\c\"d";
    store
        .add_node(
            "Thing",
            props(&[("node_id", weird), ("agent_id", "x")]),
            Some(weird),
        )
        .unwrap();
    store
        .add_node(
            "Thing",
            props(&[("node_id", "safe"), ("agent_id", "x")]),
            Some("safe"),
        )
        .unwrap();

    store.add_edge(weird, "safe", "LINKS", None).unwrap();
    store.add_edge("safe", weird, "REL2", None).unwrap();
    store.add_edge(weird, weird, "SELF", None).unwrap();

    store
        .checkpoint()
        .expect("checkpoint must commit the CSR group");

    assert!(
        store.delete_node(weird),
        "deleting an edge-bearing node with a tricky id must succeed"
    );
    assert!(
        store.get_node(weird).is_none(),
        "the escaped-id node must be gone"
    );
    assert!(
        store.get_node("safe").is_some(),
        "the neighbor must survive — escaping must target only the deleted node"
    );
    assert!(
        store
            .query_neighbors("safe", None, Direction::Both, 10)
            .is_empty(),
        "edges to/from the deleted node must be removed from the survivor"
    );
}

/// Existing behavior must still hold once rel tables exist: deleting an
/// *edgeless* node (after a checkpoint, with other relationships present in the
/// catalog) succeeds and leaves every unrelated node and edge intact. This
/// exercises the rewritten path's guard — the incident-edge passes run but
/// match nothing, then the plain `DELETE` removes the isolated node.
#[test]
fn delete_edgeless_node_with_rel_tables_present() {
    let (_tmp, mut store) = open_temp();

    for id in ["p", "q", "z"] {
        store
            .add_node(
                "Thing",
                props(&[("node_id", id), ("agent_id", "x")]),
                Some(id),
            )
            .unwrap();
    }
    // An unrelated edge so the catalog has a rel table (known_rel_tables is
    // non-empty), but `z` itself owns no relationships.
    store.add_edge("p", "q", "LINKS", None).unwrap();
    store.checkpoint().expect("checkpoint must commit");

    assert!(
        store.delete_node("z"),
        "deleting an edgeless node must still succeed when rel tables exist"
    );
    assert!(
        store.get_node("z").is_none(),
        "the edgeless node must be gone"
    );

    // The unrelated edge and its endpoints are untouched.
    assert!(
        store.get_node("p").is_some(),
        "unrelated node p must survive"
    );
    assert!(
        store.get_node("q").is_some(),
        "unrelated node q must survive"
    );
    let nbrs = store.query_neighbors("p", Some("LINKS"), Direction::Outgoing, 10);
    assert_eq!(nbrs.len(), 1, "the unrelated edge must be preserved");
    assert_eq!(nbrs[0].1.node_id, "q");
}

/// REGRESSION (#98), daemon-lifecycle variant: the consumer crashes while
/// consolidating a *reopened* store, where the CSR rel groups were committed to
/// disk in a previous process and then freshly loaded. Reproduce that shape —
/// build edges, checkpoint, `close`, reopen — then delete an edge-bearing node.
/// The reopened, committed-on-disk CSR group is the closest in-process analogue
/// of the `scanCommittedInMemRandom` state in the symbolized backtrace, so this
/// is the strongest crash tripwire; behaviorally the delete must still remove
/// the node and all of its incident edges with the survivor left intact.
#[test]
fn delete_node_with_edges_survives_close_and_reopen() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("graph.ladybug");

    {
        let mut store = LbugGraphStore::open(&path, Some("s")).unwrap();
        for id in ["a", "b", "c"] {
            store
                .add_node(
                    "Thing",
                    props(&[("node_id", id), ("agent_id", "x")]),
                    Some(id),
                )
                .unwrap();
        }
        store.add_edge("a", "b", "LINKS", None).unwrap();
        store.add_edge("c", "a", "LINKS", None).unwrap();
        store.add_edge("a", "a", "SELF", None).unwrap();
        // Fold the rels into the main DB file, then drop the handle so the next
        // process loads them as committed-on-disk CSR groups.
        store
            .checkpoint()
            .expect("checkpoint must commit the CSR group");
        store.close();
    }

    let mut store = LbugGraphStore::open(&path, Some("s")).unwrap();

    // The delete that crashes the consumer's daemon after a long uptime.
    assert!(
        store.delete_node("a"),
        "deleting an edge-bearing node after reopen must succeed"
    );
    assert!(
        store.get_node("a").is_none(),
        "the deleted node must be gone after reopen"
    );
    assert!(
        store
            .query_neighbors("b", None, Direction::Both, 10)
            .is_empty(),
        "edges incident to the deleted node must be removed after reopen"
    );
    assert!(
        store.get_node("b").is_some() && store.get_node("c").is_some(),
        "unrelated neighbors must survive the reopen delete"
    );
}
