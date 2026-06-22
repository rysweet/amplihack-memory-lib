# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Critical native crash (#98):** the persistent store
  (`LbugGraphStore`) `SIGSEGV`-crashed the consumer's daemon every ~45–90
  minutes during retention/dedup consolidation. `delete_node` issued a Cypher
  `MATCH (n) WHERE n.node_id = '…' DETACH DELETE n`, and when the node had
  relationships in a committed CSR rel table, `lbug 0.15.3`'s
  `detachDeleteForCSRRels` computed an invalid CSR node-group index
  (`getGroup(UINT32_MAX)` → null `unique_ptr` deref) and crashed. After the
  graph-enhancement work most nodes carry edges (`DERIVES_FROM`, `SIMILAR_TO`,
  `SUPERSEDES`, `PROCEDURE_DERIVES_FROM`), so consolidation routinely
  detach-deleted an edge-bearing node and brought the daemon down. Two
  complementary fixes:
  - **Edge-first, no-`DETACH` deletion** removes the trigger: `delete_node` now
    deletes every incident relationship first via two directed, label-less
    passes (`MATCH (a)-[r]->(b) WHERE a.node_id='…' DELETE r` for both
    directions, covering self-loops and parallel edges), then deletes the
    now-isolated node with a plain `DELETE` (no `DETACH`), so the buggy
    `detachDeleteForCSRRels` path is never entered. The whole operation runs
    under the write lock and is fail-closed: if the edge cleanup errors,
    `delete_node` returns `false` without touching the node, so no half-deleted,
    edge-orphaned node is left behind. Return semantics, id→table cache
    eviction, the per-write durability barrier, and auto-checkpointing are
    unchanged. See `rust/amplihack-memory/docs/safe_node_deletion.md`.
  - **Optional engine patch bump** as defense in depth: where it builds cleanly the
    pinned `lbug` engine is bumped from `=0.15.3` to `=0.15.4` (latest 0.15.x patch;
    no jump to 0.17.x), otherwise the pin stays at `=0.15.3`. The workaround stands
    on its own and does not depend on the bump.
- **Critical durability bug (#95):** a failed auto-`CHECKPOINT` could
  permanently brick the persistent store and crash-loop the consumer. The
  buffer-pool cap was hardcoded at **128 MiB**, so on a busy host a checkpoint
  exhausted it (`"Buffer manager exception: ... buffer pool is full"`); the
  failed checkpoint corrupted the catalog, and every subsequent open failed with
  `"Load table failed: table 0 doesn't exist in catalog"`. Recovery only handled
  a corrupt WAL, not a corrupt catalog, so the error propagated and the daemon
  `exit(1)`'d forever. Two complementary fixes:
  - **Larger, configurable limits** remove the trigger: the buffer-pool cap and
    max database size are now read from `AMPLIHACK_MEMORY_BUFFER_POOL_BYTES` /
    `AMPLIHACK_MEMORY_MAX_DB_BYTES` and default to **1 GiB** (buffer pool, up
    from 128 MiB) and **16 GiB** (max DB size, up from 1 GiB). lbug allocates the
    buffer pool lazily and `max_db_size` is only an mmap reservation, so the
    larger caps cost nothing until data needs them. Overrides are clamped to sane
    minimums (64 MiB / 1 GiB), `buffer_pool <= max_db_size` is enforced, and the
    effective values are logged once at open.
  - **Catalog / main-DB corruption recovery** removes the crash loop:
    `open_with_recovery` (and therefore `open_persistent`) now self-heals a
    corrupt catalog. When the main database cannot be opened even with the WAL
    fully quarantined — or with no WAL present at all — the entire database is
    quarantined to `<db_path>.corrupt-<timestamp>` (moved aside, never deleted)
    and a fresh, empty database is opened, surfaced as the new
    `WalRecoveryOutcome::RebuiltAfterCorruption`. The strict `open()` stays
    strict and still errors.
- **Critical durability bug (#88):** the persistent `CognitiveMemory`
  (`LbugGraphStore`) could no longer be opened after an unclean shutdown left
  the LadybugDB write-ahead log (WAL) partially written — `open_persistent`
  crashed on a C++ WAL-replay assertion and the only "recovery" was deleting the
  WAL, losing every uncheckpointed record. `open_persistent` now transparently
  recovers: the unreplayable WAL is quarantined to `<wal>.corrupt-<timestamp>`
  (moved aside, never deleted), the recoverable prefix is replayed, and a
  checkpoint folds the survivors into the main database file.

### Added
- `AMPLIHACK_MEMORY_BUFFER_POOL_BYTES` / `AMPLIHACK_MEMORY_MAX_DB_BYTES`
  environment overrides for the LadybugDB buffer-pool cap and max database size,
  plus `LbugGraphStore::buffer_pool_bytes()` / `max_db_bytes()` getters for the
  effective values.
- `WalRecoveryOutcome::RebuiltAfterCorruption` — reported when a corrupt
  catalog / main database file is quarantined and a fresh empty database is
  rebuilt in its place (`recovered_records = 0`, quarantine path recorded).
- `LbugGraphStore::last_checkpoint_error()` — exposes the most recent
  checkpoint failure (e.g. buffer-pool exhaustion), cleared on the next
  successful checkpoint, so consumers can surface store health.
- `CognitiveMemory::open_persistent_with_recovery` and
  `LbugGraphStore::open_with_recovery` — explicit corrupt-WAL recovery entry
  points returning a structured `WalRecovery` report (`WalRecoveryOutcome`,
  recovered record count, quarantine path). `open_persistent` delegates to the
  recovery path so existing callers gain crash-resilience with no code change.
- `CognitiveMemory::checkpoint()` / `GraphStore::checkpoint()` — force-flush the
  WAL into the durable store so a clean reopen needs no replay (no-op for the
  in-memory backend).
- Automatic checkpointing every `AUTO_CHECKPOINT_WRITES` (128) writes and on
  `close` / `Drop`, bounding how much data an unclean shutdown can strand in the
  WAL.

## [0.4.0] - 2026-03-13

### Added
- Quality audit cycle with 83 findings fixed (PRs #57–#60)
- 31 outside-in behavioral tests (`tests/outside_in.rs`)
- Property-based and concurrent tests (`tests/concurrent.rs`)
- Supply chain security with `cargo-deny` (`deny.toml`)
- Python integration tests for PyO3 bindings
- Extended `.gitignore` for Rust, editor, and Python entries

### Changed
- Split 6 oversized modules into submodules — all modules now < 400 LOC (#47, #56)
- Improved CI: kuzu tests, `cargo audit`, MSRV 1.80 verification, dependency caching (#49, #50)
- Doc comments added to all public items (#55)

### Fixed
- 21 HIGH severity issues covering security, correctness, and error handling (#57, #60)
- Credential scrubber now covers metadata and tags (#48)
- SQL/Cypher injection hardening (#48)
- Path traversal validation on Python bindings (#48)
- Kuzu `validate()` and federated doc comments (#58)

## [0.3.0] - 2026-03-12

### Added
- Complete Rust port of amplihack-memory-lib from Python
- Native LadybugDB backend replacing PyO3 Kuzu bridge
- Kuzu graph database support via PyO3 bridge
- Comprehensive security hardening (#48)
- Multi-threaded concurrent access tests (#43)
- Graph, hierarchical, concurrent, and pattern benchmarks (Criterion) (#54)
- PyO3 Python bindings with graph operations (#53)
- HiveStore: 5 public functions with 30 tests (#52)
- GitHub Actions CI pipeline for Rust and Python tests
- Reverse-edge index for O(1) incoming-edge lookup (#42)

### Changed
- Code quality improvements: DRY patterns, dead code removal, constants (#45)
- Removed 17 unnecessary `.clone()` calls across 5 files
- Module split, dead code removal, parity tests (Phase 2)
- Expanded amplihack-memory README (#51)

### Fixed
- Python–Rust behavioral parity alignment (#46)
- Replaced swallowed errors with `tracing` warnings (#44)
- String slicing panic prevention and extended experience ID hash (#41)
- 12 quality audit findings (2 HIGH, 4 MEDIUM, 6 LOW)
- All audit findings in Python bindings resolved

## [0.2.0] - 2026-02-27

### Added
- 6-type cognitive memory system backed by Kuzu (episodic, semantic, procedural, prospective, sensory, working) (#1)
- Dual-backend architecture: Kuzu (default) + SQLite
- CognitiveMemory exported from package (#7)
- Comprehensive documentation and GitHub Pages with MkDocs (#5)
- Release notes for v0.1.0

## [0.1.0] - 2026-02-15

### Added
- Initial release of amplihack-memory-lib
- 166 unit tests
- SQLite backend and InMemoryGraphStore
- Cognitive memory with 6 experience types
- Hierarchical memory with experience consolidation
- Pattern recognition and semantic search
- Security layer with credential scrubbing

[0.4.0]: https://github.com/rysweet/amplihack-memory-lib/compare/v0.2.0...HEAD
[0.3.0]: https://github.com/rysweet/amplihack-memory-lib/compare/v0.2.0...v0.2.0
[0.2.0]: https://github.com/rysweet/amplihack-memory-lib/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/rysweet/amplihack-memory-lib/releases/tag/v0.1.0
