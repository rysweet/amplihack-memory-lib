# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Creative-idea prospective memory type + typed memory links
  (`creative_idea` module).** A first-class memory-model definition for
  *creative ideas* — future-oriented candidate self-improvements stored as
  prospective memories. Owns the lifecycle `CreativeIdeaStatus` state machine
  (validated `can_transition_to`, terminal states, fail-closed `FromStr`) and
  the typed `MemoryLink`/`MemoryLinkKind` edge taxonomy (`Semantic`,
  `Episodic`, `Procedural`, `Goal`) linking an idea to its supporting cognitive
  memory nodes, plus the stable `CREATIVE_IDEA_TRIGGER` prospective sentinel and
  `CREATIVE_IDEA_PAYLOAD_VERSION`. Purely additive; layers on the existing
  prospective primitive with no schema change so consumers orchestrate around
  the library rather than re-deriving the lifecycle/link taxonomy.

### Fixed
- **Recall ranking: stop-word/punctuation-blind similarity tokenizer.**
  `semantic_search.TFIDFSimilarity.calculate` (the base scorer behind
  `retrieve_relevant_experiences` / `SemanticSearchEngine.search`) tokenized with a
  bare `str.split()`, so stop words and punctuation counted toward Jaccard overlap.
  Topically unrelated notes that merely shared question-scaffolding words ("how do
  we …", "how to …") were scored as relevant and out-ranked genuinely relevant
  experiences. It now reuses the content-aware `similarity._tokenize` (lowercase,
  strip punctuation, drop stop words and short tokens), a single source of truth
  for tokenization. On the new recall-quality benchmark corpus this raises
  precision@3 from 0.2500 → 0.4583 and MRR from 0.4938 → 1.0000. Internal scorer
  only (not part of the public API); all existing tests pass unchanged.
- **Critical durable-recall data loss on WAL recovery (#2550, Simard #2550):**
  `LbugGraphStore::open_with_recovery` could permanently drop tens of thousands of
  recovered memories. On a corrupt-WAL open it replayed the good prefix and
  attempted a `CHECKPOINT` to fold it into the main DB file — but if that
  checkpoint-after-recovery **failed** (the verified incident: "cannot open
  `cognitive.wal(.checkpoint)` / error removing `cognitive.wal`"), the error was
  swallowed and the recovered records lived only in the now-quarantined WAL. A
  later open then read the pre-recovery (near-empty) main file and **reset the
  store to empty** — with no restore path. **Fix — durable-or-salvaged recovery:**
  a failed checkpoint-after-recovery no longer returns a store whose records are
  not persisted. The recovered graph is captured (dumped) while the resilient
  handle is still open and **salvaged into a fresh, clean database** (quarantine
  the un-checkpointable original → reload nodes+edges → checkpoint), where the
  checkpoint succeeds, so the pre-corruption prefix is durable across a strict
  reopen. `rebuild_after_corruption` likewise now attempts a **read-only salvage**
  of any still-readable records before resetting to empty, so a store that still
  holds recoverable records is never reset. Both paths report `RecoveredPrefix`
  with the surviving count. New hermetic regressions (no sleeps/network):
  `open_with_recovery_preserves_every_precorruption_record` (a corrupt WAL tail is
  recovered without an assertion/crash and every pre-corruption record survives +
  is durable) and `recovery_is_durable_when_checkpoint_after_recovery_fails` (the
  incident: a forced checkpoint-after-recovery failure salvages rather than resets
  to empty, durable across reopen). Added as a blocking CI data-loss gate.
- **Critical CSR corruption / native crash (#100):** the persistent store
  (`LbugGraphStore`) still `SIGSEGV`-crashed the consumer daemon during
  retention/dedup consolidation even after the #98/#99 `DETACH`-avoidance work.
  Root cause (bisected): a **physical relationship delete out of a *committed*
  CSR rel group** — `delete_edge`, and `delete_node`'s incident-edge strip —
  interleaved with checkpoints drives lbug's CSR node-group index to the
  `UINT32_MAX` sentinel; the next scan to touch that table dereferences a null
  group (`getGroup(UINT32_MAX)`). It is version-independent (reproduced on lbug
  0.15.3 / 0.15.4 / 0.17.1), i.e. an engine-level CSR delete+checkpoint bug, not
  a consumer-side concurrency issue. **Fix — soft delete (tombstone):**
  `delete_node` / `delete_edge` no longer issue any physical `DELETE`. They mark
  rows deleted with a reserved `_deleted` column (`SET …._deleted = '1'`, a
  property write that leaves the CSR adjacency structure untouched), and every
  read filters tombstoned rows out. `add_node` revives a tombstoned id so the
  "delete then re-add the same id" consolidation pattern still works. The
  `_deleted` column is created on every node/rel table and back-filled on older
  stores via `ALTER TABLE … ADD` at reopen. The observable delete/recall
  contract is unchanged. A 400-round delete+checkpoint churn that previously
  corrupted a committed CSR group by round ~10 now completes cleanly
  (`reproduces_issue_100_csr_delete_corruption`). Tombstoned rows remain
  physically resident until a future compaction pass.

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
- **Durable recall-quality regression benchmark** (`tests/test_recall_quality_benchmark.py`).
  Internal test infrastructure that guards the "recall quality" axis: a fixed,
  hand-labelled fixture corpus of agent experiences with known query→relevant
  results, plus a precision@k and MRR harness over the existing recall/ranking path
  (`semantic_search.retrieve_relevant_experiences`). CI-gated (runs in the
  `python-tests` job) — the metrics must never drop below the recorded baselines
  (precision@3 ≥ 0.4583, MRR ≥ 1.0), so any future change that degrades recall
  ranking fails CI. Deterministic (stable ids + shared timestamp; pure Python, no
  ML deps).
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
- **Deterministic episodic→semantic distillation heuristic + fact-yield
  benchmark (#117):** new `distillation` module with `distill_episode` /
  `distill_corpus` — a pure, dependency-free extractor that segments free-text
  episodes into atomic semantic-fact candidates (sentence + conjunctive-clause
  splitting, whitespace/punctuation normalisation, trivial-fragment filtering)
  and exact-dedups them case-insensitively with insertion order preserved. Adds
  a fixed `FACT_YIELD_BENCH_CORPUS` with `fact_yield` / `fact_yield_baseline`
  helpers: the naive one-fact-per-episode baseline yields `FACT_YIELD_BASELINE`
  (7) durable facts, while atomic segmentation yields `FACT_YIELD_IMPROVED` (13)
  — an +86% measurable gain locked in by a regression test asserting the yield
  never drops below the recorded baseline.

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
