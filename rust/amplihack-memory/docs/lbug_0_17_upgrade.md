# LadybugDB engine upgrade — `lbug` 0.15.4 → 0.17.1

The persistent `GraphStore` backend (`LbugGraphStore`, which powers
`CognitiveMemory::open_persistent`) links the published LadybugDB engine through
the [`lbug`](https://crates.io/crates/lbug) crate. This page is the authoritative
description of the coordinated engine upgrade from **`lbug = "=0.15.4"`** to
**`lbug = "=0.17.1"`** across `amplihack-memory-lib` and its downstream consumer
**Simard**, tracked by issue **#100**.

> **Status — lib side landed; Simard pending.** On the `amplihack-memory-lib`
> side this is now the *implemented* state: `rust/amplihack-memory/Cargo.toml`
> pins `lbug = "=0.17.1"`, the v40-fixture data-loss regression is committed and
> **CI-blocking** (the dedicated "Data-loss gate" step in `.github/workflows/ci.yml`),
> and the full persistent suite stays green. The **Simard** side (PR #2, a
> separate checkout) is still pending — see [Simard coordination](#simard-coordination).
> Present-tense claims about Simard describe the intended post-merge behavior.

The headline guarantees are:

- **No data loss.** A live cognitive store written by `lbug 0.15.4`
  (on-disk **storage format v40**) opens **in place** under `lbug 0.17.1`
  (**storage format v41**). `0.17.1`'s `canReadStorageVersion()` accepts v40, so
  there is **no dump/reload, no quarantine, and no rebuild-from-empty**. The
  live store survives the binary swap untouched.
- **The CSR crash does not regress.** The `getGroup(UINT32_MAX)` `SIGSEGV`
  (issue #100; an upstream LadybugDB CSR-engine bug) is **version-independent** —
  it reproduces on `lbug` 0.15.3, 0.15.4, **and** 0.17.1. The fix that carries
  forward is the **soft-delete (tombstone)** design already shipped in
  [`docs/safe_node_deletion.md`](safe_node_deletion.md), *not* the version bump.
  An optional patched engine crate adds C++ defense-in-depth (see
  [Defense in depth](#defense-in-depth--the-patched-engine-crate)).
- **The bump is mechanically non-breaking.** The 0.15.4 → 0.17.1 Rust-binding API
  delta is purely *additive* (new `Value::Json`, `LogicalType::Json`,
  `Error::JsonError`, `SystemConfig::enable_multi_writes`); every `match` on
  `lbug::Value` in this crate already carries a `_ =>` catch-all, so no logic
  changes are required — only the version pin and a recompile.

> **Feature gate.** Everything on this page requires the `persistent` cargo
> feature (which pulls in the `lbug` engine):
> `cargo build --features persistent` / `cargo test --features persistent`.

---

## Table of contents

1. [Why upgrade](#why-upgrade)
2. [The crash is version-independent](#the-crash-is-version-independent)
3. [Storage format: v40 → v41 (one-way, in-place)](#storage-format-v40--v41-one-way-in-place)
4. [Recovery safety — a readable old store is never rebuilt](#recovery-safety--a-readable-old-store-is-never-rebuilt)
5. [Rust API delta (0.15.4 → 0.17.1)](#rust-api-delta-0154--0171)
6. [Build & CI configuration](#build--ci-configuration)
7. [Defense in depth — the patched engine crate](#defense-in-depth--the-patched-engine-crate)
8. [Simard coordination](#simard-coordination)
9. [Configuration reference](#configuration-reference)
10. [Examples](#examples)
11. [Operational runbook — upgrading a live store](#operational-runbook--upgrading-a-live-store)
12. [Compatibility & guarantees](#compatibility--guarantees)
13. [Testing](#testing)

---

## Why upgrade

`lbug 0.17.1` is the current stable 0.17 line. Relative to 0.15.4 it brings a
catalog/storage format revision (v41), native JSON values, an opt-in
multi-writer mode, and — relevant to CI — a **precompiled static engine archive**
that no longer requires building the C++ engine from source on every build. The
upgrade keeps the exact-pin convention (`=0.17.1`) so the engine version stays
reproducible directly from `Cargo.toml` (the engine version is fixed by the
exact `=` pin regardless of the lockfile, which this repo gitignores).

The upgrade is **orthogonal to the crash fix**: it does not by itself fix or
regress the CSR `SIGSEGV`. The two were investigated and decided independently.

---

## The crash is version-independent

Issue #100 is a null-pointer dereference inside LadybugDB's CSR rel-table scan:
a **physical relationship delete out of a *committed* CSR rel group**, interleaved
with a checkpoint, drives the CSR node-group index to the `UINT32_MAX`
(`INVALID_CHUNKED_GROUP_IDX`) sentinel; the next scan to touch that table
dereferences a null `unique_ptr`.

```
getGroup(groupIdx = 4294967295 / UINT32_MAX)   -> null unique_ptr deref
  <- CSRNodeGroup::scanCommittedInMemRandom
  <- CSRNodeGroup::scan
  <- RelTableScanState::scanNext
```

This was **reproduced identically on `lbug` 0.15.3, 0.15.4, and 0.17.1** (and on
`ladybug` main). It is an engine-level CSR delete+checkpoint bug, not a
consumer-side concurrency issue, so **a version bump alone does not fix it.**

The carried-forward fix is the **soft-delete tombstone**: `delete_node` /
`delete_edge` issue no physical `DELETE`; they set a reserved `_deleted` column
(a property write that never mutates the CSR adjacency structure) and every read
filters tombstoned rows out. `add_node` revives a tombstoned id. That keeps
`LbugGraphStore` off the crashing code path entirely, on every engine version.
See [`docs/safe_node_deletion.md`](safe_node_deletion.md) for the full design and
the `reproduces_issue_100_csr_delete_corruption` regression test, which stays
green on 0.17.1.

---

## Storage format: v40 → v41 (one-way, in-place)

| `lbug` version | On-disk storage version |
| --- | --- |
| 0.12 – 0.16.1 | `STORAGE_VERSION_40` (v40) |
| 0.17.0 / 0.17.1 | `STORAGE_VERSION_41` (v41) |

`lbug 0.17.1`'s `StorageVersionInfo::canReadStorageVersion(v)` returns `true` for
`v == 40 || v == 41`. Concretely:

- **Opening a v40 store under 0.17.1 succeeds with no conversion step.** Reads
  return the existing nodes and edges immediately — the live store is *not*
  migrated, quarantined, or rebuilt on open.
- **The first checkpoint under 0.17.1 upgrades the on-disk store to v41**
  (the v41 change adds a table-storage FORMAT field to catalog entries). The
  store auto-checkpoints after `AUTO_CHECKPOINT_WRITES` (128) mutations and on
  `close`/`Drop`, so in practice the upgrade to v41 happens shortly after the
  first writes.
- **The upgrade is forward-only.** Once a store has been checkpointed to v41,
  **`lbug` 0.15.x can no longer open it.** Rollback to a 0.15.x binary therefore
  requires a **backup taken before the first 0.17.1 checkpoint** (see the
  [runbook](#operational-runbook--upgrading-a-live-store)).

**No dump/reload migration tool is needed or provided.** Because v40 is
read-compatible, an export-from-0.15.4 / import-into-0.17.1 path would be wasted
work and additional risk. The migration *is* "open the store with the new
binary"; durability is the automatic, in-place v40→v41 upgrade.

---

## Recovery safety — a readable old store is never rebuilt

The resilient entry point `LbugGraphStore::open_with_recovery` (wrapped by
`CognitiveMemory::open_persistent`) self-heals a corrupt **WAL** or corrupt
**catalog/main file** by quarantining the bad artifact and, in the catalog case,
rebuilding from empty — see [`docs/durability_and_recovery.md`](durability_and_recovery.md).
That rebuild-from-empty path is the data-loss mechanism this upgrade must **not**
trigger for a healthy older-format store.

Guarantees for the v40 → v41 upgrade:

- **A readable v40 store opens `Clean`.** Because 0.17.1 reads v40, the strict
  open in `open_with_recovery` succeeds, returns `WalRecoveryOutcome::Clean`, and
  the rebuild path is **never entered**. A correct format upgrade is *not*
  classified as corruption.
- **A genuinely-unreadable format version is surfaced as "needs migration", not
  corruption.** As forward-looking insurance for a *future* format jump (a
  hypothetical store the engine truly cannot read), an unreadable-storage-version
  open failure is reported distinctly rather than being quarantined-and-rebuilt
  from empty. For the 0.15.4 → 0.17.1 hop this branch is never taken (v40 is
  readable); it exists so the next format crossing cannot silently destroy data.

The practical contract: **deploying the 0.17.1 binary against an existing v40
store preserves every node and edge.** The fixture regression test below proves
it.

---

## Rust API delta (0.15.4 → 0.17.1)

The consumer-visible part of the 0.15.4 → 0.17.1 binding API is **additive**. The
new surface and its impact on this crate:

| Addition | Where | Impact on `amplihack-memory` |
| --- | --- | --- |
| `Value::Json(serde_json::Value)` (+ `ConversionError::Json`) | `value.rs` | None — every `match` on `Value` (`value_to_string`, `value_as_str`, `value_as_usize`) has a `_ =>` arm. |
| `LogicalType::Json` | `logical_type.rs` | None — not matched exhaustively. |
| `Error::JsonError` (+ `From<serde_json::Error>`) | `error.rs` | None — errors are stringified. |
| `SystemConfig::enable_multi_writes` (builder; `Default = false`) | `database.rs` | None — `SystemConfig` is built via the existing builder chain (`max_db_size` → `buffer_pool_size` → `auto_checkpoint` → `throw_on_wal_replay_failure`); the new field defaults off. |
| `LBUG_LIBRARY_SOURCE` / `get_library_source()`, `CsrResult`/`csr()` (arrow) | `ffi.rs`, `query_result.rs` | None — unused. |

**Compile-impact verdict: zero match-arm edits.** The whole change in
`graph/lbug_store/{mod.rs, store_impl.rs, tests.rs}` is the version pin plus a
recompile. The unchanged `SystemConfig` builder in `system_config()` continues to
produce the same configuration; the new `enable_multi_writes` field stays at its
`false` default (single-writer semantics, matching today's behavior).

The dependency change in `rust/amplihack-memory/Cargo.toml`:

```toml
[dependencies]
# Published LadybugDB (Kùzu) bindings used by the persistent GraphStore backend.
# Same crate/version Simard depends on; only pulled in by the `persistent` feature.
lbug = { version = "=0.17.1", optional = true }   # was "=0.15.4"
```

`amplihack-memory-lib` depends on the **unpatched** crates.io `lbug` so the crate
remains publishable; the C++ defense-in-depth patch is applied only by Simard via
`[patch.crates-io]` (see below).

---

## Build & CI configuration

`lbug 0.17.x` **downloads a precompiled static `liblbug` archive by default**
instead of always compiling the C++ engine from source. This speeds up CI but
introduces a network dependency at build time and means the C++ defense-in-depth
patch is linked **only** when building from source / from the patched crate.

Relevant environment variables (read by the `lbug` build script):

| Variable | Effect |
| --- | --- |
| `LBUG_BUILD_FROM_SOURCE=1` | Force compiling the C++ engine from source (hermetic CI; required toolchain: `cmake` + a C++ compiler). |
| `LBUG_RUST_BUILD_FROM_SOURCE=1` | Force building the Rust bindings against a from-source engine. |
| `LBUG_SOURCE_DIR=<path>` | Use a local engine source tree (e.g. a vendored/patched checkout) instead of downloading. |

**CI guidance.** The `persistent`-feature job builds with the precompiled archive
by default. For fully hermetic builds (no network), set `LBUG_BUILD_FROM_SOURCE=1`
(and `LBUG_SOURCE_DIR` when using a patched tree). MSRV is unchanged at **1.85**;
`lbug 0.17.1` builds on 1.85.

```bash
# Default (fast) — precompiled engine archive:
cargo test -p amplihack-memory --features persistent

# Hermetic / patched — compile the engine (and patch) from source:
LBUG_BUILD_FROM_SOURCE=1 cargo test -p amplihack-memory --features persistent
```

---

## Defense in depth — the patched engine crate

The soft-delete tombstone already keeps `LbugGraphStore` off the crashing path,
so the C++ patch is **belt-and-suspenders** — it matters for *already-corrupted*
stores and for any other consumer that still issues physical deletes from a
committed CSR group.

The patch guards `getGroup(INVALID_CHUNKED_GROUP_IDX == UINT32_MAX)` with a
`getNumGroups()` bounds-check before the dereference in `CSRNodeGroup::scan*`,
applied to a **0.17.1** engine. It is maintained as sibling checkouts:
`ladybug-fork` (the patched engine source — `ladybug @ 7c5210f` plus the CSR
`getGroup` fix) and `lbug-patched` (the `lbug` 0.17.1 crate built against that
engine). It is **opt-in**, wired only where needed via Cargo's
`[patch.crates-io]` to redirect `lbug` to the patched crate:

```toml
# Consumer Cargo.toml — opt in to the patched 0.17.1 engine (defense in depth).
# Overrides lbug for the WHOLE dependency graph, including amplihack-memory,
# so the C++ bounds-check is linked transitively. Point it at the local
# `lbug-patched` checkout; a published git source can be substituted later.
[patch.crates-io]
lbug = { path = "../lbug-patched" }   # or { git = "<fork-url>", tag = "v0.17.1-csr-fix" }
```

`amplihack-memory-lib` itself does **not** add this `[patch.crates-io]` (it stays
on the published crate so it remains publishable from crates.io). Simard **adds**
a `[patch.crates-io] lbug` (its `Cargo.toml` has no `[patch]` section today)
pointing at the 0.17.1 patched crate so the bounds-check is linked into the
deployed binary.

---

## Simard coordination

Simard links `lbug` **twice**: directly (its own `lbug` dependency) and
transitively (through its `amplihack-memory` git dependency). Both must move to
0.17.1 together so the final binary links exactly one engine and one storage
format.

Simard-side changes (PR #2, separate checkout, lands after the
`amplihack-memory-lib` PR merges). Simard's `Cargo.toml` today has
`lbug = "0.15"`, an **unpinned** `amplihack-memory` git dependency (no
`rev`/`tag`), and **no** `[patch.crates-io]` section:

1. Bump Simard's own `lbug = "0.15"` → `=0.17.1`.
2. Pin the (currently floating) `amplihack-memory` git dependency to the merged
   `amplihack-memory-lib` commit that carries the 0.17.1 pin.
3. Add a `[patch.crates-io] lbug` pointing at the **0.17.1** patched crate (gives
   the CSR `getGroup` bounds-check; Simard has none today).
4. No code edits to `src/cognitive_memory/mod.rs` or its `as_str` / `as_i64` /
   `as_f64` helpers — they already use `_ =>` catch-all arms, so the additive
   `Value::Json` variant is non-breaking.

Verification gate:

```bash
# Exactly one lbug version in the whole graph, and it is 0.17.x:
cargo tree -i lbug
```

---

## Configuration reference

The buffer-pool / database-size limits and checkpoint behavior are unchanged by
the upgrade; they are repeated here for completeness (full reference in
[`docs/durability_and_recovery.md`](durability_and_recovery.md)).

| Setting | Env var | Default | Notes |
| --- | --- | --- | --- |
| Buffer-pool cap | `AMPLIHACK_MEMORY_BUFFER_POOL_BYTES` | 1 GiB | Allocated lazily; clamped to ≥ 64 MiB and ≤ max DB size. |
| Max DB size | `AMPLIHACK_MEMORY_MAX_DB_BYTES` | 16 GiB | mmap address-space reservation only; clamped to ≥ 1 GiB. |
| Build from source | `LBUG_BUILD_FROM_SOURCE` | unset (download) | `=1` compiles the C++ engine (needed to link the patch). |
| Engine source dir | `LBUG_SOURCE_DIR` | unset | Path to a vendored/patched engine tree. |
| Multi-writer mode | n/a (`SystemConfig::enable_multi_writes`) | `false` | Left at default; single-writer semantics preserved. |

---

## Examples

### Opening an existing v40 store under 0.17.1 (no migration step)

```rust
use amplihack_memory::cognitive_memory::CognitiveMemory;

// `~/.simard/cognitive` was written by lbug 0.15.4 (storage v40).
// Under the lbug 0.17.1 binary it opens in place — every fact/episode/edge is
// present immediately; no export/import, no quarantine, no rebuild.
// Signature is `open_persistent(path, agent_name)`.
let mem = CognitiveMemory::open_persistent("~/.simard/cognitive", "agent")?;

// Reads work straight away against the v40 store.
let recalled = mem.recall_episodes(10);
assert!(!recalled.is_empty());

// The first checkpoint upgrades the on-disk store to v41 (forward-only).
mem.checkpoint()?;
# Ok::<(), amplihack_memory::MemoryError>(())
```

### Inspecting recovery outcome (proves no rebuild)

```rust
use amplihack_memory::graph::LbugGraphStore;
use amplihack_memory::graph::lbug_store::WalRecoveryOutcome;

// Opening a healthy v40 store reports Clean — NOT RebuiltAfterCorruption.
let (_store, recovery) = LbugGraphStore::open_with_recovery(v40_path, Some("agent"))?;
assert_eq!(recovery.outcome, WalRecoveryOutcome::Clean);
assert_ne!(recovery.outcome, WalRecoveryOutcome::RebuiltAfterCorruption);
# Ok::<(), amplihack_memory::MemoryError>(())
```

### Building with the patched engine for defense in depth

```bash
# Compile the engine (and the CSR getGroup bounds-check) from the patched
# engine source. LBUG_SOURCE_DIR points at the `ladybug-fork` engine tree.
LBUG_BUILD_FROM_SOURCE=1 \
LBUG_SOURCE_DIR=/path/to/ladybug-fork \
  cargo test -p amplihack-memory --features persistent
```

---

## Operational runbook — upgrading a live store

> **Deploy is out of scope for this change.** This runbook documents the safe
> procedure for whoever performs the deploy; the upgrade PRs do **not** restart
> or redeploy the running daemon.

1. **Stop the consumer** (so nothing is mid-write against the store).
2. **Back up the v40 store** (`cp -a ~/.simard/cognitive ~/.simard/cognitive.v40.bak`).
   This is the **only** rollback path to a 0.15.x binary, because the first
   0.17.1 checkpoint upgrades the store to v41 (forward-only).
3. **Validate on a copy first.** Open the backup copy with the 0.17.1 binary and
   assert item-count parity (e.g. the recall/total counts match the v40 store).
   Never validate against the live store directly.
4. **Deploy the 0.17.1 binary** and start the consumer pointed at the original
   store. It opens v40 in place; the first checkpoint upgrades it to v41.
5. **Confirm** post-start that the item count matches the pre-upgrade count and
   that no `RebuiltAfterCorruption` / quarantine artifact (`*.corrupt-*`) was
   written.

If step 3 or 5 shows any count regression or a quarantine artifact, **stop and
restore the v40 backup** — do not run on a rebuilt-from-empty store.

---

## Compatibility & guarantees

- **Data-preserving.** An existing v40 store opens in place under 0.17.1; the
  live cognitive store is not lost, quarantined, or rebuilt.
- **API-compatible.** No public `amplihack-memory` signature changes. The bump is
  a version pin + recompile; callers compile and behave identically.
- **Crash-fix preserved.** The soft-delete tombstone (issue #100) is retained and
  stays green on 0.17.1; the optional patched engine adds C++ defense in depth.
- **Forward-only storage.** After the first 0.17.1 checkpoint the store is v41 and
  cannot be reopened by 0.15.x; rollback requires the pre-upgrade backup.
- **MSRV unchanged.** Builds on Rust 1.85.
- **In-memory backend unaffected.** `InMemoryGraphStore` never linked the engine.

---

## Testing

Behind the `persistent` feature in
`rust/amplihack-memory/src/graph/lbug_store/tests.rs`:

- **v40 fixture opens losslessly under 0.17.1 (the migration regression).** A
  small, deterministic v40 store — generated out-of-band by a 0.15.4 build and
  committed as test data (following the existing `copy_dir_files` /
  `corrupt_wal_tail` fixture-staging pattern) — is staged into a tempdir and
  opened under the 0.17.1 engine. The test asserts the recovery outcome is
  `Clean` (not `RebuiltAfterCorruption`), and that node/edge counts and content
  hashes match the fixture. After a write + `checkpoint()` it asserts the store
  reports storage version 41. **This test is CI-blocking** via the dedicated
  "Data-loss gate" step (see *CI gate shape* below), so "green CI proving no data
  loss" is real.
- **CSR crash regression stays green.**
  `reproduces_issue_100_csr_delete_corruption` (the 400-round delete+checkpoint
  churn) passes on 0.17.1, proving the tombstone fix did not regress with the
  engine bump.
- **Soft-delete contract.** `re_add_after_soft_delete_revives_node` and the
  `not_deleted` recall-filter tests stay green.

**CI gate shape (`.github/workflows/ci.yml`).** The data-loss proof is a
**dedicated blocking step** that runs only the deterministic migration tests:

```yaml
- name: Data-loss gate (v40 store opens losslessly under lbug 0.17.x)
  run: cargo test --features persistent --lib graph::lbug_store::tests::v40
```

This step links the **prebuilt** lbug 0.17.x engine archive (no from-source C++
compile, so it is fast) and is fully deterministic, so making it merge-blocking
adds no flakiness — "green CI proving no data loss" is therefore *real*. The
broader `cargo test --features persistent` job stays `continue-on-error: true`
because it also runs the probabilistic 400-round CSR delete+checkpoint churn
(`reproduces_issue_100_csr_delete_corruption`), a native stress test kept
non-blocking to absorb rare upstream native crashes.

```bash
cargo test  -p amplihack-memory --features persistent --lib graph::lbug_store
cargo build -p amplihack-memory --features persistent
cargo clippy -p amplihack-memory --all-targets --features persistent -- -D warnings
```

> **Pre-commit note.** The configured pre-commit hooks build with default
> features only and do not compile the `persistent` code; the explicit
> `--features persistent` commands above are the authoritative gate for this
> upgrade.
