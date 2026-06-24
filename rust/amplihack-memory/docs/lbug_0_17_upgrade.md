# LadybugDB engine upgrade — `lbug` 0.15.4 → 0.17.1

The persistent `GraphStore` backend (`LbugGraphStore`, which powers
`CognitiveMemory::open_persistent`) links the published LadybugDB engine through
the [`lbug`](https://crates.io/crates/lbug) crate. This page is the authoritative
description of the coordinated engine upgrade from **`lbug = "=0.15.4"`** to
**`lbug = "=0.17.1"`** across `amplihack-memory-lib` and its downstream consumer
**Simard**, tracked by issue **#100**.

> **Status — the #107 empty-read defence is now IMPLEMENTED in `amplihack-memory-lib`.**
> This page is both the design rationale and a record of the landed code. The
> root cause was confirmed **lib-side** (a `_deleted` tombstone filter against a
> legacy schema — see [Root cause (corrected)](#root-cause-corrected-the-_deleted-tombstone-filter-on-a-legacy-schema)),
> so **no cross-repo engine patch is required**. See
> [Implementation status](#implementation-status) for the per-capability table.
>
> - **Landed (`amplihack-memory-lib`, this branch):** `rust/amplihack-memory/Cargo.toml`
>   pins `lbug = "=0.17.1"`; the #100 migration regression (the **10-node** v40
>   fixture, `v40_fixture_opens_clean_and_preserves_all_data` +
>   `v40_fixture_on_disk_is_v40_and_upgrades_after_checkpoint`) is committed and
>   **CI-blocking** via the "Data-loss gate" step in `.github/workflows/ci.yml`; the
>   #95/#88 crash-recovery state machine (`Clean`, `RecoveredPrefix`,
>   `CheckpointOnly`, `RebuiltAfterCorruption`) is live in `open_with_recovery`;
>   **and the full #107 defence**: the schema-gated read-path fix, the read-only
>   **empty-read safety gate** in `open_with_recovery`, the
>   `WalRecoveryOutcome::SuspectedDataLoss` outcome + store **sealing**, the
>   `MemoryError::SuspectedDataLoss` fail-closed mapping, and the
>   `v40_legacy_store_*` / `v40_real_store_parity_*` / `v40_empty_read_gate_*`
>   regression tests (all CI-blocking via the `v40` selector). Empirically, a
>   0.17.1 build now reads a **copy** of the real `~/.simard/cognitive` v40 store
>   at full parity (`total = 2813`, matching 0.15.4), where before it read `0`.
> - **Superseded / not needed:** the cross-repo engine read fix (the engine reads
>   the real store correctly), the typed `NodeCount`, and a production-scale git-LFS
>   fixture (the bug is scale-independent; the env-gated `v40_real_store_parity_*`
>   test exercises a real-store copy when present).
> - **Simard (still pending — a separate checkout):** the 0.17.1 re-pin plus the
>   safe-update parity gate — see [Simard coordination](#simard-coordination).

> **⚠ Issue #107 — the silent empty-read.** An earlier draft of this page claimed
> an *unconditional* "no data loss". That was wrong and is retracted. `lbug 0.17.1`
> was found to **silently read certain real v40 stores as empty** (`total = 0`, no
> error, no quarantine artifact), which — if checkpointed — would upgrade an
> *empty* store to v41 and **permanently destroy** the data. This page specifies the
> [real root cause](#the-107-silent-empty-read-bug--root-cause), the
> [empty-read safety gate](#the-empty-read-safety-gate-item-count-parity) that will
> make the failure **fail-closed**, and the
> [production-scale fixture](#testing) that reproduces it. The gate is **planned, not
> yet implemented** — see [Implementation status](#implementation-status).

The headline guarantees this design will deliver are (see
[Implementation status](#implementation-status) for what is landed vs. planned):

- **No *silent* data loss — to be enforced by an item-count parity gate.** A live
  cognitive store written by `lbug 0.15.4` (on-disk **storage format v40**) is
  intended to open **in place** under `lbug 0.17.1` (**storage format v41**):
  `0.17.1`'s `canReadStorageVersion()` accepts v40, so there is **no dump/reload**.
  But because `0.17.1` could read a populated v40 store back as empty
  ([#107](#the-107-silent-empty-read-bug--root-cause)), the guarantee is
  **gated**, not blind. Two independent layers will cooperate:
  1. an **engine read-path fix** (in the patched 0.17.1 engine) restores full
     item-count parity for the real v40 catalog/CSR layout; and
  2. an always-on **empty-read safety gate** in `open_with_recovery` that
     **refuses** to proceed when a store with a material on-disk footprint reads
     back as empty — it **never checkpoints, never rebuilds, and never mutates a
     byte**, so the v40 store remains readable by a 0.15.x binary for rollback.

  The gate is intended to be the durable guarantee, holding *regardless* of whether
  the engine fix is linked; the engine fix turns the populated store from
  "gated-refused" back into "opens cleanly with every item present".
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

1. [Implementation status](#implementation-status)
2. [Why upgrade](#why-upgrade)
3. [The crash is version-independent](#the-crash-is-version-independent)
4. [Storage format: v40 → v41 (one-way, in-place)](#storage-format-v40--v41-one-way-in-place)
5. [The #107 silent empty-read bug — root cause](#the-107-silent-empty-read-bug--root-cause)
6. [The empty-read safety gate (item-count parity)](#the-empty-read-safety-gate-item-count-parity)
7. [Recovery safety — a readable old store is never rebuilt](#recovery-safety--a-readable-old-store-is-never-rebuilt)
8. [Rust API delta (0.15.4 → 0.17.1)](#rust-api-delta-0154--0171)
9. [Build & CI configuration](#build--ci-configuration)
10. [Defense in depth — the patched engine crate](#defense-in-depth--the-patched-engine-crate)
11. [Simard coordination](#simard-coordination)
12. [Configuration reference](#configuration-reference)
13. [Examples](#examples)
14. [Operational runbook — upgrading a live store](#operational-runbook--upgrading-a-live-store)
15. [Compatibility & guarantees](#compatibility--guarantees)
16. [Testing](#testing)

---

## Implementation status

This page is a **specification**: it describes the *target* state of the #107 fix.
The table below is the single source of truth for what is **landed** in
`amplihack-memory-lib` today versus what this design will **add**. When the
sections below use the present tense ("the gate refuses…", "`open_with_recovery`
runs the parity gate…"), they describe **planned** behaviour unless this table
marks it landed.

> **⚠ Root cause corrected (this branch).** The empty read is **not** an lbug
> 0.17.1 engine catalog/CSR bug (the published engine reads the real v40 store
> correctly — `count(n)` returns the full 3242). It is a **lib-side** issue: the
> store's tables **lack the `_deleted` tombstone column**, every labeled read
> appends `WHERE (n._deleted IS NULL OR n._deleted = '')`, lbug 0.17.1 strictly
> binds that to a hard `Binder exception: Cannot find property _deleted`, and the
> read helper swallows the error to an empty `Vec` ⇒ `get_statistics().total = 0`.
> See [Root cause (corrected)](#root-cause-corrected-the-_deleted-tombstone-filter-on-a-legacy-schema).
> **Consequences:** the cross-repo *engine* patch is **not required**; the fix is a
> read-path change in this crate; and the regression guard is a **small synthetic
> legacy-schema** test, not a production-scale / git-LFS fixture (rows marked
> *superseded* below).

| Capability | Status | Where |
| --- | --- | --- |
| `lbug = "=0.17.1"` version pin | ✅ landed | `rust/amplihack-memory/Cargo.toml` |
| #95/#88 crash-recovery state machine (`Clean` / `RecoveredPrefix` / `CheckpointOnly` / `RebuiltAfterCorruption`) | ✅ landed | `graph/lbug_store/mod.rs` |
| #100 migration regression — **10-node** v40 fixture + CI "Data-loss gate" | ✅ landed | `fixtures/v40_store/`, `tests.rs`, `.github/workflows/ci.yml` |
| **`MemoryError::SuspectedDataLoss { footprint_bytes, read_count }`** + display test | ✅ landed (this branch) | `src/errors.rs` |
| **`WalRecoveryOutcome::SuspectedDataLoss`** variant | ✅ landed (this branch) | `graph/lbug_store/mod.rs` |
| **#107 legacy-schema read-parity tests** (`v40_legacy_store_*`, `v40_real_store_parity_*`) — RED→GREEN | ✅ landed (this branch) | `tests.rs` |
| **Empty-read gate-trip test** (`v40_empty_read_gate_trips_and_seals_without_checkpoint`) — RED→GREEN | ✅ landed (this branch) | `tests.rs` |
| **Read-path fix** — labeled reads tolerant of a missing `_deleted` column (schema-gated tombstone filter + `ensure_schema_loaded` on the stats read path) | ✅ landed (this branch) | `graph/lbug_store/{mod,store_impl}.rs` |
| **`CommittedFootprint` probe + empty-read gate** in `open_with_recovery` (read-only peek + sealing) | ✅ landed (this branch) | `graph/lbug_store/mod.rs` |
| **Fail-closed mapping** in `open_persistent` / `open_persistent_with_recovery` | ✅ landed (this branch) | `src/cognitive_memory/mod.rs` |
| ~~Typed `count_all_nodes() -> NodeCount`~~ (**superseded** — the gate treats a `0` count over a material footprint as suspect) | ❌ not needed | — |
| ~~Production-scale v40 fixture~~ (**superseded** — bug is scale-independent; a real-store copy is exercised by the env-gated `v40_real_store_parity_*` test) | ❌ not needed | — |
| **Engine read-path fix** (cross-repo) (**superseded** — engine is not at fault; verified: `count(n)` reads the real store's full 2813) | ❌ not needed | ~~`rysweet/lbug-patched`~~ |
| **Simard** integration contract (re-pin preconditions + `Value::Json` non-breaking) | ✅ verified (this branch, read-only) | `Simard:Cargo.toml`, `Simard:src/cognitive_memory/mod.rs:290` |
| **Simard** 0.17.1 re-pin + safe-update parity gate | 🚧 planned (after PR #108 merges) | Simard checkout |

The implementation order and per-PR scope are in
[PR dependency order](#pr-dependency-order-strict-each-green-ci-merge-ready).

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

## The #107 silent empty-read bug — root cause

> **This is the bug this whole page exists to defend against.** Read it before
> deploying 0.17.1 against any real store.

### Symptom (reproduced on a *copy*, never the live store)

The same single-file v40 cognitive store reads differently under the two engines:

```text
# 0.15.4 (the writer of the store) — CORRECT
simard-0.15.4 memory stats <copy-state-root> --json
  → "access_tier":"direct-open", "total": 2813

# 0.17.1 — WRONG: silently empty, no error, no quarantine artifact
simard-0.17.1 memory stats <copy-state-root> --json
  → "access_tier":"direct-open", "total": 0
```

Both report `access_tier":"direct-open"` — `0.17.1` *opens* the store, claims it
can read v40, and then returns **zero rows**. There is no error, no `*.corrupt-*`
sibling, and no log line: the failure is **completely silent**. A daemon that
trusts this read would run "clean" OODA cycles over an empty graph, checkpoint,
upgrade v40→v41 over the empty store, and **permanently lose 2 800+ items** — and
`0.15.x` can no longer open the resulting v41 file to recover them.

> **Always validate on a copy.** Reproduce by copying a backup to a throwaway
> state-root first — `cp -a ~/.simard/cognitive.v40.bak /tmp/x/cognitive` — and
> running the 0.17.1 binary against `/tmp/x`. **Never** point an unproven 0.17.1
> binary read-write at `~/.simard/cognitive`.

### Why PR #105's fixture missed it

PR #105 added a v40 fixture regression test
(`graph::lbug_store::tests::v40_fixture_opens_clean_and_preserves_all_data`) that
**passes** — but it was generated by the *current* lib, whose `CREATE NODE TABLE`
already includes the reserved `_deleted` tombstone column. The real production
store predates that column (see the root cause below), so the fixture never
reproduces the failing condition. The trigger is **not scale** — it is the
**absence of the `_deleted` column** in the on-disk schema, which reproduces with
a handful of nodes. The guards are therefore the small *synthetic legacy-schema*
tests in [Testing](#testing), not a multi-gigabyte fixture.

### Root cause (corrected): the `_deleted` tombstone filter on a legacy schema

> **⚠ Correction.** An earlier draft of this page (and the #107 design) attributed
> the empty read to the **lbug 0.17.1 engine** mis-reconstructing the v40
> catalog/CSR metadata in its `upgradeLegacyStorageFormat` branches. **Empirical
> investigation on a copy of the real store disproves that.** The published lbug
> 0.17.1 engine reads the real v40 store **correctly**: on a copy,
> `MATCH (n) RETURN count(n)` returns **3242** and per-table `count(n)` sums to the
> exact 0.15.4 figure (`3 + 72 + 1976 + 212 + 9 + 541 = 2813`). The data is fully
> intact on disk. The bug is **lib-side**, not in the engine, and **no engine C++
> patch is required.**

The actual chain, reproduced both on a copy of `~/.simard/cognitive.v40.bak` and
synthetically (`v40_legacy_store_*` in [Testing](#testing)):

1. The real store's node tables were written by an **older lib** whose
   `CREATE NODE TABLE` did **not** include the `_deleted` tombstone column (it was
   added in the #99/#100 soft-delete era). So the on-disk tables **lack
   `_deleted`**.
2. Every *labeled* read in this crate appends the tombstone filter
   `not_deleted("n")` ⇒ `WHERE (n._deleted IS NULL OR n._deleted = '')`
   (`match_return_nodes`, used by `query_nodes` and hence `get_statistics`).
3. **lbug 0.17.1 strictly binds** a labeled `MATCH (n:Label)` against the table's
   schema, so a reference to the absent `_deleted` is a hard error:
   `Binder exception: Cannot find property _deleted for n`. (lbug 0.15.4 tolerated
   the missing property as `NULL`, which is why the *same* store read `2813` there.)
4. The read helper **swallows the error** (`Err(_) => Vec::new()`), so the labeled,
   filtered read returns an **empty `Vec`** with nothing surfaced.
5. `get_statistics()` sums those empty per-type reads ⇒ **`total = 0`** over a
   fully-populated store — the silent empty read. (`count(n)`, which never names
   `_deleted`, still returns the real total, which is why the data is provably
   intact; and `get_node` still works because its cold path uses an *unlabeled*
   `MATCH (n)` that binds `_deleted` leniently.)

The **fix is lib-side**: make the labeled read path tolerant of a store whose
tables lack the `_deleted` column — e.g. only emit the `not_deleted` predicate for
tables that actually carry the column (introspect the schema), and/or back-fill
the column via `ALTER TABLE ... ADD` on open, and/or retry the read without the
filter on a "missing property" binder error. After the fix a 0.17.1 build reads a
**copy** of the real store at `access_tier=direct-open` with full item-count
parity and upgrades it losslessly to v41. See [Testing](#testing) for the tests
that turn this from red to green.

> **The fix may be partial; the gate is not.** Even if some future store has a
> different unreadable shape, the
> [empty-read safety gate](#the-empty-read-safety-gate-item-count-parity) below
> *still* prevents the data loss by refusing to checkpoint a store that reads
> empty. The gate is the non-negotiable guarantee; the read-path fix restores the
> happy path.


---

## The empty-read safety gate (item-count parity)

> **✅ Landed (this branch).** This section describes the gate as implemented in
> `LbugGraphStore::open_with_recovery`. It runs as a **read-only peek** *before*
> any read-write open: a read-write open mutates the file on drop (the engine
> rewrites a checkpoint header even with no writes), so the gate inspects the
> store through a read-only handle that cannot checkpoint, upgrade, rebuild, or
> rewrite a single byte. The `WalRecoveryOutcome::SuspectedDataLoss` outcome, the
> typed `CommittedFootprint` probe, and store sealing are all live.

The gate is **defense-in-depth that does not depend on the engine fix**. It will
live in `LbugGraphStore::open_with_recovery` and run **immediately after a
successful open and node count, before *any* write path** — before auto-checkpoint,
before `checkpoint()`, and before any recovery rebuild. Its job: never let a store
that *should* be populated get checkpointed (and thus upgraded to v41) while it
reads empty.

### How "should be populated" is decided — without trusting the suspect read

The gate cannot use the catalog read as its own baseline (that read is the thing
under suspicion). Instead it compares the freshly-read node count against an
**independent on-disk signal**, the **committed-footprint probe**:

| Probe result | Meaning |
| --- | --- |
| `CommittedFootprint::Empty` | No durable evidence of any committed data: the database file size is at/near the fresh-store baseline (≤ `EMPTY_STORE_FOOTPRINT_THRESHOLD`, **32 KiB**). A legitimately **new or empty** store. |
| `CommittedFootprint::NonEmpty { bytes }` | The on-disk store carries a **material committed footprint** (file size above the 32 KiB threshold — e.g. a ~31 MB single-file store) — it was written with real data by a previous process. |

The probe reads **durable file-level evidence** (database file size above the
fresh-store baseline), **not** the catalog rows that the bug zeroes out — so it
stays trustworthy even when the read path is broken. The 32 KiB threshold is
empirical (lbug 0.17.1, this crate's `SystemConfig`): a checkpointed store
measures ~16 KiB empty (no tables) and ~48 KiB with a single node, so 32 KiB sits
above the empty/cognitive-empty baseline and below any store that has ever
committed a row.

### Node count — empty vs. real, never silently zero

The gate's count comes from `count_all_nodes()`, which sums the **unfiltered**
`count(n)` of every node table (`MATCH (n:Label) RETURN count(n)`). That query
**never names `_deleted`**, so it stays correct even on the #107 legacy schema
(this is *why* the data is provably intact on disk — the catalog rows are all
there; only the labeled, `_deleted`-filtered read path zeroed out). A `count(n)`
that fails to read collapses to `0`, and the gate treats **`0` over a material
footprint** — whether a genuinely-empty catalog or an unreadable one — as a
parity failure: a populated store whose count cannot be confirmed is exactly as
dangerous as one that reads `0`.

> The gate is thus a **backstop for a catalog that reads empty**, not the fix for
> #107 itself: for the actual #107 bug `count_all_nodes()` returns the *real*
> total (2813), so the gate stays `Clean` and the **read-path fix** restores the
> filtered reads. The gate fires only if even the unfiltered count comes back `0`
> over a material file — a future/worse failure mode that must still never be
> checkpointed.

### The decision

| Committed footprint | `count_all_nodes()` | Outcome |
| --- | --- | --- |
| `Empty` | anything | `Clean` — a genuinely fresh/empty store; nothing to protect. |
| `NonEmpty` | `> 0` | `Clean` — a populated store that read back populated. |
| `NonEmpty` | `0` (empty or unreadable) | **`SuspectedDataLoss`** — the gate trips. |

### What the gate does when it trips (`SuspectedDataLoss`)

A store with a material on-disk footprint read back empty. Because the gate runs
as a **read-only peek**, the suspect store was never even opened read-write, so
in order of importance:

- **Writes nothing.** The read-only handle cannot auto-checkpoint, `checkpoint()`,
  rebuild, or rewrite the file on drop. **No v41 write ever happens**, so the
  on-disk v40 bytes are byte-for-byte as they were and a `0.15.x` binary can still
  read the store. This is the property that prevents the permanent loss.
- **Quarantines nothing.** Unlike catalog corruption, the bytes are *fine* — the
  reader is the problem — so there is no `*.corrupt-*` artifact and nothing is
  moved aside.
- **Surfaces the failure loudly and fails closed:**
  - `LbugGraphStore::open_with_recovery` returns
    `Ok((store, WalRecovery { outcome: SuspectedDataLoss, recovered_records: 0,
    quarantined_wal: None }))`, and the returned `store` is **sealed**:
    `checkpoint()` (and the trait method, `do_checkpoint`, auto-checkpoint, and the
    drop checkpoint) are all hard-refused, so a caller cannot accidentally upgrade
    it to v41.
  - `CognitiveMemory::open_persistent` (the recommended entry point) maps that
    outcome to a hard **`MemoryError::SuspectedDataLoss { footprint_bytes,
    read_count }`** and refuses to return a usable handle — so an unsuspecting
    consumer aborts loudly instead of running on an empty graph.

`WalRecovery::recovered()` returns `true` for `SuspectedDataLoss` (it is not
`Clean`). The gate **never deletes or rewrites** the store; recovery from a trip
is an operator action (roll back to the 0.15.x binary, or deploy a 0.17.1 build
that carries the lib read-path fix). See the
[runbook](#operational-runbook--upgrading-a-live-store).

### A note on 2813 vs. 2831 (relative parity, no magic number)

You will see two counts: the task tracked **2831 items**, while
`memory stats --json` reports **`total = 2813`**. They are different metrics —
`2813` is the **node** total (`count(n)` summed across node tables, what the gate
checks), and `2831` includes additional relationship/edge records. The gate and
the parity test therefore assert **relative parity** — the post-open count equals
the count a *trusted* reader reported for the *same* metric — and **never** equal
a hardcoded literal. A magic constant rots as the live store grows; the bug
signature is "non-zero → 0", which relative parity catches robustly at any size.

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
- **A v40 store that *opens* but reads back empty is caught by the parity gate,
  not rebuilt (✅ #107, landed).** This is the #107 mode: the strict open succeeds
  (no error), so the old "rebuild on open failure" logic would not fire — but the
  [empty-read safety gate](#the-empty-read-safety-gate-item-count-parity) will
  compare the committed footprint against the node count and return
  `WalRecoveryOutcome::SuspectedDataLoss` *before any checkpoint or rebuild*. The
  store is **not** rebuilt-from-empty and **not** checkpointed to v41; the v40
  bytes are left intact for rollback. `SuspectedDataLoss` is distinct from both
  `Clean` and `RebuiltAfterCorruption`. (This gate is **landed** — see
  [Implementation status](#implementation-status).)

The practical contract this design delivers: **deploying the 0.17.1 binary against
an existing v40 store either preserves every node and edge, or refuses to upgrade
and preserves the v40 bytes for rollback — it never silently runs on (or
checkpoints) an empty graph.** The production-scale parity fixture and the gate-trip
test below will prove both halves.

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

> **#107 does NOT need a patched engine.** The #107 silent empty-read is a
> **lib-side** bug (the `_deleted` tombstone filter on a legacy schema), fixed in
> this crate's read path — the published crates.io `lbug 0.17.1` engine reads the
> real v40 store correctly (`count(n)` returns the full 2813). The patched crate
> below is **optional** and unrelated to #107; it carries only the **#100 CSR
> crash bounds-check** as belt-and-suspenders defense in depth.

The patched `lbug` 0.17.1 crate carries **one** fix layered on the from-source
engine (`lbug-src/`):

1. the **CSR crash bounds-check** — `getGroup(INVALID_CHUNKED_GROUP_IDX ==
   UINT32_MAX)` is guarded by a `getNumGroups()` bounds-check before the
   dereference in `CSRNodeGroup::scan*` (issue #100 belt-and-suspenders; the
   soft-delete tombstone already keeps `LbugGraphStore` off this path, so this is
   redundant insurance, not a requirement).

The patched crate lives at **`rysweet/lbug-patched`** on the **`v0.17.1-csr-fix`**
branch (the from-source `lbug-src/` C++ engine, force-compiled). The fix commit is
published under an **immutable tag** so a consumer pin is reproducible. It is
**opt-in**, wired only where needed via Cargo's `[patch.crates-io]` to redirect
`lbug` to the patched crate:

```toml
# Consumer Cargo.toml — opt in to the patched 0.17.1 engine (the #100 CSR
# bounds-check defense-in-depth ONLY; #107 is fixed lib-side and needs no patch).
# Overrides lbug for the WHOLE dependency graph, including amplihack-memory.
#
# COMMITTED form — a git URL + immutable tag, so CI can resolve it:
[patch.crates-io]
lbug = { git = "https://github.com/rysweet/lbug-patched", tag = "v0.17.1-csr-fix-pinned" }
```

> **Local-path patches are dev-only.** A `lbug = { path = "/home/azureuser/src/lbug-patched" }`
> override is convenient for local iteration but **must never land in a committed
> `Cargo.toml`** — CI cannot resolve a developer's local path.

`amplihack-memory-lib` itself does **not** add this `[patch.crates-io]` (it stays
on the published crate so it remains publishable from crates.io). Simard **may**
add the `[patch.crates-io] lbug` git+tag pin for the #100 CSR bounds-check, but —
unlike an earlier draft of this page — it is **not** required to fix #107.

---

## Simard coordination

Simard links `lbug` **twice**: directly (its own `lbug` dependency) and
transitively (through its `amplihack-memory` git dependency). Both must move to
0.17.1 together so the final binary links exactly one engine and one storage
format.

> **Strict PR order (see [Testing](#testing) → *PR dependency order*).** The
> engine fix (`lbug-patched`, tagged) must merge first, then the
> `amplihack-memory-lib` PR that pins it, and **only then** the Simard re-pin.
> **The Simard re-pin must NOT deploy 0.17.1 to the live `simard-ooda` daemon**;
> it produces a verified, merge-ready change and stops before the irreversible
> live deploy.

Simard-side changes (separate checkout, lands after the `amplihack-memory-lib` PR
merges). Simard's `Cargo.toml` today has `lbug = "0.15"`, an **unpinned**
`amplihack-memory` git dependency (no `rev`/`tag`), and **no** `[patch.crates-io]`
section:

1. Bump Simard's own `lbug = "0.15"` → `=0.17.1`.
2. Pin the (currently floating) `amplihack-memory` git dependency to the merged
   `amplihack-memory-lib` commit that carries the 0.17.1 pin **and** the
   empty-read safety gate.
3. Add a `[patch.crates-io] lbug` git+tag pin at the **0.17.1 patched crate**
   (gives the #107 read fix and the #100 CSR `getGroup` bounds-check; Simard has
   no `[patch]` section today).
4. No code edits to `src/cognitive_memory/mod.rs` or its `as_str` / `as_i64` /
   `as_f64` helpers — they already use `_ =>` catch-all arms, so the additive
   `Value::Json` variant is non-breaking.

> **✅ Integration contract — verified (this branch).** The four re-pin
> preconditions above were checked against the live Simard checkout
> (`/home/azureuser/src/Simard`, branch
> `engineer/fix-meeting-subprocess-timeout-and-structured-decisions`), **read-only**:
>
> - `Cargo.toml` carries `lbug = "0.15"`, an unpinned `amplihack-memory` git
>   dependency (no `rev`/`tag`), and **no** `[patch]` section — exactly the
>   starting state assumed by steps 1–3.
> - `src/cognitive_memory/mod.rs:290–310` defines `as_str` / `as_i64` / `as_f64`,
>   and **all three end in a `_ => None` catch-all arm**, so the additive
>   `lbug::Value::Json` variant introduced by 0.17.1 cannot break Simard's value
>   extraction (step 4 holds; no Simard code edit is required).
>
> No Simard file was modified and the live `~/.simard/cognitive` store was not
> opened — this is contract verification only. The re-pin itself is still
> **pending**: it lands in the separate Simard checkout **after** PR #108 merges,
> and **must not** deploy 0.17.1 to the live `simard-ooda` daemon.

### Simard safe-update parity gate (the second defense layer)

The lib's [empty-read safety gate](#the-empty-read-safety-gate-item-count-parity)
protects every consumer at store-open time. Simard adds a **second, independent**
parity check in its **safe-update validate phase**, because Simard's existing
validate logic only checks "the daemon runs clean OODA cycles" — which a
*silently empty* store passes. Clean cycles over an empty graph are not evidence
of a healthy upgrade.

The Simard gate:

1. **Records the pre-upgrade item count** with the *current* (0.15.4) binary,
   *before* the binary swap — an independent baseline obtained outside the suspect
   0.17.1 read path.
2. After the 0.17.1 binary opens the store (a **copy** during validation), reads
   the item count and **compares it to the recorded pre-upgrade count**.
3. On any shortfall (the #107 signature, `pre > 0` but `post == 0`, or any
   mismatch) it **fails validation, does not mark the upgrade validated, and rolls
   back** (restore from the pre-upgrade backup). It never checkpoints the copy and
   never advances the live store.

This makes the safe-update path assert **item-count parity**, not merely "no
crashes", so the class of bug in #107 can never be auto-validated.

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

### Inspecting recovery outcome (checking for rebuild / silent empty)

> **✅ Landed.** The `SuspectedDataLoss` / `MemoryError::SuspectedDataLoss` paths
> below are implemented (this branch). All of `Clean` / `RebuiltAfterCorruption` /
> `SuspectedDataLoss` compile and behave as shown (see
> [Implementation status](#implementation-status)).

```rust
use amplihack_memory::graph::{LbugGraphStore, WalRecoveryOutcome};

// Opening a healthy, populated v40 store reports Clean — NOT
// RebuiltAfterCorruption and NOT SuspectedDataLoss.
let (_store, recovery) = LbugGraphStore::open_with_recovery(v40_path, Some("agent"))?;
assert_eq!(recovery.outcome, WalRecoveryOutcome::Clean);
assert_ne!(recovery.outcome, WalRecoveryOutcome::RebuiltAfterCorruption);
assert_ne!(recovery.outcome, WalRecoveryOutcome::SuspectedDataLoss);
# Ok::<(), amplihack_memory::MemoryError>(())
```

### Handling a tripped empty-read gate (#107 fail-closed)

```rust
use amplihack_memory::CognitiveMemory;
use amplihack_memory::MemoryError;

// If a populated v40 store reads back empty (e.g. a future engine that genuinely
// reads the catalog empty), the recommended entry point FAILS CLOSED rather than
// handing back an empty graph. The on-disk v40 bytes are untouched — roll back to
// 0.15.x or deploy a build that reads the store correctly.
match CognitiveMemory::open_persistent("/tmp/copy/cognitive", "agent") {
    Ok(mem) => { /* parity held — safe to use */ let _ = mem; }
    Err(MemoryError::SuspectedDataLoss { footprint_bytes, read_count }) => {
        // footprint says the store holds data, but it read `read_count` (0) nodes.
        // NOTHING was checkpointed or rebuilt; the store is still v40-readable.
        eprintln!(
            "ABORT: store has a {footprint_bytes}-byte footprint but read {read_count} \
             nodes — refusing to upgrade. Roll back or use the patched engine."
        );
    }
    Err(other) => return Err(other),
}
# Ok::<(), amplihack_memory::MemoryError>(())
```

The low-level handle exposes the same signal as a `WalRecoveryOutcome` for tests
and tools:

```rust
use amplihack_memory::graph::{LbugGraphStore, WalRecoveryOutcome};

let (store, recovery) = LbugGraphStore::open_with_recovery(copy_path, Some("agent"))?;
if recovery.outcome == WalRecoveryOutcome::SuspectedDataLoss {
    // The store is SEALED: checkpoint() is a hard error and auto-checkpoint is
    // disabled, so no v41 write can occur. No `*.corrupt-*` artifact is written.
    assert!(store.checkpoint().is_err());
}
# Ok::<(), amplihack_memory::MemoryError>(())
```

### Building with the patched engine for defense in depth

```bash
# Compile the engine (the #107 v40 read fix + the #100 CSR getGroup bounds-check)
# from the patched lbug-src/ source. LBUG_SOURCE_DIR points at the patched engine
# tree (the `v0.17.1-csr-fix` branch of rysweet/lbug-patched).
LBUG_BUILD_FROM_SOURCE=1 \
LBUG_SOURCE_DIR=/path/to/lbug-patched/lbug-src \
  cargo test -p amplihack-memory --features persistent
```

---

## Operational runbook — upgrading a live store

> **Deploy is out of scope for this change.** This runbook documents the safe
> procedure for whoever performs the deploy; the upgrade PRs do **not** restart
> or redeploy the running daemon. **Never** point an unproven 0.17.1 binary
> read-write at the live `~/.simard/cognitive` — validate on a copy first.

1. **Stop the consumer** (so nothing is mid-write against the store).
2. **Back up the v40 store** (`cp -a ~/.simard/cognitive ~/.simard/cognitive.v40.bak`).
   This is the **only** rollback path to a 0.15.x binary, because the first
   0.17.1 checkpoint upgrades the store to v41 (forward-only).
3. **Record the pre-upgrade item count** with the *current* (0.15.4) binary —
   `simard-0.15.4 memory stats <state-root> --json` → note `total`. This is the
   trusted parity baseline (see the [2813 vs 2831 note](#a-note-on-2813-vs-2831-relative-parity-no-magic-number)).
4. **Validate on a COPY first.** `cp -a` the backup to a throwaway state-root and
   open *that* with the 0.17.1 binary. Assert **item-count parity** against the
   step-3 baseline. If the gate trips (`SuspectedDataLoss` /
   `MemoryError::SuspectedDataLoss`) or the count does not match, **stop**: the
   0.17.1 build you are about to deploy does not read this store correctly. Deploy
   a 0.17.1 build carrying the lib read-path fix (this branch) and re-validate.
   **Never** validate against the live store directly.
5. **Deploy the 0.17.1 binary** and start the consumer pointed at the original
   store. It opens v40 in place; the first checkpoint upgrades it to v41.
6. **Confirm** post-start that the item count matches the step-3 baseline and that
   **no** `SuspectedDataLoss`, `RebuiltAfterCorruption`, or quarantine artifact
   (`*.corrupt-*`) was produced.

If step 4 or 6 shows any count regression, a tripped gate, or a quarantine
artifact, **stop and restore the v40 backup** — do not run on, or checkpoint, a
store that read empty or was rebuilt-from-empty.

---

## Compatibility & guarantees

- **No silent data loss (gated).** *Once the gate lands*, an existing, populated v40
  store either opens in place under 0.17.1 with full item-count parity, **or** the
  [empty-read safety gate](#the-empty-read-safety-gate-item-count-parity) trips
  (`SuspectedDataLoss` / `MemoryError::SuspectedDataLoss`) and refuses to
  checkpoint — leaving the v40 bytes untouched for 0.15.x rollback. The store is
  **never** silently run empty, and **never** checkpointed-to-v41 while empty.
- **API additions only.** The new surface this design will add —
  `WalRecoveryOutcome::SuspectedDataLoss`,
  `MemoryError::SuspectedDataLoss { footprint_bytes, read_count }`, and the
  hardened internal `count_all_nodes()` typed result — is **additive**. Existing
  signatures (`open`, `open_with_recovery`, `checkpoint`, `WalRecovery` fields)
  stay unchanged. The crate's own `match` sites on `WalRecoveryOutcome` already
  carry `_ =>` arms; the recommended consumer surface
  (`CognitiveMemory::open_persistent`) returns a `Result`, so the new outcome
  reaches most callers only as the `MemoryError::SuspectedDataLoss` error variant,
  not as a new enum arm they must handle.
- **Crash-fix preserved.** The soft-delete tombstone (issue #100) is retained and
  stays green on 0.17.1; the patched engine adds the #107 read fix and the C++ CSR
  bounds-check in depth.
- **Forward-only storage.** After the first 0.17.1 checkpoint the store is v41 and
  cannot be reopened by 0.15.x; rollback requires the pre-upgrade backup. The gate
  guarantees that first checkpoint never happens over an empty read.
- **MSRV unchanged.** Builds on Rust 1.85.
- **In-memory backend unaffected.** `InMemoryGraphStore` never linked the engine.

---

## Testing

> **✅ Landed.** All of the tests below are in the tree and **GREEN** (this
> branch), and the "Data-loss gate" CI step runs them via the `v40` selector. The
> production-scale git-LFS fixture was **superseded** — the bug is scale-independent,
> and a real-store copy is exercised by the env-gated `v40_real_store_parity_*`
> test when present (see [Implementation status](#implementation-status)).

Behind the `persistent` feature in
`rust/amplihack-memory/src/graph/lbug_store/tests.rs`:

- **✅ Small v40 fixture opens losslessly (the original migration regression).** The
  10-node, single-node-group fixture under `fixtures/v40_store/` — generated
  out-of-band by a 0.15.4 build and committed as frozen test data — is staged into
  a tempdir and opened under the 0.17.1 engine
  (`v40_fixture_opens_clean_and_preserves_all_data`). It asserts `Clean` (not
  `RebuiltAfterCorruption`), node/edge parity, and v41-after-checkpoint
  (`v40_fixture_on_disk_is_v40_and_upgrades_after_checkpoint`). **Note:** this
  fixture is too small to reproduce #107 — it is retained, but the
  production-scale fixture below is what will actually guard against the silent
  empty read.
- **✅ Legacy-schema read parity (the #107 guard) — GREEN.**
  Three tests reproduce the corrected root cause (a node table that **lacks the
  `_deleted` column**) and assert the lib read path returns every item, not a
  silent empty:
  - `v40_legacy_store_missing_tombstone_column_reads_all_items` (store level): a
    synthetic legacy store is seeded via raw Cypher so its tables omit `_deleted`,
    then `query_nodes(type, agent_filter)` must return every seeded row.
  - `v40_legacy_store_cognitive_statistics_reports_full_total` (public-API level):
    reproduces the exact reported symptom — `CognitiveMemory::get_statistics().total
    == 0` over a populated store — and asserts the full total instead.
  - `v40_real_store_parity_preserves_all_items` (empirical, **env-gated** on
    `AMPLIHACK_V40_REAL_STORE_COPY`): opens a *copy* of a real v40 store and asserts
    the lib read path equals a **tolerant** trusted baseline (a per-type `count(n)`
    that never names `_deleted`), so there is **no hardcoded literal** — see the
    [2813 vs 2831 note](#a-note-on-2813-vs-2831-relative-parity-no-magic-number).
    Skipped (passes trivially) when the env var is unset, so CI — which has no real
    store — is unaffected; the two synthetic tests are the deterministic CI guards.
    (Verified manually against a copy of `~/.simard/cognitive.v40.bak`: the lib now
    reads `total = 2813`, matching 0.15.4, where the unfixed path read `0`.)

  All three were **RED before the read-path fix** (the labeled, `_deleted`-filtered
  read binder-errors on the missing column and the lib swallowed it to an empty
  `Vec`, reading `0` / `total 0`) and are **GREEN** now that the read path gates the
  tombstone filter on the actual schema. Because the trigger is the *absent column*,
  not data scale, the fixtures are **small and synthetic** — no
  `fixtures/v40_store_large`, no git-LFS, no real memory content committed.
- **✅ Empty-read gate trips and seals (engine-independent) — GREEN.**
  `v40_empty_read_gate_trips_and_seals_without_checkpoint` seeds a
  populated, checkpointed store, uses the `#[cfg(test)] test_seam` (thread-local, so
  it cannot pollute parallel tests) to force the open path to observe an empty read,
  and asserts the gate returns `WalRecoveryOutcome::SuspectedDataLoss`, that the
  returned store is **sealed** (`checkpoint()` is refused so it can never be upgraded
  to v41 while empty), that **no** `*.corrupt-*` artifact was written, and that the
  on-disk bytes are byte-identical (achieved because the gate peeks **read-only**).
  Holds **regardless of any engine build**.
- **✅ Fresh/empty store is NOT falsely flagged.** A genuinely new/empty store (≤32
  KiB on disk) is classified `CommittedFootprint::Empty`, so the gate is skipped and
  it opens `Clean` — covered by the existing open/round-trip tests, which all stay
  green with the gate in place.
- **✅ CSR crash regression stays green.**
  `reproduces_issue_100_csr_delete_corruption` (the 400-round delete+checkpoint
  churn) passes on 0.17.1, proving the tombstone fix did not regress with the
  engine bump.
- **✅ Soft-delete contract.** `re_add_after_soft_delete_revives_node` and the
  `not_deleted` recall-filter tests stay green.

**CI gate shape (`.github/workflows/ci.yml`).** Today the data-loss proof is a
**dedicated blocking step** that runs the `v40`-prefixed tests — which currently
means the 10-node migration regression. Under this design the parity and gate-trip
tests are named under the `v40` prefix so they are **auto-included** by the
prefix filter. The landed step shape:

```yaml
- name: Data-loss gate (v40 lossless open + #107 empty-read parity/gate)
  working-directory: rust/amplihack-memory
  # Prefix filter — covers every v40_* test (fixtures, legacy-schema parity,
  # gate-trip, and the env-gated real-store parity check):
  run: cargo test --features persistent --lib graph::lbug_store::tests::v40
```

This step is fully deterministic, so it is merge-blocking with no flakiness —
"green CI proving no *silent* data loss" is a real guarantee. The broader
`cargo test --features persistent` job stays `continue-on-error: true` because it
also runs the probabilistic 400-round CSR delete+checkpoint churn
(`reproduces_issue_100_csr_delete_corruption`), a native stress test kept
non-blocking to absorb rare upstream native crashes.

> **The gate-trip test is engine-independent and always blocking.** Even against
> the *unpatched* prebuilt archive, the gate-trip test passes — proving the lib
> refuses to checkpoint an empty read regardless of which engine is linked.

### PR dependency order (each green-CI merge-ready)

> **The engine PR is no longer on the critical path.** Because #107 is fixed
> lib-side (no engine change required), the `lbug-patched` engine PR is **optional**
> (it only adds the #100 CSR bounds-check defense-in-depth) and is **not** a
> prerequisite for the `amplihack-memory-lib` PR to go green.

1. **`amplihack-memory-lib` (this PR)** — the read-path fix, the empty-read safety
   gate + supporting types, the fail-closed mapping, the regression tests, and the
   CI-blocking gate. Builds and passes against the **published** crates.io
   `lbug 0.17.1` (no `[patch.crates-io]`, so the crate stays publishable). Files
   touched:

   | File | Change |
   | --- | --- |
   | `src/graph/lbug_store/store_impl.rs` | Schema-gate the labeled tombstone filter (`match_return_nodes`, `get_node_from_table`, `query_neighbors_one_type`) via `tombstone_filter`; call `ensure_schema_loaded` on the stats read path (`query_nodes`, `search_nodes`). |
   | `src/graph/lbug_store/mod.rs` | Add `WalRecoveryOutcome::SuspectedDataLoss`; add `tombstone_filter`/`table_has_deleted_column`; add `CommittedFootprint` + `committed_footprint` (32 KiB threshold) + the read-only `peek_empty_read_gate` run before any read-write open in `open_with_recovery`; add the `sealed` flag + `seal()` and enforce it in `do_checkpoint` and `Drop`; add `read_only_system_config`/`try_open_database_read_only`. `count_all_nodes()` stays `-> usize` (the gate treats `0` over a material footprint as suspect). |
   | `src/errors.rs` | Add `MemoryError::SuspectedDataLoss { footprint_bytes: u64, read_count: usize }` (with its `#[error(..)]` format string) + a display test. |
   | `src/cognitive_memory/mod.rs` | Map a `SuspectedDataLoss` outcome to `MemoryError::SuspectedDataLoss` in `open_persistent_with_recovery` so the entry point fails closed. |
   | `src/graph/lbug_store/tests.rs` | `v40_legacy_store_*`, `v40_real_store_parity_*`, `v40_empty_read_gate_*` (thread-local seam). |
   | `.github/workflows/ci.yml` | Update the "Data-loss gate" step name/comment to document the #107 coverage (the `…tests::v40` prefix already auto-covers the new `v40_*` tests). |
2. **`rysweet/lbug-patched` (engine fix) — OPTIONAL, not blocking.** The #100 CSR
   `getGroup` bounds-check on `v0.17.1-csr-fix`. Only needed if a deployment wants
   the C++ belt-and-suspenders; the tombstone soft-delete already keeps the lib off
   that path.
3. **Simard** — the 0.17.1 re-pin plus the safe-update parity gate. **Does not
   deploy 0.17.1 to the live daemon** — it stops at a verified, merge-ready change.

```bash
cargo test   -p amplihack-memory --features persistent --lib graph::lbug_store
cargo build  -p amplihack-memory --features persistent
cargo clippy -p amplihack-memory --all-targets --features persistent -- -D warnings
```

> **Pre-commit note.** The configured pre-commit hooks build with default
> features only and do not compile the `persistent` code; the explicit
> `--features persistent` commands above are the authoritative gate for this
> upgrade.
