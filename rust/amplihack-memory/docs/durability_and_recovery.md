# Durability & Recovery — configurable limits and self-healing recovery for `LbugGraphStore`

The persistent backend (`LbugGraphStore`, the LadybugDB-backed `GraphStore` that
powers `CognitiveMemory::open_persistent`) survives a process crash **and** an
on-disk corruption without ever requiring an operator to delete files or restart
a wedged daemon. This page is the complete reference for that durability story:
the configurable buffer-pool / database-size limits, the layered crash-recovery
state machine (clean → corrupt WAL → corrupt catalog), the checkpoint-health
signal, the public API, and an end-to-end operational tutorial.

It is the authoritative description of the fix for issue **#95** ("CRITICAL
durability bug — daemon crash-loops forever after a failed CHECKPOINT corrupts
the catalog") and builds on the corrupt-WAL recovery added for **#88**.

- **No more crash loops.** A corrupt catalog / main database file — the failure
  mode that previously made the store *permanently* unopenable and crash-looped a
  consumer for ~10 hours under systemd — is now quarantined and rebuilt from empty
  automatically. The resilient entry point (`open_with_recovery`) never returns an
  error that a restart cannot clear.
- **Never lose data.** Every corrupt artifact (WAL or whole database) is **moved
  aside, never deleted**, to a timestamped `*.corrupt-<unix_ts>` sibling so the
  incident can be investigated or salvaged offline.
- **Tunable, with safe defaults.** The buffer-pool cap and maximum database size
  are read from environment variables with larger, safer defaults than the old
  hardcoded 128 MiB / 1 GiB — the values whose exhaustion triggered #95.
- **Observable.** The effective limits are logged once at open and exposed via
  getters; a failed checkpoint is recorded and surfaced via
  `last_checkpoint_error()` so a host can report store health without scraping
  logs.
- **Strict mode preserved.** The strict `open()` still errors on corruption (so
  tests and tools that *want* to detect corruption can); only the resilient
  `open_with_recovery()` self-heals.
- **No silent empty reads (issue #107) — _implemented_.** Distinct from
  corruption: an engine that *opens* a populated store but reads it back **empty**
  is caught by an **empty-read safety gate** that runs (as a read-only peek)
  before any write. It returns
  [`WalRecoveryOutcome::SuspectedDataLoss`](#walrecoveryoutcome-reference),
  **refuses to checkpoint or rebuild**, and leaves the on-disk bytes untouched so
  the store stays readable by the writer's engine. This gate **is implemented**
  (see [`docs/lbug_0_17_upgrade.md`](lbug_0_17_upgrade.md#implementation-status));
  for the specific `lbug 0.17.1` v40 silent empty-read, the root cause was a
  **lib-side** tombstone-filter bug fixed in the read path, so the gate is the
  belt-and-suspenders backstop and the read-path fix restores full parity.

> **Feature gate.** Everything on this page requires the `persistent` cargo
> feature (which pulls in the `lbug` engine):
> `cargo build --features persistent` / `cargo test --features persistent`.

---

## Table of contents

1. [Why this exists (the #95 incident)](#why-this-exists-the-95-incident)
2. [Configuration — buffer pool & max DB size](#configuration--buffer-pool--max-db-size)
3. [The recovery state machine](#the-recovery-state-machine)
4. [The empty-read parity gate (#107)](#the-empty-read-parity-gate-107)
5. [`WalRecoveryOutcome` reference](#walrecoveryoutcome-reference)
6. [Quarantine semantics](#quarantine-semantics)
7. [Checkpoint health signal](#checkpoint-health-signal)
8. [Public API reference](#public-api-reference)
9. [Tutorial — operating a resilient persistent store](#tutorial--operating-a-resilient-persistent-store)
10. [Operational runbook](#operational-runbook)
11. [Compatibility & guarantees](#compatibility--guarantees)

---

## Why this exists (the #95 incident)

A live consumer's daemon crash-looped for ~10 hours. The chain of events:

1. `system_config()` hardcoded `buffer_pool_size(128 MiB)` and
   `max_db_size(1 GiB)`. On a host with hundreds of GiB free, **128 MiB was far
   too small**.
2. During an auto-`CHECKPOINT`, LadybugDB's buffer pool was exhausted:
   `Buffer manager exception: Unable to allocate memory! The buffer pool is full
   and no memory could be freed`.
3. The **failed checkpoint left the catalog corrupt**. Every subsequent open
   failed with `Load table failed: table 0 doesn't exist in catalog`.
4. The recovery path only understood a corrupt **WAL** (quarantine → replay good
   prefix → checkpoint-only fallback). It did **not** understand a corrupt
   **catalog / main DB file**, so the checkpoint-only fallback also failed and the
   error propagated.
5. The consumer `exit(1)`'d → systemd restarted it → it failed to open the same
   corrupt catalog → `exit(1)` → forever.

The fix has three independent parts, each of which alone would have broken the
loop, and which together make the store robust:

| Part | What it prevents | Section |
| --- | --- | --- |
| Configurable, larger limits | the **trigger** (buffer-pool exhaustion) | [Configuration](#configuration--buffer-pool--max-db-size) |
| Catalog / main-DB corruption recovery | the **permanent crash loop** | [Recovery state machine](#the-recovery-state-machine) |
| Checkpoint-failure signal | **silent** degradation before disaster | [Checkpoint health signal](#checkpoint-health-signal) |

---

## Configuration — buffer pool & max DB size

Two environment variables control the LadybugDB limits. Both take a **byte
count** (a base-10 unsigned integer, no suffix).

| Variable | Controls | Default | Minimum (clamp) |
| --- | --- | --- | --- |
| `AMPLIHACK_MEMORY_BUFFER_POOL_BYTES` | buffer-pool size cap | **1 GiB** (`1073741824`) | 64 MiB (`67108864`) |
| `AMPLIHACK_MEMORY_MAX_DB_BYTES` | maximum database size | **16 GiB** (`17179869184`) | 1 GiB (`1073741824`) |

The constants are exported for programmatic use:

```rust
use amplihack_memory::graph::lbug_store::{
    ENV_BUFFER_POOL_BYTES, // "AMPLIHACK_MEMORY_BUFFER_POOL_BYTES"
    ENV_MAX_DB_BYTES,      // "AMPLIHACK_MEMORY_MAX_DB_BYTES"
    AUTO_CHECKPOINT_WRITES, // 128 — auto-checkpoint cadence
};
```

### Why the larger defaults are free

- **Buffer pool is allocated lazily.** lbug treats `buffer_pool_size` as a
  *ceiling*, not an upfront allocation. Raising the cap from 128 MiB to 1 GiB
  does not reserve 1 GiB of RAM; it just lets the engine grow the pool when it
  actually needs to (e.g. during a large checkpoint) instead of failing. This is
  exactly what was missing in #95.
- **`max_db_size` is only an mmap reservation.** It reserves *address space*, not
  physical pages, so a 16 GiB cap costs nothing until the database genuinely
  grows. The previous 1 GiB cap was an arbitrary low ceiling on how large a
  memory store could ever become.

### Resolution rules

The env value → effective bytes mapping is deterministic and total (it never
panics, and an invalid override degrades to the default rather than erroring):

1. **Missing / empty / unparseable / `0` / negative** → the **default**.
2. **Valid positive integer below the minimum** → **clamped up** to the minimum
   (an absurdly small override can never starve the engine).
3. **Valid positive integer at or above the minimum** → used as-is.
4. **Cross-invariant:** the effective buffer pool is finally clamped so that
   `buffer_pool_size <= max_db_size` (a pool larger than the entire database is
   meaningless). If you set a buffer pool larger than the max DB size, the buffer
   pool is reduced to the max DB size.

The resolution logic is factored into pure, independently testable functions
(internal to the crate, `pub(crate)`):

- `resolve_buffer_pool_bytes(env: Option<&str>) -> u64`
- `resolve_max_db_bytes(env: Option<&str>) -> u64`
- `effective_limits(buffer_env, max_db_env) -> (u64, u64)` — applies both
  resolvers and the `buffer_pool <= max_db` invariant.

> These take the *value*, not the environment, so they can be unit-tested without
> mutating global process state. Consumers should read the resolved values back
> from a live store via [`buffer_pool_bytes()` / `max_db_bytes()`](#checkpoint-health-signal)
> rather than re-deriving them.

### Effective-limits log line

The effective limits are emitted **exactly once per process** (recovery re-opens
call the config builder repeatedly; the log is `Once`-guarded so it never spams):

```text
INFO lbug_store: effective LadybugDB limits (override via
     AMPLIHACK_MEMORY_BUFFER_POOL_BYTES / AMPLIHACK_MEMORY_MAX_DB_BYTES)
     buffer_pool_bytes=1073741824 max_db_bytes=17179869184
```

### Examples

```bash
# Busy host: give the buffer pool more headroom so a large checkpoint
# cannot exhaust it (the #95 trigger). 4 GiB pool, 64 GiB max DB.
export AMPLIHACK_MEMORY_BUFFER_POOL_BYTES=4294967296
export AMPLIHACK_MEMORY_MAX_DB_BYTES=68719476736

# Tiny edge device: an over-small override is clamped UP to the 64 MiB floor.
export AMPLIHACK_MEMORY_BUFFER_POOL_BYTES=1048576   # 1 MiB requested -> 64 MiB used

# Typo / garbage: silently falls back to the 1 GiB default (no crash, no error).
export AMPLIHACK_MEMORY_BUFFER_POOL_BYTES=lots-of-ram
```

`auto_checkpoint(true)` and the strict/resilient `throw_on_wal_replay_failure`
plumbing are unchanged by this configuration — the limits are orthogonal to the
WAL-replay strictness selected by `open()` vs `open_with_recovery()`.

---

## The recovery state machine

There are two entry points, and they differ **only** in how they react to a
failed open:

| Entry point | On clean open | On corrupt WAL | On corrupt catalog / main DB |
| --- | --- | --- | --- |
| `LbugGraphStore::open()` (strict) | opens | **errors** | **errors** |
| `LbugGraphStore::open_with_recovery()` (resilient) | opens (`Clean`) | recovers | **rebuilds** (self-heals) |

`CognitiveMemory::open_persistent()` delegates to the resilient path, so every
consumer gains self-healing automatically.

`open_with_recovery()` walks this state machine, stopping at the first step that
yields a working database:

```
open_with_recovery(db_path)
│
├─ 0. EMPTY-READ GATE  [#107, landed]  — read-only peek, before any read-write open
│      (committed footprint vs unfiltered node count; a read-only open cannot mutate the file)
│      ├─ footprint Empty ...........................► proceed to step 1 (fresh/empty store)
│      ├─ footprint NonEmpty & count > 0 ............► proceed to step 1 (populated, reads populated)
│      └─ footprint NonEmpty & count 0/unreadable ..► SuspectedDataLoss
│            (return a sealed read-only handle: no checkpoint, no rebuild, bytes untouched)
│
├─ 1. strict open
│      └─ Ok ──► Clean
│
├─ 2. strict open failed
│      ├─ WAL file present?
│      │     ├─ no  ─► main DB / catalog is corrupt with no WAL to blame
│      │     │         └─ rebuild_after_corruption ─► RebuiltAfterCorruption
│      │     └─ yes ─► continue to WAL recovery
│      │
│      ├─ 3. copy WAL aside (preserve the incident artifact first)
│      │
│      ├─ 4. resilient open (replay good prefix, ignore unreplayable tail)
│      │     └─ Ok ─► checkpoint the prefix into the main DB ─► RecoveredPrefix
│      │
│      ├─ 5. resilient open failed -> move WAL fully aside, retry strict
│      │     └─ Ok ─► open from last good checkpoint only ─► CheckpointOnly
│      │
│      └─ 6. STILL failing with the WAL gone -> the catalog itself is corrupt
│            └─ rebuild_after_corruption ─► RebuiltAfterCorruption
│
└─ only returns Err if even a FRESH empty database cannot be created
   (e.g. the parent directory is unwritable)
```

The crucial #95 change is the two paths into **`rebuild_after_corruption`**
(steps 2-no-WAL and 6): previously the no-WAL branch returned `Err` unchanged and
step 6 had no fallback, so a corrupt catalog propagated an error and crash-looped
the consumer. Now both route through quarantine-and-rebuild, so a corrupt catalog
self-heals whether or not a WAL was present.

The **#107** change (**landed**, step 0 in the diagram above) is the **empty-read
gate run as a read-only peek *before* the read-write open**. A silent empty read
is *not* an open failure — the engine returns `Ok` — so none of the corruption
branches would fire. The gate intercepts it **before any write** and, on a
populated-but-empty read, returns `SuspectedDataLoss` instead of `Clean`. It runs
read-only precisely so the inspection itself cannot mutate the store (a read-write
open rewrites a checkpoint header on drop even with no writes); it **never** routes
into `rebuild_after_corruption`, since rebuilding (or checkpointing) would write
v41 over a store whose bytes are actually fine, destroying the rollback path. See
[The empty-read parity gate (#107)](#the-empty-read-parity-gate-107).

`Clean`, `RecoveredPrefix`, and `CheckpointOnly` behave exactly as they did
before. The behavioural changes are: the previously-erroring catalog cases now
produce `RebuiltAfterCorruption` (#95, **landed**), and a populated store that
reads empty produces `SuspectedDataLoss` (#107, **landed**) instead of a silent
`Clean`.

### Detection: how "catalog corruption" is recognised

The store does not parse engine error strings. Corruption is detected
structurally: **the main database still fails to open after the WAL has been
fully moved aside** (or there was never a WAL to blame). At that point WAL replay
cannot be the cause, so the failure is attributed to the catalog / main file and
the rebuild path runs. This deliberately favours *availability*: on the resilient
entry point any persistent open failure that survives WAL quarantine triggers a
rebuild, because a running store with an empty graph is strictly better than a
daemon that `exit(1)`s forever.

> **The empty-read gate is the deliberate exception to "availability over data".**
> Everywhere else this page favours a running-but-empty store over a crash loop.
> The parity gate inverts that *only* for the silent-empty-read case, because
> there the bytes are intact and "run empty + checkpoint" would convert a
> recoverable situation into permanent loss. Fail-closed is correct precisely
> because rollback is still possible.

---

## The empty-read parity gate (#107)

> **✅ Landed.** The gate, the `WalRecoveryOutcome::SuspectedDataLoss` outcome, the
> `MemoryError::SuspectedDataLoss` error, and the `CommittedFootprint` probe
> described in this section are **implemented** (the node count uses the existing
> unfiltered `count_all_nodes() -> usize`; a `0` over a material footprint is
> treated as suspect). See
> [Implementation status](lbug_0_17_upgrade.md#implementation-status).

The parity gate defends against an engine that **opens a store successfully but
reads it back empty** — concretely, `lbug 0.17.1` reading certain real v40
cognitive stores as `total = 0` with no error and no quarantine artifact. Left
unchecked, the store would be checkpointed (upgrading v40→v41 over an empty graph)
and the data would be **permanently lost**. The gate will make this **fail-closed**.

### Where it runs

Inside `open_with_recovery`, **immediately after a successful strict open and node
count, before any write path** — before auto-checkpoint, before an explicit
`checkpoint()`, and before any `rebuild_after_corruption`. No v41 write can occur
until the gate passes.

### The two independent signals it compares

The gate cannot trust the catalog read as its own baseline (that read is the
suspect). It will compare two **independent** signals:

1. **Committed-footprint probe** — durable, file-level evidence of whether the
   store *should* hold data, read **without** going through the catalog rows that
   the bug zeroes out:

   | `CommittedFootprint` | Meaning |
   | --- | --- |
   | `Empty` | No durable evidence of committed data (file at/below the fresh-store baseline). A legitimately new/empty store. |
   | `NonEmpty { bytes, .. }` | A material on-disk footprint (e.g. a ~31 MB single-file store) written by a previous process. |

2. **Node count** — `count_all_nodes()` sums the **unfiltered** `count(n)` of every
   node table. That query never names `_deleted`, so it stays correct even on the
   #107 legacy schema; a query that fails collapses to `0`, and the gate treats
   **`0` over a material footprint** (genuinely-empty *or* unreadable) as a parity
   failure — a populated store whose count cannot be confirmed is as dangerous as
   one that reads `0`.

   > A typed `NodeCount` was considered but **not** needed: the footprint probe
   > already distinguishes "fresh/empty" (which never trips) from "material"
   > (where any `0` trips), so `count_all_nodes() -> usize` is sufficient.

### The decision and its effect

| Footprint | Count | Result |
| --- | --- | --- |
| `Empty` | any | `Clean` — fresh/empty store, nothing to protect. |
| `NonEmpty` | `Counted(n>0)` | `Clean` — populated store read back populated. |
| `NonEmpty` | `Counted(0)` / `ConfirmedEmpty` / `Unreadable` | **`SuspectedDataLoss`** — the gate trips. |

On `SuspectedDataLoss` the gate **fails closed**:

- **Writes nothing** — no auto-checkpoint, no `checkpoint()`, no
  `rebuild_after_corruption`. The on-disk v40 bytes are exactly as opened, so a
  0.15.x binary can still read them. *This is the property that prevents the loss.*
- **Quarantines nothing** — the bytes are fine; the reader is the problem. No
  `*.corrupt-*` artifact is produced and nothing is moved aside.
- **Returns a sealed store** — `open_with_recovery` returns
  `Ok((store, WalRecovery { outcome: SuspectedDataLoss, recovered_records: 0,
  quarantined_wal: None }))`, and the returned `store` refuses to write:
  `checkpoint()` is a hard error and auto-checkpoint is disabled.
- **Fails closed at the consumer entry point** —
  `CognitiveMemory::open_persistent` maps the outcome to
  `MemoryError::SuspectedDataLoss { footprint_bytes, read_count }` and never
  returns a usable handle.

The gate emits a single `error!` log (naming the footprint and the read count) and
emits **no node properties** — no memory content is logged.

> **No false positives on fresh stores.** A genuinely new/empty store probes
> `CommittedFootprint::Empty` and therefore opens `Clean`; the gate only trips when
> a *materially populated* footprint reads back empty.

---

## `WalRecoveryOutcome` reference

`open_with_recovery()` returns `(LbugGraphStore, WalRecovery)`. The
`WalRecovery.outcome` is one of:

| Variant | Meaning | `recovered_records` | `quarantined_wal` |
| --- | --- | --- | --- |
| `Clean` | WAL replayed cleanly (or no recovery needed) **and the #107 empty-read gate passed**. No artifact written, no warning. | `0` | `None` |
| `RecoveredPrefix` | A corrupt WAL tail was quarantined; the good prefix was replayed and checkpointed into the main DB. | nodes after replay | `Some(<wal>.corrupt-<ts>)` |
| `CheckpointOnly` | The WAL was unusable even in resilient mode; it was quarantined and the store opened from the last good checkpoint only. | nodes at last checkpoint | `Some(<wal>.corrupt-<ts>)` |
| `RebuiltAfterCorruption` | The catalog / main DB itself was corrupt and unopenable even with the WAL gone (the #95 mode). The whole database was quarantined and a **fresh empty** database opened. | **`0`** | `Some(<db_path>.corrupt-<ts>)` |
| `SuspectedDataLoss` *(#107, landed)* | A materially-populated on-disk footprint read back **empty** (caught by the read-only peek). The store is **sealed** (refuses to checkpoint), **nothing is written or quarantined**, and the bytes are left intact for rollback. | **`0`** | `None` |

All five variants are implemented today.
`WalRecovery::recovered()` returns `true` for every variant except `Clean`
(including `SuspectedDataLoss`).

```rust
use amplihack_memory::graph::{LbugGraphStore, WalRecoveryOutcome};

let (store, recovery) = LbugGraphStore::open_with_recovery(path, Some("agent-7"))?;
match recovery.outcome {
    WalRecoveryOutcome::Clean => { /* normal startup */ }
    WalRecoveryOutcome::RecoveredPrefix | WalRecoveryOutcome::CheckpointOnly => {
        tracing::warn!(
            recovered = recovery.recovered_records,
            artifact = ?recovery.quarantined_wal,
            "recovered from a corrupt WAL",
        );
    }
    WalRecoveryOutcome::RebuiltAfterCorruption => {
        // Data could not be salvaged in-process. The corrupt DB is preserved at
        // recovery.quarantined_wal for offline forensics; the live store is empty.
        tracing::error!(
            quarantined_db = ?recovery.quarantined_wal,
            "catalog was corrupt; rebuilt from empty — investigate the quarantine",
        );
    }
    WalRecoveryOutcome::SuspectedDataLoss => {
        // A populated store read back EMPTY (#107). NOTHING was written or
        // quarantined; the on-disk bytes are intact and still readable by the
        // writer's engine. The `store` handle is sealed: checkpoint() will error.
        // Do NOT proceed to write — roll back, or deploy an engine that reads the
        // store correctly. (CognitiveMemory::open_persistent turns this into a
        // hard MemoryError::SuspectedDataLoss instead.)
        tracing::error!(
            "store opened but read EMPTY despite a populated footprint — refusing \
             to checkpoint; the v40 bytes are preserved for rollback",
        );
    }
}
```

> **Field naming note.** The path field is historically named `quarantined_wal`
> for backward compatibility. For `RebuiltAfterCorruption` it holds the
> moved-aside **database** path (`<db_path>.corrupt-<ts>`), not a WAL. For
> `SuspectedDataLoss` it is `None` (nothing is quarantined). The field always
> means "where the corrupt artifact was moved aside", or `None` when none was.

---

## Quarantine semantics

Both recovery families preserve the incident artifact before touching the live
location. **Nothing is ever deleted.**

- **Naming.** `<original_path>.corrupt-<unix_ts>`, where `<unix_ts>` is a
  high-resolution UNIX-epoch timestamp (nanoseconds, via `as_nanos()`), which
  makes a collision between two recoveries vanishingly unlikely — the only
  theoretical clash is two rebuilds of the same path within the same nanosecond.
  A corrupt WAL becomes `cognitive.ladybug.wal.corrupt-…`; a corrupt database
  becomes `cognitive.ladybug.corrupt-…`.
- **Move, then rebuild.** For a corrupt catalog the database is moved aside with
  an atomic `rename` where possible. The rename frees `db_path` so a fresh, empty
  database can be created there.
- **Copy-then-clear fallback.** If a rename is risky (e.g. the quarantine target
  is on a different filesystem), the artifact is **copied** to the quarantine
  path first and only then is the original location cleared — the data is always
  preserved at the quarantine location *before* the live path is reused. If even
  the copy fails, the original is left untouched and an `Err` is returned rather
  than risking data loss.
- **Directory or file.** `db_path` is normally a single file, but quarantine uses
  filesystem `rename`, which also relocates a directory, so a directory-form
  database is handled defensively.
- **Sidecars.** When the WAL is moved fully aside (the `CheckpointOnly` / rebuild
  path), any stale WAL sidecars are moved with it so the engine reopens with no
  WAL to replay.

You are responsible for pruning old `*.corrupt-*` artifacts; the store never
deletes them (see the [runbook](#operational-runbook)).

---

## Checkpoint health signal

Auto-checkpointing is best-effort: a failed checkpoint does **not** fail the
write that triggered it (the write already succeeded and was `fsync`'d), and the
store retries on the *next* write rather than on every write. But a failing
checkpoint is the early warning that preceded #95, so it is now surfaced loudly
and queryably.

- **Loud log.** A failed auto-checkpoint logs at `warn!` and explicitly calls out
  that it *can indicate buffer-pool exhaustion*, naming the override env var:

  ```text
  WARN lbug_store: auto-checkpoint failed — this can indicate buffer-pool
       exhaustion (raise the cap via AMPLIHACK_MEMORY_BUFFER_POOL_BYTES);
       recorded as last_checkpoint_error, will retry on the next write
       error=Buffer manager exception: ... buffer pool is full ...
  ```

- **Queryable health.** The last checkpoint error is recorded on the store and
  exposed via `last_checkpoint_error() -> Option<String>`:
  - Set to `Some(message)` when a checkpoint fails.
  - Cleared back to `None` on the next **successful** checkpoint (health
    recovers automatically).

  A consumer can poll this to drive a readiness/health endpoint:

  ```rust
  if let Some(err) = store.last_checkpoint_error() {
      // Surface degraded durability; consider raising the buffer-pool cap.
      health.set_degraded(format!("checkpoint failing: {err}"));
  }
  ```

- **Effective config readback.** `buffer_pool_bytes()` and `max_db_bytes()`
  report the limits *this* store actually opened with (after env overrides and
  clamping), so a health endpoint can show "running with a 1 GiB pool" without
  re-reading the environment.

---

## Public API reference

All items below are gated behind the `persistent` feature.

### Constants — `amplihack_memory::graph::lbug_store`

| Item | Type | Value | Meaning |
| --- | --- | --- | --- |
| `ENV_BUFFER_POOL_BYTES` | `&str` | `"AMPLIHACK_MEMORY_BUFFER_POOL_BYTES"` | env var name for the buffer-pool cap |
| `ENV_MAX_DB_BYTES` | `&str` | `"AMPLIHACK_MEMORY_MAX_DB_BYTES"` | env var name for the max DB size |
| `AUTO_CHECKPOINT_WRITES` | `u64` | `128` | mutating ops between auto-checkpoints |

> Defaults and minimums (1 GiB / 16 GiB / 64 MiB / 1 GiB) and the
> `resolve_*` / `effective_limits` helpers are crate-internal (`pub(crate)`)
> implementation details; read effective values back via the getters below.

### Types — `amplihack_memory::graph`

`pub use graph::{LbugGraphStore, WalRecovery, WalRecoveryOutcome};`

> These are re-exported under `amplihack_memory::graph` (not at the crate root —
> unlike `CognitiveMemory`). Import them as `amplihack_memory::graph::LbugGraphStore`,
> as the examples on this page do; `amplihack_memory::LbugGraphStore` is *not* a
> valid path.

#### `enum WalRecoveryOutcome`

`Clean | RecoveredPrefix | CheckpointOnly | RebuiltAfterCorruption |
SuspectedDataLoss` (the #107 variant, **landed**). See the
[outcome reference](#walrecoveryoutcome-reference). Derives
`Debug, Clone, Copy, PartialEq, Eq`; `SuspectedDataLoss` was additive — match
sites in this crate already carry `_ =>` arms.

#### `struct WalRecovery`

| Field / method | Type | Description |
| --- | --- | --- |
| `outcome` | `WalRecoveryOutcome` | how the store was opened |
| `recovered_records` | `usize` | graph nodes present after recovery (`0` for `Clean`, `RebuiltAfterCorruption`, and `SuspectedDataLoss`) |
| `quarantined_wal` | `Option<PathBuf>` | where the corrupt artifact (WAL or DB) was moved aside (`None` for `Clean` and `SuspectedDataLoss`) |
| `recovered()` | `fn(&self) -> bool` | `true` for any non-`Clean` outcome |

Derives `Debug, Clone`.

#### `enum MemoryError` (the `SuspectedDataLoss` variant)

**Landed (#107) in `src/errors.rs`.** `MemoryError` (re-exported at the crate
root, `amplihack_memory::MemoryError`) carries a fail-closed **struct** variant
returned by `CognitiveMemory::open_persistent` when the
[empty-read gate](#the-empty-read-parity-gate-107) trips:

```rust
MemoryError::SuspectedDataLoss {
    /// On-disk committed footprint (bytes) the store was found to hold.
    footprint_bytes: u64,
    /// Node count the (suspect) read path returned — the value the gate rejected.
    read_count: usize,
}
```

`open_persistent` returns this instead of a usable handle, so a consumer cannot
accidentally run on (or checkpoint) a store that read empty.

### `impl LbugGraphStore`

| Method | Signature | Description |
| --- | --- | --- |
| `open` | `fn open(db_path: &Path, store_id: Option<&str>) -> Result<Self>` | **Strict** open. Errors on a corrupt WAL **or** corrupt catalog. Never rebuilds, never runs the empty-read gate. |
| `open_with_recovery` | `fn open_with_recovery(db_path: &Path, store_id: Option<&str>) -> Result<(Self, WalRecovery)>` | **Resilient** open. Runs the **empty-read gate** (read-only peek) *first* — returning `SuspectedDataLoss` + a sealed store if a populated store reads empty — then recovers a corrupt WAL and **rebuilds** on catalog corruption. Only errors if a fresh DB cannot be created. |
| `checkpoint` | `fn checkpoint(&self) -> Result<()>` | Force a `CHECKPOINT` + durability barrier. A **hard error on a `SuspectedDataLoss`-sealed store** (refuses to upgrade a store that read empty). |
| `buffer_pool_bytes` | `fn buffer_pool_bytes(&self) -> u64` | Effective buffer-pool cap this store opened with. |
| `max_db_bytes` | `fn max_db_bytes(&self) -> u64` | Effective max DB size this store opened with. |
| `last_checkpoint_error` | `fn last_checkpoint_error(&self) -> Option<String>` | Most recent checkpoint error, or `None` after a successful checkpoint. |
| `db_path` | `fn db_path(&self) -> &Path` | The on-disk database path. |

> **Internal (`pub(crate)`) helpers behind the gate.** `count_all_nodes() -> usize`
> sums the unfiltered `count(n)`; `committed_footprint(db_path) -> CommittedFootprint`
> (`Empty` / `NonEmpty { bytes }`) is the independent on-disk probe; `tombstone_filter`
> gates the `_deleted` predicate on the actual schema; `seal()` marks a store
> checkpoint-refusing. All are crate-internal; callers observe the gate only through
> `WalRecoveryOutcome::SuspectedDataLoss` / `MemoryError::SuspectedDataLoss`.

### `impl CognitiveMemory` (the recommended entry point)

| Method | Signature | Description |
| --- | --- | --- |
| `open_persistent` | `fn open_persistent(path, agent_name) -> Result<Self>` | Opens the LadybugDB-backed store via `open_persistent_with_recovery` (self-healing). **Fails closed with `MemoryError::SuspectedDataLoss` when the empty-read gate trips.** |
| `open_persistent_with_recovery` | `fn open_persistent_with_recovery(path, agent_name) -> Result<Self>` | Same as above; explicit name. Logs a structured `warn!`/`error!` whenever recovery (any non-`Clean` outcome) ran, and converts a `SuspectedDataLoss` outcome into the hard error. |
| `checkpoint` | `fn checkpoint(&self) -> Result<()>` | Forces the backend WAL into the main DB (no-op for the in-memory backend). |

`open_persistent` returns `MemoryError::Storage` only if the database cannot be
opened **even after** quarantining a corrupt WAL/catalog and creating a fresh one
(e.g. an unwritable parent directory) — not for ordinary corruption, which now
self-heals. It returns `MemoryError::SuspectedDataLoss` when a populated store
reads empty — a deliberate, fail-closed refusal, **not** a self-heal.

---

## Tutorial — operating a resilient persistent store

A complete lifecycle: open with recovery, inspect health, write, checkpoint,
crash, reopen.

```rust
use amplihack_memory::graph::{LbugGraphStore, WalRecoveryOutcome};
use std::path::Path;

fn run(db_path: &Path) -> amplihack_memory::Result<()> {
    // 1. Open resiliently. This NEVER crash-loops: a corrupt WAL is recovered
    //    and a corrupt catalog is rebuilt from empty.
    let (store, recovery) = LbugGraphStore::open_with_recovery(db_path, Some("agent-7"))?;

    // 2. Report durability posture at startup.
    tracing::info!(
        buffer_pool = store.buffer_pool_bytes(),
        max_db = store.max_db_bytes(),
        "persistent store open",
    );
    if recovery.recovered() {
        tracing::warn!(
            outcome = ?recovery.outcome,
            recovered = recovery.recovered_records,
            artifact = ?recovery.quarantined_wal,
            "store recovered on open",
        );
    }
    if recovery.outcome == WalRecoveryOutcome::RebuiltAfterCorruption {
        // The previous database was unsalvageable in-process and was preserved
        // at recovery.quarantined_wal. Trigger any rehydration you need here.
    }
    if recovery.outcome == WalRecoveryOutcome::SuspectedDataLoss {
        // #107: a populated store read back EMPTY. The store is SEALED — calling
        // store.checkpoint() below would return Err — and the v40 bytes are
        // intact. Do NOT write. Abort and roll back / deploy a correct engine.
        // (Most callers use CognitiveMemory, which turns this into a hard error.)
        return Err(amplihack_memory::MemoryError::Storage(
            "store read empty despite a populated footprint; refusing to write".into(),
        ));
    }

    // 3. ... writes happen via the GraphStore / CognitiveMemory API ...
    //    The store auto-checkpoints every AUTO_CHECKPOINT_WRITES (128) writes.

    // 4. Monitor checkpoint health (e.g. from a readiness probe).
    if let Some(err) = store.last_checkpoint_error() {
        tracing::warn!(%err, "checkpoint failing — consider raising the buffer-pool cap");
    }

    // 5. Force durability at a known-safe point.
    store.checkpoint()?;

    Ok(())
}
```

For most callers, prefer the `CognitiveMemory` wrapper, which logs recovery for
you:

```rust
use amplihack_memory::CognitiveMemory;

// Self-healing open: recovers a corrupt WAL, rebuilds a corrupt catalog, and
// raises the buffer-pool / max-DB limits from the AMPLIHACK_MEMORY_* env vars.
let mut mem = CognitiveMemory::open_persistent("./data/cognitive.ladybug", "agent-7")?;

// ... use mem exactly as the in-memory backend ...

mem.checkpoint()?; // bound crash loss
drop(mem);         // also checkpoints on drop
```

### Reproducing catalog-corruption recovery (what the tests assert)

The behaviour is covered by tests under `--features persistent`:

1. Open a store, write nodes, drop the handle (clean close).
2. **Corrupt the main database file** on disk — truncate it or overwrite the
   header bytes with garbage so a reopen fails with
   `table 0 doesn't exist in catalog`.
3. Call `open_with_recovery`. Assert:
   - it returns `Ok` with `outcome == RebuiltAfterCorruption` and
     `recovered_records == 0`;
   - a `*.corrupt-*` sibling of the database path now exists (the quarantined
     corrupt file);
   - the returned store is **empty and writable** (a fresh database).
4. Both the **WAL-present** and **no-WAL** cases self-heal identically (the
   no-WAL case previously returned an error).

The WAL-recovery tests that are directly asserted — `Clean`
(`open_with_recovery_is_a_noop_on_a_clean_store`) and `RecoveredPrefix`
(`open_with_recovery_survives_corrupt_wal_and_returns_checkpointed_records`) —
and the resolver unit tests (override / default / unparseable→default /
below-min→clamp, plus the `buffer_pool <= max_db` invariant) all remain green.
`CheckpointOnly` is the hard-to-reach fallback where resilient replay *itself*
fails; it has no dedicated test and is currently exercised only indirectly. The
checkpoint-health getters (`last_checkpoint_error()`, `buffer_pool_bytes()`,
`max_db_bytes()`) are likewise not yet covered by a direct test.

### Reproducing the empty-read gate (#107, the landed tests)

> **✅ Landed.** These tests are in the tree and **GREEN**, `v40`-prefixed under
> `--features persistent` so they join the existing
> [CI data-loss gate](lbug_0_17_upgrade.md#testing):

- **Legacy-schema read parity** (`v40_legacy_store_missing_tombstone_column_reads_all_items`,
  `v40_legacy_store_cognitive_statistics_reports_full_total`) — a store whose tables
  lack `_deleted` must read back every item, not a silent empty. These reproduce the
  actual #107 root cause and pass with the read-path fix.
- **Real-store parity** (`v40_real_store_parity_preserves_all_items`) — env-gated on
  `AMPLIHACK_V40_REAL_STORE_COPY`; opens a *copy* of a real v40 store and asserts the
  read count equals a tolerant trusted baseline (relative parity, no magic literal).
  Verified manually: the lib reads `total = 2813` (matching 0.15.4) where the unfixed
  path read `0`.
- **Gate trips without rebuild** (`v40_empty_read_gate_trips_and_seals_without_checkpoint`)
  — a populated-footprint store that reads empty must return
  `WalRecoveryOutcome::SuspectedDataLoss`, run **no** checkpoint, leave the store
  **not** upgraded to v41, write **no** `*.corrupt-*` artifact, and leave the bytes
  byte-identical (the gate peeks **read-only**). Holds **regardless of the engine** —
  it is the durable proof of the safety property.
- **No false positive on a fresh store** — a genuinely empty/new store (≤32 KiB)
  probes `CommittedFootprint::Empty` and opens `Clean`, never `SuspectedDataLoss`.

---

## Operational runbook

**Symptom: logs show `SuspectedDataLoss`, or `open_persistent` returns
`MemoryError::SuspectedDataLoss { footprint_bytes, read_count }`.**
The store **opened but read empty** despite a populated on-disk footprint (the
#107 silent empty-read). **Nothing was written, checkpointed, or quarantined** —
the on-disk bytes are intact and still readable by the engine that wrote them.
**Do not** retry with `--force`, delete the store, or run a binary that checkpoints
it. Instead:

1. **Roll back** to the engine/binary that reads the store correctly (the
   writer's version), or deploy a build carrying the lib read-path fix (the #107
   fix is lib-side; see [`lbug_0_17_upgrade.md`](lbug_0_17_upgrade.md#root-cause-corrected-the-_deleted-tombstone-filter-on-a-legacy-schema)).
2. **Validate on a copy** (`cp -a <store> /tmp/copy && open /tmp/copy`) and confirm
   the count matches a trusted baseline before pointing any writer at the original.

This is a **fail-closed** state by design: the gate chose "refuse and preserve"
over "run empty and lose data", because the bytes are recoverable.

**Symptom: logs show `RebuiltAfterCorruption`.**
The catalog was corrupt and the store rebuilt itself from empty. The consumer is
*running again* with an empty graph — the crash loop is broken. The old data is
preserved at the `quarantined_wal` path (`<db_path>.corrupt-<ts>`). Investigate
or attempt offline salvage from that file; rehydrate from your source of truth if
you have one.

**Symptom: logs show `auto-checkpoint failed … buffer-pool exhaustion`, or
`last_checkpoint_error()` is `Some`.**
This is the leading indicator of the #95 failure. Raise the buffer-pool cap:

```bash
export AMPLIHACK_MEMORY_BUFFER_POOL_BYTES=4294967296   # 4 GiB
# restart the consumer; confirm the once-at-open INFO line shows the new value
```

The error clears itself on the next successful checkpoint once the pool has
headroom.

**Symptom: disk filling with `*.corrupt-*` files.**
Quarantined artifacts are **never auto-deleted** (data preservation is
deliberate). Prune them out-of-band once you have confirmed they are not needed:

```bash
# Inspect first; delete only what you have triaged.
ls -lh ./data/*.corrupt-*
```

**Tuning guidance.**
- Set `AMPLIHACK_MEMORY_BUFFER_POOL_BYTES` to a comfortable fraction of free RAM
  on busy hosts (the pool is lazy, so headroom is cheap insurance against
  checkpoint-time exhaustion).
- Set `AMPLIHACK_MEMORY_MAX_DB_BYTES` above your expected long-term database size
  (it is only an address-space reservation).
- Leave `auto_checkpoint(true)` and `AUTO_CHECKPOINT_WRITES` at their defaults
  unless you have a specific reason to change the WAL-bounding cadence.

---

## Compatibility & guarantees

- **API additions only.** The implemented additions — `RebuiltAfterCorruption`, the
  env constants, and the `buffer_pool_bytes()` / `max_db_bytes()` /
  `last_checkpoint_error()` getters — are purely additive; existing signatures
  (`open`, `open_with_recovery`, `checkpoint`, `WalRecovery` fields) are unchanged,
  including the historical `quarantined_wal` field name. The #107 surface
  (`WalRecoveryOutcome::SuspectedDataLoss`, `MemoryError::SuspectedDataLoss`,
  `count_all_nodes() -> usize`, the `committed_footprint` probe) is likewise
  additive — match sites already carry `_ =>` arms.
- **Strict mode is unchanged.** `LbugGraphStore::open()` still errors on a corrupt
  WAL or catalog and never rebuilds — use it when you *want* to detect corruption
  (tests, repair tooling). It does **not** run the empty-read gate (the gate is a
  resilient-path behaviour).
- **No data is ever deleted.** Every corrupt WAL or database is moved aside to a
  timestamped sibling first. The empty-read gate goes further: on a suspected
  empty read it moves **nothing** and writes **nothing** (it peeks read-only),
  leaving the bytes exactly as found.
- **The empty-read gate is fail-closed, not fail-open.** Unlike catalog
  recovery (which favours availability and rebuilds-from-empty), the #107 gate
  **refuses** to proceed when a populated store reads empty — because there the
  bytes are intact and "run empty + checkpoint" would convert a recoverable state
  into permanent loss. Treating a `0` count over a material footprint as suspect
  ensures a *failed* read is never mistaken for an empty store.
- **Default behaviour is safer, not different.** With no env vars set, the store
  uses a 1 GiB pool / 16 GiB max DB (up from 128 MiB / 1 GiB), self-heals catalog
  corruption, and refuses to checkpoint a store that read empty — strictly more
  robust than before, with the same on-disk format.

### See also

- [`docs/lbug_0_17_upgrade.md`](lbug_0_17_upgrade.md) — the engine upgrade, the
  #107 root cause, the production-scale fixture, and the cross-repo engine fix +
  Simard safe-update parity gate.
- [`README.md` → Durability & crash recovery](../README.md) — the at-a-glance
  summary.
- [`CHANGELOG.md`](../../../CHANGELOG.md) — the `[Unreleased]` entries for #95
  (configurable limits + catalog recovery) and #88 (corrupt-WAL recovery). The #107
  empty-read parity gate will add its entry when it lands.
- Module docs on `crate::graph::lbug_store` — the rustdoc that mirrors this page.
