# Coordination Layer (Design C) — Durability Contract

**The transactionally-safe, crash-durable coordination layer over the single-writer
`lbug` cognitive store.**

The coordination layer ("Design C") lets many short-lived agent processes (engineers,
worktrees, CLI sessions) contribute writes to one shared `lbug`-backed cognitive store
without corrupting it, losing acknowledged writes, or double-applying an effect after a
crash. Every guarantee documented here is pinned to a TLA+ invariant in
[`specs/`](../specs/README.md) — the specs are the source of truth, and CI blocks any
change that drifts from them (job `tla-model-check`).

This page is the operator- and integrator-facing **durability contract** for the F1–F5
work. Every guarantee below is now **shipped** and pinned to a TLA+ invariant plus a
regression test that fails if the fix is reverted. For the module-internal narrative see
the crate docs on `amplihack_memory::coord`.

> ## ✅ Implementation status
>
> The F1–F5 hardening is complete. Every row below is **Shipped** — each fix landed with
> a crash-injection/concurrency regression test that fails without it.
>
> | Fix | Guarantee | Status | How it is enforced |
> |-----|-----------|--------|--------------------|
> | **F1** | Torn-tail recovery | ✅ **Shipped** | Torn partial tail truncated under the append lock **before** the next append; complete frames never discarded; interior-CRC fails closed |
> | **F2** | Exactly-once apply | ✅ **Shipped** | Store-resident `coord_applied_intent` ledger co-committed with the effect under one `CHECKPOINT`; the applier's dedup set is seeded from it at open, so replay survives restart |
> | **F3** | Durable append (ack = durable) | ✅ **Shipped** | Unconditional record `fsync` + parent-directory `fsync` on new-segment creation, regardless of `fsync_on_append` |
> | **F4** | Single-writer ownership | ✅ **Shipped** | Exclusive `flock(LOCK_EX\|LOCK_NB)` on `<store>.writer.lock` held for the handle lifetime; fail-closed `AlreadyLocked` |
> | **F5** | Atomic lease write | ✅ **Shipped** | `tmp → fsync → rename → dir-fsync` lease swap, serialized by a dedicated `lease.lock` |

---

## Durability guarantees at a glance

| # | Guarantee | Status | What it means for you | Backing invariant / spec |
|---|-----------|--------|-----------------------|--------------------------|
| **F1** | **Torn-tail recovery** | ✅ Shipped | A crash mid-append never discards a *complete, fsynced* write. The unfinished trailing bytes are truncated under the append lock before the next write, never left to corrupt a following acked frame. | `PrefixConsistency` (`DurableLog.tla`) |
| **F2** | **Exactly-once apply** | ✅ Shipped | An acked write is applied to the store **exactly once**, even if the daemon crashes between materializing the effect and advancing its cursor. Replays are idempotent across restarts. | `NoLostAckedWrite` + `PrefixConsistency` |
| **F3** | **Durable append (ack = durable)** | ✅ Shipped | When `append` returns success, the record and (on new segments) its containing directory are `fsync`-ed to stable storage. The ack is a real durability point, not a page-cache promise. | `NoLostAckedWrite` (`DurableLog.tla`) |
| **F4** | **Single-writer ownership** | ✅ Shipped | Exactly one process (or handle) can hold the persistent store writer at a time. A second opener **fails closed** immediately instead of racing into split-brain. | `NoSplitBrain` (`FencedApplier.tla`) |
| **F5** | **Atomic lease write** | ✅ Shipped | Updating the ownership lease is all-or-nothing. A crash mid-write can never leave a half-written lease; the previous lease record survives intact. | `NoSplitBrain` (`FencedApplier.tla`) |

The on-disk format (magic/version/CRC framing, `WriteIntent` serialization, IEEE CRC-32)
is backward-compatible across this work: an existing store opens without migration. The
F2 ledger adds a new node label (`coord_applied_intent`) rather than changing any record
layout, and F4/F5 add sidecar files (`<store>.writer.lock`, `lease.lock`), not new bytes
in existing records.

---

## Architecture recap

The diagram shows the Design C topology — every element below is shipped.

```
 ┌────────────┐   append (F3: fsync + dir-fsync = ACK)   ┌──────────────────────┐
 │  Writer A  │ ───────────────────────────────────────▶ │  Durable intent log   │
 ├────────────┤                                           │  (segmented, CRC-      │
 │  Writer B  │ ───────────────────────────────────────▶ │   framed, append-only)│
 └────────────┘                                           └──────────┬───────────┘
        ▲                                                            │ drain in order
        │ F4: LOCK_EX|LOCK_NB ownership lock (<store>.writer.lock)   ▼
        │ F5: atomic lease write (epoch)                  ┌──────────────────────┐
 ┌──────┴───────────────────────────┐   F2: co-commit     │   Single Applier      │
 │  lbug single-writer cognitive     │◀──────────────────│   (epoch-fenced)      │
 │  store  +  applied-intent ledger  │   under one         └──────────────────────┘
 │                                   │   CHECKPOINT barrier
 └───────────────────────────────────┘
```

* **Writers** append `WriteIntent`s to a durable, `fsync`-on-append shared log. The
  append is the durability acknowledgement (F3): the record is `fsync`-ed unconditionally,
  and a newly created segment's parent directory is `fsync`-ed too.
* A **single Applier** drains the log strictly in order and applies each intent to the
  `lbug` store. It fences on a monotonic **epoch** carried by the lease so it can never
  mutate the store under a stale claim, and the store handle holds an exclusive
  ownership lock (F4) that prevents a second writer from even *starting*.
* The exactly-once story (F2) is a store-resident **applied-intent ledger**
  (`coord_applied_intent`, keyed on `intent_id`) committed under the *same* `CHECKPOINT`
  barrier as the effect. The applier seeds its in-memory dedup set from the ledger at
  open, so replay after a restart in the checkpoint↔cursor window is a no-op.
* The **lease** carries the epoch and is updated by an atomic `rename` (F5); together with
  the ownership lock (F4) these are the split-brain guards.

---

## Configuration

Durability is governed by `CoordConfig`. The defaults are the safe defaults — you should
rarely change them.

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `base_dir` | `PathBuf` | `<store>.coord` | Root of the coordination directory (`0o700`). Placed **beside** the store so an ordinary directory backup captures it. |
| `segment_bytes` | `u64` | `64 MiB` | Intent-log segment size before rollover. |
| `socket_name` | `String` | `read.sock` | Read-transport Unix socket filename. |
| `max_frame_bytes` | `u32` | `16 MiB` | Hard cap on a single log record / IPC frame. Bounds a hostile or corrupt length prefix **before** any body allocation (F1 defence-in-depth). |
| `fsync_on_append` | `bool` | `true` | **Non-authoritative for the ack (F3).** The append ack always `fsync`s the record and, on a new segment, its parent directory — regardless of this flag. Retained only for wire/config compatibility. See the note below. |

### `fsync_on_append` is non-authoritative — the ack always fsyncs (F3)

The append ack `fsync`s the acknowledged record **and**, when it creates a new segment,
the containing directory — **unconditionally**, independent of `fsync_on_append`. The
`fsync` *is* the durability ack (`append` returns the `LogOffset` only once the write is
durable), so it can never be relaxed on the ack path. `fsync_on_append` is retained only
so existing `CoordConfig` literals keep compiling; setting it to `false` does **not**
weaken the ack. If you want a genuinely non-durable throwaway store, use an in-memory /
ephemeral store rather than trying to disable fsync here.

```rust
use amplihack_memory::coord::CoordConfig;

// Default, recommended: coordination dir beside the store, full durability.
let config = CoordConfig::for_store("/data/agents/hive.lbug");

// Advanced: larger segments for a very high-write daemon.
let config = CoordConfig {
    segment_bytes: 256 * 1024 * 1024,
    ..CoordConfig::for_store("/data/agents/hive.lbug")
};
```

The coordination directory layout:

```
/data/agents/hive.lbug            # the lbug store file (+ .wal, .corrupt-* sidecars)
/data/agents/hive.lbug.writer.lock # F4 exclusive single-writer ownership lock (flock)
/data/agents/hive.lbug.coord/     # coordination dir (0o700)
├── lease                         # ownership lease carrying the epoch
├── lease.lock                    # F5 dedicated RMW serialization lock (stable inode)
├── .lease.tmp                    # transient F5 temp file (renamed over `lease`; not at rest)
├── applied-index                 # durable applier cursor
├── intent-log/                   # segmented, append-only, CRC-framed durable log
│   ├── 000000000000.seg
│   └── 000000000001.seg
└── read.sock                     # read-transport UDS
```

The store's `.wal` and `.corrupt-<ts>` sidecars are lbug WAL-recovery artifacts, not
coordination-layer files.

---

## F1 — Torn-tail recovery  ✅ Shipped

The durable intent log is segmented and every record is CRC-framed. On the append path,
under the append `flock` and **before** the target segment or append offset is chosen, the
last segment's tail is validated and any torn trailing bytes are truncated away:

* The tail is scanned forward over complete frames. A frame's length prefix is checked
  against `max_frame_bytes` **before** any allocation, so a hostile/corrupt length prefix
  cannot drive an over-read or over-allocation.
* Every **complete** frame is kept — a fsynced, acked write is never discarded (even a
  complete frame whose CRC does not validate is preserved for the reader to surface, not
  truncated).
* Only a genuinely **incomplete** trailing partial (a crash mid-`write_all`, before the
  `fsync`/ack) is truncated to the last good frame boundary, so the next append lands
  cleanly and can never turn the torn bytes into committed-interior corruption or eat a
  following acked frame.

A steady-state fast path caches the clean length this writer last left behind, so the
single-writer common case skips the scan while the on-disk length is unchanged.

A CRC mismatch on a **committed interior** record (not the tail) is a different, harder
failure: it is **never skipped** (that would violate `PrefixConsistency`) and surfaces as
`MemoryError::LogCorruption { segment, offset }` so an operator can intervene.

---

## F2 — Exactly-once apply  ✅ Shipped

Every `WriteIntent` carries an `intent_id` (a `Uuid`; `append_new` assigns a fresh v4 id
and returns it so the caller can correlate the eventual apply). Exactly-once apply is
enforced by a **store-resident idempotency ledger**: a node labelled `coord_applied_intent`
keyed on `intent_id`, written through the same single-writer graph as the intent's effect
and therefore co-committed with it under the **same `CHECKPOINT` barrier** — *before* the
durable `applied-index` cursor advances.

Because the effect and its ledger marker land in one WAL flush / one checkpoint, there is
no crash window in which the effect exists but the "already applied" fact does not. The
applier:

1. **Seeds** its in-memory dedup set from the ledger at open (`applied_intent_ids()`), so a
   fresh process that resumes from a cursor a crash left *behind* the store effects still
   recognises the already-applied records.
2. Re-reads the log from the durable `applied-index`.
3. For each intent, skips it if its `intent_id` is already in the (ledger-seeded) set;
   otherwise applies the effect **and** records the ledger marker.
4. Checkpoints once at the end of the batch, co-committing every effect and marker, then
   advances the durable cursor.

This closes the "checkpoint succeeded, index advance lost to a crash → replay
double-applies" defect **across process restarts**. `lbug` exposes `execute(cypher)` + an
explicit `CHECKPOINT` barrier rather than multi-statement transactions, so exactly-once is
achieved by co-committing effect + marker under one checkpoint, not a SQL-style
`BEGIN/COMMIT`.

### Ledger properties

* **Injection-safe:** ledger writes go through the same parameterized store mutator surface
  as every other node, never Cypher string interpolation.
* **Idempotent marker:** the ledger node is keyed on `intent_id` (`node_id =
  coord_applied_<intent_id>`), so re-recording the same intent is a no-op.
* **Growth:** the ledger grows one small node per applied intent. Pruning markers strictly
  below the durable `applied-index` is a documented follow-up, not a correctness risk.

---

## F3 — Durable append (ack = durable)  ✅ Shipped

`append` `fsync`s the log segment **unconditionally** before returning, so the returned
`LogOffset` is a real durability acknowledgement for the record's bytes — the
`fsync_on_append` flag can no longer downgrade the ack (see the config note above). When
the append creates a **new segment**, it additionally `fsync`s the **parent directory** so
the new segment's directory entry (name → inode link) is durable too; without that, a power
loss after the ack could vanish the whole new segment file. A non-`fsync`'d append is never
treated as an ack.

---

## F4 — Single-writer ownership lock  ✅ Shipped

Opening the persistent store (`open_persistent` / `open_persistent_with_recovery`) acquires
an exclusive, **non-blocking** advisory writer lock (`flock(LOCK_EX | LOCK_NB)`) on a
store-adjacent `<store>.writer.lock` file **before** the store itself is opened. The lock is
bound to the store-handle lifetime and released on `Drop`. There is **no pid-liveness
probing** (`kill(pid, 0)` is never used) — ownership is purely the lock fd's lifetime plus
the monotonic epoch, which avoids TOCTOU and pid-reuse spoofing.

> Note: the `flock` calls elsewhere in the tree (`intent_log.rs` append lock, `lease.rs`
> `lease.lock`) are **brief serialization locks** around a single append or a lease
> read-modify-write — they are explicitly *not* ownership. F4's lock is a distinct,
> handle-lifetime ownership lock on a separate file.

### Behaviour

* **First opener** acquires the lock and proceeds.
* **Second opener** (a different process, or a second handle in the same process that did
  not drop the first) **fails closed immediately** with `MemoryError::AlreadyLocked { path }`
  — it never blocks, and never falls back to an unlocked open.
* On process exit or `Drop`, the OS releases the `flock`, so a *legitimate* reopen (after
  the prior handle is gone) succeeds. There is no stale-lock cleanup step and no manual
  unlock file to remove.

```rust
use amplihack_memory::CognitiveMemory;
use amplihack_memory::errors::MemoryError;

let mem = CognitiveMemory::open_persistent("/data/agents/hive.lbug", "daemon")?;
// ... this process now owns the single writer ...

// A second opener anywhere else:
match CognitiveMemory::open_persistent("/data/agents/hive.lbug", "intruder") {
    Err(MemoryError::AlreadyLocked { path }) => {
        eprintln!("store already owned by another writer: {path}");
        // back off / connect as a read client / append via the writer log instead
    }
    Ok(_second) => unreachable!("single-writer invariant"),
    Err(e) => return Err(e),
}
```

> **Same-process reopen:** because `flock` is per-open-file-description, reopening the same
> store in the same process while the first handle is still alive is treated as contention
> and fails closed. Drop the first handle before reopening.

**Epoch fence (complementary).** At every apply, the applier also **fences on the epoch**:
if the live on-disk lease epoch has advanced past the epoch the applier stamped, the apply
is rejected (`MemoryError::EpochFenced { expected, found }`) rather than mutating the store
under a stale claim. The ownership lock prevents two writers from *starting*; the epoch
fence guarantees no mutation under a stale claim across a handoff. Together they are the
shipped `NoSplitBrain` guarantee.

---

## F5 — Atomic lease write  ✅ Shipped

The ownership lease (which carries the monotonic `Epoch`) is rewritten with the proven
durable-swap idiom instead of an in-place truncate-then-write:

```
write the record to <coord>/.lease.tmp   (same dir, 0600, truncated)
    → fsync(tmp)
    → rename(tmp, lease)                  (atomic)
    → fsync(parent dir)
```

Because an in-place `set_len(0)` + write could leave the live lease **empty** if a crash
landed between the truncate and the record write — which would make `current_epoch` fail
closed and strand the store — the atomic `rename` guarantees the lease path always resolves
to a **complete** record: either the old one or the new one, never nothing.

Since the atomic `rename` replaces the lease *inode*, an `flock` on the lease file itself
would be defeated by the swap. The read-modify-write is therefore serialized on a
**dedicated `lease.lock`** file whose inode is stable, while the lease record is swapped by
`rename`. Reads (`current_epoch`) need no lock at all: a `rename` is atomic, so a reader
always sees one complete record.

Consequences you can rely on:

* A crash at any point leaves **either** the complete previous lease **or** the complete new
  lease on disk — never a torn record, and never an epoch lower than a previously-persisted
  one.
* The temp file is same-directory (no cross-filesystem `rename` failure) and `0600` (no
  world-readable window), and is renamed away — never observed at rest.
* Readers of the lease never observe a partially written epoch.

No new bytes are added to the lease record format; only the write procedure and its lock
change.

---

## Error reference

All coordination errors are variants of `amplihack_memory::errors::MemoryError`. They are
**structural only** — they carry offsets, epochs, counts, and paths, never user memory
payloads. (`amplihack_memory` uses a single `MemoryError` enum; there is no separate
`CoordError` type.)

| Variant | Status | Raised by | Meaning & recommended handling |
|---------|--------|-----------|--------------------------------|
| `LogCorruption { segment, offset }` | ✅ Shipped | F1 interior-CRC check | A committed interior log record is corrupt. **Not** auto-skipped. Operator intervention required (inspect the named segment/offset; restore from backup if needed). |
| `EpochFenced { expected, found }` | ✅ Shipped | F4 applier fence | The applier's stamped epoch is stale; another process took the lease. The apply was rejected (store untouched). Restart/reacquire cleanly. |
| `UnsupportedIntentVersion` | ✅ Shipped | Applier decode | The log contains an intent kind/version this daemon doesn't understand. Fails closed — deploy a newer daemon rather than dropping the acked write. |
| `SuspectedDataLoss { footprint_bytes, read_count }` | ✅ Shipped | Store open | A populated store read back empty. The store is left intact (not checkpointed) so the loss can't be made permanent. |
| `Storage(String)` | ✅ Shipped | I/O layer | Underlying I/O failure (fsync, rename, dir create). Message carries a payload-free context label. |
| `AlreadyLocked { path }` | ✅ Shipped | F4 open path | Another writer already owns the store's exclusive lock. Do **not** retry-spin blindly; back off, connect as a read client, or append via the writer log. Fail-closed by design. |

Guiding principle across all five fixes: **fail closed, never silently degrade.** A lock
failure (F4), a CRC failure (F1), or a rename failure (F5) always propagates a typed `Err` —
none of them fall back to an unsafe path.

---

## Tutorial: a safe multi-writer daemon

The intended topology is one long-lived **daemon** that owns the writer, plus many
short-lived **agents** that append intents and read via the socket. The examples below
compile against the current API.

### 1. Start the daemon (drains the log, epoch-fenced, single-writer locked)

```rust
use std::sync::atomic::{AtomicBool, Ordering};
use amplihack_memory::coord::{CoordConfig, Coordinator};

fn run_daemon(shutdown: &AtomicBool) -> amplihack_memory::errors::Result<()> {
    let store_path = std::path::Path::new("/data/agents/hive.lbug");
    let config = CoordConfig::for_store(store_path);

    // Acquires the lease + takes the exclusive single-writer ownership lock (F4)
    // + opens the single lbug writer + resumes from the durable applied-index.
    let coord = Coordinator::open(store_path, "daemon", config)?;

    // Drains the durable log, applying each intent under the epoch fence (F4)
    // and recovering any torn tail (F1) found on startup. Blocks until shutdown.
    coord.run_applier(|| shutdown.load(Ordering::Relaxed))?;
    Ok(())
}
```

> Starting a second daemon by mistake exits immediately with
> `MemoryError::AlreadyLocked` — no split brain, no corruption.

### 2. Agents append durable writes (the ack is durable)

```rust
use uuid::Uuid;
use amplihack_memory::coord::{CoordConfig, WriteIntent, WriterClient};

fn agent_write() -> amplihack_memory::errors::Result<()> {
    let config = CoordConfig::for_store("/data/agents/hive.lbug");
    let writer = WriterClient::connect(&config)?;

    // WriteIntent is an enum; append_new() assigns a fresh v4 intent_id
    // (so the intent_id below is a placeholder it will overwrite).
    let intent = WriteIntent::StoreFact {
        intent_id: Uuid::nil(),
        agent_name: "agent-7".to_string(),
        concept: "deployment".to_string(),
        content: "Staging uses port 8080".to_string(),
        confidence: 0.9,
        source_id: "runbook-42".to_string(),
        tags: None,
        metadata: None,
    };

    // append_new() returns (durable offset, assigned intent_id). The record is
    // fsynced unconditionally before returning (F3), so the offset is a real ack.
    let (offset, id) = writer.append_new(intent)?;

    // Safe to report success: this write survives a crash and will be applied to
    // the store; with F2's ledger it is applied exactly once even across restarts.
    println!("acked intent {id} at offset {offset:?}");
    Ok(())
}
```

### 3. Crash-safety (all shipped)

* Agent dies **after** `append` returns → the write is durable unconditionally (F3 ✅:
  segment `fsync` on every ack, plus a parent-dir `fsync` when a new segment is created) and
  will be applied by the daemon.
* Daemon dies **mid-append** to the log → the incomplete trailing frame is truncated back to
  the last complete frame on next open (F1 ✅); no complete frame is lost, and interior
  corruption is surfaced as a hard error rather than silently skipped.
* Daemon dies **between checkpoint and index-advance** → replay is deduped by the
  store-resident intent ledger (F2 ✅), so the effect is applied **exactly once** across both
  in-process replay and a full restart. The effect count stays at 1.
* Daemon dies **mid-lease-write** → the lease is swapped atomically via temp + `fsync` +
  `rename` + dir-`fsync` (F5 ✅), so on-disk you always find one complete lease — never a
  torn or empty one.
* Two daemons started at once → the second **fails closed** with
  `MemoryError::AlreadyLocked` (F4 ✅), backed by the epoch fence for handoff safety.

---

## Operational notes

* **Backups:** because the coordination directory sits beside the store, a plain recursive
  copy of `<store>` and `<store>.coord/` captures a consistent set. Take backups with the
  daemon stopped (or via a filesystem snapshot) so the applied-index and store agree.
* **Torn-tail recovery:** F1 recovers a crash-torn log by **truncating** the incomplete
  trailing frame in place — it does **not** leave `.corrupt-<ts>` sidecar files. (The
  `*.corrupt-<ts>` files you may see next to the lbug store are unrelated lbug WAL-recovery
  artifacts, not coordination-layer output.)
* **Writer lock file:** `<store>.writer.lock` is the F4 ownership lock. It is created on open
  and the advisory `flock` is released on process exit / handle `Drop`; the file itself may
  remain at rest and is safe to leave in place (it carries no state).
* **Telemetry:** all durability events are emitted via structured `tracing` + OpenTelemetry
  (no `print!`/`println!`). Logs contain `intent_id`s, offsets, epochs, and counts — never raw
  memory payloads.
* **Verifying the guarantees:** the TLA+ specs are model-checked in CI (`tla-model-check`).
  Each of F1–F5 ships with a regression test that **fails if the fix is reverted** (e.g.
  crash-between-checkpoint-and-index-advance → effect count must equal 1; second opener must
  error `AlreadyLocked`; crafted oversized length-prefix must be rejected, not allocated).

---

## Related documentation

* [`specs/README.md`](../specs/README.md) — the TLA+ models (`DurableLog.tla`,
  `FencedApplier.tla`, `FederatedLoss.tla`) and their invariants.
* [Architecture](architecture.md) — package structure and core components.
* [Kuzu Backend](kuzu_backend.md) — how the underlying graph store works.
