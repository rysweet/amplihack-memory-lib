# Exactly-Once Apply, Bounded Ledger & Verified Clean-Tail Recovery

The multi-writer coordination layer ([`coordination_layer.md`](coordination_layer.md))
drains a durable shared intent log through a **single fenced applier** and applies
each [`WriteIntent`] to the `lbug`-backed store exactly-once. This page is the
authoritative reference for the three durability/performance guarantees that make
that "exactly-once" claim hold on a **busy, crash-prone daemon** — not just in a
clean shutdown:

| ID | Guarantee | What it prevents |
| --- | --- | --- |
| **F2** | [Replay-safe (idempotent) effects](#f2--replay-safe-idempotent-effects) | A crash/auto-checkpoint landing *between* an effect and its ledger marker re-applying the effect on restart → **duplicate memory**. |
| **N2** | [Bounded applied-intent ledger](#n2--bounded-applied-intent-ledger) | The idempotency ledger growing **without bound** (one permanent marker node per applied intent) and degrading lookups forever. |
| **N1** | [Verified clean-tail log recovery](#n1--verified-clean-tail-log-recovery) | `append()` re-CRC-scanning the **entire** last segment (up to 64 MiB) on *every* append → O(segment²) work that serializes all writers. |

These build on — and **never regress** — the round-1 coordination fixes
(F1 torn-tail recovery, F3 directory `fsync` on the ack path, F4 single-writer
`flock`, F5 atomic lease writes). See the
[invariant → spec map](#invariant--spec-map) for how each interacts with the
TLA+ model.

> **Feature gate.** Everything on this page is part of the `coord` layer and the
> `persistent` engine path:
> `cargo build --features persistent` / `cargo test --features persistent`.

---

## Table of contents

1. [The failure it closes](#the-failure-it-closes)
2. [F2 — replay-safe (idempotent) effects](#f2--replay-safe-idempotent-effects)
   - [Deterministic node ids](#deterministic-node-ids)
   - [Per-variant idempotency contract](#per-variant-idempotency-contract)
   - [Idempotent links](#idempotent-links)
   - [Episode `temporal_index` is create-only](#episode-temporal_index-is-create-only)
   - [`RecordAccess` is stamp-guarded](#recordaccess-is-stamp-guarded)
3. [N2 — bounded applied-intent ledger](#n2--bounded-applied-intent-ledger)
4. [N1 — verified clean-tail log recovery](#n1--verified-clean-tail-log-recovery)
5. [On-disk layout additions](#on-disk-layout-additions)
6. [API reference](#api-reference)
7. [Configuration & test seams](#configuration--test-seams)
8. [Tutorials](#tutorials)
9. [Invariant → spec map](#invariant--spec-map)
10. [Security model](#security-model)
11. [Compatibility & downgrade safety](#compatibility--downgrade-safety)

---

## The failure it closes

The applier drains the log and, for each record, performs **two** store writes:

```text
apply_intent(&mut memory, &intent)?;   // the effect  (may be many store writes)
memory.mark_intent_applied(id)?;       // the durable idempotency marker
```

The store auto-checkpoints every `AUTO_CHECKPOINT_WRITES` (128) writes, and the
process can be `SIGKILL`ed at any instant. Because a single intent expands into a
*variable* number of store writes (node + edges + tags + provenance), the
128-write checkpoint boundary is **not pair-aligned** with the
`effect → marker` pair. So on a busy daemon this window is reachable:

```text
... effect write N-1 | effect write N | << AUTO-CHECKPOINT / CRASH >> | marker write | ...
                       \___ effect now DURABLE ___/                    \_ never ran _/
```

On restart the `applied-index` never advanced (its persist runs only at
end-of-drain), the in-memory `seen` set is empty, and `intent_applied(id)` is
`false` because the marker never landed — so the applier **re-applies the effect**.
Before F2 that minted a *fresh* duplicate node (and double-bumped
`temporal_index`). The three fixes on this page make that re-apply a **no-op**,
keep the idempotency ledger **bounded**, and make the per-append recovery scan
**O(1)** in the common case.

The core design decision (F2 **Option 1**): make the *effect itself replay-safe*
rather than trying to wrap effect + marker in one transaction or disabling
checkpoints. A replay-safe effect makes the `effect ↔ marker` window **harmless**
*and* makes N2 pruning inherently safe — a pruned-but-durable intent that happens
to replay simply upserts the same nodes/edges again.

---

## F2 — replay-safe (idempotent) effects

**Guarantee:** applying the same [`WriteIntent`] (same `intent_id`) any number of
times converges to exactly one logical effect — identical node ids, identical
edges, identical scalar fields, and **no** monotonic side effects (no extra
`temporal_index` bump, no extra `usage_count` increment). This holds for **all
eight** `WriteIntent` variants and is enforced at the shared `apply_intent`
choke point so **both** drain paths ([`Applier::drain`] and
[`Coordinator::drain_once`]) inherit it automatically.

### Deterministic node ids

Create-family intents no longer mint a random UUID. The effect node id is
**derived deterministically** from the intent's idempotency key using a
**disjoint fixed prefix** per memory family, and written through the store's
existing same-primary-key **upsert/revive** path
(`LbugGraphStore::add_node`, `store_impl.rs`). A replay therefore `MATCH … SET`s
the *same* row instead of `CREATE`-ing a new one.

| Variant | Deterministic id | Prefix |
| --- | --- | --- |
| `StoreFact` | `sem_{intent_id}` | `sem_` |
| `StoreEpisode` | `epi_{intent_id}` | `epi_` |
| `StoreProcedure` | `proc_{intent_id}`¹ | `proc_` |
| `StoreProspective` | `pro_{intent_id}` | `pro_` |
| `UpsertFact` (create arm) | `sem_{intent_id}` | `sem_` |
| applied-intent marker | `aintent_{intent_id}` | `aintent_` |

¹ `StoreProcedure` was already idempotent-by-name (upsert-by-`name`); the
deterministic id keeps it aligned and collision-free with the other families.

The prefixes are **disjoint and fixed** so that an attacker-influenced `intent_id`
in one family can never forge a primary key in another family. The `intent_id` is
a validated `Uuid` (canonical hyphenated form only), so the derived id has a
bounded charset and length and is never interpolated into a Cypher/path context
beyond the existing escaped-parameter path.

> **Threading note.** `apply_intent` today destructures the `intent_id` away
> (`..`) and calls the public `store_fact` / `store_episode` / `store_prospective`
> / `upsert_fact` methods, which **mint a random id internally**. The fix adds
> `pub(crate)` id-accepting entry points (e.g. `store_fact_with_id`, or routing
> through the existing `*_inner`) and has `apply_intent` pass the derived
> deterministic id. The public random-id methods and their existing callers are
> **unchanged** — determinism is opt-in on the applier path only.

### Per-variant idempotency contract

| Variant | Replay behavior |
| --- | --- |
| `StoreFact` | Upsert node `sem_{intent_id}`; content-hash/dedup edges re-derived identically. |
| `StoreEpisode` | Upsert node `epi_{intent_id}`; **`temporal_index` bumped create-only** (see below). |
| `StoreProcedure` | Upsert-by-(agent,name) (unchanged); already idempotent, so a replay overwrites the same canonical node with identical steps. |
| `StoreProspective` | Upsert node `pro_{intent_id}`; status/priority overwritten identically. |
| `UpsertFact` | **Both arms** are made deterministic: the create arm writes `sem_{intent_id}`; the supersede arm mutates the existing target *and* writes its new superseding revision node with a deterministic id derived from `intent_id` — so replay neither duplicates the revision node nor re-supersedes. |
| `LinkFactToEpisodes` | Edges written via [idempotent link](#idempotent-links) — a `DERIVES_FROM` edge that already exists is a no-op, so no duplicate edges accrue on replay. |
| `LinkSimilarFacts` | Same idempotent-link path for the `SIMILAR_TO` edge. |
| `RecordAccess` | [Stamp-guarded RMW](#recordaccess-is-stamp-guarded): `usage_count` is **not** re-incremented if the intent was already applied. |

### Idempotent links

Link intents previously issued a raw `add_edge` (`CREATE`), so a replay produced a
**second parallel edge**. Rather than mint a deterministic edge id and drive an
upsert, the fix takes the simpler existence-guarded path: before writing a link,
the applier checks whether a live edge of that type already connects the two
nodes and skips the write if so.

```rust
// CognitiveMemory (mod.rs):
pub(super) fn edge_of_type_exists(
    &self,
    source_id: &str,
    target_id: &str,
    edge_type: &str,
) -> bool; // true if a live `edge_type` edge already runs source → target
```

`add_provenance_edge` (used by `link_fact_to_episodes`) and `link_similar_facts`
both consult `edge_of_type_exists` and return early when the edge is already
present — so replaying a link intent converges to a single edge without needing a
deterministic edge id.

### Episode `temporal_index` is create-only

`store_episode` auto-increments the monotonic `temporal_index`. Under naive
replay that value would **diverge** (bump twice for one logical episode). The fix
is **get-before-put**: before bumping, the applier probes for the deterministic
id with `get_node("epi_{intent_id}")` (equivalently `physical_node_exists`) and
bumps `temporal_index` **only when that node is absent** (i.e. a genuine first
apply). A replay that resolves to an already-present `epi_{intent_id}` reuses the
stored `temporal_index` and leaves the counter untouched — the value stays
identical across any number of replays.

> **Implementation note.** The probe is an explicit read *before* `add_node`, not
> a return value of `add_node` — `add_node` returns `Result<GraphNode>` and does
> **not** expose a created-vs-updated discriminant. Do not rely on the store's
> upsert path to signal novelty; check existence first.

### `RecordAccess` is stamp-guarded

`RecordAccess` is a read-modify-write (`usage_count += 1`, refresh
`last_accessed`), which is inherently non-idempotent. It is made replay-safe with
a **per-node applied stamp**: the same `update_node` that bumps the counter also
sets `last_access_intent = {intent_id}`. On replay, if the node's stored
`last_access_intent` already equals the intent's id, the `usage_count` bump is
skipped (the timestamp refresh is naturally idempotent). One logical access =
exactly one increment, regardless of replays.

---

## N2 — bounded applied-intent ledger

**Guarantee:** the `NT_APPLIED_INTENT` marker set is **bounded** — markers whose
intents are durably at/below the persisted `applied-index` are pruned, so the
cognitive store does not grow without bound and `intent_applied` lookups do not
degrade over time.

### Why it is safe to prune below the watermark

The `applied-index` is a **durable low-watermark**: once
`persist_applied_index(pos)` has committed (checkpoint → `fsync` → atomic
rename), every intent strictly *before* `pos` is durably applied. A restart
resumes reading *from* `pos`, so those intents are **never re-read from the log**
and therefore never need a marker. Their markers are pure dead weight and safe to
delete. Markers for the **current, not-yet-persisted** window are **retained** —
they are the only thing that closes the F2 window if a crash rewinds the index.

### Ordering (crash-safe, both drain paths)

Pruning runs **strictly after** `persist_applied_index` succeeds, in **both**
[`Applier::drain`] and [`Coordinator::drain_once`]:

```text
1. apply effects + write markers   (per record, replay-safe via F2)
2. memory.checkpoint()             (effects + markers durable)
3. persist_applied_index(pos)      (atomic: temp → fsync → rename → dir-fsync)
4. prune_applied_below(order_key(pos))   ← STRICTLY AFTER step 3
```

Deleting **strictly `<` the watermark** and only **after** the watermark is
durable makes the prune **idempotent and crash-safe**: a crash mid-prune leaves
*extra* markers (harmless — they collapse on next prune), never *fewer* than the
safe set. Because F2 makes replay a no-op, even a marker that is pruned while its
effect is durably past the index cannot cause a duplicate if it somehow replays.

Each marker records an opaque **`applied_offset`** order-key (the log
[`LogOffset`] serialized to a total-order string) so the ledger can be compared
against the watermark without importing the `coord::LogOffset` type into the
lower `cognitive_memory` layer (that would introduce a **dependency cycle** — the
watermark always crosses the layer boundary as an opaque string).

> **Prune cost & cadence.** `prune_applied_below` is a **watermark-bounded
> delete** — it removes only markers whose `applied_offset` is `<` the watermark
> (the window that just went durable), so its cost is proportional to the number
> of *newly prunable* markers, not to the total store size. It runs once per
> successful `persist_applied_index` in each drain path. Because the delete is
> idempotent and watermark-bounded, running it on every persist is correct and
> cheap; an implementation may additionally **throttle** it (skip passes that did
> not advance the index, or batch across a few persists) with **no** effect on
> correctness — the durable `applied-index`, not the prune cadence, is the
> low-watermark. If the marker set is stored such that a range/predicate delete is
> not directly available, the implementation must still bound the scan to the
> prunable window rather than re-walking the entire marker set each pass.

> **Ordering contract (critical).** `prune_applied_below` compares
> `applied_offset` against the watermark with a **string** `<`, but `LogOffset`
> is `(segment: u64, offset: u64)`. `order_key()` **must** therefore encode a
> *lexicographically* total order that matches `LogOffset`'s numeric `Ord` — i.e.
> fixed-width zero-padded fields, e.g. `format!("{segment:020}:{offset:020}")`.
> A naive `"{segment}:{offset}"` breaks at 10 vs 2 and would let prune delete
> markers **above** the true watermark (data loss / F2 regression). The
> `order_key` unit test must assert `a < b (LogOffset)  ⇔  a.order_key() <
> b.order_key()` over segment/offset rollovers.

---

## N1 — verified clean-tail log recovery

**Guarantee:** `append()` performs **O(1)** recovery work in the common
clean-tail case instead of re-CRC-scanning the whole last segment, while
preserving F1's torn-tail and interior-corruption semantics **exactly**.

Before N1, every `append` called `recover_tail`, which re-read and CRC-validated
the **entire** last segment (up to `segment_bytes`, default 64 MiB) under the
append `flock` — O(segment²) per full segment, serializing all writers.

### The `.clean-tail` lower-bound hint

A verified clean-tail **offset** is persisted as an atomic sidecar
(`<intent-log>/.clean-tail`) using the same `temp → fsync → rename → dir-fsync`
pattern as `persist_applied_index`. It is a **monotone lower bound** on the
byte offset up to which the current segment is known frame-aligned and
CRC-clean.

`append` behaves as follows:

- **Fast path (tail is the known-clean boundary):** if the observed segment
  length equals the recorded clean-tail boundary, the full re-scan is skipped
  entirely — the tail is already known clean, so recovery is O(1).
- **Slow path (tail past the clean boundary):** only the **unverified suffix**
  from the clean-tail offset forward is scanned and CRC-validated via
  `recover_tail_from(index, start)`; on success the new clean boundary is
  persisted. A full `recover_tail` runs only when the observed tail is **not** a
  clean frame boundary relative to the hint.

### F1 semantics are preserved exactly

The clean-tail offset is a **hint only** and is validated before use — it must be
`≤` the segment length and land on a frame boundary; otherwise it is ignored and
a full suffix scan runs. Every **consumed** frame is still CRC-checked. All F1
guarantees are unchanged:

- a **torn tail** (partial/implausible/bad-CRC *final* frame) is truncated to the
  last clean boundary, and
- a **bad-CRC interior** frame (bytes follow it) is still a hard
  [`MemoryError::LogCorruption`] — **never** truncated away.

The `.clean-tail` sidecar is never allowed to cause a consumed frame to skip its
CRC check, so a corrupt or tampered hint can at worst force one extra full scan —
it can never hide corruption.

---

## On-disk layout additions

Relative to the [coordination directory layout](coordination_layer.md#directory-layout),
this feature adds one sidecar and one node property. Nothing else on disk
changes.

```text
<base_dir>/                     # e.g. <store>.coord/
├── lease                       # (F5) atomic lease record
├── lease.lock                  # (F5) RMW flock target
├── applied-index               # durable applier cursor (LogOffset)
├── intent-log/
│   ├── .append.lock            # (F1/F4) append serialization flock
│   ├── .clean-tail             # NEW (N1): atomic clean-tail lower-bound hint
│   ├── 000000000000.seg
│   └── 000000000001.seg
└── <socket_name>               # (ipc) read-plane UDS
```

- **`.clean-tail`** — small atomic JSON sidecar `{ "segment": u64, "offset": u64 }`,
  created/updated at runtime by the intent log. Absent ⇒ treated as "no hint"
  (one full suffix scan). Mode `0o600`.
- **`NT_APPLIED_INTENT` marker node** — gains an **`applied_offset`** string
  property (the opaque order-key). Old markers lacking it are treated fail-safe
  (skipped by prune, never over-pruned).

---

## API reference

All new symbols are `pub(crate)`-or-narrower — no public prune / deterministic-id
/ clean-tail API is exposed, preserving crate SemVer and keeping durability
internals private.

### Bounded ledger — `cognitive_memory/applied_ledger.rs`

```rust
impl CognitiveMemory {
    /// `true` if `intent_id` has a durable applied-marker in the store.
    pub(crate) fn intent_applied(&self, intent_id: Uuid) -> bool;

    /// Record `intent_id` as applied, stamping the opaque log order-key so the
    /// marker can later be pruned below a durable watermark. Idempotent (same
    /// primary key collapses to one node).
    pub(crate) fn mark_intent_applied(
        &mut self,
        intent_id: Uuid,
        applied_key: &str,      // opaque LogOffset order-key
    ) -> Result<()>;

    /// Delete every applied-intent marker whose `applied_offset` is STRICTLY
    /// LESS THAN `watermark`. Idempotent and crash-safe. Markers lacking a
    /// parseable `applied_offset` are left intact (fail-safe: never over-prune).
    pub(crate) fn prune_applied_below(&mut self, watermark: &str) -> Result<()>;

    /// Current count of applied-intent marker nodes (for bound assertions/tests).
    pub(crate) fn applied_intent_count(&self) -> usize;
}
```

### Idempotent links — `cognitive_memory/mod.rs`

```rust
impl CognitiveMemory {
    /// `true` if a live `edge_type` edge already runs source → target. The link
    /// applier (`add_provenance_edge`, `link_similar_facts`) skips the write when
    /// this is already `true`, so a replay never accrues a duplicate edge.
    pub(super) fn edge_of_type_exists(
        &self,
        source_id: &str,
        target_id: &str,
        edge_type: &str,
    ) -> bool;
}
```

### Clean-tail log — `coord/intent_log.rs`

```rust
impl LogOffset {
    /// Total-order string key for cross-layer watermark comparison. Encodes
    /// `segment`/`offset` as **fixed-width zero-padded** fields so lexicographic
    /// string order matches `LogOffset`'s numeric `Ord`. Never leaks the
    /// `LogOffset` type into the `cognitive_memory` layer.
    #[cfg(feature = "persistent")]
    pub(crate) fn order_key(&self) -> String;
}

impl SegmentedLog {
    /// Scan + CRC-validate only the suffix of `index` from `start` forward,
    /// returning the verified clean boundary offset. Preserves F1 torn-tail
    /// truncate / interior-corruption hard-fail.
    fn recover_tail_from(&self, index: u64, start: u64) -> Result<u64>;
}

/// Frames CRC-validated by the most recent append-path recovery — for N1
/// scan-bound tests. A module-level counter, reset via `reset_frames_scanned`.
pub fn frames_scanned() -> u64;
pub fn reset_frames_scanned();
```

### Fenced applier — `coord/applier.rs`

`apply_intent` remains the shared choke point (both `Applier::drain` and
`Coordinator::drain_once` call it), so the F2 idempotency fix lives in one place.
Both paths thread the record's [`LogOffset`] into `mark_intent_applied(id,
pos.order_key())` and call `prune_applied_below(pos.order_key())` **strictly
after** `persist_applied_index`.

---

## Configuration & test seams

There are **no new runtime configuration knobs** — the guarantees are always on
and require no tuning. Two existing/added seams matter for testing:

| Env var | Purpose | Default |
| --- | --- | --- |
| `AMPLIHACK_COORD_CRASH_AFTER_EFFECTS` | **Test-only** crash-injection seam. When set, the applier aborts in the `effect ↔ marker` window (after the effect's last write, before the marker) to exercise the F2 replay path deterministically. | unset (never fires) |

The seam is compiled into the normal path but **inert unless the env var is
set**, so production never carries an artificial abort. Tests set it, kill/restart
the applier, and assert exactly-once.

---

## Tutorials

Each tutorial is a **sketch** — pseudocode that names the assertion, not a
copy-paste test. `applied_intent_count()` and `frames_scanned()` are specified in
the [API reference](#api-reference); the other read-side accessors used below
(`count_nodes(node_type)`, `temporal_index()`, `edges_between(src, tgt)`) are
**illustrative test helpers**. Implementers must confirm the equivalent accessor
already exists on `CognitiveMemory` / the store and use it, or add a small
`#[cfg(test)]` helper — the guarantee under test, not the exact accessor name, is
what each tutorial pins down.

### Tutorial 1 — assert exactly-once across a mid-drain crash (F2)

```rust
// tests/coord_f2_crash_injection.rs (sketch)
// 1. Append > AUTO_CHECKPOINT_WRITES intents so an auto-checkpoint lands mid-drain.
// 2. Set AMPLIHACK_COORD_CRASH_AFTER_EFFECTS=1 and drain in a child; it aborts
//    AFTER an effect's last write but BEFORE its marker (or SIGKILL there).
// 3. Restart the applier (empty `seen`, applied-index NOT advanced past the victim).
// 4. Drain to completion, then assert:
assert_eq!(memory.count_nodes(NT_SEMANTIC)?, expected);      // no duplicate node
assert_eq!(memory.temporal_index(), expected_tidx);          // no double bump
assert_eq!(edges_between(fact, episode).len(), 1);           // no duplicate edge
// This test FAILS without the F2 fix (duplicate node + double temporal_index).
```

### Tutorial 2 — assert the ledger is bounded (N2)

```rust
// tests/coord_n2_ledger_prune.rs (sketch)
// 1. Apply N intents, N >> any prune threshold; drain fully so the index advances.
// 2. Read the bound:
let count = memory.applied_intent_count()?;
assert!(count <= WINDOW_MAX);            // below-watermark markers are gone
// 3. Restart AFTER prune and re-drain the same log prefix:
assert_eq!(applier.drain()?, 0);         // nothing re-applied (index covers them)
// Exactly-once still holds because the applied-index — not the ledger — is the
// durable low-watermark. FAILS without the prune (unbounded ledger).
```

### Tutorial 3 — assert append does not re-CRC the whole segment (N1)

```rust
// tests/coord_n1_scan_bound.rs (sketch)
// 1. Fill the last segment with many clean frames.
// 2. Append once and read the instrumented scan count:
let scanned = log.frames_scanned();
assert!(scanned <= SMALL_CONST);         // independent of segment fill
// 3. Re-run coord_f1_append_recovery.rs UNCHANGED — torn-tail truncate and
//    interior bad-CRC -> LogCorruption still hold.
// FAILS without the clean-tail fast path (scan grows with fill).
```

---

## Invariant → spec map

| Guarantee | TLA+ property (`specs/`) | How it is upheld |
| --- | --- | --- |
| Exactly-once apply (F2) | exactly-once / `PrefixConsistency` | Replay-safe deterministic-id effects + durable ledger marker; both drain paths share `apply_intent`. |
| Bounded ledger (N2) | `PrefixConsistency` (+ ledger-GC action) | Prune strictly below the durable `applied-index` **after** it is persisted; retains the current window. |
| No lost acked write (N1) | `NoLostAckedWrite` | Clean-tail is a lower-bound *hint*; every consumed frame is still CRC-checked; torn-tail truncate / interior-corruption hard-fail preserved (F1). |
| No split brain | `NoSplitBrain` | Unchanged — fence-before-apply (F4 lease/epoch) still gates every apply. |

The model adds a **ledger-GC** action (delete markers below a durable
lower-bound) and a **clean-tail lower-bound** action (monotone advance of a
verified offset) so the checker exercises N2/N1 alongside the existing
invariants.

---

## Security model

- **Disjoint id prefixes.** `sem_ / epi_ / pro_ / aintent_` are fixed and
  disjoint, so an attacker-influenced `intent_id` in one
  family cannot forge another family's primary key. `intent_id` is a validated
  `Uuid` (bounded charset/length); ids flow only through the existing
  escaped-parameter path — never string-interpolated into a query or filesystem
  path.
- **Total, non-panicking parsing.** `applied_offset` and `.clean-tail` parsing is
  total: a malformed/tampered value **fails safe** — prune is *skipped* (never
  over-prunes → never data loss) and the clean-tail hint falls back to a full
  suffix scan.
- **Untrusted hint.** `.clean-tail` is validated (`≤` file length, on a frame
  boundary) before it is trusted; consumed frames are always CRC-checked, so a
  bad hint costs at most one extra scan.
- **Integrity over confidentiality (unchanged model).** All new persistence uses
  atomic `temp → fsync → rename → dir-fsync`. No `intent_id`/payload contents are
  logged in errors, no lock is held across an error path, and replay is
  non-amplifying — the bounded ledger (N2) and bounded scan (N1) are themselves
  DoS mitigations.
- **Private surface.** All new symbols are `pub(crate)`-or-narrower; no durability
  internals are exposed in the public API.

---

## Compatibility & downgrade safety

- **Old markers (no `applied_offset`).** Treated fail-safe by `prune_applied_below`:
  skipped, never over-pruned. The ledger simply converges to the bounded set as
  new pruning passes run.
- **Old log dirs (no `.clean-tail`).** Absence is "no hint" — the first append
  performs one full suffix scan and then writes the sidecar. No migration step.
- **Deterministic ids are additive.** New effects use deterministic ids; existing
  random-id nodes already in a store are untouched and continue to resolve.
- **Safe downgrade.** An older binary ignores `.clean-tail` and the
  `applied_offset` property (extra node property is inert), and its unbounded
  ledger still functions — correctness is preserved, only the N1/N2 optimizations
  are lost until re-upgraded.
- **No public API change; no on-disk format break.** Segment framing, lease, and
  `applied-index` formats are unchanged.

---

### See also

- [`coordination_layer.md`](coordination_layer.md) — the Design C layer this
  hardens (writer client, log, lease, applier, read plane).
- [`durability_and_recovery.md`](durability_and_recovery.md) — the
  `LbugGraphStore` crash/corruption recovery beneath the coord store.
