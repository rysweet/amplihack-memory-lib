# CSR rel-delete corruption (#100) — rebuild-on-delete + typed per-rel-type scans

Issue **#100** was a native `SIGSEGV` in the pinned LadybugDB engine, the same
`getGroup(UINT32_MAX)` null-pointer dereference family as the
[`DETACH DELETE` crash](safe_node_deletion.md) (#98). It crashed the consumer
daemon every ~45–90 min during retention/dedup consolidation.

The #100 work delivered **two** complementary changes:

1. **The fix — rebuild-on-delete.** `delete_edge` and `delete_node`'s incident-edge
   strip no longer issue an in-place `DELETE r` against a committed CSR rel group
   (the operation that corrupts the group). They instead **rebuild** the affected
   rel table: copy the surviving edges into a scratch table and atomically swap it
   in. This removes the corrupting operation entirely.
2. **Read-side hardening — typed per-rel-type scans.** Every label-less
   multi-rel-table scan in the `lbug_store` read hot path was replaced with
   **typed, per-rel-type scans**, so amplihack-memory never drives lbug's
   multi-rel-table scanner (one of #100's two crash backtraces).

> **Feature gate.** Everything here requires the `persistent` cargo feature
> (which pulls in the `lbug` engine):
> `cargo build --features persistent` / `cargo test --features persistent`.

---

## Table of contents

1. [The two #100 backtraces](#1-the-two-100-backtraces)
2. [The root cause](#2-the-root-cause)
3. [The fix — rebuild-on-delete](#3-the-fix--rebuild-on-delete)
4. [Read-side hardening — typed per-rel-type scans](#4-read-side-hardening--typed-per-rel-type-scans)
5. [What did *not* change](#5-what-did-not-change)
6. [Trade-offs and follow-up](#6-trade-offs-and-follow-up)
7. [Testing](#7-testing)

---

## 1. The two #100 backtraces

Both fault on the same null `unique_ptr` deref —
`getGroup(groupIdx = 4294967295 / UINT32_MAX)` inside a CSR rel group — but reach
it from different call sites:

**Backtrace 1 — `DETACH DELETE` (already handled by #98/#99).**

```
getGroup(UINT32_MAX) -> null deref
  <- CSRNodeGroup::scanCommittedInMemRandom
  <- RelTable::detachDeleteForCSRRels
  <- RelTable::detachDelete            (Cypher DETACH DELETE)
```

The [edge-first, no-`DETACH` deletion](safe_node_deletion.md) already keeps
amplihack-memory out of `detachDeleteForCSRRels` entirely.

**Backtrace 2 — label-less multi-rel-table *read* (the #100 read scan).**

```
getGroup(UINT32_MAX) -> null deref
  <- CSRNodeGroup::scanCommittedInMemRandom
  <- RelTableCollectionScanner::scan
  <- ScanMultiRelTable::getNextTuplesInternal   (MATCH (a)-[r]->(b), no rel label)
```

A `MATCH (a)-[r]->(b)` with **no rel-type label** fans across *every* rel table at
once via lbug's multi-rel-table scanner. The `lbug_store` read path
(`query_neighbors(.., None, ..)` and the `traverse(.., None, ..)` BFS that calls
it) used to emit exactly this shape.

Both backtraces are only *messengers*: they dereference a CSR group whose index
has **already** been set to the `UINT32_MAX` sentinel. The corruption happens
earlier — see §2.

---

## 2. The root cause

A bisecting reproduction (`run_csr_delete_churn` in
`src/graph/lbug_store/tests.rs`) isolates the trigger:

| Workload | Outcome |
| --- | --- |
| writes only | OK |
| writes + label-less reads | OK |
| writes + **deletes** (interleaved with checkpoints) | **CORRUPT** |
| `delete_edge` alone | sufficient to corrupt |
| deleting **edgeless** nodes | safe |

The driver is **deleting relationships out of a *committed* CSR rel group**
(`delete_edge`, and `delete_node`'s Phase-A edge strip) while checkpoints are
interleaved. That sets the CSR node group's index to the `UINT32_MAX` sentinel;
the **next** scan to touch that table dereferences it — a checkpoint flush, a
typed read, *or* the label-less read of backtrace 2.

This is **version-independent** — reproduced on lbug `0.15.3`, `0.15.4`, and
`0.17.1` — i.e. an engine bug in committed-CSR relationship deletion plus
checkpointing. Skipping a CSR node group whose index is the `UINT32_MAX` sentinel
*before* dereferencing it would be the direct engine fix, but that check lives
inside lbug's C++ and is not exposed to the Rust bindings. The feasible Rust-side
fix is therefore to **never perform the corrupting operation** — see §3.

**Debug vs release surfacing.** In a debug/assertions build lbug reports the bad
group as a *recoverable* `Storage` error from the bindings containing
`group_collection.h ... groupIdx < groups.size()`. In a release/NDEBUG build the
same condition is the raw null-pointer-deref `SIGSEGV` that killed the consumer
daemon.

---

## 3. The fix — rebuild-on-delete

`delete_edge` and `delete_node`'s incident-edge strip never delete relationship
rows in place. To remove a set of edges from rel table `T`, they **rebuild** `T`:

1. **Snapshot survivors** — `MATCH (a)-[r:T]->(b) WHERE <keep>` returns the
   endpoint ids and every (`STRING`) column of the edges that must *survive*
   (`<keep>` is "not incident to the deleted node", or "not the deleted edge").
2. **Build a scratch table** — `CREATE REL TABLE T__rebuild_tmp(FROM … TO …, …)`
   and re-insert the survivors into it. The original `T` is still fully intact.
3. **fsync** — the durability barrier makes the scratch table durable while the
   original still holds every edge, so a crash here loses nothing (the delete
   simply has not happened yet).
4. **Swap** — `DROP TABLE T; ALTER TABLE T__rebuild_tmp RENAME TO T;` then fsync.

This is implemented in `LbugGraphStore::rebuild_rel_table_keeping`
(`src/graph/lbug_store/mod.rs`). Because the deleted edges are simply absent from
the rebuilt table, lbug never runs an in-place rel delete against a committed CSR
group, so the group is never corrupted.

### 3.1 Survivor safety and crash recovery

lbug DDL (`DROP` / `CREATE` / `ALTER … RENAME`) auto-commits, so the rebuild
cannot be wrapped in one explicit transaction. The ordering above keeps survivors
safe regardless: the original table is dropped **only after** the scratch table is
built and fsync'd. The single crash-loss window is between the `DROP` and the
`RENAME`. A leftover `T__rebuild_tmp` from a crash in that window is recovered on
the next open by `recover_rebuild_tmp_tables` (called from `ensure_schema_loaded`):

- If the canonical `T` still exists, the scratch table is a **pre-swap straggler**
  (the original is intact, the delete never took effect) → **drop** it.
- If `T` is missing, the crash landed mid-swap and the rebuilt survivors live only
  in the scratch table → **promote** it (`RENAME T__rebuild_tmp TO T`).

Either way no acknowledged edge is lost, and the catalog converges to a clean
state. This is strictly *more* durable than the pre-fix behaviour, where the
delete corrupted the catalog and forced a quarantine + rebuild
([durability_and_recovery.md](durability_and_recovery.md)).

### 3.2 The edgeless fast path

Deleting an **edgeless** node is safe (bisection: it never corrupts a CSR group),
so `delete_node` keeps the plain `DELETE n` fast path for it. Only rel tables the
node actually has an incident edge in are rebuilt (`rel_has_incident_node` gates
each rebuild). The common working-/sensory-memory eviction case — evicting a node
with no relationships — therefore rebuilds nothing.

---

## 4. Read-side hardening — typed per-rel-type scans

Independently of the delete fix, every label-less rel `MATCH` was removed from the
`lbug_store` read hot path (`store_impl.rs`) and replaced with a typed
`MATCH (a)-[r:TYPE]->(b)` fan-out over the rel types the catalog knows about. A
typed scan resolves to a single rel table's CSR storage (lbug's `ScanRelTable`)
and **never** enters the multi-rel-table scanner of backtrace 2.

### 4.1 Source of truth — `distinct_rel_names()`

The known rel types come from the `known_rel_tables` catalog cache (keyed
`(rel_name, from, to)`, populated reopen-safely by `ensure_schema_loaded`). A rel
name can appear with several endpoint pairs, but each edge lives in exactly one
rel table, so the helper collapses to **distinct, sorted names**:

```rust
fn distinct_rel_names(&self) -> Vec<String> {
    let mut names: Vec<String> = self
        .known_rel_tables
        .borrow()
        .iter()
        .map(|(rel, _, _)| rel.clone())
        .collect();
    names.sort();
    names.dedup();
    names
}
```

Catalog names are always valid identifiers; callers still re-check with
`is_valid_identifier` before interpolating a name as a bare rel label, and skip
(reads) or fail-closed (deletes) anything that is not.

### 4.2 Reads — `query_neighbors_directed`

`edge_type = Some(t)` issues a single typed scan as before. `edge_type = None`
now **fans out one typed scan per distinct known rel type and unions the
results**. Because every edge lives in exactly one rel table the union returns
each edge **exactly once** (ordering across rel types was never part of the
contract), `limit` is honored across the union, and a failure on one rel type is
logged and skipped (best-effort read). `traverse(.., None, ..)` inherits the fix
for free — its BFS calls `query_neighbors` and dedups by visited set.

---

## 5. What did *not* change

- **No trait or public-API change.** `GraphStore` is untouched; no
  `cognitive_memory/*` consumer changed. The `query_neighbors` / `traverse` /
  `delete_node` / `delete_edge` contracts (edge type populated, `≤ limit`, all
  incident edges removed or none, idempotent no-op delete of a missing edge) are
  preserved.
- **Other backends are untouched.** Only `lbug` has the C++ CSR storage, so
  `in_memory_store`, `hive_store`, `federated_store`, and `kuzu_store` keep their
  existing semantics. The fix is contained in the `lbug_store` brick.
- **Durability/recovery is preserved and extended.** The per-write barrier and
  auto-checkpoint bookkeeping are unchanged; the rebuild adds its own fsync before
  the swap and a `__rebuild_tmp` recovery step on open (§3.1). It never loses an
  acknowledged write.

---

## 6. Trade-offs and follow-up

- **Cost.** A rebuild is `O(edges in the affected rel table)` per delete, and the
  reinsert is one `CREATE` per surviving edge. Deletes are infrequent
  (retention/dedup, eviction) and the edgeless fast path skips them entirely, so
  this trades a rare, bounded slow-down for the elimination of a daemon-killing
  crash. Very large rel tables with frequent edge-bearing deletes are the worst
  case; batch the survivor reinsert (`UNWIND`) if that ever becomes hot.
- **Upstream lbug fix still desirable.** Guarding `getGroup` against the
  `UINT32_MAX` sentinel / fixing the committed-CSR rel-delete + checkpoint path
  would let us delete in place again and drop the rebuild. The rebuild stands on
  its own until then.

---

## 7. Testing

Coverage lives in `rust/amplihack-memory/src/graph/lbug_store/tests.rs` behind the
`persistent` feature.

**Root-cause regression (runs in CI, must stay green):**

- `csr_delete_checkpoint_churn_stays_consistent` — the former `#[ignore]`d
  reproduction, now a passing guard. Drives the delete + checkpoint consolidation
  churn that corrupted a committed CSR rel group within ~10 rounds pre-fix; with
  rebuild-on-delete it completes cleanly. A regression that reintroduces an
  in-place rel delete fails it (debug: `getGroup(UINT32_MAX)` corruption error;
  release: `SIGSEGV` aborts the binary).

**Rebuild correctness:**

- `delete_edge_rebuild_preserves_other_edges_and_props` — deleting one edge keeps
  every other edge (and its properties) across a checkpoint and a reopen.
- `delete_missing_edge_is_noop_success` — deleting a non-existent edge is an
  idempotent success that does not disturb the table.
- `delete_node_rebuild_preserves_unrelated_edges` — deleting an edge-bearing node
  removes only its incident edges, leaving unrelated edges in the same and other
  rel tables intact.
- `delete_edgeless_node_fast_path_after_fix` — an edgeless node deletes via the
  plain-`DELETE` fast path with rel tables present.
- `stale_rebuild_tmp_is_dropped_on_open` / `interrupted_rebuild_tmp_is_promoted_on_open`
  — the `__rebuild_tmp` crash-recovery cases (§3.1).

**Behavioral regression — typed reads:**

- `label_less_neighbor_read_over_committed_multi_rel_tables_does_not_crash`,
  `label_less_traverse_over_committed_multi_rel_tables_does_not_crash`,
  `consolidation_cycle_with_label_less_reads_does_not_crash`,
  `label_less_reads_survive_close_and_reopen_with_committed_csr` — the typed
  per-rel-type read fan-out returns every incident edge exactly once and never
  enters the multi-rel-table scanner.

Validate the green suite with the feature-gated tests (the original failure was a
probabilistic native crash, so run a few times, including `--release` for the
NDEBUG configuration the daemon uses):

```bash
cargo test  -p amplihack-memory --features persistent --lib graph::lbug_store
cargo test  -p amplihack-memory --release --features persistent --lib graph::lbug_store
cargo build -p amplihack-memory --features persistent
cargo clippy -p amplihack-memory --features persistent -- -D warnings
```

> **Pre-commit note.** The configured pre-commit hooks build with default
> features only and do not compile the `persistent` code; the explicit
> `--features persistent` commands above are the authoritative gate.

---

## See also

- [`docs/safe_node_deletion.md`](safe_node_deletion.md) — the resolved #98/#99
  edge-first, no-`DETACH` node deletion (whose Phase A now rebuilds incident rel
  tables rather than deleting edges in place).
- [`docs/durability_and_recovery.md`](durability_and_recovery.md) — checkpointing,
  recovery, and quarantine behavior.
