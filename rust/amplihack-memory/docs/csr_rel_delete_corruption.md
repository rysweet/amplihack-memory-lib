# CSR rel-delete corruption — typed per-rel-type scans (#100) and the unresolved `getGroup(UINT32_MAX)` root cause

Issue **#100** is a native `SIGSEGV` in the pinned LadybugDB engine, the same
`getGroup(UINT32_MAX)` null-pointer dereference family as the
[`DETACH DELETE` crash](safe_node_deletion.md) (#98). This page documents the
**two** things the #100 work delivered:

1. A **read-side + delete-side hardening** — every label-less multi-rel-table
   scan in the `lbug_store` hot path was replaced with **typed, per-rel-type
   scans**, so amplihack-memory never drives lbug's multi-rel-table scanner.
2. A faithful, bisecting **reproduction** that pins the *actual* root cause to an
   lbug **engine** bug: deleting relationships out of a *committed* CSR rel group
   while interleaving checkpoints. That root cause is **not yet resolved**; the
   typed-scan rewrite is hardening, **not** a fix.

> **Read this first.** The typed-scan rewrite removes one of #100's two crash
> *messengers* and is worth keeping, but it does **not** make `delete_edge` /
> retention deletes safe on its own. The unresolved engine bug and its follow-up
> options are in [§4](#4-the-unresolved-root-cause) and [§5](#5-follow-up).

> **Feature gate.** Everything here requires the `persistent` cargo feature
> (which pulls in the `lbug` engine):
> `cargo build --features persistent` / `cargo test --features persistent`.

---

## Table of contents

1. [The two #100 backtraces](#1-the-two-100-backtraces)
2. [The hardening — typed per-rel-type scans](#2-the-hardening--typed-per-rel-type-scans)
3. [What did *not* change](#3-what-did-not-change)
4. [The unresolved root cause](#4-the-unresolved-root-cause)
5. [Follow-up](#5-follow-up)
6. [Testing](#6-testing)

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

---

## 2. The hardening — typed per-rel-type scans

The fix removes **every** label-less rel `MATCH` from the `lbug_store` hot path
(`src/graph/lbug_store/store_impl.rs`) and replaces each with a typed
`MATCH (a)-[r:TYPE]->(b)` fan-out over the rel types the catalog knows about. A
typed scan resolves to a single rel table's CSR storage (lbug's `ScanRelTable`)
and **never** enters the multi-rel-table scanner.

### 2.1 Source of truth — `distinct_rel_names()`

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

### 2.2 Reads — `query_neighbors_directed`

`edge_type = Some(t)` issues a single typed scan as before. `edge_type = None`
now **fans out one typed scan per distinct known rel type and unions the
results**, instead of one label-less scan:

- The edge's type is taken from the **loop variable**, not
  `RelVal::get_label_name` (which only mattered for the old multi-table scan).
- Because every edge lives in exactly one rel table, the union returns each edge
  **exactly once** — matching the old single-query result set. Ordering across
  rel types was never part of the contract.
- `limit` is honored across the union: the loop breaks once `pairs.len() >=
  limit` and the result is truncated to `limit`.
- A failure on one rel type is logged and skipped (best-effort read), consistent
  with the prior read semantics.

`traverse(.., None, ..)` inherits the fix for free — its BFS calls
`query_neighbors` and dedups by visited set, so the per-type fan-out cannot
duplicate nodes or paths.

### 2.3 Deletes — `delete_node` Phase A

The incident-edge strip in [`delete_node`](safe_node_deletion.md) likewise moved
from a single label-less pass to **typed, per-rel-type** directed `DELETE` passes
(outgoing + incoming) over `distinct_rel_names()`, fail-closed on a non-identifier
rel name. An empty rel-type set means no passes run and control falls through to
the plain Phase-B `DELETE`. (`delete_edge` already emitted a typed
`-[r:TYPE]->` and was unaffected.)

---

## 3. What did *not* change

- **No trait or public-API change.** `GraphStore` is untouched; no
  `cognitive_memory/*` consumer changed. The `query_neighbors` / `traverse` /
  `delete_node` contracts (edge type populated, `≤ limit`, all incident edges
  removed or none) are preserved.
- **Other backends are untouched.** Only `lbug` has the C++ CSR multi-rel-table
  scanner, so `in_memory_store`, `hive_store`, `federated_store`, and
  `kuzu_store` keep their label-less semantics. The fix is contained in the
  `lbug_store` brick.
- **Durability/recovery unchanged.** Skipping a CSR node group whose index is the
  `UINT32_MAX` sentinel *before* dereferencing it would be the direct fix, but
  that check lives inside lbug's C++ engine and is **not exposed to the Rust
  bindings**, so it cannot be done here. Avoiding the multi-rel-table scanner via
  typed scans is the feasible, durability-neutral mitigation; the post-write
  barrier and auto-checkpoint bookkeeping are unchanged.

---

## 4. The unresolved root cause

The typed-scan rewrite changes *which* scans are issued; it does not stop the CSR
group from being corrupted in the first place. A bisecting reproduction
(`run_csr_delete_churn` / `reproduces_issue_100_csr_delete_corruption` in
`src/graph/lbug_store/tests.rs`) isolates the trigger:

| Workload | Outcome |
| --- | --- |
| writes only | OK |
| writes + label-less reads | OK |
| writes + **deletes** (interleaved with checkpoints) | **CORRUPT** |
| `delete_edge` alone | sufficient to corrupt |
| deleting **edgeless** nodes | safe |

So the driver is **deleting relationships out of a *committed* CSR rel group**
(`delete_edge`, and `delete_node`'s Phase-A edge strip) while checkpoints are
interleaved. That sets the CSR node group's index to the `UINT32_MAX` sentinel;
the **next** scan to touch that table dereferences it — which can be a checkpoint
flush, a typed read, *or* the label-less read of backtrace 2. The label-less read
is one possible messenger, **not** the cause, which is why removing it does not
remove the crash.

This is **version-independent** — reproduced on lbug `0.15.3`, `0.15.4`, and
`0.17.1` — i.e. an engine bug in CSR relationship deletion plus checkpointing, not
a single-version regression and not something the Rust-side scan rewrite can fix.

**Debug vs release surfacing.** In a debug/assertions build lbug reports the bad
group as a *recoverable* `Storage` error from the bindings containing
`group_collection.h ... groupIdx < groups.size()`. In a release/NDEBUG build the
same condition is the raw null-pointer-deref `SIGSEGV` that kills the consumer
daemon. **Do not run the reproduction under `--release`** — there it aborts the
test binary via `SIGSEGV` instead of returning an error.

---

## 5. Follow-up

Until lbug fixes the engine bug, candidate workarounds (none landed yet):

- **Upstream lbug fix** — guard `getGroup` against the `UINT32_MAX` sentinel /
  fix the committed-CSR rel-delete + checkpoint path. This is the real fix.
- **Rust-side soft-delete** — tombstone edges (e.g. a `deleted` property) and
  filter them in reads instead of issuing `DELETE r` against committed CSR
  groups, so the corrupting operation never runs.
- **Rel-table rebuild** — periodically rebuild affected rel tables out of band
  rather than deleting in place.

The reproduction test is marked `#[ignore]` so CI stays green; **un-ignore it
once the root cause is fixed** (it asserts the desired post-fix outcome: the
delete + checkpoint churn completes with no CSR-corruption error).

---

## 6. Testing

Coverage lives in `rust/amplihack-memory/src/graph/lbug_store/tests.rs` behind the
`persistent` feature.

**Behavioral regression — typed reads (run in CI, must stay green):**

- `label_less_neighbor_read_over_committed_multi_rel_tables_does_not_crash` —
  a `query_neighbors(.., None, ..)` over a node owning *committed* edges in
  several rel tables returns every incident edge of every type exactly once
  (directions correct) and does not crash.
- `label_less_traverse_over_committed_multi_rel_tables_does_not_crash` — the BFS
  `traverse(.., None, ..)` analogue crosses every rel type, visiting each
  reachable node once.
- `consolidation_cycle_with_label_less_reads_does_not_crash` — a full
  consolidation cycle (idempotent edge churn → `delete_edge` → `delete_node` of an
  edge-bearing fact via typed Phase-A deletes → checkpoint → label-less
  reads/traverse) completes consistently.
- `label_less_reads_survive_close_and_reopen_with_committed_csr` — the same reads
  over CSR groups committed in a previous process and freshly reloaded.

**Root-cause reproduction (`#[ignore]`d — reproduces the *unresolved* bug):**

- `reproduces_issue_100_csr_delete_corruption` — drives the delete + checkpoint
  consolidation churn and asserts it completes without corrupting a committed CSR
  rel group. It currently fails (debug: `getGroup(UINT32_MAX)` corruption error;
  release: `SIGSEGV`), so it is ignored until the engine bug is fixed.

Validate the green suite with the feature-gated tests (the original failure was a
probabilistic native crash, so run a few times):

```bash
cargo test  -p amplihack-memory --features persistent --lib graph::lbug_store
cargo build -p amplihack-memory --features persistent
cargo clippy -p amplihack-memory --all-targets --features persistent -- -D warnings
```

> **Pre-commit note.** The configured pre-commit hooks build with default
> features only and do not compile the `persistent` code; the explicit
> `--features persistent` commands above are the authoritative gate.

---

## See also

- [`docs/safe_node_deletion.md`](safe_node_deletion.md) — the resolved #98/#99
  edge-first, no-`DETACH` node deletion (whose Phase A now uses the typed deletes
  described above).
- [`docs/durability_and_recovery.md`](durability_and_recovery.md) — checkpointing,
  recovery, and quarantine behavior.
