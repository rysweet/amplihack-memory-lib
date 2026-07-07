# Ordered & Trigger-Filtered Prospective Recall

`CognitiveMemory` enumerates prospective memories so that the database
**sorts by priority and then truncates** to your limit — never the reverse — and
exposes a trigger-scoped read (`get_prospective_by_trigger`) that pushes the
`trigger_condition` filter into the query so the limit bounds only *matching*
nodes. This page is the complete reference for the feature: the ordering model,
the new API, the per-backend behaviour, the fail-closed error model, the
security model, and an end-to-end tutorial reproducing the creative-ideas
dashboard that motivated it.

- **Sort-then-truncate, in the database.** `get_all_prospective(limit)` returns
  the true **top-`limit` by priority**. Ordering is pushed into the query
  (`ORDER BY … DESC` before `LIMIT`) so a large store can no longer hand back an
  arbitrary window that happens to omit the rows you asked for.
- **Numeric priority ordering.** `priority` is stored as a string property but
  ordered **numerically** (`CAST(n.priority AS INT64)`), so `10` correctly
  outranks `9` and negative priorities sort below zero — not lexicographically.
- **Trigger-scoped recall.** `get_prospective_by_trigger(trigger, limit)` filters
  on `trigger_condition` **inside** the query, so every matching prospective is
  returned even when the store holds far more than `limit` unrelated nodes.
- **Additive and backward compatible.** `get_all_prospective` keeps its exact
  signature (`-> Vec<ProspectiveMemory>`), serde shape, and priority-descending
  order; existing callers and tests are untouched. The trigger-scoped read and
  the ordered graph primitive are *new* surfaces you opt into.
- **Fail-closed.** The new `Result`-returning API **propagates** a backend read
  error as `Err` instead of masking it as an empty result. No fallbacks, no
  swallowed errors, no false "there are zero ideas."
- **Same-agent only.** A `CognitiveMemory` instance is scoped to one agent; both
  reads always AND in the agent filter, so another agent's prospectives are
  never scored, matched, or returned.

---

## Table of contents

1. [The bug this fixes](#the-bug-this-fixes)
2. [When to use it](#when-to-use-it)
3. [Ordering semantics](#ordering-semantics)
4. [API reference](#api-reference)
   - [`get_prospective_by_trigger`](#get_prospective_by_triggertrigger-limit---resultvecprospectivememory)
   - [`get_all_prospective`](#get_all_prospectivelimit---vecprospectivememory)
   - [`GraphStore::query_nodes_ordered`](#graphstorequery_nodes_ordered)
   - [`ProspectiveMemory`](#prospectivememory)
5. [Per-backend behaviour](#per-backend-behaviour)
6. [Fail-closed error model](#fail-closed-error-model)
7. [Security](#security)
8. [Tutorial: the creative-ideas dashboard](#tutorial-the-creative-ideas-dashboard)
9. [Compatibility and guarantees](#compatibility-and-guarantees)

---

## The bug this fixes

Before this feature, `get_all_prospective(limit)` fetched rows with a bare
`RETURN n LIMIT {limit}` (no `ORDER BY`) and sorted by priority **in Rust,
afterwards**:

```text
DB:    MATCH (n:ProspectiveMemory) WHERE n.agent_id = $a RETURN n LIMIT 512   ← arbitrary 512 rows
Rust:  out.sort_by_key(|pm| Reverse(pm.priority))                             ← too late
```

The database therefore **truncated first and sorted second**. In a store holding
more than `limit` prospective nodes, a non-deterministic subset came back and
newly-written or lower-priority nodes were silently dropped.

Downstream, the creative-ideas dashboard called `get_all_prospective(512)` and
then filtered the result in Rust for a sentinel `trigger_condition`
(`"creative-idea-thread"`). Because creative-idea nodes are low priority, in a
large store they fell **outside** the arbitrary 512-row window — so the dashboard
showed **zero ideas** even though every idea had been persisted. That is the
user-facing symptom: *"creative ideas never show up in the dashboard."*

The fix moves ordering (and the trigger filter) into the query:

```text
DB:  MATCH (n:ProspectiveMemory)
     WHERE n.agent_id = $a AND n.trigger_condition = $t
     RETURN n ORDER BY CAST(n.priority AS INT64) DESC LIMIT 512   ← matching nodes only, sorted first
```

---

## When to use it

| You want to… | Use |
|--------------|-----|
| Get every prospective carrying a specific `trigger_condition` (a dashboard, a channel, a workflow thread), regardless of store size | [`get_prospective_by_trigger`](#get_prospective_by_triggertrigger-limit---resultvecprospectivememory) |
| Get the genuine **top-`limit` by priority** across *all* statuses (e.g. a backup/export or an "N most urgent" view) | [`get_all_prospective`](#get_all_prospectivelimit---vecprospectivememory) |
| Build your own ordered, filtered read over any node type at the graph layer | [`GraphStore::query_nodes_ordered`](#graphstorequery_nodes_ordered) |
| Distinguish a **confirmed-empty** result from a **transient read failure** | any `Result`-returning method here (`get_prospective_by_trigger`, `query_nodes_ordered`) — they fail closed |

If you only ever store a handful of prospectives (fewer than your limit),
`get_all_prospective` already returned everything; the ordering fix simply
guarantees the *order* and the trigger-scoped read saves you the Rust-side
filter.

---

## Ordering semantics

Both reads follow the same contract, and it is the whole point of the feature:

1. **Filter in the query.** The agent filter (and, for the trigger-scoped read,
   `trigger_condition = $trigger`) is applied in the `WHERE` clause. All filters
   are ANDed.
2. **Order in the query.** Matching rows are ordered by
   `CAST(n.priority AS INT64) DESC` — **numeric**, highest priority first.
3. **Truncate in the query.** `LIMIT {limit}` is applied *after* the sort, so you
   receive the top-`limit` matching rows, deterministically.
4. **Stable final sort in Rust.** The mapped `Vec<ProspectiveMemory>` is
   re-sorted `by_key(Reverse(priority))` so the observable Rust ordering is
   identical across every backend, including the in-memory default that sorts in
   Rust.

Why numeric and not lexicographic: `priority` is persisted as a string
(`priority.to_string()`). A string `ORDER BY` would rank `"9"` above `"10"` and
mis-sort negatives. The DB backends therefore `CAST(... AS INT64)`; the in-memory
default parses each `priority` to `i64` symmetrically (an unparsable or missing
value sorts as `i64::MIN`, i.e. last for a descending sort) so ordering matches
across backends.

`limit` semantics: pass `usize::MAX` to mean "no cap." A `limit` of `0` returns an
empty `Vec` (`Ok(vec![])` for the fallible reads).

---

## API reference

### `get_prospective_by_trigger(trigger, limit) -> Result<Vec<ProspectiveMemory>>`

```rust
pub fn get_prospective_by_trigger(
    &self,
    trigger: &str,
    limit: usize,
) -> crate::Result<Vec<ProspectiveMemory>>
```

Return up to `limit` prospective memories for **this agent** whose
`trigger_condition` **equals** `trigger`, ordered by priority descending
(highest first), across every status (`pending` / `triggered` / `resolved`).

A **pure read** (`&self`): it neither mutates node status nor runs the
keyword-overlap heuristic used by [`check_triggers`](../README.md). The
match is an **exact-equality** filter on the `trigger_condition` property, pushed
into the query — not the fuzzy word-overlap that `check_triggers` performs.

Because the filter is in the query, the `LIMIT` bounds only *matching* nodes: a
sentinel-tagged prospective is returned even if the store holds millions of
higher-priority nodes with other triggers.

| Parameter | Type | Description |
|-----------|------|-------------|
| `trigger` | `&str` | The exact `trigger_condition` string to match (e.g. a dashboard sentinel like `"creative-idea-thread"`). |
| `limit` | `usize` | Maximum number of **matching** rows to return (top-`limit` by priority). Use `usize::MAX` for all. |

**Returns** `Ok(Vec<ProspectiveMemory>)` — the top-`limit` matching prospectives,
priority-descending. `Err(MemoryError)` if the backend read fails
(fail-closed — see [Fail-closed error model](#fail-closed-error-model)).

```rust
// Every creative idea, regardless of how large the store is:
let ideas = cog.get_prospective_by_trigger("creative-idea-thread", 512)?;
for idea in &ideas {
    println!("💡 {} → {}", idea.description, idea.action_on_trigger);
}
```

### `get_all_prospective(limit) -> Vec<ProspectiveMemory>`

```rust
pub fn get_all_prospective(&self, limit: usize) -> Vec<ProspectiveMemory>
```

Return up to `limit` prospective memories for this agent, in every status,
ordered by priority (highest first). The signature, serde shape, and
priority-descending order are **unchanged**; what changed is *which* rows you get
when the store holds more than `limit` nodes: it is now the **true top-`limit` by
priority** (sort-then-truncate), not an arbitrary window.

| Parameter | Type | Description |
|-----------|------|-------------|
| `limit` | `usize` | Maximum rows to return (top-`limit` by priority). Use `usize::MAX` for all. |

**Returns** `Vec<ProspectiveMemory>`, priority-descending.

!!! note "Frozen signature, fail-closed via logging"
    This method's `-> Vec` signature is preserved for backward compatibility, so
    it cannot return `Err`. On a genuine backend read error it logs via
    `tracing::error!` (the **error only** — never node payloads) and returns an
    empty `Vec`. A caller that must distinguish a confirmed-empty store from an
    unreadable one should use a `Result`-returning read
    ([`get_prospective_by_trigger`](#get_prospective_by_triggertrigger-limit---resultvecprospectivememory)
    or [`query_nodes_ordered`](#graphstorequery_nodes_ordered)), which
    propagates the error.

```rust
// The five most urgent prospectives, correctly the true top-5:
let top = cog.get_all_prospective(5);
```

### `GraphStore::query_nodes_ordered`

The graph-layer primitive both reads are built on. It is an object-safe trait
method with a default implementation, so every existing `GraphStore` backend gets
correct behaviour for free and only the durable backends override it.

```rust
fn query_nodes_ordered(
    &self,
    node_type: &str,
    filters: Option<&HashMap<String, String>>,
    order_by: &str,
    numeric: bool,
    descending: bool,
    limit: usize,
) -> crate::Result<Vec<GraphNode>>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `node_type` | `&str` | Node label to match (e.g. `"ProspectiveMemory"`). |
| `filters` | `Option<&HashMap<String, String>>` | Equality filters, ANDed into the `WHERE` clause (same semantics as `query_nodes`). |
| `order_by` | `&str` | Property to order by. Must be a valid identifier (see [Security](#security)); an invalid value returns `Err`. |
| `numeric` | `bool` | When `true`, order by the value cast to `INT64` (`CAST(n.{order_by} AS INT64)`) — required for string-stored numbers like `priority`. When `false`, order lexicographically. |
| `descending` | `bool` | `true` for `DESC` (highest first), `false` for `ASC`. |
| `limit` | `usize` | Max rows **after** ordering. |

**Returns** `Ok(Vec<GraphNode>)` ordered and truncated by the backend, or
`Err(MemoryError)` on a read failure or an invalid `order_by`.

**Contract:** ordering and truncation happen **together, in the backend**, so the
`LIMIT` always applies *after* the sort. `query_nodes` (the original, infallible,
unordered method) is unchanged and still available.

Default implementation (in-memory / test / composite backends that cannot fail):
fetches all matching rows (`query_nodes(node_type, filters, usize::MAX)`), sorts
them in Rust by `order_by` (parsing to `i64` when `numeric`; missing/unparsable →
`i64::MIN`), applies `descending`, then truncates to `limit`, and returns `Ok`.

### `ProspectiveMemory`

Unchanged by this feature; reproduced here for convenience.

| Field | Type | Description |
|-------|------|-------------|
| `node_id` | `String` | Unique graph identifier |
| `description` | `String` | Human-readable description of the intention |
| `trigger_condition` | `String` | Condition/sentinel that fires the action (the field `get_prospective_by_trigger` matches on) |
| `action_on_trigger` | `String` | Action to perform when triggered |
| `status` | `String` | `pending` / `triggered` / `resolved` |
| `priority` | `i32` | Priority (higher = more urgent); the field both reads order by |
| `created_at` | `DateTime<Utc>` | When the intention was recorded |

---

## Per-backend behaviour

`query_nodes_ordered` has a correct default; only the two durable backends and
the composite override it to push ordering into their query engines.

| Backend | Feature flag | `query_nodes_ordered` |
|---------|-------------|-----------------------|
| **In-memory** | default | Inherits the default: fetch-all → Rust sort → truncate. |
| **Hive** (read-only composite member) | default | Inherits the default. |
| **Kuzu** | `kuzu` | Override: emits `MATCH (n:{type}){WHERE …} RETURN n ORDER BY CAST(n.{order_by} AS INT64) DESC LIMIT {limit}` (plain `n.{order_by}` when `!numeric`; `ASC` when `!descending`). Filter values bind via `$fN` parameters. `order_by` validated first; a query-engine error propagates as `Err`. |
| **LadybugDB (lbug)** | `persistent` | Override: same ordered-limit shape, reusing the existing equality-filter and tombstone (soft-delete) handling, with `ORDER BY CAST(n.{order_by} AS INT64) DESC` inserted before the limit clause. `order_by` validated first; a read error propagates as `Err`. |
| **Federated** | — | Override: calls each member's `query_nodes_ordered(…, limit)?` (local + hive), merges, `deduplicate_nodes`, then **re-sorts globally** by `order_by`/`descending` and truncates to `limit`. The global top-`limit` is a subset of the union of each member's top-`limit`, so a per-shard limit followed by a global re-sort yields the correct global order. A member error propagates as `Err`. |

The federated override must re-sort **before** truncating; sorting per-shard and
merely concatenating would re-introduce the truncation bug across shards.

---

## Fail-closed error model

The original bug is a *false empty* — a store that is actually populated reads
back as "zero ideas." This feature is deliberately fail-closed so an unreadable
store never masquerades as an empty one:

| Surface | On read error | Rationale |
|---------|---------------|-----------|
| `get_prospective_by_trigger` | Propagates `Err(MemoryError)` via `?` | New API; `Result` is the correct fail-closed shape. Never returns `Ok(vec![])` on error. |
| `GraphStore::query_nodes_ordered` (DB overrides) | Propagates `Err(MemoryError)` | A false-empty here is exactly the original bug class. |
| `get_all_prospective` | Logs `tracing::error!` (error only), returns empty `Vec` | Signature is frozen at `-> Vec`, so it surfaces via logging rather than a swallowed-but-silent empty. |

There are **no fallbacks**: the fallible paths do not degrade to an unordered or
empty result on error, and they never retry against a different backend. An
invalid `order_by` is an `Err(MemoryError::InvalidInput(...))`, not a silent
switch to unordered results.

---

## Security

The only new interpolated value is `order_by`, which is spliced into the
`ORDER BY` clause. Every database override **validates it before interpolation**
with the crate's existing identifier validator (`validate_identifier`, i.e.
`^[A-Za-z_][A-Za-z0-9_]*$`); an invalid value returns
`Err(MemoryError::InvalidInput(...))` and the query is never executed.

- **No new validator is introduced** — the feature reuses the audited
  `validate_identifier` / `is_valid_identifier` helpers already guarding
  `node_type` and filter keys, avoiding validation drift.
- **Filter *values* remain parameter-bound** (Kuzu `$fN` bindings; lbug's
  existing escaping) exactly as in `query_nodes`; the trigger string and agent
  name are values, never interpolated into Cypher.
- `numeric`, `descending` are `bool` and `limit` is `usize` — type-safe, no
  injection surface.
- **No panics on malformed data:** priority parsing in the default impl is a
  checked parse (unparsable → `i64::MIN`), so a corrupt `priority` value cannot
  panic the read (a panic would be a denial-of-service).
- **Log hygiene:** the frozen `get_all_prospective` logs the error only — never
  node contents or PII.

Injection attempts such as `order_by = "x) DELETE n //"`, `"1;DROP"`, `"a b"`, or
`""` are all rejected with `Err(InvalidInput)` and never reach the query engine.

---

## Tutorial: the creative-ideas dashboard

This reproduces the scenario the feature was built for: a store dominated by
high-priority operational prospectives, plus a handful of low-priority
creative-idea prospectives that a dashboard must surface.

```rust
use amplihack_memory::CognitiveMemory;

const SENTINEL: &str = "creative-idea-thread";

fn main() -> amplihack_memory::Result<()> {
    let mut cog = CognitiveMemory::new("simard")?;

    // 590 high-priority operational prospectives with an unrelated trigger.
    for i in 0..590 {
        cog.store_prospective(
            &format!("ops alert {i}"),
            "deployment failed error",   // unrelated trigger
            "page on-call",
            /* priority = */ 100,        // HIGH — these dominate any top-N window
        )?;
    }

    // 10 low-priority creative ideas carrying the dashboard sentinel.
    for i in 0..10 {
        cog.store_prospective(
            &format!("idea {i}: what if the agent journaled its dead ends?"),
            SENTINEL,                    // the trigger the dashboard filters on
            "surface in the ideas dashboard",
            /* priority = */ 1,          // LOW — falls outside an arbitrary window
        )?;
    }

    // ✅ With the trigger filter in the query, all 10 ideas come back —
    //    regardless of the 590 higher-priority nodes.
    let ideas = cog.get_prospective_by_trigger(SENTINEL, 512)?;
    assert_eq!(ideas.len(), 10);
    for idea in &ideas {
        println!("💡 {}", idea.description);
    }

    // ⚠️ Contrast — enumerating the top-512 by priority and filtering in Rust
    //    (the old dashboard approach) returns ZERO ideas: the 10 low-priority
    //    sentinel nodes sit outside the top-512 window the ops alerts dominate.
    //    Ordering get_all_prospective correctly does NOT rescue this access
    //    pattern — only pushing the trigger filter into the query (above) does.
    //    (Under the prior truncate-then-sort code this count was also
    //    non-deterministic: some ideas could land in the arbitrary window.)
    let leaked = cog
        .get_all_prospective(512)
        .into_iter()
        .filter(|p| p.trigger_condition == SENTINEL)
        .count();
    assert_eq!(leaked, 0); // enumerate-then-filter cannot surface low-priority matches

    cog.close();
    Ok(())
}
```

And the top-N ordering guarantee, independent of insertion order (a fresh store):

```rust
// Insert in ascending priority so a truncate-then-sort backend would have
// returned the *lowest* priorities. Ordered recall returns the highest.
cog.store_prospective("low",  "t", "a", 2)?;
cog.store_prospective("mid",  "t", "a", 9)?;
cog.store_prospective("high", "t", "a", 10)?;   // 10 > 9 numerically, not "10" < "9"
cog.store_prospective("top",  "t", "a", 100)?;

let top2 = cog.get_all_prospective(2);
assert_eq!(top2.iter().map(|p| p.priority).collect::<Vec<_>>(), vec![100, 10]);
```

---

## Compatibility and guarantees

- **Additive.** `get_all_prospective` keeps its exact signature, serde shape, and
  priority-descending order. `query_nodes` is untouched. `query_nodes_ordered`
  and `get_prospective_by_trigger` are new. Existing tests
  (`test_prospective_priority_ordering`,
  `test_get_all_prospective_returns_every_status_without_mutating`) continue to
  pass unchanged.
- **Object-safe trait extension.** `query_nodes_ordered` ships with a default
  impl, so third-party `GraphStore` implementers need no changes (MINOR semver).
- **Deterministic ordering** across every backend: DB backends sort-then-truncate
  numerically in-engine; the in-memory default reproduces the same order in Rust;
  the final Rust re-sort makes the observable order backend-independent.
- **Rust-only surface.** These reads are additive Rust APIs; the Python bindings
  are unchanged (there is no Python binding for prospective enumeration).
- **No `println!`/`eprintln!`** in production paths — diagnostics go through
  `tracing`.

See also: the crate [`README`](../README.md) for the `store_prospective` /
`check_triggers` / `resolve_prospective` quick-start, and
[Ranked Recall](ranked_recall.md) for scored retrieval over semantic and
episodic memory.
