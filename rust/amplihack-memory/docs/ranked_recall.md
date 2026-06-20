# Ranked Recall — scored retrieval over `CognitiveMemory`

`CognitiveMemory` can rank recalled memories by a blend of six signals —
**text relevance, confidence, importance, recency, usage, and graph proximity** —
instead of returning only keyword matches sorted by confidence. This page is the
complete reference for the feature: the scoring model, configuration, the public
API, access tracking, the security model, and an end-to-end tutorial.

- **Deterministic.** Scoring is plain arithmetic over fields the crate already
  stores — Jaccard keyword overlap, exponential recency decay, a logarithmic
  usage boost, and neighbour overlap over existing `DERIVES_FROM` / `SIMILAR_TO`
  edges. No embeddings, no model downloads, no network calls.
- **Additive and backward compatible.** `search_facts`, `get_all_facts`, and
  `search_episodes_by_keyword` are unchanged in signature, ordering, and serde
  shape. Ranked recall is a *new* path you opt into; nothing about the existing
  keyword path moves.
- **Same-agent only.** A `CognitiveMemory` instance is scoped to one agent.
  Candidates and graph neighbours that belong to other agents are never scored,
  never returned, and never contribute to the graph term.
- **Lifecycle-aware.** Archived and superseded facts (and compressed episodes)
  are excluded by default; you opt back in per call.
- **Access-tracking.** Ranked recall (and the explicit `record_access`) bump a
  memory's `usage_count` and `last_accessed_at`, persisted across reopen, so
  frequently-used and recently-used memories naturally float to the top over
  time.

---

## Table of contents

1. [When to use it](#when-to-use-it)
2. [How the score is computed](#how-the-score-is-computed)
   - [The six signals](#the-six-signals)
   - [The graph term (neighbour boost)](#the-graph-term-neighbour-boost)
   - [`reasons` — why an item ranked where it did](#reasons--why-an-item-ranked-where-it-did)
3. [Configuration](#configuration)
   - [`RecallWeights`](#recallweights)
   - [`RecallOptions`](#recalloptions)
4. [API reference](#api-reference)
   - [`recall_facts_ranked`](#recall_facts_rankedquery-options---resultvecscoredsemanticfact)
   - [`recall_episodes_ranked`](#recall_episodes_rankedquery-options---resultvecscoredepisodicmemory)
   - [`record_access`](#record_accessnode_id-kind---result)
   - [`Scored<T>`](#scoredt)
   - [`AccessKind`](#accesskind)
   - [Pure scoring primitives](#pure-scoring-primitives)
5. [Candidate gathering, filtering, and ordering](#candidate-gathering-filtering-and-ordering)
6. [Access tracking and persistence](#access-tracking-and-persistence)
7. [Security and tenant isolation](#security-and-tenant-isolation)
8. [Errors](#errors)
9. [Tutorial: ranking a knowledge base](#tutorial-ranking-a-knowledge-base)
10. [Compatibility and guarantees](#compatibility-and-guarantees)

---

## When to use it

| You want to… | Use |
|--------------|-----|
| Retrieve the *most relevant* facts for a query, blending freshness, importance, and usage | [`recall_facts_ranked`](#recall_facts_rankedquery-options---resultvecscoredsemanticfact) |
| Retrieve the most relevant episodes for a query | [`recall_episodes_ranked`](#recall_episodes_rankedquery-options---resultvecscoredepisodicmemory) |
| Have retrieval *learn* over time (used memories rank higher next time) | either ranked method with `record_access: true` (the default) |
| Manually mark a memory as used (e.g. after a non-recall read) | [`record_access`](#record_accessnode_id-kind---result) |
| Plain keyword lookup, confidence-sorted, no side effects | the existing `search_facts` / `search_episodes_by_keyword` (unchanged) |

If you only need exact keyword matches and never care about freshness, usage, or
graph proximity, you do not need ranked recall — the existing keyword search is
still there and still behaves identically.

---

## How the score is computed

For a candidate memory `x` and query `q`, the score is a single **un-normalized**
weighted sum. Higher is better; the absolute value is meaningless — only the
relative ordering matters. The weights (and only the weights) are the tuning
knob.

```text
score(x) = w_text       · keyword_jaccard(q, text(x))
         + w_confidence  · confidence(x)                  // facts only; 0 for episodes
         + w_importance  · importance(x)                  // facts only; 0 for episodes
         + w_recency     · exp_decay(age(x), half_life)
         + w_usage       · usage_boost(usage_count(x))
         + w_graph       · best_edge_score(x, q, max_graph_hops)
```

Because the sum is un-normalized, a strong showing on several signals can
outrank a perfect keyword match that is otherwise stale, low-confidence, and
unused. That is the entire point: a recent, high-importance, frequently-used
fact should beat an old throwaway that merely happens to share a word.

### The six signals

| Signal | Term | Range | Source |
|--------|------|-------|--------|
| Text relevance | `keyword_jaccard(q, text(x))` | `[0, 1]` | `concept + " " + content` for facts; `content` for episodes |
| Confidence | `confidence(x)` | `[0, 1]` | `SemanticFact.confidence` (episodes contribute `0`) |
| Importance | `importance(x)` | `[0, 1]` | `SemanticFact.importance` (episodes contribute `0`) |
| Recency | `exp_decay(age(x), half_life)` | `(0, 1]` | `now − reference_time(x)` |
| Usage | `usage_boost(usage_count(x))` | `[0, ∞)` | `usage_count` property |
| Graph | `best_edge_score(x, q, hops)` | `[0, 1]` | best agent-owned neighbour overlap |

Precise definitions:

- **`keyword_jaccard(q, t)`** — the [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index)
  of the lower-cased, whitespace-tokenized **word sets** of `q` and `t`:
  `|A ∩ B| / |A ∪ B|`. An empty query or empty text yields `0.0` (empty union ⇒
  `0.0`, never a divide-by-zero). Ranked recall does **not** special-case an
  empty query by delegating to `get_all_*`; it ranks every non-excluded
  candidate by the remaining signals.

- **`reference_time(x) = last_accessed_at.unwrap_or(created_at)`**, and
  `age(x) = max(0, now − reference_time(x))` in seconds. Because access updates
  `last_accessed_at`, *using* a memory also refreshes its recency. (To measure
  pure stored-age decay in tests or analytics, recall with
  `record_access: false`.)

- **`exp_decay(age, hl) = 2^(−age / hl)`** — a half-life decay in `(0, 1]`:
  `age = 0 ⇒ 1.0`, `age = hl ⇒ 0.5`, `age = 2·hl ⇒ 0.25`. A non-positive
  half-life (`hl ≤ 0`) is guarded to yield `0.0` (the recency term is switched
  off, never `NaN`).

- **`usage_boost(n) = ln(1 + max(0, n))`** — logarithmic, so the first few uses
  matter a lot and later uses add progressively less (`n ≤ 0 ⇒ 0.0`). Sub-linear
  growth deliberately damps runaway inflation from hot memories.

> **`confidence` vs `importance`.** These are scored as two independent signals,
> but a fact's stored `importance` *defaults to its `confidence`* when not set
> explicitly (`store_fact` / `store_fact_with_provenance` always do this). For
> facts stored that way the two signals move together, so their combined effect is
> roughly `(w_confidence + w_importance)·confidence`. To make `importance` an
> independent dial, store the fact via `upsert_fact` with an explicit
> `FactInput { importance: Some(_), .. }`. Episodes carry neither field and
> contribute `0` to both terms.

### The graph term (neighbour boost)

The graph term pulls a candidate up when it sits next to *other* memories that
match the query, following the edges the crate already maintains:

```text
best_edge_score(x, q, hops) =
    max over agent-owned neighbours n reachable within min(hops, 3) of
        keyword_jaccard(q, text(n)) / hop_distance(n)
    or 0.0 if there are none
```

Traversal direction depends on what you are recalling and which edge type is
followed (`DERIVES_FROM` points fact → episode):

| Recalling | `DERIVES_FROM` | `SIMILAR_TO` |
|-----------|----------------|--------------|
| Facts (`recall_facts_ranked`) | `Outgoing` | `Both` |
| Episodes (`recall_episodes_ranked`) | `Incoming` | `Both` |

- Hop-1 neighbours are gathered directly; `max_graph_hops > 1` runs a bounded
  breadth-first traversal with a visited-set cycle guard. The default is
  **1 hop**, and the effective hop count is always clamped to **≤ 3** to bound
  fan-out (a `max_graph_hops` of `0` disables the graph term entirely).
- Each additional hop divides a neighbour's contribution by its hop distance, so
  closer neighbours dominate.
- **Tenant safety:** any neighbour whose `agent_id` differs from the current
  agent is dropped *before* it can contribute to the score or appear in
  `reasons`. Another agent's graph can never leak in, even via a shared edge.

### `reasons` — why an item ranked where it did

Every `Scored<T>` carries a `reasons: Vec<String>` explaining its score. There is
one entry per **positive** weighted term, for example:

```text
["text 0.40 (jaccard=0.50)",
 "confidence 0.63",
 "importance 0.35",
 "recency 0.40 (age=120s)",
 "usage 0.21 (n=3)",
 "graph 0.30 (SIMILAR_TO hop1)"]
```

If every term is zero, a single baseline entry is emitted
(`"baseline (no positive signals)"`), so `reasons` is **never empty**. Reasons
are intentionally **numeric/label-only**: they may include the weighted term
value, the contributing edge type and hop, and the opaque neighbour `node_id` as
a label, but they **never** embed raw fact/episode content, neighbour body text,
or the query string.

---

## Configuration

### `RecallWeights`

`RecallWeights` is `Copy` and `Serialize`/`Deserialize`. Every weight is an
independent `f64`; set one to `0.0` to switch a signal off, or raise it to make
that signal dominate. Weights are not required to sum to anything.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RecallWeights {
    pub text_relevance: f64,
    pub confidence: f64,
    pub importance: f64,
    pub recency: f64,
    pub usage: f64,
    pub graph: f64,
}
```

| Field | Default | Effect |
|-------|---------|--------|
| `text_relevance` | `1.0` | Weight on keyword Jaccard overlap with the query. |
| `confidence` | `0.7` | Weight on a fact's stored confidence (no effect on episodes). |
| `importance` | `0.5` | Weight on a fact's stored importance (no effect on episodes). |
| `recency` | `0.4` | Weight on exponential recency decay. |
| `usage` | `0.3` | Weight on the logarithmic usage boost. |
| `graph` | `0.6` | Weight on the best neighbour overlap. |

```rust
// Defaults
RecallWeights {
    text_relevance: 1.0,
    confidence: 0.7,
    importance: 0.5,
    recency: 0.4,
    usage: 0.3,
    graph: 0.6,
}

// Example: rank almost purely by freshness + usage, ignore the graph.
let weights = RecallWeights {
    recency: 2.0,
    usage: 1.0,
    graph: 0.0,
    ..RecallWeights::default()
};
```

> **Default weighting note.** With the defaults, `confidence` (`0.7`) and
> `importance` (`0.5`) *both* act on a fact's confidence for the common
> `store_fact` path, because stored `importance` defaults to `confidence` there
> (see the [`confidence` vs `importance` note](#the-six-signals)). So those two
> weights together contribute ≈ `1.2 · confidence` unless you set `importance`
> independently via `upsert_fact`. Tune `importance` *and* store with an explicit
> importance if you want that dial to move on its own.

### `RecallOptions`

`RecallOptions` is `Clone` and `Serialize`/`Deserialize`. Construct it with
`..Default::default()` so future additive fields don't break your call sites.

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecallOptions {
    pub limit: usize,
    pub min_confidence: f64,
    pub include_archived: bool,
    pub include_superseded: bool,
    pub max_graph_hops: usize,
    pub recency_half_life_seconds: f64,
    pub record_access: bool,
    pub weights: RecallWeights,
}
```

| Field | Default | Meaning |
|-------|---------|---------|
| `limit` | `10` | Max items returned (after scoring + sorting). `0` ⇒ `Ok(vec![])`. |
| `min_confidence` | `0.0` | Drop facts below this confidence. **Facts only** — a no-op for episodes. |
| `include_archived` | `false` | When `false`, archived facts are excluded. |
| `include_superseded` | `false` | When `false`, superseded facts (`superseded_by.is_some()`) are excluded. |
| `max_graph_hops` | `1` | Neighbour traversal depth. `0` disables the graph term; clamped to `≤ 3`. |
| `recency_half_life_seconds` | `604_800.0` | Recency half-life (7 days). `≤ 0` switches the recency term off. |
| `record_access` | `true` | When `true`, each returned item gets a `record_access(_, Recall)` bump. Set `false` for a pure, side-effect-free read. |
| `weights` | `RecallWeights::default()` | The six weights above. |

```rust
// Top 5 high-confidence facts, no graph traversal, no side effects.
let opts = RecallOptions {
    limit: 5,
    min_confidence: 0.5,
    max_graph_hops: 0,
    record_access: false,
    ..RecallOptions::default()
};
```

---

## API reference

All three methods take `&mut self` because the default path records access
(writes `usage_count` / `last_accessed_at`). The existing read-only `search_*` /
`get_all_*` methods keep `&self` and are untouched.

### `recall_facts_ranked(query, options) -> Result<Vec<Scored<SemanticFact>>>`

```rust
pub fn recall_facts_ranked(
    &mut self,
    query: &str,
    options: RecallOptions,
) -> Result<Vec<Scored<SemanticFact>>>;
```

Scores every agent-owned semantic fact against `query` using all six signals,
excludes archived/superseded facts and those below `min_confidence` (unless opted
in), sorts by descending score, truncates to `limit`, and — unless
`record_access` is `false` — records a `Recall` access for each returned fact.
Returns a `Vec<Scored<SemanticFact>>` already in rank order.

```rust
use amplihack_memory::{CognitiveMemory, RecallOptions};

let mut mem = CognitiveMemory::new("research-agent")?;
let hits = mem.recall_facts_ranked("rust memory safety", RecallOptions::default())?;
for Scored { item, score, reasons } in &hits {
    println!("{score:.3}  {}  {:?}", item.content, reasons);
}
```

### `recall_episodes_ranked(query, options) -> Result<Vec<Scored<EpisodicMemory>>>`

```rust
pub fn recall_episodes_ranked(
    &mut self,
    query: &str,
    options: RecallOptions,
) -> Result<Vec<Scored<EpisodicMemory>>>;
```

The episode analogue. Episodes carry no confidence or importance, so those terms
contribute `0` and `min_confidence` is a no-op; text relevance, recency, usage,
and the graph term still apply. Compressed episodes are excluded (parity with
`search_episodes_by_keyword`). `usage_count` and `last_accessed_at` are read
directly from the episode's graph node — `EpisodicMemory` has **no field for
either**, so a recorded access persists on the node and shapes the recency/usage
terms of *future* recalls, but is never surfaced on the returned
`Scored<EpisodicMemory>.item`. (`SemanticFact` does carry both fields; see
[access tracking](#access-tracking-and-persistence) for why a recall still returns
the pre-bump value.)

```rust
let hits = mem.recall_episodes_ranked("deployment incident", RecallOptions::default())?;
```

### `record_access(node_id, kind) -> Result<()>`

```rust
pub fn record_access(&mut self, node_id: &str, kind: AccessKind) -> Result<()>;
```

Increments a node's `usage_count` (saturating) and sets `last_accessed_at` to the
current time, persisting both. Works for any node owned by the current agent —
fact or episode. Call it yourself when a memory is consumed outside of ranked
recall (e.g. surfaced through your own UI), so future ranked recalls reflect that
use.

```rust
use amplihack_memory::AccessKind;

mem.record_access(&fact_id, AccessKind::Read)?;
```

Failure modes:

- the node does not exist ⇒ `MemoryError::Storage`;
- the node belongs to a different agent ⇒ `MemoryError::SecurityViolation`
  (**no write happens** — it fails closed);
- the underlying store update fails ⇒ `MemoryError::Storage`.

### `Scored<T>`

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scored<T> {
    pub item: T,
    pub score: f64,
    pub reasons: Vec<String>,
}
```

The ranked wrapper returned by both recall methods. `item` is the unmodified
`SemanticFact` / `EpisodicMemory`; `score` is the un-normalized combined score;
`reasons` explains the score (see [`reasons`](#reasons--why-an-item-ranked-where-it-did)).

### `AccessKind`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessKind { Read, Recall }
```

Serialized as `"Read"` / `"Recall"`. Both variants increment `usage_count`
identically today; the distinction is informational and reserved for future
differential weighting. Ranked recall records `Recall`; use `Read` for manual
non-recall reads.

### Pure scoring primitives

The three building blocks are public and re-exported from the crate root, so you
can reproduce or unit-test the scoring math independently of any store. They are
pure functions — no `self`, no clock, fully deterministic.

```rust
/// Jaccard overlap of lower-cased whitespace word sets. 0.0 on an empty union.
pub fn keyword_jaccard(query: &str, text: &str) -> f64;

/// 2^(-age/half_life), in (0, 1]. 1.0 at age 0; 0.0 when half_life <= 0.
pub fn exp_decay(age_seconds: f64, half_life_seconds: f64) -> f64;

/// ln(1 + max(0, usage_count)). 0.0 at n <= 0; monotonic, sub-linear.
pub fn usage_boost(usage_count: i64) -> f64;
```

```rust
use amplihack_memory::{keyword_jaccard, exp_decay, usage_boost};

assert_eq!(keyword_jaccard("rust async", "rust async runtime"), 2.0 / 3.0);
assert_eq!(exp_decay(0.0, 604_800.0), 1.0);
assert_eq!(exp_decay(604_800.0, 604_800.0), 0.5);
assert_eq!(usage_boost(0), 0.0);
assert!(usage_boost(10) > usage_boost(1));
```

---

## Candidate gathering, filtering, and ordering

A single `recall_*_ranked` call runs these steps:

1. **Gather** all nodes of the requested type (`SemanticMemory` /
   `EpisodicMemory`) **owned by the current agent** — the same agent-scoped query
   path `search_facts` uses. Other agents' nodes are never fetched.
2. **Convert** graph nodes to `SemanticFact` / `EpisodicMemory` via the crate's
   existing converters.
3. **Filter by lifecycle:**
   - *Facts:* drop `archived` unless `include_archived`; drop superseded
     (`superseded_by.is_some()`) unless `include_superseded`; drop
     `confidence < min_confidence`.
   - *Episodes:* archived / superseded / `min_confidence` are no-ops; `compressed`
     episodes are excluded (matching `search_episodes_by_keyword`).
4. **Score** the survivors and **sort by descending score** using a NaN-safe
   total order (`f64::total_cmp`). Ties break by `confidence` descending, then
   `node_id` ascending (for episodes, by `temporal_index` descending then
   `node_id`, since confidence is uniformly `0`). Then `truncate(limit)`.
5. **Record access** (only if `record_access`): for each *returned* (post-truncation)
   item, after scoring, record a `Recall` access. This affects the *next* recall,
   not the current result — the `Scored` items in *this* result carry each memory's
   state **before** its bump. The writes are best-effort and **non-transactional**:
   items are bumped one at a time and the first write failure aborts the loop and
   returns that `Err`. There is no rollback, so a mid-loop failure can leave the
   earlier items already bumped (a partial write) while the caller receives the
   `Err` instead of the ranked `Vec`.

---

## Access tracking and persistence

`record_access` (and the recall side-effect) write exactly two properties on the
target node:

| Property | Type | Written value |
|----------|------|---------------|
| `usage_count` | integer | previous value `saturating_add(1)` (a missing/unparseable value reads as `0`) |
| `last_accessed_at` | unix seconds | current time |

Nothing else is touched — the query and content are never copied into the graph.
Both property keys are static literals. On the in-memory backend the updates live
for the lifetime of the process; on the `persistent` (LadybugDB) backend they are
durable — the two columns are created lazily on first write (**no migration, no
DDL**) and survive `checkpoint()`, `drop`, and reopen. A fact accessed, then
checkpointed, then reopened, comes back with the incremented `usage_count` and
the recorded `last_accessed_at` intact.

A recall's own access writes run **after** scoring and conversion, so the items a
recall returns carry their **pre-bump** values: for a `SemanticFact` the
`usage_count` / `last_accessed_at` you read on a result reflect the state *before*
this call, and the increment first becomes visible on the *next* recall. Episodes
never expose these values on the struct at all (see
[`recall_episodes_ranked`](#recall_episodes_rankedquery-options---resultvecscoredepisodicmemory)),
so for episodes the access is observable only as changed ranking on later recalls.

---

## Security and tenant isolation

Ranked recall is built to be safe under hostile input and multi-agent stores:

- **Agent-scoped candidates.** Recall only ever queries nodes owned by the
  current agent. Agent A's recall can never return agent B's memory.
- **Agent-scoped neighbours.** The graph term filters neighbours by `agent_id`;
  a cross-agent neighbour contributes `0` and never appears in `reasons`.
- **`record_access` fails closed.** Writing to a node owned by another agent is
  refused with `MemoryError::SecurityViolation` and performs no write.
- **Query is data, never code.** The query is only ever tokenized for Jaccard
  comparison. It is never interpolated into Cypher/SQL, an identifier, or a
  filter key.
- **No panics on hostile floats.** `NaN`/`inf` weights or half-lives still
  produce a sorted result — sorting uses `f64::total_cmp`, decay guards
  non-positive half-lives, and parsing failures degrade to `0`. No
  `unwrap`/`expect`/`panic` is reachable from untrusted input.
- **Bounded fan-out.** Graph traversal is capped at 3 effective hops with a
  visited-set, so a densely linked store cannot blow up traversal cost.
- **Leak-free explanations.** `reasons` are numeric/label-only and never log
  payloads.

---

## Errors

| Method | Variant | Trigger |
|--------|---------|---------|
| `record_access` | `MemoryError::Storage` | node id not found |
| `record_access` | `MemoryError::SecurityViolation` | node owned by another agent (no write) |
| `record_access` | `MemoryError::Storage` | underlying store update failed |
| `recall_*_ranked` | *(none)* | normal results, empty store, or `limit = 0` ⇒ `Ok` |
| `recall_*_ranked` | propagated | only when `record_access == true` and a post-rank access write fails (non-transactional — earlier items may already be bumped) |

Degenerate options never error and never panic: a non-positive half-life,
`max_graph_hops = 0`, `limit = 0`, or a negative usage count simply become
zero-contribution terms.

---

## Tutorial: ranking a knowledge base

```rust
use amplihack_memory::{
    CognitiveMemory, RecallOptions, RecallWeights, Scored, AccessKind,
};

fn main() -> amplihack_memory::Result<()> {
    let mut mem = CognitiveMemory::new("research-agent")?;

    // 1. Store a few facts. An old, low-confidence keyword match…
    let rust = ["rust".to_string()];
    let old = mem.store_fact(
        "rust-async",
        "early notes on rust async runtimes",
        0.2, "s-old", Some(&rust), None,
    )?;

    // …and a recent, high-importance, frequently-used fact on the same topic.
    let fresh = mem.store_fact(
        "rust-async",
        "tokio is the de-facto rust async runtime",
        0.9, "s-fresh", Some(&rust), None,
    )?;
    // Bump usage so the fresh fact has a usage history.
    for _ in 0..5 { mem.record_access(&fresh, AccessKind::Read)?; }

    // 2. Rank by the combined score. Despite both matching "rust async",
    //    the fresh/important/used fact ranks first.
    let hits = mem.recall_facts_ranked("rust async runtime", RecallOptions::default())?;
    assert_eq!(hits[0].item.content, "tokio is the de-facto rust async runtime");
    for Scored { item, score, reasons } in &hits {
        println!("{score:.3}  {}\n      {:?}", item.content, reasons);
    }
    let _ = (old,); // `old` still appears, just lower.

    // 3. Tune the weights: ignore the graph, weight freshness heavily,
    //    and take a side-effect-free read of the top 3.
    let opts = RecallOptions {
        limit: 3,
        record_access: false,
        weights: RecallWeights { recency: 2.0, graph: 0.0, ..RecallWeights::default() },
        ..RecallOptions::default()
    };
    let _fresh_first = mem.recall_facts_ranked("rust async", opts)?;

    // 4. Graph boost: a weak fact linked SIMILAR_TO a strong match is pulled up.
    let weak = mem.store_fact(
        "rust-tooling", "cargo is rust's build tool", 0.5, "s-weak", Some(&rust), None,
    )?;
    mem.link_similar_facts(&weak, &fresh, 0.8)?;
    let boosted = mem.recall_facts_ranked(
        "tokio runtime",
        RecallOptions { max_graph_hops: 1, ..RecallOptions::default() },
    )?;
    // `weak` now carries a positive "graph (...)" reason.
    assert!(boosted.iter().any(|s| s.item.content.contains("cargo")
        && s.reasons.iter().any(|r| r.starts_with("graph"))));

    Ok(())
}
```

---

## Compatibility and guarantees

- **Backward compatible.** `search_facts`, `get_all_facts`, and
  `search_episodes_by_keyword` keep their exact signatures, ordering, and serde
  shapes. Ranked recall is purely additive.
- **Both backends, one implementation.** All logic sits behind the existing
  `GraphStore` seam, so behaviour is identical on the default in-memory backend
  and the `--features persistent` (LadybugDB) backend. No new edge types, no
  schema migration, no `GraphStore` trait change, no new dependencies.
- **Deterministic.** For a fixed store, query, options, and clock, the result —
  scores, order, and `reasons` — is fully deterministic.
- **Monotonic in each signal.** With non-negative weights, increasing any single
  signal (importance, usage, fact confidence, keyword overlap, recency, or best
  neighbour overlap) never lowers an item's score.
- **`reasons` is never empty.** Every returned item carries at least one reason.
- **SemVer-additive evolution.** New `RecallOptions` / `RecallWeights` fields are
  introduced via `Default`, so `..Default::default()` call sites keep compiling.
  serde keys are snake_case; `AccessKind` wire values `"Read"` / `"Recall"` are
  stable.

See the crate [`README`](../README.md) for the quick-start summary, and
[`docs/similarity_linking.md`](similarity_linking.md) for the `SIMILAR_TO` edges
that feed the graph term.
