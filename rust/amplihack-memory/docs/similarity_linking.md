# Automatic `SIMILAR_TO` Linking Between Facts

`CognitiveMemory` can automatically connect related semantic facts with
`SIMILAR_TO` edges so that knowledge about the same topic becomes traversable —
without any manual bookkeeping. This page is the complete reference for the
feature: how similarity is scored, how to configure linking, the public API, the
edge data model, and an end-to-end tutorial.

- **Deterministic.** Similarity is computed with the crate's existing
  `similarity` helpers — plain text/tag/concept Jaccard math. No embeddings, no
  model downloads, no network calls.
- **Additive and opt-in.** `store_fact` and `store_fact_with_provenance` are
  unchanged and never create `SIMILAR_TO` edges on their own. You choose when
  linking happens.
- **Same-agent only.** A `CognitiveMemory` instance is scoped to one agent;
  linking only ever considers facts owned by that agent. Facts belonging to
  other agents are never connected.
- **Idempotent.** Re-running any entry point never duplicates an existing edge.
- **Write-only for now.** This feature *creates* `SIMILAR_TO` edges; it does not
  yet add a public API to read them back. See
  [Reading edges back](#reading-edges-back).

---

## Table of contents

1. [When to use it](#when-to-use-it)
2. [How similarity is computed](#how-similarity-is-computed)
3. [Configuration — `SimilarityOptions`](#configuration--similarityoptions)
4. [API reference](#api-reference)
   - [`auto_link_similar_facts`](#auto_link_similar_factsfact_id-options---resultusize)
   - [`rebuild_similarity_links`](#rebuild_similarity_linksoptions---resultsimilarityreport)
   - [`store_fact_with_options`](#store_fact_with_options---resultstring)
   - [`SimilarityReport`](#similarityreport)
   - [`StoreFactOptions`](#storefactoptions)
5. [Edge data model](#edge-data-model)
   - [Reading edges back](#reading-edges-back)
6. [Idempotency and directionality](#idempotency-and-directionality)
7. [Persistence](#persistence)
8. [Tutorial: building a linked knowledge base](#tutorial-building-a-linked-knowledge-base)
9. [Relationship to manual `link_similar_facts`](#relationship-to-manual-link_similar_facts)
10. [Compatibility and guarantees](#compatibility-and-guarantees)

---

## When to use it

| You want to… | Use |
|--------------|-----|
| Link one freshly created or known fact to its neighbours | [`auto_link_similar_facts`](#auto_link_similar_factsfact_id-options---resultusize) |
| Backfill links across an entire agent's existing facts (import, migration, threshold change) | [`rebuild_similarity_links`](#rebuild_similarity_linksoptions---resultsimilarityreport) |
| Wire up every new fact automatically as it is stored | [`store_fact_with_options`](#store_fact_with_options---resultstring) |
| Connect two specific facts with a score you computed yourself | the existing `link_similar_facts(a, b, score)` |

If you only ever search facts by keyword (`search_facts`) and never traverse the
graph, you do not need `SIMILAR_TO` edges at all — linking is purely additive.

---

## How similarity is computed

Two facts are compared with the crate's composite **Jaccard** similarity
(`amplihack_memory::similarity::compute_similarity`). The score is a weighted
blend of three independent Jaccard overlaps:

```text
score = 0.5 · word_overlap(content_a, content_b)
      + 0.2 · tag_overlap(tags_a,    tags_b)
      + 0.3 · concept_overlap(concept_a, concept_b)
```

| Component | Weight | Source field | Overlap of |
|-----------|--------|--------------|------------|
| Word      | 0.5    | `content`    | tokenized word sets (see below) |
| Tag       | 0.2    | `tags`       | lower-cased, trimmed tag sets |
| Concept   | 0.3    | `concept`    | tokenized word sets (same tokenizer as `content`) |

**Tokenization matters.** Both the word and concept components run their text
through the crate's tokenizer before the Jaccard comparison. Tokenization
lower-cases the text, splits on whitespace, strips surrounding punctuation
(`.,;:!?()[]{}"'`), and then **removes English stop words** (`the`, `is`, `of`,
`with`, `and`, …) and **any token of two characters or fewer**. Only the
remaining meaningful words count toward overlap — short connective words and
one/two-letter tokens are ignored entirely. This affects tuning: two facts that
differ only in stop words score higher than a naive word-for-word comparison
would suggest, while very short distinguishing tokens contribute nothing.

The result is always finite and in `[0.0, 1.0]`. An edge is created when

```text
score >= options.threshold      (inclusive)
```

### Picking a threshold

Because word overlap is capped at `0.5` of the total, a high threshold like the
default **`0.60`** effectively requires facts to share *both* substantial
content overlap *and* an overlapping concept (and ideally a shared tag).
Unrelated facts score at or near `0.0`.

| Threshold | Behaviour |
|-----------|-----------|
| `0.40`    | Loose — links facts that merely share a topic word or two. More edges, more noise. |
| `0.60` (default) | Balanced — links facts that clearly describe the same thing. |
| `0.80`    | Strict — near-duplicates only. |

Tune the threshold to your corpus and re-run
[`rebuild_similarity_links`](#rebuild_similarity_linksoptions---resultsimilarityreport)
to apply the change. Because rebuild is additive, **lowering** the threshold and
re-running adds the newly qualifying edges; raising it does not remove existing
edges (delete those manually if you need to).

---

## Configuration — `SimilarityOptions`

```rust
pub struct SimilarityOptions {
    pub enabled: bool,
    pub threshold: f64,
    pub candidate_limit: usize,
    pub bidirectional: bool,
}
```

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `enabled` | `bool` | `true` | Master switch. When `false`, every entry point becomes an inert no-op (returns `Ok(0)` / an empty report and writes nothing). Lets a single options value also gate the store-time hook. |
| `threshold` | `f64` | `0.60` | Inclusive composite score (`>=`) required to create an edge. Valid range `[0.0, 1.0]`. |
| `candidate_limit` | `usize` | `50` | Upper bound on how many *other* same-agent facts are scored per source fact. Candidates are taken **highest-confidence first** (the same order as `get_all_facts`), then truncated to this limit — so on a corpus larger than the limit only the most-confident facts are considered, and a textually similar but low-confidence fact beyond the cutoff may never be scored. Bounds cost on large corpora. |
| `bidirectional` | `bool` | `true` | When `true`, also create the reciprocal `B → A` edge so directional (`Outgoing`) queries find the pair from either side. |

`SimilarityOptions` derives `Debug, Clone, Copy, PartialEq` (it contains an
`f64`, so it is intentionally **not** `Eq`/`Hash`). `Default` yields
`{ enabled: true, threshold: 0.60, candidate_limit: 50, bidirectional: true }`.

```rust
use amplihack_memory::SimilarityOptions;

// Defaults.
let opts = SimilarityOptions::default();

// A stricter, one-directional, capped configuration.
let strict = SimilarityOptions {
    threshold: 0.75,
    candidate_limit: 20,
    bidirectional: false,
    ..SimilarityOptions::default()
};
```

All methods take `options` by reference (`&SimilarityOptions`) for signature
consistency even though the type is `Copy`; pass `&SimilarityOptions::default()`
for the defaults.

---

## API reference

All three methods are inherent methods on `CognitiveMemory` and return
`amplihack_memory::Result<T>` (`Result<T, MemoryError>`).

### `auto_link_similar_facts(fact_id, options) -> Result<usize>`

```rust
pub fn auto_link_similar_facts(
    &mut self,
    fact_id: &str,
    options: &SimilarityOptions,
) -> Result<usize>;
```

Creates `SIMILAR_TO` edges from `fact_id` to every other same-agent fact whose
composite similarity is `>= options.threshold`.

**Parameters**

- `fact_id` — the `node_id` of the source semantic fact (as returned by
  `store_fact` / `store_fact_with_provenance`).
- `options` — see [`SimilarityOptions`](#configuration--similarityoptions).

**Returns** — the number of **new unordered pairs** linked by this call. A
reciprocal edge created under `bidirectional == true` is part of the same pair
and is *not* counted separately.

**Behaviour and guarantees**

- `options.enabled == false` → returns `Ok(0)` and writes nothing.
- `fact_id` not found among this agent's semantic facts → logs a `warn!` and
  returns `Ok(0)` (lenient; not an error).
- Scores at most `candidate_limit` other facts — selected **highest-confidence
  first** (the `get_all_facts` order) — with the source fact itself always
  excluded (no self-loops). On a corpus larger than the limit, low-confidence
  facts beyond the cutoff are not scored and so are never linked.
- For each candidate `c` with `score >= threshold` that is **not already**
  connected to `fact_id` by a `SIMILAR_TO` edge in *either* direction: creates
  `fact_id → c`, plus `c → fact_id` when `bidirectional`.
- **Idempotent** — a candidate already linked in either direction is skipped, so
  calling the method again returns `Ok(0)`.

**Errors** — propagates `MemoryError::Storage` only if the backend fails to write
an edge between two existing nodes (degenerate; both endpoints come from the
agent's own fact set). There is no `InvalidInput` path.

```rust
let n = mem.auto_link_similar_facts(&fact_id, &SimilarityOptions::default())?;
println!("linked {n} new neighbour(s)");
```

### `rebuild_similarity_links(options) -> Result<SimilarityReport>`

```rust
pub fn rebuild_similarity_links(
    &mut self,
    options: &SimilarityOptions,
) -> Result<SimilarityReport>;
```

Backfill / maintenance: applies the same above-threshold linking across **all**
of the agent's semantic facts. Use it after importing facts in bulk, after a
migration, or after changing the threshold.

**Returns** — a [`SimilarityReport`](#similarityreport).

**Behaviour and guarantees**

- `options.enabled == false` → returns
  `Ok(SimilarityReport { facts_processed: 0, links_created: 0 })`.
- **Non-destructive** — never deletes edges. Existing `SIMILAR_TO` edges
  (including ones you created manually with `link_similar_facts`) are preserved
  and suppress duplicate creation.
- **Idempotent** — every unordered pair is counted and created at most once even
  when `bidirectional == false` (once `A → B` exists, processing `B` sees `A`
  and skips). An immediate second run reports `links_created == 0`.
- Bounded cost — at most `facts_processed · candidate_limit` similarity
  computations.

```rust
let report = mem.rebuild_similarity_links(&SimilarityOptions::default())?;
println!(
    "processed {} facts, created {} new links",
    report.facts_processed, report.links_created,
);
```

### `store_fact_with_options(...) -> Result<String>`

```rust
pub fn store_fact_with_options(
    &mut self,
    concept: &str,
    content: &str,
    confidence: f64,
    source_id: &str,
    tags: Option<&[String]>,
    metadata: Option<&HashMap<String, serde_json::Value>>,
    source_episode_ids: &[String],
    options: &StoreFactOptions,
) -> Result<String>;
```

Stores a semantic fact exactly like `store_fact_with_provenance`, then —
**only** if `options.similarity` is `Some(opts)` and `opts.enabled` — calls
`auto_link_similar_facts(new_id, &opts)` on the freshly stored fact.

**Returns** — the new fact's `node_id` (identical to
`store_fact_with_provenance`). The auto-link count is intentionally discarded; if
you need it, call `auto_link_similar_facts` yourself.

**Behaviour**

- `options.similarity == None` (the default) → **byte-for-byte equivalent** to
  `store_fact_with_provenance(...)`; zero `SIMILAR_TO` edges. Safe drop-in.
- `Some(opts)` with `opts.enabled == false` → fact stored, no linking.
- `Some(opts)` with `opts.enabled == true` → fact stored *and* linked to its
  similar neighbours.

**Errors** — inherits everything `store_fact_with_provenance` can return
(`MemoryError::InvalidInput` when `confidence ∉ [0.0, 1.0]`,
`MemoryError::Storage` on a node/edge write failure), plus any error propagated
from the auto-link step.

### `SimilarityReport`

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimilarityReport {
    pub facts_processed: usize,
    pub links_created: usize,
}
```

| Field | Meaning |
|-------|---------|
| `facts_processed` | Number of same-agent semantic facts iterated during the rebuild. |
| `links_created` | Number of **new unordered `SIMILAR_TO` pairs** created this pass. |

It derives `Eq`, so tests can assert exact values:

```rust
assert_eq!(
    mem.rebuild_similarity_links(&SimilarityOptions::default())?.links_created,
    0, // second pass adds nothing
);
```

### `StoreFactOptions`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct StoreFactOptions {
    pub similarity: Option<SimilarityOptions>,
}
```

| Field | Default | Meaning |
|-------|---------|---------|
| `similarity` | `None` | `Some(opts)` (with `opts.enabled`) ⇒ auto-link the freshly stored fact; `None` ⇒ inert, behaves like `store_fact_with_provenance`. |

`StoreFactOptions::default()` is `{ similarity: None }`, so the store-time hook is
**off by default** for backward compatibility.

---

## Edge data model

Each auto-created edge is:

```text
SemanticMemory --SIMILAR_TO--> SemanticMemory
```

between two facts owned by the **same agent**, carrying a single property:

| Property | Value |
|----------|-------|
| `similarity_score` | The composite score formatted as a fixed **4-decimal** string, e.g. `"0.7321"`. Always parseable with `parse::<f64>()`, and equal to the score to 4 dp. |

The fixed `{:.4}` format makes persistence assertions exact and round-trips
cleanly: `props["similarity_score"].parse::<f64>()` always succeeds and equals
the score to 4 decimal places.

### Reading edges back

> **This feature is write-only for now: there is currently no public API to read
> `SIMILAR_TO` edges back.** `CognitiveMemory` owns its graph through a
> **private** field, and the only edge readers it exposes today
> (`fact_provenance`, `procedure_provenance`) are specific to `DERIVES_FROM`
> provenance — not `SIMILAR_TO`. The crate's own tests verify these edges via
> crate-internal field access
> (`cm.graph.query_neighbors(id, Some("SIMILAR_TO"), Direction::Both, limit)`),
> which is **not** reachable from outside the crate.
>
> A higher-level convenience reader — e.g.
> `similar_facts(fact_id) -> Vec<(String, f64)>` returning each neighbour id with
> its parsed `similarity_score` — is the natural complement to this write side. It
> is recorded as a deliberate, **backward-compatible future addition** (see the
> API contract §9) and is intentionally out of scope for the initial feature.
> Until it lands, auto-linking builds graph structure that a graph-layer consumer
> can use; there is no supported way to read the edges back through the public
> `CognitiveMemory` surface.

---

## Idempotency and directionality

**Idempotency** is keyed on the **unordered pair** `{A, B}`. Before linking, the
pair is checked for an existing `SIMILAR_TO` edge in *either* direction; if one
exists, the whole pair is skipped (no reciprocal is added either). Consequences:

- Calling `auto_link_similar_facts` twice on the same fact: the second call
  returns `Ok(0)`.
- `rebuild_similarity_links` run twice: the second run reports
  `links_created == 0`.
- Manually-created `link_similar_facts` edges count as "already linked" and are
  never duplicated by the auto-linker.

**Directionality** is controlled by `bidirectional`:

| `bidirectional` | Edges written for a new pair `{A, B}` (source `A`) | `Direction::Outgoing` from `A` finds `B` | from `B` finds `A` |
|-----------------|---------------------------------------------------|------------------------------------------|--------------------|
| `true` (default) | `A → B` **and** `B → A` | yes | yes |
| `false`          | `A → B` only | yes | no (use `Direction::Both`) |

In both cases a `Direction::Both` neighbour query sees the pair from either
endpoint, so default traversal is unaffected by the flag.

---

## Persistence

On the `persistent` backend (`CognitiveMemory::open_persistent(...)`), the
`SIMILAR_TO` relationship table is created on first use and the edges — including
the `similarity_score` property — are durable. They survive `checkpoint()`,
`drop`, and reopen exactly like every other node and edge:

```rust
# #[cfg(feature = "persistent")]
# fn demo() -> amplihack_memory::Result<()> {
use amplihack_memory::{CognitiveMemory, SimilarityOptions};

let path = "/var/lib/agent/cognitive.ladybug";
{
    let mut mem = CognitiveMemory::open_persistent(path, "agent-1")?;
    let a = mem.store_fact("rust", "rust borrow checker memory safety", 0.9, "s", None, None)?;
    let _b = mem.store_fact("rust", "rust borrow checker enforces memory safety", 0.9, "s", None, None)?;
    mem.auto_link_similar_facts(&a, &SimilarityOptions::default())?;
    mem.checkpoint()?;
} // dropped → checkpointed

// Reopen — the SIMILAR_TO edge and its similarity_score persist with the database.
let _mem = CognitiveMemory::open_persistent(path, "agent-1")?;
// The edge is durable; reading it back requires a graph-layer consumer —
// no public reader is exposed yet (see "Reading edges back").
# Ok(())
# }
```

No schema migration is required: enabling linking on an existing database simply
starts creating `SIMILAR_TO` edges on the next call.

---

## Tutorial: building a linked knowledge base

This walkthrough stores a mix of related and unrelated facts, links them, backfills,
and shows idempotency.

```rust
use amplihack_memory::{CognitiveMemory, SimilarityOptions, StoreFactOptions};

fn main() -> amplihack_memory::Result<()> {
    let mut mem = CognitiveMemory::new("research-agent")?;

    // 1. Store three facts: two about Rust memory safety (related), one about cooking.
    let rust_tags = ["rust".to_string(), "memory".to_string()];
    let a = mem.store_fact(
        "rust-safety",
        "the rust borrow checker guarantees memory safety without a garbage collector",
        0.90, "src-1", Some(&rust_tags), None,
    )?;
    let b = mem.store_fact(
        "rust-safety",
        "rust enforces memory safety at compile time through its borrow checker",
        0.85, "src-2", Some(&rust_tags), None,
    )?;
    let cooking_tags = ["food".to_string()];
    let _c = mem.store_fact(
        "cooking",
        "simmer onions gently in butter until soft and translucent",
        0.80, "src-3", Some(&cooking_tags), None,
    )?;

    // 2. Explicitly link `a`. Only `b` clears the default 0.60 threshold; `c` does not.
    let linked = mem.auto_link_similar_facts(&a, &SimilarityOptions::default())?;
    assert_eq!(linked, 1, "a should link to b only");

    // 3. Idempotency: a second call creates nothing new.
    assert_eq!(mem.auto_link_similar_facts(&a, &SimilarityOptions::default())?, 0);

    // 4. Backfill the whole agent. The a<->b pair already exists, so 0 new links.
    let report = mem.rebuild_similarity_links(&SimilarityOptions::default())?;
    assert_eq!(report.facts_processed, 3);
    assert_eq!(report.links_created, 0);

    // 5. Add a new fact that auto-links on store.
    let opts = StoreFactOptions { similarity: Some(SimilarityOptions::default()) };
    let _d = mem.store_fact_with_options(
        "rust-safety",
        "memory safety in rust comes from ownership and the borrow checker",
        0.88, "src-4", Some(&rust_tags), None, &[], &opts,
    )?;
    // `d` is already connected to a and b.

    Ok(())
}
```

To tune the corpus, adjust `SimilarityOptions::threshold` and re-run
`rebuild_similarity_links`. Lowering the threshold adds the newly-qualifying
edges; the call remains idempotent and non-destructive.

---

## Relationship to manual `link_similar_facts`

`CognitiveMemory::link_similar_facts(a, b, score)` predates this feature and is
**unchanged**. It creates a single directed `SIMILAR_TO` edge with whatever
score you supply, stored as the raw `score.to_string()`.

| | `link_similar_facts` (manual) | auto-link path (this feature) |
|---|---|---|
| Score source | caller-supplied `f64` | computed via `compute_similarity` |
| `similarity_score` format | raw `score.to_string()` | fixed `format!("{score:.4}")` |
| Direction | single `a → b` | `a → b` (+ reciprocal if `bidirectional`) |
| Dedup | none (caller's responsibility) | unordered-pair idempotency |

The two coexist on the same `SIMILAR_TO` edge type. The auto-linker treats
manually-created edges as "already linked" (so it never duplicates them), and
`rebuild_similarity_links` preserves them. Because the two producers may write
the `similarity_score` property in different precisions, **consumers should treat
`similarity_score` as a parseable `f64` string** rather than assuming a fixed
number of decimal places across both.

---

## Compatibility and guarantees

- **Purely additive.** No existing public item changes signature or behaviour:
  `store_fact`, `store_fact_with_provenance`, `store_fact_with_provenance_strict`,
  `link_similar_facts`, `link_fact_to_episode(s)`, and `get_all_facts` are
  untouched. `store_fact` still never creates `SIMILAR_TO` edges.
- **New public surface** (re-exported at the crate root,
  `amplihack_memory::{...}`): the types `SimilarityOptions`, `SimilarityReport`,
  `StoreFactOptions`, and the three `CognitiveMemory` methods above.
- **No new dependencies and no new math** — built entirely on the existing
  `similarity` module helpers.
- **Same-agent isolation** is never crossed.
- **Works on both backends** — default in-memory and `--features persistent` —
  with identical semantics; edges and scores are durable on the persistent
  backend.

See the crate [`README.md`](../README.md) for the short version embedded in the
cognitive-memory quick start.
