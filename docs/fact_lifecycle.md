# Fact Lifecycle: Dedup, Supersession & Retention

`CognitiveMemory` semantic facts are long-lived knowledge. Without lifecycle
management, agents that store facts on every turn accumulate **duplicates**
(the same fact written many times), **stale revisions** (an older value left
behind when a newer one arrives), and **unbounded growth** (the fact store
never shrinks). This page documents the three capabilities that solve that:

| Capability | What it does | Entry point |
|------------|--------------|-------------|
| **Deduplication** | Detects and collapses facts that are the same (by content hash, caller-supplied key, or same-concept similarity) | [`upsert_fact`](#upsert_fact), [`find_duplicate_facts`](#find_duplicate_facts) |
| **Supersession** | Replaces an old fact with a newer one, keeping the history queryable via a `SUPERSEDES` edge | [`supersede_fact`](#supersede_fact) |
| **Retention / Pruning** | Archives then deletes facts under a configurable policy (per-concept caps, TTL, importance floor), with a non-destructive dry run | [`prune_semantic_memory`](#prune_semantic_memory) |

All three are **additive and backward compatible**. Existing calls
([`store_fact`](cognitive_memory.md#4-semantic-memory),
`store_fact_with_provenance`, `get_all_facts`) behave exactly as before; the new
fields default sensibly for facts written by earlier versions, and no schema
migration is required on either the in-memory or the persistent (LadybugDB)
backend.

> **Scope.** This feature lives in the Rust crate `amplihack-memory`
> (`CognitiveMemory`). The Rust API is the primary, authoritative surface and is
> documented first. The Python `CognitiveMemory` additively exposes the new fact
> fields; see [Python parity](#python-parity).

> **Note.** The Rust snippets below use the `?` operator and therefore assume a
> caller that returns `Result<(), MemoryError>` (e.g. the body of `fn main() ->
> amplihack_memory::Result<()>`).

---

## Concepts

```
                       supersede_fact(old, new, reason)
                       ┌──────────────────────────────┐
                       │                               │
                       v                               │
  new fact ──SUPERSEDES──> old fact (archived=true, superseded_by=new)
  (current)                (history, excluded from default reads)


  upsert_fact(input, { dedup }):

      DedupMode::None                 -> always Inserted
      DedupMode::ExactContentHash     -> identical content_hash  -> Reused
      DedupMode::CallerKey("k")       -> same dedup_key,
                                           same content           -> Reused
                                           changed content        -> Superseded
      DedupMode::SameConceptSimilarity-> score >= threshold       -> Reused


  prune_semantic_memory(policy):

      candidate && !archived  ->  archive   (run 1)
      candidate &&  archived  ->  delete     (run 2)   (archive-before-delete)
```

A fact is a **candidate** for pruning when it exceeds a per-concept cap, has
passed its concept TTL, or is a superseded record (when
`include_superseded` is set). Provenance-bearing facts whose `importance` is at
or above `min_importance_to_keep` are **protected** and never deleted.

---

## Extended `SemanticFact`

[`SemanticFact`](cognitive_memory.md#4-semantic-memory) gains eight fields. Each
is `#[serde(default)]`, so older serialized facts and pre-existing graph nodes
deserialize cleanly. When a property is absent from the stored node it is
default-populated on read (no migration break).

| Field | Type | Default on read | Description |
|-------|------|-----------------|-------------|
| `importance` | `f64` | `confidence` | Retention weight in `[0.0, 1.0]`. When a fact is written without an explicit importance it mirrors `confidence`. Pruning protects facts with `importance >= min_importance_to_keep`. |
| `usage_count` | `i64` | `0` | Number of times this fact has been reused via dedup. Incremented with `saturating_add` on each `Reused` outcome. |
| `last_accessed_at` | `Option<DateTime<Utc>>` | `None` | Set to "now" whenever the fact is reused by `upsert_fact`. |
| `expires_at` | `Option<DateTime<Utc>>` | `None` | Optional absolute expiry. Independent of the per-concept TTL applied at prune time. |
| `archived` | `bool` | `false` | `true` once the fact has been superseded or archived by pruning. Archived facts are excluded from [`get_all_facts`](#reads-and-archived-facts) / `search_facts` by default. |
| `superseded_by` | `Option<String>` | `None` | `node_id` of the fact that replaced this one (set by [`supersede_fact`](#supersede_fact)). |
| `content_hash` | `String` | computed | Stable fingerprint of `concept` + `content` (see [content_hash](#content_hash-and-dedup_key)). Recomputed on read if the stored value is empty. |
| `dedup_key` | `Option<String>` | `None` | Caller-supplied identity used by `DedupMode::CallerKey`. |

### Persistence

The write path always emits all eight properties for every fact, so reads
round-trip uniformly across the `InMemoryGraphStore` and the persistent
LadybugDB backend. On the persistent backend the new columns are created lazily
on first write (LadybugDB auto-`ALTER`s the table) — **there is no migration
step**. The fields survive close/reopen:

```rust
// Requires the `persistent` feature.
use amplihack_memory::CognitiveMemory;

{
    let mut cog = CognitiveMemory::open_persistent("/tmp/cog_db", "agent")?;
    cog.store_fact("ports", "staging uses 8080", 0.9, "", None, None)?;
} // dropped / closed

let cog = CognitiveMemory::open_persistent("/tmp/cog_db", "agent")?;
let fact = &cog.get_all_facts(10)[0];
assert_eq!(fact.importance, 0.9);          // defaulted from confidence
assert_eq!(fact.usage_count, 0);
assert!(!fact.content_hash.is_empty());    // recomputed if it was empty
```

### Reads and archived facts

`get_all_facts` and `search_facts` keep their existing signatures and behaviour:
they return **all** of this agent's facts sorted by confidence descending,
including any that have been archived. Each archived fact carries
`archived == true`, so callers that want to exclude supersession/retention
history filter on that flag. Reads stay `&self` and never mutate:
`last_accessed_at` and `usage_count` are written only by `upsert_fact`,
`supersede_fact`, and `prune_semantic_memory`.

### `content_hash` and `dedup_key`

`content_hash` is a hex-encoded SHA-256 over the concept and content joined by a
unit-separator byte:

```
content_hash = hex( SHA256( concept ‖ 0x1F ‖ content ) )
```

* The `0x1F` (ASCII Unit Separator) delimiter prevents boundary-collision: the
  facts `("ab", "c")` and `("a", "bc")` hash differently.
* The hash is computed over the **exact bytes** — no trimming or case folding —
  so `ExactContentHash` dedup is exact.
* `content_hash` is a fingerprint for equality, **not** a security token.

`dedup_key` is opaque and caller-controlled. It expresses "these writes are
about the same thing" independent of content, which is what enables
supersession when the content changes (see [CallerKey](#dedupmode)).

---

## Deduplication

### `DedupMode`

```rust
pub enum DedupMode {
    /// No deduplication — every upsert inserts a new fact.
    None,
    /// Reuse an existing fact with an identical `content_hash`.
    ExactContentHash,
    /// Identity is the caller-supplied `dedup_key`. Same key + same content
    /// reuses; same key + changed content supersedes.
    CallerKey(String),
    /// Reuse the most similar same-concept fact whose composite similarity is
    /// at or above `similarity_threshold`.
    SameConceptSimilarity,
}
```

### `DedupOptions`

```rust
pub struct DedupOptions {
    pub mode: DedupMode,
    pub similarity_threshold: f64,
    pub same_concept_only: bool,
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `DedupMode` | `None` | Strategy used to detect a duplicate. |
| `similarity_threshold` | `f64` | `0.60` | Minimum composite similarity (`0.5*word + 0.2*tag + 0.3*concept`, the same score used by `SIMILAR_TO` linking) for `SameConceptSimilarity` to treat two facts as duplicates. |
| `same_concept_only` | `bool` | `true` | Restrict candidate scanning to facts that share the new fact's `concept`. Keeps dedup cheap and concept-scoped. |

`DedupOptions::default()` is `{ mode: None, similarity_threshold: 0.60,
same_concept_only: true }`, i.e. **dedup off** — opt in explicitly.

`similarity_threshold` only applies to `SameConceptSimilarity`; the other modes
ignore it.

### `upsert_fact`

```rust
pub fn upsert_fact(
    &mut self,
    input: FactInput,
    options: &StoreFactOptions,
) -> Result<StoreFactOutcome>
```

Store a fact, deduplicating per `options.dedup`. On a dedup hit it either
**reuses** the existing fact (updating `confidence` to the new value,
incrementing `usage_count`, and stamping `last_accessed_at`) or **supersedes**
it. Provenance (`options.provenance`) adds a `DERIVES_FROM` edge from the
surviving node to each source episode, exactly as
[`store_fact_with_provenance`](cognitive_memory.md#4-semantic-memory) does, and
`SIMILAR_TO` auto-linking (`options.similarity`) is applied to the surviving
node exactly as in
[`store_fact_with_options`](cognitive_memory.md#4-semantic-memory). `upsert_fact`
is the **only** method that reads `options.provenance` and `options.dedup` (see
[StoreFactOptions](#storefactoptions)).

#### `FactInput`

```rust
pub struct FactInput {
    pub concept: String,
    pub content: String,
    pub confidence: f64,
    pub source_id: String,
    pub tags: Vec<String>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub importance: Option<f64>,
    pub expires_at: Option<DateTime<Utc>>,
    pub dedup_key: Option<String>,
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `concept` | `String` | required | Concept / lookup key. |
| `content` | `String` | required | Factual content. |
| `confidence` | `f64` | required | Confidence in `[0.0, 1.0]`; out-of-range/`NaN` is `MemoryError::InvalidInput`. |
| `source_id` | `String` | `""` | Opaque provenance string (never auto-converted to an edge). |
| `tags` | `Vec<String>` | `[]` | Tags used for similarity and filtering. |
| `metadata` | `HashMap<String, Value>` | `{}` | Arbitrary structured metadata. |
| `importance` | `Option<f64>` | `None` ⇒ `confidence` | Retention weight; defaults to `confidence` when unset. |
| `expires_at` | `Option<DateTime<Utc>>` | `None` | Optional absolute expiry used by retention. |
| `dedup_key` | `Option<String>` | `None` | Required by `DedupMode::CallerKey`; ignored by other modes. |

`FactInput` implements `Default`, and a convenience constructor
`FactInput::new(concept, content, confidence)` fills the rest with defaults.

#### `StoreFactOptions`

The existing `StoreFactOptions` (introduced for `SIMILAR_TO` auto-linking) is
extended in place:

```rust
pub struct StoreFactOptions {
    pub similarity: Option<SimilarityOptions>,
    pub provenance: ProvenanceOptions,
    pub dedup: DedupOptions,
}

pub struct ProvenanceOptions {
    pub source_episode_ids: Vec<String>,
    pub strict: bool,
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `similarity` | `Option<SimilarityOptions>` | `None` | When `Some(opts)` and enabled, auto-link the surviving fact to its `SIMILAR_TO` neighbours. |
| `provenance` | `ProvenanceOptions` | empty / non-strict | Source episode ids; each existing episode gets a `DERIVES_FROM` edge from the surviving fact. With `strict = true`, a missing episode fails the whole insert atomically. |
| `dedup` | `DedupOptions` | `DedupOptions::default()` (mode `None`) | Deduplication strategy. |

> **Which methods read which fields.** `provenance` and `dedup` are consumed
> **only** by [`upsert_fact`](#upsert_fact) (and `dedup` also by
> [`find_duplicate_facts`](#find_duplicate_facts)). The pre-existing
> [`store_fact_with_options`](cognitive_memory.md#4-semantic-memory) keeps its
> exact positional signature and reads **only** `similarity`; it ignores
> `provenance` and `dedup`. Provenance for that legacy method continues to come
> from its own `source_episode_ids` argument, so every existing call behaves
> identically.

`StoreFactOptions` derives `Debug, Clone, PartialEq, Default`. It **no longer
derives `Copy`** (it now owns a `Vec` and `String`). Construct it with struct
update syntax:

```rust
let options = StoreFactOptions {
    dedup: DedupOptions { mode: DedupMode::ExactContentHash, ..Default::default() },
    ..Default::default()
};
```

`StoreFactOptions::default()` (all fields default) makes `upsert_fact` behave
like a plain insert with no dedup, no provenance, and no similarity linking.

#### `StoreFactOutcome` and `DedupAction`

```rust
pub struct StoreFactOutcome {
    pub node_id: String,
    pub dedup_action: DedupAction,
    pub content_hash: String,
    pub similarity_links_created: usize,
    pub provenance_edges_created: usize,
}

pub enum DedupAction {
    /// A brand-new fact node was created.
    Inserted,
    /// An existing fact matched and was reinforced; no new node created.
    Reused { existing_id: String },
    /// A new node was created and an older one was archived behind it.
    Superseded { old_id: String, new_id: String },
}
```

`node_id` is always the **surviving / current** fact: the new node for
`Inserted` and `Superseded`, and the reused node for `Reused`.

#### Decision table

| `mode` | Match condition | Outcome | Side effects |
|--------|-----------------|---------|--------------|
| `None` | — | `Inserted` | New node. |
| `ExactContentHash` | a same-agent fact has an identical `content_hash` | `Reused { existing_id }` | `confidence = new`, `usage_count += 1`, `last_accessed_at = now`. No new node. |
| `ExactContentHash` | no match | `Inserted` | New node. |
| `CallerKey(k)` | same `dedup_key`, **identical** content | `Reused { existing_id }` | Reinforced as above (no churn for an unchanged re-write). |
| `CallerKey(k)` | same `dedup_key`, **changed** content | `Superseded { old_id, new_id }` | New node inserted; old node archived; `SUPERSEDES` edge `new -> old`. |
| `CallerKey(k)` | no fact with that key | `Inserted` | New node carrying `dedup_key = k`. |
| `SameConceptSimilarity` | a same-concept fact scores `>= similarity_threshold` | `Reused { existing_id }` | Reinforced; reuses the highest-scoring match. |
| `SameConceptSimilarity` | no match | `Inserted` | New node. |

Reuse updates **confidence, usage_count, and last_accessed_at only**;
`importance` is intentionally left untouched so a hand-tuned retention weight is
never silently lowered by a reuse.

#### Example

```rust
use amplihack_memory::{
    CognitiveMemory, DedupMode, DedupOptions, DedupAction, FactInput, StoreFactOptions,
};

let mut cog = CognitiveMemory::new("agent")?;

let exact = StoreFactOptions {
    dedup: DedupOptions { mode: DedupMode::ExactContentHash, ..Default::default() },
    ..Default::default()
};

let first = cog.upsert_fact(
    FactInput::new("ports", "staging uses 8080", 0.9),
    &exact,
)?;
assert!(matches!(first.dedup_action, DedupAction::Inserted));

// Identical content -> reused, not a second node.
let second = cog.upsert_fact(
    FactInput::new("ports", "staging uses 8080", 0.95),
    &exact,
)?;
assert!(matches!(second.dedup_action, DedupAction::Reused { .. }));
assert_eq!(first.node_id, second.node_id);
assert_eq!(cog.get_all_facts(50).len(), 1);
```

```rust
// CallerKey: same key, changed content -> supersede, keeping the latest.
use amplihack_memory::{CognitiveMemory, DedupMode, DedupOptions, DedupAction, FactInput, StoreFactOptions};
let mut cog = CognitiveMemory::new("agent")?;

let by_key = |k: &str| StoreFactOptions {
    dedup: DedupOptions { mode: DedupMode::CallerKey(k.into()), ..Default::default() },
    ..Default::default()
};

let v1 = cog.upsert_fact(
    FactInput { dedup_key: Some("staging-port".into()), ..FactInput::new("ports", "staging uses 8080", 0.9) },
    &by_key("staging-port"),
)?;

let v2 = cog.upsert_fact(
    FactInput { dedup_key: Some("staging-port".into()), ..FactInput::new("ports", "staging uses 9090", 0.9) },
    &by_key("staging-port"),
)?;

match v2.dedup_action {
    DedupAction::Superseded { old_id, new_id } => {
        assert_eq!(old_id, v1.node_id);
        assert_eq!(new_id, v2.node_id);
    }
    _ => panic!("expected supersession"),
}
assert_eq!(cog.get_all_facts(50).len(), 1);
```

### `find_duplicate_facts`

```rust
pub fn find_duplicate_facts(
    &self,
    options: &DedupOptions,
    limit: usize,
) -> Vec<DuplicateFactGroup>
```

Group this agent's **active** facts into duplicate clusters under the given mode
without modifying anything. Useful for auditing an existing store before
enabling dedup, or for a periodic cleanup job that feeds
[`supersede_fact`](#supersede_fact). Only groups with two or more members are
returned; at most `limit` groups.

```rust
pub struct DuplicateFactGroup {
    /// The shared identity for the group: the `content_hash`
    /// (ExactContentHash), the `dedup_key` (CallerKey), or the representative
    /// node id (SameConceptSimilarity).
    pub key: String,
    /// `node_id`s in the group, oldest first (by creation time, ties broken by
    /// node id). Always contains at least two ids.
    pub fact_ids: Vec<String>,
    /// The representative (oldest) member, equal to `fact_ids[0]`.
    pub representative_id: String,
}
```

| `mode` | Grouping key |
|--------|--------------|
| `None` | returns no groups |
| `ExactContentHash` | identical `content_hash` |
| `CallerKey(_)` | identical `dedup_key` |
| `SameConceptSimilarity` | same concept and pairwise similarity `>= similarity_threshold` |

---

## Supersession

### `SUPERSEDES` edge

A new edge type connects a current fact to the record it replaced:

```
SemanticMemory --SUPERSEDES--> SemanticMemory   (new -> old)
```

The edge carries a `reason` property and a `superseded_at` timestamp. It is the
inverse view of the old fact's `superseded_by` field, so supersession history is
queryable from either direction.

### `supersede_fact`

```rust
pub fn supersede_fact(
    &mut self,
    old_id: &str,
    new_id: &str,
    reason: &str,
) -> Result<()>
```

Mark `old_id` as replaced by `new_id`:

1. Set `old.superseded_by = new_id` and `old.archived = true` (it disappears
   from default reads).
2. Add a `SUPERSEDES` edge `new_id -> old_id` carrying `reason` and
   `superseded_at`.

Both ids must be this agent's `SemanticMemory` nodes — a guard that prevents one
agent from archiving another agent's fact by guessing an id. The operation is
idempotent: re-superseding the same pair re-asserts the state and the single
edge without error.

#### Errors

* `MemoryError::InvalidInput` — `old_id`/`new_id` is unknown, is not a semantic
  fact, belongs to another agent, or `old_id == new_id`.
* `MemoryError::Storage` — the backend rejects the node update or edge write.

```rust
use amplihack_memory::CognitiveMemory;
let mut cog = CognitiveMemory::new("agent")?;

let old = cog.store_fact("ports", "staging uses 8080", 0.9, "", None, None)?;
let new = cog.store_fact("ports", "staging uses 9090", 0.95, "", None, None)?;

cog.supersede_fact(&old, &new, "port reassigned in infra change #214")?;

// `get_all_facts` returns both facts; the superseded `old` is flagged
// `archived` (and linked behind the SUPERSEDES edge), so callers filter it out.
let facts = cog.get_all_facts(50);
assert!(facts.iter().any(|f| f.node_id == new && !f.archived));
assert!(facts.iter().any(|f| f.node_id == old && f.archived));
```

---

## Retention / Pruning

### `RetentionPolicy`

```rust
pub struct RetentionPolicy {
    pub max_facts_per_concept: Option<usize>,
    pub ttl_seconds_by_concept: HashMap<String, i64>,
    pub min_importance_to_keep: f64,
    pub include_superseded: bool,
    pub dry_run: bool,
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_facts_per_concept` | `Option<usize>` | `None` | Keep at most N active facts per concept; lowest-importance (tie-broken by oldest) beyond N become candidates. `None` = no cap. |
| `ttl_seconds_by_concept` | `HashMap<String, i64>` | `{}` | Per-concept maximum age in seconds (measured from `created_at`). A concept with no entry has no TTL. |
| `min_importance_to_keep` | `f64` | `0.0` | Facts with `importance >= this` are protected from deletion (but may still be archived). Clamped to `[0.0, 1.0]`. |
| `include_superseded` | `bool` | `false` | When `true`, already-superseded (archived) facts are pruning candidates too. |
| `dry_run` | `bool` | `false` | When `true`, compute and report counts but mutate nothing. |

`RetentionPolicy::default()` is inert: no cap, no TTLs, importance floor `0.0`,
superseded excluded, not a dry run — a default-constructed policy archives and
deletes nothing.

### `prune_semantic_memory`

```rust
pub fn prune_semantic_memory(
    &mut self,
    policy: &RetentionPolicy,
) -> Result<PruneReport>
```

Apply the retention policy with **archive-before-delete** semantics so nothing
is destroyed in a single irreversible step:

* A **candidate** fact that is **not yet archived** is archived
  (`archived = true`) — counted in `archived`.
* A **candidate** fact that is **already archived** is deleted — counted in
  `deleted`.

Run the same policy twice and the first pass archives, the second deletes; this
is intentional and gives operators a window to inspect or restore between
passes.

```rust
pub struct PruneReport {
    /// Facts archived this run (real run only).
    pub archived: usize,
    /// Facts deleted this run (real run only).
    pub deleted: usize,
    /// Facts that *would* be archived (dry run only).
    pub would_archive: usize,
    /// Facts that *would* be deleted (dry run only).
    pub would_delete: usize,
}
```

In a real run `would_archive`/`would_delete` are `0`; in a dry run
`archived`/`deleted` are `0` and no data changes.

#### Protection rules

A candidate is **never deleted** when **both** hold:

* it has provenance — at least one `DERIVES_FROM` edge, and
* `importance >= policy.min_importance_to_keep`.

The explicit override is the policy itself: lowering a fact's importance below
`min_importance_to_keep` (or raising the floor above it) is the deliberate way
to make a high-value, provenance-bearing fact eligible for deletion. This keeps
durable, sourced knowledge from being silently swept away by an aggressive cap
or TTL.

#### Dry run vs. real run

```rust
use std::collections::HashMap;
use amplihack_memory::{CognitiveMemory, RetentionPolicy};

let mut cog = CognitiveMemory::new("agent")?;
for i in 0..5 {
    cog.store_fact("logs", &format!("log line {i}"), 0.2, "", None, None)?;
}

let policy = || RetentionPolicy {
    max_facts_per_concept: Some(2),
    ttl_seconds_by_concept: HashMap::new(),
    min_importance_to_keep: 0.0,
    include_superseded: false,
    dry_run: true,
};

// Dry run: counts only, store unchanged.
let preview = cog.prune_semantic_memory(&policy())?;
assert_eq!(preview.would_archive, 3);
assert_eq!(preview.archived, 0);
assert_eq!(cog.get_all_facts(50).len(), 5);

// Real run 1: archive the 3 over-cap facts. Archived facts are retained (still
// returned by `get_all_facts`, flagged `archived = true`) until a later pass.
let run1 = cog.prune_semantic_memory(&RetentionPolicy { dry_run: false, ..policy() })?;
assert_eq!(run1.archived, 3);
assert_eq!(run1.deleted, 0);
assert_eq!(cog.get_all_facts(50).len(), 5); // archived, not yet deleted

// Real run 2: the now-archived candidates are deleted.
let run2 = cog.prune_semantic_memory(&RetentionPolicy { dry_run: false, ..policy() })?;
assert_eq!(run2.deleted, 3);
assert_eq!(cog.get_all_facts(50).len(), 2);
```

#### TTL example

```rust
use std::collections::HashMap;
use amplihack_memory::RetentionPolicy;

// Facts under concept "sensor-reading" older than 1 hour become candidates.
let mut ttl = HashMap::new();
ttl.insert("sensor-reading".to_string(), 3_600);

let policy = RetentionPolicy {
    ttl_seconds_by_concept: ttl,
    min_importance_to_keep: 0.8, // protect high-importance + provenance-bearing facts
    dry_run: false,
    ..Default::default()
};
```

---

## Backward compatibility

| Surface | Guarantee |
|---------|-----------|
| `store_fact`, `store_fact_with_provenance`, `store_fact_with_provenance_strict` | Unchanged signatures and behavior; delegate to the shared write path with new fields defaulted (`importance = confidence`, `usage_count = 0`, `archived = false`, etc.). |
| `store_fact_with_options` | Unchanged signature and behavior. It reads **only** `options.similarity`; the new `options.provenance` and `options.dedup` fields are ignored on this path (they are honored exclusively by `upsert_fact`). Provenance still comes from its `source_episode_ids` argument, so existing calls behave identically. |
| `get_all_facts`, `search_facts` | Unchanged signatures; still return all facts by confidence descending. Archived facts are included, each flagged `archived = true` so callers can filter them. |
| Existing `SemanticFact` consumers | New fields are `#[serde(default)]`; deserializing an old fact yields the documented defaults. |
| Stored graph nodes (both backends) | Missing properties are default-populated on read; `content_hash` is recomputed when absent. No migration required. |

The one source-level change for callers that constructed `StoreFactOptions`
literally is that it no longer derives `Copy` (it now owns a `Vec`/`String`).
Use `..Default::default()` in the struct literal — a compile-time-visible,
mechanical update.

---

## Python parity

The Python `SemanticFact` dataclass (returned by `search_facts` and
`get_all_facts` on the `amplihack_memory.CognitiveMemory` class) gains the new
fields **additively** — existing attributes are unchanged, so code that reads
`node_id`/`concept`/`content`/`confidence` continues to work:

```python
from amplihack_memory import CognitiveMemory

cog = CognitiveMemory("agent", "/tmp/cog_db")
cog.store_fact("ports", "staging uses 8080", confidence=0.9)

fact = cog.get_all_facts(limit=1)[0]
fact.importance        # 0.9    (defaults to confidence)
fact.usage_count       # 0
fact.last_accessed_at  # None   (datetime once reused via dedup)
fact.expires_at        # None   (datetime | None)
fact.archived          # False
fact.superseded_by     # None   (str | None)
fact.content_hash      # "…"    (hex SHA-256)
fact.dedup_key         # None   (str | None)
```

The deduplication, supersession, and retention **operations** (`upsert_fact`,
`supersede_fact`, `prune_semantic_memory`, `find_duplicate_facts`) are part of
the Rust API in this iteration; the Python surface gains the read-only fields
above so callers can observe lifecycle state.

---

## Configuration reference

| Setting | Type | Default | Used by |
|---------|------|---------|---------|
| `DedupOptions.mode` | `DedupMode` | `None` | `upsert_fact`, `find_duplicate_facts` |
| `DedupOptions.similarity_threshold` | `f64` | `0.60` | `SameConceptSimilarity` |
| `DedupOptions.same_concept_only` | `bool` | `true` | dedup candidate scan |
| `RetentionPolicy.max_facts_per_concept` | `Option<usize>` | `None` | `prune_semantic_memory` |
| `RetentionPolicy.ttl_seconds_by_concept` | `HashMap<String, i64>` | `{}` | `prune_semantic_memory` |
| `RetentionPolicy.min_importance_to_keep` | `f64` | `0.0` | prune protection |
| `RetentionPolicy.include_superseded` | `bool` | `false` | prune candidacy |
| `RetentionPolicy.dry_run` | `bool` | `false` | prune preview |

---

## See also

* [Cognitive Memory](cognitive_memory.md) — the six memory types and the base
  semantic-fact API.
* [API Reference](api_reference.md) — full method and data-class tables.
* [Architecture](architecture.md) — backends, edges, and data flow.
