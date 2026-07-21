# TLA+ Specifications: Multi-Writer Coordination for the lbug Store

This directory contains a durable, CI-enforced set of TLA+ specifications that
formalize the multi-writer coordination design for the lbug-backed cognitive
store used by amplihack-memory-lib (Simard). The specifications are
model-checked with TLC on every push (see the `tla-model-check` job in
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml)), so the safety and
liveness invariants described here are enforced going forward — if a future
change regresses the design, CI fails.

> **Scope.** This is a specification + documentation + CI artifact. It does not
> change any Rust code or runtime behavior. Implementing the epoch-fencing token
> and the durable shared log in the Rust crate is a separate follow-up.

---

## 1. The problem

`lbug` (the embedded graph store) is **single-process-exclusive for writes**:
only one OS process may hold the writable database at a time. Simard's
architecture has two kinds of writers that both need to persist knowledge into
**one** durable shared store:

- A **long-running daemon** (Simard itself).
- Many **ephemeral "engineer" processes** — short-lived workers, each in its own
  worktree, that Simard reaps unpredictably on heartbeat-staleness.

Two hazards fall out of this:

1. **Split-brain writes.** When a would-be writer finds the store "locked", the
   current code reaps the recorded open-lock if `kill(pid, 0)` reports the
   recorded PID is dead (`memory_ipc::reap_stale_open_lock`). PID liveness is
   **not** lock ownership: under PID reuse, or an alive-but-paused holder, the
   reaper can steal the lock while the real holder is still running. Two writers
   then open the store at once → WAL/catalog corruption → the "memory-wipe"
   recovery path.

2. **Lost ephemeral writes.** An engineer's learnings must survive the reaping
   of its worktree. Any design where a write is only durable *inside* the
   engineer's own local store loses that write if the engineer dies before its
   contents reach the shared store.

---

## 2. Designs considered

Three candidate designs were evaluated against the two hazards above.

| Design | Description | Verdict |
|--------|-------------|---------|
| **A — Single applier / server** | A single writer process owns the lbug store; all others RPC their writes to it. Correct in principle, but concentrates all durability in one live process and adds a full request/response server on the hot path. | Rejected: heavier than needed; durability still depends on one live process rather than a durable medium. |
| **B — Federated per-agent + hive** | Each engineer writes to its **own** local store (durable only within its worktree lifetime) and later **consolidates** into a shared "hive". | **Rejected — formally.** Reaping is not coordinated with consolidation, so an engineer that dies after producing but before consolidating loses an already-acked write. See [`FederatedLoss.tla`](FederatedLoss.tla). |
| **C — Durable shared log + single fenced applier** | Every engineer **appends** its write to a durable, append-only shared log (the ack / durability point). A **single applier**, protected by a monotonic **epoch fencing token**, consumes the log in order and applies to the lbug store. | **Chosen.** The append makes the write durable before the engineer can die; the fencing token makes the applier a safe single writer even under a bad reap. Proven by [`DurableLog.tla`](DurableLog.tla) and [`FencedApplier.tla`](FencedApplier.tla). |

**Why C.** Design C decouples *durability* (append to the shared log) from
*application* (single fenced applier). Durability no longer depends on any
ephemeral process staying alive, and single-writer safety no longer depends on
the unsafe `kill(pid, 0)` liveness check. Design A also achieves single-writer
safety but keeps durability tied to one live process and adds a server; design B
fails durability outright.

---

## 3. What each spec proves

### `FencedApplier.tla` — single-writer safety (the fencing fix)

Models the write-apply path with a monotonic lease **epoch** as a fencing token.
`Acquire` deliberately allows an **unsafe steal** (it may fire while the previous
holder is still active), reproducing the real `kill(pid, 0)` false-negative
hazard. `Apply` is guarded by the epoch **iff** `Fencing = TRUE`.

- Invariant **`NoSplitBrain`**: the shared store is never mutated by two
  different lease epochs (single writer holds).
- With `Fencing = FALSE` (today's unfenced open) TLC finds a split-brain —
  `NoSplitBrain` is **violated**. This is the current bug in
  `memory_ipc::reap_stale_open_lock`, reproduced formally.
- With `Fencing = TRUE` (the proposed epoch-fenced apply) a stale-epoch writer's
  apply is rejected, so `NoSplitBrain` **holds**.

### `DurableLog.tla` — durability of ephemeral-writer knowledge (design C)

Models design C: engineers append to a durable shared log, a single applier
consumes it in order.

- Invariant **`PrefixConsistency`** (safety): the applied portion never runs
  ahead of the durable log (`appliedIdx <= Len(log)`). Because `Apply` advances
  one entry at a time, in order, over an append-only log, the store is therefore
  always an in-order prefix of the log — no gaps, no duplicates (exactly-once
  apply).
- Property **`NoLostAckedWrite`** (liveness): every appended (acked) write is
  eventually applied, **even if the submitting engineer has died**. Holds.

### `FederatedLoss.tla` — why design B is rejected

Model-checks the **same** liveness goal (`NoLostAckedWrite`) under the federated
model B, where a write is "acked" when it lands in the engineer's local store
but must later be consolidated into the hive. Because the reaper is not
coordinated with consolidation, an engineer can die after producing but before
consolidating and its worktree is destroyed.

- Property **`NoLostAckedWrite`**: **violated** under model B — a formal
  justification for rejecting the federated design. (This is generous to B: it
  even assumes weak fairness on consolidation, and the property still fails.)

> Both configs that check the `FencedApplier` model also assert a structural
> **`TypeOK`** invariant (state well-formedness) alongside `NoSplitBrain`;
> `TypeOK` holds in both, so the unfenced negative check fails specifically on
> `NoSplitBrain`. The `DurableLog` positive config likewise asserts `TypeOK`
> alongside `PrefixConsistency`. `FederatedLoss.cfg` asserts only the
> `NoLostAckedWrite` property whose violation it demonstrates.

---

## 4. Reproducing the checks

There are **four** checks across the three modules. Two are **positive**
(the good designs — TLC must find no error) and two are **negative** (the
rejected/buggy designs — TLC must find the expected violation).

> **The two "violation" configs are intentional.** `FencedApplier_unfenced.cfg`
> and `FederatedLoss.cfg` are **negative tests**: they demonstrate the rejected
> designs and are **expected to violate** their invariant/property. A violation
> there is a *pass*; the *absence* of a violation is a failure (it would mean the
> bug is no longer demonstrable / the model drifted). CI asserts the correct
> polarity for all four.

### Run everything (recommended)

The check logic lives in **one place** — [`run-checks.sh`](run-checks.sh) — so
local runs and CI are identical. From the repository root:

```bash
bash specs/run-checks.sh
```

The script provisions `tla2tools.jar` (pinned by URL + sha256, or reused from
the `TLA_TOOLS_JAR` environment variable if set), runs all four checks, asserts
the expected polarity of each, and exits `0` **iff** all four match. Any positive
check finding an error, or any negative check *not* finding its expected
violation, makes the script exit non-zero.

To reuse a locally-cached tools jar instead of downloading:

```bash
TLA_TOOLS_JAR=/path/to/tla2tools.jar bash specs/run-checks.sh
```

### Run an individual check

Each check is a single `tlc` invocation of the form
`tlc -deadlock -config <cfg> <module>`. Assuming `tla2tools.jar` is on your
classpath (e.g. `alias tlc='java -cp /path/to/tla2tools.jar tlc2.TLC'`):

| # | Command | Kind | Expected TLC result |
|---|---------|------|---------------------|
| 1 | `tlc -deadlock -config specs/FencedApplier_fenced.cfg specs/FencedApplier.tla` | positive | `No error has been found` (`NoSplitBrain` holds under `Fencing = TRUE`) |
| 2 | `tlc -deadlock -config specs/DurableLog.cfg specs/DurableLog.tla` | positive | `No error has been found` (`PrefixConsistency` holds, `NoLostAckedWrite` holds) |
| 3 | `tlc -deadlock -config specs/FencedApplier_unfenced.cfg specs/FencedApplier.tla` | **negative** | `Invariant NoSplitBrain is violated` (split-brain reproduced under `Fencing = FALSE`) |
| 4 | `tlc -deadlock -config specs/FederatedLoss.cfg specs/FederatedLoss.tla` | **negative** | `Temporal properties were violated` (`NoLostAckedWrite` fails under design B) |

Note that TLC prints `is violated` for an invariant (check 3) and `were
violated` for a temporal property (check 4); the polarity logic in
`run-checks.sh` matches either by looking for `violated` case-insensitively.

---

## 5. File inventory

| File | Role |
|------|------|
| [`FencedApplier.tla`](FencedApplier.tla) | Single-writer safety model with an epoch fencing token. |
| [`FencedApplier_fenced.cfg`](FencedApplier_fenced.cfg) | Positive config (`Fencing = TRUE`); `NoSplitBrain` must hold. |
| [`FencedApplier_unfenced.cfg`](FencedApplier_unfenced.cfg) | Negative config (`Fencing = FALSE`); `NoSplitBrain` must be violated. |
| [`DurableLog.tla`](DurableLog.tla) | Design C durability model (durable shared log + single applier). |
| [`DurableLog.cfg`](DurableLog.cfg) | Positive config; `PrefixConsistency` and `NoLostAckedWrite` must hold. |
| [`FederatedLoss.tla`](FederatedLoss.tla) | Design B model showing acked writes are lost. |
| [`FederatedLoss.cfg`](FederatedLoss.cfg) | Negative config; `NoLostAckedWrite` must be violated. |
| [`run-checks.sh`](run-checks.sh) | Single source of truth for running all four checks and asserting polarity. Used by both CI and local devs. |
| `README.md` | This document. |

TLC scratch artifacts (`*_TTrace_*`, `*.bin`, `states/`, and the downloaded
`tla2tools.jar`) are transient and are git-ignored; they are not part of the
committed specification set.

---

## 6. How this is enforced in CI

The `tla-model-check` job in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml):

- runs on `ubuntu-latest` with `permissions: contents: read` and no secrets;
- provisions Temurin JDK 21;
- downloads `tla2tools.jar` pinned by URL + sha256 (verified **before** it is
  executed — reproducible, no floating "latest");
- invokes `specs/run-checks.sh` (the CI job contains no duplicated grep logic —
  it calls the same script devs run locally).

The job fails if any positive check finds an error, or any negative check does
not find its expected violation. In other words: **if the single-writer safety
invariant ever regresses, or the rejected designs silently become "safe" (model
drift), CI blocks the change.**
