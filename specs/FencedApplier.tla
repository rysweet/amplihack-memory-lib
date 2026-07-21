-------------------------- MODULE FencedApplier --------------------------
(***************************************************************************)
(* Transactionally-safe single-writer coordination for the lbug-backed    *)
(* cognitive store (amplihack-memory-lib / Simard).                        *)
(*                                                                         *)
(* CONTEXT                                                                 *)
(*   lbug is single-process-exclusive for writes: only one OS process may  *)
(*   hold the writable Database.  Simard runs many ephemeral engineer      *)
(*   processes that all need to persist knowledge to ONE durable shared    *)
(*   store (federating per-engineer stores loses an engineer's learnings   *)
(*   when its worktree is reaped -- so the shared store is required).      *)
(*                                                                         *)
(*   Today, a would-be writer that finds the store "locked" reaps the      *)
(*   lock when kill(pid,0) says the recorded PID is dead                   *)
(*   (memory_ipc::reap_stale_open_lock).  PID-liveness is NOT lock         *)
(*   ownership: under PID reuse, or an alive-but-paused holder, the reaper  *)
(*   can steal the lock while the real holder is still running -> two      *)
(*   writers -> WAL/catalog corruption -> the "memory-wipe" recovery path. *)
(*                                                                         *)
(* THIS SPEC                                                               *)
(*   Models the write-apply path with a monotonic lease EPOCH as a fencing *)
(*   token.  Acquire() deliberately allows an UNSAFE steal (it may fire    *)
(*   while the previous holder is still `active`), reproducing the real    *)
(*   hazard.  Apply() is guarded by the epoch iff Fencing = TRUE.          *)
(*                                                                         *)
(*   With Fencing = FALSE  -> TLC finds a split-brain (NoSplitBrain fails). *)
(*   With Fencing = TRUE   -> NoSplitBrain holds: a stale-epoch writer's    *)
(*                            apply is rejected, so the store is only ever  *)
(*                            mutated by the current epoch (single writer). *)
(***************************************************************************)
EXTENDS Naturals

CONSTANTS Procs,      \* set of writer processes (ephemeral engineers + daemon)
          MaxEpoch,   \* bound on lease acquisitions (keeps the model finite)
          Fencing,    \* TRUE = epoch-fenced apply; FALSE = today's unfenced open
          None        \* sentinel model value: "no lease holder" (None \notin Procs)

VARIABLES
  epoch,      \* global monotonic lease counter: the source of fencing tokens
  holder,     \* proc that most recently acquired the lease (or None)
  myEpoch,    \* [Procs -> Nat]: the epoch each proc BELIEVES it holds
  active,     \* [Procs -> BOOLEAN]: proc has acquired and not yet applied/crashed
  corrupted   \* set TRUE once a stale-epoch writer mutates the shared store

vars == <<epoch, holder, myEpoch, active, corrupted>>

TypeOK ==
  /\ epoch \in 0..MaxEpoch
  /\ holder \in Procs \cup {None}
  /\ myEpoch \in [Procs -> 0..MaxEpoch]
  /\ active \in [Procs -> BOOLEAN]
  /\ corrupted \in BOOLEAN

Init ==
  /\ epoch = 0
  /\ holder = None
  /\ myEpoch = [p \in Procs |-> 0]
  /\ active = [p \in Procs |-> FALSE]
  /\ corrupted = FALSE

(* Acquire / steal the write lease.  Models BOTH a clean handoff (previous  *)
(* holder already crashed) AND an unsafe steal (previous holder still       *)
(* `active` -- the kill(pid,0) false-negative).  Bumping `epoch` mints a     *)
(* fresh fencing token; the acquirer records it in myEpoch.                 *)
Acquire(p) ==
  /\ epoch < MaxEpoch
  /\ epoch' = epoch + 1
  /\ holder' = p
  /\ myEpoch' = [myEpoch EXCEPT ![p] = epoch + 1]
  /\ active' = [active EXCEPT ![p] = TRUE]
  /\ UNCHANGED corrupted

(* Apply a write to the shared lbug store.                                  *)
(*   Fencing = TRUE : accepted only while the proc's token is still current.*)
(*   Fencing = FALSE: accepted unconditionally (today: the writer just      *)
(*                    opens the DB / holds the reaped lock and writes).     *)
(* A write accepted from a proc whose token is NOT the current epoch means  *)
(* a SECOND, stale writer mutated the store == corruption.                  *)
Apply(p) ==
  /\ active[p]
  /\ (Fencing => myEpoch[p] = epoch)
  /\ corrupted' = (corrupted \/ (myEpoch[p] # epoch))
  /\ active' = [active EXCEPT ![p] = FALSE]
  /\ UNCHANGED <<epoch, holder, myEpoch>>

(* The holder simply dies (ephemeral engineer reaped / process exits).      *)
Crash(p) ==
  /\ active[p]
  /\ active' = [active EXCEPT ![p] = FALSE]
  /\ UNCHANGED <<epoch, holder, myEpoch, corrupted>>

Next == \E p \in Procs : Acquire(p) \/ Apply(p) \/ Crash(p)

Spec == Init /\ [][Next]_vars

(*-----------------------------  PROPERTIES  ------------------------------*)

(* SAFETY: the shared store is never mutated by two different lease epochs, *)
(* i.e. the single-writer / no-split-brain invariant holds.                 *)
NoSplitBrain == corrupted = FALSE

=============================================================================
