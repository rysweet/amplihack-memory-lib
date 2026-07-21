--------------------------- MODULE DurableLog ---------------------------
(***************************************************************************)
(* Durability of ephemeral-writer knowledge in design C.                   *)
(*                                                                         *)
(* THE EPHEMERAL-ENGINEER PROBLEM                                          *)
(*   Simard's engineers are short-lived and reaped unpredictably           *)
(*   (heartbeat-stale).  A per-engineer local store (the "federated"       *)
(*   model B) makes an engineer's learnings durable only in a store that   *)
(*   is DESTROYED when its worktree is reaped -- so anything not           *)
(*   consolidated into the shared "hive" BEFORE the reap is lost.  Because  *)
(*   reaping is not coordinated with consolidation, that is an unavoidable  *)
(*   lost-write window.                                                     *)
(*                                                                         *)
(* DESIGN C                                                                *)
(*   An engineer's write is appended to a DURABLE SHARED log (the ack /    *)
(*   durability point).  A single fenced applier consumes the log in order *)
(*   and applies to the lbug store.  Once appended, the write survives the *)
(*   engineer's death -- no consolidation race.                            *)
(*                                                                         *)
(*   We check:                                                             *)
(*     PrefixConsistency (safety)  -- the materialized store is always an  *)
(*       in-order prefix of the durable log: no gaps, no reorder, no       *)
(*       duplicates == exactly-once apply.  This is falsifiable: a buggy   *)
(*       applier that skipped, reordered, or replayed an entry would make  *)
(*       `store` diverge from the log prefix and TLC would report it.      *)
(*     NoLostAckedWrite (liveness) -- every appended (acked) write is       *)
(*       eventually applied, EVEN IF the submitting engineer has died.     *)
(***************************************************************************)
EXTENDS Naturals, Sequences

CONSTANTS Clients   \* the ephemeral engineer processes

(* Per-client lifecycle:                                                    *)
(*   "start"    - has a write to contribute, not yet durable               *)
(*   "appended" - write is in the durable shared log (acked); may die now  *)
(*   "dead"     - process exited                                           *)
VARIABLES
  log,    \* Seq(Clients): durable, append-only order of accepted writes
  store,  \* Seq(Clients): what the single applier has actually materialized
          \*   into the lbug store, in the order it applied them
  pc      \* [Clients -> {"start","appended","dead"}]

vars == <<log, store, pc>>

(* Set of writes durably applied to the store.                             *)
AppliedSet == { store[i] : i \in 1..Len(store) }
(* Set of writes that were durably acked (appended), regardless of author's *)
(* liveness.                                                                *)
AckedSet == { log[i] : i \in 1..Len(log) }

TypeOK ==
  /\ Len(store) \in 0..Len(log)
  /\ \A i \in 1..Len(store) : store[i] \in Clients
  /\ pc \in [Clients -> {"start","appended","dead"}]

Init ==
  /\ log = << >>
  /\ store = << >>
  /\ pc = [c \in Clients |-> "start"]

(* Engineer durably submits its write: append to the shared log.  This is   *)
(* the ack point -- after this the engineer may die safely.                 *)
Submit(c) ==
  /\ pc[c] = "start"
  /\ log' = Append(log, c)
  /\ pc' = [pc EXCEPT ![c] = "appended"]
  /\ UNCHANGED store

(* Engineer dies BEFORE durably appending: its write was never acked, so    *)
(* losing it is correct (nothing promised).                                 *)
CrashBeforeAck(c) ==
  /\ pc[c] = "start"
  /\ pc' = [pc EXCEPT ![c] = "dead"]
  /\ UNCHANGED <<log, store>>

(* Engineer dies AFTER durably appending: the acked write MUST still be     *)
(* applied.  This is the ephemeral-writer case design C must survive.       *)
CrashAfterAck(c) ==
  /\ pc[c] = "appended"
  /\ pc' = [pc EXCEPT ![c] = "dead"]
  /\ UNCHANGED <<log, store>>

(* The single fenced applier materializes the next log entry, in order.     *)
Apply ==
  /\ Len(store) < Len(log)
  /\ store' = Append(store, log[Len(store) + 1])
  /\ UNCHANGED <<log, pc>>

Next ==
  \/ \E c \in Clients : Submit(c) \/ CrashBeforeAck(c) \/ CrashAfterAck(c)
  \/ Apply

(* Weak fairness on Apply: the applier keeps making progress while the log  *)
(* has unapplied entries (a live daemon).                                   *)
Spec == Init /\ [][Next]_vars /\ WF_vars(Apply)

(*-----------------------------  PROPERTIES  ------------------------------*)

(* SAFETY: the materialized store is exactly an in-order prefix of the      *)
(* durable log.  Equivalent to: no gaps, no reorder, no duplicates ->       *)
(* exactly-once apply.  Falsifiable -- a broken applier makes store diverge *)
(* from the log prefix of the same length.                                  *)
PrefixConsistency == store = SubSeq(log, 1, Len(store))

(* LIVENESS: every durably-acked write is eventually applied, regardless of *)
(* whether its author is still alive.                                       *)
NoLostAckedWrite == \A c \in Clients : (c \in AckedSet) ~> (c \in AppliedSet)

=============================================================================
