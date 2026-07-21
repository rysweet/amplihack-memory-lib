--------------------------- MODULE FederatedLoss ---------------------------
(***************************************************************************)
(* Why the FEDERATED per-agent model (design B) is WRONG for Simard's      *)
(* ephemeral engineers.                                                    *)
(*                                                                         *)
(* In model B each engineer writes to its OWN local store (durable only    *)
(* within that engineer's lifetime / worktree) and must later CONSOLIDATE  *)
(* those writes into the shared hive.  The engineer treats a write as done *)
(* ("acked") the moment it lands in its local store -- that is the API it   *)
(* is given.  But Simard reaps engineers on heartbeat-staleness, NOT in    *)
(* coordination with consolidation.  So an engineer can die after          *)
(* producing but before consolidating, and its local store (worktree) is   *)
(* destroyed -> the acked write is permanently lost.                       *)
(*                                                                         *)
(* This module model-checks the SAME liveness property design C satisfies  *)
(*   NoLostAckedWrite: every acked write eventually reaches the shared      *)
(*   durable store.                                                        *)
(* and shows it is VIOLATED under model B -- a formal justification for     *)
(* rejecting the federated model, complementing DurableLog.tla which shows *)
(* design C satisfies it.                                                  *)
(***************************************************************************)
EXTENDS Naturals

CONSTANTS Clients

(* pc:                                                                      *)
(*   "start"        - nothing produced yet                                  *)
(*   "produced"     - write is in the engineer's LOCAL store (acked to it)  *)
(*   "consolidated" - write has been merged into the shared hive (durable)  *)
(*   "dead"         - engineer reaped; local store destroyed                *)
VARIABLES
  acked,   \* set of clients whose write reached their local store (acked)
  hive,    \* set of clients whose write reached the shared durable hive
  pc       \* [Clients -> {"start","produced","consolidated","dead"}]

vars == <<acked, hive, pc>>

Init ==
  /\ acked = {}
  /\ hive  = {}
  /\ pc = [c \in Clients |-> "start"]

(* Engineer writes to its local store.  From the engineer's viewpoint this  *)
(* is the ack -- the write "succeeded".                                     *)
Produce(c) ==
  /\ pc[c] = "start"
  /\ acked' = acked \cup {c}
  /\ pc' = [pc EXCEPT ![c] = "produced"]
  /\ UNCHANGED hive

(* Engineer merges its local store into the shared hive.                    *)
Consolidate(c) ==
  /\ pc[c] = "produced"
  /\ hive' = hive \cup {c}
  /\ pc' = [pc EXCEPT ![c] = "consolidated"]
  /\ UNCHANGED acked

(* Reaper kills the engineer AFTER it produced but BEFORE it consolidated.  *)
(* The local store (worktree) is destroyed -> the acked write is lost.      *)
ReapAfterProduce(c) ==
  /\ pc[c] = "produced"
  /\ pc' = [pc EXCEPT ![c] = "dead"]
  /\ UNCHANGED <<acked, hive>>

(* Reaper kills an engineer that never produced -- nothing lost.            *)
ReapBeforeProduce(c) ==
  /\ pc[c] = "start"
  /\ pc' = [pc EXCEPT ![c] = "dead"]
  /\ UNCHANGED <<acked, hive>>

Next ==
  \E c \in Clients :
       Produce(c) \/ Consolidate(c)
    \/ ReapAfterProduce(c) \/ ReapBeforeProduce(c)

(* Be maximally generous to model B: assume the engine WOULD consolidate    *)
(* if given the chance (weak fairness on Consolidate).  The property still  *)
(* fails, because the reaper is not coordinated with consolidation.         *)
Spec == Init /\ [][Next]_vars /\ \A c \in Clients : WF_vars(Consolidate(c))

(* Same liveness goal as design C: every acked write eventually durable.    *)
NoLostAckedWrite == \A c \in Clients : (c \in acked) ~> (c \in hive)

=============================================================================
