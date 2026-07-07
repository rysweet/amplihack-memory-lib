use super::super::types::{agent_filter, NT_PROSPECTIVE};
use super::super::*;

fn make_cm() -> CognitiveMemory {
    CognitiveMemory::new(&format!("test-agent-{}", uuid::Uuid::new_v4())).unwrap()
}

// -- prospective memory --

#[test]
fn test_store_and_check_triggers() {
    let mut cm = make_cm();
    cm.store_prospective(
        "notify on deploy",
        "deploy production",
        "send notification",
        5,
    )
    .unwrap();

    // Content that matches
    let triggered = cm.check_triggers("starting deploy to staging");
    assert_eq!(triggered.len(), 1);
    assert_eq!(triggered[0].status, "triggered");
    assert_eq!(triggered[0].action_on_trigger, "send notification");
}

#[test]
fn test_check_triggers_no_match() {
    let mut cm = make_cm();
    cm.store_prospective("remind", "deploy", "do stuff", 1)
        .unwrap();

    let triggered = cm.check_triggers("completely unrelated text");
    assert!(triggered.is_empty());
}

#[test]
fn test_triggered_not_re_triggered() {
    let mut cm = make_cm();
    cm.store_prospective("notify", "deploy", "action", 1)
        .unwrap();

    let t1 = cm.check_triggers("deploy happening");
    assert_eq!(t1.len(), 1);

    // Already triggered — should not trigger again
    let t2 = cm.check_triggers("deploy happening again");
    assert!(t2.is_empty());
}

#[test]
fn test_resolve_prospective() {
    let mut cm = make_cm();
    let id = cm
        .store_prospective("notify", "deploy", "action", 1)
        .unwrap();

    let triggered = cm.check_triggers("deploy now");
    assert_eq!(triggered.len(), 1);

    cm.resolve_prospective(&id);

    // After resolving, it should not appear in pending or triggered checks
    let filter = agent_filter(&cm.agent_name);
    let nodes = cm
        .graph
        .query_nodes(NT_PROSPECTIVE, Some(&filter), usize::MAX);
    let node = nodes.iter().find(|n| n.node_id == id).unwrap();
    assert_eq!(node.properties.get("status").unwrap(), "resolved");
}

#[test]
fn test_prospective_priority_ordering() {
    let mut cm = make_cm();
    cm.store_prospective("low", "deploy", "low-action", 1)
        .unwrap();
    cm.store_prospective("high", "deploy", "high-action", 10)
        .unwrap();

    let triggered = cm.check_triggers("deploy now");
    assert_eq!(triggered.len(), 2);
    // Higher priority first
    assert_eq!(triggered[0].priority, 10);
}

// -- get_all_prospective (Simard #2550): read-only enumeration of ALL statuses --

#[test]
fn test_get_all_prospective_returns_every_status_without_mutating() {
    let mut cm = make_cm();
    cm.store_prospective("goal A", "alphacond", "do A", 1)
        .unwrap();
    let id_b = cm
        .store_prospective("goal B", "betacond", "do B", 9)
        .unwrap();

    // Fire B so it is "triggered" (not "pending") — get_all_prospective must
    // still surface it, unlike check_triggers which only reads pending nodes.
    // Distinct single-token triggers so keyword-overlap fires B only.
    let fired = cm.check_triggers("betacond happening");
    assert_eq!(fired.len(), 1);

    let all = cm.get_all_prospective(usize::MAX);
    assert_eq!(
        all.len(),
        2,
        "must return prospective memories in every status"
    );
    // Priority-ordered (highest first).
    assert_eq!(all[0].priority, 9);
    assert_eq!(all[0].node_id, id_b);
    assert_eq!(all[0].status, "triggered");
    assert_eq!(all[1].status, "pending");

    // Pure read: calling it does not change any status.
    let again = cm.get_all_prospective(usize::MAX);
    assert_eq!(again.len(), 2);
    assert_eq!(again[0].status, "triggered");
    assert_eq!(again[1].status, "pending");
}

#[test]
fn test_get_all_prospective_empty_store() {
    let cm = make_cm();
    assert!(cm.get_all_prospective(usize::MAX).is_empty());
}

// -- issue #124: ordered + trigger-scoped prospective recall --
//
// The truncation bug: `get_all_prospective(limit)` fetched an arbitrary `limit`
// rows (bare `RETURN n LIMIT k`, no ORDER BY) and sorted by priority in Rust
// *afterwards*. In a store holding more than `limit` prospectives a
// non-deterministic subset came back, and low-priority nodes (e.g. creative
// ideas) were silently dropped. The dashboard called `get_all_prospective(512)`
// then filtered by a sentinel `trigger_condition` in Rust, so in a large store
// it surfaced ZERO ideas even though every idea had been persisted.

/// The sentinel `trigger_condition` a dashboard filters on.
const SENTINEL: &str = "creative-idea-thread";

#[test]
fn test_get_prospective_by_trigger_surfaces_all_matches_in_large_store() {
    let mut cm = make_cm();

    // 590 HIGH-priority operational prospectives with an unrelated trigger.
    // These dominate any arbitrary or priority-ordered top-512 window.
    for i in 0..590 {
        cm.store_prospective(
            &format!("ops alert {i}"),
            "deployment failed error",
            "page on-call",
            100,
        )
        .unwrap();
    }
    // 10 LOW-priority creative-idea prospectives carrying the dashboard sentinel.
    for i in 0..10 {
        cm.store_prospective(
            &format!("idea {i}"),
            SENTINEL,
            "surface in the ideas dashboard",
            1,
        )
        .unwrap();
    }

    // The trigger filter is pushed into the query, so the 512 limit bounds only
    // *matching* nodes: all 10 ideas come back regardless of the 590 noisy nodes.
    let ideas = cm
        .get_prospective_by_trigger(SENTINEL, 512)
        .expect("trigger-scoped read must succeed");
    assert_eq!(
        ideas.len(),
        10,
        "every sentinel-tagged prospective must be returned even in a >limit store"
    );
    assert!(
        ideas.iter().all(|p| p.trigger_condition == SENTINEL),
        "only sentinel-tagged nodes should be returned"
    );

    // Contrast documenting the original bug: the old dashboard approach —
    // enumerate the top-512 by priority, then filter in Rust — surfaces ZERO
    // ideas, because the 10 low-priority sentinels sit outside the top-512
    // window the 590 priority-100 ops alerts fill.
    let leaked = cm
        .get_all_prospective(512)
        .into_iter()
        .filter(|p| p.trigger_condition == SENTINEL)
        .count();
    assert_eq!(
        leaked, 0,
        "enumerate-then-filter cannot surface low-priority matches — only the \
         trigger-scoped query can"
    );
}

#[test]
fn test_get_all_prospective_returns_true_top_n_by_priority() {
    let mut cm = make_cm();
    // Insert in ASCENDING priority so a truncate-then-sort backend could have
    // returned the *lowest* priorities. Multi-digit values also make numeric vs
    // lexicographic ordering differ: 10 must outrank 9 (not "10" < "9").
    cm.store_prospective("low", "t", "a", 2).unwrap();
    cm.store_prospective("mid", "t", "a", 9).unwrap();
    cm.store_prospective("high", "t", "a", 10).unwrap();
    cm.store_prospective("top", "t", "a", 100).unwrap();

    let top2 = cm.get_all_prospective(2);
    assert_eq!(
        top2.iter().map(|p| p.priority).collect::<Vec<_>>(),
        vec![100, 10],
        "must return the true top-2 by numeric priority (DB sorts, then truncates)"
    );
}

#[test]
fn test_get_prospective_by_trigger_orders_and_limits() {
    let mut cm = make_cm();
    for p in [3, 1, 5, 2, 4] {
        cm.store_prospective(&format!("idea p{p}"), SENTINEL, "act", p)
            .unwrap();
    }
    // Unrelated high-priority noise that must never crowd out the matches.
    cm.store_prospective("noise", "other-trigger", "act", 99)
        .unwrap();

    let top3 = cm
        .get_prospective_by_trigger(SENTINEL, 3)
        .expect("trigger-scoped read must succeed");
    assert_eq!(top3.len(), 3, "limit must bound the matching rows");
    assert_eq!(
        top3.iter().map(|p| p.priority).collect::<Vec<_>>(),
        vec![5, 4, 3],
        "top-3 matching by priority, highest first"
    );
    assert!(top3.iter().all(|p| p.trigger_condition == SENTINEL));
}

#[test]
fn test_get_prospective_by_trigger_no_match_is_empty_ok() {
    let mut cm = make_cm();
    cm.store_prospective("x", "some-trigger", "a", 1).unwrap();
    let got = cm
        .get_prospective_by_trigger("nonexistent-trigger", 10)
        .expect("a no-match read is a confirmed-empty Ok, not an error");
    assert!(got.is_empty());
}

#[test]
fn test_get_prospective_by_trigger_returns_every_status() {
    let mut cm = make_cm();
    // A pending sentinel node and one that we fire so it becomes "triggered".
    cm.store_prospective("pending idea", SENTINEL, "a", 1)
        .unwrap();
    cm.store_prospective("betacond idea", SENTINEL, "a", 5)
        .unwrap();
    // Fire only the second via keyword overlap on its (multi-word) trigger?  The
    // sentinel is a single token, so firing would mark BOTH. Instead update via a
    // separate trigger node to exercise the triggered status.
    let fired = cm.check_triggers("creative-idea-thread happening");
    assert_eq!(
        fired.len(),
        2,
        "both sentinel nodes fire on the sentinel word"
    );

    // Like get_all_prospective, the trigger-scoped read is status-agnostic.
    let all = cm
        .get_prospective_by_trigger(SENTINEL, usize::MAX)
        .expect("read must succeed");
    assert_eq!(all.len(), 2);
    assert!(all.iter().all(|p| p.status == "triggered"));
}

// -- Extended coverage: store_prospective --

#[test]
fn test_store_prospective_returns_valid_id() {
    let mut cm = make_cm();
    let id = cm
        .store_prospective("remind me", "meeting starts", "join call", 5)
        .unwrap();
    assert!(id.starts_with("pro_"));
}

#[test]
fn test_store_prospective_initial_status_pending() {
    let mut cm = make_cm();
    let id = cm
        .store_prospective("task", "trigger", "action", 1)
        .unwrap();

    let filter = agent_filter(&cm.agent_name);
    let nodes = cm
        .graph
        .query_nodes(NT_PROSPECTIVE, Some(&filter), usize::MAX);
    let node = nodes.iter().find(|n| n.node_id == id).unwrap();
    assert_eq!(node.properties.get("status").unwrap(), "pending");
}

// -- Extended coverage: check_triggers --

#[test]
fn test_check_triggers_partial_word_no_match() {
    let mut cm = make_cm();
    cm.store_prospective("notify", "deployment", "run script", 1)
        .unwrap();

    // "deploy" is not the same word as "deployment" in word-based matching
    let triggered = cm.check_triggers("deploy is happening");
    // This depends on tokenization — "deployment" != "deploy"
    // The trigger uses word-level overlap, so exact word match is needed
    assert!(triggered.is_empty());
}

#[test]
fn test_check_triggers_multiple_triggers() {
    let mut cm = make_cm();
    cm.store_prospective("alert1", "error", "log error", 2)
        .unwrap();
    cm.store_prospective("alert2", "error", "notify admin", 5)
        .unwrap();

    let triggered = cm.check_triggers("an error occurred");
    assert_eq!(triggered.len(), 2);
    // Higher priority first
    assert_eq!(triggered[0].priority, 5);
    assert_eq!(triggered[1].priority, 2);
}
