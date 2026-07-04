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
