use super::super::*;

// -- AgentCapabilities validation --

#[test]
fn test_agent_capabilities_validation() {
    let result = AgentCapabilities::new(ScopeLevel::SessionOnly, vec![], 0, false, 10);
    assert!(result.is_err());

    let result = AgentCapabilities::new(ScopeLevel::SessionOnly, vec![], 10, false, 0);
    assert!(result.is_err());
}

#[test]
fn test_experience_type_filtering() {
    let caps = AgentCapabilities::new(
        ScopeLevel::SessionOnly,
        vec![ExperienceType::Success, ExperienceType::Failure],
        50,
        false,
        10,
    )
    .unwrap();

    assert!(caps.can_retrieve_experience_type(ExperienceType::Success));
    assert!(!caps.can_retrieve_experience_type(ExperienceType::Pattern));
    assert!(!caps.can_retrieve_experience_type(ExperienceType::Insight));
}

#[test]
fn test_pattern_access_control() {
    let caps_no_pattern = AgentCapabilities::new(
        ScopeLevel::SessionOnly,
        vec![ExperienceType::Pattern],
        50,
        false, // can_access_patterns = false
        10,
    )
    .unwrap();
    assert!(!caps_no_pattern.can_retrieve_experience_type(ExperienceType::Pattern));

    let caps_with_pattern = AgentCapabilities::new(
        ScopeLevel::SessionOnly,
        vec![ExperienceType::Pattern],
        50,
        true, // can_access_patterns = true
        10,
    )
    .unwrap();
    assert!(caps_with_pattern.can_retrieve_experience_type(ExperienceType::Pattern));
}
