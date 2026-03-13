use super::super::*;

/// Minimal in-memory backend for testing SecureMemoryBackend.
struct MockBackend {
    experiences: Vec<Experience>,
}

impl MockBackend {
    fn new() -> Self {
        Self {
            experiences: Vec::new(),
        }
    }
}

impl crate::backends::ExperienceBackend for MockBackend {
    fn add(&mut self, experience: &Experience) -> crate::Result<String> {
        self.experiences.push(experience.clone());
        Ok(experience.experience_id.clone())
    }

    fn search(
        &self,
        _query: &str,
        experience_type: Option<ExperienceType>,
        min_confidence: f64,
        limit: usize,
    ) -> crate::Result<Vec<Experience>> {
        Ok(self
            .experiences
            .iter()
            .filter(|e| {
                experience_type.map_or(true, |et| e.experience_type == et)
                    && e.confidence >= min_confidence
            })
            .take(limit)
            .cloned()
            .collect())
    }

    fn get_statistics(&self) -> crate::Result<crate::backends::base::StorageStatistics> {
        Ok(Default::default())
    }
}

fn make_experience(exp_type: ExperienceType, context: &str, outcome: &str) -> Experience {
    Experience::new(exp_type, context.into(), outcome.into(), 0.8).unwrap()
}

#[test]
fn test_secure_backend_rejects_unauthorized_type() {
    let caps = AgentCapabilities::new(
        ScopeLevel::SessionOnly,
        vec![ExperienceType::Success],
        50,
        false,
        10,
    )
    .unwrap();
    let mut backend = SecureMemoryBackend::new(MockBackend::new(), caps);

    let exp = make_experience(ExperienceType::Pattern, "ctx", "outcome");
    let result = backend.add_experience(&exp);
    assert!(result.is_err());
    match result.unwrap_err() {
        MemoryError::SecurityViolation(_) => {}
        other => panic!("expected SecurityViolation, got {other:?}"),
    }
}

#[test]
fn test_secure_backend_scrubs_credentials() {
    let caps = AgentCapabilities::new(
        ScopeLevel::SessionOnly,
        vec![ExperienceType::Success],
        50,
        false,
        10,
    )
    .unwrap();
    let mut backend = SecureMemoryBackend::new(MockBackend::new(), caps);

    let exp = make_experience(
        ExperienceType::Success,
        "found key sk-abcdefghijklmnopqrst12345",
        "stored it",
    );
    backend.add_experience(&exp).unwrap();

    let stored = &backend.store.experiences[0];
    assert!(
        stored.context.contains("[REDACTED]"),
        "Credential should be scrubbed"
    );
    assert!(
        !stored.context.contains("sk-"),
        "Original credential should not remain"
    );
}

#[test]
fn test_secure_backend_search_filters_type() {
    let caps = AgentCapabilities::new(
        ScopeLevel::SessionOnly,
        vec![ExperienceType::Success],
        50,
        false,
        10,
    )
    .unwrap();
    let backend = SecureMemoryBackend::new(MockBackend::new(), caps);

    let result = backend.search("query", Some(ExperienceType::Pattern), 0.0, 10);
    assert!(result.is_err());
    match result.unwrap_err() {
        MemoryError::SecurityViolation(_) => {}
        other => panic!("expected SecurityViolation, got {other:?}"),
    }
}

#[test]
fn test_secure_backend_validate_rejects_destructive() {
    let caps = AgentCapabilities::new(ScopeLevel::SessionOnly, vec![], 50, false, 10).unwrap();
    let backend = SecureMemoryBackend::new(MockBackend::new(), caps);

    let result = backend.validate_custom_query("DROP TABLE experiences");
    assert!(result.is_err());
}

#[test]
fn test_secure_backend_validate_rejects_over_cost() {
    let caps = AgentCapabilities::new(
        ScopeLevel::SessionOnly,
        vec![],
        2, // very low cost limit
        false,
        10,
    )
    .unwrap();
    let backend = SecureMemoryBackend::new(MockBackend::new(), caps);

    // This query has full_scan + no LIMIT penalty > 2
    let result = backend.validate_custom_query("SELECT * FROM experiences");
    assert!(result.is_err());
    match result.unwrap_err() {
        MemoryError::QueryCostExceeded(_) => {}
        other => panic!("expected QueryCostExceeded, got {other:?}"),
    }
}

#[test]
fn test_scrub_experience_credential_in_tags() {
    let scrubber = CredentialScrubber::new();
    let mut exp = make_experience(ExperienceType::Success, "clean context", "clean outcome");
    exp.tags = vec!["password=SuperSecret123".into(), "safe-tag".into()];
    let (scrubbed, _) = scrubber.scrub_experience(&exp).unwrap();
    assert!(scrubbed.tags.contains(&"safe-tag".to_string()));
}

#[test]
fn test_scrub_experience_credential_in_metadata() {
    let scrubber = CredentialScrubber::new();
    let mut exp = make_experience(ExperienceType::Success, "clean context", "clean outcome");
    exp.metadata
        .insert("api_key".into(), serde_json::json!("abc123def456ghi789"));
    let (scrubbed, _) = scrubber.scrub_experience(&exp).unwrap();
    assert!(scrubbed.metadata.contains_key("api_key"));
}

// -- QA audit --

#[test]
fn test_qa_scrub_metadata() {
    let scrubber = CredentialScrubber::new();
    let mut metadata = std::collections::HashMap::new();
    metadata.insert(
        "notes".to_string(),
        serde_json::Value::String("key is sk-abcdefghijklmnopqrst12345".to_string()),
    );
    let exp = Experience::from_parts(
        "m1".into(),
        ExperienceType::Success,
        "ctx".into(),
        "out".into(),
        0.8,
        chrono::Utc::now(),
        metadata,
        vec![],
    )
    .unwrap();
    let (s, w) = scrubber.scrub_experience(&exp).unwrap();
    assert!(w);
    assert!(s
        .metadata
        .get("notes")
        .unwrap()
        .as_str()
        .unwrap()
        .contains("[REDACTED]"));
}

#[test]
fn test_qa_scrub_tags() {
    let scrubber = CredentialScrubber::new();
    let exp = Experience::from_parts(
        "t1".into(),
        ExperienceType::Success,
        "ctx".into(),
        "out".into(),
        0.8,
        chrono::Utc::now(),
        std::collections::HashMap::new(),
        vec!["sk-abcdefghijklmnopqrst12345".into()],
    )
    .unwrap();
    let (s, w) = scrubber.scrub_experience(&exp).unwrap();
    assert!(w);
    assert!(s.tags.iter().any(|t| t.contains("[REDACTED]")));
}

#[test]
fn test_qa_search_sanitizes_quotes() {
    let caps = AgentCapabilities::new(ScopeLevel::SessionOnly, vec![], 50, false, 10).unwrap();
    let mut backend = SecureMemoryBackend::new(MockBackend::new(), caps);
    let exp = make_experience(ExperienceType::Success, "test ctx", "test out");
    backend.add_experience(&exp).unwrap();
    let result = backend.search("test' OR '1'='1", None, 0.0, 10);
    assert!(result.is_ok());
}
