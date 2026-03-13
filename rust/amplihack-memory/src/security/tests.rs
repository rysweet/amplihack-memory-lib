use super::*;

#[test]
fn test_credential_scrubber() {
    let scrubber = CredentialScrubber::new();
    assert!(scrubber.contains_credentials("my key sk-abcdefghijklmnopqrst12345"));
    let (scrubbed, modified) = scrubber.scrub_text("key is sk-abcdefghijklmnopqrst12345");
    assert!(modified);
    assert!(scrubbed.contains("[REDACTED]"));
    assert!(!scrubbed.contains("sk-"));
}

#[test]
fn test_query_validator_safe() {
    assert!(QueryValidator::is_safe_query("SELECT * FROM experiences"));
    assert!(!QueryValidator::is_safe_query("DELETE FROM experiences"));
    assert!(!QueryValidator::is_safe_query("DROP TABLE experiences"));
}

#[test]
fn test_query_cost() {
    let cost = QueryValidator::estimate_cost("SELECT * FROM t");
    assert!(cost > 1); // Has full_scan penalty
}

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

#[test]
fn test_plain_md5_not_scrubbed() {
    let scrubber = CredentialScrubber::new();
    let md5 = "d41d8cd98f00b204e9800998ecf8427e";
    let (scrubbed, modified) = scrubber.scrub_text(md5);
    assert!(!modified, "Plain MD5 hash should not be scrubbed");
    assert_eq!(scrubbed, md5);
}

#[test]
fn test_azure_key_with_context_scrubbed() {
    let scrubber = CredentialScrubber::new();
    let text = r#"azure_key="abcdef01234567890abcdef012345678""#;
    let (scrubbed, modified) = scrubber.scrub_text(text);
    assert!(modified, "Azure key with context should be scrubbed");
    assert!(scrubbed.contains("[REDACTED]"));
}

#[test]
fn test_is_safe_query_with_created_at() {
    assert!(
        QueryValidator::is_safe_query("SELECT * FROM t WHERE created_at > '2024-01-01'"),
        "Column name 'created_at' should not trigger CREATE keyword match"
    );
}

#[test]
fn test_is_safe_query_with_updated_at() {
    assert!(
        QueryValidator::is_safe_query("SELECT * FROM t WHERE updated_at IS NOT NULL"),
        "Column name 'updated_at' should not trigger UPDATE keyword match"
    );
}

#[test]
fn test_is_safe_query_rejects_real_create() {
    assert!(!QueryValidator::is_safe_query(
        "SELECT 1; CREATE TABLE evil (id INT)"
    ));
}

// --- tst-001: SecureMemoryBackend tests ---

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

// ====================================================================
// Extended coverage: CredentialScrubber::scrub_text
// ====================================================================

#[test]
fn test_scrub_aws_access_key() {
    let scrubber = CredentialScrubber::new();
    let (scrubbed, modified) = scrubber.scrub_text("key is AKIAIOSFODNN7EXAMPLE");
    assert!(modified);
    assert!(scrubbed.contains("[REDACTED]"));
    assert!(!scrubbed.contains("AKIA"));
}

#[test]
fn test_scrub_aws_secret_key() {
    let scrubber = CredentialScrubber::new();
    let text = r#"aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY1""#;
    let (scrubbed, modified) = scrubber.scrub_text(text);
    assert!(modified);
    assert!(scrubbed.contains("[REDACTED]"));
}

#[test]
fn test_scrub_github_token() {
    let scrubber = CredentialScrubber::new();
    let token = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij1234";
    let (scrubbed, modified) = scrubber.scrub_text(&format!("token: {token}"));
    assert!(modified);
    assert!(!scrubbed.contains("ghp_"));
}

#[test]
fn test_scrub_password_field() {
    let scrubber = CredentialScrubber::new();
    let (scrubbed, modified) = scrubber.scrub_text(r#"password="SuperSecret123!""#);
    assert!(modified);
    assert!(!scrubbed.contains("SuperSecret123!"));
}

#[test]
fn test_scrub_generic_api_key() {
    let scrubber = CredentialScrubber::new();
    let (scrubbed, modified) = scrubber.scrub_text(r#"api_key="abc123def456ghi789""#);
    assert!(modified);
    assert!(!scrubbed.contains("abc123def456ghi789"));
}

#[test]
fn test_scrub_generic_secret() {
    let scrubber = CredentialScrubber::new();
    let (scrubbed, modified) = scrubber.scrub_text(r#"secret="myTopSecretValue""#);
    assert!(modified);
    assert!(!scrubbed.contains("myTopSecretValue"));
}

#[test]
fn test_scrub_ssh_key() {
    let scrubber = CredentialScrubber::new();
    let text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpA...";
    let (scrubbed, modified) = scrubber.scrub_text(text);
    assert!(modified);
    assert!(scrubbed.contains("[REDACTED]"));
}

#[test]
fn test_scrub_database_url() {
    let scrubber = CredentialScrubber::new();
    let text = "postgres://admin:s3cret@localhost:5432/mydb";
    let (scrubbed, modified) = scrubber.scrub_text(text);
    assert!(modified);
    assert!(!scrubbed.contains("s3cret"));
}

#[test]
fn test_scrub_no_credentials() {
    let scrubber = CredentialScrubber::new();
    let text = "This is a normal text with no credentials at all.";
    let (scrubbed, modified) = scrubber.scrub_text(text);
    assert!(!modified);
    assert_eq!(scrubbed, text);
}

#[test]
fn test_scrub_multiple_credentials_in_one_text() {
    let scrubber = CredentialScrubber::new();
    let text = r#"key=sk-abcdefghijklmnopqrst12345 and password="hunter2secret""#;
    let (scrubbed, modified) = scrubber.scrub_text(text);
    assert!(modified);
    assert!(!scrubbed.contains("sk-"));
    assert!(!scrubbed.contains("hunter2secret"));
}

#[test]
fn test_contains_credentials_true() {
    let scrubber = CredentialScrubber::new();
    assert!(scrubber.contains_credentials("AKIAIOSFODNN7EXAMPLE"));
    assert!(scrubber.contains_credentials("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij1234"));
}

#[test]
fn test_contains_credentials_false() {
    let scrubber = CredentialScrubber::new();
    assert!(!scrubber.contains_credentials("just a normal string"));
}

// ====================================================================
// Extended coverage: QueryValidator::estimate_cost
// ====================================================================

#[test]
fn test_estimate_cost_simple_with_limit() {
    let cost = QueryValidator::estimate_cost("SELECT id FROM t LIMIT 10");
    // Base(1) only — no full_scan, has LIMIT
    assert_eq!(cost, 1);
}

#[test]
fn test_estimate_cost_with_join() {
    let cost = QueryValidator::estimate_cost("SELECT * FROM a JOIN b ON a.id = b.id LIMIT 10");
    // Base(1) + full_scan(10) + join(5) = 16, has LIMIT so no +20
    assert!(cost >= 16, "Expected cost >= 16, got {cost}");
}

#[test]
fn test_estimate_cost_multiple_joins() {
    let cost = QueryValidator::estimate_cost(
        "SELECT * FROM a JOIN b ON a.id = b.id JOIN c ON b.id = c.id LIMIT 10",
    );
    // Base(1) + full_scan(10) + 2*join(5) = 21
    assert!(
        cost >= 21,
        "Expected cost >= 21 for double join, got {cost}"
    );
}

#[test]
fn test_estimate_cost_order_by() {
    let cost = QueryValidator::estimate_cost("SELECT * FROM t ORDER BY name LIMIT 10");
    // Base(1) + full_scan(10) + order_by(2) = 13
    assert!(cost >= 13, "Expected cost >= 13, got {cost}");
}

#[test]
fn test_estimate_cost_no_limit_penalty() {
    let cost_no_limit = QueryValidator::estimate_cost("SELECT * FROM t");
    let cost_with_limit = QueryValidator::estimate_cost("SELECT * FROM t LIMIT 10");
    assert!(
        cost_no_limit > cost_with_limit,
        "No-limit query should cost more"
    );
    // Difference should be the 20-point penalty
    assert_eq!(cost_no_limit - cost_with_limit, 20);
}

#[test]
fn test_estimate_cost_subquery() {
    let cost =
        QueryValidator::estimate_cost("SELECT * FROM t WHERE id IN (SELECT id FROM s) LIMIT 10");
    // Base(1) + full_scan(10) + subquery(3) = 14
    assert!(cost >= 14, "Expected cost >= 14 for subquery, got {cost}");
}

// ====================================================================
// Extended coverage: QueryValidator::validate_query
// ====================================================================

#[test]
fn test_validate_query_passes_under_limit() {
    assert!(QueryValidator::validate_query("SELECT id FROM t LIMIT 10", 50).is_ok());
}

#[test]
fn test_validate_query_fails_over_limit() {
    let result = QueryValidator::validate_query("SELECT * FROM t", 5);
    assert!(result.is_err());
    match result.unwrap_err() {
        MemoryError::QueryCostExceeded(msg) => {
            assert!(msg.contains("exceeds limit"));
        }
        other => panic!("expected QueryCostExceeded, got {other:?}"),
    }
}

#[test]
fn test_validate_query_at_exact_limit() {
    let cost = QueryValidator::estimate_cost("SELECT id FROM t LIMIT 10");
    // Passes when max_cost equals cost
    assert!(QueryValidator::validate_query("SELECT id FROM t LIMIT 10", cost).is_ok());
    // Fails when max_cost is one less
    assert!(QueryValidator::validate_query("SELECT id FROM t LIMIT 10", cost - 1).is_err());
}

#[test]
fn test_is_safe_query_insert_rejected() {
    assert!(!QueryValidator::is_safe_query(
        "INSERT INTO t VALUES (1, 'a')"
    ));
}

#[test]
fn test_is_safe_query_grant_rejected() {
    assert!(!QueryValidator::is_safe_query(
        "SELECT 1; GRANT ALL ON t TO user"
    ));
}

#[test]
fn test_is_safe_query_simple_select_accepted() {
    assert!(QueryValidator::is_safe_query(
        "SELECT id, name FROM users WHERE active = 1 LIMIT 100"
    ));
}

#[test]
fn test_scrub_short_password() {
    let scrubber = CredentialScrubber::new();
    let (scrubbed, modified) = scrubber.scrub_text(r#"password="abc""#);
    assert!(modified, "short password should be scrubbed");
    assert!(
        !scrubbed.contains("abc"),
        "password value should be removed"
    );
    assert!(scrubbed.contains("[REDACTED]"));
}

#[test]
fn test_scrub_single_char_password() {
    let scrubber = CredentialScrubber::new();
    let (scrubbed, modified) = scrubber.scrub_text("password=x");
    assert!(modified, "single-char password should be scrubbed");
    assert!(!scrubbed.contains("=x"));
}

#[test]
fn test_redaction_full_replacement_no_prefix() {
    let scrubber = CredentialScrubber::new();
    let (scrubbed, _) = scrubber.scrub_text("key is AKIAIOSFODNN7EXAMPLE");
    assert_eq!(
        scrubbed, "key is [REDACTED]",
        "AWS key should be fully replaced"
    );
    let (scrubbed, _) = scrubber.scrub_text("found sk-abcdefghijklmnopqrst12345");
    assert_eq!(
        scrubbed, "found [REDACTED]",
        "OpenAI key should be fully replaced"
    );
}

#[test]
fn test_redaction_password_full_replacement() {
    let scrubber = CredentialScrubber::new();
    let (scrubbed, _) = scrubber.scrub_text(r#"password="SuperSecret123!""#);
    assert!(
        scrubbed.starts_with("[REDACTED]"),
        "should be fully redacted: {scrubbed}"
    );
}

// ====================================================================
// #22: Blocked keywords — sqlite_master, UNION, ATTACH, PRAGMA
// ====================================================================

#[test]
fn test_is_safe_query_blocks_sqlite_master() {
    assert!(!QueryValidator::is_safe_query(
        "SELECT * FROM sqlite_master"
    ));
    assert!(!QueryValidator::is_safe_query(
        "SELECT * FROM sqlite_temp_master"
    ));
}

#[test]
fn test_is_safe_query_blocks_union() {
    assert!(!QueryValidator::is_safe_query("SELECT 1 UNION SELECT 2"));
}

#[test]
fn test_is_safe_query_blocks_attach() {
    assert!(!QueryValidator::is_safe_query(
        "SELECT 1; ATTACH DATABASE ':memory:' AS evil"
    ));
}

#[test]
fn test_is_safe_query_blocks_pragma() {
    assert!(!QueryValidator::is_safe_query(
        "SELECT 1; PRAGMA table_info(experiences)"
    ));
}

// ====================================================================
// #23: New credential patterns — GCP, Slack, Stripe, JWT, Bearer
// ====================================================================

#[test]
fn test_scrub_gcp_service_account() {
    let scrubber = CredentialScrubber::new();
    let text = r#"{"type": "service_account", "project_id": "my-project"}"#;
    assert!(scrubber.contains_credentials(text));
    let (scrubbed, modified) = scrubber.scrub_text(text);
    assert!(modified);
    assert!(scrubbed.contains("[REDACTED]"));
}

#[test]
fn test_scrub_slack_webhook() {
    let scrubber = CredentialScrubber::new();
    // Use a truncated URL that matches our pattern but won't trigger secret scanners.
    let text = "webhook: https://hooks.slack.com/services/T0/B0/XXXX";
    assert!(scrubber.contains_credentials(text));
    let (scrubbed, modified) = scrubber.scrub_text(text);
    assert!(modified);
    assert!(!scrubbed.contains("hooks.slack.com"));
}

#[test]
fn test_scrub_stripe_key() {
    let scrubber = CredentialScrubber::new();
    // Build fake key at runtime to avoid push-protection false positives.
    let fake_key = format!("sk_liv{}{}", "e_", "0".repeat(24));
    let text = format!("stripe_key={fake_key}");
    assert!(scrubber.contains_credentials(&text));
    let (scrubbed, modified) = scrubber.scrub_text(&text);
    assert!(modified);
    assert!(!scrubbed.contains("sk_liv"));
}

#[test]
fn test_scrub_stripe_test_key() {
    let scrubber = CredentialScrubber::new();
    let fake_key = format!("sk_tes{}{}", "t_", "0".repeat(24));
    assert!(scrubber.contains_credentials(&fake_key));
}

#[test]
fn test_scrub_jwt_token() {
    let scrubber = CredentialScrubber::new();
    let text = "auth: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U";
    assert!(scrubber.contains_credentials(text));
    let (scrubbed, modified) = scrubber.scrub_text(text);
    assert!(modified);
    assert!(!scrubbed.contains("eyJ"));
}

#[test]
fn test_scrub_bearer_token() {
    let scrubber = CredentialScrubber::new();
    let text = "Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345";
    assert!(scrubber.contains_credentials(text));
    let (scrubbed, modified) = scrubber.scrub_text(text);
    assert!(modified);
    assert!(!scrubbed.contains("abcdefghijklmnopqrstuvwxyz012345"));
}

// ====================================================================
// #36: Cypher safety validation
// ====================================================================

#[test]
fn test_is_safe_cypher_allows_match_return() {
    assert!(QueryValidator::is_safe_cypher(
        "MATCH (e:Experience) WHERE e.agent = $agent RETURN e.experience_id"
    ));
}

#[test]
fn test_is_safe_cypher_rejects_delete() {
    assert!(!QueryValidator::is_safe_cypher(
        "MATCH (e:Experience) DELETE e"
    ));
}

#[test]
fn test_is_safe_cypher_rejects_set() {
    assert!(!QueryValidator::is_safe_cypher(
        "MATCH (e:Experience) SET e.name = 'evil'"
    ));
}

#[test]
fn test_is_safe_cypher_rejects_create() {
    assert!(!QueryValidator::is_safe_cypher(
        "CREATE (:Experience {id: '1'})"
    ));
}

#[test]
fn test_is_safe_cypher_rejects_non_match() {
    assert!(!QueryValidator::is_safe_cypher(
        "DETACH DELETE (e:Experience)"
    ));
}
