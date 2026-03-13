//! Security layer for memory-enabled agents.
//!
//! Provides capability-based access control, credential scrubbing,
//! and query validation to ensure safe memory operations.

use regex::Regex;
use std::sync::LazyLock;

use crate::errors::MemoryError;
use crate::experience::{Experience, ExperienceType};

/// Memory access scope levels, ordered from most restrictive to least.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScopeLevel {
    /// Access limited to the current session only.
    SessionOnly,
    /// Read access across sessions of the same agent.
    CrossSessionRead,
    /// Read and write access across sessions of the same agent.
    CrossSessionWrite,
    /// Read access to any agent's memory in the system.
    GlobalRead,
    /// Full read/write access to any agent's memory in the system.
    GlobalWrite,
}

impl ScopeLevel {
    fn hierarchy_index(&self) -> usize {
        match self {
            Self::SessionOnly => 0,
            Self::CrossSessionRead => 1,
            Self::CrossSessionWrite => 2,
            Self::GlobalRead => 3,
            Self::GlobalWrite => 4,
        }
    }
}

/// Security capabilities governing what a memory-enabled agent can do.
///
/// Capabilities are checked before every store/retrieve/query operation
/// by [`SecureMemoryBackend`].
pub struct AgentCapabilities {
    /// The maximum scope this agent is allowed to access.
    pub scope: ScopeLevel,
    /// Which experience types the agent may store and retrieve.
    /// An empty list means all types are allowed.
    pub allowed_experience_types: Vec<ExperienceType>,
    /// Maximum estimated query cost before a query is rejected.
    pub max_query_cost: i32,
    /// Whether the agent may access pattern-type experiences.
    pub can_access_patterns: bool,
    /// Maximum storage quota in megabytes.
    pub memory_quota_mb: i32,
}

impl AgentCapabilities {
    /// Create agent capabilities with the given scope, allowed experience types,
    /// query cost limit, pattern access flag, and memory quota.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::InvalidInput`] if `max_query_cost` or
    /// `memory_quota_mb` is not positive.
    pub fn new(
        scope: ScopeLevel,
        allowed_experience_types: Vec<ExperienceType>,
        max_query_cost: i32,
        can_access_patterns: bool,
        memory_quota_mb: i32,
    ) -> crate::Result<Self> {
        if max_query_cost <= 0 {
            return Err(MemoryError::InvalidInput(
                "max_query_cost must be positive integer".into(),
            ));
        }
        if memory_quota_mb <= 0 {
            return Err(MemoryError::InvalidInput(
                "memory_quota_mb must be positive integer".into(),
            ));
        }
        Ok(Self {
            scope,
            allowed_experience_types,
            max_query_cost,
            can_access_patterns,
            memory_quota_mb,
        })
    }

    /// Check if agent can store this experience type.
    pub fn can_store_experience(&self, experience: &Experience) -> bool {
        if self.allowed_experience_types.is_empty() {
            return true;
        }
        self.allowed_experience_types
            .contains(&experience.experience_type)
    }

    /// Check if agent can retrieve this experience type.
    pub fn can_retrieve_experience_type(&self, experience_type: ExperienceType) -> bool {
        if self.allowed_experience_types.is_empty() {
            return true;
        }
        if experience_type == ExperienceType::Pattern && !self.can_access_patterns {
            return false;
        }
        self.allowed_experience_types.contains(&experience_type)
    }

    /// Check if agent has sufficient scope permission.
    pub fn can_access_scope(&self, target_scope: ScopeLevel) -> bool {
        self.scope.hierarchy_index() >= target_scope.hierarchy_index()
    }
}

/// Detect and redact sensitive credentials in experience data.
pub struct CredentialScrubber {
    patterns: Vec<(&'static str, Regex)>,
}

static CREDENTIAL_PATTERNS: LazyLock<Vec<(&str, Regex)>> = LazyLock::new(|| {
    vec![
        ("aws_key", Regex::new(r"AKIA[0-9A-Z]{16}").unwrap()),
        (
            "aws_secret",
            Regex::new(r#"(aws_secret_access_key["\'\s:=]+)([A-Za-z0-9/+=]{40})"#).unwrap(),
        ),
        ("openai_key", Regex::new(r"sk-[A-Za-z0-9]{20,}").unwrap()),
        (
            "github_token",
            Regex::new(r"gh[pousr]_[A-Za-z0-9]{36,}").unwrap(),
        ),
        (
            "azure_key",
            Regex::new(r#"(?i)((?:azure|subscription|account)[_\-\s]*(?:key|token|secret)["\s:=]*)([0-9a-fA-F]{32,})"#).unwrap(),
        ),
        (
            "password",
            Regex::new(r#"(?i)(password["\'\s]*[=:]\s*"?)([^\s"\']{8,})"#).unwrap(),
        ),
        (
            "token",
            Regex::new(r#"(?i)(token["\'\s]*[=:]\s*"?)([A-Za-z0-9\-._~+/]{8,}=*)"#).unwrap(),
        ),
        (
            "api_key",
            Regex::new(r#"(?i)(api[_\-]?key["\'\s]*[=:]\s*"?)([A-Za-z0-9\-._~+/]{8,}=*)"#).unwrap(),
        ),
        (
            "secret",
            Regex::new(r#"(?i)(secret["\'\s]*[=:]\s*"?)([A-Za-z0-9\-._~+/]{8,}=*)"#).unwrap(),
        ),
        (
            "ssh_key",
            Regex::new(r"-----BEGIN (?:RSA|DSA|EC|OPENSSH) PRIVATE KEY-----").unwrap(),
        ),
        (
            "db_url",
            Regex::new(r"((?:postgres|mysql|mongodb)://[^:]+:)([^@]+)(@)").unwrap(),
        ),
    ]
});

const REDACTION_TEXT: &str = "[REDACTED]";

impl Default for CredentialScrubber {
    fn default() -> Self {
        Self::new()
    }
}

impl CredentialScrubber {
    /// Create a new scrubber initialised with the default credential patterns.
    pub fn new() -> Self {
        Self {
            patterns: CREDENTIAL_PATTERNS.clone(),
        }
    }

    /// Scrub sensitive data from experience. Returns (scrubbed, was_scrubbed).
    pub fn scrub_experience(&self, experience: &Experience) -> crate::Result<(Experience, bool)> {
        let (scrubbed_context, ctx_modified) = self.scrub_text(&experience.context);
        let (scrubbed_outcome, out_modified) = self.scrub_text(&experience.outcome);
        let was_scrubbed = ctx_modified || out_modified;

        let mut tags = experience.tags.clone();
        if was_scrubbed && !tags.contains(&"credential_scrubbed".to_string()) {
            tags.push("credential_scrubbed".to_string());
        }

        let scrubbed = Experience::from_parts(
            experience.experience_id.clone(),
            experience.experience_type,
            scrubbed_context,
            scrubbed_outcome,
            experience.confidence,
            experience.timestamp,
            experience.metadata.clone(),
            tags,
        )
        .map_err(|e| {
            MemoryError::Internal(format!("failed to reconstruct scrubbed experience: {e}"))
        })?;

        Ok((scrubbed, was_scrubbed))
    }

    /// Scrub credentials from text. Returns (scrubbed_text, was_modified).
    pub fn scrub_text(&self, text: &str) -> (String, bool) {
        let mut scrubbed = text.to_string();
        let mut modified = false;

        for (name, pattern) in &self.patterns {
            if pattern.is_match(&scrubbed) {
                let replacement = match *name {
                    "db_url" => {
                        format!("${{1}}{REDACTION_TEXT}${{3}}")
                    }
                    _ => REDACTION_TEXT.to_string(),
                };
                scrubbed = pattern
                    .replace_all(&scrubbed, replacement.as_str())
                    .to_string();
                modified = true;
            }
        }

        (scrubbed, modified)
    }

    /// Check if text contains credentials without scrubbing.
    pub fn contains_credentials(&self, text: &str) -> bool {
        self.patterns
            .iter()
            .any(|(_, pattern)| pattern.is_match(text))
    }
}

/// Validate and estimate cost of database queries.
///
/// Assigns cost points based on SQL patterns (JOINs, subqueries,
/// unbounded SELECTs) and rejects queries exceeding a configured limit.
pub struct QueryValidator;

static COST_PATTERNS: LazyLock<Vec<(&str, Regex, i32)>> = LazyLock::new(|| {
    vec![
        (
            "full_scan",
            Regex::new(r"(?i)SELECT\s+\*\s+FROM\s+\w+\s*(?:WHERE)?").unwrap(),
            10,
        ),
        ("join", Regex::new(r"(?i)\bJOIN\b").unwrap(), 5),
        (
            "subquery",
            Regex::new(r"(?is)SELECT.*\(.*SELECT").unwrap(),
            3,
        ),
        ("order_by", Regex::new(r"(?i)\bORDER BY\b").unwrap(), 2),
        // no_limit is checked separately since Rust regex doesn't support lookahead
    ]
});

static LIMIT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)\bLIMIT\b").unwrap());

impl QueryValidator {
    /// Estimate the cost of a SQL query based on pattern analysis.
    ///
    /// Cost factors: full table scans (+10), JOINs (+5 each), subqueries (+3),
    /// ORDER BY (+2), and missing LIMIT (+20). Base cost is 1.
    pub fn estimate_cost(sql: &str) -> i32 {
        let mut cost = 1; // Base cost

        for (name, pattern, points) in COST_PATTERNS.iter() {
            if *name == "join" {
                let matches = pattern.find_iter(sql).count();
                cost += (matches as i32) * points;
            } else if pattern.is_match(sql) {
                cost += points;
            }
        }

        // Check for no LIMIT (penalize unbounded queries)
        let sql_upper = sql.to_uppercase();
        if sql_upper.contains("SELECT") && !LIMIT_RE.is_match(sql) {
            cost += 20;
        }

        cost
    }

    /// Validate that a query's estimated cost does not exceed `max_cost`.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::QueryCostExceeded`] if the cost exceeds the limit.
    pub fn validate_query(sql: &str, max_cost: i32) -> crate::Result<()> {
        let cost = Self::estimate_cost(sql);
        if cost > max_cost {
            return Err(MemoryError::QueryCostExceeded(format!(
                "Query cost ({cost}) exceeds limit ({max_cost}). \
                 Consider adding LIMIT, reducing JOINs, or simplifying query."
            )));
        }
        Ok(())
    }

    /// Check if a SQL query is non-destructive (read-only SELECT).
    ///
    /// Returns `false` for DELETE, UPDATE, INSERT, DROP, and other DDL/DML.
    pub fn is_safe_query(sql: &str) -> bool {
        let sql_trimmed = sql.trim();

        if !sql_trimmed
            .chars()
            .take(6)
            .collect::<String>()
            .eq_ignore_ascii_case("select")
        {
            return false;
        }

        static DESTRUCTIVE_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
            [
                "DELETE", "UPDATE", "INSERT", "DROP", "TRUNCATE", "ALTER", "CREATE", "REPLACE",
                "GRANT", "REVOKE",
            ]
            .iter()
            .map(|kw| Regex::new(&format!(r"(?i)\b{kw}\b")).unwrap())
            .collect()
        });

        for pattern in DESTRUCTIVE_PATTERNS.iter() {
            if pattern.is_match(sql_trimmed) {
                return false;
            }
        }

        true
    }
}

/// Wrapper around an experience backend that enforces capability-based access control.
///
/// All write operations are credential-scrubbed before reaching the inner store.
pub struct SecureMemoryBackend<S> {
    /// The underlying experience storage backend.
    pub store: S,
    /// Capabilities controlling what operations are permitted.
    pub capabilities: AgentCapabilities,
    scrubber: CredentialScrubber,
}

impl<S: crate::backends::ExperienceBackend> SecureMemoryBackend<S> {
    /// Wrap an experience backend with security enforcement.
    ///
    /// Applies `capabilities` for access control and scrubs credentials on write.
    pub fn new(store: S, capabilities: AgentCapabilities) -> Self {
        Self {
            store,
            capabilities,
            scrubber: CredentialScrubber::new(),
        }
    }

    /// Add experience with security checks.
    pub fn add_experience(&mut self, experience: &Experience) -> crate::Result<String> {
        if !self.capabilities.can_store_experience(experience) {
            return Err(MemoryError::SecurityViolation(format!(
                "Agent not allowed to store {} experiences",
                experience.experience_type
            )));
        }

        let (scrubbed, _) = self.scrubber.scrub_experience(experience)?;
        self.store.add(&scrubbed)
    }

    /// Search experiences with security checks.
    pub fn search(
        &self,
        query: &str,
        experience_type: Option<ExperienceType>,
        min_confidence: f64,
        limit: usize,
    ) -> crate::Result<Vec<Experience>> {
        if let Some(et) = experience_type {
            if !self.capabilities.can_retrieve_experience_type(et) {
                return Err(MemoryError::SecurityViolation(format!(
                    "Agent not allowed to retrieve {} experiences",
                    et
                )));
            }
        }

        self.store
            .search(query, experience_type, min_confidence, limit)
    }

    /// Validate custom SQL query.
    pub fn validate_custom_query(&self, sql: &str) -> crate::Result<()> {
        if !QueryValidator::is_safe_query(sql) {
            return Err(MemoryError::SecurityViolation(
                "Only SELECT queries are allowed".into(),
            ));
        }
        QueryValidator::validate_query(sql, self.capabilities.max_query_cost)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scope_hierarchy() {
        let caps =
            AgentCapabilities::new(ScopeLevel::CrossSessionRead, vec![], 50, false, 10).unwrap();
        assert!(caps.can_access_scope(ScopeLevel::SessionOnly));
        assert!(caps.can_access_scope(ScopeLevel::CrossSessionRead));
        assert!(!caps.can_access_scope(ScopeLevel::CrossSessionWrite));
    }

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
        let cost = QueryValidator::estimate_cost(
            "SELECT * FROM t WHERE id IN (SELECT id FROM s) LIMIT 10",
        );
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
}
