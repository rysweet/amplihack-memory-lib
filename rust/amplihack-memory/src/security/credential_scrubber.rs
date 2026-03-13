//! Credential detection and redaction for experience data.

use regex::Regex;
use std::sync::LazyLock;

use crate::errors::MemoryError;
use crate::experience::Experience;

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
        // Design choice: unlike Python which matches any 32-char hex string,
        // the Rust regex requires a context label (azure, subscription, account)
        // before the hex value. This avoids false positives on arbitrary hex
        // strings like UUIDs, checksums, and commit hashes.
        (
            "azure_key",
            Regex::new(r#"(?i)((?:azure|subscription|account)[_\-\s]*(?:key|token|secret)["\s:=]*)([0-9a-fA-F]{32,})"#).unwrap(),
        ),
        (
            "password",
            Regex::new(r#"(?i)(password["\'\s]*[=:]\s*"?)([^\s"\']+)"#).unwrap(),
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
        (
            "gcp_service_account",
            Regex::new(r#""type"\s*:\s*"service_account""#).unwrap(),
        ),
        (
            "slack_webhook",
            Regex::new(r"https://hooks\.slack\.com/services/[A-Za-z0-9/]+").unwrap(),
        ),
        (
            "stripe_key",
            Regex::new(r"sk_(?:live|test)_[A-Za-z0-9]{24,}").unwrap(),
        ),
        (
            "jwt_token",
            Regex::new(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}").unwrap(),
        ),
        (
            "bearer_token",
            Regex::new(r"(Bearer\s+)([A-Za-z0-9._~+/=-]{20,})").unwrap(),
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
                    "bearer_token" => {
                        format!("${{1}}{REDACTION_TEXT}")
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
