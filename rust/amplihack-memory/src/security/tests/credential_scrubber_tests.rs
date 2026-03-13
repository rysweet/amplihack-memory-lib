use super::super::*;

// -- CredentialScrubber --

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
    assert!(scrubbed.contains("password="), "label: {scrubbed}");
    assert!(scrubbed.contains("[REDACTED]"), "redacted: {scrubbed}");
    assert!(
        !scrubbed.contains("SuperSecret123!"),
        "secret gone: {scrubbed}"
    );
}

// -- New credential patterns: GCP, Slack, Stripe, JWT, Bearer --

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

// -- QA audit: label preservation --

#[test]
fn test_qa_preserves_password_label() {
    let scrubber = CredentialScrubber::new();
    let (s, m) = scrubber.scrub_text(r#"password="mysecret""#);
    assert!(m);
    assert!(s.contains("password="));
    assert!(s.contains("[REDACTED]"));
}

#[test]
fn test_qa_preserves_api_key_label() {
    let scrubber = CredentialScrubber::new();
    let (s, _) = scrubber.scrub_text(r#"api_key="abcdef1234567890""#);
    assert!(s.contains("api_key="));
    assert!(s.contains("[REDACTED]"));
}

#[test]
fn test_qa_preserves_secret_label() {
    let scrubber = CredentialScrubber::new();
    let (s, _) = scrubber.scrub_text(r#"secret="TopSecretValue123""#);
    assert!(s.contains("secret="));
    assert!(s.contains("[REDACTED]"));
}
