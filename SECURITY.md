# Security Policy

## Security model overview

amplihack-memory implements defense-in-depth across four layers:

1. **Credential scrubbing** — detects and redacts secrets before they reach storage.
2. **Query validation** — blocks SQL/Cypher injection and limits query cost.
3. **Path traversal protection** — validates file paths in Python bindings.
4. **Scope enforcement** — per-agent capability restrictions (documented stub, enforced at API boundary).

All layers are composed in `SecureMemoryBackend<S>`, which wraps any experience backend and applies checks on every operation. Violations return `MemoryError::SecurityViolation`.

## Credential scrubbing

`CredentialScrubber` scans experience content, metadata, and tags using compiled regex patterns. Matched values are replaced with `[REDACTED]`.

### Detected patterns

| Pattern | What it matches |
|---------|----------------|
| AWS Access Key | `AKIA` followed by 16 alphanumeric characters |
| AWS Secret Key | `aws_secret_access_key` followed by a 40-character base64 value |
| OpenAI API Key | `sk-` prefix with 20+ alphanumeric characters |
| GitHub Token | `gh[pousr]_` prefix with 36+ alphanumeric characters |
| Azure Key | Context-aware — requires `azure`, `subscription`, or `account` label before a 32+ hex value (avoids false positives on UUIDs and commit hashes) |
| Password | `password=` or `password:` followed by a value |
| Generic Token | `token=` or `token:` followed by 8+ characters |
| API Key | `api_key=` / `api-key=` followed by 8+ characters |
| Secret | `secret=` or `secret:` followed by 8+ characters |
| SSH Private Key | `-----BEGIN (RSA\|DSA\|EC\|OPENSSH) PRIVATE KEY-----` header |
| Database URL | `postgres://`, `mysql://`, `mongodb://` with embedded credentials |
| GCP Service Account | JSON containing `"type": "service_account"` |

Scrubbing is applied to all three fields of every experience: **content**, **metadata**, and **tags**.

## Query validation

`QueryValidator` enforces read-only access and query cost limits.

### SQL safety

- Only `SELECT` statements are allowed.
- Semicolons are rejected (prevents statement chaining).
- Blocked keywords: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `CREATE`, `ALTER`, `TRUNCATE`, `EXEC`, `EXECUTE`, `GRANT`, `REVOKE`.
- SQL comments (`--` and `/* */`) are stripped before validation.

### Cypher safety

- Only `MATCH` queries are allowed (read-only).
- Blocked operations: `DELETE`, `DETACH`, `SET`, `REMOVE`, `MERGE`, `CREATE`, `CALL`, `LOAD`, `FOREACH`.

### Query cost estimation

Each query is scored against heuristic cost rules:

| Pattern | Cost |
|---------|------|
| Base cost | +1 |
| `SELECT * FROM` (full scan) | +10 |
| Each `JOIN` | +5 |
| Each subquery | +3 |
| `ORDER BY` | +2 |
| Missing `LIMIT` clause | +20 |

Queries exceeding an agent's `max_query_cost` are rejected with a descriptive error.

## Path traversal protection

Python bindings validate file paths to prevent directory traversal attacks. Paths containing `..` sequences or absolute paths outside the expected working directory are rejected before reaching the Rust backend.

## Scope enforcement

`ScopeEnforcer` restricts per-agent capabilities through the `AgentCapabilities` struct:

| Capability | Description |
|------------|-------------|
| `scope` | Access level — `SessionOnly`, `CrossSessionRead`, `CrossSessionWrite`, `GlobalRead`, `GlobalWrite` |
| `allowed_experience_types` | Allowlist of experience types the agent may store or retrieve |
| `max_query_cost` | Maximum query cost budget (1–100+) |
| `can_access_patterns` | Whether the agent may read pattern-type memories |
| `memory_quota_mb` | Per-agent memory quota in megabytes |

Scope enforcement is currently documented as a structured stub. The `AgentCapabilities` struct is defined and validated at the `SecureMemoryBackend` API boundary, but dynamic policy loading is not yet implemented.

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.4.x | ✅ Current |
| 0.3.x | ⚠️ Best-effort |
| < 0.3 | ❌ No |

## Reporting a vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not open a public issue.**
2. Email **[security@amplihack.dev](mailto:security@amplihack.dev)** with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
3. You will receive an acknowledgment within **48 hours**.
4. A fix will be developed privately and released as a patch version.

For non-security bugs, please [open an issue](https://github.com/rysweet/amplihack-memory-lib/issues).
