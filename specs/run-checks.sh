#!/usr/bin/env bash
#
# run-checks.sh — single source of truth for the TLA+ model checks.
#
# Runs the four TLC checks that guard the multi-writer coordination design for
# the lbug-backed store and asserts the EXPECTED polarity of each:
#
#   POSITIVE (must report "No error has been found"):
#     * FencedApplier_fenced.cfg on FencedApplier.tla   (epoch fencing => NoSplitBrain holds)
#     * DurableLog.cfg           on DurableLog.tla       (durable log     => NoLostAckedWrite holds)
#
#   NEGATIVE (must report a violation — the buggy/rejected designs are still
#             demonstrable; a negative check PASSES only when its violation is present):
#     * FencedApplier_unfenced.cfg on FencedApplier.tla  (kill(pid,0) reap => NoSplitBrain VIOLATED)
#     * FederatedLoss.cfg          on FederatedLoss.tla  (federated design => NoLostAckedWrite VIOLATED)
#
# The script exits 0 iff ALL four checks match their expected polarity, and
# non-zero otherwise. Both CI and local developers invoke this one script so the
# grep/polarity logic lives in exactly one place (see specs/README.md).
#
# The tla2tools.jar is pinned by URL + sha256 for reproducibility (no floating
# "latest"). Set TLA_TOOLS_JAR to reuse a locally cached jar and skip the
# download; if set, its sha256 is still verified against the pin.
#
set -euo pipefail

# --- Pinned tla2tools.jar (TLA+ tools v1.7.4 stable release) -----------------
TLA_TOOLS_URL="https://github.com/tlaplus/tlaplus/releases/download/v1.7.4/tla2tools.jar"
TLA_TOOLS_SHA256="936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88"

# Positive matcher: the literal TLC success string.
POSITIVE_MATCH="No error has been found"

# Directory of this script == where the .tla / .cfg artifacts live.
SPECS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Hard-coded (module, cfg) pairs — never derived from args/env (no path-injection
# surface). Each pair is "<module.tla> <config.cfg>".
POSITIVE_CHECKS=(
  "FencedApplier.tla FencedApplier_fenced.cfg"
  "DurableLog.tla DurableLog.cfg"
)
NEGATIVE_CHECKS=(
  "FencedApplier.tla FencedApplier_unfenced.cfg"
  "FederatedLoss.tla FederatedLoss.cfg"
)

log() { printf '%s\n' "$*" >&2; }

verify_sha256() {
  # verify_sha256 <file> <expected-hex>
  local file="$1" expected="$2" actual
  actual="$(sha256sum "$file" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    log "ERROR: sha256 mismatch for $file"
    log "  expected: $expected"
    log "  actual:   $actual"
    return 1
  fi
}

# Resolve the jar: reuse TLA_TOOLS_JAR if provided (verifying its hash), else
# download the pinned release into a temp dir and verify before use.
TMP_DIR=""
cleanup() {
  if [[ -n "$TMP_DIR" && -d "$TMP_DIR" ]]; then
    rm -rf "$TMP_DIR"
  fi
}
trap cleanup EXIT

resolve_jar() {
  if [[ -n "${TLA_TOOLS_JAR:-}" ]]; then
    if [[ ! -f "$TLA_TOOLS_JAR" ]]; then
      log "ERROR: TLA_TOOLS_JAR=$TLA_TOOLS_JAR does not exist"
      return 1
    fi
    log "Using cached jar from TLA_TOOLS_JAR: $TLA_TOOLS_JAR"
    verify_sha256 "$TLA_TOOLS_JAR" "$TLA_TOOLS_SHA256"
    JAR="$TLA_TOOLS_JAR"
    return 0
  fi
  TMP_DIR="$(mktemp -d)"
  JAR="$TMP_DIR/tla2tools.jar"
  log "Downloading pinned tla2tools.jar ..."
  # Retry to tolerate transient network/GitHub failures in CI. Safe because the
  # download is fail-closed: the sha256 pin is verified before the jar is used.
  curl --fail --location --show-error --silent \
    --retry 3 --retry-delay 2 --retry-connrefused \
    -o "$JAR" "$TLA_TOOLS_URL"
  verify_sha256 "$JAR" "$TLA_TOOLS_SHA256"
  log "Verified pinned tla2tools.jar (sha256 OK)"
}

# run_tlc <module.tla> <config.cfg> — returns TLC's combined output on stdout.
run_tlc() {
  local module="$1" cfg="$2"
  ( cd "$SPECS_DIR" && java -cp "$JAR" tlc2.TLC -deadlock -config "$cfg" "$module" 2>&1 )
}

FAILURES=0

check_positive() {
  local module="$1" cfg="$2" output
  log ""
  log "=== POSITIVE: tlc -deadlock -config $cfg $module (expect clean) ==="
  output="$(run_tlc "$module" "$cfg")" || true
  if grep -qF "$POSITIVE_MATCH" <<<"$output"; then
    log "PASS: '$POSITIVE_MATCH'"
  else
    log "FAIL: expected '$POSITIVE_MATCH' but it was not found. TLC output:"
    log "$output"
    FAILURES=$((FAILURES + 1))
  fi
}

check_negative() {
  local module="$1" cfg="$2" output
  log ""
  log "=== NEGATIVE: tlc -deadlock -config $cfg $module (expect a violation) ==="
  output="$(run_tlc "$module" "$cfg")" || true
  # Case-insensitive so both an INVARIANT ('is violated') and a PROPERTY
  # ('were violated') are matched. A negative check PASSES iff its expected
  # violation is present.
  if grep -qi "violated" <<<"$output"; then
    log "PASS: expected violation is present (intentional negative test)"
  else
    log "FAIL: expected a violation but none was found. TLC output:"
    log "$output"
    FAILURES=$((FAILURES + 1))
  fi
}

main() {
  resolve_jar

  local pair module cfg
  for pair in "${POSITIVE_CHECKS[@]}"; do
    read -r module cfg <<<"$pair"
    check_positive "$module" "$cfg"
  done
  for pair in "${NEGATIVE_CHECKS[@]}"; do
    read -r module cfg <<<"$pair"
    check_negative "$module" "$cfg"
  done

  log ""
  if [[ "$FAILURES" -eq 0 ]]; then
    log "All 4 TLA+ checks matched their expected polarity (2 positive clean, 2 negative violated)."
    exit 0
  fi
  log "$FAILURES TLA+ check(s) did not match the expected polarity."
  exit 1
}

main "$@"
