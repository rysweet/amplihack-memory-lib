"""Contract tests for the TLA+ specification set (issue #135).

This is a SPEC + DOC + CI deliverable: a durable, CI-enforced set of TLA+
specifications that formalize the multi-writer coordination design (design C:
durable shared log + single epoch-fenced applier) for the lbug-backed store.
There is no Rust/Python behavior change, so the "contract" these tests enforce
is the shape and integrity of the committed artifacts, the check harness, the
documentation, and the CI gate.

Written test-first (TDD): these tests define the acceptance criteria and are
expected to FAIL until the implementation lands the artifacts and edits.

Test groups
-----------
* ``TestSpecArtifacts``       -- the 7 verified files exist and are byte-identical
                                 (sha256) to the model-checked oracle; no scratch.
* ``TestRunChecksScript``     -- ``specs/run-checks.sh`` exists, is executable, and
                                 is the single source of truth for the 4 checks.
* ``TestSpecsReadme``         -- ``specs/README.md`` is a durable design doc.
* ``TestCiGate``              -- ``.github/workflows/ci.yml`` has the ``tla-model-check`` job.
* ``TestRootReadmeLink``      -- root ``README.md`` links to ``specs/README.md``.
* ``TestPreCommitByteIdentity`` -- mutating hooks skip the verbatim ``.tla``/``.cfg``.
* ``TestGitignoreScratch``    -- TLC scratch is git-ignored durably.
* ``TestModelCheckIntegration`` -- (opt-in via ``RUN_TLC=1``) actually runs the harness.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SPECS_DIR = REPO_ROOT / "specs"

# sha256 oracle of the verified, TLC-passing artifacts. Copying must be
# byte-identical; any drift here means the committed spec no longer matches the
# model-checked source of truth. These digests ARE the durable oracle (the
# original verified files live only in an ephemeral session directory).
SPEC_SHA256 = {
    "FencedApplier.tla": "88264a7425049a5f2bd3b44c5c48bc42f485b945171224eb8b1f727a1df0fb9d",
    "FencedApplier_unfenced.cfg": "f1e155e2a61b6e4a14bf50af6f43dbea2759944539119b1e73c9cb045cc542eb",
    "FencedApplier_fenced.cfg": "91dd5deb33bebf8bacfaa51a7a9f57c16c09ca824a6c24d1a945b71230817e8f",
    "DurableLog.tla": "d433b5ca8f2a22ca1b036d8db7167b10c3e5605cb2efaa63e5e734db35cecca8",
    "DurableLog.cfg": "fda39894875039c8a7fc3210cf0a0c466bf71af3646b7d237845f7d8265a088a",
    "FederatedLoss.tla": "affc4191fb23e5dc6767c63240dcce946610e086037b8eadfcdb7f668ce4ef2c",
    "FederatedLoss.cfg": "4d6ca5c4333ad600eb3b8832b282abbd5205c68afda369b4609ec6d486426022",
}

# The four model-check invocations, as (config, module) pairs.
POSITIVE_CHECKS = [
    ("FencedApplier_fenced.cfg", "FencedApplier.tla"),
    ("DurableLog.cfg", "DurableLog.tla"),
]
NEGATIVE_CHECKS = [
    ("FencedApplier_unfenced.cfg", "FencedApplier.tla"),
    ("FederatedLoss.cfg", "FederatedLoss.tla"),
]

RUN_CHECKS = SPECS_DIR / "run-checks.sh"
SPECS_README = SPECS_DIR / "README.md"
ROOT_README = REPO_ROOT / "README.md"
CI_YAML = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PRE_COMMIT = REPO_ROOT / ".pre-commit-config.yaml"
GITIGNORE = REPO_ROOT / ".gitignore"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class TestSpecArtifacts:
    """The 7 verified TLA+ files land verbatim under specs/."""

    def test_specs_dir_exists(self):
        assert SPECS_DIR.is_dir(), "specs/ directory must exist at repo root"

    @pytest.mark.parametrize("filename", sorted(SPEC_SHA256))
    def test_artifact_present(self, filename):
        assert (SPECS_DIR / filename).is_file(), f"missing spec artifact specs/{filename}"

    @pytest.mark.parametrize("filename,expected", sorted(SPEC_SHA256.items()))
    def test_artifact_is_byte_identical(self, filename, expected):
        """Each artifact must match the model-checked oracle digest exactly."""
        path = SPECS_DIR / filename
        if not path.is_file():
            pytest.fail(f"missing spec artifact specs/{filename}")
        actual = _sha256(path)
        assert actual == expected, (
            f"specs/{filename} is not byte-identical to the verified oracle "
            f"(expected {expected}, got {actual}). The spec must be copied verbatim."
        )

    def test_no_scratch_files_committed(self):
        """TLC scratch must never be committed into specs/."""
        if not SPECS_DIR.is_dir():
            pytest.fail("specs/ directory must exist at repo root")
        offenders = []
        for p in SPECS_DIR.rglob("*"):
            name = p.name
            rel = p.relative_to(SPECS_DIR).as_posix()
            if "_TTrace_" in name:
                offenders.append(rel)
            elif p.is_file() and name.endswith((".bin", ".jar")):
                offenders.append(rel)
            elif p.is_dir() and name == "states":
                offenders.append(rel + "/")
        assert not offenders, f"TLC scratch must not be committed: {offenders}"

    def test_only_expected_files_in_specs(self):
        """specs/ contains exactly the 7 artifacts plus the harness and docs."""
        if not SPECS_DIR.is_dir():
            pytest.fail("specs/ directory must exist at repo root")
        allowed = set(SPEC_SHA256) | {"run-checks.sh", "README.md"}
        present = {p.name for p in SPECS_DIR.iterdir() if p.is_file()}
        unexpected = present - allowed
        assert not unexpected, f"unexpected files in specs/: {sorted(unexpected)}"


class TestRunChecksScript:
    """specs/run-checks.sh is the single source of truth for the 4 checks."""

    def test_exists(self):
        assert RUN_CHECKS.is_file(), "specs/run-checks.sh must exist"

    def test_is_executable(self):
        if not RUN_CHECKS.is_file():
            pytest.fail("specs/run-checks.sh must exist")
        mode = RUN_CHECKS.stat().st_mode
        assert mode & stat.S_IXUSR, "specs/run-checks.sh must be executable (chmod +x)"

    def test_has_bash_shebang_and_strict_mode(self):
        if not RUN_CHECKS.is_file():
            pytest.fail("specs/run-checks.sh must exist")
        content = _read(RUN_CHECKS)
        first = content.splitlines()[0] if content else ""
        assert first.startswith("#!") and "bash" in first, "must have a bash shebang"
        assert "set -euo pipefail" in content, "must use 'set -euo pipefail' (fail closed)"

    def test_references_all_four_checks(self):
        if not RUN_CHECKS.is_file():
            pytest.fail("specs/run-checks.sh must exist")
        content = _read(RUN_CHECKS)
        for cfg, module in POSITIVE_CHECKS + NEGATIVE_CHECKS:
            assert cfg in content, f"run-checks.sh must reference config {cfg}"
            assert module in content, f"run-checks.sh must reference module {module}"

    def test_uses_deadlock_flag(self):
        if not RUN_CHECKS.is_file():
            pytest.fail("specs/run-checks.sh must exist")
        assert "-deadlock" in _read(RUN_CHECKS), "checks must run with 'tlc -deadlock'"

    def test_positive_matcher_present(self):
        if not RUN_CHECKS.is_file():
            pytest.fail("specs/run-checks.sh must exist")
        assert "No error has been found" in _read(RUN_CHECKS), (
            "positive checks must assert the literal 'No error has been found'"
        )

    def test_negative_matcher_asserts_specific_violation_strings(self):
        """Negative checks must assert the SPECIFIC violation each demonstrates.

        A bare 'violated' grep lets an unrelated failure (e.g. a broken TypeOK)
        masquerade as the intended demonstration and keep the gate green while
        the design property is no longer actually exercised. Each negative check
        must therefore assert the exact invariant/property TLC names.
        """
        if not RUN_CHECKS.is_file():
            pytest.fail("specs/run-checks.sh must exist")
        content = _read(RUN_CHECKS)
        # The two rejected designs must each be pinned to their specific violation.
        assert "Invariant NoSplitBrain is violated" in content, (
            "unfenced check must assert the specific 'NoSplitBrain is violated'"
        )
        assert "Temporal properties were violated" in content, (
            "federated check must assert the specific 'Temporal properties were violated'"
        )
        # A fixed-string match (grep -F) is required so the specific expected
        # string is asserted literally, not as a loose/partial pattern.
        assert re.search(r"grep[^\n|]*-[a-zA-Z]*F", content), (
            "negative matcher must assert its expected string with a fixed-string grep (-F)"
        )

    def test_pins_jar_by_sha256(self):
        if not RUN_CHECKS.is_file():
            pytest.fail("specs/run-checks.sh must exist")
        content = _read(RUN_CHECKS)
        assert re.search(r"sha256", content, re.IGNORECASE), (
            "the tla2tools.jar must be pinned/verified by sha256 (no floating latest)"
        )

    def test_honors_tla_tools_jar_env(self):
        if not RUN_CHECKS.is_file():
            pytest.fail("specs/run-checks.sh must exist")
        assert "TLA_TOOLS_JAR" in _read(RUN_CHECKS), (
            "run-checks.sh must reuse a cached jar via the TLA_TOOLS_JAR env var"
        )

    def test_no_forbidden_references(self):
        if not RUN_CHECKS.is_file():
            pytest.fail("specs/run-checks.sh must exist")
        content = _read(RUN_CHECKS).lower()
        assert "kuzu" not in content, "no kuzu references allowed"
        assert "curl | bash" not in content and "curl|bash" not in content, (
            "must not pipe curl straight into a shell"
        )


class TestSpecsReadme:
    """specs/README.md is a durable, evergreen design document."""

    def test_exists(self):
        assert SPECS_README.is_file(), "specs/README.md must exist"

    def test_states_problem_and_designs(self):
        if not SPECS_README.is_file():
            pytest.fail("specs/README.md must exist")
        text = _read(SPECS_README)
        low = text.lower()
        assert "kill(pid" in low or "reap_stale_open_lock" in low, (
            "must state the unsafe kill(pid,0) reap problem"
        )
        # Three candidate designs A/B/C must all be discussed.
        for token in ("A", "B", "C"):
            assert re.search(rf"\b{token}\b", text), f"must discuss design {token}"
        assert "chosen" in low or "why c" in low, "must explain why design C was chosen"

    def test_describes_each_spec(self):
        if not SPECS_README.is_file():
            pytest.fail("specs/README.md must exist")
        text = _read(SPECS_README)
        for token in (
            "FencedApplier",
            "DurableLog",
            "FederatedLoss",
            "NoSplitBrain",
            "NoLostAckedWrite",
        ):
            assert token in text, f"specs/README.md must mention {token}"

    def test_gives_reproduce_commands(self):
        if not SPECS_README.is_file():
            pytest.fail("specs/README.md must exist")
        text = _read(SPECS_README)
        assert "tlc -deadlock" in text, "must give the tlc -deadlock reproduce command"
        assert "run-checks.sh" in text, "must reference the run-checks.sh helper"

    def test_marks_negative_tests_as_intentional(self):
        if not SPECS_README.is_file():
            pytest.fail("specs/README.md must exist")
        low = _read(SPECS_README).lower()
        assert "negative" in low, "must call out the negative tests"
        assert "expected to violate" in low or "intentional" in low, (
            "must state the violation configs are intentional negative tests"
        )

    def test_no_forbidden_references(self):
        if not SPECS_README.is_file():
            pytest.fail("specs/README.md must exist")
        low = _read(SPECS_README).lower()
        assert "kuzu" not in low, "no kuzu references allowed"

    def test_is_evergreen_not_a_snapshot(self):
        if not SPECS_README.is_file():
            pytest.fail("specs/README.md must exist")
        low = _read(SPECS_README).lower()
        for phrase in ("as of today", "as of now", "at the time of writing", "point-in-time report"):
            assert phrase not in low, f"docs must be evergreen; found snapshot phrase '{phrase}'"


class TestCiGate:
    """A tla-model-check CI job enforces the invariants on every change."""

    def test_ci_yaml_exists(self):
        assert CI_YAML.is_file(), ".github/workflows/ci.yml must exist"

    def test_has_tla_model_check_job(self):
        if not CI_YAML.is_file():
            pytest.fail(".github/workflows/ci.yml must exist")
        assert re.search(r"^\s{2}tla-model-check:", _read(CI_YAML), re.MULTILINE), (
            "ci.yml must define a 'tla-model-check' job"
        )

    def test_job_calls_run_checks_script(self):
        if not CI_YAML.is_file():
            pytest.fail(".github/workflows/ci.yml must exist")
        assert "specs/run-checks.sh" in _read(CI_YAML), (
            "the CI job must invoke specs/run-checks.sh (single source of truth, no duplicated grep)"
        )

    def test_job_provisions_temurin_21(self):
        if not CI_YAML.is_file():
            pytest.fail(".github/workflows/ci.yml must exist")
        content = _read(CI_YAML)
        assert "setup-java" in content, "CI job must set up a JDK via actions/setup-java"
        assert "temurin" in content.lower(), "CI job must use the Temurin distribution"
        assert re.search(r"java-version:\s*['\"]?21", content), "CI job must use JDK 21"

    def test_job_pins_jar_by_sha256(self):
        if not CI_YAML.is_file():
            pytest.fail(".github/workflows/ci.yml must exist")
        # The pin can live in the script; the workflow must not rely on a
        # floating latest. Accept sha256 pinning either in the workflow text
        # or delegated to run-checks.sh (which is separately tested).
        content = _read(CI_YAML)
        assert "run-checks.sh" in content or re.search(r"sha256", content, re.IGNORECASE), (
            "jar must be pinned by sha256 (in workflow or via run-checks.sh)"
        )

    def test_job_has_least_privilege_permissions(self):
        if not CI_YAML.is_file():
            pytest.fail(".github/workflows/ci.yml must exist")
        content = _read(CI_YAML)
        block = re.search(
            r"^\s{2}tla-model-check:.*?(?=^\s{2}\S|\Z)", content, re.MULTILINE | re.DOTALL
        )
        assert block, "tla-model-check job block not found"
        job_text = block.group(0)
        assert re.search(r"permissions:\s*\n\s+contents:\s*read", job_text), (
            "tla-model-check must declare 'permissions: contents: read'"
        )


class TestRootReadmeLink:
    """The design doc is discoverable from the repo root README."""

    def test_links_to_specs_readme(self):
        assert ROOT_README.is_file(), "root README.md must exist"
        text = _read(ROOT_README)
        assert "specs/README.md" in text, "root README must link to specs/README.md"


class TestPreCommitByteIdentity:
    """Mutating hooks must not rewrite the verbatim .tla/.cfg artifacts."""

    def test_excludes_specs_tla_and_cfg(self):
        assert PRE_COMMIT.is_file(), ".pre-commit-config.yaml must exist"
        content = _read(PRE_COMMIT)
        assert "specs/" in content and re.search(r"tla|cfg", content), (
            "pre-commit must exclude specs/*.tla and specs/*.cfg so end-of-file-fixer / "
            "trailing-whitespace cannot break byte-identity"
        )


class TestGitignoreScratch:
    """TLC scratch is durably ignored so it can never be committed."""

    def test_ignores_tlc_scratch(self):
        assert GITIGNORE.is_file(), ".gitignore must exist"
        content = _read(GITIGNORE)
        assert "specs/states/" in content, ".gitignore must ignore specs/states/"
        assert re.search(r"specs/\*\.jar", content), ".gitignore must ignore specs/*.jar"
        assert "_TTrace_" in content, ".gitignore must ignore specs/*_TTrace_* scratch"


@pytest.mark.skipif(
    os.environ.get("RUN_TLC") != "1",
    reason="opt-in end-to-end model check; set RUN_TLC=1 (needs java + network or TLA_TOOLS_JAR)",
)
class TestModelCheckIntegration:
    """End-to-end: the harness runs all 4 checks and asserts the polarity.

    This is the ultimate contract: positive configs are clean, negative configs
    show their expected violation, and the script exits 0 overall. Opt-in
    because it needs a JDK and either network access or a cached TLA_TOOLS_JAR.
    """

    def test_run_checks_passes(self):
        if not RUN_CHECKS.is_file():
            pytest.fail("specs/run-checks.sh must exist")
        if shutil.which("java") is None:
            pytest.skip("java not available")
        result = subprocess.run(
            ["bash", str(RUN_CHECKS)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=1800,
        )
        assert result.returncode == 0, (
            "specs/run-checks.sh must exit 0 (2 positive clean, 2 negative violated).\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
