# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-03-13

### Added
- Quality audit cycle with 83 findings fixed (PRs #57–#60)
- 31 outside-in behavioral tests (`tests/outside_in.rs`)
- Property-based and concurrent tests (`tests/concurrent.rs`)
- Supply chain security with `cargo-deny` (`deny.toml`)
- Python integration tests for PyO3 bindings
- Extended `.gitignore` for Rust, editor, and Python entries

### Changed
- Split 6 oversized modules into submodules — all modules now < 400 LOC (#47, #56)
- Improved CI: kuzu tests, `cargo audit`, MSRV 1.80 verification, dependency caching (#49, #50)
- Doc comments added to all public items (#55)

### Fixed
- 21 HIGH severity issues covering security, correctness, and error handling (#57, #60)
- Credential scrubber now covers metadata and tags (#48)
- SQL/Cypher injection hardening (#48)
- Path traversal validation on Python bindings (#48)
- Kuzu `validate()` and federated doc comments (#58)

## [0.3.0] - 2026-03-12

### Added
- Complete Rust port of amplihack-memory-lib from Python
- Native LadybugDB backend replacing PyO3 Kuzu bridge
- Kuzu graph database support via PyO3 bridge
- Comprehensive security hardening (#48)
- Multi-threaded concurrent access tests (#43)
- Graph, hierarchical, concurrent, and pattern benchmarks (Criterion) (#54)
- PyO3 Python bindings with graph operations (#53)
- HiveStore: 5 public functions with 30 tests (#52)
- GitHub Actions CI pipeline for Rust and Python tests
- Reverse-edge index for O(1) incoming-edge lookup (#42)

### Changed
- Code quality improvements: DRY patterns, dead code removal, constants (#45)
- Removed 17 unnecessary `.clone()` calls across 5 files
- Module split, dead code removal, parity tests (Phase 2)
- Expanded amplihack-memory README (#51)

### Fixed
- Python–Rust behavioral parity alignment (#46)
- Replaced swallowed errors with `tracing` warnings (#44)
- String slicing panic prevention and extended experience ID hash (#41)
- 12 quality audit findings (2 HIGH, 4 MEDIUM, 6 LOW)
- All audit findings in Python bindings resolved

## [0.2.0] - 2026-02-27

### Added
- 6-type cognitive memory system backed by Kuzu (episodic, semantic, procedural, prospective, sensory, working) (#1)
- Dual-backend architecture: Kuzu (default) + SQLite
- CognitiveMemory exported from package (#7)
- Comprehensive documentation and GitHub Pages with MkDocs (#5)
- Release notes for v0.1.0

## [0.1.0] - 2026-02-15

### Added
- Initial release of amplihack-memory-lib
- 166 unit tests
- SQLite backend and InMemoryGraphStore
- Cognitive memory with 6 experience types
- Hierarchical memory with experience consolidation
- Pattern recognition and semantic search
- Security layer with credential scrubbing

[0.4.0]: https://github.com/rysweet/amplihack-memory-lib/compare/v0.2.0...HEAD
[0.3.0]: https://github.com/rysweet/amplihack-memory-lib/compare/v0.2.0...v0.2.0
[0.2.0]: https://github.com/rysweet/amplihack-memory-lib/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/rysweet/amplihack-memory-lib/releases/tag/v0.1.0
