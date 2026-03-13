# Contributing to amplihack-memory

Thank you for your interest in contributing! This guide covers everything you need to get started.

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Rust | 1.80+ (MSRV) | Core development |
| Python | 3.10+ | PyO3 bindings and integration tests |
| pre-commit | latest | Git hooks for formatting, linting, testing |

## Setup

```bash
# Clone the repository
git clone https://github.com/rysweet/amplihack-memory-lib.git
cd amplihack-memory-lib

# Build the crate
cargo build --manifest-path rust/amplihack-memory/Cargo.toml

# Run all tests
cargo test --manifest-path rust/amplihack-memory/Cargo.toml

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Feature flags

The crate has optional features that can be tested individually:

```bash
# Default (no optional features)
cargo test --manifest-path rust/amplihack-memory/Cargo.toml

# With LadybugDB backend
cargo test --manifest-path rust/amplihack-memory/Cargo.toml --features ladybug

# With PyO3 Python bindings
cargo test --manifest-path rust/amplihack-memory/Cargo.toml --features python
```

## Pre-commit hooks

The project uses [pre-commit](https://pre-commit.com/) to enforce quality on every commit. After `pre-commit install`, each commit automatically runs:

- **`cargo fmt`** — formatting check
- **`cargo clippy -- -D warnings`** — lint with zero warnings
- **`cargo test --quiet`** — full test suite
- **trailing-whitespace**, **end-of-file-fixer**, **check-yaml**, **check-toml**, **check-merge-conflict**

## Pull request process

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```
2. **Write tests first** — new functionality must include tests.
3. **Run the full check suite locally**:
   ```bash
   cd rust/amplihack-memory
   cargo fmt --check
   cargo clippy -- -D warnings
   cargo test
   ```
4. **Push and open a draft PR** for early feedback.
5. **CI must pass** — the pipeline runs tests, clippy, formatting, `cargo audit`, and MSRV verification on Rust 1.80.
6. **Request review** when ready. At least one approval is required before merging.

## Coding standards

### Module size
Every module must stay **under 400 lines of code**. If a module grows beyond this limit, split it into submodules with a `mod.rs` barrel file.

### Documentation
All `pub` items must have doc comments (`///`). Run `cargo doc --no-deps` to verify documentation builds without warnings.

### Error handling
- Use `thiserror` for error types. All error variants live in `src/errors.rs`.
- Annotate `Result`-returning public functions with `#[must_use]` where the caller should not silently ignore the return value.
- Never swallow errors — use `tracing::warn!` at minimum when an error cannot be propagated.

### Security
- All experience data passes through `CredentialScrubber` before storage.
- Custom queries are validated by `QueryValidator` (SQL and Cypher).
- See [SECURITY.md](SECURITY.md) for the full security model.

## Testing

The project maintains four layers of tests:

| Layer | Location | Purpose |
|-------|----------|---------|
| **Unit tests** | `src/**/tests.rs` | Per-module logic verification (~250 tests) |
| **Integration tests** | `tests/parity_test.rs` | Python–Rust behavioral parity |
| **Outside-in tests** | `tests/outside_in.rs` | 31 end-to-end behavioral tests |
| **Concurrent tests** | `tests/concurrent.rs` | Multi-threaded correctness (8+ threads) |

### Running benchmarks

```bash
cd rust/amplihack-memory
cargo bench
```

Benchmarks use the [Criterion](https://github.com/bheisler/criterion.rs) framework and produce HTML reports in `target/criterion/`.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
