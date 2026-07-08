# lbug libstdc++ `std::format` ABI portability — the `initBufferManager` SIGSEGV and its fix

The persistent `GraphStore` backend (`LbugGraphStore`, which powers
`CognitiveMemory::open_persistent`) links the embedded LadybugDB engine through
the [`lbug`](https://crates.io/crates/lbug) crate (`lbug = "=0.17.1"`). This page
is the authoritative root-cause and fix for a **portability crash** observed when
a Simard-family binary built on one Linux release is run on a **newer** one
(issue **#3119**).

## Symptom

On a host with a newer libstdc++ (Ubuntu **26.04 LTS**, `g++ 15.2.0`, libstdc++
exposing `GLIBCXX_3.4.35`) the binary **segfaults during cognitive-memory
initialization**. The same source runs cleanly on Ubuntu **25.10** (libstdc++
`GLIBCXX_3.4.34`). The backtrace is deterministic:

```
…::run_ooda_daemon
  └─ LibraryCognitiveMemory::open
       └─ amplihack_memory::graph::lbug_store::open_database
            └─ lbug::main::Database::initBufferManager
                 └─ std::vformat / std::__format::__formatter_str   ← SIGSEGV
```

## Root cause: two libstdc++ `std::format` ABIs in one process

In static-link mode, `lbug`'s `build.rs` **prefers a prebuilt engine archive**
rather than compiling the C++ engine locally. That prebuilt is compiled on a
builder image against **that image's** libstdc++. C++20 `std::format` is largely
header-only, inlined, and templated: the `std::__format` formatter types and the
argument-store layout are **instantiated into the prebuilt at its own compile
time**. Those instantiations are *not* exported, symbol-versioned entry points,
so the usual `GLIBCXX_3.4.NN` backward-compatibility guarantee (which only covers
exported versioned symbols) does **not** protect them.

When that prebuilt is linked into a binary that runs on a host whose libstdc++
changed the `std::format` internals (25.10 `3.4.34` → 26.04 `3.4.35`),
`Database::initBufferManager` builds a formatted string, reaches `std::vformat`,
and dereferences a mismatched formatter layout — a hard SIGSEGV rather than a
clean, catchable error. On 25.10 the prebuilt's assumed ABI happens to match the
host, so the same path is fine. This is a classic *do not mix C++
standard-library ABIs in one process* failure; `std::format` is simply the part
of libstdc++ whose ABI moved between the two releases.

Because the mismatch is frozen into the prebuilt at *its* build time, no
runtime detect-and-retry at the application layer can recover it — the fix must
prevent the mismatch from existing.

## The fix: build lbug from source against the host toolchain

`lbug`'s `build.rs` honours `LBUG_BUILD_FROM_SOURCE=1` (alias
`LBUG_RUST_BUILD_FROM_SOURCE=1`): when set, it prints *"Skipping prebuilt liblbug
because source build was requested"* and drives its bundled cmake build instead.
Compiling lbug from source against the **host's own** `g++`/libstdc++ means there
is exactly one `std::format` ABI in the process, and `initBufferManager` runs
cleanly on any host libstdc++ — including future ones.

### What this crate owns, and its limits

Cargo runs a dependency's build script (`lbug`) **before** its dependent's
(`amplihack-memory`), so `amplihack-memory` **cannot** set
`LBUG_BUILD_FROM_SOURCE` for `lbug` from its own `build.rs` — the ordering makes
upstream influence impossible, and `lbug` exposes no Cargo *feature* to force
source builds. What this crate therefore provides is:

1. **A build-time guard** (`build.rs`). When the `persistent` feature is active
   and neither source-build variable is set, it emits a prominent
   `cargo:warning` at every consumer's build, naming the ABI-mismatch SIGSEGV and
   how to avoid it. This surfaces the requirement in Simard/Crocutus builds
   rather than letting a prebuilt slip in silently.
2. **A from-source ABI proof in CI** (`lbug-from-source` job). It installs the
   native toolchain, sets `LBUG_BUILD_FROM_SOURCE=1`, compiles lbug from source,
   and **opens a real persistent store** so `initBufferManager` actually runs —
   proving the from-source path initializes cleanly. The pre-existing fast
   data-loss gates deliberately keep using the prebuilt (the CI runner's ABI
   matches it), so they stay fast; this job is the additive from-source proof.
3. **This document** — the durable root-cause record.

### Who sets the environment variable

The variable is set **where the deployment binary is built**: the Simard-family
[platform installer](https://github.com/rysweet/Crocutus) exports
`LBUG_BUILD_FROM_SOURCE=1` in its build phase (`scripts/install.sh`,
`installer_build_env`) and its preflight doctor provisions the native toolchain
(`build-essential`, `cmake`, `clang`, `pkg-config`, `libssl-dev`) the source
build needs. That guarantees the *produced daemon binary* never contains a
mismatched prebuilt lbug, on 25.10 **and** 26.04, from a single install command.

### Why not "just pick a matching prebuilt variant"?

`lbug` exposes `LBUG_LINUX_VARIANT` to select a prebuilt for a given libstdc++,
but that is brittle: it requires a published variant for every host libstdc++ the
fleet will ever run on, and it fails the day a host ships a libstdc++ newer than
any published variant (exactly the 26.04 case). Source-build needs only the
toolchain the installer already provisions and is correct for **any** host
libstdc++. Source-build is the fix; a variant match is at best a
non-relied-upon fallback.

## Known from-source build issues in lbug 0.17.1 (tracked: #130)

Building lbug 0.17.1 from source resolves the open-time `initBufferManager`/
`std::vformat` SIGSEGV, but the end-to-end on host `dev` (Ubuntu 26.04) surfaced
two *further* defects in lbug's from-source path. Both currently need a
downstream workaround (applied by the platform installer); the durable engine
fixes are tracked in issue #130.

1. **Duplicate-symbol link failure.** `LBUG_BUILD_FROM_SOURCE=1 cargo build`
   fails at link with rust-lld `duplicate symbol` errors for `utf8proc`/`antlr4`
   symbols: lbug bundles those third-party objects *inside* `liblbug.a` **and**
   also links them as separate `--whole-archive` libraries. The installer works
   around it with `RUSTFLAGS=-Clink-arg=-Wl,--allow-multiple-definition`; without
   it the from-source build does not link at all.

2. **Post-verdict static-teardown SIGSEGV.** A *clean from-source* binary still
   SIGSEGVs in C++ **global/static-destructor teardown** on normal process exit,
   *after* program logic has returned success (reproduces on both prebuilt and
   from-source, so it is not the open-time ABI mismatch). A fail-closed guardrail
   gate that runs `<binary> check` would see exit 139 and false-fail a
   proven-safe verdict; the installer's gate therefore keys on the explicit
   affirmative verdict marker (fail-closed preserved) rather than the exit code
   alone.

Until #130 lands, the from-source *cognitive-memory store* opens and runs
cleanly (the centerpiece fix); these two items are build-link and
process-teardown hygiene, worked around downstream.

## Downstream

Simard consumes this fix by **bumping its `amplihack-memory` pin** to the commit
that carries this guard + proof + doc — no lbug/engine logic is forked into
Simard. See Simard `Cargo.toml` and `docs/concepts/lbug-portability-libstdcxx-abi.md`.
