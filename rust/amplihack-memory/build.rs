//! Build script for `amplihack-memory`.
//!
//! Its sole job is the **lbug libstdc++ ABI portability guard** (issue #3119).
//!
//! Cognitive memory's `persistent` feature links the embedded LadybugDB engine
//! (`lbug`). By default `lbug`'s own build script prefers a *prebuilt* engine
//! archive compiled against *some* builder image's libstdc++. When that
//! prebuilt is linked into a binary that then runs on a host with a **newer**
//! libstdc++, the C++20 `std::format` machinery baked into the prebuilt no
//! longer matches the host runtime and cognitive-memory initialization
//! **segfaults** in `std::vformat` during `Database::initBufferManager` (seen on
//! Ubuntu 26.04 / `GLIBCXX_3.4.35`; fine on 25.10 / `GLIBCXX_3.4.34`).
//!
//! The durable fix is to build `lbug` **from source** against the host
//! toolchain by setting `LBUG_BUILD_FROM_SOURCE=1` (alias
//! `LBUG_RUST_BUILD_FROM_SOURCE=1`) in the build environment, so exactly one
//! libstdc++ `std::format` ABI is present in the process.
//!
//! This build script cannot set that environment variable for `lbug`'s build
//! script — Cargo runs a dependency's build script (`lbug`) *before* its
//! dependent's (`amplihack-memory`), so the ordering makes it impossible to
//! influence upstream from here. What it CAN do, and does, is **surface the
//! requirement to every consumer at build time**: when the `persistent` feature
//! is active and neither source-build variable is set, it emits a prominent
//! `cargo:warning` telling the operator that a prebuilt `lbug` is being linked
//! and how to avoid the ABI-mismatch SIGSEGV. The platform installer
//! (`Crocutus scripts/install.sh`) sets `LBUG_BUILD_FROM_SOURCE=1` for its
//! deployment build so the produced daemon binary never contains a mismatched
//! prebuilt. See `docs/lbug_libstdcxx_abi.md`.

fn main() {
    // Re-run this guard whenever the source-build selection changes.
    println!("cargo:rerun-if-env-changed=LBUG_BUILD_FROM_SOURCE");
    println!("cargo:rerun-if-env-changed=LBUG_RUST_BUILD_FROM_SOURCE");

    // Only relevant when the persistent (lbug-backed) store is compiled in.
    let persistent = std::env::var_os("CARGO_FEATURE_PERSISTENT").is_some();
    if !persistent {
        return;
    }

    let from_source = std::env::var_os("LBUG_BUILD_FROM_SOURCE").is_some()
        || std::env::var_os("LBUG_RUST_BUILD_FROM_SOURCE").is_some();

    if !from_source {
        println!(
            "cargo:warning=amplihack-memory[persistent]: building against a PREBUILT lbug. \
             On a host whose libstdc++ std::format ABI differs from the prebuilt's, cognitive \
             memory can SIGSEGV in std::vformat during Database::initBufferManager (issue #3119). \
             Set LBUG_BUILD_FROM_SOURCE=1 to compile lbug from source against this host's \
             toolchain. The platform installer does this automatically; see \
             amplihack-memory docs/lbug_libstdcxx_abi.md."
        );
    } else {
        println!(
            "cargo:warning=amplihack-memory[persistent]: LBUG_BUILD_FROM_SOURCE is set — lbug \
             will be compiled from source against this host's libstdc++ (issue #3119 ABI fix)."
        );
    }
}
