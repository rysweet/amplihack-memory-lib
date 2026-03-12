//! PyO3 bindings exposing amplihack-memory as a Python extension module.
//!
//! Compiled only with `--features python`. Produces a `amplihack_memory_rs`
//! native module importable from Python.
#![allow(clippy::useless_conversion)]

mod cognitive;
mod connectors;
mod converters;
mod experience;
mod helpers;
mod hierarchical;

pub use cognitive::PyCognitiveMemory;
pub use connectors::{PyExperienceStore, PyMemoryConnector};
pub use experience::PyExperience;
pub use hierarchical::PyHierarchicalMemory;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::contradiction::detect_contradiction;
use crate::entity_extraction::extract_entity_name;
use crate::security::CredentialScrubber;
use crate::similarity::compute_word_similarity;

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Extract an entity name from text given a concept hint.
#[pyfunction]
#[pyo3(name = "extract_entities")]
fn py_extract_entities(text: &str, concept: &str) -> String {
    extract_entity_name(text, concept)
}

/// Compute Jaccard word-level similarity between two strings (0.0–1.0).
#[pyfunction]
#[pyo3(name = "jaccard_similarity")]
fn py_jaccard_similarity(a: &str, b: &str) -> f64 {
    compute_word_similarity(a, b)
}

/// Detect whether two facts contradict each other.
/// Returns a dict with `contradiction` (bool) and `conflicting_values` (str),
/// or None if no contradiction is detected.
#[pyfunction]
#[pyo3(name = "detect_contradiction")]
fn py_detect_contradiction(
    py: Python<'_>,
    fact1: &str,
    fact2: &str,
    concept1: &str,
    concept2: &str,
) -> PyResult<Option<PyObject>> {
    match detect_contradiction(fact1, fact2, concept1, concept2) {
        Some(cr) => {
            let d = PyDict::new_bound(py);
            d.set_item("contradiction", cr.contradiction)?;
            d.set_item("conflicting_values", &cr.conflicting_values)?;
            Ok(Some(d.to_object(py)))
        }
        None => Ok(None),
    }
}

/// Scrub credentials from text. Returns (scrubbed_text, had_credentials).
#[pyfunction]
#[pyo3(name = "scrub_credentials")]
fn py_scrub_credentials(text: &str) -> (String, bool) {
    let scrubber = CredentialScrubber::new();
    scrubber.scrub_text(text)
}

// ---------------------------------------------------------------------------
// Module entry point
// ---------------------------------------------------------------------------

/// Native Rust implementation of amplihack-memory, exposed as a Python module.
#[pymodule]
#[pyo3(name = "amplihack_memory_rs")]
pub fn amplihack_memory_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<PyExperience>()?;
    m.add_class::<PyMemoryConnector>()?;
    m.add_class::<PyExperienceStore>()?;
    m.add_class::<PyCognitiveMemory>()?;
    m.add_class::<PyHierarchicalMemory>()?;

    // Functions
    m.add_function(wrap_pyfunction!(py_extract_entities, m)?)?;
    m.add_function(wrap_pyfunction!(py_jaccard_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(py_detect_contradiction, m)?)?;
    m.add_function(wrap_pyfunction!(py_scrub_credentials, m)?)?;

    Ok(())
}
