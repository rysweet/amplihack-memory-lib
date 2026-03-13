//! Helper functions for PyO3 ↔ Rust type conversions.

use std::collections::HashMap;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::backends::base::StorageStatistics;
use crate::experience::ExperienceType;
use crate::memory_types::MemoryCategory;

/// Convert a [`crate::MemoryError`] into a Python `RuntimeError`.
pub(crate) fn mem_err(e: crate::MemoryError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Parse a string into an [`ExperienceType`], raising `ValueError` on failure.
pub(crate) fn parse_experience_type(s: &str) -> PyResult<ExperienceType> {
    s.parse::<ExperienceType>().map_err(PyValueError::new_err)
}

/// Convert a `serde_json::Value` to a Python object recursively.
pub(crate) fn json_to_py(py: Python<'_>, val: &serde_json::Value) -> PyResult<PyObject> {
    match val {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok(b.to_object(py)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.to_object(py))
            } else {
                Ok(n.as_f64().unwrap_or(0.0).to_object(py))
            }
        }
        serde_json::Value::String(s) => Ok(s.to_object(py)),
        serde_json::Value::Array(arr) => {
            let items: Vec<PyObject> = arr
                .iter()
                .map(|v| json_to_py(py, v))
                .collect::<PyResult<_>>()?;
            Ok(PyList::new_bound(py, items).to_object(py))
        }
        serde_json::Value::Object(map) => {
            let d = PyDict::new_bound(py);
            for (k, v) in map {
                d.set_item(k, json_to_py(py, v)?)?;
            }
            Ok(d.to_object(py))
        }
    }
}

/// Convert a Python object to a `serde_json::Value` recursively.
pub(crate) fn py_to_json(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<serde_json::Value> {
    if obj.is_none() {
        Ok(serde_json::Value::Null)
    } else if let Ok(b) = obj.extract::<bool>() {
        Ok(serde_json::Value::Bool(b))
    } else if let Ok(i) = obj.extract::<i64>() {
        Ok(serde_json::json!(i))
    } else if let Ok(f) = obj.extract::<f64>() {
        Ok(serde_json::json!(f))
    } else if let Ok(s) = obj.extract::<String>() {
        Ok(serde_json::Value::String(s))
    } else if let Ok(list) = obj.downcast::<PyList>() {
        let items: PyResult<Vec<serde_json::Value>> =
            list.iter().map(|item| py_to_json(&item)).collect();
        Ok(serde_json::Value::Array(items?))
    } else if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (k, v) in dict {
            let key: String = k.extract()?;
            map.insert(key, py_to_json(&v)?);
        }
        Ok(serde_json::Value::Object(map))
    } else {
        Ok(serde_json::Value::String(obj.str()?.to_string()))
    }
}

/// Convert an optional Python dict to a Rust `HashMap<String, serde_json::Value>`.
pub(crate) fn pydict_to_hashmap(
    dict: Option<&Bound<'_, PyDict>>,
) -> PyResult<HashMap<String, serde_json::Value>> {
    match dict {
        None => Ok(HashMap::new()),
        Some(d) => {
            let mut map = HashMap::new();
            for (k, v) in d {
                let key: String = k.extract()?;
                map.insert(key, py_to_json(&v)?);
            }
            Ok(map)
        }
    }
}

/// Convert a Rust `HashMap<String, serde_json::Value>` to a Python dict.
pub(crate) fn hashmap_to_pydict(
    py: Python<'_>,
    map: &HashMap<String, serde_json::Value>,
) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    for (k, v) in map {
        d.set_item(k, json_to_py(py, v)?)?;
    }
    Ok(d.to_object(py))
}

/// Convert a [`StorageStatistics`] struct to a Python dictionary.
pub(crate) fn storage_stats_to_dict(
    py: Python<'_>,
    stats: &StorageStatistics,
) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    d.set_item("total_experiences", stats.total_experiences)?;
    d.set_item("storage_size_kb", stats.storage_size_kb)?;
    d.set_item("compressed_experiences", stats.compressed_experiences)?;
    d.set_item("compression_ratio", stats.compression_ratio)?;
    let by_type = PyDict::new_bound(py);
    for (et, count) in &stats.by_type {
        by_type.set_item(et.as_str(), count)?;
    }
    d.set_item("by_type", by_type)?;
    Ok(d.to_object(py))
}

/// Parse a string into a [`MemoryCategory`], raising `ValueError` on failure.
pub(crate) fn parse_category(s: &str) -> PyResult<MemoryCategory> {
    s.parse::<MemoryCategory>().map_err(PyValueError::new_err)
}
