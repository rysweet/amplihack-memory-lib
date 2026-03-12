#!/usr/bin/env python3
"""Compare Python and Rust parity test results.

Loads both JSON result files, matches tests by name, and reports
MATCH / DIVERGENCE / MISSING for each test case.
"""

import json
import math
import sys

PYTHON_FILE = "/tmp/python_parity_results.json"
RUST_FILE = "/tmp/rust_parity_results.json"

NUMERIC_TOLERANCE = 0.05  # 5%


def normalize(val):
    """Normalize a value for comparison."""
    if isinstance(val, str):
        return " ".join(val.split()).strip().lower()
    if isinstance(val, list):
        return sorted(normalize(v) for v in val)
    if isinstance(val, dict):
        return {k: normalize(v) for k, v in sorted(val.items())}
    return val


def values_equal(py_val, rs_val, path=""):
    """Deep-compare two values with tolerance for numerics and normalization for strings."""
    diffs = []

    if isinstance(py_val, (int, float)) and isinstance(rs_val, (int, float)):
        if py_val == 0 and rs_val == 0:
            return diffs
        if py_val == 0 or rs_val == 0:
            if abs(py_val - rs_val) > NUMERIC_TOLERANCE:
                diffs.append((path, py_val, rs_val))
            return diffs
        if abs(py_val - rs_val) / max(abs(py_val), abs(rs_val)) > NUMERIC_TOLERANCE:
            diffs.append((path, py_val, rs_val))
        return diffs

    if isinstance(py_val, bool) and isinstance(rs_val, bool):
        if py_val != rs_val:
            diffs.append((path, py_val, rs_val))
        return diffs

    if isinstance(py_val, str) and isinstance(rs_val, str):
        if normalize(py_val) != normalize(rs_val):
            diffs.append((path, py_val, rs_val))
        return diffs

    if isinstance(py_val, list) and isinstance(rs_val, list):
        # Compare as sets if items are hashable primitives
        try:
            py_set = set(normalize(v) for v in py_val)
            rs_set = set(normalize(v) for v in rs_val)
            if py_set != rs_set:
                diffs.append((path, sorted(py_set), sorted(rs_set)))
        except TypeError:
            # Fall back to ordered comparison
            for i in range(max(len(py_val), len(rs_val))):
                if i >= len(py_val):
                    diffs.append((f"{path}[{i}]", "<missing>", rs_val[i]))
                elif i >= len(rs_val):
                    diffs.append((f"{path}[{i}]", py_val[i], "<missing>"))
                else:
                    diffs.extend(values_equal(py_val[i], rs_val[i], f"{path}[{i}]"))
        return diffs

    if isinstance(py_val, dict) and isinstance(rs_val, dict):
        all_keys = set(py_val.keys()) | set(rs_val.keys())
        for key in sorted(all_keys):
            child_path = f"{path}.{key}" if path else key
            if key not in py_val:
                diffs.append((child_path, "<missing>", rs_val[key]))
            elif key not in rs_val:
                diffs.append((child_path, py_val[key], "<missing>"))
            else:
                diffs.extend(values_equal(py_val[key], rs_val[key], child_path))
        return diffs

    # Type mismatch or other — try coercing int↔float
    if type(py_val) != type(rs_val):
        try:
            return values_equal(float(py_val), float(rs_val), path)
        except (TypeError, ValueError):
            pass
        diffs.append((path, f"{py_val} (type={type(py_val).__name__})",
                       f"{rs_val} (type={type(rs_val).__name__})"))
    elif py_val != rs_val:
        diffs.append((path, py_val, rs_val))

    return diffs


def main():
    with open(PYTHON_FILE) as f:
        py_results = json.load(f)
    with open(RUST_FILE) as f:
        rs_results = json.load(f)

    py_by_name = {r["test"]: r for r in py_results}
    rs_by_name = {r["test"]: r for r in rs_results}

    all_tests = sorted(set(py_by_name.keys()) | set(rs_by_name.keys()))

    matches = 0
    divergences = 0
    missing = 0
    py_failures = 0
    rs_failures = 0

    print("=" * 78)
    print("  A/B PARITY COMPARISON: Python ↔ Rust")
    print("=" * 78)
    print()

    for test in all_tests:
        py = py_by_name.get(test)
        rs = rs_by_name.get(test)

        if py is None:
            print(f"  MISSING (Python)  {test}")
            missing += 1
            continue
        if rs is None:
            print(f"  MISSING (Rust)    {test}")
            missing += 1
            continue

        if not py["success"]:
            py_failures += 1
            print(f"  PY-FAIL           {test}")
            err = py["result"]
            if isinstance(err, dict) and "error" in err:
                print(f"                    Python error: {err['error'][:100]}")
            continue

        if not rs["success"]:
            rs_failures += 1
            print(f"  RS-FAIL           {test}")
            err = rs["result"]
            if isinstance(err, dict) and "error" in err:
                print(f"                    Rust error: {err['error'][:100]}")
            continue

        diffs = values_equal(py["result"], rs["result"])

        if not diffs:
            matches += 1
            print(f"  MATCH             {test}")
        else:
            divergences += 1
            print(f"  DIVERGENCE        {test}")
            for path, py_val, rs_val in diffs:
                loc = f"  .{path}" if path else ""
                print(f"                    {loc}")
                print(f"                      Python: {py_val}")
                print(f"                      Rust:   {rs_val}")

    print()
    print("=" * 78)
    print(f"  SUMMARY")
    print(f"  Total tests:  {len(all_tests)}")
    print(f"  Matches:      {matches}")
    print(f"  Divergences:  {divergences}")
    print(f"  Missing:      {missing}")
    print(f"  Py failures:  {py_failures}")
    print(f"  Rs failures:  {rs_failures}")
    parity_pct = (matches / len(all_tests) * 100) if all_tests else 0
    print(f"  Parity:       {parity_pct:.1f}%")
    print("=" * 78)

    sys.exit(0 if divergences == 0 and missing == 0 else 1)


if __name__ == "__main__":
    main()
