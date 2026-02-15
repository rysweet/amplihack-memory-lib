#!/usr/bin/env python3
"""Verification script for dual-backend architecture.

Demonstrates that both SQLite and Kuzu backends work with the same API.
"""

from amplihack_memory import ExperienceStore, Experience, ExperienceType, MemoryConnector
from datetime import datetime
import shutil
from pathlib import Path


def cleanup():
    """Clean up test directories."""
    test_paths = [
        Path.home() / ".amplihack" / "memory" / "verify-sqlite",
        Path.home() / ".amplihack" / "memory" / "verify-kuzu",
    ]
    for path in test_paths:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


def test_backend(backend_name: str):
    """Test a specific backend."""
    print(f"\n{'='*60}")
    print(f"Testing {backend_name.upper()} Backend")
    print('='*60)

    # Create store with specified backend
    store = ExperienceStore(agent_name=f"verify-{backend_name}", backend=backend_name)
    print(f"✓ Created store with {backend_name} backend")

    # Add experiences
    experiences = []
    for i in range(10):
        exp = Experience(
            experience_type=ExperienceType.SUCCESS if i % 2 == 0 else ExperienceType.PATTERN,
            context=f"Test context {i} with searchable keyword",
            outcome=f"Outcome {i}: Successfully completed task",
            confidence=0.6 + (i * 0.04),
            timestamp=datetime.now(),
            metadata={"test_id": i, "backend": backend_name},
            tags=["test", backend_name],
        )
        exp_id = store.add(exp)
        experiences.append(exp_id)
    print(f"✓ Added {len(experiences)} experiences")

    # Retrieve all
    all_results = store.connector.retrieve_experiences(limit=20)
    assert len(all_results) == 10
    print(f"✓ Retrieved {len(all_results)} experiences")

    # Filter by type
    patterns = store.connector.retrieve_experiences(
        experience_type=ExperienceType.PATTERN
    )
    assert len(patterns) == 5
    print(f"✓ Filtered by type: {len(patterns)} patterns")

    # Filter by confidence
    high_conf = store.connector.retrieve_experiences(min_confidence=0.85)
    assert len(high_conf) > 0
    print(f"✓ Confidence filter: {len(high_conf)} high-confidence experiences")

    # Search
    search_results = store.search("searchable")
    assert len(search_results) > 0
    print(f"✓ Search found {len(search_results)} results for 'searchable'")

    # Search with filters
    filtered_search = store.search(
        "keyword",
        experience_type=ExperienceType.SUCCESS,
        min_confidence=0.7,
        limit=5,
    )
    assert len(filtered_search) > 0
    print(f"✓ Filtered search found {len(filtered_search)} results")

    # Statistics
    stats = store.get_statistics()
    assert stats["total_experiences"] == 10
    assert len(stats["by_type"]) == 2  # SUCCESS and PATTERN
    print(f"✓ Statistics:")
    print(f"  - Total: {stats['total_experiences']}")
    print(f"  - By type: {dict((k.value, v) for k, v in stats['by_type'].items())}")
    print(f"  - Storage: {stats['storage_size_kb']:.2f} KB")

    # Test backend type
    assert store.connector.backend_type == backend_name
    print(f"✓ Backend type verified: {backend_name}")

    print(f"\n{backend_name.upper()} Backend: ALL TESTS PASSED ✓")


def main():
    """Main test runner."""
    print("\n" + "=" * 60)
    print("Dual-Backend Architecture Verification")
    print("=" * 60)

    cleanup()

    try:
        # Test SQLite backend
        test_backend("sqlite")

        # Test Kuzu backend (default)
        test_backend("kuzu")

        # Test default backend is Kuzu
        print("\n" + "=" * 60)
        print("Testing Default Backend")
        print("=" * 60)
        default_store = ExperienceStore(agent_name="verify-default")
        assert default_store.connector.backend_type == "kuzu"
        print("✓ Default backend is Kuzu")

        print("\n" + "=" * 60)
        print("SUCCESS: All backends working perfectly!")
        print("=" * 60)
        print("\nSummary:")
        print("- SQLite backend: ✓ Fully functional")
        print("- Kuzu backend: ✓ Fully functional (DEFAULT)")
        print("- API compatibility: ✓ Same API for both backends")
        print("- Backward compatibility: ✓ SQLite still available")
        print("\nImplementation: Complete ✓")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        cleanup()

    return 0


if __name__ == "__main__":
    exit(main())
