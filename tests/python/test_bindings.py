"""Integration tests for amplihack_memory_rs Python bindings."""
import pytest
import sys
import tempfile
import os

try:
    import amplihack_memory_rs as amr
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(not HAS_MODULE, reason="amplihack_memory_rs not built")


class TestPyExperience:
    def test_create_experience(self):
        exp = amr.PyExperience("learning", "context here", "outcome here", ["tag1"])
        assert exp.experience_type == "learning"
        assert exp.context == "context here"

    def test_experience_repr(self):
        exp = amr.PyExperience("test", "ctx", "out", [])
        assert "test" in repr(exp)


class TestPyMemoryConnector:
    def test_create_connector(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = amr.PyMemoryConnector("test-agent", os.path.join(tmp, "test.db"))
            assert conn is not None

    def test_store_and_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = amr.PyMemoryConnector("test-agent", os.path.join(tmp, "test.db"))
            exp = amr.PyExperience("learning", "rust is fast", "good performance", ["rust"])
            conn.store_experience(exp)
            results = conn.search("rust", 5)
            assert len(results) > 0

    def test_path_traversal_rejected(self):
        with pytest.raises(Exception):
            amr.PyMemoryConnector("test-agent", "../../etc/passwd")


class TestPyGraphStore:
    def test_add_and_get_node(self):
        store = amr.PyGraphStore()
        store.add_node("n1", "test", {"key": "value"})
        node = store.get_node("n1")
        assert node is not None
        assert node["node_type"] == "test"

    def test_add_edge(self):
        store = amr.PyGraphStore()
        store.add_node("n1", "type1", {})
        store.add_node("n2", "type2", {})
        store.add_edge("n1", "n2", "CONNECTS", {})
        edges = store.get_edges("n1", "outgoing", None)
        assert len(edges) > 0

    def test_search_nodes(self):
        store = amr.PyGraphStore()
        store.add_node("n1", "concept", {"name": "rust programming"})
        results = store.search_nodes("rust", 10)
        assert len(results) > 0

    def test_node_count(self):
        store = amr.PyGraphStore()
        assert store.node_count() == 0
        store.add_node("n1", "test", {})
        assert store.node_count() == 1


class TestUtilityFunctions:
    def test_extract_entities(self):
        result = amr.extract_entities("Python is a programming language")
        assert isinstance(result, str)

    def test_jaccard_similarity(self):
        sim = amr.jaccard_similarity("hello world", "hello earth")
        assert 0.0 <= sim <= 1.0

    def test_detect_contradiction(self):
        result = amr.detect_contradiction("The sky is blue", "The sky is red", "color", "color")
        assert isinstance(result, bool)

    def test_scrub_credentials(self):
        result = amr.scrub_credentials("my password=secret123 is here")
        assert "secret123" not in result


class TestPySecureMemoryBackend:
    def test_create_secure_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = amr.PySecureMemoryBackend(
                os.path.join(tmp, "secure.db"),
                "test-agent",
                100,
                "session_only",
            )
            assert backend is not None


class TestPyCognitiveMemory:
    def test_store_and_recall_fact(self):
        with tempfile.TemporaryDirectory() as tmp:
            cm = amr.PyCognitiveMemory("test-agent", os.path.join(tmp, "cog.db"))
            cm.store_fact("The Earth orbits the Sun", "astronomy")
            facts = cm.recall_facts("Earth Sun", 5)
            assert len(facts) > 0
