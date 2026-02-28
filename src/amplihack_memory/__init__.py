"""amplihack-memory-lib: Standalone memory system for goal-seeking agents."""

__version__ = "0.3.0"

from .cognitive_memory import CognitiveMemory
from .connector import MemoryConnector
from .contradiction import detect_contradiction
from .entity_extraction import extract_entity_name
from .exceptions import (
    ExperienceNotFoundError,
    InvalidExperienceError,
    MemoryError,
    MemoryQuotaExceededError,
)
from .experience import Experience, ExperienceType
from .hierarchical_memory import (
    HierarchicalMemory,
    KnowledgeEdge,
    KnowledgeNode,
    KnowledgeSubgraph,
    MemoryClassifier,
)
from .memory_types import (
    ConsolidatedEpisode,
    EpisodicMemory,
    MemoryCategory,
    ProceduralMemory,
    ProspectiveMemory,
    SemanticFact,
    SensoryItem,
    WorkingMemorySlot,
)
from .security import (
    AgentCapabilities,
    CredentialScrubber,
    QueryCostExceededError,
    QueryValidator,
    ScopeLevel,
    SecureMemoryBackend,
    SecurityViolationError,
)
from .similarity import (
    compute_similarity,
    compute_tag_similarity,
    compute_word_similarity,
    rerank_facts_by_query,
)
from .store import ExperienceStore

__all__ = [
    # Hierarchical memory (Graph RAG)
    "HierarchicalMemory",
    "KnowledgeNode",
    "KnowledgeEdge",
    "KnowledgeSubgraph",
    "MemoryClassifier",
    # Shared utilities
    "extract_entity_name",
    "detect_contradiction",
    "compute_similarity",
    "compute_word_similarity",
    "compute_tag_similarity",
    "rerank_facts_by_query",
    # Cognitive memory
    "CognitiveMemory",
    "MemoryCategory",
    "SensoryItem",
    "WorkingMemorySlot",
    "EpisodicMemory",
    "SemanticFact",
    "ProceduralMemory",
    "ProspectiveMemory",
    "ConsolidatedEpisode",
    # Experience store (existing)
    "Experience",
    "ExperienceType",
    "MemoryConnector",
    "ExperienceStore",
    # Security (existing)
    "AgentCapabilities",
    "ScopeLevel",
    "CredentialScrubber",
    "QueryValidator",
    "SecureMemoryBackend",
    # Exceptions (existing)
    "MemoryError",
    "ExperienceNotFoundError",
    "InvalidExperienceError",
    "MemoryQuotaExceededError",
    "SecurityViolationError",
    "QueryCostExceededError",
]
