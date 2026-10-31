"""
Abstract base class for all vector search indexes.

Every algorithm (Brute Force, IVF, HNSW, PQ, IVF+PQ) implements this
interface. The rest of the codebase interacts only with VectorSearchIndex
and never with algorithm-specific classes directly.

This is the cornerstone of the Factory Pattern used in this project.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class HadithResult:
    """A single search result."""
    rank: int               # 1-indexed rank in the result list
    hadith_id: str          # ID from the dataset
    score: float            # Similarity score (higher = more similar for cosine)
    text: Optional[str] = None
    book: Optional[str] = None
    chapter: Optional[str] = None
    narrator: Optional[str] = None
    source: Optional[str] = None
    reference: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "hadith_id": self.hadith_id,
            "score": self.score,
            "text": self.text,
            "book": self.book,
            "chapter": self.chapter,
            "narrator": self.narrator,
            "source": self.source,
            "reference": self.reference,
        }


@dataclass
class SearchResult:
    """
    Container for search results plus timing metrics.

    The metrics dict contains millisecond timings:
      - embedding_time_ms
      - search_time_ms
      - postprocess_time_ms
      - total_time_ms
    """
    results: List[HadithResult] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    algorithm: str = ""
    query: str = ""
    top_k: int = 0


@dataclass
class IndexStats:
    """Statistics about a built index."""
    algorithm: str
    num_vectors: int
    embedding_dim: int
    build_time_s: float
    index_size_bytes: int
    params: Dict[str, Any] = field(default_factory=dict)

    def index_size_mb(self) -> float:
        return self.index_size_bytes / 1024 / 1024

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "num_vectors": self.num_vectors,
            "embedding_dim": self.embedding_dim,
            "build_time_s": self.build_time_s,
            "index_size_bytes": self.index_size_bytes,
            "index_size_mb": self.index_size_mb(),
            "params": self.params,
        }


class VectorSearchIndex(ABC):
    """
    Abstract interface for all vector search algorithms.

    Subclasses:
        BruteForceIndex — exact cosine search (baseline)
        IVFIndex        — inverted file index (approximate)
        HNSWIndex       — hierarchical navigable small worlds (approximate)
        PQIndex         — product quantization (compressed approximate)
        IVFPQIndex      — IVF + PQ combined (compressed approximate)

    Usage::

        index = VectorSearchFactory.create("hnsw", config)
        index.build(embeddings, hadith_ids)
        index.save(path)

        # Later:
        index.load(path)
        result = index.search(query_vector, top_k=10)
    """

    @abstractmethod
    def build(self, embeddings: np.ndarray, hadith_ids: List[str]) -> None:
        """
        Build the search index from precomputed embeddings.

        Args:
            embeddings: Float32 array of shape (N, D).
            hadith_ids: Ordered list of N Hadith IDs (row i → ID i).
        """
        ...

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
    ) -> SearchResult:
        """
        Search for the top-K nearest Hadiths.

        Args:
            query_vector: Float32 array of shape (D,).
            top_k:        Number of results to return.

        Returns:
            SearchResult with ranked HadithResult list and timing metrics.
        """
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Persist the index to disk.

        Args:
            path: Directory to store index files.
        """
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """
        Load the index from disk.

        Args:
            path: Directory containing saved index files.

        Raises:
            FileNotFoundError: If index files are missing.
        """
        ...

    @abstractmethod
    def get_stats(self) -> IndexStats:
        """
        Return statistics about the current index.

        Raises:
            RuntimeError: If the index has not been built or loaded.
        """
        ...

    def is_built(self) -> bool:
        """Return True if the index is ready to search."""
        try:
            self.get_stats()
            return True
        except RuntimeError:
            return False
