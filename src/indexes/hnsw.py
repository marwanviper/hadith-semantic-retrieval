"""
Hierarchical Navigable Small World (HNSW) approximate nearest-neighbor index.

Algorithm overview:
  HNSW builds a multi-layer proximity graph where:

  - Layer 0 contains ALL vectors, densely connected.
  - Higher layers contain exponentially fewer vectors ("skip list" structure).
  - Each node at layer L is also present in all layers below L.

  During search:
    1. Enter at the top layer at the entry point.
    2. Greedily navigate to the closest neighbor.
    3. Descend to the next layer and repeat.
    4. At layer 0, perform a beam search (ef_search candidates kept).
    5. Return top-K from the beam search result set.

Key parameters:
  M: Max number of bidirectional edges per node.
    - Higher M = denser graph = better recall, more RAM, slower build.
    - Typical: M=8–32. Start with M=16.

  ef_construction: Beam width during index construction.
    - Higher = better quality graph = better recall, slower build.
    - Typical: 100–400. Default 200 is a good balance.

  ef_search: Beam width during search (dynamic candidate list size).
    - Higher = more candidates explored = better recall, slower.
    - ef_search >= top_k required for correctness.
    - Typical: 64–256. Tune for your latency budget.
    - This is the PRIMARY parameter to tune during benchmarking.

Speed vs recall tradeoff:
  ef_search=16   → very fast, recall ~0.70–0.80
  ef_search=64   → balanced, recall ~0.90–0.95
  ef_search=200  → near-exact, recall ~0.98+

Complexity:
  - Build: O(N·M·log(N)) time, O(N·M) space for graph edges
  - Search: O(log(N)·ef_search·M) per query
  - Storage: typically 50–200 MB per million 384-dim vectors at M=16

Android portability:
  hnswlib saves binary index files.  The hnswlib C++ library can be compiled
  for Android via the NDK.  Alternatively, export the graph to a custom
  format and implement HNSW navigation in Kotlin for the Android runtime.
  This index is the RECOMMENDED choice for mobile deployment.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.indexes.base import (
    HadithResult,
    IndexStats,
    SearchResult,
    VectorSearchIndex,
)
from src.logging.logger import get_logger

logger = get_logger(__name__)


class HNSWIndex(VectorSearchIndex):
    """
    HNSW approximate nearest-neighbor index backed by hnswlib.

    Args:
        M:              Max connections per node (graph density).
        ef_construction: Build-time beam width (quality vs speed).
        ef_search:      Query-time beam width (recall vs latency).
        metric:         'cosine' or 'l2' (hnswlib: 'cosine' maps to inner product
                        after normalization, 'l2' for Euclidean).
    """

    ALGORITHM = "hnsw"
    INDEX_FILE = "hnsw.bin"
    META_FILE = "hnsw_meta.json"

    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        metric: str = "cosine",
    ) -> None:
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.metric = metric

        self._index = None          # hnswlib.Index
        self._hadith_ids: List[str] = []
        self._dim: int | None = None
        self._n_elements: int = 0
        self._build_time_s: float = 0.0

    def _hnswlib_space(self) -> str:
        """Map our metric name to hnswlib space string."""
        return {"cosine": "cosine", "l2": "l2", "dot": "ip"}.get(
            self.metric, "cosine"
        )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, embeddings: np.ndarray, hadith_ids: List[str]) -> None:
        """
        Construct the HNSW graph from embeddings.

        hnswlib uses integer IDs (0…N-1) internally. We maintain a
        parallel list that maps integer ID → Hadith string ID.
        """
        try:
            import hnswlib
        except ImportError as e:
            raise ImportError(
                "hnswlib is required for HNSW index. "
                "Install with: pip install hnswlib"
            ) from e

        t0 = time.perf_counter()

        n_vecs, dim = embeddings.shape
        self._dim = dim
        self._hadith_ids = list(hadith_ids)
        self._n_elements = n_vecs

        logger.info(
            "HNSW: Building index (%d vectors, dim=%d, M=%d, ef_construction=%d) …",
            n_vecs, dim, self.M, self.ef_construction
        )

        space = self._hnswlib_space()
        index = hnswlib.Index(space=space, dim=dim)
        index.init_index(
            max_elements=n_vecs,
            ef_construction=self.ef_construction,
            M=self.M,
        )
        index.set_ef(self.ef_search)
        index.set_num_threads(4)

        # Add all vectors with sequential integer IDs
        ids = np.arange(n_vecs, dtype=np.int64)
        index.add_items(embeddings.astype(np.float32), ids)

        self._index = index
        self._build_time_s = time.perf_counter() - t0
        logger.info(
            "HNSW index built in %.2fs",
            self._build_time_s
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_vector: np.ndarray, top_k: int) -> SearchResult:
        """
        Navigate the HNSW graph to find top-K approximate nearest neighbors.

        Graph traversal:
          - Start at entry point in the top layer.
          - Greedy neighbor hop until reaching layer 0.
          - Beam search at layer 0 with ef_search candidates.
          - Return top-K from the beam.

        Increasing ef_search increases recall at the cost of latency.
        """
        if self._index is None:
            raise RuntimeError("HNSW index not built. Call build() or load() first.")

        t_start = time.perf_counter()

        actual_k = min(top_k, self._n_elements)
        if top_k > self._n_elements:
            logger.warning(
                "Requested top_k=%d but only %d elements. Returning %d.",
                top_k, self._n_elements, actual_k
            )

        # Ensure ef >= k for correct results
        effective_ef = max(self.ef_search, actual_k)
        self._index.set_ef(effective_ef)

        q = query_vector.astype(np.float32).reshape(1, -1)
        labels, distances = self._index.knn_query(q, k=actual_k)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # hnswlib returns L2 distance for 'cosine' space; convert to similarity
        # For 'cosine' space: dist = 1 - cosine_similarity (hnswlib convention)
        if self.metric == "cosine":
            scores = 1.0 - distances[0]
        else:
            scores = -distances[0]  # Negate so higher = better

        results = [
            HadithResult(
                rank=rank + 1,
                hadith_id=self._hadith_ids[int(label)],
                score=float(score),
            )
            for rank, (label, score) in enumerate(zip(labels[0], scores))
        ]

        return SearchResult(
            results=results,
            metrics={
                "search_time_ms": round(elapsed_ms, 3),
                "total_time_ms": round(elapsed_ms, 3),
                "ef_search_used": effective_ef,
            },
            algorithm=self.ALGORITHM,
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # hnswlib has its own binary serialization
        self._index.save_index(str(path / self.INDEX_FILE))

        meta = {
            "hadith_ids": self._hadith_ids,
            "dim": self._dim,
            "n_elements": self._n_elements,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "metric": self.metric,
            "build_time_s": self._build_time_s,
        }
        with open(path / self.META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        size_bytes = (path / self.INDEX_FILE).stat().st_size
        logger.info(
            "HNSW index saved to %s (%.1f MB)", path, size_bytes / 1024 / 1024
        )

    def load(self, path: Path) -> None:
        try:
            import hnswlib
        except ImportError as e:
            raise ImportError("hnswlib is required. pip install hnswlib") from e

        path = Path(path)
        index_file = path / self.INDEX_FILE
        meta_file = path / self.META_FILE

        if not index_file.exists():
            raise FileNotFoundError(
                f"HNSW index not found at {index_file}. "
                "Run 'python app.py build --algorithm hnsw' first."
            )

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self._dim = meta["dim"]
        self._hadith_ids = meta["hadith_ids"]
        self._n_elements = meta["n_elements"]
        self.M = meta["M"]
        self.ef_construction = meta["ef_construction"]
        self.ef_search = meta["ef_search"]
        self.metric = meta["metric"]
        self._build_time_s = meta["build_time_s"]

        space = self._hnswlib_space()
        index = hnswlib.Index(space=space, dim=self._dim)
        index.load_index(str(index_file), max_elements=self._n_elements)
        index.set_ef(self.ef_search)
        self._index = index

        size_mb = index_file.stat().st_size / 1024 / 1024
        logger.info(
            "HNSW index loaded: %d vectors, dim=%d, M=%d, ef=%d (%.1f MB)",
            self._n_elements, self._dim, self.M, self.ef_search, size_mb
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> IndexStats:
        if self._index is None:
            raise RuntimeError("HNSW index not built.")
        index_file_size = 0
        return IndexStats(
            algorithm=self.ALGORITHM,
            num_vectors=self._n_elements,
            embedding_dim=self._dim,
            build_time_s=self._build_time_s,
            index_size_bytes=index_file_size,  # Updated after save()
            params={
                "M": self.M,
                "ef_construction": self.ef_construction,
                "ef_search": self.ef_search,
                "metric": self.metric,
            },
        )
