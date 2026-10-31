"""
Brute Force vector search index.

Algorithm:
  For each query, compute cosine similarity against ALL N stored vectors.
  Select top-K by score.

Complexity:
  - Build: O(1) — just stores vectors
  - Search: O(N·D) per query
  - Storage: O(N·D·4) bytes (float32)

Role:
  This is the ground-truth baseline.  Because it examines every vector,
  its recall is always 1.0 (perfect).  All ANN indexes are compared
  against this baseline to measure recall loss.

Android portability:
  Yes — the stored matrix + NumPy-style dot product can be run in Java/Kotlin
  using float arrays and BLAS.  Practical only for small datasets (<50K hadiths).
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import List

import numpy as np

from src.indexes.base import (
    HadithResult,
    IndexStats,
    SearchResult,
    VectorSearchIndex,
)
from src.logging.logger import get_logger

logger = get_logger(__name__)


class BruteForceIndex(VectorSearchIndex):
    """
    Exact cosine similarity search against all stored vectors.

    This is the reference implementation.  Every other index is benchmarked
    against the top-K results produced by this index (Recall@K).

    Args:
        metric: 'cosine' (recommended) or 'dot' (equivalent when normalized).
    """

    ALGORITHM = "brute_force"
    INDEX_FILE = "index.pkl"

    def __init__(self, metric: str = "cosine") -> None:
        self.metric = metric
        self._embeddings: np.ndarray | None = None
        self._hadith_ids: List[str] = []
        self._build_time_s: float = 0.0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, embeddings: np.ndarray, hadith_ids: List[str]) -> None:
        """
        Store embeddings in memory.  No preprocessing needed.

        Brute force requires no training — it stores vectors as-is.
        L2 normalization (done in EmbeddingModel) ensures cosine similarity
        equals the dot product, which is computed cheaply via matrix multiply.
        """
        t0 = time.perf_counter()
        self._embeddings = embeddings.astype(np.float32)
        self._hadith_ids = list(hadith_ids)
        self._build_time_s = time.perf_counter() - t0
        logger.info(
            "BruteForce index built: %d vectors, dim=%d",
            len(hadith_ids), embeddings.shape[1]
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_vector: np.ndarray, top_k: int) -> SearchResult:
        """
        Compute cosine similarity with every stored vector.

        Implementation:
          1. Dot product: scores = embeddings @ query  (vectorized, fast)
          2. argpartition: find top-K indices without full sort
          3. Sort top-K by score descending

        This is O(N·D) per query — linear scan, maximum recall.
        """
        if self._embeddings is None:
            raise RuntimeError("Index not built. Call build() or load() first.")

        t_start = time.perf_counter()

        actual_k = min(top_k, len(self._hadith_ids))
        if top_k > len(self._hadith_ids):
            logger.warning(
                "Requested top_k=%d but only %d candidates available. "
                "Returning %d results.",
                top_k, len(self._hadith_ids), actual_k
            )

        q = query_vector.astype(np.float32)
        # Cosine similarity = dot product when both vectors are L2-normalized
        scores = self._embeddings @ q  # shape: (N,)

        # Efficient top-K selection
        if actual_k < len(scores):
            top_indices = np.argpartition(scores, -actual_k)[-actual_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        else:
            top_indices = np.argsort(scores)[::-1]

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        results = [
            HadithResult(
                rank=rank + 1,
                hadith_id=self._hadith_ids[idx],
                score=float(scores[idx]),
            )
            for rank, idx in enumerate(top_indices[:actual_k])
        ]

        return SearchResult(
            results=results,
            metrics={
                "search_time_ms": round(elapsed_ms, 3),
                "total_time_ms": round(elapsed_ms, 3),
                "n_candidates_examined": len(self._hadith_ids),
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
        state = {
            "embeddings": self._embeddings,
            "hadith_ids": self._hadith_ids,
            "build_time_s": self._build_time_s,
            "metric": self.metric,
        }
        with open(path / self.INDEX_FILE, "wb") as f:
            pickle.dump(state, f, protocol=4)
        size_mb = (path / self.INDEX_FILE).stat().st_size / 1024 / 1024
        logger.info("BruteForce index saved to %s (%.1f MB)", path, size_mb)

    def load(self, path: Path) -> None:
        path = Path(path)
        index_file = path / self.INDEX_FILE
        if not index_file.exists():
            raise FileNotFoundError(
                f"BruteForce index not found at {index_file}. "
                "Run 'python app.py build --algorithm brute_force' first."
            )
        with open(index_file, "rb") as f:
            state = pickle.load(f)
        self._embeddings = state["embeddings"]
        self._hadith_ids = state["hadith_ids"]
        self._build_time_s = state["build_time_s"]
        self.metric = state["metric"]
        logger.info(
            "BruteForce index loaded: %d vectors, dim=%d",
            len(self._hadith_ids), self._embeddings.shape[1]
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> IndexStats:
        if self._embeddings is None:
            raise RuntimeError("Index not built.")
        from pathlib import Path as _Path
        return IndexStats(
            algorithm=self.ALGORITHM,
            num_vectors=len(self._hadith_ids),
            embedding_dim=self._embeddings.shape[1],
            build_time_s=self._build_time_s,
            index_size_bytes=self._embeddings.nbytes,
            params={"metric": self.metric},
        )
