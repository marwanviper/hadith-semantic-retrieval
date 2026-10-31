"""
Inverted File Index (IVF) for approximate nearest-neighbor search.

Algorithm overview:
  1. BUILD:
     a. Cluster all N vectors into K centroids using K-Means.
     b. Assign each vector to its nearest centroid.
     c. Store vectors in inverted lists (centroid → [vector, ...])

  2. SEARCH:
     a. Compute distance from query to all K centroids.
     b. Select the nprobe nearest centroids.
     c. Linear scan only the nprobe inverted lists.
     d. Return top-K results.

Key parameters:
  n_clusters (K): Number of Voronoi cells. Controls partitioning granularity.
    - Higher K = smaller clusters = faster search, lower recall.
    - Rule of thumb: K ≈ sqrt(N), e.g. K=256 for 65536 vectors.

  nprobe: Number of clusters searched per query.
    - Increasing nprobe improves recall at the cost of search time.
    - nprobe=1 → fastest but lowest recall.
    - nprobe=K → exact search (equivalent to brute force).
    - Typical trade-off: nprobe=8-32 for good recall.

Complexity:
  - Build:  O(N·K·D·iters) for K-Means
  - Search: O(K·D + (N/K)·nprobe·D)
  - Storage: O(N·D·4) bytes (same as brute force, no compression)

Android portability:
  Yes — centroids + inverted lists are plain arrays. K-Means lookup and
  linear scan can be implemented in Kotlin with float arrays.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.indexes.base import (
    HadithResult,
    IndexStats,
    SearchResult,
    VectorSearchIndex,
)
from src.logging.logger import get_logger

logger = get_logger(__name__)


class IVFIndex(VectorSearchIndex):
    """
    Approximate nearest-neighbor search using Inverted File Index.

    Args:
        n_clusters: Number of K-Means clusters (Voronoi cells).
        nprobe:     Number of clusters to search per query.
        metric:     Distance metric ('cosine' or 'l2').
        random_seed: Seed for K-Means reproducibility.
    """

    ALGORITHM = "ivf"
    INDEX_FILE = "ivf_index.pkl"

    def __init__(
        self,
        n_clusters: int = 256,
        nprobe: int = 8,
        metric: str = "cosine",
        random_seed: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.nprobe = nprobe
        self.metric = metric
        self.random_seed = random_seed

        self._centroids: np.ndarray | None = None          # (K, D)
        self._inverted_lists: list[list[int]] | None = None # centroid_id → [global_idx]
        self._embeddings: np.ndarray | None = None         # (N, D) all vectors
        self._hadith_ids: List[str] = []
        self._build_time_s: float = 0.0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, embeddings: np.ndarray, hadith_ids: List[str]) -> None:
        """
        Cluster vectors with K-Means and build inverted lists.

        Steps:
          1. Run K-Means on all embeddings to find centroids.
          2. Assign each vector to its nearest centroid.
          3. Store the mapping centroid → [vector_indices].
        """
        from sklearn.cluster import MiniBatchKMeans

        t0 = time.perf_counter()

        n_vecs, dim = embeddings.shape
        effective_clusters = min(self.n_clusters, n_vecs)
        if effective_clusters < self.n_clusters:
            logger.warning(
                "n_clusters=%d exceeds dataset size %d; using %d clusters instead.",
                self.n_clusters, n_vecs, effective_clusters
            )

        logger.info(
            "IVF: Training K-Means with %d clusters on %d vectors (dim=%d) …",
            effective_clusters, n_vecs, dim
        )

        kmeans = MiniBatchKMeans(
            n_clusters=effective_clusters,
            random_state=self.random_seed,
            n_init=3,
            max_iter=100,
        )
        labels = kmeans.fit_predict(embeddings)  # shape (N,)

        self._centroids = kmeans.cluster_centers_.astype(np.float32)  # (K, D)
        # Normalize centroids for cosine search
        if self.metric == "cosine":
            norms = np.linalg.norm(self._centroids, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            self._centroids = self._centroids / norms

        self._inverted_lists = [[] for _ in range(effective_clusters)]
        for vec_idx, cluster_id in enumerate(labels):
            self._inverted_lists[cluster_id].append(vec_idx)

        self._embeddings = embeddings.astype(np.float32)
        self._hadith_ids = list(hadith_ids)
        self._build_time_s = time.perf_counter() - t0

        avg_list_size = n_vecs / effective_clusters
        logger.info(
            "IVF index built in %.2fs: %d clusters, avg list size=%.1f",
            self._build_time_s, effective_clusters, avg_list_size
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_vector: np.ndarray, top_k: int) -> SearchResult:
        """
        Search nprobe nearest clusters and return top-K results.

        Steps:
          1. Dot product with all centroids to find nearest.
          2. Gather all vectors from selected clusters.
          3. Compute exact similarity within those candidates.
          4. Return top-K.
        """
        if self._centroids is None:
            raise RuntimeError("IVF index not built. Call build() or load() first.")

        t_start = time.perf_counter()

        q = query_vector.astype(np.float32)

        # Step 1: find nprobe nearest centroids
        centroid_scores = self._centroids @ q  # (K,)
        effective_nprobe = min(self.nprobe, len(self._centroids))
        nearest_centroids = np.argpartition(centroid_scores, -effective_nprobe)[-effective_nprobe:]

        # Step 2: collect candidate vector indices from selected clusters
        candidate_indices: list[int] = []
        for c_id in nearest_centroids:
            candidate_indices.extend(self._inverted_lists[c_id])

        if not candidate_indices:
            return SearchResult(
                results=[],
                metrics={"search_time_ms": 0.0, "total_time_ms": 0.0},
                algorithm=self.ALGORITHM,
                top_k=top_k,
            )

        # Step 3: exact similarity within candidates
        candidate_indices_arr = np.array(candidate_indices, dtype=np.int64)
        candidate_vecs = self._embeddings[candidate_indices_arr]  # (M, D)
        scores = candidate_vecs @ q  # (M,)

        # Step 4: top-K
        actual_k = min(top_k, len(scores))
        if actual_k < len(scores):
            top_local = np.argpartition(scores, -actual_k)[-actual_k:]
            top_local = top_local[np.argsort(scores[top_local])[::-1]]
        else:
            top_local = np.argsort(scores)[::-1]

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        results = [
            HadithResult(
                rank=rank + 1,
                hadith_id=self._hadith_ids[candidate_indices_arr[local_idx]],
                score=float(scores[local_idx]),
            )
            for rank, local_idx in enumerate(top_local[:actual_k])
        ]

        return SearchResult(
            results=results,
            metrics={
                "search_time_ms": round(elapsed_ms, 3),
                "total_time_ms": round(elapsed_ms, 3),
                "n_candidates_examined": len(candidate_indices),
                "n_clusters_searched": effective_nprobe,
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
            "centroids": self._centroids,
            "inverted_lists": self._inverted_lists,
            "embeddings": self._embeddings,
            "hadith_ids": self._hadith_ids,
            "n_clusters": self.n_clusters,
            "nprobe": self.nprobe,
            "metric": self.metric,
            "build_time_s": self._build_time_s,
        }
        with open(path / self.INDEX_FILE, "wb") as f:
            pickle.dump(state, f, protocol=4)
        size_mb = (path / self.INDEX_FILE).stat().st_size / 1024 / 1024
        logger.info("IVF index saved to %s (%.1f MB)", path, size_mb)

    def load(self, path: Path) -> None:
        path = Path(path)
        index_file = path / self.INDEX_FILE
        if not index_file.exists():
            raise FileNotFoundError(
                f"IVF index not found at {index_file}. "
                "Run 'python app.py build --algorithm ivf' first."
            )
        with open(index_file, "rb") as f:
            state = pickle.load(f)
        self._centroids = state["centroids"]
        self._inverted_lists = state["inverted_lists"]
        self._embeddings = state["embeddings"]
        self._hadith_ids = state["hadith_ids"]
        self.n_clusters = state["n_clusters"]
        self.nprobe = state["nprobe"]
        self.metric = state["metric"]
        self._build_time_s = state["build_time_s"]
        logger.info(
            "IVF index loaded: %d vectors, %d clusters",
            len(self._hadith_ids), len(self._centroids)
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> IndexStats:
        if self._embeddings is None:
            raise RuntimeError("IVF index not built.")
        return IndexStats(
            algorithm=self.ALGORITHM,
            num_vectors=len(self._hadith_ids),
            embedding_dim=self._embeddings.shape[1],
            build_time_s=self._build_time_s,
            index_size_bytes=self._embeddings.nbytes + self._centroids.nbytes,
            params={
                "n_clusters": self.n_clusters,
                "nprobe": self.nprobe,
                "metric": self.metric,
            },
        )
