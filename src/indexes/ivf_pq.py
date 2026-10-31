"""
IVF + Product Quantization combined index.

Search flow:
  Query
    ↓
  Embed query
    ↓
  Compute distance to all IVF centroids
    ↓
  Select nprobe nearest clusters
    ↓
  For each cluster, retrieve PQ-compressed residuals
    ↓
  ADC over PQ codes → approximate scores
    ↓
  Top-K results

How it works:
  IVF reduces the number of vectors to score from N to N/K×nprobe.
  PQ compresses each vector so that scoring is fast (byte lookups instead
  of float dot products) and memory usage is reduced.

When to use:
  Large datasets (>100K) where you need both:
    - Fast search (IVF partitioning reduces candidate set)
    - Low memory (PQ compression reduces vector size)
  Typical: 1M vectors, M=8 PQ, K=1024 IVF → ~10x faster than PQ alone
           with similar recall.

Parameters:
  n_clusters: IVF coarse quantizer clusters. Larger = faster search, lower recall.
  nprobe:     IVF clusters searched. Larger = better recall, slower.
  subquantizers (M): PQ subspaces. Must divide embedding_dim.
  bits_per_code: PQ bits per subspace.

Android portability:
  Yes — IVF centroids + PQ codebooks + byte codes are all plain arrays.
  This is the recommended combination for large-scale mobile search.
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


class IVFPQIndex(VectorSearchIndex):
    """
    Inverted File Index with Product Quantization.

    Args:
        n_clusters:    IVF number of coarse clusters.
        nprobe:        IVF clusters searched per query.
        subquantizers: PQ number of subspaces.
        bits_per_code: PQ bits per subspace centroid.
        metric:        'cosine' or 'l2'.
        random_seed:   Reproducibility seed.
    """

    ALGORITHM = "ivf_pq"
    INDEX_FILE = "ivf_pq_index.pkl"

    def __init__(
        self,
        n_clusters: int = 64,
        nprobe: int = 4,
        subquantizers: int = 8,
        bits_per_code: int = 8,
        metric: str = "cosine",
        random_seed: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.nprobe = nprobe
        self.subquantizers = subquantizers
        self.bits_per_code = bits_per_code
        self.metric = metric
        self.random_seed = random_seed

        self._ivf_centroids: np.ndarray | None = None          # (K, D)
        self._inverted_lists: list[list[int]] | None = None    # cluster → [global_idx]
        self._pq_codebooks: np.ndarray | None = None           # (M, K_pq, sub_dim)
        self._pq_codes: np.ndarray | None = None               # (N, M) uint8
        self._hadith_ids: List[str] = []
        self._dim: int | None = None
        self._n_pq_centroids: int = 0
        self._sub_dim: int = 0
        self._build_time_s: float = 0.0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, embeddings: np.ndarray, hadith_ids: List[str]) -> None:
        """
        Build IVF partitioning then PQ-encode vectors within clusters.

        Steps:
          1. K-Means on full vectors → IVF centroids
          2. Assign each vector to a cluster → inverted lists
          3. K-Means per subspace → PQ codebooks (trained on full dataset)
          4. Encode all vectors into PQ byte codes
        """
        from sklearn.cluster import MiniBatchKMeans

        t0 = time.perf_counter()

        n_vecs, dim = embeddings.shape
        self._dim = dim

        if dim % self.subquantizers != 0:
            raise ValueError(
                f"Embedding dim={dim} not divisible by subquantizers={self.subquantizers}."
            )

        self._sub_dim = dim // self.subquantizers
        self._n_pq_centroids = min(2 ** self.bits_per_code, n_vecs)
        effective_clusters = min(self.n_clusters, n_vecs)

        # --- Step 1 & 2: IVF clustering ---
        logger.info(
            "IVFPQ: Training IVF with %d clusters on %d vectors …",
            effective_clusters, n_vecs
        )
        ivf_kmeans = MiniBatchKMeans(
            n_clusters=effective_clusters,
            random_state=self.random_seed,
            n_init=3, max_iter=100,
        )
        labels = ivf_kmeans.fit_predict(embeddings)
        self._ivf_centroids = ivf_kmeans.cluster_centers_.astype(np.float32)
        if self.metric == "cosine":
            norms = np.linalg.norm(self._ivf_centroids, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            self._ivf_centroids /= norms

        self._inverted_lists = [[] for _ in range(effective_clusters)]
        for vec_idx, cluster_id in enumerate(labels):
            self._inverted_lists[cluster_id].append(vec_idx)

        # --- Step 3 & 4: PQ codebook training and encoding ---
        logger.info(
            "IVFPQ: Training PQ with %d subspaces × %d centroids …",
            self.subquantizers, self._n_pq_centroids
        )
        codebooks = np.zeros(
            (self.subquantizers, self._n_pq_centroids, self._sub_dim), dtype=np.float32
        )
        codes = np.zeros((n_vecs, self.subquantizers), dtype=np.uint8)

        for m in range(self.subquantizers):
            start_col = m * self._sub_dim
            end_col = start_col + self._sub_dim
            sub_vecs = embeddings[:, start_col:end_col].astype(np.float32)
            pq_kmeans = MiniBatchKMeans(
                n_clusters=self._n_pq_centroids,
                random_state=self.random_seed + m + 1000,
                n_init=3, max_iter=100,
            )
            assignments = pq_kmeans.fit_predict(sub_vecs)
            codebooks[m] = pq_kmeans.cluster_centers_.astype(np.float32)
            codes[:, m] = assignments.astype(np.uint8)
            logger.debug("IVFPQ: PQ subspace %d/%d trained.", m + 1, self.subquantizers)

        self._pq_codebooks = codebooks
        self._pq_codes = codes
        self._hadith_ids = list(hadith_ids)
        self._build_time_s = time.perf_counter() - t0
        logger.info("IVFPQ index built in %.2fs", self._build_time_s)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_vector: np.ndarray, top_k: int) -> SearchResult:
        """
        IVF coarse quantization + PQ fine scoring.

        Steps:
          1. Find nprobe nearest IVF centroids.
          2. Collect candidate vector indices from those clusters.
          3. Precompute PQ distance table for the query.
          4. ADC lookup over candidate PQ codes.
          5. Return top-K.
        """
        if self._ivf_centroids is None:
            raise RuntimeError("IVFPQ index not built. Call build() or load() first.")

        t_start = time.perf_counter()

        q = query_vector.astype(np.float32)

        # Step 1: IVF centroid lookup
        centroid_scores = self._ivf_centroids @ q
        effective_nprobe = min(self.nprobe, len(self._ivf_centroids))
        nearest_centroids = np.argpartition(centroid_scores, -effective_nprobe)[-effective_nprobe:]

        # Step 2: Gather candidates
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

        candidate_arr = np.array(candidate_indices, dtype=np.int64)

        # Step 3: Precompute PQ distance table
        dist_table = np.zeros((self.subquantizers, self._n_pq_centroids), dtype=np.float32)
        for m in range(self.subquantizers):
            start_col = m * self._sub_dim
            end_col = start_col + self._sub_dim
            q_sub = q[start_col:end_col]
            dist_table[m] = self._pq_codebooks[m] @ q_sub

        # Step 4: ADC over candidate codes
        candidate_codes = self._pq_codes[candidate_arr]  # (n_cands, M)
        scores = np.zeros(len(candidate_arr), dtype=np.float32)
        for m in range(self.subquantizers):
            scores += dist_table[m][candidate_codes[:, m].astype(np.int32)]

        # Step 5: top-K
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
                hadith_id=self._hadith_ids[int(candidate_arr[local_idx])],
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
            "ivf_centroids": self._ivf_centroids,
            "inverted_lists": self._inverted_lists,
            "pq_codebooks": self._pq_codebooks,
            "pq_codes": self._pq_codes,
            "hadith_ids": self._hadith_ids,
            "n_clusters": self.n_clusters,
            "nprobe": self.nprobe,
            "subquantizers": self.subquantizers,
            "bits_per_code": self.bits_per_code,
            "metric": self.metric,
            "dim": self._dim,
            "n_pq_centroids": self._n_pq_centroids,
            "sub_dim": self._sub_dim,
            "build_time_s": self._build_time_s,
        }
        with open(path / self.INDEX_FILE, "wb") as f:
            pickle.dump(state, f, protocol=4)
        size_mb = (path / self.INDEX_FILE).stat().st_size / 1024 / 1024
        logger.info("IVFPQ index saved to %s (%.1f MB)", path, size_mb)

    def load(self, path: Path) -> None:
        path = Path(path)
        index_file = path / self.INDEX_FILE
        if not index_file.exists():
            raise FileNotFoundError(
                f"IVFPQ index not found at {index_file}. "
                "Run 'python app.py build --algorithm ivf_pq' first."
            )
        with open(index_file, "rb") as f:
            state = pickle.load(f)
        self._ivf_centroids = state["ivf_centroids"]
        self._inverted_lists = state["inverted_lists"]
        self._pq_codebooks = state["pq_codebooks"]
        self._pq_codes = state["pq_codes"]
        self._hadith_ids = state["hadith_ids"]
        self.n_clusters = state["n_clusters"]
        self.nprobe = state["nprobe"]
        self.subquantizers = state["subquantizers"]
        self.bits_per_code = state["bits_per_code"]
        self.metric = state["metric"]
        self._dim = state["dim"]
        self._n_pq_centroids = state["n_pq_centroids"]
        self._sub_dim = state["sub_dim"]
        self._build_time_s = state["build_time_s"]
        logger.info(
            "IVFPQ index loaded: %d vectors, %d IVF clusters, PQ M=%d K=%d",
            len(self._hadith_ids), len(self._ivf_centroids),
            self.subquantizers, self._n_pq_centroids
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> IndexStats:
        if self._pq_codes is None:
            raise RuntimeError("IVFPQ index not built.")
        return IndexStats(
            algorithm=self.ALGORITHM,
            num_vectors=len(self._hadith_ids),
            embedding_dim=self._dim,
            build_time_s=self._build_time_s,
            index_size_bytes=(
                self._pq_codes.nbytes
                + self._pq_codebooks.nbytes
                + self._ivf_centroids.nbytes
            ),
            params={
                "n_clusters": self.n_clusters,
                "nprobe": self.nprobe,
                "subquantizers": self.subquantizers,
                "bits_per_code": self.bits_per_code,
                "metric": self.metric,
            },
        )
