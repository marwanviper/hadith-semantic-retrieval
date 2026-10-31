"""
Product Quantization (PQ) vector index.

Algorithm overview:
  Product Quantization compresses high-dimensional vectors into compact
  byte codes by:

  1. TRAINING (codebook learning):
     a. Split each D-dimensional vector into M equal subspaces of size D/M.
     b. Run K-Means (K=2^bits centroids) independently on each subspace.
     c. Result: M codebooks, each with K centroids of size D/M.

  2. ENCODING (quantization):
     For each database vector:
       For each subspace m:
         Find the nearest centroid in codebook m.
         Store its index (0…K-1) as one byte.
     Result: each vector → M bytes instead of D×4 bytes.

  3. SEARCH (Asymmetric Distance Computation, ADC):
     For a query vector:
       a. Split query into M subspaces.
       b. For each subspace m, compute distance from query subspace
          to all K centroids in codebook m.
          This gives an M×K distance table.
       c. For each database code (M bytes), look up its approximate
          distance to the query using the precomputed table.
          Total distance = sum of M table lookups.
       d. Return top-K by approximate distance.

Key parameters:
  subquantizers (M): Number of subspaces.
    - Must divide embedding_dim evenly.
    - Higher M = better recall but larger index.
    - Typical: M=8 for 384-dim (D/M=48), M=16 for 768-dim (D/M=48).

  bits_per_code: Bits per subspace centroid index.
    - 8 bits = 256 centroids per subspace (standard PQ8).
    - Higher bits = better quality, more memory for codebook.

Storage:
  Uncompressed: N×D×4 bytes (float32)
  PQ-compressed: N×M bytes
  Compression ratio: D/M×4  (e.g., 384-dim, M=8 → 384/8*4=192x smaller)

Android portability:
  Yes — codebooks and byte codes are plain arrays.
  ADC lookup is simple table indexing, easily implemented in Kotlin/Java.
  This makes PQ an excellent choice for mobile deployment with constrained
  storage and RAM.
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


class PQIndex(VectorSearchIndex):
    """
    Searchable Product Quantization index.

    This is NOT just a compression wrapper — it implements the full
    Asymmetric Distance Computation (ADC) search pipeline.

    Args:
        subquantizers: Number of subspaces (M). Must divide embedding_dim.
        bits_per_code:  Bits per centroid index. 8 → 256 centroids.
        metric:         'cosine' or 'l2'.
        random_seed:    K-Means reproducibility seed.
    """

    ALGORITHM = "pq"
    INDEX_FILE = "pq_index.pkl"

    def __init__(
        self,
        subquantizers: int = 8,
        bits_per_code: int = 8,
        metric: str = "cosine",
        random_seed: int = 42,
    ) -> None:
        self.subquantizers = subquantizers
        self.bits_per_code = bits_per_code
        self.metric = metric
        self.random_seed = random_seed

        self._codebooks: np.ndarray | None = None  # (M, K, sub_dim)
        self._codes: np.ndarray | None = None      # (N, M) uint8
        self._hadith_ids: List[str] = []
        self._dim: int | None = None
        self._n_centroids: int = 0
        self._sub_dim: int = 0
        self._build_time_s: float = 0.0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, embeddings: np.ndarray, hadith_ids: List[str]) -> None:
        """
        Train codebooks and encode all vectors.

        Steps:
          1. Verify dimension is divisible by subquantizers.
          2. Train K-Means per subspace.
          3. Encode all vectors into byte codes.
        """
        from sklearn.cluster import MiniBatchKMeans

        t0 = time.perf_counter()

        n_vecs, dim = embeddings.shape
        self._dim = dim

        if dim % self.subquantizers != 0:
            raise ValueError(
                f"Embedding dim={dim} is not divisible by "
                f"subquantizers={self.subquantizers}. "
                f"Try subquantizers={self._suggest_divisor(dim)}."
            )

        self._sub_dim = dim // self.subquantizers
        self._n_centroids = min(2 ** self.bits_per_code, n_vecs)

        logger.info(
            "PQ: Training %d codebooks (×%d centroids, sub_dim=%d) on %d vectors …",
            self.subquantizers, self._n_centroids, self._sub_dim, n_vecs
        )

        codebooks = np.zeros(
            (self.subquantizers, self._n_centroids, self._sub_dim), dtype=np.float32
        )
        codes = np.zeros((n_vecs, self.subquantizers), dtype=np.uint8)

        for m in range(self.subquantizers):
            start_col = m * self._sub_dim
            end_col = start_col + self._sub_dim
            sub_vecs = embeddings[:, start_col:end_col].astype(np.float32)

            kmeans = MiniBatchKMeans(
                n_clusters=self._n_centroids,
                random_state=self.random_seed + m,
                n_init=3,
                max_iter=100,
            )
            assignments = kmeans.fit_predict(sub_vecs)  # (N,)
            codebooks[m] = kmeans.cluster_centers_.astype(np.float32)
            codes[:, m] = assignments.astype(np.uint8)

            logger.debug("PQ subspace %d/%d trained.", m + 1, self.subquantizers)

        self._codebooks = codebooks
        self._codes = codes
        self._hadith_ids = list(hadith_ids)
        self._build_time_s = time.perf_counter() - t0

        raw_size_mb = n_vecs * dim * 4 / 1024 / 1024
        pq_size_mb = n_vecs * self.subquantizers / 1024 / 1024
        compression_ratio = raw_size_mb / max(pq_size_mb, 1e-9)
        logger.info(
            "PQ index built in %.2fs | Raw: %.1f MB -> PQ: %.1f MB (%.0fx compression)",
            self._build_time_s, raw_size_mb, pq_size_mb, compression_ratio
        )

    @staticmethod
    def _suggest_divisor(dim: int) -> int:
        """Find a reasonable divisor of dim close to 8."""
        for candidate in [8, 4, 12, 16, 6, 2]:
            if dim % candidate == 0:
                return candidate
        return 1

    # ------------------------------------------------------------------
    # Search (Asymmetric Distance Computation)
    # ------------------------------------------------------------------

    def search(self, query_vector: np.ndarray, top_k: int) -> SearchResult:
        """
        ADC search:
          1. Compute distance table: for each subspace m and each centroid k,
             compute dist(query_subvec_m, codebook[m, k]).
          2. For each database code, sum the M table lookups.
          3. Return top-K by total distance.

        This is approximate because each vector is represented by its
        nearest-centroid approximation, not its exact float values.
        """
        if self._codes is None:
            raise RuntimeError("PQ index not built. Call build() or load() first.")

        t_start = time.perf_counter()

        q = query_vector.astype(np.float32)

        # Step 1: Precompute distance table (M, K)
        # For cosine: use dot product (vectors are L2-normalized)
        dist_table = np.zeros((self.subquantizers, self._n_centroids), dtype=np.float32)
        for m in range(self.subquantizers):
            start_col = m * self._sub_dim
            end_col = start_col + self._sub_dim
            q_sub = q[start_col:end_col]  # (sub_dim,)
            # dot product: shape (K,)
            dist_table[m] = self._codebooks[m] @ q_sub

        # Step 2: ADC lookup for all N codes
        # self._codes is (N, M) uint8
        # For each row i, score[i] = sum_m dist_table[m, codes[i, m]]
        scores = np.zeros(len(self._codes), dtype=np.float32)
        for m in range(self.subquantizers):
            scores += dist_table[m][self._codes[:, m].astype(np.int32)]

        # Step 3: top-K
        actual_k = min(top_k, len(scores))
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
                "n_candidates_examined": len(self._codes),
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
            "codebooks": self._codebooks,
            "codes": self._codes,
            "hadith_ids": self._hadith_ids,
            "subquantizers": self.subquantizers,
            "bits_per_code": self.bits_per_code,
            "metric": self.metric,
            "dim": self._dim,
            "n_centroids": self._n_centroids,
            "sub_dim": self._sub_dim,
            "build_time_s": self._build_time_s,
        }
        with open(path / self.INDEX_FILE, "wb") as f:
            pickle.dump(state, f, protocol=4)
        size_mb = (path / self.INDEX_FILE).stat().st_size / 1024 / 1024
        logger.info("PQ index saved to %s (%.1f MB)", path, size_mb)

    def load(self, path: Path) -> None:
        path = Path(path)
        index_file = path / self.INDEX_FILE
        if not index_file.exists():
            raise FileNotFoundError(
                f"PQ index not found at {index_file}. "
                "Run 'python app.py build --algorithm pq' first."
            )
        with open(index_file, "rb") as f:
            state = pickle.load(f)
        self._codebooks = state["codebooks"]
        self._codes = state["codes"]
        self._hadith_ids = state["hadith_ids"]
        self.subquantizers = state["subquantizers"]
        self.bits_per_code = state["bits_per_code"]
        self.metric = state["metric"]
        self._dim = state["dim"]
        self._n_centroids = state["n_centroids"]
        self._sub_dim = state["sub_dim"]
        self._build_time_s = state["build_time_s"]
        logger.info(
            "PQ index loaded: %d vectors, M=%d, K=%d",
            len(self._hadith_ids), self.subquantizers, self._n_centroids
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> IndexStats:
        if self._codes is None:
            raise RuntimeError("PQ index not built.")
        codes_size = self._codes.nbytes
        codebook_size = self._codebooks.nbytes
        return IndexStats(
            algorithm=self.ALGORITHM,
            num_vectors=len(self._hadith_ids),
            embedding_dim=self._dim,
            build_time_s=self._build_time_s,
            index_size_bytes=codes_size + codebook_size,
            params={
                "subquantizers": self.subquantizers,
                "bits_per_code": self.bits_per_code,
                "n_centroids": self._n_centroids,
                "metric": self.metric,
            },
        )
