"""
High-level Hadith search engine.

This is the main public API for the search system. Callers interact only
with HadithSearchEngine and never with specific index classes directly.
The algorithm is selected via config; the caller doesn't need to know which
one is running.

Usage::

    config = load_config()
    engine = HadithSearchEngine(config)
    engine.load_or_build(algorithm="hnsw")

    results = engine.search(
        "حديث عن الأعمال والنية",
        top_k=10,
    )
    for r in results.results:
        print(r.rank, r.hadith_id, r.score, r.text[:80])
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

from src.config.config import Config, load_config
from src.datasets.base import HadithDataset
from src.datasets.json_loader import JSONHadithDataset
from src.embeddings.base import EmbeddingModel
from src.embeddings.cache import EmbeddingCache
from src.embeddings.factory import EmbeddingFactory
from src.indexes.base import HadithResult, SearchResult
from src.indexes.factory import VectorSearchFactory
from src.logging.logger import get_logger

logger = get_logger(__name__)


class HadithSearchEngine:
    """
    End-to-end Hadith retrieval engine.

    Responsibilities:
      - Load and cache embeddings.
      - Build or load a vector search index.
      - Encode queries and run searches.
      - Attach text/metadata to results from the dataset.

    The algorithm used (brute_force, ivf, hnsw, pq, ivf_pq) is determined
    by config or by the algorithm argument to load_or_build().

    Args:
        config: Application configuration object.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._embedding_model: Optional[EmbeddingModel] = None
        self._index = None
        self._dataset: Optional[HadithDataset] = None
        self._hadith_ids: List[str] = []
        self._algorithm: str = config.search.algorithm

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _get_embedding_model(self) -> EmbeddingModel:
        if self._embedding_model is None:
            self._embedding_model = EmbeddingFactory.create(self.config)
        return self._embedding_model

    def _get_dataset(self) -> HadithDataset:
        if self._dataset is None:
            dataset_path = self.config.raw_dataset_file()
            self._dataset = JSONHadithDataset(dataset_path)
            self._dataset.load()
        return self._dataset

    def load_embeddings(self) -> tuple:
        """
        Load cached embeddings from disk.

        Returns:
            Tuple (embeddings: np.ndarray, hadith_ids: list[str]).

        Raises:
            FileNotFoundError: If embeddings cache doesn't exist.
        """
        cache = EmbeddingCache(
            npz_path=self.config.embeddings_file(),
            meta_path=self.config.embeddings_meta_file(),
        )
        embeddings, hadith_ids, _ = cache.load(
            expected_model=self.config.embedding.model
        )
        self._hadith_ids = hadith_ids
        return embeddings, hadith_ids

    def load_or_build(self, algorithm: Optional[str] = None) -> None:
        """
        Load an existing index or build a new one from cached embeddings.

        If the index files exist on disk, they are loaded.
        Otherwise, embeddings are loaded and the index is built and saved.

        Args:
            algorithm: Algorithm name override. Defaults to config.search.algorithm.
        """
        if algorithm:
            self._algorithm = algorithm

        index = VectorSearchFactory.create(self._algorithm, self.config)
        index_path = self.config.index_path(self._algorithm)

        if index_path.exists() and any(index_path.iterdir()):
            logger.info("Loading %s index from %s", self._algorithm, index_path)
            index.load(index_path)
        else:
            logger.info("No saved index found. Building %s index ...", self._algorithm)
            embeddings, hadith_ids = self.load_embeddings()
            index.build(embeddings, hadith_ids)
            index.save(index_path)

        self._index = index
        logger.info("%s index ready.", self._algorithm)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> SearchResult:
        """
        Search for the most semantically similar Hadiths.

        Timing breakdown:
          - query_embed_ms: Time to encode the query text.
          - search_ms:      Time for the vector search.
          - postprocess_ms: Time to attach text/metadata from the dataset.
          - total_ms:       End-to-end latency.

        Args:
            query:  Natural language query (Arabic or other language).
            top_k:  Number of results. Defaults to config.search.top_k.

        Returns:
            SearchResult with ranked results and timing metrics.

        Raises:
            RuntimeError: If the index has not been loaded or built.
        """
        if self._index is None:
            raise RuntimeError(
                "Search engine not ready. Call load_or_build() first."
            )

        if top_k is None:
            top_k = self.config.search.top_k

        t_total_start = time.perf_counter()

        # Step 1: Encode query
        t0 = time.perf_counter()
        model = self._get_embedding_model()
        query_vector = model.encode_query(query)
        embed_ms = (time.perf_counter() - t0) * 1000.0

        # Step 2: Vector search
        result = self._index.search(query_vector, top_k=top_k)
        result.query = query

        # Step 3: Attach text/metadata
        t0 = time.perf_counter()
        self._attach_metadata(result)
        postprocess_ms = (time.perf_counter() - t0) * 1000.0

        total_ms = (time.perf_counter() - t_total_start) * 1000.0

        result.metrics["query_embed_ms"] = round(embed_ms, 3)
        result.metrics["postprocess_ms"] = round(postprocess_ms, 3)
        result.metrics["total_time_ms"] = round(total_ms, 3)

        logger.info(
            "Search completed in %.2f ms (embed=%.2f ms, search=%.2f ms) | "
            "algorithm=%s top_k=%d",
            total_ms,
            embed_ms,
            result.metrics.get("search_time_ms", 0.0),
            self._algorithm,
            top_k,
        )

        return result

    def _attach_metadata(self, result: SearchResult) -> None:
        """
        Populate text and metadata fields in each HadithResult.

        If the dataset is not loaded, results are returned with IDs only.
        """
        try:
            dataset = self._get_dataset()
            ids = [r.hadith_id for r in result.results]
            hadiths = dataset.get_metadata(ids)
            for hadith_result, hadith in zip(result.results, hadiths):
                if hadith is not None:
                    hadith_result.text = hadith.text
                    hadith_result.book = hadith.book
                    hadith_result.chapter = hadith.chapter
                    hadith_result.narrator = hadith.narrator
                    hadith_result.source = hadith.source
                    hadith_result.reference = hadith.reference
        except Exception as e:
            logger.warning("Could not attach metadata: %s", e)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return index statistics as a dict."""
        if self._index is None:
            raise RuntimeError("No index loaded.")
        return self._index.get_stats().to_dict()
