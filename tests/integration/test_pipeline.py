"""
Integration test: full pipeline from data loading to search results.

Tests the complete flow:
  synthetic data -> embeddings -> index -> search -> results

This verifies that all components connect correctly and the search engine
API works end-to-end for each algorithm.

Note: Uses synthetic embeddings, not a real model, to avoid network access.
"""

from __future__ import annotations

import json
import numpy as np
import pytest
from pathlib import Path
from typing import List

from src.config.config import load_config
from src.datasets.json_loader import JSONHadithDataset
from src.embeddings.cache import EmbeddingCache
from src.indexes.factory import VectorSearchFactory, ALL_ALGORITHMS
from src.indexes.brute_force import BruteForceIndex
from src.indexes.base import SearchResult


N = 50   # Small dataset for fast tests
D = 16   # Embedding dim (divisible by 4 for PQ)


@pytest.fixture
def project_config(tmp_path):
    """Config pointing to tmp_path as project root."""
    return load_config(project_root=str(tmp_path))


@pytest.fixture
def sample_dataset_file(tmp_path) -> Path:
    """Create a small JSON dataset file."""
    hadiths = [
        {"id": str(i), "text": f"نص الحديث رقم {i} في هذا الكتاب الكريم", "book": "البخاري"}
        for i in range(N)
    ]
    path = tmp_path / "data" / "raw" / "hadiths.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hadiths, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def synthetic_embeddings_cache(tmp_path, project_config):
    """Create a fake embeddings cache with synthetic vectors."""
    rng = np.random.default_rng(42)
    embeddings = rng.random((N, D)).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= norms
    ids = [str(i) for i in range(N)]

    cache = EmbeddingCache(
        npz_path=project_config.embeddings_file(),
        meta_path=project_config.embeddings_meta_file(),
    )
    cache.save(
        embeddings=embeddings,
        hadith_ids=ids,
        model_info={"model_name": project_config.embedding.model, "embedding_dim": str(D)},
    )
    return embeddings, ids


class TestEmbeddingCacheRoundtrip:
    def test_save_and_load(self, tmp_path, project_config, synthetic_embeddings_cache):
        embeddings, ids = synthetic_embeddings_cache
        cache = EmbeddingCache(
            npz_path=project_config.embeddings_file(),
            meta_path=project_config.embeddings_meta_file(),
        )
        loaded_emb, loaded_ids, meta = cache.load(
            expected_model=project_config.embedding.model
        )
        assert loaded_emb.shape == embeddings.shape
        assert loaded_ids == ids
        assert meta["model_name"] == project_config.embedding.model

    def test_model_mismatch_raises(self, project_config, synthetic_embeddings_cache):
        from src.embeddings.cache import CacheMismatchError
        cache = EmbeddingCache(
            npz_path=project_config.embeddings_file(),
            meta_path=project_config.embeddings_meta_file(),
        )
        with pytest.raises(CacheMismatchError):
            cache.load(expected_model="completely-different-model")


class TestBruteForceIntegration:
    def test_full_search_flow(self, project_config, synthetic_embeddings_cache, tmp_path):
        embeddings, ids = synthetic_embeddings_cache
        index = BruteForceIndex()
        index.build(embeddings, ids)
        index.save(tmp_path / "bf_test")

        index2 = BruteForceIndex()
        index2.load(tmp_path / "bf_test")

        query = embeddings[0]
        result = index2.search(query, top_k=5)
        assert isinstance(result, SearchResult)
        assert len(result.results) == 5
        assert result.results[0].hadith_id == "0"  # Exact match is rank 1


class TestAllAlgorithmsPipeline:
    """Verify build-search-save-load for every algorithm."""

    @pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
    def test_algorithm_pipeline(self, algorithm, project_config, synthetic_embeddings_cache, tmp_path):
        embeddings, ids = synthetic_embeddings_cache

        # Override PQ params for small test data
        overrides = {}
        if algorithm in ("pq", "ivf_pq"):
            overrides["pq.subquantizers"] = 4
            overrides["pq.bits_per_code"] = 4
            overrides["ivf_pq.subquantizers"] = 4
            overrides["ivf_pq.bits_per_code"] = 4
            overrides["ivf_pq.n_clusters"] = 10
        if algorithm == "ivf":
            overrides["ivf.n_clusters"] = 10

        config = load_config(
            project_root=str(project_config.project_root),
            overrides=overrides,
        )

        index = VectorSearchFactory.create(algorithm, config)
        index.build(embeddings, ids)

        index_path = tmp_path / "indexes" / algorithm
        index.save(index_path)

        index2 = VectorSearchFactory.create(algorithm, config)
        index2.load(index_path)

        query = embeddings[5]
        result = index2.search(query, top_k=5)

        assert isinstance(result, SearchResult), f"{algorithm} search failed"
        assert len(result.results) == 5
        assert "search_time_ms" in result.metrics
        assert result.metrics["search_time_ms"] > 0
