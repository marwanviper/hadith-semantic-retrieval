"""
Unit tests for all vector search indexes.

Uses small synthetic embeddings (100 vectors, dim=8) to verify:
  - Build succeeds
  - Search returns results in correct shape
  - Save/load round-trip works
  - All indexes implement the same interface
"""

import numpy as np
import pytest
from pathlib import Path

from src.indexes.brute_force import BruteForceIndex
from src.indexes.ivf import IVFIndex
from src.indexes.hnsw import HNSWIndex
from src.indexes.pq import PQIndex
from src.indexes.ivf_pq import IVFPQIndex
from src.indexes.base import VectorSearchIndex, SearchResult


N = 100   # Number of test vectors
D = 16    # Must be divisible by subquantizers (8) and by 2 for PQ
K = 10    # top_k


@pytest.fixture
def synthetic_data():
    """Normalized random embeddings."""
    rng = np.random.default_rng(42)
    embeddings = rng.random((N, D)).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    ids = [str(i) for i in range(N)]
    return embeddings, ids


@pytest.fixture
def query_vector(synthetic_data):
    embeddings, _ = synthetic_data
    return embeddings[0]  # Use first embedding as query


def _assert_valid_result(result: SearchResult, top_k: int):
    """Common assertions for search results."""
    assert isinstance(result, SearchResult)
    assert len(result.results) == top_k
    for r in result.results:
        assert r.rank >= 1
        assert r.hadith_id is not None
        assert isinstance(r.score, float)
    # Ranks should be sequential
    ranks = [r.rank for r in result.results]
    assert ranks == list(range(1, top_k + 1))


class TestBruteForceIndex:
    def test_build_and_search(self, synthetic_data, query_vector):
        embeddings, ids = synthetic_data
        index = BruteForceIndex()
        index.build(embeddings, ids)
        result = index.search(query_vector, top_k=K)
        _assert_valid_result(result, K)

    def test_first_result_is_query(self, synthetic_data, query_vector):
        """Brute force should find the query vector itself as rank 1."""
        embeddings, ids = synthetic_data
        index = BruteForceIndex()
        index.build(embeddings, ids)
        result = index.search(query_vector, top_k=1)
        assert result.results[0].hadith_id == "0"
        assert result.results[0].score > 0.99

    def test_save_load_roundtrip(self, synthetic_data, query_vector, tmp_path):
        embeddings, ids = synthetic_data
        index = BruteForceIndex()
        index.build(embeddings, ids)
        index.save(tmp_path / "bf")

        index2 = BruteForceIndex()
        index2.load(tmp_path / "bf")
        result = index2.search(query_vector, top_k=K)
        _assert_valid_result(result, K)

    def test_get_stats(self, synthetic_data):
        embeddings, ids = synthetic_data
        index = BruteForceIndex()
        index.build(embeddings, ids)
        stats = index.get_stats()
        assert stats.num_vectors == N
        assert stats.embedding_dim == D
        assert stats.algorithm == "brute_force"

    def test_not_built_raises(self):
        index = BruteForceIndex()
        with pytest.raises(RuntimeError):
            index.search(np.zeros(D, dtype=np.float32), top_k=1)


class TestIVFIndex:
    def test_build_and_search(self, synthetic_data, query_vector):
        embeddings, ids = synthetic_data
        index = IVFIndex(n_clusters=10, nprobe=3)
        index.build(embeddings, ids)
        result = index.search(query_vector, top_k=5)
        assert len(result.results) == 5

    def test_save_load_roundtrip(self, synthetic_data, query_vector, tmp_path):
        embeddings, ids = synthetic_data
        index = IVFIndex(n_clusters=10, nprobe=3)
        index.build(embeddings, ids)
        index.save(tmp_path / "ivf")

        index2 = IVFIndex()
        index2.load(tmp_path / "ivf")
        result = index2.search(query_vector, top_k=5)
        assert len(result.results) == 5

    def test_get_stats(self, synthetic_data):
        embeddings, ids = synthetic_data
        index = IVFIndex(n_clusters=10, nprobe=3)
        index.build(embeddings, ids)
        stats = index.get_stats()
        assert stats.num_vectors == N
        assert stats.algorithm == "ivf"


class TestHNSWIndex:
    def test_build_and_search(self, synthetic_data, query_vector):
        embeddings, ids = synthetic_data
        index = HNSWIndex(M=8, ef_construction=50, ef_search=20)
        index.build(embeddings, ids)
        result = index.search(query_vector, top_k=5)
        assert len(result.results) == 5

    def test_save_load_roundtrip(self, synthetic_data, query_vector, tmp_path):
        embeddings, ids = synthetic_data
        index = HNSWIndex(M=8, ef_construction=50, ef_search=20)
        index.build(embeddings, ids)
        index.save(tmp_path / "hnsw")

        index2 = HNSWIndex()
        index2.load(tmp_path / "hnsw")
        result = index2.search(query_vector, top_k=5)
        assert len(result.results) == 5

    def test_get_stats(self, synthetic_data):
        embeddings, ids = synthetic_data
        index = HNSWIndex(M=8, ef_construction=50)
        index.build(embeddings, ids)
        stats = index.get_stats()
        assert stats.num_vectors == N
        assert stats.algorithm == "hnsw"


class TestPQIndex:
    def test_build_and_search(self, synthetic_data, query_vector):
        embeddings, ids = synthetic_data
        # D=16, subquantizers=4 -> sub_dim=4, n_centroids=min(256, 100)=100
        index = PQIndex(subquantizers=4, bits_per_code=4)
        index.build(embeddings, ids)
        result = index.search(query_vector, top_k=5)
        assert len(result.results) == 5

    def test_save_load_roundtrip(self, synthetic_data, query_vector, tmp_path):
        embeddings, ids = synthetic_data
        index = PQIndex(subquantizers=4, bits_per_code=4)
        index.build(embeddings, ids)
        index.save(tmp_path / "pq")

        index2 = PQIndex()
        index2.load(tmp_path / "pq")
        result = index2.search(query_vector, top_k=5)
        assert len(result.results) == 5

    def test_invalid_subquantizers_raises(self, synthetic_data):
        embeddings, ids = synthetic_data
        index = PQIndex(subquantizers=7)  # 16 % 7 != 0
        with pytest.raises(ValueError):
            index.build(embeddings, ids)

    def test_get_stats(self, synthetic_data):
        embeddings, ids = synthetic_data
        index = PQIndex(subquantizers=4, bits_per_code=4)
        index.build(embeddings, ids)
        stats = index.get_stats()
        assert stats.num_vectors == N
        assert stats.algorithm == "pq"


class TestIVFPQIndex:
    def test_build_and_search(self, synthetic_data, query_vector):
        embeddings, ids = synthetic_data
        index = IVFPQIndex(n_clusters=10, nprobe=3, subquantizers=4, bits_per_code=4)
        index.build(embeddings, ids)
        result = index.search(query_vector, top_k=5)
        assert len(result.results) == 5

    def test_save_load_roundtrip(self, synthetic_data, query_vector, tmp_path):
        embeddings, ids = synthetic_data
        index = IVFPQIndex(n_clusters=10, nprobe=3, subquantizers=4, bits_per_code=4)
        index.build(embeddings, ids)
        index.save(tmp_path / "ivfpq")

        index2 = IVFPQIndex()
        index2.load(tmp_path / "ivfpq")
        result = index2.search(query_vector, top_k=5)
        assert len(result.results) == 5

    def test_get_stats(self, synthetic_data):
        embeddings, ids = synthetic_data
        index = IVFPQIndex(n_clusters=10, nprobe=3, subquantizers=4, bits_per_code=4)
        index.build(embeddings, ids)
        stats = index.get_stats()
        assert stats.num_vectors == N
        assert stats.algorithm == "ivf_pq"


class TestAllIndexesSameInterface:
    """Verify all indexes are interchangeable through VectorSearchIndex."""

    def _make_indexes(self):
        return [
            BruteForceIndex(),
            IVFIndex(n_clusters=10, nprobe=3),
            HNSWIndex(M=8, ef_construction=50, ef_search=20),
            PQIndex(subquantizers=4, bits_per_code=4),
            IVFPQIndex(n_clusters=10, nprobe=3, subquantizers=4, bits_per_code=4),
        ]

    def test_all_implement_interface(self):
        for index in self._make_indexes():
            assert isinstance(index, VectorSearchIndex)

    def test_all_return_search_result(self, synthetic_data, query_vector):
        embeddings, ids = synthetic_data
        for index in self._make_indexes():
            index.build(embeddings, ids)
            result = index.search(query_vector, top_k=5)
            assert isinstance(result, SearchResult), f"{type(index).__name__} failed"
            assert len(result.results) == 5
