"""Unit tests for the vector search factory."""

import numpy as np
import pytest

from src.config.config import load_config
from src.indexes.factory import VectorSearchFactory, ALL_ALGORITHMS
from src.indexes.brute_force import BruteForceIndex
from src.indexes.hnsw import HNSWIndex
from src.indexes.ivf import IVFIndex
from src.indexes.pq import PQIndex
from src.indexes.ivf_pq import IVFPQIndex


@pytest.fixture
def config(tmp_path):
    return load_config(project_root=str(tmp_path))


class TestVectorSearchFactory:
    def test_create_brute_force(self, config):
        index = VectorSearchFactory.create("brute_force", config)
        assert isinstance(index, BruteForceIndex)

    def test_create_ivf(self, config):
        index = VectorSearchFactory.create("ivf", config)
        assert isinstance(index, IVFIndex)

    def test_create_hnsw(self, config):
        index = VectorSearchFactory.create("hnsw", config)
        assert isinstance(index, HNSWIndex)

    def test_create_pq(self, config):
        index = VectorSearchFactory.create("pq", config)
        assert isinstance(index, PQIndex)

    def test_create_ivf_pq(self, config):
        index = VectorSearchFactory.create("ivf_pq", config)
        assert isinstance(index, IVFPQIndex)

    def test_unknown_algorithm_raises(self, config):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            VectorSearchFactory.create("magic_index", config)

    def test_list_algorithms_contains_all(self, config):
        algorithms = VectorSearchFactory.list_algorithms()
        for algo in ["brute_force", "ivf", "hnsw", "pq", "ivf_pq"]:
            assert algo in algorithms

    def test_register_new_algorithm(self, config):
        from src.indexes.base import VectorSearchIndex, SearchResult, IndexStats

        class DummyIndex(VectorSearchIndex):
            def build(self, e, ids): pass
            def search(self, q, top_k): return SearchResult()
            def save(self, p): pass
            def load(self, p): pass
            def get_stats(self): raise RuntimeError()

        VectorSearchFactory.register("dummy", DummyIndex)
        index = VectorSearchFactory.create("dummy", config)
        assert isinstance(index, DummyIndex)
