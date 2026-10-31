"""
Vector search index factory.

This is the single point where algorithm names are mapped to classes.
All other code uses VectorSearchFactory.create() and never has
algorithm-specific conditional logic (if/elif chains).

Adding a new algorithm:
  1. Implement VectorSearchIndex in a new module.
  2. Add it to _REGISTRY below.
  3. Done.

Usage:
    from src.indexes.factory import VectorSearchFactory
    from src.config.config import load_config

    config = load_config()
    index = VectorSearchFactory.create("hnsw", config)
    index.build(embeddings, hadith_ids)
    result = index.search(query_vector, top_k=10)
"""

from __future__ import annotations

from typing import Dict, Type

from src.config.config import Config
from src.indexes.base import VectorSearchIndex
from src.indexes.brute_force import BruteForceIndex
from src.indexes.hnsw import HNSWIndex
from src.indexes.ivf import IVFIndex
from src.indexes.ivf_pq import IVFPQIndex
from src.indexes.pq import PQIndex
from src.logging.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Registry: algorithm name → class
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, Type[VectorSearchIndex]] = {
    "brute_force": BruteForceIndex,
    "ivf": IVFIndex,
    "hnsw": HNSWIndex,
    "pq": PQIndex,
    "ivf_pq": IVFPQIndex,
}

ALL_ALGORITHMS = list(_REGISTRY.keys())


class VectorSearchFactory:
    """Creates VectorSearchIndex instances from name and configuration."""

    @staticmethod
    def create(name: str, config: Config) -> VectorSearchIndex:
        """
        Instantiate the requested search index with parameters from config.

        Args:
            name:   Algorithm name: 'brute_force', 'ivf', 'hnsw', 'pq', 'ivf_pq'.
            config: Application configuration.

        Returns:
            VectorSearchIndex instance (not yet built).

        Raises:
            ValueError: If name is not in the registry.
        """
        name = name.lower().strip()
        if name not in _REGISTRY:
            available = ", ".join(_REGISTRY.keys())
            raise ValueError(
                f"Unknown algorithm '{name}'. Available: {available}"
            )

        cls = _REGISTRY[name]
        index = VectorSearchFactory._build_with_params(cls, name, config)
        logger.debug("Created %s index", name)
        return index

    @staticmethod
    def _build_with_params(
        cls: Type[VectorSearchIndex],
        name: str,
        config: Config,
    ) -> VectorSearchIndex:
        """Instantiate the class with the correct parameters from config."""
        if name == "brute_force":
            return BruteForceIndex(metric=config.search.metric)

        elif name == "ivf":
            return IVFIndex(
                n_clusters=config.ivf.n_clusters,
                nprobe=config.ivf.nprobe,
                metric=config.search.metric,
            )

        elif name == "hnsw":
            return HNSWIndex(
                M=config.hnsw.M,
                ef_construction=config.hnsw.ef_construction,
                ef_search=config.hnsw.ef_search,
                metric=config.search.metric,
            )

        elif name == "pq":
            return PQIndex(
                subquantizers=config.pq.subquantizers,
                bits_per_code=config.pq.bits_per_code,
                metric=config.search.metric,
            )

        elif name == "ivf_pq":
            return IVFPQIndex(
                n_clusters=config.ivf_pq.n_clusters,
                nprobe=config.ivf_pq.nprobe,
                subquantizers=config.ivf_pq.subquantizers,
                bits_per_code=config.ivf_pq.bits_per_code,
                metric=config.search.metric,
            )

        else:
            return cls()

    @staticmethod
    def list_algorithms() -> list[str]:
        """Return all registered algorithm names."""
        return list(_REGISTRY.keys())

    @staticmethod
    def register(name: str, cls: Type[VectorSearchIndex]) -> None:
        """
        Register a new algorithm at runtime.

        Args:
            name: Algorithm identifier string.
            cls:  VectorSearchIndex subclass.
        """
        if not issubclass(cls, VectorSearchIndex):
            raise TypeError(f"{cls} must be a subclass of VectorSearchIndex")
        _REGISTRY[name] = cls
        logger.info("Registered new algorithm: %s", name)
