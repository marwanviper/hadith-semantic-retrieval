"""
Centralized configuration management.

Loads configuration from a YAML file and exposes it as typed dataclasses.
CLI overrides are applied on top of the file configuration.

Design decisions:
  - Single source of truth: Config object is created once and passed around.
  - No global state: callers receive a Config instance explicitly.
  - Simple > clever: we use plain dataclasses rather than pydantic to keep
    dependencies minimal and Android-portability clear.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingConfig:
    model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: int = 32
    normalize_embeddings: bool = True


@dataclass
class SearchConfig:
    algorithm: str = "hnsw"
    top_k: int = 10
    metric: str = "cosine"


@dataclass
class HNSWConfig:
    M: int = 16
    ef_construction: int = 200
    ef_search: int = 64


@dataclass
class IVFConfig:
    n_clusters: int = 256
    nprobe: int = 8


@dataclass
class PQConfig:
    subquantizers: int = 8
    bits_per_code: int = 8


@dataclass
class IVFPQConfig:
    n_clusters: int = 64
    nprobe: int = 4
    subquantizers: int = 8
    bits_per_code: int = 8


@dataclass
class PathsConfig:
    raw_data: str = "data/raw"
    processed_data: str = "data/processed"
    indexes: str = "data/indexes"
    benchmark_results: str = "data/benchmark_results"
    models: str = "models"


@dataclass
class BenchmarkConfig:
    n_queries: int = 20
    top_k: int = 10
    random_seed: int = 42
    warmup_queries: int = 3


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "{asctime} | {levelname:<8} | {name} | {message}"
    datefmt: str = "%Y-%m-%d %H:%M:%S"


@dataclass
class Config:
    """Root configuration object. Pass this to all components."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    hnsw: HNSWConfig = field(default_factory=HNSWConfig)
    ivf: IVFConfig = field(default_factory=IVFConfig)
    pq: PQConfig = field(default_factory=PQConfig)
    ivf_pq: IVFPQConfig = field(default_factory=IVFPQConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    # Root directory of the project (set at load time)
    project_root: str = "."

    def resolve_path(self, relative: str) -> Path:
        """Resolve a path relative to project_root."""
        return Path(self.project_root) / relative

    def index_path(self, algorithm: str) -> Path:
        """Return the directory where an algorithm's index is stored."""
        return self.resolve_path(self.paths.indexes) / algorithm

    def embeddings_file(self) -> Path:
        """Return the path of the cached embeddings .npz file."""
        return self.resolve_path(self.paths.processed_data) / "embeddings.npz"

    def embeddings_meta_file(self) -> Path:
        """Return the path of the cached embeddings metadata JSON."""
        return self.resolve_path(self.paths.processed_data) / "embeddings_meta.json"

    def raw_dataset_file(self) -> Path:
        """Return the expected location of the raw JSON dataset."""
        return self.resolve_path(self.paths.raw_data) / "hadiths.json"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _dict_to_dataclass(cls, data: dict[str, Any]) -> Any:
    """Shallow recursive conversion from dict to dataclass."""
    if not isinstance(data, dict):
        return data
    kwargs = {}
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(cls)}
    for key, value in data.items():
        if key not in field_names:
            continue
        field_type = cls.__dataclass_fields__[key].type
        # Resolve string annotations
        if isinstance(field_type, str):
            import typing
            field_type = eval(field_type, {**vars(typing), **globals()})
        if dataclasses.is_dataclass(field_type) and isinstance(value, dict):
            kwargs[key] = _dict_to_dataclass(field_type, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[dict[str, Any]] = None,
    project_root: Optional[str] = None,
) -> Config:
    """
    Load configuration from YAML file with optional overrides.

    Args:
        config_path: Path to YAML config file. Defaults to 'configs/default.yaml'
                     relative to project_root.
        overrides: Dict of dotted-key overrides, e.g. {'search.algorithm': 'ivf'}.
        project_root: Root directory of the project. Defaults to cwd.

    Returns:
        Config: Fully populated configuration object.
    """
    if project_root is None:
        project_root = os.getcwd()

    if config_path is None:
        config_path = str(Path(project_root) / "configs" / "default.yaml")

    raw: dict[str, Any] = {}
    if Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    # Apply overrides (dotted keys like 'search.algorithm')
    if overrides:
        for dotted_key, value in overrides.items():
            parts = dotted_key.split(".")
            d = raw
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value

    config = _dict_to_dataclass(Config, raw)
    config.project_root = project_root
    return config
