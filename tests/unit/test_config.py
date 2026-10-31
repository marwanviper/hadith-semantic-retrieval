"""Unit tests for configuration loading."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.config.config import load_config, Config


class TestLoadConfig:
    def test_defaults_when_no_file(self, tmp_path):
        """Load config with no YAML file returns sensible defaults."""
        config = load_config(config_path=str(tmp_path / "nonexistent.yaml"), project_root=str(tmp_path))
        assert isinstance(config, Config)
        assert config.embedding.batch_size == 32
        assert config.search.top_k == 10
        assert config.hnsw.M == 16

    def test_yaml_values_loaded(self, tmp_path):
        """Values from YAML file are correctly parsed."""
        yaml_content = {
            "embedding": {"model": "test-model", "batch_size": 64},
            "search": {"algorithm": "ivf", "top_k": 5},
        }
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump(yaml_content))
        config = load_config(config_path=str(config_file), project_root=str(tmp_path))
        assert config.embedding.model == "test-model"
        assert config.embedding.batch_size == 64
        assert config.search.algorithm == "ivf"
        assert config.search.top_k == 5

    def test_overrides_applied(self, tmp_path):
        """CLI overrides take precedence over YAML values."""
        config = load_config(
            config_path=str(tmp_path / "none.yaml"),
            overrides={"search.algorithm": "hnsw", "hnsw.M": 32},
            project_root=str(tmp_path),
        )
        assert config.search.algorithm == "hnsw"
        assert config.hnsw.M == 32

    def test_project_root_set(self, tmp_path):
        """project_root is correctly assigned."""
        config = load_config(project_root=str(tmp_path))
        assert config.project_root == str(tmp_path)

    def test_resolve_path(self, tmp_path):
        """resolve_path joins project_root with relative path."""
        config = load_config(project_root=str(tmp_path))
        resolved = config.resolve_path("data/raw")
        assert resolved == tmp_path / "data" / "raw"

    def test_embeddings_file(self, tmp_path):
        """embeddings_file() returns correct path."""
        config = load_config(project_root=str(tmp_path))
        assert config.embeddings_file().name == "embeddings.npz"
