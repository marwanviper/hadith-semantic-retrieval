"""
Embedding cache — persist and reload precomputed embeddings.

Stores two files:
  embeddings.npz         — compressed NumPy array (float32, shape N×D)
  embeddings_meta.json   — model info, dimension, count, timestamp, preprocessing config

On load, the metadata is validated against the current model config.
If the model has changed, a CacheMismatchError is raised so the user
knows to regenerate embeddings rather than silently getting wrong results.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.logging.logger import get_logger

logger = get_logger(__name__)


class CacheMismatchError(Exception):
    """
    Raised when cached embeddings were produced by a different model
    than the one currently configured.
    """


class EmbeddingCache:
    """
    Save and load precomputed Hadith embeddings.

    Args:
        npz_path:  Path to the .npz file for the embedding matrix.
        meta_path: Path to the JSON metadata file.
    """

    def __init__(self, npz_path: Path, meta_path: Path) -> None:
        self.npz_path = Path(npz_path)
        self.meta_path = Path(meta_path)

    def exists(self) -> bool:
        """Return True if both the embeddings and metadata files exist."""
        return self.npz_path.exists() and self.meta_path.exists()

    def save(
        self,
        embeddings: np.ndarray,
        hadith_ids: List[str],
        model_info: Dict[str, str],
        extra_meta: Optional[Dict] = None,
    ) -> None:
        """
        Persist embeddings to disk.

        Args:
            embeddings:  Float32 array of shape (N, D).
            hadith_ids:  List of Hadith IDs in row order.
            model_info:  Dict from EmbeddingModel.model_info().
            extra_meta:  Any additional metadata to store.
        """
        self.npz_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(str(self.npz_path), embeddings=embeddings)

        meta = {
            "model_name": model_info.get("model_name", "unknown"),
            "embedding_dim": int(embeddings.shape[1]),
            "n_hadiths": int(embeddings.shape[0]),
            "hadith_ids": hadith_ids,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dtype": str(embeddings.dtype),
            **model_info,
        }
        if extra_meta:
            meta.update(extra_meta)

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        size_mb = self.npz_path.stat().st_size / 1024 / 1024
        logger.info(
            "Saved %d embeddings (dim=%d) to %s (%.1f MB)",
            embeddings.shape[0], embeddings.shape[1], self.npz_path, size_mb
        )

    def load(
        self,
        expected_model: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[str], Dict]:
        """
        Load embeddings from disk.

        Args:
            expected_model: If provided, raises CacheMismatchError if the
                            cached model name does not match.

        Returns:
            Tuple of (embeddings array, hadith_ids list, metadata dict).

        Raises:
            FileNotFoundError:  If cache files are missing.
            CacheMismatchError: If model name doesn't match.
        """
        if not self.exists():
            raise FileNotFoundError(
                f"Embedding cache not found at {self.npz_path}. "
                "Run 'python app.py embeddings' first."
            )

        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        cached_model = meta.get("model_name", "")
        if expected_model and cached_model != expected_model:
            raise CacheMismatchError(
                f"Cached embeddings were generated with model '{cached_model}' "
                f"but current config uses '{expected_model}'. "
                "Delete the cache and re-run 'python app.py embeddings'."
            )

        data = np.load(str(self.npz_path))
        embeddings = data["embeddings"].astype(np.float32)
        hadith_ids = meta.get("hadith_ids", [str(i) for i in range(len(embeddings))])

        logger.info(
            "Loaded %d embeddings (dim=%d) from %s | model=%s",
            embeddings.shape[0], embeddings.shape[1], self.npz_path, cached_model
        )
        return embeddings, hadith_ids, meta
