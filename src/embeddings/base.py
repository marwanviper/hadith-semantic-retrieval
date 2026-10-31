"""
Abstract embedding model interface.

All embedding models must implement this interface so that the rest of
the codebase never has hard-coded references to a specific model.

Android portability:
  The encode/encode_query logic itself is NOT ported to Android.
  Instead, the offline pipeline exports the model to ONNX, and the
  Android runtime uses ONNX Runtime Mobile to run inference locally.
  The abstract interface here documents the contract that both the
  Python training side and the Android inference side must honour.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np


class EmbeddingModel(ABC):
    """Abstract sentence embedding model."""

    @abstractmethod
    def encode(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of texts into embeddings.

        Args:
            texts:         List of text strings.
            show_progress: Show a tqdm progress bar.

        Returns:
            Float32 array of shape (len(texts), embedding_dim).
        """
        ...

    @abstractmethod
    def encode_query(self, text: str) -> np.ndarray:
        """
        Encode a single query string.

        This may differ from encode() in some models (e.g., asymmetric
        retrieval models use a different prompt for queries vs documents).
        For symmetric models this simply calls encode([text])[0].

        Args:
            text: Query string.

        Returns:
            Float32 array of shape (embedding_dim,).
        """
        ...

    @abstractmethod
    def model_info(self) -> Dict[str, str]:
        """
        Return model metadata for cache validation and logging.

        Returns:
            Dict with at least: model_name, embedding_dim.
        """
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Embedding vector dimension."""
        ...
