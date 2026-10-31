"""
Sentence-Transformers embedding model wrapper.

Default model: paraphrase-multilingual-MiniLM-L12-v2
  - 50+ languages including Arabic
  - ~120 MB on disk
  - 384-dimensional embeddings
  - Good recall quality vs size tradeoff for mobile deployment

Model is downloaded automatically on first use and cached in models/.
The model name comes from Config, so it can be changed without touching
any search algorithm code.

Android portability:
  Export to ONNX with:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    model.save_to_hub(...)  # or export via optimum/onnxruntime
  Then use ONNX Runtime Mobile on Android with the same tokenizer.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.embeddings.base import EmbeddingModel
from src.logging.logger import get_logger

logger = get_logger(__name__)


class SentenceTransformerModel(EmbeddingModel):
    """
    Embedding model backed by the sentence-transformers library.

    Args:
        model_name:            HuggingFace model ID or local path.
        normalize_embeddings:  If True, L2-normalize all embeddings.
                               Allows cosine similarity = dot product.
        batch_size:            Batch size for bulk encoding.
        cache_dir:             Directory to cache downloaded models.
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        cache_dir: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._model = None
        self._dim: int | None = None

    def _load_model(self) -> None:
        """Lazy-load model on first use."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required. "
                "Install it with: pip install sentence-transformers"
            ) from e

        logger.info("Loading embedding model: %s", self.model_name)
        kwargs: dict = {}
        if self.cache_dir:
            kwargs["cache_folder"] = self.cache_dir
        self._model = SentenceTransformer(self.model_name, **kwargs)
        # Probe dimension with a dummy encode
        dummy = self._model.encode(["test"], normalize_embeddings=False)
        self._dim = dummy.shape[1]
        logger.info(
            "Model loaded: %s | embedding_dim=%d",
            self.model_name, self._dim
        )

    def encode(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode texts in batches.

        Returns:
            Float32 ndarray of shape (N, D).
        """
        self._load_model()
        logger.debug(
            "Encoding %d texts (batch_size=%d, normalize=%s)",
            len(texts), self.batch_size, self.normalize_embeddings
        )
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        """
        Encode a single query string.

        For paraphrase/symmetric models, query and document encoding
        are identical. Override this for asymmetric retrieval models.

        Returns:
            Float32 ndarray of shape (D,).
        """
        self._load_model()
        result = self._model.encode(
            [text],
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return result[0].astype(np.float32)

    def model_info(self) -> Dict[str, str]:
        self._load_model()
        return {
            "model_name": self.model_name,
            "embedding_dim": str(self._dim),
            "normalize": str(self.normalize_embeddings),
            "backend": "sentence-transformers",
        }

    @property
    def embedding_dim(self) -> int:
        self._load_model()
        return self._dim
