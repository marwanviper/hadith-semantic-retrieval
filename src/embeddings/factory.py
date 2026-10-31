"""
Embedding model factory.

Usage:
    from src.embeddings.factory import EmbeddingFactory
    from src.config.config import Config

    config = load_config()
    model = EmbeddingFactory.create(config)
"""

from __future__ import annotations

from typing import Dict, Type

from src.config.config import Config
from src.embeddings.base import EmbeddingModel
from src.embeddings.sentence_transformer import SentenceTransformerModel
from src.logging.logger import get_logger

logger = get_logger(__name__)

# Registry: backend name → class
_REGISTRY: Dict[str, Type[EmbeddingModel]] = {
    "sentence_transformer": SentenceTransformerModel,
}


class EmbeddingFactory:
    """Creates EmbeddingModel instances from configuration."""

    @staticmethod
    def create(config: Config) -> EmbeddingModel:
        """
        Create an EmbeddingModel from config.

        Currently the only backend is sentence-transformers.
        Add new backends by inserting into _REGISTRY.

        Args:
            config: Application configuration.

        Returns:
            EmbeddingModel instance (not yet loaded — models are lazy-loaded).
        """
        model = SentenceTransformerModel(
            model_name=config.embedding.model,
            normalize_embeddings=config.embedding.normalize_embeddings,
            batch_size=config.embedding.batch_size,
        )
        logger.debug("Created embedding model: %s", config.embedding.model)
        return model
