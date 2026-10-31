"""
Embedding generation script.

Loads the Hadith dataset, generates embeddings for all texts, and saves
them to data/processed/embeddings.npz with metadata.

Embeddings are generated ONCE and cached. Subsequent runs load from cache
unless --force is specified.

Usage:
    python scripts/generate_embeddings.py
    python scripts/generate_embeddings.py --force
    python app.py embeddings
    python app.py embeddings --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.config import load_config
from src.datasets.json_loader import JSONHadithDataset
from src.embeddings.cache import EmbeddingCache, CacheMismatchError
from src.embeddings.factory import EmbeddingFactory
from src.logging.logger import configure_logging, get_logger

configure_logging("INFO")
logger = get_logger(__name__)


def main(force: bool = False) -> None:
    config = load_config(project_root=str(project_root))

    cache = EmbeddingCache(
        npz_path=config.embeddings_file(),
        meta_path=config.embeddings_meta_file(),
    )

    if not force and cache.exists():
        try:
            _, hadith_ids, meta = cache.load(expected_model=config.embedding.model)
            logger.info(
                "Embeddings already cached: %d vectors, model=%s. Use --force to regenerate.",
                len(hadith_ids), meta.get("model_name")
            )
            print(f"\nEmbeddings already cached ({len(hadith_ids)} Hadiths). Use --force to regenerate.")
            return
        except CacheMismatchError as e:
            logger.warning("%s. Regenerating ...", e)

    # Load dataset
    dataset_path = config.raw_dataset_file()
    if not dataset_path.exists():
        logger.error(
            "Dataset not found at %s. Run 'python app.py prepare' first.",
            dataset_path
        )
        sys.exit(1)

    dataset = JSONHadithDataset(dataset_path)
    dataset.load()
    texts = dataset.get_texts()
    hadith_ids = [h.id for h in dataset.get_all()]

    logger.info("Generating embeddings for %d Hadiths ...", len(texts))
    t0 = time.perf_counter()

    model = EmbeddingFactory.create(config)
    embeddings = model.encode(texts, show_progress=True)

    elapsed_s = time.perf_counter() - t0
    logger.info(
        "Embeddings generated: %d × %d in %.2fs (%.0f/s)",
        len(texts), embeddings.shape[1], elapsed_s, len(texts) / elapsed_s
    )

    # Save
    cache.save(
        embeddings=embeddings,
        hadith_ids=hadith_ids,
        model_info=model.model_info(),
        extra_meta={"n_texts": len(texts), "elapsed_s": round(elapsed_s, 2)},
    )

    print(f"\nEmbeddings saved: {len(texts)} Hadiths, dim={embeddings.shape[1]}")
    print(f"Output: {config.embeddings_file()}")
    print(f"Time: {elapsed_s:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Hadith embeddings")
    parser.add_argument("--force", action="store_true", help="Regenerate even if cache exists")
    args = parser.parse_args()
    main(force=args.force)
