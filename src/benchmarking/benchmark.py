"""
Benchmarking framework.

Runs each search algorithm against a query set and produces:
  - Timing statistics (avg, P50, P95, P99 latency)
  - Recall@K vs Brute Force ground truth
  - Index build time
  - Index size on disk
  - Structured JSON + CSV output
  - Human-readable summary table

Usage::

    runner = BenchmarkRunner(config)
    runner.run(algorithms=["hnsw", "ivf"], rebuild=False)
    runner.print_summary()
    runner.save_results()
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.benchmarking.metrics import (
    BenchmarkMetrics,
    QueryMetrics,
    aggregate_metrics,
    compute_precision,
    compute_recall,
)
from src.config.config import Config
from src.embeddings.cache import EmbeddingCache
from src.embeddings.factory import EmbeddingFactory
from src.indexes.base import VectorSearchIndex
from src.indexes.brute_force import BruteForceIndex
from src.indexes.factory import VectorSearchFactory, ALL_ALGORITHMS
from src.logging.logger import get_logger

logger = get_logger(__name__)


class BenchmarkRunner:
    """
    Orchestrates benchmarking of multiple search algorithms.

    Args:
        config: Application configuration.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._results: Dict[str, BenchmarkMetrics] = {}
        self._embeddings: Optional[np.ndarray] = None
        self._hadith_ids: Optional[List[str]] = None
        self._query_vectors: Optional[np.ndarray] = None
        self._query_texts: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_embeddings(self) -> None:
        """Load cached embeddings once, reuse for all algorithms."""
        if self._embeddings is not None:
            return
        cache = EmbeddingCache(
            npz_path=self.config.embeddings_file(),
            meta_path=self.config.embeddings_meta_file(),
        )
        self._embeddings, self._hadith_ids, _ = cache.load(
            expected_model=self.config.embedding.model
        )
        logger.info("Benchmark: loaded %d embeddings.", len(self._hadith_ids))

    def _prepare_query_set(self) -> None:
        """
        Select benchmark queries from the dataset.

        Uses a random sample of Hadith embeddings as query vectors.
        This means ground-truth rank 1 will always be the query itself,
        which verifies that the index can retrieve exact matches.
        """
        if self._query_vectors is not None:
            return
        self._load_embeddings()

        rng = np.random.default_rng(self.config.benchmark.random_seed)
        n_queries = min(self.config.benchmark.n_queries, len(self._hadith_ids))
        query_indices = rng.choice(len(self._hadith_ids), size=n_queries, replace=False)

        self._query_vectors = self._embeddings[query_indices]  # (Q, D)
        self._query_texts = [self._hadith_ids[i] for i in query_indices]
        logger.info("Benchmark: prepared %d query vectors.", n_queries)

    # ------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------

    def _compute_ground_truth(
        self, query_vectors: np.ndarray, k: int
    ) -> List[List[str]]:
        """
        Compute exact top-K using Brute Force as ground truth.

        Returns:
            List of lists of top-K Hadith IDs per query.
        """
        logger.info("Computing ground truth with Brute Force ...")
        bf = BruteForceIndex(metric=self.config.search.metric)
        bf.build(self._embeddings, self._hadith_ids)

        ground_truth = []
        for qv in query_vectors:
            result = bf.search(qv, top_k=k)
            ground_truth.append([r.hadith_id for r in result.results])
        return ground_truth

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        algorithms: Optional[List[str]] = None,
        rebuild: bool = False,
    ) -> Dict[str, BenchmarkMetrics]:
        """
        Run benchmarks for the specified algorithms.

        Args:
            algorithms: Algorithm names to benchmark.
                        None or ['all'] → all registered algorithms.
            rebuild:    If True, rebuild indexes even if they exist on disk.

        Returns:
            Dict mapping algorithm name → BenchmarkMetrics.
        """
        if algorithms is None or algorithms == ["all"]:
            algorithms = ALL_ALGORITHMS

        self._prepare_query_set()
        k = self.config.benchmark.top_k
        warmup = self.config.benchmark.warmup_queries

        # Compute ground truth (Brute Force exact search)
        ground_truth = self._compute_ground_truth(self._query_vectors, k)

        for algo in algorithms:
            logger.info("=== Benchmarking: %s ===", algo.upper())
            try:
                metrics = self._benchmark_single(
                    algo, ground_truth, k, warmup, rebuild
                )
                self._results[algo] = metrics
            except Exception as e:
                logger.error("Benchmark failed for %s: %s", algo, e, exc_info=True)

        return self._results

    def _benchmark_single(
        self,
        algorithm: str,
        ground_truth: List[List[str]],
        k: int,
        warmup: int,
        rebuild: bool,
    ) -> BenchmarkMetrics:
        """Benchmark a single algorithm."""
        index = VectorSearchFactory.create(algorithm, self.config)
        index_path = self.config.index_path(algorithm)

        # Build or load
        t_build_start = time.perf_counter()
        if not rebuild and index_path.exists() and any(index_path.iterdir()):
            logger.info("Loading %s index from disk ...", algorithm)
            index.load(index_path)
            build_time_s = 0.0  # Not rebuilt
        else:
            logger.info("Building %s index ...", algorithm)
            index.build(self._embeddings, self._hadith_ids)
            build_time_s = time.perf_counter() - t_build_start
            index.save(index_path)

        stats = index.get_stats()
        if build_time_s == 0.0:
            build_time_s = stats.build_time_s

        # Get index size from disk
        index_size_mb = self._get_index_size_mb(index_path)

        # Warmup queries (excluded from metrics)
        warmup_count = min(warmup, len(self._query_vectors))
        for i in range(warmup_count):
            index.search(self._query_vectors[i], top_k=k)

        # Benchmark queries
        per_query: List[QueryMetrics] = []
        for i, (qv, gt_ids) in enumerate(zip(self._query_vectors, ground_truth)):
            t0 = time.perf_counter()
            result = index.search(qv, top_k=k)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            candidate_ids = [r.hadith_id for r in result.results]
            recall = compute_recall(gt_ids, candidate_ids, k)
            precision = compute_precision(gt_ids, candidate_ids, k)

            per_query.append(QueryMetrics(
                query=self._query_texts[i],
                latency_ms=round(latency_ms, 3),
                recall_at_k=recall,
                precision_at_k=precision,
                n_results=len(result.results),
            ))

        metrics = aggregate_metrics(
            algorithm=algorithm,
            per_query=per_query,
            build_time_s=build_time_s,
            index_size_mb=index_size_mb,
            k=k,
            index_params=stats.params,
        )

        logger.info(
            "%s: avg_latency=%.2f ms | P95=%.2f ms | Recall@%d=%.3f | size=%.1f MB",
            algorithm, metrics.avg_latency_ms, metrics.p95_latency_ms,
            k, metrics.avg_recall_at_k, metrics.index_size_mb
        )
        return metrics

    @staticmethod
    def _get_index_size_mb(path: Path) -> float:
        """Sum size of all files in the index directory."""
        if not path.exists():
            return 0.0
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return round(total / 1024 / 1024, 2)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Print a formatted comparison table to stdout."""
        if not self._results:
            print("No benchmark results available.")
            return

        header = f"{'Algorithm':<15} {'Build':>10} {'Avg ms':>9} {'P50 ms':>9} {'P95 ms':>9} {'Recall@K':>10} {'Size MB':>10}"
        separator = "-" * len(header)
        print()
        print("=" * len(header))
        print(" BENCHMARK RESULTS ")
        print("=" * len(header))
        print(header)
        print(separator)
        for algo, m in self._results.items():
            build_str = f"{m.build_time_s:.2f}s" if m.build_time_s > 0 else "cached"
            row = (
                f"{algo:<15} "
                f"{build_str:>10} "
                f"{m.avg_latency_ms:>9.2f} "
                f"{m.p50_latency_ms:>9.2f} "
                f"{m.p95_latency_ms:>9.2f} "
                f"{m.avg_recall_at_k:>10.4f} "
                f"{m.index_size_mb:>10.2f}"
            )
            print(row)
        print(separator)
        print()

    def save_results(self, output_dir: Optional[Path] = None) -> Path:
        """
        Save benchmark results as JSON and CSV.

        Returns:
            Path to the JSON results file.
        """
        if output_dir is None:
            output_dir = self.config.resolve_path(self.config.paths.benchmark_results)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        # JSON
        json_path = output_dir / f"benchmark_{timestamp}.json"
        results_dict = {
            "timestamp": timestamp,
            "config": {
                "embedding_model": self.config.embedding.model,
                "k": self.config.benchmark.top_k,
                "n_queries": self.config.benchmark.n_queries,
                "random_seed": self.config.benchmark.random_seed,
            },
            "results": {algo: m.to_dict() for algo, m in self._results.items()},
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

        # CSV (summary)
        csv_path = output_dir / f"benchmark_{timestamp}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "algorithm", "build_time_s", "avg_latency_ms",
                "p50_ms", "p95_ms", "p99_ms",
                f"recall_at_{self.config.benchmark.top_k}",
                "index_size_mb"
            ])
            for algo, m in self._results.items():
                writer.writerow([
                    algo, m.build_time_s, m.avg_latency_ms,
                    m.p50_latency_ms, m.p95_latency_ms, m.p99_latency_ms,
                    m.avg_recall_at_k, m.index_size_mb
                ])

        logger.info("Results saved: %s and %s", json_path, csv_path)
        return json_path
