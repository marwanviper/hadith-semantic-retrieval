"""
Retrieval quality metrics for benchmarking.

Metrics:
  Recall@K: Fraction of true top-K (ground truth) found in ANN top-K.
    recall@K = |ground_truth_ids ∩ candidate_ids| / K

  Precision@K: Fraction of ANN results that are in ground truth.
    precision@K = |ground_truth_ids ∩ candidate_ids| / |candidate_ids|

  Latency percentiles: P50, P95, P99.

The ground truth is always the Brute Force exact search results,
which have perfect recall (1.0) by definition.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class QueryMetrics:
    """Metrics for a single benchmark query."""
    query: str
    latency_ms: float
    recall_at_k: float
    precision_at_k: float
    n_results: int


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics across all benchmark queries."""
    algorithm: str
    n_queries: int
    k: int
    build_time_s: float
    index_size_mb: float
    index_params: Dict = field(default_factory=dict)

    # Latency
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    # Quality
    avg_recall_at_k: float = 0.0
    avg_precision_at_k: float = 0.0

    # Per-query results
    per_query: List[QueryMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "algorithm": self.algorithm,
            "n_queries": self.n_queries,
            "k": self.k,
            "build_time_s": self.build_time_s,
            "index_size_mb": self.index_size_mb,
            "index_params": self.index_params,
            "latency": {
                "avg_ms": self.avg_latency_ms,
                "p50_ms": self.p50_latency_ms,
                "p95_ms": self.p95_latency_ms,
                "p99_ms": self.p99_latency_ms,
                "min_ms": self.min_latency_ms,
                "max_ms": self.max_latency_ms,
            },
            "quality": {
                "avg_recall_at_k": self.avg_recall_at_k,
                "avg_precision_at_k": self.avg_precision_at_k,
            },
        }


def compute_recall(ground_truth_ids: List[str], candidate_ids: List[str], k: int) -> float:
    """
    Compute Recall@K.

    Args:
        ground_truth_ids: True top-K Hadith IDs (from Brute Force).
        candidate_ids:    ANN top-K Hadith IDs.
        k:               K value.

    Returns:
        Recall@K in [0, 1]. Higher is better.
    """
    if k == 0:
        return 0.0
    gt_set = set(ground_truth_ids[:k])
    cand_set = set(candidate_ids[:k])
    return len(gt_set & cand_set) / max(len(gt_set), 1)


def compute_precision(ground_truth_ids: List[str], candidate_ids: List[str], k: int) -> float:
    """
    Compute Precision@K.

    Args:
        ground_truth_ids: True top-K Hadith IDs.
        candidate_ids:    ANN top-K Hadith IDs.
        k:               K value.

    Returns:
        Precision@K in [0, 1].
    """
    if not candidate_ids:
        return 0.0
    gt_set = set(ground_truth_ids[:k])
    n_relevant = sum(1 for cid in candidate_ids[:k] if cid in gt_set)
    return n_relevant / min(k, len(candidate_ids))


def compute_latency_stats(
    latencies_ms: List[float],
) -> Dict[str, float]:
    """
    Compute descriptive latency statistics.

    Args:
        latencies_ms: List of per-query latencies in milliseconds.

    Returns:
        Dict with avg, p50, p95, p99, min, max.
    """
    if not latencies_ms:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0,
                "p99_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}

    arr = sorted(latencies_ms)
    n = len(arr)

    def percentile(data: list, p: float) -> float:
        idx = int(p / 100 * (n - 1))
        return data[idx]

    return {
        "avg_ms": round(statistics.mean(arr), 3),
        "p50_ms": round(percentile(arr, 50), 3),
        "p95_ms": round(percentile(arr, 95), 3),
        "p99_ms": round(percentile(arr, 99), 3),
        "min_ms": round(arr[0], 3),
        "max_ms": round(arr[-1], 3),
    }


def aggregate_metrics(
    algorithm: str,
    per_query: List[QueryMetrics],
    build_time_s: float,
    index_size_mb: float,
    k: int,
    index_params: Dict,
) -> BenchmarkMetrics:
    """Compute aggregate BenchmarkMetrics from per-query results."""
    latencies = [q.latency_ms for q in per_query]
    lat_stats = compute_latency_stats(latencies)
    avg_recall = statistics.mean(q.recall_at_k for q in per_query) if per_query else 0.0
    avg_precision = statistics.mean(q.precision_at_k for q in per_query) if per_query else 0.0

    return BenchmarkMetrics(
        algorithm=algorithm,
        n_queries=len(per_query),
        k=k,
        build_time_s=round(build_time_s, 3),
        index_size_mb=round(index_size_mb, 2),
        index_params=index_params,
        avg_latency_ms=lat_stats["avg_ms"],
        p50_latency_ms=lat_stats["p50_ms"],
        p95_latency_ms=lat_stats["p95_ms"],
        p99_latency_ms=lat_stats["p99_ms"],
        min_latency_ms=lat_stats["min_ms"],
        max_latency_ms=lat_stats["max_ms"],
        avg_recall_at_k=round(avg_recall, 4),
        avg_precision_at_k=round(avg_precision, 4),
        per_query=per_query,
    )
