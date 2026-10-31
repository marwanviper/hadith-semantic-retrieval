"""Unit tests for benchmarking metrics."""

import pytest

from src.benchmarking.metrics import (
    compute_recall,
    compute_precision,
    compute_latency_stats,
    aggregate_metrics,
    QueryMetrics,
)


class TestRecall:
    def test_perfect_recall(self):
        gt = ["a", "b", "c"]
        cands = ["a", "b", "c"]
        assert compute_recall(gt, cands, k=3) == 1.0

    def test_zero_recall(self):
        gt = ["a", "b", "c"]
        cands = ["x", "y", "z"]
        assert compute_recall(gt, cands, k=3) == 0.0

    def test_partial_recall(self):
        gt = ["a", "b", "c", "d"]
        cands = ["a", "b", "x", "y"]
        recall = compute_recall(gt, cands, k=4)
        assert abs(recall - 0.5) < 1e-6

    def test_k_zero_returns_zero(self):
        assert compute_recall([], [], k=0) == 0.0


class TestPrecision:
    def test_perfect_precision(self):
        gt = ["a", "b", "c"]
        cands = ["a", "b", "c"]
        assert compute_precision(gt, cands, k=3) == 1.0

    def test_empty_candidates(self):
        assert compute_precision(["a"], [], k=3) == 0.0


class TestLatencyStats:
    def test_stats_computed(self):
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = compute_latency_stats(latencies)
        assert stats["min_ms"] == 10.0
        assert stats["max_ms"] == 50.0
        assert stats["avg_ms"] == 30.0

    def test_empty_returns_zeros(self):
        stats = compute_latency_stats([])
        assert stats["avg_ms"] == 0.0

    def test_p50_median(self):
        latencies = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = compute_latency_stats(latencies)
        assert stats["p50_ms"] == 3.0


class TestAggregateMetrics:
    def test_aggregate(self):
        per_query = [
            QueryMetrics(query="q1", latency_ms=10.0, recall_at_k=1.0, precision_at_k=1.0, n_results=5),
            QueryMetrics(query="q2", latency_ms=20.0, recall_at_k=0.8, precision_at_k=0.8, n_results=5),
        ]
        metrics = aggregate_metrics(
            algorithm="hnsw",
            per_query=per_query,
            build_time_s=1.5,
            index_size_mb=50.0,
            k=5,
            index_params={"M": 16},
        )
        assert metrics.algorithm == "hnsw"
        assert metrics.avg_recall_at_k == 0.9
        assert metrics.n_queries == 2
        assert metrics.build_time_s == 1.5
