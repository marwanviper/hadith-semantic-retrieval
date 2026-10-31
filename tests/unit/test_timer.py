"""Unit tests for timing utilities."""

import time

import pytest

from src.profiling.timer import Timer, TimingRecord, timed
from src.logging.logger import get_logger


class TestTimer:
    def test_elapsed_ms_positive(self):
        with Timer("test") as t:
            time.sleep(0.01)
        assert t.elapsed_ms > 5  # at least 5ms
        assert t.elapsed_ms < 500  # sanity upper bound

    def test_elapsed_s(self):
        with Timer("test") as t:
            time.sleep(0.05)
        assert t.elapsed_s > 0.01

    def test_records_to_timing_record(self):
        record = TimingRecord()
        with Timer("step1", record=record):
            time.sleep(0.01)
        assert "step1" in record.measurements
        assert record.measurements["step1"] > 0

    def test_timing_record_total(self):
        record = TimingRecord()
        record.record("a", 10.0)
        record.record("b", 20.0)
        assert record.total_ms() == 30.0
        assert record.total_ms("a") == 10.0

    def test_timing_record_to_dict(self):
        record = TimingRecord()
        record.record("search", 7.5)
        d = record.to_dict()
        assert d["search"] == 7.5


class TestTimedDecorator:
    def test_timed_decorator_returns_result(self):
        logger = get_logger("test.timer")

        @timed(logger, "my_func")
        def my_func(x: int) -> int:
            return x * 2

        result = my_func(5)
        assert result == 10

    def test_timed_decorator_preserves_name(self):
        logger = get_logger("test.timer")

        @timed(logger, "my_named_func")
        def my_named_func():
            pass

        assert my_named_func.__name__ == "my_named_func"
