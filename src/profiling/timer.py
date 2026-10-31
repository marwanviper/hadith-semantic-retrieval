"""
Reusable timing utilities.

Provides:
  - Timer: context manager that records elapsed time.
  - timed: decorator that logs function execution time.
  - TimingRecord: structured timing data.

Usage:
    with Timer("embedding") as t:
        embeddings = model.encode(texts)
    print(t.elapsed_ms)  # milliseconds

    @timed(logger, "index build")
    def build_index(...):
        ...
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional
import logging


@dataclass
class TimingRecord:
    """Holds named timing measurements (in milliseconds)."""
    measurements: Dict[str, float] = field(default_factory=dict)

    def record(self, name: str, elapsed_ms: float) -> None:
        self.measurements[name] = round(elapsed_ms, 3)

    def total_ms(self, *keys: str) -> float:
        """Sum selected measurements (or all if none specified)."""
        if not keys:
            return sum(self.measurements.values())
        return sum(self.measurements.get(k, 0.0) for k in keys)

    def to_dict(self) -> Dict[str, float]:
        return dict(self.measurements)


class Timer:
    """
    Context manager for measuring wall-clock time.

    Example::

        with Timer("search") as t:
            results = index.search(query, top_k=10)
        print(f"Search took {t.elapsed_ms:.2f} ms")
    """

    def __init__(self, name: str = "", record: Optional[TimingRecord] = None):
        """
        Args:
            name:   Human-readable label for this measurement.
            record: Optional TimingRecord to store result in.
        """
        self.name = name
        self.record = record
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    @property
    def elapsed_s(self) -> float:
        """Elapsed time in seconds."""
        return self.end_time - self.start_time

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed_s * 1000.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.end_time = time.perf_counter()
        if self.record is not None and self.name:
            self.record.record(self.name, self.elapsed_ms)


def timed(
    logger: logging.Logger,
    label: str,
    level: int = logging.DEBUG,
) -> Callable:
    """
    Decorator that logs the execution time of the wrapped function.

    Args:
        logger: Logger to use.
        label:  Human-readable label for the timing message.
        level:  Log level (default DEBUG).

    Example::

        @timed(logger, "build index")
        def build(self, embeddings):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.log(level, "%s completed in %.2f ms", label, elapsed_ms)
            return result
        return wrapper
    return decorator
