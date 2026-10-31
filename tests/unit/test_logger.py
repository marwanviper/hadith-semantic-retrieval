"""Unit tests for logging configuration."""

import logging

import pytest

from src.logging.logger import configure_logging, get_logger


class TestLogger:
    def test_get_logger_returns_logger(self):
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_get_logger_different_names(self):
        logger1 = get_logger("module.a")
        logger2 = get_logger("module.b")
        assert logger1 is not logger2

    def test_configure_sets_level(self):
        configure_logging(level="DEBUG")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_configure_sets_level_info(self):
        configure_logging(level="INFO")
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_logger_can_log(self, caplog):
        configure_logging(level="DEBUG")
        logger = get_logger("test.caplog")
        with caplog.at_level(logging.DEBUG, logger="test.caplog"):
            logger.debug("test debug message")
            logger.info("test info message")
        assert "test debug message" in caplog.text
        assert "test info message" in caplog.text
