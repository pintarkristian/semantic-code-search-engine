"""Unit tests for semcode.logging."""

import logging

from semcode.logging import configure_logging, get_logger


def test_get_logger_has_expected_methods() -> None:
    configure_logging(log_level="DEBUG", log_format="pretty")
    log = get_logger("test.module")
    # Verify the returned object is usable as a logger
    assert callable(getattr(log, "info", None))
    assert callable(getattr(log, "debug", None))
    assert callable(getattr(log, "warning", None))
    assert callable(getattr(log, "error", None))


def test_configure_logging_sets_root_level() -> None:
    configure_logging(log_level="WARNING", log_format="pretty")
    assert logging.getLogger().level == logging.WARNING


def test_configure_logging_sets_debug_level() -> None:
    configure_logging(log_level="DEBUG", log_format="pretty")
    assert logging.getLogger().level == logging.DEBUG


def test_configure_logging_json_does_not_raise() -> None:
    configure_logging(log_level="INFO", log_format="json")
    log = get_logger("test.json")
    log.info("json log test", key="value")


def test_configure_logging_pretty_does_not_raise() -> None:
    configure_logging(log_level="INFO", log_format="pretty")
    log = get_logger("test.pretty")
    log.info("pretty log test", key="value")


def test_logger_bind_does_not_raise() -> None:
    configure_logging(log_level="DEBUG", log_format="pretty")
    log = get_logger("semcode.ingest")
    bound = log.bind(repo="test-repo")
    bound.debug("context bound", extra_field=42)


def test_configure_logging_replaces_handlers() -> None:
    # Calling configure_logging twice should not double-add handlers
    configure_logging(log_level="INFO", log_format="pretty")
    configure_logging(log_level="INFO", log_format="pretty")
    assert len(logging.getLogger().handlers) == 1
