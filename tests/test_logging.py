"""Tests for dust2dusty logging module."""

import logging


def test_get_logger():
    """Test that get_logger returns the correct logger."""
    from dust2dusty.logging import LOGGER_NAME, get_logger

    logger = get_logger()
    assert logger.name == LOGGER_NAME
    assert isinstance(logger, logging.Logger)


def test_setup_logging_default():
    """Test setup_logging with default settings."""
    from dust2dusty.logging import setup_logging

    logger = setup_logging(debug=False)
    assert logger.level == logging.INFO


def test_setup_logging_debug():
    """Test setup_logging with debug enabled."""
    from dust2dusty.logging import setup_logging

    logger = setup_logging(debug=True)
    assert logger.level == logging.DEBUG


def test_setup_walker_logger():
    """Test walker-specific logger creation."""
    from dust2dusty.logging import LOGGER_NAME, setup_walker_logger

    walker_logger = setup_walker_logger("test_walker", debug=False)
    assert f"{LOGGER_NAME}.walker_test_walker" == walker_logger.name
