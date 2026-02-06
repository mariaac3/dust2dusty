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
    assert f"{LOGGER_NAME}.worker_salt2mu_test_walker" == walker_logger.name


def test_add_file_handler():
    """Test that add_file_handler creates a log file and writes to it."""
    import tempfile
    from pathlib import Path

    from dust2dusty.logging import LOGGER_NAME, add_file_handler

    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = str(Path(tmpdir) / "test.log")
        add_file_handler(log_file)

        logger = logging.getLogger(LOGGER_NAME)
        logger.warning("test message from add_file_handler")

        # Flush all handlers
        for h in logger.handlers:
            h.flush()

        content = Path(log_file).read_text()
        assert "test message from add_file_handler" in content

        # Clean up: remove the handler we added
        for h in logger.handlers[:]:
            if isinstance(h, logging.FileHandler) and h.baseFilename == str(
                Path(log_file).resolve()
            ):
                logger.removeHandler(h)
                h.close()
