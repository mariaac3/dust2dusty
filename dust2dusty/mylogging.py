"""
Shared logging configuration for DUST2DUSTY package.

This module provides a unified logging setup for all modules in the package.
All modules should use the logger obtained from get_logger().

Usage:
    from dust2dusty.logging import setup_logging, get_logger

    # In main script:
    setup_logging(debug=True)  # Call once at startup

    # In any module:
    logger = get_logger()
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
"""

from __future__ import annotations

import logging
import sys

# Package-wide logger name
LOGGER_NAME: str = "dust2dusty"

# Track if logging has been configured
_logging_configured: bool = False


def setup_logging(
    debug: bool = False, log_file: str | None = None, verbose: bool = False
) -> logging.Logger:
    """
    Configure logging for the DUST2DUSTY package.

    Sets up a package-wide logger with console output and optional file output.
    Should be called once at the start of the main program.

    Args:
        debug: If True, set logging level to DEBUG; otherwise INFO.
        log_file: Optional path to log file. If provided, logs will also be
            written to this file.
        verbose: If True, show INFO level messages on console; otherwise only
            show WARNING and above on console. File logging is unaffected.

    Returns:
        Configured logger instance for DUST2DUSTY.
    """
    global _logging_configured

    logger = logging.getLogger(LOGGER_NAME)

    # Avoid adding handlers multiple times
    if _logging_configured:
        # Just update the level if already configured
        level = logging.DEBUG if debug else logging.INFO
        logger.setLevel(level)
        return logger

    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    # Console handler - level depends on verbose flag
    console_handler = logging.StreamHandler(stream=sys.stdout)
    if debug:
        console_level = logging.DEBUG
    elif verbose:
        console_level = logging.INFO
    else:
        console_level = logging.WARNING
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(
        fmt="[%(levelname)8s |%(filename)21s:%(lineno)3d]   %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)8s |%(filename)21s:%(lineno)3d]   %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Suppress verbose output from matplotlib and seaborn
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("seaborn").setLevel(logging.ERROR)

    _logging_configured = True
    return logger


def get_logger() -> logging.Logger:
    """
    Get the package-wide logger.

    Returns the DUST2DUSTY logger. If setup_logging() hasn't been called yet,
    returns an unconfigured logger (messages may not appear until setup_logging()
    is called).

    Returns:
        The DUST2DUSTY package logger.
    """
    return logging.getLogger(LOGGER_NAME)


def setup_walker_logger(
    walker_id: int | str, log_dir: str = "logs", debug: bool = False
) -> logging.Logger:
    """
    Create a logger for a specific MCMC walker subprocess.

    Each walker gets its own log file for detailed debugging of subprocess
    communication with SALT2mu.exe. Logs are always written to files (never
    to terminal).

    Args:
        walker_id: Integer or string identifier for the walker.
        log_dir: Directory for log files.
        debug: If True, log level is DEBUG; otherwise INFO.

    Returns:
        Logger instance for this specific walker.
    """
    logger_name = f"{LOGGER_NAME}.walker_{walker_id}"
    logger = logging.getLogger(logger_name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    # File handler for walker-specific log (always write to file)
    file_handler = logging.FileHandler(f"{log_dir}/walker_{walker_id}.log", mode="w")
    file_handler.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Prevent propagation to root logger (avoid terminal output)
    logger.propagate = False

    return logger
