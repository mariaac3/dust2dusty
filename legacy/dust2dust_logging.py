"""
Shared logging configuration for DUST2DUST package.

This module provides a unified logging setup for both DUST2DUST.py and callSALT2mu.py.
All modules in the package should use the logger obtained from get_logger().

Usage:
    from dust2dust_logging import setup_logging, get_logger

    # In main script (DUST2DUST.py):
    setup_logging(debug=True)  # Call once at startup

    # In any module:
    logger = get_logger()
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
"""

import logging
import sys

# Package-wide logger name
LOGGER_NAME = "dust2dust"

# Track if logging has been configured
_logging_configured = False


def setup_logging(debug=False, log_file=None):
    """
    Configure logging for the DUST2DUST package.

    Sets up a package-wide logger with console output and optional file output.
    Should be called once at the start of the main program.

    Args:
        debug: If True, set logging level to DEBUG; otherwise INFO (default: False)
        log_file: Optional path to log file. If provided, logs will also be written
                  to this file (default: None)

    Returns:
        logging.Logger: Configured logger instance for DUST2DUST
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

    # Console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        fmt="[%(levelname)8s |%(filename)21s:%(lineno)3d]   %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a")
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


def get_logger():
    """
    Get the package-wide logger.

    Returns the DUST2DUST logger. If setup_logging() hasn't been called yet,
    returns an unconfigured logger (messages may not appear until setup_logging()
    is called).

    Returns:
        logging.Logger: The DUST2DUST package logger
    """
    return logging.getLogger(LOGGER_NAME)


def setup_salt2mu_logger(walker_id, log_dir="logs", debug=False):
    """
    Create a logger for a specific MCMC walker subprocess.

    Each walker gets its own log file for detailed debugging of subprocess
    communication with SALT2mu.exe.

    Args:
        walker_id: Integer or string identifier for the walker
        log_dir: Directory for log files (default: "logs")
        debug: If True, enable logging; otherwise use NullHandler (default: False)

    Returns:
        logging.Logger: Logger instance for this specific walker
    """
    logger_name = f"{LOGGER_NAME}.walker_{walker_id}"
    logger = logging.getLogger(logger_name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    if debug:
        # File handler for walker-specific log
        file_handler = logging.FileHandler(f"{log_dir}/walker_{walker_id}.log", mode="a+")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        # Null handler when not debugging
        logger.addHandler(logging.NullHandler())

    return logger
