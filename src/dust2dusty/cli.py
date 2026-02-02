"""
Command-line interface for DUST2DUSTY.

This module provides the main entry point for running DUST2DUSTY from the command line.

Usage:
    dust2dusty --CONFIG config.yml [--DEBUG] [--test_run] [--NOWEIGHT]

Example:
    dust2dusty --CONFIG IN_DUST2DUST.yml --DEBUG
"""

import sys

from dust2dusty.dust2dust import (
    MCMC,
    _init_worker,
    get_args,
    init_dust2dust,
    input_cleaner,
    load_config,
    log_probability,
)
from dust2dusty.logging import get_logger, setup_logging


def main():
    """
    Main entry point for the dust2dusty command-line tool.

    Parses command-line arguments, sets up logging, loads configuration,
    and runs either a test evaluation or full MCMC sampling.
    """
    # Parse arguments and load configuration
    args = get_args()

    # Set up logging before loading config (uses shared logging module)
    DEBUG = args.DEBUG or args.test_run
    setup_logging(debug=DEBUG)
    logger = get_logger()

    # Load and validate configuration
    config = load_config(args.CONFIG, args)

    # Initialize real data
    realdata = init_dust2dust(config, debug=DEBUG)

    if DEBUG:
        nwalkers = 1
    else:
        pos, nwalkers, ndim = input_cleaner(
            config.inp_params,
            config.paramshapesdict,
            config.splitdict,
            config.parameter_initialization,
            config.PARAMETER_OVERRIDES,
            walkfactor=3,
        )

    # Test run mode - single likelihood evaluation
    if config.test_run:
        _init_worker(config, realdata, debug=DEBUG)
        logger.info(f"Test run result: {log_probability(config.params)}")
        sys.exit(0)

    # Full MCMC run
    logger.debug("\n" + "=" * 60)
    logger.debug("Starting MCMC sampling...")
    logger.debug(f"  Walkers: {nwalkers}")
    logger.debug(f"  Dimensions: {ndim}")
    logger.debug(f"  Parameters: {', '.join(config.inp_params)}")
    logger.debug("=" * 60 + "\n")

    sampler = MCMC(config, pos, nwalkers, ndim, realdata, debug=DEBUG)

    logger.info("DUST2DUSTY complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
