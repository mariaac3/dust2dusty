"""
DUST2DUSTY: Supernova Cosmology Analysis with MCMC

A Python package for Markov Chain Monte Carlo (MCMC) fitting of supernova intrinsic
scatter distributions while accounting for selection effects using reweighting.

The package fits distributions for supernova properties (color, stretch, extinction, etc.)
by comparing real data to reweighted simulations via the SALT2mu.exe executable.

Main components:
    - dust2dust: Main MCMC fitting module
    - salt2mu: Interface to SALT2mu.exe subprocess
    - logging: Shared logging configuration

Example usage:
    Command line:
        dust2dusty --CONFIG config.yml --DEBUG

    Python API:
        from dust2dusty import Config, load_config, run_mcmc
        from dust2dusty.logging import setup_logging

        setup_logging(debug=True)
        config = load_config("config.yml", args)
        run_mcmc(config)
"""

__version__ = "0.1.0"
__author__ = "B. Popovic, D. Brout, B. Carreres, M. Acevedo"

# Import main components for convenient access
from dust2dusty.dust2dust import (
    MCMC,
    Config,
    init_dust2dust,
    load_config,
    log_likelihood,
    log_prior,
    log_probability,
)
from dust2dusty.logging import get_logger, setup_logging
from dust2dusty.salt2mu import SALT2mu

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Logging
    "setup_logging",
    "get_logger",
    # Configuration
    "Config",
    "load_config",
    # Main functions
    "init_dust2dust",
    "MCMC",
    "log_likelihood",
    "log_prior",
    "log_probability",
    # SALT2mu interface
    "SALT2mu",
]
