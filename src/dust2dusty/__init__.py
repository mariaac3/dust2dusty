"""
DUST2DUSTY: Supernova Cosmology Analysis with MCMC.

A Python package for Markov Chain Monte Carlo (MCMC) fitting of supernova
intrinsic scatter distributions while accounting for selection effects
using reweighting.

The package fits distributions for supernova properties (color, stretch,
extinction, etc.) by comparing real data to reweighted simulations via
the SALT2mu.exe executable.

Modules:
    cli: Command-line interface and Config dataclass.
    dust2dust: Worker-dependent likelihood functions and SALT2mu connection.
    mcmc: Main MCMC sampling function using emcee or nautilus.
    salt2mu: Interface to SALT2mu.exe subprocess.
    log: Shared logging configuration.
    utils: Utility functions for parameter handling and normalization.

Example Usage:
    Command line::

        dust2dusty --CONFIG config.yml --DEBUG

    Python API::

        from dust2dusty import Config, load_config, MCMC
        from dust2dusty.log import setup_logging, get_logger
        from dust2dusty.utils import init_salt2mu_realdata, input_cleaner

        setup_logging(debug=True)
        logger = get_logger()
        config = load_config("config.yml", args, logger)
        realdata = init_salt2mu_realdata(config, logger)
        pos, nwalkers, ndim = input_cleaner(...)
        MCMC(config, pos, nwalkers, ndim, realdata)
"""

__version__: str = "0.1.0"
__author__: str = "B. Popovic, D. Brout, B. Carreres, M. Acevedo"

# Import main components for convenient access
from dust2dusty import cli, utils
from dust2dusty.likelihood_worker import (
    log_likelihood,
    log_prior,
    log_probability,
)
from dust2dusty.log import get_logger, setup_logging
from dust2dusty.mcmc import MCMC
from dust2dusty.salt2mu import SALT2mu

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Logging
    "setup_logging",
    "get_logger",
    # Main functions
    "MCMC",
    "log_likelihood",
    "log_prior",
    "log_probability",
    # SALT2mu interface
    "SALT2mu",
]
