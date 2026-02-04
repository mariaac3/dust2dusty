"""
Command-line interface for DUST2DUSTY.

This module provides the main entry point for running DUST2DUSTY from the
command line, as well as configuration loading and the Config dataclass.

Usage:
    dust2dusty --CONFIG config.yml [--DEBUG] [--TEST_RUN] [--NOWEIGHT]

Example:
    dust2dusty --CONFIG IN_DUST2DUST.yml --DEBUG
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import _MISSING_TYPE, dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import yaml
from numpy.typing import NDArray

from dust2dusty.logging import get_logger, setup_logging
from dust2dusty.utils import __dust2dust_str__


@dataclass
class Config:
    """
    Configuration dataclass for DUST2DUSTY.

    Provides type-safe access to all configuration parameters with attribute
    access syntax (config.data_input instead of config['DATA_INPUT']).

    Attributes:
        data_input: Path to real data input file for SALT2mu.
        sim_input: Path to simulation input file for SALT2mu.
        simref_file: Path to simulation reference file.
        outdir: Output directory for results.
        chains: Path to existing chains file (for resuming).
        inp_params: List of parameter names to fit.
        params: Initial parameter values for test runs.
        paramshapesdict: Maps parameters to distribution shapes.
        splitdict: Defines parameter splits by host properties.
        splitparam: Primary split parameter name.
        parameter_initialization: Initialization specs for each parameter.
        splitarr: Array generation strings for split variables.
        CMD_DATA: Command-line override for data input.
        CMD_SIM: Command-line override for simulation input.
        TEST_RUN: If True, run single likelihood evaluation only.
        debug: If True, enable verbose debug output.
        NOWEIGHT: If True, disable reweighting function.

    Class Attributes:
        PARAM_TO_SALT2MU: Maps internal names to SALT2mu column names.
        SUBPROCESS_TO_SNANA: Maps subprocess names to SNANA names.
        DEFAULT_PARAMETER_RANGES: Value grids for PDF generation.
        SPLIT_PARAMETER_FORMATS: Binning specs for split parameters.
        PARAMETER_OVERRIDES: Fixed parameters (not fitted).
        DISTRIBUTION_PARAMETERS: Parameter names for each distribution type.
    """

    # Parameter name mappings for SALT2mu format
    PARAM_TO_SALT2MU: ClassVar[dict[str, str]] = {
        "c": "SIM_c",
        "x1": "SIM_x1",
        "HOST_LOGMASS": "HOST_LOGMASS",
        "Mass": "HOST_LOGMASS",
        "RV": "SIM_RV",
        "EBV": "SIM_EBV",
        "beta": "SIM_beta",
        "SIM_ZCMB": "SIM_ZCMB",
        "EBVZ": "SIM_EBV",
        "ZTRUE": "SIM_ZCMB",
        "z": "SIM_ZCMB",
        "HOST_COLOR": "HOST_COLOR",
    }

    # SNANA output format mappings
    SUBPROCESS_TO_SNANA: ClassVar[dict[str, str]] = {
        "SIM_c": "SALT2c",
        "SIM_RV": "RV",
        "HOST_LOGMASS": "LOGMASS",
        "SIM_EBV": "EBV",
        "SIM_ZCMB": "ZTRUE",
        "SIM_beta": "SALT2BETA",
        "HOST_COLOR": "COLOR",
    }

    # Default value ranges for parameter arrays
    DEFAULT_PARAMETER_RANGES: ClassVar[dict[str, NDArray[np.float64]]] = {
        "c": np.arange(-0.5, 0.5, 0.001),
        "x1": np.arange(-5, 5, 0.01),
        "RV": np.arange(0, 8, 0.1),
        "EBV": np.arange(0.0, 1.5, 0.02),
        "EBVZ": np.arange(0.0, 1.5, 0.02),
    }

    # Split parameter format specifications
    SPLIT_PARAMETER_FORMATS: ClassVar[dict[str, str]] = {
        "HOST_LOGMASS": "HOST_LOGMASS(2,0:20)",
        "HOST_COLOR": "HOST_COLOR(2,-.5:2.5)",
        "zHD": "zHD(2,0:1)",
    }

    # Parameter override dictionary (for fixing parameters)
    PARAMETER_OVERRIDES: ClassVar[dict[str, float]] = {}

    # Distribution parameter specifications
    DISTRIBUTION_PARAMETERS: ClassVar[dict[str, list[str]]] = {
        "Gaussian": ["mu", "std"],
        "Skewed Gaussian": ["mu", "std_l", "std_r"],
        "Exponential": ["Tau"],
        "LogNormal": ["ln_mu", "ln_std"],
        "Double Gaussian": ["a1", "mu1", "std1", "mu2", "std2"],
    }

    # File paths (required)
    data_input: str
    sim_input: str
    simref_file: str

    # File paths (optional)
    outdir: str = "./dust2dust_output/"
    chains: str | None = None

    # Parameter configuration
    inp_params: list[str] = field(default_factory=list)
    params: list[float] = field(default_factory=list)
    paramshapesdict: dict[str, str] = field(default_factory=dict)
    splitdict: dict[str, dict[str, float]] = field(default_factory=dict)
    splitparam: str = "HOST_LOGMASS"
    parameter_initialization: dict[str, list[Any]] = field(default_factory=dict)
    splitarr: dict[str, str] = field(default_factory=dict)

    # Command line arguments
    # - Command-line overrides
    CMD_DATA: str | None = None
    CMD_SIM: str | None = None

    # - Runtime flags
    USE_MPI: bool = False
    TEST_RUN: bool = False
    DEBUG: bool = False
    NOWEIGHT: bool = False
    VERBOSE: bool = False

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], args: argparse.Namespace) -> Config:
        """
        Create Config object from YAML dictionary and command-line arguments.

        Args:
            config_dict: Dictionary loaded from YAML configuration file.
            args: Parsed command-line arguments.

        Returns:
            Configured Config dataclass instance.
        """
        return cls(
            # File paths
            data_input=config_dict["DATA_INPUT"],
            sim_input=config_dict["SIM_INPUT"],
            simref_file=config_dict["SIMREF_FILE"],
            outdir=config_dict.get("OUTDIR", ""),
            chains=config_dict.get("CHAINS"),
            # Parameter configuration
            inp_params=config_dict["INP_PARAMS"],
            params=config_dict.get("PARAMS", []),
            paramshapesdict=config_dict["PARAMSHAPESDICT"],
            splitdict=config_dict["SPLITDICT"],
            splitparam=config_dict.get("SPLITPARAM", "HOST_LOGMASS"),
            parameter_initialization=config_dict["PARAMETER_INITIALIZATION"],
            splitarr=config_dict["SPLITARR"],
            # Command-line arguments
            CMD_DATA=args.CMD_DATA,
            CMD_SIM=args.CMD_SIM,
            TEST_RUN=args.TEST_RUN,
            DEBUG=args.DEBUG or args.TEST_RUN,
            NOWEIGHT=args.NOWEIGHT,
            USE_MPI=args.USE_MPI,
            VERBOSE=args.VERBOSE,
        )

    def __post_init__(self):
        # Loop through the fields
        for f in fields(self):
            # If there is a default and the value of the field is none we can assign a value
            if not isinstance(f.default, _MISSING_TYPE) and getattr(self, f.name) is None:
                setattr(self, f.name, f.default)


def create_output_directories(outdir: str, logger: logging.Logger):
    """
    Create output directory structure for DUST2DUSTY results.

    Creates the main output directory and required subdirectories:
        - chains: MCMC chain outputs
        - figures: Diagnostic plots
        - parallel: Subprocess communication files
        - logs: Log files
        - realdata_files: Real data SALT2mu outputs
        - worker_files: Worker subprocess files

    Args:
        outdir: Path to main output directory (can be relative or absolute).
        logger: Logger instance for output messages.

    Returns:
        Absolute path to output directory with trailing slash.

    Raises:
        SystemExit: If directory structure cannot be created.
    """
    # Use current directory if none specified
    outdir = Path(outdir)
    logger.debug(f"Create main directory {outdir.absolute()}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Create required subdirectories
    required_subdirs = [
        "chains",
        "figures",
        "parallel",
        "logs",
        "realdata_files",
        "worker_files",
    ]
    for subdir in required_subdirs:
        subdir_path = outdir / subdir
        logger.debug(f"Create sub directory {subdir_path.absolute()}")
        subdir_path.mkdir(parents=True, exist_ok=True)


def load_config(config_path: str, args: argparse.Namespace, logger: logging.Logger) -> Config:
    """
    Load configuration from YAML file and set up output directories.

    Performs complete configuration setup:
        1. Loads and validates YAML configuration file
        2. Creates Config dataclass instance
        3. Sets up output directory structure
        4. Logs configuration summary

    Args:
        config_path: Path to YAML configuration file.
        args: Parsed command-line arguments.
        logger: Logger instance for output messages.

    Returns:
        Fully configured Config instance with output directories created.

    Raises:
        SystemExit: If config file doesn't exist, has invalid syntax,
            is missing required keys, or output directories cannot be created.
    """
    # Validate config file path
    if not config_path:
        logger.error("No configuration file specified. Use --CONFIG <path>")
        sys.exit(1)

    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    # Load YAML file
    try:
        with open(config_path) as cfgfile:
            config_dict = yaml.safe_load(cfgfile)
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML syntax in {config_path}")
        logger.error(e)
        sys.exit(1)

    # Validate required keys
    required_keys = [
        "DATA_INPUT",
        "SIM_INPUT",
        "INP_PARAMS",
        "PARAMSHAPESDICT",
        "SPLITDICT",
        "PARAMETER_INITIALIZATION",
        "SPLITARR",
        "SIMREF_FILE",
    ]
    missing_keys = [key for key in required_keys if key not in config_dict]
    if missing_keys:
        logger.error(f"Missing required configuration keys: {missing_keys}")
        sys.exit(1)

    # Create Config object from dictionary and args
    config = Config.from_dict(config_dict, args)

    logger.info(f"Loaded configuration from: {config_path}")

    # Set up output directory structure
    create_output_directories(config.outdir, logger)

    # Log configuration summary
    logger.info("Configuration finalized successfully:")
    logger.info(f"---- Data: {Path(config.data_input).absolute()}")
    logger.info(f"---- Simulation: {Path(config.sim_input).absolute()}")
    logger.info(f"---- Parameters to fit: {', '.join(config.inp_params)}")
    logger.info(f"---- Output directory: {Path(config.outdir).absolute()}")

    return config


def get_args() -> argparse.Namespace:
    """
    Parse command-line arguments for DUST2DUSTY.

    Defines and parses all command-line flags including configuration file
    path, debug modes, and SALT2mu command overrides.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="DUST2DUSTY: MCMC fitting of supernova intrinsic scatter distributions"
    )

    parser.add_argument(
        "--CONFIG",
        type=str,
        default="",
        help="Path to YAML configuration file (required)",
    )

    parser.add_argument(
        "--TEST_RUN",
        action="store_true",
        help="Run single likelihood evaluation for testing (does not launch MCMC)",
    )

    parser.add_argument(
        "--DEBUG",
        action="store_true",
        help="Enable debug mode with verbose output",
    )

    parser.add_argument(
        "--NOWEIGHT",
        action="store_true",
        help="Disable reweighting function (use for unweighted sims like G10, C11)",
    )

    parser.add_argument(
        "--CMD_DATA",
        type=str,
        default=None,
        help="Command-line override for SALT2mu data input file",
    )

    parser.add_argument(
        "--CMD_SIM",
        type=str,
        default=None,
        help="Command-line override for SALT2mu simulation input file",
    )

    parser.add_argument(
        "--USE_MPI",
        action="store_true",
        help="Use MPI to distribute process",
    )

    parser.add_argument(
        "--VERBOSE",
        action="store_true",
        help="Show INFO level logging on terminal (default: only WARNING and above)",
    )

    return parser.parse_args()


def _get_mpi_info() -> tuple[int, int]:
    """
    Get MPI rank and size.

    Returns:
        Tuple of (rank, size). Returns (0, 1) if MPI is not available.
    """
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        return comm.Get_rank(), comm.Get_size()
    except ImportError:
        return 0, 1


def main() -> int:
    """
    Main entry point for the dust2dusty command-line tool.

    Parses command-line arguments, sets up logging, loads configuration,
    and runs either a test evaluation or full MCMC sampling.

    For MPI runs, only the master process (rank 0) performs full setup.
    Worker processes (rank > 0) skip heavy initialization and go directly
    to the MCMC function where they wait in the pool.

    Returns:
        Exit code (0 for success).
    """
    # Import here to avoid circular imports
    from dust2dusty.dust2dust import (
        _init_worker,
        log_probability,
    )
    from dust2dusty.mcmc import MCMC
    from dust2dusty.utils import init_salt2mu_realdata, input_cleaner

    # Check MPI status early - workers should not do heavy setup
    rank, size = _get_mpi_info()
    is_master = rank == 0

    if is_master:
        # Master process (rank 0) does full setup
        args = get_args()
        debug = args.DEBUG or args.TEST_RUN
        setup_logging(debug=debug, verbose=args.VERBOSE)
        logger = get_logger()
        logger.info(__dust2dust_str__)

        config = load_config(args.CONFIG, args, logger)
        realdata_salt2mu_results = init_salt2mu_realdata(config, logger, debug=debug)

        pos, nwalkers, ndim = input_cleaner(
            config.inp_params,
            config.paramshapesdict,
            config.splitdict,
            config.DISTRIBUTION_PARAMETERS,
            config.parameter_initialization,
            config.PARAMETER_OVERRIDES,
            walkfactor=2,
        )

        # Test run mode - single likelihood evaluation (no MPI needed)
        if config.TEST_RUN:
            _init_worker(config, realdata_salt2mu_results, debug=debug)
            logger.info(f"Test run result: {log_probability(config.params)}")
            sys.exit(0)

        # Full MCMC run
        logger.info("=" * 60)
        logger.info("Starting MCMC sampling...")
        logger.info(f"  Walkers: {nwalkers}")
        logger.info(f"  Dimensions: {ndim}")
        logger.info(f"  Parameters: {', '.join(config.inp_params)}")
        logger.info("=" * 60 + "\n")
        logger.debug("DEBUG MODE ON")

        MCMC(config, pos, nwalkers, ndim, realdata_salt2mu_results, debug=debug)

        logger.info("DUST2DUST(Y) complete.")
    else:
        # Worker processes (rank > 0) go directly to MCMC with None values
        # They will receive config via the pool initializer
        MCMC(None, None, 0, 0, None, debug=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
