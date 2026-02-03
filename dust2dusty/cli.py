"""
Command-line interface for DUST2DUSTY.

This module provides the main entry point for running DUST2DUSTY from the command line,
as well as configuration loading and the Config dataclass.

Usage:
    dust2dusty --CONFIG config.yml [--DEBUG] [--test_run] [--NOWEIGHT]

Example:
    dust2dusty --CONFIG IN_DUST2DUST.yml --DEBUG
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np
import yaml

from dust2dusty.logging import get_logger, setup_logging

# ===========================================================================================================================================
############################################################# Configuration Class ###################################################
# ===========================================================================================================================================


@dataclass
class Config:
    """
    Configuration dataclass for DUST2DUST.

    Provides type-safe access to all configuration parameters with
    attribute access syntax (config.data_input instead of config['DATA_INPUT']).

    Attributes organized by purpose:
    - File paths: data_input, sim_input, simref_file, outdir, chains
    - Parameters: inp_params, params, paramshapesdict, splitdict, splitparam, parameter_initialization, splitarr
    - Command-line overrides: cmd_data, cmd_sim
    - Flags: single, debug, noweight
    """

    # Parameter name mappings for SALT2mu format
    # Converts internal parameter names to SALT2mu/simulation column names
    PARAM_TO_SALT2MU: ClassVar[Dict[str, str]] = {
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
    # Converts SUBPROCESS column names to SNANA standard names
    SUBPROCESS_TO_SNANA: ClassVar[Dict[str, str]] = {
        "SIM_c": "SALT2c",
        "SIM_RV": "RV",
        "HOST_LOGMASS": "LOGMASS",
        "SIM_EBV": "EBV",
        "SIM_ZCMB": "ZTRUE",
        "SIM_beta": "SALT2BETA",
        "HOST_COLOR": "COLOR",
    }

    # Default value ranges for parameter arrays
    # Defines the grid of values used for PDF generation for each parameter
    DEFAULT_PARAMETER_RANGES: ClassVar[Dict[str, np.ndarray]] = {
        "c": np.arange(-0.5, 0.5, 0.001),
        "x1": np.arange(-5, 5, 0.01),
        "RV": np.arange(0, 8, 0.1),
        "EBV": np.arange(0.0, 1.5, 0.02),
        "EBVZ": np.arange(0.0, 1.5, 0.02),
    }

    # Split parameter format specifications
    # Defines how parameters are split into bins for SALT2mu output
    # Format: 'PARAM(nbins, min:max)'
    SPLIT_PARAMETER_FORMATS: ClassVar[Dict[str, str]] = {
        "HOST_LOGMASS": "HOST_LOGMASS(2,0:20)",
        "HOST_COLOR": "HOST_COLOR(2,-.5:2.5)",
        "zHD": "zHD(2,0:1)",
    }

    # Parameter override dictionary
    # Used to fix specific parameters during fitting (not fitted, held constant)
    # Populated programmatically based on user input or left empty for standard fitting
    PARAMETER_OVERRIDES: ClassVar[Dict[str, float]] = {}

    # Distribution parameter specifications
    # Maps distribution types to their required parameter names
    DISTRIBUTION_PARAMETERS: ClassVar[Dict[str, List[str]]] = {
        "Gaussian": ["mu", "std"],
        "Skewed Gaussian": ["mu", "std_l", "std_r"],
        "Exponential": ["Tau"],
        "LogNormal": ["ln_mu", "ln_std"],
        "Double Gaussian": ["a1", "mu1", "std1", "mu2", "std2"],
    }

    # File paths
    data_input: str
    sim_input: str
    simref_file: str
    outdir: str = ""
    chains: Optional[str] = None

    # Parameter configuration
    inp_params: List[str] = field(default_factory=list)
    params: List[float] = field(default_factory=list)
    paramshapesdict: Dict[str, str] = field(default_factory=dict)
    splitdict: Dict[str, Dict[str, float]] = field(default_factory=dict)
    splitparam: str = "HOST_LOGMASS"
    parameter_initialization: Dict[str, List[Any]] = field(default_factory=dict)
    splitarr: Dict[str, str] = field(default_factory=dict)

    # Command-line overrides (set by args, not config file)
    cmd_data: Optional[str] = None
    cmd_sim: Optional[str] = None

    # Runtime flags (set by args, not config file)
    test_run: bool = False
    debug: bool = False
    noweight: bool = False

    @classmethod
    def from_dict(cls, config_dict: dict, args: argparse.Namespace) -> "Config":
        """
        Create Config object from YAML dictionary and command-line arguments.

        Args:
            config_dict: Dictionary loaded from YAML file
            args: Parsed command-line arguments

        Returns:
            Config: Configured dataclass instance
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
            cmd_data=args.CMD_DATA,
            cmd_sim=args.CMD_SIM,
            test_run=args.test_run,
            debug=args.DEBUG or args.test_run,  # SINGLE implies DEBUG
            noweight=args.NOWEIGHT,
        )


# ===========================================================================================================================================
############################################################# IO ###################################################
# ===========================================================================================================================================


def create_output_directories(outdir, logger):
    """
    Create output directory structure for DUST2DUST results.

    Creates the main output directory and required subdirectories:
    - chains: MCMC chain outputs
    - figures: Diagnostic plots
    - parallel: Subprocess communication files
    - logs: Log files

    Args:
        outdir: Path to main output directory (can be relative or absolute)
        logger: Logger instance for output messages

    Returns:
        str: Absolute path to output directory with trailing slash

    Raises:
        SystemExit: If directory structure cannot be created
    """
    # Use current directory if none specified
    if not outdir:
        outdir = os.getcwd()

    # Expand ~ and ensure absolute path
    outdir = os.path.abspath(os.path.expanduser(outdir))

    # Ensure trailing slash
    if not outdir.endswith("/"):
        outdir += "/"

    # Create main directory if it doesn't exist
    if not os.path.exists(outdir):
        logger.debug(f"Creating output directory: {outdir}")
        try:
            os.makedirs(outdir)
        except OSError as e:
            logger.error(f"Could not create directory {outdir}: {e}")
            sys.exit(1)
    else:
        logger.debug(f"Using existing directory: {outdir}")

    # Create required subdirectories
    required_subdirs = ["chains", "figures", "parallel", "logs"]
    for subdir in required_subdirs:
        subdir_path = os.path.join(outdir, subdir)
        if not os.path.exists(subdir_path):
            try:
                os.makedirs(subdir_path)
                logger.debug(f"  Created subdirectory: {subdir}/")
            except OSError as e:
                logger.error(f"Could not create subdirectory {subdir_path}: {e}")
                sys.exit(1)

    # Verify all required subdirectories exist
    missing_dirs = [d for d in required_subdirs if not os.path.isdir(os.path.join(outdir, d))]
    if missing_dirs:
        logger.error(f"Missing required subdirectories: {missing_dirs}")
        logger.error("Required subdirectories: chains, figures, parallel, logs")
        sys.exit(1)

    return outdir


def load_config(config_path: str, args: argparse.Namespace, logger) -> Config:
    """
    Load configuration from YAML file, create Config object, and set up output directories.

    Performs complete configuration setup:
    1. Loads and validates YAML configuration file
    2. Creates Config dataclass instance
    3. Sets up output directory structure
    4. Prints configuration summary

    Args:
        config_path: Path to YAML configuration file
        args: Parsed command-line arguments
        logger: Logger instance for output messages

    Returns:
        Config: Fully configured Config instance with output directories created

    Raises:
        SystemExit: If config file doesn't exist, has invalid syntax, missing required keys,
                   or output directories cannot be created
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
            config_dict = yaml.load(cfgfile, Loader=yaml.FullLoader)
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

    logger.debug(f"Loaded configuration from: {config_path}")

    # Set up output directory structure
    config.outdir = create_output_directories(config.outdir, logger)

    # Log configuration summary
    logger.debug("Configuration finalized successfully.")
    logger.debug(f"  Data: {config.data_input}")
    logger.debug(f"  Simulation: {config.sim_input}")
    logger.debug(f"  Parameters to fit: {', '.join(config.inp_params)}")
    logger.debug(f"  Output directory: {config.outdir}")

    return config


def get_args():
    """
    Parse command-line arguments for DUST2DUST.

    Defines and parses all command-line flags including configuration file path,
    debug modes, plotting options, and SALT2mu command overrides.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="DUST2DUST: MCMC fitting of supernova intrinsic scatter distributions"
    )

    parser.add_argument(
        "--CONFIG", type=str, default="", help="Path to YAML configuration file (required)"
    )

    parser.add_argument(
        "--test_run",
        action="store_true",
        help="Run single likelihood evaluation for testing (does not launch MCMC)",
    )

    parser.add_argument(
        "--DEBUG", action="store_true", help="Enable debug mode with verbose output"
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

    return parser.parse_args()


def main():
    """
    Main entry point for the dust2dusty command-line tool.

    Parses command-line arguments, sets up logging, loads configuration,
    and runs either a test evaluation or full MCMC sampling.
    """
    # Import here to avoid circular imports
    from dust2dusty.dust2dust import (
        MCMC,
        _init_worker,
        init_dust2dust,
        input_cleaner,
        log_probability,
    )

    # Parse arguments and load configuration
    args = get_args()

    # Set up logging before loading config (uses shared logging module)
    DEBUG = args.DEBUG or args.test_run
    setup_logging(debug=DEBUG)
    logger = get_logger()

    # Load and validate configuration
    config = load_config(args.CONFIG, args, logger)

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
