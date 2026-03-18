"""
Command-line interface for DUST2DUSTY.

This module provides the main entry point for running DUST2DUSTY from the
command line, as well as configuration loading and the Config dataclass.

Usage:
    dust2dusty --CONFIG config.yml [--DEBUG] [--DEBUG_FULL] [--TEST_RUN] [--NOWEIGHT]

Example:
    dust2dusty --CONFIG IN_DUST2DUST.yml --DEBUG
    dust2dusty --CONFIG IN_DUST2DUST.yml --DEBUG_FULL
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from dataclasses import _MISSING_TYPE, dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import yaml
from numpy.typing import NDArray

from dust2dusty.log import add_file_handler, get_logger, setup_logging
from dust2dusty.utils import (
    __dust2dust_str__,
    _init_salt2mu_realdata,
    get_sampled_par_names_and_init,
)


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
        fitted_params: List of parameter names to fit.
        params: Initial parameter values for test runs.
        param_dists: Maps parameters to distribution shapes.
        splitdict: Defines parameter splits by host properties.
        splitparam: Primary split parameter name.
        parameter_initialization: Initialization specs for each parameter.
        splitarr: Array generation strings for split variables.
        CMD_DATA: Command-line override for data input.
        CMD_SIM: Command-line override for simulation input.
        TEST_RUN: If True, run single likelihood evaluation only.
        DEBUG: If True, enable debug mode (3 iterations, verbose output,
            SALT2mu FITRES output).
        DEBUG_FULL: If True, run full MCMC with DEBUG-level logging
            but production SALT2mu settings.
        NOWEIGHT: If True, disable reweighting function.

    Class Attributes:
        PARAM_TO_SALT2MU: Maps internal names to SALT2mu column names.
        SUBPROCESS_TO_SNANA: Maps subprocess names to SNANA names.
        DEFAULT_PARAMETER_RANGES: Value grids for PDF generation.
        SPLIT_PARAMETER_FORMATS: Binning specs for split parameters.
        PARAMETER_OVERRIDES: Fixed parameters (not fitted).
        DISTRIBUTION_PARAMETERS: Parameter names for each distribution type.
    """

    # # Split parameter format specifications
    # _SPLIT_PARAMETER_FORMATS: ClassVar[dict[str, str]] = {
    #     "HOST_LOGMASS": "HOST_LOGMASS(2,0:20)",
    #     "HOST_COLOR": "HOST_COLOR(2,-.5:2.5)",
    #     "zHD": "zHD(2,0:1)",
    # }

    # Distribution parameter specifications
    _DISTRIBUTION_PARAMETERS: ClassVar[dict[str, list[str]]] = {
        "Gaussian": ["mu", "std"],
        "Skewed Gaussian": ["mu", "std_low", "std_high"],
        "Exponential": ["tau"],
        "LogNormal": ["ln_mu", "ln_std"],
        "Double Gaussian": ["a_1", "mu_1", "std_1", "mu_2", "std_2"],
    }

    _SALT2MU_EXE = "SALT2mu.exe"  # Default SALT2mu executable name (assumed to be in PATH)

    # File paths (required)
    OUTPUT_DIR: str | Path
    data_input: str | Path
    sim_input: str | Path
    simref_file: str | Path

    # File paths (optional)
    chains: str | None = None

    # Parameter configuration
    salt2mu_genpdf_grid: dict[str, Any] = field(default_factory=dict)
    fitted_params: dict[str, Any] = field(default_factory=dict)
    parameter_inits: dict[str, dict[str, Any]] = field(default_factory=dict)
    splitarr: dict[str, dict[str, Any]] = field(default_factory=dict)

    # - MPI flag
    USE_MPI: bool = False

    # - Runtime flags
    TEST_RUN: bool = False
    DEBUG_RUN: bool = False
    DEBUG_FULL: bool = False
    NOWEIGHT: bool = False
    VERBOSE: bool = False
    SAMPLER: str = "emcee"
    NWALKERS: int | None = None

    @classmethod
    def from_dict(
        cls, config_dict: dict[str, Any], args: argparse.Namespace, USE_MPI=False
    ) -> Config:
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
            data_input=Path(config_dict["DATA_INPUT"]),
            sim_input=Path(config_dict["SIM_INPUT"]),
            simref_file=Path(config_dict["SIMREF_FILE"]),
            # Parameter configuration
            fitted_params=config_dict["FITTED_PARAMS"],
            parameter_inits=config_dict["PARAMETER_INITS"],
            salt2mu_genpdf_grid=config_dict.get("SALT2MU_GENPDF_GRID", {}),
            # Command-line arguments
            OUTPUT_DIR=Path(args.OUTPUT_DIR),
            TEST_RUN=args.TEST_RUN,
            DEBUG_RUN=args.DEBUG_RUN,
            DEBUG_FULL=args.DEBUG_FULL,
            NOWEIGHT=args.NOWEIGHT,
            VERBOSE=args.VERBOSE,
            SAMPLER=args.SAMPLER,
            # Add MPI flag
            USE_MPI=USE_MPI,  # MPI is auto-detected
        )

    def __post_init__(self):
        # Loop through the fields
        for f in fields(self):
            # If there is a default and the value of the field is none we can assign a value
            if not isinstance(f.default, _MISSING_TYPE) and getattr(self, f.name) is None:
                setattr(self, f.name, f.default)

    @property
    def split_pars(self):
        return np.unique(
            sum(
                [list(v["splits"].keys()) for _, v in self.fitted_params.items() if "splits" in v],
                [],
            )
        )

    @property
    def split_dist_par(self):
        split_pars_list = [k for k in self.split_pars if k != "SIM_ZCMB"]
        if len(split_pars_list) > 1:
            raise NotImplementedError(
                "Multiple distribution split parameters not currently supported."
            )
        return split_pars_list[0]


def _create_output_directories(
    outdir: Path, config_file: Path, logger: logging.Logger, force: bool = False
):
    """
    Create output directory structure for DUST2DUSTY results.

    Creates the main output directory and required subdirectories:
        - chains: MCMC chain outputs
        - logs: Log files
        - realdata_files: Real data SALT2mu outputs
        - worker_salt2mu_files: Per-worker SALT2mu subprocess files

    Args:
        outdir: Path to main output directory (can be relative or absolute).
        logger: Logger instance for output messages.
        force: If True, remove existing output directory before creating.

    Returns:
        None.

    Raises:
        FileExistsError: If output directory already exists and force is False.
        SystemExit: If directory structure cannot be created.
    """
    # Use current directory if none specified
    if outdir.exists():
        if force:
            logger.warning(f"Removing existing output directory: {outdir.absolute()}")
            shutil.rmtree(outdir)
        else:
            raise FileExistsError(
                f"Output directory already exists: {outdir.absolute()}. Use --FORCE_OVERRIDE to overwrite."
            )

    logger.debug(f"Create main directory {outdir.absolute()}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Copy config
    shutil.copy2(config_file, outdir)

    # Create required subdirectories
    required_subdirs = [
        "chains",
        "logs",
        "realdata_salt2mu_files",
        "worker_salt2mu_files",
    ]
    for subdir in required_subdirs:
        subdir_path = outdir / subdir
        logger.debug(f"Create sub directory {subdir_path.absolute()}")
        subdir_path.mkdir(parents=True, exist_ok=True)


def _load_config(args: argparse.Namespace, logger: logging.Logger, USE_MPI=False) -> Config:
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
    config_file = Path(args.CONFIG_FILE)
    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_file}")
        sys.exit(1)

    # Load YAML file
    try:
        with open(config_file) as cfgfile:
            config_dict = yaml.safe_load(cfgfile)
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML syntax in {config_file}")
        logger.error(e)
        sys.exit(1)

    # Validate required keys
    required_keys = set(
        [
            "DATA_INPUT",
            "SIM_INPUT",
            "FITTED_PARAMS",
            "PARAMETER_INITS",
            "SIMREF_FILE",
        ]
    )

    missing_keys = [key for key in required_keys if key not in config_dict]

    if missing_keys:
        logger.error(f"Missing required configuration keys: {missing_keys}")
        sys.exit(1)

    # Create Config object from dictionary and args
    config = Config.from_dict(config_dict, args)

    logger.info(f"Loaded configuration from: {config_file}")

    # Set up output directory structure
    _create_output_directories(
        config.OUTPUT_DIR, config_file, logger, force=args.FORCE_OVERRIDE, USE_MPI=USE_MPI
    )

    # Log configuration summary
    logger.info("Configuration finalized successfully:")
    logger.info(f"-- Data: {Path(config.data_input).absolute()}")
    logger.info(f"-- Simulation: {Path(config.sim_input).absolute()}")
    logger.info(f"-- Parameters to fit: {', '.join(config.fitted_params.keys())}")
    logger.info(f"-- Output directory: {Path(config.OUTPUT_DIR).absolute()}")

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
        "CONFIG_FILE",
        type=str,
        help="Path to YAML configuration file (required)",
    )

    parser.add_argument(
        "OUTPUT_DIR",
        type=str,
        help="Path to output directory (required)",
    )

    parser.add_argument(
        "--TEST_RUN",
        action="store_true",
        help="Run single likelihood evaluation for testing (does not launch MCMC)",
    )

    parser.add_argument(
        "--DEBUG_RUN",
        action="store_true",
        help="Run 3 MCMC iterations with DEBUG-level logging output",
    )

    parser.add_argument(
        "--DEBUG_FULL",
        action="store_true",
        help="Run full MCMC with DEBUG-level logging output",
    )

    parser.add_argument(
        "--NOWEIGHT",
        action="store_true",
        help="Disable reweighting function (use for unweighted sims like G10, C11)",
    )

    parser.add_argument(
        "--VERBOSE",
        action="store_true",
        help="Show INFO level logging on terminal (default: only WARNING and above)",
    )

    parser.add_argument(
        "--FORCE_OVERRIDE",
        action="store_true",
        help="Remove and recreate output directory if it already exists",
    )

    parser.add_argument(
        "--SAMPLER",
        type=str,
        default="emcee",
        choices=["emcee", "zeus", "nautilus"],
        help="MCMC sampler to use: 'emcee' (default), 'zeus', or 'nautilus'",
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


def _install_mpi_excepthook() -> None:
    """
    Install a sys.excepthook that calls MPI Abort on unhandled exceptions.

    Without this, a single crashing MPI process leaves the others hanging
    forever (waiting for messages that never arrive), and SLURM never
    kills the job.  MPI.COMM_WORLD.Abort terminates every process in the
    communicator so the job fails immediately.
    """
    from mpi4py import MPI

    _original_hook = sys.excepthook

    def _mpi_excepthook(exc_type, exc_value, exc_tb):
        _original_hook(exc_type, exc_value, exc_tb)
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort(1)

    sys.excepthook = _mpi_excepthook


def main() -> int:
    """
    Main entry point for the dust2dusty command-line tool.

    Parses command-line arguments, sets up logging, loads configuration,
    and runs either a test evaluation or full MCMC sampling.

    Two independent concerns are controlled separately:
        - ``debug`` (from --DEBUG_RUN/--TEST_RUN): controls short-run behavior
          (3 iterations, no HDF5 backend, SALT2mu optmask=1).
        - ``debug_logging`` (--TEST_RUN, or --DEBUG_FULL):
          controls log verbosity (DEBUG level to console and log files).

    For MPI runs, only the master process (rank 0) performs full setup.
    Worker processes (rank > 0) skip heavy initialization and go directly
    to the MCMC function where they wait in the pool.

    Returns:
        Exit code (0 for success).
    """
    # Import here to avoid circular imports
    from dust2dusty.likelihood_worker import (
        _init_worker,
        log_probability,
    )
    from dust2dusty.mcmc import MCMC
    from dust2dusty.utils import input_cleaner

    # Check MPI status early - workers should not do heavy setup
    USE_MPI = False
    rank, size = _get_mpi_info()
    is_master = rank == 0

    # Ensure any unhandled exception on *any* rank aborts the whole MPI job
    # instead of leaving the other ranks hanging forever.
    if size > 1:
        USE_MPI = True
        _install_mpi_excepthook()

    if is_master:
        # Master process (rank 0) does full setup
        args = get_args()
        # Set debug flag
        debug = args.TEST_RUN or args.DEBUG_RUN or args.DEBUG_FULL
        setup_logging(debug=debug, verbose=args.VERBOSE)
        logger = get_logger()
        logger.info(__dust2dust_str__)

        config = _load_config(args, logger, USE_MPI=USE_MPI)
        add_file_handler(str(Path(config.OUTPUT_DIR) / "logs" / "master.log"))
        logger.info("Master log file created.")

        realdata_salt2mu_results = _init_salt2mu_realdata(config, logger, debug=debug)

        # Test run mode - single likelihood evaluation (no MPI needed)
        if config.TEST_RUN:
            _init_worker(config, realdata_salt2mu_results, debug=debug)
            _, p0_init, _, _, _ = get_sampled_par_names_and_init(config)

            logger.info(f"Test run result: {log_probability(p0_init, last=True)}")
            sys.exit(0)

        # Full MCMC run
        MCMC(
            config,
            realdata_salt2mu_results,
            debug=debug,
        )

        logger.info("DUST2DUST(Y) complete.")
    else:
        # Worker processes (rank > 0) go directly to MCMC with None values
        # They will receive config via the pool initializer
        MCMC(None, None, debug=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
