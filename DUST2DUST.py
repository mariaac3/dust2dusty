#!/usr/bin/env python
"""
DUST2DUST: Supernova Cosmology Analysis with MCMC

This module performs Markov Chain Monte Carlo (MCMC) fitting of supernova intrinsic
scatter distributions while accounting for selection effects using reweighting.

The code fits distributions for supernova properties (color, stretch, extinction, etc.)
by comparing real data to reweighted simulations via the SALT2mu.exe executable.

Main workflow:
    1. Load configuration from YAML file specifying parameters to fit
    2. Initialize connections to SALT2mu.exe subprocesses (one per MCMC walker)
    3. Run MCMC using emcee, where each likelihood evaluation:
       - Writes PDF functions for proposed parameters
       - Calls SALT2mu.exe to reweight simulation
       - Compares data vs simulation distributions
    4. Save chains and create diagnostic plots

Key components:
    - Parameter configuration via YAML (distributions, splits, priors)
    - SALT2mu.exe interface via callSALT2mu module
    - Likelihood calculation comparing multiple observables
    - Support for parameter splits by mass, redshift, etc.

Usage:
    python DUST2DUST.py --CONFIG IN_DUST2DUST.yml

    Optional flags:
    --SINGLE: Run single likelihood evaluation for testing
    --GENPDF_ONLY: Generate GENPDF file without running MCMC
    --DOPLOT: Create plots from existing chains
    --DEBUG: Enable verbose output
"""

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from yaml.tokens import FlowSequenceStartToken

import callSALT2mu

matplotlib.use("Agg")
import argparse
import itertools
import logging
import os
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count, current_process
from typing import Any, Dict, List, Optional

import corner
import emcee
import pylab as plt
import yaml

JOBNAME_SALT2mu = "SALT2mu.exe"  # public default code
os.environ["OMP_NUM_THREADS"] = "1"
ncbins = 6

# Module-level configuration object
# Set in main() and accessed throughout the module
# Replaces 20+ individual global variables with a single config object
config: Optional["Config"] = None

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
    - Flags: single, debug, noweight, doplot, genpdf_only
    """

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
    single: bool = False
    debug: bool = False
    noweight: bool = False
    doplot: bool = False
    genpdf_only: bool = False

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
            single=args.SINGLE,
            debug=args.DEBUG or args.SINGLE,  # SINGLE implies DEBUG
            noweight=args.NOWEIGHT,
            doplot=args.DOPLOT,
            genpdf_only=args.GENPDF_ONLY,
        )


# ===========================================================================================================================================
############################################################# IO ###################################################
# ===========================================================================================================================================


def create_output_directories(outdir):
    """
    Create output directory structure for DUST2DUST results.

    Creates the main output directory and required subdirectories:
    - chains: MCMC chain outputs
    - figures: Diagnostic plots
    - parallel: Subprocess communication files
    - logs: Log files

    Args:
        outdir: Path to main output directory (can be relative or absolute)

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
        print(f"Creating output directory: {outdir}")
        try:
            os.makedirs(outdir)
        except OSError as e:
            print(f"ERROR: Could not create directory {outdir}: {e}")
            sys.exit(1)
    else:
        print(f"Using existing directory: {outdir}")

    # Create required subdirectories
    required_subdirs = ["chains", "figures", "parallel", "logs"]
    for subdir in required_subdirs:
        subdir_path = os.path.join(outdir, subdir)
        if not os.path.exists(subdir_path):
            try:
                os.makedirs(subdir_path)
                print(f"  Created subdirectory: {subdir}/")
            except OSError as e:
                print(f"ERROR: Could not create subdirectory {subdir_path}: {e}")
                sys.exit(1)

    # Verify all required subdirectories exist
    missing_dirs = [d for d in required_subdirs if not os.path.isdir(os.path.join(outdir, d))]
    if missing_dirs:
        print(f"ERROR: Missing required subdirectories: {missing_dirs}")
        print("Required subdirectories: chains, figures, parallel, logs")
        sys.exit(1)

    return outdir


def setup_logging():
    """
    Configure logging settings for the main program.

    Sets up basic logging configuration with INFO level and custom format.
    Suppresses verbose output from matplotlib and seaborn.
    """
    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)8s |%(filename)21s:%(lineno)3d]   %(message)s"
    )
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("seaborn").setLevel(logging.ERROR)
    # END setup_logging


def load_config(config_path: str, args: argparse.Namespace) -> Config:
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

    Returns:
        Config: Fully configured Config instance with output directories created

    Raises:
        SystemExit: If config file doesn't exist, has invalid syntax, missing required keys,
                   or output directories cannot be created
    """
    # Validate config file path
    if not config_path:
        print("ERROR: No configuration file specified. Use --CONFIG <path>")
        sys.exit(1)

    if not os.path.exists(config_path):
        print(f"ERROR: Configuration file not found: {config_path}")
        sys.exit(1)

    # Load YAML file
    try:
        with open(config_path, "r") as cfgfile:
            config_dict = yaml.load(cfgfile, Loader=yaml.FullLoader)
    except yaml.YAMLError as e:
        print(f"ERROR: Invalid YAML syntax in {config_path}")
        print(e)
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
        print(f"ERROR: Missing required configuration keys: {missing_keys}")
        sys.exit(1)

    print(f"Loaded configuration from: {config_path}")

    # Create Config object from dictionary and args
    config = Config.from_dict(config_dict, args)

    # Set up output directory structure
    config.outdir = create_output_directories(config.outdir)

    # Print configuration summary
    print("Configuration finalized successfully.")
    print(f"  Data: {config.data_input}")
    print(f"  Simulation: {config.sim_input}")
    print(f"  Parameters to fit: {', '.join(config.inp_params)}")
    print(f"  Output directory: {config.outdir}")

    return config
    # END load_config


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
        "--SINGLE",
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
        "--DOPLOT",
        action="store_true",
        help="Create corner and chain plots from existing chains (requires --CHAINS)",
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
        "--GENPDF_ONLY",
        action="store_true",
        help="Generate GENPDF file and quit (does not run MCMC)",
    )

    args = parser.parse_args()
    return args
    # END get_args


# ===================================================
############### Parameter Configuration Constants
# ===================================================

# Distribution parameter specifications
# Maps distribution types to their required parameter names
DISTRIBUTION_PARAMETERS: Dict[str, List[str]] = {
    "Gaussian": ["mu", "std"],
    "Skewed Gaussian": ["mu", "std_l", "std_r"],
    "Exponential": ["Tau"],
    "LogNormal": ["ln_mu", "ln_std"],
    "Double Gaussian": ["a1", "mu1", "std1", "mu2", "std2"],
}

# Parameter name mappings for SALT2mu format
# Converts internal parameter names to SALT2mu/simulation column names
PARAM_TO_SALT2MU: Dict[str, str] = {
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
SUBPROCESS_TO_SNANA: Dict[str, str] = {
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
DEFAULT_PARAMETER_RANGES: Dict[str, np.ndarray] = {
    "c": np.arange(-0.5, 0.5, 0.001),
    "x1": np.arange(-5, 5, 0.01),
    "RV": np.arange(0, 8, 0.1),
    "EBV": np.arange(0.0, 1.5, 0.02),
    "EBVZ": np.arange(0.0, 1.5, 0.02),
}

# Split parameter format specifications
# Defines how parameters are split into bins for SALT2mu output
# Format: 'PARAM(nbins, min:max)'
SPLIT_PARAMETER_FORMATS: Dict[str, str] = {
    "HOST_LOGMASS": "HOST_LOGMASS(2,0:20)",
    "HOST_COLOR": "HOST_COLOR(2,-.5:2.5)",
    "zHD": "zHD(2,0:1)",
}

# Parameter override dictionary
# Used to fix specific parameters during fitting (not fitted, held constant)
# Populated programmatically based on user input or left empty for standard fitting
PARAMETER_OVERRIDES: Dict[str, float] = {}
# =======================================================
################### FUNCTIONS ##########################
# =======================================================


def thetaconverter(theta):
    """
    Create mapping from input parameters to theta array indices.

    For each parameter in INP_PARAMS, identifies which positions in the theta array
    correspond to that parameter's distribution parameters (after expansion for splits).

    Example:
        If INP_PARAMS = ['c', 'RV'] and expanded params are
        ['c_mu', 'c_std', 'RV_mu_HOST_LOGMASS_low', 'RV_mu_HOST_LOGMASS_high', ...]
        then thetadict['c'] = [0, 1] and thetadict['RV'] = [2, 3, ...]

    Args:
        theta: Array of parameter values (length = ndim)

    Returns:
        dict: Mapping from parameter name to list of indices in theta array
              Key: parameter name (e.g., 'c', 'RV')
              Value: list of integer indices
    """
    thetadict = {}
    extparams = pconv(
        config.inp_params, config.paramshapesdict, config.splitdict
    )  # expanded list of all variables. len is ndim.
    for p in config.inp_params:
        thetalist = []
        for n, ep in enumerate(extparams):
            if p in ep:  # for instance, if 'c' is in 'c_l', then this records that position.
                thetalist.append(n)
        thetadict[p] = thetalist
    return thetadict  # key gives location of relevant parameters in extparams
    # END thetaconverter


def thetawriter(theta, key, names=False):
    """
    Extract subset of theta array corresponding to a specific parameter.

    Uses thetaconverter to identify which elements of theta belong to the
    specified parameter, then returns that slice.

    Args:
        theta: Array of parameter values (length = ndim)
        key: Parameter name (e.g., 'c', 'RV', 'EBV')
        names: If True, returns parameter names instead of values (default: False)

    Returns:
        numpy.ndarray or list: Subset of theta (or parameter names) for this parameter
                               E.g., for 'RV' might return [mu_low, std_low, mu_high, std_high]
    """
    thetadict = thetaconverter(theta)
    lowbound = thetadict[key][0]
    highbound = thetadict[key][-1] + 1
    if names:
        return names[lowbound:highbound]
    else:
        return theta[
            lowbound:highbound
        ]  # Returns theta in the range of first to last index for relevant parameter. For example, inp_param = ['c', 'RV'], thetawriter(theta, 'c') would give theta[0:2] which is ['c_m', 'c_std']


def input_cleaner(INP_PARAMS, PARAMETER_INITIALIZATION, parameter_overrides, walkfactor=2):
    """
    Initialize MCMC walker starting positions with appropriate constraints.

    Generates initial walker positions for emcee sampler, ensuring all parameters
    start within their valid bounds and with appropriate spreads.

    Args:
        INP_PARAMS: List of parameter names to fit (e.g., ['c', 'RV', 'EBV'])
        PARAMETER_INITIALIZATION: Dictionary containing initialization info for each expanded parameter:
                   {param_name: [mean, std, require_positive, [lower_bound, upper_bound]]}
        parameter_overrides: Dictionary of parameters to fix (not fit)
        walkfactor: Multiplier for number of walkers (nwalkers = ndim * walkfactor, default: 2)

    Returns:
        tuple: (pos, nwalkers, ndim)
               pos: array of shape (nwalkers, ndim) with initial walker positions
               nwalkers: number of MCMC walkers
               ndim: number of dimensions (parameters)
    """
    plist = pconv(INP_PARAMS, PARAMSHAPESDICT, SPLITDICT)
    for element in parameter_overrides.keys():
        plist.remove(element)
    pos = np.abs(0.1 * np.random.randn(len(plist) * walkfactor, len(plist)))
    for entry in range(len(plist)):
        newpos_param = PARAMETER_INITIALIZATION[plist[entry]]
        pos[:, entry] = np.random.normal(newpos_param[0], newpos_param[1], len(pos[:, entry]))
        if newpos_param[2]:
            pos[:, entry] = np.abs(pos[:, entry])
        while any(ele < newpos_param[3][0] for ele in pos[:, entry]) or any(
            ele > newpos_param[3][1] for ele in pos[:, entry]
        ):
            pos[:, entry] = np.random.normal(newpos_param[0], newpos_param[1], len(pos[:, entry]))
            if newpos_param[2]:
                pos[:, entry] = np.abs(pos[:, entry])
    return pos, len(plist) * walkfactor, len(plist)
    # END input_cleaner


def pconv(INP_PARAMS, paramshapesdict, splitdict):
    """
    Convert input parameters to expanded parameter list accounting for distribution shapes and splits.

    Takes high-level parameter names and expands them into full list of distribution parameters,
    accounting for:
    1. Distribution shape (Gaussian needs mu+std, Exponential needs tau, etc.)
    2. Parameter splits (e.g., different values for low/high mass, low/high redshift)

    Example:
        INP_PARAMS = ['RV']
        paramshapesdict = {'RV': 'Gaussian'}  # needs mu, std
        splitdict = {'RV': {'HOST_LOGMASS': 10}}  # split at mass=10

        Returns: ['RV_HOST_LOGMASS_low_mu', 'RV_HOST_LOGMASS_low_std',
                  'RV_HOST_LOGMASS_high_mu', 'RV_HOST_LOGMASS_high_std']

    Args:
        INP_PARAMS: List of high-level parameter names (e.g., ['c', 'RV', 'EBV'])
        paramshapesdict: Maps parameter to distribution shape (e.g., {'c': 'Gaussian'})
        splitdict: Nested dict defining parameter splits
                   {param: {split_var: split_value}}
                   e.g., {'RV': {'HOST_LOGMASS': 10, 'SIM_ZCMB': 0.1}}

    Returns:
        list: Expanded parameter names (length = ndim for MCMC)
              Format: 'PARAM_SPLITVAR1_lowhigh_SPLITVAR2_lowhigh_..._DISTRIBUTIONPARAM'
    """
    inpfull = []
    for i in INP_PARAMS:
        initial_dimension = DISTRIBUTION_PARAMETERS[paramshapesdict[i]]
        if i in splitdict.keys():
            things_to_split_on = splitdict[i]  # {"Mass": 10, "z": 0.1}
            nsplits = len(things_to_split_on)  # 2
            params_to_split_on = things_to_split_on.keys()  # ["Mass", "z"]
            # Create format string like "{}_{}_{}_{}" for nsplits*2 parameters
            format_string = "_".join(["{}"] * nsplits * 2)
            lowhigh_array = np.tile(
                ["low", "high"], [nsplits, 1]
            )  # [["low", "high"], ["low", "high"]]
            splitlist = []
            for lowhigh_combo in itertools.product(*lowhigh_array):
                to_format = [val for pair in zip(params_to_split_on, lowhigh_combo) for val in pair]
                final = format_string.format(*to_format)
                splitlist.append(final)
            # initial_dimension = [tmp[0]+'_'+tmp[1] for tmp in itertools.product(initial_dimension,splitlist)]
            initial_dimension = [
                tmp[1] + "_" + tmp[0] for tmp in itertools.product(splitlist, initial_dimension)
            ]
        final_dimension = [i + "_" + s for s in initial_dimension]
        inpfull.append(final_dimension)
    inpfull = [item for sublist in inpfull for item in sublist]
    return inpfull
    # END split_cleaner


def array_conv(inp, SPLITDICT, SPLITARR):
    """
    Generate arrays for PDF evaluation based on parameter and its splits.

    Creates list of arrays needed to evaluate and write PDF functions for a parameter.
    First array is the parameter values, subsequent arrays are split variable values.

    Args:
        inp: Parameter name (e.g., 'c', 'RV', 'EBV')
        SPLITDICT: Dictionary defining splits for this parameter
        SPLITARR: Dictionary mapping split variables to array generation strings
                  (e.g., {'HOST_LOGMASS': 'np.arange(5,15,1)'})

    Returns:
        list: [param_array, split1_array, split2_array, ...]
              Empty list if inp is 'beta' or 'alpha' (handled differently)

    Example:
        For RV split on mass: [[0, 0.1, 0.2, ...], [5, 6, 7, ..., 15]]
    """
    if (inp == "beta") or (inp == "alpha"):
        return []
    arrlist = []
    arrlist.append(DEFAULT_PARAMETER_RANGES[inp])
    if inp in SPLITDICT.keys():
        for s in SPLITDICT[inp].keys():
            arrlist.append(eval((SPLITARR[s])))
    return arrlist
    # END array_conv


def dffixer(df, RET, ifdata):
    """
    Extract binned statistics from SALT2mu output dataframe.

    Parses the pandas dataframe returned by SALT2mu to extract color and x1 histograms,
    Hubble residuals, and scatter statistics split by the config.splitparam variable
    (typically HOST_LOGMASS).

    Args:
        df: pandas DataFrame from SALT2mu output containing binned statistics.
            Expected columns: ibin_c, ibin_x1, ibin_{splitparam}, NEVT, MURES_SUM, STD_ROBUST
        RET: Return type string:
            - 'HIST': Return only histogram counts
            - 'ANALYSIS': Return full statistics dictionary
        ifdata: Boolean indicating if this is real data (True) or simulation (False).
                Currently unused but kept for potential future differentiation.

    Returns:
        If RET == 'HIST':
            tuple: (color_hist, x1_hist) - numpy arrays of histogram counts per bin
        If RET == 'ANALYSIS':
            dict: Dictionary with keys:
                'color_hist': Color histogram counts (array)
                'x1_hist': x1 histogram counts (array, empty if ibin_x1 not in df)
                'mures_high': High-mass Hubble residuals per color bin (array)
                'mures_low': Low-mass Hubble residuals per color bin (array)
                'rms_high': High-mass robust scatter per color bin (array)
                'rms_low': Low-mass robust scatter per color bin (array)
                'nevt_high': High-mass event counts per color bin (array)
                'nevt_low': Low-mass event counts per color bin (array)
        Else:
            str: 'No output'
    """
    cpops = []
    x1pops = []
    rmspops = []

    dflow = df.loc[df[f"ibin_{config.splitparam}"] == 0]
    dfhigh = df.loc[df[f"ibin_{config.splitparam}"] == 1]

    lowNEVT = dflow.NEVT.values
    highNEVT = dfhigh.NEVT.values
    lowrespops = dflow.MURES_SUM.values
    highrespops = dfhigh.MURES_SUM.values

    # Color histogram (existing)
    for q in np.unique(df.ibin_c.values):
        cpops.append(np.sum(df.loc[df.ibin_c == q].NEVT))
    cpops = np.array(cpops)

    # x1 (stretch) histogram
    # Check if x1 bins exist in dataframe
    if "ibin_x1" in df.columns:
        for q in np.unique(df.ibin_x1.values):
            x1pops.append(np.sum(df.loc[df.ibin_x1 == q].NEVT))
        x1pops = np.array(x1pops)
    else:
        # If no x1 bins, return empty array
        x1pops = np.array([])

    lowRMS = dflow.STD_ROBUST.values
    highRMS = dfhigh.STD_ROBUST.values

    if RET == "HIST":
        return cpops, x1pops
    elif RET == "ANALYSIS":
        # Return dictionary structure for cleaner access
        return {
            "color_hist": cpops,
            "x1_hist": x1pops,
            "mures_high": highrespops / dfhigh.NEVT.values,
            "mures_low": lowrespops / dflow.NEVT.values,
            "rms_high": highRMS,
            "rms_low": lowRMS,
            "nevt_high": highNEVT,
            "nevt_low": lowNEVT,
        }
    else:
        return "No output"
    # END dffixer


def LL_Creator(realdata, sim, inparr, returnall=False, RMS_weight=1):
    """
    Calculate log-likelihood by comparing data and simulation observables.

    Computes chi-squared statistics between observed and simulated distributions for:
    - Color (c) histogram
    - Stretch (x1) histogram
    - Hubble residuals (MURES) split by mass (high/low)
    - Hubble residual scatter (RMS) split by mass (high/low)
    - Beta parameter (color-luminosity relation)
    - Intrinsic scatter (sigint)

    Args:
        realdata: SALT2mu object containing real data fit results (beta, betaerr, sigint, siginterr)
        inparr: Dictionary with [data, sim] pairs for each observable:
                Keys: 'color_hist', 'x1_hist', 'mures_high', 'mures_low',
                      'rms_high', 'rms_low', 'nevt_high', 'nevt_low'
                Each value is [real_data, sim_data]
        simbeta: Beta parameter from simulation fit
        simsigint: Intrinsic scatter from simulation fit
        returnall: If True, return detailed components (default: False)
        RMS_weight: Weight factor for RMS terms in likelihood (default: 1)

    Returns:
        If returnall is False:
            float: Total log-likelihood (sum of all components)
        If returnall is True:
            tuple: (LL_dict, datacount_dict, simcount_dict, poisson_dict)
                   LL_dict: Individual chi-squared contributions by observable name
                   datacount_dict: Data values for each observable
                   simcount_dict: Simulation values for each observable
                   poisson_dict: Poisson errors for each observable
    """
    LL_dict = defaultdict(float)

    if returnall:
        # Always create detail dicts (minimal memory overhead)
        # Only return them if returnall=True at the end
        datacount_dict = defaultdict(float)
        simcount_dict = defaultdict(float)
        poisson_dict = defaultdict(float)

    # ========== Parameter likelihood terms ==========
    # Beta (color-luminosity relation)
    print(
        "real beta, sim beta, real beta error", realdata.beta, sim.beta, realdata.betaerr, flush=True
    )

    LL_dict["beta"] = -0.5 * ((realdata.beta - sim.beta) / realdata.betaerr) ** 2

    # Intrinsic scatter
    LL_dict["sigint"] = -0.5 * ((realdata.sigint - sim.sigint) / realdata.siginterr) ** 2

    # ========== Observable distributions ==========
    # Get event counts for error calculations
    nevt_high = inparr["nevt_high"][0]
    nevt_low = inparr["nevt_low"][0]

    # Color histogram
    data_color, sim_color = inparr["color_hist"]
    datacount_color, simcount_color, poisson_color, ww = normhisttodata(data_color, sim_color)
    LL_dict["color_hist"] = -0.5 * np.sum(
        (datacount_color - simcount_color) ** 2 / poisson_color**2
    )
    datacount_dict["color_hist"] = datacount_color
    simcount_dict["color_hist"] = simcount_color
    poisson_dict["color_hist"] = poisson_color

    # X1 (stretch) histogram
    data_x1, sim_x1 = inparr["x1_hist"]
    if len(data_x1) > 0 and len(sim_x1) > 0:
        datacount_x1, simcount_x1, poisson_x1, ww = normhisttodata(data_x1, sim_x1)
        LL_dict["x1_hist"] = -0.5 * np.sum((datacount_x1 - simcount_x1) ** 2 / poisson_x1**2)
        datacount_dict["x1_hist"] = datacount_x1
        simcount_dict["x1_hist"] = simcount_x1
        poisson_dict["x1_hist"] = poisson_x1
    else:
        # Skip if x1 histogram not available
        LL_dict["x1_hist"] = 0.0
        datacount_dict["x1_hist"] = np.array([])
        simcount_dict["x1_hist"] = np.array([])
        poisson_dict["x1_hist"] = np.array([])

    # High-mass MURES
    data_mures_high, sim_mures_high = inparr["mures_high"]
    poisson_mures_high = inparr["rms_high"][0] / np.sqrt(nevt_high)
    LL_dict["mures_high"] = -0.5 * np.sum(
        (data_mures_high - sim_mures_high) ** 2 / poisson_mures_high**2
    )
    datacount_dict["mures_high"] = data_mures_high
    simcount_dict["mures_high"] = sim_mures_high
    poisson_dict["mures_high"] = poisson_mures_high

    # Low-mass MURES
    data_mures_low, sim_mures_low = inparr["mures_low"]
    poisson_mures_low = inparr["rms_low"][0] / np.sqrt(nevt_low)
    LL_dict["mures_low"] = -0.5 * np.sum(
        (data_mures_low - sim_mures_low) ** 2 / poisson_mures_low**2
    )
    datacount_dict["mures_low"] = data_mures_low
    simcount_dict["mures_low"] = sim_mures_low
    poisson_dict["mures_low"] = poisson_mures_low

    # High-mass RMS
    data_rms_high, sim_rms_high = inparr["rms_high"]
    poisson_rms_high = data_rms_high / np.sqrt(2 * nevt_high)
    LL_dict["rms_high"] = (
        -0.5 * np.sum((data_rms_high - sim_rms_high) ** 2 / poisson_rms_high**2) * RMS_weight
    )
    datacount_dict["rms_high"] = data_rms_high
    simcount_dict["rms_high"] = sim_rms_high
    poisson_dict["rms_high"] = poisson_rms_high

    # Low-mass RMS
    data_rms_low, sim_rms_low = inparr["rms_low"]
    poisson_rms_low = data_rms_low / np.sqrt(2 * nevt_low)
    LL_dict["rms_low"] = (
        -0.5 * np.sum((data_rms_low - sim_rms_low) ** 2 / poisson_rms_low**2) * RMS_weight
    )
    datacount_dict["rms_low"] = data_rms_low
    simcount_dict["rms_low"] = sim_rms_low
    poisson_dict["rms_low"] = poisson_rms_low

    # Calculate total log-likelihood with error checking
    # Check for NaN or inf values in any component
    invalid_components = []
    for key, value in LL_dict.items():
        if not np.isfinite(value) or (
            isinstance(value, np.ndarray) and not np.all(np.isfinite(value))
        ):
            invalid_components.append(key)

    if invalid_components:
        print(f"WARNING: Invalid (NaN/inf) likelihood components: {invalid_components}", flush=True)
        print(f"LL_dict values: {LL_dict}", flush=True)
        # Return -inf for MCMC rejection, but still provide detail dicts if requested
        if returnall:
            return LL_dict, datacount_dict, simcount_dict, poisson_dict
        else:
            return -np.inf

    if returnall:
        return sum(LL_dict.values()), LL_dict, datacount_dict, simcount_dict, poisson_dict

    return sum(LL_dict.values())
    # END LL_Creator


def pltting_func(samples, INP_PARAMS, ndim):
    """
    Create chain trace plots and corner plot from MCMC samples.

    Generates two diagnostic plots:
    1. Trace plot: Shows parameter evolution over MCMC steps for all walkers.
       Each parameter gets its own subplot showing walker trajectories.
    2. Corner plot: Shows 1D and 2D marginalized posteriors with correlations
       between all parameter pairs.

    Args:
        samples: MCMC samples array of shape (nsteps, nwalkers, ndim)
        INP_PARAMS: List of input parameter names (used to generate labels via pconv())
        ndim: Number of dimensions (parameters)

    Side effects:
        - Saves chain trace plot to config.outdir/figures/*-chains.pdf
        - Saves corner plot to config.outdir/figures/*-corner.pdf
        - Prints "upload <filepath>" messages for both plots
        - Uses LaTeX rendering for plot labels (text.usetex not explicitly set here)

    Note:
        Parameter labels are generated by expanding INP_PARAMS through pconv(),
        excluding any parameters in PARAMETER_OVERRIDES.
    """
    labels = pconv(INP_PARAMS, config.paramshapesdict, config.splitdict)
    for k in PARAMETER_OVERRIDES.keys():
        labels.remove(k)
    plt.clf()
    fig, axes = plt.subplots(ndim, figsize=(10, 2 * ndim), sharex=True)
    for it in range(ndim):
        ax = axes[it]
        ax.plot(samples[:, :, it], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        # ax.set_ylabel(r'$'+str(labels[it])+'$')
        ax.set_ylabel(it)
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    plt.savefig(
        config.outdir + "figures/" + config.data_input.split(".")[0].split("/")[-1] + "-chains.pdf",
        bbox_inches="tight",
    )
    print("upload " + config.outdir + "figures/chains.pdf")
    plt.close()

    flat_samples = samples.reshape(-1, samples.shape[-1])

    plt.clf()
    fig = corner.corner(flat_samples, labels=labels, smooth=True)
    plt.savefig(
        config.outdir + "figures/" + config.data_input.split(".")[0].split("/")[-1] + "-corner.pdf"
    )
    print("upload " + config.outdir + "figures/corner.pdf")
    plt.close()
    # END pltting_func


def Criteria_Plotter(theta, genpdf_only=False):
    """
    Create diagnostic plots comparing data and simulation for a given parameter set.

    Runs log_likelihood with given parameters and creates a 3-panel figure showing:
    - Panel (a): Color histogram comparison (data points vs simulation line)
    - Panel (b): Hubble residuals vs color, split by high/low mass
    - Panel (c): Hubble residual scatter (RMS) vs color, split by high/low mass

    Each panel includes chi-squared values for the comparison.

    Args:
        theta: Array of parameter values to evaluate (length = ndim)
        genpdf_only: If True, generate GENPDF file and return without plotting.
                    The GENPDF file is renamed and converted for SNANA use.

    Side effects:
        - Creates test connection (ID=299) via init_connection()
        - If genpdf_only=False:
          Saves plot to config.outdir/figures/*overplot_observed_DATA_SIM_OVERVIEW.pdf
        - If genpdf_only=True:
          Renames parallel/299_PYTHONCROSSTALK_OUT.DAT to GENPDF.DAT

    Returns:
        str: 'Done' if plotting successful
        None: If genpdf_only=True (after generating GENPDF file)
        None: If log_likelihood returns invalid results (bad parameters)
    """
    tc = init_connection(299, real=False, debug=True)[1]
    try:
        chisq, datacount_list, simcount_list, poisson_list = log_likelihood(
            (theta), returnall=True, connection=tc, genpdf_only=genpdf_only
        )
    except TypeError:
        if genpdf_only:
            # Read in and change the GENPDF filenames to be appropriate for SNANA-usage. The SUBPROCESS and SIM parameters are not the same by default.
            print(
                "If you did not mean to generate the GENPDF only, something has gone wrong. Otherwise this is working properly."
            )
            os.rename("parallel/299_PYTHONCROSSTALK_OUT.DAT", "GENPDF.DAT")
            return ()
        else:
            print(
                "LL was not returned after running log_likelihood, which is likely due to bad parameters. Will skip plotting."
            )
            return
    cbins = np.linspace(-0.2, 0.25, ncbins)
    chisq = -2 * chisq
    if config.debug:
        print("RESULT!", chisq, flush=True)
    sys.stdout.flush()
    plt.rcParams.update({"text.usetex": True, "font.size": 12})
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    ##### Color Hist
    ax = axs[0]
    ax.errorbar(
        cbins, datacount_list[0], yerr=(poisson_list[0]), fmt="o", c="darkmagenta", label="Data"
    )
    ax.plot(cbins, simcount_list[0], c="dimgray", label="Simulation")
    ax.legend()
    ax.set_xlabel(r"$c$")
    ax.set_ylabel("Count")
    thestring = r"$\chi^2_c =$ " + str(np.around(chisq[0], 1))
    ax.text(
        -0.2,
        50,
        thestring,
    )
    ax.text(-0.2, 450, "a)")
    ###### MURES hi and lo
    ax = axs[1]
    ax.errorbar(
        cbins, datacount_list[1], yerr=(poisson_list[1]), fmt="^", c="k", label="Data, High"
    )
    ax.plot(cbins, simcount_list[1], c="tab:orange", label="Simulation, High", ls="--")
    ax.errorbar(
        cbins, datacount_list[2], yerr=(poisson_list[2]), fmt="s", c="tab:green", label="Data, Low"
    )
    ax.plot(cbins, simcount_list[2], c="tab:blue", label="Simulation, Low")
    ax.legend(bbox_to_anchor=[1.7, 1.2], ncol=2)
    ax.set_xlabel(r"$c$")
    ax.set_ylabel(r"$\mu - \mu_{\rm model}$")
    thestring = r"High $\chi^2_{\mu_{\rm res}} =$ " + str(np.around(chisq[1], 1))
    ax.text(
        -0.2,
        0.205,
        thestring,
    )
    thestring = r"Low $\chi^2_{\mu_{\rm res}} =$ " + str(np.around(chisq[2], 1))
    ax.text(
        -0.2,
        0.18,
        thestring,
    )
    ax.text(-0.2, 0.275, "b)")
    ####### RMS hi and lo
    ax = axs[2]
    ax.errorbar(
        cbins, datacount_list[3], yerr=(poisson_list[3]), fmt="^", c="k", label="REAL DATA HIGH"
    )
    ax.plot(cbins, simcount_list[3], c="tab:orange", label="SIMULATION HI", ls="--")
    ax.errorbar(
        cbins,
        datacount_list[4],
        yerr=(poisson_list[4]),
        fmt="s",
        c="tab:green",
        label="REAL DATA LOW",
    )
    ax.plot(cbins, simcount_list[4], c="tab:blue", label="SIMULATION LOW ")
    ax.set_xlabel(r"$c$")
    ax.set_ylabel(r"$\sigma_{\rm r}$")
    thestring = r"High $\chi^2_{\sigma_{\rm r}} =$ " + str(np.around(chisq[3], 1))
    ax.text(
        -0.2,
        0.42,
        thestring,
    )
    thestring = r"Low $\chi^2_{\sigma_{\rm r}} =$ " + str(np.around(chisq[4], 1))
    ax.text(
        -0.2,
        0.395,
        thestring,
    )
    ax.text(-0.2, 0.48, "c)")
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=None)
    plt.savefig(
        config.outdir
        + "figures/"
        + config.data_input.split(".")[0].split("/")[-1]
        + "overplot_observed_DATA_SIM_OVERVIEW.pdf",
        pad_inches=0.01,
        bbox_inches="tight",
    )
    print("upload " + config.outdir + "figures/overplot_observed_DATA_SIM_OVERVIEW.pdf")
    plt.close()
    return "Done"
    # END Criteria_Plotter


def subprocess_to_snana(OUTDIR, snana_mapping):
    """
    Convert GENPDF file from SUBPROCESS format to SNANA-compatible format.

    Reads GENPDF.DAT file, removes the first line (header), and replaces variable
    names from subprocess format (e.g., 'SIM_c', 'SIM_RV') to SNANA format
    (e.g., 'SALT2c', 'RV') so the file can be used directly in SNANA simulations.

    Args:
        OUTDIR: Output directory containing GENPDF.DAT (should end with '/')
        snana_mapping: Dictionary mapping subprocess names to SNANA names.
                       Uses SUBPROCESS_TO_SNANA constant:
                       {'SIM_c': 'SALT2c', 'SIM_RV': 'RV', 'HOST_LOGMASS': 'LOGMASS', ...}

    Side effects:
        - Removes and recreates GENPDF.DAT file with:
          - First line removed
          - All variable names converted to SNANA format

    Returns:
        str: 'Done' upon completion
    """
    filein = OUTDIR + "GENPDF.DAT"
    f = open(filein, "r")
    lines = f.readlines()
    f.close()
    del lines[0]
    os.remove(filein)
    f = open(filein, "w+")
    for line in lines:
        f.write(line)
    f.close()
    f = open(filein, "r")
    filedata = f.read()
    f.close()
    for i in snana_mapping.keys():
        if i in filedata:
            filedata = filedata.replace(i, snana_mapping[i])
    os.remove(filein)
    f = open(filein, "w")
    f.write(filedata)
    f.close()
    return "Done"
    # END subprocess_to_snana


# =======================================================
################### CONNECTIONS #######################
# =======================================================


def generate_genpdf_varnames(inp_params, splitparam):
    """
    Generate SUBPROCESS_VARNAMES_GENPDF string for SALT2mu from input parameters.

    Builds the comma-separated list of variable names that should be included
    in the GENPDF output file for SNANA simulations. Translates internal parameter
    names to SALT2mu column names using PARAM_TO_SALT2MU mapping.

    Args:
        inp_params: List of parameter names being fit (e.g., ['c', 'RV', 'EBV', 'x1'])
        splitparam: Primary split parameter (e.g., 'HOST_LOGMASS')

    Returns:
        str: Comma-separated SALT2mu variable names
             (e.g., 'SIM_c,HOST_LOGMASS,SIM_RV,SIM_x1,SIM_ZCMB,SIM_beta')

    Example:
        >>> generate_genpdf_varnames(['c', 'RV', 'x1'], 'HOST_LOGMASS')
        'SIM_c,HOST_LOGMASS,SIM_RV,SIM_x1,SIM_ZCMB,SIM_beta'

    Note:
        Always includes SIM_ZCMB and SIM_beta even if not in inp_params,
        as these are required for SALT2mu output.
    """
    varnames = []

    # Add parameter variables in SALT2mu format
    for param in inp_params:
        if param in PARAM_TO_SALT2MU:
            salt2mu_name = PARAM_TO_SALT2MU[param]
            if salt2mu_name not in varnames:  # Avoid duplicates
                varnames.append(salt2mu_name)

    # Add split parameter if not already included
    if splitparam not in varnames:
        varnames.insert(1, splitparam)  # Insert after first parameter

    # Always include redshift and beta if not already present
    if "SIM_ZCMB" not in varnames:
        varnames.append("SIM_ZCMB")
    if "SIM_beta" not in varnames:
        varnames.append("SIM_beta")

    return ",".join(varnames)


def init_connection(index, real=True, debug=False):
    """
    Initialize connection(s) to SALT2mu.exe subprocess(es).

    Creates SALT2mu connection objects for real data and/or simulation.
    Each connection maintains a persistent subprocess that can be called repeatedly
    with different PDF functions for reweighting.

    Args:
        index: Integer ID for this connection (used for file naming in parallel/)
        real: If True, also create connection for real data (default: True)
        debug: If True, use OPTMASK=1 to create FITRES files (default: False)

    Returns:
        tuple: (realdata, connection)
               realdata: SALT2mu object for real data, or None if real=False
               connection: SALT2mu object for simulation

    Side effects:
        - Creates temporary files in config.outdir/parallel/ for subprocess I/O:
          - {index}_SUBPROCESS_REALDATA_OUT.DAT
          - {index}_SUBROCESS_SIM_OUT.DAT
          - {index}_PYTHONCROSSTALK_OUT.DAT
          - {index}_SUBPROCESS_LOG_DATA.STDOUT
          - {index}_SUBPROCESS_LOG_SIM.STDOUT
        - Launches SALT2mu.exe subprocess(es)

    OPTMASK values:
        1: Creates FITRES file (used in DEBUG/SINGLE modes)
        2: Creates M0DIF file
        4: Implements randomseed option (default for production)

    Note:
        Uses config.data_input, config.simref_file, config.inp_params,
        config.splitparam, config.debug to configure SALT2mu command.
    """

    OPTMASK = 4
    directory = "parallel"
    if config.debug:
        OPTMASK = 1
    elif config.single:
        OPTMASK = 1

    realdataout = f"{config.outdir}{directory}/{index}_SUBPROCESS_REALDATA_OUT.DAT"
    Path(realdataout).touch()
    simdataout = f"{config.outdir}{directory}/{index}_SUBROCESS_SIM_OUT.DAT"
    Path(simdataout).touch()
    mapsout = f"{config.outdir}{directory}/{index}_PYTHONCROSSTALK_OUT.DAT"
    Path(mapsout).touch()
    subprocess_log_data = f"{config.outdir}{directory}/{index}_SUBPROCESS_LOG_DATA.STDOUT"
    Path(subprocess_log_data).touch()
    subprocess_log_sim = f"{config.outdir}{directory}/{index}_SUBPROCESS_LOG_SIM.STDOUT"
    Path(subprocess_log_sim).touch()

    # Generate output table specification (color bins x split parameter bins)
    arg_outtable = f"'c(6,-0.2:0.25)*{SPLIT_PARAMETER_FORMATS[config.splitparam]}'"

    # Generate GENPDF variable names from input parameters
    GENPDF_NAMES = generate_genpdf_varnames(config.inp_params, config.splitparam)

    cmd_exe = f"{JOBNAME_SALT2mu} {config.data_input} SUBPROCESS_FILES=%s,%s,%s "
    cmd = cmd_exe + (
        f"SUBPROCESS_VARNAMES_GENPDF={GENPDF_NAMES} "
        f"SUBPROCESS_OUTPUT_TABLE={arg_outtable} "
        f"SUBPROCESS_OPTMASK={OPTMASK} "
        f"SUBPROCESS_SIMREF_FILE={config.simref_file} "
        f"debug_flag=930"
    )
    connection = callSALT2mu.SALT2mu(
        cmd, mapsout, simdataout, subprocess_log_sim, debug=config.debug
    )
    realdata = None
    if real:
        cmd = cmd_exe + f"SUBPROCESS_OUTPUT_TABLE={arg_outtable} debug_flag=930"
        if OPTMASK < 4:
            cmd += f" SUBPROCESS_OPTMASK={OPTMASK}"
        realdata = callSALT2mu.SALT2mu(
            cmd,
            config.outdir + "NOTHING.DAT",
            realdataout,
            subprocess_log_data,
            realdata=True,
            debug=config.debug,
        )
    else:  # connection is an object that is equal to SUBPROCESS_SIM/DATA
        connection.getResult()  # Gets result, as it were
    return realdata, connection
    # END init_connection


def connection_prepare(connection):
    """
    Prepare SALT2mu connection for next likelihood evaluation.

    Increments iteration counter and opens output file for writing PDF functions.

    Args:
        connection: SALT2mu connection object

    Returns:
        SALT2mu: Updated connection object with incremented iteration

    Side effects:
        - Increments connection.iter
        - Opens/truncates crosstalk file for new iteration
    """

    return connection
    # END connection_prepare


def connection_next(connection):
    """
    Finalize current iteration and execute SALT2mu with written PDFs.

    Closes iteration in output file, sends iteration number to SALT2mu subprocess,
    waits for results, and parses output.

    Args:
        connection: SALT2mu connection object

    Returns:
        SALT2mu: Connection object with updated results

    Side effects:
        - Writes iteration end marker to crosstalk file
        - Sends iteration number to SALT2mu stdin
        - Reads and parses SALT2mu output
    """
    connection.write_iterend()
    print("wrote end")
    connection.next()
    print("submitted next iter")
    connection.getResult()
    return connection
    # END connection_next


def normhisttodata(datacount, simcount):
    """
    Normalize simulation histogram to match total counts in data.

    Scales simulation counts to have same total as data, computes Poisson errors,
    and masks bins where both data and sim are zero. This ensures fair comparison
    between data and simulation histograms regardless of total event counts.

    Args:
        datacount: Array of data histogram counts per bin (will be converted to numpy array)
        simcount: Array of simulation histogram counts per bin (will be converted to numpy array)

    Returns:
        tuple: (datacount_masked, simcount_normalized, poisson_errors, mask)
            datacount_masked: Data counts with zero bins removed
            simcount_normalized: Simulation counts scaled by (datatot/simtot), zeros removed
            poisson_errors: sqrt(datacount) per bin, minimum value 1 to avoid division by zero
            mask: Boolean array indicating which bins are non-zero (True = kept)
    """
    datacount = np.array(datacount)
    simcount = np.array(simcount)
    datatot = np.sum(datacount)
    simtot = np.sum(simcount)
    simcount = simcount * datatot / simtot

    ww = (datacount != 0) | (simcount != 0)

    poisson = np.sqrt(datacount)
    poisson[datacount == 0] = 1
    poisson[~np.isfinite(poisson)] = 1
    return datacount[ww], simcount[ww], poisson[ww], ww
    # END normhisttodata


# =======================================================
################### SCIENCE FUNCTIONS ##################
# =======================================================


def log_likelihood(realdata, connections, theta, returnall : bool = False):
    """
    Calculate log-likelihood for proposed parameter values.

    Core likelihood function for MCMC. For each parameter set:
    1. Writes PDF functions to file via connection.write_generic_PDF()
    2. Calls SALT2mu.exe to reweight simulation with those PDFs
    3. Parses binned output from SALT2mu (color histograms, MURES, RMS by mass)
    4. Compares reweighted simulation to real data via LL_Creator()

    Args:
        theta: Array of parameter values (length = ndim)
        connection: SALT2mu connection object. If False, retrieves from global
                   connections list based on current process identity.
        returnall: If True, return detailed likelihood components (default: False)
        genpdf_only: If True, write GENPDF file and return without running SALT2mu.
                    Used for generating input files for SNANA simulations.

    Returns:
        float: Log-likelihood value
        If returnall=True: tuple of (LL_dict, datacount_dict, simcount_dict, poisson_dict)
        Returns -inf if:
            - MAXPROB > 1.001 (PDF hitting boundary of bounding function)
            - Beta is NaN (SALT2mu fit failed)
            - Subprocess error or BrokenPipeError occurs
        Returns None if genpdf_only=True (after writing GENPDF file)

    Side effects:
        - Writes PDF functions to connection's crosstalk file
        - Calls SALT2mu.exe subprocess via connection_next()
        - May regenerate connection if BrokenPipeError occurs (production mode)
        - Prints diagnostic information during execution

    Module-level variables used:
        config: Config object with inp_params, splitdict, paramshapesdict, splitarr, debug
        connections: List of SALT2mu connection objects (one per walker)
        realdata: SALT2mu object containing real data results
        DISTRIBUTION_PARAMETERS, PARAM_TO_SALT2MU: Parameter mapping dictionaries
    """
    print(f"Current PID is {os.getpid()}")
    # Get walker connection
    connection = connections[(current_process()._identity[0] - 1)]

    # Cycle iteration
    connection.iter += 1  # tick up iteration by one
    connection.write_iterbegin()  # open SOMETHING.DAT for that iteration

    print("writing 1d pdf", flush=True)

    for inp in config.inp_params:  # TODO - need to generalise to 2d functions as well
        connection.write_generic_PDF(
            inp,
            config.splitdict,
            thetawriter(theta, inp),
            config.paramshapesdict[inp],
            DISTRIBUTION_PARAMETERS,
            PARAM_TO_SALT2MU,
            array_conv(inp, config.splitdict, config.splitarr),
        )

    print("next", flush=True)

    # NOW RUN SALT2mu with these new distributions
    connection.write_iterend()
    print("wrote end")
    connection.next()
    print("submitted next iter")
    connection.getResult()
    print("got result", flush=True)

    if connection.maxprob > 1.001:
        print(
            connection.maxprob,
            "MAXPROB parameter greater than 1! Coming up against the bounding function! Returning -np.inf to account, caught right after connection",
            flush=True,
        )
        return -np.inf

    # try:
    #     if np.isnan(connection.beta):
    #         print("WARNING! oops negative infinity!")
    #         newcon = (
    #             current_process()._identity[0] - 1
    #         )  # % see above at original connection generator, this has been changed
    #         tc = init_connection(newcon, real=False)[1]
    #         connections[newcon] = tc
    #         return -np.inf
    # except AttributeError:
    #     print("WARNING! We tripped an AttributeError here.")
    #     if config.debug:
    #         print("inp", tempin)
    #         return -np.inf
    #     else:
    #         newcon = (
    #             current_process()._identity[0] - 1
    #         )  # % see above at original connection generator, this has been changed
    #         tc = init_connection(newcon, real=False)[1]
    #         connections[newcon] = tc
    #         return -np.inf

    # ANALYSIS returns c, highres, lowres, rms
    print("Right before calculation", flush=True)
    bindf = connection.bindf.dropna() # THIS IS THE PANDAS DATAFRAME OF THE OUTPUT FROM SALT2mu
    sim_vals = dffixer(bindf, "ANALYSIS", False)

    realbindf = realdata.bindf .dropna() # same for the real data (was a global variable)
    real_vals = dffixer(realbindf, "ANALYSIS", True)

    # Build dictionary pairing data and simulation values
    inparr = {key: [real_vals[key], sim_vals[key]] for key in real_vals.keys()}

    # except Exception as e:
    #     print(e)
    #     print("WARNING! something went wrong in reading in stuff for the LL calc")
    #     return -np.inf
    # except BrokenPipeError:
    #     if DEBUG:
    #         print("WARNING! we landed in a Broken Pipe error")
    #         quit()
    #     else:
    #         print("WARNING! Slurm Broken Pipe Error!")  # REGENERATE THE CONNECTION
    #         print("before regenerating")
    #         newcon = (
    #             current_process()._identity[0] - 1
    #         )  # % see above at original connection generator, this has been changed
    #         tc = init_connection(newcon, real=False)[1]
    #         connections[newcon] = tc
    #         return log_likelihood(theta, connection=tc)

    print("Right before calling LL Creator", flush=True)

    out_result = LL_Creator(realdata, connection, inparr, returnall=returnall)
    # print(
    #     "for ",
    #     pconv(INP_PARAMS, PARAMSHAPESDICT, SPLITDICT),
    #     " parameters = ",
    #     theta,
    #     "we found an LL of",
    #     out_result, flush=True
    # )
    return out_result
    # END log_likelihood


def log_prior(theta, debug=False):
    """
    Calculate log-prior probability for parameter values.

    Checks if all parameters are within their allowed bounds specified in
    config.parameter_initialization. Uses uniform (flat) priors within bounds,
    returning 0 (log(1)) if all parameters are valid or -inf if any parameter
    is outside its allowed range.

    Args:
        theta: Array of parameter values (length = ndim)
        debug: Unused parameter kept for backwards compatibility. Debug output
               is controlled by config.debug instead.

    Returns:
        float: 0.0 if all parameters within bounds, -np.inf otherwise

    Module-level variables used:
        config: Config object with inp_params, paramshapesdict, splitdict,
               parameter_initialization, and debug flag
    """
    thetadict = thetaconverter(theta)
    plist = pconv(config.inp_params, config.paramshapesdict, config.splitdict)
    if config.debug:
        print("plist", plist)
    tlist = False  # if all parameters are good, this remains false
    for key in thetadict.keys():
        if config.debug:
            print("key", key)
        temp_ps = thetawriter(
            theta, key
        )  # I hate this but it works. Creates expanded list for this parameter
        if config.debug:
            print("temp_ps", temp_ps)
        plist_n = thetawriter(theta, key, names=plist)
        for t in range(len(temp_ps)):  # then goes through
            if config.debug:
                print("plist name", plist_n[t])
            lowb = config.parameter_initialization[plist_n[t]][3][0]
            highb = config.parameter_initialization[plist_n[t]][3][1]
            if config.debug:
                print(lowb, temp_ps[t], highb)
            if not lowb < temp_ps[t] < highb:  # and compares to valid boundaries.
                tlist = True
    if tlist:
        return -np.inf
    else:
        return 0
    # END log_prior


def log_probability(theta):
    """
    Calculate log-probability (posterior) for MCMC sampling.

    Combines log-prior and log-likelihood following Bayes' theorem:
    log P(theta|data) = log P(theta) + log P(data|theta)

    This is the main function passed to emcee.EnsembleSampler. It first checks
    the prior (parameter bounds), and only evaluates the expensive likelihood
    if parameters are within bounds.

    Args:
        theta: Array of parameter values (length = ndim)

    Returns:
        float: Log-posterior probability (log_prior + log_likelihood)
               Returns -np.inf if prior returns -inf (parameter out of bounds)
               or if likelihood evaluation fails
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        print("WARNING! We returned -inf from small parameters!")
        sys.stdout.flush()
        return -np.inf
    else:
        sys.stdout.flush()
        return lp + log_likelihood(connections[(current_process()._identity[0] - 1)], theta)
    # END log_probability


def init_dust2dust():
    """
    Initialize DUST2DUST by running SALT2mu on real data.

    Runs SALT2mu on real data to get baseline values for beta, betaerr,
    sigint, and siginterr that will be compared against in likelihood calculations.
    This establishes the "truth" values from observed data.

    Returns:
        SALT2mu: Connection object containing real data fit results with attributes:
                 - beta: Color-luminosity parameter
                 - betaerr: Uncertainty on beta
                 - sigint: Intrinsic scatter
                 - siginterr: Uncertainty on sigint
                 - bindf: Pandas DataFrame with binned statistics

    Side effects:
        - Creates SALT2mu connection with ID=299 (DEBUG/SINGLE/DOPLOT modes)
          or ID=0 (production mode)
        - Launches SALT2mu.exe subprocess for real data

    Note:
        Uses module-level DEBUG, SINGLE, DOPLOT flags to determine connection ID.
    """

    index = 0
    if DEBUG or SINGLE or DOPLOT:
        index = 299

    realdata, _ = init_connection(index, real=True, debug=DEBUG)

    return realdata
    # END init_dust2dust


def init_connections(nwalkers: int, DEBUG=False):
    """
    Initialize SALT2mu subprocess connections for all MCMC walkers.

    Creates one SALT2mu connection per walker (or 1 in DEBUG mode).
    Each connection is a persistent subprocess that can be called repeatedly
    during MCMC sampling for likelihood evaluations.

    Args:
        nwalkers: Number of MCMC walkers (connections to create)
        DEBUG: If True, create only 1 connection for testing (default: False)

    Returns:
        list: List of SALT2mu connection objects, one per walker (or 1 if DEBUG)

    Side effects:
        - Launches nwalkers SALT2mu.exe subprocesses (1 if DEBUG)
        - Creates temporary files for each connection in config.outdir/parallel/
        - Prints progress messages for each walker initialized
    """
    connections = []
    if DEBUG:
        print("we are in debug mode now")
        nwalkers = 1
    for i in range(nwalkers):
        print("generated", i, "walker.")
        sys.stdout.flush()
        _, tc = init_connection(i, real=False, debug=DEBUG)  # set this back to DEBUG=DEBUG
        connections.append(tc)
    print("Done initialising walkers")
    return connections
    # END init_connections


def MCMC(nwalkers, ndim):
    """
    Run MCMC sampling using emcee ensemble sampler.

    Executes MCMC in blocks of iterations, saving chains after each block.
    Uses multiprocessing Pool to parallelize walker evaluations across
    all available CPU cores.

    Args:
        nwalkers: Number of MCMC walkers
        ndim: Number of parameters (dimensions)

    Returns:
        str: "Hi!" upon completion

    Side effects:
        - Runs MCMC for 50 blocks of iterations:
          - First block: 100 steps (burn-in)
          - Subsequent blocks: 300 steps each
          - Total: 100 + 49*300 = 14,800 steps
        - Saves chains to config.outdir/chains/*-samples.npz after each block
        - Progress bar displayed via emcee's progress=True

    Module-level variables used:
        pos: Initial walker positions array of shape (nwalkers, ndim)
        config: Config object with outdir and data_input for output naming
        log_probability: Function evaluating posterior for each walker
    """
    with Pool(nwalkers) as pool:
        # Instantiate the sampler once (in parallel)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
        for qb in range(50):
            print("Starting loop iteration", qb)
            print("begun", cpu_count(), "CPUs with", nwalkers, ndim, "walkers and dimensions")
            sys.stdout.flush()
            # Run the sampler
            if qb == 0:
                state2 = sampler.run_mcmc(pos, 100, progress=True)
            else:
                state2 = sampler.run_mcmc(None, 300, progress=True)
            sys.stdout.flush()
            # Save the output for later
            samples = sampler.get_chain()
            np.savez(
                config.outdir
                + "chains/"
                + config.data_input.split(".")[0].split("/")[-1]
                + "-samples.npz",
                samples,
            )
            # pltting_func(samples, INP_PARAMS, ndim)
    return "Hi!"
    # END MCMC


# =================================================================================================
###############################
# =================================================================================================

if __name__ == "__main__":
    #    try:
    #        setup_logging()
    #        logging.info("# ========== BEGIN DUST2DUST ===============")

    # Parse arguments and load configuration
    args = get_args()
    # Set module-level config (replaces 20+ individual globals)
    config = load_config(args.CONFIG, args)

    # Initialize MCMC
    pos, nwalkers, ndim = input_cleaner(
        config.inp_params, config.parameter_initialization, PARAMETER_OVERRIDES, walkfactor=3
    )
    realdata = init_dust2dust()

    # Handle different run modes
    if config.single:
        if len(config.params) != ndim:
            print(f"ERROR: Parameter count mismatch. Expected {ndim}, got {len(config.params)}")
            print("Quitting to avoid confusion.")
            sys.exit(1)
        print(f"Running single likelihood evaluation with parameters: {config.params}")
        Criteria_Plotter(config.params)
        sys.exit(0)

    elif config.genpdf_only:
        print("Generating GENPDF file...")
        if len(config.params) != ndim:
            print(f"ERROR: Parameter count mismatch. Expected {ndim}, got {len(config.params)}")
            print("Quitting to avoid confusion.")
            sys.exit(1)
        print(f"Input parameters: {config.params}")
        Criteria_Plotter(config.params, genpdf_only=True)
        subprocess_to_snana(config.outdir, SUBPROCESS_TO_SNANA)
        sys.exit(0)

    elif config.doplot:
        if not config.chains:
            print("ERROR: No chains file specified. Use CHAINS: <path> in config file.")
            print("Quitting gracefully.")
            sys.exit(1)
        print("DOPLOT mode enabled. Creating plots from existing chains...")
        old_chains = np.load(config.chains)
        past_results = old_chains.f.arr_0
        Criteria_Plotter(config.params)
        pltting_func(past_results, config.params, ndim)
        print("Plotting complete.")
        sys.exit(0)

    # Run full MCMC
    print("\n" + "=" * 60)
    print("Starting MCMC sampling...")
    print(f"  Walkers: {nwalkers}")
    print(f"  Dimensions: {ndim}")
    print(f"  Parameters: {', '.join(config.inp_params)}")
    print("=" * 60 + "\n")

    connections = init_connections(nwalkers)
    MCMC(nwalkers, ndim)
    # except Exception as e:
    #    logging.exception(e)
    #    raise e
# end:
