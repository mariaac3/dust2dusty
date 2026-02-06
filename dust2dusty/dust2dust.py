"""
DUST2DUSTY: Supernova Cosmology Analysis with MCMC.

This module performs Markov Chain Monte Carlo (MCMC) fitting of supernova
intrinsic scatter distributions while accounting for selection effects
using reweighting.

The code fits distributions for supernova properties (color, stretch,
extinction, etc.) by comparing real data to reweighted simulations via
the SALT2mu.exe executable.

Main Workflow:
    1. Load configuration from YAML file specifying parameters to fit
    2. Initialize connections to SALT2mu.exe subprocesses (one per MCMC walker)
    3. Run MCMC using emcee, where each likelihood evaluation:
       - Writes PDF functions for proposed parameters
       - Calls SALT2mu.exe to reweight simulation
       - Compares data vs simulation distributions
    4. Save chains and create diagnostic plots

Key Components:
    - Parameter configuration via YAML (distributions, splits, priors)
    - SALT2mu.exe interface via salt2mu module
    - Likelihood calculation comparing multiple observables
    - Support for parameter splits by mass, redshift, etc.

Usage:
    python dust2dust.py --CONFIG IN_DUST2DUST.yml

    Optional flags:
        --TEST_RUN: Run single likelihood evaluation for testing
        --DEBUG: Enable verbose output
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dust2dusty.utils import cmd_exe, normhisttodata, pconv, set_numpy_threads

# Call BEFORE importing numpy
set_numpy_threads(4)


import numpy as np
import pandas as pd
from numpy.typing import NDArray

from dust2dusty.logging import get_logger
from dust2dusty.salt2mu import SALT2mu

if TYPE_CHECKING:
    from dust2dusty.cli import Config

# =============================================================================
# GLOBAL VARIABLES & CONSTANTS
# =============================================================================

# Constants
JOBNAME_SALT2MU: str = "SALT2mu.exe"
NCBINS: int = 6  # Number of color bins

# Module-level logger
logger: logging.Logger = get_logger()

# Worker-local global variables for multiprocessing
# These are set by _init_worker() for each Pool worker process
_WORKER_REALDATA_SALT2MU_RESULTS: dict[str, Any] | None = None
_WORKER_SALT2MU_CONNECTION: SALT2mu | None = None
_WORKER_DEBUGFLAG: bool = False
_WORKER_INDEX: int | None = None
_CONFIG: Config | None = None


# =============================================================================
# PARAMETER CONVERSION HELPERS
# =============================================================================


def thetaconverter(theta: NDArray[np.float64]) -> dict[str, list[int]]:
    """
    Create mapping from input parameters to theta array indices.

    For each parameter in inp_params, identifies which positions in the
    theta array correspond to that parameter's distribution parameters
    (after expansion for splits).

    Example:
        If inp_params = ['c', 'RV'] and expanded params are
        ['c_mu', 'c_std', 'RV_mu_HOST_LOGMASS_low', 'RV_mu_HOST_LOGMASS_high', ...]
        then thetadict['c'] = [0, 1] and thetadict['RV'] = [2, 3, ...]

    Args:
        theta: Array of parameter values (length = ndim).

    Returns:
        Mapping from parameter name to list of indices in theta array.
    """
    thetadict: dict[str, list[int]] = {}
    extparams = pconv(
        _CONFIG.inp_params,
        _CONFIG.paramshapesdict,
        _CONFIG.splitdict,
        _CONFIG.DISTRIBUTION_PARAMETERS,
    )
    for p in _CONFIG.inp_params:
        thetalist: list[int] = []
        for n, ep in enumerate(extparams):
            if p in ep:
                thetalist.append(n)
        thetadict[p] = thetalist
    return thetadict


def thetawriter(
    theta: NDArray[np.float64],
    key: str,
    names: bool | list[str] = False,
) -> NDArray[np.float64] | list[str]:
    """
    Extract subset of theta array corresponding to a specific parameter.

    Uses thetaconverter to identify which elements of theta belong to the
    specified parameter, then returns that slice.

    Args:
        theta: Array of parameter values (length = ndim).
        key: Parameter name (e.g., 'c', 'RV', 'EBV').
        names: If a list, returns parameter names instead of values.

    Returns:
        Subset of theta values or parameter names for this parameter.
        E.g., for 'RV' might return [mu_low, std_low, mu_high, std_high].
    """
    thetadict = thetaconverter(theta)
    lowbound = thetadict[key][0]
    highbound = thetadict[key][-1] + 1
    if isinstance(names, list):
        return names[lowbound:highbound]
    else:
        return theta[lowbound:highbound]


def array_conv(
    inp: str,
    splitdict: dict[str, dict[str, float]],
    splitarr: dict[str, str],
) -> list[NDArray[np.float64]]:
    """
    Generate arrays for PDF evaluation based on parameter and its splits.

    Creates list of arrays needed to evaluate and write PDF functions for a
    parameter. First array is the parameter values, subsequent arrays are
    split variable values.

    Args:
        inp: Parameter name (e.g., 'c', 'RV', 'EBV').
        splitdict: Dictionary defining splits for this parameter.
        splitarr: Dictionary mapping split variables to array generation
            strings (e.g., {'HOST_LOGMASS': 'np.arange(5,15,1)'}).

    Returns:
        List of [param_array, split1_array, split2_array, ...].
        Empty list if inp is 'beta' or 'alpha' (handled differently).

    Example:
        For RV split on mass: [[0, 0.1, 0.2, ...], [5, 6, 7, ..., 15]]
    """
    if inp in ("beta", "alpha"):
        return []
    arrlist: list[NDArray[np.float64]] = []
    arrlist.append(_CONFIG.DEFAULT_PARAMETER_RANGES[inp])
    if inp in splitdict.keys():
        for s in splitdict[inp].keys():
            arrlist.append(eval(splitarr[s]))
    return arrlist


# =============================================================================
# DATA PROCESSING
# =============================================================================


def dffixer(
    df: pd.DataFrame,
    return_type: str,
) -> tuple[NDArray, NDArray] | dict[str, NDArray] | str:
    """
    Extract binned statistics from SALT2mu output dataframe.

    Parses the pandas dataframe returned by SALT2mu to extract color and
    x1 histograms, Hubble residuals, and scatter statistics split by the
    splitparam variable (typically HOST_LOGMASS).

    Args:
        df: pandas DataFrame from SALT2mu output containing binned statistics.
            Expected columns: ibin_c, ibin_x1, ibin_{splitparam}, NEVT,
            MURES_SUM, STD_ROBUST.
        return_type: Return type string:
            - 'HIST': Return only histogram counts
            - 'ANALYSIS': Return full statistics dictionary

    Returns:
        If return_type == 'HIST':
            Tuple of (color_hist, x1_hist) - numpy arrays of histogram counts.
        If return_type == 'ANALYSIS':
            Dictionary with keys: 'color_hist', 'x1_hist', 'mures_high',
            'mures_low', 'rms_high', 'rms_low', 'nevt_high', 'nevt_low'.
        Otherwise:
            String 'No output'.
    """
    cpops: list[float] = []
    x1pops: list[float] = []

    dflow = df.loc[df[f"ibin_{_CONFIG.splitparam}"] == 0]
    dfhigh = df.loc[df[f"ibin_{_CONFIG.splitparam}"] == 1]

    lowNEVT = dflow.NEVT.values
    highNEVT = dfhigh.NEVT.values
    lowrespops = dflow.MURES_SUM.values
    highrespops = dfhigh.MURES_SUM.values

    # Color histogram
    for q in np.unique(df.ibin_c.values):
        cpops.append(np.sum(df.loc[df.ibin_c == q].NEVT))
    cpops_arr = np.array(cpops)

    # x1 (stretch) histogram
    if "ibin_x1" in df.columns:
        for q in np.unique(df.ibin_x1.values):
            x1pops.append(np.sum(df.loc[df.ibin_x1 == q].NEVT))
        x1pops_arr = np.array(x1pops)
    else:
        x1pops_arr = np.array([])

    lowRMS = dflow.STD_ROBUST.values
    highRMS = dfhigh.STD_ROBUST.values

    if return_type == "HIST":
        return cpops_arr, x1pops_arr
    elif return_type == "ANALYSIS":
        return {
            "color_hist": cpops_arr,
            "x1_hist": x1pops_arr,
            "mures_high": highrespops / dfhigh.NEVT.values,
            "mures_low": lowrespops / dflow.NEVT.values,
            "rms_high": highRMS,
            "rms_low": lowRMS,
            "nevt_high": highNEVT,
            "nevt_low": lowNEVT,
        }
    else:
        return "No output"


# =============================================================================
# SALT2MU CONNECTION MANAGEMENT
# =============================================================================


def generate_genpdf_varnames(inp_params: list[str], splitparam: str) -> str:
    """
    Generate SUBPROCESS_VARNAMES_GENPDF string for SALT2mu.

    Builds the comma-separated list of variable names that should be included
    in the GENPDF output file for SNANA simulations. Translates internal
    parameter names to SALT2mu column names using PARAM_TO_SALT2MU mapping.

    Args:
        inp_params: List of parameter names being fit (e.g., ['c', 'RV', 'x1']).
        splitparam: Primary split parameter (e.g., 'HOST_LOGMASS').

    Returns:
        Comma-separated SALT2mu variable names.
        Example: 'SIM_c,HOST_LOGMASS,SIM_RV,SIM_x1,SIM_ZCMB,SIM_beta'

    Note:
        Always includes SIM_ZCMB and SIM_beta even if not in inp_params,
        as these are required for SALT2mu output.
    """
    varnames: list[str] = []

    # Add parameter variables in SALT2mu format
    for param in inp_params:
        if param in _CONFIG.PARAM_TO_SALT2MU:
            salt2mu_name = _CONFIG.PARAM_TO_SALT2MU[param]
            if salt2mu_name not in varnames:
                varnames.append(salt2mu_name)

    # Add split parameter if not already included
    if splitparam not in varnames:
        varnames.insert(1, splitparam)

    # Always include redshift and beta if not already present
    if "SIM_ZCMB" not in varnames:
        varnames.append("SIM_ZCMB")
    if "SIM_beta" not in varnames:
        varnames.append("SIM_beta")

    return ",".join(varnames)


def get_worker_index() -> int:
    """
    Get worker rank/index for MPI.

    Returns MPI rank when running under MPI, or 0 for serial execution.

    Returns:
        MPI rank (0 for master, >0 for workers) or 0 if MPI not available.
    """
    try:
        from mpi4py import MPI

        return MPI.COMM_WORLD.Get_rank()
    except ImportError:
        return 0


def init_salt2mu_worker_connection() -> SALT2mu:
    """
    Initialize connection to SALT2mu.exe subprocess for this worker.

    Creates a SALT2mu connection object for simulation. Each connection
    maintains a persistent subprocess that can be called repeatedly with
    different PDF functions for reweighting.

    Uses module-level globals _CONFIG, _WORKER_INDEX, and _WORKER_DEBUGFLAG
    which must be set by _init_worker() before calling this function.

    Returns:
        SALT2mu connection object for simulation.

    Side Effects:
        - Creates temporary files in config.outdir/worker_files/ for subprocess I/O
        - Launches SALT2mu.exe subprocess

    Note:
        OPTMASK values:
        - 1: Creates FITRES file (used in DEBUG modes)
        - 2: Creates M0DIF file
        - 4: Implements randomseed option (default for production)
    """
    optmask = 4
    directory = "worker_files"
    # if _WORKER_DEBUGFLAG:
    #     optmask = 1

    outdir = Path(_CONFIG.outdir)
    sim_data_out = outdir / f"{directory}/{_WORKER_INDEX}_SUBPROCESS_SIM_OUT.DAT"
    sim_data_out.touch()

    maps_out = outdir / f"{directory}/{_WORKER_INDEX}_PYTHONCROSSTALK_OUT.DAT"
    maps_out.touch()

    subprocess_log_sim = outdir / f"{directory}/{_WORKER_INDEX}_SUBPROCESS_LOG_SIM.STDOUT"
    subprocess_log_sim.touch()

    # Generate output table specification (color bins x split parameter bins)
    arg_outtable = f"'c(6,-0.2:0.25)*{_CONFIG.SPLIT_PARAMETER_FORMATS[_CONFIG.splitparam]}'"

    # Generate GENPDF variable names from input parameters
    genpdf_names = generate_genpdf_varnames(_CONFIG.inp_params, _CONFIG.splitparam)

    cmd = cmd_exe(JOBNAME_SALT2MU, _CONFIG.sim_input) + (
        f"SUBPROCESS_VARNAMES_GENPDF={genpdf_names} "
        f"SUBPROCESS_OUTPUT_TABLE={arg_outtable} "
        f"SUBPROCESS_OPTMASK={optmask} "
        f"SUBPROCESS_SIMREF_FILE={_CONFIG.simref_file} "
        f"debug_flag=930"
    )

    connection = SALT2mu(cmd, maps_out, sim_data_out, subprocess_log_sim, debug=_WORKER_DEBUGFLAG)

    return connection


# =============================================================================
# LIKELIHOOD & PRIOR FUNCTIONS
# =============================================================================


def compute_and_sum_loglikelihoods(
    inparr: dict[str, list[NDArray]],
    returnall: bool = False,
    rms_weight: float = 1.0,
) -> float | tuple[float, dict, dict, dict, dict]:
    """
    Calculate log-likelihood by comparing data and simulation observables.

    Computes chi-squared statistics between observed and simulated
    distributions for:
    - Color (c) histogram
    - Stretch (x1) histogram
    - Hubble residuals (MURES) split by mass (high/low)
    - Hubble residual scatter (RMS) split by mass (high/low)
    - Beta parameter (color-luminosity relation)
    - Intrinsic scatter (sigint)

    Args:
        inparr: Dictionary with [data, sim] pairs for each observable.
            Keys: 'color_hist', 'x1_hist', 'mures_high', 'mures_low',
            'rms_high', 'rms_low', 'nevt_high', 'nevt_low'.
            Each value is [real_data, sim_data].
        returnall: If True, return detailed components.
        rms_weight: Weight factor for RMS terms in likelihood.

    Returns:
        If returnall is False:
            Total log-likelihood (sum of all components).
        If returnall is True:
            Tuple of (total_ll, ll_dict, datacount_dict, simcount_dict, poisson_dict).
    """
    ll_dict: dict[str, float] = defaultdict(float)
    datacount_dict: dict[str, Any] = defaultdict(float)
    simcount_dict: dict[str, Any] = defaultdict(float)
    poisson_dict: dict[str, Any] = defaultdict(float)

    # ========== Parameter likelihood terms ==========
    # Beta (color-luminosity relation)
    logger.debug(
        f"real beta, sim beta, real beta error: "
        f"{_WORKER_REALDATA_SALT2MU_RESULTS['beta']}, "
        f"{_WORKER_SALT2MU_CONNECTION.salt2mu_results['beta']}, "
        f"{_WORKER_REALDATA_SALT2MU_RESULTS['betaerr']}"
    )

    ll_dict["beta"] = (
        -0.5
        * (
            (
                _WORKER_REALDATA_SALT2MU_RESULTS["beta"]
                - _WORKER_SALT2MU_CONNECTION.salt2mu_results["beta"]
            )
            / _WORKER_REALDATA_SALT2MU_RESULTS["betaerr"]
        )
        ** 2
    )

    # Intrinsic scatter
    ll_dict["sigint"] = (
        -0.5
        * (
            (
                _WORKER_REALDATA_SALT2MU_RESULTS["sigint"]
                - _WORKER_SALT2MU_CONNECTION.salt2mu_results["sigint"]
            )
            / _WORKER_REALDATA_SALT2MU_RESULTS["siginterr"]
        )
        ** 2
    )

    # ========== Observable distributions ==========
    nevt_high = inparr["nevt_high"][0]
    nevt_low = inparr["nevt_low"][0]

    # Color histogram
    data_color, sim_color = inparr["color_hist"]
    if len(data_color) > 0 and len(sim_color) > 0:
        datacount_color, simcount_color, poisson_color, _ = normhisttodata(data_color, sim_color)
        ll_dict["color_hist"] = -0.5 * np.sum(
            (datacount_color - simcount_color) ** 2 / poisson_color**2
        )
        datacount_dict["color_hist"] = datacount_color
        simcount_dict["color_hist"] = simcount_color
        poisson_dict["color_hist"] = poisson_color

    # X1 (stretch) histogram
    data_x1, sim_x1 = inparr["x1_hist"]
    if len(data_x1) > 0 and len(sim_x1) > 0:
        datacount_x1, simcount_x1, poisson_x1, _ = normhisttodata(data_x1, sim_x1)
        ll_dict["x1_hist"] = -0.5 * np.sum((datacount_x1 - simcount_x1) ** 2 / poisson_x1**2)
        datacount_dict["x1_hist"] = datacount_x1
        simcount_dict["x1_hist"] = simcount_x1
        poisson_dict["x1_hist"] = poisson_x1

    # High-mass MURES
    data_mures_high, sim_mures_high = inparr["mures_high"]
    poisson_mures_high = inparr["rms_high"][0] / np.sqrt(nevt_high)
    ll_dict["mures_high"] = -0.5 * np.sum(
        (data_mures_high - sim_mures_high) ** 2 / poisson_mures_high**2
    )
    datacount_dict["mures_high"] = data_mures_high
    simcount_dict["mures_high"] = sim_mures_high
    poisson_dict["mures_high"] = poisson_mures_high

    # Low-mass MURES
    data_mures_low, sim_mures_low = inparr["mures_low"]
    poisson_mures_low = inparr["rms_low"][0] / np.sqrt(nevt_low)
    ll_dict["mures_low"] = -0.5 * np.sum(
        (data_mures_low - sim_mures_low) ** 2 / poisson_mures_low**2
    )
    datacount_dict["mures_low"] = data_mures_low
    simcount_dict["mures_low"] = sim_mures_low
    poisson_dict["mures_low"] = poisson_mures_low

    # High-mass RMS
    data_rms_high, sim_rms_high = inparr["rms_high"]
    poisson_rms_high = data_rms_high / np.sqrt(2 * nevt_high)
    ll_dict["rms_high"] = (
        -0.5 * np.sum((data_rms_high - sim_rms_high) ** 2 / poisson_rms_high**2) * rms_weight
    )
    datacount_dict["rms_high"] = data_rms_high
    simcount_dict["rms_high"] = sim_rms_high
    poisson_dict["rms_high"] = poisson_rms_high

    # Low-mass RMS
    data_rms_low, sim_rms_low = inparr["rms_low"]
    poisson_rms_low = data_rms_low / np.sqrt(2 * nevt_low)
    ll_dict["rms_low"] = (
        -0.5 * np.sum((data_rms_low - sim_rms_low) ** 2 / poisson_rms_low**2) * rms_weight
    )
    datacount_dict["rms_low"] = data_rms_low
    simcount_dict["rms_low"] = sim_rms_low
    poisson_dict["rms_low"] = poisson_rms_low

    # Check for invalid values
    invalid_components = []
    for key, value in ll_dict.items():
        if not np.isfinite(value) or (
            isinstance(value, np.ndarray) and not np.all(np.isfinite(value))
        ):
            invalid_components.append(key)

    if invalid_components:
        logger.warning(f"Invalid (NaN/inf) likelihood components: {invalid_components}")
        logger.warning(f"ll_dict values: {dict(ll_dict)}")
        if returnall:
            return (
                float(sum(ll_dict.values())),
                dict(ll_dict),
                dict(datacount_dict),
                dict(simcount_dict),
                dict(poisson_dict),
            )
        else:
            return -np.inf

    if returnall:
        return (
            float(sum(ll_dict.values())),
            dict(ll_dict),
            dict(datacount_dict),
            dict(simcount_dict),
            dict(poisson_dict),
        )

    return float(sum(ll_dict.values()))


def log_likelihood(
    theta: NDArray[np.float64] | list[float],
    returnall: bool = False,
) -> float | tuple[dict, dict, dict, dict]:
    """
    Calculate log-likelihood for proposed parameter values.

    Core likelihood function for MCMC. For each parameter set:
    1. Writes PDF functions to file via connection.write_generic_PDF()
    2. Calls SALT2mu.exe to reweight simulation with those PDFs
    3. Parses binned output from SALT2mu (color histograms, MURES, RMS by mass)
    4. Compares reweighted simulation to real data

    Args:
        theta: Array of parameter values (length = ndim).
        returnall: If True, return detailed likelihood components.

    Returns:
        Log-likelihood value (float).
        If returnall=True: tuple of (ll_dict, datacount_dict, simcount_dict, poisson_dict).
        Returns -inf if MAXPROB > 1.001 (PDF hitting boundary).
    """
    logger.debug(f"Current PID is {os.getpid()}")
    logger.debug("writing PDF")

    theta_index_dic = thetaconverter(theta)
    logger.debug(f"theta = {theta}, theta_dic={theta_index_dic}")

    # Run SALT2mu with these PDFs
    _WORKER_SALT2MU_CONNECTION.next_iter(theta, theta_index_dic, _CONFIG)

    if _WORKER_SALT2MU_CONNECTION.salt2mu_results["maxprob"] > 1.001:
        logger.debug(
            f"{_WORKER_SALT2MU_CONNECTION.salt2mu_results['maxprob']} MAXPROB > 1! "
            "Returning -np.inf"
        )
        return -np.inf

    logger.debug("Right before calculation")
    bindf = _WORKER_SALT2MU_CONNECTION.salt2mu_results["bindf"].dropna()
    sim_vals = dffixer(bindf, "ANALYSIS")

    realbindf = _WORKER_REALDATA_SALT2MU_RESULTS["bindf"].dropna()
    real_vals = dffixer(realbindf, "ANALYSIS")

    # Build dictionary pairing data and simulation values
    inparr = {key: [real_vals[key], sim_vals[key]] for key in real_vals.keys()}

    logger.debug("Right before calling LL Creator")
    out_result = compute_and_sum_loglikelihoods(inparr, returnall=returnall)

    _WORKER_SALT2MU_CONNECTION.iter += 1
    return out_result


def log_prior(theta: NDArray[np.float64] | list[float]) -> float:
    """
    Calculate log-prior probability for parameter values.

    Checks if all parameters are within their allowed bounds specified in
    config.parameter_initialization. Uses uniform (flat) priors within
    bounds, returning 0 (log(1)) if all parameters are valid or -inf if
    any parameter is outside its allowed range.

    Args:
        theta: Array of parameter values (length = ndim).

    Returns:
        0.0 if all parameters within bounds, -np.inf otherwise.
    """
    thetadict = thetaconverter(theta)
    plist = pconv(
        _CONFIG.inp_params,
        _CONFIG.paramshapesdict,
        _CONFIG.splitdict,
        _CONFIG.DISTRIBUTION_PARAMETERS,
    )
    logger.debug(f"plist: {plist}")

    out_of_bounds = False
    for key in thetadict.keys():
        logger.debug(f"key: {key}")
        temp_ps = thetawriter(theta, key)
        logger.debug(f"temp_ps: {temp_ps}")
        plist_n = thetawriter(theta, key, names=plist)
        for t in range(len(temp_ps)):
            logger.debug(f"plist name: {plist_n[t]}")
            lowb = _CONFIG.parameter_initialization[plist_n[t]][3][0]
            highb = _CONFIG.parameter_initialization[plist_n[t]][3][1]
            logger.debug(f"{lowb} < {temp_ps[t]} < {highb}")
            if not lowb < temp_ps[t] < highb:
                out_of_bounds = True

    if out_of_bounds:
        return -np.inf
    else:
        return 0.0


def log_probability(theta: NDArray[np.float64] | list[float]) -> float:
    """
    Calculate log-probability (posterior) for MCMC sampling.

    Combines log-prior and log-likelihood following Bayes' theorem.
    Must be called after _init_worker has set up the worker state.

    Args:
        theta: Array of parameter values (length = ndim).

    Returns:
        Log-posterior probability (log_prior + log_likelihood).
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        logger.debug("WARNING! We returned -inf from small parameters!")
        return -np.inf
    return lp + log_likelihood(theta)


# =============================================================================
# INITIALIZATION & WORKER SETUP
# =============================================================================


def _init_worker(
    config: Config,
    realdata_salt2mu_results: dict[str, Any],
    debug: bool = False,
) -> None:
    """
    Initializer function for Pool workers.

    Sets up worker-local state by storing the appropriate connection for
    this worker based on its process identity. Called once per worker
    when the Pool is created.

    Args:
        config: Configuration object with parameters and paths.
        realdata_salt2mu_results: Dictionary containing real data fit results
            from SALT2mu (shared across workers).
        debug: If True, enable debug mode.
    """
    global _WORKER_REALDATA_SALT2MU_RESULTS
    global _WORKER_SALT2MU_CONNECTION
    global _WORKER_DEBUGFLAG
    global _CONFIG
    global _WORKER_INDEX

    _WORKER_DEBUGFLAG = debug
    _CONFIG = config

    _WORKER_INDEX = get_worker_index()

    log_path = str(Path(config.outdir) / "logs" / f"worker_{_WORKER_INDEX}.log")
    add_file_handler(log_path)
    logger.info(f"Worker {_WORKER_INDEX} logging to {log_path}")

    _WORKER_SALT2MU_CONNECTION = init_salt2mu_worker_connection()
    _WORKER_REALDATA_SALT2MU_RESULTS = realdata_salt2mu_results
